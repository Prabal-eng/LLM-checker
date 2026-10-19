"""Candidate detectors. Every candidate maps texts to scores where higher means more likely AI."""

from typing import Protocol, Sequence

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_union


class Candidate(Protocol):
    name: str
    trainable: bool

    def fit(self, train: pd.DataFrame) -> None: ...

    def score(self, texts: Sequence[str]) -> np.ndarray: ...


class LegacyDetector:
    """The shipped hybrid model, unchanged, as the baseline to beat."""

    name, trainable = "legacy", False

    def __init__(self) -> None:
        from ..detectors.trained_detector import TrainedInDomainDetector

        self._detector = TrainedInDomainDetector()

    def fit(self, train: pd.DataFrame) -> None:
        pass

    def score(self, texts: Sequence[str]) -> np.ndarray:
        return np.array([r.ai_score for r in self._detector.detect_batch(list(texts))])


class TfidfBaseline:
    """Hashed word and character n-grams with a linear classifier, trained in batches.

    Hashing is stateless and training streams batches, so memory stays bounded however
    large the training split is; a fitted vocabulary did not fit in RAM.
    """

    name, trainable = "tfidf", True

    # The opening of a text carries the style signal and keeps feature extraction cheap.
    MAX_CHARS = 1500
    BATCH = 4000
    EPOCHS = 1

    def __init__(self, max_train: int = 60_000) -> None:
        self.max_train = max_train
        self._features = make_union(
            HashingVectorizer(ngram_range=(1, 2), n_features=2**20, alternate_sign=False, dtype=np.float32),
            HashingVectorizer(analyzer="char_wb", ngram_range=(3, 5), n_features=2**20, alternate_sign=False,
                              dtype=np.float32),
        )
        self._model = None

    def _transform(self, texts: Sequence[str]):
        return self._features.transform([t[: self.MAX_CHARS] for t in texts])

    def fit(self, train: pd.DataFrame) -> None:
        if len(train) > self.max_train:
            train = train.sample(self.max_train, random_state=0)
        y = train["y"].to_numpy()
        # partial_fit cannot balance classes itself, so weight each sample by inverse class frequency.
        weights = np.where(y == 1, len(y) / (2 * max(y.sum(), 1)), len(y) / (2 * max((1 - y).sum(), 1)))
        texts = train["text"].tolist()
        self._model = SGDClassifier(loss="log_loss", alpha=1e-6, random_state=0)
        rng = np.random.default_rng(0)
        for _ in range(self.EPOCHS):
            order = rng.permutation(len(texts))
            for start in range(0, len(order), self.BATCH):
                idx = order[start:start + self.BATCH]
                self._model.partial_fit(self._transform([texts[i] for i in idx]), y[idx],
                                        classes=[0, 1], sample_weight=weights[idx])

    def score(self, texts: Sequence[str]) -> np.ndarray:
        texts = list(texts)
        return np.concatenate([self._model.decision_function(self._transform(texts[i:i + self.BATCH]))
                               for i in range(0, len(texts), self.BATCH)]) if texts else np.array([])
