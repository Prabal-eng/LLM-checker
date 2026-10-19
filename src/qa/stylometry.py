"""Stylometric feature extraction for machine learning AI detectors."""
import logging
import re
import math
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

logger = logging.getLogger(__name__)


class StylometricFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extracts dense stylometric and linguistic features from text."""

    def __init__(self):
        pass

    def fit(self, x, y=None):
        return self

    def transform(self, texts):
        features = []
        for text in texts:
            clean = str(text or "").strip()
            words = re.findall(r"\b\w+\b", clean.lower())
            word_count = len(words)
            char_count = len(clean)

            # Sentences & burstiness
            sentences = [s.strip() for s in re.split(r"[.!?]+", clean) if s.strip()]
            sent_lens = [len(re.findall(r"\b\w+\b", s)) for s in sentences if s]

            if len(sent_lens) > 1:
                mean_len = sum(sent_lens) / len(sent_lens)
                variance = sum((l - mean_len) ** 2 for l in sent_lens) / len(sent_lens)
                burstiness = math.sqrt(variance) / (mean_len + 1e-5)
            else:
                burstiness = 0.5

            # Lexical diversity (TTR)
            unique_words = len(set(words))
            ttr = unique_words / (word_count + 1e-5)

            # Punctuation & case statistics
            comma_density = clean.count(",") / (word_count + 1.0)
            semicolon_count = clean.count(";")
            uppercase_words = sum(1 for w in re.findall(r"\b[A-Z]{2,}\b", clean))
            avg_word_length = char_count / (word_count + 1e-5)

            # Formal transition indicators
            formal_transitions = sum(1 for w in [
                "furthermore", "moreover", "conclusion", "consequently", "testament",
                "crucial", "seamlessly", "meticulously", "notwithstanding", "comprehensive"
            ] if w in clean.lower())

            features.append([
                word_count,
                burstiness,
                ttr,
                comma_density,
                semicolon_count,
                uppercase_words,
                avg_word_length,
                formal_transitions
            ])

        return np.array(features, dtype=np.float32)
