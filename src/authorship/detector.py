"""The selected detector behind the shared decision policy.

One forward pass gives both the authorship score and a text embedding. The embedding feeds an
out-of-distribution check: the mean cosine distance to the k nearest training texts, compared
with a threshold set on validation. torch and transformers are imported only when the model is
first used, so the service and tests run without them.

LengthRoutedDetector puts a short-text specialist in front of the main model for the word-count
bands the specialist won on validation.
"""

import logging
import threading
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

from ..detectors.base import BaseDetector, DetectionResult
from .calibration import probability

logger = logging.getLogger(__name__)

WINDOW_WORDS = 160  # sentence-started windows of this many words fit the 256-token model window
PRECISIONS = ("bf16", "fp16", "fp32")  # GPU autocast dtype for inference; fp32 means none, and the CPU runs fp32


class OodGate:
    """k-nearest-neighbour cosine distance to a bank of L2-normalised training embeddings."""

    def __init__(self, bank: np.ndarray, k: int, threshold: Optional[float]) -> None:
        self.bank, self.k, self.threshold = bank.astype(np.float32), k, threshold

    CHUNK = 1024  # rows per similarity block, so memory stays bounded for large batches and banks

    def distances(self, embeddings: np.ndarray) -> np.ndarray:
        out = np.empty(len(embeddings), dtype=np.float64)
        k = min(self.k, len(self.bank))
        for start in range(0, len(embeddings), self.CHUNK):
            sims = embeddings[start:start + self.CHUNK].astype(np.float32) @ self.bank.T
            nearest = np.partition(sims, sims.shape[1] - k, axis=1)[:, -k:]
            out[start:start + self.CHUNK] = 1.0 - nearest.mean(axis=1)
        return out

    def in_distribution(self, distances: np.ndarray) -> np.ndarray:
        if self.threshold is None:
            return np.ones(len(distances), dtype=bool)
        return distances <= self.threshold


class AuthorshipModelDetector(BaseDetector):
    """Fine-tuned DistilRoBERTa scores, mapped to probabilities, with an OOD verdict per text."""

    provider = "distilroberta-authorship"

    # Inference batch 16 (Amendment 16): 32 peaked at 3.94 GB with a large main and short model loaded.
    def __init__(self, model_dir: Path, platt: Tuple[float, float], gate: Optional[OodGate] = None,
                 max_tokens: int = 256, batch_size: int = 16, device: Optional[str] = None,
                 text_view: Optional[str] = None, expected_files_sha256: Optional[str] = None,
                 precision: str = "bf16", half_weights: bool = False) -> None:
        super().__init__(threshold=0.5)
        from ..corpus.text import text_view as view_of

        if precision not in PRECISIONS:
            # Read from the registry record (Increment 4d); fail here, before any scoring.
            raise ValueError(f"unknown inference precision {precision!r}; use one of {PRECISIONS}")
        self.precision = precision
        # Records that state their precision load their GPU weights in it, so a large main and short model
        # fit in 4 GB together; older records keep the fp32 weights they were calibrated with.
        self.half_weights = half_weights
        self.text_view = text_view
        self._view = view_of(text_view)  # an unknown recorded view fails here, before any scoring
        self.expected_files_sha256 = expected_files_sha256
        self.model_dir, self.platt, self.gate = Path(model_dir), platt, gate
        self.max_tokens, self.batch_size, self.device = max_tokens, batch_size, device
        self._model = self._tokenizer = None
        self._lock = threading.Lock()

    def _ensure_loaded(self) -> None:
        with self._lock:
            if self._model is not None:
                return
            if self.expected_files_sha256:
                # Checked again at load: the registry verified the files when the service was built,
                # and the weights are read only on the first request (Ralph R07-8).
                from .registry import directory_sha256

                if directory_sha256(self.model_dir) != self.expected_files_sha256:
                    raise RuntimeError(f"{self.model_dir.name}: model files changed since they were verified")
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.model_dir, dtype=self.load_dtype(torch.device(device).type)).to(device).eval()
            logger.info("loaded %s on %s", self.model_dir.name, device)

    def load_dtype(self, device_type: str):
        """The dtype the weights load in: the recorded precision on the GPU when half_weights, else fp32."""
        import torch

        dtype = self.autocast_dtype() if self.half_weights and device_type == "cuda" else None
        return dtype or torch.float32

    def autocast_dtype(self):
        """The GPU autocast dtype for the recorded precision; None means plain fp32."""
        import torch

        return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": None}[self.precision]

    def raw_and_embeddings(self, texts: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Raw logit margins (AI minus human) and L2-normalised mean-pooled embeddings."""
        import torch

        self._ensure_loaded()
        texts = [self._prep(t) for t in texts]
        device = next(self._model.parameters()).device
        dtype = self.autocast_dtype()
        order = np.argsort([len(t) for t in texts])
        raw = np.empty(len(texts))
        emb: Optional[np.ndarray] = None
        with torch.inference_mode():
            for start in range(0, len(texts), self.batch_size):
                idx = order[start:start + self.batch_size]
                enc = self._tokenizer([texts[i] for i in idx], return_tensors="pt", padding=True,
                                      truncation=True, max_length=self.max_tokens).to(device)
                with torch.autocast(device.type, dtype=dtype or torch.float32,
                                    enabled=device.type == "cuda" and dtype is not None):
                    out = self._model(**enc, output_hidden_states=True)
                logits = out.logits.float()
                if not torch.isfinite(logits).all():
                    # A decision must never rest on a NaN or infinite score (half-precision weights can overflow).
                    raise FloatingPointError(f"{self.model_dir.name}: non-finite scores")
                hidden = out.hidden_states[-1].float()
                mask = enc["attention_mask"].unsqueeze(-1).float()
                pooled = (hidden * mask).sum(1) / mask.sum(1)
                pooled = torch.nn.functional.normalize(pooled, dim=1)
                if emb is None:
                    emb = np.empty((len(texts), pooled.shape[1]), dtype=np.float32)
                raw[idx] = (logits[:, 1] - logits[:, 0]).cpu().numpy()
                emb[idx] = pooled.cpu().numpy()
        return raw, emb if emb is not None else np.empty((0, 0), dtype=np.float32)

    def windows(self, text: str) -> List[str]:
        """The text itself, then (when it is longer than one model window) sentence-started windows
        covering the rest. The first entry is scored exactly as in calibration: read from its start
        and cut at max_tokens."""
        from .excerpts import sentence_windows

        self._ensure_loaded()
        prepared = self._prep(text)
        if len(self._tokenizer(prepared, add_special_tokens=False, verbose=False)["input_ids"]) <= self.max_tokens - 2:
            return [text]
        return [text] + sentence_windows(prepared, WINDOW_WORDS)[1:]

    def _prep(self, text: str) -> str:
        return self._view(text)

    def detect_batch(self, texts: List[str], max_workers: int = 1) -> List[DetectionResult]:
        """One result per text. A text longer than one model window is scored in full, window by
        window (Ralph R07-2); its windows are listed in the analysis for the policy to combine."""
        if not texts:
            return []
        per_text = [self.windows(t) for t in texts]
        flat = [w for ws in per_text for w in ws]
        raw, emb = self.raw_and_embeddings(flat)
        probs = probability(raw, *self.platt)
        if self.gate is not None:
            distances = self.gate.distances(emb)
            inside = self.gate.in_distribution(distances)
        else:
            distances, inside = np.full(len(flat), np.nan), np.ones(len(flat), dtype=bool)
        entries = [{"text": w, "score": float(np.clip(p, 0.0, 1.0)), "raw_score": round(float(r), 4),
                    "ood_distance": None if np.isnan(d) else round(float(d), 4), "in_distribution": bool(i)}
                   for w, p, r, d, i in zip(flat, probs, raw, distances, inside)]
        results, start = [], 0
        for ws in per_text:
            mine, start = entries[start:start + len(ws)], start + len(ws)
            first = mine[0]
            analysis = {k: first[k] for k in ("raw_score", "ood_distance", "in_distribution")}
            if len(mine) > 1:
                analysis["windows"] = mine
                analysis["in_distribution"] = all(m["in_distribution"] for m in mine)
            results.append(DetectionResult(provider=self.provider, ai_score=first["score"],
                                           is_flagged=bool(first["score"] >= self.threshold), analysis=analysis))
        return results

    def detect(self, text: str) -> DetectionResult:
        return self.detect_batch([text])[0]


class LengthRoutedDetector(BaseDetector):
    """Scores each text with the short-text specialist when its word count falls in a range the
    specialist serves, and with the main model otherwise. Word counts use the policy's rule, so a
    text is scored by the model whose thresholds its band uses."""

    provider = "distilroberta-authorship-routed"

    def __init__(self, main: BaseDetector, short: BaseDetector, short_ranges: Sequence[Tuple[int, Optional[int]]]):
        super().__init__(threshold=0.5)
        self.main, self.short, self.short_ranges = main, short, [tuple(r) for r in short_ranges]

    def serves_short(self, text: str) -> bool:
        from ..detection.policy import DecisionPolicy

        words = DecisionPolicy.word_count(text)
        return any(lo <= words < (hi if hi is not None else np.inf) for lo, hi in self.short_ranges)

    def detect_batch(self, texts: List[str], max_workers: int = 1) -> List[DetectionResult]:
        results: List[Optional[DetectionResult]] = [None] * len(texts)
        routes = [self.serves_short(t) for t in texts]
        for name, detector, wanted in (("short", self.short, True), ("main", self.main, False)):
            idx = [i for i, r in enumerate(routes) if r is wanted]
            if idx:
                for i, result in zip(idx, detector.detect_batch([texts[i] for i in idx])):
                    result.analysis["model"] = name
                    results[i] = result
        return results  # type: ignore[return-value]

    def detect(self, text: str) -> DetectionResult:
        return self.detect_batch([text])[0]
