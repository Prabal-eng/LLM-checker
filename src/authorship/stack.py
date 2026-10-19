"""Increment 4d optional step: does a perplexity stack add detection? Measured on validation only.

    python -m src.authorship.stack --candidate modernbert-base

For each trained model (the main model on 100+-word validation texts, the short specialist on
25-99-word texts and excerpts), a logistic stack combines the classifier's raw score with the
Increment 3 zero-shot scores (Qwen2.5-0.5B observer and its instruct performer: Binoculars and
Fast-DetectGPT), all read on the model's text view. The stack is fitted on one half of validation and
measured on the other; halves split by parent text, so an excerpt never sits apart from its source.
It is adopted only if it adds at least 2 points of detection at 1% false AI and the GPU limits hold
with every model loaded (pre-registration). Writes a text-free report; reads only validation rows.
"""

import argparse
import hashlib
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ..bakeoff.metrics import at_threshold, threshold_at_fpr
from ..corpus.paths import REPO_BENCHMARK_DIR, VERSION
from ..corpus.text import MODEL_TEXT_VIEW, text_view

logger = logging.getLogger(__name__)

FEATURES = ("raw", "binoculars", "fast_base", "fast_instruct")
MIN_GAIN_POINTS = 2.0
GPU_PEAK_GB, GPU_P95_MS, GPU_SHEET_S = 3.5, 500.0, 120.0
SHEET_TEXTS = 3000  # a 500-row sheet with six text columns


def half_of(sample_ids: pd.Series) -> np.ndarray:
    """0 or 1 per row, by a stable hash of the parent text's id (an excerpt's id is its parent's plus ':x')."""
    parents = sample_ids.astype(str).str.removesuffix(":x")
    return parents.map(lambda s: int(hashlib.sha256(s.encode("utf-8")).hexdigest(), 16) % 2).to_numpy()


def detection(y: np.ndarray, scores: np.ndarray) -> float:
    """Share of AI texts above the threshold that keeps 1% of these humans above it."""
    return float(at_threshold(y, scores, threshold_at_fpr(scores[y == 0], 0.01))["tpr"])


def stack_gain(frame: pd.DataFrame) -> dict:
    """Fit the stack on half 0; measure classifier-alone and stacked detection at 1% false AI on half 1."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    half = half_of(frame["sample_id"])
    fit, test = half == 0, half == 1
    features = frame[list(FEATURES)]
    features = features.fillna(features[fit].median())  # zero-shot scores are NaN for one-token texts
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(features[fit], frame.loc[fit, "y"])
    y = frame.loc[test, "y"].to_numpy()
    alone = detection(y, frame.loc[test, "raw"].to_numpy())
    stacked = detection(y, model.decision_function(features[test]))
    return {"fit_rows": int(fit.sum()), "test_rows": int(test.sum()),
            "test_humans": int((y == 0).sum()), "test_ais": int((y == 1).sum()),
            "classifier_tpr_at_1pct_fpr": round(alone, 4), "stack_tpr_at_1pct_fpr": round(stacked, 4),
            "gain_points": round(100 * (stacked - alone), 2),
            "coefficients": dict(zip(FEATURES, np.round(model[-1].coef_[0], 4).tolist()))}


def adopted(gain_points: float, limits: dict) -> bool:
    return (gain_points >= MIN_GAIN_POINTS and limits["peak_vram_gb"] <= GPU_PEAK_GB
            and limits["p95_ms"] <= GPU_P95_MS and limits["sheet_seconds"] <= GPU_SHEET_S)


def gpu_limits(detectors: list, scorer, texts: list) -> dict:
    """Peak GPU memory with every model loaded, p95 single-text time for the main model plus the
    zero-shot pair, and a 500 x 6 sheet estimate from batched throughput."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("the GPU limits are measured on the GPU")
    for d in detectors:
        d._ensure_loaded()
    torch.cuda.reset_peak_memory_stats()
    singles = []
    for t in texts[:50]:
        start = time.perf_counter()
        detectors[0].raw_and_embeddings([t])
        scorer.score_frame([t])
        singles.append(1000 * (time.perf_counter() - start))
    batch = texts[:500]
    start = time.perf_counter()
    detectors[0].raw_and_embeddings(batch)
    scorer.score_frame(batch)
    per_text = (time.perf_counter() - start) / len(batch)
    return {"peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3),
            "p95_ms": round(float(np.percentile(singles, 95)), 1),
            "sheet_seconds": round(per_text * SHEET_TEXTS, 1)}


def _release_gpu() -> None:
    import gc

    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def report_file(candidate: str) -> Path:
    return REPO_BENCHMARK_DIR / VERSION / "model" / candidate / "stack.json"


def run(candidate: str) -> dict:
    from ..bakeoff.neural import ZeroShotScorer
    from .calibrate import _detector, load_split, short_rows, with_excerpts
    from .train import record_file

    val = load_split("val")
    val = pd.concat([val.assign(excerpt=False), with_excerpts(val)], ignore_index=True)
    view = text_view(MODEL_TEXT_VIEW)
    report = {"rule": f"adopted when the stack adds {MIN_GAIN_POINTS} points of detection at 1% false AI on "
                      f"the held-out half and the GPU limits hold", "models": {}}
    # Each classifier scores alone and is freed before the zero-shot pair loads: together they spill a
    # 4 GB card into shared memory, which made scoring many times slower. The GPU limits below still load
    # every model at once, as the pre-registration measures them.
    trainings, scored = {}, {}
    for variant, rows in (("main", val[val["words"] >= 100]), ("short", short_rows(val))):
        path = record_file(variant=variant, candidate=candidate)
        if not path.exists():
            continue
        trainings[variant] = json.loads(path.read_text(encoding="utf-8"))
        detector = _detector(trainings[variant])
        logger.info("%s: scoring %d validation texts", variant, len(rows))
        raw, _ = detector.raw_and_embeddings(rows["text"].tolist())
        scored[variant] = (rows, raw)
        del detector
        _release_gpu()
    scorer = ZeroShotScorer()
    for variant, (rows, raw) in scored.items():
        logger.info("%s: zero-shot scores for %d validation texts", variant, len(rows))
        zero_shot = scorer.score_frame([view(t) for t in rows["text"]])
        frame = pd.concat([rows[["sample_id", "y"]].reset_index(drop=True), zero_shot], axis=1).assign(raw=raw)
        report["models"][variant] = {"model_id": trainings[variant]["model_id"], **stack_gain(frame)}
    detectors = [_detector(training) for training in trainings.values()]
    report["gpu_limits"] = gpu_limits(detectors, scorer, val["text"].sample(n=500, random_state=0).tolist())
    for entry in report["models"].values():
        entry["adopted"] = adopted(entry["gain_points"], report["gpu_limits"])
    report["locked_split_reads"] = 0
    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    out = report_file(candidate)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--candidate", required=True)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for noisy in ("transformers", "httpx", "huggingface_hub", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.ERROR)
    print(json.dumps(run(args.candidate), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
