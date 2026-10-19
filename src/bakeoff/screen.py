"""Increment 4d model screening: rank candidate encoders on validation before any full training.

    python -m src.bakeoff.screen [--candidates distilroberta-base,roberta-base] [--tracks main,short]
                                 [--train-rows 24000]

Each candidate trains for one epoch on a label-balanced sample of the train split, with the same
source-balanced weights as full training, and is scored on the validation split. It reports AUROC,
detection at a 1% false-AI threshold overall and per word band, training time, peak GPU memory and
scoring speed. The main track uses the main model's data at 256 tokens; the short track uses 25-99-word
texts and excerpts at 160 tokens. Every candidate uses length-grouped batches.

Results are text-free JSON, one file per candidate and track, so an interrupted run resumes where it
stopped; a candidate that fails (out of memory, non-finite loss, missing tokenizer) is recorded and the
rest still run. Only train and validation rows are read.
"""

import argparse
import gc
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from ..corpus.paths import REPO_BENCHMARK_DIR, VERSION
from ..corpus.text import MODEL_TEXT_VIEW
from .data import folds, load_fit_data, prepare
from .metrics import at_threshold, auroc, threshold_at_fpr
from .run import selection_data

logger = logging.getLogger(__name__)

OUT_DIR = REPO_BENCHMARK_DIR / "v4" / "screening"
TRACKS = {"main": 256, "short": 160}
BANDS = ((25, 49, "25-49"), (50, 99, "50-99"), (100, 199, "100-199"), (200, None, "200+"))

# Base encoders train fully; large ones train LoRA adapters so they fit in 4 GB. DeBERTa runs in fp16
# with loss scaling because DeBERTa-v3-small gave NaN scores under bf16 in Increment 3.
CANDIDATES = (
    {"name": "distilroberta-base", "kwargs": {"model_name": "distilroberta-base"}},
    {"name": "roberta-base", "kwargs": {"model_name": "FacebookAI/roberta-base", "lr": 2e-5,
                                        "gradient_checkpointing": True}},
    {"name": "modernbert-base", "kwargs": {"model_name": "answerdotai/ModernBERT-base", "lr": 3e-5,
                                           "optimizer": "adamw8bit", "gradient_checkpointing": True}},
    {"name": "deberta-v3-base", "kwargs": {"model_name": "microsoft/deberta-v3-base", "lr": 2e-5,
                                           "precision": "fp16", "optimizer": "adamw8bit",
                                           "gradient_checkpointing": True}},
    {"name": "modernbert-large-lora", "kwargs": {"model_name": "answerdotai/ModernBERT-large", "lr": 2e-4,
                                                 "lora": {"r": 16, "alpha": 32}, "gradient_checkpointing": True,
                                                 "score_batch_size": 16}},
    {"name": "deberta-v3-large-lora", "kwargs": {"model_name": "microsoft/deberta-v3-large", "lr": 2e-4,
                                                 "precision": "fp16", "lora": {"r": 16, "alpha": 32},
                                                 "gradient_checkpointing": True, "score_batch_size": 16}},
)


def build_classifier(spec: dict, track: str):
    from .neural import FineTunedClassifier

    return FineTunedClassifier(max_train=None, epochs=1, text_view=MODEL_TEXT_VIEW, max_tokens=TRACKS[track],
                               group_by_length=True, **spec["kwargs"])


def track_data(df: pd.DataFrame, track: str) -> pd.DataFrame:
    if track == "short":
        from ..authorship.excerpts import short_frame

        return prepare(short_frame(df))
    return df


def _detection(y: np.ndarray, scores: np.ndarray) -> dict:
    threshold = threshold_at_fpr(scores[y == 0], 0.01)
    return {"auroc": auroc(y, scores), "tpr_at_1pct_fpr": float(at_threshold(y, scores, threshold)["tpr"])}


def band_metrics(df: pd.DataFrame, scores: np.ndarray) -> dict:
    """Detection per word band, each band's threshold set on its own human texts."""
    words = df["text"].map(lambda t: len(str(t).split())).to_numpy()
    y = df["y"].to_numpy()
    out = {}
    for lo, hi, label in BANDS:
        mask = (words >= lo) & (words <= (hi if hi is not None else np.inf))
        humans, ais = int(((y == 0) & mask).sum()), int(((y == 1) & mask).sum())
        if humans and ais:
            out[label] = {"humans": humans, "ais": ais, **_detection(y[mask], scores[mask])}
    return out


def _balanced(fit: pd.DataFrame, rows: int) -> pd.DataFrame:
    per_label = min(rows // 2, int(fit["y"].value_counts().min()))
    return fit.groupby("y", group_keys=False).sample(n=per_label, random_state=0)


def _release(clf) -> None:
    del clf
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _screen_one(spec: dict, track: str, fit: pd.DataFrame, val: pd.DataFrame) -> dict:
    clf = build_classifier(spec, track)
    try:
        started = time.perf_counter()
        clf.fit(fit, balance_sources=True)
        trained = time.perf_counter()
        scores = np.asarray(clf.score(val["text"].tolist()), dtype=float)
        scored = time.perf_counter()
        peak = getattr(clf, "peak_vram_gb", None)
    finally:
        _release(clf)
    return {
        "name": spec["name"],
        "track": track,
        "settings": spec["kwargs"],
        "max_tokens": TRACKS[track],
        "train_rows": len(fit),
        "validation_rows": len(val),
        "validation": _detection(val["y"].to_numpy(), scores),
        "bands": band_metrics(val, scores),
        "train_seconds": round(trained - started, 1),
        "score_ms_per_text": round(1000 * (scored - trained) / max(len(val), 1), 2),
        "peak_train_vram_gb": round(peak, 2) if peak is not None else None,
        "corpus_version": VERSION,
        "locked_split_reads": 0,
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }


def _write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run(specs: Sequence[dict] = CANDIDATES, *, tracks: Sequence[str] = ("main", "short"), train_rows: int = 24_000,
        out_dir: Path = OUT_DIR, data: Optional[pd.DataFrame] = None) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = data if data is not None else selection_data(load_fit_data())
    for track in tracks:
        fold = next(iter(folds(track_data(df, track))))
        fit, val = _balanced(fold.fit, train_rows), fold.calib
        for spec in specs:
            done, failed = out_dir / f"{spec['name']}-{track}.json", out_dir / f"{spec['name']}-{track}.error.json"
            if done.exists():
                continue
            logger.info("screening %s on the %s track: %d train rows, %d validation rows",
                        spec["name"], track, len(fit), len(val))
            try:
                result = _screen_one(spec, track, fit, val)
            except Exception as exc:  # one candidate's failure must not stop an overnight run
                logger.exception("%s failed on the %s track", spec["name"], track)
                _write(failed, {"name": spec["name"], "track": track, "settings": spec["kwargs"],
                                "error": f"{type(exc).__name__}: {exc}"[:2000],
                                "failed_at": datetime.now(timezone.utc).isoformat()})
                continue
            failed.unlink(missing_ok=True)
            _write(done, result)
            logger.info("%s %s: %s", spec["name"], track, result["validation"])
    summary = {}
    for track in tracks:
        rows = []
        for spec in specs:
            for path in (out_dir / f"{spec['name']}-{track}.json", out_dir / f"{spec['name']}-{track}.error.json"):
                if path.exists():
                    rows.append(json.loads(path.read_text(encoding="utf-8")))
                    break
        summary[track] = rows
    _write(out_dir / "summary.json", summary)
    return summary


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--candidates", default=",".join(c["name"] for c in CANDIDATES))
    parser.add_argument("--tracks", default="main,short")
    parser.add_argument("--train-rows", type=int, default=24_000)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    wanted = args.candidates.split(",")
    unknown = set(wanted) - {c["name"] for c in CANDIDATES}
    if unknown:
        parser.error(f"unknown candidates: {sorted(unknown)}")
    specs = [c for c in CANDIDATES if c["name"] in wanted]
    summary = run(specs, tracks=args.tracks.split(","), train_rows=args.train_rows, out_dir=args.out_dir)
    for track, rows in summary.items():
        for row in rows:
            if "error" in row:
                print(f"{track:5} {row['name']:24} ERROR {row['error'][:80]}")
            else:
                print(f"{track:5} {row['name']:24} TPR@1%FPR {row['validation']['tpr_at_1pct_fpr']:.4f} "
                      f"AUROC {row['validation']['auroc']:.4f} train {row['train_seconds']:.0f}s "
                      f"VRAM {row['peak_train_vram_gb']} GB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
