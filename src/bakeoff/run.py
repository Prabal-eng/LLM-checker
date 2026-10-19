"""Model selection bake-off. Reads only the train and validation splits.

    python -m src.bakeoff.run select [--candidates legacy,tfidf,...]
    python -m src.bakeoff.run bench  [--candidates ...]

Results are written to data/benchmark/<version>/bakeoff/ (metrics only) and per-text
scores to CORPUS_DIR/<version>/bakeoff/.
"""

import argparse
import json
import logging
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ..corpus.paths import REPO_BENCHMARK_DIR, VERSION, corpus_dir
from ..corpus.sources._common import take_per_stratum
from .candidates import LegacyDetector, TfidfBaseline
from .data import LOFO_FAMILIES, RECENT_OPENAI, folds, load_fit_data
from .metrics import fold_metrics

logger = logging.getLogger(__name__)

CANDIDATES = ("legacy", "tfidf", "zs-fast-base", "zs-fast-instruct", "zs-binoculars", "distilroberta")
VAL_PER_STRATUM = 1000
SHEET_CELLS = 1500  # a 500-row x 6-column sheet after short cells are skipped


def build_candidates(names: list[str]) -> list:
    built, zero_shot = [], None
    for name in names:
        if name == "legacy":
            built.append(LegacyDetector())
        elif name == "tfidf":
            built.append(TfidfBaseline())
        elif name.startswith("zs-"):
            from .neural import ZeroShotCandidate, ZeroShotScorer

            if zero_shot is None:
                zero_shot = (ZeroShotScorer(), {})
            column = {"zs-fast-base": "fast_base", "zs-fast-instruct": "fast_instruct",
                      "zs-binoculars": "binoculars"}[name]
            built.append(ZeroShotCandidate(zero_shot[0], column, zero_shot[1]))
        elif name == "distilroberta":
            from .neural import FineTunedClassifier

            built.append(FineTunedClassifier())
        else:
            raise ValueError(f"unknown candidate {name}")
    return built


def selection_data(df: pd.DataFrame) -> pd.DataFrame:
    """All train rows plus a stratified validation sample shared by every candidate."""
    val = df[df["split"] == "val"]
    sample = take_per_stratum(val, ["source_dataset", "y"], VAL_PER_STRATUM, "sample_id")
    unseen = val[val["generator_family"].isin(LOFO_FAMILIES)]
    val_eval = pd.concat([sample, unseen]).drop_duplicates("sample_id").assign(eval_only=False)
    # Every newest-GPT validation row, for the recent:openai fold only; other folds ignore them,
    # so adding this fold leaves their data (and checkpoints) unchanged.
    recent = val[val["generator"].isin(RECENT_OPENAI) & ~val["sample_id"].isin(val_eval["sample_id"])]
    return pd.concat([df[df["split"] == "train"].assign(eval_only=False), val_eval,
                      recent.assign(eval_only=True)], ignore_index=True)


def summarize(fold_results: dict) -> dict:
    loso = {k: v for k, v in fold_results.items() if k.startswith("loso:")}
    lofo = {k: v for k, v in fold_results.items() if k.startswith("lofo:")}

    def values(group: dict, *path: str) -> list:
        out = []
        for result in group.values():
            v = result
            for key in path:
                v = v.get(key) if isinstance(v, dict) else None
            if v is not None:
                out.append(v)
        return out

    loso_auc = values(loso, "auroc")
    return {
        "val_auroc": fold_results.get("val", {}).get("auroc"),
        "val_tpr_at_1pct_fpr": fold_results.get("val", {}).get("at_fpr_0.01", {}).get("tpr"),
        "loso_mean_auroc": round(statistics.mean(loso_auc), 4) if loso_auc else None,
        "loso_worst_auroc": min(loso_auc) if loso_auc else None,
        "loso_mean_tpr_at_1pct": round(statistics.mean(values(loso, "at_fpr_0.01", "tpr")), 4)
        if values(loso, "at_fpr_0.01", "tpr") else None,
        # How far the false-positive rate drifts when a threshold set on other sources is reused.
        "loso_worst_transferred_fpr": max(values(loso, "at_fpr_0.01", "fpr"), default=None),
        "lofo_mean_tpr_at_1pct": round(statistics.mean(values(lofo, "at_fpr_0.01", "tpr")), 4)
        if values(lofo, "at_fpr_0.01", "tpr") else None,
        "recent_openai_auroc": fold_results.get("recent:openai", {}).get("auroc"),
        "recent_openai_tpr_at_1pct": fold_results.get("recent:openai", {}).get("at_fpr_0.01", {}).get("tpr"),
    }


CUDA_RETRIES = 3


def _with_retries(compute, what: str):
    """Retry transient CUDA driver faults (laptop GPUs can drop out briefly), then give up."""
    for attempt in range(1, CUDA_RETRIES + 1):
        try:
            return compute()
        except RuntimeError as exc:
            if "CUDA" not in str(exc) or attempt == CUDA_RETRIES:
                raise
            logger.warning("%s: CUDA fault (%s); retry %d/%d in 30 s", what, str(exc).splitlines()[0],
                           attempt, CUDA_RETRIES - 1)
            try:
                import torch

                torch.cuda.empty_cache()
            except Exception:
                pass
            time.sleep(30)


def _write_summary(results_dir: Path, out_file: Path, extra: dict) -> dict:
    """Rebuild the selection summary from per-candidate files, so parallel runs never clobber each other."""
    candidates = {}
    for path in sorted(results_dir.glob("*.json")):
        item = json.loads(path.read_text(encoding="utf-8"))
        candidates[item["name"]] = {k: v for k, v in item.items() if k != "name"}
    record = {**extra, "candidates": candidates}
    out_file.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return record


def select(names: list[str], out_file: Path, scores_dir: Path) -> dict:
    df = selection_data(load_fit_data())
    extra = {"version": VERSION, "locked_split_reads": 0,
             "selection_rows": {"train": int((df["split"] == "train").sum()),
                                "val_eval": int((df["split"] == "val").sum())}}
    results_dir = out_file.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    scores_dir.mkdir(parents=True, exist_ok=True)
    fold_list = list(folds(df))

    for candidate in build_candidates(names):
        started = time.time()
        safe = candidate.name.replace(":", "_").replace("/", "_")
        # Each finished fold is checkpointed, so a crash costs one fold, not the whole candidate.
        partial = results_dir / "partial" / safe
        partial.mkdir(parents=True, exist_ok=True)
        results, frames = {}, []
        cached = None
        for fold in fold_list:
            done = partial / f"{fold.name.replace(':', '_')}.json"
            if done.exists():
                results[fold.name] = json.loads(done.read_text(encoding="utf-8"))
                frames.append(pd.read_parquet(done.with_suffix(".parquet")))
                if not candidate.trainable:
                    # A fixed scorer gives each text one score, so checkpointed scores seed later folds.
                    saved = frames[-1].set_index("sample_id")["score"]
                    cached = saved if cached is None else pd.concat([cached, saved])
                    cached = cached[~cached.index.duplicated()]
                logger.info("%s %s: reusing checkpoint", candidate.name, fold.name)
                continue
            if candidate.trainable:
                def compute(fold=fold):
                    logger.info("%s: fitting fold %s on %d rows", candidate.name, fold.name, len(fold.fit))
                    candidate.fit(fold.fit)
                    needed = pd.concat([fold.calib, fold.evaluation]).drop_duplicates("sample_id")
                    return pd.Series(candidate.score(needed["text"].tolist()), index=needed["sample_id"].to_numpy())
                scores = _with_retries(compute, f"{candidate.name} {fold.name}")
            else:
                cached = cached if cached is not None else pd.Series(dtype=float)
                needed = pd.concat([fold.calib, fold.evaluation]).drop_duplicates("sample_id")
                needed = needed[~needed["sample_id"].isin(cached.index)]
                if len(needed):
                    logger.info("%s: scoring %d validation texts", candidate.name, len(needed))
                    new = _with_retries(
                        lambda: pd.Series(candidate.score(needed["text"].tolist()), index=needed["sample_id"].to_numpy()),
                        f"{candidate.name} scoring")
                    cached = pd.concat([cached, new])
                scores = cached
            calib_s = scores.loc[fold.calib["sample_id"]].to_numpy()
            eval_s = scores.loc[fold.evaluation["sample_id"]].to_numpy()
            results[fold.name] = fold_metrics(fold.calib, calib_s, fold.evaluation, eval_s)
            frame = pd.DataFrame({"fold": fold.name, "sample_id": fold.evaluation["sample_id"].to_numpy(),
                                  "y": fold.evaluation["y"].to_numpy(), "score": eval_s})
            frames.append(frame)
            frame.to_parquet(done.with_suffix(".parquet"), index=False)
            done.write_text(json.dumps(results[fold.name], sort_keys=True), encoding="utf-8")
            logger.info("%s %s AUROC %s", candidate.name, fold.name, results[fold.name]["auroc"])
        pd.concat(frames).to_parquet(scores_dir / f"{safe}.parquet", index=False)
        (results_dir / f"{safe}.json").write_text(json.dumps({
            "name": candidate.name, "summary": summarize(results), "folds": results,
            "seconds": round(time.time() - started, 1), "finished_at": datetime.now(timezone.utc).isoformat(),
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        record = _write_summary(results_dir, out_file, extra)
    return record


def _time_scoring(candidate, texts: list[str]) -> float:
    started = time.perf_counter()
    candidate.score(texts)
    return time.perf_counter() - started


def bench(names: list[str], out_file: Path, cpu_sheet_sample: int = 100) -> dict:
    df = load_fit_data()
    val = df[df["split"] == "val"].sample(SHEET_CELLS, random_state=0)
    single = val.sample(30, random_state=1)["text"].tolist()
    record = json.loads(out_file.read_text(encoding="utf-8")) if out_file.exists() else {}
    for candidate in build_candidates(names):
        if candidate.trainable:
            candidate.fit(df[df["split"] == "train"].sample(2000, random_state=0))
        result = {}
        devices = ["cuda", "cpu"] if hasattr(candidate, "to_device") or hasattr(candidate, "scorer") else ["cpu"]
        for device in devices:
            mover = getattr(candidate, "scorer", candidate)
            if hasattr(mover, "to_device"):
                mover.to_device(device)
            if hasattr(candidate, "cache"):
                candidate.cache.clear()
            _time_scoring(candidate, single[:2])  # warm-up
            if hasattr(candidate, "cache"):
                candidate.cache.clear()
            latencies = []
            for text in single:
                latencies.append(_time_scoring(candidate, [text]))
                if hasattr(candidate, "cache"):
                    candidate.cache.clear()
            n = SHEET_CELLS if device == "cuda" else min(SHEET_CELLS, cpu_sheet_sample)
            sheet_texts = val["text"].tolist()[:n]
            elapsed = _time_scoring(candidate, sheet_texts)
            if hasattr(candidate, "cache"):
                candidate.cache.clear()
            result[device] = {
                "single_text_p95_s": round(float(np.percentile(latencies, 95)), 4),
                "sheet_cells_timed": n,
                "sheet_seconds_estimated": round(elapsed * SHEET_CELLS / n, 1),
                "estimated": n < SHEET_CELLS,
            }
            logger.info("%s on %s: %s", candidate.name, device, result[device])
        record[candidate.name] = result
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return record


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["select", "bench"])
    parser.add_argument("--candidates", default=",".join(CANDIDATES))
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for noisy in ("transformers", "httpx", "huggingface_hub", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.ERROR)

    names = args.candidates.split(",")
    out_dir = REPO_BENCHMARK_DIR / VERSION / "bakeoff"
    if args.command == "select":
        record = select(names, out_dir / "selection.json", corpus_dir() / VERSION / "bakeoff")
        print(json.dumps({k: v["summary"] for k, v in record["candidates"].items()}, indent=2))
    else:
        print(json.dumps(bench(names, out_dir / "bench.json"), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
