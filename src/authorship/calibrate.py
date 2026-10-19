"""Calibrate the trained model on validation and register it (still unvalidated).

    python -m src.authorship.calibrate

Scores every validation text once, fits the probability map, sets per-band thresholds that
bound every source slice, builds the out-of-distribution bank from training texts, and compares
shared with per-source thresholds (U4). Validation includes sentence-aligned excerpts of longer
texts, so the bands under 100 words are calibrated on enough text. With a short-text specialist
trained, it is calibrated on the same rows and serves the bands under 100 words it wins. Writes
the registry entry with benchmark = null, so the service keeps abstaining until
`python -m src.authorship.benchmark` passes. Only train and validation rows are read.
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ..corpus.paths import REPO_BENCHMARK_DIR, VERSION, corpus_dir
from ..corpus.text import MODEL_TEXT_VIEW, text_view_fingerprint
from ..corpus.sources._common import take_per_stratum
from ..detection.policy import DecisionPolicy
from . import calibration as cal
from . import registry
from .detector import AuthorshipModelDetector, OodGate
from .train import record_file as training_record_file

logger = logging.getLogger(__name__)

BANK_PER_STRATUM = 2000          # training texts per (source, label) in the OOD bank
BANK_EXCERPTS_PER_STRATUM = 700  # plus excerpts, so short in-distribution text has near neighbours
OOD_K = 10
OOD_KEEP = 0.99                  # share of validation text the gate must accept
SHORT_UPPER = 100                # the specialist may serve only bands below this many words
ROUTING_MARGIN = 0.01            # pre-registered: the specialist must win coverage by a point
_COLUMNS = ["sample_id", "text", "label", "split", "source_dataset", "domain", "generator_family"]


def calibration_file(version: str = VERSION, candidate: str = "distilroberta-base") -> Path:
    return REPO_BENCHMARK_DIR / version / "model" / candidate / "calibration.json"


def load_split(split: str) -> pd.DataFrame:
    if split not in ("train", "val"):
        raise ValueError("calibration reads only train and validation rows")
    df = pd.read_parquet(corpus_dir() / VERSION / "corpus.parquet", columns=_COLUMNS,
                         filters=[("split", "==", split)])
    if set(df["split"]) != {split}:
        raise RuntimeError("calibration read an unexpected split")
    df["y"] = (df["label"] == "ai").astype(int)
    df["words"] = df["text"].map(DecisionPolicy.word_count)
    return df.reset_index(drop=True)


def per_source_comparison(val: pd.DataFrame, shared: list, **targets) -> dict:
    """U4: coverage under shared thresholds versus thresholds fitted to each source alone."""
    shared_report = cal.evaluate(val, shared)
    out = {}
    for source, part in val.groupby("source_dataset"):
        own = cal.choose_thresholds(part, **targets)
        own_report = cal.evaluate(part, own)[source]
        out[source] = {"shared_coverage": shared_report[source]["coverage_supported"],
                       "own_coverage": own_report["coverage_supported"],
                       "shared_false_ai_upper95": shared_report[source]["false_ai_upper95"],
                       "own_false_ai_upper95": own_report["false_ai_upper95"]}
    return out


def with_excerpts(df: pd.DataFrame, view: Optional[str] = MODEL_TEXT_VIEW) -> pd.DataFrame:
    """Sentence-aligned excerpts of df's longer texts that fall in the bands under 100 words, cut from view."""
    from .excerpts import short_frame

    ex = short_frame(df.drop(columns=["words"], errors="ignore"), view=view)
    ex = ex[ex["excerpt"]].copy()
    ex["words"] = ex["text"].map(DecisionPolicy.word_count)
    return ex[(ex["words"] >= cal.BANDS[0]) & (ex["words"] < SHORT_UPPER)].reset_index(drop=True)


def short_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df[(df["words"] >= cal.BANDS[0]) & (df["words"] < SHORT_UPPER)].reset_index(drop=True)


def band_coverage(df: pd.DataFrame, bands: list, *, score: str = "prob") -> dict:
    """Decisive share of the rows in each band (0 for an unsupported band)."""
    by_band = {b.lower_words: b for b in bands}
    out = {}
    for lower, part in df.assign(band=df["words"].map(cal.band_of)).groupby("band"):
        b = by_band.get(lower)
        if b is None or not b.supported:
            out[int(lower)] = 0.0
            continue
        decisive = (part[score] >= b.ai_min) | (part[score] <= b.human_max)
        out[int(lower)] = round(float(decisive.mean()), 4)
    return out


def route(main_coverage: dict, short_coverage: dict, short_bands: list, margin: float = ROUTING_MARGIN) -> list:
    """Bands under 100 words the specialist serves: supported, and ahead of the main model by margin."""
    supported = {b.lower_words for b in short_bands if b.supported}
    return sorted(lower for lower in supported
                  if lower < SHORT_UPPER and short_coverage.get(lower, 0.0) >= main_coverage.get(lower, 0.0) + margin)


def combine_bands(main_bands: list, short_bands: list, routed: list) -> list:
    """Policy bands: the specialist's thresholds for routed bands, the main model's elsewhere."""
    short_by_band = {b["lower_words"]: b for b in cal.as_dicts(short_bands)}
    return [dict(short_by_band[b["lower_words"]], model="short") if b["lower_words"] in routed
            else dict(b, model="main") for b in cal.as_dicts(main_bands)]


def _score(detector, artifact: Path, sha: str, sets: dict) -> dict:
    """Raw scores and embeddings per named row set, cached beside the artifact by row ids."""
    cache = artifact.parent / f"calibration_scores_{sha[:12]}.npz"
    ids = {name: rows["sample_id"].to_numpy(dtype=str) for name, rows in sets.items()}
    saved = dict(np.load(cache)) if cache.exists() else {}
    if saved and all(np.array_equal(saved.get(f"{n}_ids", np.array([])), i) for n, i in ids.items()):
        logger.info("reusing cached scores for %s", artifact.parent.name)
        return saved
    import torch

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    out = {}
    for name, rows in sets.items():
        logger.info("%s: scoring %d %s texts", artifact.parent.name, len(rows), name)
        out[f"{name}_raw"], out[f"{name}_emb"] = detector.raw_and_embeddings(rows["text"].tolist())
        out[f"{name}_ids"] = ids[name]
    peak = round(torch.cuda.max_memory_allocated() / 1e9, 3) if torch.cuda.is_available() else float("nan")
    out["peak_vram_gb"] = np.array(peak)
    np.savez(cache, **out)
    return out


def _fit(val: pd.DataFrame, raw: np.ndarray, val_emb: np.ndarray, bank: np.ndarray,
         max_false_ai: float, max_false_human: float) -> tuple:
    a, b = cal.fit_platt(raw, val["y"].to_numpy())
    val = val.assign(raw=raw, prob=cal.probability(raw, a, b))
    distances = OodGate(bank, OOD_K, None).distances(val_emb)
    threshold = float(np.quantile(distances, OOD_KEEP))
    val = val.assign(ood_distance=distances, in_distribution=distances <= threshold)
    bands = cal.choose_thresholds(val, max_fpr=max_false_ai, max_fnr=max_false_human)
    return val, (a, b), threshold, bands


def _model_report(val: pd.DataFrame, platt, threshold: float, bands: list, bank_rows: int) -> dict:
    inside = val[val["in_distribution"]]
    return {
        "validation_rows": len(val), "validation_excerpts": int(val["excerpt"].sum()),
        "platt": list(platt), "bands": cal.as_dicts(bands),
        "slices_all": cal.evaluate(val, bands), "slices_in_distribution": cal.evaluate(inside, bands),
        "band_coverage": band_coverage(val, bands),
        "ood": {"k": OOD_K, "threshold": threshold, "bank_rows": bank_rows,
                "validation_accept_rate": round(float(val["in_distribution"].mean()), 4),
                "accept_rate_by_source": val.groupby("source_dataset")["in_distribution"].mean().round(4).to_dict()},
    }


def _detector(training: dict) -> AuthorshipModelDetector:
    return AuthorshipModelDetector(Path(training["artifact_path"]), platt=(1.0, 0.0),
                                   max_tokens=training.get("max_tokens") or 256,
                                   text_view=training.get("text_normalization"),
                                   precision=training.get("precision", registry.LEGACY_PRECISION),
                                   half_weights="precision" in training)


def calibrate(record_path: Optional[Path] = None, registry_path: Optional[Path] = None,
              max_false_ai: float = 0.008, max_false_human: float = 0.08,
              short_record_path: Optional[Path] = None, candidate: str = "distilroberta-base") -> dict:
    """max_false_ai and max_false_human are calibration targets; setting them below the release
    bounds (1% and 10%) leaves a margin for the locked benchmark."""
    training = json.loads((record_path or training_record_file(candidate=candidate)).read_text(encoding="utf-8"))
    short_file = short_record_path or training_record_file(variant="short", candidate=candidate)
    short_training = json.loads(short_file.read_text(encoding="utf-8")) if short_file.exists() else None
    artifact = Path(training["artifact_path"])
    root = corpus_dir()

    val, train = load_split("val"), load_split("train")
    val = pd.concat([val.assign(excerpt=False), with_excerpts(val)], ignore_index=True)
    natural_bank = take_per_stratum(train, ["source_dataset", "y"], BANK_PER_STRATUM, "sample_id")
    excerpt_bank = with_excerpts(take_per_stratum(train, ["source_dataset", "y"], BANK_EXCERPTS_PER_STRATUM,
                                                  "sample_id"))
    bank_rows = pd.concat([natural_bank, excerpt_bank], ignore_index=True)

    scores = _score(_detector(training), artifact, training["artifact_sha256"], {"val": val, "bank": bank_rows})
    main_val, platt, threshold, bands = _fit(val, scores["val_raw"], scores["val_emb"], scores["bank_emb"],
                                             max_false_ai, max_false_human)
    report = {"targets": {"max_false_ai_upper95": max_false_ai, "max_false_human_upper95": max_false_human},
              "main": _model_report(main_val, platt, threshold, bands, len(bank_rows)),
              "u4_shared_vs_per_source": per_source_comparison(main_val, bands, max_fpr=max_false_ai,
                                                               max_fnr=max_false_human),
              "peak_inference_vram_gb": float(scores["peak_vram_gb"])}

    combined, short_entry = combine_bands(bands, [], []), None
    if short_training is not None:
        short_artifact = Path(short_training["artifact_path"])
        short_val = short_rows(val)
        short_bank_rows = pd.concat([short_rows(natural_bank), with_excerpts(natural_bank)], ignore_index=True)
        short_scores = _score(_detector(short_training), short_artifact, short_training["artifact_sha256"],
                              {"val": short_val, "bank": short_bank_rows})
        s_val, s_platt, s_threshold, s_bands = _fit(short_val, short_scores["val_raw"], short_scores["val_emb"],
                                                    short_scores["bank_emb"], max_false_ai, max_false_human)
        main_cov, short_cov = band_coverage(short_rows(main_val), bands), band_coverage(s_val, s_bands)
        routed = route(main_cov, short_cov, s_bands)
        combined = combine_bands(bands, s_bands, routed)
        report["short"] = _model_report(s_val, s_platt, s_threshold, s_bands, len(short_bank_rows))
        report["routing"] = {"rule": f"the specialist serves a band under {SHORT_UPPER} words when its decisive "
                                     f"coverage beats the main model's by {ROUTING_MARGIN}",
                             "main_coverage": main_cov, "short_coverage": short_cov, "short_bands": routed}
        if routed:
            short_bank_file = short_artifact.parent / "ood_bank.npy"
            np.save(short_bank_file, short_scores["bank_emb"].astype(np.float32))
            short_entry = {
                "model_id": short_training["model_id"], "max_tokens": short_training.get("max_tokens") or 160,
                "precision": short_training.get("precision", registry.LEGACY_PRECISION),
                "platt": list(s_platt),
                "ranges": [[b["lower_words"], b["upper_words"]] for b in combined if b["model"] == "short"],
                "artifact": {"relative_path": short_artifact.relative_to(root).as_posix(),
                             "sha256": short_training["artifact_sha256"],
                             "files_sha256": registry.directory_sha256(short_artifact)},
                "ood": {"bank_relative_path": short_bank_file.relative_to(root).as_posix(),
                        "bank_sha256": registry.sha256_file(short_bank_file), "k": OOD_K, "threshold": s_threshold},
                "training_record": f"data/benchmark/{VERSION}/model/{candidate}/training_record_short.json",
            }
    report["policy_bands"] = combined
    report["locked_split_reads"] = 0
    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    out = calibration_file(candidate=candidate)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True, default=float) + "\n", encoding="utf-8")

    # The locked splits stay "seen" across models: keep every earlier benchmark run on record.
    previous = registry.load(registry_path) or {}
    history = list(previous.get("locked_benchmark_history", []))
    if previous.get("benchmark"):
        run = {k: previous["benchmark"].get(k) for k in ("ran_at", "passed", "failures")}
        run["model_id"] = previous.get("model_id")
        run["corpus_version"] = previous.get("corpus_version", "v1")
        if not any(h.get("ran_at") == run["ran_at"] for h in history):
            history.append(run)
    bank_file = artifact.parent / "ood_bank.npy"
    np.save(bank_file, scores["bank_emb"].astype(np.float32))
    entry = {
        "model_id": training["model_id"],
        "policy_version": registry.POLICY_VERSION,
        "corpus_version": VERSION,
        "max_tokens": training.get("max_tokens") or 256,
        "precision": training.get("precision", registry.LEGACY_PRECISION),
        "text_normalization": training.get("text_normalization"),
        "text_view_sha256": text_view_fingerprint(training.get("text_normalization")),
        "scoring_code_sha256": registry.scoring_code_sha256(),
        "artifact": {"relative_path": artifact.relative_to(root).as_posix(), "sha256": training["artifact_sha256"],
                     "files_sha256": registry.directory_sha256(artifact)},
        "calibration": {
            "platt": list(platt),
            "bands": combined,
            "main_bands": combine_bands(bands, [], []),
            "ood": {"bank_relative_path": bank_file.relative_to(root).as_posix(),
                    "bank_sha256": registry.sha256_file(bank_file), "k": OOD_K, "threshold": threshold},
            "report": f"data/benchmark/{VERSION}/model/{candidate}/calibration.json",
            "targets": {"max_false_ai_upper95": max_false_ai, "max_false_human_upper95": max_false_human},
        },
        "training_record": f"data/benchmark/{VERSION}/model/{candidate}/training_record.json",
        "benchmark": None,
        "locked_benchmark_history": history,
    }
    if short_entry:
        entry["short_model"] = short_entry
    registry.save(entry, registry_path)
    return report


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--max-false-ai", type=float, default=0.008)  # the pre-registered targets (Ralph R12-18)
    parser.add_argument("--max-false-human", type=float, default=0.08)
    parser.add_argument("--candidate", required=True)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for noisy in ("transformers", "httpx", "huggingface_hub", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.ERROR)
    report = calibrate(max_false_ai=args.max_false_ai, max_false_human=args.max_false_human,
                       candidate=args.candidate)
    print(json.dumps({k: report.get(k) for k in ("policy_bands", "routing", "peak_inference_vram_gb")},
                     indent=2, default=float))
    print(json.dumps({m: {"band_coverage": report[m]["band_coverage"], "ood": report[m]["ood"]}
                      for m in ("main", "short") if m in report}, indent=2, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
