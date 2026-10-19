"""One-shot locked benchmark of the registered model; opens the validation gate only on a pass.

    python -m src.authorship.benchmark [--rerun-reason "..."]

Runs the real service (language gate, OOD gate, length bands, flag-only rater context) over the
locked splits. The pass criteria below are fixed before the first run:

- test split, each source slice with >= 300 human texts: false-AI 95% upper bound <= 1%
- test split, each source slice with >= 100 AI texts: false-human 95% upper bound <= 10%
- test split: decisive coverage >= 40% of supported text (English, long enough, in a
  calibrated band, in distribution)

Reported but not gating: the unseen-generator split, stress sets (FR12), and the in-house rater
texts, whose "human" labels are unverified (a flag there may be real AI). In-house texts whose
model outputs were viewed before this run are reported apart from the rest.

The benchmark runs once. A second run requires --rerun-reason, which is recorded.
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from ..bakeoff.metrics import upper_bound
from ..corpus.lock import locked_hash, read as read_lock
from ..corpus.paths import REPO_BENCHMARK_DIR, VERSION, corpus_dir, lock_path
from ..corpus.schema import LOCKED_SPLITS
from ..corpus.text import MODEL_TEXT_VIEW
from ..detection import TextContext
from ..detection.policy import DecisionPolicy
from . import registry
from .calibration import BANDS, band_of

logger = logging.getLogger(__name__)

MAX_FALSE_AI, MAX_FALSE_HUMAN, MIN_COVERAGE = 0.01, 0.10, 0.40
MIN_HUMANS, MIN_AIS = 300, 100
UNSUPPORTED = {"INSUFFICIENT_TEXT", "UNSUPPORTED_LANGUAGE", "UNSUPPORTED_DOMAIN", "UNSUPPORTED_LENGTH",
               "UNSUPPORTED_INPUT_TYPE", "UNSUPPORTED_INPUT_LENGTH"}
PSEUDO_ROW_SIZE = 6
CHUNK = 4000
_COLUMNS = ["sample_id", "text", "label", "split", "source_dataset", "domain", "generator_family",
            "transform", "writer_group", "group_key", "source_record_id"]


def benchmark_file(version: str = VERSION) -> Path:
    return REPO_BENCHMARK_DIR / version / "model" / "benchmark.json"


def previewed_ids() -> set:
    """In-house texts whose model outputs were viewed before v1's locked run (the ids do not change)."""
    path = REPO_BENCHMARK_DIR / "v1" / "model" / "inhouse_previewed_ids.json"
    return set(json.loads(path.read_text(encoding="utf-8"))["sample_ids"]) if path.exists() else set()


def by_band(part: pd.DataFrame) -> dict:
    bands = part["text"].map(DecisionPolicy.word_count).map(lambda w: band_of(w) if w >= BANDS[0] else 0)
    return {("<25" if b == 0 else str(b)): slice_stats(p) for b, p in part.groupby(bands)}


def excerpt_report(test: pd.DataFrame, services: dict, view: Optional[str] = MODEL_TEXT_VIEW) -> dict:
    """Report-only short-text falloff: sentence-aligned excerpts of test texts under 100 words, cut from the
    scored record's own view (a model is read the way it was trained; final review I2)."""
    from .calibrate import with_excerpts

    ex = with_excerpts(test[["sample_id", "text", "y", "source_dataset"]], view=view)
    out = {"excerpts": len(ex)}
    for name, service in services.items():
        decided = pd.concat([ex.reset_index(drop=True), decide_all(service, ex["text"].tolist(), TextContext.GENERAL)],
                            axis=1)
        out[name] = {"overall_by_band": by_band(decided),
                     "by_source_band": {s: by_band(p) for s, p in decided.groupby("source_dataset")}}
    return out


def decide_all(service, texts: list, context: TextContext) -> pd.DataFrame:
    rows = []
    for start in range(0, len(texts), CHUNK):
        for r in service.check_batch(texts[start:start + CHUNK], context):
            rows.append({"decision": r.decision.value, "reason": r.reason_codes[0] if r.reason_codes else "",
                         "score": r.score})
        logger.info("decided %d/%d", min(start + CHUNK, len(texts)), len(texts))
    return pd.DataFrame(rows)


def slice_stats(part: pd.DataFrame) -> dict:
    h, a = part[part["y"] == 0], part[part["y"] == 1]
    false_ai = int((h["decision"] == "likely_ai_generated").sum())
    false_human = int((a["decision"] == "likely_human_written").sum())
    supported = part[~part["reason"].isin(UNSUPPORTED)]
    return {
        "humans": len(h), "ais": len(a),
        "false_ai": false_ai, "false_ai_upper95": upper_bound(false_ai, len(h)),
        "false_human": false_human, "false_human_upper95": upper_bound(false_human, len(a)),
        "detected_ai": int((a["decision"] == "likely_ai_generated").sum()),
        "supported": len(supported),
        "coverage_supported": round(float((supported["decision"] != "inconclusive").mean()), 4) if len(supported) else None,
        "reasons": part["reason"].value_counts().to_dict(),
    }


def gate(test: pd.DataFrame) -> tuple[bool, list]:
    failures = []
    for source, part in test.groupby("source_dataset"):
        s = slice_stats(part)
        if s["humans"] >= MIN_HUMANS and s["false_ai_upper95"] > MAX_FALSE_AI:
            failures.append(f"{source}: false-AI upper bound {s['false_ai_upper95']:.4f} > {MAX_FALSE_AI}")
        if s["ais"] >= MIN_AIS and s["false_human_upper95"] > MAX_FALSE_HUMAN:
            failures.append(f"{source}: false-human upper bound {s['false_human_upper95']:.4f} > {MAX_FALSE_HUMAN}")
    coverage = slice_stats(test)["coverage_supported"]
    if coverage is None or coverage < MIN_COVERAGE:
        failures.append(f"coverage {coverage} < {MIN_COVERAGE}")
    return not failures, failures


def pseudo_rows(inhouse: pd.DataFrame) -> pd.DataFrame:
    """Join each rater's texts in groups of six, approximating a spreadsheet row."""
    out = []
    for rater, part in inhouse.sort_values("source_record_id").groupby("group_key"):
        texts, ids = part["text"].tolist(), part["sample_id"].tolist()
        for i in range(0, len(texts), PSEUDO_ROW_SIZE):
            out.append({"text": "\n\n".join(texts[i:i + PSEUDO_ROW_SIZE]), "ids": ids[i:i + PSEUDO_ROW_SIZE]})
    return pd.DataFrame(out)


def main_only_service(record: dict, routed):
    """The routed service's main model alone, under the main model's own bands, for the excerpt comparison.
    It reuses the loaded main model rather than loading a second copy onto the GPU (Plan C final review I1)."""
    from ..detection import AuthorshipService

    main = getattr(routed.detector, "main", None)
    plain = registry.main_only(record)
    if main is None:
        return registry.service_from_record(plain, validated=True)
    return AuthorshipService(main, policy=registry.build_policy(plain), model_validated=True,
                             detector_version=plain["model_id"], require_english=True)


def read_diagnostic() -> pd.DataFrame:
    """The rasbt50k diagnostic rows (outside the lock), read before a locked run is recorded (Ralph R13-7)."""
    diag = pd.read_parquet(corpus_dir() / VERSION / "corpus.parquet", columns=_COLUMNS,
                           filters=[("split", "==", "diagnostic"), ("source_dataset", "==", "rasbt50k")])
    return diag.reset_index(drop=True).assign(y=lambda d: (d["label"] == "ai").astype(int))


def model_report(record: dict, service, df: pd.DataFrame, main_only=None,
                 diag: Optional[pd.DataFrame] = None) -> tuple[pd.DataFrame, dict]:
    """Decisions on the general locked rows and one model's report sections: the test split by source,
    band and generator family; short excerpts cut from the model's own view; unseen generators; stress;
    the rasbt50k diagnostic; and unverified in-house flags. df holds the locked rows with y. main_only is
    the main model alone for the excerpt comparison; it is built from the record when not given."""
    general = df[df["split"] != "inhouse"].reset_index(drop=True)
    decided = pd.concat([general, decide_all(service, general["text"].tolist(), TextContext.GENERAL)], axis=1)
    test = decided[decided["split"] == "test"]
    report: dict = {"test": {"overall": slice_stats(test),
                             "by_source": {k: slice_stats(v) for k, v in test.groupby("source_dataset")},
                             "by_band": by_band(test),
                             "by_generator_family": {k: slice_stats(v)
                                                     for k, v in test[test["y"] == 1].groupby("generator_family")},
                             # Report only: the gate holds the held-out sites as one slice (Ralph R12-5).
                             "stackexchange_heldout_by_site": {
                                 k: slice_stats(v)
                                 for k, v in test[test["source_dataset"] == "stackexchange_heldout"].groupby("domain")}}}
    services = {"routed" if record.get("short_model") else "main": service}
    if record.get("short_model"):
        services["main_only"] = main_only or main_only_service(record, service)
    report["short_text_excerpts"] = excerpt_report(test, services, view=record.get("text_normalization"))
    unseen = decided[decided["split"] == "test_unseen_generator"]
    report["test_unseen_generator"] = {"overall": slice_stats(unseen),
                                       "by_family": {k: slice_stats(v) for k, v in unseen.groupby("generator_family")}}
    stress = decided[decided["split"] == "stress"].fillna({"transform": "none", "writer_group": "unknown"})
    report["stress"] = {f"{k[0]}|{k[1]}|{k[2]}": slice_stats(v)
                        for k, v in stress.groupby(["source_dataset", "transform", "writer_group"])}

    # Diagnostic (outside the lock): current-model texts from a source too artifact-laden to fit on.
    diag = read_diagnostic() if diag is None else diag
    if len(diag):
        diag = pd.concat([diag.reset_index(drop=True), decide_all(service, diag["text"].tolist(), TextContext.GENERAL)],
                         axis=1)
        report["diagnostic_rasbt50k"] = {
            "note": "Report only. Human rows are mid-sentence chunks with code stripped, so their flags are not "
                    "comparable to other sources; AI rows are 2026-model answers.",
            "ai_by_generator": {k: slice_stats(v) for k, v in diag[diag["y"] == 1].groupby("generator_family")},
            "human_by_domain": {k: slice_stats(v) for k, v in diag[diag["y"] == 0].groupby("domain")},
        }

    inhouse = df[df["split"] == "inhouse"].reset_index(drop=True)
    seen = previewed_ids()
    cells = pd.concat([inhouse, decide_all(service, inhouse["text"].tolist(), TextContext.RATER_COMMENT)], axis=1)
    rows = pseudo_rows(inhouse)
    rows = pd.concat([rows, decide_all(service, rows["text"].tolist(), TextContext.RATER_COMMENT)], axis=1)
    rows["previewed"] = rows["ids"].map(lambda ids: any(i in seen for i in ids))

    def flags(part: pd.DataFrame) -> dict:
        n = len(part)
        flagged = int((part["decision"] == "likely_ai_generated").sum())
        return {"texts": n, "flagged": flagged, "flag_rate": round(flagged / n, 4) if n else None,
                "reasons": part["reason"].value_counts().to_dict()}

    report["inhouse_unverified"] = {
        "note": "Report only. Labels are presumed human but unverified; a flag may be real AI use.",
        "cells_never_viewed": flags(cells[~cells["sample_id"].isin(seen)]),
        "cells_previewed": flags(cells[cells["sample_id"].isin(seen)]),
        "pseudo_rows_never_viewed": flags(rows[~rows["previewed"]]),
        "pseudo_rows_previewed": flags(rows[rows["previewed"]]),
    }
    return decided, report


def run(rerun_reason: Optional[str] = None, registry_path: Optional[Path] = None,
        report_path: Optional[Path] = None) -> dict:
    record = registry.load(registry_path)
    if record is None:
        raise SystemExit("no registry entry; run python -m src.authorship.calibrate first")
    out = report_path or benchmark_file()
    history = list((record.get("benchmark") or {}).get("history", []))
    # A locked test is "seen" once any run on it started (finished or not), or once a report for
    # this version exists; runs recorded before versions were tracked were v1 (Ralph R07-9).
    seen = [h for h in record.get("locked_benchmark_history") or [] if h.get("corpus_version", "v1") == VERSION]
    if (record.get("benchmark") or seen or out.exists()) and not rerun_reason:
        raise SystemExit("the locked benchmark already ran; pass --rerun-reason to run it again")

    df = pd.read_parquet(corpus_dir() / VERSION / "corpus.parquet", columns=_COLUMNS,
                         filters=[("split", "in", [s.value for s in LOCKED_SPLITS])])
    lock = read_lock(lock_path())
    if lock is None or locked_hash(df) != lock["sha256"]:
        raise SystemExit("locked splits do not match the committed lock; refusing to benchmark")
    df["y"] = (df["label"] == "ai").astype(int)

    ok, why = registry.verify_files(record)
    if not ok:
        # The run must score exactly the files the gate would later open for (Ralph R07-4).
        raise SystemExit(f"registered files do not verify ({why}); refusing to benchmark")
    service = registry.build_service(registry_path)
    if service is None:
        raise SystemExit("registered model files are missing")
    service.model_validated = True  # the candidate under test; the registry decides afterwards
    # Recorded before any locked text is scored, so an interrupted run still counts as a run.
    entry = {"started_at": datetime.now(timezone.utc).isoformat(), "status": "started",
             "model_id": record["model_id"], "rerun_reason": rerun_reason, "corpus_version": VERSION,
             "lock_sha256": lock["sha256"], "short_model_id": (record.get("short_model") or {}).get("model_id")}
    record.setdefault("locked_benchmark_history", []).append(entry)
    registry.save(record, registry_path)

    import torch

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    report: dict = {"model_id": record["model_id"], "policy_version": record["policy_version"],
                    "criteria": {"max_false_ai_upper95": MAX_FALSE_AI, "max_false_human_upper95": MAX_FALSE_HUMAN,
                                 "min_coverage_supported": MIN_COVERAGE, "min_humans": MIN_HUMANS, "min_ais": MIN_AIS}}

    decided, sections = model_report(record, service, df)
    report.update(sections)
    passed, failures = gate(decided[decided["split"] == "test"])
    report["peak_inference_vram_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 3) if torch.cuda.is_available() else None
    report["passed"], report["failures"] = passed, failures
    report["ran_at"] = datetime.now(timezone.utc).isoformat()

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    history.append({"ran_at": report["ran_at"], "passed": passed, "rerun_reason": rerun_reason,
                    "model_id": record["model_id"]})
    entry.update({"status": "finished", "ran_at": report["ran_at"], "passed": passed, "failures": failures})
    record["benchmark"] = registry.seal(record, passed=passed, ran_at=report["ran_at"], failures=failures,
                                        report=f"data/benchmark/{VERSION}/model/benchmark.json",
                                        report_sha256=registry.text_sha256(out), history=history)
    registry.save(record, registry_path)
    return report


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rerun-reason")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for noisy in ("transformers", "httpx", "huggingface_hub", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.ERROR)
    report = run(args.rerun_reason)
    print(json.dumps({"passed": report["passed"], "failures": report["failures"],
                      "test_overall": report["test"]["overall"], "inhouse": report["inhouse_unverified"]},
                     indent=2, default=str))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
