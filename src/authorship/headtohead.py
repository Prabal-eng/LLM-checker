"""Increment 4d head-to-head: today's models against the 4d models on the v4 locked test, in one run.

    python -m src.authorship.headtohead [--rerun-reason "..."]

Both model sets run through the full service, each through its own registry record: today's models
with their format2 view, bf16 precision, bands and excerpt view; the 4d models with theirs. The
pre-registered pass criteria apply to each on the test split. The improvement rule compares AI
detection on the shared-ground slices, and the outcome table decides which model set is registered.
The run is recorded before any locked text is scored and happens once; a second run needs a reason.
"""

import argparse
import gc
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from ..corpus.lock import locked_hash, read as read_lock
from ..corpus.paths import REPO_BENCHMARK_DIR, VERSION, corpus_dir, lock_path
from ..corpus.schema import LOCKED_SPLITS
from . import benchmark, registry
from .benchmark import MIN_AIS

logger = logging.getLogger(__name__)

INCUMBENT_PATH = Path(__file__).resolve().parents[2] / "models" / "incumbent_registry.json"
INCUMBENT_MODEL_ID = "distilroberta-authorship-v5"


def load_incumbent(path: Optional[Path] = None) -> dict:
    """Today's registry record, frozen at the 4d branch point, refused unless its v3 seal still verifies:
    the head-to-head must score exactly the models, thresholds and routing that passed."""
    path = path or INCUMBENT_PATH
    if not path.exists():
        raise SystemExit(f"{path} is missing; freeze today's record first")
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("model_id") != INCUMBENT_MODEL_ID:
        raise SystemExit(f"the incumbent record must be {INCUMBENT_MODEL_ID}, not {record.get('model_id')}")
    benchmark = record.get("benchmark") or {}
    if not benchmark.get("passed") or benchmark.get("record_sha256") != registry.record_sha256(record, benchmark):
        raise SystemExit("the incumbent record no longer matches its v3 seal")
    return record


# Pre-registration "Improvement rule" with Amendment 2: rows neither side has the in-distribution edge on.
SHARED_FRESH_SOURCES = ("hc3", "mage", "mirage", "raid", "beemo", "no_robots", "aita2026", "dolly", "oasst1")
FRONTIER_SOURCES = ("wasitai2026", "lede")
IMPROVEMENT_POINTS = 5.0     # 4d must detect this many points more AI on the shared ground together
MAX_SLICE_LOSS_POINTS = 5.0  # and no gated shared-ground slice may lose more than this
GOAL_OVERALL, GOAL_SHORT, GOAL_FRONTIER = 0.75, 0.25, 0.70


# Unseen-generator rows count from these sources only: the rest are test portions of sources 4d trained on
# (maga, dactyl, arena140k, openturingbench, stackexchange), which share its training distribution
# (Amendment 4).
UNSEEN_SHARED_SOURCES = SHARED_FRESH_SOURCES + FRONTIER_SOURCES + ("stackexchange_heldout",)
DETECTOR_FAILURES = ("DETECTOR_ERROR", "DETECTOR_INVALID_RESPONSE", "DETECTOR_REPORTED_ERROR",
                     "DETECTOR_INVALID_EVIDENCE")


def shared_ground(decided: pd.DataFrame) -> pd.Series:
    """Test rows of the existing and frontier sources, and unseen-generator rows of the shared-ground sources.
    The held-out Stack Exchange sites' test rows come from the training generators, which 4d fitted on Stack
    Exchange, so only their held-out-generator answers count, through the unseen-generator split
    (Amendment 2)."""
    test = decided["split"] == "test"
    listed = decided["source_dataset"].isin(SHARED_FRESH_SOURCES + FRONTIER_SOURCES)
    unseen = (decided["split"] == "test_unseen_generator") & decided["source_dataset"].isin(UNSEEN_SHARED_SOURCES)
    return (test & listed) | unseen


def detection(part: pd.DataFrame) -> Optional[float]:
    """Share of AI rows decided Likely AI."""
    ai = part[part["y"] == 1]
    return round(float((ai["decision"] == "likely_ai_generated").mean()), 4) if len(ai) else None


def improvement(four_d: pd.DataFrame, incumbent: pd.DataFrame) -> dict:
    """The pre-registered rule on paired decisions: pooled shared-ground detection gains at least 5 points,
    and no gated shared-ground slice (a test-split source slice with MIN_AIS AI rows; the unseen-generator
    split is report-only in the pass criteria) loses more than 5."""
    if not four_d["sample_id"].reset_index(drop=True).equals(incumbent["sample_id"].reset_index(drop=True)):
        raise ValueError("the two models' decisions are not on the same rows in the same order")
    mask = shared_ground(four_d).to_numpy()
    mine, theirs = four_d[mask], incumbent[mask]
    pooled = {"4d": detection(mine), "incumbent": detection(theirs), "ais": int((mine["y"] == 1).sum())}
    gain = round(100 * ((pooled["4d"] or 0.0) - (pooled["incumbent"] or 0.0)), 2)
    slices, losing = {}, []
    for source, part in mine[mine["split"] == "test"].groupby("source_dataset"):
        other = theirs.loc[part.index]
        ais = int((part["y"] == 1).sum())
        entry = {"ais": ais, "4d": detection(part), "incumbent": detection(other), "gated": ais >= MIN_AIS}
        entry["change_points"] = (round(100 * (entry["4d"] - entry["incumbent"]), 2)
                                  if entry["4d"] is not None else None)
        if entry["gated"] and entry["change_points"] < -MAX_SLICE_LOSS_POINTS:
            losing.append(source)
        slices[source] = entry
    unseen = mine["split"] == "test_unseen_generator"
    excluded = (four_d["split"] == "test_unseen_generator").to_numpy() & ~mask
    slices["test_unseen_generator"] = {
        "ais": int((mine[unseen]["y"] == 1).sum()), "4d": detection(mine[unseen]),
        "incumbent": detection(theirs[unseen]), "gated": False,
        "by_source": {source: {"ais": int((part["y"] == 1).sum()), "4d": detection(part),
                               "incumbent": detection(theirs.loc[part.index])}
                      for source, part in mine[unseen].groupby("source_dataset")},
        # Reported by source, not counted (Amendment 4 item 1, Ralph R12-16).
        "excluded_by_source": {source: {"ais": int((part["y"] == 1).sum()), "4d": detection(part),
                                        "incumbent": detection(incumbent.loc[part.index])}
                               for source, part in four_d[excluded].groupby("source_dataset")}}
    # Report only (Ralph R12-15): LEDE is most of the pooled AI rows, so the pooled figure without it is shown too.
    rest = (mine["source_dataset"] != "lede").to_numpy()
    without = {"4d": detection(mine[rest]), "incumbent": detection(theirs[rest]), "ais": int((mine[rest]["y"] == 1).sum())}
    without["gain_points"] = round(100 * ((without["4d"] or 0.0) - (without["incumbent"] or 0.0)), 2)
    return {"pooled": pooled, "gain_points": gain, "slices": slices, "losing_slices": sorted(losing),
            "without_lede": without, "holds": bool(gain >= IMPROVEMENT_POINTS and not losing)}


def outcome(four_d_passed: bool, rule_holds: bool, incumbent_passed: bool) -> str:
    """The pre-registered outcome table, completed before the lock (Plan C Task 6 amendment): when 4d passes
    the gate but not the rule and today's models fail the gate, 4d is the only model set meeting the
    contract, so it is registered."""
    if four_d_passed and rule_holds:
        return "register_4d"
    if incumbent_passed:
        return "keep_incumbent"
    if four_d_passed:
        return "register_4d"
    return "close_gate"


def goals(decided: pd.DataFrame) -> dict:
    """Reported, not gating: AI detection overall, on 25-99-word texts, and on the frontier sets (test split)."""
    from ..detection.policy import DecisionPolicy

    test = decided[decided["split"] == "test"]
    words = test["text"].map(DecisionPolicy.word_count)
    parts = {"overall": (test, GOAL_OVERALL), "words_25_99": (test[(words >= 25) & (words < 100)], GOAL_SHORT),
             "frontier": (test[test["source_dataset"].isin(FRONTIER_SOURCES)], GOAL_FRONTIER)}
    return {name: {"detected": detection(part), "ais": int((part["y"] == 1).sum()), "goal": goal}
            for name, (part, goal) in parts.items()}


def report_file(version: str = VERSION) -> Path:
    return REPO_BENCHMARK_DIR / version / "model" / "head_to_head.json"


def run(rerun_reason: Optional[str] = None, registry_path: Optional[Path] = None,
        incumbent_path: Optional[Path] = None, report_path: Optional[Path] = None,
        lock_file: Optional[Path] = None, factories: Optional[dict] = None) -> dict:
    """factories ({"4d": ..., "incumbent": ...}, each returning a service) replace the registered services in
    tests. Each model set is loaded only for its own pass and released after it, so the two never share the
    4 GB GPU (final review I1)."""
    record = registry.load(registry_path)
    if record is None:
        raise SystemExit("no registry entry; calibrate the 4d models first")
    if record.get("corpus_version") != VERSION or record.get("model_id") == INCUMBENT_MODEL_ID:
        raise SystemExit(f"the registry holds {record.get('model_id')} ({record.get('corpus_version')}), not a "
                         f"{VERSION} 4d record; calibrate the 4d models first (Ralph R12-17)")
    incumbent = load_incumbent(incumbent_path)
    out = report_path or report_file()
    seen = [h for h in record.get("locked_benchmark_history") or [] if h.get("corpus_version", "v1") == VERSION]
    if (record.get("benchmark") or seen or out.exists()) and not rerun_reason:
        raise SystemExit("the locked head-to-head already ran; pass --rerun-reason to run it again")

    df = pd.read_parquet(corpus_dir() / VERSION / "corpus.parquet", columns=benchmark._COLUMNS,
                         filters=[("split", "in", [s.value for s in LOCKED_SPLITS])])
    lock = read_lock(lock_file or lock_path())
    if lock is None or locked_hash(df) != lock["sha256"]:
        raise SystemExit("locked splits do not match the committed lock; refusing to run")
    df["y"] = (df["label"] == "ai").astype(int)
    injected = factories is not None
    if not injected:
        # The 4d record must meet everything verify() will ask of it once sealed (Ralph R12-6); today's record
        # is held to its v3 seal and its files, since its scoring-code digest predates the 4d branch.
        for name, rec, check in (("4d", record, registry.verify_calibrated), ("incumbent", incumbent, registry.verify_files)):
            ok, why = check(rec)
            if not ok:
                raise SystemExit(f"the {name} record does not verify ({why}); refusing to run")
        # Both are candidates under test here; the outcome decides what the registry opens for.
        factories = {"4d": lambda: registry.service_from_record(record, validated=True),
                     "incumbent": lambda: registry.service_from_record(incumbent, validated=True)}
    diag = benchmark.read_diagnostic()  # read before the run is recorded (Ralph R13-7)
    # Recorded before any locked text is scored, so an interrupted run still counts as a run.
    entry = {"started_at": datetime.now(timezone.utc).isoformat(), "status": "started",
             "model_id": record["model_id"], "short_model_id": (record.get("short_model") or {}).get("model_id"),
             "incumbent_model_id": incumbent["model_id"], "head_to_head": True, "rerun_reason": rerun_reason,
             "corpus_version": VERSION, "lock_sha256": lock["sha256"]}
    record.setdefault("locked_benchmark_history", []).append(entry)
    registry.save(record, registry_path)

    torch = None
    if not injected:  # tests never touch CUDA (final review I2)
        try:
            import torch
        except ImportError:
            torch = None
    probe = torch is not None and torch.cuda.is_available()
    models, decided = {}, {}
    for name, rec in (("4d", record), ("incumbent", incumbent)):
        logger.info("scoring the locked splits with %s (%s)", name, rec["model_id"])
        if probe:
            torch.cuda.reset_peak_memory_stats()
        service = factories[name]()
        # Injected services stand in for their own main-only excerpt service; a real routed service lends its
        # loaded main model to one (benchmark.main_only_service).
        frame, sections = benchmark.model_report(rec, service, df, main_only=service if injected else None, diag=diag)
        errors = frame["reason"].isin(DETECTOR_FAILURES)
        if errors.any():
            # A failed detector call turns its whole chunk Inconclusive; scored on, the run would seal a gate
            # computed on missing decisions (Ralph R12-2). It stays recorded, unsealed: a rerun needs a reason.
            why = f"{name}: {int(errors.sum())} detector error(s) ({dict(frame.loc[errors, 'reason'].value_counts())})"
            entry.update({"status": "detector_error", "failures": [why],
                          "stopped_at": datetime.now(timezone.utc).isoformat()})
            registry.save(record, registry_path)
            raise SystemExit(f"{why} while scoring the locked splits; the run is recorded unsealed")
        passed, failures = benchmark.gate(frame[frame["split"] == "test"])
        models[name] = {"model_id": rec["model_id"], "short_model_id": (rec.get("short_model") or {}).get("model_id"),
                        "text_normalization": rec.get("text_normalization"), "passed": passed,
                        "failures": failures, "goals": goals(frame),
                        "peak_inference_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3) if probe else None,
                        **sections}
        decided[name] = frame
        del service
        gc.collect()
        if probe:
            torch.cuda.empty_cache()
    rule = improvement(decided["4d"], decided["incumbent"])
    result = outcome(models["4d"]["passed"], rule["holds"], models["incumbent"]["passed"])
    report = {"criteria": {"max_false_ai_upper95": benchmark.MAX_FALSE_AI,
                           "max_false_human_upper95": benchmark.MAX_FALSE_HUMAN,
                           "min_coverage_supported": benchmark.MIN_COVERAGE, "min_humans": benchmark.MIN_HUMANS,
                           "min_ais": benchmark.MIN_AIS, "improvement_points": IMPROVEMENT_POINTS,
                           "max_slice_loss_points": MAX_SLICE_LOSS_POINTS},
              "models": models, "improvement_rule": rule, "outcome": result,
              "ran_at": datetime.now(timezone.utc).isoformat()}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")

    registered = result == "register_4d"
    failures = list(models["4d"]["failures"])
    if not registered:
        failures.append(f"outcome {result}: improvement rule {'held' if rule['holds'] else 'failed'} "
                        f"(gain {rule['gain_points']} points, losing {rule['losing_slices']}), today's models "
                        f"{'passed' if models['incumbent']['passed'] else 'failed'}")
    entry.update({"status": "finished", "ran_at": report["ran_at"], "passed": registered, "failures": failures,
                  "outcome": result})
    history = list((record.get("benchmark") or {}).get("history", []))
    history.append({"ran_at": report["ran_at"], "passed": registered, "rerun_reason": rerun_reason,
                    "model_id": record["model_id"], "outcome": result})
    record["benchmark"] = registry.seal(record, passed=registered, ran_at=report["ran_at"], failures=failures,
                                        report=f"data/benchmark/{VERSION}/model/head_to_head.json",
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
    print(json.dumps({"outcome": report["outcome"], "improvement_rule": {k: report["improvement_rule"][k] for k in
                                                                         ("pooled", "gain_points", "losing_slices",
                                                                          "holds")},
                      "models": {k: {"passed": v["passed"], "failures": v["failures"], "goals": v["goals"]}
                                 for k, v in report["models"].items()}}, indent=2, default=str))
    return 0 if report["outcome"] == "register_4d" else 1


if __name__ == "__main__":
    sys.exit(main())
