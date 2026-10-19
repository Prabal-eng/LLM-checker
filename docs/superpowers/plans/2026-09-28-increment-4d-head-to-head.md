# Increment 4d Head-to-Head Implementation Plan (Plan C)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Score today's models and the 4d models on the v4 locked test in one run, apply the pre-registered pass criteria, improvement rule and outcome table, check the speed and memory limits, and seal the result in the registry.

**Architecture:** Today's registry record is frozen on the 4d branch (`models/incumbent_registry.json`) and loaded only if its v3 seal still verifies. The per-model report sections move out of `benchmark.run` into `benchmark.model_report`, so the single-model benchmark and the new head-to-head (`src/authorship/headtohead.py`) build them the same way. The head-to-head scores both services over the same locked rows (each through its own record: its view, precision, bands and excerpt view), computes each gate, the shared-ground improvement rule and the outcome, writes one report, and seals the 4d record with `passed` true only when the outcome registers 4d. `src/authorship/speed.py` measures the system under test against the GPU and CPU limits.

**Tech Stack:** Python 3.12, pandas, numpy, PyTorch (only at run time), pytest.

**Spec:** `docs/superpowers/specs/2026-09-28-increment-4d-detection-design.md` (sections "The v4 locked test", "Shared-ground slices", "Process" steps 8–9, "Error handling"), bound by `docs/bob/llm-authorship-checker/increment-4d-preregistration.md` (sections "System under test", "Pass criteria", "Improvement rule", "Reported, not gating", "Review before the lock is read", Amendments 1–2).

## Global Constraints

- Pass criteria (unchanged since v1), on the v4 `test` split through the full service: every source slice with ≥300 human texts has a false-AI 95% upper bound ≤ 1%; every source slice with ≥100 AI texts has a false-human 95% upper bound ≤ 10%; decisive coverage ≥ 40% of supported text.
- Shared-ground slices: fresh rows of hc3, mage, mirage, raid, beemo, no_robots, aita2026, dolly and oasst1; wasitai2026, lede and the unseen-generator split; stackexchange_heldout only through its held-out-generator answers (Amendment 2 item 4). Test portions of maga and dactyl are reported but excluded.
- Outcome table: 4d passes the gate, detects ≥ 5 points more AI on the shared-ground slices together, and no gated shared-ground slice loses more than 5 points → 4d is registered; 4d fails and today's models pass → today's models stay registered; both fail → the gate closes.
- Today's models are scored through their own registry records, including their `format2` view; each model's excerpts are cut from its own view.
- GPU limits: peak inference memory ≤ 3.5 GB; single text ≤ 500 ms at p95; a 500 × 6 sheet ≤ 120 s. CPU limits (build brief Revision 3): single text ≤ 10 s; a sheet ≤ 30 minutes.
- The locked test runs once. A second run needs a recorded reason, and only for a bug, never tuning. The run is recorded in the registry before any locked text is scored.
- An independent Ralph review runs before the lock is read, in the mode the user chooses (paste mode in a separate chat, or a subagent); its findings become numbered amendments first.
- Registry stays fail-closed; the seal covers the run's result and report.
- The full suite passes with no GPU and no model files; tests never touch CUDA.
- Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Never open, commit or inspect `x` or `inspect_wb.py`. Never train on the in-house root sheets; in-house results are report-only. Don't name individual raters.

## Review Focus

1. **The incumbent record drifting from what passed v3** (a hand edit, a merge picking up 4d's calibration): the head-to-head must refuse to score it. (Task 1 `test_a_changed_incumbent_record_is_refused`.)
2. **Rows misaligned between the two models' decisions:** every comparison must pair the same text. (Task 4 `test_head_to_head_pairs_the_same_rows`.)
3. **An interrupted head-to-head:** it must still count as a run on the v4 lock. (Task 4 `test_an_interrupted_head_to_head_still_counts_as_a_run`.)
4. **A shared-ground slice with too few AI rows deciding the outcome:** only gated slices (≥ 100 AI) can block by losing more than 5 points. (Task 3 `test_only_gated_slices_can_block_the_rule`.)
5. **An outcome the pre-registered table leaves open** (4d passes the gate but not the rule, today's models fail): must resolve by a rule written before the lock. (Task 3 `test_outcome_covers_every_case` and the Task 6 amendment.)

---

### Task 1: Freeze today's registry record

**Files:**
- Create: `models/incumbent_registry.json` (today's v5 record as committed at the 4d branch point, `6221d3b`)
- Create: `src/authorship/headtohead.py` (loader only in this task)
- Create: `tests/test_head_to_head.py`

**Interfaces:**
- Produces: `headtohead.INCUMBENT_PATH: Path`, `headtohead.INCUMBENT_MODEL_ID = "distilroberta-authorship-v5"`, `headtohead.load_incumbent(path: Optional[Path] = None) -> dict` (raises `SystemExit` on a missing file, a different model or a broken seal).

- [ ] **Step 1: Freeze the record**

Run: `git show 6221d3b:models/registry.json > models/incumbent_registry.json`
Expected: a file whose `model_id` is `distilroberta-authorship-v5` and whose `benchmark.passed` is true.

- [ ] **Step 2: Write the failing tests**

`tests/test_head_to_head.py`:

```python
"""Increment 4d head-to-head: today's models against the 4d models on the v4 locked test."""

import json

import pytest

from src.authorship import headtohead, registry


def test_the_frozen_incumbent_record_is_the_one_that_passed_v3():
    record = headtohead.load_incumbent()
    assert record["model_id"] == "distilroberta-authorship-v5"
    assert record["text_normalization"] == "format2+typography+collapse_whitespace"
    assert record["benchmark"]["passed"] is True


def test_a_changed_incumbent_record_is_refused(tmp_path):
    record = json.loads(headtohead.INCUMBENT_PATH.read_text(encoding="utf-8"))
    record["calibration"]["bands"][0]["ai_min"] = 0.5  # a hand edit after the v3 run
    path = tmp_path / "incumbent.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(SystemExit, match="seal"):
        headtohead.load_incumbent(path)
    other = {**json.loads(headtohead.INCUMBENT_PATH.read_text(encoding="utf-8")), "model_id": "someone-else"}
    path.write_text(json.dumps(other), encoding="utf-8")
    with pytest.raises(SystemExit, match="distilroberta-authorship-v5"):
        headtohead.load_incumbent(path)
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `python -m pytest tests/test_head_to_head.py -q`
Expected: FAIL — `ImportError: cannot import name 'headtohead'`.

- [ ] **Step 4: Implement the loader**

`src/authorship/headtohead.py`:

```python
"""Increment 4d head-to-head: today's models against the 4d models on the v4 locked test, in one run.

    python -m src.authorship.headtohead [--rerun-reason "..."]

Both model sets run through the full service, each through its own registry record: today's models
with their format2 view, bf16 precision, bands and excerpt view; the 4d models with theirs. The
pre-registered pass criteria apply to each on the test split. The improvement rule compares AI
detection on the shared-ground slices, and the outcome table decides which model set is registered.
The run is recorded before any locked text is scored and happens once; a second run needs a reason.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

from . import registry

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
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python -m pytest tests/test_head_to_head.py -q`
Expected: PASS (2 tests).

- [ ] **Step 6: Commit**

```bash
git add models/incumbent_registry.json src/authorship/headtohead.py tests/test_head_to_head.py
git commit -m "feat: freeze today's registry record for the 4d head-to-head"
```

---

### Task 2: One report builder for a model's locked run

**Files:**
- Modify: `src/authorship/benchmark.py` (extract `model_report`; `run` calls it)
- Test: `tests/test_head_to_head.py`

**Interfaces:**
- Produces: `benchmark.model_report(record: dict, service, df: pd.DataFrame) -> tuple[pd.DataFrame, dict]`. `df` is the locked rows with `y`; the returned frame is the general (non-in-house) rows with `decision`, `reason`, `score`; the dict holds `test`, `short_text_excerpts`, `test_unseen_generator`, `stress`, optional `diagnostic_rasbt50k`, and `inhouse_unverified`.
- Consumes: `benchmark.excerpt_report(test, services, view)` (Plan A review fix), `registry.main_only`, `registry.service_from_record`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_head_to_head.py`:

```python
import pandas as pd

from src.authorship import benchmark
from src.corpus.paths import VERSION


class _Result:
    def __init__(self, decision, reason, score):
        from src.detection import DetectionDecision

        self.decision, self.reason_codes, self.score = DetectionDecision(decision), [reason], score


class Flagger:
    """Flags texts containing `word`; labels the rest human. Records what it saw."""

    def __init__(self, word="robot"):
        self.word, self.seen = word, []

    def check_batch(self, texts, context):
        self.seen += list(texts)
        return [_Result("likely_ai_generated", "CALIBRATED_HIGH_SCORE", 0.99) if self.word in t
                else _Result("likely_human_written", "CALIBRATED_LOW_SCORE", 0.01) for t in texts]


def _locked(n=40):
    long_ai = " ".join(f"The robot wrote sentence {{i}} with care." for _ in range(20))
    long_human = " ".join(f"My cat sat on mat number {{i}} today." for _ in range(20))
    rows = []
    for i in range(n):
        for split, source, family in (("test", "raid", "openai"), ("test", "wasitai2026", "openai"),
                                      ("test", "maga", "openai"), ("test_unseen_generator", "stackexchange_heldout", "tii")):
            ai = i % 2 == 1 or split == "test_unseen_generator"
            rows.append({"sample_id": f"{source}{split}{i}", "split": split, "source_dataset": source,
                         "label": "ai" if ai else "human", "generator_family": family if ai else None,
                         "text": (long_ai if ai else long_human).replace("{i}", str(i)), "domain": "d",
                         "transform": "none", "writer_group": None, "group_key": f"g{source}{i}",
                         "source_record_id": str(i)})
    rows.append({"sample_id": "stress0", "split": "stress", "source_dataset": "raid", "label": "ai",
                 "generator_family": "openai", "text": long_ai.replace("{i}", "s"), "domain": "d",
                 "transform": "attack:paraphrase", "writer_group": None, "group_key": "gs", "source_record_id": "s"})
    for j in range(3):
        rows.append({"sample_id": f"in{j}", "split": "inhouse", "source_dataset": "legacy_inhouse", "label": "human",
                     "generator_family": None, "text": f"rater comment {j} about the chart", "domain": "rater",
                     "transform": "none", "writer_group": None, "group_key": "rater-a", "source_record_id": f"row:{j}"})
    df = pd.DataFrame(rows)
    return df.assign(y=(df["label"] == "ai").astype(int))


def _corpus_dir(tmp_path, monkeypatch, df):
    (tmp_path / "corpus" / VERSION).mkdir(parents=True)
    df.drop(columns="y").to_parquet(tmp_path / "corpus" / VERSION / "corpus.parquet", index=False)
    monkeypatch.setenv("CORPUS_DIR", str(tmp_path / "corpus"))


def test_model_report_builds_every_section_from_one_models_decisions(tmp_path, monkeypatch):
    df = _locked()
    _corpus_dir(tmp_path, monkeypatch, df)
    record = {"model_id": "m", "text_normalization": "collapse_whitespace"}
    decided, sections = benchmark.model_report(record, Flagger(), df)
    assert len(decided) == int((df["split"] != "inhouse").sum())
    assert {"test", "short_text_excerpts", "test_unseen_generator", "stress", "inhouse_unverified"} <= set(sections)
    assert sections["test"]["by_source"]["raid"]["detected_ai"] == 20
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_head_to_head.py -q`
Expected: FAIL — `AttributeError: module 'src.authorship.benchmark' has no attribute 'model_report'`.

- [ ] **Step 3: Extract `model_report`**

In `src/authorship/benchmark.py`, add before `def run(`:

```python
def model_report(record: dict, service, df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Decisions on the general locked rows and one model's report sections: the test split by source,
    band and generator family; short excerpts cut from the model's own view; unseen generators; stress;
    the rasbt50k diagnostic; and unverified in-house flags. df holds the locked rows with y."""
    general = df[df["split"] != "inhouse"].reset_index(drop=True)
    decided = pd.concat([general, decide_all(service, general["text"].tolist(), TextContext.GENERAL)], axis=1)
    test = decided[decided["split"] == "test"]
    report: dict = {"test": {"overall": slice_stats(test),
                             "by_source": {k: slice_stats(v) for k, v in test.groupby("source_dataset")},
                             "by_band": by_band(test),
                             "by_generator_family": {k: slice_stats(v)
                                                     for k, v in test[test["y"] == 1].groupby("generator_family")}}}
    services = {"routed" if record.get("short_model") else "main": service}
    if record.get("short_model"):
        services["main_only"] = registry.service_from_record(registry.main_only(record), validated=True)
    report["short_text_excerpts"] = excerpt_report(test, services, view=record.get("text_normalization"))
    unseen = decided[decided["split"] == "test_unseen_generator"]
    report["test_unseen_generator"] = {"overall": slice_stats(unseen),
                                       "by_family": {k: slice_stats(v) for k, v in unseen.groupby("generator_family")}}
    stress = decided[decided["split"] == "stress"].fillna({"transform": "none", "writer_group": "unknown"})
    report["stress"] = {f"{k[0]}|{k[1]}|{k[2]}": slice_stats(v)
                        for k, v in stress.groupby(["source_dataset", "transform", "writer_group"])}

    # Diagnostic (outside the lock): current-model texts from a source too artifact-laden to fit on.
    diag = pd.read_parquet(corpus_dir() / VERSION / "corpus.parquet", columns=_COLUMNS,
                           filters=[("split", "==", "diagnostic"), ("source_dataset", "==", "rasbt50k")])
    if len(diag):
        diag = diag.reset_index(drop=True).assign(y=lambda d: (d["label"] == "ai").astype(int))
        diag = pd.concat([diag, decide_all(service, diag["text"].tolist(), TextContext.GENERAL)], axis=1)
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
```

In `run`, replace everything from `general = df[df["split"] != "inhouse"].reset_index(drop=True)` down to (and including) the `report["inhouse_unverified"] = {...}` block with:

```python
    decided, sections = model_report(record, service, df)
    report.update(sections)
    passed, failures = gate(decided[decided["split"] == "test"])
```

(`report["peak_inference_vram_gb"]`, `report["passed"], report["failures"] = passed, failures` and everything after stay as they are.)

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_head_to_head.py tests/test_authorship_benchmark.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/authorship/benchmark.py tests/test_head_to_head.py
git commit -m "refactor: one report builder for a model's locked run"
```

---

### Task 3: Shared ground, the improvement rule and the outcome

**Files:**
- Modify: `src/authorship/headtohead.py`
- Test: `tests/test_head_to_head.py`

**Interfaces:**
- Produces: `headtohead.SHARED_FRESH_SOURCES`, `headtohead.FRONTIER_SOURCES`, `headtohead.shared_ground(decided: pd.DataFrame) -> pd.Series`, `headtohead.detection(part: pd.DataFrame) -> Optional[float]`, `headtohead.improvement(four_d: pd.DataFrame, incumbent: pd.DataFrame) -> dict` (keys `pooled`, `gain_points`, `slices`, `losing_slices`, `holds`), `headtohead.outcome(four_d_passed: bool, rule_holds: bool, incumbent_passed: bool) -> str` (one of `register_4d`, `keep_incumbent`, `close_gate`), `headtohead.goals(decided: pd.DataFrame) -> dict`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_head_to_head.py`:

```python
def _decided(rows):
    """rows: (split, source, y, decision) tuples."""
    return pd.DataFrame([{"sample_id": f"s{i}", "split": s, "source_dataset": src, "y": y, "decision": d,
                          "text": "word " * (30 if src == "short" else 150)}
                         for i, (s, src, y, d) in enumerate(rows)])


AI, HUMAN = "likely_ai_generated", "likely_human_written"


def test_shared_ground_keeps_fresh_frontier_and_unseen_rows_only():
    d = _decided([("test", "raid", 1, AI), ("test", "wasitai2026", 1, AI), ("test", "maga", 1, AI),
                  ("test", "stackexchange_heldout", 1, AI), ("test_unseen_generator", "stackexchange_heldout", 1, AI),
                  ("test", "arena140k", 1, AI), ("stress", "raid", 1, AI)])
    assert headtohead.shared_ground(d).tolist() == [True, True, False, False, True, False, False]


def test_the_rule_needs_five_points_on_the_pooled_shared_ground():
    four_d = _decided([("test", "raid", 1, AI)] * 60 + [("test", "raid", 1, HUMAN)] * 40)
    incumbent = four_d.assign(decision=[AI] * 56 + [HUMAN] * 44)
    rule = headtohead.improvement(four_d, incumbent)
    assert rule["gain_points"] == pytest.approx(4.0) and not rule["holds"]
    incumbent = four_d.assign(decision=[AI] * 55 + [HUMAN] * 45)
    assert headtohead.improvement(four_d, incumbent)["holds"]


def test_only_gated_slices_can_block_the_rule():
    # lede loses 50 points on 20 AI rows (not gated); raid gains; the rule holds.
    four_d = _decided([("test", "raid", 1, AI)] * 200 + [("test", "lede", 1, HUMAN)] * 20)
    incumbent = four_d.assign(decision=[AI] * 100 + [HUMAN] * 100 + [AI] * 20)
    rule = headtohead.improvement(four_d, incumbent)
    assert rule["holds"] and rule["losing_slices"] == []
    # With 120 lede AI rows the same loss is on a gated slice and blocks the rule.
    four_d = _decided([("test", "raid", 1, AI)] * 400 + [("test", "lede", 1, HUMAN)] * 120)
    incumbent = four_d.assign(decision=[AI] * 100 + [HUMAN] * 300 + [AI] * 120)
    rule = headtohead.improvement(four_d, incumbent)
    assert rule["gain_points"] > 5 and rule["losing_slices"] == ["lede"] and not rule["holds"]


def test_head_to_head_rows_must_pair():
    four_d = _decided([("test", "raid", 1, AI)] * 3)
    with pytest.raises(ValueError):
        headtohead.improvement(four_d, four_d.iloc[::-1].reset_index(drop=True))


def test_outcome_covers_every_case():
    table = {(True, True, True): "register_4d", (True, True, False): "register_4d",
             (True, False, True): "keep_incumbent", (False, True, True): "keep_incumbent",
             (False, False, True): "keep_incumbent", (True, False, False): "register_4d",
             (False, True, False): "close_gate", (False, False, False): "close_gate"}
    for (four_d, rule, incumbent), expected in table.items():
        assert headtohead.outcome(four_d, rule, incumbent) == expected


def test_goals_report_overall_short_and_frontier_detection():
    d = _decided([("test", "raid", 1, AI), ("test", "raid", 1, HUMAN), ("test", "short", 1, AI),
                  ("test", "lede", 1, AI), ("test", "wasitai2026", 1, HUMAN), ("test", "raid", 0, HUMAN)])
    g = headtohead.goals(d)
    assert g["overall"]["detected"] == pytest.approx(3 / 5)
    assert g["words_25_99"]["detected"] == pytest.approx(1.0) and g["words_25_99"]["ais"] == 1
    assert g["frontier"]["detected"] == pytest.approx(0.5)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_head_to_head.py -q`
Expected: FAIL — `AttributeError: module 'src.authorship.headtohead' has no attribute 'shared_ground'`.

- [ ] **Step 3: Implement**

Add to `src/authorship/headtohead.py` (imports: `import numpy as np`, `import pandas as pd`, `from .benchmark import MIN_AIS`):

```python
# Pre-registration "Improvement rule" with Amendment 2: rows neither side has the in-distribution edge on.
SHARED_FRESH_SOURCES = ("hc3", "mage", "mirage", "raid", "beemo", "no_robots", "aita2026", "dolly", "oasst1")
FRONTIER_SOURCES = ("wasitai2026", "lede")
IMPROVEMENT_POINTS = 5.0     # 4d must detect this many points more AI on the shared ground together
MAX_SLICE_LOSS_POINTS = 5.0  # and no gated shared-ground slice may lose more than this
GOAL_OVERALL, GOAL_SHORT, GOAL_FRONTIER = 0.75, 0.25, 0.70


def shared_ground(decided: pd.DataFrame) -> pd.Series:
    """Test rows of the existing and frontier sources, and every unseen-generator row. The held-out Stack
    Exchange sites' test rows come from the training generators, which 4d fitted on Stack Exchange, so
    only their held-out-generator answers count, through the unseen-generator split (Amendment 2)."""
    test = decided["split"] == "test"
    listed = decided["source_dataset"].isin(SHARED_FRESH_SOURCES + FRONTIER_SOURCES)
    return (test & listed) | (decided["split"] == "test_unseen_generator")


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
    slices["test_unseen_generator"] = {"ais": int((mine[unseen]["y"] == 1).sum()), "4d": detection(mine[unseen]),
                                       "incumbent": detection(theirs[unseen]), "gated": False}
    return {"pooled": pooled, "gain_points": gain, "slices": slices, "losing_slices": sorted(losing),
            "holds": bool(gain >= IMPROVEMENT_POINTS and not losing)}


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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_head_to_head.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/authorship/headtohead.py tests/test_head_to_head.py
git commit -m "feat: shared-ground improvement rule and outcome table for the 4d head-to-head"
```

---

### Task 4: The head-to-head run

**Files:**
- Modify: `src/authorship/headtohead.py`
- Test: `tests/test_head_to_head.py`

**Interfaces:**
- Consumes: `benchmark.model_report`, `benchmark.gate`, `benchmark._COLUMNS`, `benchmark.MAX_FALSE_AI`, `MAX_FALSE_HUMAN`, `MIN_COVERAGE`, `MIN_HUMANS`, `MIN_AIS`; `registry.load`, `registry.save`, `registry.verify_files`, `registry.service_from_record`, `registry.seal`, `registry.text_sha256`; `corpus.lock.read`, `corpus.lock.locked_hash`; Task 3 functions.
- Produces: `headtohead.report_file(version=VERSION) -> Path` (`data/benchmark/<version>/model/head_to_head.json`); `headtohead.run(rerun_reason=None, registry_path=None, incumbent_path=None, report_path=None, lock_file=None, services=None) -> dict`; CLI `--rerun-reason`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_head_to_head.py`:

```python
from src.corpus import lock as corpus_lock
from src.corpus.schema import LOCKED_SPLITS


def _setup(tmp_path, monkeypatch):
    df = _locked()
    _corpus_dir(tmp_path, monkeypatch, df)
    locked = df[df["split"].isin([s.value for s in LOCKED_SPLITS])].drop(columns="y")
    lock_file = tmp_path / "test.lock.json"
    lock_file.write_text(json.dumps({"sha256": corpus_lock.locked_hash(locked)}), encoding="utf-8")
    reg = tmp_path / "registry.json"
    registry.save({"model_id": "4d-main", "policy_version": "v", "text_normalization": "collapse_whitespace",
                   "benchmark": None, "locked_benchmark_history": []}, reg)
    return dict(registry_path=reg, report_path=tmp_path / "head_to_head.json", lock_file=lock_file)


def test_head_to_head_registers_4d_when_it_passes_and_beats_today(tmp_path, monkeypatch):
    paths = _setup(tmp_path, monkeypatch)
    four_d, today = Flagger("robot"), Flagger("never-present")
    report = headtohead.run(services={"4d": four_d, "incumbent": today}, **paths)
    assert report["outcome"] == "register_4d"
    assert report["improvement_rule"]["holds"] and report["improvement_rule"]["gain_points"] == 100.0
    record = registry.load(paths["registry_path"])
    assert record["benchmark"]["passed"] is True
    assert record["benchmark"]["record_sha256"] == registry.record_sha256(record, record["benchmark"])
    assert record["locked_benchmark_history"][-1]["incumbent_model_id"] == "distilroberta-authorship-v5"
    with pytest.raises(SystemExit, match="already ran"):
        headtohead.run(services={"4d": four_d, "incumbent": today}, **paths)


def test_head_to_head_pairs_the_same_rows(tmp_path, monkeypatch):
    paths = _setup(tmp_path, monkeypatch)
    four_d, today = Flagger("robot"), Flagger("robot")
    headtohead.run(services={"4d": four_d, "incumbent": today}, **paths)
    assert four_d.seen[:200] == today.seen[:200]


def test_an_interrupted_head_to_head_still_counts_as_a_run(tmp_path, monkeypatch):
    paths = _setup(tmp_path, monkeypatch)

    class Crashes(Flagger):
        def check_batch(self, texts, context):
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        headtohead.run(services={"4d": Crashes(), "incumbent": Flagger()}, **paths)
    record = registry.load(paths["registry_path"])
    assert record["locked_benchmark_history"][-1]["status"] == "started"
    with pytest.raises(SystemExit, match="already ran"):
        headtohead.run(services={"4d": Flagger(), "incumbent": Flagger()}, **paths)


def test_head_to_head_refuses_a_changed_lock(tmp_path, monkeypatch):
    paths = _setup(tmp_path, monkeypatch)
    paths["lock_file"].write_text(json.dumps({"sha256": "0" * 64}), encoding="utf-8")
    with pytest.raises(SystemExit, match="lock"):
        headtohead.run(services={"4d": Flagger(), "incumbent": Flagger()}, **paths)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_head_to_head.py -q`
Expected: FAIL — `AttributeError: module 'src.authorship.headtohead' has no attribute 'run'`.

- [ ] **Step 3: Implement**

Add to `src/authorship/headtohead.py` (imports: `from datetime import datetime, timezone`; `from ..corpus.lock import locked_hash, read as read_lock`; `from ..corpus.paths import REPO_BENCHMARK_DIR, VERSION, corpus_dir, lock_path`; `from ..corpus.schema import LOCKED_SPLITS`; `from . import benchmark`):

```python
def report_file(version: str = VERSION) -> Path:
    return REPO_BENCHMARK_DIR / version / "model" / "head_to_head.json"


def run(rerun_reason: Optional[str] = None, registry_path: Optional[Path] = None,
        incumbent_path: Optional[Path] = None, report_path: Optional[Path] = None,
        lock_file: Optional[Path] = None, services: Optional[dict] = None) -> dict:
    """services ({"4d": ..., "incumbent": ...}) replaces the registered services in tests."""
    record = registry.load(registry_path)
    if record is None:
        raise SystemExit("no registry entry; calibrate the 4d models first")
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
    if services is None:
        for name, rec in (("4d", record), ("incumbent", incumbent)):
            ok, why = registry.verify_files(rec)
            if not ok:
                raise SystemExit(f"{name} files do not verify ({why}); refusing to run")
        # Both are candidates under test here; the outcome decides what the registry opens for.
        services = {"4d": registry.service_from_record(record, validated=True),
                    "incumbent": registry.service_from_record(incumbent, validated=True)}
    # Recorded before any locked text is scored, so an interrupted run still counts as a run.
    entry = {"started_at": datetime.now(timezone.utc).isoformat(), "status": "started",
             "model_id": record["model_id"], "short_model_id": (record.get("short_model") or {}).get("model_id"),
             "incumbent_model_id": incumbent["model_id"], "head_to_head": True, "rerun_reason": rerun_reason,
             "corpus_version": VERSION, "lock_sha256": lock["sha256"]}
    record.setdefault("locked_benchmark_history", []).append(entry)
    registry.save(record, registry_path)

    try:
        import torch
    except ImportError:
        torch = None
    if torch is not None and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    models, decided = {}, {}
    for name, rec in (("4d", record), ("incumbent", incumbent)):
        logger.info("scoring the locked splits with %s (%s)", name, rec["model_id"])
        frame, sections = benchmark.model_report(rec, services[name], df)
        passed, failures = benchmark.gate(frame[frame["split"] == "test"])
        models[name] = {"model_id": rec["model_id"], "short_model_id": (rec.get("short_model") or {}).get("model_id"),
                        "text_normalization": rec.get("text_normalization"), "passed": passed,
                        "failures": failures, "goals": goals(frame), **sections}
        decided[name] = frame
    rule = improvement(decided["4d"], decided["incumbent"])
    result = outcome(models["4d"]["passed"], rule["holds"], models["incumbent"]["passed"])
    report = {"criteria": {"max_false_ai_upper95": benchmark.MAX_FALSE_AI,
                           "max_false_human_upper95": benchmark.MAX_FALSE_HUMAN,
                           "min_coverage_supported": benchmark.MIN_COVERAGE, "min_humans": benchmark.MIN_HUMANS,
                           "min_ais": benchmark.MIN_AIS, "improvement_points": IMPROVEMENT_POINTS,
                           "max_slice_loss_points": MAX_SLICE_LOSS_POINTS},
              "models": models, "improvement_rule": rule, "outcome": result,
              "peak_inference_vram_gb": (round(torch.cuda.max_memory_allocated() / 1e9, 3)
                                         if torch is not None and torch.cuda.is_available() else None),
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_head_to_head.py tests/test_authorship_benchmark.py -q`
Expected: PASS.

- [ ] **Step 5: Run the whole suite and commit**

Run: `python -m pytest -q > .superpowers/sdd/2026-09-28-increment-4d-head-to-head/suite-task4.txt 2>&1; tail -3 .superpowers/sdd/2026-09-28-increment-4d-head-to-head/suite-task4.txt`
Expected: all pass.

```bash
git add src/authorship/headtohead.py tests/test_head_to_head.py
git commit -m "feat: one-shot head-to-head of today's and the 4d models on the v4 locked test"
```

---

### Task 5: Speed and memory limits of the system under test

**Files:**
- Create: `src/authorship/speed.py`
- Test: `tests/test_head_to_head.py`

**Interfaces:**
- Consumes: `registry.load`, `registry.service_from_record`, `calibrate.load_split`.
- Produces: `speed.LIMITS`, `speed.measure(service, singles: list, sheet: list, sheet_cells: int = 3000) -> dict` (`p95_ms`, `sheet_seconds`, `sheet_cells_timed`), `speed.within(result: dict, device: str) -> dict` (`limits`, `checks`, `passed`), `speed.run(registry_path=None, out=None) -> dict`; report `data/benchmark/v4/model/speed.json` keyed by device.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_head_to_head.py`:

```python
from src.authorship import speed


class Slow(Flagger):
    def __init__(self, seconds):
        super().__init__()
        self.seconds = seconds

    def check_batch(self, texts, context):
        import time

        time.sleep(self.seconds * len(texts))
        return super().check_batch(texts, context)


def test_speed_times_single_texts_and_scales_the_sheet_sample():
    result = speed.measure(Slow(0.01), singles=["a"] * 10, sheet=["b"] * 20, sheet_cells=3000)
    assert 8 <= result["p95_ms"] <= 60
    assert 20 <= result["sheet_seconds"] <= 90 and result["sheet_cells_timed"] == 20


def test_speed_limits_follow_the_device():
    gpu = speed.within({"p95_ms": 400.0, "sheet_seconds": 100.0, "peak_vram_gb": 3.6}, "cuda")
    assert not gpu["passed"] and gpu["checks"] == {"peak_vram_gb": False, "p95_ms": True, "sheet_seconds": True}
    cpu = speed.within({"p95_ms": 9_000.0, "sheet_seconds": 1_700.0}, "cpu")
    assert cpu["passed"]
    assert not speed.within({"p95_ms": 11_000.0, "sheet_seconds": 100.0}, "cpu")["passed"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_head_to_head.py -q`
Expected: FAIL — `ImportError: cannot import name 'speed'`.

- [ ] **Step 3: Implement `src/authorship/speed.py`**

```python
"""Speed and memory limits of the system under test (pre-registration; build brief Revision 3).

    python -m src.authorship.speed                          # the GPU limits
    CUDA_VISIBLE_DEVICES= python -m src.authorship.speed    # the CPU limits

Runs the registered service over validation texts: 30 single texts for the p95 time, and a 500 x 6
sheet (3,000 cells; on the CPU a 100-cell sample, scaled to 3,000). On the GPU, peak inference memory is
recorded too. Results merge into data/benchmark/<version>/model/speed.json by device. Reads only
validation rows.
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from ..corpus.paths import REPO_BENCHMARK_DIR, VERSION
from ..detection import TextContext
from . import registry

logger = logging.getLogger(__name__)

LIMITS = {"cuda": {"peak_vram_gb": 3.5, "p95_ms": 500.0, "sheet_seconds": 120.0},
          "cpu": {"p95_ms": 10_000.0, "sheet_seconds": 1_800.0}}
SHEET_CELLS, SINGLES, CPU_SHEET_SAMPLE = 3000, 30, 100


def measure(service, singles: list, sheet: list, sheet_cells: int = SHEET_CELLS) -> dict:
    service.check_batch(singles[:2], TextContext.GENERAL)  # warm-up: loads the models
    times = []
    for text in singles:
        start = time.perf_counter()
        service.check_batch([text], TextContext.GENERAL)
        times.append(1000 * (time.perf_counter() - start))
    start = time.perf_counter()
    service.check_batch(sheet, TextContext.GENERAL)
    elapsed = time.perf_counter() - start
    return {"p95_ms": round(float(np.percentile(times, 95)), 1),
            "sheet_seconds": round(elapsed * sheet_cells / len(sheet), 1), "sheet_cells_timed": len(sheet)}


def within(result: dict, device: str) -> dict:
    limits = LIMITS[device]
    checks = {k: result.get(k) is not None and result[k] <= v for k, v in limits.items()}
    return {"limits": limits, "checks": checks, "passed": all(checks.values())}


def report_file() -> Path:
    return REPO_BENCHMARK_DIR / VERSION / "model" / "speed.json"


def run(registry_path: Optional[Path] = None, out: Optional[Path] = None) -> dict:
    import torch

    from .calibrate import load_split

    device = "cuda" if torch.cuda.is_available() else "cpu"
    record = registry.load(registry_path)
    service = registry.service_from_record(record, validated=True)
    val = load_split("val")
    sheet = val.sample(n=SHEET_CELLS, random_state=0)["text"].tolist()
    singles = val.sample(n=SINGLES, random_state=1)["text"].tolist()
    if device == "cpu":
        sheet = sheet[:CPU_SHEET_SAMPLE]
    else:
        torch.cuda.reset_peak_memory_stats()
    result = measure(service, singles, sheet)
    if device == "cuda":
        result["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 3)
    result.update(within(result, device), measured_at=datetime.now(timezone.utc).isoformat())
    out = out or report_file()
    saved = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    saved.update({"model_id": record["model_id"], device: result, "locked_split_reads": 0})
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(saved, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return saved


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    print(json.dumps(run(), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the tests to verify they pass, then commit**

Run: `python -m pytest tests/test_head_to_head.py -q`
Expected: PASS.

```bash
git add src/authorship/speed.py tests/test_head_to_head.py
git commit -m "feat: measure the 4d system against the GPU and CPU limits"
```

---

### Task 6: The head-to-head amendment and the Ralph brief

**Files:**
- Modify: `docs/bob/llm-authorship-checker/increment-4d-preregistration.md` (next numbered amendment)
- Create: `docs/bob/llm-authorship-checker/ralph/prompt-increment-4d.md`

- [ ] **Step 1: Write the amendment**

Append a section `## Amendment N — head-to-head definitions (before the lock is read)` (N = the next free number) that states, in the document's plain style:
1. The v4 locked run is one head-to-head run (`python -m src.authorship.headtohead`): both model sets run through the full service on the same locked rows, each through its own registry record; today's record is `models/incumbent_registry.json`, frozen at `6221d3b` and refused unless its v3 seal verifies.
2. Pooled shared-ground detection counts every AI row of the shared-ground slices (test rows of the listed existing and frontier sources, every unseen-generator row).
3. A gated shared-ground slice is a test-split source slice with at least 100 AI rows; the unseen-generator split is report-only in the pass criteria, so it counts in the pooled figure but can't block by losing.
4. The outcome table's open case: 4d passes the gate but not the improvement rule, and today's models fail the gate → 4d is registered, as the only model set meeting the contract. The improvement rule governs replacing a model set that passes.
5. Speed and memory limits are measured on validation text with `python -m src.authorship.speed` (GPU, then with the GPU hidden for the CPU) before the locked run; a model set breaking a limit can't be registered.

- [ ] **Step 2: Write the Ralph brief**

Create `docs/bob/llm-authorship-checker/ralph/prompt-increment-4d.md` in the format of `ralph/prompt-increment-4c.md` (read it first and keep its structure: role, what to review, files to read, what to return). Scope, from the pre-registration's "Review before the lock is read": the v4 corpus and audit (`data/benchmark/v4/build_record.json`, fresh rows per source, retirement over every earlier split, held-out sources, near-duplicates); the `format3` view and disguise handling (`src/corpus/text.py`, `src/detection/obfuscation.py`); the validation results of the chosen models (training records, `calibration.json`, `stack.json`, `registers.json`, `speed.json`); and the head-to-head procedure (`src/authorship/headtohead.py`, Amendments 2 and N). Ask for numbered findings (R12-1, R12-2, ...) with severity and a proposed fix, and a verdict of PASS or FAIL for running the lock.

- [ ] **Step 3: Commit and push**

```bash
git add docs/bob/llm-authorship-checker/increment-4d-preregistration.md docs/bob/llm-authorship-checker/ralph/prompt-increment-4d.md
git commit -m "docs: 4d head-to-head amendment and Ralph brief"
git push
```

---

### Task 7: Ralph review, limits, the locked run and the outcome

Runs after Plan B Task 6 (4d models trained and calibrated, optional steps decided).

- [ ] **Step 1: Ralph review.** Ask the user which mode they want: paste mode in a separate chat (their 2026-09-26 preference) or a subagent (how 4c's R08–R11 ran). Then run `ralph/prompt-increment-4d.md` that way; the review lands in `docs/bob/llm-authorship-checker/ralph/12-increment-4d.md`. Fix each finding as a numbered amendment (code fixes test-first), committed before the lock is read. A FAIL verdict repeats this step after the fixes.

- [ ] **Step 2: Limits.** Run `python -m src.authorship.speed --device cuda`, then `python -m src.authorship.speed --device cpu` (it hides the GPU with `CUDA_VISIBLE_DEVICES=-1`; an empty value doesn't hide it on Windows, Ralph R13-8), detached. Expected: `speed.json` with `cuda.passed` and `cpu.passed` true. A failed limit stops here: it is recorded as an amendment and the plan returns to Plan B.

- [ ] **Step 3: The locked run, once.** Run `python -m src.authorship.headtohead`, detached, with its log on `D:`. Expected: `head_to_head.json` written, the registry's 4d record sealed, the printed outcome one of `register_4d`, `keep_incumbent`, `close_gate`. Commit `data/benchmark/v4/model/head_to_head.json` and `models/registry.json`, and push.

- [ ] **Step 4: The report.** Write `docs/bob/llm-authorship-checker/increment-4d-report-<date>.md`: the outcome; both gates with every failing slice; the improvement rule (pooled and per slice); the goals; results by source, band and family for both model sets; excerpts; stress; in-house flags (report only, no rater names); limits; and what was ruled or amended along the way.

- [ ] **Step 5: Act on the outcome — ask the user first.** Each action leaves this worktree, so it waits for the user's go-ahead:
  - `register_4d`: merge `increment-4d` into `llm-authorship-checker` (superpowers:finishing-a-development-branch), so the served registry becomes the 4d record.
  - `keep_incumbent`: nothing changes in the main checkout; the 4d branch stays unmerged, with its report.
  - `close_gate`: record the failed v4 run in the main checkout's registry (append the run to `locked_benchmark_history` and re-seal `benchmark` with `passed: false`), so the served gate closes as the pre-registration requires.
