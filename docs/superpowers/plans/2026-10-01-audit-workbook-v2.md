# Audit Workbook v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** One verdict tier per audited row with measured rates, trigger sentences for high scores, and a reorganised audit workbook and dashboard.

**Architecture:** Everything lives in the report layer (`src/report/`) on top of the decisions the sealed service already makes. `tiers.py` labels a row from its decision and score; `tier_rates.py` measures each tier on saved validation scores; `triggers.py` re-scores a row's text with one sentence removed at a time; `rows.py` holds one `RowReport` per row, which the pipeline builds and the workbook, summary and API read.

**Tech Stack:** Python 3.12, pandas, numpy, openpyxl 3.1.5 (rich text), FastAPI, pytest.

**Spec:** `docs/superpowers/specs/2026-10-01-audit-workbook-v2-design.md`

## Global Constraints

- No change to any decision, score or threshold; Likely AI means exactly what it means today.
- Never edit a sealed scoring file: `src/detection/*.py`, `src/authorship/{detector,excerpts,language,registry,calibration}.py`, `src/corpus/text.py`, `src/detectors/base.py`. Importing from them is fine.
- `/api/v2/check-text` and the detection contract are unchanged.
- Rates come from validation data only: never the locked test, never the in-house sheets.
- Trigger sentences never run on paid API detectors (GPTZero, Originality).
- Tier bands: the sponsor's 75 / 50 / 30 (2026-09-28). Tier names: "Likely AI", "High AI score – review" (en dash), "Moderate AI score", "Needs a second look", "Low AI score", "Not checked", "Likely human".
- Trigger defaults: min drop 10 points, at most 12 candidate sentences, top 2, candidates start within the first 160 words.
- Every value from an upload stays inert text in the workbook (formula safety).
- Commits end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Work in `..\excel-ai-qa-pipeline-4d` on `increment-4d`.
- Never read cell text from the rater workbooks; real-run checks report structure and counts only. Never name raters.

## Review Focus

1. A rationale with no sentence punctuation (one long line, or bullet lines): `split_sentences` must still give usable sentences (line breaks split) and never an empty span.
2. A row where only the second of two text columns is filled: decided cells, the rationale cell and the trigger spans must point at that column, not the first.
3. A trigger error on one row (e.g. CUDA out of memory): that row reads "trigger not found (error)" and the audit finishes.
4. A rationale starting with `=` that also has a trigger sentence (rich text): it must stay text, never a formula.
5. A sheet with no structural issues and no agent column: no Structural Issues sheet, agents read "(no agent)", and nothing crashes.

---

### Task 1: Verdict tiers and config

**Files:**
- Create: `src/report/tiers.py`
- Modify: `src/config.py` (add `TierConfig`, `TriggerConfig`; keep `ReviewTagConfig` until Task 5)
- Test: `tests/test_tiers.py`

**Interfaces:**
- Produces: `LIKELY_AI, HUMAN, HIGH, MODERATE, SECOND_LOOK, LOW, NOT_CHECKED, TIER_ORDER, EXPLAINED` (str constants and tuples); `not_checked_reason(codes: Iterable[str]) -> str`; `TierRules(high_min=0.75, moderate_min=0.50, second_look_min=0.30)` with `.from_config(cfg)`, `.by_score(score: float) -> str`, `.tier(result: TextCheckResponse) -> Tuple[str, Optional[str]]`, `.meanings() -> Dict[str, str]`, `.as_dict() -> dict`; `bar_for(policy: DecisionPolicy, text: str) -> Optional[float]`. Config: `PipelineConfig.tiers: TierConfig(high_min, moderate_min, second_look_min)`, `PipelineConfig.triggers: TriggerConfig(enabled=True, min_drop=0.10, max_sentences=12, top=2)`.

- [ ] **Step 1: Write the failing tests** — `tests/test_tiers.py`:

```python
"""Verdict tiers (spec 2026-10-01): one label per row that never contradicts the calibrated decision."""
import pytest

from src.config import PipelineConfig, TierConfig, TriggerConfig
from src.detection import DetectionDecision, DetectorStatus, TextCheckResponse
from src.detection.policy import BandRule, DecisionPolicy
from src.report.tiers import (EXPLAINED, HIGH, HUMAN, LIKELY_AI, LOW, MODERATE, NOT_CHECKED, SECOND_LOOK,
                              TIER_ORDER, TierRules, bar_for, not_checked_reason)

RULES = TierRules()


def _result(decision, score, codes=("X",)):
    evidence = [] if decision is DetectionDecision.INCONCLUSIVE else [
        {"code": "E", "description": "evidence", "supports": decision}]
    return TextCheckResponse(decision=decision, score=score, score_valid=score is not None,
                             detector_status=DetectorStatus.AVAILABLE, reason_codes=list(codes),
                             evidence=evidence, detector_version="t", policy_version="p")


UNSURE, AI, HUMAN_D = (DetectionDecision.INCONCLUSIVE, DetectionDecision.LIKELY_AI_GENERATED,
                       DetectionDecision.LIKELY_HUMAN_WRITTEN)


def test_defaults_are_the_sponsors_bands_and_the_trigger_rules():
    cfg = PipelineConfig()
    assert (cfg.tiers.high_min, cfg.tiers.moderate_min, cfg.tiers.second_look_min) == (0.75, 0.50, 0.30)
    assert (cfg.triggers.enabled, cfg.triggers.min_drop, cfg.triggers.max_sentences, cfg.triggers.top) == (
        True, 0.10, 12, 2)
    assert TIER_ORDER[0] == LIKELY_AI and EXPLAINED == (LIKELY_AI, HIGH)
    assert HIGH == "High AI score – review"


def test_only_a_likely_ai_decision_is_called_likely_ai():
    assert RULES.tier(_result(AI, 0.995)) == (LIKELY_AI, None)
    assert RULES.tier(_result(UNSURE, 0.98)) == (HIGH, None)       # under the bar: ranked, not decided
    assert RULES.tier(_result(UNSURE, 0.75)) == (HIGH, None)       # bands include their lower edge
    assert RULES.tier(_result(UNSURE, 0.60)) == (MODERATE, None)
    assert RULES.tier(_result(UNSURE, 0.30)) == (SECOND_LOOK, None)
    assert RULES.tier(_result(UNSURE, 0.29)) == (LOW, None)
    assert RULES.tier(_result(HUMAN_D, 0.02)) == (HUMAN, None)


def test_a_mixed_signal_row_above_the_bar_is_ranked_by_its_score():
    assert RULES.tier(_result(UNSURE, 0.995, ("MIXED_AUTHORSHIP_SIGNALS",))) == (HIGH, None)


@pytest.mark.parametrize("code,reason", [
    ("UNSUPPORTED_LANGUAGE", "not English"), ("INSUFFICIENT_TEXT", "too short"),
    ("UNSUPPORTED_INPUT_LENGTH", "too long"), ("DISGUISE_SUSPECTED", "disguised characters"),
    ("MODEL_NOT_VALIDATED", "model not validated"), ("UNSUPPORTED_LENGTH", "no bar for this length"),
    ("UNSUPPORTED_DOMAIN", "unlike calibrated text"), ("SCORE_UNAVAILABLE", "detector unavailable"),
    ("DETECTOR_ERROR", "detector unavailable"), ("SOMETHING_NEW", "SOMETHING_NEW")])
def test_unscored_rows_are_not_checked_with_a_reason(code, reason):
    assert RULES.tier(_result(UNSURE, None, (code,))) == (NOT_CHECKED, reason)
    assert not_checked_reason([code]) == reason


def test_bands_come_from_config():
    rules = TierRules.from_config(TierConfig(high_min=0.9, moderate_min=0.8, second_look_min=0.7))
    assert [rules.by_score(s) for s in (0.95, 0.85, 0.75, 0.65)] == [HIGH, MODERATE, SECOND_LOOK, LOW]
    assert rules.as_dict() == {"high_min": 0.9, "moderate_min": 0.8, "second_look_min": 0.7}


def test_every_tier_has_a_meaning():
    assert set(RULES.meanings()) == set(TIER_ORDER)


def test_the_bar_is_the_length_band_of_the_text():
    policy = DecisionPolicy(min_word_count=3, bands=(BandRule(3, 10, 0.1, 0.99), BandRule(10, None, 0.1, 0.94)))
    assert bar_for(policy, "one two three four") == 0.99
    assert bar_for(policy, " ".join(["word"] * 12)) == 0.94
    assert bar_for(policy, "one two") is None                     # no band this short
    assert bar_for(DecisionPolicy(), "any text at all") == DecisionPolicy().ai_min_score
    assert bar_for(policy, "") is None


def test_trigger_config_rejects_nonsense():
    with pytest.raises(ValueError):
        TriggerConfig(min_drop=0.0)
    with pytest.raises(ValueError):
        TriggerConfig(max_sentences=0)
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_tiers.py -q`
Expected: FAIL at import (`cannot import name 'TierConfig'` / `No module named 'src.report.tiers'`).

- [ ] **Step 3: Add the config classes** — in `src/config.py`, after `ReviewTagConfig`:

```python
class TierConfig(BaseModel):
    """Score bands below the calibrated bar (the sponsor's 75 / 50 / 30 from 2026-09-28; spec 2026-10-01)."""
    high_min: float = Field(default=0.75, ge=0.0, le=1.0, description="High AI score - review at or above this score")
    moderate_min: float = Field(default=0.50, ge=0.0, le=1.0, description="Moderate AI score at or above this score")
    second_look_min: float = Field(default=0.30, ge=0.0, le=1.0, description="Needs a second look at or above this score")


class TriggerConfig(BaseModel):
    """Trigger sentences: the sentences whose removal lowers a high AI score most (spec 2026-10-01)."""
    enabled: bool = True
    min_drop: float = Field(default=0.10, gt=0.0, le=1.0, description="Smallest score drop that counts as a trigger")
    max_sentences: int = Field(default=12, ge=1, description="Most sentences tried per row")
    top: int = Field(default=2, ge=1, description="Most trigger sentences shown per row")
```

and in `PipelineConfig` add:

```python
    tiers: TierConfig = Field(default_factory=TierConfig)
    triggers: TriggerConfig = Field(default_factory=TriggerConfig)
```

- [ ] **Step 4: Write `src/report/tiers.py`**

```python
"""Verdict tiers for the audit workbook (spec 2026-10-01).

One label per row that never contradicts the calibrated decision: Likely AI means the row cleared the
calibrated bar for its length. Below the bar, rows are ranked by AI score in the sponsor's bands
(2026-09-28: 75 / 50 / 30). Rater rationales are flag-only, so they are never called Likely human.
"""

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, Optional, Tuple

from ..detection import DetectionDecision, TextCheckResponse
from ..detection.obfuscation import deobfuscate
from ..detection.policy import DecisionPolicy

LIKELY_AI = "Likely AI"
HUMAN = "Likely human"
HIGH = "High AI score – review"
MODERATE = "Moderate AI score"
SECOND_LOOK = "Needs a second look"
LOW = "Low AI score"
NOT_CHECKED = "Not checked"
TIER_ORDER = (LIKELY_AI, HIGH, MODERATE, SECOND_LOOK, LOW, NOT_CHECKED, HUMAN)
EXPLAINED = (LIKELY_AI, HIGH)  # tiers whose trigger sentences are looked for

_NOT_CHECKED = {
    "UNSUPPORTED_LANGUAGE": "not English",
    "INSUFFICIENT_TEXT": "too short",
    "UNSUPPORTED_INPUT_LENGTH": "too long",
    "DISGUISE_SUSPECTED": "disguised characters",
    "MODEL_NOT_VALIDATED": "model not validated",
    "UNSUPPORTED_LENGTH": "no bar for this length",
    "UNSUPPORTED_DOMAIN": "unlike calibrated text",
    "SCORE_UNAVAILABLE": "detector unavailable",
}


def not_checked_reason(codes: Iterable[str]) -> str:
    """Why a row has no score, in words, from its reason codes."""
    codes = list(codes)
    for code in codes:
        if code in _NOT_CHECKED:
            return _NOT_CHECKED[code]
        if code.startswith("DETECTOR_"):
            return "detector unavailable"
    return ", ".join(codes) or "no score"


@dataclass(frozen=True)
class TierRules:
    high_min: float = 0.75
    moderate_min: float = 0.50
    second_look_min: float = 0.30

    @classmethod
    def from_config(cls, config) -> "TierRules":
        return cls(config.high_min, config.moderate_min, config.second_look_min)

    def as_dict(self) -> dict:
        return asdict(self)

    def by_score(self, score: float) -> str:
        if score >= self.high_min:
            return HIGH
        if score >= self.moderate_min:
            return MODERATE
        return SECOND_LOOK if score >= self.second_look_min else LOW

    def tier(self, result: TextCheckResponse) -> Tuple[str, Optional[str]]:
        """(tier, the reason a row was not checked, else None). A row above the bar whose decision is
        Inconclusive (mixed signals across a long text) is ranked by its score."""
        if result.decision is DetectionDecision.LIKELY_AI_GENERATED:
            return LIKELY_AI, None
        if result.decision is DetectionDecision.LIKELY_HUMAN_WRITTEN:
            return HUMAN, None
        if result.score is None:
            return NOT_CHECKED, not_checked_reason(result.reason_codes)
        return self.by_score(result.score), None

    def meanings(self) -> Dict[str, str]:
        pct = lambda x: f"{x * 100:.0f}%"  # noqa: E731
        return {
            LIKELY_AI: "Cleared the calibrated bar for its length, set so that under 1% of human writing in "
                       "validation clears it. The only tier that is a decision.",
            HIGH: f"AI score of {pct(self.high_min)} or more, but under the bar. Review.",
            MODERATE: f"AI score from {pct(self.moderate_min)} to under {pct(self.high_min)}.",
            SECOND_LOOK: f"AI score from {pct(self.second_look_min)} to under {pct(self.moderate_min)}.",
            LOW: f"AI score under {pct(self.second_look_min)}. Not proof of human writing.",
            NOT_CHECKED: "No score: not English, too short, disguised characters, or the model could not score it.",
            HUMAN: "The model's calibrated human decision (never given to rater rationales).",
        }


def bar_for(policy: DecisionPolicy, text: str) -> Optional[float]:
    """The score a text needs for a Likely AI decision: its length band's bar, the policy's single bar when it
    has no bands, or None when no band covers the text's length."""
    if not isinstance(text, str) or not text.strip():
        return None
    if not policy.bands:
        return policy.ai_min_score
    band = policy.band_for(policy.word_count(deobfuscate(text).text))
    return band.ai_min_score if band else None
```

After writing, check the en dash survived as the escape or the character: `grep -n "High AI score" src/report/tiers.py`.

- [ ] **Step 5: Run to verify it passes**

Run: `python -m pytest tests/test_tiers.py -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/report/tiers.py src/config.py tests/test_tiers.py
git commit -m "feat: verdict tiers for the audit workbook (one label per row, the bar per length)"
```

---

### Task 2: Measured tier rates

**Files:**
- Create: `src/report/tier_rates.py`, `data/report/tier_rates.json` (generated)
- Test: `tests/test_tier_rates.py`

**Interfaces:**
- Consumes: `TierRules`, tier constants (Task 1); `registry.load`, `registry.build_policy`, `registry.artifact_dir`, `registry.short_parts`, `registry.sha256_file`; `calibration.probability(raw, a, b)`; `calibrate.load_split`, `calibrate.with_excerpts`.
- Produces: `RATES_FILE: Path`; `RATED_TIERS = (LIKELY_AI, HIGH, MODERATE, SECOND_LOOK, LOW)`; `SLICES = {"25-100": (25, 100), "all": (0, None)}`; `scored_frame(val, main, short, record) -> DataFrame` (adds `prob`); `rate_table(frame, rules, policy) -> dict` = `{slice: {"human_texts", "ai_texts", "tiers": {tier: {"human": float, "ai": float}}}}`; `build(record=None, path=RATES_FILE, rules=TierRules()) -> dict`; `load_rates(model_id, path=RATES_FILE, rules=None) -> Optional[dict]` (entry keys: `model_id, measured_at, bands, slices, sources, notes`).

- [ ] **Step 1: Write the failing tests** — `tests/test_tier_rates.py`:

```python
"""Measured tier rates from saved validation scores (spec 2026-10-01, section 1)."""
import json
import math

import numpy as np
import pandas as pd
import pytest

from src.report.tier_rates import build, load_rates, rate_table, scored_frame
from src.report.tiers import HIGH, LIKELY_AI, LOW, TierRules
from src.authorship import registry

RECORD = {"model_id": "m1", "calibration": {"platt": [1.0, 0.0], "bands": [
    {"lower_words": 25, "upper_words": 100, "human_max": 0.05, "ai_min": 0.99, "supported": True},
    {"lower_words": 100, "upper_words": None, "human_max": 0.05, "ai_min": 0.90, "supported": True}]},
    "short_model": {"platt": [2.0, 0.0], "ranges": [[25, 100]]}}


def _raw(p, a):
    return math.log(p / (1 - p)) / a


VAL = pd.DataFrame({"sample_id": ["a", "b", "c", "d", "e"], "y": [1, 0, 1, 0, 0], "words": [30, 30, 150, 150, 10]})
MAIN = {"ids": np.array(["a", "b", "c", "d", "e"]), "raw": np.array([0.0, 0.0, _raw(0.95, 1), _raw(0.10, 1), 0.0])}
SHORT = {"ids": np.array(["a", "b"]), "raw": np.array([_raw(0.995, 2), _raw(0.80, 2)])}


def test_texts_are_routed_and_mapped_as_the_service_does():
    frame = scored_frame(VAL, MAIN, SHORT, RECORD)
    assert frame.set_index("sample_id")["prob"].round(3).to_dict() == {
        "a": 0.995, "b": 0.8, "c": 0.95, "d": 0.1, "e": 0.5}


def test_each_tier_holds_its_share_of_human_and_ai_texts_per_slice():
    table = rate_table(scored_frame(VAL, MAIN, SHORT, RECORD), TierRules(), registry.build_policy(RECORD))
    short, every = table["25-100"], table["all"]
    assert (short["human_texts"], short["ai_texts"]) == (1, 1)
    assert short["tiers"][HIGH] == {"human": 1.0, "ai": 0.0}
    assert short["tiers"][LIKELY_AI] == {"human": 0.0, "ai": 1.0}
    assert (every["human_texts"], every["ai_texts"]) == (2, 2)   # "e" has no bar and is left out
    assert every["tiers"][HIGH]["human"] == 0.5 and every["tiers"][LOW]["human"] == 0.5
    assert every["tiers"][LIKELY_AI]["ai"] == 1.0


def test_a_text_without_a_saved_score_fails_rather_than_guessing():
    short = {"ids": np.array(["a"]), "raw": np.array([1.0])}
    with pytest.raises(RuntimeError, match="no saved score"):
        scored_frame(VAL, MAIN, short, RECORD)


def test_rates_are_stored_per_model_and_only_apply_to_their_bands(tmp_path, monkeypatch):
    path = tmp_path / "rates.json"
    monkeypatch.setattr("src.report.tier_rates.validation_frame", lambda record: VAL)
    monkeypatch.setattr("src.report.tier_rates.saved_scores", lambda record: (MAIN, SHORT, {"main": "x"}))
    entry = build(dict(RECORD, text_normalization="v", corpus_version="c"), path=path)
    assert entry["model_id"] == "m1" and entry["bands"] == TierRules().as_dict()
    assert load_rates("m1", path)["slices"]["25-100"]["tiers"][HIGH]["human"] == 1.0
    assert load_rates("m1", path, TierRules(high_min=0.9)) is None   # other bands: these rates don't apply
    assert load_rates("other", path) is None
    assert json.loads(path.read_text(encoding="utf-8"))["m1"]["notes"]
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_tier_rates.py -q`
Expected: FAIL at import (`No module named 'src.report.tier_rates'`).

- [ ] **Step 3: Write `src/report/tier_rates.py`**

```python
"""Measured rates for the verdict tiers (spec 2026-10-01, section 1).

    python -m src.report.tier_rates

For the registered model, reads the validation scores its calibration saved beside each model
(calibration_scores_<sha12>.npz) and the validation frame calibration built (validation rows plus
sentence-aligned excerpts), routes each text as the service routes it, maps it through that model's Platt
parameters and tiers it with the record's bands. Writes, per tier, the share of human and of AI validation
texts in it, for 25-100 words and for all lengths with a bar, to data/report/tier_rates.json keyed by
model_id. No GPU; only validation rows are read.
"""

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from ..authorship import registry
from ..authorship.calibration import probability
from .tiers import HIGH, LIKELY_AI, LOW, MODERATE, SECOND_LOOK, TierRules

logger = logging.getLogger(__name__)

RATES_FILE = Path(__file__).resolve().parents[2] / "data" / "report" / "tier_rates.json"
RATED_TIERS = (LIKELY_AI, HIGH, MODERATE, SECOND_LOOK, LOW)
SLICES = {"25-100": (25, 100), "all": (0, None)}
NOTES = (
    "Rates are measured on public validation writing, not on rater rationales; rater text may differ.",
    "Validation includes sentence-aligned excerpts of longer texts, as calibration did.",
    "Later-window vetoes on long texts and disguised-character handling are not replayed.",
)


def scored_frame(val: pd.DataFrame, main: dict, short: Optional[dict], record: dict) -> pd.DataFrame:
    """val (sample_id, y, words) with each text's probability from the model the service would route it to."""
    def probs(cache: dict, platt) -> pd.Series:
        series = pd.Series(probability(cache["raw"], *platt), index=np.asarray(cache["ids"], dtype=str))
        if not series.index.is_unique:
            raise RuntimeError("a score cache repeats a sample id")
        return series

    ranges = [tuple(r) for r in (record.get("short_model") or {}).get("ranges", [])]
    routed = val["words"].map(lambda w: any(lo <= w < (hi if hi is not None else np.inf) for lo, hi in ranges))
    prob = val["sample_id"].map(probs(main, record["calibration"]["platt"]))
    if short is not None and routed.any():
        prob = prob.where(~routed, val["sample_id"].map(probs(short, record["short_model"]["platt"])))
    if prob.isna().any():
        raise RuntimeError(f"{int(prob.isna().sum())} validation texts have no saved score; re-run calibration")
    return val.assign(prob=prob.to_numpy(dtype=float))


def rate_table(frame: pd.DataFrame, rules: TierRules, policy) -> dict:
    """Per slice: how many human and AI texts have a bar, and each tier's share of them."""
    bars = frame["words"].map(lambda w: getattr(policy.band_for(int(w)), "ai_min_score", np.nan))
    tiers = [LIKELY_AI if p >= b else rules.by_score(p) for p, b in zip(frame["prob"], bars.fillna(np.inf))]
    frame = frame.assign(bar=bars, tier=tiers)
    out = {}
    for name, (lo, hi) in SLICES.items():
        part = frame[frame["bar"].notna() & (frame["words"] >= lo) & ((frame["words"] < hi) if hi else True)]
        humans, ais = part[part["y"] == 0], part[part["y"] == 1]
        share = lambda group, tier: round(float((group["tier"] == tier).mean()), 4) if len(group) else None  # noqa: E731
        out[name] = {"human_texts": int(len(humans)), "ai_texts": int(len(ais)),
                     "tiers": {t: {"human": share(humans, t), "ai": share(ais, t)} for t in RATED_TIERS}}
    return out


def validation_frame(record: dict) -> pd.DataFrame:
    """The validation frame calibration scored: validation rows plus their excerpts (sample_id, y, words)."""
    from ..authorship.calibrate import load_split, with_excerpts
    from ..corpus.paths import VERSION
    from ..corpus.text import MODEL_TEXT_VIEW

    if record.get("corpus_version") != VERSION or record.get("text_normalization") != MODEL_TEXT_VIEW:
        raise RuntimeError("rates can only be rebuilt for a model calibrated on this branch's corpus and text view")
    val = load_split("val")
    val = pd.concat([val.assign(excerpt=False), with_excerpts(val)], ignore_index=True)
    return val[["sample_id", "y", "words"]]


def _cache(model_dir: Path, sha: str) -> Tuple[dict, str]:
    path = Path(model_dir).parent / f"calibration_scores_{sha[:12]}.npz"
    with np.load(path) as saved:
        cache = {"ids": saved["val_ids"].astype(str), "raw": saved["val_raw"].astype(float)}
    return cache, registry.sha256_file(path)


def saved_scores(record: dict) -> Tuple[dict, Optional[dict], dict]:
    """(main cache, short cache or None, {name: file sha256}) for the record's models."""
    main, main_sha = _cache(registry.artifact_dir(record), record["artifact"]["sha256"])
    parts = registry.short_parts(record)
    if not parts:
        return main, None, {"main": main_sha}
    short, short_sha = _cache(parts[0], record["short_model"]["artifact"]["sha256"])
    return main, short, {"main": main_sha, "short": short_sha}


def build(record: Optional[dict] = None, path: Path = RATES_FILE, rules: TierRules = TierRules()) -> dict:
    record = record or registry.load()
    if not record:
        raise RuntimeError("no registered model")
    main, short, sources = saved_scores(record)
    frame = scored_frame(validation_frame(record), main, short, record)
    entry = {"model_id": record["model_id"], "measured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
             "bands": rules.as_dict(), "slices": rate_table(frame, rules, registry.build_policy(record)),
             "sources": sources, "notes": list(NOTES)}
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    data[record["model_id"]] = entry
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return entry


def load_rates(model_id: str, path: Path = RATES_FILE, rules: Optional[TierRules] = None) -> Optional[dict]:
    """The model's measured rates, or None when they were not measured (or were measured for other bands)."""
    if not Path(path).exists():
        return None
    entry = json.loads(Path(path).read_text(encoding="utf-8")).get(model_id)
    if entry and rules is not None and entry.get("bands") != rules.as_dict():
        return None
    return entry


def main(argv=None) -> None:
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    entry = build()
    for name, part in entry["slices"].items():
        print(f"{name}: {part['human_texts']} human, {part['ai_texts']} AI texts")
        for tier, rate in part["tiers"].items():
            print(f"  {tier:28s} human {rate['human']:.4f}  AI {rate['ai']:.4f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_tier_rates.py -q`
Expected: all pass.

- [ ] **Step 5: Measure the registered 4d model (CPU only)**

Run: `CUDA_VISIBLE_DEVICES=-1 python -m src.report.tier_rates`
Expected: prints two slices with roughly 15-20k human and 25-40k AI texts each; `data/report/tier_rates.json` holds `deberta-v3-large-lora-authorship-4d`. Sanity check: in "25-100", Likely AI human share is at or under about 0.01 (the bar's contract); if it is far above, stop and investigate routing before committing.

- [ ] **Step 6: Commit**

```bash
git add src/report/tier_rates.py tests/test_tier_rates.py data/report/tier_rates.json
git commit -m "feat: measured tier rates for the 4d model from its saved validation scores"
```

---

### Task 3: Trigger sentences

**Files:**
- Create: `src/report/triggers.py`
- Test: `tests/test_triggers.py`

**Interfaces:**
- Consumes: `deobfuscate` (sealed, import only); `LengthRoutedDetector`, `AuthorshipModelDetector`, `WINDOW_WORDS` from `src.authorship.detector` (import only); `probability`.
- Produces: `JOIN = "\n\n"`; `Sentence(column: str, start: int, end: int, text: str, word_start: int)`; `split_sentences(segments: Sequence[Tuple[str, str]]) -> List[Sentence]`; `without(segments, sentence) -> str`; `TriggerResult(kind, baseline=None, sentences=[], drops=[], note="")` with `.summary() -> str` (kinds: "sentences", "spread", "single", "skipped", "error"); `scorer_for(detector, text) -> Optional[Callable[[List[str]], np.ndarray]]`; `TriggerFinder(detector, min_drop=0.10, max_sentences=12, top=2, window_words=WINDOW_WORDS)` with `.explain(segments) -> TriggerResult`.

- [ ] **Step 1: Write the failing tests** — `tests/test_triggers.py`:

```python
"""Trigger sentences (spec 2026-10-01, section 2): the sentences whose removal lowers the score most."""
import numpy as np
import pytest

from src.authorship.detector import LengthRoutedDetector
from src.detectors import GPTZeroDetector
from src.detectors.base import BaseDetector, DetectionResult
from src.report.triggers import JOIN, TriggerFinder, TriggerResult, scorer_for, split_sentences, without

PLANT = "We must delve into the rich tapestry of this response."


class Planted(BaseDetector):
    """0.95 when the planted sentence is present, else 0.20; records every text it scores."""

    def __init__(self):
        super().__init__()
        self.seen = []

    def detect(self, text):
        self.seen.append(text)
        score = 0.95 if "rich tapestry" in text else 0.20
        return DetectionResult(provider="planted", ai_score=score, is_flagged=score >= 0.5)


def test_sentences_split_at_sentence_ends_and_line_breaks_and_map_back_to_cells():
    segs = [("first", "The answer is right. It cites the source!"), ("second", "- point one\n- point two")]
    got = split_sentences(segs)
    assert [(s.column, s.text) for s in got] == [
        ("first", "The answer is right."), ("first", "It cites the source!"),
        ("second", "- point one"), ("second", "- point two")]
    for s in got:
        assert dict(segs)[s.column][s.start:s.end] == s.text
    assert [s.word_start for s in got] == [0, 4, 8, 11]


def test_no_split_inside_abbreviations_or_lowercase_continuations():
    got = split_sentences([("c", "It follows e.g. the rubric. then it ends")])
    assert [s.text for s in got] == ["It follows e.g. the rubric. then it ends"]


def test_a_text_without_punctuation_is_one_sentence_and_empty_cells_give_none():
    assert [s.text for s in split_sentences([("c", "no punctuation at all here")])] == ["no punctuation at all here"]
    assert split_sentences([("c", ""), ("d", "   ")]) == []


def test_removing_a_sentence_keeps_the_rest_of_the_row_joined_as_the_pipeline_joins_it():
    segs = [("a", "One. Two."), ("b", "Three.")]
    first = split_sentences(segs)[0]
    assert without(segs, first) == "Two." + JOIN + "Three."
    last = split_sentences(segs)[2]
    assert without(segs, last) == "One. Two."


def test_the_planted_sentence_is_found_with_its_drop():
    finder = TriggerFinder(Planted())
    result = finder.explain([("why", f"The rating is fair. {PLANT} It matches the rubric.")])
    assert result.kind == "sentences"
    (sentence, drop), = result.sentences
    assert sentence.text == PLANT and drop == pytest.approx(0.75)
    assert result.baseline == pytest.approx(0.95)
    assert len(result.drops) == 3
    assert result.summary() == "Without it: 95% -> 20%"


def test_a_signal_spread_across_the_text_names_no_sentence():
    class Flat(Planted):
        def detect(self, text):
            return DetectionResult(provider="flat", ai_score=0.9, is_flagged=True)

    result = TriggerFinder(Flat()).explain([("why", "One thing. Another thing. A third thing.")])
    assert result.kind == "spread" and result.sentences == []
    assert result.summary().startswith("Spread across the text")


def test_a_single_sentence_is_the_whole_text():
    result = TriggerFinder(Planted()).explain([("why", PLANT)])
    assert result.kind == "single" and result.summary() == "One sentence: the whole text"


def test_only_sentences_inside_the_model_window_are_tried_and_at_most_max_sentences():
    long_text = " ".join(f"Sentence number {i} is here." for i in range(40))   # 5 words each
    detector = Planted()
    result = TriggerFinder(detector, max_sentences=12, window_words=30).explain([("c", long_text)])
    assert len(result.drops) == 6                       # sentences starting at words 0..25
    result = TriggerFinder(detector, max_sentences=3).explain([("c", long_text)])
    assert len(result.drops) == 3


def test_paid_apis_are_never_called():
    assert scorer_for(GPTZeroDetector(api_key="x"), "text") is None
    assert TriggerFinder(GPTZeroDetector(api_key="x")).explain([("c", "One. Two.")]).kind == "skipped"


def test_the_routed_detector_scores_with_the_model_that_scored_the_row():
    class Model(BaseDetector):
        def __init__(self, name):
            super().__init__()
            self.name, self.calls = name, 0

        def detect(self, text):
            self.calls += 1
            return DetectionResult(provider=self.name, ai_score=0.5, is_flagged=False)

    short, main = Model("short"), Model("main")
    routed = LengthRoutedDetector(main, short, [(25, 100)])
    scorer_for(routed, " ".join(["word"] * 30))(["a", "b"])       # 30 words: the short model's range
    assert (short.calls, main.calls) == (2, 0)
    scorer_for(routed, " ".join(["word"] * 150))(["a"])
    assert main.calls == 1


def test_the_model_path_uses_one_truncated_pass_and_the_models_own_probability_map():
    class FakeModel:  # duck-typed like AuthorshipModelDetector: raw_and_embeddings and platt
        platt = (1.0, 0.0)

        def raw_and_embeddings(self, texts):
            return np.zeros(len(texts)), np.zeros((len(texts), 2))

    assert list(scorer_for(FakeModel(), "text")(["a", "b"])) == [0.5, 0.5]
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_triggers.py -q`
Expected: FAIL at import (`No module named 'src.report.triggers'`).

- [ ] **Step 3: Write `src/report/triggers.py`**

```python
"""Trigger sentences: which sentences drive a high AI score (spec 2026-10-01, section 2).

The model scores a whole text, so nothing inside it points at a sentence. This module removes one
sentence at a time, re-scores the text with the model that scored the row, and reports the sentences
whose removal lowers the score most. It explains a score; it never changes a decision. Indicative only:
how well these explanations match what a careful reader would pick has not been measured.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from ..authorship.calibration import probability
from ..authorship.detector import WINDOW_WORDS, LengthRoutedDetector
from ..detection.obfuscation import deobfuscate
from ..detectors import GPTZeroDetector, OriginalityDetector

logger = logging.getLogger(__name__)

JOIN = "\n\n"  # how the pipeline joins a row's cells into the row's combined text
# A sentence ends at . ! ? or an ellipsis (with any closing quotes or brackets) when whitespace and an
# uppercase letter, digit or opening quote follow; a line break always ends one.
_END = re.compile(r"[.!?…]+[\"'”’)\]]*(?=\s+[\"'(\[“‘A-Z0-9])")


@dataclass(frozen=True)
class Sentence:
    column: str      # the cell it came from
    start: int       # character offsets in that cell's (stripped) text
    end: int
    text: str
    word_start: int  # words before it in the row's decided text


def _spans(text: str) -> List[Tuple[int, int]]:
    spans = []
    for line in re.finditer(r"[^\n]+", text):
        base, body, begin = line.start(), line.group(0), 0
        for cut in [m.end() for m in _END.finditer(body)] + [len(body)]:
            piece = body[begin:cut]
            if piece.strip():
                lead = len(piece) - len(piece.lstrip())
                spans.append((base + begin + lead, base + begin + len(piece.rstrip())))
            begin = cut
    return spans


def split_sentences(segments: Sequence[Tuple[str, str]]) -> List[Sentence]:
    """The sentences of a row's decided text, in order, each mapped back to its cell."""
    out, words = [], 0
    for column, text in segments:
        for start, end in _spans(text or ""):
            piece = text[start:end]
            out.append(Sentence(column, start, end, piece, words))
            words += len(piece.split())
    return out


def without(segments: Sequence[Tuple[str, str]], sentence: Sentence) -> str:
    """The decided text with one sentence removed, cells joined as the pipeline joins them."""
    cells = []
    for column, text in segments:
        if column == sentence.column:
            text = (text[:sentence.start] + " " + text[sentence.end:]).strip()
        if text:
            cells.append(text)
    return JOIN.join(cells)


@dataclass
class TriggerResult:
    kind: str                                   # "sentences", "spread", "single", "skipped" or "error"
    baseline: Optional[float] = None            # the decided text's score, re-scored here
    sentences: List[Tuple[Sentence, float]] = field(default_factory=list)  # triggers, with their drops
    drops: List[Tuple[Sentence, float]] = field(default_factory=list)      # every sentence tried
    note: str = ""

    def summary(self) -> str:
        if self.kind == "sentences":
            parts = [f"{self.baseline:.0%} -> {self.baseline - drop:.0%}" for _, drop in self.sentences]
            return "Without it: " + parts[0] if len(parts) == 1 else (
                "Without the first: " + parts[0] + "; without the second: " + parts[1])
        if self.kind == "spread":
            return "Spread across the text: no single sentence lowers the score by 10 points or more"
        if self.kind == "single":
            return "One sentence: the whole text"
        return self.note


def scorer_for(detector, text: str) -> Optional[Callable[[List[str]], np.ndarray]]:
    """A function scoring texts with the model that scored `text` (its deobfuscated form): the routed
    detector's short or main model, each through its own probability map in one truncated pass per text,
    as calibration scored. Other local detectors use their own scores. None for paid APIs."""
    if isinstance(detector, (GPTZeroDetector, OriginalityDetector)):
        return None
    if isinstance(detector, LengthRoutedDetector):
        detector = detector.short if detector.serves_short(text) else detector.main
    if hasattr(detector, "raw_and_embeddings") and hasattr(detector, "platt"):
        def score(texts: List[str]) -> np.ndarray:
            raw, _ = detector.raw_and_embeddings(list(texts))
            return np.clip(probability(raw, *detector.platt), 0.0, 1.0)
        return score
    return lambda texts: np.array([r.ai_score for r in detector.detect_batch(list(texts))], dtype=float)


class TriggerFinder:
    def __init__(self, detector, min_drop: float = 0.10, max_sentences: int = 12, top: int = 2,
                 window_words: int = WINDOW_WORDS) -> None:
        self.detector, self.min_drop, self.max_sentences = detector, min_drop, max_sentences
        self.top, self.window_words = top, window_words

    def explain(self, segments: Sequence[Tuple[str, str]]) -> TriggerResult:
        segments = [(c, t) for c, t in segments if t and t.strip()]
        sentences = split_sentences(segments)
        if not sentences:
            return TriggerResult("skipped", note="not run (no text)")
        full = deobfuscate(JOIN.join(t for _, t in segments)).text
        scorer = scorer_for(self.detector, full)
        if scorer is None:
            return TriggerResult("skipped", note="not run (paid API)")
        if len(sentences) == 1:
            return TriggerResult("single", sentences=[(sentences[0], 0.0)])
        # Text past the model's window never reached the score, so only sentences starting inside it are tried.
        tried = [s for s in sentences if s.word_start < self.window_words][: self.max_sentences]
        scores = scorer([full] + [deobfuscate(without(segments, s)).text for s in tried])
        baseline = float(scores[0])
        drops = [(s, baseline - float(v)) for s, v in zip(tried, scores[1:])]
        ranked = sorted((d for d in drops if d[1] >= self.min_drop), key=lambda d: -d[1])[: self.top]
        return TriggerResult("sentences" if ranked else "spread", baseline=baseline, sentences=ranked, drops=drops)
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_triggers.py -q`
Expected: all pass. If `test_sentences_split...` fails on word starts, recount: "The answer is right." = 4 words, "It cites the source!" = 4, "- point one" = 3 -> starts 0, 4, 8, 11.

- [ ] **Step 5: Commit**

```bash
git add src/report/triggers.py tests/test_triggers.py
git commit -m "feat: trigger sentences - the sentences whose removal lowers a high AI score most"
```

---

### Task 4: Row reports in the pipeline

**Files:**
- Create: `src/report/rows.py`
- Modify: `src/pipeline.py` (constructor, row loop, summary)
- Test: `tests/test_row_reports.py`

**Interfaces:**
- Consumes: Tasks 1-3.
- Produces: `RowReport(position, task_id, agent, result, tier, reason, bar, cells, decided, combined=None, trigger=None, proxy_warnings=[])` with properties `score`, `master_row` (position + 2), `decided_on` (column name or "all columns together"); `decided_cells(cells, row_result, combined_checked: bool) -> List[Tuple[str, str]]`; `tier_counts(reports) -> Dict[str, int]`; `agent_table(reports) -> List[dict]` (keys `agent, rows, likely_ai, high, moderate, not_checked, highest_score, share_likely_ai`). Pipeline: `QAPipeline.tier_rules`, `QAPipeline.trigger_finder` (None when off), `QAPipeline.last_reports: List[RowReport]`; summary keys `tier_counts`, `agents`, `detector_version`, `tier_rates`, `triggers` (`{"enabled", "rows", "seconds"}`).

- [ ] **Step 1: Write the failing tests** — `tests/test_row_reports.py`:

```python
"""Row reports: the pipeline's one-per-row record of tier, bar, decided text and trigger (spec 2026-10-01)."""
import pandas as pd

from src.config import PipelineConfig
from src.detection import AuthorshipService
from src.detection.policy import BandRule, DecisionPolicy
from src.detectors.base import BaseDetector, DetectionResult
from src.pipeline import QAPipeline
from src.report.rows import agent_table, decided_cells, tier_counts
from src.report.tiers import HIGH, LIKELY_AI, LOW, NOT_CHECKED
from src.report.triggers import TriggerResult

PLANT = "We must delve into the rich tapestry of this response."


class Planted(BaseDetector):
    def detect(self, text):
        score = 0.995 if "certainly" in text else 0.95 if "rich tapestry" in text else 0.20
        return DetectionResult(provider="planted", ai_score=score, is_flagged=score >= 0.5,
                               analysis={"evidence": ["planted"]})


def _service():
    policy = DecisionPolicy(min_word_count=5, bands=(BandRule(5, None, 0.05, 0.99),))
    return AuthorshipService(Planted(), policy=policy, model_validated=True, detector_version="planted-v1")


def _run(tmp_path, rows, cols=("why",), **trigger_cfg):
    src = tmp_path / "in.csv"
    pd.DataFrame(rows).to_csv(src, index=False)
    config = PipelineConfig()
    config.structural.text_columns = list(cols)
    for key, value in trigger_cfg.items():
        setattr(config.triggers, key, value)
    pipe = QAPipeline(config=config, authorship_service=_service())
    return pipe, pipe.run(str(src), str(tmp_path / "out.xlsx"))


ROWS = [
    {"rater_id": "ann", "task_id": "T1", "why": f"The rating is fair overall. {PLANT} It matches the rubric well."},
    {"rater_id": "bo", "task_id": "T2", "why": "The rating is fair and the response cites the source it used."},
    {"rater_id": "ann", "task_id": "T3", "why": "It is certainly a clear and complete answer to the question."},
    {"rater_id": "cy", "task_id": "T4", "why": "Too short."},
]


def test_each_row_gets_a_tier_a_bar_and_its_decided_text(tmp_path):
    pipe, res = _run(tmp_path, ROWS)
    tiers = [r.tier for r in pipe.last_reports]
    assert tiers == [HIGH, LOW, LIKELY_AI, NOT_CHECKED]
    assert pipe.last_reports[0].bar == 0.99 and pipe.last_reports[3].bar is None
    assert pipe.last_reports[3].reason == "too short"
    assert pipe.last_reports[0].decided == [("why", ROWS[0]["why"])]
    assert pipe.last_reports[0].task_id == "T1" and pipe.last_reports[0].agent == "ann"
    assert res.summary["tier_counts"][HIGH] == 1 and res.summary["tier_counts"][LIKELY_AI] == 1
    assert res.ai_flagged_count == 1                       # decisions unchanged


def test_triggers_are_found_for_high_and_likely_ai_rows_only(tmp_path):
    pipe, res = _run(tmp_path, ROWS)
    high, low = pipe.last_reports[0], pipe.last_reports[1]
    assert high.trigger.kind == "sentences" and high.trigger.sentences[0][0].text == PLANT
    assert low.trigger is None
    assert res.summary["triggers"]["enabled"] is True and res.summary["triggers"]["rows"] == 2


def test_triggers_can_be_switched_off(tmp_path):
    pipe, res = _run(tmp_path, ROWS, enabled=False)
    assert all(r.trigger is None for r in pipe.last_reports)
    assert res.summary["triggers"] == {"enabled": False, "rows": 0, "seconds": 0.0}


def test_a_trigger_error_never_fails_the_audit(tmp_path, monkeypatch):
    from src.report import triggers

    def boom(self, segments):
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(triggers.TriggerFinder, "explain", boom)
    pipe, res = _run(tmp_path, ROWS)
    assert pipe.last_reports[0].trigger.kind == "error"
    assert pipe.last_reports[0].trigger.summary() == "trigger not found (error)"


def test_agents_are_ranked_most_flagged_first(tmp_path):
    _, res = _run(tmp_path, ROWS)
    agents = res.summary["agents"]
    assert [a["agent"] for a in agents] == ["ann", "bo", "cy"]
    assert agents[0] == {"agent": "ann", "rows": 2, "likely_ai": 1, "high": 1, "moderate": 0, "not_checked": 0,
                         "highest_score": 0.995, "share_likely_ai": 0.5}
    assert agents[2]["not_checked"] == 1 and agents[2]["highest_score"] is None


def test_a_row_decided_on_its_combined_text_uses_both_cells_and_a_lone_second_cell_is_its_own(tmp_path):
    rows = [{"rater_id": "a", "first": "The rating is fair overall here.", "second": f"{PLANT} It matches."},
            {"rater_id": "b", "first": "", "second": f"The rating is fair overall. {PLANT}"}]
    pipe, _ = _run(tmp_path, rows, cols=("first", "second"))
    both, lone = pipe.last_reports
    assert [c for c, _ in both.decided] == ["first", "second"] and both.decided_on == "all columns together"
    assert lone.decided == [("second", rows[1]["second"])] and lone.decided_on == "second"
    assert lone.trigger.sentences[0][0].column == "second"


def test_decided_cells_prefers_a_cell_that_flagged_the_row_on_its_own():
    class Row:
        evidence = [type("E", (), {"code": "CELL_FLAGGED_AI"})()]
        likely_ai_columns = ["b"]
        column_scores = {"a": 0.2, "b": 0.995}

    assert decided_cells([("a", "x y"), ("b", "z w")], Row(), combined_checked=True) == [("b", "z w")]
    assert decided_cells([("a", ""), ("b", "")], Row(), combined_checked=False) == []


def test_tier_counts_and_agent_table_handle_no_rows():
    assert set(tier_counts([])) and all(v == 0 for v in tier_counts([]).values())
    assert agent_table([]) == []


def test_a_sheet_without_an_agent_column_reads_no_agent(tmp_path):
    pipe, res = _run(tmp_path, [{"why": "The rating is fair and the response cites the source it used."}])
    assert pipe.last_reports[0].agent == "(no agent)"
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_row_reports.py -q`
Expected: FAIL at import (`No module named 'src.report.rows'`).

- [ ] **Step 3: Write `src/report/rows.py`**

```python
"""One record per audited row (spec 2026-10-01, section 3): the workbook, the summary and the API read these
instead of parallel lists."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ..detection import RowAuthorshipResult, TextCheckResponse
from .tiers import HIGH, LIKELY_AI, MODERATE, NOT_CHECKED, TIER_ORDER
from .triggers import TriggerResult


@dataclass
class RowReport:
    position: int                       # 0-based row of the audited sheet (as written to Annotated Master)
    task_id: Optional[str]
    agent: str
    result: RowAuthorshipResult
    tier: str
    reason: Optional[str]               # why the row was not checked
    bar: Optional[float]                # the score its decided text needed for Likely AI
    cells: List[Tuple[str, str]]        # every text column's (stripped) cell, in text-column order
    decided: List[Tuple[str, str]]      # the cells whose text decided the row
    combined: Optional[TextCheckResponse] = None   # the row's cells checked together, when there were several
    trigger: Optional[TriggerResult] = None
    proxy_warnings: List[str] = field(default_factory=list)

    @property
    def score(self) -> Optional[float]:
        return self.result.score

    @property
    def master_row(self) -> int:
        return self.position + 2

    @property
    def decided_on(self) -> str:
        names = [c for c, _ in self.decided]
        return "all columns together" if len(names) > 1 else (names[0] if names else "")


def decided_cells(cells: List[Tuple[str, str]], row_result, combined_checked: bool) -> List[Tuple[str, str]]:
    """The cells whose text decided the row: every filled cell when the row was decided on its combined text,
    else the one cell whose result became the row's (a cell that flagged the row on its own, or the only one)."""
    filled = [(c, t) for c, t in cells if t]
    if not filled:
        return []
    flagged_alone = any(getattr(e, "code", None) == "CELL_FLAGGED_AI" for e in row_result.evidence)
    if combined_checked and not flagged_alone:
        return filled
    pool = [(c, t) for c, t in filled if c in row_result.likely_ai_columns] or filled
    score = lambda ct: row_result.column_scores.get(ct[0]) if row_result.column_scores.get(ct[0]) is not None else -1.0  # noqa: E731
    return [max(pool, key=score)]


def tier_counts(reports: List[RowReport]) -> Dict[str, int]:
    return {tier: sum(r.tier == tier for r in reports) for tier in TIER_ORDER}


def agent_table(reports: List[RowReport]) -> List[dict]:
    """One row per agent, most flagged first: Likely AI, then High AI score, then highest score."""
    table: Dict[str, dict] = {}
    for r in reports:
        a = table.setdefault(r.agent, {"agent": r.agent, "rows": 0, "likely_ai": 0, "high": 0, "moderate": 0,
                                       "not_checked": 0, "highest_score": None, "share_likely_ai": 0.0})
        a["rows"] += 1
        a["likely_ai"] += r.tier == LIKELY_AI
        a["high"] += r.tier == HIGH
        a["moderate"] += r.tier == MODERATE
        a["not_checked"] += r.tier == NOT_CHECKED
        if r.score is not None and (a["highest_score"] is None or r.score > a["highest_score"]):
            a["highest_score"] = round(float(r.score), 4)
    for a in table.values():
        a["share_likely_ai"] = round(a["likely_ai"] / a["rows"], 4)
    return sorted(table.values(), key=lambda a: (-a["likely_ai"], -a["high"],
                                                 -(a["highest_score"] if a["highest_score"] is not None else -1.0),
                                                 a["agent"]))
```

- [ ] **Step 4: Wire the pipeline** — in `src/pipeline.py`:

Imports (after the `review_tags` import, which stays until Task 5):

```python
from .report.rows import RowReport, agent_table, decided_cells, tier_counts
from .report.tier_rates import load_rates
from .report.tiers import EXPLAINED, NOT_CHECKED, TierRules, bar_for
from .report.triggers import JOIN, TriggerFinder, TriggerResult
```

In `__init__`, after `self.review_tag_rules = ...`:

```python
        self.tier_rules = TierRules.from_config(self.config.tiers)
        trig = self.config.triggers
        self.trigger_finder: Optional[TriggerFinder] = (
            TriggerFinder(self.authorship_service.detector, min_drop=trig.min_drop,
                          max_sentences=trig.max_sentences, top=trig.top) if trig.enabled else None)
        self.last_reports: List[RowReport] = []
```

Add two methods before `run`:

```python
    @staticmethod
    def _cell(df: pd.DataFrame, idx: int, column: Optional[str]) -> Optional[str]:
        if not column or column not in df.columns or pd.isna(df.loc[idx, column]):
            return None
        return str(df.loc[idx, column]).strip() or None

    def _find_triggers(self, reports: List[RowReport], progress_callback=None) -> Dict[str, Any]:
        """Trigger sentences for rows in the explained tiers (spec 2026-10-01). A failure on one row is noted
        on that row and never fails the audit."""
        if self.trigger_finder is None:
            return {"enabled": False, "rows": 0, "seconds": 0.0}
        wanted = [r for r in reports if r.tier in EXPLAINED and r.decided]
        started = time.perf_counter()
        for n, report in enumerate(wanted):
            if progress_callback and n % 10 == 0:
                progress_callback(85 + int(5 * n / max(1, len(wanted))), 100,
                                  f"Finding trigger sentences {n + 1}/{len(wanted)}...")
            try:
                report.trigger = self.trigger_finder.explain(report.decided)
            except Exception as exc:  # e.g. CUDA out of memory: note it on the row, keep auditing
                logger.warning("trigger sentences failed on row %d: %s", report.master_row, exc.__class__.__name__)
                report.trigger = TriggerResult("error", note="trigger not found (error)")
        return {"enabled": True, "rows": len(wanted), "seconds": round(time.perf_counter() - started, 2)}
```

In `run`, before the row loop add `reports: List[RowReport] = []`. In the loop, replace the `row_texts` block:

```python
            # The row's rationales checked together decide the row; cells stay as detail.
            cells = [(c, "" if pd.isna(df.loc[idx, c]) else str(df.loc[idx, c]).strip())
                     for c in text_cols if c in df.columns]
            row_texts = [t for _, t in cells if t]
            row_det = self._cached_detect(JOIN.join(row_texts)) if len(row_texts) > 1 else None
            row_result = combine_column_results(col_results, row_result=row_det)
```

and after `proxy_results.append(prox)` add:

```python
            tier, reason = self.tier_rules.tier(row_result)
            decided = decided_cells(cells, row_result, combined_checked=row_det is not None)
            reports.append(RowReport(
                position=idx, task_id=self._cell(df, idx, row_id_col),
                agent=self._cell(df, idx, rater_col) or "(no agent)", result=row_result, tier=tier, reason=reason,
                bar=None if tier == NOT_CHECKED else bar_for(self.authorship_service.policy,
                                                             JOIN.join(t for _, t in decided)),
                cells=cells, decided=decided, combined=row_det, proxy_warnings=list(prox.warnings)))
```

After the loop (before `ai_flagged_total = ...`):

```python
        trigger_stats = self._find_triggers(reports, progress_callback)
        self.last_reports = reports
```

After `summary["verdict_column"] = verdict_col`:

```python
        summary["detector_version"] = self.authorship_service.detector_version
        summary["tier_counts"] = tier_counts(reports)
        summary["agents"] = agent_table(reports)
        summary["tier_rates"] = load_rates(summary["detector_version"].split(":")[0], rules=self.tier_rules)
        summary["triggers"] = trigger_stats
```

- [ ] **Step 5: Run the new and existing tests**

Run: `python -m pytest tests/test_row_reports.py -q` then `python -m pytest -q -x`
Expected: new tests pass; the full suite passes (the old workbook is still written).

- [ ] **Step 6: Commit**

```bash
git add src/report/rows.py src/pipeline.py tests/test_row_reports.py
git commit -m "feat: one row report per audited row - tier, bar, decided text, trigger sentences, agents"
```

---

### Task 5: The v2 workbook

**Files:**
- Modify: `src/report/excel_writer.py` (rewrite), `src/pipeline.py` (writer calls; drop review tags), `src/config.py` (drop `ReviewTagConfig`, `review_tags`)
- Delete: `src/report/review_tags.py`, `tests/test_review_tags.py`
- Modify tests: `tests/test_multi_column.py`, `tests/test_pipeline.py`
- Test: `tests/test_workbook_v2.py`

**Interfaces:**
- Consumes: `RowReport`, `agent_table` output, `TierRules`, `bar_for`, `RATED_TIERS`, `JOIN`, `EXPLAINED`.
- Produces: `AuditedWorkbookBuilder(tier_rules=TierRules())` with `add_summary_sheet(summary, rates, warning=None)`, `add_agents_sheet(agents)`, `add_ai_review_sheet(reports)`, `add_annotated_master_sheet(df, reports, quality_results, row_issues_count, text_cols)`, `add_technical_sheet(reports, policy)`, `add_structural_issues_sheet(issues)`, `save(path)`; helpers `verdict_text(report)`, `rationale_cell(report)`, `trigger_text(report)`.

- [ ] **Step 1: Write the failing tests** — `tests/test_workbook_v2.py`:

```python
"""The v2 audit workbook (spec 2026-10-01, section 4)."""
import openpyxl
import pandas as pd
from openpyxl.cell.rich_text import CellRichText, TextBlock

from tests.test_row_reports import PLANT, _service
from src.config import PipelineConfig
from src.pipeline import QAPipeline
from src.report.tiers import HIGH, LIKELY_AI, LOW, NOT_CHECKED

ROWS = [
    {"rater_id": "ann", "task_id": "T1", "why": f"The rating is fair overall. {PLANT} It matches the rubric well."},
    {"rater_id": "bo", "task_id": "T2", "why": "The rating is fair and the response cites the source it used."},
    {"rater_id": "ann", "task_id": "T3", "why": "It is certainly a clear and complete answer to the question."},
    {"rater_id": "cy", "task_id": "T4", "why": "Too short."},
]


def _audit(tmp_path, rows=ROWS, cols=("why",)):
    src, out = tmp_path / "in.csv", tmp_path / "out.xlsx"
    pd.DataFrame(rows).to_csv(src, index=False)
    config = PipelineConfig()
    config.structural.text_columns = list(cols)
    config.structural.skip_unassigned = False
    QAPipeline(config=config, authorship_service=_service()).run(str(src), str(out))
    return out


def _headers(ws):
    return [c.value for c in ws[1]]


def test_sheets_in_order_and_no_structural_sheet_without_issues(tmp_path):
    wb = openpyxl.load_workbook(_audit(tmp_path, ROWS[:3]))
    assert wb.sheetnames == ["Summary Dashboard", "Agents", "AI Review", "Annotated Master", "Technical"]


def test_ai_review_has_one_verdict_the_bar_and_the_full_rationale_strongest_first(tmp_path):
    ws = openpyxl.load_workbook(_audit(tmp_path))["AI Review"]
    h = _headers(ws)
    assert h == ["Row #", "Task ID", "Agent", "Verdict", "AI Score", "Bar", "Decided On", "Rationale", "Trigger",
                 "Each Column Alone", "Human Reviewer Decision", "Reviewer Notes"]
    col = {name: i + 1 for i, name in enumerate(h)}
    verdicts = [ws.cell(r, col["Verdict"]).value for r in range(2, 6)]
    assert verdicts == [LIKELY_AI, HIGH, LOW, f"{NOT_CHECKED} (too short)"]
    assert ws.cell(3, col["Bar"]).value == 0.99 and ws.cell(3, col["Task ID"]).value == "T1"
    assert ws.cell(3, col["Rationale"]).value == ROWS[0]["why"]          # full text, not cut
    assert ws.cell(3, col["Trigger"]).value == "Without it: 95% -> 20%"
    assert "Review Tag" not in h and "Detector Version" not in h


def test_the_trigger_sentence_is_bold_red_inside_the_rationale(tmp_path):
    ws = openpyxl.load_workbook(_audit(tmp_path), rich_text=True)["AI Review"]
    col = _headers(ws).index("Rationale") + 1
    cell = ws.cell(3, col).value
    assert isinstance(cell, CellRichText)
    red = [b for b in cell if isinstance(b, TextBlock) and b.font.b]
    assert [b.text for b in red] == [PLANT]
    assert str(red[0].font.color.rgb).endswith("C00000")


def test_agents_sheet_ranks_agents(tmp_path):
    ws = openpyxl.load_workbook(_audit(tmp_path))["Agents"]
    assert _headers(ws)[:4] == ["Agent", "Rows", LIKELY_AI, HIGH]
    assert [ws.cell(r, 1).value for r in range(2, 5)] == ["ann", "bo", "cy"]


def test_master_has_the_short_qa_block(tmp_path):
    ws = openpyxl.load_workbook(_audit(tmp_path))["Annotated Master"]
    qa = [h for h in _headers(ws) if str(h).startswith("QA ")]
    assert qa[:7] == ["QA Verdict", "QA AI Score", "QA Bar", "QA Decided On", "QA Trigger Sentence",
                      "QA Quality Score", "QA Quality Flags"]
    col = _headers(ws).index("QA Trigger Sentence") + 1
    assert ws.cell(2, col).value == PLANT


def test_technical_sheet_has_a_line_per_scored_text_with_sentence_drops(tmp_path):
    rows = [{"rater_id": "a", "first": "The rating is fair overall here.", "second": f"{PLANT} It matches."}]
    ws = openpyxl.load_workbook(_audit(tmp_path, rows, cols=("first", "second")))["Technical"]
    texts = [ws.cell(r, 3).value for r in range(2, ws.max_row + 1)]
    assert texts == ["first", "second", "all columns together"]
    drops_col = _headers(ws).index("Sentence Drops") + 1
    assert "pts" in ws.cell(4, drops_col).value and ws.cell(2, drops_col).value in (None, "")


def test_summary_has_one_tier_table_with_rates_or_a_not_measured_note(tmp_path):
    ws = openpyxl.load_workbook(_audit(tmp_path))["Summary Dashboard"]
    text = [str(c.value) for row in ws.iter_rows() for c in row if c.value is not None]
    assert HIGH in text and "Agents with any Likely AI" in text
    assert "not measured for this model" in text                       # planted-v1 has no measured rates
    assert "Likely Human-written" not in text and "Review Tags (strongest first)" not in text


def test_a_formula_like_rationale_with_a_trigger_stays_text(tmp_path):
    rows = [{"rater_id": "a", "why": f"=SUM(1,2) is what the rater typed. {PLANT} It ends here."}]
    wb = openpyxl.load_workbook(_audit(tmp_path, rows))
    assert all(c.data_type != "f" for ws in wb.worksheets for row in ws.iter_rows() for c in row)
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_workbook_v2.py -q`
Expected: FAIL (sheet names and headers are the old ones).

- [ ] **Step 3: Rewrite `src/report/excel_writer.py`**

```python
"""Audit workbook (v2, spec 2026-10-01): Summary Dashboard, Agents, AI Review, Annotated Master, Technical,
and Structural Issues when there are any. One verdict tier per row; trigger sentences in bold red."""
import logging
from typing import Any, Dict, List, Optional

import openpyxl
import pandas as pd
from openpyxl.cell.rich_text import CellRichText, TextBlock
from openpyxl.cell.text import InlineFont
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation

from ..qa.quality import QualityAssessment
from ..qa.structural import StructuralIssue
from .rows import RowReport
from .tier_rates import RATED_TIERS
from .tiers import EXPLAINED, HIGH, HUMAN, LIKELY_AI, MODERATE, NOT_CHECKED, SECOND_LOOK, TIER_ORDER, TierRules, bar_for
from .triggers import JOIN, TriggerResult

logger = logging.getLogger(__name__)

EXCEL_CELL_LIMIT = 32_767


def _fill(rgb: str) -> PatternFill:
    return PatternFill(start_color=rgb, end_color=rgb, fill_type="solid")


NAVY_HEADER, GRAY_HEADER = _fill("1F4E78"), _fill("595959")
RED_FILL, YELLOW_FILL = _fill("FFC7CE"), _fill("FFEB9C")
TIER_FILLS = {LIKELY_AI: _fill("F4A7A7"), HIGH: _fill("FFC7CE"), MODERATE: _fill("FFEB9C"),
              SECOND_LOOK: _fill("FFF2CC"), NOT_CHECKED: _fill("D9D9D9"), HUMAN: _fill("C6EFCE")}
FONT_HEADER = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
FONT_BOLD = Font(name="Calibri", size=11, bold=True)
FONT_DEFAULT = Font(name="Calibri", size=10)
FONT_NOTE = Font(name="Calibri", size=10, italic=True)
TRIGGER_FONT = InlineFont(b=True, color="C00000")
LABEL_FONT = InlineFont(b=True)
THIN_BORDER = Border(*(Side(style="thin", color="D9D9D9"),) * 4)
WRAP = Alignment(wrap_text=True, vertical="top")
REVIEW_CHOICES = '"Approved (Human),Confirmed AI,False Positive,Needs Discussion"'
RATE_COLUMNS = (("25-100", "human", "Human texts, 25-100 words"), ("25-100", "ai", "AI texts, 25-100 words"),
                ("all", "human", "Human texts, all lengths"), ("all", "ai", "AI texts, all lengths"))
NOTES = (
    "Likely AI is the only decision. Its bar depends on the text's length (see the Bar column in AI Review) and "
    "is set so that under 1% of human writing in validation clears it.",
    "The other tiers rank rows for review by AI score; they are not decisions.",
    "Rater rationales are never called Likely human: missed AI on rater text has not been measured, so a low "
    "score is not proof of human writing.",
    "Trigger sentences are indicative: the model scores the whole text, and the highlighted sentence is the one "
    "whose removal lowers the score most.",
)


def verdict_text(report: RowReport) -> str:
    return f"{report.tier} ({report.reason})" if report.reason else report.tier


def trigger_text(report: RowReport) -> str:
    if report.trigger is None:
        return "not run" if report.tier in EXPLAINED else ""
    return report.trigger.summary()


def trigger_sentences(report: RowReport) -> str:
    if report.trigger is not None and report.trigger.kind == "sentences":
        return " | ".join(s.text for s, _ in report.trigger.sentences)
    return trigger_text(report)


def drops_text(trigger: Optional[TriggerResult]) -> str:
    if trigger is None or not trigger.drops:
        return ""
    return "\n".join(f"{-drop * 100:+.0f} pts: [{s.column}] {s.text[:80]}" for s, drop in trigger.drops)


def rationale_cell(report: RowReport):
    """The decided text, each cell labelled when there are several, with trigger sentences in bold red."""
    marks: Dict[str, list] = {}
    if report.trigger is not None and report.trigger.kind == "sentences":
        for sentence, _ in report.trigger.sentences:
            marks.setdefault(sentence.column, []).append((sentence.start, sentence.end))
    parts: list = []
    for i, (column, text) in enumerate(report.decided):
        if i:
            parts.append(JOIN)
        if len(report.decided) > 1:
            parts.append(TextBlock(LABEL_FONT, f"[{column}] "))
        pos = 0
        for start, end in sorted(marks.get(column, [])):
            parts += [text[pos:start], TextBlock(TRIGGER_FONT, text[start:end])]
            pos = end
        parts.append(text[pos:])
    plain = "".join(p if isinstance(p, str) else p.text for p in parts)
    if len(plain) > EXCEL_CELL_LIMIT or not any(isinstance(p, TextBlock) for p in parts):
        return plain[:EXCEL_CELL_LIMIT]
    return CellRichText(*[p for p in parts if (p if isinstance(p, str) else p.text)])


def column_scores(report: RowReport, rules: TierRules) -> str:
    results = report.result.column_results
    if len(results) < 2:
        return ""
    lines = []
    for name, res in results.items():
        tier, reason = rules.tier(res)
        detail = f"{res.score:.0%}" if res.score is not None else reason
        lines.append(f"{name}: {tier} ({detail})")
    return "\n".join(lines)


def _header(ws, row: int, headers: List[str], fill: PatternFill = NAVY_HEADER) -> None:
    for col, name in enumerate(headers, 1):
        cell = ws.cell(row=row, column=col, value=name)
        cell.font, cell.fill = FONT_HEADER, fill
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)


def _widths(ws, widths: Dict[int, float]) -> None:
    for col, width in widths.items():
        ws.column_dimensions[get_column_letter(col)].width = width


def _autofit_columns(ws, max_len_cap: int = 50) -> None:
    for col in ws.columns:
        longest = max((len(str(c.value or "")) for c in col), default=0)
        ws.column_dimensions[get_column_letter(col[0].column)].width = min(max_len_cap, max(longest + 3, 12))


def _box(ws, row: int, ncols: int, wrap: bool = False) -> None:
    for col in range(1, ncols + 1):
        cell = ws.cell(row=row, column=col)
        cell.border = THIN_BORDER
        if wrap:
            cell.alignment = WRAP


class AuditedWorkbookBuilder:
    """Builds the audit workbook from the pipeline's row reports."""

    def __init__(self, tier_rules: TierRules = TierRules()):
        self.tier_rules = tier_rules
        self.wb = openpyxl.Workbook()
        self.wb.remove(self.wb.active)

    def add_summary_sheet(self, summary: Dict[str, Any], rates: Optional[dict], warning: Optional[str] = None):
        ws = self.wb.create_sheet(title="Summary Dashboard")
        ws.cell(row=1, column=1, value="EVALUATION QA AUDIT REPORT").font = Font(
            name="Calibri", size=16, bold=True, color="1F4E78")
        ws.cell(row=2, column=1, value="AI authorship tiers, trigger sentences and structural checks").font = FONT_NOTE
        if warning:  # e.g. an unvalidated preview: shown above every figure
            banner = ws.cell(row=3, column=1, value=warning)
            banner.font = Font(name="Calibri", size=11, bold=True, color="9C0006")
            banner.fill = RED_FILL
        agents = summary.get("agents") or []
        triggers = summary.get("triggers") or {}
        facts = [
            ("Model", summary.get("detector_version")),
            ("Locale", summary.get("locale") or "all rows"),
            ("Sheet", summary.get("detected_sheet")),
            ("Agent column", summary.get("rater_column")),
            ("Columns checked", ", ".join(summary.get("text_columns") or [])),
            ("Rows checked", summary.get("total_rows")),
            ("Agents with any Likely AI", f"{sum(1 for a in agents if a['likely_ai'])} of {len(agents)}"),
            ("Average justification quality", f"{(summary.get('avg_quality_score') or 0) * 100:.1f}%"),
            ("Trigger sentences", f"looked for in {triggers.get('rows', 0)} rows ({triggers.get('seconds', 0):.0f} s)"
             if triggers.get("enabled") else "off"),
        ]
        if summary.get("structural_errors_count") or summary.get("structural_warnings_count"):
            facts += [("Structural errors", summary.get("structural_errors_count", 0)),
                      ("Structural warnings", summary.get("structural_warnings_count", 0))]
        row = 5
        _header(ws, row, ["Run", "Value"])
        for label, value in facts:
            row += 1
            ws.cell(row=row, column=1, value=label).font = FONT_DEFAULT
            ws.cell(row=row, column=2, value=value).font = FONT_BOLD
            _box(ws, row, 2)

        row += 2
        ws.cell(row=row, column=1, value="Verdict tiers (strongest first)").font = FONT_BOLD
        row += 1
        _header(ws, row, ["Tier", "Rows", "Meaning"] + [label for _, _, label in RATE_COLUMNS], GRAY_HEADER)
        meanings, counts = self.tier_rules.meanings(), summary.get("tier_counts") or {}
        for tier in TIER_ORDER:
            if tier == HUMAN and not counts.get(tier):
                continue  # rater rationales are flag-only, so this row would always read 0
            row += 1
            tier_cell = ws.cell(row=row, column=1, value=tier)
            if tier in TIER_FILLS:
                tier_cell.fill = TIER_FILLS[tier]
            ws.cell(row=row, column=2, value=counts.get(tier, 0)).font = FONT_BOLD
            ws.cell(row=row, column=3, value=meanings[tier]).font = FONT_DEFAULT
            if tier in RATED_TIERS:
                if rates:
                    for col, (slice_name, who, _) in enumerate(RATE_COLUMNS, 4):
                        cell = ws.cell(row=row, column=col, value=rates["slices"][slice_name]["tiers"][tier][who])
                        cell.number_format = "0.0%"
                else:
                    ws.cell(row=row, column=4, value="not measured for this model").font = FONT_NOTE
            _box(ws, row, 3 + len(RATE_COLUMNS), wrap=True)

        row += 2
        notes = list(NOTES)
        if rates:
            every = rates["slices"]["all"]
            notes.append(f"Rates: the share of {every['human_texts']:,} human and {every['ai_texts']:,} AI validation "
                         f"texts that land in each tier (measured {rates['measured_at'][:10]}).")
            notes += rates.get("notes", [])
        for note in notes:
            ws.cell(row=row, column=1, value=note).font = FONT_NOTE
            row += 1
        _widths(ws, {1: 34, 2: 40, 3: 60, 4: 15, 5: 15, 6: 15, 7: 15})

    def add_agents_sheet(self, agents: List[dict]):
        ws = self.wb.create_sheet(title="Agents")
        headers = ["Agent", "Rows", LIKELY_AI, HIGH, MODERATE, NOT_CHECKED, "Highest AI Score", "Share Likely AI"]
        _header(ws, 1, headers)
        for row, a in enumerate(agents, 2):
            values = [a["agent"], a["rows"], a["likely_ai"], a["high"], a["moderate"], a["not_checked"],
                      a["highest_score"], a["share_likely_ai"]]
            for col, value in enumerate(values, 1):
                ws.cell(row=row, column=col, value=value).font = FONT_DEFAULT
            ws.cell(row=row, column=7).number_format = ws.cell(row=row, column=8).number_format = "0.0%"
            if a["likely_ai"]:
                ws.cell(row=row, column=3).fill = TIER_FILLS[LIKELY_AI]
            if a["high"]:
                ws.cell(row=row, column=4).fill = TIER_FILLS[HIGH]
            _box(ws, row, len(headers))
        ws.freeze_panes = "B2"
        ws.auto_filter.ref = f"A1:{get_column_letter(len(headers))}{len(agents) + 1}"
        _widths(ws, {1: 34, 2: 8, 3: 11, 4: 22, 5: 18, 6: 12, 7: 16, 8: 15})

    def add_ai_review_sheet(self, reports: List[RowReport]):
        """One row per task, strongest first, with the full decided text and its trigger sentences."""
        ws = self.wb.create_sheet(title="AI Review")
        with_ids = any(r.task_id for r in reports)
        headers = (["Row #"] + (["Task ID"] if with_ids else []) +
                   ["Agent", "Verdict", "AI Score", "Bar", "Decided On", "Rationale", "Trigger", "Each Column Alone",
                    "Human Reviewer Decision", "Reviewer Notes"])
        _header(ws, 1, headers)
        col = {name: i for i, name in enumerate(headers, 1)}
        choices = DataValidation(type="list", formula1=REVIEW_CHOICES, allow_blank=True)
        ws.add_data_validation(choices)
        order = sorted(reports, key=lambda r: (TIER_ORDER.index(r.tier),
                                               -(r.score if r.score is not None else -1.0), r.position))
        for row, rep in enumerate(order, 2):
            values = {"Row #": rep.master_row, "Task ID": rep.task_id, "Agent": rep.agent,
                      "Verdict": verdict_text(rep), "AI Score": rep.score, "Bar": rep.bar,
                      "Decided On": rep.decided_on, "Rationale": rationale_cell(rep), "Trigger": trigger_text(rep),
                      "Each Column Alone": column_scores(rep, self.tier_rules)}
            for name, value in values.items():
                if name in col:
                    ws.cell(row=row, column=col[name], value=value)
            if rep.tier in TIER_FILLS:
                ws.cell(row=row, column=col["Verdict"]).fill = TIER_FILLS[rep.tier]
            for name in ("AI Score", "Bar"):
                ws.cell(row=row, column=col[name]).number_format = "0.0%"
            choices.add(ws.cell(row=row, column=col["Human Reviewer Decision"]))
            _box(ws, row, len(headers), wrap=True)
        ws.freeze_panes = ws.cell(row=2, column=col["Agent"] + 1)
        ws.auto_filter.ref = f"A1:{get_column_letter(len(headers))}{len(order) + 1}"
        widths = {"Row #": 7, "Task ID": 16, "Agent": 28, "Verdict": 24, "AI Score": 9, "Bar": 8, "Decided On": 20,
                  "Rationale": 80, "Trigger": 30, "Each Column Alone": 34, "Human Reviewer Decision": 22,
                  "Reviewer Notes": 30}
        _widths(ws, {col[name]: width for name, width in widths.items() if name in col})

    def add_annotated_master_sheet(self, df: pd.DataFrame, reports: List[RowReport],
                                   quality_results: List[QualityAssessment], row_issues_count: Dict[int, int],
                                   text_cols: Optional[List[str]] = None):
        """The audited rows with a short QA block appended."""
        ws = self.wb.create_sheet(title="Annotated Master")
        orig = list(df.columns)
        present = [c for c in (text_cols or []) if c in df.columns]
        truncated = {i: [c for c in present if not pd.isna(df.loc[i, c]) and len(str(df.loc[i, c])) > EXCEL_CELL_LIMIT]
                     for i in range(len(df))}
        any_issues, any_truncated = any(row_issues_count.values()), any(truncated.values())
        qa = (["QA Verdict", "QA AI Score", "QA Bar", "QA Decided On", "QA Trigger Sentence", "QA Quality Score",
               "QA Quality Flags"] + (["QA Structural Issues"] if any_issues else [])
              + (["QA Truncated Columns"] if any_truncated else []))
        for col, name in enumerate(orig + qa, 1):
            cell = ws.cell(row=1, column=col, value=name)
            cell.font, cell.fill = FONT_HEADER, NAVY_HEADER if col <= len(orig) else GRAY_HEADER
        for i, rep in enumerate(reports):
            row = i + 2
            for col, name in enumerate(orig, 1):
                value = df.loc[i, name]
                rendered = None if pd.isna(value) else str(value)[:EXCEL_CELL_LIMIT]
                cell = ws.cell(row=row, column=col, value=rendered)
                if isinstance(rendered, str):
                    cell.data_type = "s"
            qual = quality_results[i]
            values = [verdict_text(rep), rep.score, rep.bar, rep.decided_on, trigger_sentences(rep),
                      qual.quality_score, "; ".join(qual.flags) or "None"]
            if any_issues:
                values.append(row_issues_count.get(i, 0))
            if any_truncated:
                values.append(", ".join(truncated[i]) or "None")
            base = len(orig)
            for k, value in enumerate(values, 1):
                ws.cell(row=row, column=base + k, value=value)
            if rep.tier in TIER_FILLS:
                ws.cell(row=row, column=base + 1).fill = TIER_FILLS[rep.tier]
            for k in (2, 3, 6):
                ws.cell(row=row, column=base + k).number_format = "0.0%"
            if any_issues and row_issues_count.get(i, 0):
                ws.cell(row=row, column=base + 8).fill = RED_FILL
        ws.freeze_panes = "A2"
        _autofit_columns(ws, max_len_cap=35)

    def add_technical_sheet(self, reports: List[RowReport], policy):
        """Everything moved out of the other sheets: one line per scored text (each cell, and the row's cells
        checked together)."""
        ws = self.wb.create_sheet(title="Technical")
        headers = ["Row #", "Agent", "Text", "Decision", "Score", "Bar", "Words", "Reason Codes", "Evidence",
                   "Detector Status", "Detector Version", "Policy Version", "Proxy Warnings", "Sentence Drops"]
        _header(ws, 1, headers)
        row = 1
        for rep in reports:
            texts = dict(rep.cells)
            entries = [(name, res, texts.get(name, "")) for name, res in rep.result.column_results.items()]
            if rep.combined is not None:
                entries.append(("all columns together", rep.combined, JOIN.join(t for _, t in rep.cells if t)))
            for n, (name, res, text) in enumerate(entries):
                row += 1
                values = [rep.master_row, rep.agent, name, res.decision.value, res.score, bar_for(policy, text),
                          policy.word_count(text) if text else 0, ", ".join(res.reason_codes) or "None",
                          "; ".join(e.description for e in res.evidence) or "None", res.detector_status.value,
                          res.detector_version, res.policy_version,
                          "; ".join(rep.proxy_warnings) if n == 0 else "",
                          drops_text(rep.trigger) if name == rep.decided_on else ""]
                for col, value in enumerate(values, 1):
                    ws.cell(row=row, column=col, value=value).font = FONT_DEFAULT
                ws.cell(row=row, column=5).number_format = ws.cell(row=row, column=6).number_format = "0.0%"
                _box(ws, row, len(headers), wrap=True)
        ws.freeze_panes = "D2"
        ws.auto_filter.ref = f"A1:{get_column_letter(len(headers))}{max(row, 1)}"
        _widths(ws, {1: 7, 2: 24, 3: 22, 4: 22, 5: 9, 6: 8, 7: 7, 8: 30, 9: 60, 10: 14, 11: 30, 12: 18, 13: 30,
                     14: 60})

    def add_structural_issues_sheet(self, issues: List[StructuralIssue]):
        ws = self.wb.create_sheet(title="Structural Issues")
        _header(ws, 1, ["Excel Row", "Column", "Severity", "Rule Violated", "Offending Value", "Description"])
        for row, issue in enumerate(issues, start=2):
            ws.cell(row=row, column=1, value=issue.excel_row).alignment = Alignment(horizontal="center")
            ws.cell(row=row, column=2, value=issue.column)
            severity = ws.cell(row=row, column=3, value=issue.severity)
            severity.alignment = Alignment(horizontal="center")
            severity.fill = RED_FILL if issue.severity == "ERROR" else YELLOW_FILL
            if issue.severity == "ERROR":
                severity.font = FONT_BOLD
            ws.cell(row=row, column=4, value=issue.rule_violated)
            ws.cell(row=row, column=5, value=str(issue.value)[:80])
            ws.cell(row=row, column=6, value=issue.description)
            _box(ws, row, 6)
        _autofit_columns(ws, max_len_cap=55)

    def save(self, filepath: str):
        # Reports contain no executable formulas. Keep every formula-like value
        # originating from uploaded data or detector evidence as inert text.
        for worksheet in self.wb.worksheets:
            for row in worksheet.iter_rows():
                for cell in row:
                    if cell.data_type == "f":
                        cell.data_type = "s"
        self.wb.save(filepath)
```

- [ ] **Step 4: Switch the pipeline to the new writer and drop review tags** — in `src/pipeline.py`:
  - remove the import `from .report.review_tags import TAG_ORDER, ReviewTagRules`, the constructor parameter `review_tag_rules`, the line `self.review_tag_rules = ...`, the list `review_tags`, the `review_tags.append(...)` line and `summary["review_tag_counts"] = ...`;
  - replace everything from `wb_builder = AuditedWorkbookBuilder()` through `wb_builder.save(output_file)` with:

```python
        wb_builder = AuditedWorkbookBuilder(self.tier_rules)
        wb_builder.add_summary_sheet(summary, rates=summary["tier_rates"],
                                     warning=(self.unvalidated_preview or {}).get("note"))
        wb_builder.add_agents_sheet(summary["agents"])
        wb_builder.add_ai_review_sheet(reports)
        wb_builder.add_annotated_master_sheet(df, reports, quality_results, summary["row_issues_count"], text_cols)
        wb_builder.add_technical_sheet(reports, self.authorship_service.policy)
        if structural_result.issues:
            wb_builder.add_structural_issues_sheet(structural_result.issues)
        wb_builder.save(output_file)
```

  In `src/config.py` delete `ReviewTagConfig` and the `review_tags` field. Delete `src/report/review_tags.py` and `tests/test_review_tags.py` (`git rm`).

- [ ] **Step 5: Move the old layout tests to the new layout**
  - `tests/test_multi_column.py`: replace the block from `# Verify Annotated Master has columns for both` to the end with:

```python
    master_ws = wb["Annotated Master"]
    header_vals = [cell.value for cell in master_ws[1]]
    assert "QA Verdict" in header_vals and "QA Decided On" in header_vals
    assert any("image_factuality" in str(h) for h in header_vals)
    assert any("overall_assessment" in str(h) for h in header_vals)
    assert not any(str(h).startswith("QA_Text_") for h in header_vals)   # per-cell detail is on Technical

    technical = wb["Technical"]
    texts = [technical.cell(row=r, column=3).value for r in range(2, technical.max_row + 1)]
    assert texts.count("image_factuality_rationale") == 3 and texts.count("all columns together") == 2
    tech_headers = [cell.value for cell in technical[1]]
    assert "Detector Version" in tech_headers and "Policy Version" in tech_headers

    review_headers = [cell.value for cell in wb["AI Review"][1]]
    assert "Each Column Alone" in review_headers and "Decided On" in review_headers
```

  (Row T3's cells are both too short, so only T1 and T2 have a combined line: 3 cell lines per column and 2 combined lines.)
  - `tests/test_pipeline.py` `test_pipeline_end_to_end`: replace the two review-header asserts with

```python
    review_headers = [cell.value for cell in wb["AI Review"][1]]
    assert "Verdict" in review_headers and "Bar" in review_headers
    assert "Technical" in wb.sheetnames and "Agents" in wb.sheetnames
```

  and in `test_csv_text_over_excel_limit_is_explicitly_rejected_and_marked` replace the last two asserts with

```python
    assert values["QA Verdict"] == "Not checked (too long)"
    assert values["QA Truncated Columns"] == "justification"
```

- [ ] **Step 6: Run the tests**

Run: `python -m pytest tests/test_workbook_v2.py -q` then `python -m pytest -q`
Expected: all pass. `tests/test_preview.py::test_a_preview_run_marks_the_summary_and_the_workbook` still finds the banner in the first rows of Summary Dashboard.

- [ ] **Step 7: Commit**

```bash
git add -A src/report src/pipeline.py src/config.py tests
git commit -m "feat: the v2 audit workbook - verdict tiers, agents, full rationales with trigger sentences, Technical sheet"
```

---

### Task 6: API, CLI and dashboard

**Files:**
- Modify: `src/api.py` (`run_audit_endpoint`), `src/cli.py`, `src/static/index.html`
- Test: `tests/test_dashboard_v2.py`

**Interfaces:**
- Consumes: summary keys `tier_counts`, `agents`, `triggers` (Task 4); `PipelineConfig.triggers.enabled`.
- Produces: form field `explain_triggers` (default true); CLI flag `--no-triggers`; dashboard elements `#explainTriggers`, `#statHigh`, `#statNotChecked`, `#agentsBox`, `#agentsTable`.

- [ ] **Step 1: Write the failing tests** — `tests/test_dashboard_v2.py`:

```python
"""Dashboard, API and CLI for the v2 audit (spec 2026-10-01, section 5)."""
import json
import sys
from unittest.mock import patch

import pandas as pd
from fastapi.testclient import TestClient

from src.api import app

CSV = "text/csv"


def _upload(tmp_path):
    path = tmp_path / "rows.csv"
    pd.DataFrame({"rater_id": ["a", "b"], "justification": [
        "The response corrects the false premise politely and offers grounded alternatives.",
        "It cites the right source and the rating follows from the rubric for this task."]}).to_csv(path, index=False)
    return {"file": (path.name, path.read_bytes(), CSV)}


def test_the_dashboard_has_the_agents_table_the_new_cards_and_the_trigger_checkbox():
    page = TestClient(app).get("/").text
    for marker in ('id="explainTriggers"', 'id="statHigh"', 'id="statNotChecked"', 'id="agentsTable"',
                   "formData.append('explain_triggers'", "renderAgents("):
        assert marker in page


def test_the_api_returns_tiers_and_agents_and_can_skip_triggers(tmp_path):
    client = TestClient(app)
    res = client.post("/api/audit", files=_upload(tmp_path), data={"provider": "heuristic", "only_assigned": "true"})
    summary = res.json()["summary"]
    assert set(summary["tier_counts"]) >= {"Likely AI", "Not checked"} and len(summary["agents"]) == 2
    assert summary["triggers"]["enabled"] is True
    res = client.post("/api/audit", files=_upload(tmp_path),
                      data={"provider": "heuristic", "only_assigned": "true", "explain_triggers": "false"})
    assert res.json()["summary"]["triggers"]["enabled"] is False


def test_the_cli_can_skip_triggers(tmp_path, capsys):
    from src.cli import main

    path = tmp_path / "rows.csv"
    pd.DataFrame({"rater_id": ["a"], "justification": ["The response is honest and grounded about the hurdles."]}
                 ).to_csv(path, index=False)
    with patch.object(sys, "argv", ["cli.py", "audit", str(path), "-o", str(tmp_path / "o.xlsx"), "-p", "heuristic",
                                    "--no-triggers", "--json"]):
        main()
    assert json.loads(capsys.readouterr().out)["summary"]["triggers"]["enabled"] is False
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_dashboard_v2.py -q`
Expected: FAIL (markers missing, `--no-triggers` unknown).

- [ ] **Step 3: API** — in `run_audit_endpoint` add the parameter `explain_triggers: bool = Form(True),` after `locale`, and after `config.structural.skip_unassigned = only_assigned` add `config.triggers.enabled = explain_triggers`.

- [ ] **Step 4: CLI** — after the `--only-assigned` argument add

```python
    audit_parser.add_argument("--no-triggers", action="store_true",
                              help="Skip finding trigger sentences (the sentences behind a high AI score)")
```

and after `if args.only_assigned: ...` add

```python
        if args.no_triggers:
            config.triggers.enabled = False
```

- [ ] **Step 5: Dashboard** — in `src/static/index.html`:
  - after the `onlyAssigned` checkbox block (`</div>` closing its `margin-top: 1.25rem` wrapper) add:

```html
                <div style="margin-top: 0.6rem;">
                    <label style="display: flex; align-items: center; gap: 0.5rem; cursor: pointer; font-size: 0.9rem; font-weight: normal; color: var(--text-muted);">
                        <input type="checkbox" id="explainTriggers" name="explain_triggers" checked style="width: 1.1rem; height: 1.1rem; accent-color: var(--accent);">
                        <span>Find trigger sentences (which sentence drove a high AI score; adds a little time)</span>
                    </label>
                </div>
```

  - change the `statAIFlagged` card label from `AI Flagged` to `Likely AI`, and after that card add:

```html
                <div class="stat-card">
                    <div class="stat-val" id="statHigh" style="color: #fb923c;">0</div>
                    <div class="stat-label">High AI score &ndash; review</div>
                </div>
                <div class="stat-card">
                    <div class="stat-val" id="statNotChecked" style="color: #94a3b8;">0</div>
                    <div class="stat-label">Not checked</div>
                </div>
```

  - after the `stats-grid` closing `</div>` add:

```html
            <div id="agentsBox" style="display: none; margin-top: 1.5rem;">
                <h3 style="margin-bottom: 0.5rem;">Flagged agents</h3>
                <table id="agentsTable" style="width: 100%; border-collapse: collapse; font-size: 0.9rem; text-align: left;">
                    <thead><tr><th>Agent</th><th>Rows</th><th>Likely AI</th><th>High AI score</th><th>Highest score</th></tr></thead>
                    <tbody></tbody>
                </table>
            </div>
```

  - in the script, before the submit handler, add:

```javascript
        function renderAgents(summary) {
            const box = document.getElementById('agentsBox');
            const body = document.querySelector('#agentsTable tbody');
            body.innerHTML = '';
            const flagged = (summary.agents || []).filter(a => a.likely_ai > 0 || a.high > 0);
            flagged.forEach(a => {
                const tr = document.createElement('tr');
                const best = a.highest_score === null ? '' : (a.highest_score * 100).toFixed(1) + '%';
                [a.agent, a.rows, a.likely_ai, a.high, best].forEach(v => {
                    const td = document.createElement('td');
                    td.textContent = v;   // agent names come from the upload: never innerHTML
                    td.style.padding = '0.35rem 0.5rem';
                    td.style.borderBottom = '1px solid var(--border)';
                    tr.appendChild(td);
                });
                body.appendChild(tr);
            });
            box.style.display = flagged.length ? 'block' : 'none';
        }
```

  - next to `formData.append('only_assigned', ...)` add `formData.append('explain_triggers', document.getElementById('explainTriggers').checked);`
  - after `document.getElementById('statAIFlagged').textContent = data.ai_flagged_count;` add:

```javascript
                const tiers = data.summary.tier_counts || {};
                const high = Object.entries(tiers).find(([name]) => name.startsWith('High AI score'));
                document.getElementById('statHigh').textContent = high ? high[1] : 0;
                document.getElementById('statNotChecked').textContent = tiers['Not checked'] || 0;
                renderAgents(data.summary);
```

- [ ] **Step 6: Run the tests**

Run: `python -m pytest tests/test_dashboard_v2.py -q` then `python -m pytest -q`
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add src/api.py src/cli.py src/static/index.html tests/test_dashboard_v2.py
git commit -m "feat: dashboard flagged-agents table, tier cards and trigger switch; API and CLI trigger switch"
```

---

### Task 7: Real run and timing

**Files:**
- Modify: none expected (a config default change only if the timing demands it, with a ledger ruling)

- [ ] **Step 1: Re-audit one of today's sheets with the 4d preview** (GPU; the web app must not be running). Use the same workbook and locale the user audited (Replication 2, fr_FR: the longest rationales, the worst case for triggers), columns `Rating_human_rationale` and ` AR Rationale (Required — 2–4 sentences)`, assigned rows only, output to the scratchpad:

```bash
python -m src.cli audit "<workbook path>" --locale fr_FR -p preview-unvalidated --only-assigned \
  --text-cols "Rating_human_rationale" " AR Rationale (Required — 2–4 sentences)" -o "<scratchpad>/fr_FR_v2.xlsx" --json > "<scratchpad>/fr_FR_v2.json"
```

Expected: completes; the JSON's `summary.triggers.seconds` is the trigger cost.

- [ ] **Step 2: Compare counts only** (no cell text): Likely AI count equals the old file's (`audited_0fdc45b2_..._fr_FR.xlsx`, 71); tier counts sum to the rows; agents table length; trigger kinds counted (sentences / spread / single / error). Peak VRAM stays at or under 3.5 GB (`nvidia-smi` during the run).

- [ ] **Step 3: Decide on timing.** If triggers add more than 300 s, lower `TriggerConfig.max_sentences` default to 8 with a ledger ruling and re-run; otherwise record the time. Delete the scratchpad output workbook afterwards (it holds rater text).

- [ ] **Step 4: Commit any change and push**

```bash
git push origin increment-4d
```
