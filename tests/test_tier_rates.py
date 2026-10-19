"""Measured tier rates from saved validation scores (spec 2026-10-01, section 1)."""
import json
import math

import numpy as np
import pandas as pd
import pytest

from src.authorship import registry
from src.report.tier_rates import build, load_rates, rate_table, scored_frame
from src.report.tiers import HIGH, LIKELY_AI, LOW, TierRules

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
