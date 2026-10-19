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
    trig = cfg.triggers
    assert (trig.enabled, trig.min_share, trig.max_sentences, trig.top) == (True, 0.20, 12, 2)
    assert (trig.words, trig.max_words, trig.top_words, trig.min_word_share) == (True, 60, 5, 0.05)
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
        TriggerConfig(min_share=0.0)
    with pytest.raises(ValueError):
        TriggerConfig(max_words=0)
    with pytest.raises(ValueError):
        TriggerConfig(max_sentences=0)
