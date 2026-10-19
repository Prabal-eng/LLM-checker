"""Unit tests for AI detection module."""
import pytest
from src.detectors import get_detector, TrainedInDomainDetector, HeuristicDetector


def test_unified_detector_with_ai_patterns():
    detector = get_detector("trained", threshold=0.60)
    ai_text = (
        "Furthermore, it is crucial to delve into the comprehensive nature of this summary. "
        "It seamlessly encapsulates the primary themes while meticulously maintaining brevity. "
        "In conclusion, it serves as a testament to high fidelity."
    )
    res = detector.detect(ai_text)
    assert res.ai_score >= 0.60
    assert res.is_flagged is True
    assert len(res.analysis["llm_markers_found"]) >= 2
    assert "burstiness_ratio" in res.analysis
    assert "ttr" in res.analysis
    assert len(res.analysis["evidence"]) >= 1


def test_unified_detector_with_human_informal_text():
    detector = get_detector("trained", threshold=0.60)
    human_text = "idk why but clicked the button and it didn't work, lol seems like a typo in the api url"
    res = detector.detect(human_text)
    assert res.ai_score < 0.40
    assert res.is_flagged is False
    assert len(res.analysis["human_markers_found"]) >= 1


def test_unified_detector_empty_and_short():
    detector = TrainedInDomainDetector(threshold=0.70)
    assert detector.detect("").is_flagged is False
    assert detector.detect("looks ok").is_flagged is False
    assert detector.detect("").ai_score == 0.0
