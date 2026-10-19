"""Unit tests for the trained in-domain detector and hybrid scoring engine."""
import pytest
from src.detectors.trained_detector import TrainedInDomainDetector


def test_trained_detector_predicts_human():
    det = TrainedInDomainDetector(threshold=0.70)
    human_text = "Good concept but the garbled caption text under panel 2 need fixing before it loos polished"
    res = det.detect(human_text)
    assert res.ai_score < 0.50
    assert res.is_flagged is False
    assert res.analysis["predicted_label"] == "Human"


def test_trained_detector_predicts_ai_image_critique():
    det = TrainedInDomainDetector(threshold=0.70)
    ai_text = (
        "Upon close inspection, the generated visual effectively encapsulates the user's intent, "
        "although minor discrepancies in typography and label alignment diminish overall readability. "
        "Consequently, targeted refinement would substantially elevate the quality score."
    )
    res = det.detect(ai_text)
    assert res.ai_score >= 0.70
    assert res.is_flagged is True
    assert res.analysis["predicted_label"] == "AI"


def test_trained_detector_predicts_comparative_eval_ai():
    det = TrainedInDomainDetector(threshold=0.70)
    comparative_text = (
        "Response A is the strongest overall because it directly addresses the user’s argument, "
        "recognizes the appeal of a 3-day work week, and offers a practical counterpoint about implementation. "
        "It feels more focused and balanced while maintaining a natural, constructive tone."
    )
    res = det.detect(comparative_text)
    assert res.ai_score >= 0.70
    assert res.is_flagged is True
    assert "Formulaic comparative opening pattern" in res.analysis["evidence"]
    assert len(res.analysis["llm_markers_found"]) > 0


def test_trained_detector_human_counter_signal_dampens_score():
    det = TrainedInDomainDetector(threshold=0.70)
    text_with_slang = "Response C is better. It is east and did math right, unlike response B. It explains the weird time wrap that feels natural."
    res = det.detect(text_with_slang)
    assert res.ai_score < 0.70
    assert res.is_flagged is False
    assert any("human" in ev.lower() for ev in res.analysis["evidence"])


def test_trained_detector_batch_processing():
    det = TrainedInDomainDetector(threshold=0.70)
    texts = [
        "",
        "Too short",
        "Response A is the strongest overall because it gives clear constructive pushback while maintaining a balanced tone.",
        "Response B is nice but idk didn't work for me lol."
    ]
    results = det.detect_batch(texts)
    assert len(results) == 4
    assert results[0].ai_score == 0.0
    assert results[1].ai_score == 0.10
    assert results[2].is_flagged is True
    assert results[3].is_flagged is False
