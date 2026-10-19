"""Tests for selective classification and detector failure handling."""

import pytest

from src.detection import (
    AuthorshipService,
    DecisionPolicy,
    DetectionDecision,
    DetectorStatus,
    combine_column_results,
)
from src.detectors.base import BaseDetector, DetectionResult


class FixedDetector(BaseDetector):
    def __init__(self, score: float):
        super().__init__(threshold=0.80)
        self.score = score
        self.calls = 0

    def detect(self, text: str) -> DetectionResult:
        self.calls += 1
        return DetectionResult(
            provider="fixed",
            ai_score=self.score,
            is_flagged=self.score >= self.threshold,
            analysis={"evidence": ["Observed test indicator"]},
            raw_response={"ok": True},
        )


class FailingDetector(BaseDetector):
    def detect(self, text: str) -> DetectionResult:
        raise TimeoutError("provider timed out")


class ReportedFailureDetector(BaseDetector):
    def detect(self, text: str) -> DetectionResult:
        return DetectionResult(
            provider="failed",
            ai_score=0.99,
            is_flagged=True,
            analysis={"error": "upstream failed"},
            raw_response={"status": 500},
        )


@pytest.mark.parametrize(
    ("score", "expected"),
    [
        (0.10, DetectionDecision.LIKELY_HUMAN_WRITTEN),
        (0.30, DetectionDecision.LIKELY_HUMAN_WRITTEN),
        (0.31, DetectionDecision.INCONCLUSIVE),
        (0.79, DetectionDecision.INCONCLUSIVE),
        (0.80, DetectionDecision.LIKELY_AI_GENERATED),
        (0.95, DetectionDecision.LIKELY_AI_GENERATED),
    ],
)
def test_validated_detector_uses_three_way_score_regions(score, expected):
    service = AuthorshipService(
        FixedDetector(score),
        model_validated=True,
        detector_version="fixed-v1",
    )
    result = service.check_text("This sufficiently long response contains enough words for a decision.")

    assert result.decision is expected
    assert result.score == score
    assert result.score_valid is True
    assert result.detector_status is DetectorStatus.AVAILABLE
    if expected is not DetectionDecision.INCONCLUSIVE:
        assert result.evidence
    if expected is DetectionDecision.INCONCLUSIVE:
        assert any(item.code == "SCORE_IN_ABSTENTION_BAND" for item in result.evidence)


def test_short_text_abstains_without_calling_detector():
    detector = FixedDetector(0.95)
    service = AuthorshipService(detector, model_validated=True)

    result = service.check_text("Too short")

    assert result.decision is DetectionDecision.INCONCLUSIVE
    assert result.reason_codes == ["INSUFFICIENT_TEXT"]
    assert result.detector_status is DetectorStatus.NOT_EVALUATED
    assert detector.calls == 0


def test_detector_exception_becomes_unavailable_abstention():
    service = AuthorshipService(FailingDetector(), model_validated=True)

    result = service.check_text("This sufficiently long response triggers the unavailable detector path.")

    assert result.decision is DetectionDecision.INCONCLUSIVE
    assert result.detector_status is DetectorStatus.UNAVAILABLE
    assert result.score is None
    assert result.reason_codes == ["DETECTOR_ERROR"]


def test_missing_detector_result_becomes_stable_abstention():
    class NoneDetector(BaseDetector):
        def detect(self, text: str):
            return None

    result = AuthorshipService(NoneDetector(), model_validated=True).check_text(
        "one two three four five six seven eight"
    )

    assert result.decision is DetectionDecision.INCONCLUSIVE
    assert result.reason_codes == ["DETECTOR_INVALID_RESPONSE"]


@pytest.mark.parametrize("error", [TimeoutError(), ConnectionError(), RuntimeError()])
def test_detector_exceptions_share_one_stable_public_reason(error):
    class ExceptionDetector(BaseDetector):
        def detect(self, text: str) -> DetectionResult:
            raise error

    result = AuthorshipService(ExceptionDetector(), model_validated=True).check_text(
        "one two three four five six seven eight"
    )

    assert result.reason_codes == ["DETECTOR_ERROR"]


def test_detector_reported_error_with_raw_payload_still_abstains():
    service = AuthorshipService(ReportedFailureDetector(), model_validated=True)

    result = service.check_text("one two three four five six seven eight")

    assert result.decision is DetectionDecision.INCONCLUSIVE
    assert result.detector_status is DetectorStatus.UNAVAILABLE
    assert result.reason_codes == ["DETECTOR_REPORTED_ERROR"]


def test_invalid_detector_evidence_becomes_unavailable_abstention():
    class InvalidEvidenceDetector(BaseDetector):
        def detect(self, text: str) -> DetectionResult:
            return DetectionResult(
                provider="invalid-evidence",
                ai_score=0.50,
                is_flagged=False,
                analysis={"evidence": ["x" * 501]},
                raw_response={"ok": True},
            )

    result = AuthorshipService(InvalidEvidenceDetector(), model_validated=True).check_text(
        "one two three four five six seven eight"
    )

    assert result.decision is DetectionDecision.INCONCLUSIVE
    assert result.detector_status is DetectorStatus.UNAVAILABLE
    assert result.reason_codes == ["DETECTOR_INVALID_EVIDENCE"]


@pytest.mark.parametrize("value", [None, 123, ["text"], {"text": "value"}])
def test_unsupported_input_type_abstains_without_calling_detector(value):
    detector = FixedDetector(0.99)
    result = AuthorshipService(detector, model_validated=True).check_text(value)

    assert result.decision is DetectionDecision.INCONCLUSIVE
    assert result.reason_codes == ["UNSUPPORTED_INPUT_TYPE"]
    assert result.detector_status is DetectorStatus.NOT_EVALUATED
    assert detector.calls == 0


def test_unsupported_input_length_abstains_without_calling_detector():
    detector = FixedDetector(0.99)
    result = AuthorshipService(detector, model_validated=True).check_text("x" * 32_768)

    assert result.decision is DetectionDecision.INCONCLUSIVE
    assert result.reason_codes == ["UNSUPPORTED_INPUT_LENGTH"]
    assert detector.calls == 0


def test_unvalidated_model_never_runs_or_returns_decisive_result():
    detector = FixedDetector(0.99)
    service = AuthorshipService(detector, model_validated=False)

    result = service.check_text("This response is long enough but the model has not passed validation.")

    assert result.decision is DetectionDecision.INCONCLUSIVE
    assert result.reason_codes == ["MODEL_NOT_VALIDATED"]
    assert result.score is None
    assert detector.calls == 0


def test_policy_rejects_invalid_threshold_order():
    with pytest.raises(ValueError, match="thresholds"):
        DecisionPolicy(human_max_score=0.8, ai_min_score=0.8)


def test_row_combination_prioritizes_likely_ai_and_preserves_columns():
    human = AuthorshipService(FixedDetector(0.10), model_validated=True).check_text(
        "This sufficiently long human sample has enough words for classification."
    )
    ai = AuthorshipService(FixedDetector(0.95), model_validated=True).check_text(
        "This sufficiently long machine sample has enough words for classification."
    )

    result = combine_column_results({"comment": human, "rationale": ai})

    assert result.decision is DetectionDecision.LIKELY_AI_GENERATED
    assert result.score == 0.95
    assert result.likely_ai_columns == ["rationale"]
    assert result.column_decisions["comment"] is DetectionDecision.LIKELY_HUMAN_WRITTEN
    assert result.column_results["rationale"].score_valid is True


def test_row_combination_keeps_mixed_non_ai_results_inconclusive():
    human = AuthorshipService(FixedDetector(0.10), model_validated=True).check_text(
        "This sufficiently long human sample has enough words for classification."
    )
    short = AuthorshipService(FixedDetector(0.95), model_validated=True).check_text("Too short")

    result = combine_column_results({"comment": human, "note": short})

    assert result.decision is DetectionDecision.INCONCLUSIVE
    assert result.score == 0.10
    assert "INSUFFICIENT_TEXT" in result.reason_codes


# --- Increment 4: length bands, flag-only context, language and domain support ---

from src.detection import BandRule, TextContext  # noqa: E402

BANDED = DecisionPolicy(min_word_count=25, bands=(
    BandRule(25, 50, human_max_score=0.05, ai_min_score=0.99),
    BandRule(100, None, human_max_score=0.30, ai_min_score=0.80),
))


def _text(n: int) -> str:
    base = "the report is clear and the chart is fine but the legend is cut off".split()
    return " ".join((base * (n // len(base) + 1))[:n])


class OodDetector(FixedDetector):
    def detect(self, text: str) -> DetectionResult:
        result = super().detect(text)
        result.analysis.update({"in_distribution": False, "ood_distance": 0.9})
        return result


def _service(score: float, detector_cls=FixedDetector, **kwargs) -> AuthorshipService:
    return AuthorshipService(detector_cls(score), policy=BANDED, model_validated=True,
                             detector_version="test", require_english=True, **kwargs)


@pytest.mark.parametrize(("words", "score", "expected"), [
    (30, 0.90, DetectionDecision.INCONCLUSIVE),       # short band needs 0.99
    (30, 0.995, DetectionDecision.LIKELY_AI_GENERATED),
    (150, 0.90, DetectionDecision.LIKELY_AI_GENERATED),
    (150, 0.20, DetectionDecision.LIKELY_HUMAN_WRITTEN),
])
def test_thresholds_follow_the_word_count_band(words, score, expected):
    assert _service(score).check_text(_text(words)).decision is expected


def test_length_between_supported_bands_abstains():
    result = _service(0.99).check_text(_text(70))
    assert result.decision is DetectionDecision.INCONCLUSIVE
    assert result.reason_codes == ["UNSUPPORTED_LENGTH"]


def test_flag_only_context_never_labels_human():
    low = _service(0.01).check_text(_text(150), context=TextContext.RATER_COMMENT)
    high = _service(0.95).check_text(_text(150), context=TextContext.RATER_COMMENT)
    assert low.decision is DetectionDecision.INCONCLUSIVE
    assert low.reason_codes == ["BELOW_AI_THRESHOLD_FLAG_ONLY"]
    assert high.decision is DetectionDecision.LIKELY_AI_GENERATED


def test_non_english_text_abstains_without_calling_detector():
    service = _service(0.99)
    text = ("L'image montre les bons monuments, mais les étiquettes du côté gauche sont coupées et la "
            "légende ne correspond pas aux couleurs utilisées dans le graphique, ce qui la rend difficile "
            "à lire pour les utilisateurs qui consultent le rapport.")
    result = service.check_text(text)
    assert result.reason_codes == ["UNSUPPORTED_LANGUAGE"]
    assert service.detector.calls == 0


def test_out_of_distribution_text_abstains_in_general_context_only():
    general = _service(0.95, OodDetector).check_text(_text(150))
    rater = _service(0.95, OodDetector).check_text(_text(150), context=TextContext.RATER_COMMENT)
    assert general.decision is DetectionDecision.INCONCLUSIVE
    assert general.reason_codes == ["UNSUPPORTED_DOMAIN"]
    assert general.score is None
    assert rater.decision is DetectionDecision.LIKELY_AI_GENERATED


def test_band_thresholds_are_validated():
    with pytest.raises(ValueError):
        DecisionPolicy(bands=(BandRule(25, None, human_max_score=0.9, ai_min_score=0.5),))


def test_row_result_decides_the_row_and_cells_stay_as_detail():
    short = AuthorshipService(FixedDetector(0.95), policy=BANDED, model_validated=True,
                              detector_version="test").check_text("too short to judge")
    row = _service(0.95).check_text(_text(150), context=TextContext.RATER_COMMENT)
    result = combine_column_results({"a": short, "b": short}, row_result=row)
    assert result.decision is DetectionDecision.LIKELY_AI_GENERATED
    assert result.evidence[0].code == "ROW_COMBINED_TEXT"
    assert result.likely_ai_columns == []
    assert set(result.column_results) == {"a", "b"}


def test_batch_decisions_match_single_decisions():
    texts = [_text(30), _text(70), _text(150), "short", 42,
             "La imagen muestra los monumentos correctos, pero las etiquetas del lado izquierdo están "
             "cortadas y la leyenda no coincide con los colores que se usan en el gráfico."]
    for context in (TextContext.GENERAL, TextContext.RATER_COMMENT):
        batch = _service(0.95).check_batch(texts, context)
        single = [_service(0.95).check_text(t, context) for t in texts]
        assert [r.model_dump() for r in batch] == [r.model_dump() for r in single]


def test_batch_detector_failure_abstains_for_every_scored_text():
    service = AuthorshipService(FailingDetector(), policy=BANDED, model_validated=True, detector_version="t")
    results = service.check_batch([_text(150), _text(160), "short"])
    assert [r.reason_codes for r in results] == [["DETECTOR_ERROR"], ["DETECTOR_ERROR"], ["INSUFFICIENT_TEXT"]]
