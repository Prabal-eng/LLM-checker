"""Selective three-way policy applied after a detector produces an observation."""

import re
from dataclasses import dataclass
from typing import Any, Iterable, Optional, Tuple

from .contracts import (
    DecisionEvidence,
    DetectionDecision,
    DetectorStatus,
    TextCheckResponse,
)


@dataclass(frozen=True)
class BandRule:
    """Calibrated thresholds for texts with lower_words <= words < upper_words (None: no upper limit)."""

    lower_words: int
    upper_words: Optional[int]
    human_max_score: float
    ai_min_score: float


@dataclass(frozen=True)
class DecisionPolicy:
    """Convert a validated detector score into a decision or explicit abstention."""

    human_max_score: float = 0.30
    ai_min_score: float = 0.80
    min_word_count: int = 8
    # Excel's cell limit is the shared ceiling so standalone and workbook inputs
    # use the same support policy and workbook exports never imply full preservation.
    max_char_count: int = 32_767
    version: str = "v2.0.0-provisional"
    # Calibrated per-length thresholds. When set, they replace the single thresholds above and
    # text in a length range without a band abstains with UNSUPPORTED_LENGTH.
    bands: Tuple[BandRule, ...] = ()

    def __post_init__(self) -> None:
        if not 0.0 <= self.human_max_score < self.ai_min_score <= 1.0:
            raise ValueError("thresholds must satisfy 0 <= human < AI <= 1")
        for band in self.bands:
            if not 0.0 <= band.human_max_score < band.ai_min_score <= 1.0:
                raise ValueError("band thresholds must satisfy 0 <= human < AI <= 1")
        if self.min_word_count < 1:
            raise ValueError("min_word_count must be positive")
        if self.max_char_count < 1:
            raise ValueError("max_char_count must be positive")

    def decide(
        self,
        *,
        text: Any,
        detector_version: str,
        detector_status: DetectorStatus,
        score: Optional[float],
        model_validated: bool,
        observed_evidence: Iterable[DecisionEvidence] = (),
        failure_reason: Optional[str] = None,
        unsupported_reason: Optional[str] = None,
        flag_only: bool = False,
    ) -> TextCheckResponse:
        """Apply sufficiency, availability, validation, support, and score-band gates in order.

        unsupported_reason marks text outside the calibrated support (language or domain).
        flag_only allows Likely AI but never Likely human, for slices where missed AI is unmeasured.
        """
        evidence = list(observed_evidence)

        if not isinstance(text, str):
            return self._abstain(
                detector_version=detector_version,
                detector_status=DetectorStatus.NOT_EVALUATED,
                reason="UNSUPPORTED_INPUT_TYPE",
                evidence=[DecisionEvidence(
                    code="INPUT_TYPE",
                    description="Text input must be a string.",
                    value=type(text).__name__,
                )],
            )

        if len(text) > self.max_char_count:
            return self._abstain(
                detector_version=detector_version,
                detector_status=DetectorStatus.NOT_EVALUATED,
                reason="UNSUPPORTED_INPUT_LENGTH",
                evidence=[DecisionEvidence(
                    code="CHAR_COUNT",
                    description=f"Input exceeds the {self.max_char_count} character limit.",
                    value=len(text),
                )],
            )

        word_count = self.word_count(text)

        if word_count < self.min_word_count:
            return self._abstain(
                detector_version=detector_version,
                detector_status=DetectorStatus.NOT_EVALUATED,
                reason="INSUFFICIENT_TEXT",
                evidence=[DecisionEvidence(
                    code="WORD_COUNT",
                    description=f"Only {word_count} words; at least {self.min_word_count} are required.",
                    value=word_count,
                )],
            )

        if detector_status is not DetectorStatus.AVAILABLE:
            return self._abstain(
                detector_version=detector_version,
                detector_status=detector_status,
                reason=failure_reason or "DETECTOR_UNAVAILABLE",
                evidence=evidence,
            )

        if not model_validated:
            return self._abstain(
                detector_version=detector_version,
                detector_status=DetectorStatus.AVAILABLE,
                reason="MODEL_NOT_VALIDATED",
                evidence=evidence,
            )

        if unsupported_reason:
            return self._abstain(
                detector_version=detector_version,
                detector_status=DetectorStatus.AVAILABLE,
                reason=unsupported_reason,
                evidence=evidence,
            )

        if score is None:
            return self._abstain(
                detector_version=detector_version,
                detector_status=DetectorStatus.UNAVAILABLE,
                reason="SCORE_UNAVAILABLE",
                evidence=evidence,
            )

        human_max, ai_min = self.human_max_score, self.ai_min_score
        if self.bands:
            band = self.band_for(word_count)
            if band is None:
                return self._abstain(
                    detector_version=detector_version,
                    detector_status=DetectorStatus.AVAILABLE,
                    reason="UNSUPPORTED_LENGTH",
                    evidence=evidence + [DecisionEvidence(
                        code="WORD_COUNT",
                        description="No calibrated thresholds exist for text of this length.",
                        value=word_count,
                    )],
                )
            human_max, ai_min = band.human_max_score, band.ai_min_score

        if flag_only and score < ai_min:
            evidence.append(DecisionEvidence(
                code="FLAG_ONLY_CONTEXT",
                description="Only a Likely AI result can be issued for this kind of text; "
                            "the score is below the AI threshold.",
                value=score,
            ))
            return TextCheckResponse(
                decision=DetectionDecision.INCONCLUSIVE,
                score=score,
                score_valid=True,
                detector_status=DetectorStatus.AVAILABLE,
                reason_codes=["BELOW_AI_THRESHOLD_FLAG_ONLY"],
                evidence=evidence,
                detector_version=detector_version,
                policy_version=self.version,
            )

        if score <= human_max:
            evidence.append(DecisionEvidence(
                code="SCORE_BELOW_HUMAN_THRESHOLD",
                description="The calibrated score is inside the likely-human decision region.",
                supports=DetectionDecision.LIKELY_HUMAN_WRITTEN,
                value=score,
            ))
            return TextCheckResponse(
                decision=DetectionDecision.LIKELY_HUMAN_WRITTEN,
                score=score,
                score_valid=True,
                detector_status=DetectorStatus.AVAILABLE,
                reason_codes=["CALIBRATED_LOW_SCORE"],
                evidence=evidence,
                detector_version=detector_version,
                policy_version=self.version,
            )

        if score >= ai_min:
            evidence.append(DecisionEvidence(
                code="SCORE_ABOVE_AI_THRESHOLD",
                description="The calibrated score is inside the likely-AI decision region.",
                supports=DetectionDecision.LIKELY_AI_GENERATED,
                value=score,
            ))
            return TextCheckResponse(
                decision=DetectionDecision.LIKELY_AI_GENERATED,
                score=score,
                score_valid=True,
                detector_status=DetectorStatus.AVAILABLE,
                reason_codes=["CALIBRATED_HIGH_SCORE"],
                evidence=evidence,
                detector_version=detector_version,
                policy_version=self.version,
            )

        evidence.append(DecisionEvidence(
            code="SCORE_IN_ABSTENTION_BAND",
            description="The calibrated score is between the likely-human and likely-AI thresholds.",
            value=score,
        ))
        return TextCheckResponse(
            decision=DetectionDecision.INCONCLUSIVE,
            score=score,
            score_valid=True,
            detector_status=DetectorStatus.AVAILABLE,
            reason_codes=["SCORE_IN_ABSTENTION_BAND"],
            evidence=evidence,
            detector_version=detector_version,
            policy_version=self.version,
        )

    def band_for(self, word_count: int) -> Optional[BandRule]:
        """The calibrated band covering word_count, or None where no band was supported."""
        for band in self.bands:
            if word_count >= band.lower_words and (band.upper_words is None or word_count < band.upper_words):
                return band
        return None

    @staticmethod
    def word_count(text: str) -> int:
        """Count Unicode words consistently at every entry point."""
        return len(re.findall(r"\b\w+\b", text, flags=re.UNICODE))

    def has_sufficient_text(self, text: Any) -> bool:
        return (
            isinstance(text, str)
            and len(text) <= self.max_char_count
            and self.word_count(text) >= self.min_word_count
        )

    def _abstain(
        self,
        *,
        detector_version: str,
        detector_status: DetectorStatus,
        reason: str,
        evidence: list[DecisionEvidence],
    ) -> TextCheckResponse:
        return TextCheckResponse(
            decision=DetectionDecision.INCONCLUSIVE,
            score=None,
            score_valid=False,
            detector_status=detector_status,
            reason_codes=[reason],
            evidence=evidence,
            detector_version=detector_version,
            policy_version=self.version,
        )
