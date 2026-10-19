"""Public v2 contract shared by standalone and spreadsheet detection flows."""

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator


class DetectionDecision(str, Enum):
    """The only user-facing authorship decisions."""

    LIKELY_AI_GENERATED = "likely_ai_generated"
    LIKELY_HUMAN_WRITTEN = "likely_human_written"
    INCONCLUSIVE = "inconclusive"


class DetectorStatus(str, Enum):
    """Whether a detector produced usable evidence."""

    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    NOT_EVALUATED = "not_evaluated"


class DecisionEvidence(BaseModel):
    """One observed signal supporting or limiting a decision."""

    code: str = Field(min_length=1, max_length=80)
    description: str = Field(min_length=1, max_length=500)
    supports: Optional[DetectionDecision] = None
    value: Optional[Any] = None


class TextContext(str, Enum):
    """What kind of text is being checked; it selects the calibrated policy."""

    GENERAL = "general"
    # Evaluator rationales: missed AI is unmeasured there, so only Likely AI can be issued.
    # ponytail: D6 flag-only on public-corpus thresholds until verified rater rationales exist.
    RATER_COMMENT = "rater_comment"


class CheckTextRequest(BaseModel):
    """Standalone text-check request."""

    text: Any
    context: TextContext = TextContext.GENERAL


class TextCheckResponse(BaseModel):
    """Stable response contract for all authorship-check entry points."""

    decision: DetectionDecision
    score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    score_valid: bool = False
    detector_status: DetectorStatus
    reason_codes: List[str] = Field(default_factory=list)
    evidence: List[DecisionEvidence] = Field(default_factory=list)
    detector_version: str = Field(min_length=1, max_length=100)
    policy_version: str = Field(min_length=1, max_length=100)

    @model_validator(mode="after")
    def validate_decision_integrity(self) -> "TextCheckResponse":
        """Reject contradictory availability, score, and decision states."""
        if self.score_valid != (self.score is not None):
            raise ValueError("score_valid must be true exactly when score is present")

        if self.detector_status is not DetectorStatus.AVAILABLE:
            if self.decision is not DetectionDecision.INCONCLUSIVE:
                raise ValueError("an unavailable detector must return inconclusive")
            if self.score is not None:
                raise ValueError("an unavailable detector cannot return a score")
            if not self.reason_codes:
                raise ValueError("an unavailable detector must provide a reason code")

        if self.decision is not DetectionDecision.INCONCLUSIVE:
            if self.detector_status is not DetectorStatus.AVAILABLE or not self.score_valid:
                raise ValueError("a decisive result requires an available detector and valid score")
            if not self.evidence:
                raise ValueError("a decisive result requires observed evidence")

        return self


class RowAuthorshipResult(TextCheckResponse):
    """A row decision plus the independently evaluated column decisions."""

    column_decisions: Dict[str, DetectionDecision] = Field(default_factory=dict)
    column_scores: Dict[str, Optional[float]] = Field(default_factory=dict)
    column_results: Dict[str, TextCheckResponse] = Field(default_factory=dict)
    likely_ai_columns: List[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_column_integrity(self) -> "RowAuthorshipResult":
        keys = set(self.column_results)
        if keys != set(self.column_decisions) or keys != set(self.column_scores):
            raise ValueError("column result projections must have identical keys")
        for name, result in self.column_results.items():
            if self.column_decisions[name] is not result.decision:
                raise ValueError("column decision projection is inconsistent")
            if self.column_scores[name] != result.score:
                raise ValueError("column score projection is inconsistent")
        expected_ai = {
            name for name, result in self.column_results.items()
            if result.decision is DetectionDecision.LIKELY_AI_GENERATED
        }
        if set(self.likely_ai_columns) != expected_ai:
            raise ValueError("likely_ai_columns is inconsistent with column results")
        return self
