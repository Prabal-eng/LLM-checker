"""Versioned contracts for LLM authorship decisions."""

from .contracts import (
    CheckTextRequest,
    DecisionEvidence,
    DetectionDecision,
    DetectorStatus,
    RowAuthorshipResult,
    TextCheckResponse,
    TextContext,
)
from .policy import BandRule, DecisionPolicy
from .service import AuthorshipService, combine_column_results

__all__ = [
    "CheckTextRequest",
    "DecisionEvidence",
    "DetectionDecision",
    "DetectorStatus",
    "RowAuthorshipResult",
    "TextCheckResponse",
    "TextContext",
    "BandRule",
    "DecisionPolicy",
    "AuthorshipService",
    "combine_column_results",
]
