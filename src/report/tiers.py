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
