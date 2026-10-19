"""Justification Quality Pass (Provenance vs. Quality)."""
import re
import logging
from typing import List, Optional
from pydantic import BaseModel, Field
from ..config import QualityConfig

logger = logging.getLogger(__name__)


class QualityAssessment(BaseModel):
    quality_score: float = Field(ge=0.0, le=1.0, description="Quality rating from 0.0 to 1.0")
    is_acceptable: bool = True
    char_length: int = 0
    word_count: int = 0
    flags: List[str] = Field(default_factory=list)
    reasoning: str = ""


# Negative problem terms to expect when verdict is FAIL
PROBLEM_INDICATOR_WORDS = [
    "error", "fail", "bug", "broken", "incorrect", "invalid", "missing",
    "crash", "defect", "inaccurate", "misleading", "poor", "issue", "problem", "typo"
]

# Positive approval terms to expect when verdict is PASS
POSITIVE_INDICATOR_WORDS = [
    "correct", "accurate", "matches", "meets", "valid", "satisfied", "clean",
    "working", "expected", "verified", "passed", "appropriate", "consistent"
]


def evaluate_justification_quality(
    text: Optional[str],
    verdict: Optional[str],
    config: QualityConfig
) -> QualityAssessment:
    """Evaluates justification quality, specificity, and alignment with the assigned verdict."""
    if not text or not str(text).strip():
        return QualityAssessment(
            quality_score=0.0,
            is_acceptable=False,
            char_length=0,
            word_count=0,
            flags=["MISSING_TEXT"],
            reasoning="Justification text is completely empty"
        )

    clean_text = str(text).strip()
    words = re.findall(r"\b\w+\b", clean_text.lower())
    word_count = len(words)
    char_len = len(clean_text)
    flags: List[str] = []

    # 1. Length & word count
    if char_len < config.min_justification_length or word_count < config.min_word_count:
        flags.append(f"TOO_SHORT (words: {word_count}, min: {config.min_word_count})")

    # 2. Generic fluff detection
    lower_text = clean_text.lower()
    for fluff in config.generic_fluff_phrases:
        if lower_text == fluff or (len(words) <= 3 and fluff in lower_text):
            flags.append(f"GENERIC_FLUFF ('{fluff}')")
            break

    # 3. Verdict Alignment Checks
    verdict_clean = str(verdict or "").strip().upper()
    if verdict_clean in ("FAIL", "NEEDS_REVIEW"):
        has_problem_desc = any(term in words for term in PROBLEM_INDICATOR_WORDS)
        has_positive_only = any(term in words for term in POSITIVE_INDICATOR_WORDS) and not has_problem_desc
        if has_positive_only:
            flags.append("VERDICT_MISMATCH (Verdict is FAIL but text contains only positive feedback)")
        elif not has_problem_desc and word_count < 8:
            flags.append("VAGUE_FAILURE (Failed verdict lacks concrete description of defect)")

    elif verdict_clean == "PASS":
        has_problem_desc = any(term in words for term in ["fatal", "crash", "broken", "severe defect"])
        if has_problem_desc:
            flags.append("VERDICT_MISMATCH (Verdict is PASS but text describes critical breakage)")

    # Score calculation
    base_score = 1.0
    if "MISSING_TEXT" in flags:
        base_score = 0.0
    else:
        if any("TOO_SHORT" in f for f in flags):
            base_score -= 0.35
        if any("GENERIC_FLUFF" in f for f in flags):
            base_score -= 0.40
        if any("VERDICT_MISMATCH" in f for f in flags):
            base_score -= 0.45
        if any("VAGUE_FAILURE" in f for f in flags):
            base_score -= 0.25

    final_score = round(max(0.0, min(1.0, base_score)), 2)
    is_acceptable = final_score >= 0.60 and not any("MISMATCH" in f for f in flags)

    return QualityAssessment(
        quality_score=final_score,
        is_acceptable=is_acceptable,
        char_length=char_len,
        word_count=word_count,
        flags=flags,
        reasoning="; ".join(flags) if flags else "Adequate and aligned with verdict"
    )
