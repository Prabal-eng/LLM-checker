import logging
from typing import Dict, List, Optional
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class DetectorConfig(BaseModel):
    provider: str = Field(default="trained", description="'trained' (local In-Domain ML), 'gptzero', or 'originality'")
    api_key: Optional[str] = None
    ai_threshold: float = Field(default=0.70, ge=0.0, le=1.0, description="Threshold above which text is flagged as AI")
    request_timeout: int = 15


class StructuralConfig(BaseModel):
    sheet_name: Optional[str] = None
    required_columns: List[str] = Field(default_factory=lambda: [
        "rater_id", "row_id", "category", "verdict", "justification"
    ])
    rater_column: Optional[str] = None
    row_id_column: Optional[str] = None
    verdict_column: str = "verdict"
    justification_column: Optional[str] = "justification"
    text_columns: Optional[List[str]] = None  # One or multiple columns to detect AI from
    valid_verdicts: Optional[List[str]] = None  # None means do not strictly enforce verdict enum if custom
    fail_verdicts: List[str] = Field(default_factory=lambda: ["FAIL", "NEEDS_REVIEW", "1", "2", "POOR", "INADEQUATE"])
    check_duplicates: bool = True
    duplicate_subset: Optional[List[str]] = Field(default_factory=lambda: ["rater_id", "row_id"])
    skip_unassigned: bool = Field(default=False, description="If True, skips rows where rater_id is empty/unassigned")


class QualityConfig(BaseModel):
    min_justification_length: int = 20
    min_word_count: int = 4
    check_specificity: bool = True
    # Generic filler phrases that indicate low quality justifications
    generic_fluff_phrases: List[str] = Field(default_factory=lambda: [
        "looks good", "as expected", "fine", "no issue", "all good", "okay",
        "same as above", "n/a", "none", "not applicable", "worked well"
    ])


class TierConfig(BaseModel):
    """Score bands below the calibrated bar (the sponsor's 75 / 50 / 30 from 2026-09-28; spec 2026-10-01)."""
    high_min: float = Field(default=0.75, ge=0.0, le=1.0, description="High AI score - review at or above this score")
    moderate_min: float = Field(default=0.50, ge=0.0, le=1.0, description="Moderate AI score at or above this score")
    second_look_min: float = Field(default=0.30, ge=0.0, le=1.0, description="Needs a second look at or above this score")


class TriggerConfig(BaseModel):
    """Trigger sentences and words: what takes away the largest share of a high row's AI signal (its log-odds) when
    removed (spec 2026-10-01; words 2026-10-02)."""
    enabled: bool = True
    min_share: float = Field(default=0.20, gt=0.0, le=1.0,
                             description="Smallest share of the AI signal that makes a sentence a trigger")
    max_sentences: int = Field(default=12, ge=1, description="Most sentences tried per row")
    top: int = Field(default=2, ge=1, description="Most trigger sentences shown per row")
    words: bool = True
    max_words: int = Field(default=60, ge=1, description="Most words tried per row")
    top_words: int = Field(default=5, ge=1, description="Most trigger words or phrases shown per row")
    min_word_share: float = Field(default=0.05, gt=0.0, le=1.0,
                                  description="Smallest share of the AI signal that makes a word a trigger")


class PipelineConfig(BaseModel):
    detector: DetectorConfig = Field(default_factory=DetectorConfig)
    structural: StructuralConfig = Field(default_factory=StructuralConfig)
    quality: QualityConfig = Field(default_factory=QualityConfig)
    tiers: TierConfig = Field(default_factory=TierConfig)
    triggers: TriggerConfig = Field(default_factory=TriggerConfig)
