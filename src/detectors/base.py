"""Base class and common models for AI text detection."""
from abc import ABC, abstractmethod
import concurrent.futures
import logging
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class DetectionResult(BaseModel):
    provider: str
    ai_score: float = Field(ge=0.0, le=1.0, description="Probability of AI-generated content (0.0 to 1.0)")
    is_flagged: bool = Field(description="True if score >= configured threshold")
    confidence: Optional[float] = Field(default=None, description="Confidence metric if provided by API")
    analysis: Dict[str, Any] = Field(default_factory=dict, description="Detailed breakdown or heuristics")
    raw_response: Optional[Dict[str, Any]] = None
    column_scores: Optional[Dict[str, float]] = Field(default=None, description="Per-column AI scores if multiple columns analyzed")
    flagged_columns: Optional[List[str]] = Field(default=None, description="List of columns that triggered AI flag")


class BaseDetector(ABC):
    """Abstract interface for AI text detectors."""

    def __init__(self, threshold: float = 0.70):
        self.threshold = threshold

    @abstractmethod
    def detect(self, text: str) -> DetectionResult:
        """Analyze text and return DetectionResult."""
        pass

    def detect_batch(self, texts: List[str], max_workers: int = 5) -> List[DetectionResult]:
        """Analyzes a batch of texts concurrently via ThreadPoolExecutor.
        
        Subclasses may override for native batch endpoints or vectorized execution.
        """
        if not texts:
            return []
        if len(texts) == 1 or max_workers <= 1:
            return [self.detect(t) for t in texts]

        logger.debug("Executing detect_batch for %d texts with max_workers=%d", len(texts), max_workers)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            return list(executor.map(self.detect, texts))
