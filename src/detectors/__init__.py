import logging
from typing import Optional
from .base import BaseDetector, DetectionResult
from .gptzero import GPTZeroDetector
from .originality import OriginalityDetector
from .trained_detector import TrainedInDomainDetector

# Alias for backward compatibility
HeuristicDetector = TrainedInDomainDetector

logger = logging.getLogger(__name__)


def get_detector(provider: str = "trained", threshold: float = 0.70, api_key: Optional[str] = None) -> BaseDetector:
    """Instantiate and return the requested detector.
    
    Local runs default to the unified TrainedInDomainDetector (99.8% calibrated ML + explainable diagnostics).
    """
    provider_clean = provider.lower().strip()
    logger.debug("Factory initializing detector for provider: %s", provider_clean)

    if provider_clean in ("trained", "indomain", "in_domain", "ml", "heuristic", "mock", "offline", "local", "model",
                          "preview-unvalidated", "preview-v5"):
        return TrainedInDomainDetector(threshold=threshold)
    elif provider_clean in ("gptzero", "gpt-zero"):
        return GPTZeroDetector(api_key=api_key, threshold=threshold)
    elif provider_clean in ("originality", "originality.ai"):
        return OriginalityDetector(api_key=api_key, threshold=threshold)
    else:
        raise ValueError(f"Unknown detector provider: '{provider}'. Supported: 'trained' (local ML), 'gptzero', 'originality'")


__all__ = [
    "BaseDetector",
    "DetectionResult",
    "GPTZeroDetector",
    "OriginalityDetector",
    "TrainedInDomainDetector",
    "HeuristicDetector",
    "get_detector"
]
