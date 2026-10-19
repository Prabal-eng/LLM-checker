"""QA module imports."""
from .structural import run_structural_qa, StructuralIssue, StructuralQAResult
from .proxies import compute_rater_uniformity, evaluate_row_proxies, ProxySignals
from .quality import evaluate_justification_quality, QualityAssessment
from .discovery import ColumnDiscovery, DiscoveredColumns

__all__ = [
    "run_structural_qa",
    "StructuralIssue",
    "StructuralQAResult",
    "compute_rater_uniformity",
    "evaluate_row_proxies",
    "ProxySignals",
    "evaluate_justification_quality",
    "QualityAssessment",
    "ColumnDiscovery",
    "DiscoveredColumns"
]
