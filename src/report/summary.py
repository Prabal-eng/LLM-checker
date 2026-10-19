"""Summary analytics and aggregation for QA audit runs."""
from typing import Any, Dict, List

import pandas as pd

from ..detection import DetectionDecision, RowAuthorshipResult
from ..qa.quality import QualityAssessment
from ..qa.structural import StructuralIssue


def compute_audit_summary(
    df: pd.DataFrame,
    detection_results: List[RowAuthorshipResult],
    quality_results: List[QualityAssessment],
    structural_issues: List[StructuralIssue],
    rater_col: str = "rater_id",
) -> Dict[str, Any]:
    """Aggregate three-way decisions without treating missing scores as zero."""
    total_rows = len(df)
    ai_count = sum(r.decision is DetectionDecision.LIKELY_AI_GENERATED for r in detection_results)
    human_count = sum(r.decision is DetectionDecision.LIKELY_HUMAN_WRITTEN for r in detection_results)
    inconclusive_count = sum(r.decision is DetectionDecision.INCONCLUSIVE for r in detection_results)
    structural_errors_count = sum(s.severity == "ERROR" for s in structural_issues)
    structural_warnings_count = sum(s.severity == "WARNING" for s in structural_issues)
    avg_quality = sum(q.quality_score for q in quality_results) / total_rows if total_rows else 0.0

    row_issues_count: Dict[int, int] = {}
    for issue in structural_issues:
        if issue.row_index >= 0:
            row_issues_count[issue.row_index] = row_issues_count.get(issue.row_index, 0) + 1

    rater_stats: Dict[str, Dict[str, Any]] = {}
    if rater_col in df.columns:
        for idx, result in enumerate(detection_results):
            rater = str(df.loc[idx, rater_col])
            stat = rater_stats.setdefault(rater, {
                "total": 0, "likely_ai": 0, "likely_human": 0,
                "inconclusive": 0, "structural_issues": 0,
                "ai_scores_sum": 0.0, "valid_score_count": 0,
                "avg_ai_score": None,
            })
            stat["total"] += 1
            if result.decision is DetectionDecision.LIKELY_AI_GENERATED:
                stat["likely_ai"] += 1
            elif result.decision is DetectionDecision.LIKELY_HUMAN_WRITTEN:
                stat["likely_human"] += 1
            else:
                stat["inconclusive"] += 1
            stat["structural_issues"] += row_issues_count.get(idx, 0)
            if result.score_valid and result.score is not None:
                stat["ai_scores_sum"] += result.score
                stat["valid_score_count"] += 1

        for stat in rater_stats.values():
            if stat["valid_score_count"]:
                stat["avg_ai_score"] = round(stat["ai_scores_sum"] / stat["valid_score_count"], 3)

    return {
        "total_rows": total_rows,
        "ai_flagged_count": ai_count,
        "likely_human_count": human_count,
        "inconclusive_count": inconclusive_count,
        "structural_errors_count": structural_errors_count,
        "structural_warnings_count": structural_warnings_count,
        "avg_quality_score": round(avg_quality, 3),
        "rater_stats": rater_stats,
        "row_issues_count": row_issues_count,
    }
