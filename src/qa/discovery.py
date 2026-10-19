"""Column discovery and schema inference for evaluation spreadsheets."""
import logging
import re
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
import pandas as pd
from ..config import StructuralConfig

logger = logging.getLogger(__name__)


@dataclass
class DiscoveredColumns:
    rater_col: str
    text_cols: List[str]
    text_col: str
    verdict_col: Optional[str]
    row_id_col: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rater_col": self.rater_col,
            "text_cols": self.text_cols,
            "text_col": self.text_col,
            "verdict_col": self.verdict_col,
            "row_id_col": self.row_id_col,
        }

    def __getitem__(self, item: str) -> Any:
        return getattr(self, item)


def norm_name(name: Any) -> str:
    """Header text with case, spaces and hyphens folded to underscores, so 'Task ID' matches 'task_id'."""
    return re.sub(r"[\s\-]+", "_", str(name).strip().lower())


class ColumnDiscovery:
    """Intelligent discovery of rater, rationale, verdict, and identifier columns."""

    RATER_KEYWORDS = ["task_assignee", "rater_id", "rater", "annotator", "evaluator", "assignee", "submitted_by", "email"]
    # A column named like a rater but holding free text (e.g. raterComment) is the rationale, not the rater.
    NOT_RATER_KEYWORDS = ["comment", "rationale", "justification", "feedback", "reason", "note", "explanation"]
    RATIONALE_PRIORITY = [
        "overall_assessment_rationale",
        "justification",
        "rationale",
        "image_factuality_rationale",
        "design_clarity_quality_rationale",
        "comments",
        "feedback",
        "explanation",
        "notes"
    ]
    VERDICT_KEYWORDS = ["overall_assessment", "verdict", "score", "rating", "grade", "decision"]
    ROW_ID_KEYWORDS = ["prompt_id", "row_id", "task_id", "item_id", "task_number", "sample_id"]

    @classmethod
    def discover_rater_column(cls, cols: List[str], cols_lower: Dict[str, str], configured: Optional[str] = None) -> str:
        """Finds the rater / evaluator ID column."""
        if configured and configured in cols:
            return configured

        for c in cols:
            c_low = norm_name(c)
            if any(k in c_low for k in cls.NOT_RATER_KEYWORDS):
                continue
            if any(k in c_low for k in cls.RATER_KEYWORDS):
                return c

        return cols[0] if cols else "rater"

    @classmethod
    def discover_text_columns(
        cls,
        df: pd.DataFrame,
        cols: List[str],
        cols_lower: Dict[str, str],
        configured_cols: Optional[List[str]] = None,
        configured_justification: Optional[str] = None
    ) -> List[str]:
        """Finds one or more justification / rationale text columns."""
        configured = configured_cols or ([configured_justification] if configured_justification else None)
        selected: List[str] = []

        if configured:
            for req_col in configured:
                req_clean = req_col.strip().lower()
                match = next((c for c in cols if c.strip().lower() == req_clean), None)
                if match and match not in selected:
                    selected.append(match)
                elif req_col in cols and req_col not in selected:
                    selected.append(req_col)

        if not selected:
            # Check for all substantial rationale/justification/comment columns
            matched_cols = []
            for c in cols:
                c_low = cols_lower[c]
                if any(skip in c_low for skip in ["qc", "l2", "admin", "status", "date", "time", "for rater", "reviewer", "lead"]):
                    continue
                if any(k in c_low for k in ["rationale", "justification", "specific comment", "comment", "feedback"]):
                    if c in df.columns and (pd.api.types.is_string_dtype(df[c]) or df[c].dtype == object):
                        non_empty = df[c].dropna().astype(str).str.strip()
                        non_empty = non_empty[non_empty != ""]
                        if len(non_empty) > 0 and non_empty.str.len().mean() > 15:
                            matched_cols.append(c)

            if matched_cols:
                # Prioritize primary justification/overall rationale first
                def sort_key(col_name: str) -> int:
                    low = col_name.lower()
                    if "winner" in low or "overall" in low or "justification" in low:
                        return 0
                    return 1

                selected = sorted(matched_cols, key=sort_key)

        if not selected:
            for p in cls.RATIONALE_PRIORITY:
                for c, c_low in cols_lower.items():
                    if p in c_low and c not in selected:
                        selected.append(c)
                if selected:
                    break

        if not selected:
            for c in cols:
                if df[c].dtype == object and df[c].dropna().astype(str).str.len().mean() > 20:
                    selected.append(c)
                    break

        if not selected:
            selected = [cols[-1]] if cols else ["justification"]

        return selected

    @classmethod
    def discover_verdict_column(cls, cols: List[str], cols_lower: Dict[str, str], configured: Optional[str] = None) -> Optional[str]:
        """Finds the verdict / evaluation score column."""
        if configured and configured in cols:
            return configured

        for c, c_low in cols_lower.items():
            if any(k in c_low for k in cls.VERDICT_KEYWORDS) and "rationale" not in c_low:
                return c

        return None

    @classmethod
    def discover_row_id_column(
        cls,
        cols: List[str],
        cols_lower: Dict[str, str],
        rater_col: str,
        configured: Optional[str] = None
    ) -> Optional[str]:
        """Finds the item / prompt identifier column."""
        if configured and configured in cols:
            return configured

        for c in cols:
            c_low = norm_name(c)
            if c == rater_col:
                continue
            if any(k in c_low for k in cls.ROW_ID_KEYWORDS) or c_low == "id":
                return c

        return None

    @classmethod
    def discover(cls, df: pd.DataFrame, config: Optional[StructuralConfig] = None) -> DiscoveredColumns:
        """Runs full column discovery across a DataFrame."""
        cfg = config or StructuralConfig()
        cols = list(df.columns)
        cols_lower = {c: str(c).lower().strip() for c in cols}

        rater_col = cls.discover_rater_column(cols, cols_lower, configured=cfg.rater_column)
        text_cols = cls.discover_text_columns(
            df=df,
            cols=cols,
            cols_lower=cols_lower,
            configured_cols=cfg.text_columns,
            configured_justification=cfg.justification_column
        )
        verdict_col = cls.discover_verdict_column(cols, cols_lower, configured=cfg.verdict_column)
        row_id_col = cls.discover_row_id_column(cols, cols_lower, rater_col=rater_col, configured=cfg.row_id_column)

        logger.debug(
            "ColumnDiscovery: rater='%s', text=%s, verdict='%s', row_id='%s'",
            rater_col, text_cols, verdict_col, row_id_col
        )

        return DiscoveredColumns(
            rater_col=rater_col,
            text_cols=text_cols,
            text_col=text_cols[0],
            verdict_col=verdict_col,
            row_id_col=row_id_col
        )
