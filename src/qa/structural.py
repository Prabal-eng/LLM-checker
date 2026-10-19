"""Structural and mechanical QA validation using pandas."""
import logging
from typing import List, Any, Optional
import pandas as pd
from pydantic import BaseModel
from ..config import StructuralConfig

logger = logging.getLogger(__name__)


class StructuralIssue(BaseModel):
    row_index: int
    excel_row: int  # 1-indexed Excel row (header is 1, data starts at 2)
    column: str
    value: Any
    rule_violated: str
    severity: str  # "ERROR" or "WARNING"
    description: str


class StructuralQAResult(BaseModel):
    total_rows: int
    issues_count: int
    error_count: int
    warning_count: int
    issues: List[StructuralIssue]


def run_structural_qa(df: pd.DataFrame, config: StructuralConfig) -> StructuralQAResult:
    """Executes structural and mechanical checks across the DataFrame."""
    issues: List[StructuralIssue] = []
    total_rows = len(df)

    # 1. Check Missing Required Columns
    for col in config.required_columns:
        if col not in df.columns:
            issues.append(StructuralIssue(
                row_index=-1,
                excel_row=1,
                column=col,
                value=None,
                rule_violated="MISSING_COLUMN",
                severity="ERROR",
                description=f"Mandatory column '{col}' is missing from the dataset"
            ))

    # If any required column is completely missing, we can't do row-level checks on it
    present_cols = [c for c in config.required_columns if c in df.columns]

    # 2. Blank / Missing Required Fields
    for col in present_cols:
        # Boolean mask for empty or whitespace-only cells
        blank_mask = df[col].isna() | (df[col].astype(str).str.strip() == "") | (df[col].astype(str).str.lower() == "nan")
        for idx in df[blank_mask].index:
            excel_row = idx + 2  # Assuming 1 header line
            val = df.loc[idx, col]
            issues.append(StructuralIssue(
                row_index=idx,
                excel_row=excel_row,
                column=col,
                value="" if pd.isna(val) else str(val),
                rule_violated="BLANK_REQUIRED_FIELD",
                severity="ERROR",
                description=f"Field '{col}' cannot be empty or blank"
            ))

    # 3. Duplicate Detection
    if config.check_duplicates and config.duplicate_subset:
        subset_cols = [c for c in config.duplicate_subset if c in df.columns]
        if subset_cols:
            dup_mask = df.duplicated(subset=subset_cols, keep=False)
            for idx in df[dup_mask].index:
                excel_row = idx + 2
                val = {c: str(df.loc[idx, c]) for c in subset_cols}
                issues.append(StructuralIssue(
                    row_index=idx,
                    excel_row=excel_row,
                    column=", ".join(subset_cols),
                    value=str(val),
                    rule_violated="DUPLICATE_RECORD",
                    severity="WARNING",
                    description=f"Duplicate submission detected on key columns: {val}"
                ))

    # 4. Valid Verdict Enum Checks
    verdict_col = config.verdict_column
    if config.valid_verdicts and verdict_col in df.columns:
        valid_upper = [v.upper() for v in config.valid_verdicts]
        invalid_verdict_mask = ~df[verdict_col].isna() & ~df[verdict_col].astype(str).str.strip().str.upper().isin(valid_upper)
        for idx in df[invalid_verdict_mask].index:
            excel_row = idx + 2
            val = df.loc[idx, verdict_col]
            issues.append(StructuralIssue(
                row_index=idx,
                excel_row=excel_row,
                column=verdict_col,
                value=str(val),
                rule_violated="INVALID_VERDICT_VALUE",
                severity="ERROR",
                description=f"Value '{val}' is not a recognized verdict. Allowed: {config.valid_verdicts}"
            ))

    # 5. Cross-Column Conditional Consistency: FAIL Verdict Requires Justification
    just_col = config.justification_column
    if verdict_col in df.columns and just_col in df.columns:
        fail_upper = [f.upper() for f in config.fail_verdicts]
        is_fail = df[verdict_col].astype(str).str.strip().str.upper().isin(fail_upper)
        empty_just = df[just_col].isna() | (df[just_col].astype(str).str.strip() == "")

        inconsistent_mask = is_fail & empty_just
        for idx in df[inconsistent_mask].index:
            excel_row = idx + 2
            issues.append(StructuralIssue(
                row_index=idx,
                excel_row=excel_row,
                column=just_col,
                value="",
                rule_violated="FAIL_WITHOUT_JUSTIFICATION",
                severity="ERROR",
                description=f"Row marked with verdict '{df.loc[idx, verdict_col]}' must provide a detailed justification."
            ))

    err_count = sum(1 for i in issues if i.severity == "ERROR")
    warn_count = sum(1 for i in issues if i.severity == "WARNING")

    return StructuralQAResult(
        total_rows=total_rows,
        issues_count=len(issues),
        error_count=err_count,
        warning_count=warn_count,
        issues=issues
    )
