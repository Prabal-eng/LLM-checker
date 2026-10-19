"""Unit tests for structural and mechanical QA."""
import pandas as pd
from src.config import StructuralConfig
from src.qa.structural import run_structural_qa


def test_structural_qa_flags_blanks_and_mismatches():
    config = StructuralConfig(
        required_columns=["rater_id", "row_id", "verdict", "justification"],
        fail_verdicts=["FAIL"],
        valid_verdicts=["PASS", "FAIL"]
    )

    df = pd.DataFrame([
        # Row 0: Valid
        {"rater_id": "R1", "row_id": "T1", "verdict": "PASS", "justification": "Checked and works fine."},
        # Row 1: Blank required field (rater_id)
        {"rater_id": "", "row_id": "T2", "verdict": "PASS", "justification": "Passed test."},
        # Row 2: Invalid verdict
        {"rater_id": "R2", "row_id": "T3", "verdict": "MAYBE", "justification": "Uncertain"},
        # Row 3: FAIL without justification
        {"rater_id": "R3", "row_id": "T4", "verdict": "FAIL", "justification": ""},
        # Row 4: Duplicate of Row 0
        {"rater_id": "R1", "row_id": "T1", "verdict": "PASS", "justification": "Duplicate submission"},
    ])

    result = run_structural_qa(df, config)

    rules_violated = [issue.rule_violated for issue in result.issues]
    assert "BLANK_REQUIRED_FIELD" in rules_violated
    assert "INVALID_VERDICT_VALUE" in rules_violated
    assert "FAIL_WITHOUT_JUSTIFICATION" in rules_violated
    assert "DUPLICATE_RECORD" in rules_violated
    assert result.error_count >= 3
