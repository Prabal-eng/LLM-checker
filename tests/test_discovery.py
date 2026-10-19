"""Unit tests for ColumnDiscovery."""
import pandas as pd
from src.config import StructuralConfig
from src.qa.discovery import ColumnDiscovery, DiscoveredColumns


def test_column_discovery_standard():
    df = pd.DataFrame({
        "rater_id": ["user1", "user2"],
        "prompt_id": [101, 102],
        "category": ["A", "B"],
        "overall_assessment_rationale": ["Good text explanation here", "Another explanation"],
        "verdict": ["PASS", "FAIL"]
    })
    disc = ColumnDiscovery.discover(df)
    assert disc.rater_col == "rater_id"
    assert disc.text_col == "overall_assessment_rationale"
    assert disc.verdict_col == "verdict"
    assert disc.row_id_col == "prompt_id"


def test_column_discovery_fallbacks():
    df = pd.DataFrame({
        "annotator": ["user1"],
        "sample_id": [1],
        "notes": ["A brief review note."],
        "score": ["5"]
    })
    disc = ColumnDiscovery.discover(df)
    assert disc.rater_col == "annotator"
    assert "notes" in disc.text_cols
    assert disc.verdict_col == "score"
    assert disc.row_id_col == "sample_id"


def test_column_discovery_configured_overrides():
    df = pd.DataFrame({
        "col_a": ["user1"],
        "col_b": ["Custom rationale string here"],
        "col_c": ["PASS"]
    })
    cfg = StructuralConfig(
        rater_column="col_a",
        text_columns=["col_b"],
        verdict_column="col_c"
    )
    disc = ColumnDiscovery.discover(df, config=cfg)
    assert disc.rater_col == "col_a"
    assert disc.text_col == "col_b"
    assert disc.verdict_col == "col_c"


def test_column_discovery_average_length_fallback():
    long_texts = ["This is a very long descriptive text evaluation written by a human tester." * 2]
    df = pd.DataFrame({
        "id": [1],
        "misc_description": long_texts
    })
    disc = ColumnDiscovery.discover(df)
    assert "misc_description" in disc.text_cols


def test_column_discovery_export_headers():
    """Spaced/cased ID header, email submitter and a rater*-named comment column (data (6).xlsx)."""
    df = pd.DataFrame({
        "Task ID": ["01a1147d-6cc9", "01a1147d-6cd4"],
        "Submitted by (email)": ["a@x.ai", "b@x.ai"],
        "replicate": ["1", "2"],
        "skipReason": ["", ""],
        "raterComment": ["The prompt is clear because it is asking", "Passes all 3 criteria for the task"],
    })
    disc = ColumnDiscovery.discover(df)
    assert disc.rater_col == "Submitted by (email)"
    assert disc.row_id_col == "Task ID"
    assert disc.text_cols == ["raterComment"]
