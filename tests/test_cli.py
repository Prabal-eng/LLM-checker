"""Tests for the CLI module."""
import os
import sys
import json
import pytest
import pandas as pd
from unittest.mock import patch
from src.cli import main


def test_cli_audit_json(tmp_path, capsys):
    df = pd.DataFrame({
        "rater_id": ["user1", "user2"],
        "prompt_id": [1, 2],
        "justification": [
            "The model output is clear, concise, and meets all criteria.",
            "Furthermore this is crucial and robust."
        ],
        "verdict": ["PASS", "PASS"]
    })
    csv_path = os.path.join(tmp_path, "cli_test.csv")
    out_path = os.path.join(tmp_path, "cli_out.xlsx")
    df.to_csv(csv_path, index=False)

    test_args = [
        "cli.py", "audit", csv_path,
        "-o", out_path,
        "-p", "heuristic",
        "--json"
    ]
    with patch.object(sys, "argv", test_args):
        main()

    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data["total_rows"] == 2
    assert "summary" in data
    assert os.path.exists(out_path)


def test_cli_help_exit(capsys):
    with patch.object(sys, "argv", ["cli.py"]):
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 1
