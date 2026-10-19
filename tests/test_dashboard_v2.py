"""Dashboard, API and CLI for the v2 audit (spec 2026-10-01, section 5)."""
import json
import sys
from unittest.mock import patch

import pandas as pd
from fastapi.testclient import TestClient

from src.api import app

CSV = "text/csv"


def _upload(tmp_path):
    path = tmp_path / "rows.csv"
    pd.DataFrame({"rater_id": ["a", "b"], "justification": [
        "The response corrects the false premise politely and offers grounded alternatives.",
        "It cites the right source and the rating follows from the rubric for this task."]}).to_csv(path, index=False)
    return {"file": (path.name, path.read_bytes(), CSV)}


def test_the_dashboard_has_the_agents_table_the_new_cards_and_the_trigger_checkbox():
    page = TestClient(app).get("/").text
    for marker in ('id="explainTriggers"', 'id="statHigh"', 'id="statNotChecked"', 'id="agentsTable"',
                   "formData.append('explain_triggers'", "renderAgents(", 'id="triggerWordsTable"',
                   "renderTriggerWords("):
        assert marker in page


def test_the_api_returns_tiers_and_agents_and_can_skip_triggers(tmp_path):
    client = TestClient(app)
    res = client.post("/api/audit", files=_upload(tmp_path), data={"provider": "heuristic", "only_assigned": "true"})
    summary = res.json()["summary"]
    assert set(summary["tier_counts"]) >= {"Likely AI", "Not checked"} and len(summary["agents"]) == 2
    assert summary["triggers"]["enabled"] is True and isinstance(summary["trigger_words"], list)
    res = client.post("/api/audit", files=_upload(tmp_path),
                      data={"provider": "heuristic", "only_assigned": "true", "explain_triggers": "false"})
    assert res.json()["summary"]["triggers"]["enabled"] is False


def test_the_cli_can_skip_triggers(tmp_path, capsys):
    from src.cli import main

    path = tmp_path / "rows.csv"
    pd.DataFrame({"rater_id": ["a"], "justification": ["The response is honest and grounded about the hurdles."]}
                 ).to_csv(path, index=False)
    with patch.object(sys, "argv", ["cli.py", "audit", str(path), "-o", str(tmp_path / "o.xlsx"), "-p", "heuristic",
                                    "--no-triggers", "--json"]):
        main()
    assert json.loads(capsys.readouterr().out)["summary"]["triggers"]["enabled"] is False
