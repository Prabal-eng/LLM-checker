"""The registered 4d model gives verdicts despite its closed gate, with no unvalidated label, through the default
engine and the preview engines alike (user decisions 2026-10-01 and 2026-10-08: "this model works for us for now")."""
import json
import os

import openpyxl
import pandas as pd
import pytest

from src.authorship import registry
from src.config import PipelineConfig
from src.detection import AuthorshipService
from src.detectors import get_detector
from src.pipeline import PREVIEW_PROVIDER, QAPipeline

RECORD = {"model_id": "deberta-4d", "benchmark": {"passed": False, "failures": [
    "no_robots: false-AI upper bound 0.0189 > 0.01", "wasitai2026: false-human upper bound 0.1973 > 0.1"]}}


@pytest.fixture
def preview(monkeypatch):
    calls = []

    def fake_service(record, root=None, fallback=None, validated=None):
        calls.append(validated)
        return AuthorshipService(get_detector("heuristic"), model_validated=bool(validated), detector_version=record["model_id"])

    monkeypatch.setattr(registry, "load", lambda path=None: RECORD)
    monkeypatch.setattr(registry, "service_from_record", fake_service)
    return calls


def _pipeline(provider):
    config = PipelineConfig()
    config.detector.provider = provider
    config.structural.skip_unassigned = False
    return QAPipeline(config=config)


def test_the_preview_builds_the_registered_model_with_verdicts_on_and_no_label(preview):
    pipe = _pipeline(PREVIEW_PROVIDER)
    assert preview == [True]
    assert pipe.authorship_service.model_validated is True
    assert pipe.authorship_service.detector_version == "deberta-4d"
    assert pipe.unvalidated_preview is None


def test_a_run_carries_no_unvalidated_label_in_the_summary_or_the_workbook(preview, tmp_path):
    src = tmp_path / "sheet.csv"
    pd.DataFrame({"rater_id": ["r1", "r2"], "row_id": [1, 2], "justification": [
        "The response corrects the false premise politely and offers grounded alternatives to the user.",
        "Furthermore, it is crucial to delve into the comprehensive nature of this robust response."]}).to_csv(src, index=False)
    out = tmp_path / "out.xlsx"
    res = _pipeline("trained").run(str(src), str(out))
    assert res.summary["unvalidated_preview"] is None and res.summary["detector_version"] == "deberta-4d"
    wb = openpyxl.load_workbook(out)
    cells = " ".join(str(c.value) for ws in wb.worksheets for row in ws.iter_rows() for c in row if c.value)
    assert "UNVALIDATED" not in cells.upper()


def test_the_default_engine_serves_the_registered_model_with_verdicts_on(preview):
    pipe = _pipeline("trained")
    assert preview == [True]
    assert pipe.authorship_service.model_validated is True and pipe.authorship_service.detector_version == "deberta-4d"


def test_without_a_registered_model_the_default_engine_falls_back_to_the_legacy_model(monkeypatch):
    monkeypatch.setattr(registry, "load", lambda path=None: None)
    pipe = _pipeline("trained")
    assert pipe.authorship_service.model_validated is False
    assert pipe.authorship_service.detector_version == "trained-legacy-unvalidated"


def test_pasted_text_uses_the_registered_model_with_verdicts_on(preview):
    from src.api import _get_v2_authorship_service

    _get_v2_authorship_service.cache_clear()       # the app builds it once; earlier tests may have filled the cache
    try:
        service = _get_v2_authorship_service()
        assert service.model_validated is True and service.detector_version == "deberta-4d"
    finally:
        _get_v2_authorship_service.cache_clear()


def test_the_preview_refuses_without_a_registered_model(monkeypatch):
    monkeypatch.setattr(registry, "load", lambda path=None: None)
    with pytest.raises(RuntimeError, match="registered model"):
        _pipeline(PREVIEW_PROVIDER)


def test_the_cli_offers_the_preview(preview, tmp_path, capsys):
    import sys
    from unittest.mock import patch

    from src.cli import main

    src = tmp_path / "sheet.csv"
    pd.DataFrame({"rater_id": ["r1"], "justification": ["The response is honest and grounded about the hurdles."]}).to_csv(src, index=False)
    with patch.object(sys, "argv", ["cli.py", "audit", str(src), "-o", str(tmp_path / "o.xlsx"), "--json"]):
        main()
    summary = json.loads(capsys.readouterr().out)["summary"]
    assert summary["detector_version"] == "deberta-4d" and summary["unvalidated_preview"] is None


def test_the_dashboard_shows_no_unvalidated_label():
    from fastapi.testclient import TestClient

    from src.api import app

    import re

    page = TestClient(app).get("/").text
    for value in ("trained", "preview-v5"):     # the paid APIs stay unvalidated: they always abstain
        assert "unvalidated" not in re.search(rf'<option value="{value}"[^>]*>[^<]*', page).group(0).lower()


# --- the previous model (v5) as a preview too ------------------------------------------------------------

V5 = {"model_id": "distilroberta-v5", "benchmark": {"passed": True, "failures": []}}  # its frozen v3 pass


def test_the_previous_model_is_offered_as_a_preview_with_its_v4_failures(monkeypatch, tmp_path):
    from src.authorship import headtohead
    from src.pipeline import PREVIEW_V5

    loaded = []
    monkeypatch.setattr(registry, "load", lambda path=None: loaded.append(path) or (V5 if path == headtohead.INCUMBENT_PATH else RECORD))
    monkeypatch.setattr(registry, "service_from_record", lambda record, root=None, fallback=None, validated=None:
                        AuthorshipService(get_detector("heuristic"), model_validated=bool(validated),
                                          detector_version=record["model_id"]))
    report = tmp_path / "head_to_head.json"
    report.write_text(json.dumps({"models": {"incumbent": {"model_id": "distilroberta-v5", "failures": [
        "maga: false-AI upper bound 0.0240 > 0.01"]}, "4d": {"model_id": "deberta-4d", "failures": []}}}), encoding="utf-8")
    monkeypatch.setattr(headtohead, "report_file", lambda version=None: report)
    pipe = _pipeline(PREVIEW_V5)
    assert loaded == [headtohead.INCUMBENT_PATH]
    assert pipe.authorship_service.detector_version == "distilroberta-v5" and pipe.unvalidated_preview is None


def test_the_dashboard_and_cli_offer_the_previous_model():
    from fastapi.testclient import TestClient

    from src.api import app
    from src.pipeline import PREVIEW_V5

    assert f'value="{PREVIEW_V5}"' in TestClient(app).get("/").text
    from src import cli
    assert PREVIEW_V5 in open(cli.__file__, encoding="utf-8").read()


def test_the_dashboard_is_never_cached_and_preselects_the_newest_model():
    """After a restart the browser showed a cached page without the 4d option, so the page isn't cached; the 4d
    model is the default engine, labelled as the newest and pre-selected."""
    import re

    from fastapi.testclient import TestClient

    from src.api import app

    res = TestClient(app).get("/")
    assert "no-store" in res.headers.get("cache-control", "")
    option = re.search(r'<option value="trained"[^>]*>[^<]*', res.text).group(0)
    assert "selected" in option and "newest" in option and "4d" in option
