"""Tests for FastAPI endpoints."""
import io
import pytest
from fastapi.testclient import TestClient
import pandas as pd
from src.api import app, MAX_UPLOAD_BYTES
from src.detection import DetectionDecision, DetectorStatus, TextCheckResponse


@pytest.fixture
def client():
    return TestClient(app)


def test_serve_index(client):
    res = client.get("/")
    assert res.status_code == 200
    assert "Excel AI" in res.text or "html" in res.headers.get("content-type", "")


def test_inspect_columns_invalid_ext(client):
    res = client.post(
        "/api/inspect-columns",
        files={"file": ("test.txt", b"dummy content", "text/plain")}
    )
    assert res.status_code == 400
    assert "supported" in res.json()["detail"].lower()


def test_inspect_columns_csv(client):
    csv_data = "rater_id,overall_assessment_rationale,verdict\nuser1,Looks good and verified,PASS\n"
    res = client.post(
        "/api/inspect-columns",
        files={"file": ("test.csv", csv_data.encode("utf-8"), "text/csv")}
    )
    assert res.status_code == 200
    data = res.json()
    assert "columns" in data
    assert "overall_assessment_rationale" in data["columns"]
    assert "overall_assessment_rationale" in data["recommended"]


def test_run_audit_endpoint_csv(client):
    csv_data = (
        "rater_id,prompt_id,justification,verdict\n"
        "rater_1,1,The model responded with high quality and followed all instructions.,PASS\n"
        "rater_2,2,Furthermore this is crucial and robust meticulously.,PASS\n"
    )
    res = client.post(
        "/api/audit",
        files={"file": ("sample.csv", csv_data.encode("utf-8"), "text/csv")},
        data={"provider": "heuristic", "threshold": "0.70", "only_assigned": "true"}
    )
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"
    assert data["total_rows"] == 2
    assert "output_filename" in data
    assert "token_usage" in data["summary"]
    assert "cache_stats" in data["summary"]

    # Test download endpoint
    out_file = data["output_filename"]
    dl_res = client.get(f"/api/download/{out_file}")
    assert dl_res.status_code == 200
    assert len(dl_res.content) > 0


def test_download_not_found(client):
    res = client.get("/api/download/non_existent_audit_file.xlsx")
    assert res.status_code == 404


def test_file_too_large(client, monkeypatch):
    # Temporarily set max upload to 100 bytes
    monkeypatch.setattr("src.api.MAX_UPLOAD_BYTES", 100)
    big_data = b"x" * 200
    res = client.post(
        "/api/inspect-columns",
        files={"file": ("sample.csv", big_data, "text/csv")}
    )
    assert res.status_code == 413
    assert "too large" in res.json()["detail"].lower()


def test_v2_text_check_skeleton_contract(client):
    health = client.get("/api/v2/health")
    assert health.status_code == 200
    assert health.json()["contract_version"] == "v2"

    before = client.get("/api/v2/metrics").json()
    response = client.post(
        "/api/v2/check-text",
        json={"text": "A standalone response that needs an authorship assessment."},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["decision"] == "inconclusive"
    assert body["score"] is None
    assert body["score_valid"] is False
    assert body["detector_status"] == "available"
    assert body["reason_codes"] == ["MODEL_NOT_VALIDATED"]
    assert body["evidence"] == []
    assert body["detector_version"] == "trained-indomain-legacy-unvalidated"
    assert body["policy_version"] == "v2.0.0-provisional"
    assert set(body) == {
        "decision", "score", "score_valid", "detector_status", "reason_codes",
        "evidence", "detector_version", "policy_version",
    }
    assert {item.value for item in DetectionDecision} == {
        "likely_ai_generated", "likely_human_written", "inconclusive",
    }

    after = client.get("/api/v2/metrics").json()
    assert after["text_checks_total"] == before["text_checks_total"] + 1
    assert after["inconclusive_total"] == before["inconclusive_total"] + 1


@pytest.mark.parametrize(
    ("value", "reason"),
    [
        ("", "INSUFFICIENT_TEXT"),
        (123, "UNSUPPORTED_INPUT_TYPE"),
        (None, "UNSUPPORTED_INPUT_TYPE"),
        pytest.param("x" * 32_768, "UNSUPPORTED_INPUT_LENGTH", id="too-long"),
    ],
)
def test_v2_unsupported_or_insufficient_input_abstains(client, value, reason):
    response = client.post("/api/v2/check-text", json={"text": value})

    assert response.status_code == 200
    body = response.json()
    assert body["decision"] == "inconclusive"
    assert body["score_valid"] is False
    assert body["reason_codes"] == [reason]


@pytest.mark.parametrize(
    "invalid_fields",
    [
        {"score": None, "score_valid": True},
        {"score": 0.9, "score_valid": False},
        {"decision": DetectionDecision.LIKELY_AI_GENERATED},
        {"reason_codes": []},
        {"detector_version": ""},
        {"policy_version": ""},
    ],
)
def test_v2_contract_rejects_contradictory_results(invalid_fields):
    fields = {
        "decision": DetectionDecision.INCONCLUSIVE,
        "score": None,
        "score_valid": False,
        "detector_status": DetectorStatus.NOT_EVALUATED,
        "reason_codes": ["NOT_EVALUATED"],
        "evidence": [],
        "detector_version": "skeleton",
        "policy_version": "v2-skeleton",
    }
    fields.update(invalid_fields)
    with pytest.raises(ValueError):
        TextCheckResponse(**fields)


def test_v2_check_text_accepts_a_known_context_and_rejects_others(client):
    text = "A standalone response that needs an authorship assessment."
    ok = client.post("/api/v2/check-text", json={"text": text, "context": "rater_comment"})
    assert ok.status_code == 200 and ok.json()["decision"] == "inconclusive"
    bad = client.post("/api/v2/check-text", json={"text": text, "context": "essay"})
    assert bad.status_code == 422
