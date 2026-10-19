"""Tests for API detector connectors retry logic and Airtable sync."""
import pytest
from unittest.mock import patch, MagicMock
from src.detectors.gptzero import GPTZeroDetector
from src.detectors.originality import OriginalityDetector
from src.detection import AuthorshipService, DetectionDecision, DetectorStatus
from src.integrations.airtable import AirtableSyncClient, AirtableConfig


def test_gptzero_missing_key():
    detector = GPTZeroDetector(api_key="")
    with pytest.raises(ValueError, match="GPTZero API Key is missing"):
        detector.detect("Sample text")


def test_gptzero_empty_text():
    detector = GPTZeroDetector(api_key="fake-key")
    res = detector.detect("")
    assert res.ai_score == 0.0
    assert not res.is_flagged


@patch("src.detectors.gptzero.requests.post")
@patch("src.detectors.gptzero.time.sleep")
def test_gptzero_retry_on_429(mock_sleep, mock_post):
    mock_429 = MagicMock()
    mock_429.status_code = 429

    mock_200 = MagicMock()
    mock_200.status_code = 200
    mock_200.json.return_value = {
        "documents": [{
            "completely_generated_prob": 0.88,
            "confidence_category": "high",
            "predicted_class": "AI"
        }]
    }

    mock_post.side_effect = [mock_429, mock_200]

    detector = GPTZeroDetector(api_key="test-key")
    res = detector.detect("This is some sample text for testing retry.")

    assert res.ai_score == 0.88
    assert res.is_flagged is True
    assert mock_post.call_count == 2
    mock_sleep.assert_called_once()


def test_originality_missing_key():
    detector = OriginalityDetector(api_key="")
    with pytest.raises(ValueError, match="Originality API Key is missing"):
        detector.detect("Sample text")


@patch("src.detectors.originality.requests.post")
@patch("src.detectors.originality.time.sleep")
def test_originality_retry_and_success(mock_sleep, mock_post):
    mock_429 = MagicMock()
    mock_429.status_code = 429

    mock_200 = MagicMock()
    mock_200.status_code = 200
    mock_200.json.return_value = {
        "score": {
            "ai": 0.95,
            "original": 0.05,
            "confidence": 0.9
        }
    }

    mock_post.side_effect = [mock_429, mock_200]

    detector = OriginalityDetector(api_key="test-key")
    res = detector.detect("Sample test text here.")

    assert res.ai_score == 0.95
    assert res.is_flagged is True
    assert mock_post.call_count == 2


@pytest.mark.parametrize(
    ("detector_class", "patch_target"),
    [
        (GPTZeroDetector, "src.detectors.gptzero.requests.post"),
        (OriginalityDetector, "src.detectors.originality.requests.post"),
    ],
)
def test_malformed_external_response_abstains(detector_class, patch_target):
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {}

    with patch(patch_target, return_value=response):
        result = AuthorshipService(
            detector_class(api_key="test-key"), model_validated=True
        ).check_text("one two three four five six seven eight")

    assert result.decision is DetectionDecision.INCONCLUSIVE
    assert result.detector_status is DetectorStatus.UNAVAILABLE
    assert result.reason_codes == ["DETECTOR_REPORTED_ERROR"]


@pytest.mark.parametrize(
    ("detector_class", "patch_target", "payload"),
    [
        (GPTZeroDetector, "src.detectors.gptzero.requests.post", {"documents": [{"completely_generated_prob": True}]}),
        (GPTZeroDetector, "src.detectors.gptzero.requests.post", {"documents": [{"completely_generated_prob": False}]}),
        (OriginalityDetector, "src.detectors.originality.requests.post", {"score": {"ai": True}}),
        (OriginalityDetector, "src.detectors.originality.requests.post", {"score": {"ai": False}}),
    ],
)
def test_boolean_external_score_abstains(detector_class, patch_target, payload):
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = payload

    with patch(patch_target, return_value=response):
        result = AuthorshipService(
            detector_class(api_key="test-key"), model_validated=True
        ).check_text("one two three four five six seven eight")

    assert result.decision is DetectionDecision.INCONCLUSIVE
    assert result.detector_status is DetectorStatus.UNAVAILABLE
    assert result.reason_codes == ["DETECTOR_REPORTED_ERROR"]


def test_airtable_sync_unconfigured():
    client = AirtableSyncClient(config=AirtableConfig(api_token="", base_id=""))
    res = client.sync_flagged_rows([{"row": 1}])
    assert res["status"] == "skipped"


@patch("src.integrations.airtable.requests.post")
def test_airtable_sync_success(mock_post):
    mock_res = MagicMock()
    mock_res.status_code = 200
    mock_res.json.return_value = {"records": [{"id": "rec123"}]}
    mock_post.return_value = mock_res

    client = AirtableSyncClient(config=AirtableConfig(api_token="pat123", base_id="app123"))
    records = [{"prompt_id": 1, "rater_id": "r1", "ai_score": 0.85}]
    res = client.sync_flagged_rows(records)

    assert res["status"] == "completed"
    assert res["synced_count"] == 1
