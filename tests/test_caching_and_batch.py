"""Tests for pipeline caching, batch detection, progress tracking, and cost estimation."""
import os
import pandas as pd
from src.config import PipelineConfig
from src.pipeline import QAPipeline
from src.detectors import get_detector
from src.detectors.trained_detector import TrainedInDomainDetector


def test_pipeline_caching_and_progress(tmp_path):
    df = pd.DataFrame({
        "rater_id": ["user1", "user1", "user2"],
        "row_id": [1, 2, 3],
        "justification": [
            "The model followed all instructions accurately.",
            "The model followed all instructions accurately.",  # Duplicate! Should trigger cache hit
            "Unique comment here for testing."
        ],
        "verdict": ["PASS", "PASS", "FAIL"]
    })
    csv_file = os.path.join(tmp_path, "cache_test.csv")
    out_file = os.path.join(tmp_path, "cache_test_out.xlsx")
    df.to_csv(csv_file, index=False)

    progress_log = []

    def on_progress(curr, total, stage):
        progress_log.append((curr, total, stage))

    cfg = PipelineConfig()
    cfg.detector.provider = "heuristic"
    pipeline = QAPipeline(config=cfg)

    res = pipeline.run(csv_file, output_file=out_file, progress_callback=on_progress)
    assert res.total_rows == 3
    assert len(progress_log) > 0

    # Verify cache hits
    assert "cache_stats" in res.summary
    assert res.summary["cache_stats"]["cache_hits"] >= 1

    # Verify token and cost estimation
    assert "token_usage" in res.summary
    assert res.summary["token_usage"]["total_words"] > 0
    assert "estimated_cost_usd" in res.summary["token_usage"]


def test_trained_detector_vectorized_batch():
    detector = TrainedInDomainDetector()
    texts = [
        "",  # Empty
        "Hi",  # <3 words
        "The model accurately followed all system guidelines and provided high quality code.",
        "Furthermore this robust framework is meticulously crafted."
    ]
    results = detector.detect_batch(texts)
    assert len(results) == 4
    assert results[0].ai_score == 0.0
    assert results[1].ai_score == 0.10
    assert results[2].confidence > 0.0
    assert results[3].confidence > 0.0


def test_base_detector_batch_fallback():
    from src.detectors.base import BaseDetector, DetectionResult

    class DummyDetector(BaseDetector):
        def detect(self, text: str) -> DetectionResult:
            return DetectionResult(provider="dummy", ai_score=0.5, is_flagged=False)

    detector = DummyDetector()
    texts = ["Simple test one.", "Simple test two."]
    results = detector.detect_batch(texts, max_workers=2)
    assert len(results) == 2
    assert results[0].provider == "dummy"
    assert results[1].provider == "dummy"
