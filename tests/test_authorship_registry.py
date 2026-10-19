"""Registry gate and the out-of-distribution check; no torch or model file needed."""

import numpy as np
import pytest

from src.authorship import registry
from src.corpus.text import text_view_fingerprint
from src.authorship.detector import OodGate
from src.detection import DetectionDecision


def _record(root, *, passed=True):
    art = root / "models" / "m" / "epoch-1"
    art.mkdir(parents=True)
    (art / "model.safetensors").write_bytes(b"weights")
    bank = root / "models" / "m" / "bank.npy"
    np.save(bank, np.eye(4, dtype=np.float32))
    record = {
        "model_id": "m",
        "precision": "bf16",
        "text_normalization": "collapse_whitespace",
        "text_view_sha256": text_view_fingerprint("collapse_whitespace"),
        "scoring_code_sha256": registry.scoring_code_sha256(),
        "artifact": {"relative_path": "models/m/epoch-1", "sha256": registry.sha256_file(art / "model.safetensors"),
                     "files_sha256": registry.directory_sha256(art)},
        "calibration": {
            "platt": [1.0, 0.0],
            "bands": [{"lower_words": 50, "upper_words": 100, "supported": False, "ai_min": None, "human_max": None},
                      {"lower_words": 100, "upper_words": None, "supported": True, "ai_min": 0.9, "human_max": 0.2}],
            "ood": {"bank_relative_path": "models/m/bank.npy", "bank_sha256": registry.sha256_file(bank),
                    "k": 2, "threshold": 0.5},
        },
    }
    return _sealed(record, passed)


def _sealed(record, passed=True):
    """The benchmark's record of a run, bound to the registry entry it scored."""
    record.pop("benchmark", None)
    record["benchmark"] = registry.seal(record, passed=passed)
    return record


def test_verify_passes_only_with_benchmark_and_matching_hashes(tmp_path):
    record = _record(tmp_path)
    assert registry.verify(record, tmp_path) == (True, "validated")
    assert registry.verify({**record, "benchmark": {"passed": False}}, tmp_path)[0] is False
    assert registry.verify(None, tmp_path) == (False, "no registry")


def test_changed_weights_close_the_gate(tmp_path):
    record = _record(tmp_path)
    (tmp_path / "models" / "m" / "epoch-1" / "model.safetensors").write_bytes(b"tampered weights")
    assert registry.verify(record, tmp_path) == (False, "model artifact hash mismatch")


def test_missing_artifact_closes_the_gate(tmp_path):
    record = _record(tmp_path)
    (tmp_path / "models" / "m" / "epoch-1" / "model.safetensors").unlink()
    assert registry.verify(record, tmp_path) == (False, "model artifact missing")


def test_policy_uses_only_supported_bands(tmp_path):
    policy = registry.build_policy(_record(tmp_path))
    assert [b.lower_words for b in policy.bands] == [100]
    assert policy.min_word_count == 100


def test_service_falls_back_without_a_model_file(tmp_path):
    record = _record(tmp_path)
    path = tmp_path / "registry.json"
    registry.save(record, path)
    (tmp_path / "models" / "m" / "epoch-1" / "model.safetensors").unlink()
    sentinel = object()
    assert registry.build_service(path, tmp_path, fallback=sentinel) is sentinel


def test_unvalidated_registered_model_abstains_without_loading(tmp_path):
    record = _record(tmp_path, passed=False)
    path = tmp_path / "registry.json"
    registry.save(record, path)
    service = registry.build_service(path, tmp_path)
    result = service.check_text(" ".join(["the chart labels are clear and accurate"] * 20))
    assert result.decision is DetectionDecision.INCONCLUSIVE
    assert result.reason_codes == ["MODEL_NOT_VALIDATED"]
    assert service.detector._model is None


def test_ood_gate_distance_and_threshold():
    bank = np.eye(3, dtype=np.float32)
    gate = OodGate(bank, k=1, threshold=0.2)
    near = np.array([[1.0, 0.0, 0.0]])
    far = np.array([[0.577, 0.577, 0.577]])
    d = gate.distances(np.vstack([near, far]))
    assert d[0] == pytest.approx(0.0, abs=1e-6) and d[1] > 0.4
    assert gate.in_distribution(d).tolist() == [True, False]
    assert OodGate(bank, 1, None).in_distribution(d).all()


def _with_short(root, record):
    art = root / "models" / "s" / "epoch-0"
    art.mkdir(parents=True)
    (art / "model.safetensors").write_bytes(b"short weights")
    bank = root / "models" / "s" / "bank.npy"
    np.save(bank, np.eye(4, dtype=np.float32))
    record["short_model"] = {
        "model_id": "s", "max_tokens": 160, "precision": "bf16", "platt": [1.0, 0.0], "ranges": [[25, 100]],
        "artifact": {"relative_path": "models/s/epoch-0", "sha256": registry.sha256_file(art / "model.safetensors"),
                     "files_sha256": registry.directory_sha256(art)},
        "ood": {"bank_relative_path": "models/s/bank.npy", "bank_sha256": registry.sha256_file(bank),
                "k": 2, "threshold": 0.5},
    }
    return _sealed(record)


def test_short_model_is_verified_and_routed(tmp_path):
    from src.authorship.detector import LengthRoutedDetector

    record = _with_short(tmp_path, _record(tmp_path))
    assert registry.verify(record, tmp_path) == (True, "validated")
    path = tmp_path / "registry.json"
    registry.save(record, path)
    service = registry.build_service(path, tmp_path)
    assert isinstance(service.detector, LengthRoutedDetector)
    assert service.detector.short.max_tokens == 160
    (tmp_path / "models" / "s" / "epoch-0" / "model.safetensors").write_bytes(b"tampered")
    assert registry.verify(record, tmp_path) == (False, "short-text model artifact hash mismatch")


def test_length_router_sends_each_text_to_its_model():
    from src.authorship.detector import LengthRoutedDetector
    from src.detectors.base import BaseDetector, DetectionResult

    class Fixed(BaseDetector):
        def __init__(self, score):
            super().__init__(0.5)
            self.score, self.calls = score, []

        def detect(self, text):
            return DetectionResult(provider="f", ai_score=self.score, is_flagged=False)

        def detect_batch(self, texts, max_workers=1):
            self.calls.append(list(texts))
            return [self.detect(t) for t in texts]

    main, short = Fixed(0.1), Fixed(0.9)
    router = LengthRoutedDetector(main, short, [(25, 50), (50, 100)])
    texts = ["word " * 30, "word " * 150, "word " * 80, "word " * 10]
    results = router.detect_batch(texts)
    assert [r.ai_score for r in results] == [0.9, 0.1, 0.9, 0.1]
    assert [r.analysis["model"] for r in results] == ["short", "main", "short", "main"]
    assert len(short.calls) == 1 and len(main.calls) == 1  # one batch per model


def test_main_only_drops_the_specialist_and_restores_main_thresholds(tmp_path):
    record = _with_short(tmp_path, _record(tmp_path))
    record["calibration"]["main_bands"] = [dict(b, model="main") for b in record["calibration"]["bands"]]
    record["calibration"]["bands"] = [dict(record["calibration"]["bands"][0], supported=True, ai_min=0.8,
                                           human_max=0.1, model="short")] + record["calibration"]["bands"][1:]
    plain = registry.main_only(record)
    assert "short_model" not in plain and "short_model" in record
    assert plain["calibration"]["bands"][0]["supported"] is False
    service = registry.service_from_record(plain, tmp_path, validated=True)
    assert service.model_validated and not hasattr(service.detector, "short")


def test_editing_the_registry_after_the_benchmark_closes_the_gate(tmp_path):
    """Ralph R08-4: a pass vouches for the exact thresholds, calibration and text view it scored."""
    record = _record(tmp_path)
    record["calibration"]["bands"][1]["ai_min"] = 0.5
    assert registry.verify(record, tmp_path) == (False, "registry changed after the locked benchmark")
    unsealed = _record(tmp_path / "c")
    unsealed["benchmark"] = {"passed": True}
    assert registry.verify(unsealed, tmp_path / "c") == (False, "registry changed after the locked benchmark")


def test_a_record_without_config_hashes_or_a_text_view_stays_closed(tmp_path):
    record = _record(tmp_path)
    del record["artifact"]["files_sha256"]
    _sealed(record)
    assert registry.verify(record, tmp_path) == (False, "model has no recorded config and tokenizer hash")
    record = _record(tmp_path / "b")
    del record["text_normalization"]
    _sealed(record)
    assert registry.verify(record, tmp_path / "b") == (False, "no recorded text view")


def test_a_changed_text_view_closes_the_gate():
    """Ralph R09-3: the seal bound the view's name; a changed view behind the same name kept it open."""
    record = _record(__import__("pathlib").Path(__import__("tempfile").mkdtemp()))
    record["text_view_sha256"] = "0" * 64
    _sealed(record)
    assert registry.verify(record, None)[1] == "text view changed since calibration"


def test_the_benchmark_result_itself_is_sealed(tmp_path):
    """Ralph R09-4: flipping a sealed failure to passed must not open the gate."""
    record = _record(tmp_path, passed=False)
    record["benchmark"]["passed"] = True
    assert registry.verify(record, tmp_path) == (False, "registry changed after the locked benchmark")


def test_changed_scoring_code_closes_the_gate(tmp_path):
    """Ralph R10-3: the seal bound the view's name and a 3-input fingerprint, not the deobfuscation or
    policy code that also decides outcomes."""
    record = _record(tmp_path)
    record["scoring_code_sha256"] = "0" * 64
    _sealed(record)
    assert registry.verify(record, tmp_path) == (False, "scoring code changed since calibration")


def test_the_run_history_is_sealed(tmp_path):
    """Ralph R10-5: deleting or editing a locked run, or its rerun reason, must close the gate."""
    record = _record(tmp_path)
    record["locked_benchmark_history"] = [{"corpus_version": "v3", "status": "finished", "rerun_reason": None}]
    _sealed(record)
    assert registry.verify(record, tmp_path) == (True, "validated")
    record["locked_benchmark_history"][0]["rerun_reason"] = "edited"
    assert registry.verify(record, tmp_path) == (False, "registry changed after the locked benchmark")


def test_text_digest_ignores_line_endings(tmp_path):
    """Ralph R10-5: a report hashed on a Windows CRLF checkout must match the committed LF blob."""
    (tmp_path / "a.json").write_bytes(b'{"x": 1}\r\n')
    (tmp_path / "b.json").write_bytes(b'{"x": 1}\n')
    assert registry.text_sha256(tmp_path / "a.json") == registry.text_sha256(tmp_path / "b.json")


def test_every_scoring_module_change_moves_the_digest(tmp_path, monkeypatch):
    """Ralph R11-2: the Platt map (calibration.py) and the detector base were outside the digest, and
    the earlier test only overwrote the recorded value instead of changing a file."""
    import shutil
    assert {"src/authorship/calibration.py", "src/detectors/base.py"} <= set(registry._SCORING_CODE)
    for rel in registry._SCORING_CODE:
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(registry._REPO / rel, tmp_path / rel)
    monkeypatch.setattr(registry, "_REPO", tmp_path)
    before = registry.scoring_code_sha256()
    target = tmp_path / "src/authorship/calibration.py"
    target.write_text(target.read_text(encoding="utf-8") + "\n# shifted\n", encoding="utf-8")
    assert registry.scoring_code_sha256() != before


def test_a_missing_benchmark_report_closes_the_gate(tmp_path):
    """Ralph R11-4: deleting the report kept the gate open."""
    record = _record(tmp_path)
    record["benchmark"]["report"] = "data/benchmark/does-not-exist/benchmark.json"
    record["benchmark"] = registry.seal({k: v for k, v in record.items() if k != "benchmark"},
                                        **{k: v for k, v in record["benchmark"].items() if k != "record_sha256"})
    assert registry.verify(record, tmp_path) == (False, "benchmark report missing")


def test_a_record_without_an_inference_precision_stays_closed(tmp_path):
    record = _record(tmp_path)
    del record["precision"]
    assert registry.verify(_sealed(record), tmp_path) == (False, "no recorded inference precision")
    record["precision"] = "int8"
    assert registry.verify(_sealed(record), tmp_path) == (False, "no recorded inference precision")


def test_a_short_model_without_an_inference_precision_stays_closed(tmp_path):
    record = _with_short(tmp_path, _record(tmp_path))
    del record["short_model"]["precision"]
    assert registry.verify(_sealed(record), tmp_path) == (
        False, "no recorded inference precision for the short-text model")


def test_the_service_scores_in_the_recorded_precision(tmp_path):
    record = _with_short(tmp_path, _record(tmp_path))
    record["precision"], record["short_model"]["precision"] = "fp16", "fp32"
    service = registry.service_from_record(_sealed(record), tmp_path, validated=True)
    assert service.detector.main.precision == "fp16" and service.detector.short.precision == "fp32"


def test_a_pre_4d_record_scores_under_bf16_as_it_was_calibrated(tmp_path):
    record = _record(tmp_path)
    del record["precision"]
    service = registry.service_from_record(_sealed(record), tmp_path, validated=True)
    assert service.detector.precision == registry.LEGACY_PRECISION == "bf16"
    assert registry.verify(record, tmp_path)[0] is False  # the gate itself stays closed


def test_records_that_state_a_precision_load_half_weights_on_the_gpu(tmp_path):
    record = _with_short(tmp_path, _record(tmp_path))
    service = registry.service_from_record(_sealed(record), tmp_path, validated=True)
    assert service.detector.main.half_weights and service.detector.short.half_weights
    legacy = _record(tmp_path / "old")
    del legacy["precision"]
    assert not registry.service_from_record(_sealed(legacy), tmp_path / "old", validated=True).detector.half_weights
