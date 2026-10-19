"""Increment 4d full training: checks inside an epoch, early stopping, LoRA merging."""

import pytest

from src.bakeoff import neural
from src.bakeoff.neural import check_points
from tests.tiny_encoder import rows, tiny_encoder


def test_check_points_fall_on_optimizer_steps_and_end_the_epoch():
    assert check_points(100, 4, 2) == [26, 50, 76, 100]
    assert check_points(7, 4, 2) == [2, 4, 6, 7]
    assert check_points(10, 1, 2) == [10]


def _tiny_classifier(tmp_path, monkeypatch, epochs):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(neural, "_device", lambda: torch.device("cpu"))  # never touch a busy GPU
    return neural.FineTunedClassifier(model_name=str(tiny_encoder(tmp_path / "tiny")), max_tokens=16,
                                      max_train=None, epochs=epochs, batch_size=8, accumulate=2,
                                      text_view="collapse_whitespace")


def test_fit_checks_each_quarter_epoch_and_stops_when_told(tmp_path, monkeypatch):
    clf = _tiny_classifier(tmp_path, monkeypatch, epochs=3)
    calls = []

    def on_check(state):
        calls.append((state["epoch"], state["check"], state["micro"]))
        return len(calls) == 3

    clf.fit(rows(64), balance_sources=True, checks_per_epoch=4, on_check=on_check)
    assert calls == [(0, 1, 2), (0, 2, 4), (0, 3, 6)]


def test_fit_resumes_inside_an_epoch_from_a_check_state(tmp_path, monkeypatch):
    first = _tiny_classifier(tmp_path, monkeypatch, epochs=2)
    saved = {}

    def stop_at_second(state):
        if state["check"] == 2:
            saved.update(state)
            return True
        return False

    first.fit(rows(64), balance_sources=True, checks_per_epoch=4, on_check=stop_at_second)
    second = _tiny_classifier(tmp_path, monkeypatch, epochs=2)
    calls = []
    second.fit(rows(64), balance_sources=True, checks_per_epoch=4, resume=saved,
               on_check=lambda s: calls.append((s["epoch"], s["check"])) or False)
    assert calls == [(0, 3), (0, 4), (1, 1), (1, 2), (1, 3), (1, 4)]


def test_export_merged_matches_the_adapter_model(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    base_dir = tiny_encoder(tmp_path / "tiny")
    model = neural.apply_lora(AutoModelForSequenceClassification.from_pretrained(base_dir), {"r": 4, "alpha": 8})
    torch.manual_seed(1)
    for name, p in model.named_parameters():
        if "lora_B" in name:
            torch.nn.init.normal_(p, std=0.5)  # LoRA B starts at zero; make the adapter matter
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(base_dir)
    adapter = tmp_path / "check-0.3"
    model.save_pretrained(adapter)
    tokenizer.save_pretrained(adapter)
    out = neural.export_merged(str(base_dir), adapter, tmp_path / "merged-0.3")
    merged = AutoModelForSequenceClassification.from_pretrained(out).eval()
    enc = tokenizer(["the chart labels vocab_a", "river stone"], return_tensors="pt", padding=True)
    with torch.inference_mode():
        assert torch.allclose(merged(**enc).logits, model(**enc).logits, atol=1e-5)
    assert not (out / "adapter_config.json").exists()
    assert (out / "tokenizer.json").exists()


def test_the_detector_rejects_an_unknown_precision_before_loading():
    from src.authorship.detector import AuthorshipModelDetector

    with pytest.raises(ValueError):
        AuthorshipModelDetector("unused", platt=(1.0, 0.0), precision="int8")
    assert AuthorshipModelDetector("unused", platt=(1.0, 0.0)).precision == "bf16"


def test_the_detector_autocasts_in_its_recorded_precision():
    torch = pytest.importorskip("torch")
    from src.authorship.detector import AuthorshipModelDetector

    def dtype(p):
        return AuthorshipModelDetector("unused", platt=(1.0, 0.0), precision=p).autocast_dtype()

    assert dtype("fp16") is torch.float16 and dtype("bf16") is torch.bfloat16 and dtype("fp32") is None


def test_a_fp16_detector_scores_on_the_cpu(tmp_path):
    pytest.importorskip("torch")
    from src.authorship.detector import AuthorshipModelDetector

    detector = AuthorshipModelDetector(tiny_encoder(tmp_path / "tiny"), platt=(1.0, 0.0), max_tokens=16,
                                       device="cpu", text_view="collapse_whitespace", precision="fp16")
    raw, emb = detector.raw_and_embeddings(["the chart labels vocab_a", "river stone"])
    assert raw.shape == (2,) and emb.shape[0] == 2


def test_calibration_passes_the_training_precision_and_writes_per_candidate():
    from src.authorship import calibrate

    training = {"artifact_path": "unused", "text_normalization": "collapse_whitespace", "max_tokens": 16}
    assert calibrate._detector({**training, "precision": "fp16"}).precision == "fp16"
    assert calibrate._detector(training).precision == "bf16"  # records trained before 4d
    assert calibrate.calibration_file(candidate="deberta-v3-base").parts[-2:] == ("deberta-v3-base",
                                                                                   "calibration.json")


def test_fit_releases_the_resume_state_once_applied(tmp_path, monkeypatch):
    first = _tiny_classifier(tmp_path, monkeypatch, epochs=1)
    saved = {}
    first.fit(rows(64), balance_sources=True, checks_per_epoch=4,
              on_check=lambda s: saved.update(s) or s["check"] == 2)
    second = _tiny_classifier(tmp_path, monkeypatch, epochs=1)
    second.fit(rows(64), balance_sources=True, checks_per_epoch=4, resume=saved, on_check=lambda s: False)
    assert saved == {}  # the multi-GB state is not held for the rest of the run


def test_export_merged_restores_base_weights_the_checkpoint_lacked(tmp_path):
    # DeBERTa-v3-large's checkpoint has no pooler: the classifier's pooler starts random and stays frozen
    # under LoRA. The merged model must carry the pooler the adapter model was validated with.
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DebertaV2Config,
                              DebertaV2Model)

    config = DebertaV2Config(vocab_size=16, hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                             intermediate_size=64, max_position_embeddings=64, num_labels=2, pooler_hidden_size=32)
    base_dir = tmp_path / "base"
    torch.manual_seed(0)
    DebertaV2Model(config).save_pretrained(base_dir)  # the encoder alone: no pooler, no classifier
    model, new_weights = neural.load_classifier(str(base_dir), torch.float32)
    assert any(k.startswith("pooler.") for k in new_weights)
    model = neural.apply_lora(model, {"r": 4, "alpha": 8})
    torch.manual_seed(1)
    for name, p in model.named_parameters():
        if "lora_B" in name:
            torch.nn.init.normal_(p, std=0.5)
    model.eval()
    adapter = tmp_path / "check-0.1"
    model.save_pretrained(adapter)
    neural.save_new_weights(new_weights, adapter)
    AutoTokenizer.from_pretrained(tiny_encoder(tmp_path / "tiny")).save_pretrained(adapter)
    out = neural.export_merged(str(base_dir), adapter, tmp_path / "merged-0.1")
    merged = AutoModelForSequenceClassification.from_pretrained(out).eval()
    ids = torch.tensor([[1, 5, 6, 7, 2], [1, 8, 9, 2, 0]])
    mask = (ids != 0).long()
    with torch.inference_mode():
        assert torch.allclose(merged(input_ids=ids, attention_mask=mask).logits,
                              model(input_ids=ids, attention_mask=mask).logits, atol=1e-5)


def test_gpu_weights_load_in_the_recorded_precision_and_cpu_stays_fp32():
    torch = pytest.importorskip("torch")
    from src.authorship.detector import AuthorshipModelDetector

    def dtype(precision, half, device):
        return AuthorshipModelDetector("unused", platt=(1.0, 0.0), precision=precision,
                                       half_weights=half).load_dtype(device)

    assert dtype("fp16", True, "cuda") is torch.float16 and dtype("bf16", True, "cuda") is torch.bfloat16
    assert dtype("fp16", True, "cpu") is torch.float32
    assert dtype("bf16", False, "cuda") is torch.float32  # a pre-4d record keeps the weights it was calibrated with
    assert dtype("fp32", True, "cuda") is torch.float32


def test_non_finite_scores_fail_loudly(tmp_path):
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace

    from src.authorship.detector import AuthorshipModelDetector

    class NaNModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.zeros(1))

        def forward(self, input_ids, attention_mask, output_hidden_states=True):
            n = input_ids.shape[0]
            return SimpleNamespace(logits=torch.full((n, 2), float("nan")),
                                   hidden_states=[torch.zeros(n, input_ids.shape[1], 4)])

    from transformers import AutoTokenizer

    detector = AuthorshipModelDetector("unused", platt=(1.0, 0.0), max_tokens=16, device="cpu",
                                       text_view="collapse_whitespace")
    detector._tokenizer = AutoTokenizer.from_pretrained(tiny_encoder(tmp_path / "tiny"))
    detector._model = NaNModel()
    with pytest.raises(FloatingPointError):
        detector.raw_and_embeddings(["the chart labels vocab_a"])


def test_lora_keeps_its_frozen_base_in_half_precision_on_the_gpu_only():
    torch = pytest.importorskip("torch")
    assert neural.base_load_dtype({"r": 16}, "fp16", "cuda") is torch.float16
    assert neural.base_load_dtype({"r": 16}, "bf16", "cuda") is torch.bfloat16
    assert neural.base_load_dtype({"r": 16}, "fp16", "cpu") is torch.float32
    assert neural.base_load_dtype(None, "fp16", "cuda") is torch.float32  # full fine-tunes keep fp32 master weights
    assert neural.base_load_dtype({"r": 16}, "fp32", "cuda") is torch.float32
