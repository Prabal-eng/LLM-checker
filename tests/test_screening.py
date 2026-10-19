"""Increment 4d model screening: cheaper training options and the validation-only screening run."""

import json

import numpy as np
import pandas as pd
import pytest

from src.bakeoff import screen
from src.bakeoff.neural import autocast_dtype, length_grouped_order
from tests.test_authorship_train import _data


def test_length_grouped_order_is_a_deterministic_permutation():
    lengths = np.random.default_rng(1).integers(5, 400, size=1000)
    order = length_grouped_order(lengths, batch_size=8, seed=3)
    assert sorted(order.tolist()) == list(range(1000))
    assert order.tolist() == length_grouped_order(lengths, batch_size=8, seed=3).tolist()
    assert order.tolist() != length_grouped_order(lengths, batch_size=8, seed=4).tolist()


def test_length_grouped_order_pads_less_than_a_random_order():
    lengths = np.random.default_rng(2).integers(5, 400, size=4000)

    def padded(order):  # tokens a batch spends on padding up to its longest text
        return sum(int(lengths[b].max()) * len(b) - int(lengths[b].sum()) for b in np.split(order, len(order) // 8))

    random_order = np.random.default_rng(0).permutation(4000)
    assert padded(length_grouped_order(lengths, batch_size=8, seed=0)) < 0.3 * padded(random_order)


def test_autocast_dtype_follows_the_precision_setting():
    torch = pytest.importorskip("torch")
    assert autocast_dtype("bf16") is torch.bfloat16
    assert autocast_dtype("fp16") is torch.float16
    assert autocast_dtype("fp32") is None
    with pytest.raises(ValueError):
        autocast_dtype("int8")


def test_lora_trains_a_small_share_of_weights_and_the_classifier_head():
    pytest.importorskip("peft")
    from transformers import RobertaConfig, RobertaForSequenceClassification

    from src.bakeoff.neural import apply_lora

    config = RobertaConfig(vocab_size=100, hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                           intermediate_size=64, max_position_embeddings=64, num_labels=2)
    model = apply_lora(RobertaForSequenceClassification(config), {"r": 4, "alpha": 8})
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    total = sum(p.numel() for p in model.parameters())
    assert sum(p.numel() for n, p in model.named_parameters() if n in trainable) < 0.5 * total
    assert any("lora_" in n for n in trainable)
    assert any("classifier" in n for n in trainable)


def test_large_candidates_score_in_small_batches():
    pytest.importorskip("torch")  # CI installs requirements.txt, which leaves torch out
    # A 64-text batch of a large encoder overflows 4 GB and Windows spills it to system RAM, which
    # made DeBERTa-v3-large score at 182 ms a text in the smoke test.
    for spec in screen.CANDIDATES:
        if "large" in spec["name"]:
            assert spec["kwargs"]["score_batch_size"] <= 16
    clf = screen.build_classifier(next(c for c in screen.CANDIDATES if "large" in c["name"]), "main")
    assert clf.score_batch_size <= 16


def test_band_metrics_set_each_bands_threshold_on_its_own_humans():
    rng = np.random.default_rng(0)
    rows = []
    for band_words, ai_shift in ((30, 0.5), (150, 3.0)):
        for y in (0, 1):
            for s in rng.normal(loc=ai_shift * y, size=2000):
                rows.append({"text": " ".join(["w"] * band_words), "y": y, "score": s})
    df = pd.DataFrame(rows)
    out = screen.band_metrics(df, df.pop("score").to_numpy())
    assert set(out) == {"25-49", "100-199"}
    assert out["100-199"]["tpr_at_1pct_fpr"] > out["25-49"]["tpr_at_1pct_fpr"]
    assert out["25-49"]["humans"] == 2000 and out["25-49"]["ais"] == 2000


class _FakeClassifier:
    """Scores AI rows higher; records how many rows it trained on."""

    def __init__(self, **kwargs):
        self.kwargs, self.fitted_rows, self.peak_vram_gb = kwargs, None, 0.0

    def fit(self, train, *, balance_sources):
        assert balance_sources
        self.fitted_rows = len(train)

    def score(self, texts):
        return np.array([("vocab_a" in t) + 0.01 * i for i, t in enumerate(texts)], dtype=float)


def test_screen_writes_a_text_free_result_per_candidate_and_skips_finished_ones(tmp_path, monkeypatch):
    built = []

    def fake_build(spec, track):
        clf = _FakeClassifier(**spec["kwargs"])
        built.append((spec["name"], track))
        return clf

    monkeypatch.setattr(screen, "build_classifier", fake_build)
    specs = [{"name": "a", "kwargs": {}}, {"name": "b", "kwargs": {}}]
    data = _data()
    screen.run(specs, tracks=("main",), train_rows=40, out_dir=tmp_path, data=data)
    assert built == [("a", "main"), ("b", "main")]
    saved = json.loads((tmp_path / "a-main.json").read_text(encoding="utf-8"))
    assert saved["train_rows"] <= 40 and saved["locked_split_reads"] == 0
    assert saved["validation"]["tpr_at_1pct_fpr"] == 1.0
    assert "vocab" not in json.dumps(saved)  # no corpus text in the record

    built.clear()
    screen.run(specs, tracks=("main",), train_rows=40, out_dir=tmp_path, data=data)
    assert built == []  # both finished, nothing re-trained
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert [row["name"] for row in summary["main"]] == ["a", "b"]


def test_a_failing_candidate_is_recorded_and_the_rest_still_run(tmp_path, monkeypatch):
    def fake_build(spec, track):
        if spec["name"] == "broken":
            raise RuntimeError("CUDA out of memory")
        return _FakeClassifier()

    monkeypatch.setattr(screen, "build_classifier", fake_build)
    specs = [{"name": "broken", "kwargs": {}}, {"name": "ok", "kwargs": {}}]
    screen.run(specs, tracks=("main",), train_rows=40, out_dir=tmp_path, data=_data())
    assert "out of memory" in json.loads((tmp_path / "broken-main.error.json").read_text(encoding="utf-8"))["error"]
    assert not (tmp_path / "broken-main.json").exists()  # a rerun tries it again
    assert (tmp_path / "ok-main.json").exists()
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert [row["name"] for row in summary["main"]] == ["broken", "ok"] and "error" in summary["main"][0]
