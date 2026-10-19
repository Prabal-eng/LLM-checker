"""Training orchestration: epoch checkpoints, best-epoch choice, resume, and the record."""

import json

import numpy as np
import pytest

from src.authorship import train as train_mod
from src.bakeoff.run import selection_data
from tests.test_bakeoff import _corpus


class _Saver:
    def save_pretrained(self, path):
        path.mkdir(parents=True, exist_ok=True)
        (path / "model.safetensors").write_bytes(b"weights-" + path.name.encode())


class FakeClassifier:
    """Validation improves for three checks, then worsens; records which checks it trained."""

    model_name, max_tokens = "fake", 16
    NOISE = [2.0, 1.0, 0.1, 0.5, 1.5, 2.0, 2.5, 3.0, 3.0, 3.0, 3.0, 3.0]  # per check; lower scores better

    def __init__(self, epochs=3):
        self.epochs, self.trained, self.noise = epochs, [], 2.0
        self.model = self.tokenizer = _Saver()

    def fit(self, train, *, balance_sources, checks_per_epoch, on_check, resume):
        assert balance_sources
        start = 0 if resume is None else resume["epoch"] * checks_per_epoch + resume["check"]
        for i in range(start, self.epochs * checks_per_epoch):
            epoch, check = divmod(i, checks_per_epoch)
            self.noise = self.NOISE[i]
            self.trained.append(f"{epoch}.{check + 1}")
            if on_check({"epoch": epoch, "check": check + 1, "micro": 0}):
                return

    def score(self, texts):
        rng = np.random.default_rng(len(self.trained))
        return np.array([("vocab_a" in t) + rng.normal(scale=self.noise) for t in texts])


def _data():
    df = _corpus(n_groups=12)
    df.loc[df["y"] == 1, "text"] = df.loc[df["y"] == 1, "text"] + " vocab_a"
    return selection_data(df)


def test_train_stops_two_checks_after_the_best_and_keeps_only_its_weights(tmp_path):
    pytest.importorskip("torch")  # CI installs requirements.txt, which leaves torch out
    clf = FakeClassifier()
    out = tmp_path / "m"
    result = train_mod.train(clf, out_dir=out, record=tmp_path / "rec.json", data=_data())
    assert clf.trained == ["0.1", "0.2", "0.3", "0.4", "1.1"]
    assert result["chosen_check"] == "0.3" and result["stopped_early"] is True
    assert sorted(p.name for p in out.glob("check-*")) == ["check-0.3"]
    assert result["artifact_sha256"] == train_mod.sha256_file(out / "check-0.3" / "model.safetensors")
    assert result["locked_split_reads"] == 0 and result["lora_merged"] is False
    assert not (out / "resume.pt").exists()
    saved = json.loads((tmp_path / "rec.json").read_text(encoding="utf-8"))
    assert saved["text_normalization"] == "format4+typography+collapse_whitespace"
    assert saved["candidate"] == "distilroberta-base" and saved["precision"] == "bf16"
    # No corpus text leaks into the committed record: every value is a number, hash, path, or setting.
    assert "vocab" not in json.dumps(saved)


def test_train_resumes_inside_an_epoch_after_a_crash(tmp_path):
    pytest.importorskip("torch")  # CI installs requirements.txt, which leaves torch out
    class Crash(FakeClassifier):
        def fit(self, train, **kwargs):
            inner = kwargs["on_check"]

            def on_check(state):
                inner(state)
                if state["check"] == 2:
                    raise KeyboardInterrupt  # the process dies after the second check's resume state
                return False

            super().fit(train, **{**kwargs, "on_check": on_check})

    out = tmp_path / "m"
    with pytest.raises(KeyboardInterrupt):
        train_mod.train(Crash(), out_dir=out, record=tmp_path / "rec.json", data=_data())
    second = FakeClassifier()
    result = train_mod.train(second, out_dir=out, record=tmp_path / "rec.json", data=_data())
    assert second.trained[0] == "0.3"
    assert set(result["checks"]) == {"0.1", "0.2", "0.3", "0.4", "1.1"}


def test_a_lora_run_records_the_merged_model(tmp_path, monkeypatch):
    pytest.importorskip("torch")  # CI installs requirements.txt, which leaves torch out
    from src.bakeoff import neural

    def fake_export(base, adapter, out):
        out.mkdir(parents=True, exist_ok=True)
        (out / "model.safetensors").write_bytes(b"merged-" + adapter.name.encode())
        return out

    monkeypatch.setattr(neural, "export_merged", fake_export)
    clf = FakeClassifier()
    clf.lora = {"r": 16, "alpha": 32}
    result = train_mod.train(clf, out_dir=tmp_path / "m", record=tmp_path / "rec.json", data=_data())
    assert result["artifact_path"].endswith("merged-0.3") and result["lora_merged"] is True
    assert result["artifact_sha256"] == train_mod.sha256_file(tmp_path / "m" / "merged-0.3" / "model.safetensors")


def test_best_check_prefers_detection_then_auroc_then_the_earlier_check():
    h = {"0.1": {"tpr_at_1pct_fpr": 0.5, "auroc": 0.9}, "0.2": {"tpr_at_1pct_fpr": 0.6, "auroc": 0.8},
         "0.3": {"tpr_at_1pct_fpr": 0.6, "auroc": 0.8}, "1.1": {"tpr_at_1pct_fpr": 0.4, "auroc": 0.99}}
    assert train_mod.best_check(h) == "0.2"
    assert train_mod.stale_checks(h) == 2


def test_model_ids_and_records_are_per_candidate():
    assert train_mod.model_id("modernbert-base") == "modernbert-base-authorship-4d"
    assert train_mod.model_id("modernbert-base", "short") == "modernbert-base-authorship-short-4d"
    assert train_mod.record_file(variant="short", candidate="modernbert-base").parts[-2:] == (
        "modernbert-base", "training_record_short.json")
    with pytest.raises(ValueError):
        train_mod.recipe("no-such-model")


def test_source_balance_weights_equalize_labels_within_each_source():
    import pandas as pd

    from src.bakeoff.neural import source_balance_weights

    df = pd.DataFrame({"source_dataset": ["a"] * 10 + ["b"] * 4, "y": [1] * 9 + [0] + [1, 0, 0, 0]})
    w = source_balance_weights(df)
    for source, n in (("a", 10), ("b", 4)):
        rows = df["source_dataset"] == source
        ai, human = w[rows & (df["y"] == 1)].sum(), w[rows & (df["y"] == 0)].sum()
        assert abs(ai - human) < 1e-9 and abs(ai + human - n) < 1e-9


def test_short_variant_trains_on_short_texts_and_excerpts(tmp_path):
    pytest.importorskip("torch")  # CI installs requirements.txt, which leaves torch out
    df = _data()
    df.loc[df.index[::2], "text"] = df.loc[df.index[::2], "text"].map(lambda t: " ".join([t + "."] * 12))
    seen = {}

    class ShortFake(FakeClassifier):
        def fit(self, train, **kwargs):
            seen["words"] = train["text"].str.split().str.len()
            seen["excerpts"] = int(train["excerpt"].sum())
            super().fit(train, **kwargs)

    result = train_mod.train(ShortFake(), out_dir=tmp_path / "s", record=tmp_path / "rec.json",
                             data=df, variant="short")
    assert seen["words"].between(25, 99).all()
    assert seen["excerpts"] > 0 and result["train_excerpts"] == seen["excerpts"]
    assert result["model_id"] == "distilroberta-base-authorship-short-4d" and result["variant"] == "short"


def test_excerpts_are_deterministic_sentence_runs_of_25_to_99_words():
    from src.authorship.excerpts import excerpt, short_frame
    import pandas as pd

    text = " ".join(f"Sentence {i} has exactly seven words here." for i in range(60))
    first = excerpt(text, "id-1")
    assert first == excerpt(text, "id-1")
    assert 25 <= len(first.split()) <= 99 and first.endswith(".")
    assert excerpt("too short to cut " * 5, "id-1") is None
    assert 25 <= len(excerpt("word " * 400, "id-2").split()) <= 99  # one endless sentence is cut
    frame = short_frame(pd.DataFrame({"sample_id": ["a", "b", "c"], "text": [text, "w " * 40, "w " * 10],
                                      "y": [1, 0, 0]}))
    assert list(frame["excerpt"]) == [False, True]
    assert frame.loc[frame["excerpt"], "sample_id"].item() == "a:x"
    assert frame.loc[frame["excerpt"], "parent_id"].item() == "a"


def test_a_crash_while_saving_the_resume_state_never_loses_the_best_weights(tmp_path, monkeypatch):
    pytest.importorskip("torch")  # CI installs requirements.txt, which leaves torch out
    # Check 0.3 is a new best; the process dies while writing its resume state. On resume, check 0.3 is
    # re-run and scores worse than 0.2, so 0.2 is the best again and its weights must still exist.
    real, calls = train_mod._write_resume, []

    def dies_on_third(path, state):
        calls.append(state["check"])
        if len(calls) == 3:
            raise MemoryError("while writing the resume state")
        real(path, state)

    monkeypatch.setattr(train_mod, "_write_resume", dies_on_third)
    out = tmp_path / "m"
    with pytest.raises(MemoryError):
        train_mod.train(FakeClassifier(), out_dir=out, record=tmp_path / "rec.json", data=_data())
    monkeypatch.setattr(train_mod, "_write_resume", real)

    class WorseOnRerun(FakeClassifier):
        NOISE = [2.0, 1.0, 1.8, 2.5, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0]

    second = WorseOnRerun()
    result = train_mod.train(second, out_dir=out, record=tmp_path / "rec.json", data=_data())
    assert second.trained[0] == "0.3" and result["chosen_check"] == "0.2"
    assert result["artifact_sha256"] == train_mod.sha256_file(out / "check-0.2" / "model.safetensors")


def test_a_changed_run_refuses_to_reuse_or_resume_old_checks(tmp_path):
    pytest.importorskip("torch")  # CI installs requirements.txt, which leaves torch out
    out = tmp_path / "m"
    train_mod.train(FakeClassifier(), out_dir=out, record=tmp_path / "rec.json", data=_data())
    again = FakeClassifier()
    train_mod.train(again, out_dir=out, record=tmp_path / "rec.json", data=_data())
    assert again.trained == []  # the same run, finished: nothing retrains
    data = _data()
    changed = data.drop(index=data.index[data["split"] == "train"][0])  # say an amendment dropped a row
    with pytest.raises(SystemExit, match="different run"):
        train_mod.train(FakeClassifier(), out_dir=out, record=tmp_path / "rec.json", data=changed)
    with pytest.raises(SystemExit, match="different run"):
        train_mod.train(FakeClassifier(), out_dir=out, record=tmp_path / "rec.json", data=_data(), patience=3)


def test_the_resume_state_is_saved_with_torch_and_read_onto_the_cpu(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")

    class Crash(FakeClassifier):
        def fit(self, train, **kwargs):
            inner = kwargs["on_check"]

            def on_check(state):
                inner(state)
                if state["check"] == 2:
                    raise KeyboardInterrupt
                return False

            super().fit(train, **{**kwargs, "on_check": on_check})

    out = tmp_path / "m"
    with pytest.raises(KeyboardInterrupt):
        train_mod.train(Crash(), out_dir=out, record=tmp_path / "rec.json", data=_data())
    assert torch.load(out / "resume.pt", map_location="cpu", weights_only=False)["check"] == 2
    seen, real_load = {}, torch.load
    monkeypatch.setattr(torch, "load", lambda *a, **k: seen.update(k) or real_load(*a, **k))
    train_mod.train(FakeClassifier(), out_dir=out, record=tmp_path / "rec.json", data=_data())
    assert seen.get("map_location") == "cpu"


def test_fast_settings_keep_the_effective_batch_and_are_recorded(tmp_path):
    pytest.importorskip("torch")  # CI installs requirements.txt, which leaves torch out
    screening = train_mod.settings_for("deberta-v3-large-lora", "screening")
    fast = train_mod.settings_for("deberta-v3-large-lora", "fast")
    assert screening == train_mod.recipe("deberta-v3-large-lora")
    assert fast["gradient_checkpointing"] is False
    assert fast["batch_size"] * fast["accumulate"] == 16  # the same effective batch as screening's 8 x 2
    assert {k: v for k, v in fast.items() if k not in ("batch_size", "accumulate", "gradient_checkpointing")} == \
        {k: v for k, v in screening.items() if k not in ("batch_size", "accumulate", "gradient_checkpointing")}
    assert train_mod.settings_for("modernbert-base", "fast") == train_mod.recipe("modernbert-base")
    with pytest.raises(ValueError):
        train_mod.settings_for("deberta-v3-large-lora", "turbo")
    result = train_mod.train(FakeClassifier(), candidate="deberta-v3-large-lora", settings="fast",
                             out_dir=tmp_path / "m", record=tmp_path / "rec.json", data=_data())
    assert result["settings"] == "fast" and result["recipe"]["gradient_checkpointing"] is False


def test_fullbatch_settings_take_the_effective_batch_in_one_step(tmp_path):
    """Amendment 12: one micro-batch of 16 with checkpointing, 11% faster than 8 x 2 at a 2.1 GB peak."""
    pytest.importorskip("torch")  # CI installs requirements.txt, which leaves torch out
    screening = train_mod.settings_for("deberta-v3-large-lora", "screening")
    full = train_mod.settings_for("deberta-v3-large-lora", "fullbatch")
    assert (full["batch_size"], full["accumulate"], full["gradient_checkpointing"]) == (16, 1, True)
    assert full["batch_size"] * full["accumulate"] == 16  # screening's 8 x 2 (the classifier's defaults)
    assert {k: v for k, v in full.items() if k not in ("batch_size", "accumulate")} == screening
    assert train_mod.settings_for("modernbert-base", "fullbatch") == train_mod.recipe("modernbert-base")
    result = train_mod.train(FakeClassifier(), candidate="deberta-v3-large-lora", settings="fullbatch",
                             out_dir=tmp_path / "m", record=tmp_path / "rec.json", data=_data())
    assert result["settings"] == "fullbatch" and result["recipe"]["batch_size"] == 16


def test_the_record_binds_validation_ids_and_the_defaults_are_the_amended_ones(tmp_path):
    """Ralph R12-18: the committed record carries a validation-ids digest; training defaults to the
    Amendment 12 settings and calibration to the pre-registered 0.8% / 8% targets."""
    pytest.importorskip("torch")  # CI installs requirements.txt, which leaves torch out
    import inspect

    from src.authorship import calibrate
    from src.authorship.train import ids_digest

    data = _data()
    result = train_mod.train(FakeClassifier(), out_dir=tmp_path / "m", record=tmp_path / "rec.json", data=data)
    val_ids = data[data["split"] == "val"]["sample_id"]
    assert result["validation_ids_sha256"] == ids_digest(val_ids)
    assert inspect.signature(train_mod.train).parameters["settings"].default == "fullbatch"
    assert inspect.signature(calibrate.calibrate).parameters["max_false_ai"].default == 0.008
    assert inspect.signature(calibrate.calibrate).parameters["max_false_human"].default == 0.08


def test_the_scoring_code_digest_covers_the_detection_package():
    from src.authorship import registry

    assert "src/detection/__init__.py" in registry._SCORING_CODE
