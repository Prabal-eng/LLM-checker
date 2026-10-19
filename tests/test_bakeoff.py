"""Bake-off harness: thresholds, folds, locked-split isolation, end-to-end selection. No GPU needed."""

import json
import random

import numpy as np
import pandas as pd
import pytest

from src.bakeoff import data as bakeoff_data
from src.bakeoff import run as bakeoff_run
from src.bakeoff.data import folds, load_fit_data, prepare
from src.bakeoff.metrics import at_threshold, fold_metrics, threshold_at_fpr, upper_bound


def test_threshold_keeps_calibration_fpr_at_or_below_target():
    rng = np.random.default_rng(0)
    humans = rng.normal(size=10_000)
    t = threshold_at_fpr(humans, 0.01)
    assert (humans > t).mean() <= 0.01


def test_upper_bound_matches_rule_of_three():
    assert upper_bound(0, 300) == pytest.approx(0.00994, abs=1e-4)
    assert upper_bound(0, 0) is None
    assert upper_bound(5, 5) == 1.0


def test_at_threshold_counts_errors():
    y = np.array([0, 0, 0, 1, 1])
    s = np.array([0.1, 0.9, 0.2, 0.8, 0.1])
    r = at_threshold(y, s, 0.5)
    assert (r["fpr"], r["tpr"], r["n_human"], r["n_ai"]) == (pytest.approx(1 / 3, abs=1e-4), 0.5, 3, 2)


def _corpus(n_groups=80, seed=0):
    rng = random.Random(seed)
    vocab_h = [f"h{i}" for i in range(300)]
    vocab_a = [f"a{i}" for i in range(300)]
    rows = []
    for split in ("train", "val"):
        for source in bakeoff_data.LOSO_SOURCES:
            for g in range(n_groups):
                for y, vocab, family in ((0, vocab_h, None), (1, vocab_a, ["openai", "anthropic", "google", "deepseek"][g % 4])):
                    text = " ".join(rng.choice(vocab) for _ in range(30))
                    generator = None if family is None else ("GPT-4o" if g % 8 == 0 else f"{family}-model")
                    rows.append({"sample_id": f"{split}{source}{g}{y}", "text": text, "label": "ai" if y else "human",
                                 "split": split, "source_dataset": source, "domain": "d", "generator_family": family,
                                 "generator": generator})
    return prepare(pd.DataFrame(rows))


def test_folds_hold_out_the_named_source_and_family():
    df = _corpus()
    by_name = {f.name: f for f in folds(df)}
    assert set(by_name) == {"val", *(f"loso:{s}" for s in bakeoff_data.LOSO_SOURCES),
                            *(f"lofo:{f}" for f in bakeoff_data.LOFO_FAMILIES), "recent:openai"}
    loso = by_name["loso:mage"]
    assert "mage" not in set(loso.fit["source_dataset"]) | set(loso.calib["source_dataset"])
    assert set(loso.evaluation["source_dataset"]) == {"mage"}
    lofo = by_name["lofo:anthropic"]
    assert "anthropic" not in set(lofo.fit["generator_family"].dropna()) | set(lofo.calib["generator_family"].dropna())
    assert set(lofo.evaluation.loc[lofo.evaluation["y"] == 1, "generator_family"]) == {"anthropic"}
    assert all(set(f.fit["split"]) == {"train"} for f in by_name.values())
    recent = by_name["recent:openai"]
    assert "GPT-4o" not in set(recent.fit["generator"].dropna()) | set(recent.calib["generator"].dropna())
    assert set(recent.evaluation.loc[recent.evaluation["y"] == 1, "generator"]) == {"GPT-4o"}
    assert "openai-model" in set(recent.fit["generator"].dropna())


def test_eval_only_rows_reach_only_the_recent_openai_fold():
    df = _corpus()
    extra = df[(df["split"] == "val") & (df["generator"] == "GPT-4o")].copy()
    extra["sample_id"] += "x"
    extra["eval_only"] = True
    with_extra = {f.name: f for f in folds(pd.concat([df, extra], ignore_index=True))}
    without = {f.name: f for f in folds(df)}
    for name, fold in without.items():
        if name == "recent:openai":
            continue
        for part in ("fit", "calib", "evaluation"):
            assert set(getattr(with_extra[name], part)["sample_id"]) == set(getattr(fold, part)["sample_id"])
    assert set(extra["sample_id"]) <= set(with_extra["recent:openai"].evaluation["sample_id"])


def test_loader_never_returns_locked_splits(tmp_path):
    df = _corpus(n_groups=5).drop(columns=["y", "length_bucket"])
    locked = df.head(10).assign(split="test", sample_id=lambda d: d["sample_id"] + "t")
    (tmp_path / "v1").mkdir()
    pd.concat([df, locked]).to_parquet(tmp_path / "v1" / "corpus.parquet", index=False)
    loaded = load_fit_data(root=tmp_path, version="v1")
    assert set(loaded["split"]) == {"train", "val"}
    assert len(loaded) == len(df)


def test_fold_metrics_reports_slices():
    df = _corpus(n_groups=20)
    val = df[df["split"] == "val"].reset_index(drop=True)
    s = val["y"].to_numpy() + np.random.default_rng(0).normal(scale=0.1, size=len(val))
    m = fold_metrics(val, s, val, s)
    assert m["auroc"] == 1.0
    assert set(m["auroc_by_source_dataset"]) == set(bakeoff_data.LOSO_SOURCES)
    assert m["at_fpr_0.01"]["fpr"] <= 0.01


def test_select_end_to_end_with_tfidf(tmp_path, monkeypatch):
    monkeypatch.setattr(bakeoff_run, "load_fit_data", lambda: _corpus())
    out = tmp_path / "selection.json"
    record = bakeoff_run.select(["tfidf"], out, tmp_path / "scores")
    summary = record["candidates"]["tfidf"]["summary"]
    assert record["locked_split_reads"] == 0
    assert summary["val_auroc"] > 0.95 and summary["loso_worst_auroc"] > 0.9
    assert json.loads(out.read_text())["candidates"]["tfidf"]["folds"]["loso:raid"]["n"] > 0
    assert (tmp_path / "scores" / "tfidf.parquet").exists()


def test_select_resumes_from_fold_checkpoints(tmp_path, monkeypatch):
    from src.bakeoff.candidates import TfidfBaseline

    monkeypatch.setattr(bakeoff_run, "load_fit_data", lambda: _corpus())
    out = tmp_path / "selection.json"
    first = bakeoff_run.select(["tfidf"], out, tmp_path / "scores")

    fits = []
    original = TfidfBaseline.fit
    monkeypatch.setattr(TfidfBaseline, "fit", lambda self, train: (fits.append(1), original(self, train)))
    second = bakeoff_run.select(["tfidf"], out, tmp_path / "scores")

    assert fits == []
    assert second["candidates"]["tfidf"]["folds"] == first["candidates"]["tfidf"]["folds"]


def test_transient_cuda_fault_is_retried_and_other_errors_are_not(monkeypatch):
    monkeypatch.setattr(bakeoff_run.time, "sleep", lambda s: None)
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) < 2:
            raise RuntimeError("CUDA driver error: device not ready")
        return "ok"

    assert bakeoff_run._with_retries(flaky, "test") == "ok" and len(calls) == 2
    with pytest.raises(RuntimeError, match="shape"):
        bakeoff_run._with_retries(lambda: (_ for _ in ()).throw(RuntimeError("shape mismatch")), "test")


def test_fixed_scorer_resume_scores_only_rows_missing_from_checkpoints(tmp_path, monkeypatch):
    class Counting:
        name, trainable = "fixed", False

        def __init__(self):
            self.seen = []

        def score(self, texts):
            self.seen.extend(texts)
            return np.array([len(t) % 7 for t in texts], dtype=float)

    df = _corpus()
    monkeypatch.setattr(bakeoff_run, "load_fit_data", lambda: df)
    first = Counting()
    monkeypatch.setattr(bakeoff_run, "build_candidates", lambda names: [first])
    bakeoff_run.select(["fixed"], tmp_path / "selection.json", tmp_path / "scores")
    # Drop one fold's checkpoint; the rerun must recompute it from the other folds' saved scores.
    for path in (tmp_path / "results" / "partial" / "fixed").glob("recent_openai.*"):
        path.unlink()
    again = Counting()
    monkeypatch.setattr(bakeoff_run, "build_candidates", lambda names: [again])
    record = bakeoff_run.select(["fixed"], tmp_path / "selection.json", tmp_path / "scores")
    assert again.seen == []
    assert record["candidates"]["fixed"]["folds"]["recent:openai"]["auroc"] is not None
