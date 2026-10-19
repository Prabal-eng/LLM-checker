"""Increment 4d optional steps, measured on validation: the perplexity stack and style-aware thresholds."""

import numpy as np
import pandas as pd
import pytest

from src.authorship import registers, stack

OK_LIMITS = {"peak_vram_gb": 3.0, "p95_ms": 300.0, "sheet_seconds": 90.0}


def test_halves_keep_excerpts_with_their_parent():
    ids = pd.Series([f"s{i}" for i in range(200)] + [f"s{i}:x" for i in range(200)])
    half = stack.half_of(ids)
    assert (half[:200] == half[200:]).all()
    assert 60 < half[:200].sum() < 140


def _frame(signal: float, n: int = 4000) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    y = np.arange(n) % 2
    return pd.DataFrame({"sample_id": [f"s{i}" for i in range(n)], "y": y,
                         "raw": y * 1.0 + rng.normal(size=n), "binoculars": y * signal + rng.normal(size=n),
                         "fast_base": rng.normal(size=n), "fast_instruct": rng.normal(size=n)})


def test_the_stack_is_adopted_when_a_zero_shot_score_adds_detection():
    result = stack.stack_gain(_frame(signal=2.0))
    assert result["gain_points"] >= 2.0
    assert stack.adopted(result["gain_points"], OK_LIMITS)


def test_the_stack_is_dropped_when_zero_shot_scores_are_noise():
    result = stack.stack_gain(_frame(signal=0.0))
    assert result["gain_points"] < 2.0
    assert not stack.adopted(result["gain_points"], OK_LIMITS)


def test_adoption_needs_every_gpu_limit():
    assert not stack.adopted(5.0, {**OK_LIMITS, "peak_vram_gb": 3.6})
    assert not stack.adopted(5.0, {**OK_LIMITS, "p95_ms": 600.0})
    assert not stack.adopted(5.0, {**OK_LIMITS, "sheet_seconds": 130.0})


# Every core (source, domain) pair that trains in corpus v4: v3's pairs plus the new sources.
V4_TRAINING_PAIRS = (
    [("aita2026", "aita"), ("oasst1", "assistant_reply"), ("arena140k", "chat")]
    + [("beemo", d) for d in ("closed_qa", "generation", "open_qa", "rewrite", "summarize")]
    + [("dolly", d) for d in ("brainstorming", "classification", "closed_qa", "creative_writing", "general_qa",
                              "information_extraction", "open_qa", "summarization")]
    + [("hc3", d) for d in ("finance", "medicine", "open_qa", "reddit_eli5", "wiki_csai")]
    + [("mage", d) for d in ("cmv", "cnn", "dialogsum", "eli5", "hswag", "imdb", "pubmed", "roct", "sci_gen",
                             "squad", "tldr", "wp", "xsum", "yelp")]
    + [("mirage", d) for d in ("academic", "comment", "news", "website")]
    + [("no_robots", d) for d in ("brainstorm", "chat", "classify", "closed_qa", "coding", "extract",
                                  "generation", "open_qa", "rewrite", "summarize")]
    + [("raid", d) for d in ("abstracts", "books", "news", "poetry", "recipes", "reddit", "reviews", "wiki")]
    + [("stackexchange", d) for d in ("cooking", "fitness", "gardening", "interpersonal", "money", "outdoors",
                                      "parenting", "pets")]
    + [("maga", d) for d in ("amazon_reviews", "cc_news", "natural_questions", "npr_news", "reddit", "s2orc",
                             "trustpilot_reviews", "wikipedia", "yahoo_answers")]
    + [("dactyl", d) for d in ("abstracts", "news", "reviews", "student_essays", "tweets")]
)


def test_every_training_pair_has_a_register():
    assert {registers.register_label(s, d) for s, d in V4_TRAINING_PAIRS} == set(registers.REGISTERS)


def test_an_unmapped_pair_fails_loudly():
    with pytest.raises(KeyError):
        registers.register_label("mage", "no_such_domain")
    with pytest.raises(KeyError):
        registers.register_label("no_such_source", "x")


def test_the_register_classifier_learns_registers_from_text():
    vocab = {"qa_forum": "answer question thread upvote", "assistant": "certainly here is a helpful list",
             "news_expository": "officials said on tuesday report", "essay_academic": "we propose results method",
             "social_review": "loved it five stars lol"}
    rng = np.random.default_rng(0)
    texts, labels = [], []
    for register, words in vocab.items():
        for _ in range(60):
            texts.append(" ".join(rng.permutation(words.split() + ["common", "words", "here"])))
            labels.append(register)
    clf = registers.RegisterClassifier().fit(texts, labels)
    assert (clf.predict(texts) == np.array(labels)).mean() > 0.95


def _cells(n_humans_a: int, n_humans_b: int) -> pd.DataFrame:
    """Band 100, two registers. Register a's humans score low, so its own bar can sit below the pooled bar."""
    rng = np.random.default_rng(1)
    parts = []
    for register, humans, human_scores in (("a", n_humans_a, (0.0, 0.3)), ("b", n_humans_b, (0.0, 0.95))):
        parts.append(pd.DataFrame({"register": register, "y": 0, "words": 150, "source_dataset": f"src_{register}",
                                   "prob": rng.uniform(*human_scores, size=humans)}))
        parts.append(pd.DataFrame({"register": register, "y": 1, "words": 150, "source_dataset": f"src_{register}",
                                   "prob": rng.uniform(0.4, 1.0, size=humans)}))
    return pd.concat(parts, ignore_index=True)


def test_a_cell_with_600_humans_gets_its_own_bar_and_a_smaller_cell_keeps_the_pooled_bar():
    val = _cells(700, 500)
    bars = registers.cell_bars(val, {100: 0.97}, max_fpr=0.008, max_fnr=0.08)
    assert bars[(100, "a")]["own"] and bars[(100, "a")]["ai_min"] < 0.97
    assert not bars[(100, "b")]["own"] and bars[(100, "b")]["ai_min"] == 0.97


def test_style_thresholds_are_adopted_only_with_gain_and_every_cell_in_target():
    good = registers.register_gain(_cells(700, 500), {100: 0.97}, max_fpr=0.008, max_fnr=0.08)
    assert good["gain_points"] >= 2.0 and good["every_cell_in_target"] and good["adopted"]
    # A pooled bar that lets register b's humans through breaks that cell's false-AI target.
    bad = registers.register_gain(_cells(700, 500), {100: 0.5}, max_fpr=0.008, max_fnr=0.08)
    assert not bad["every_cell_in_target"] and not bad["adopted"]


def test_style_thresholds_also_hold_every_source_slice():
    # src_c's humans are split across registers a and b (290 each), so no cell is held to src_c, but
    # together they are a gated slice. Register a's low bar lets 10 of them through. Register a is large
    # (src_d's humans score 0), so every cell stays within target while src_c breaks its false-AI bound,
    # and the step can't be adopted.
    rng = np.random.default_rng(2)

    def part(register, source, y, scores):
        return pd.DataFrame({"register": register, "y": y, "words": 150, "source_dataset": source, "prob": scores})

    val = pd.concat([
        part("a", "src_a", 0, rng.uniform(0.0, 0.2, 3000)), part("a", "src_a", 1, rng.uniform(0.4, 1.0, 3000)),
        part("a", "src_c", 0, np.r_[np.full(10, 0.5), np.full(280, 0.1)]), part("a", "src_d", 0, np.zeros(20000)),
        part("b", "src_b", 0, rng.uniform(0.0, 0.2, 250)), part("b", "src_b", 1, rng.uniform(0.4, 1.0, 250)),
        part("b", "src_c", 0, np.full(290, 0.1)),
    ], ignore_index=True)
    result = registers.register_gain(val, {100: 0.97}, max_fpr=0.008, max_fnr=0.08)
    assert result["gain_points"] >= 2.0 and result["every_cell_in_target"]
    assert not result["every_source_in_target"] and not result["adopted"]
    assert [s["source"] for s in result["sources"] if not s["within_target"]] == ["src_c"]


def test_sources_the_build_may_admit_have_a_register_too():
    # The loader dry run found openturingbench's news rows unmapped; the AI-only probe is expected to exclude
    # the source, but an amendment admitting it must not break the register step.
    assert registers.register_label("openturingbench", "news") == "news_expository"


def test_the_zero_shot_pair_loads_only_after_every_classifier_is_released(tmp_path, monkeypatch):
    """On a 4 GB card the classifiers plus the zero-shot pair spill into shared memory and score many
    times slower, so the stack scores with each classifier alone, frees it, then loads the pair. The
    GPU limits still load every model at once."""
    import gc
    import json
    import weakref

    from src.authorship import calibrate, train
    from src.bakeoff import neural

    rng = np.random.default_rng(0)
    n = 1200
    words = np.where(np.arange(n) % 2 == 0, 150, 60)
    val = pd.DataFrame({"sample_id": [f"s{i}" for i in range(n)], "y": np.arange(n) // 2 % 2,
                        "text": [f"text {i}" for i in range(n)], "words": words})
    monkeypatch.setattr(calibrate, "load_split", lambda split: val)
    monkeypatch.setattr(calibrate, "with_excerpts", lambda df, view=None: df.iloc[0:0])
    monkeypatch.setattr(calibrate, "short_rows", lambda df: df[df["words"] < 100])
    for variant in ("main", "short"):
        (tmp_path / f"{variant}.json").write_text(json.dumps({"model_id": f"m-{variant}"}), encoding="utf-8")
    monkeypatch.setattr(train, "record_file", lambda variant, candidate: tmp_path / f"{variant}.json")

    detectors, scorers = [], []

    def alive(refs):
        gc.collect()
        return [r for r in refs if r() is not None]

    class FakeDetector:
        def raw_and_embeddings(self, texts):
            assert not alive(scorers), "a classifier scored while the zero-shot pair was loaded"
            return rng.normal(size=len(texts)), None

    def make_detector(training):
        d = FakeDetector()
        detectors.append(weakref.ref(d))
        return d

    class FakeScorer:
        def __init__(self):
            scorers.append(weakref.ref(self))

        def score_frame(self, texts):
            assert not alive(detectors), "the zero-shot pair scored while a classifier was loaded"
            return pd.DataFrame({k: rng.normal(size=len(texts)) for k in ("binoculars", "fast_base", "fast_instruct")})

    monkeypatch.setattr(calibrate, "_detector", make_detector)
    monkeypatch.setattr(neural, "ZeroShotScorer", FakeScorer)
    loaded_for_limits = []
    monkeypatch.setattr(stack, "gpu_limits", lambda detectors, scorer, texts: loaded_for_limits.append(len(detectors))
                        or OK_LIMITS)
    monkeypatch.setattr(stack, "report_file", lambda candidate: tmp_path / "stack.json")
    report = stack.run("x")
    assert set(report["models"]) == {"main", "short"}
    assert loaded_for_limits == [2]  # the limits still see every model loaded
