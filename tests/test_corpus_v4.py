"""Corpus v4 (Increment 4d): nothing an earlier model fitted or was tested on may be evaluated again."""

import pandas as pd

from src.corpus import build as B
from src.corpus.audit import earlier_seen_groups
from src.corpus.paths import VERSION


def _manifest(root, version, rows):
    (root / version).mkdir(parents=True)
    pd.DataFrame(rows).to_parquet(root / version / "manifest.parquet", index=False)


def test_version_is_v4():
    assert VERSION == "v4"


def test_retirement_covers_fitted_and_tested_rows_but_not_report_only_ones(tmp_path):
    for v, rows in (("v1", [{"sample_id": "a", "group_key": "g:a", "split": "train"},
                            {"sample_id": "b", "group_key": "g:b", "split": "stress"}]),
                    ("v2", [{"sample_id": "c", "group_key": "g:c", "split": "val"},
                            {"sample_id": "d", "group_key": "g:d", "split": "inhouse"}]),
                    ("v3", [{"sample_id": "e", "group_key": "g:e", "split": "test"},
                            {"sample_id": "f", "group_key": "g:f", "split": "test_unseen_generator"}])):
        _manifest(tmp_path, v, rows)
    ids, groups = B.retirement(tmp_path, ("v1", "v2", "v3"))
    assert ids == {"a", "c", "e", "f"}
    assert groups == {"g:a", "g:c", "g:e", "g:f"}
    assert earlier_seen_groups(tmp_path, "v4") == groups


# --- near-duplicate retirement against earlier fitted or tested text (Task 3) -------------------------

from src.corpus.audit import audit  # noqa: E402
from tests.test_corpus import built  # noqa: E402,F401  (fixture: a built toy corpus)

LONG = ("The river bends twice before the old mill, where the path narrows and the reeds grow tall "
        "enough to hide a heron; walkers usually turn back at the stile unless the ground is dry.")


def test_cluster_hits_tell_fitted_text_from_report_only_text():
    # The build-level behaviour lives in test_corpus_v4_review.py (final review I7).
    fresh = "A completely different paragraph about sourdough starters and how often to feed them daily."
    earlier = pd.DataFrame({"sample_id": ["x", "y"], "text": [LONG, fresh + " Reported once as stress."],
                            "seen": [True, False]})
    seen, every = B.earlier_cluster_hits([LONG + " Quoted again.", fresh + " Reported once as stress.", "new"],
                                         earlier)
    assert seen == [True, False, False] and every == [True, True, False]


def test_audit_fails_when_an_evaluation_row_matches_earlier_fitted_text(built):  # noqa: F811
    df = built.df
    victim = df.index[df["split"] == "test"][0]
    report = audit(df, {"sha256": built.record["test_lock_sha256"]}, earlier_texts=[df.loc[victim, "text"]])
    assert any("earlier version fitted or tested" in e for e in report.errors)
    clean = audit(df, {"sha256": built.record["test_lock_sha256"]}, earlier_texts=["unrelated words " * 20])
    assert clean.passed


# --- held-out families and sources (Task 4) ----------------------------------------------------------

from src.corpus.split import HOLDOUT_FAMILIES, HOLDOUT_SOURCES  # noqa: E402
from src.corpus.sources._common import generator_family  # noqa: E402


def test_v4_holdouts():
    # Cohere and Moonshot are fitted now (Arena has Command A and Kimi K2); Phi, Granite and Falcon
    # answer only the held-out Stack Exchange sites, and no training source has BLOOM.
    assert HOLDOUT_FAMILIES == {"bigscience", "microsoft", "ibm", "tii"}
    assert HOLDOUT_SOURCES == {"stackexchange_heldout", "wasitai2026", "lede"}
    assert generator_family("falcon3:3b") == "tii"


# --- AI-only source probe (Task 7) --------------------------------------------------------------------

from src.corpus import confound  # noqa: E402


def _probe_rows(ai_text, human_text, human_source="dolly", human_domain="open_qa"):
    rows = [{"source_dataset": "arena140k", "domain": "chat", "label": "ai", "role": "core",
             "text": ai_text(i)} for i in range(200)]
    rows += [{"source_dataset": human_source, "domain": human_domain, "label": "human", "role": "core",
              "text": human_text(i)} for i in range(200)]
    rows += [{"source_dataset": human_source, "domain": "other", "label": "human", "role": "core",
              "text": "ignored row"} for _ in range(5)]
    return pd.DataFrame(rows)


def test_ai_only_probe_flags_a_layout_giveaway():
    df = _probe_rows(lambda i: f"**Answer {i}**\n\n- point one\n- point two\n\n## Summary\nDone.",
                     lambda i: f"i think its fine honestly, number {i} worked for me last week")
    auc = confound.ai_only_probe(df, {"arena140k": ("dolly",)})
    assert auc["arena140k"] > confound.WARN_AUC


def test_ai_only_probe_passes_when_surfaces_match_and_reads_domain_pools():
    same = lambda i: f"Water the basil when the soil feels dry, and give it about {i % 9 + 3} hours of sun."  # noqa: E731
    df = _probe_rows(same, same, human_source="raid", human_domain="news")
    auc = confound.ai_only_probe(df, {"arena140k": ("raid:news",)})
    assert auc["arena140k"] < 0.7
    assert confound.ai_only_probe(df, {"arena140k": ("raid:recipes",)}) == {}  # no humans in the pool: no probe


def test_ai_only_probe_reads_only_the_models_window():
    # Past the model's window, extra length is invisible to the model, so it must not count as a shortcut.
    body = lambda i: " ".join(f"word{(i * 7 + k) % 50}" for k in range(confound.PROBE_WORDS))  # noqa: E731
    df = _probe_rows(lambda i: body(i) + " tail" * 400, body)
    assert confound.ai_only_probe(df, {"arena140k": ("dolly",)})["arena140k"] < 0.7
