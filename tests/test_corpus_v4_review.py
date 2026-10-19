"""Corpus v4 final-review fixes (Plan A): fresh rows from existing sources, retirement of held-out sources,
the leakage rule over every earlier split, build and audit agreeing, format3 idempotence, the v3 generator
pool, and excerpts cut from each model's own view."""

import json
import random
from types import SimpleNamespace

import pandas as pd
import pytest

from src.corpus import build as B
from src.corpus.schema import Label, LicenceClass, Role
from src.corpus.sources import hc3, raid
from src.corpus.sources._common import SourceSpec, make_sample, take_per_stratum
from src.corpus.split import assign_splits
from src.corpus.text import sample_id, text_view
from tests.test_corpus import VOCAB, fake_source

# --- C1: existing sources yield rows no earlier version took ----------------------------------------


def test_skip_used_takes_the_rows_earlier_versions_left_and_falls_back_when_short():
    df = pd.DataFrame({"k": [f"r{i}" for i in range(10)], "s": ["a"] * 10})
    used = take_per_stratum(df, ["s"], 3, "k")
    fresh = take_per_stratum(df, ["s"], 3, "k", skip_used=True)
    assert len(fresh) == 3 and set(fresh["k"]).isdisjoint(used["k"])
    small = df.head(4)
    fresh_small = take_per_stratum(small, ["s"], 3, "k", skip_used=True)
    assert len(fresh_small) == 3
    assert len(set(fresh_small["k"]) - set(take_per_stratum(small, ["s"], 3, "k")["k"])) == 1


def test_hc3_reads_answers_no_earlier_version_took(tmp_path):
    rng = random.Random(3)
    for d in hc3._DOMAINS:
        answers = [" ".join(rng.choice(VOCAB) for _ in range(30)) for _ in range(12)]
        (tmp_path / f"{d}.jsonl").write_text(json.dumps(
            {"question": d, "human_answers": answers[:6], "chatgpt_answers": answers[6:]}) + "\n", encoding="utf-8")
    earlier = set(take_per_stratum(hc3._rows(tmp_path), ["domain", "label"], 3, "record")["record"])
    got = {s.source_record_id for s in hc3.load(tmp_path, 3)}
    assert len(got) == 30 and got.isdisjoint(earlier)


def test_raid_streams_enough_rows_to_skip_the_used_ones(tmp_path):
    rng = random.Random(4)
    text = lambda: " ".join(rng.choice(VOCAB) for _ in range(30))  # noqa: E731
    rows = [{"id": f"h{i}", "source_id": None, "model": "human", "attack": None, "domain": "news",
             "generation": text()} for i in range(8)]
    pd.DataFrame(rows).to_csv(tmp_path / raid._FILE, index=False)
    human = pd.DataFrame(rows).assign(source_id=lambda d: d["id"])
    earlier = set(take_per_stratum(human, ["domain"], 2, "id")["id"])
    got = {s.source_record_id for s in raid.load(tmp_path, 2) if s.label is Label.HUMAN}
    assert len(got) == 2 and got.isdisjoint(earlier)


@pytest.mark.parametrize("module", ["mage", "mirage", "no_robots", "aita2026"])
def test_other_existing_sources_opt_in_to_fresh_rows(module):
    import inspect

    from src.corpus import sources

    assert "skip_used=True" in inspect.getsource(getattr(sources, module))


@pytest.mark.parametrize("module", ["dolly", "oasst1", "stackexchange"])
def test_sources_whose_answers_were_generated_keep_their_items(module):
    # Their AI rows were generated for the items earlier versions chose; new items would have no answers.
    import inspect

    from src.corpus import sources

    assert "skip_used" not in inspect.getsource(getattr(sources, module))


# --- C2: a held-out source row that an earlier version saw is never tested ---------------------------


def _rows(ids, source="lede"):
    return pd.DataFrame({"sample_id": ids, "group_key": [f"{source}:{i}" for i in ids],
                         "cluster_id": range(len(ids)), "source_dataset": source, "role": "core",
                         "generator_family": "openai"})


def test_a_retired_held_out_row_is_excluded_not_tested():
    split = assign_splits(_rows(["h1", "h2"]), holdout_sources=frozenset({"lede"}), retired=frozenset({"h1"}))
    assert pd.isna(split.iloc[0]) and split.iloc[1] == "test"


# --- I5, I6, I7: the build retires against every earlier split and agrees with its audit ---------------

rng = random.Random(11)


def _text(n=80):
    return " ".join(rng.choice(VOCAB) for _ in range(n))


T1, T2, T3, T4 = _text(), _text(), _text(), _text()   # earlier: fitted, stress, stress, stress
NEAR_T1 = T1 + " and then the walkers turned back at the stile"
NEAR_T4 = T4 + " which the reviewers reported last spring"
FRESH = [_text() for _ in range(6)]


def _custom(name, rows):
    spec = SourceSpec(name=name, licence="MIT", licence_class=LicenceClass.PERMISSIVE)

    def load(raw, cap):
        for i, (text, role) in enumerate(rows):
            yield make_sample(spec, text=text, label=Label.HUMAN, role=role, record_id=str(i), domain="d",
                              group_key=f"{name}{i}", **({"transform": "none"} if role is Role.STRESS else {}))

    return SimpleNamespace(SPEC=spec, load=load)


def _earlier_version(root):
    rows = [(T1, "train"), (T2, "stress"), (T3, "stress"), (T4, "stress")]
    df = pd.DataFrame({"text": [t for t, _ in rows], "split": [s for _, s in rows],
                       "group_key": [f"old:{i}" for i in range(len(rows))]})
    df["sample_id"] = df["text"].map(sample_id)
    (root / "v1").mkdir(parents=True)
    df.to_parquet(root / "v1" / "corpus.parquet", index=False)
    df.drop(columns="text").to_parquet(root / "v1" / "manifest.parquet", index=False)


def _build(tmp_path, **overrides):
    root = tmp_path / "corpus"
    _earlier_version(root)
    sources = {"a": fake_source("a", 1), "b": fake_source("b", 2),
               "wasitai2026": _custom("wasitai2026", [(T2, Role.CORE), (NEAR_T1, Role.CORE), (NEAR_T4, Role.CORE)]
                                      + [(t, Role.CORE) for t in FRESH]),
               "detector_bias": _custom("detector_bias", [(T2, Role.STRESS), (T3, Role.STRESS)])}
    kwargs = dict(sources=sources, cap=10_000, root=root, lock_file=tmp_path / "repo" / "test.lock.json",
                  record_file=tmp_path / "repo" / "build_record.json", version="v2", previous_versions=("v1",))
    record = B.build(**{**kwargs, **overrides})
    return record, pd.read_parquet(root / "v2" / "corpus.parquet")


def _split_of(df, text):
    return df.loc[df["sample_id"] == sample_id(text), "split"].tolist()


def test_the_build_never_tests_text_any_earlier_version_held(tmp_path):
    record, df = _build(tmp_path)
    assert _split_of(df, NEAR_T1) == []          # held-out near-duplicate of fitted text (C2)
    assert _split_of(df, NEAR_T4) == []          # near-duplicate of earlier stress text (I5)
    assert _split_of(df, T2) == ["stress"]       # the stress copy survives; the core copy is dropped (I5)
    assert df.loc[df["sample_id"] == sample_id(T2), "source_dataset"].tolist() == ["detector_bias"]
    assert _split_of(df, T3) == ["stress"]       # a stress text reported before stays a stress text
    assert all(_split_of(df, t) == ["test"] for t in FRESH)
    assert record["audit"]["passed"]
    assert record["core_copies_of_reported_text_dropped"] == 1
    assert record["near_duplicate_retired_by_source"]["wasitai2026"] == 2


def test_the_build_retires_what_its_audit_would_flag(tmp_path, monkeypatch):
    # Clustering the open rows can miss a link the audit's own clustering finds (boilerplate cut-offs
    # depend on which texts are clustered); the build must settle on the audit's check, not abort.
    monkeypatch.setattr(B, "earlier_cluster_hits", lambda texts, earlier: ([False] * len(texts),) * 2)
    record, df = _build(tmp_path)
    assert _split_of(df, NEAR_T1) == [] and _split_of(df, NEAR_T4) == []
    assert record["audit"]["passed"] and record["retired_by_audit_check"] == 2


def test_earlier_texts_cover_every_split_once_per_text(tmp_path):
    for version in ("v1", "v2"):
        (tmp_path / version).mkdir()
        df = pd.DataFrame({"text": ["fit", "tested", "stressed"], "split": ["train", "test", "stress"]})
        df.assign(sample_id=df["text"].map(sample_id)).to_parquet(tmp_path / version / "corpus.parquet", index=False)
    frame = B.earlier_frame(tmp_path, ("v1", "v2"))
    assert sorted(frame["text"]) == ["fit", "stressed", "tested"]
    assert dict(zip(frame["text"], frame["seen"])) == {"fit": True, "tested": True, "stressed": False}
    assert sorted(B.earlier_texts(tmp_path, ("v1", "v2"))) == ["fit", "stressed", "tested"]


def test_the_build_drops_an_ai_only_source_the_probe_flags(tmp_path):
    ai = SourceSpec(name="arena140k", licence="MIT", licence_class=LicenceClass.PERMISSIVE)
    human = SourceSpec(name="dolly", licence="MIT", licence_class=LicenceClass.PERMISSIVE)

    def arena(raw, cap):  # a giveaway the view keeps: shouting and exclamation marks
        for i in range(120):
            yield make_sample(ai, text=f"ABSOLUTELY {_text(30).upper()} GREAT!!! {i}", label=Label.AI,
                              role=Role.CORE, record_id=str(i), domain="chat", group_key=f"x{i}",
                              generator="gpt-4o")

    def dolly(raw, cap):
        for i in range(120):
            yield make_sample(human, text=f"{_text(30)} number {i}.", label=Label.HUMAN, role=Role.CORE,
                              record_id=str(i), domain="open_qa", group_key=f"y{i}")

    root = tmp_path / "corpus"
    record = B.build(sources={"a": fake_source("a", 1), "arena140k": SimpleNamespace(SPEC=ai, load=arena),
                              "dolly": SimpleNamespace(SPEC=human, load=dolly)},
                     cap=10_000, root=root, lock_file=tmp_path / "lock.json", record_file=tmp_path / "rec.json",
                     version="v1", retired=frozenset(), retired_groups=frozenset(), previous_versions=())
    assert record["ai_only_probe"]["excluded"] == ["arena140k"]
    assert "arena140k" not in set(pd.read_parquet(root / "v1" / "corpus.parquet")["source_dataset"])


# --- I3: format3 is final after one application ------------------------------------------------------

FORMAT3 = text_view("format3+typography+collapse_whitespace")


@pytest.mark.parametrize("text", ["1.\n2.\n3. text", " 1. item", "　# Heading", "> > > > > deep quote",
                                  "> > quoted\n> > > deeper reply\nplain"])
def test_format3_is_idempotent_on_nested_markers(text):
    once = FORMAT3(text)
    assert FORMAT3(once) == once


def test_format3_is_idempotent_on_random_markup():
    pieces = ["#", "##", ">", "> ", "-", "--", "*", "**", "_", "`", "1.", "2)", "(3)", "•", "·",
              "—", "–", "\n", " ", "\t", "　", "word", "Word.", "it’s", "…", "x-y"]
    r = random.Random(0)
    for _ in range(3000):
        text = "".join(r.choice(pieces) + r.choice(["", " ", "\n"]) for _ in range(r.randint(1, 25)))
        once = FORMAT3(text)
        assert FORMAT3(once) == once, repr(text)


def test_format3_leaves_ordinary_prose_alone():
    prose = "On 3 May the well-known chef said it was fine. Prices rose 10% in 2020."
    assert FORMAT3(prose) == prose


# --- I4: the default variant keeps v3's generator pools ----------------------------------------------


def test_the_default_variant_keeps_v3s_generator_pools():
    from src.corpus import generate as G

    assert G.DEFAULT_POOLS["stackexchange"] == ("llama3.2:3b", "qwen3.5:2b", "gemma3:4b", "ministral-3:3b",
                                                "phi4-mini", "granite4:3b")
    assert G.DEFAULT_POOLS["dolly"] == G.DEFAULT_POOLS["oasst1"] == G.TRAINING_GENERATORS


# --- I2: excerpts are cut from each scored model's own view -------------------------------------------


def test_excerpts_are_cut_from_the_scored_models_view():
    from src.authorship import benchmark

    seen = {}

    class Recorder:
        def __init__(self, name):
            self.name = name

        def check_batch(self, texts, context):
            seen[self.name] = list(texts)
            return [SimpleNamespace(decision=SimpleNamespace(value="inconclusive"),
                                    reason_codes=["SCORE_IN_ABSTENTION_BAND"], score=0.5) for _ in texts]

    long = " ".join(f"We had tea—then walked to stop {i}." for i in range(40))
    test = pd.DataFrame({"sample_id": ["a"], "text": [long], "y": [0], "source_dataset": ["s"]})
    benchmark.excerpt_report(test, {"v5": Recorder("v5")}, view="format2+typography+collapse_whitespace")
    benchmark.excerpt_report(test, {"4d": Recorder("4d")}, view="format3+typography+collapse_whitespace")
    assert "tea-then" in seen["v5"][0] and "tea then" in seen["4d"][0]
