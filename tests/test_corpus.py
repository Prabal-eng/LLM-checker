"""Corpus ingestion, splitting, locking, and audit. No network or GPU required."""

import json
import random
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from pydantic import ValidationError

from src.corpus import dedup, pii
from src.corpus.audit import audit
from src.corpus.build import BuildError, build
from src.corpus.schema import Label, LicenceClass, Role, Sample
from src.corpus.sources import beemo, detector_bias, hc3, legacy, mage, mirage, raid
from src.corpus.sources._common import SourceSpec, generator_family, make_sample
from src.corpus.split import assign_splits, partition
from src.corpus.text import sample_id

VOCAB = [f"w{i}" for i in range(4000)]


def sentence(rng: random.Random, n: int = 40) -> str:
    return " ".join(rng.choice(VOCAB) for _ in range(n))


def fake_source(name: str, seed: int, groups: int = 120, stress_groups: int = 10,
                family_generator: str = "gpt-4o", holdout_generator: str = "granite4:3b"):
    spec = SourceSpec(name=name, licence="MIT", licence_class=LicenceClass.PERMISSIVE)

    def load(raw: Path, cap: int):
        rng = random.Random(seed)
        for g in range(groups):
            common = dict(domain=f"d{g % 3}", group_key=f"g{g}")
            yield make_sample(spec, text=sentence(rng), label=Label.HUMAN, role=Role.CORE,
                              record_id=f"{g}:h", **common)
            yield make_sample(spec, text=sentence(rng), label=Label.AI, role=Role.CORE,
                              record_id=f"{g}:a", generator=family_generator, **common)
            if g % 10 == 0:
                yield make_sample(spec, text=sentence(rng), label=Label.AI, role=Role.CORE,
                                  record_id=f"{g}:x", generator=holdout_generator, **common)
            if g < stress_groups:
                yield make_sample(spec, text=sentence(rng), label=Label.AI, role=Role.STRESS,
                                  record_id=f"{g}:s", generator=family_generator,
                                  transform="human_edit", **common)

    return SimpleNamespace(SPEC=spec, load=load)


@pytest.fixture
def built(tmp_path):
    kwargs = dict(sources={"a": fake_source("a", 1), "b": fake_source("b", 2)}, cap=10_000,
                  root=tmp_path / "corpus", lock_file=tmp_path / "repo" / "test.lock.json",
                  record_file=tmp_path / "repo" / "build_record.json", version="v1", retired=frozenset(),
                  retired_groups=frozenset(), previous_versions=())
    record = build(**kwargs)
    df = pd.read_parquet(tmp_path / "corpus" / "v1" / "corpus.parquet")
    return SimpleNamespace(kwargs=kwargs, record=record, df=df, tmp=tmp_path)


# --- schema and text -------------------------------------------------------

def _sample(**overrides):
    fields = dict(sample_id="0" * 64, text="t", label=Label.HUMAN, role=Role.CORE, source_dataset="s",
                  source_record_id="r", licence="MIT", licence_class=LicenceClass.PERMISSIVE,
                  verification="dataset_label", domain="d", group_key="g")
    fields.update(overrides)
    return Sample(**fields)


def test_schema_requires_generator_for_ai_and_forbids_it_for_human():
    with pytest.raises(ValidationError, match="generator"):
        _sample(label=Label.AI)
    with pytest.raises(ValidationError, match="generator"):
        _sample(generator="gpt-4o", generator_family="openai")
    assert _sample(label=Label.AI, generator="gpt-4o", generator_family="openai")


def test_schema_requires_licence():
    with pytest.raises(ValidationError):
        _sample(licence="")


def test_sample_id_ignores_case_and_whitespace():
    assert sample_id("Hello   World\n") == sample_id("hello world")
    assert sample_id("hello world") != sample_id("hello, world")


# --- PII ---------------------------------------------------------------------

@pytest.mark.parametrize("text, kind", [
    ("mail jane.doe@example.com now", "EMAIL"),
    ("call (555) 123-4567 today", "PHONE"),
    ("call +44 20 7946 0958 today", "PHONE"),
    ("card 4111 1111 1111 1111 ok", "CARD"),
    ("host 192.168.10.4 down", "IP"),
])
def test_pii_redacts_each_kind(text, kind):
    redacted, hits = pii.redact(text)
    assert hits[kind] == 1
    assert f"[{kind}]" in redacted
    assert not pii.find(redacted)


def test_pii_redacts_numbers_exposed_by_an_earlier_redaction():
    redacted, hits = pii.redact("tel (713) 853-3989(713) 646-2551 Fax")
    assert hits["PHONE"] == 2
    assert not pii.find(redacted)


@pytest.mark.parametrize("text", [
    "Between 1998-2003 there were 1,234 items.",
    "Version 1.2.3 shipped with 16 GB of memory.",
    "Order number 1234567890123 is not a valid card.",
])
def test_pii_leaves_ordinary_numbers_alone(text):
    assert pii.redact(text) == (text, pii.find(text))
    assert not pii.find(text)


# --- generator families and source parsers -----------------------------------

@pytest.mark.parametrize("generator, family", [
    ("gpt-3.5-trubo", "openai"), ("GPT-4o", "openai"), ("GPT-o3-mini", "openai"),
    ("text-davinci-003", "openai"), ("gpt_j", "eleutherai"), ("gpt-neox", "eleutherai"),
    ("Claude-3.7-sonnet", "anthropic"), ("gemini-2.0-flash", "google"), ("flan_t5_xxl", "google"),
    ("t0_11b", "bigscience"), ("bloom_7b", "bigscience"), ("llama-chat", "meta"), ("opt_30b", "meta"),
    ("mistral-chat", "mistral"), ("HuggingFaceH4/zephyr-7b-beta", "mistral"), ("allenai/tulu-2-7b", "meta"),
    ("cohere-chat", "cohere"), ("mpt-chat", "mosaicml"), ("Qwen2.5-7B-Instruct", "alibaba"),
    ("Qwen2.5-7B-Instruct-R1-Distill", "deepseek"), ("grok2", "xai"), ("moonshot-v1", "moonshot"),
    ("Doubao-1.5-pro-32k", "bytedance"), ("GLM130B", "zhipu"), ("13B", "meta"), ("7B", "meta"),
])
def test_generator_family(generator, family):
    assert generator_family(generator) == family


@pytest.mark.parametrize("src, expected", [
    ("cmv_human", ("cmv", Label.HUMAN, None)),
    ("sci_gen_human", ("sci_gen", Label.HUMAN, None)),
    ("eli5_machine_specified_gpt-3.5-trubo", ("eli5", Label.AI, "gpt-3.5-trubo")),
    ("xsum_machine_continuation_flan_t5_xxl", ("xsum", Label.AI, "flan_t5_xxl")),
    ("cnn_gpt4", ("cnn", Label.AI, "gpt-4")),
    ("unparseable", None),
])
def test_mage_src_parsing(src, expected):
    assert mage.parse_src(src) == expected


def test_beemo_edit_variants_parse_prompt_lists():
    assert beemo._edit_variants("[{'P1': 'a'}, {'P2': 'b'}]") == [("P1", "a"), ("P2", "b")]
    assert beemo._edit_variants("not a list") == []
    assert beemo._edit_variants(None) == []


def _texts(rng, n):
    return [sentence(rng, 30) for _ in range(n)]


def test_loaders_read_their_file_formats(tmp_path):
    rng = random.Random(7)
    t = _texts(rng, 40)

    m = tmp_path / "mage"
    m.mkdir()
    core = pd.DataFrame({"text": t[:4], "label": [1, 0, 0, 1],
                         "src": ["cmv_human", "cmv_machine_topical_gpt-4", "cmv_human", "cmv_human"]})
    for f in mage._CORE_FILES:
        core.assign(text=[x + f for x in core["text"]]).to_csv(m / f, index=False)
    pd.DataFrame({"text": [t[4]], "label": [0], "src": ["cnn_gpt4"]}).to_csv(m / mage._PARAPHRASE_FILE, index=False)
    got = list(mage.load(m, 100))
    # The row labelled 0 with a human src disagrees with itself and is dropped.
    assert {s.role for s in got} == {Role.CORE, Role.STRESS}
    assert all(s.label is Label.HUMAN for s in got if s.domain == "cmv" and s.generator is None)
    assert [s.transform for s in got if s.role is Role.STRESS] == ["paraphrase"]

    h = tmp_path / "hc3"
    h.mkdir()
    for d in hc3._DOMAINS:
        (h / f"{d}.jsonl").write_text(json.dumps(
            {"question": d, "human_answers": [t[5] + d], "chatgpt_answers": [t[6] + d]}) + "\n", encoding="utf-8")
    got = list(hc3.load(h, 100))
    assert len(got) == 10 and {s.generator for s in got} == {None, "chatgpt"}
    assert len({s.group_key for s in got}) == 5

    r = tmp_path / "raid"
    r.mkdir()
    pd.DataFrame({
        "id": ["h0", "a0", "z0", "c0"], "source_id": [None, "h0", "h0", None],
        "model": ["human", "gpt4", "gpt4", "human"], "attack": [None, None, "paraphrase", None],
        "domain": ["news", "news", "news", "czech"], "generation": [t[7], t[20], t[30], t[31]],
    }).to_csv(r / raid._FILE, index=False)
    got = list(raid.load(r, 100))
    assert {s.label for s in got if s.role is Role.CORE} == {Label.HUMAN, Label.AI}
    assert all(s.group_key.startswith("raid:h") for s in got)
    assert all(s.transform == "attack:paraphrase" for s in got if s.role is Role.STRESS)
    assert all(s.domain == "news" for s in got)

    b = tmp_path / "beemo"
    b.mkdir()
    pd.DataFrame([{"id": 0, "category": "Open QA", "model": "google/gemma-7b-it", "prompt_id": "p",
                   "prompt": "q", "model_output": t[17], "human_output": t[18], "human_edits": t[19],
                   "llama-3.1-70b_edits": str([{"P1": t[36]}]), "gpt-4o_edits": str([{"P1": t[37]}])}]
                 ).to_parquet(b / beemo._FILE, index=False)
    got = list(beemo.load(b, 100))
    assert sorted(s.transform for s in got) == ["human_edit", "llm_humanize", "llm_humanize", "none", "none"]
    assert {s.generator_family for s in got if s.transform == "llm_humanize"} == {"meta", "openai"}

    g = tmp_path / "mirage"
    for setting in mirage._SETTINGS:
        (g / setting).mkdir(parents=True)
        for task in mirage._TASKS:
            (g / setting / f"{task}.json").write_text(json.dumps([
                {"original": t[38], "rewritten": t[39] + setting + task, "domain": "news", "model": "GPT-4o"}
            ]), encoding="utf-8")
    got = list(mirage.load(g, 100))
    assert sum(s.label is Label.HUMAN for s in got) == 1
    assert {s.transform for s in got if s.role is Role.STRESS} == {"llm_polish", "llm_rewrite"}

    d = tmp_path / "bias"
    d.mkdir()
    for name in detector_bias._SETS:
        (d / f"{name}.json").write_text(json.dumps([{"document": t[0] + name}]), encoding="utf-8")
    got = list(detector_bias.load(d, 100))
    assert all(s.role is Role.STRESS for s in got)
    assert {s.writer_group for s in got if s.domain == "toefl_essay"} == {"non_native"}


def test_legacy_loader_keeps_unverified_rows_out_of_training(tmp_path):
    csv = tmp_path / "legacy.csv"
    pd.DataFrame({
        "text": ["rater wrote this", "template text", "copied text", "another rater"],
        "label": [0, 1, 1, 0],
        "source": ["human_gempix", "synthetic_ai", "real_ai_copypasta", "human_personality"],
        "column": ["c"] * 4, "rater": ["rater_1", "synthetic_llm", "rater_2", "rater_3"],
    }).to_csv(csv, index=False)
    got = list(legacy.load(tmp_path, 100, dataset=csv))
    assert {s.role for s in got if s.label is Label.HUMAN} == {Role.INHOUSE}
    assert {s.role for s in got if s.label is Label.AI} == {Role.DIAGNOSTIC}
    assert all(s.licence_class is LicenceClass.INTERNAL for s in got)


# --- deduplication and splitting ---------------------------------------------

def test_near_duplicates_cluster_and_distinct_texts_do_not():
    rng = random.Random(3)
    base = sentence(rng, 80)
    near = base.replace(base.split()[10], "changed", 1)
    other = sentence(rng, 80)
    ids = dedup.cluster([base, near, other])
    assert ids[0] == ids[1] != ids[2]


def _frame(rows):
    df = pd.DataFrame(rows)
    df["sample_id"] = [sample_id(f"text {i}") for i in range(len(df))]
    df["cluster_id"] = range(len(df))
    return df


def test_splits_keep_groups_together_and_hold_out_families():
    rows = []
    for g in range(300):
        rows += [{"group_key": f"g{g}", "role": "core", "generator_family": None},
                 {"group_key": f"g{g}", "role": "core", "generator_family": "openai"},
                 {"group_key": f"g{g}", "role": "core", "generator_family": "ibm"}]
    df = _frame(rows)
    df["split"] = assign_splits(df)

    assert df.groupby("group_key")["split"].apply(lambda s: s.dropna().map(partition).nunique()).max() == 1
    fit = df["split"].isin(["train", "val"])
    assert not (fit & (df["generator_family"] == "ibm")).any()
    assert (df["split"] == "test_unseen_generator").any()
    assert df.loc[df["generator_family"] == "ibm", "split"].isna().any()


def test_stress_samples_pull_their_group_into_evaluation():
    rows = [{"group_key": f"g{g}", "role": r, "generator_family": None}
            for g in range(50) for r in ("core", "stress")]
    df = _frame(rows)
    split = assign_splits(df)
    assert set(split[df["role"] == "core"]) == {"test"}
    assert set(split[df["role"] == "stress"]) == {"stress"}


def test_split_assignment_does_not_depend_on_row_order():
    rows = [{"group_key": f"g{g}", "role": "core", "generator_family": None} for g in range(200)]
    df = _frame(rows)
    forward = assign_splits(df)
    shuffled = df.sample(frac=1, random_state=0)
    assert assign_splits(shuffled).sort_index().equals(forward)


def test_held_out_sources_are_tested_whole_unless_they_share_a_component():
    rows = [{"group_key": f"g{g}", "role": "core", "generator_family": None, "source_dataset": "dolly"}
            for g in range(100)]
    rows += [{"group_key": "g0", "role": "core", "generator_family": None, "source_dataset": "mage"}]
    df = _frame(rows)
    split = assign_splits(df, holdout_sources=frozenset({"dolly"}))
    dolly = df["source_dataset"] == "dolly"
    assert set(split[dolly & (df["group_key"] != "g0")]) == {"test"}
    assert split[dolly & (df["group_key"] == "g0")].isna().all()


def test_retired_components_are_never_tested_again():
    rows = [{"group_key": f"g{g}", "role": "core", "generator_family": None} for g in range(400)]
    rows += [{"group_key": f"g{g}", "role": "stress", "generator_family": None} for g in range(20)]
    df = _frame(rows)
    before = assign_splits(df)
    retired = frozenset(df.loc[before == "test", "sample_id"])
    after = assign_splits(df, retired=retired, salt="v2")
    was_tested = df["sample_id"].isin(retired)
    assert not (after[was_tested & (df["role"] == "core")] == "test").any()
    # Stress variants would force a retired parent back into evaluation, so the component leaves whole.
    stressed = df["group_key"].isin([f"g{g}" for g in range(20)])
    assert after[stressed].isna().all()
    assert (after[~was_tested] == "test").any()  # the fresh test comes from previously fitted rows


def test_stress_variants_can_follow_their_parent():
    rows = []
    for g in range(300):
        rows += [{"group_key": f"g{g}", "role": "core", "generator_family": None, "source_dataset": "beemo"},
                 {"group_key": f"g{g}", "role": "stress", "generator_family": None, "source_dataset": "beemo"}]
    df = _frame(rows)
    split = assign_splits(df, stress_follows_parent=frozenset({"beemo"}))
    core, stress = split[df["role"] == "core"], split[df["role"] == "stress"]
    assert {"train", "val", "test"} <= set(core)
    parent = core.reindex(stress.index - 1).to_numpy()
    assert (stress[parent == "test"] == "stress").all()
    assert stress[parent != "test"].isna().all()


def test_retired_seen_ids_read_the_previous_manifest(tmp_path):
    from src.corpus.build import retired_seen_ids
    (tmp_path / "v1").mkdir()
    pd.DataFrame({"sample_id": ["a", "b", "c", "d"], "split": ["test", "train", "test", "stress"]}).to_parquet(
        tmp_path / "v1" / "manifest.parquet")
    assert retired_seen_ids(tmp_path, ("v1",)) == frozenset({"a", "b", "c"})  # Increment 4d: fitted rows too
    with pytest.raises(BuildError):
        retired_seen_ids(tmp_path, ("v0",))


def test_new_source_helpers():
    from src.corpus.sources.no_robots import first_answer
    from src.corpus.sources.rasbt50k import generator_name
    assert generator_family(generator_name("openai", "luna-medium")) == "openai"
    assert generator_family(generator_name("ollama", "qwen3.6:35b")) == "alibaba"
    assert generator_family(generator_name("openrouter", "moonshotai/kimi-k3")) == "moonshot"
    assert generator_family("nvidia/nemotron-3.5-lightning") == "nvidia"
    assert generator_family("z-ai/glm-5.3") == "zhipu" and generator_family("x-ai/grok-4.6") == "xai"
    assert first_answer([{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]) == "a"


def test_make_sample_licence_override():
    spec = SourceSpec(name="s", licence="MIT", licence_class=LicenceClass.PERMISSIVE)
    s = make_sample(spec, text="x", label=Label.HUMAN, role=Role.CORE, record_id="1", domain="d",
                    group_key="g", licence=("CC BY-SA 4.0", LicenceClass.COPYLEFT))
    assert (s.licence, s.licence_class) == ("CC BY-SA 4.0", LicenceClass.COPYLEFT)


# --- build, lock, and audit ---------------------------------------------------

def test_build_passes_audit_and_writes_lock_and_record(built):
    assert built.record["audit"]["passed"]
    assert (built.tmp / "repo" / "test.lock.json").exists()
    assert set(built.df["split"]) >= {"train", "val", "test", "test_unseen_generator", "stress"}
    assert built.record["counts"]["split_label"]
    assert built.record["shortcut_probe"]["status"] == "ok"


def test_second_build_reproduces_the_lock(built):
    again = build(**built.kwargs)
    assert again["test_lock_sha256"] == built.record["test_lock_sha256"]


def test_changed_evaluation_data_requires_explicit_relock(built):
    changed = dict(built.kwargs, sources={"a": fake_source("a", 1), "b": fake_source("b", 99)})
    with pytest.raises(BuildError, match="relock"):
        build(**changed)
    record = build(**changed, relock_reason="replaced source b")
    lock = json.loads((built.tmp / "repo" / "test.lock.json").read_text())
    assert lock["sha256"] == record["test_lock_sha256"]
    assert lock["history"][0]["reason"] == "replaced source b"


def _lock(df):
    from src.corpus.lock import locked_hash
    return {"sha256": locked_hash(df)}


def test_audit_detects_group_spanning_train_and_test(built):
    df = built.df.copy()
    test_row = df.index[df["split"] == "test"][0]
    train_row = df.index[df["split"] == "train"][0]
    df.loc[train_row, "group_key"] = df.loc[test_row, "group_key"]
    assert any("span" in e for e in audit(df, _lock(df)).errors)


def test_audit_detects_pii(built):
    df = built.df.copy()
    row = df.index[df["split"] == "train"][0]
    df.loc[row, "text"] = df.loc[row, "text"] + " contact me at jane@example.com"
    df.loc[row, "sample_id"] = sample_id(df.loc[row, "text"])
    assert any("PII" in e for e in audit(df, _lock(df)).errors)


def test_audit_detects_missing_licence(built):
    df = built.df.copy()
    df.loc[df.index[0], "licence"] = ""
    assert any("schema" in e for e in audit(df, _lock(df)).errors)


def test_audit_detects_holdout_family_in_training(built):
    df = built.df.copy()
    row = df.index[(df["split"] == "train") & (df["label"] == "ai")][0]
    df.loc[row, ["generator", "generator_family"]] = ["granite4:3b", "ibm"]
    assert any("held-out" in e for e in audit(df, _lock(df)).errors)


def test_audit_detects_lock_mismatch_and_missing_lock(built):
    df = built.df.copy()
    assert any("lock" in e for e in audit(df, {"sha256": "0" * 64}).errors)
    assert any("no test lock" in e for e in audit(df, None).errors)


def test_audit_detects_edited_text_with_stale_id(built):
    df = built.df.copy()
    df.loc[df.index[0], "text"] = "edited after the build"
    assert any("do not match" in e for e in audit(df, _lock(df)).errors)


# --- downloads ----------------------------------------------------------------

class _FlakyServer:
    """Serves a payload in pieces, dropping the connection after each piece."""

    def __init__(self, payload: bytes, piece: int, drops: int):
        self.payload, self.piece, self.drops = payload, piece, drops
        self.ranges = []

    def get(self, url, stream, timeout, headers):
        import requests

        start = int(headers["Range"].split("=")[1].rstrip("-")) if headers else 0
        self.ranges.append(start)
        server = self

        class Response:
            status_code = 206 if start else 200

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def raise_for_status(self):
                pass

            def iter_content(self, chunk_size):
                end = len(server.payload) if server.drops <= 0 else start + server.piece
                yield server.payload[start:end]
                if server.drops > 0:
                    server.drops -= 1
                    raise requests.exceptions.ChunkedEncodingError("connection broken")

        return Response()


def test_download_resumes_through_more_drops_than_the_retry_limit(tmp_path, monkeypatch):
    import hashlib
    from src.corpus.sources import _common

    payload = bytes(range(256)) * 40
    server = _FlakyServer(payload, piece=500, drops=_common.DOWNLOAD_ATTEMPTS + 5)
    monkeypatch.setattr(_common.requests, "get", server.get)

    digest = _common.download("https://example.test/file", tmp_path / "file.bin")

    assert (tmp_path / "file.bin").read_bytes() == payload
    assert digest == hashlib.sha256(payload).hexdigest()
    assert server.ranges[:3] == [0, 500, 1000]


def test_download_gives_up_when_attempts_stop_making_progress(tmp_path, monkeypatch):
    import requests
    from src.corpus.sources import _common

    def always_fails(*args, **kwargs):
        raise requests.ConnectionError("down")

    monkeypatch.setattr(_common.requests, "get", always_fails)
    with pytest.raises(requests.ConnectionError):
        _common.download("https://example.test/file", tmp_path / "file.bin")
    assert not (tmp_path / "file.bin").exists()


def test_containment_and_shared_openings_join_one_cluster():
    rng = random.Random(3)
    base = sentence(rng, 400)
    contained = " ".join(base.split()[50:250])
    unrelated = sentence(rng, 300)
    same_prompt = " ".join(base.split()[:30]) + " " + sentence(rng, 300)
    ids = dedup.cluster_with_containment([base, contained, unrelated, same_prompt])
    assert ids[0] == ids[1] == ids[3] and ids[2] != ids[0]
    assert dedup.containment_pairs([base, contained, unrelated]) == [(0, 1)]


def test_named_sources_get_a_larger_test_share():
    rows = [{"group_key": f"{s}:g{g}", "role": "core", "generator_family": None, "source_dataset": s}
            for s in ("small", "big") for g in range(2000)]
    df = _frame(rows)
    split = assign_splits(df, test_shares={"small": 0.35})
    share = (split == "test").groupby(df["source_dataset"]).mean()
    assert 0.31 < share["small"] < 0.39 and 0.12 < share["big"] < 0.18
    assert 0.12 < (split == "val").groupby(df["source_dataset"]).mean()["small"] < 0.18


def test_components_in_a_retired_group_are_never_tested_again():
    """Amendment 5: v3 re-selects oasst1 replies and strips markdown, so a retired group's rows can
    carry new sample ids. The group itself was tested, so none of its rows may be tested again."""
    rows = [{"group_key": f"g{g}", "role": "core", "generator_family": None} for g in range(400)]
    df = _frame(rows)
    retired_groups = frozenset(f"g{g}" for g in range(200))
    split = assign_splits(df, retired_groups=retired_groups, salt="v3")
    in_retired = df["group_key"].isin(retired_groups)
    assert set(split[in_retired]) <= {"train", "val"}
    assert (split[~in_retired] == "test").any()


def test_settled_splits_survive_reclustering_of_the_kept_rows():
    """Amendment 5: the audit reclusters only the kept rows, where boilerplate shared by fewer texts can
    link texts the build left apart. The build reclusters the kept rows the same way until none span."""
    from src.corpus.build import settle_splits
    rows = [{"group_key": f"g{g}", "role": "core", "generator_family": None} for g in range(200)]
    df = _frame(rows)
    df["text"] = [f"text {i}" for i in range(len(df))]
    pairs = lambda texts: [int(t.split()[1]) // 2 for t in texts]  # noqa: E731  links rows 0-1, 2-3, ...
    unsettled = assign_splits(df, salt="v3")
    assert (unsettled.map(partition).groupby(df.index // 2).nunique() > 1).any()  # the fixture does leak
    split = settle_splits(df, lambda d: assign_splits(d, salt="v3"), pairs)
    assert (split.map(partition).groupby(df.index // 2).nunique() == 1).all()


def test_mirage_leaves_out_its_email_domain(tmp_path):
    """Amendment 5: MIRAGE's human Enron emails carry formatting damage (glued words, document tags)
    that its AI versions lack, so the email domain is excluded."""
    import json as _json
    from src.corpus.sources import mirage
    for setting in ("DIG", "SIG"):
        (tmp_path / setting).mkdir()
        for task in ("generate", "polish", "rewrite"):
            items = [{"domain": d, "model": "gpt-4o", "original": f"{d} original text {setting} {task} " * 8,
                      "rewritten": f"{d} rewritten text by the model {setting} {task} " * 8}
                     for d in ("email", "news")]
            (tmp_path / setting / f"{task}.json").write_text(_json.dumps(items), encoding="utf-8")
    samples = list(mirage.load(tmp_path, cap=100))
    assert samples and {s.domain for s in samples} == {"news"}


def test_lock_covers_exact_text_and_source():
    """Ralph R08-5: sample_id ignores case and whitespace, and the gate slices by source."""
    from src.corpus.lock import locked_hash
    base = pd.DataFrame({"sample_id": ["a", "b"], "split": ["test", "test"], "label": ["human", "ai"],
                         "text": ["Plain text here.", "Other text."], "source_dataset": ["hc3", "mage"]})
    recased = base.assign(text=["plain TEXT here.", "Other text."])
    relabelled = base.assign(source_dataset=["mage", "mage"])
    assert len({locked_hash(base), locked_hash(recased), locked_hash(relabelled)}) == 3


def test_retirement_reads_earlier_groups_even_when_ids_are_given(tmp_path):
    """Ralph R08-10: passing retired ids used to switch off group retirement."""
    from src.corpus.build import retirement
    (tmp_path / "v1").mkdir()
    pd.DataFrame({"sample_id": ["x"], "group_key": ["g1"], "split": ["test"]}).to_parquet(
        tmp_path / "v1" / "manifest.parquet")
    ids, groups = retirement(tmp_path, versions=("v1",), retired=frozenset({"given"}))
    assert ids == frozenset({"given"}) and groups == frozenset({"g1"})


def test_audit_refuses_when_an_earlier_manifest_is_missing(tmp_path):
    from src.corpus.audit import earlier_seen_groups
    (tmp_path / "v1").mkdir()
    pd.DataFrame({"sample_id": ["x"], "group_key": ["g1"], "split": ["test"]}).to_parquet(
        tmp_path / "v1" / "manifest.parquet")
    assert earlier_seen_groups(tmp_path, "v2") == frozenset({"g1"})
    try:
        earlier_seen_groups(tmp_path, "v3")
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("a missing earlier manifest must stop the audit, not shrink it")
