"""Local paired generation (Increment 4c): deterministic, resumable, provenance-tracked; no Ollama needed."""

import json

import pandas as pd

from src.corpus import generate as G
from src.corpus.sources import stackexchange
from src.corpus.sources._common import generated_licence, strip_markdown


def _items(n=12):
    return pd.DataFrame({"item_id": [f"s:{i}" for i in range(n)], "prompt": [f"Question {i}?" for i in range(n)],
                         "human_text": ["word " * (20 + i * 10) for i in range(n)]})


def _fake_chat(calls):
    def chat(model, message, seed, max_tokens, **options):
        calls.append((model, message, seed, max_tokens))
        return {"message": {"content": f"answer from {model}"}, "done_reason": "stop", "eval_count": 5}
    return chat


INFO = lambda model: {"digest": "d", "licence": "MIT License", "quantization": "Q4_K_M"}  # noqa: E731


def test_generation_is_deterministic_resumable_and_recorded(tmp_path):
    calls = []
    pool = ("m1", "m2")
    first = G.generate("src", _items(), pool, root=tmp_path, limit=8, chat_fn=_fake_chat(calls), info_fn=INFO)
    assert sum(first.values()) == 8
    again = G.generate("src", _items(), pool, root=tmp_path, limit=8, chat_fn=_fake_chat(calls), info_fn=INFO)
    assert again == {} and len(calls) == 8  # the same subset, already answered
    records = [json.loads(line) for p in (tmp_path / "raw" / "src" / "generated").glob("*.jsonl")
               for line in p.read_text(encoding="utf-8").splitlines()]
    assert {r["generator"] for r in records} <= set(pool)
    assert all(r["generator"] == G.pick(r["item_id"], pool) for r in records)
    r = records[0]
    assert r["options"]["seed"] == G.seed_of(r["item_id"]) and r["model_licence"] == "MIT License"
    assert r["options"]["num_predict"] == 2 * r["target_words"] + 60
    assert set(G.load_generated(tmp_path / "raw" / "src")["item_id"]) == {x["item_id"] for x in records}


def test_target_length_follows_the_human_answer_within_bounds():
    assert G.target_words("word " * 10) == G.MIN_TARGET
    assert G.target_words("word " * 90) == 90
    assert G.target_words("word " * 900) == G.MAX_TARGET
    assert G.user_message("Why?", 90).endswith("Answer in about 90 words.")


def test_models_filter_limits_one_run_to_the_ready_generators(tmp_path):
    calls = []
    G.generate("src", _items(), ("m1", "m2"), root=tmp_path, models={"m1"}, chat_fn=_fake_chat(calls), info_fn=INFO)
    assert {c[0] for c in calls} == {"m1"}


def test_markdown_markers_and_licences():
    assert strip_markdown("## Plan\n**Step** one and __two__") == "Plan\nStep one and two"
    assert generated_licence("MIT License")[1].value == "permissive"
    assert generated_licence("LLAMA 3.2 COMMUNITY LICENSE AGREEMENT")[1].value == "unknown"


def test_stack_exchange_html_becomes_plain_text():
    text = stackexchange.to_text("<p>Use <em>fresh</em> basil &amp; salt.</p><ul><li>one</li><li>two</li></ul>")
    assert text == "Use fresh basil & salt.\n\n- one\n\n- two"


def _write_generated(raw, records):
    out = raw / "generated"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "m1.jsonl").open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps({"generator": "m1", "model_licence": "MIT License", "done_reason": "stop", **r}) + "\n")


def test_load_generated_drops_cut_off_answers_and_word_count_artifacts(tmp_path):
    """Amendment 4: an answer cut off at the token cap, or one that restates the requested length at its
    start or end, carries an artifact of our prompt rather than of AI writing."""
    body = "Basil likes warm soil and regular water. " * 6
    _write_generated(tmp_path, [
        {"item_id": "keep", "text": body},
        {"item_id": "keep-mid", "text": "Plan first.\n" + body + "Aim for 500 words/day early on. " + body},
        {"item_id": "cut", "text": body + "and then the", "done_reason": "length"},
        {"item_id": "opening", "text": "Okay, here's a guide to basil, aiming for around 117 words:\n\n" + body},
        {"item_id": "ending", "text": body + "\n\n*(Word count: ~180)*"},
    ])
    assert set(G.load_generated(tmp_path)["item_id"]) == {"keep", "keep-mid"}


class _Reply:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload


SHOW_PHI = {"license": "Microsoft.\nCopyright (c) Microsoft Corporation.\n\nMIT License\n\nPermission is hereby granted",
            "modelfile": "FROM x", "template": "{{ .Prompt }}", "modified_at": "2026-09-27T15:02:53.3301033+05:30",
            "details": {"parent_model": "", "format": "gguf", "family": "phi3", "families": ["phi3"],
                        "parameter_size": "3.8B", "quantization_level": "Q4_K_M"},
            "model_info": {}, "tensors": [], "capabilities": ["completion"]}
TAGS = {"models": [
    {"name": "phi4-mini:latest", "model": "phi4-mini:latest", "modified_at": "2026-09-27T15:02:53.3301033+05:30",
     "size": 1, "digest": "78fad5d182a7", "details": {}},
    {"name": "llama3.2:3b", "model": "llama3.2:3b", "modified_at": "2026-09-27T14:02:45+05:30",
     "size": 1, "digest": "a80c4f17acd5", "details": {}},
]}


def _fake_ollama(monkeypatch, show):
    monkeypatch.setattr(G.requests, "post", lambda url, json=None, timeout=None: _Reply(show))
    monkeypatch.setattr(G.requests, "get", lambda url, timeout=None: _Reply(TAGS))


def test_model_info_takes_the_digest_from_tags_because_show_has_none(monkeypatch):
    _fake_ollama(monkeypatch, SHOW_PHI)
    info = G.model_info("phi4-mini")
    assert info["digest"] == "78fad5d182a7"
    assert info["modified_at"] == "2026-09-27T15:02:53.3301033+05:30"


def test_model_info_names_the_licence_from_its_full_text(monkeypatch):
    _fake_ollama(monkeypatch, SHOW_PHI)
    assert G.model_info("phi4-mini")["licence"] == "MIT License"
    _fake_ollama(monkeypatch, {**SHOW_PHI, "license": "LLAMA 3.2 COMMUNITY LICENSE AGREEMENT\nLlama 3.2 Version"})
    assert G.model_info("llama3.2:3b")["licence"] == "LLAMA 3.2 COMMUNITY LICENSE AGREEMENT"
    _fake_ollama(monkeypatch, {**SHOW_PHI, "license": "Some preamble.\nNothing named here."})
    assert G.model_info("llama3.2:3b")["licence"] == "Some preamble."


def _records(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_backfill_fills_missing_provenance_and_keeps_a_backup(tmp_path):
    out = tmp_path / "generated"
    out.mkdir()
    (out / "phi4-mini.jsonl").write_text(
        json.dumps({"item_id": "a", "generator": "phi4-mini", "model_digest": None, "model_licence": "Microsoft.",
                    "text": "answer", "created_at": "2026-09-27T16:30:29+00:00"}) + "\n", encoding="utf-8")
    info = {"phi4-mini": {"digest": "78fad5d182a7", "licence": "MIT License",
                          "modified_at": "2026-09-27T15:02:53.3301033+05:30"}}
    assert G.backfill_provenance(out, info, note="test") == {"phi4-mini": 1}
    r = _records(out / "phi4-mini.jsonl")[0]
    assert (r["model_digest"], r["model_licence"], r["text"]) == ("78fad5d182a7", "MIT License", "answer")
    assert r["provenance_backfill"] == "test"
    assert _records(out / "backup" / "phi4-mini.jsonl")[0]["model_digest"] is None
    assert G.backfill_provenance(out, info, note="test") == {}  # nothing left to fill


def test_backfill_refuses_a_model_changed_after_its_first_answer(tmp_path):
    out = tmp_path / "generated"
    out.mkdir()
    (out / "phi4-mini.jsonl").write_text(
        json.dumps({"item_id": "a", "generator": "phi4-mini", "model_digest": None, "model_licence": "Microsoft.",
                    "text": "answer", "created_at": "2026-09-27T09:00:00+00:00"}) + "\n", encoding="utf-8")
    info = {"phi4-mini": {"digest": "78fad5d182a7", "licence": "MIT License",
                          "modified_at": "2026-09-27T15:02:53.3301033+05:30"}}  # 09:32 UTC, after the answer
    try:
        G.backfill_provenance(out, info, note="test")
    except ValueError as exc:
        assert "phi4-mini" in str(exc)
    else:
        raise AssertionError("a model pulled after its first answer must not be backfilled")
    assert _records(out / "phi4-mini.jsonl")[0]["model_digest"] is None


def test_backfill_mode_updates_every_source_without_downloading(tmp_path, monkeypatch):
    monkeypatch.setenv("CORPUS_DIR", str(tmp_path))
    out = tmp_path / "raw" / "dolly" / "generated"
    out.mkdir(parents=True)
    (out / "llama3.2_3b.jsonl").write_text(
        json.dumps({"item_id": "dolly:1", "generator": "llama3.2:3b", "model_digest": None,
                    "model_licence": "LLAMA 3.2 COMMUNITY LICENSE AGREEMENT", "text": "answer",
                    "created_at": "2026-09-27T08:39:43+00:00"}) + "\n", encoding="utf-8")
    monkeypatch.setattr(G, "model_info", lambda m: {"digest": "a80c4f17acd5", "modified_at": "2026-09-27T14:02:45+05:30",
                                                    "licence": "LLAMA 3.2 COMMUNITY LICENSE AGREEMENT"})
    import src.corpus.sources._common as common
    monkeypatch.setattr(common, "download", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no downloads")))
    assert G.main(["--backfill-provenance", "--sources", "dolly"]) == 0
    assert _records(out / "llama3.2_3b.jsonl")[0]["model_digest"] == "a80c4f17acd5"


# --- Increment 4d generation variants ----------------------------------------------------------------

def _variant_records(folder):
    return [json.loads(line) for p in folder.glob("*.jsonl") for line in p.read_text(encoding="utf-8").splitlines()]


def test_variant_answers_go_to_their_own_folder_with_their_style(tmp_path):
    calls = []
    G.generate("src", _items(), ("m1",), root=tmp_path, limit=4, variant="short4d",
               chat_fn=_fake_chat(calls), info_fn=INFO)
    records = _variant_records(tmp_path / "raw" / "src" / "generated_short4d")
    assert len(records) == 4
    assert all(25 <= r["target_words"] <= 99 for r in records)
    assert all(set(r["style"]) == {"persona", "plain", "temperature"} for r in records)
    assert all(0.6 <= r["options"]["temperature"] <= 1.1 for r in records)
    assert all(r["options"]["system"] == G.PERSONAS[r["style"]["persona"]] for r in records)
    assert all(r["variant"] == "short4d" for r in records)
    assert not (tmp_path / "raw" / "src" / "generated").exists()


def test_plain_style_asks_for_plain_text_and_default_variant_keeps_v3_settings(tmp_path):
    calls = []
    G.generate("src", _items(40), ("m1",), root=tmp_path, variant="train4d", chat_fn=_fake_chat(calls), info_fn=INFO)
    plain = {r["item_id"] for r in _variant_records(tmp_path / "raw" / "src" / "generated_train4d") if r["style"]["plain"]}
    assert plain and all((G.PLAIN_NOTE in message) == (f"s:{message.split()[1].rstrip('?')}" in plain)
                         for _, message, _, _ in calls)
    calls.clear()
    G.generate("v3", _items(4), ("m1",), root=tmp_path, chat_fn=_fake_chat(calls), info_fn=INFO)
    v3 = _variant_records(tmp_path / "raw" / "v3" / "generated")
    assert all(r["options"]["temperature"] == G.TEMPERATURE and r["options"]["system"] == G.SYSTEM_PROMPT
               and r["seed_salt"] == "" for r in v3)
    assert all(G.PLAIN_NOTE not in message for _, message, _, _ in calls)


def test_style_is_deterministic_and_varied():
    styles = [G.style_of(f"s:{i}", "train4d") for i in range(200)]
    assert styles == [G.style_of(f"s:{i}", "train4d") for i in range(200)]
    assert {s["persona"] for s in styles} == set(G.PERSONAS)
    assert 0.2 < sum(s["plain"] for s in styles) / 200 < 0.4
    assert styles != [G.style_of(f"s:{i}", "short4d") for i in range(200)]


def test_humanize_rewrites_a_tenth_with_a_different_model(tmp_path):
    calls = []
    G.generate("src", _items(40), ("m1", "m2"), root=tmp_path, variant="train4d",
               chat_fn=_fake_chat(calls), info_fn=INFO)
    n = G.humanize("src", "train4d", ("m1", "m2"), root=tmp_path, chat_fn=_fake_chat(calls), info_fn=INFO)
    rewritten = G.load_generated(tmp_path / "raw" / "src", variant="train4d_humanized")
    assert 1 <= len(rewritten) <= 10 and len(rewritten) == n
    original = G.load_generated(tmp_path / "raw" / "src", variant="train4d").set_index("item_id")
    assert all(r.generator != original.loc[r.item_id, "generator"] == r.rewritten_from for r in rewritten.itertuples())
    assert G.humanize("src", "train4d", ("m1", "m2"), root=tmp_path, chat_fn=_fake_chat(calls), info_fn=INFO) == 0


def test_4d_pools_keep_held_out_generators_off_fitted_sites():
    for (source, variant), pool in G.POOLS_4D.items():
        held = set(pool) & set(G.HELD_OUT_GENERATORS)
        assert (source == "stackexchange_heldout") == bool(held), (source, variant)
        assert (source, variant) in G.GENERATION_LIMITS_4D
    assert "falcon3:3b" in G.HELD_OUT_GENERATORS


def test_loaders_read_every_variant_and_tag_humanized_rows(tmp_path):
    raw = tmp_path / "raw" / "src"
    items = _items(12)
    for variant in ("", "short4d", "train4d"):
        G.generate("src", items, ("m1",), root=tmp_path, limit=3, variant=variant,
                   chat_fn=lambda *a, **k: {"message": {"content": "a generated answer " * 5}, "done_reason": "stop"},
                   info_fn=INFO)
    G.humanize("src", "train4d", ("m1", "m2"), root=tmp_path, share=1.0,
               chat_fn=lambda *a, **k: {"message": {"content": "casual rewrite of it " * 5}, "done_reason": "stop"},
               info_fn=INFO)
    every = G.load_all_generated(raw)
    assert set(every["variant"]) == {"", "short4d", "train4d", "train4d_humanized"}
    assert set(every.loc[every["variant"] == "train4d_humanized", "transform"]) == {"llm_rewrite_casual"}
    assert set(every.loc[every["variant"] != "train4d_humanized", "transform"]) == {"none"}


def test_stack_exchange_loader_reads_variants_and_tags_humanized_rows(tmp_path):
    import pandas as pd
    from src.corpus.schema import Label

    raw = tmp_path / "raw" / "stackexchange"
    answer = "<p>" + "Soak the beans overnight and cook them slowly with a bay leaf and salt. " * 3 + "</p>"
    for site in stackexchange.SITES:
        (raw / f"{site}.stackexchange.com").mkdir(parents=True)
        pd.DataFrame({"qid": [1], "date": ["2020/01/01"],
                      "question": ["<p>How long should dried beans soak before cooking them at home?</p>"],
                      "answers": [[{"text": answer, "pm_score": 3, "selected": True, "answer_id": 7}]]}
                     ).to_parquet(raw / f"{site}.stackexchange.com" / "train-00000-of-00001.parquet", index=False)
    items = stackexchange.pairs(raw)
    reply = lambda *a, **k: {"message": {"content": "Soak them for about eight hours in cold water first. " * 2},  # noqa: E731
                             "done_reason": "stop"}
    G.generate("stackexchange", items, ("m1",), root=tmp_path, variant="short4d", chat_fn=reply, info_fn=INFO)
    G.humanize("stackexchange", "short4d", ("m1", "m2"), root=tmp_path, share=1.0,
               chat_fn=lambda *a, **k: {"message": {"content": "just soak em overnight honestly, works fine " * 2},
                                        "done_reason": "stop"}, info_fn=INFO)
    got = [s for s in stackexchange.load(raw, 100) if s.label is Label.AI]
    assert len(got) == 8  # one short answer per site; the humanized folder of short4d isn't a loaded variant
    assert {s.transform for s in got} == {"none"}
    assert all(s.source_record_id.endswith(":short4d") for s in got)


def test_main_runs_a_variant_with_its_pool_and_limit(tmp_path, monkeypatch):
    import pandas as pd

    monkeypatch.setenv("CORPUS_DIR", str(tmp_path))
    import src.corpus.sources._common as common
    monkeypatch.setattr(common, "download", lambda *a, **k: "sha")
    from src.corpus.sources import stackexchange_heldout
    monkeypatch.setattr(stackexchange_heldout, "pairs", lambda raw: pd.DataFrame(
        {"item_id": ["q1"], "prompt": ["?"], "human_text": ["word " * 50]}))
    monkeypatch.setattr(G, "available_models", lambda: set(G.TRAINING_GENERATORS + G.HELD_OUT_GENERATORS))
    runs = []
    monkeypatch.setattr(G, "generate", lambda source, items, pool, **kw: runs.append((source, pool, kw)) or {})
    assert G.main(["--sources", "stackexchange_heldout", "--variant", "test4d"]) == 0
    assert {r[0] for r in runs} == {"stackexchange_heldout"}
    assert all(r[1] == G.POOLS_4D[("stackexchange_heldout", "test4d")] for r in runs)
    assert all(r[2]["variant"] == "test4d" and r[2]["limit"] == 2100 for r in runs)
    assert [next(iter(r[2]["models"])) for r in runs] == list(G.TRAINING_GENERATORS + G.HELD_OUT_GENERATORS)


def test_trim_greeting_removes_only_a_leading_greeting():
    """Amendment 8: the casual persona opens with greetings that real answers almost never have."""
    assert G.trim_greeting("Hey, that looks like root rot.") == "That looks like root rot."
    assert G.trim_greeting("Hey! 6 weeks? That's too soon.") == "6 weeks? That's too soon."
    assert G.trim_greeting("Hi there, try a lower heat.") == "Try a lower heat."
    assert G.trim_greeting("Hello friend!\n\nWater it weekly.") == "Water it weekly."
    assert G.trim_greeting("Hey, honestly, it depends.") == "Honestly, it depends."  # fillers stay
    assert G.trim_greeting("Okay, so it depends.") == "Okay, so it depends."
    assert G.trim_greeting("Hey Jude is a song by the Beatles.") == "Hey Jude is a song by the Beatles."
    assert G.trim_greeting("Hayden said hi to everyone.") == "Hayden said hi to everyone."
    assert G.trim_greeting("Hey!") == "Hey!"  # nothing left to keep
    assert G.trim_greeting("hey man, nah, don't panic.") == "Nah, don't panic."
    assert G.trim_greeting("Hey Tony, hope you're doing well!") == "Hope you're doing well!"
    assert G.trim_greeting("yo hey, the chain is a mess.") == "The chain is a mess."
    assert G.trim_greeting("Hi all - short answer: yes.") == "Short answer: yes."


def test_trim_opener_also_removes_a_leading_okay():
    """Amendment 9: Gemma opens 34% of its answers with "Okay," against 0.21% of human texts."""
    assert G.trim_opener("Okay, so it depends.") == "So it depends."
    assert G.trim_opener("Okay, wow, that's a lot to unpack!") == "Wow, that's a lot to unpack!"
    assert G.trim_opener("OK. Drain the pan first.") == "Drain the pan first."
    assert G.trim_opener("Okay Marta, that's wild!") == "That's wild!"
    assert G.trim_opener("Hey! Okay, so the chain is a mess.") == "So the chain is a mess."
    assert G.trim_opener("Okay, hey there, try a lower heat.") == "Try a lower heat."
    assert G.trim_opener("Hey, that looks like root rot.") == "That looks like root rot."  # greetings as before
    assert G.trim_opener("Okay so it depends.") == "Okay so it depends."  # no punctuation: part of the sentence
    assert G.trim_opener("OK Go is a band.") == "OK Go is a band."
    assert G.trim_opener("Okayama is a city.") == "Okayama is a city."
    assert G.trim_opener("It's okay, really.") == "It's okay, really."
    assert G.trim_opener("Okay!") == "Okay!"  # nothing left to keep
    assert G.trim_opener("Sure, it depends.") == "Sure, it depends."  # humans open with "Sure" too
    assert G.trim_opener("Hey bro, so your supervisor is wrong.") == "So your supervisor is wrong."
    assert G.trim_opener("Hey bro/sis! Frustrated, huh?") == "Frustrated, huh?"
    assert G.trim_opener("Yo what up! Yeah man, it takes time.") == "Yeah man, it takes time."


def test_load_generated_trims_okay_openers(tmp_path):
    body = "basil likes warm soil and regular water. " * 6
    _write_generated(tmp_path, [{"item_id": "okay", "text": "Okay, " + body}])
    assert G.load_generated(tmp_path).set_index("item_id")["text"]["okay"].startswith("Basil likes")


def test_load_generated_trims_greetings_and_keeps_the_raw_file(tmp_path):
    body = "basil likes warm soil and regular water. " * 6
    _write_generated(tmp_path, [{"item_id": "greet", "text": "Hey there, " + body},
                                {"item_id": "plain", "text": body}])
    loaded = G.load_generated(tmp_path).set_index("item_id")["text"]
    assert loaded["greet"].startswith("Basil likes") and loaded["plain"] == body
    assert "Hey there" in (tmp_path / "generated" / "m1.jsonl").read_text(encoding="utf-8")
