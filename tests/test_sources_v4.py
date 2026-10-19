"""Increment 4d sources: test-only (WasItAIGenerated 2026-09, LEDE frontier news, eight new Stack Exchange
sites) and new training sources."""

import pandas as pd

from src.corpus.schema import Label, LicenceClass, Role
from src.corpus.sources import SOURCES, lede, stackexchange, stackexchange_heldout, wasitai2026


def test_wasitai_keeps_english_full_text_rows_and_makes_humanized_rows_stress(tmp_path):
    (tmp_path / "data").mkdir()
    base = {"group": "g1", "words": 60, "vendor": "OpenAI", "humanizer": None, "humanizer_mode": None,
            "source": "fineweb", "source_dataset": "x", "source_url": "http://example.org", "text_available": True}
    pd.DataFrame([
        {**base, "id": "1", "label": "human", "lang": "en", "generator": None, "text": "A person wrote this " * 5,
         "text_license": "CC-BY-4.0"},
        {**base, "id": "2", "label": "ai", "lang": "en", "generator": "gpt-6", "text": "A model wrote this " * 5,
         "text_license": "CC-BY-4.0"},
        {**base, "id": "3", "label": "humanized", "lang": "en", "generator": "claude-opus-5.5",
         "humanizer": "undetectable", "text": "A model then a humanizer " * 5, "text_license": "CC-BY-4.0"},
        {**base, "id": "4", "label": "human", "lang": "de", "generator": None, "text": "Ein Mensch " * 5,
         "text_license": "CC-BY-4.0"},
        {**base, "id": "5", "label": "human", "lang": "en", "generator": None, "text": None,
         "text_license": "not redistributed (third-party copyright)", "text_available": False},
        {**base, "id": "6", "label": "ai_edited", "lang": "en", "generator": "gpt-6", "text": "Edited " * 5,
         "text_license": "MIT"},
        {**base, "id": "7", "label": "humanized", "lang": "en", "generator": None, "vendor": None,
         "humanizer": "stealth", "text": "Unknown origin then a humanizer " * 5, "text_license": "CC-BY-4.0"},
    ]).to_parquet(tmp_path / "data" / "benchmark.parquet", index=False)
    got = list(wasitai2026.load(tmp_path, 100))
    assert sorted(s.source_record_id for s in got) == ["1", "2", "3", "7"]
    assert next(s for s in got if s.source_record_id == "7").generator == "unknown"  # 30 real rows name none
    stress = next(s for s in got if s.source_record_id == "3")
    assert stress.role is Role.STRESS and stress.label is Label.AI and stress.transform == "humanizer:undetectable"
    assert {s.licence for s in got} == {"CC-BY-4.0"} and {s.licence_class for s in got} == {LicenceClass.PERMISSIVE}
    assert all(s.group_key == "wasitai2026:g1" for s in got)
    assert next(s for s in got if s.source_record_id == "2").generator_family == "openai"


def test_wasitai_licence_classes():
    assert wasitai2026.row_licence("CC-BY-NC-SA-4.0")[1] is LicenceClass.NON_COMMERCIAL
    assert wasitai2026.row_licence("MIT")[1] is LicenceClass.PERMISSIVE
    assert wasitai2026.row_licence("something else")[1] is LicenceClass.UNKNOWN


def test_lede_loads_only_frontier_models_and_generated_strategies(tmp_path):
    eng = tmp_path / "data" / "Eng"
    eng.mkdir(parents=True)
    for stem in ("Claude_Opus_4.6", "GPT_5.4_pro", "Gemini_3.1_Pro", "GPT_4o", "Claude_Opus_4.1"):
        pd.DataFrame({"Unnamed: 0": [0, 1, 2], "human_rid": ["r1", "r2", "r3"], "human_fid": ["f1", "f2", "f3"],
                      "title": ["t"] * 3, "summary": ["s"] * 3,
                      "ai_article": [f"{stem} article {i} about the harbour council vote " * 4 for i in range(3)],
                      "category": ["politics and government"] * 3, "model": [stem] * 3,
                      "strategy": ["sc", "ng", "we"], "language": ["eng"] * 3, "num_sentences": [5] * 3,
                      "num_words": [40] * 3}).to_csv(eng / f"{stem}_Eng.csv", index=False)
    got = list(lede.load(tmp_path, 100))
    assert {s.generator for s in got} == {"Claude Opus 4.6", "GPT 5.4 pro", "Gemini 3.1 Pro"}
    assert {s.generator_family for s in got} == {"anthropic", "openai", "google"}
    assert len(got) == 6  # the 'we' (writing enhancement) row of each file is dropped
    assert all(s.label is Label.AI and s.role is Role.CORE and s.domain == "news" for s in got)
    assert {s.group_key for s in got} == {"lede:r1", "lede:r2"}
    assert {s.licence_class for s in got} == {LicenceClass.NON_COMMERCIAL}


def test_stackexchange_heldout_uses_only_the_new_sites():
    assert set(stackexchange_heldout.SITES).isdisjoint(stackexchange.SITES)
    assert len(stackexchange_heldout.SITES) == 8
    assert stackexchange_heldout.SPEC.name == "stackexchange_heldout"
    assert all(rel.split("/")[0].split(".")[0] in stackexchange_heldout.SITES
               for _, rel in stackexchange_heldout.SPEC.files)


def test_stackexchange_heldout_selects_like_the_v3_loader(tmp_path):
    answer = "<p>" + "Keep the receipt and ask the airline desk before you leave the airport. " * 3 + "</p>"
    for site in stackexchange_heldout.SITES:
        folder = tmp_path / f"{site}.stackexchange.com"
        folder.mkdir()
        pd.DataFrame({"qid": [1, 2], "date": ["2020/05/01", "2023/01/01"],
                      "question": ["<p>What should I do when my bag is lost on a connecting flight?</p>"] * 2,
                      "answers": [[{"text": answer, "pm_score": 5, "selected": True, "answer_id": 10}]] * 2}
                     ).to_parquet(folder / "train-00000-of-00001.parquet", index=False)
    items = stackexchange_heldout.pairs(tmp_path)
    assert len(items) == 8  # one pre-2022 question per site; the 2023 question is after ChatGPT
    got = list(stackexchange_heldout.load(tmp_path, 100))
    assert {s.source_dataset for s in got} == {"stackexchange_heldout"}
    assert all(s.label is Label.HUMAN for s in got)


def test_v4_sources_are_registered():
    assert {"wasitai2026", "lede", "stackexchange_heldout"} <= set(SOURCES)


def test_stack_exchange_reads_every_shard_of_a_site(tmp_path):
    answer = "<p>" + "Turn the valve off at the main before you open the pipe under the sink. " * 3 + "</p>"
    folder = tmp_path / "diy.stackexchange.com"
    folder.mkdir()
    for shard, qid in ((0, 1), (1, 2)):
        pd.DataFrame({"qid": [qid], "date": ["2019/03/02"],
                      "question": ["<p>How do I replace the trap under my kitchen sink without a plumber?</p>"],
                      "answers": [[{"text": answer, "pm_score": 4, "selected": True, "answer_id": qid * 10}]]}
                     ).to_parquet(folder / f"train-0000{shard}-of-00002.parquet", index=False)
    assert sorted(stackexchange.select_pairs(tmp_path, ("diy",), 10)["qid"]) == ["1", "2"]
    assert ("https://huggingface.co/datasets/HuggingFaceH4/stack-exchange-preferences/resolve/main/data/"
            "diy.stackexchange.com/train-00001-of-00002.parquet",
            "diy.stackexchange.com/train-00001-of-00002.parquet") in stackexchange_heldout.SPEC.files


# --- training sources (Task 6) -----------------------------------------------------------------------

import json  # noqa: E402

from src.corpus.sources import arena140k, dactyl, maga, openturingbench  # noqa: E402
from src.corpus.sources._common import LENGTH_BANDS, length_matched  # noqa: E402


def _words(n, seed):
    return " ".join(f"w{seed}x{i}" for i in range(n))


def test_length_matched_keeps_equal_counts_per_band_and_label():
    rows = [{"d": "x", "y": "human", "text": _words(40, i)} for i in range(30)]
    rows += [{"d": "x", "y": "ai", "text": _words(300, i)} for i in range(30)]
    rows += [{"d": "x", "y": "ai", "text": _words(45, 100 + i)} for i in range(5)]
    rows += [{"d": "x", "y": "human", "text": _words(310, 200 + i)} for i in range(8)]
    out = length_matched(pd.DataFrame(rows), ["d"], "y", "text", key="text")
    counts = out.assign(long=out["text"].str.split().str.len() >= 100).groupby(["long", "y"]).size().to_dict()
    assert counts == {(False, "ai"): 5, (False, "human"): 5, (True, "ai"): 8, (True, "human"): 8}


def test_a_capped_length_match_keeps_equal_counts_per_band_and_label():
    """R12-1: capping each label after matching must not undo the match. Humans lean short and AI long,
    with pools far over the cap, as in MAGA."""
    import numpy as np

    rng = np.random.default_rng(0)
    rows = [{"d": "x", "y": "human", "text": _words(int(n), i)}
            for i, n in enumerate(np.clip(rng.lognormal(np.log(80), 0.6, 4000), 5, 900))]
    rows += [{"d": "x", "y": "ai", "text": _words(int(n), 10_000 + i)}
             for i, n in enumerate(np.clip(rng.lognormal(np.log(220), 0.5, 4000), 5, 900))]
    out = length_matched(pd.DataFrame(rows), ["d"], "y", "text", key="text", cap=500)
    words = out["text"].str.split().str.len()
    band = np.searchsorted(np.asarray(LENGTH_BANDS), words, side="right")
    counts = out.assign(band=band).groupby(["band", "y"]).size().unstack(fill_value=0)
    assert (counts["ai"] == counts["human"]).all()
    assert counts["ai"].sum() == 500
    from sklearn.metrics import roc_auc_score
    assert roc_auc_score(out["y"] == "ai", words) < 0.55  # length no longer separates the labels


def test_maga_pairs_humans_with_plain_and_aligned_ai_and_drops_wikihow(tmp_path):
    for folder in ("train", "val"):
        (tmp_path / folder).mkdir()
    records = {"MGB": [], "MAGA": []}
    for i in range(6):
        human = {"id": f"h{i}", "human_source_id": f"h{i}", "prompt_id": None, "model": "human",
                 "domain": "Yahoo Answers" if i < 5 else "wikiHow", "title": "t", "text": _words(150, i),
                 "system_prompt": None, "user_prompt": None}
        for config in records:
            records[config].append(human)
            records[config].append({**human, "id": f"{config}{i}", "model": "Qwen3-8B" if config == "MGB" else "GPT-4o-mini",
                                    "prompt_id": f"p{i}", "text": _words(152, 1000 * (config == "MAGA") + 50 + i)})
    for config, rows in records.items():
        (tmp_path / "train" / f"{config}_train.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
        (tmp_path / "val" / f"{config}_val.jsonl").write_text("", encoding="utf-8")
    got = list(maga.load(tmp_path, 100))
    assert {s.domain for s in got} == {"yahoo_answers"}
    assert sum(s.label is Label.HUMAN for s in got) == 5  # humans appear in both files but load once
    assert {s.transform for s in got if s.label is Label.AI} == {"none", "maga_aligned"}
    assert {s.group_key for s in got} == {f"maga:h{i}" for i in range(5)}


def test_dactyl_drops_writing_prompts_and_keeps_labels(tmp_path):
    rows = [{"id": "1", "text": _words(60, 1), "domain": "reviews", "model": "human", "target": 0, "prompt": None},
            {"id": "2", "text": _words(60, 2), "domain": "reviews", "model": "gpt-4o-mini", "target": 1, "prompt": "p"},
            {"id": "3", "text": _words(60, 3), "domain": "RedditWritingPrompts", "model": "human", "target": 0,
             "prompt": None},
            {"id": "4", "text": _words(60, 4), "domain": "tweets",
             "model": "ShantanuT01/fine-tuned-Llama-3.2-1B-Instruct-apollo-mini-tweets-testing", "target": 1,
             "prompt": "q"},
            {"id": "5", "text": _words(60, 5), "domain": "tweets", "model": "human", "target": 0, "prompt": None}]
    for name in ("train", "validation", "test"):
        pd.DataFrame([{**r, "id": f"{name}{r['id']}", "category_info": None, "generation_info": None,
                       "created_at": None} for r in rows]).to_parquet(tmp_path / f"{name}.parquet", index=False)
    got = list(dactyl.load(tmp_path, 100))
    assert "RedditWritingPrompts" not in {s.domain for s in got}
    assert {(s.domain, s.label) for s in got} == {("reviews", Label.HUMAN), ("reviews", Label.AI),
                                                  ("tweets", Label.HUMAN), ("tweets", Label.AI)}
    assert {s.generator_family for s in got if s.label is Label.AI} == {"openai", "meta"}


def _turn(role, text):
    return {"role": role, "content": [{"type": "text", "text": text, "image": None, "mimeType": None}]}


def test_arena_keeps_english_first_answers_and_drops_held_out_families(tmp_path):
    (tmp_path / "data").mkdir()
    q = _turn("user", "How do I keep basil alive indoors?")
    pd.DataFrame([
        {"id": "c1", "model_a": "gemini-2.5-pro", "model_b": "command-a-03-2025", "language": "en", "is_code": False,
         "conversation_a": [q, _turn("assistant", "Give it bright light and water when dry. " * 4)],
         "conversation_b": [q, _turn("assistant", "Basil loves sun; pinch the tops often. " * 4)]},
        {"id": "c2", "model_a": "phi-4", "model_b": "gpt-4.1", "language": "en", "is_code": False,
         "conversation_a": [q, _turn("assistant", "Phi answer here about basil care. " * 4)],
         "conversation_b": [q, _turn("assistant", "GPT answer here about basil care. " * 4)]},
        {"id": "c3", "model_a": "gpt-4.1", "model_b": "o3", "language": "pl", "is_code": False,
         "conversation_a": [q, _turn("assistant", "Polish " * 20)], "conversation_b": [q, _turn("assistant", "x " * 20)]},
        {"id": "c4", "model_a": "gpt-4.1", "model_b": "o3", "language": "en", "is_code": True,
         "conversation_a": [q, _turn("assistant", "code " * 20)], "conversation_b": [q, _turn("assistant", "y " * 20)]},
    ]).to_parquet(tmp_path / "data" / "train-00000-of-00007.parquet", index=False)
    got = list(arena140k.load(tmp_path, 100))
    assert sorted(s.generator for s in got) == ["command-a-03-2025", "gemini-2.5-pro", "gpt-4.1"]
    assert all(s.label is Label.AI for s in got)
    assert {s.group_key for s in got} == {"arena140k:c1", "arena140k:c2"}
    assert {s.licence_class for s in got} == {LicenceClass.UNKNOWN}


def test_openturingbench_loads_in_domain_ai_news(tmp_path):
    (tmp_path / "in-domain").mkdir()
    for name in ("train", "val", "test"):
        (tmp_path / "in-domain" / f"{name}.json").write_text(json.dumps([
            {"url": f"story-{name}", "content": _words(300, 7), "model": "Qwen/Qwen2.5-7B-Instruct"},
            {"url": f"story-{name}", "content": _words(300, 8), "model": "google/gemma-2-9b-it"}]), encoding="utf-8")
    got = list(openturingbench.load(tmp_path, 100))
    # The 700 MB train file is not read: validation and test already hold more rows than the cap keeps.
    assert len(got) == 4 and all(s.label is Label.AI and s.domain == "news" for s in got)
    assert {s.group_key for s in got} == {f"openturingbench:story-{n}" for n in ("val", "test")}
    assert {s.licence_class for s in got} == {LicenceClass.NON_COMMERCIAL}


def test_magistral_is_a_mistral_model():
    from src.corpus.sources._common import generator_family
    assert generator_family("magistral-medium-2506") == "mistral"
