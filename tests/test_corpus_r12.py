"""Ralph R12 fixes to corpus v4 loading (Amendment 13)."""
import pytest

from src.corpus.sources._common import trim_header, trim_preamble


@pytest.mark.parametrize("text, expected", [
    ("Here is the body of the Reddit post:\n\nSo I went to the store.", "So I went to the store."),
    ("Certainly! Here's an improved version of your abstract:\nWe study graphs.", "We study graphs."),
    ("Sure, here's the body of an Amazon review:\n\nGreat blender.", "Great blender."),
    ("Here's a lyrical article starting with the exact phrase you specified:The river ran.", "The river ran."),
    ("Here's the article:a low-noise amplifier.", "A low-noise amplifier."),
    ("Sure! Here's an original 150-word article:---**Have you seen it?**", "**Have you seen it?**"),
    ("Certainly! Below is an improved version of the article:\n\nThe council met.", "The council met."),
])
def test_self_announcing_preambles_are_trimmed(text, expected):
    assert trim_preamble(text) == expected


@pytest.mark.parametrize("text", [
    "Here's the thing about cats: they sleep a lot.",
    "Here is what happened to me last week. I lost my keys.",
    "DECEMBER 22--Nearly 12 months in the making, here is the report: it is long.",
    "The article: a summary of the debate.",
    "Here is the body of the post:",  # nothing would be left
])
def test_ordinary_openers_are_left_alone(text):
    assert trim_preamble(text) == text


def test_news_template_lines_are_trimmed_and_the_title_kept():
    text = '"Tips for Brides"\n\nDate: November 26, 2013\n\nCategory: WEDDINGS\n\nAs the season nears, brides plan.'
    assert trim_header(text) == '"Tips for Brides"\n\nAs the season nears, brides plan.'
    text = ("HEADLINE: Mugabe Speaks\nSUBHEADLINE: A long talk\nBy [Your Name]\nDATE: June 13, 2016\n"
            "CATEGORY: WORLD NEWS\n\nIn the aftermath, crowds gathered.")
    assert trim_header(text) == "Mugabe Speaks\nA long talk\n\nIn the aftermath, crowds gathered."


@pytest.mark.parametrize("category", ["Parenting", "U.S. News", "Culture & Arts", "Latino Voices", "WorldPost"])
def test_title_case_categories_are_template_lines_too(category):
    text = f'"A Title"\nCategory: {category}\nDate: May 2, 2016\n\nCrowds gathered in the square.'
    assert trim_header(text) == '"A Title"\n\nCrowds gathered in the square.'


def test_template_words_later_in_the_text_stay():
    text = "The vote came first.\nDate: set by the council.\nCategory: none was given."
    assert trim_header(text) == text


import json  # noqa: E402

from src.corpus.schema import Label, Role  # noqa: E402

_BODY = " ".join(f"w{i}" for i in range(60))


def test_maga_trims_preambles_from_its_rows(tmp_path):
    from src.corpus.sources import maga
    for folder in ("train", "val"):
        (tmp_path / folder).mkdir()
    rows = [{"id": "h1", "human_source_id": "h1", "prompt_id": None, "model": "human", "domain": "Reddit",
             "title": "t", "text": "Went hiking today. " + _BODY, "system_prompt": None, "user_prompt": None},
            {"id": "a1", "human_source_id": "h1", "prompt_id": "p1", "model": "Qwen3-8B", "domain": "Reddit",
             "title": "t", "text": "Here is the body of the Reddit post:\n\nWent biking today. " + _BODY,
             "system_prompt": None, "user_prompt": None}]
    for config in ("MGB", "MAGA"):
        (tmp_path / "train" / f"{config}_train.jsonl").write_text(
            "\n".join(json.dumps({**r, "id": r["id"] + config if r["model"] != "human" else r["id"]}) for r in rows),
            encoding="utf-8")
        (tmp_path / "val" / f"{config}_val.jsonl").write_text("", encoding="utf-8")
    got = list(maga.load(tmp_path, 100))
    ai = [s.text for s in got if s.label is Label.AI]
    assert ai and all(t.startswith("Went biking today.") for t in ai)


def test_mirage_trims_core_rows_and_leaves_stress_rows_alone(tmp_path):
    from src.corpus.sources import mirage
    for setting in mirage._SETTINGS:
        (tmp_path / setting).mkdir()
        for task in mirage._TASKS:
            (tmp_path / setting / f"{task}.json").write_text(json.dumps([
                {"original": "The storm hit the coast. " + _BODY, "domain": "news", "model": "GPT-4o",
                 "rewritten": f"Here's the article:the storm {setting} {task} hit. " + _BODY}]), encoding="utf-8")
    got = list(mirage.load(tmp_path, 100))
    core_ai = [s.text for s in got if s.label is Label.AI and s.role is Role.CORE]
    stress = [s.text for s in got if s.role is Role.STRESS]
    assert core_ai and all(t.startswith("The storm") for t in core_ai)
    assert stress and all(t.startswith("Here's the article:") for t in stress)


def test_openturingbench_trims_its_template_lines(tmp_path):
    from src.corpus.sources import openturingbench
    for rel in openturingbench._FILES:
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text(json.dumps([
            {"url": f"u{rel}", "model": "gpt-4o",
             "content": f'"A Title {rel}"\nDate: May 2, 2016\nCategory: WORLD NEWS\n\nCrowds gathered. ' + _BODY}]),
            encoding="utf-8")
    got = list(openturingbench.load(tmp_path, 100))
    assert got and all("Date:" not in s.text and "Category:" not in s.text for s in got)
    assert all(s.text.startswith('"A Title') for s in got)


def test_generated_answers_lose_self_announcing_preambles_too():
    from src.corpus.generate import trim_opener
    assert trim_opener("Hey there! Here's a casual version of the answer:\nJust reboot it.") == "Just reboot it."
    assert trim_opener("Here's the thing: reboot it.") == "Here's the thing: reboot it."


@pytest.mark.parametrize("text, reason", [
    ("I can't provide veterinary advice, but a vet can help with that.", "refusal"),
    ("I cannot provide information on the preservation of animal heads.", "refusal"),
    ("Sure. As an AI language model, I don't have opinions, but here goes.", "refusal"),
    ("Reboot it first, then unplug the charger, wait thirty seconds and plug it back in again, which clears "
     "most faults. That fixes most cases (about 150 words). Then check the cable, the adapter and the socket, "
     "and try another outlet before you replace anything.", "length_note_inside"),
    ("Pick an apple that is firm. It is an temperate fruit that keeps well in the cold.".replace("an temperate",
                                                                                              "an \u6e29\u5e26"), "code_switch"),
])
def test_generated_answers_with_these_artifacts_are_dropped(text, reason):
    from src.corpus.generate import artifact
    assert artifact(text, "stop") == reason


@pytest.mark.parametrize("text", ["Write it in five words or fewer, then read it aloud to someone.",
                                  "I can't stand slow laptops either; reboot it and check the cable."])
def test_ordinary_answers_are_kept(text):
    from src.corpus.generate import artifact
    assert artifact(text, "stop") is None
