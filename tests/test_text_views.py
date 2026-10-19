"""format3 (Increment 4d, debts D14 and D15): every bullet glyph and (1) marker goes, dashes follow one
rule, and the view is idempotent. Earlier views keep their exact behaviour."""

import pytest

from src.corpus.text import MODEL_TEXT_VIEW, text_view, text_view_fingerprint

F3 = text_view("format3+typography+collapse_whitespace")
F2 = text_view("format2+typography+collapse_whitespace")

# The fingerprint the v5 registry was sealed with (training_record.json text_view_sha256) must not move.
FORMAT2_FINGERPRINT = "f3289b1f8878b99eae20375b518b4e5bb6986540e86268047f58918fcc2e8eaa"


def test_earlier_view_fingerprints_are_unchanged():
    assert text_view_fingerprint("format2+typography+collapse_whitespace") == FORMAT2_FINGERPRINT


def test_format3_fingerprint_covers_its_new_rules():
    assert text_view_fingerprint("format3+typography+collapse_whitespace") != FORMAT2_FINGERPRINT


@pytest.mark.parametrize("glyph", ["·", "∙", "●", "○", "■", "□", "▫",
                                   "‧", "▸", "►", "✓", "✔", "➤", "•"])
def test_every_bullet_glyph_is_dropped(glyph):
    assert F3(f"{glyph} salt {glyph} pepper") == "salt pepper"


def test_parenthesised_and_bare_numbered_markers_are_dropped():
    assert F3("Do this: (1) mix (2) bake 3) serve") == "Do this: mix bake serve"
    assert F3("version (2024) stays") == "version (2024) stays"  # four digits are not a marker


def test_view_is_idempotent_on_bold_numbers():
    once = F3("**1.** Step one\n**2.** Step two")
    assert once == "Step one Step two"
    assert F3(once) == once


@pytest.mark.parametrize("text", ["tea—then bed", "tea — then bed", "tea -- then bed",
                                  "tea–then bed", "tea - then bed"])
def test_spaced_and_unspaced_dashes_follow_one_rule(text):
    assert F3(text) == "tea then bed"


def test_word_hyphens_stay():
    assert F3("a well-known trick") == "a well-known trick"


def test_format2_behaviour_is_untouched():
    assert F2("tea—then bed") == "tea-then bed"  # the D15 behaviour stays under its old name


# --- format4 (Amendment 13, Ralph R12-4) -----------------------------------------------------------------

F4 = text_view("format4+typography+collapse_whitespace")
FORMAT3_FINGERPRINT = text_view_fingerprint("format3+typography+collapse_whitespace")


def test_model_view_is_format4():
    assert MODEL_TEXT_VIEW == "format4+typography+collapse_whitespace"
    assert text_view_fingerprint(MODEL_TEXT_VIEW) not in (FORMAT2_FINGERPRINT, FORMAT3_FINGERPRINT)


@pytest.mark.parametrize("dash", ["\ufe58", "\u2e3a", "\u2e3b", "\u2012", "\ufe31", "\u2e17", "\u301c", "\u2014"])
def test_every_unicode_dash_is_a_space(dash):
    assert F4(f"tea{dash}then bed") == "tea then bed"


def test_word_hyphens_still_stay_in_format4():
    assert F4("a well-known trick, a well\u2010known one") == "a well-known trick, a well-known one"


@pytest.mark.parametrize("bullet", ["\u25c6", "\u2192", "\u2705", "\U0001f539", "\u2605", "\u27a1\ufe0f", "\u25b6"])
def test_symbol_bullets_opening_a_line_are_dropped(bullet):
    assert F4(f"{bullet} salt\n{bullet} pepper") == "salt pepper"


def test_emoji_inside_a_sentence_stay():
    assert F4("done for today \U0001f600 see you") == "done for today \U0001f600 see you"


def test_circled_and_letter_markers_are_dropped():
    assert F4("\u2460 mix \u2461 bake") == "mix bake"
    assert F4("a) mix\nb) bake\n(c) serve") == "mix bake serve"
    assert F4("I saw a) nothing") == "I saw a) nothing"  # not at a line start


def test_format4_is_idempotent_and_leaves_prose_alone():
    import random
    prose = "On 3 May the well-known chef said it was fine. Prices rose 10% in 2020."
    assert F4(prose) == prose
    pieces = ["#", ">", "-", "--", "*", "1.", "2)", "(3)", "a)", "(b)", "\u2022", "\u25c6", "\u2192", "\u2705",
              "\u2460", "\ufe58", "\u2014", "\n", " ", "\t", "word", "Word.", "it\u2019s", "x-y", "\U0001f600"]
    r = random.Random(1)
    for _ in range(3000):
        text = "".join(r.choice(pieces) + r.choice(["", " ", "\n"]) for _ in range(r.randint(1, 25)))
        once = F4(text)
        assert F4(once) == once, repr(text)


def test_format3_behaviour_is_untouched_by_format4():
    assert F3("tea\ufe58then bed") != "tea then bed"  # format3 keeps its sealed behaviour by name
