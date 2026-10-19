"""Text normalization shared by identity, deduplication, and auditing."""

import hashlib
import re
import unicodedata

_WS = re.compile(r"\s+")
_WORD = re.compile(r"\w+", flags=re.UNICODE)


def normalize(text: str) -> str:
    return _WS.sub(" ", unicodedata.normalize("NFKC", text).casefold()).strip()


def collapse_whitespace(text: str) -> str:
    """Every run of whitespace, including line breaks, becomes one space.

    Model inputs use this so layout cannot stand in for authorship: public datasets kept
    paragraph breaks in AI text but stripped them from human text.
    """
    return _WS.sub(" ", text).strip()


# Pre-registration Amendment 5: curly quotes, dashes and ellipses come from the typing device or
# editor (smart quotes in Word, phones, chat UIs), not the author. v3's probe found them separating
# paired human and AI answers, so the model sees them as ASCII.
_TYPOGRAPHY = str.maketrans({
    "\u2018": "'", "\u2019": "'", "\u201a": "'", "\u201b": "'", "\u2032": "'",
    "\u201c": '"', "\u201d": '"', "\u201e": '"', "\u201f": '"', "\u2033": '"', "\u00ab": '"', "\u00bb": '"',
    "\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-", "\u2014": "-", "\u2015": "-", "\u2212": "-",
    "\u00a0": " ", "\u2007": " ", "\u2009": " ", "\u202f": " ", "\u2026": "...",
})


def normalize_typography(text: str) -> str:
    return text.translate(_TYPOGRAPHY)


# Amendment 6 (Ralph R08-2): markdown survived into the model's view, and asterisks marked 17% of
# Dolly's AI answers against 1% of its human ones, while Stack Exchange's converter had already
# flattened its human lists. Line-leading headings, quotes and list markers, emphasis markers and
# code ticks are dropped, so formatting can't stand in for authorship in any source.
_LINE_MARKUP = re.compile(r"^[ \t]*(?:#{1,6}[ \t]*|>+[ \t]?|(?:[-*+\u2022\u25aa\u25e6]|\d{1,3}[.)])[ \t]+)", re.M)
_RULE = re.compile(r"^[ \t]*(?:[-*_][ \t]*){3,}$", re.M)
_EMPHASIS = re.compile(r"\*+|(?<!\w)_+(?=\S)|(?<=\S)_+(?!\w)|`+")


def strip_formatting(text: str) -> str:
    return _EMPHASIS.sub("", _LINE_MARKUP.sub("", _RULE.sub("", text)))


# Amendment 7 (Ralph R09-1, R09-6): markers inside a line survived. A one-line dash list evaded the
# model, and a spaced hyphen marked HC3's human text. Bullets anywhere and free-standing dashes go;
# hyphens inside words stay.
_INLINE_MARKUP = re.compile(r"[\u2022\u2023\u2043\u25aa\u25e6]|(?<!\S)-{1,3}(?!\S)")


def strip_formatting_inline(text: str) -> str:
    return _INLINE_MARKUP.sub(" ", strip_formatting(text))


# Increment 4d (debts D14, D15): other bullet glyphs and (1)/1) markers survived format2, a bold number
# ("**1.**") needed two passes, and an unspaced em dash became a word hyphen while a spaced one was
# dropped. format3 drops every bullet glyph and short numbered marker, treats every en or em dash (and
# a free-standing "--") as a space before typography runs, and repeats the whole view until nothing
# changes, so one application is final: nested markers ("> > > quote", "1.\n2.\n3.") and markers behind
# other whitespace surface one pass at a time (final review I3).
_BULLETS3 = re.compile(r"[\u2022\u2023\u2043\u25aa\u25e6\u00b7\u2219\u25cf\u25cb\u25a0\u25a1\u25ab\u2027\u25b8\u25ba\u2713\u2714\u27a4]")
_NUMBERED3 = re.compile(r"(?<!\S)(?:\(\d{1,2}\)|\d{1,2}\))(?!\S)")
_DASHES3 = re.compile(r"[\u2013\u2014\u2015]|(?<![-\w])-{2,3}(?![-\w])")


MAX_VIEW_PASSES = 50


def strip_formatting3(text: str) -> str:
    # Repeated while lines still exist, so a marker uncovered by a pass (a bold "**2.**") goes at its line start.
    for _ in range(MAX_VIEW_PASSES):
        cleaned = _NUMBERED3.sub(" ", _BULLETS3.sub(" ", strip_formatting_inline(text)))
        if cleaned == text:
            break
        text = cleaned
    return text


def format3_view(text: str) -> str:
    # Dashes go before typography, which would otherwise turn en and em dashes into word hyphens. The
    # whole view repeats too: a marker behind other whitespace reaches a line start only once it collapses.
    for _ in range(MAX_VIEW_PASSES):
        cleaned = collapse_whitespace(strip_formatting3(normalize_typography(_DASHES3.sub(" ", text))))
        if cleaned == text:
            break
        text = cleaned
    return text


# Amendment 13 (Ralph R12-4): format3's closed lists missed other dash forms (small and two-em dashes),
# symbol bullets (diamonds, arrows, check marks), circled numbers and letter markers. format4 treats every
# Unicode dash but a word hyphen (the ASCII hyphen, U+2010, U+2011; the fullwidth and small hyphens, which
# the service reads back first) as a space; drops a symbol (category So or Sm) that opens a line before a
# space, circled and parenthesised numbers, and one-letter markers ("a)", "(b)") at a line start or "(b)"
# standing free. Emoji inside a sentence stay. The dashes are listed here, not read from the Unicode
# database, so the view can't move with a Python upgrade.
_DASHES4 = re.compile(r"[\u058a\u05be\u1400\u1806\u2012\u2013\u2014\u2015\u2e17\u2e1a\u2e3a\u2e3b\u2e40\u2e5d"
                      r"\u301c\u3030\u30a0\ufe31\ufe32\ufe58\U00010ead]|(?<![-\w])-{2,3}(?![-\w])")
_CIRCLED4 = re.compile(r"[\u2460-\u24b5\u24ea-\u24ff\u2776-\u2793\u3251-\u325f\u32b1-\u32bf\U0001f100-\U0001f10c]")
_LINE_SYMBOL4 = re.compile(r"^[ \t]*([^\x00-\x7f\s])[\ufe0f\u20e3]?[ \t]+", re.M)
_LETTER_MARKER4 = re.compile(r"^[ \t]*\(?[A-Za-z]\)[ \t]+", re.M)
_FREE_LETTER4 = re.compile(r"(?<!\S)\([A-Za-z]\)(?!\S)")


def _drop_symbol_bullet(match: re.Match) -> str:
    return "" if unicodedata.category(match.group(1)) in ("So", "Sm") else match.group(0)


def strip_formatting4(text: str) -> str:
    for _ in range(MAX_VIEW_PASSES):
        cleaned = _FREE_LETTER4.sub(" ", _CIRCLED4.sub(" ", strip_formatting3(text)))
        cleaned = _LETTER_MARKER4.sub("", _LINE_SYMBOL4.sub(_drop_symbol_bullet, cleaned))
        if cleaned == text:
            break
        text = cleaned
    return text


def format4_view(text: str) -> str:
    for _ in range(MAX_VIEW_PASSES):
        cleaned = collapse_whitespace(strip_formatting4(normalize_typography(_DASHES4.sub(" ", text))))
        if cleaned == text:
            break
        text = cleaned
    return text


# A view is recorded by name, and each name keeps its exact behaviour: a model is always read the way
# it was trained. A changed behaviour gets a new name (Ralph R09-3).
_TEXT_VIEWS = {
    "collapse_whitespace": collapse_whitespace,
    "typography+collapse_whitespace": lambda text: collapse_whitespace(normalize_typography(text)),
    "format+typography+collapse_whitespace":
        lambda text: collapse_whitespace(strip_formatting(normalize_typography(text))),
    "format2+typography+collapse_whitespace":
        lambda text: collapse_whitespace(strip_formatting_inline(normalize_typography(text))),
    "format3+typography+collapse_whitespace": format3_view,
    "format4+typography+collapse_whitespace": format4_view,
}

# The view current models train on; the shortcut probe reads the same view. The registered v5 models
# keep format2 by name through their registry record.
MODEL_TEXT_VIEW = "format4+typography+collapse_whitespace"

_FINGERPRINT_CASES = (
    "## Heading\n1. *One* item\n2) __two__ items\n- dash\n* star\n\u2022 bullet\n> quote `code`\n---\nsnake_case",
    "it\u2019s \u201cquoted\u201d \u2014 with\u00a0spaces\u2026 and 2\u20133 ranges",
    "Steps - mix -- bake \u2022 serve, a well-known trick\r\n\tend",
)


# Views added from Increment 4d on also fingerprint the cases their new rules cover; earlier views keep
# exactly the cases they were sealed with.
_EXTRA_FINGERPRINT_CASES = {
    "format3+typography+collapse_whitespace": ("\u00b7 dot (1) one 2) two tea\u2014then", "**1.** Bold\n**2.** Two"),
    "format4+typography+collapse_whitespace": ("\u00b7 dot (1) one 2) two tea\u2014then", "**1.** Bold\n**2.** Two",
                                               "tea\ufe58then \u2e3a x\n\u25c6 a\n\u2192 b\n\u2705 c\n\u2460 d (e) f\na) g",
                                               "well\u2010known \U0001f600 stays"),
}


def text_view_fingerprint(name) -> str:
    """SHA-256 of what a view does to fixed inputs: the registry checks it, so a view whose code
    changed behind an unchanged name can't serve a model calibrated on the old one (Ralph R09-3)."""
    view = text_view(name)
    cases = _FINGERPRINT_CASES + _EXTRA_FINGERPRINT_CASES.get(name, ())
    return hashlib.sha256("\x00".join(view(case) for case in cases).encode("utf-8")).hexdigest()


def text_view(name):
    """The model-input view a model was trained with, by its recorded name (None: raw text)."""
    if name is None:
        return lambda text: text
    if name not in _TEXT_VIEWS:
        raise ValueError(f"unknown text view {name!r}")
    return _TEXT_VIEWS[name]


def sample_id(text: str) -> str:
    return hashlib.sha256(normalize(text).encode("utf-8")).hexdigest()


def words(text: str) -> list[str]:
    return _WORD.findall(normalize(text))
