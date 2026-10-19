"""Sentence-aligned short excerpts of longer texts, for the short-text specialist.

An excerpt is a run of whole sentences from a text, 25 to 99 words long. Its length and starting
sentence come from a hash of the sample id, so the same text always yields the same excerpt, and
the rule never looks at the label. Every excerpt keeps its parent's split, source, and label, so
excerpts of a validation text stay validation data.

ponytail: D7 excerpts stand in for natural short texts until a verified short-text set exists.
"""

import hashlib
import re
from typing import Optional

import pandas as pd

from ..corpus.text import MODEL_TEXT_VIEW, collapse_whitespace, text_view

SHORT_MIN, SHORT_MAX = 25, 99
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[“‘A-Z0-9])")


def _draw(key: str, salt: str, n: int) -> int:
    return int(hashlib.sha256(f"{salt}:{key}".encode("utf-8")).hexdigest()[:12], 16) % n


def excerpt(text: str, key: str, lo: int = SHORT_MIN, hi: int = SHORT_MAX, view: str = MODEL_TEXT_VIEW) -> Optional[str]:
    """A lo..hi-word run of sentences from text, or None when the text is not longer than hi.

    The excerpt is cut from the model's view of the whole text: collapsing first would move a list's
    line-start markers inside a line, past the view's reach (Ralph R09-1)."""
    text = text_view(view)(text)
    if len(text.split()) <= hi:
        return None
    target = lo + _draw(key, "length", hi - lo + 1)
    sentences = [s.split() for s in _SENTENCE_END.split(text) if s.strip()]
    remaining = [sum(len(s) for s in sentences[i:]) for i in range(len(sentences))]
    starts = [i for i, r in enumerate(remaining) if r >= target] or [0]
    words: list = []
    for sentence in sentences[starts[_draw(key, "start", len(starts))]:]:
        if len(words) + len(sentence) > hi:
            if len(words) < lo:  # a very long sentence: cut it at the target length
                words = (words + sentence)[:target]
            break
        words += sentence
        if len(words) >= target:
            break
    return " ".join(words) if lo <= len(words) <= hi else None


def sentence_windows(text: str, length: int = 160, min_step: int = 100) -> list:
    """Windows of exactly `length` words that together cover the text.

    Each window starts at a sentence start, the way a long training text is read from its
    opening, and the next one starts at the last sentence start inside it (at least `min_step`
    words on), so consecutive windows overlap and nothing is skipped. The last window is the
    text's final `length` words. Every window has the same length, so all of them fall in the
    same calibrated band and are scored by the same model.
    """
    words = collapse_whitespace(text).split()
    n = len(words)
    if n <= length:
        return [" ".join(words)]
    starts, position = [], 0
    for sentence in _SENTENCE_END.split(" ".join(words)):
        starts.append(position)
        position += len(sentence.split())
    windows, begin = [], 0
    while True:
        windows.append(" ".join(words[begin:begin + length]))
        if begin + length >= n:
            return windows
        inside = [s for s in starts if begin + min_step < s <= begin + length]
        following = max(inside) if inside else begin + length - (length - min_step) // 2
        begin = n - length if n - following < min_step else following


def short_frame(df: pd.DataFrame, lo: int = SHORT_MIN, hi: int = SHORT_MAX,
                view: Optional[str] = MODEL_TEXT_VIEW) -> pd.DataFrame:
    """Natural lo..hi-word rows plus one excerpt of every longer row, flagged by `excerpt`, cut from view."""
    words = df["text"].str.split().str.len()
    natural = df[(words >= lo) & (words <= hi)].assign(excerpt=False)
    longer = df[words > hi]
    texts = [excerpt(t, k, lo, hi, view) for t, k in zip(longer["text"], longer["sample_id"])]
    derived = longer.assign(text=texts, excerpt=True).dropna(subset=["text"])
    derived = derived.assign(parent_id=derived["sample_id"], sample_id=derived["sample_id"] + ":x")
    return pd.concat([natural, derived], ignore_index=True)
