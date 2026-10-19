"""Trigger sentences and words: what drives a high AI score (spec 2026-10-01, section 2; words 2026-10-02).

The model scores a whole text, so nothing inside it points at a sentence or a word. This module removes one
sentence, then one word, at a time, re-scores the text with the model that scored the row, and reports what takes
away the largest share of the row's AI signal. The AI signal is the score's log-odds: how far the text sits above
50%. Near the top of the scale a percent barely moves (99% -> 97% is a quarter of the signal), so shares are taken
in log-odds, where it still does. It explains a score; it never changes a decision. Indicative only: how well these
explanations match what a careful reader would pick has not been measured.
"""

import logging
import math
import re
from dataclasses import dataclass, field, replace
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from ..authorship.detector import WINDOW_WORDS, LengthRoutedDetector
from ..detection.obfuscation import deobfuscate
from ..detectors import GPTZeroDetector, OriginalityDetector

logger = logging.getLogger(__name__)

JOIN = "\n\n"  # how the pipeline joins a row's cells into the row's combined text
# A sentence ends at . ! ? or an ellipsis (with any closing quotes or brackets) when whitespace and an
# uppercase letter, digit or opening quote follow; a line break always ends one.
_END = re.compile(r"[.!?…]+[\"'”’)\]]*(?=\s+[\"'(\[“‘A-Z0-9])")
# A word: letters or digits, with inner apostrophes or hyphens (it's, well-known); punctuation stays behind.
_WORD = re.compile(r"[^\W_]+(?:['’\-][^\W_]+)*")
FOCUS_SENTENCES = 3   # words are tried in the sentences that take away the most signal
WORD_DROPS_KEPT = 10  # the tried words kept for the Technical sheet
# Words too common to act on: tried like any other, never shown as a trigger on their own.
STOPWORDS = frozenset("""
a an the and or but nor so of to in on at for with by from as into than then is are was were be been being am
do does did has have had it its it's this that these those there here i me my we our you your he him his she her
they them their not no
""".split())


@dataclass(frozen=True)
class Sentence:
    column: str      # the cell it came from
    start: int       # character offsets in that cell's (stripped) text
    end: int
    text: str
    word_start: int  # words before it in the row's decided text


@dataclass(frozen=True)
class Phrase:
    column: str      # the cell it came from
    start: int       # character offsets in that cell's (stripped) text
    end: int
    text: str
    share: float     # the share of the row's AI signal that goes away without it


def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


def pct(share: float) -> str:
    """A share of the AI signal for display: removing something can take away more than all of it."""
    return f"{min(max(share, 0.0), 1.0):.0%}"


def _spans(text: str) -> List[Tuple[int, int]]:
    spans = []
    for line in re.finditer(r"[^\n]+", text):
        base, body, begin = line.start(), line.group(0), 0
        for cut in [m.end() for m in _END.finditer(body)] + [len(body)]:
            piece = body[begin:cut]
            if piece.strip():
                lead = len(piece) - len(piece.lstrip())
                spans.append((base + begin + lead, base + begin + len(piece.rstrip())))
            begin = cut
    return spans


def split_sentences(segments: Sequence[Tuple[str, str]]) -> List[Sentence]:
    """The sentences of a row's decided text, in order, each mapped back to its cell."""
    out, words = [], 0
    for column, text in segments:
        for start, end in _spans(text or ""):
            piece = text[start:end]
            out.append(Sentence(column, start, end, piece, words))
            words += len(piece.split())
    return out


def without(segments: Sequence[Tuple[str, str]], span) -> str:
    """The decided text with one sentence or phrase removed, cells joined as the pipeline joins them."""
    cells = []
    for column, text in segments:
        if column == span.column:
            text = (text[:span.start] + " " + text[span.end:]).strip()
        if text:
            cells.append(text)
    return JOIN.join(cells)


def only_common_words(text: str) -> bool:
    return all(w.lower() in STOPWORDS for w in _WORD.findall(text))


@dataclass
class TriggerResult:
    kind: str                                   # "sentences", "spread", "single", "skipped" or "error"
    baseline: Optional[float] = None            # the decided text's score, re-scored here
    sentences: List[Tuple[Sentence, float]] = field(default_factory=list)  # triggers, with their shares
    drops: List[Tuple[Sentence, float]] = field(default_factory=list)      # every sentence tried, with its share
    note: str = ""
    signal: Optional[float] = None              # the decided text's log-odds
    words: List[Phrase] = field(default_factory=list)       # trigger words and phrases, largest share first
    word_drops: List[Phrase] = field(default_factory=list)  # the words tried with the largest shares

    def after(self, share: float) -> float:
        """The score once something taking away `share` of the AI signal is removed."""
        return _sigmoid(self.signal * (1.0 - share))

    def words_text(self) -> str:
        return "Words: " + ", ".join(f'"{w.text}" ({pct(w.share)})' for w in self.words)

    def summary(self) -> str:
        if self.kind == "sentences":
            parts = [f"{self.baseline:.0%} -> {self.after(share):.0%} ({pct(share)} of the AI signal)"
                     for _, share in self.sentences]
            text = "Without it: " + parts[0] if len(parts) == 1 else (
                "Without the first: " + parts[0] + "; without the second: " + parts[1])
            return text + ("\n" + self.words_text() if self.words else "")
        if self.kind in ("spread", "single"):
            if self.words:
                return self.words_text()
            return ("No single sentence or word stands out: the AI signal is spread across the text"
                    if self.kind == "spread" else "One sentence, and no single word stands out")
        return self.note


def scorer_for(detector, text: str) -> Optional[Callable[[List[str]], np.ndarray]]:
    """A function giving texts' log-odds (the AI signal) with the model that scored `text` (its deobfuscated
    form): the routed detector's short or main model, each through its own probability map in one truncated pass
    per text, as calibration scored. Other local detectors give the log-odds of their scores. None for paid APIs."""
    if isinstance(detector, (GPTZeroDetector, OriginalityDetector)):
        return None
    if isinstance(detector, LengthRoutedDetector):
        detector = detector.short if detector.serves_short(text) else detector.main
    if hasattr(detector, "raw_and_embeddings") and hasattr(detector, "platt"):
        def score(texts: List[str]) -> np.ndarray:
            raw, _ = detector.raw_and_embeddings(list(texts))
            a, b = detector.platt  # calibration.probability is sigmoid(a * raw + b)
            return a * np.asarray(raw, dtype=float) + b
        return score

    def score_other(texts: List[str]) -> np.ndarray:
        p = np.clip(np.array([r.ai_score for r in detector.detect_batch(list(texts))], dtype=float), 1e-6, 1 - 1e-6)
        return np.log(p / (1.0 - p))
    return score_other


class TriggerFinder:
    def __init__(self, detector, min_share: float = 0.20, max_sentences: int = 12, top: int = 2,
                 words: bool = True, max_words: int = 60, top_words: int = 5, min_word_share: float = 0.05,
                 window_words: int = WINDOW_WORDS) -> None:
        self.detector, self.min_share, self.max_sentences, self.top = detector, min_share, max_sentences, top
        self.words, self.max_words, self.top_words, self.min_word_share = words, max_words, top_words, min_word_share
        self.window_words = window_words

    def explain(self, segments: Sequence[Tuple[str, str]]) -> TriggerResult:
        segments = [(c, t) for c, t in segments if t and t.strip()]
        sentences = split_sentences(segments)
        if not sentences:
            return TriggerResult("skipped", note="not run (no text)")
        full = deobfuscate(JOIN.join(t for _, t in segments)).text
        scorer = scorer_for(self.detector, full)
        if scorer is None:
            return TriggerResult("skipped", note="not run (paid API)")
        many = len(sentences) > 1
        # Text past the model's window never reached the score, so only sentences starting inside it are tried.
        tried = [s for s in sentences if s.word_start < self.window_words][: self.max_sentences] if many else []
        scores = scorer([full] + [deobfuscate(without(segments, s)).text for s in tried])
        signal = float(scores[0])
        if signal <= 0:
            return TriggerResult("skipped", baseline=_sigmoid(signal), signal=signal,
                                 note="not run (re-scored at 50% or under)")
        drops = [(s, (signal - float(v)) / signal) for s, v in zip(tried, scores[1:])]
        ranked = sorted((d for d in drops if d[1] >= self.min_share), key=lambda d: -d[1])[: self.top]
        result = TriggerResult(("sentences" if ranked else "spread") if many else "single",
                               baseline=_sigmoid(signal), signal=signal, sentences=ranked, drops=drops)
        if self.words:
            focus = [s for s, _ in sorted(drops, key=lambda d: -d[1])[:FOCUS_SENTENCES]] if many else sentences
            result.words, result.word_drops = self._words(segments, focus, scorer, signal)
        return result

    def _words(self, segments, focus: List[Sentence], scorer, signal: float) -> Tuple[List[Phrase], List[Phrase]]:
        """Each word of the focus sentences removed in turn; neighbouring words that each pass the bar are joined
        into one phrase and re-scored as a whole. Returns (shown, the tried words with the largest shares)."""
        tokens = []
        for s in focus:
            for m in _WORD.finditer(s.text):
                if s.word_start + len(s.text[:m.start()].split()) < self.window_words:
                    tokens.append(Phrase(s.column, s.start + m.start(), s.start + m.end(), m.group(0), 0.0))
        tokens = tokens[: self.max_words]
        if not tokens:
            return [], []
        scores = scorer([deobfuscate(without(segments, t)).text for t in tokens])
        tokens = [replace(t, share=(signal - float(v)) / signal) for t, v in zip(tokens, scores)]
        texts, order = dict(segments), {c: i for i, (c, _) in enumerate(segments)}
        groups: List[List[Phrase]] = []
        for t in sorted((t for t in tokens if t.share >= self.min_word_share), key=lambda t: (order[t.column], t.start)):
            last = groups[-1][-1] if groups else None
            if last is not None and last.column == t.column and not texts[t.column][last.end:t.start].strip():
                groups[-1].append(t)
            else:
                groups.append([t])
        phrases = [g[0] if len(g) == 1 else
                   Phrase(g[0].column, g[0].start, g[-1].end, texts[g[0].column][g[0].start:g[-1].end], 0.0)
                   for g in groups]
        joined = [i for i, g in enumerate(groups) if len(g) > 1]
        if joined:
            again = scorer([deobfuscate(without(segments, phrases[i])).text for i in joined])
            for i, v in zip(joined, again):
                phrases[i] = replace(phrases[i], share=(signal - float(v)) / signal)
        shown = sorted((p for p in phrases if p.share >= self.min_word_share and not only_common_words(p.text)),
                       key=lambda p: -p.share)[: self.top_words]
        return shown, sorted(tokens, key=lambda t: -t.share)[:WORD_DROPS_KEPT]
