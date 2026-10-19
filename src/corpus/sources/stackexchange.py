"""Stack Exchange (HuggingFaceH4/stack-exchange-preferences): pre-ChatGPT human answers, paired
with local model answers to the same questions (Increment 4c).

v3 held these eight sites out as its locked test's unseen, verified-human, helpful-answer register;
v4 (Increment 4d) fits them, now that test is spent, and tests on eight new sites
(stackexchange_heldout), which share this selection. The human side is the accepted answer (else the top-voted one, score >= 2) to a question
dated 2021 or earlier, from eight non-code sites. Answers with code blocks or more than three links
are skipped, so code and URLs can't separate the labels.
"""

import html
import re
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable

import pandas as pd

from ..generate import load_all_generated
from ..schema import Label, LicenceClass, Role, Sample
from ._common import SourceSpec, compact, generated_licence, make_sample, strip_markdown, take_per_stratum

SITES = ("cooking", "gardening", "money", "parenting", "pets", "outdoors", "fitness", "interpersonal")
_BASE = "https://huggingface.co/datasets/HuggingFaceH4/stack-exchange-preferences/resolve/main/data/"
LATEST_QUESTION = "2021/12/31"
MIN_WORDS, MAX_WORDS, MAX_LINKS, MIN_SCORE = 25, 600, 3, 2
PER_SITE = 375

SPEC = SourceSpec(
    name="stackexchange",
    licence="CC-BY-SA-4.0",
    licence_class=LicenceClass.COPYLEFT,
    files=tuple((f"{_BASE}{s}.stackexchange.com/train-00000-of-00001.parquet",
                 f"{s}.stackexchange.com/train-00000-of-00001.parquet") for s in SITES),
)


class _Text(HTMLParser):
    """HTML to plain text: paragraphs and line breaks become newlines, list items get a dash."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list = []

    def handle_starttag(self, tag, attrs) -> None:
        if tag in ("p", "br", "div", "blockquote", "ul", "ol", "h1", "h2", "h3", "h4"):
            self.parts.append("\n")
        elif tag == "li":
            self.parts.append("\n- ")

    def handle_endtag(self, tag) -> None:
        if tag in ("p", "blockquote", "li", "h1", "h2", "h3", "h4"):
            self.parts.append("\n")

    def handle_data(self, data) -> None:
        self.parts.append(data)


def to_text(markup: str) -> str:
    parser = _Text()
    parser.feed(markup or "")
    text = html.unescape("".join(parser.parts))
    return re.sub(r"\n{3,}", "\n\n", re.sub(r"[ \t]+", " ", text)).strip()


def _best(answers) -> dict:
    ranked = sorted((a for a in answers if a.get("pm_score", 0) >= MIN_SCORE),
                    key=lambda a: (bool(a.get("selected")), a.get("pm_score", 0)), reverse=True)
    return ranked[0] if ranked else {}


def select_pairs(raw: Path, sites, per_site: int) -> pd.DataFrame:
    """Pre-ChatGPT questions with a well-scored, code-free best answer, up to per_site per site."""
    rows = []
    for site in sites:
        shards = sorted((raw / f"{site}.stackexchange.com").glob("train-*.parquet"))
        df = pd.concat([pd.read_parquet(path) for path in shards], ignore_index=True)
        df = df[df["date"] <= LATEST_QUESTION]
        for r in df.itertuples():
            best = _best(list(r.answers))
            markup = best.get("text", "")
            if not markup or "<pre" in markup or "<code" in markup or markup.count("<a ") > MAX_LINKS:
                continue
            answer, question = to_text(markup), to_text(r.question)
            if not (MIN_WORDS <= len(answer.split()) <= MAX_WORDS) or len(question.split()) < 10:
                continue
            rows.append({"item_id": f"{site}:{r.qid}", "site": site, "qid": str(r.qid), "date": r.date,
                         "prompt": question, "human_text": answer, "answer_id": str(best.get("answer_id"))})
    return take_per_stratum(pd.DataFrame(rows), ["site"], per_site, "item_id").reset_index(drop=True)


def pairs(raw: Path) -> pd.DataFrame:
    return select_pairs(raw, SITES, PER_SITE)


def samples(spec: SourceSpec, raw: Path, items: pd.DataFrame) -> Iterable[Sample]:
    """Human answers and their paired generated answers as samples of `spec` (shared by both Stack
    Exchange sources)."""
    yield from compact(
        make_sample(spec, text=strip_markdown(r.human_text), label=Label.HUMAN, role=Role.CORE,
                    record_id=f"{r.item_id}:{r.answer_id}", domain=r.site, group_key=r.item_id,
                    verification=f"pre-ChatGPT by question date ({r.date})")
        for r in items.itertuples()
    )
    generated = load_all_generated(raw).merge(items[["item_id", "site"]], on="item_id")
    yield from compact(
        make_sample(spec, text=strip_markdown(r.text), label=Label.AI, role=Role.CORE,
                    record_id=f"{r.item_id}:{r.generator}" + (f":{r.variant}" if r.variant else ""), domain=r.site, group_key=r.item_id,
                    generator=r.generator, verification="generated locally", transform=r.transform,
                    licence=generated_licence(r.model_licence))
        for r in generated.itertuples() if len(str(r.text).split()) >= 10
    )


def load(raw: Path, cap: int) -> Iterable[Sample]:
    yield from samples(SPEC, raw, pairs(raw))
