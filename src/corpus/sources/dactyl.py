"""DACTYL (ShantanuT01/DACTYL; card: MIT, paper: CC-BY-4.0; https://huggingface.co/datasets/ShantanuT01/DACTYL):
human tweets, reviews, abstracts, student essays and news, with few-shot answers from eleven 2024 models
and two fine-tuned Llama 3.2 1B models (Increment 4d, for short texts).

RedditWritingPrompts rows are dropped because MAGE already holds WritingPrompts. AI rows are grouped by
their prompt, human rows by id; rows are length-matched per domain before capping so length can't stand
in for authorship.
"""

import hashlib
from pathlib import Path
from typing import Iterable

import pandas as pd

from ..schema import Label, LicenceClass, Role, Sample
from ._common import SourceSpec, compact, length_matched, make_sample

_BASE = "https://huggingface.co/datasets/ShantanuT01/DACTYL/resolve/main/"
_FILES = ("train.parquet", "validation.parquet", "test.parquet")
_COLUMNS = ["id", "text", "domain", "model", "target", "prompt"]
DROP_DOMAINS = frozenset({"RedditWritingPrompts"})
PER_CELL = 2000

SPEC = SourceSpec(
    name="dactyl",
    licence="MIT (dataset card; the paper says CC-BY-4.0)",
    licence_class=LicenceClass.PERMISSIVE,
    files=tuple((f"{_BASE}{f}", f) for f in _FILES),
)


def _group(row) -> str:
    if row.target == 1 and isinstance(row.prompt, str):
        return "p:" + hashlib.sha256(row.prompt.encode("utf-8")).hexdigest()[:16]
    return f"h:{row.id}"


def load(raw: Path, cap: int) -> Iterable[Sample]:
    df = pd.concat([pd.read_parquet(raw / f, columns=_COLUMNS) for f in _FILES], ignore_index=True)
    df = df[~df["domain"].isin(DROP_DOMAINS)].drop_duplicates("id")
    df = length_matched(df, ["domain"], "target", "text", key="id", cap=min(cap, PER_CELL))
    yield from compact(
        make_sample(SPEC, text=r.text, label=Label.AI if r.target == 1 else Label.HUMAN, role=Role.CORE,
                    record_id=str(r.id), domain=str(r.domain).lower(), group_key=_group(r),
                    generator=r.model if r.target == 1 else None)
        for r in df.itertuples()
    )
