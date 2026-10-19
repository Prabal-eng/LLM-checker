"""databricks-dolly-15k: instruction answers written by Databricks employees (2023), paired with
local model answers to the same prompts (Increment 4c).

v2 held Dolly out whole as unseen assistant-style human text; v2's locked test is retired, so v3
fits it, now with AI answers to the same prompts so assistant voice alone can't mean AI. Answers
under 25 words fall below the checker's support floor and are skipped.
"""

import hashlib
from pathlib import Path
from typing import Iterable

import pandas as pd

from ..generate import load_all_generated
from ..schema import Label, LicenceClass, Role, Sample
from ._common import SourceSpec, compact, generated_licence, make_sample, strip_markdown, take_per_stratum

_FILE = "databricks-dolly-15k.jsonl"
MIN_WORDS = 25
PER_CATEGORY = 500

SPEC = SourceSpec(
    name="dolly",
    licence="CC-BY-SA-3.0",
    licence_class=LicenceClass.COPYLEFT,
    files=((f"https://huggingface.co/datasets/databricks/databricks-dolly-15k/resolve/main/{_FILE}", _FILE),),
)


def pairs(raw: Path) -> pd.DataFrame:
    df = pd.read_json(raw / _FILE, lines=True).reset_index(names="row")
    df = df[df["response"].str.split().str.len() >= MIN_WORDS]
    df = df.assign(group=df["instruction"].map(lambda t: hashlib.sha256(t.encode("utf-8")).hexdigest()[:16]),
                   item_id=df["row"].map(lambda r: f"dolly:{r}"),
                   prompt=[f"{i}\n\n{c}".strip() if c else i for i, c in zip(df["instruction"], df["context"])],
                   human_text=df["response"])
    return take_per_stratum(df, ["category"], PER_CATEGORY, "row").reset_index(drop=True)


def load(raw: Path, cap: int) -> Iterable[Sample]:
    items = pairs(raw)
    yield from compact(
        make_sample(SPEC, text=strip_markdown(r.human_text), label=Label.HUMAN, role=Role.CORE,
                    record_id=str(r.row), domain=r.category, group_key=r.group)
        for r in items.itertuples()
    )
    generated = load_all_generated(raw).merge(items[["item_id", "category", "group"]], on="item_id")
    yield from compact(
        make_sample(SPEC, text=strip_markdown(r.text), label=Label.AI, role=Role.CORE,
                    record_id=f"{r.item_id}:{r.generator}" + (f":{r.variant}" if r.variant else ""), domain=r.category, group_key=r.group,
                    generator=r.generator, verification="generated locally", transform=r.transform,
                    licence=generated_licence(r.model_licence))
        for r in generated.itertuples() if len(str(r.text).split()) >= 10
    )
