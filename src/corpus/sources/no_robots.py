"""No Robots: instruction answers written by paid annotators in an assistant's voice.

Human-only. Prompts that Beemo reuses are left to Beemo, which pairs them with model answers.
"""

from pathlib import Path
from typing import Iterable

import pandas as pd

from ..schema import Label, LicenceClass, Role, Sample
from ._common import SourceSpec, compact, make_sample, take_per_stratum

_BASE = "https://huggingface.co/datasets/HuggingFaceH4/no_robots/resolve/main/data/"
_FILES = ("train-00000-of-00001.parquet", "test-00000-of-00001.parquet")
_BEEMO = "beemo.parquet"
PER_CATEGORY = 600

SPEC = SourceSpec(
    name="no_robots",
    licence="CC-BY-NC-4.0",
    licence_class=LicenceClass.NON_COMMERCIAL,
    files=tuple((f"{_BASE}{f}", f) for f in _FILES)
    + (("https://huggingface.co/datasets/toloka/beemo/resolve/main/data/train-00000-of-00001.parquet", _BEEMO),),
)


def first_answer(messages) -> str:
    return next((m["content"] for m in messages if m["role"] == "assistant"), "")


def load(raw: Path, cap: int) -> Iterable[Sample]:
    df = pd.concat([pd.read_parquet(raw / f) for f in _FILES], ignore_index=True)
    beemo = set(pd.read_parquet(raw / _BEEMO, columns=["prompt_id"])["prompt_id"])
    df = df[~df["prompt_id"].isin(beemo)]
    df = df.assign(text=df["messages"].map(first_answer))
    df = take_per_stratum(df, ["category"], min(cap, PER_CATEGORY), "prompt_id", skip_used=True)
    yield from compact(
        make_sample(SPEC, text=r.text, label=Label.HUMAN, role=Role.CORE, record_id=r.prompt_id,
                    domain=r.category.lower().replace(" ", "_"), group_key=r.prompt_id)
        for r in df.itertuples()
    )
