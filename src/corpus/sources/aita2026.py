"""AITA human-vs-AI (2026 generators): r/AmItheAsshole posts and same-title posts by seven 2026
models. The human side is redistributed only by AI2's Scruples, so it is rebuilt from there."""

import json
import tarfile
from pathlib import Path
from typing import Iterable

import pandas as pd

from ..schema import Label, LicenceClass, Role, Sample
from ._common import SourceSpec, compact, make_sample, take_per_stratum

_ANECDOTES = "anecdotes.tar.gz"
_SPLITS = ("train", "dev", "test")
_HUMAN_LICENCE = ("Reddit user content via AI2 Scruples (Apache-2.0 distribution)", LicenceClass.UNKNOWN)

SPEC = SourceSpec(
    name="aita2026",
    licence="Apache-2.0",
    licence_class=LicenceClass.PERMISSIVE,
    files=(("https://huggingface.co/datasets/mild-rgb/aita-human-vs-ai/resolve/main/dataset.jsonl", "dataset.jsonl"),
           ("https://storage.googleapis.com/ai2-mosaic-public/projects/scruples/v1.0/data/anecdotes.tar.gz",
            _ANECDOTES)),
)


def scruples_posts(raw: Path) -> dict:
    posts = {}
    with tarfile.open(raw / _ANECDOTES) as archive:
        for split in _SPLITS:
            for line in archive.extractfile(f"anecdotes/{split}.scruples-anecdotes.jsonl"):
                item = json.loads(line)
                posts[item["post_id"]] = item["text"]
    return posts


def load(raw: Path, cap: int) -> Iterable[Sample]:
    df = pd.read_json(raw / "dataset.jsonl", lines=True, dtype={"q_id": str})
    posts = scruples_posts(raw)
    human = df[df["label"] == "human"].drop(columns="text")
    human = human.assign(text=human["q_id"].map(posts)).dropna(subset=["text"])
    human = take_per_stratum(human, ["label"], cap, "doc_id", skip_used=True)
    ai = take_per_stratum(df[df["label"] == "ai"], ["generator"], cap, "doc_id", skip_used=True)
    yield from compact(
        make_sample(SPEC, text=r.text, label=Label.HUMAN, role=Role.CORE, record_id=r.doc_id,
                    domain="aita", group_key=r.q_id, licence=_HUMAN_LICENCE)
        for r in human.itertuples()
    )
    yield from compact(
        make_sample(SPEC, text=r.text, label=Label.AI, role=Role.CORE, record_id=r.doc_id,
                    domain="aita", group_key=r.q_id, generator=r.generator)
        for r in ai.itertuples()
    )
