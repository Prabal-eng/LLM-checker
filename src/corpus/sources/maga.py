"""MAGA-Bench, English (anyangsong/MAGA, MIT; https://huggingface.co/datasets/anyangsong/MAGA): human texts
from ten web registers (Reddit, Yahoo Answers, Natural Questions, Amazon and Trustpilot reviews, CC News,
NPR, S2ORC, Wikipedia, wikiHow) paired by title with answers from twelve 2024-25 models (Increment 4d).

Two configurations share the same 60,000 human texts: MGB is plain generation and MAGA is "aligned"
generation built to read as human (role play, self-refinement, adversarial training); aligned answers are
tagged transform "maga_aligned". The extra/ configurations stay out unless INCLUDE_EXTRA is set. wikiHow
rows are dropped (the source was withdrawn after a DMCA notice). Human texts run far shorter than the AI
ones, so rows are length-matched per domain before capping.
"""

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from ..schema import Label, LicenceClass, Role, Sample
from ._common import SourceSpec, compact, length_matched, make_sample, trim_preamble

_BASE = "https://huggingface.co/datasets/anyangsong/MAGA/resolve/main/"
_FILES = {"MGB": ("train/MGB_train.jsonl", "val/MGB_val.jsonl"),
          "MAGA": ("train/MAGA_train.jsonl", "val/MAGA_val.jsonl")}
_EXTRA = {f"MAGA_extra_{m}": (f"extra/train/MAGA_extra_train_{m}.jsonl", f"extra/val/MAGA_extra_val_{m}.jsonl")
          for m in ("BPO", "GDARL_CDMMR", "role_playing", "self_refine")}
INCLUDE_EXTRA = False
DROP_DOMAINS = frozenset({"wikiHow"})
PER_DOMAIN = 1500


def _configs() -> dict:
    return _FILES | (_EXTRA if INCLUDE_EXTRA else {})


SPEC = SourceSpec(
    name="maga",
    licence="MIT",
    licence_class=LicenceClass.PERMISSIVE,
    files=tuple((f"{_BASE}{rel}", rel) for files in _configs().values() for rel in files),
)


def _rows(raw: Path) -> pd.DataFrame:
    rows = []
    for config, files in _configs().items():
        for rel in files:
            with (raw / rel).open(encoding="utf-8") as fh:
                rows.extend({**json.loads(line), "config": config} for line in fh if line.strip())
    return pd.DataFrame(rows)


def load(raw: Path, cap: int) -> Iterable[Sample]:
    df = _rows(raw)
    df = df[~df["domain"].isin(DROP_DOMAINS)]
    human = df["model"] == "human"
    df = pd.concat([df[human].drop_duplicates("id"), df[~human]], ignore_index=True)
    df["y"] = np.where(df["model"] == "human", "human", "ai")
    df["text"] = df["text"].map(lambda t: trim_preamble(str(t)))  # Amendment 13 (R12-3), before matching
    df = length_matched(df, ["domain"], "y", "text", key="id", cap=min(cap, PER_DOMAIN))
    yield from compact(
        make_sample(SPEC, text=r.text, label=Label.HUMAN if r.y == "human" else Label.AI, role=Role.CORE,
                    record_id=str(r.id), domain=r.domain.lower().replace(" ", "_"), group_key=str(r.human_source_id),
                    generator=r.model if r.y == "ai" else None,
                    transform="maga_aligned" if r.y == "ai" and r.config != "MGB" else "none")
        for r in df.itertuples()
    )
