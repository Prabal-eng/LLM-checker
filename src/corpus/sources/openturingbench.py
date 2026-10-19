"""OpenTuringBench (MLNTeam-Unical/OpenTuringBench, CC-BY-NC-4.0;
https://huggingface.co/datasets/MLNTeam-Unical/OpenTuringBench): news articles written from HuffPost
headlines by seven 2024 open models (Qwen 2.5, Llama 3.1, Phi-3.5, Mistral, SOLAR, neural-chat, Gemma 2)
(Increment 4d, for news from open models).

AI only: the human articles must be fetched from HuffPost URLs, so they are not used; the corpus balances
this source against the human news already in it, and the AI-only probe checks it. Only the in-domain
validation and test files are read (58,000 articles, more than the cap keeps); the 700 MB train file would
strain memory. Articles written from one headline share a group. Capped at PER_MODEL articles per model.
"""

import json
from pathlib import Path
from typing import Iterable

import pandas as pd

from ..schema import Label, LicenceClass, Role, Sample
from ._common import SourceSpec, compact, make_sample, take_per_stratum, trim_header, trim_preamble

_BASE = "https://huggingface.co/datasets/MLNTeam-Unical/OpenTuringBench/resolve/main/"
_FILES = ("in-domain/val.json", "in-domain/test.json")
PER_MODEL = 1500

SPEC = SourceSpec(
    name="openturingbench",
    licence="CC-BY-NC-4.0",
    licence_class=LicenceClass.NON_COMMERCIAL,
    files=tuple((f"{_BASE}{f}", f) for f in _FILES),
)


def load(raw: Path, cap: int) -> Iterable[Sample]:
    rows = []
    for rel in _FILES:
        with (raw / rel).open(encoding="utf-8") as fh:
            rows.extend({"url": r["url"], "text": r["content"], "generator": r["model"],
                         "key": f"{rel}:{r['url']}:{r['model']}"} for r in json.load(fh))
    df = take_per_stratum(pd.DataFrame(rows), ["generator"], min(cap, PER_MODEL), "key")
    yield from compact(
        make_sample(SPEC, text=trim_preamble(trim_header(r.text)), label=Label.AI, role=Role.CORE,  # Amendment 13
                    record_id=r.key, domain="news",
                    group_key=str(r.url), generator=r.generator,
                    verification="dataset label (model named by the authors)")
        for r in df.itertuples()
    )
