"""MIRAGE: human originals regenerated, polished, or rewritten by 17 recent LLMs."""

import hashlib
import json
from pathlib import Path
from typing import Iterable

import pandas as pd

from ..schema import Label, LicenceClass, Role, Sample
from ._common import SourceSpec, compact, make_sample, take_per_stratum, trim_preamble

_BASE = "https://raw.githubusercontent.com/fjc2005/DetectAnyLLM/main/data/MIRAGE_BENCH/"
_SETTINGS = ("DIG", "SIG")
_TASKS = {"generate": (Role.CORE, "none"), "polish": (Role.STRESS, "llm_polish"),
          "rewrite": (Role.STRESS, "llm_rewrite")}
# Amendment 5: the human Enron emails carry formatting damage their AI versions lack (words glued
# where line breaks were lost, document tags, headers); the probe separated that domain at 0.97.
EXCLUDED_DOMAINS = frozenset({"email"})

SPEC = SourceSpec(
    name="mirage",
    licence="Pi-Lab-1.0",
    licence_class=LicenceClass.NON_COMMERCIAL,
    files=tuple((f"{_BASE}{s}/{t}.json", f"{s}/{t}.json") for s in _SETTINGS for t in _TASKS),
)


def _records(raw: Path) -> pd.DataFrame:
    rows = []
    for setting in _SETTINGS:
        for task in _TASKS:
            with open(raw / setting / f"{task}.json", encoding="utf-8") as fh:
                for i, item in enumerate(json.load(fh)):
                    original = item.get("original") or ""
                    rows.append({
                        "record": f"{setting}/{task}:{i}", "task": task, "domain": item.get("domain", "unknown"),
                        "model": item.get("model"), "original": original, "rewritten": item.get("rewritten"),
                        "group": hashlib.sha256(original.encode("utf-8")).hexdigest()[:16],
                    })
    df = pd.DataFrame(rows).dropna(subset=["model", "rewritten"])
    return df[~df["domain"].isin(EXCLUDED_DOMAINS)]


def load(raw: Path, cap: int) -> Iterable[Sample]:
    df = _records(raw)
    human = take_per_stratum(df.drop_duplicates("group"), ["domain"], cap, "group", skip_used=True)
    generated = take_per_stratum(df[df["task"] == "generate"], ["domain", "model"], cap, "record", skip_used=True)
    revised = take_per_stratum(df[df["task"] != "generate"], ["task", "model"], cap, "record")

    yield from compact(
        make_sample(SPEC, text=trim_preamble(r.original), label=Label.HUMAN, role=Role.CORE,
                    record_id=f"{r.record}:original", domain=r.domain, group_key=r.group)
        for r in human.itertuples()
    )
    # Amendment 13 (R12-3): core rows lose a leading "Here's the article:"; stress rows keep their text.
    for frame in (generated, revised):
        yield from compact(
            make_sample(SPEC, text=trim_preamble(r.rewritten) if _TASKS[r.task][0] is Role.CORE else r.rewritten,
                        label=Label.AI, role=_TASKS[r.task][0],
                        record_id=r.record, domain=r.domain, group_key=r.group,
                        generator=r.model, transform=_TASKS[r.task][1])
            for r in frame.itertuples()
        )
