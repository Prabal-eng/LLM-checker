"""HC3 (English): human and ChatGPT answers to the same questions in five domains."""

import hashlib
import json
from pathlib import Path
from typing import Iterable

import pandas as pd

from ..schema import Label, LicenceClass, Role, Sample
from ._common import SourceSpec, compact, make_sample, take_per_stratum

_BASE = "https://huggingface.co/datasets/Hello-SimpleAI/HC3/resolve/main/"
_DOMAINS = ("finance", "medicine", "open_qa", "reddit_eli5", "wiki_csai")

SPEC = SourceSpec(
    name="hc3",
    licence="CC-BY-SA-4.0",
    licence_class=LicenceClass.COPYLEFT,
    files=tuple((f"{_BASE}{d}.jsonl", f"{d}.jsonl") for d in _DOMAINS),
)


def _rows(raw: Path) -> pd.DataFrame:
    records = []
    for domain in _DOMAINS:
        with open(raw / f"{domain}.jsonl", encoding="utf-8") as fh:
            for line in fh:
                item = json.loads(line)
                question = hashlib.sha256(item["question"].encode("utf-8")).hexdigest()[:16]
                for column, label in (("human_answers", Label.HUMAN), ("chatgpt_answers", Label.AI)):
                    for i, answer in enumerate(item.get(column) or []):
                        records.append({
                            "domain": domain, "question": question, "label": label,
                            "record": f"{domain}:{question}:{column}:{i}", "text": answer,
                        })
    return pd.DataFrame(records)


def load(raw: Path, cap: int) -> Iterable[Sample]:
    df = take_per_stratum(_rows(raw), ["domain", "label"], cap, "record", skip_used=True)
    yield from compact(
        make_sample(
            SPEC,
            text=r.text,
            label=r.label,
            role=Role.CORE,
            record_id=r.record,
            domain=r.domain,
            group_key=f"{r.domain}:{r.question}",
            generator="chatgpt" if r.label == Label.AI else None,
        )
        for r in df.itertuples()
    )
