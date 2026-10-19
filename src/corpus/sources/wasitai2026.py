"""WasItAIGenerated AI-detector benchmark 2026-09: human text and answers from 2026 frontier models (GPT-6,
Claude Opus 5.5, Gemini 3.x, Grok 4.7, DeepSeek V4, Qwen 3.8), plus commercially humanized answers.

Held out whole (split.HOLDOUT_SOURCES) as v4's frontier test. Only English rows whose text is in the file
are used: rows the publisher could not redistribute are never re-fetched from their URLs, since the page
may have changed. Humanized rows are a report-only stress slice; the seven AI-edited rows need a labelling
policy and are skipped. Thirty humanized rows name no generator and are recorded as "unknown". Licences
are recorded per row (dataset card: licence "other", mixed per row).
https://huggingface.co/datasets/wasitaigeneratedcom/ai-detector-benchmark-2026-09
"""

from pathlib import Path
from typing import Iterable

import pandas as pd

from ..schema import Label, LicenceClass, Role, Sample
from ._common import SourceSpec, compact, make_sample

_FILE = "data/benchmark.parquet"
SPEC = SourceSpec(
    name="wasitai2026",
    licence="mixed per row (dataset card: other)",
    licence_class=LicenceClass.UNKNOWN,
    files=((f"https://huggingface.co/datasets/wasitaigeneratedcom/ai-detector-benchmark-2026-09/resolve/main/{_FILE}",
            _FILE),),
)


def row_licence(name) -> tuple:
    name = name if isinstance(name, str) and name.strip() else "unknown"
    parts = name.lower().split("-")
    if "nc" in parts:
        return name, LicenceClass.NON_COMMERCIAL
    if name.lower().startswith(("cc-by-4.0", "mit", "apache")):
        return name, LicenceClass.PERMISSIVE
    return name, LicenceClass.UNKNOWN


def load(raw: Path, cap: int) -> Iterable[Sample]:
    df = pd.read_parquet(raw / _FILE)
    df = df[(df["lang"] == "en") & df["text"].notna() & df["label"].isin(["human", "ai", "humanized"])]
    yield from compact(
        make_sample(SPEC, text=r.text, label=Label.HUMAN if r.label == "human" else Label.AI,
                    role=Role.STRESS if r.label == "humanized" else Role.CORE, record_id=str(r.id),
                    domain=str(r.source), group_key=str(r.group),
                    generator=(r.generator if isinstance(r.generator, str) else "unknown") if r.label != "human" else None,
                    transform=f"humanizer:{r.humanizer}" if r.label == "humanized" else "none",
                    verification="dataset label (publisher)", licence=row_licence(r.text_license))
        for r in df.itertuples()
    )
