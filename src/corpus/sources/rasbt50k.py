"""rasbt/human-vs-ai-50k: openly licensed human text and 2026-model texts written from the same
source documents (Wikipedia, StackExchange, arXiv, PLOS, PMC, Gutenberg, OpenStax, blogs).

Diagnostic only: never fitted, never gating. The v2 build's shortcut probe separated its labels
from surface features alone (AUC 0.98). Human rows are fixed-length chunks cut mid-sentence (23%
end a sentence, against 99.7% of AI rows), code is stripped from the StackExchange answers, and
outside StackExchange the AI text answers a question about the document instead of matching its
genre. Its AI rows still show how often current (2026) models are caught.
"""

from pathlib import Path
from typing import Iterable

import pandas as pd

from ..schema import Label, LicenceClass, Role, Sample
from ._common import SourceSpec, compact, make_sample, take_per_stratum

_BASE = "https://huggingface.co/datasets/rasbt/human-vs-ai-50k/resolve/main/data/"
_FILES = tuple(f"{s}-00000-of-00001.parquet" for s in ("train", "validation", "test"))
_CLASSES = {"CC0 1.0": LicenceClass.PERMISSIVE, "CC BY 4.0": LicenceClass.PERMISSIVE,
            "Public domain in the United States": LicenceClass.PERMISSIVE}
_AI_LICENCE = ("rasbt/human-vs-ai-50k dataset licence: other", LicenceClass.UNKNOWN)

SPEC = SourceSpec(
    name="rasbt50k",
    licence="per-row source licence (CC0, CC BY, CC BY-SA, public domain)",
    licence_class=LicenceClass.COPYLEFT,
    files=tuple((f"{_BASE}{f}", f) for f in _FILES),
)


def generator_name(provider: str, model: str) -> str:
    # OpenAI's models here carry codenames only, so the provider names the family. Other providers
    # are hosts (ollama, openrouter) whose names would mislead the family patterns.
    return f"openai/{model}" if provider == "openai" else model


def load(raw: Path, cap: int) -> Iterable[Sample]:
    df = pd.concat([pd.read_parquet(raw / f) for f in _FILES], ignore_index=True)
    # Seven texts were repaired by hand after generation; they are neither purely model nor human.
    df = df[df["generator_provider"].fillna("") != "codex-manual"]
    human = take_per_stratum(df[df["label"] == 0], ["source_collection"], cap, "id")
    ai = df[df["label"] == 1]
    ai = ai.assign(generator=[generator_name(p, m) for p, m in zip(ai["generator_provider"], ai["generator_model"])])
    ai = take_per_stratum(ai, ["source_collection", "generator"], cap, "id")
    yield from compact(
        make_sample(SPEC, text=r.text, label=Label.HUMAN, role=Role.DIAGNOSTIC, record_id=str(r.id),
                    domain=r.source_collection, group_key=r.group_id,
                    licence=(r.source_license, _CLASSES.get(r.source_license, LicenceClass.COPYLEFT)))
        for r in human.itertuples()
    )
    yield from compact(
        make_sample(SPEC, text=r.text, label=Label.AI, role=Role.DIAGNOSTIC, record_id=str(r.id),
                    domain=r.source_collection, group_key=r.group_id, generator=r.generator,
                    licence=_AI_LICENCE)
        for r in ai.itertuples()
    )
