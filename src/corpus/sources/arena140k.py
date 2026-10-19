"""Arena Human Preference 140k (lmarena-ai/arena-human-preference-140k): anonymous side-by-side answers from
53 models collected April-July 2025, including Claude 4, Gemini 2.5, o3, GPT-4.1, Grok, DeepSeek, Llama 4
and Cohere Command A (Increment 4d). https://huggingface.co/datasets/lmarena-ai/arena-human-preference-140k

AI only: each English, non-code conversation gives the first assistant answer of each side. Prompts are
CC-BY-4.0; model outputs remain under their providers' terms, recorded as such (licence class unknown).
Sides from a held-out family are dropped. Both sides of a conversation share one group. Balanced in the
corpus against human assistant-style answers (Dolly, OpenAssistant, No Robots) and checked by the AI-only
probe. Capped at PER_MODEL answers per model.
"""

from pathlib import Path
from typing import Iterable

import pandas as pd

from ..schema import Label, LicenceClass, Role, Sample
from ..split import HOLDOUT_FAMILIES
from ._common import SourceSpec, compact, generator_family, make_sample, take_per_stratum

_BASE = "https://huggingface.co/datasets/lmarena-ai/arena-human-preference-140k/resolve/main/"
_FILES = tuple(f"data/train-0000{i}-of-00007.parquet" for i in range(7))
_COLUMNS = ["id", "model_a", "model_b", "language", "is_code", "conversation_a", "conversation_b"]
PER_MODEL = 400

SPEC = SourceSpec(
    name="arena140k",
    licence="prompts CC-BY-4.0; outputs under model providers' terms",
    licence_class=LicenceClass.UNKNOWN,
    files=tuple((f"{_BASE}{f}", f) for f in _FILES),
)


def first_answer(conversation) -> str:
    for turn in conversation:
        if turn["role"] == "assistant":
            content = turn["content"]
            if isinstance(content, str):
                return content
            return "\n".join(part["text"] for part in content if part.get("type") == "text" and part.get("text"))
    return ""


def load(raw: Path, cap: int) -> Iterable[Sample]:
    rows = []
    for rel in _FILES:
        path = raw / rel
        if not path.exists():
            continue
        df = pd.read_parquet(path, columns=_COLUMNS)
        df = df[(df["language"] == "en") & ~df["is_code"].astype(bool)]
        for r in df.itertuples():
            for side, model in (("a", r.model_a), ("b", r.model_b)):
                if generator_family(model) in HOLDOUT_FAMILIES:
                    continue
                rows.append({"key": f"{r.id}:{side}", "id": r.id, "generator": model,
                             "text": first_answer(getattr(r, f"conversation_{side}"))})
    if not rows:
        return
    df = take_per_stratum(pd.DataFrame(rows), ["generator"], min(cap, PER_MODEL), "key")
    yield from compact(
        make_sample(SPEC, text=r.text, label=Label.AI, role=Role.CORE, record_id=r.key, domain="chat",
                    group_key=str(r.id), generator=r.generator, verification="dataset label (model named by Arena)")
        for r in df.itertuples()
    )
