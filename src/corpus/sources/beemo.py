"""Beemo: human answers, LLM answers, and expert- or LLM-edited LLM answers to the same prompts."""

import ast
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from ..schema import Label, LicenceClass, Role, Sample
from ._common import SourceSpec, compact, make_sample

_FILE = "train-00000-of-00001.parquet"
_EDITORS = {"llama-3.1-70b_edits": "llama-3.1-70b-instruct", "gpt-4o_edits": "gpt-4o"}

SPEC = SourceSpec(
    name="beemo",
    licence="MIT",
    licence_class=LicenceClass.PERMISSIVE,
    files=((f"https://huggingface.co/datasets/toloka/beemo/resolve/main/data/{_FILE}", _FILE),),
)


def _edit_variants(value: Optional[str]) -> list[tuple[str, str]]:
    """Parse ``[{'P1': text}, {'P2': text}, ...]`` into (prompt_name, text) pairs."""
    if not value:
        return []
    try:
        parsed = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return []
    return [(k, v) for item in parsed if isinstance(item, dict) for k, v in item.items() if isinstance(v, str)]


def load(raw: Path, cap: int) -> Iterable[Sample]:
    df = pd.read_parquet(raw / _FILE)

    def rows() -> Iterable[Optional[Sample]]:
        for r in df.to_dict("records"):
            common = dict(domain=r["category"].lower().replace(" ", "_"), group_key=r["prompt_id"])
            yield make_sample(SPEC, text=r["human_output"], label=Label.HUMAN, role=Role.CORE,
                              record_id=f"{r['id']}:human", **common)
            yield make_sample(SPEC, text=r["model_output"], label=Label.AI, role=Role.CORE,
                              record_id=f"{r['id']}:model", generator=r["model"], **common)
            yield make_sample(SPEC, text=r["human_edits"], label=Label.AI, role=Role.STRESS,
                              record_id=f"{r['id']}:human_edits", generator=r["model"],
                              transform="human_edit", **common)
            for column, editor in _EDITORS.items():
                for prompt_name, text in _edit_variants(r.get(column)):
                    yield make_sample(SPEC, text=text, label=Label.AI, role=Role.STRESS,
                                      record_id=f"{r['id']}:{column}:{prompt_name}", generator=editor,
                                      transform="llm_humanize", **common)

    yield from compact(rows())
