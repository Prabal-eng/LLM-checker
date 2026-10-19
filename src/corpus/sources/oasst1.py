"""OpenAssistant oasst1: first-turn assistant replies written by volunteers (2023), English only,
paired with local model answers to the same prompts (Increment 4c).

v2 held oasst1 out whole; v2's locked test is retired, so v3 fits it. Kept replies passed community
review, were not deleted, and are not synthetic. Volunteers role-played the assistant, and some
replies say so ("as an AI language model"); those cannot be told apart from pasted model output,
so replies with such giveaways are left out of the human side.
"""

import re
from pathlib import Path
from typing import Iterable

import pandas as pd

from ..generate import load_all_generated
from ..schema import Label, LicenceClass, Role, Sample
from ._common import SourceSpec, compact, generated_licence, make_sample, strip_markdown, take_per_stratum

_BASE = "https://huggingface.co/datasets/OpenAssistant/oasst1/resolve/main/data/"
_FILES = ("train-00000-of-00001-b42a775f407cee45.parquet", "validation-00000-of-00001-134b8fd0c89408b6.parquet")
MIN_WORDS = 25
AI_VOICE = re.compile(r"\bas an ai\b|\bas a language model\b|\bai language model\b|"
                      r"\bi(?:'m| am) an? (?:ai|artificial intelligence|language model)\b|\bopenai\b|\bchatgpt\b",
                      re.IGNORECASE)

SPEC = SourceSpec(
    name="oasst1",
    licence="Apache-2.0",
    licence_class=LicenceClass.PERMISSIVE,
    files=tuple((f"{_BASE}{f}", f) for f in _FILES),
)


def pairs(raw: Path) -> pd.DataFrame:
    df = pd.concat([pd.read_parquet(raw / f) for f in _FILES], ignore_index=True)
    roots = df[df["parent_id"].isna() & (df["role"] == "prompter") & (df["lang"] == "en")]
    roots = roots.set_index("message_id")["text"]
    reviewed = df["review_result"].fillna(False).astype(bool)
    keep = ((df["role"] == "assistant") & (df["lang"] == "en") & ~df["deleted"].astype(bool) & reviewed
            & ~df["synthetic"].astype(bool) & df["parent_id"].isin(roots.index)
            & (df["text"].str.split().str.len() >= MIN_WORDS) & ~df["text"].str.contains(AI_VOICE))
    replies = df[keep].assign(prompt=lambda d: d["parent_id"].map(roots), human_text=lambda d: d["text"],
                              item_id=lambda d: d["message_id"])
    # One reply per prompt, so each prompt's human and AI answers form one pair.
    replies = replies.sort_values("message_id").drop_duplicates("parent_id")
    return take_per_stratum(replies, ["lang"], 3000, "message_id").reset_index(drop=True)


def load(raw: Path, cap: int) -> Iterable[Sample]:
    items = pairs(raw)
    yield from compact(
        make_sample(SPEC, text=strip_markdown(r.human_text), label=Label.HUMAN, role=Role.CORE,
                    record_id=r.message_id, domain="assistant_reply", group_key=r.message_tree_id)
        for r in items.itertuples()
    )
    generated = load_all_generated(raw).merge(items[["item_id", "message_tree_id"]], on="item_id")
    yield from compact(
        make_sample(SPEC, text=strip_markdown(r.text), label=Label.AI, role=Role.CORE,
                    record_id=f"{r.item_id}:{r.generator}" + (f":{r.variant}" if r.variant else ""), domain="assistant_reply", group_key=r.message_tree_id,
                    generator=r.generator, verification="generated locally", transform=r.transform,
                    licence=generated_licence(r.model_licence))
        for r in generated.itertuples() if len(str(r.text).split()) >= 10
    )
