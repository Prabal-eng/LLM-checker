"""RAID train split: eight English domains, eleven generators, and adversarial attacks."""

from pathlib import Path
from typing import Iterable

import pandas as pd

from ..schema import Label, LicenceClass, Role, Sample
from ..split import bucket
from ._common import SourceSpec, compact, make_sample, take_per_stratum

_FILE = "train.csv"
_URL = "https://huggingface.co/datasets/liamdugan/raid/resolve/main/train.csv"
# The auto-converted Parquet copy of RAID is truncated to four domains, so read the full CSV.
_CHUNK_ROWS = 100_000
_AI_MODELS = 11  # chatgpt, gpt4, gpt3, gpt2, llama-chat, mistral(-chat), mpt(-chat), cohere(-chat)
_COLUMNS = ["id", "source_id", "model", "attack", "domain", "generation"]
_ENGLISH = {"abstracts", "books", "news", "poetry", "recipes", "reddit", "reviews", "wiki"}
# Attacked texts are drawn only from this share of source documents, which the
# splitter then keeps in evaluation, so attacks never share content with training.
_STRESS_SOURCE_SHARE = 0.15

SPEC = SourceSpec(
    name="raid",
    licence="MIT",
    licence_class=LicenceClass.PERMISSIVE,
    files=((_URL, _FILE),),
)


def _keep(kept: pd.DataFrame, new: pd.DataFrame, strata: list[str], cap: int) -> pd.DataFrame:
    """Merge a chunk into the running selection and re-cap, so memory stays bounded by the caps."""
    return take_per_stratum(pd.concat([kept, new], ignore_index=True), strata, cap, "id")


def _fresh(rows: pd.DataFrame, strata: list[str], cap: int) -> pd.DataFrame:
    """The cap rows per stratum that earlier versions left (Increment 4d); the running selection keeps
    twice the cap so they are still there to take."""
    return take_per_stratum(rows, strata, cap, "id", skip_used=True)


def _candidates(raw: Path, cap: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Humans form one stratum per domain while AI spreads the cap across the generators.
    ai_cap = max(1, cap // _AI_MODELS)
    human = ai = stress = pd.DataFrame(columns=_COLUMNS)
    for df in pd.read_csv(raw / _FILE, usecols=_COLUMNS, chunksize=_CHUNK_ROWS):
        df = df[df["domain"].isin(_ENGLISH)].dropna(subset=["generation"])
        df["source_id"] = df["source_id"].fillna(df["id"])
        df["attack"] = df["attack"].fillna("none")
        clean = df[df["attack"] == "none"]
        human = _keep(human, clean[clean["model"] == "human"], ["domain"], 2 * cap)
        ai = _keep(ai, clean[clean["model"] != "human"], ["domain", "model"], 2 * ai_cap)
        attacked = df[(df["attack"] != "none") & (df["model"] != "human")]
        attacked = attacked[attacked["source_id"].map(bucket) < _STRESS_SOURCE_SHARE]
        stress = _keep(stress, attacked, ["attack"], cap)
    return _fresh(human, ["domain"], cap), _fresh(ai, ["domain", "model"], ai_cap), stress


def load(raw: Path, cap: int) -> Iterable[Sample]:
    human, ai, stress = _candidates(raw, cap)

    def rows(df: pd.DataFrame, role: Role) -> Iterable:
        for r in df.itertuples():
            is_human = r.model == "human"
            yield make_sample(
                SPEC,
                text=r.generation,
                label=Label.HUMAN if is_human else Label.AI,
                role=role,
                record_id=r.id,
                domain=r.domain,
                group_key=r.source_id,
                generator=None if is_human else r.model,
                transform="none" if r.attack == "none" else f"attack:{r.attack}",
            )

    yield from compact(rows(pd.concat([human, ai]), Role.CORE))
    yield from compact(rows(stress, Role.STRESS))
