"""Selection data and folds. Only train and validation rows are ever read here.

Locked evaluation splits are filtered out at the Parquet reader, so model selection
cannot see them even by accident.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import pandas as pd

from ..corpus.paths import VERSION, corpus_dir
from ..corpus.schema import Label, Split

FIT_SPLITS = (Split.TRAIN.value, Split.VAL.value)
LOSO_SOURCES = ("hc3", "mage", "mirage", "raid")
LOFO_FAMILIES = ("anthropic", "deepseek", "google")
# OpenAI is too large a share of the data to hold out whole; instead its newest models are held
# out, which is the realistic risk: the detector has seen older GPT versions and a new one ships.
RECENT_OPENAI = ("GPT-4o", "GPT-4o-mini", "GPT-o3-mini")
LENGTH_BUCKETS = (0, 25, 50, 100, 200, np.inf)
_COLUMNS = ["sample_id", "text", "label", "split", "source_dataset", "domain", "generator_family", "generator"]


def load_fit_data(root: Optional[Path] = None, version: str = VERSION) -> pd.DataFrame:
    path = (root or corpus_dir()) / version / "corpus.parquet"
    df = pd.read_parquet(path, columns=_COLUMNS, filters=[("split", "in", list(FIT_SPLITS))])
    if not set(df["split"]).issubset(FIT_SPLITS):
        raise RuntimeError("selection data contains a non-fit split")
    return prepare(df)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.reset_index(drop=True)
    df["y"] = (df["label"] == Label.AI.value).astype(int)
    if "generator" not in df:
        df["generator"] = None
    # eval_only rows are extra validation rows used by the recent-OpenAI fold alone.
    df["eval_only"] = df["eval_only"].fillna(False).astype(bool) if "eval_only" in df else False
    words = df["text"].str.split().str.len()
    df["length_bucket"] = pd.cut(words, LENGTH_BUCKETS, right=False,
                                 labels=["<25", "25-49", "50-99", "100-199", "200+"]).astype(str)
    return df


@dataclass(frozen=True)
class Fold:
    """fit trains the model; calib sets thresholds; evaluation measures them."""

    name: str
    fit: pd.DataFrame
    calib: pd.DataFrame
    evaluation: pd.DataFrame


def folds(df: pd.DataFrame) -> Iterator[Fold]:
    train, all_val = df[df["split"] == Split.TRAIN.value], df[df["split"] == Split.VAL.value]
    val = all_val[~all_val["eval_only"]]
    yield Fold("val", fit=train, calib=val, evaluation=val)
    for source in LOSO_SOURCES:
        yield Fold(f"loso:{source}",
                   fit=train[train["source_dataset"] != source],
                   calib=val[val["source_dataset"] != source],
                   evaluation=val[val["source_dataset"] == source])
    for family in LOFO_FAMILIES:
        unseen = val["generator_family"] == family
        yield Fold(f"lofo:{family}",
                   fit=train[train["generator_family"] != family],
                   calib=val[~unseen],
                   evaluation=val[unseen | (val["y"] == 0)])
    recent_train = train["generator"].isin(RECENT_OPENAI)
    yield Fold("recent:openai",
               fit=train[~recent_train],
               calib=val[~val["generator"].isin(RECENT_OPENAI)],
               evaluation=all_val[all_val["generator"].isin(RECENT_OPENAI) | ((all_val["y"] == 0) & ~all_val["eval_only"])])
