"""Eight Stack Exchange sites never used before (Increment 4d): the v4 locked test's unseen Q&A register.

Same selection as the v3 loader (questions dated 2021 or earlier, best answer scored at least 2, no code
blocks, at most three links), paired with answers from generators no model trains on (Phi-4-mini,
Granite 4, Falcon 3). Held out whole (split.HOLDOUT_SOURCES).
"""

from pathlib import Path
from typing import Iterable

import pandas as pd

from ..schema import LicenceClass, Sample
from . import stackexchange as se
from ._common import SourceSpec

SITES = ("travel", "diy", "workplace", "academia", "bicycles", "mechanics", "photo", "writers")
PER_SITE = 375
# The larger sites are published as two parquet shards.
SHARDS = {"diy": 2, "workplace": 2, "academia": 2}


def _files(site: str) -> list:
    n = SHARDS.get(site, 1)
    return [f"{site}.stackexchange.com/train-{i:05d}-of-{n:05d}.parquet" for i in range(n)]


SPEC = SourceSpec(
    name="stackexchange_heldout",
    licence="CC-BY-SA-4.0",
    licence_class=LicenceClass.COPYLEFT,
    files=tuple((f"{se._BASE}{rel}", rel) for site in SITES for rel in _files(site)),
)


def pairs(raw: Path) -> pd.DataFrame:
    return se.select_pairs(raw, SITES, PER_SITE)


def load(raw: Path, cap: int) -> Iterable[Sample]:
    yield from se.samples(SPEC, raw, pairs(raw))
