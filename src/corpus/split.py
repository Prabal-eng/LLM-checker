"""Group-aware split assignment.

Samples that share a prompt/source group or a near-duplicate cluster form one
component, and a component never spans train, validation, and evaluation.
"""

import hashlib
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from .dedup import UnionFind
from .schema import Role, Split

TRAIN_SHARE = 0.70
VAL_SHARE = 0.15
# Families withheld from fitting so the unseen-generator test measures generalization. Increment 4d fits
# Cohere and Moonshot (Arena supplies Command A and Kimi K2); microsoft (Phi-4-mini), ibm (Granite 4) and
# tii (Falcon 3) answer only the held-out Stack Exchange sites, and no training source has BLOOM.
HOLDOUT_FAMILIES = frozenset({"bigscience", "microsoft", "ibm", "tii"})
# Whole sources withheld from fitting. v4 tests on eight new pre-ChatGPT Stack Exchange sites and two
# frontier sets no model trains on; v3's eight sites are fitted now that its test is spent.
HOLDOUT_SOURCES = frozenset({"stackexchange_heldout", "wasitai2026", "lede"})
# Sources whose test slice would otherwise hold too few human texts to show a 1% bound: with
# 300-600, even one or two errors put the upper bound over 1% (Increment 4c pre-registration).
SOURCE_TEST_SHARE = {"aita2026": 0.35, "no_robots": 0.35, "mirage": 0.35, "maga": 0.20, "dactyl": 0.20}
# Sources whose stress variants follow their parent's split instead of pulling the parent into
# evaluation. Variants of a fitted parent are dropped, since stress rows are never trained on.
STRESS_FOLLOWS_PARENT = frozenset({"beemo"})


def components(group_keys: Iterable[str], cluster_ids: Iterable[object]) -> list[int]:
    nodes: dict[str, int] = {}
    edges = [
        (nodes.setdefault(f"g:{g}", len(nodes)), nodes.setdefault(f"c:{c}", len(nodes)))
        for g, c in zip(group_keys, cluster_ids)
    ]
    uf = UnionFind(len(nodes))
    for a, b in edges:
        uf.union(a, b)
    return [uf.find(a) for a, _ in edges]


def bucket(key: str) -> float:
    """Stable value in [0, 1) derived only from the key."""
    return int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:13], 16) / 16**13


def assign_splits(
    df: pd.DataFrame,
    holdout_families: frozenset = HOLDOUT_FAMILIES,
    *,
    holdout_sources: frozenset = frozenset(),
    stress_follows_parent: frozenset = frozenset(),
    retired: frozenset = frozenset(),
    retired_groups: frozenset = frozenset(),
    salt: str = "",
    test_shares: Optional[dict] = None,
) -> pd.Series:
    """Return the split per row; None marks a sample excluded from the corpus.

    retired holds sample ids of an earlier locked test, and retired_groups its group keys, so a group
    stays retired when its rows are re-selected or their text changes (Amendment 5). Their components
    may be fitted but never tested again; a retired component that stress variants would force into evaluation is dropped.
    salt re-draws the train/validation/test buckets, so a new version gets a fresh test set.
    test_shares raises the test share for named sources (by each component's anchor row).
    """
    comp = pd.Series(components(df["group_key"], df["cluster_id"]), index=df.index)
    source = df["source_dataset"] if "source_dataset" in df else pd.Series("", index=df.index)
    # Anchor on the smallest sample id so assignment does not depend on row order.
    anchor = df.groupby(comp)["sample_id"].transform("min")
    follows = (df["role"] == Role.STRESS.value) & source.isin(stress_follows_parent)
    forcing = df["role"].isin([Role.STRESS.value, Role.INHOUSE.value]) & ~follows
    forced_eval = forcing.groupby(comp).transform("any")
    was_tested = (df["sample_id"].isin(retired) | df["group_key"].isin(retired_groups)).groupby(comp).transform("any")
    b = anchor.map(lambda k: bucket(salt + k))
    test_share = 1.0 - TRAIN_SHARE - VAL_SHARE
    if test_shares:
        first_source = source.groupby(comp).transform(lambda s: s.loc[df.loc[s.index, "sample_id"].idxmin()])
        test_share = first_source.map(lambda s: test_shares.get(s, 1.0 - TRAIN_SHARE - VAL_SHARE)).to_numpy()
    fresh = np.where(b < 1.0 - test_share - VAL_SHARE, Split.TRAIN.value,
                     np.where(b < 1.0 - test_share, Split.VAL.value, Split.TEST.value))
    refit = np.where(b < TRAIN_SHARE / (TRAIN_SHARE + VAL_SHARE), Split.TRAIN.value, Split.VAL.value)
    base = np.where(forced_eval, np.where(was_tested, None, Split.TEST.value),
                    np.where(was_tested, refit, fresh))
    split = pd.Series(base, index=df.index, dtype=object)

    core = df["role"] == Role.CORE.value
    held = source.isin(holdout_sources)
    # A held-out source is tested whole, except rows sharing a component with other sources and rows an
    # earlier version saw, which can't be fitted in its place (Increment 4d final review C2).
    only_held = held.groupby(comp).transform("all")
    fresh_held = only_held & ~was_tested
    split[core & held] = np.where(fresh_held[core & held], Split.TEST.value, None)
    holdout = df["generator_family"].isin(holdout_families)
    split[core & holdout & (split == Split.TEST.value)] = Split.TEST_UNSEEN_GENERATOR.value
    split[core & holdout & split.isin([Split.TRAIN.value, Split.VAL.value])] = None
    for role, target in ((Role.STRESS, Split.STRESS), (Role.INHOUSE, Split.INHOUSE),
                         (Role.DIAGNOSTIC, Split.DIAGNOSTIC)):
        split[df["role"] == role.value] = target.value
    split[follows & (pd.Series(base, index=df.index) != Split.TEST.value)] = None
    # A retired component that stress variants would force into evaluation leaves whole: its
    # variants were scored beside the retired parent, so they are not fresh either (Ralph R07-3).
    split[(df["role"] == Role.STRESS.value) & forced_eval & was_tested] = None
    return split


def partition(split: Optional[str]) -> Optional[str]:
    """Train, validation, and evaluation are the partitions a component must not span."""
    if split in (Split.TRAIN.value, Split.VAL.value):
        return split
    if split in (Split.TEST.value, Split.TEST_UNSEEN_GENERATOR.value,
                 Split.STRESS.value, Split.INHOUSE.value):
        return "evaluation"
    return None
