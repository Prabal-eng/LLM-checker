"""Near-duplicate clustering with MinHash signatures and banded LSH.

Whole-text Jaccard misses one text mostly contained in another (Ralph R07-6, repaid debt D9), so
cluster_with_containment also links texts sharing most of their sampled word 13-grams.
"""

import zlib
from collections import defaultdict
from typing import Sequence

import numpy as np

from .text import words

NUM_PERM = 64
BANDS = 16
ROWS = NUM_PERM // BANDS
SHINGLE = 3
MAX_WORDS = 2000
SIMILARITY = 0.7
MAX_HEADS = 50  # ponytail: D4 bounded bucket verification, exhaustive pairs if an audit finds misses.
_PRIME = np.uint64(4294967311)  # smallest prime above 2**32

_rng = np.random.default_rng(20260926)
_A = _rng.integers(1, 2**31, size=NUM_PERM, dtype=np.uint64)
_B = _rng.integers(0, 2**31, size=NUM_PERM, dtype=np.uint64)


def signature(text: str) -> np.ndarray:
    tokens = words(text)[:MAX_WORDS]
    if len(tokens) < SHINGLE:
        grams = [" ".join(tokens)]
    else:
        grams = [" ".join(tokens[i:i + SHINGLE]) for i in range(len(tokens) - SHINGLE + 1)]
    x = np.fromiter(
        (zlib.crc32(g.encode("utf-8")) for g in set(grams)), dtype=np.uint64
    )
    return ((np.outer(x, _A) + _B) % _PRIME).min(axis=0)


class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def cluster(texts: Sequence[str]) -> list[int]:
    """Return a cluster id per text; texts with estimated Jaccard >= SIMILARITY share one."""
    if not len(texts):
        return []
    sigs = np.vstack([signature(t) for t in texts])
    uf = UnionFind(len(texts))
    for band in range(BANDS):
        buckets: dict[bytes, list[int]] = defaultdict(list)
        block = sigs[:, band * ROWS:(band + 1) * ROWS]
        for i in range(len(texts)):
            buckets[block[i].tobytes()].append(i)
        for members in buckets.values():
            if len(members) < 2:
                continue
            # Star verification against successive heads keeps template buckets near-linear.
            remaining, heads = members, 0
            while len(remaining) > 1 and heads < MAX_HEADS:
                head, unmatched = remaining[0], []
                for other in remaining[1:]:
                    if np.mean(sigs[head] == sigs[other]) >= SIMILARITY:
                        uf.union(head, other)
                    else:
                        unmatched.append(other)
                remaining, heads = unmatched, heads + 1
    return [uf.find(i) for i in range(len(texts))]


CONTAINMENT_N = 13          # word n-gram length for the containment check (Ralph R07-6)
CONTAINMENT_SAMPLE = 8      # keep the 1-in-8 n-grams chosen by hash, so shared passages align anywhere
CONTAINMENT_SHARE = 0.5     # linked when half of the smaller text's sampled n-grams occur in the other
CONTAINMENT_MIN_SHARED = 3
MAX_POSTING = 20            # n-grams in more texts than this are boilerplate and link nothing


def sampled_ngrams(text: str) -> np.ndarray:
    tokens = words(text)[:MAX_WORDS]
    grams = (zlib.crc32(" ".join(tokens[i:i + CONTAINMENT_N]).encode("utf-8"))
             for i in range(len(tokens) - CONTAINMENT_N + 1))
    return np.unique(np.fromiter((g for g in grams if g % CONTAINMENT_SAMPLE == 0), dtype=np.uint32))


def containment_pairs(texts: Sequence[str]) -> list[tuple[int, int]]:
    """Pairs of texts where one is largely contained in the other, which whole-text Jaccard misses."""
    per_text = [sampled_ngrams(t) for t in texts]
    sizes = np.array([len(g) for g in per_text], dtype=np.int64)
    if not sizes.sum():
        return []
    grams = np.concatenate(per_text)
    owner = np.repeat(np.arange(len(texts), dtype=np.int64), sizes)
    order = np.argsort(grams, kind="stable")
    grams, owner = grams[order], owner[order]
    starts = np.flatnonzero(np.r_[True, grams[1:] != grams[:-1]])
    counts = np.diff(np.r_[starts, len(grams)])
    keys = []
    for start, count in zip(starts[(counts > 1) & (counts <= MAX_POSTING)],
                            counts[(counts > 1) & (counts <= MAX_POSTING)]):
        members = np.unique(owner[start:start + count])
        if len(members) > 1:
            a, b = np.triu_indices(len(members), 1)
            keys.append(members[a] * len(texts) + members[b])
    if not keys:
        return []
    pair_keys, shared = np.unique(np.concatenate(keys), return_counts=True)
    i, j = pair_keys // len(texts), pair_keys % len(texts)
    smaller = np.minimum(sizes[i], sizes[j])
    linked = (shared >= CONTAINMENT_MIN_SHARED) & (shared >= CONTAINMENT_SHARE * smaller)
    return list(zip(i[linked].tolist(), j[linked].tolist()))


OPENING_WORDS = 24  # texts opening with the same 24 words share a prompt (MAGE has no prompt ids, D3)


def opening_pairs(texts: Sequence[str]) -> list[tuple[int, int]]:
    first: dict = {}
    pairs = []
    for i, text in enumerate(texts):
        tokens = words(text)
        if len(tokens) >= OPENING_WORDS + 6:
            key = " ".join(tokens[:OPENING_WORDS])
            if key in first:
                pairs.append((first[key], i))
            else:
                first[key] = i
    return pairs


def cluster_with_containment(texts: Sequence[str]) -> list[int]:
    """cluster(), plus links between texts where one is mostly contained in the other or both open
    with the same words, which whole-text Jaccard misses (Ralph R07-6)."""
    ids = cluster(texts)
    uf = UnionFind(len(texts))
    for i, root in enumerate(ids):
        uf.union(i, root)
    for i, j in containment_pairs(texts) + opening_pairs(texts):
        uf.union(i, j)
    return [uf.find(i) for i in range(len(texts))]
