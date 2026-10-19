"""Assemble the authorship corpus: download, ingest, redact, deduplicate, split, lock, audit.

    python -m src.corpus.build [--sources mage,hc3,...] [--cap 3000] [--relock "reason"]

Full text is written under CORPUS_DIR (default D:/llm-authorship-corpus). The build
record and test lock are written into the repo under data/benchmark/<version>/.
"""

import argparse
import json
import logging
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Mapping, Optional

import numpy as np
import pandas as pd

from . import confound, dedup, lock, pii
from .audit import audit, earlier_matches
from .paths import PREVIOUS_VERSIONS, VERSION, corpus_dir, lock_path, record_path
from .schema import LOCKED_SPLITS, Role
from .sources import SOURCES
from .sources._common import download
from .split import (HOLDOUT_FAMILIES, HOLDOUT_SOURCES, SOURCE_TEST_SHARE, STRESS_FOLLOWS_PARENT, assign_splits,
                    components, partition)
from .text import sample_id

logger = logging.getLogger(__name__)

DEFAULT_CAP = 3000
# Increment 4d: AI-only sources and the human rows each is balanced against; the AI-only probe drops a
# source whose surface features alone separate it from that register.
AI_ONLY_POOLS = {
    "arena140k": ("dolly", "oasst1", "no_robots"),
    "openturingbench": ("raid:news", "dactyl:news", "maga:cc_news", "maga:npr_news"),
}
_ROLE_PRIORITY = {Role.CORE.value: 0, Role.INHOUSE.value: 1, Role.STRESS.value: 2, Role.DIAGNOSTIC.value: 3}


class BuildError(RuntimeError):
    pass


def _collect(sources: Mapping[str, ModuleType], raw_root: Path, cap: int) -> tuple[pd.DataFrame, list[dict]]:
    frames, records = [], []
    for name, module in sources.items():
        raw = raw_root / name
        files = [{"url": url, "path": rel, "sha256": download(url, raw / rel)} for url, rel in module.SPEC.files]
        rows = [s.model_dump(mode="json") for s in module.load(raw, cap)]
        logger.info("%s: %d samples", name, len(rows))
        records.append({"name": name, "licence": module.SPEC.licence,
                        "licence_class": module.SPEC.licence_class.value, "files": files, "loaded": len(rows)})
        frames.append(pd.DataFrame(rows))
    return pd.concat(frames, ignore_index=True), records


# Increment 4d: an earlier model fitted train and validation rows and was judged on test rows, so none
# may be evaluated again. Stress and in-house rows are report-only and stay available.
SEEN_SPLITS = ("train", "val", "test", "test_unseen_generator")
TESTED_SPLITS = ("test", "test_unseen_generator")


def _retired(root: Path, versions, column: str) -> frozenset:
    values: set = set()
    for version in versions:
        manifest = root / version / "manifest.parquet"
        if not manifest.exists():
            raise BuildError(f"cannot retire {version}'s rows: {manifest} is missing")
        values |= set(pd.read_parquet(manifest, columns=[column],
                                      filters=[("split", "in", list(SEEN_SPLITS))])[column])
    return frozenset(values)


def retired_seen_ids(root: Path, versions=PREVIOUS_VERSIONS) -> frozenset:
    """Sample ids of every row an earlier version fitted or tested. They may be fitted, never evaluated."""
    return _retired(root, versions, "sample_id")


def retired_seen_groups(root: Path, versions=PREVIOUS_VERSIONS) -> frozenset:
    """Group keys of every row an earlier version fitted or tested, for the audit's retirement check."""
    return _retired(root, versions, "group_key")


def retirement(root: Path, versions=PREVIOUS_VERSIONS, retired: Optional[frozenset] = None,
               retired_groups: Optional[frozenset] = None) -> tuple:
    """(ids, groups) of every row earlier versions fitted or tested; each is read from the manifests
    unless given."""
    ids = retired_seen_ids(root, versions) if retired is None else retired
    groups = retired_seen_groups(root, versions) if retired_groups is None else retired_groups
    return ids, groups


def earlier_frame(root: Path, versions=PREVIOUS_VERSIONS) -> pd.DataFrame:
    """sample_id, text and seen (some copy was fitted or tested) for every row of the earlier versions, once
    per text. The spec bars a v4 test text from any split of v1-v3, report-only ones included (final review
    I5); one row per text keeps memory down, since versions repeat most rows (I6)."""
    frames = []
    for version in versions:
        path = root / version / "corpus.parquet"
        if not path.exists():
            raise BuildError(f"cannot check near-duplicates of {version}'s rows: {path} is missing")
        part = pd.read_parquet(path, columns=["sample_id", "text", "split"])
        frames.append(part.assign(seen=part["split"].isin(SEEN_SPLITS)).drop(columns="split"))
    if not frames:
        return pd.DataFrame({"sample_id": pd.Series(dtype=str), "text": pd.Series(dtype=str),
                             "seen": pd.Series(dtype=bool)})
    rows = pd.concat(frames, ignore_index=True)
    rows["seen"] = rows.groupby("sample_id")["seen"].transform("any")
    return rows.drop_duplicates("sample_id").reset_index(drop=True)


def earlier_texts(root: Path, versions=PREVIOUS_VERSIONS) -> list:
    """Every earlier version's texts, once each: no locked-test row may match one."""
    return earlier_frame(root, versions)["text"].tolist()


def earlier_cluster_hits(texts: list, earlier: pd.DataFrame) -> tuple:
    """(seen, any): per text, whether it shares a containment cluster with an earlier fitted or tested text,
    and with any earlier text, so a quoted or lightly edited copy is retired like the original."""
    if not texts or earlier.empty:
        return [False] * len(texts), [False] * len(texts)
    clusters = dedup.cluster_with_containment(list(texts) + earlier["text"].tolist())
    mine, old = clusters[:len(texts)], clusters[len(texts):]
    seen = {c for c, s in zip(old, earlier["seen"]) if s}
    every = set(old)
    return [c in seen for c in mine], [c in every for c in mine]


MAX_EARLIER_PASSES = 4


def _redact(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    totals: Counter = Counter()
    texts = []
    for text in df["text"]:
        clean, hits = pii.redact(text)
        totals.update(hits)
        texts.append(clean)
    df = df.assign(text=texts)
    df["sample_id"] = df["text"].map(sample_id)
    return df, dict(totals)


def _resolve_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Drop texts whose sources disagree on the label; otherwise keep the highest-priority copy."""
    conflicting = df.groupby("sample_id")["label"].transform("nunique") > 1
    kept = df[~conflicting].assign(_p=df["role"].map(_ROLE_PRIORITY))
    kept = kept.sort_values(["sample_id", "_p", "source_dataset", "source_record_id"])
    deduped = kept.drop_duplicates("sample_id").drop(columns="_p")
    return deduped, {"label_conflicts_dropped": int(conflicting.sum()),
                     "exact_duplicates_dropped": int(len(kept) - len(deduped))}


def settle_splits(df: pd.DataFrame, assign, recluster, max_passes: int = 6) -> pd.Series:
    """Assign splits, then recluster the kept rows as the audit does and merge any link that puts one
    component in two partitions, until none does. Clustering ignores boilerplate by how many texts
    share it, so dropping the excluded rows can link texts the full clustering kept apart
    (Amendment 5). df["cluster_id"] is updated with the merged clusters."""
    for _ in range(max_passes):
        split = assign(df)
        kept = split.notna()
        clusters = recluster(df.loc[kept, "text"].tolist())
        parts = split[kept].map(partition)
        comp = pd.Series(components(df.loc[kept, "group_key"], clusters), index=parts.index)
        spans = parts.dropna().groupby(comp[parts.notna()]).nunique()
        if not (spans > 1).any():
            return split
        logger.info("%d components span partitions after reclustering the kept rows; merging", int((spans > 1).sum()))
        uf = dedup.UnionFind(len(df))
        position = pd.Series(range(len(df)), index=df.index)
        for key, frame in (("cluster_id", df), ("new", pd.DataFrame({"new": clusters}, index=parts.index))):
            for members in frame.groupby(key).indices.values():
                rows = position[frame.index[members]].to_numpy()
                for other in rows[1:]:
                    uf.union(rows[0], other)
        df["cluster_id"] = [uf.find(i) for i in range(len(df))]
    raise BuildError(f"splits did not settle after {max_passes} reclustering passes")


def _counts(df: pd.DataFrame, *columns: str) -> dict:
    return {"|".join(map(str, k)) if isinstance(k, tuple) else str(k): int(v)
            for k, v in df.groupby(list(columns)).size().items()}


def build(
    *,
    sources: Mapping[str, ModuleType],
    cap: int = DEFAULT_CAP,
    root: Optional[Path] = None,
    version: str = VERSION,
    lock_file: Optional[Path] = None,
    record_file: Optional[Path] = None,
    relock_reason: Optional[str] = None,
    retired: Optional[frozenset] = None,
    retired_groups: Optional[frozenset] = None,
    salt: Optional[str] = None,
    previous_versions: tuple = PREVIOUS_VERSIONS,
) -> dict:
    """retired and salt default to the ids of every row the previous versions fitted or tested and the
    version name, so each version draws a fresh test set no earlier model has seen. Near-duplicates of
    that earlier text are retired too."""
    root = root or corpus_dir()
    lock_file = lock_file or lock_path(version)
    record_file = record_file or record_path(version)

    df, source_records = _collect(sources, root / "raw", cap)
    df, redactions = _redact(df)
    earlier = earlier_frame(root, previous_versions)
    # A text an earlier version only reported (stress, diagnostic) can't be tested fresh. Its core copy
    # goes, so duplicate resolution keeps the report-only copy where it was (final review I5).
    reported = frozenset(earlier.loc[~earlier["seen"], "sample_id"])
    reused = (df["role"] == Role.CORE.value) & df["sample_id"].isin(reported)
    df = df[~reused]
    df, duplicates = _resolve_duplicates(df)
    ai_only = confound.ai_only_probe(df, {s: p for s, p in AI_ONLY_POOLS.items() if s in sources})
    ai_only_excluded = sorted(s for s, auc in ai_only.items() if auc > confound.WARN_AUC)
    if ai_only_excluded:
        logger.warning("AI-only probe excludes %s (AUC over %.2f)", ai_only_excluded, confound.WARN_AUC)
        df = df[~df["source_dataset"].isin(ai_only_excluded)]
    df = df.sort_values("sample_id").reset_index(drop=True)
    logger.info("clustering %d samples", len(df))
    df["cluster_id"] = dedup.cluster_with_containment(df["text"].tolist())
    retired, retired_groups = retirement(root, previous_versions, retired=retired, retired_groups=retired_groups)
    # Only rows that could still be evaluated are clustered against earlier text: rows retired by id or
    # group can't be, and clustering everything would not fit this machine's memory. A core row is retired
    # when it matches any earlier text; a stress row only when it matches fitted or tested text, since
    # stress texts recur in every version by design.
    open_rows = ~(df["sample_id"].isin(retired) | df["group_key"].isin(retired_groups))
    hit_seen, hit_any = earlier_cluster_hits(df.loc[open_rows, "text"].tolist(), earlier)
    near = pd.Series(False, index=df.index)
    near[open_rows] = np.where((df.loc[open_rows, "role"] == Role.CORE.value).to_numpy(), hit_any, hit_seen)
    retired = frozenset(retired) | frozenset(df.loc[near, "sample_id"])
    near_by_source = _counts(df[near], "source_dataset") if near.any() else {}
    logger.info("%d rows retired as near-duplicates of earlier text", int(near.sum()))
    salt = version if salt is None else salt
    earlier_list = earlier["text"].tolist()
    late = 0
    # Settle on the audit's own check: which texts are clustered moves the boilerplate cut-off, so a link
    # the open-row clustering missed can still show when the tested rows are clustered alone (review I6).
    for _ in range(MAX_EARLIER_PASSES):
        df["split"] = settle_splits(
            df, lambda d: assign_splits(d, holdout_sources=HOLDOUT_SOURCES,
                                        stress_follows_parent=STRESS_FOLLOWS_PARENT, retired=retired,
                                        retired_groups=retired_groups, salt=salt, test_shares=SOURCE_TEST_SHARE),
            dedup.cluster_with_containment)
        tested = df[df["split"].isin(TESTED_SPLITS)]
        hits = earlier_matches(tested, earlier_list)
        if not hits:
            break
        logger.info("%d tested rows match earlier text under the audit's clustering; retiring them", len(hits))
        retired, late = retired | frozenset(hits), late + len(hits)
    else:
        raise BuildError(f"tested rows still matched earlier text after {MAX_EARLIER_PASSES} passes")
    excluded = int(df["split"].isna().sum())
    df = df[df["split"].notna()].reset_index(drop=True)

    digest = lock.locked_hash(df)
    existing = lock.read(lock_file)
    history = list(existing.get("history", [])) if existing else []
    if existing and existing.get("sha256") != digest:
        if not relock_reason:
            raise BuildError(
                f"evaluation splits changed ({existing['sha256'][:12]} -> {digest[:12]}); "
                "rerun with --relock REASON only if the change is intended"
            )
        history.append({"previous": existing["sha256"], "reason": relock_reason,
                        "at": datetime.now(timezone.utc).isoformat()})
    out = root / version
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / "corpus.parquet", index=False)
    df.drop(columns="text").to_parquet(out / "manifest.parquet", index=False)

    # Mismatches without a relock reason were rejected above, so the new digest is the lock to audit against.
    report = audit(df, {"sha256": digest}, retired_groups=retired_groups, earlier_texts=earlier_list)
    record = {
        "version": version,
        "built_at": datetime.now(timezone.utc).isoformat(),
        "cap_per_stratum": cap,
        "holdout_families": sorted(HOLDOUT_FAMILIES),
        "holdout_sources": sorted(HOLDOUT_SOURCES),
        "stress_follows_parent": sorted(STRESS_FOLLOWS_PARENT),
        "split_salt": salt,
        "source_test_share": SOURCE_TEST_SHARE,
        "retired_seen_ids": len(retired),
        "retired_rows_present": int(df["sample_id"].isin(retired).sum()),
        "near_duplicate_retired": int(near.sum()),
        "near_duplicate_retired_by_source": near_by_source,
        "retired_by_audit_check": late,
        "core_copies_of_reported_text_dropped": int(reused.sum()),
        "earlier_texts": len(earlier_list),
        "sources": source_records,
        "redactions": redactions,
        "duplicates": duplicates,
        "excluded_holdout_from_fit": excluded,
        "total": int(len(df)),
        "counts": {
            "split_label": _counts(df, "split", "label"),
            "split_source": _counts(df, "split", "source_dataset"),
            "split_generator_family": _counts(df[df["label"] == "ai"], "split", "generator_family"),
            "split_licence_class": _counts(df, "split", "licence_class"),
            "stress_transform_label": _counts(df[df["split"] == "stress"], "transform", "label"),
        },
        "shortcut_probe": confound.shortcut_probe(df),
        "ai_only_probe": {"auc": ai_only, "pools": {s: list(p) for s, p in AI_ONLY_POOLS.items()},
                          "excluded": ai_only_excluded, "warn_threshold": confound.WARN_AUC},
        "audit": {"passed": report.passed, "errors": report.errors, "warnings": report.warnings,
                  "slices": report.slices},
        "test_lock_sha256": digest,
    }
    record_file.parent.mkdir(parents=True, exist_ok=True)
    record_file.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not report.passed:
        raise BuildError(f"audit failed: {report.errors}")
    if not existing or existing.get("sha256") != digest:
        locked = df[df["split"].isin([s.value for s in LOCKED_SPLITS])]
        lock.write(lock_file, digest, _counts(locked, "split", "label"), history)
    return record


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sources", default=",".join(SOURCES), help="comma-separated source names")
    parser.add_argument("--cap", type=int, default=DEFAULT_CAP, help="max samples per stratum")
    parser.add_argument("--corpus-dir", type=Path, default=None)
    parser.add_argument("--relock", default=None, metavar="REASON")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    unknown = set(args.sources.split(",")) - set(SOURCES)
    if unknown:
        parser.error(f"unknown sources: {sorted(unknown)}")
    try:
        record = build(sources={n: SOURCES[n] for n in args.sources.split(",")}, cap=args.cap,
                       root=args.corpus_dir, relock_reason=args.relock)
    except BuildError as exc:
        print(f"BUILD FAILED: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({k: record[k] for k in ("total", "counts", "shortcut_probe", "audit", "test_lock_sha256")},
                     indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
