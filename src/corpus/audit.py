"""Independent corpus audit. Exits non-zero on any integrity violation.

    python -m src.corpus.audit [--corpus-dir D:/llm-authorship-corpus] [--version v1]

Clusters and components are recomputed from the stored text rather than trusted
from the build, so a faulty build step cannot vouch for itself.
"""

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd
from pydantic import ValidationError

from . import dedup, lock, pii
from .paths import corpus_dir, lock_path, previous_versions, VERSION
from .schema import Label, Role, Sample, Split
from .split import HOLDOUT_FAMILIES, HOLDOUT_SOURCES, components, partition
from .text import sample_id

MIN_HUMAN_MVT = 300     # zero errors in 300 gives a one-sided 95% upper bound of about 1.0%
MIN_HUMAN_TARGET = 600  # about 0.5%

_ROLE_SPLITS = {
    Role.CORE.value: {Split.TRAIN.value, Split.VAL.value, Split.TEST.value, Split.TEST_UNSEEN_GENERATOR.value},
    Role.STRESS.value: {Split.STRESS.value},
    Role.INHOUSE.value: {Split.INHOUSE.value},
    Role.DIAGNOSTIC.value: {Split.DIAGNOSTIC.value},
}
_SAMPLE_FIELDS = tuple(Sample.model_fields)


@dataclass
class AuditReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    slices: list[dict] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.errors


def audit(df: pd.DataFrame, locked: Optional[dict], retired_groups: frozenset = frozenset(),
          earlier_texts: Sequence[str] = ()) -> AuditReport:
    """retired_groups holds the group keys of every row earlier versions fitted or tested; no evaluation
    row may belong to one (Ralph R07-3). earlier_texts holds those rows' texts; no locked-test row may
    share a near-duplicate cluster with one (Increment 4d)."""
    report = AuditReport()
    df = df.sort_values("sample_id").reset_index(drop=True)

    invalid = []
    for row in df[list(_SAMPLE_FIELDS)].to_dict("records"):
        try:
            Sample(**{k: (None if isinstance(v, float) and pd.isna(v) else v) for k, v in row.items()})
        except ValidationError as exc:
            invalid.append(f"{row.get('sample_id')}: {exc.errors()[0]['msg']}")
    if invalid:
        report.errors.append(f"{len(invalid)} rows fail the provenance schema, e.g. {invalid[:3]}")

    wrong_id = (df["text"].map(sample_id) != df["sample_id"]).sum()
    if wrong_id:
        report.errors.append(f"{wrong_id} sample_id values do not match their text")
    if df["sample_id"].duplicated().any():
        report.errors.append(f"{df['sample_id'].duplicated().sum()} duplicate sample_id values")

    bad_role = [r for r in df[["role", "split"]].itertuples() if r.split not in _ROLE_SPLITS.get(r.role, set())]
    if bad_role:
        report.errors.append(f"{len(bad_role)} samples are in a split their role does not allow")

    parts = df["split"].map(partition)
    comp = pd.Series(components(df["group_key"], dedup.cluster_with_containment(df["text"].tolist())), index=df.index)
    spans = parts.dropna().groupby(comp[parts.notna()]).nunique()
    leaking = spans[spans > 1]
    if len(leaking):
        example = df.loc[comp == leaking.index[0], ["source_dataset", "group_key", "split"]].head(3).to_dict("records")
        report.errors.append(
            f"{len(leaking)} groups or near-duplicate clusters span train/validation/evaluation, e.g. {example}"
        )

    fitted_holdout = df[df["split"].isin([Split.TRAIN.value, Split.VAL.value])
                        & df["generator_family"].isin(HOLDOUT_FAMILIES)]
    if len(fitted_holdout):
        report.errors.append(f"{len(fitted_holdout)} held-out-family samples are in train or validation")

    fitted_source = df[df["split"].isin([Split.TRAIN.value, Split.VAL.value])
                       & df["source_dataset"].isin(HOLDOUT_SOURCES)]
    if len(fitted_source):
        report.errors.append(f"{len(fitted_source)} held-out-source samples are in train or validation")

    if retired_groups:
        evaluation = df["split"].isin([Split.TEST.value, Split.TEST_UNSEEN_GENERATOR.value, Split.STRESS.value])
        reused = df[evaluation & df["group_key"].isin(retired_groups)]
        if len(reused):
            report.errors.append(f"{len(reused)} evaluation samples belong to an earlier version's locked-test "
                                 f"group, e.g. {reused['group_key'].iloc[0]}")

    if len(earlier_texts):
        tested = df[df["split"].isin([Split.TEST.value, Split.TEST_UNSEEN_GENERATOR.value])]
        hits = earlier_matches(tested, earlier_texts)
        if hits:
            report.errors.append(f"{len(hits)} evaluation samples match text an earlier version fitted or "
                                 f"tested, e.g. {hits[0][:12]}")

    pii_hits = df["text"].map(pii.find)
    flagged = pii_hits[pii_hits.map(bool)]
    if len(flagged):
        kinds = sum(flagged, Counter())
        report.errors.append(f"{len(flagged)} samples contain PII patterns: {dict(kinds)}")

    if locked is None:
        report.errors.append("no test lock exists for this corpus version")
    elif locked.get("sha256") != lock.locked_hash(df):
        report.errors.append("evaluation splits differ from the committed test lock")

    human = df[(df["label"] == Label.HUMAN.value) & df["split"].isin([Split.TEST.value, Split.STRESS.value,
                                                                     Split.INHOUSE.value])]
    for (split, source, domain), n in human.groupby(["split", "source_dataset", "domain"]).size().items():
        report.slices.append({"split": split, "source": source, "domain": domain, "human": int(n),
                              "meets_mvt": n >= MIN_HUMAN_MVT, "meets_target": n >= MIN_HUMAN_TARGET})
    short = [s for s in report.slices if s["split"] == Split.TEST.value and not s["meets_mvt"]]
    if short:
        report.warnings.append(f"{len(short)} test slices have fewer than {MIN_HUMAN_MVT} human samples")
    return report


def earlier_matches(tested: pd.DataFrame, earlier_texts: Sequence[str]) -> list:
    """Sample ids of tested rows that share a containment cluster with an earlier version's text. The build
    settles its splits on this same check, so the two can't disagree (Increment 4d final review I6)."""
    if tested.empty or not len(earlier_texts):
        return []
    clusters = dedup.cluster_with_containment(tested["text"].tolist() + list(earlier_texts))
    old = set(clusters[len(tested):])
    return [sid for sid, c in zip(tested["sample_id"], clusters[:len(tested)]) if c in old]


def earlier_seen_groups(base: Path, version: str) -> frozenset:
    """Group keys of every row an earlier version fitted or tested (Increment 4d). A missing manifest is
    an error: skipping it would silently let a retired group be tested again (Ralph R08-10)."""
    from .build import SEEN_SPLITS

    retired: set = set()
    for earlier in previous_versions(version):
        manifest = base / earlier / "manifest.parquet"
        if not manifest.exists():
            raise FileNotFoundError(f"{manifest} is missing, so {earlier}'s retired groups can't be checked")
        retired |= set(pd.read_parquet(manifest, columns=["group_key"],
                                       filters=[("split", "in", list(SEEN_SPLITS))])["group_key"])
    return frozenset(retired)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus-dir", type=Path, default=None)
    parser.add_argument("--version", default=VERSION)
    args = parser.parse_args(argv)

    base = args.corpus_dir or corpus_dir()
    root = base / args.version
    from .build import earlier_texts

    report = audit(pd.read_parquet(root / "corpus.parquet"), lock.read(lock_path(args.version)),
                   retired_groups=earlier_seen_groups(base, args.version),
                   earlier_texts=earlier_texts(base, previous_versions(args.version)))
    print(json.dumps({"passed": report.passed, "errors": report.errors, "warnings": report.warnings,
                      "slices": report.slices}, indent=2))
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
