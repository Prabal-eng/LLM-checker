"""MAGE (DeepfakeTextDetect): human and machine text across ten domains and 27 generators."""

from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from ..schema import Label, LicenceClass, Role, Sample
from ._common import SourceSpec, compact, make_sample, take_per_stratum

_BASE = "https://huggingface.co/datasets/yaful/MAGE/resolve/main/"
_CORE_FILES = ("train.csv", "valid.csv", "test.csv", "test_ood_set_gpt.csv")
_PARAPHRASE_FILE = "test_ood_set_gpt_para.csv"

SPEC = SourceSpec(
    name="mage",
    licence="Apache-2.0",
    licence_class=LicenceClass.PERMISSIVE,
    files=tuple((_BASE + f, f) for f in (*_CORE_FILES, _PARAPHRASE_FILE)),
)


def parse_src(src: str) -> Optional[tuple[str, Label, Optional[str]]]:
    """Map a MAGE ``src`` value to (domain, label, generator)."""
    if src.endswith("_human"):
        return src[: -len("_human")], Label.HUMAN, None
    if "_machine_" in src:
        domain, rest = src.split("_machine_", 1)
        parts = rest.split("_", 1)
        return domain, Label.AI, parts[1] if len(parts) == 2 else parts[0]
    if src.endswith("_gpt4"):
        return src[: -len("_gpt4")], Label.AI, "gpt-4"
    return None


def _frame(raw: Path, files: Iterable[str]) -> pd.DataFrame:
    frames = [pd.read_csv(raw / f).rename_axis("row").reset_index().assign(file_name=f) for f in files]
    df = pd.concat(frames, ignore_index=True).dropna(subset=["text", "src"])
    parsed = df["src"].map(parse_src)
    df = df[parsed.notna()].copy()
    df[["domain", "parsed_label", "generator"]] = pd.DataFrame(parsed[parsed.notna()].tolist(), index=df.index)
    # MAGE encodes human as label 1; keep only rows whose label agrees with src.
    agrees = (df["label"] == 1) == (df["parsed_label"] == Label.HUMAN)
    return df[agrees]


def load(raw: Path, cap: int) -> Iterable[Sample]:
    core = take_per_stratum(_frame(raw, _CORE_FILES), ["domain", "parsed_label"], cap, "text", skip_used=True)
    para = _frame(raw, [_PARAPHRASE_FILE])
    para = para[para["parsed_label"] == Label.AI]

    def rows(df: pd.DataFrame, role: Role, transform: str) -> Iterable[Optional[Sample]]:
        for r in df.itertuples():
            yield make_sample(
                SPEC,
                text=r.text,
                label=r.parsed_label,
                role=role,
                record_id=f"{r.file_name}:{r.row}",
                domain=r.domain,
                # ponytail: D3 no source ids in MAGE, per-row groups rely on near-duplicate clustering.
                group_key=f"{r.file_name}:{r.row}",
                generator=r.generator,
                transform=transform,
            )

    yield from compact(rows(core, Role.CORE, "none"))
    yield from compact(rows(para, Role.STRESS, "paraphrase"))
