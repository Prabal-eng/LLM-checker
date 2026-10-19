"""Fingerprint of the evaluation splits so any later change to them is detected."""

import hashlib
import json
from pathlib import Path
from typing import Optional

import pandas as pd

from .schema import LOCKED_SPLITS


def locked_hash(df: pd.DataFrame) -> str:
    """sample_id ignores case, whitespace and NFKC forms, so each line also carries the exact text's
    hash and the source slice the gate reports by (Ralph R08-5)."""
    locked = df[df["split"].isin([s.value for s in LOCKED_SPLITS])]
    lines = sorted(f"{r.sample_id}|{r.split}|{r.label}|{r.source_dataset}|"
                   f"{hashlib.sha256(r.text.encode('utf-8')).hexdigest()}" for r in locked.itertuples())
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def read(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, digest: str, counts: dict, history: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(
        {"locked_splits": [s.value for s in LOCKED_SPLITS], "sha256": digest,
         "counts": counts, "history": history},
        indent=2, sort_keys=True,
    ) + "\n", encoding="utf-8")
