"""Helpers shared by the dataset loaders."""

import hashlib
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
import requests

from ..schema import Label, LicenceClass, Role, Sample
from ..text import sample_id
from ..trims import trim_header, trim_preamble  # noqa: F401  (loaders import them from here)

logger = logging.getLogger(__name__)

DOWNLOAD_ATTEMPTS = 8

# Order matters: more specific names come before the generic patterns they contain.
_FAMILIES = [
    (r"r1-distill", "deepseek"),
    (r"zephyr", "mistral"),
    (r"^(7|13|30|65)b$", "meta"),  # MAGE names LLaMA checkpoints by size only
    (r"tulu", "meta"),
    (r"gpt-j|gpt_j|gptj|neox|pythia", "eleutherai"),
    (r"bloom|(^|[^a-z])t0([^a-z0-9]|$)|t0pp", "bigscience"),
    (r"falcon", "tii"),
    (r"gpt|davinci|chatgpt|openai|(^|[^a-z])o[134](-mini)?($|[^a-z])", "openai"),
    (r"claude", "anthropic"),
    (r"gemini|palm|bard|flan|t5|gemma", "google"),
    (r"llama|(^|[^a-z])opt([^a-z]|$)|galactica", "meta"),
    (r"mistral|mixtral|ministral|magistral", "mistral"),
    (r"cohere|command", "cohere"),
    (r"mpt", "mosaicml"),
    (r"qwen|qwq", "alibaba"),
    (r"deepseek", "deepseek"),
    (r"grok", "xai"),
    (r"moonshot|kimi", "moonshot"),
    (r"doubao", "bytedance"),
    (r"glm", "zhipu"),
    (r"nemotron|nvidia", "nvidia"),
    (r"(^|[^a-z])phi-?\d|microsoft", "microsoft"),
    (r"granite|ibm", "ibm"),
]


def generator_family(generator: str) -> str:
    name = generator.lower()
    for pattern, family in _FAMILIES:
        if re.search(pattern, name):
            return family
    return f"other:{name}"


@dataclass(frozen=True)
class SourceSpec:
    name: str
    licence: str
    licence_class: LicenceClass
    files: tuple[tuple[str, str], ...] = field(default_factory=tuple)  # (url, relative path)


def download(url: str, dest: Path) -> str:
    """Download once to dest and return the file's SHA-256."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        partial = dest.with_suffix(dest.suffix + ".part")
        # Only consecutive attempts that add no bytes count toward giving up, so a
        # large file on a flaky connection keeps resuming as long as it progresses.
        stalled, last_size = 0, -1
        while True:
            have = partial.stat().st_size if partial.exists() else 0
            stalled = 0 if have > last_size else stalled + 1
            last_size = have
            logger.info("downloading %s (resuming at %d bytes)", url, have)
            try:
                headers = {"Range": f"bytes={have}-"} if have else {}
                with requests.get(url, stream=True, timeout=60, headers=headers) as response:
                    response.raise_for_status()
                    # A server that ignores Range sends the whole file again.
                    mode = "ab" if have and response.status_code == 206 else "wb"
                    with open(partial, mode) as out:
                        for chunk in response.iter_content(chunk_size=1 << 20):
                            out.write(chunk)
                break
            except (requests.ConnectionError, requests.exceptions.ChunkedEncodingError,
                    requests.Timeout) as exc:
                if stalled + 1 >= DOWNLOAD_ATTEMPTS:
                    raise
                logger.warning("download interrupted (%s); retrying", exc.__class__.__name__)
        partial.replace(dest)
    digest = hashlib.sha256()
    with open(dest, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def take_per_stratum(df: pd.DataFrame, strata: list[str], cap: int, key: str, *,
                     skip_used: bool = False) -> pd.DataFrame:
    """Keep up to cap rows per stratum, chosen by a stable hash of key.

    skip_used (Increment 4d) passes over each stratum's first cap rows, the ones every earlier corpus
    version took with the same cap, and takes them only when the stratum has too few others, so a new
    version can test rows no earlier model saw. Retirement still keeps a reused row out of evaluation."""
    order = df[key].astype(str).map(lambda k: hashlib.sha256(k.encode("utf-8")).hexdigest())
    work = df.assign(_order=order).sort_values("_order")
    if skip_used:
        work = work.assign(_used=work.groupby(strata, sort=False).cumcount() < cap)
        work = work.sort_values(["_used", "_order"], kind="stable").drop(columns="_used")
    return work.groupby(strata, sort=False).head(cap).drop(columns="_order")


# Word-count band edges for length matching, each band about 25% wider than the last (R12-1: bands a
# factor of two wide still let one MAGA domain's lengths separate the labels at AUC 0.83 within bands).
LENGTH_BANDS = (10, 13, 16, 20, 25, 32, 40, 50, 63, 80, 100, 126, 158, 200, 252, 317, 400, 504, 635, 800,
                1000, 1260, 1600, 2000)


def length_matched(df: pd.DataFrame, strata: list, label: str, text: str, *, key: str,
                   bands: tuple = LENGTH_BANDS, cap: int | None = None) -> pd.DataFrame:
    """Keep, per stratum and word-count band, equally many rows of each label (chosen by a stable hash of
    key), so text length can't stand in for authorship (Increment 4d: MAGA's human texts run far shorter
    than its AI ones). A band holding only one label is dropped.

    cap keeps at most that many rows per stratum and label, shared out over the bands in proportion to
    their matched sizes, so every band keeps equal counts. Capping afterwards with take_per_stratum
    undid the match (R12-1): both use the same hash order, so it kept each label's majority bands."""
    import numpy as np

    words = df[text].map(lambda t: len(str(t).split())).to_numpy()
    order = df[key].astype(str).map(lambda k: hashlib.sha256(k.encode("utf-8")).hexdigest())
    work = df.assign(_band=np.searchsorted(np.asarray(bands), words, side="right"), _order=order)
    work = work.sort_values("_order")
    labels = df[label].nunique()
    parts = []
    for _, stratum in work.groupby(list(strata), sort=False):
        groups, sizes = [], []
        for _, group in stratum.groupby("_band", sort=True):
            counts = group[label].value_counts()
            if len(counts) == labels:
                groups.append(group)
                sizes.append(int(counts.min()))
        for group, keep in zip(groups, _share(sizes, cap)):
            parts.append(group.groupby(label, group_keys=False).head(keep))
    if not parts:
        return df.iloc[0:0]
    return pd.concat(parts).drop(columns=["_band", "_order"])


def _share(sizes: list[int], cap: int | None) -> list[int]:
    """Split cap over bands in proportion to sizes (largest remainder), never above a band's size."""
    total = sum(sizes)
    if cap is None or total <= cap:
        return sizes
    exact = [size * cap / total for size in sizes]
    keep = [int(x) for x in exact]
    for i in sorted(range(len(sizes)), key=lambda i: exact[i] - keep[i], reverse=True)[:cap - sum(keep)]:
        keep[i] += 1
    return keep


def make_sample(
    spec: SourceSpec,
    *,
    text: str,
    label: Label,
    role: Role,
    record_id: str,
    domain: str,
    group_key: str,
    verification: str = "dataset_label",
    generator: Optional[str] = None,
    writer_group: str = "unknown",
    transform: str = "none",
    licence: Optional[tuple[str, LicenceClass]] = None,
) -> Optional[Sample]:
    """licence overrides the source licence for sources that record one per row."""
    text = (text or "").strip() if isinstance(text, str) else ""
    if not text:
        return None
    # pandas may hand enum values back as plain strings.
    label, role = Label(label), Role(role)
    return Sample(
        sample_id=sample_id(text),
        text=text,
        label=label,
        role=role,
        source_dataset=spec.name,
        source_record_id=str(record_id),
        licence=licence[0] if licence else spec.licence,
        licence_class=licence[1] if licence else spec.licence_class,
        verification=verification,
        generator=generator if label is Label.AI else None,
        generator_family=generator_family(generator) if label is Label.AI and generator else None,
        domain=domain,
        writer_group=writer_group,
        transform=transform,
        group_key=f"{spec.name}:{group_key}",
    )


_MARKDOWN = re.compile(r"\*\*|__|^[ 	]*#{1,6}[ 	]+", re.MULTILINE)
_PERMISSIVE_LICENCES = ("mit", "apache")


def strip_markdown(text: str) -> str:
    """Drop emphasis and heading markers, applied to both labels of a paired source so formatting
    can't stand in for authorship (Increment 4c)."""
    return _MARKDOWN.sub("", text or "")


def generated_licence(model_licence: str) -> tuple:
    name = (model_licence or "unknown").strip()
    permissive = any(k in name.lower() for k in _PERMISSIVE_LICENCES)
    return (f"generated locally; model licence: {name[:120]}",
            LicenceClass.PERMISSIVE if permissive else LicenceClass.UNKNOWN)


def compact(samples: Iterable[Optional[Sample]]) -> Iterable[Sample]:
    return (s for s in samples if s is not None)
