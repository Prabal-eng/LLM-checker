"""Measured rates for the verdict tiers (spec 2026-10-01, section 1).

    python -m src.report.tier_rates

For the registered model, reads the validation scores its calibration saved beside each model
(calibration_scores_<sha12>.npz) and the validation frame calibration built (validation rows plus
sentence-aligned excerpts), routes each text as the service routes it, maps it through that model's Platt
parameters and tiers it with the record's bands. Writes, per tier, the share of human and of AI validation
texts in it, for 25-100 words and for all lengths with a bar, to data/report/tier_rates.json keyed by
model_id. No GPU; only validation rows are read.
"""

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from ..authorship import registry
from ..authorship.calibration import probability
from .tiers import HIGH, LIKELY_AI, LOW, MODERATE, SECOND_LOOK, TierRules

logger = logging.getLogger(__name__)

RATES_FILE = Path(__file__).resolve().parents[2] / "data" / "report" / "tier_rates.json"
RATED_TIERS = (LIKELY_AI, HIGH, MODERATE, SECOND_LOOK, LOW)
SLICES = {"25-100": (25, 100), "all": (0, None)}
NOTES = (
    "Rates are measured on public validation writing, not on rater rationales; rater text may differ.",
    "Validation includes sentence-aligned excerpts of longer texts, as calibration did.",
    "Later-window vetoes on long texts and disguised-character handling are not replayed.",
)


def scored_frame(val: pd.DataFrame, main: dict, short: Optional[dict], record: dict) -> pd.DataFrame:
    """val (sample_id, y, words) with each text's probability from the model the service would route it to."""
    def probs(cache: dict, platt) -> pd.Series:
        series = pd.Series(probability(cache["raw"], *platt), index=np.asarray(cache["ids"], dtype=str))
        if not series.index.is_unique:
            raise RuntimeError("a score cache repeats a sample id")
        return series

    ranges = [tuple(r) for r in (record.get("short_model") or {}).get("ranges", [])]
    routed = val["words"].map(lambda w: any(lo <= w < (hi if hi is not None else np.inf) for lo, hi in ranges))
    prob = val["sample_id"].map(probs(main, record["calibration"]["platt"]))
    if short is not None and routed.any():
        prob = prob.where(~routed, val["sample_id"].map(probs(short, record["short_model"]["platt"])))
    if prob.isna().any():
        raise RuntimeError(f"{int(prob.isna().sum())} validation texts have no saved score; re-run calibration")
    return val.assign(prob=prob.to_numpy(dtype=float))


def rate_table(frame: pd.DataFrame, rules: TierRules, policy) -> dict:
    """Per slice: how many human and AI texts have a bar, and each tier's share of them."""
    bars = frame["words"].map(lambda w: getattr(policy.band_for(int(w)), "ai_min_score", np.nan))
    tiers = [LIKELY_AI if p >= b else rules.by_score(p) for p, b in zip(frame["prob"], bars.fillna(np.inf))]
    frame = frame.assign(bar=bars, tier=tiers)
    out = {}
    for name, (lo, hi) in SLICES.items():
        part = frame[frame["bar"].notna() & (frame["words"] >= lo) & ((frame["words"] < hi) if hi else True)]
        humans, ais = part[part["y"] == 0], part[part["y"] == 1]
        share = lambda group, tier: round(float((group["tier"] == tier).mean()), 4) if len(group) else None  # noqa: E731
        out[name] = {"human_texts": int(len(humans)), "ai_texts": int(len(ais)),
                     "tiers": {t: {"human": share(humans, t), "ai": share(ais, t)} for t in RATED_TIERS}}
    return out


def validation_frame(record: dict) -> pd.DataFrame:
    """The validation frame calibration scored: validation rows plus their excerpts (sample_id, y, words)."""
    from ..authorship.calibrate import load_split, with_excerpts
    from ..corpus.paths import VERSION
    from ..corpus.text import MODEL_TEXT_VIEW

    if record.get("corpus_version") != VERSION or record.get("text_normalization") != MODEL_TEXT_VIEW:
        raise RuntimeError("rates can only be rebuilt for a model calibrated on this branch's corpus and text view")
    val = load_split("val")
    val = pd.concat([val.assign(excerpt=False), with_excerpts(val)], ignore_index=True)
    return val[["sample_id", "y", "words"]]


def _cache(model_dir: Path, sha: str) -> Tuple[dict, str]:
    path = Path(model_dir).parent / f"calibration_scores_{sha[:12]}.npz"
    with np.load(path) as saved:
        cache = {"ids": saved["val_ids"].astype(str), "raw": saved["val_raw"].astype(float)}
    return cache, registry.sha256_file(path)


def saved_scores(record: dict) -> Tuple[dict, Optional[dict], dict]:
    """(main cache, short cache or None, {name: file sha256}) for the record's models."""
    main, main_sha = _cache(registry.artifact_dir(record), record["artifact"]["sha256"])
    parts = registry.short_parts(record)
    if not parts:
        return main, None, {"main": main_sha}
    short, short_sha = _cache(parts[0], record["short_model"]["artifact"]["sha256"])
    return main, short, {"main": main_sha, "short": short_sha}


def build(record: Optional[dict] = None, path: Path = RATES_FILE, rules: TierRules = TierRules()) -> dict:
    record = record or registry.load()
    if not record:
        raise RuntimeError("no registered model")
    main, short, sources = saved_scores(record)
    frame = scored_frame(validation_frame(record), main, short, record)
    entry = {"model_id": record["model_id"], "measured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
             "bands": rules.as_dict(), "slices": rate_table(frame, rules, registry.build_policy(record)),
             "sources": sources, "notes": list(NOTES)}
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    data[record["model_id"]] = entry
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return entry


def load_rates(model_id: str, path: Path = RATES_FILE, rules: Optional[TierRules] = None) -> Optional[dict]:
    """The model's measured rates, or None when they were not measured (or were measured for other bands)."""
    if not Path(path).exists():
        return None
    entry = json.loads(Path(path).read_text(encoding="utf-8")).get(model_id)
    if entry and rules is not None and entry.get("bands") != rules.as_dict():
        return None
    return entry


def main(argv=None) -> None:
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    entry = build()
    for name, part in entry["slices"].items():
        print(f"{name}: {part['human_texts']} human, {part['ai_texts']} AI texts")
        for tier, rate in part["tiers"].items():
            print(f"  {tier:28s} human {rate['human']:.4f}  AI {rate['ai']:.4f}")


if __name__ == "__main__":
    main()
