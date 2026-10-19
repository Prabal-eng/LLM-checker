"""Score calibration and threshold selection with per-slice error bounds.

Length is observable when text is checked; source and domain are not. Thresholds are
therefore set per word-count band and shared across sources: within a band, the AI threshold
is the lowest score at which every verified source slice keeps its false-AI rate's 95% upper
bound within the target, and the human threshold is the highest score at which every verified
slice keeps its false-human upper bound within its target. A band with no such pair is
unsupported and always abstains.
"""

from dataclasses import asdict, dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from ..bakeoff.metrics import upper_bound

# Word-count bands (lower bound inclusive). Texts shorter than the first band never score.
BANDS = (25, 50, 100, 200)
MIN_HUMANS = 300  # below this, a zero-error slice cannot show a 1% false-AI upper bound
MIN_AIS = 100     # enough for a 10% false-human bound to be informative


def band_of(words: int) -> Optional[int]:
    """Lower bound of the band a word count falls in, or None if shorter than every band."""
    chosen = None
    for lower in BANDS:
        if words >= lower:
            chosen = lower
    return chosen


def fit_platt(raw: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Logistic map from raw scores to probabilities, fitted on validation."""
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(C=1e6, max_iter=1000).fit(raw.reshape(-1, 1), y)
    return float(model.coef_[0, 0]), float(model.intercept_[0])


def probability(raw, a: float, b: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-(a * np.asarray(raw, dtype=float) + b)))


@dataclass
class BandThresholds:
    lower_words: int
    upper_words: Optional[int]
    supported: bool
    ai_min: Optional[float] = None
    human_max: Optional[float] = None
    false_ai_slices: list = field(default_factory=list)     # slices constraining the AI threshold
    false_human_slices: list = field(default_factory=list)  # slices constraining the human threshold
    unverified_slices: list = field(default_factory=list)
    reason: str = ""


def slice_target(target: float, n: int) -> float:
    """The target a slice of n texts can be held to: never below what zero errors can show.

    A calibration margin (a target under the release bound) would otherwise be unreachable for
    small slices and block a whole band; such slices are held to their zero-error bound instead,
    which is within the release bound because slices below MIN_HUMANS are not constrained.
    """
    return max(target, upper_bound(0, n))


def _ai_threshold(scores_by_slice: dict, max_fpr: float) -> Optional[float]:
    """Lowest candidate threshold at which every slice's false-AI upper bound is within its target."""
    targets = {k: slice_target(max_fpr, len(h)) for k, h in scores_by_slice.items()}
    pool = np.concatenate(list(scores_by_slice.values()))
    # Just above each human score is a candidate too: a slice allowed no false AI needs a threshold
    # above its own top score, which may be the highest human score of all.
    candidates = np.unique(np.concatenate([pool, np.nextafter(pool, np.inf), np.array([1.0 + 1e-9])]))
    for t in candidates:
        if all(upper_bound(int((h >= t).sum()), len(h)) <= targets[k] for k, h in scores_by_slice.items()):
            return float(t)
    return None


def _human_threshold(scores_by_slice: dict, max_fnr: float, below: float) -> Optional[float]:
    """Highest threshold under `below` at which every slice's false-human upper bound is within its target."""
    targets = {k: slice_target(max_fnr, len(a)) for k, a in scores_by_slice.items()}
    pool = np.concatenate(list(scores_by_slice.values()))
    # Just under `below` is always a candidate: when no AI text scores below the AI threshold,
    # every lower threshold misses nothing.
    candidates = np.unique(np.append(pool[pool < below], np.nextafter(below, -np.inf)))[::-1]
    for h in candidates:
        if all(upper_bound(int((a <= h).sum()), len(a)) <= targets[k] for k, a in scores_by_slice.items()):
            return float(h)
    return None


def choose_thresholds(df: pd.DataFrame, *, score: str = "prob", slice_col: str = "source_dataset",
                      max_fpr: float = 0.01, max_fnr: float = 0.10,
                      min_humans: int = MIN_HUMANS, min_ais: int = MIN_AIS) -> list:
    """Per-band shared thresholds. df needs columns y, words, slice_col, and score.

    A slice constrains the AI threshold when it has min_humans human texts in the band, and the
    human threshold when it has min_ais AI texts; a band needs at least one of each.
    """
    out = []
    df = df.assign(band=df["words"].map(band_of))
    for i, lower in enumerate(BANDS):
        upper = BANDS[i + 1] if i + 1 < len(BANDS) else None
        band = df[df["band"] == lower]
        humans, ais, unverified = {}, {}, []
        for name, part in band.groupby(slice_col):
            h, a = part.loc[part["y"] == 0, score].to_numpy(), part.loc[part["y"] == 1, score].to_numpy()
            if len(h) >= min_humans:
                humans[name] = h
            if len(a) >= min_ais:
                ais[name] = a
            if len(h) < min_humans and len(a) < min_ais:
                unverified.append(name)
        result = BandThresholds(lower, upper, False, false_ai_slices=sorted(humans),
                                false_human_slices=sorted(ais), unverified_slices=unverified)
        if not humans or not ais:
            result.reason = f"needs a slice with {min_humans} human and one with {min_ais} AI texts"
        else:
            ai_min = _ai_threshold(humans, max_fpr)
            human_max = _human_threshold(ais, max_fnr, ai_min) if ai_min is not None else None
            if ai_min is None or ai_min > 1.0:
                result.reason = "no threshold keeps every slice's false-AI bound within target"
            elif human_max is None:
                result.reason = "no threshold keeps every slice's false-human bound within target"
            else:
                result.supported, result.ai_min, result.human_max = True, ai_min, human_max
        out.append(result)
    return out


def evaluate(df: pd.DataFrame, bands: list, *, score: str = "prob", slice_col: str = "source_dataset",
             flag_only: bool = False) -> dict:
    """Coverage and bounded error rates per slice under the chosen band thresholds."""
    by_band = {b.lower_words: b for b in bands}
    df = df.assign(band=df["words"].map(band_of))

    def decide(row) -> str:
        b = by_band.get(row["band"])
        if b is None or not b.supported:
            return "inconclusive"
        if row[score] >= b.ai_min:
            return "ai"
        if not flag_only and row[score] <= b.human_max:
            return "human"
        return "inconclusive"

    df = df.assign(decision=df.apply(decide, axis=1))
    report = {}
    for name, part in df.groupby(slice_col):
        h, a = part[part["y"] == 0], part[part["y"] == 1]
        false_ai, false_human = int((h["decision"] == "ai").sum()), int((a["decision"] == "human").sum())
        supported = part[part["band"].map(lambda x: x in by_band and by_band[x].supported)]
        report[name] = {
            "humans": len(h), "ais": len(a),
            "false_ai": false_ai, "false_ai_upper95": upper_bound(false_ai, len(h)),
            "false_human": false_human, "false_human_upper95": upper_bound(false_human, len(a)) if len(a) else None,
            "coverage_supported": round(float((supported["decision"] != "inconclusive").mean()), 4) if len(supported) else None,
            "supported_share": round(len(supported) / len(part), 4) if len(part) else None,
        }
    return report


def as_dicts(bands: list) -> list:
    return [asdict(b) for b in bands]
