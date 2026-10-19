"""Increment 4d optional step: style-aware thresholds, measured on validation only.

    python -m src.authorship.registers --candidate modernbert-base

A TF-IDF logistic classifier assigns each text one of five registers; it trains on train rows
labelled by source and domain. At check time the register would be predicted from the text, never
taken from the source, so validation rows are assigned by prediction too. A bar is set per length band
x register cell on that cell's own validation humans at the pre-registered targets; a cell with fewer
than 600 validation humans keeps the pooled band bar. The step is adopted only if it adds at least 2
points of validation detection with every cell within target. Writes a text-free report. Run after
calibration: it reuses calibration's cached validation scores and bars.
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ..bakeoff.metrics import upper_bound
from ..corpus.paths import REPO_BENCHMARK_DIR, VERSION
from ..corpus.text import MODEL_TEXT_VIEW, text_view
from . import calibration as cal

logger = logging.getLogger(__name__)

REGISTERS = ("qa_forum", "assistant", "news_expository", "essay_academic", "social_review")
MIN_CELL_HUMANS = 600
MIN_GAIN_POINTS = 2.0
FIT_PER_CELL = 10_000  # train rows per (register, label) for the classifier

# A source's register, then (source, domain) pairs that differ. Creative writing (stories, poetry,
# books) has no register of its own; it reads as long-form prose, so it sits with essays.
_SOURCE_REGISTER = {
    "aita2026": "social_review", "arena140k": "assistant", "beemo": "assistant", "dolly": "assistant",
    "hc3": "qa_forum", "no_robots": "assistant", "oasst1": "assistant", "stackexchange": "qa_forum",
    "openturingbench": "news_expository",
}
_PAIR_REGISTER = {
    ("hc3", "wiki_csai"): "news_expository",
    ("mage", "cmv"): "essay_academic", ("mage", "cnn"): "news_expository", ("mage", "dialogsum"): "news_expository",
    ("mage", "eli5"): "qa_forum", ("mage", "hswag"): "essay_academic", ("mage", "imdb"): "social_review",
    ("mage", "pubmed"): "essay_academic", ("mage", "roct"): "essay_academic", ("mage", "sci_gen"): "essay_academic",
    ("mage", "squad"): "news_expository", ("mage", "tldr"): "news_expository", ("mage", "wp"): "essay_academic",
    ("mage", "xsum"): "news_expository", ("mage", "yelp"): "social_review",
    ("mirage", "academic"): "essay_academic", ("mirage", "comment"): "social_review",
    ("mirage", "news"): "news_expository", ("mirage", "website"): "news_expository",
    ("raid", "abstracts"): "essay_academic", ("raid", "books"): "essay_academic",
    ("raid", "news"): "news_expository", ("raid", "poetry"): "essay_academic",
    ("raid", "recipes"): "news_expository", ("raid", "reddit"): "qa_forum",
    ("raid", "reviews"): "social_review", ("raid", "wiki"): "news_expository",
    ("maga", "amazon_reviews"): "social_review", ("maga", "cc_news"): "news_expository",
    ("maga", "natural_questions"): "qa_forum", ("maga", "npr_news"): "news_expository",
    ("maga", "reddit"): "qa_forum", ("maga", "s2orc"): "essay_academic",
    ("maga", "trustpilot_reviews"): "social_review", ("maga", "wikipedia"): "news_expository",
    ("maga", "yahoo_answers"): "qa_forum",
    ("dactyl", "abstracts"): "essay_academic", ("dactyl", "news"): "news_expository",
    ("dactyl", "reviews"): "social_review", ("dactyl", "student_essays"): "essay_academic",
    ("dactyl", "tweets"): "social_review",
}


def register_label(source: str, domain: str) -> str:
    """The register a (source, domain) trains as; an unmapped pair fails, so no source trains under a
    guessed register."""
    if (source, domain) in _PAIR_REGISTER:
        return _PAIR_REGISTER[(source, domain)]
    if source in _SOURCE_REGISTER:
        return _SOURCE_REGISTER[source]
    raise KeyError(f"no register for {source}:{domain}")


class RegisterClassifier:
    """Word 1-2-gram TF-IDF and a multinomial logistic regression: CPU only, well under a millisecond a text."""

    def fit(self, texts, labels) -> "RegisterClassifier":
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline

        self.model = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200_000,
                                                   sublinear_tf=True),
                                   LogisticRegression(max_iter=2000, C=4.0))
        self.model.fit(list(texts), list(labels))
        return self

    def predict(self, texts) -> np.ndarray:
        return np.asarray(self.model.predict(list(texts)))


def cell_bars(val: pd.DataFrame, pooled: dict, max_fpr: float, max_fnr: float) -> dict:
    """ai_min per (band, register): the cell's own bar where it has MIN_CELL_HUMANS validation humans and
    calibration supports it there, else the pooled band bar. val needs y, words, prob, register and
    source_dataset."""
    val = val.assign(band=val["words"].map(cal.band_of))
    out = {}
    for (band, register), cell in val.groupby(["band", "register"]):
        humans = int((cell["y"] == 0).sum())
        bar, own = pooled.get(band), False
        if humans >= MIN_CELL_HUMANS:
            fitted = {b.lower_words: b for b in cal.choose_thresholds(cell, max_fpr=max_fpr, max_fnr=max_fnr)}
            mine = fitted.get(band)
            if mine is not None and mine.supported:
                bar, own = mine.ai_min, True
        out[(int(band), register)] = {"ai_min": bar, "own": own, "humans": humans,
                                      "ais": int((cell["y"] == 1).sum())}
    return out


def _flagged(val: pd.DataFrame, bars: dict) -> np.ndarray:
    keys = zip(val["words"].map(cal.band_of), val["register"])
    ai_min = np.array([np.inf if bars.get((b, r), {}).get("ai_min") is None else bars[(b, r)]["ai_min"]
                       for b, r in keys], dtype=float)
    return val["prob"].to_numpy() >= ai_min


def register_gain(val: pd.DataFrame, pooled: dict, max_fpr: float, max_fnr: float) -> dict:
    """Validation detection under pooled band bars and under per-cell bars, and whether every cell with
    MIN_HUMANS humans keeps its false-AI bound within target under the per-cell bars."""
    bars = cell_bars(val, pooled, max_fpr, max_fnr)
    pooled_bars = {k: {"ai_min": pooled.get(k[0])} for k in bars}
    ai = val["y"].to_numpy() == 1
    before = float(_flagged(val, pooled_bars)[ai].mean())
    after = float(_flagged(val, bars)[ai].mean())
    flagged = _flagged(val, bars)
    cells, in_target = [], True
    for (band, register), entry in bars.items():
        rows = (val["words"].map(cal.band_of).to_numpy() == band) & (val["register"].to_numpy() == register)
        humans = rows & ~ai
        n = int(humans.sum())
        false_ai = int((flagged & humans).sum())
        bound = upper_bound(false_ai, n)
        ok = n < cal.MIN_HUMANS or bound <= cal.slice_target(max_fpr, n)
        in_target &= ok
        cells.append({"band": band, "register": register, **entry, "false_ai": false_ai,
                      "false_ai_upper95": bound, "within_target": ok})
    # The gate is per source, and a source spread thinly over registers constrains no cell's bar, so each
    # (band, source) slice with MIN_HUMANS humans must stay within target too (final review I5).
    frame = val.assign(_band=val["words"].map(cal.band_of).to_numpy(), _flag=flagged)
    sources, sources_ok = [], True
    for (band, source), part in frame[frame["y"] == 0].groupby(["_band", "source_dataset"]):
        n, false_ai = len(part), int(part["_flag"].sum())
        bound = upper_bound(false_ai, n)
        ok = n < cal.MIN_HUMANS or bound <= cal.slice_target(max_fpr, n)
        sources_ok &= ok
        sources.append({"band": int(band), "source": source, "humans": n, "false_ai": false_ai,
                        "false_ai_upper95": bound, "within_target": ok})
    gain = round(100 * (after - before), 2)
    return {"pooled_detection": round(before, 4), "register_detection": round(after, 4), "gain_points": gain,
            "every_cell_in_target": bool(in_target), "every_source_in_target": bool(sources_ok),
            "cells": cells, "sources": sources,
            "adopted": bool(gain >= MIN_GAIN_POINTS and in_target and sources_ok)}


def report_file(candidate: str) -> Path:
    return REPO_BENCHMARK_DIR / VERSION / "model" / candidate / "registers.json"


def _served_probabilities(val: pd.DataFrame, candidate: str, calibration: dict) -> pd.Series:
    """Each validation row's probability from the model that serves its band, read from calibration's
    cached raw scores and mapped through that model's recorded Platt map."""
    from .train import record_file

    serving = {b["lower_words"]: b.get("model", "main") for b in calibration["policy_bands"]}
    probs = {}
    for variant in ("main", "short"):
        path = record_file(variant=variant, candidate=candidate)
        if variant not in calibration or not path.exists():
            continue
        training = json.loads(path.read_text(encoding="utf-8"))
        cache = Path(training["artifact_path"]).parent / f"calibration_scores_{training['artifact_sha256'][:12]}.npz"
        scores = np.load(cache)
        probs[variant] = pd.Series(cal.probability(scores["val_raw"], *calibration[variant]["platt"]),
                                   index=scores["val_ids"].astype(str))
    band = val["words"].map(cal.band_of)
    return pd.Series([probs[serving.get(b, "main")].get(sid, np.nan) for b, sid in zip(band, val["sample_id"])],
                     index=val.index)


def run(candidate: str) -> dict:
    from ..corpus.sources._common import take_per_stratum
    from .calibrate import calibration_file, load_split, with_excerpts

    calibration = json.loads(calibration_file(candidate=candidate).read_text(encoding="utf-8"))
    targets = calibration["targets"]
    view = text_view(MODEL_TEXT_VIEW)
    train = load_split("train")  # its columns include domain
    train = train.assign(register=[register_label(s, d) for s, d in zip(train["source_dataset"], train["domain"])])
    fit = take_per_stratum(train, ["register", "y"], FIT_PER_CELL, "sample_id")
    classifier = RegisterClassifier().fit(fit["text"].map(view), fit["register"])
    val = load_split("val")
    val = pd.concat([val.assign(excerpt=False), with_excerpts(val)], ignore_index=True)
    val = val.assign(register=classifier.predict(val["text"].map(view)),
                     prob=_served_probabilities(val, candidate, calibration)).dropna(subset=["prob"])
    pooled = {b["lower_words"]: b["ai_min"] for b in calibration["policy_bands"] if b["supported"]}
    report = register_gain(val, pooled, targets["max_false_ai_upper95"], targets["max_false_human_upper95"])
    report["classifier_fit_rows"] = len(fit)
    report["locked_split_reads"] = 0
    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    out = report_file(candidate)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True, default=float) + "\n", encoding="utf-8")
    return report


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--candidate", required=True)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    report = run(args.candidate)
    print(json.dumps({k: report[k] for k in ("pooled_detection", "register_detection", "gain_points",
                                             "every_cell_in_target", "adopted")}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
