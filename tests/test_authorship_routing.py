"""Short-text routing: which model serves each band under 100 words, decided on validation."""

import numpy as np
import pandas as pd

from src.authorship import calibration as cal
from src.authorship.calibrate import band_coverage, combine_bands, route, with_excerpts


def _band(words, separation, n=1500, seed=0):
    rng = np.random.default_rng(seed)
    raw = np.concatenate([rng.normal(-separation, 1.0, n), rng.normal(separation, 1.0, n)])
    df = pd.DataFrame({"source_dataset": "s", "words": words, "y": [0] * n + [1] * n, "raw": raw})
    a, b = cal.fit_platt(df["raw"].to_numpy(), df["y"].to_numpy())
    return df.assign(prob=cal.probability(df["raw"], a, b))


def test_specialist_serves_only_bands_it_wins_by_the_margin():
    main = pd.concat([_band(30, 1.0, seed=1), _band(60, 2.5, seed=2), _band(150, 3.0, seed=3)], ignore_index=True)
    short = pd.concat([_band(30, 2.5, seed=4), _band(60, 2.5, seed=2)], ignore_index=True)
    main_bands, short_bands = cal.choose_thresholds(main), cal.choose_thresholds(short)
    main_cov, short_cov = band_coverage(main, main_bands), band_coverage(short, short_bands)
    assert short_cov[25] > main_cov[25] + 0.01
    assert abs(short_cov[50] - main_cov[50]) < 0.01  # identical scores: a tie keeps the main model
    routed = route(main_cov, short_cov, short_bands)
    assert routed == [25]
    combined = combine_bands(main_bands, short_bands, routed)
    assert [b["model"] for b in combined] == ["short", "main", "main", "main"]
    assert combined[0]["ai_min"] == next(b.ai_min for b in short_bands if b.lower_words == 25)


def test_specialist_never_serves_long_or_unsupported_bands():
    unsupported = [cal.BandThresholds(25, 50, False)]
    assert route({25: 0.1}, {25: 0.9}, unsupported) == []
    long_band = [cal.BandThresholds(100, 200, True, ai_min=0.9, human_max=0.1)]
    assert route({100: 0.1}, {100: 0.9}, long_band) == []


def test_validation_excerpts_stay_under_100_policy_words():
    text = " ".join(f"Sentence {i} has exactly seven words here." for i in range(60))
    df = pd.DataFrame({"sample_id": ["a", "b"], "text": [text, "tiny text"], "y": [1, 0],
                       "source_dataset": "s", "words": [420, 2]})
    ex = with_excerpts(df)
    assert len(ex) == 1 and ex["excerpt"].all()
    assert (ex["words"] >= 25).all() and (ex["words"] < 100).all()
