"""Threshold selection keeps every verified slice within its error bounds."""

import numpy as np
import pandas as pd

from src.authorship import calibration as cal
from src.bakeoff.metrics import upper_bound


def _slice(name, n, human_mean, ai_mean, words=120, seed=0):
    rng = np.random.default_rng(seed)
    h = rng.normal(human_mean, 1.0, n)
    a = rng.normal(ai_mean, 1.0, n)
    return pd.DataFrame({"source_dataset": name, "words": words,
                         "y": [0] * n + [1] * n, "raw": np.concatenate([h, a])})


def _data():
    easy = _slice("easy", 1500, -3, 3, seed=1)
    hard = _slice("hard", 1500, -1, 1.5, seed=2)
    df = pd.concat([easy, hard], ignore_index=True)
    a, b = cal.fit_platt(df["raw"].to_numpy(), df["y"].to_numpy())
    return df.assign(prob=cal.probability(df["raw"], a, b))


def test_band_of():
    assert cal.band_of(10) is None and cal.band_of(25) == 25 and cal.band_of(99) == 50 and cal.band_of(5000) == 200


def test_platt_is_monotone():
    a, b = cal.fit_platt(np.array([-2.0, -1, 1, 2]), np.array([0, 0, 1, 1]))
    p = cal.probability([-3, 0, 3], a, b)
    assert p[0] < p[1] < p[2]


def test_shared_thresholds_bound_every_slice():
    df = _data()
    bands = cal.choose_thresholds(df)
    band = next(b for b in bands if b.lower_words == 100)
    assert band.supported and band.human_max < band.ai_min
    report = cal.evaluate(df, bands)
    for name in ("easy", "hard"):
        assert report[name]["false_ai_upper95"] <= 0.01
        assert report[name]["false_human_upper95"] <= 0.10
    # The shared AI threshold is set by the harder slice, so the easy slice keeps most coverage.
    assert report["easy"]["coverage_supported"] > report["hard"]["coverage_supported"]


def test_band_without_enough_samples_is_unsupported():
    df = _data()
    bands = cal.choose_thresholds(df)
    assert not next(b for b in bands if b.lower_words == 25).supported


def test_uninformative_score_stays_safe_by_almost_never_deciding():
    # Bounds alone cannot reject a useless score; the coverage target (>=40%) does.
    df = _slice("noise", 1500, 0, 0, seed=3).assign(prob=lambda d: 1 / (1 + np.exp(-d["raw"])))
    bands = cal.choose_thresholds(df)
    report = cal.evaluate(df, bands)["noise"]
    assert report["false_ai_upper95"] <= 0.01 and report["false_human_upper95"] <= 0.10
    assert report["coverage_supported"] < 0.10


def test_flag_only_never_labels_human():
    df = _data()
    bands = cal.choose_thresholds(df)
    report = cal.evaluate(df, bands, flag_only=True)
    assert all(r["false_human"] == 0 for r in report.values())


def test_upper_bound_rule_of_three_sanity():
    assert abs(upper_bound(0, 300) - 0.00994) < 1e-3


def test_margin_target_does_not_block_a_band_with_a_small_slice():
    # 326 humans cannot show 0.8% even with zero errors (zero-error bound ~0.92%).
    small = _slice("small", 326, -4, 4, seed=5)
    big = _slice("big", 1500, -3, 3, seed=6)
    df = pd.concat([small, big], ignore_index=True)
    df = df.assign(prob=1 / (1 + np.exp(-df["raw"])))
    band = next(b for b in cal.choose_thresholds(df, max_fpr=0.008, max_fnr=0.08) if b.lower_words == 100)
    assert band.supported
    report = cal.evaluate(df, [band])
    assert report["big"]["false_ai_upper95"] <= 0.008
    assert report["small"]["false_ai_upper95"] <= 0.01


def test_ai_threshold_can_sit_just_above_the_top_human_score():
    # The strictest slice allows no false AI and holds the highest human score overall; the
    # threshold must land just above that score, not jump past 1.0 and drop the band.
    strict = _slice("strict", 400, -2, 3, seed=5)
    strict.loc[strict["y"].eq(0).idxmax(), "raw"] = 2.9
    loose = _slice("loose", 2000, -2, 3, seed=6)
    df = pd.concat([strict, loose], ignore_index=True)
    a, b = cal.fit_platt(df["raw"].to_numpy(), df["y"].to_numpy())
    df = df.assign(prob=cal.probability(df["raw"], a, b))
    band = next(x for x in cal.choose_thresholds(df, max_fpr=0.008, max_fnr=0.08) if x.lower_words == 100)
    top_strict_human = df.loc[(df["source_dataset"] == "strict") & (df["y"] == 0), "prob"].max()
    assert band.supported and top_strict_human < band.ai_min < top_strict_human + 1e-9
