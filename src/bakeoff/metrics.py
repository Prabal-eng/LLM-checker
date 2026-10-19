"""Selection metrics. Thresholds always come from calibration data, never from the data they are scored on."""

from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import beta
from sklearn.metrics import roc_auc_score

TARGET_FPRS = (0.01, 0.005)


def threshold_at_fpr(human_scores: np.ndarray, fpr: float) -> float:
    """Smallest threshold whose false-positive rate on these human scores is at most fpr."""
    return float(np.quantile(human_scores, 1 - fpr, method="higher"))


def upper_bound(errors: int, n: int, confidence: float = 0.95) -> Optional[float]:
    """One-sided Clopper-Pearson upper bound on an error rate."""
    if n == 0:
        return None
    return 1.0 if errors >= n else float(beta.ppf(confidence, errors + 1, n - errors))


def auroc(y: np.ndarray, s: np.ndarray) -> Optional[float]:
    return round(float(roc_auc_score(y, s)), 4) if len(set(y)) == 2 else None


def at_threshold(y: np.ndarray, s: np.ndarray, threshold: float) -> dict:
    flagged = s > threshold
    humans, ais = y == 0, y == 1
    fp, tp = int((flagged & humans).sum()), int((flagged & ais).sum())
    return {
        "threshold": threshold,
        "fpr": round(fp / humans.sum(), 5) if humans.any() else None,
        "fpr_upper95": upper_bound(fp, int(humans.sum())),
        "tpr": round(tp / ais.sum(), 4) if ais.any() else None,
        "n_human": int(humans.sum()), "n_ai": int(ais.sum()),
    }


def fold_metrics(calib: pd.DataFrame, calib_scores: np.ndarray,
                 evaluation: pd.DataFrame, eval_scores: np.ndarray) -> dict:
    y, s = evaluation["y"].to_numpy(), np.asarray(eval_scores)
    human_calib = np.asarray(calib_scores)[calib["y"].to_numpy() == 0]
    result = {"auroc": auroc(y, s), "n": int(len(y))}
    for fpr in TARGET_FPRS:
        result[f"at_fpr_{fpr}"] = at_threshold(y, s, threshold_at_fpr(human_calib, fpr))
    for column in ("length_bucket", "source_dataset"):
        result[f"auroc_by_{column}"] = {
            str(k): auroc(y[idx], s[idx])
            for k, idx in evaluation.reset_index(drop=True).groupby(column).indices.items()
        }
    return result
