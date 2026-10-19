"""Shortcut probe: how well do surface features alone separate the labels?

A high AUC from length, punctuation, or formatting means a detector can score well
without learning authorship, and the affected sources need rebalancing or slicing.

The gating probe reads the model's view (whitespace collapsed, typography as ASCII) and runs both as one shared
probe and as one probe per source (Ralph R07-7: a shared probe on raw text could not check that
claim, and a per-source probe separated MIRAGE where the shared one did not). It is fitted on
train and scored on validation, so building the corpus never reads test text. The raw-text
shared probe is kept for reference.
"""

import re

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .schema import Label, Role, Split
from .text import MODEL_TEXT_VIEW, text_view

WARN_AUC = 0.85
FEATURES = ("log_chars", "log_words", "mean_word_len", "newline_rate", "upper_ratio", "digit_ratio",
            "punct_ratio", "non_ascii_ratio", "starts_lower", "no_terminal_punct", "placeholders",
            "has_line_break", "has_blank_line", "log_line_breaks",
            # Amendment 7 (Ralph R09-6): formatting cues, read on the model's view
            "asterisks", "backticks", "spaced_dashes", "inline_numbered", "bullets")
_BLANK_LINE = re.compile(r"\n\s*\n")
_PLACEHOLDER = re.compile(r"\[(EMAIL|PHONE|CARD|IP)\]")
_SPACED_DASH = re.compile(r"(?<!\S)-{1,3}(?!\S)")
_INLINE_NUMBERED = re.compile(r"(?<!\S)\d{1,3}[.)](?=\s)")
_BULLETS = re.compile("[\u2022\u2023\u2043\u25aa\u25e6]")


def surface_features(texts: pd.Series) -> np.ndarray:
    rows = []
    for t in texts:
        n = max(len(t), 1)
        tokens = t.split()
        rows.append((
            np.log1p(len(t)), np.log1p(len(tokens)),
            sum(map(len, tokens)) / max(len(tokens), 1),
            t.count("\n") / n, sum(c.isupper() for c in t) / n, sum(c.isdigit() for c in t) / n,
            sum(not c.isalnum() and not c.isspace() for c in t) / n, sum(ord(c) > 127 for c in t) / n,
            float(t[:1].islower()), float(t.rstrip()[-1:] not in ".!?\"'"), len(_PLACEHOLDER.findall(t)),
            # Layout: v1 found blank lines in public AI text but not in its human text.
            float("\n" in t), float(bool(_BLANK_LINE.search(t))), np.log1p(t.count("\n")),
            t.count("*") / n * 100, t.count("`") / n * 100, len(_SPACED_DASH.findall(t)) / n * 100,
            len(_INLINE_NUMBERED.findall(t)) / n * 100, len(_BULLETS.findall(t)) / n * 100,
        ))
    return np.array(rows, dtype=float)


def _probe():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))


def _shared(train: pd.DataFrame, held: pd.DataFrame, texts) -> tuple:
    model = _probe().fit(surface_features(texts(train)), train["label"] == Label.AI.value)
    scores = pd.Series(model.predict_proba(surface_features(texts(held)))[:, 1], index=held.index)
    y = held["label"] == Label.AI.value
    per_source = {}
    for source, idx in held.groupby("source_dataset").groups.items():
        if y[idx].nunique() == 2:
            per_source[source] = round(float(roc_auc_score(y[idx], scores[idx])), 4)
    return round(float(roc_auc_score(y, scores)), 4), per_source


def shortcut_probe(df: pd.DataFrame) -> dict:
    core = df[df["role"] == Role.CORE.value]
    train = core[core["split"] == Split.TRAIN.value]
    held = core[core["split"] == Split.VAL.value]
    if train["label"].nunique() < 2 or held["label"].nunique() < 2:
        return {"status": "skipped", "reason": "train and validation each need both labels"}

    def model_view(part):
        return part["text"].map(text_view(MODEL_TEXT_VIEW))

    overall, per_source = _shared(train, held, model_view)
    raw_overall, raw_per_source = _shared(train, held, lambda part: part["text"])
    own = {}
    for source, part in train.groupby("source_dataset"):
        mine = held[held["source_dataset"] == source]
        if part["label"].nunique() == 2 and mine["label"].nunique() == 2:
            model = _probe().fit(surface_features(model_view(part)), part["label"] == Label.AI.value)
            scores = model.predict_proba(surface_features(model_view(mine)))[:, 1]
            own[source] = round(float(roc_auc_score(mine["label"] == Label.AI.value, scores)), 4)
    flagged = sorted({s for probe in (per_source, own) for s, auc in probe.items() if auc > WARN_AUC})
    return {
        "status": "ok", "features": list(FEATURES), "text_view": MODEL_TEXT_VIEW, "evaluated_on": "val",
        "overall_auc": overall, "per_source_auc": per_source, "per_source_own_probe_auc": own,
        "raw_text_reference": {"overall_auc": raw_overall, "per_source_auc": raw_per_source},
        "warn_threshold": WARN_AUC,
        "flagged_sources": flagged + (["<overall>"] if overall > WARN_AUC else []),
        "train_ai_share_by_source": {
            s: round(float(v), 4)
            for s, v in (train["label"] == Label.AI.value).groupby(train["source_dataset"]).mean().items()
        },
    }


AI_ONLY_SAMPLE = 4000  # rows per side, so the probe stays quick on large sources
# The models read at most 256 tokens (about 200 words), so the AI-only probe reads the same window: length
# beyond it is invisible to a model and is not a shortcut it can learn.
PROBE_WORDS = 200


def _in_pool(df: pd.DataFrame, entry: str) -> pd.Series:
    source, _, domain = entry.partition(":")
    mask = df["source_dataset"] == source
    return mask & (df["domain"] == domain) if domain else mask


def ai_only_probe(df: pd.DataFrame, pools: dict) -> dict:
    """For each AI-only source, the cross-validated AUC of the surface probe separating its AI rows from the
    human rows it is balanced against (Increment 4d). pools maps a source to entries naming a source or a
    source:domain. Read on the model's view and window (PROBE_WORDS), before splitting, so the build can drop a giveaway source
    before any training; a source with no human pool rows gets no probe."""
    core = df[df["role"] == Role.CORE.value]
    model_view = text_view(MODEL_TEXT_VIEW)

    def view(text: str) -> str:
        return " ".join(model_view(text).split()[:PROBE_WORDS])

    out = {}
    for source, entries in pools.items():
        ai = core[(core["source_dataset"] == source) & (core["label"] == Label.AI.value)]
        pool = pd.Series(False, index=core.index)
        for entry in entries:
            pool |= _in_pool(core, entry)
        human = core[pool & (core["label"] == Label.HUMAN.value)]
        if ai.empty or human.empty:
            continue
        ai = ai.sample(n=min(len(ai), AI_ONLY_SAMPLE), random_state=0)
        human = human.sample(n=min(len(human), AI_ONLY_SAMPLE), random_state=0)
        both = pd.concat([ai, human])
        y = (both["label"] == Label.AI.value).to_numpy()
        folds = min(5, int(y.sum()), int((~y).sum()))
        if folds < 2:
            continue
        scores = cross_val_predict(_probe(), surface_features(both["text"].map(view)), y, cv=folds,
                                   method="predict_proba")[:, 1]
        out[source] = round(float(roc_auc_score(y, scores)), 4)
    return out
