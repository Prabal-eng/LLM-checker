# Legacy detector performance check — 2026-09-26

## Result

The current model scores extremely well on its existing test split, but that result is not a credible estimate of general authorship detection. Most positive test rows are synthetic template text, and hundreds of those exact texts also occur in training. On an exact-duplicate-filtered diagnostic subset, the model retained high apparent coverage and made few observed errors, but the AI sample is too small and its provenance too weak to support the release reliability claims.

The current `/api/v2/check-text` and default spreadsheet paths keep the model validation gate closed. Their live product behavior is therefore 100% Inconclusive for sufficiently long ordinary inputs (decisive coverage 0%), regardless of the offline model scores below.

## Method

- Loaded the existing `train.csv`, `val.csv`, and `test.csv`; did not retrain, alter model files, or tune thresholds.
- Evaluated the checked-in `TrainedInDomainDetector` on all 1,133 test rows.
- Applied the current provisional policy directly to its scores: `<= 0.30` Likely human-written, `>= 0.80` Likely AI-generated, otherwise Inconclusive.
- Measured exact overlap after case folding and whitespace normalization using SHA-256 of each text.
- Calculated one-sided 95% Clopper–Pearson upper bounds for observed error rates. These bounds describe only the sampled labels and do not correct for label or sampling bias.

Reproduction from the repository root:

```powershell
@'
import hashlib
import numpy as np
import pandas as pd
from pathlib import Path
from src.detectors.trained_detector import TrainedInDomainDetector

frames = {n: pd.read_csv(Path('data/training') / f'{n}.csv') for n in ('train', 'val', 'test')}
test = frames['test'].reset_index(drop=True)
model = TrainedInDomainDetector()
results = model.detect_batch(test.text.astype(str).tolist())
scores = np.array([r.ai_score for r in results])
ai = scores >= 0.80
human = scores <= 0.30
labels = test.label.astype(int).to_numpy()
print('rows', len(test), 'AI labels', int(ai.sum()), 'human labels', int(human.sum()), 'inconclusive', int((~(ai | human)).sum()))
print('human to AI', int((ai & (labels == 0)).sum()), 'AI to human', int((human & (labels == 1)).sum()))
normalize = lambda t: hashlib.sha256(' '.join(str(t).casefold().split()).encode()).hexdigest()
train_texts = set(frames['train'].text.map(normalize))
overlap = test.text.map(normalize).isin(train_texts)
print('exact normalized train/test overlap rows', int(overlap.sum()))
'@ | python -
```

## Measurements

Existing split composition:

| Split | Rows | Synthetic AI | Human Gempix | Human personality | Inferred AI copypasta |
|---|---:|---:|---:|---:|---:|
| Train | 5,280 | 2,599 | 2,351 | 295 | 35 |
| Validation | 1,131 | 550 | 504 | 70 | 7 |
| Test | 1,133 | 572 | 489 | 63 | 9 |

Exact normalized overlap exists across every pair of splits: train/validation has 511 overlapping validation rows, train/test has 534 overlapping test rows, and validation/test has 477 overlapping test rows. All overlapping rows in the later split are labeled `synthetic_ai`.

On the full test set, the legacy 0.5 classification cutoff scored 99.82% accuracy (1,131/1,133) and ROC-AUC 0.99997. Under the selective policy:

| Measure | Observed |
|---|---:|
| Likely AI-generated | 580 / 1,133 |
| Likely human-written | 546 / 1,133 |
| Inconclusive | 7 / 1,133 |
| Decisive coverage | 99.38% |
| Human labeled Likely AI | 1 / 552 (0.18%); one-sided 95% upper bound 0.86% |
| AI labeled Likely human | 0 / 581 (0%); one-sided 95% upper bound 0.51% |

After removing the 534 test rows that exactly overlap training, the diagnostic subset has 599 rows: 552 human, 38 synthetic AI, and 9 inferred AI copypasta. It has 98.83% decisive coverage, 1/552 observed human-to-AI errors (0.18%, upper bound 0.86%), and 0/47 observed AI-to-human errors (0%, upper bound 6.18%). The 47 positive non-overlap examples are not a verified, diverse AI-authorship sample; nine are in the inferred copypasta category.

The only false Likely AI result on this test split is in the small `human_personality` slice: 1/63. Its one-sided 95% upper bound is 7.31%. The no-overlap AI error bounds are 7.58% for the 38 synthetic positives and 28.31% for the 9 inferred copypasta positives. This does not establish the required per-major-slice reliability.

## Assessment

These measurements do **not** validate a reliable LLM detector. Exact train/test text reuse explains 534 test predictions with direct leakage, synthetic positives dominate the test set, and the purported real-AI category is both tiny and inferred. Removing exact copies still leaves a biased, narrow, small positive set. There is no representative generator, domain, language, editing, or mixed-authorship coverage here.

The result supports the existing decision to keep the production model gate closed. A defensible performance claim needs verified human and AI provenance, removal of near-duplicate/template groups across splits, a locked test set, and enough examples per supported slice to meet the false-positive and false-human confidence bounds.
