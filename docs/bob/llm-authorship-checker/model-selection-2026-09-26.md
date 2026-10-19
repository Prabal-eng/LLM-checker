# Model selection bake-off — 2026-09-26

## Decision

**DistilRoBERTa fine-tuned on the corpus is selected** as the detector family for Increment 4 (ADR-009). It is best or level on 8 of 9 folds, keeps a 79% detection rate at a 1% false-positive rate on unseen generator families, detects the newest GPT models at 0.998 AUROC without training on them, and meets every speed target on CPU and GPU.

Its weakness is **threshold transfer to an unfamiliar kind of writing**. A threshold set for 1% false positives on the other sources gives 10.8% on MAGE. One global threshold is therefore not safe. Increment 4 must calibrate per supported domain and return `Inconclusive` outside the supported set.

No locked split was read: `locked_split_reads = 0` in `data/benchmark/v1/bakeoff/selection.json`, enforced by the Parquet filter in `src/bakeoff/data.py`.

## Method

- **Data:** train split (99,702 rows) for fitting, and a validation sample of 10,430 rows for calibration and evaluation: 1,000 per source and label, all validation rows from the held-out families, and 623 extra newest-GPT rows used only by the `recent:openai` fold.
- **Folds:**
  - `val`: in-distribution.
  - `loso:<source>` ×4: fit and calibrate without HC3, MAGE, MIRAGE, or RAID, then evaluate on it.
  - `lofo:<family>` ×3: fit without Anthropic, DeepSeek, or Google.
  - `recent:openai`: fit without GPT-4o, GPT-4o-mini and o3-mini, keep every older GPT model, and evaluate on the three newest (812 AI texts against 3,728 human).
- **Thresholds** are taken from calibration data only. The `fpr` reported for a held-out fold is the false-positive rate observed when that threshold transfers, with a one-sided 95% Clopper–Pearson upper bound.
- **Candidates:**
  - `legacy`: the shipped model, unchanged.
  - `tfidf`: hashed word and character n-grams with a streaming linear classifier.
  - `distilroberta`: `distilroberta-base`, one epoch on a 24,000-row stratified subsample of each fold's fit data, bf16, micro-batch 8 × accumulation 2.
  - Zero-shot `fast_base`, `fast_instruct` and `binoculars`: analytic Fast-DetectGPT and Binoculars from Qwen2.5-0.5B base and instruct at 256 tokens, with no training.
- **Speed:** `python -m src.bakeoff.run bench`.
  - Single-text p95 is measured over 30 validation texts.
  - The sheet estimate covers 1,500 cells (500 rows × 6 columns after short cells are skipped). They are fully timed on GPU; on CPU, 100 are timed and the result is scaled.
  - Only scoring time is measured, not model load or report writing.
  - The GPU was held to its 30 W platform limit during the benchmark, so GPU times are pessimistic.

## Results: AUROC

| Fold | DistilRoBERTa | TF-IDF | Fast-DetectGPT base | Fast-DetectGPT instruct | Binoculars | Legacy |
|---|---|---|---|---|---|---|
| val | **0.967** | 0.892 | 0.704 | 0.739 | 0.735 | 0.615 |
| loso:hc3 | **0.976** | 0.740 | 0.936 | 0.958 | 0.946 | 0.519 |
| loso:mage | **0.653** | 0.569 | 0.576 | 0.576 | 0.558 | 0.510 |
| loso:mirage | 0.726 | 0.653 | 0.661 | 0.729 | 0.740 | **0.754** |
| loso:raid | **0.934** | 0.819 | 0.792 | 0.809 | 0.806 | 0.577 |
| lofo:anthropic | **0.997** | 0.938 | 0.394 | 0.454 | 0.476 | 0.808 |
| lofo:deepseek | **0.999** | 0.962 | 0.710 | 0.776 | 0.780 | 0.690 |
| lofo:google | **0.784** | 0.778 | 0.606 | 0.629 | 0.609 | 0.543 |
| recent:openai | **0.998** | 0.979 | 0.571 | 0.660 | 0.658 | 0.752 |

## Results: operating point (threshold for 1% FPR on calibration data)

| Summary | DistilRoBERTa | TF-IDF | Fast-DetectGPT instruct | Binoculars | Legacy |
|---|---|---|---|---|---|
| Detection rate, val | **0.795** | 0.401 | 0.319 | 0.185 | 0.029 |
| Mean detection rate, unseen source | **0.549** | 0.139 | 0.390 | 0.262 | 0.032 |
| Mean detection rate, unseen family | **0.789** | 0.391 | 0.118 | 0.026 | 0.049 |
| Detection rate, newest GPT | **0.996** | 0.707 | 0.076 | 0.007 | 0.068 |
| Worst transferred FPR (target 1%) | 10.8% (MAGE) | 4.8% | **1.7%** | 3.4% | 3.3% |

DistilRoBERTa's transferred FPR per unseen source: HC3 1.6%, MAGE 10.8%, MIRAGE 4.3%, RAID 4.7%. On unseen families and newest GPT, the human texts come from familiar sources and FPR holds at or below 1%.

## Results: speed against the NFRs

| Candidate | GPU single-text p95 (≤500 ms) | CPU single-text p95 (≤2.0 s) | GPU sheet (≤120 s) | CPU sheet (≤600 s) |
|---|---|---|---|---|
| DistilRoBERTa | **9 ms** ✅ | **86 ms** ✅ | **4.5 s** ✅ | **109 s** ✅ |
| TF-IDF | — | 2.5 ms ✅ | — | 1.4 s ✅ |
| Binoculars (and Fast-DetectGPT, which runs the same two models) | 294 ms ✅ | 1.99 s ⚠️ borderline | 297 s ❌ | 2,151 s ❌ |

## Findings

1. **Trained beats zero-shot on modern generators.** All three zero-shot scores fall below chance on unseen Claude text (0.39–0.48) and near chance on the newest GPT models. They work on ChatGPT-era text (HC3 0.94–0.96), which matches their published results, but that doesn't carry over to current models.
2. **Unseen generators are not the main risk; unseen writing is.** DistilRoBERTa holds 0.997–0.999 on unseen Anthropic, DeepSeek, and newest GPT, but drops to 0.653 on MAGE and 0.726 on MIRAGE.
   - MAGE is also the hardest source in-distribution: 0.840, against 0.994–0.999 for the others.
   - That is a deficit, not an inflated score, so it does not point to the cross-split leakage that debt D3 guards against, and D3 is not triggered.
   - The likely causes are MAGE's many short texts and its older, smaller generators.
3. **Short text is weak (U3).** In-distribution AUROC by word count: <25 words 0.83, 25–49 words 0.84, 50–99 words 0.94, 100+ words 0.99. On unseen MAGE text under 50 words it is at or below chance (0.37–0.59). Increment 4 should treat 50 words as the working minimum for a decisive label and confirm it by calibration.
4. **Google's lower family score comes from old models, not Gemini.** Within `lofo:google`, MIRAGE (Gemini) scores 0.996 and MAGE (older T5/FLAN-family models) 0.524.
5. **The MIRAGE length confound (U7) is visible.**
   - MIRAGE texts cluster at 100–199 words (median about 150), but the two labels have different tails. Human texts run from 21 to 199 words, and AI texts from 78 to 219. Outside 100–199 words each length bucket holds only one label, so only that bucket can be scored.
   - The legacy model's best fold is MIRAGE (0.754), even though it fails everywhere else.
   - Both points are consistent with the Increment 2 shortcut probe (AUC 0.79). MIRAGE scores should not be read as pure authorship signal.

## Decision matrix (ADR-009)

Scores run from 1 to 5. The weights reflect the sponsor's instruction that reliability comes first.

| Criterion | Weight | DistilRoBERTa | TF-IDF | Fast-DetectGPT instruct | Binoculars | Legacy |
|---|---|---|---|---|---|---|
| Generalization: unseen source, family, newest GPT | 0.35 | 5 | 3 | 2 | 2 | 1 |
| Detection at 1% FPR | 0.20 | 5 | 3 | 2 | 1 | 1 |
| Threshold transfer (FPR drift) | 0.15 | 2 | 3 | 4 | 4 | 4 |
| Speed against NFRs | 0.15 | 5 | 5 | 1 | 1 | 5 |
| Footprint and dependencies | 0.05 | 3 | 5 | 2 | 2 | 5 |
| Maintenance (retraining, label needs) | 0.10 | 3 | 3 | 5 | 5 | 1 |
| **Weighted total** | 1.00 | **4.25** | 3.40 | 2.45 | 2.25 | 2.25 |

The decision holds under reasonable reweighting. With the other weights scaled proportionally, DistilRoBERTa stays first until threshold transfer is weighted at 0.55 or more, and that concern is answered directly by per-domain calibration in Increment 4.

## Carried into Increment 4

- **Per-domain calibration and a support gate.** Thresholds are set per supported domain, and text outside the calibrated domains returns `Inconclusive`. This addresses the 10.8% MAGE transfer.
- **A minimum length** for decisive labels, starting at 50 words.
- **A final training run** on the full train split, replacing the 24,000-row, one-epoch bake-off budget (debt D5). The run must be followed by calibration on validation and the one locked benchmark run.
- **Peak GPU memory during inference** has not been measured yet (NFR ≤3.5 GB). Training peaked at 1.85 GB, so inference is expected to be well below the limit, but it must still be measured.
- **The end-to-end spreadsheet timing** (model load and report write included) is measured when the model is wired into the pipeline.
- **Optional:** test whether adding a zero-shot score to DistilRoBERTa improves MIRAGE. Binoculars is the only candidate ahead there. Adopt it only if the gain survives calibration.

## Reproduction

```bash
export HF_HOME=D:/llm-authorship-corpus/hf-cache
python -m src.bakeoff.run select --candidates legacy,tfidf,distilroberta,zs-fast-base,zs-fast-instruct,zs-binoculars
python -m src.bakeoff.run bench  --candidates tfidf,distilroberta,zs-binoculars
# laptop GPUs can drop out mid-run; this reruns until success, resuming from fold checkpoints
bash scripts/run_bakeoff_supervised.sh <log file> distilroberta,zs-fast-base,zs-fast-instruct,zs-binoculars
```

Hardware: Dell G15 5520, RTX 3050 Laptop 4 GB, i5-12500H, 16 GB RAM, PyTorch 2.14 (CUDA 13.0), Transformers 5.17.
