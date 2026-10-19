# Increment 4 report — 2026-09-27

## Result

**The registered model (`distilroberta-authorship-v2`) did not pass the locked benchmark, so the validation gate stays closed.** Every check still returns `Inconclusive` with `MODEL_NOT_VALIDATED`, as the design requires.

- **Passes:** the model meets every pre-registered bound on the four sources it was trained on (HC3, MAGE, MIRAGE, RAID). It flags none of the 3,770 in-house rater texts.
- **Fails:** Beemo, a source it never saw. Beemo's human-written, assistant-style answers were flagged as AI 10% of the time (95% upper bound 11.2%, against a 1% limit), and missed AI came in at a 10.9% upper bound (limit 10%).

The failure is a data gap, not a defect in the policy. Every assistant-style answer in the training data is AI-written, so the model partly learned "sounds like an assistant = AI". The next increment adds human-written assistant-style sources and a fresh locked set (see *Next*).

## What was built

| Part | Where | What it does |
|---|---|---|
| Final training | `src/authorship/train.py` | Trains on all 99,702 training rows over 2 epochs. Each source's human and AI texts get equal weight, so the source can't act as a shortcut. Saves every epoch, resumes after a crash, and keeps the best epoch by validation. |
| Text normalization | `src/corpus/text.py` `collapse_whitespace` | Every whitespace run becomes one space, both in training and when checking, so layout can't stand in for authorship. |
| Calibration | `src/authorship/calibration.py`, `calibrate.py` | Maps scores to probabilities (Platt scaling). Sets thresholds per word-count band, shared across sources: every source slice keeps its 95% upper bound on false AI within target, and on missed AI within target. Calibration targets were 0.8% and 8%, a margin under the 1% and 10% release bounds. |
| Support gate | `src/authorship/language.py`, `detector.py` | Checks for English using function words and script, with no new dependency. Flags text as out of distribution when its mean cosine distance to the 10 nearest of 20,000 training embeddings is too high (the cutoff accepts 99% of validation text). Length bands handle short text. |
| Contexts | `src/detection/contracts.py` `TextContext` | `general`, the default for standalone text, gives the full three-way result. `rater_comment` is used for spreadsheets: it is flag-only, never labels text "Likely human", and skips the domain gate (debt D6). |
| Row-level sheets | `src/pipeline.py`, `combine_column_results(row_result=...)` | A row's rationales are checked together as one text and decide the row. The individual cells are kept as supporting detail. |
| Registry | `models/registry.json`, `src/authorship/registry.py` | Replaces the hard-coded gate (D1). A model counts as validated only when its benchmark passed **and** the weights and embedding bank on disk still match their recorded hashes. Every locked run is kept in `locked_benchmark_history`. |
| Locked benchmark | `src/authorship/benchmark.py` | Runs once with criteria fixed in advance, and checks the lock hash before reading. A second run needs a recorded reason. In-house text is report-only. |

The test suite has 195 tests and needs neither a GPU nor the model file.

## Training and calibration (validation only)

| | v1 (layout kept) | v2 (whitespace collapsed) |
|---|---|---|
| Validation AUROC | 0.992 | 0.988 |
| Detection at 1% FPR | 88.0% | 87.0% |
| Peak inference VRAM | 0.81 GB | 0.81 GB (NFR ≤3.5 GB ✅) |

v2's validation calibration, with targets of 0.8% false AI and 8% missed AI:

| Words | AI threshold | Human threshold |
|---|---|---|
| 25–49 | 0.915 | 0.009 |
| 50–99 | 0.954 | 0.491 |
| 100–199 | 0.949 | 0.799 |
| 200+ | 0.934 | 0.881 |

**U4 answered:** one shared policy per length band is enough.
- Thresholds fitted to each source separately would add at most 5 points of coverage: RAID goes from 90% to 95%, MIRAGE from 98% to 100%, and MAGE is unchanged. HC3 alone has too few samples to support thresholds of its own.
- Per-source thresholds would also need the source when a text is checked, and that is never known. Length always is.

**U5 answered (evidence):** a result shows the calibrated score, the band's thresholds, word count, and whether the row's combined text was used. When it abstains, it gives the reason: language shares, out-of-distribution distance, or an unsupported length. It shows no token-level "AI words". The bake-off found no per-token signal stable enough to present without overstating causality.

## Locked benchmark: two runs

| Test slice | v1: false AI | v1: missed AI (upper) | v2: false AI | v2: missed AI (upper) |
|---|---|---|---|---|
| HC3 | 1 / 1,308 | 0.5% | 2 / 1,308 | 0.6% |
| MAGE | 10 / 4,739 | 6.3% | 4 / 4,739 | 5.3% |
| MIRAGE | 11 / 10,650 | 0.3% | 21 / 10,650 | 0.1% |
| RAID | 2 / 3,681 | 1.7% | 9 / 3,681 | 4.7% |
| **Beemo (never trained on)** | **922 / 2,180 (42%)** | 4.6% | **219 / 2,180 (10%)** | **10.9%** |
| Decisive coverage on supported text | 90% | | 86% | |

| In-house rater text (unverified labels; report only) | v1 flagged | v2 flagged |
|---|---|---|
| Single rationales, never viewed (1,621) | 4 | **0** |
| Single rationales, previewed (2,149) | 1 | **0** |
| Pseudo-rows (6 rationales from one rater), never viewed (249) | 237 (95%) | **0** |
| Pseudo-rows, previewed (393) | 386 (98%) | **0** |

Reported alongside, not used to gate (v2):
- **Unseen generator families:** Moonshot 99% detected, BLOOM 39%, Cohere 43% (24% labelled human).
- **Stress sets:**
  - Paraphrased, polished, or rewritten AI text is labelled human about 20% of the time.
  - RAID's word-level attacks are missed 8–13% of the time.
  - Homoglyph and zero-width attacks are always rejected as non-English. That means `Inconclusive`, never mislabelled.
- **Human essays:** 7 of 90 non-native essays and 36 of 215 unlabelled-group essays were flagged.

## What went wrong, and what changed

1. **Layout confound (found by the first locked run).** The public datasets kept paragraph breaks in AI text but stripped them from human text: 68.5% of RAID AI texts have blank lines, against 0.3% of RAID human texts. v1 learned "paragraphs = AI".
   - The effect was clear: the same six rater rationales scored 0.976 joined with blank lines and 0.001 joined with spaces.
   - Fix: v2 trains and scores on whitespace-collapsed text.
   - The Increment 2 shortcut probe missed this because it measured line breaks only as a rate. With presence features added, it flags RAID at AUC 0.93.
2. **Locked in-house text was exposed before the benchmark.** A workbook preview scored the GemPix Replication 1 sheet, which supplied 2,149 of the 3,770 in-house texts.
   - Nothing was tuned on it.
   - Those texts are reported separately, and the IDs are in `data/benchmark/v1/model/inhouse_previewed_ids.json`.
   - Rule going forward: check whether a file feeds a locked split before running anything on it.
3. **In-house labels are unverified.** The sponsor can't say which rater texts are human. They are never used for training and never gate a release (debt D6).
4. **The test split has now been seen twice.** The second run's reason is recorded in the registry. Another run on the same split would start to fit the model to the test, so the next benchmark needs a fresh locked set.

## Next (Increment 4b)

- Add human-written, assistant-style answers paired with AI answers to the same prompts (candidates: Dolly-15k, CC-BY-SA; OpenAssistant oasst1, Apache-2.0). Add human essays too.
- Build corpus v2 with a **fresh locked test**. It must include at least one assistant-style source that no model has seen, and it keeps today's test as a historical record.
- Retrain, recalibrate, and run the new locked benchmark once.
- If sponsor-verified rater rationales can be obtained (human and AI-written), use them to lift the flag-only restriction (D6).

## Reproduction

```bash
export HF_HOME=D:/llm-authorship-corpus/hf-cache
python -m src.authorship.train --epochs 2
python -m src.authorship.calibrate --max-false-ai 0.008 --max-false-human 0.08
python -m src.authorship.benchmark --rerun-reason "..."   # the test split has already been run
```

Records (no text): `data/benchmark/v1/model/` holds training, calibration, and benchmark JSON for v1 and v2. `models/registry.json` holds the registry.
