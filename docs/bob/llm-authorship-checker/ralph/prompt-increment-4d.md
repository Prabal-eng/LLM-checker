# Ralph prompt: Increment 4d before the lock is read (R12)

You are wreck-it-ralph, an adversarial reviewer. Read `bob-the-builder/wreck-it-ralph.md` first and follow it. Your job is to break the claims below, not confirm them.

Work in the worktree at `C:\Users\prabd\.gemini\antigravity-ide\scratch\excel-ai-qa-pipeline-4d`, on branch `increment-4d`. The range is `6221d3b..HEAD`: corpus v4, generation variants, the Plan A review fixes, 4d training, precision in the registry, the optional-step measurements, the head-to-head, and pre-registration Amendments 1 onward.

**Rules:**
- Don't edit any existing file and don't commit. The only file you write is the report.
- Don't run `python -m src.authorship.headtohead` or `python -m src.authorship.benchmark`: either consumes the v4 locked test. Don't run training, calibration, generation, the corpus build or the speed check.
- Don't use the GPU. Run tests with it hidden: `CUDA_VISIBLE_DEVICES= python -m pytest -q -p no:cacheprovider`.
- RAM is tight (about 4 GB free). Don't load `corpus.parquet` whole.
  - For split and component checks, read `manifest.parquet` (no text) with column selection.
  - The build's own audit result is in `data/benchmark/v4/build_record.json`.
  - Corpus text lives under `D:/llm-authorship-corpus` and may be read in bounded slices.
- Don't read, open or run anything on the root-level `.xlsx` rater sheets or `data/training/dataset_full.csv`. They are unverified in-house data. Don't name raters.
- Don't open the files `x` or `inspect_wb.py`.

## Claims to check

1. **The v4 locked test is fresh** (`src/corpus/build.py`, `split.py`, `audit.py`, `data/benchmark/v4/build_record.json`, the v1–v4 manifests).
   - No v4 `test` or `test_unseen_generator` row matches, by sample id, group or near-duplicate, any row of v1–v3 in any split, report-only ones included (Amendment 2).
   - Held-out sources (stackexchange_heldout, wasitai2026, lede) are tested whole, except rows an earlier version saw, which are excluded.
   - Existing sources took rows no earlier version took where their pools allowed (`take_per_stratum(..., skip_used=True)`); the build record's test rows per source match what Amendment 2 says.
   - Stress slices kept their rows; a core copy of an earlier stress text was dropped, not the stress copy.
   - The build settled its splits on the audit's own earlier-text check, and the audit passed.
2. **Corpus v4 has no shortcut left.**
   - `shortcut_probe` and `ai_only_probe` in the build record: no source is over 0.85 without a recorded decision.
   - Look for surface differences that separate the labels on the model's view (`format3`): length, list markers, dashes, openers, casing, preambles in humanized rewrites ("Here's a casual version:"), personas, plain-text answers.
   - MAGA and DACTYL are length-matched per domain.
3. **`format3` and disguise handling** (`src/corpus/text.py`, `src/detection/obfuscation.py`).
   - `format3` is idempotent (one application is final) and leaves ordinary prose alone.
   - Every bullet glyph, `(1)` / `1)` marker and dash form in the pre-registration goes.
   - A formatting or character trick can't turn Likely AI into Likely human.
4. **Generation** (`src/corpus/generate.py`, `raw/*/generated*/`).
   - Every variant record has its variant, seed salt, style, options, model digest and licence.
   - The counts match the pre-registration's table.
   - The default variant keeps v3's pools; held-out generators never answer training items.
   - Look for artifacts: refusals, meta-commentary, word-count echoes, preambles, wrong language.
5. **Training read only train and validation** (`src/authorship/train.py`, `data/benchmark/v4/model/<candidate>/training_record*.json`).
   - Checks ran every quarter epoch with early stopping as pre-registered; the best check was kept.
   - A LoRA model (if any) was merged, and the merged artifact is what calibration scored.
   - The training data digests match train and validation ids only.
6. **Calibration and the registry** (`calibration.json`, `models/registry.json`, `src/authorship/registry.py`, `detector.py`).
   - Targets were 0.8% false AI and 8% missed AI, per word band; routing followed the 1-point rule.
   - The inference precision is recorded for each model and is what the detector uses; a record without one stays closed.
   - The seal covers the precision, the scoring code and the report.
7. **The optional steps were decided on validation only** (`stack.json`, `registers.json`, `src/authorship/stack.py`, `registers.py`).
   - The stack was fitted on one parent-hashed half and measured on the other.
   - Register cells under 600 validation humans kept the pooled bar; adoption needed 2 points with every cell within target.
   - Each verdict was recorded as an amendment before the lock.
8. **The head-to-head is fair and one-shot** (`src/authorship/headtohead.py`, `benchmark.py`, Amendments 2 and 3).
   - Today's models are scored through `models/incumbent_registry.json` (their `format2` view, bf16 precision, their bands, their excerpt view); the record is refused if its v3 seal breaks.
   - Both models see the same rows; shared ground, gated slices and the outcome table match the pre-registration and its amendments.
   - The run is recorded before any locked text is scored; a second run needs a reason.
   - Each model set is loaded only for its own pass; check the run can't run out of GPU memory after it
     is recorded (both models of a set, plus the main-only excerpt service, which reuses the loaded main
     model).
   - Unseen-generator rows count as shared ground only from shared-ground sources (Amendment 4); look at
     the pooled figure by source.
9. **Speed and memory** (`speed.json`, `src/authorship/speed.py`): the GPU and CPU limits are measured as Amendments 3 and 4 say, the sheet on the app's per-cell path.
10. **The pre-registration came first.** Each amendment was committed before the results it could affect (generation, the v4 build, training, calibration, the optional steps). Nothing in the pass criteria moved since v1.
11. **Tests.** The suite passes with no GPU and no model files.

## Report format

Write `docs/bob/llm-authorship-checker/ralph/12-increment-4d.md` in the same format as `11-increment-4c-final.md`:
- Verdict: PASS or FAIL for reading the lock.
- Test counts.
- A claims table with evidence.
- Findings, each with an ID (R12-n), severity, category, location, repro steps, expected vs actual, evidence and confidence.

Return the verdict and the findings list as your final message.
