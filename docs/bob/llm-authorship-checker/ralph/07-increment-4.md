# wreck-it-ralph report: increments 4 and 4b

Verdict: **FAIL**

Summary: tests **212/212** (no GPU, no model files). Live probes used the real v3 and short-v1 models on v2 validation text, with the gate forced open the way `benchmark.run` opens it. Findings **C0/H2/M4/L4**.

Range: `6158657^..HEAD` (`6158657`..`7d09d41`).

How the rules were kept:
- The benchmark was not run.
- No root `.xlsx` sheet, `dataset_full.csv`, `x` or `inspect_wb.py` was opened.
- No model scored any locked-split text.
- Locked-split text was read once, only for the verbatim-overlap check in R07-6.
- Stress results were read from the committed `data/benchmark/v2/model/benchmark.json`.
- The guard probe called `benchmark.run` on scratch registry copies with `pandas.read_parquet`, `build_service` and `registry.save` patched to raise, so nothing could be read or written.

## Claims checked

| # | Claim | Result | Evidence |
|---|---|---|---|
| 1 | Gate integrity | met, with gaps | **Gate closed.** On the committed registry, `verify` returns `(False, 'locked benchmark not passed')`. Both contexts return `MODEL_NOT_VALIDATED` without loading the model.<br>**Hash failures close it.** On a scratch root, a hash mismatch, a missing short model and a missing short bank each close the gate.<br>**Gaps (R07-8):** tokenizer and config files are not hashed, and a same-size rewrite with the mtime restored passes. |
| 2 | Routing | partly met | **Runtime:** `serves_short` sends 24 words to main, 25–99 to short and 100 to main. It counts words with the policy's rule, so the band and the scoring model agree. `route` needs a supported short band and a win of at least 1 point: 45.6% vs 29.3% and 57.9% vs 50.1%.<br>**Benchmark path (R07-4):** with the short model missing, `validated=True` gives a main-only detector under the short model's thresholds. |
| 3 | No locked data in fitting | met | **Loaders:** `load_split` rejects anything but `train`/`val` and asserts the split. `load_fit_data` asserts `FIT_SPLITS`.<br>**Excerpts:** they copy the parent's row, split included.<br>**Records:** the main model's `train_ids_sha256` equals the digest of v2's 111,434 `train` ids. `calibration.json` has `locked_split_reads: 0`. |
| 4 | Fresh test | partly met | **Audit:** `audit --version v2` passes, with 0 errors and 1 warning (30 test slices under 300 humans).<br>**v1 overlap:** v2 `test` shares 0 sample ids and 0 group keys with v1 `test`.<br>**Held-out sources:** all 6,997 Dolly and oasst1 rows are in `test`.<br>**Gaps:** 49,691 v2 stress rows come from retired v1-test components (R07-3). 74 v2 unseen-generator rows are reused from v1's unseen split. 549 MAGE test texts are at least half verbatim in train/val (R07-6). |
| 5 | Shortcuts | partly met | **Shared probe, collapsed text, train → val:** highest AUC is mirage 0.794; raid 0.614; overall 0.692.<br>**rasbt50k:** 0 rows in train/val; 47,205 diagnostic.<br>**Gaps (R07-7):** a per-source probe separates mirage at 0.866. The committed build probe runs on raw text and flags raid at 0.924, which the 4b report does not mention. |
| 6 | One-shot guard | met, with gap | **Blocking:** the current registry with no reason is blocked, and with a reason is allowed.<br>**Versions:** v1-only history lets v2's first run through.<br>**History:** `benchmark=None` after recalibration still blocks. `calibrate` keeps `locked_benchmark_history`.<br>**Gap (R07-9):** a run is recorded only when it finishes. |
| 7 | Pre-registration | met | **Timeline:** pre-reg committed 00:42:30 (`433ec0e`); the first v2 build began 00:43:29 (`build-v2.log`). Amendment 1 was committed 01:10:06 with the relock (`bc68767`). Main training ran 11:23–12:09 and the short model finished at 12:39.<br>**Criteria:** set once in `6158657` and never changed. Pre-reg edits are append-only.<br>**Calibration fix:** `d3dc2c0` (12:47:04) came before the first benchmark attempt (12:47:18). The interrupted log shows only progress lines.<br>**Caveat:** Amendment 1 says no test result informed it, but the probe AUC it cites was scored on the first build's test split. |
| 8 | Failure paths | not met | **Abstain correctly:** empty, whitespace-only, 500 zero-width characters, French, over 32,767 characters, `None` and an int. A missing or tampered short model also abstains.<br>**Fail:** light homoglyph or zero-width edits turn Likely AI into Likely human (R07-1). A long text with a human opening is labelled human (R07-2). Formula-like text is labelled (R07-10). |
| 9 | Flag-only context | met | **Live:** 300 texts that the general context labels Likely human all return Inconclusive (`BELOW_AI_THRESHOLD_FLAG_ONLY`) under `rater_comment`.<br>**Pipeline:** cells and rows both use `RATER_COMMENT`. See R07-5 for how rows combine. |
| 10 | Tests | met | `CUDA_VISIBLE_DEVICES="" CORPUS_DIR=<empty dir> python -m pytest -q` gives `212 passed in 24.31s`. |

## Findings

### R07-1: Light homoglyph or zero-width edits turn Likely AI into Likely human

- **Severity:** High
- **Category:** robustness / evasion
- **Location:**
  - `src/authorship/language.py:30-36` (a 0.9 Latin share admits partial substitution)
  - `src/corpus/text.py:17-24` (`collapse_whitespace` leaves U+200B and U+00AD in place)
  - `src/detection/service.py:_precheck`
- **Repro steps:**
  1. Build `svc = registry.service_from_record(registry.load(), validated=True)`.
  2. Take 300 v2 `val` AI texts of 60–179 words from hc3/mage/raid/mirage/aita2026, sorted by `sample_id`, that `svc` labels `likely_ai_generated`.
  3. Edit each text in one of two ways:
     - Homoglyphs: replace each of `a e o c p x i y` with its Cyrillic twin (`а е о с р х і у`) with probability p.
     - Zero-width: insert U+200B mid-word in 30% of words longer than 3 letters.
  4. Re-check the edited texts.
- **Expected vs actual:** Claim 8 says these inputs abstain. Instead, cheap edits produce confident "Likely human" labels:

  | Edit | Likely AI | Likely human | Inconclusive |
  |---|---|---|---|
  | none (baseline) | 300 | 0 | 0 |
  | homoglyph 3% | 112 | 13 | 175 |
  | homoglyph 8% | 19 | 82 | 199 |
  | homoglyph 15% | 0 | **186** | 114 (only 2 `UNSUPPORTED_LANGUAGE`) |
  | U+200B inside 30% of words | 35 | **77** | 188 |

- **Evidence:**
  - The same edits on 300 decisive-human texts caused no false AI.
  - RAID's full-strength `attack:homoglyph` and `attack:zero_width_space` stress rows all abstain as `UNSUPPORTED_LANGUAGE`, because the stress set only has heavy edits.
  - Nothing folds confusables or strips format characters (Cf) before scoring.
  - The gate is closed today, so this is what a passing benchmark would ship.
- **Confidence:** high

### R07-2: Long texts are decided from their first 256 tokens

- **Severity:** High
- **Category:** functional bug
- **Location:**
  - `src/authorship/detector.py:87-88` (`truncation=True, max_length=self.max_tokens`)
  - `src/detection/policy.py` (the 200+ band has no upper limit and there is no truncation evidence)
- **Repro steps:**
  1. Take 200 v2 `val` human texts of 300–1,499 words that the service labels Likely human.
  2. Append whitespace-collapsed v2 `val` AI text until the input reaches 32,000 characters.
  3. Check each input with the gate forced open.
- **Expected vs actual:** A document that is about 93% AI should not come back as confidently human. All 200 came back `likely_human_written`, with `SCORE_BELOW_HUMAN_THRESHOLD` as the only evidence. One example is 5,403 words with an AI share of 0.94.
- **Evidence:**
  - The reverse case (an AI opening of 300+ words, then human text to 32,000 characters) gave 158 of 200 Likely AI on mostly-human documents.
  - A 32,000-character input passes every length check, since the limit is 32,767 characters.
  - The response gives no sign that only the first ~190 words were scored.
- **Confidence:** high

### R07-3: The v2 locked stress split re-tests 49,691 variants of retired v1-test components

- **Severity:** Medium
- **Category:** data integrity / pre-registration
- **Location:**
  - `src/corpus/split.py:84-86` (the role loop sets every stress-role row to `stress` after `base` set retired components to `None`)
  - `src/corpus/audit.py` (no retirement check)
- **Repro steps:**
  1. Compute v1 components with `components(group_key, cluster_id)` on `v1/manifest.parquet`.
  2. Collect the sample ids in components that hold a v1 `test` row.
  3. Look those ids up in `v2/manifest.parquet`.
- **Expected vs actual:** The pre-registration says components containing a v1 test sample are never tested again, and a retired component that stress variants would force into evaluation is dropped. Instead:
  - The 15,528 v1-test originals are gone from v2.
  - Their 49,691 attacked variants (raid 31,434, mirage 18,257) remain in v2 `stress`, which is a locked split.
- **Evidence:**
  - Example: group `raid:21f5611a-…` has its v1 `test` original dropped, while its paraphrase, homoglyph and zero-width variants stay in `stress` under the same group key.
  - The v2 audit passes because the parents are absent and it never consults v1.
  - This does not gate the result, but v2's stress numbers include texts whose v1 originals were scored and viewed.
- **Confidence:** high

### R07-4: The benchmark can pass a configuration that differs from the one the gate later opens

- **Severity:** Medium
- **Category:** gate integrity
- **Location:**
  - `src/authorship/registry.py:134-147` (the short model is silently skipped, and the OOD gate is set to `None` when a bank is missing)
  - `src/authorship/benchmark.py:146-150` (no `verify` before scoring)
- **Repro steps:**
  1. On a scratch root with a valid record, move the short model directory away.
  2. Call `service_from_record(rec, root, validated=True)`.
  3. Separately, delete the short model's bank and repeat.
- **Expected vs actual:** Every text should be scored by the model whose thresholds its band uses. Instead:
  - With the short model moved away, the detector is a plain `AuthorshipModelDetector`. Yet `band_for(30)` returns the short model's thresholds (0.0263 / 0.9035), not the main model's (0.0269 / 0.9062).
  - With the bank deleted, the short gate is `None`, so no OOD check runs.
  - `benchmark.run` scores whatever it gets and records `passed`. When the files return, `verify` opens the gate for the routed system, which never ran on the locked test.
- **Evidence:** `verify` correctly reports `short-text model artifact missing`, but `benchmark.run` never calls it.
- **Confidence:** high

### R07-5: A row's decision hides cells that are flagged Likely AI on their own

- **Severity:** Medium
- **Category:** functional bug
- **Location:** `src/detection/service.py` (`combine_column_results`, `row_result` branch), with the truncation from R07-2
- **Repro steps:**
  1. Build rows of two cells: a v2 `val` human rationale of 220–399 words, then a v2 `val` AI text of 40–98 words that `rater_comment` flags Likely AI on its own.
  2. Check the cells and the `"\n\n"`-joined row as `pipeline.py` does.
  3. Combine them with `row_result`.
- **Expected vs actual:** A row with a decisively AI cell should not read as Inconclusive. 100 of 102 rows came back `inconclusive`. `likely_ai_columns` still lists the flagged cell in all 102.
- **Evidence:** The combined text puts the AI cell after about 300 tokens, so the model never sees it. The row decision then overrides the cell decisions.
- **Confidence:** high

### R07-6: MAGE test texts are largely verbatim in train and validation

- **Severity:** Medium
- **Category:** data integrity / leakage
- **Location:**
  - `src/corpus/dedup.py` (whole-text MinHash Jaccard ≥ 0.7 cannot see containment or shared prefixes)
  - MAGE group keys, which do not link a prompt's texts across generators
- **Repro steps:**
  1. Sample 1 in 8 of the word 13-grams (normalized words) from each v2 `test` text.
  2. Look them up in the 13-grams of every v2 `train`/`val` text.
- **Expected vs actual:** No test text should be mostly present in fitted data. Instead:
  - 2,725 of 7,375 MAGE test texts share at least one 13-gram with fitted MAGE text.
  - 549 have at least 50% of their sampled 13-grams in fitted text, and 181 have at least 90%.
  - 0 of these share a `group_key` with their fitted partner.
  - Label pairs (test → fitted): AI → AI 2,161; AI → human 229; human → AI 201; human → human 134.
- **Evidence:**
  - Example: a MAGE `wp` test text begins with the same 30 words as a fitted text.
  - Other sources overlap far less: raid 207, hc3 183, mirage 20, dolly 16 and oasst1 15, mostly single shared 13-grams.
  - MAGE passes its slice gates, and part of that slice measures memorized text.
- **Confidence:** high that the overlap exists; medium on its effect on MAGE's numbers

### R07-7: The shortcut probe checks raw text, and a per-source probe still separates mirage

- **Severity:** Low
- **Category:** validity
- **Location:**
  - `src/corpus/confound.py:26-40` (features are taken from raw text)
  - `data/benchmark/v2/build_record.json` (`shortcut_probe.flagged_sources: ["raid"]`, raid AUC 0.9243)
- **Repro steps:** Fit the confound pipeline on `train` and score `val`, using core rows only. Run it on raw and on `collapse_whitespace` text, once with one shared probe and once with a probe fitted per source.
- **Expected vs actual:** No source should exceed 0.85 on the text the model sees.

  | Probe | Text | Result |
  |---|---|---|
  | shared | raw | raid 0.919 |
  | shared | collapsed | all ≤ 0.794 (pass) |
  | per-source | collapsed | **mirage 0.866**, hc3 0.829 |

- **Evidence:** The committed probe cannot check the claim as worded, because it runs on raw text. Its raid flag is not mentioned in the 4b report.
- **Confidence:** medium

### R07-8: Gate hashes cover only the weights file and the bank

- **Severity:** Low
- **Category:** gate integrity
- **Location:**
  - `src/authorship/registry.py:46-57` (`lru_cache` keyed on path, size and mtime)
  - `src/authorship/registry.py:80-89`
  - `src/authorship/detector.py:_ensure_loaded` (weights load after `verify`)
- **Repro steps:** On a scratch root with a record that validates:
  1. Rewrite `tokenizer.json` and add `config.json`.
  2. Rewrite `model.safetensors` with the same size, then restore its mtime with `os.utime`.
- **Expected vs actual:** Changing what the model computes should close the gate.
  - Step 1 still returns `(True, 'validated')`.
  - Step 2 still returns `(True, 'validated')`, although the true SHA no longer matches.
  - After `_sha256.cache_clear()`, step 2 returns `model artifact hash mismatch`.
- **Evidence:** `verify` runs once when the service is built. The weights are read later, on the first request.
- **Confidence:** high

### R07-9: The one-shot guard records only runs that finish

- **Severity:** Low
- **Category:** process integrity
- **Location:** `src/authorship/benchmark.py:128-131` and `:209-219`
- **Repro steps:** Call `run()` with reads patched out, first on a registry copy with no history, then on the current registry.
- **Expected vs actual:** Any earlier touch of the locked test should need `--rerun-reason`. Instead, a registry with no history is let through. The v2 `benchmark.json` already on disk is ignored.
- **Evidence:** An interrupted run leaves no record. The 12:47 attempt left none, and the rerun reason was given voluntarily. Restoring an older `registry.json` would also clear the history.
- **Confidence:** high

### R07-10: Formula-like inputs are labelled instead of abstaining

- **Severity:** Low
- **Category:** claim mismatch
- **Location:** `src/detection/service.py:_precheck` (no formula check)
- **Repro steps:** Prefix the R07-1 texts with `=`, wrap them as `="…"`, or prefix them with `@`.
- **Expected vs actual:** Claim 8 says these abstain. Instead:
  - AI texts come back Likely AI 272, 290 and 262 of 300.
  - Human texts come back Likely human 167, 87 and 233 of 300.
- **Evidence:** No false AI was seen. Either the claim or the behaviour needs to change.
- **Confidence:** high

## Triage (build chat, 2026-09-27)

Live re-checks below used the real v3 and short-v1 models on v2 validation text only, with the gate forced open as `benchmark.run` does. No locked text was scored. The suite is now 226 tests, all passing.

| ID | Triage | What changed | Evidence after the fix |
|---|---|---|---|
| R07-1 | **Fix** | New `src/detection/obfuscation.py`, applied first in `AuthorshipService.check_batch`. It removes Cf format characters and maps compatibility forms and lookalike letters to Latin, but only inside words made of Latin and lookalikes; real Cyrillic or Greek words stay. Results carry `OBFUSCATION_NORMALIZED` evidence with counts. | 300 decisive-AI texts under 3%, 15% and 100% homoglyphs, or U+200B in 30% of words: 300/300 stay Likely AI in every case (was 186 Likely human at 15%). Russian text is unchanged and still `UNSUPPORTED_LANGUAGE`. |
| R07-2 | **Fix** | Text longer than the model window is scored in full: the opening (the whole text, cut at 256 tokens as in calibration) plus sentence-started 160-word windows covering the rest. The opening decides in its full-length band; any later window that decisively points the other way returns `Inconclusive` with `MIXED_AUTHORSHIP_SIGNALS`. Later windows can veto a label but never create one. `WINDOWED_SCORING` evidence lists every window score. | Human opening + ~93% AI body: 100/100 Inconclusive (71 mixed, 29 out of distribution), was 200/200 Likely human. On 2,151 clean long validation texts, false AI went from 9 to 5, missed AI stayed at 9, and decisive coverage fell from 81% to 74%. A majority vote over windows was tried and rejected: missed AI rose to 63 of 1,060, because later windows of AI text score lower. |
| R07-3 | **Fix (takes effect in corpus v3)** | `assign_splits` now drops stress rows of retired components that would be forced into evaluation. The audit takes the group keys of earlier versions' locked tests and fails on any evaluation row in one; the build and the audit CLI both supply them. | Split test updated. `audit --version v2` now reports the 49,691 rows. v2 is not rebuilt; its stress numbers carry the caveat in the 4b report. |
| R07-4 | **Fix** | `registry.verify_files` checks both models' weights, the whole artifact directory and both banks. `benchmark.run` refuses to score unless it passes. `service_from_record` falls back to `main_only` thresholds whenever the specialist's files are missing or changed, so a model is never paired with another model's thresholds. | Moving the short model away gives a plain detector with the main model's own bands (`band_for(30)` is the main band), and `verify_files` returns `short-text model artifact missing`. |
| R07-5 | **Fix** | `combine_column_results(row_result=...)`: a cell that is Likely AI on its own flags the row (`CELL_FLAGGED_AI`), even when the combined text is Inconclusive or human. | 60/60 rows with a flagged AI cell are now Likely AI (was 100/102 Inconclusive). |
| R07-6 | **Fix in the 4c corpus build** | The build adds a containment check: sampled word 13-grams link texts that are mostly contained in one another into one component, so they can't span train and test. | Scheduled with corpus v3. v2's MAGE numbers carry the caveat in the 4b report. |
| R07-7 | **Fix (takes effect in corpus v3)** | The shortcut probe now reads whitespace-collapsed text (the model's view), runs shared and per-source probes, and is fitted on train and scored on validation, so building never reads test text. The raw-text probe is kept for reference. | The 4b report now records the raw-text RAID flag (0.924) and the per-source MIRAGE result (0.866). The 4c build investigates MIRAGE if it flags again. |
| R07-8 | **Fix** | `sha256_file` no longer caches on size and mtime. `directory_sha256` covers weights, config and tokenizer, and calibration records it for both models. The detector re-checks the directory at load time, before reading weights. | A same-size rewrite with the mtime restored gives `model artifact hash mismatch`, and a changed `tokenizer.json` gives `model config or tokenizer files changed`. |
| R07-9 | **Fix** | `benchmark.run` appends a `status: started` history entry and saves the registry before any locked text is scored, then marks it finished. An existing report file for the version also counts as a run. | A run that crashes mid-scoring leaves a `started` entry, and the next run needs `--rerun-reason`. |
| R07-10 | **Claim corrected (no code change)** | The review prompt's claim was wrong. A leading `=` or `@` is ordinary text to the checker and is scored like any text. Spreadsheet output keeps formula-like values inert (`src/report/excel_writer.py:414`), which is where formula injection is handled. | No false AI was seen. |

Amendment 1 caveat: the probe AUC it cites was scored on the first v2 build's test surface features. That is model-free, but it is still a read of test text. The probe now uses validation.
