# wreck-it-ralph report: the R07 fixes and Increment 4c

Verdict: **FAIL**

Summary: tests **247/247** (no GPU, no model files, run twice: once with an empty `CORPUS_DIR`, once as is). No end-to-end examples were given. Findings **C0/H2/M2/L6**.

Range: `0037761^..HEAD` (`0037761`..`010f700`).

How the rules were kept:
- Nothing in the repo was edited or committed except this report.
- The benchmark, training, calibration and `src.corpus.audit` were not run. `corpus.parquet` was not opened.
- No root `.xlsx` sheet, `dataset_full.csv`, `x` or `inspect_wb.py` was opened.
- Split and component checks used `manifest.parquet` with column selection. Corpus text came from the raw files under `D:/llm-authorship-corpus/raw/`, mapped to v3 splits by re-deriving `sample_id` through the loaders, redaction and markdown stripping.
- **Model use:** one CPU-only run (`CUDA_VISIBLE_DEVICES=""`, 2 threads, about 1 GB RAM) scored 214 v3 **validation** AI texts, the only model scoring in this review. It used the committed registry's main model (`distilroberta-authorship-v3`) with the gate forced open, as `benchmark.run` does.
- **Locked-split text** (Stack Exchange and OpenAssistant test rows) was read from raw files for model-free surface counts only (R08-2).
- **Disclosure:** a first tokenizer-only pass, with no model, took its texts before I filtered to validation, so it may have tokenized some of the 80 OpenAssistant test AI answers. It was rerun on validation only, and only the rerun's numbers are reported.

## Claims checked

| # | Claim | Result | Evidence |
|---|---|---|---|
| 1 | R07 fixes hold | **partly met** | **R07-1: not met (R08-1).** U+200B and Cyrillic lookalikes are undone: 214/214 give identical token ids after the fix. Invisible combining marks and Latin-script lookalikes are not, and a live run turns 28 of 31 Likely AI into Likely human.<br>**R07-2: met, with a gap (R08-7).** Code: the opening decides, and a later window can only veto to `MIXED_AUTHORSHIP_SIGNALS` (`service.py:202-214`). Token-dense texts leave words unscored.<br>**R07-3: met.** v3 stress has 0 rows whose id, group or component touches a v1/v2 test row.<br>**R07-4 and R07-8: met for new records, fail-open for old ones (R08-4).** `benchmark.run` refuses unless `verify_files` passes (`benchmark.py:150-153`), and the detector re-hashes its directory before loading (`detector.py:73-79`). A record without `files_sha256` skips the config and tokenizer check, and the committed registry has none.<br>**R07-5: met** (code, `service.py:236-251`).<br>**R07-6: met, with a short-text residual (R08-8):** v2 had 549 MAGE test texts at least 50% contained in fitted text; v3 has 16.<br>**R07-7: met.** The probe reads `typography+collapse_whitespace` text, shared and per source, fitted on train and scored on val (`confound.py:65-83`).<br>**R07-9: met.** A `started` entry is saved before scoring (`benchmark.py:158-163`). |
| 2 | Generation provenance | **met**, with residual artifacts | **Fields:** 4,487 records, and none lacks a digest, options, seed, target length, licence, quantization or `created_at`.<br>**Assignment:** `generator == pick(item_id)` in all records, and `seed == seed_of(item_id)` and `num_predict == 2N+60` everywhere. `prompt_sha256` matches the current `pairs()` in every record.<br>**Subset:** every item in the first 1,500/1,500/1,200 of the hash order is answered, plus 137/102/48 extras written before the cut.<br>**Digests:** each recorded digest equals the SHA-256 of the model's manifest under `D:/ollama/models/manifests`. Every manifest predates that model's first answer (Llama: manifest 08:32:45Z, first answer 08:39:43Z).<br>**Backfill:** against `generated/backup/`, it changed only `model_digest` (all), `model_licence` (Gemma, a trailing space; phi4-mini, "Microsoft." to "MIT License") and `provenance_backfill`. Text changed in 0 records, order in 0.<br>**Amendment 4 counts** reproduce exactly: 75/106/59 dropped, 1,562/1,496/1,189 kept, and the same per-model losses.<br>**Artifacts left:** 18 fitted answers still restate the requested length (R08-6). Refusals and self-identification (0.4–1.6%) and single CJK characters mid-sentence (Qwen and Gemma, 0.1–0.6%) are real model output, not prompt echoes. There are no think tags. |
| 3 | Pairing can't leak or shortcut | **leak: met. Shortcut: not met (R08-2)** | **Leak:** 0 group keys and 0 components span train, val and evaluation. No OpenAssistant test prompt equals a fitted prompt (0 of 188).<br>**Stripping:** markdown is stripped from both labels, as specified.<br>**Shortcut:** on the model's view, single `*` and numbered lists still separate the labels. Stack Exchange's HTML conversion also flattens human formatting that the AI side keeps. |
| 4 | v3 split | **met** | **Retired tests:** v1/v2 test ids and groups in v3 test, unseen, stress and inhouse are all 0. v3 train/val hold 25,204 + 23,825 of them, which is the build record's 49,029.<br>**Stack Exchange:** 2,973 human and 824 AI in `test`, 364 AI in `test_unseen_generator`, and 0 elsewhere.<br>**Phi and Granite:** microsoft (167) and ibm (197) appear only in `test_unseen_generator`.<br>**Test share:** 34.4%, 33.7% and 36.2% of fresh (non-retired) components for aita2026, no_robots and mirage. Overall it is 26–29%, because retired components are refitted.<br>**Audit:** the build record's audit passed, with 0 errors. |
| 5 | Shortcut probe | **met as built**; caveat R08-2 | **Result:** no flagged sources. The highest own-probe AUCs are dolly 0.8464, mirage 0.8417 and hc3 0.83, all fitted on train and scored on val. My reconstruction from raw text reproduces dolly (0.8466) and oasst1 (0.7735).<br>**Caveat:** adding eight formatting features takes dolly to 0.8626, over the pre-registered 0.85. Stack Exchange is never probed. |
| 6 | Pre-registration came first | **met**, with record errors (R08-9) | **Generation:**<br>- The pre-registration was committed at 08:31:06Z (`a1c7fd8`).<br>- The first Ollama chat was at 08:37:05Z. It was the speed measurement Amendment 1 cites.<br>- Amendment 1 was committed at 08:39:16Z, and the earliest answer is 08:39:43Z.<br>**Amendments and build:**<br>- Amendment 4 was committed at 00:08:38 IST and the first build started at 00:09:16.<br>- Amendment 5 was committed at 00:40:57 and the final build ran 00:41:09–01:03:14. The record says `built_at 19:31:36Z`, and the lock was committed at 01:04:08.<br>**Criteria:** unchanged. `benchmark.py` keeps 0.01/0.10/0.40/300/100, and the 4b text is identical. |
| 7 | Fitting reads only train and val | **met in code; records unverifiable** | **Loaders:** `load_fit_data` asserts `FIT_SPLITS`, and `calibrate.load_split` rejects anything but train/val. Excerpts copy their parent's row, split included.<br>**Records:** no `data/benchmark/v3/model/training_record*.json` exists yet.<br>**Training log:** `train-v4.log` shows 14,386 micro-batches of 8 per epoch, which is ceil(115,086/8), and 115,086 is exactly v3 `train`. |
| 8 | Tests | **met** | `CUDA_VISIBLE_DEVICES="" python -m pytest -q` gives `247 passed in 26.76s`. With an empty `CORPUS_DIR` it gives `247 passed in 23.97s`. |

## Findings

### R08-1: The disguise fix misses invisible combining marks and Latin-script lookalikes, which still turn Likely AI into Likely human

- **Severity:** High
- **Category:** robustness / evasion (R07-1 not closed)
- **Location:**
  - `src/detection/obfuscation.py:53-54`: `_is_latin` accepts every letter named LATIN, including ı ɑ ɡ ᴏ ᴄ ꜱ ʏ.
  - `src/detection/obfuscation.py:58-60`: only category Cf is removed; Mn characters such as U+FE0F, U+034F and U+180B stay.
  - `tests/test_ralph_r07.py:50-68`: the tests cover only U+200B, U+00AD and Cyrillic о/а/е.
- **Repro steps:**
  1. Take v3 **val** AI answers of 60–179 words from Dolly and OpenAssistant (raw generated JSONL, mapped to splits through the manifest).
  2. Build `svc = registry.service_from_record(registry.main_only(registry.load()), validated=True)` on CPU.
  3. Keep the texts `svc` labels `likely_ai_generated` (31 of 214).
  4. Edit each kept text: insert U+FE0F or U+034F mid-word in 30% of words longer than 3 letters, or swap `i a g o c s y` for `ı ɑ ɡ ᴏ ᴄ ꜱ ʏ` with p = 0.15.
  5. Run `check_batch` on the edited texts.
- **Expected vs actual:** Disguised text should be normalized or abstain, as R07-1's triage claims. Instead:

  | Edit (31 decisive-AI val texts) | Likely AI | Likely human | Inconclusive | `OBFUSCATION_NORMALIZED` |
  |---|---|---|---|---|
  | none | 31 | 0 | 0 | 0 |
  | U+200B in 30% of words (fixed case) | 31 | 0 | 0 | 31 |
  | U+FE0F in 30% of words | 0 | **7** | 24 (12 OOD, 12 abstention band) | 0 |
  | U+034F in 30% of words | 0 | **11** | 20 | 0 |
  | Latin-script lookalikes, 15% | 0 | **28** | 3 | 0 |

- **Evidence:**
  - Tokenizer check on all 214 val texts after `deobfuscate` and the model view: 0/214 keep identical token ids, and token counts grow by 20–47%.
  - 208–210 of 214 still pass the English gate.
  - For comparison, U+200B and Cyrillic edits give 214/214 and 213/214 identical token ids.
  - Example: `'The quic️k brow️n fox…'` becomes the tokens `'Ġqu','ic','ï¸ı','k'`, with `format_removed=0` and `letters_mapped=0`.
  - The fix is preprocessing and does not depend on the model, so `distilroberta-authorship-v4` inherits the gap.
- **Confidence:** high

### R08-2: Formatting still separates the paired labels, the Stack Exchange human side is stripped of it by our converter, and the probe cannot see either

- **Severity:** High
- **Category:** validity / shortcut (pre-registered formatting control not met)
- **Location:**
  - `src/corpus/sources/_common.py:207`: only `**`, `__` and leading `#` are removed. Single-`*` emphasis and bullets, and `1.` lists, survive.
  - `src/corpus/sources/stackexchange.py:262-287`: human `<ol><li>` becomes `- `, and `<em>`, `<i>`, `<strong>`, `<b>` and `<img>` vanish.
  - `src/corpus/confound.py:26-28`: no list-marker or asterisk feature.
  - `src/corpus/confound.py:78-83`: a source without both labels in train and val (Stack Exchange) is never probed.
- **Repro steps:**
  1. Rebuild the paired texts from raw with each loader's `pairs()`, `load_generated`, `strip_markdown`, `pii.redact` and the `typography+collapse_whitespace` view.
  2. Map the texts to v3 splits by `sample_id`.
  3. Count formatting features by label.
  4. Fit the build's probe per source (train to val), then fit it again with eight formatting-only features added: asterisk, numbered item, dash item, `---`, backtick, colon rate, `[Placeholder]` and asterisk rate.
- **Expected vs actual:** The pre-registration says emphasis markers are stripped "so formatting can't stand in for authorship", and a source over 0.85 must be fixed before training. Instead:
  - Asterisks: AI 17% vs human 1% on Dolly, 26% vs 6% on OpenAssistant, and **35.3% vs 0.9% on the gated Stack Exchange test**.
  - Numbered items on Stack Exchange test: AI 21.7% vs human 2.1%. Held-out Phi and Granite use them 32%, so this is real AI style, which the human side cannot show.
  - Of 3,000 Stack Exchange human answers, 171 had `<ol>` flattened to dashes, 671 lost `<em>`/`<i>`, 569 lost `<strong>`/`<b>`, and 115 lost images.
  - A formatting-only probe fitted on Dolly and OpenAssistant train scores AUC **0.743** on the Stack Exchange test and 0.611 on its own validation. The gated slice is more separable by format than the data the model learns from.
  - Build features plus formatting features: **dolly 0.8466 → 0.8626**, over 0.85. OpenAssistant goes 0.7735 → 0.8186.
- **Evidence:**
  - The Stack Exchange false-human rate, which is gated, can be helped by a cue our conversion creates.
  - The Stack Exchange false-AI rate cannot reveal errors on human text that uses markdown lists, because its humans have none.
  - Training started at 01:04 on this corpus.
- **Confidence:** high that the asymmetry exists; medium on its effect on the v4 model's gate numbers (v4 is not trained yet)

### R08-3: The two sources that motivated 4c cannot fail the v3 gate

- **Severity:** Medium
- **Category:** validity / pre-registration scope
- **Location:**
  - `src/authorship/benchmark.py:106-113` (`MIN_HUMANS`/`MIN_AIS` per source)
  - `src/corpus/split.py:83-85` (retired components are refitted)
  - Pre-registration *Why* and Amendment 5 C
- **Repro steps:**
  1. Read `pd.read_parquet(v3/manifest.parquet, columns=[...])`.
  2. Count test rows by source and label.
- **Expected vs actual:** 4c exists because v2's test failed on OpenAssistant (592/2,999 false AI, bound 0.2097) and Dolly (0.0122). The v3 gate should test that failure, or the pre-registration should say it no longer does. Instead:
  - v3 `test` holds **oasst1 188 human / 80 AI**, below both the 300-human and 100-AI minimums, so the source is ungated.
  - Dolly has 0 test rows.
  - At v2's OpenAssistant error rate (about 37 of 188), v3 would still pass.
- **Evidence:**
  - Amendment 5 C notes that Dolly is fit-only and that OpenAssistant test rows come from fresh trees. It never says OpenAssistant falls below the gate.
  - Assistant-voiced human text is gated only through No Robots (1,003) and Stack Exchange (2,973). Stack Exchange answers are forum posts, not assistant replies.
- **Confidence:** high

### R08-4: After a pass, the gate stays open when the serving thresholds, calibration or text view change, and old records skip the config check

- **Severity:** Medium
- **Category:** gate integrity
- **Location:**
  - `src/authorship/registry.py:112-120`: `verify` checks only `benchmark.passed` and the files.
  - `src/authorship/registry.py:90`: the `files_sha256` check runs only if the key exists.
  - `src/authorship/registry.py:157` with `src/corpus/text.py:50-51`: a missing `text_normalization` becomes the raw-text view.
- **Repro steps:** On a scratch root with tiny fake weights, config, tokenizer and bank, and a copy of the committed record with `benchmark.passed=True` and fresh hashes:
  1. Change `tokenizer.json`.
  2. Drop `files_sha256` from the record.
  3. Restore the tokenizer, then set every band to 0.45/0.55, Platt to [5, 3], the OOD threshold to 10 and `text_normalization` to None.
- **Expected vs actual:** The gate should open only for the configuration that ran on the locked test, and a missing view should fail closed. Output:

  ```
  1. committed registry has files_sha256? main: False  short: False
  2. baseline verify: (True, 'validated')
  3. tokenizer changed, record WITH files_sha256: (False, 'model config or tokenizer files changed')
  4. tokenizer changed, record WITHOUT files_sha256: (True, 'validated')
  5. after the pass: bands 0.45/0.55, new Platt, OOD threshold 10, text view removed -> (True, 'validated')
  ```

- **Evidence:**
  - `build_policy` serves the edited bands: `BandRule(25, 50, human_max_score=0.45, ai_min_score=0.55)`, and so on.
  - The benchmark's `started` entry records `lock_sha256` and `model_id`, but no digest of calibration, bands, routing or text view.
  - This is the R07-4 class again: a configuration can differ from the one that was benchmarked.
- **Confidence:** high

### R08-5: The test lock does not cover the gate's slice labels or case, NFKC and whitespace changes to test text

- **Severity:** Low
- **Category:** data integrity
- **Location:**
  - `src/corpus/lock.py:13-16` (hashes `sample_id|split|label`)
  - `src/corpus/text.py:11-12,57-58` (`sample_id` is the hash of casefolded, NFKC, whitespace-collapsed text)
  - `src/authorship/benchmark.py:108` (slices by `source_dataset`)
- **Repro steps:** Compute `lock.locked_hash` on a two-row frame, then change `source_dataset`, and change the text to upper case with extra spaces.
- **Expected vs actual:** A change to what the gate reads should break the lock. Instead the hash is unchanged, and `sample_id('Hello world.') == sample_id('HELLO   WORLD.')` is `True`.
- **Evidence:**
  - DistilRoBERTa-base is cased, so such an edit changes scores without a relock.
  - A reattributed `source_dataset` changes which slices are gated.
  - No current pipeline step does either, so this is hardening.
- **Confidence:** high

### R08-6: Amendment 4's length-note filter misses notes past character 120 of the opening line

- **Severity:** Low
- **Category:** data integrity / generation artifact
- **Location:** `src/corpus/generate.py:79-81` (opening check limited to `[:120]`, ending check to `[-80:]`)
- **Repro steps:** Run `load_generated` on each source. Search the kept answers for `_LENGTH_NOTE` anywhere and record where it matches.
- **Expected vs actual:** Amendment 4 says answers restating the requested length are dropped. Instead 18 kept answers still do:
  - 16 are opening-line notes past character 120. Example (Gemma, Dolly train): `…positive impact of social media on our day-to-day lives, aiming for approximately 400 words:`
  - 2 are near the end. Example (Ministral, OpenAssistant val): `*Word count: ~230.* Would you like me to…`
- **Evidence:**
  - By split: dolly train 8, val 1; oasst1 train 8, val 1.
  - None is in the locked test. The eight Stack Exchange test hits of a looser "aiming for" pattern are false positives.
- **Confidence:** high

### R08-7: Windowed scoring leaves words unscored in token-dense text

- **Severity:** Low
- **Category:** functional gap (R07-2 claims the text is "scored in full")
- **Location:**
  - `src/authorship/detector.py:124-126`
  - `src/authorship/excerpts.py:48-72` (160-word windows assume at most 254 tokens; each piece is cut at `max_tokens`)
- **Repro steps:**
  1. With the model's tokenizer only, take every long train/val paired text.
  2. Mark the words covered by the opening, cut at 254 tokens.
  3. Mark the words covered by each later window, cut at 254 tokens.
- **Expected vs actual:** Every word should be scored. Instead 135 texts have unscored words: median 22, 75th percentile 48, maximum 343.
- **Evidence:** Most are code, ASCII art or lorem ipsum (OpenAssistant 122, Dolly 13). A mixed document could hide text there.
- **Confidence:** high

### R08-8: Short near-copies still cross MAGE test and training

- **Severity:** Low
- **Category:** data integrity / leakage (R07-6 residual)
- **Location:**
  - `src/corpus/dedup.py:87` (`CONTAINMENT_MIN_SHARED = 3`)
  - `src/corpus/dedup.py:134` (the opening check needs 30 or more words)
- **Repro steps:**
  1. Read the v3 MAGE rows from the raw CSVs by `source_record_id`, in chunks. 58,862 of 58,862 reproduce their `sample_id`.
  2. Compare each evaluation text's 1-in-8 sampled 13-grams with the union of the 13-grams of all train/val MAGE text.
- **Expected vs actual:** No test text should be mostly present in fitted text. Instead 16 of 4,953 evaluation texts are at least 50% contained, and 12 at least 90% (9 AI, 7 human). v2 had 549 and 181.
- **Evidence:**
  - All 16 are 16–52 words long with only 1–3 sampled grams, below both link rules.
  - Example: three test texts open "You don't need to do anything special in the shower. Just shower as you normally would…" beside a train text.
- **Confidence:** high

### R08-9: Pre-registration record errors

- **Severity:** Low
- **Category:** process integrity
- **Location:** `increment-4c-preregistration.md`
- **Repro steps:**
  1. `git show ab0c1f9 -- docs/bob/llm-authorship-checker/increment-4c-preregistration.md`
  2. Count `options.num_ctx` by generator in `raw/*/generated/*.jsonl`.
  3. Merge the Stack Exchange pairs with `v3/manifest.parquet`.
- **Expected vs actual:** Amendments should be append-only and accurate. Instead:
  - **Amendment 2 was rewritten in place.** Its 15:00 text (Gemma one request at a time) was replaced, and the timestamp changed to 15:25 with "All of them run batched". Amendment 3 then reversed that.
  - **Context sizes:** Amendment 2 and the `generate.py` comment say only "the first 1,201 Llama answers used 3072". **177 ministral-3:3b answers** did too.
  - **Stack Exchange count:** Amendment 4 says the gated slice keeps 825 AI answers. The lock has **824**; the `money:38926` pair left through a component shared with another source.
  - **Amendment 5's timestamp** was first written as 01:20, 40 minutes after its own commit, and then corrected.
  - **Build record timing:** the failed first build wrote `data/benchmark/v3/build_record.json` before Amendment 5 existed, because `build.py:321-324` writes the record before raising. Amendment 5 discloses the build. Only the final record postdates every amendment.
  - **Source of Amendment 5:** it says everything came from the audit and the probe. The builder's diagnosis script (`scratchpad/diag_audit.py`, output `diag_audit.out`) also printed text of first-build test rows. This was a model-free read, like the Amendment 1 caveat in R07.
- **Evidence:** as listed. The data are unaffected, since each record stores its own options.
- **Confidence:** high

### R08-10: Retirement checks can be switched off silently

- **Severity:** Low
- **Category:** technical soundness
- **Location:**
  - `src/corpus/build.py:264`: passing `retired=` sets `retired_groups = frozenset()` for both the split and the audit.
  - `src/corpus/audit.py:133-135`: the audit CLI skips a missing earlier manifest, while the build raises.
- **Repro steps:** Read the lines above. `build(sources=..., retired=frozenset({...}))` then retires by id only, and the audit's group check sees an empty set.
- **Expected vs actual:** Retirement by group, the Amendment 5 fix, should not depend on how `build` is called, and the audit should fail when it cannot consult a retired test. Instead both checks turn off without an error.
- **Evidence:** The CLI path used for v3 did not pass `retired=`, so the v3 build was unaffected. This is hardening.
- **Confidence:** high


## Triage (build chat, 2026-09-28)

Training on the old view was stopped at 01:37 because of R08-2, and its model directory was deleted. The live re-check below used the v3 model on CPU, on validation text only, with the gate forced open as `benchmark.run` does. No locked text was scored. The suite is now 255 tests, all passing. Details are in Amendment 6.

| ID | Triage | What changed | Evidence after the fix |
|---|---|---|---|
| R08-1 | **Fix** | `obfuscation.py` also removes default-ignorable marks outside Cf, and maps Latin-script lookalikes (small capitals, dotless i and j, script g, Latin alpha, long s, iota). Accented letters stay. | 40 validation texts the v3 model labels Likely AI: 15% lookalikes, U+034F in 30% of words and U+FE0F in 30% of words each give 40/40 Likely AI, all with the obfuscation note. |
| R08-2 | **Fix, retrain** | New model view `format+typography+collapse_whitespace`. It drops headings, quote and list markers, rule lines, emphasis and code ticks. The probe reads it. | Probe with added formatting features: dolly 0.858 → 0.841 and mirage 0.860 → 0.842; every source is under 0.85. |
| R08-3 | **Acknowledged** | No change to the gate. OpenAssistant's false-AI rate is reported beside it. Stack Exchange and No Robots carry the gated assistant-style human test. | — |
| R08-4 | **Fix** | `registry.seal` binds a benchmark result to a SHA-256 of the entry it scored. `verify()` also requires a text view, and config and tokenizer hashes for both models. | New registry tests: an edited band after a pass, an unsealed pass, a missing hash and a missing view all keep the gate closed. |
| R08-5 | **Fix** | The lock line adds the source and the exact text's SHA-256. v3 is relocked with its reason. | New lock test: recasing text or relabelling a source changes the hash. |
| R08-6 | **Recorded** | Amendment 4's rule is left as written, since changing fitted rows could move components. All 18 are fitted, and none is tested. | — |
| R08-7 | **Debt** | Token-dense text (code, ASCII art) can leave words unscored. | Ledger. |
| R08-8 | **Debt** | 16 short MAGE test texts fall under the containment link minimums. | Ledger; reported with the MAGE slice. |
| R08-9 | **Corrected** | Record errors are corrected in Amendment 6. | — |
| R08-10 | **Fix** | `build.retirement` reads groups unless they are given. `audit.earlier_test_groups` raises on a missing manifest. | Two new tests. |
