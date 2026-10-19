# wreck-it-ralph report: R12 part 1 (code, corpus v4, head-to-head)

Verdict: **FAIL** for part 1 (two High findings: R12-1 and R12-2).

Summary: tests **430/430** passed (`CUDA_VISIBLE_DEVICES= python -m pytest -q -p no:cacheprovider`, 84.7 s, run once). No end-to-end examples were given. Findings **C0/H2/M5/L11**.

Range: `6221d3b..HEAD`, where HEAD is `8740ef7` (Amendment 12, 2026-09-29 19:56:52 +0530). HEAD didn't move during the review, and the working tree stayed clean apart from this report.

**Scope of part 1.**
- Checked fully: claims 1, 2, 3, 4, 8, 10 and 11.
- Code only for claims 5, 6, 7 and 9 ("code checked; artifacts in part 2"). Full training was running during the review, so `training_record*.json`, `calibration.json`, `stack.json`, `registers.json`, `speed.json` and the 4d registry record don't exist yet or aren't final.

**How the rules were kept.**
- **Files:** nothing in the repo was edited or committed except this report. Scratch scripts and outputs are in the session scratchpad under `scratchpad/r12/`.
- **Commands not run:** headtohead, benchmark, training, calibration, generation, the corpus build and the speed check.
- **Models folder:** nothing under `D:\llm-authorship-corpus\models\` was opened or listed, and no model was loaded.
- **GPU:** not used. Every Python process ran with `CUDA_VISIBLE_DEVICES=` set.
- **Files not opened:** `x`, `inspect_wb.py`, the root `.xlsx` sheets and `dataset_full.csv`. No rater is named.
- **RAM:** manifests were read with column selection only. `corpus.parquet` was only streamed in 3,000–4,000-row batches with the columns needed (peak about 1.4 GB per process). It was never loaded whole.
- **Locked text.** No locked row was scored by any model, and no locked text is quoted in this report. Test texts were read only to compute n-gram hashes, word counts and sample ids; nothing was printed. That happened in three processes (two near-duplicate passes and the MAGA/DACTYL word counts), plus a reload of the LEDE and WasItAI raw files. All quoted examples are train or validation rows, or raw generator output for training sources.
- **The training log** (`D:\llm-authorship-corpus\train-4d-main.log`, outside the models folder) was read for run start times and check times only.

**Helpers.** Two sub-reviewers read the code in parallel:
- one for claims 5, 6, 7 and 9;
- one for claim 8.

I checked their findings against the code before including them. R12-2 was found by both independently, and I reran its repro.

## Claims checked

| # | Claim | Result | Evidence |
|---|---|---|---|
| 1a | No v4 `test` / `test_unseen_generator` row matches a v1–v3 row (any split) by sample id, group or near-duplicate | **met for ids and for the project's near-duplicate rule. Not met for groups (R12-7). Stricter check: R12-8** | 41,399 tested rows (`scratchpad/r12/c1_ids.py`):<br>- **Sample id:** 0 in any earlier split.<br>- **(source, source_record_id):** 0.<br>- **Group key:** 0 in earlier fitted or tested splits, but **7,749 in earlier stress splits**: 7,746 MIRAGE human test rows whose LLM polishes and rewrites v1–v3 scored as stress, and 3 RAID AI rows.<br>- **Near-duplicates:** an independent all-8-gram check finds 7 tested rows sharing a verbatim run of 30 or more words with earlier text (6 MIRAGE humans against their v1 stress polishes), and short MAGE prefixes the 30-word opening rule can't see (R12-8). |
| 1b | Held-out sources are tested whole, except rows an earlier version saw | **met, with a small exception (R12-10)** | Loaded vs kept:<br>- wasitai2026: 1,556 loaded, all kept (1,135 test + 421 stress).<br>- stackexchange_heldout: 7,072 loaded, 7,063 kept. The 9 lost rows are exact duplicates within the source (7,063 distinct ids), and every distinct id is in v4 test or unseen.<br>- lede: 4,999 loaded, 4,996 kept. The 3 missing rows are **not** in v1–v3 by id or group, and are not duplicates, so they left for a cross-source component (`split.py:93-95`), not for an earlier sighting.<br>No held-out row is in train or validation (audit check, and the manifest). |
| 1c | Existing sources took rows no earlier version took where the pool allowed; test rows per source match Amendment 2 | **mechanism met. Amendment 2's numbers don't match (R12-9)** | **Fresh share of v4 core rows:**<br>- mage ~97%.<br>- hc3 42–45%.<br>- no_robots 44%.<br>- mirage humans 100%, mirage AI 3% (pool used up).<br>- raid AI 100%, raid humans 0%.<br>- aita2026 and beemo 0%.<br>**Against Amendment 2:** it says HC3 "took every row in v3" and gives no fresh test rows. The build has **358 HC3 human and 402 HC3 AI test rows** (finance and reddit_eli5), none of them seen before, so HC3 is a gated slice.<br>**Other sources:** aita2026, beemo, dolly and oasst1 have 0 test rows, and RAID has 11 AI and 0 human. |
| 1d | Stress slices kept their rows; a core copy of earlier stress text was dropped, not the stress copy | **mostly met (R12-10)** | 596 core copies of report-only text were dropped before duplicate resolution (`build.py:216-217`). v4 has 0 core rows with a report-only earlier id.<br>Of v3's 41,538 stress rows, v4 keeps **41,117**. The 421 dropped are MIRAGE polish (324) and rewrite (97) rows. Their groups are absent from v4, and only 1 of 421 is in an earlier fitted or tested group: their human parents were retired as near-duplicates of earlier text, and the whole component left (`split.py:105`). |
| 1e | The build settled splits on the audit's own earlier-text check, and the audit passed | **met** | `build.py:244-257` loops over `audit.earlier_matches` until no tested row matches (`retired_by_audit_check: 0`). The build record has `audit.passed: true` and 0 errors. The one warning is 28 test slices under 300 humans. |
| 2a | Shortcut and AI-only probes: no source over 0.85 without a recorded decision | **met; the decision's rationale is wrong (R12-5)** | **Shortcut probe:** only Stack Exchange is over the line (own probe 0.8911), ruled on in Amendment 10.<br>**AI-only probe:** arena140k 0.8424, openturingbench 0.8459, both under 0.85. The pre-registration's dry run had predicted OpenTuringBench over the line (0.863); see R12-12. |
| 2b | No surface difference separates the labels on the model's view | **not met (R12-1, R12-3, R12-11)** | Measured on train and validation only, on the format3 view (`scratchpad/r12/c2_surface.py`, `surface.out`):<br>- **Assistant preambles** ("Here is the body of the Reddit post:", "Certainly! Here's an improved version of your NPR news article:"): MAGA AI 12.8–13.3%, MIRAGE AI 5.4%, against 0.04% and 0.12% of their humans.<br>- **News template** ("Date: … Category: …", "HEADLINE:"): 26.8% of OpenTuringBench AI rows, against ~0% of its human news pool.<br>- **Emoji:** 2.95% of AI rows against 0.29% of human rows.<br>- **Stack Exchange length:** 52% of AI answers are under 100 words, against 24% of human answers.<br>- **Humanized rewrites:** "Here's the casual rewrite: …" appears (6 of 233). |
| 2c | MAGA and DACTYL are length-matched per domain | **not met (R12-1)** | Band counts per domain are far apart, for example MAGA amazon_reviews 50–99 words: 103 AI vs 744 human, and 200–399 words: 501 vs 209. Word count alone separates MAGA at AUC 0.645. The cause is in `_common.py`: the cap reuses the length matcher's hash order. Reproduced with the project's own functions. |
| 3a | `format3` is idempotent and leaves ordinary prose alone | **met, with minor prose edits** | **Idempotence:** 0 non-idempotent cases in 20,000 random markup strings (`c3_format3.py`), plus the repo's own nested-marker tests.<br>**Prose:** stays apart from the designed dash rule, but a free-standing `(1)`, `10)` or `8)` in a sentence is removed ("He scored 10) points" → "He scored points"). |
| 3b | Every bullet glyph, `(1)` / `1)` marker and dash form goes | **partly met (R12-4)** | **Removed:** the 19 listed glyphs, `(1)`, `1)`, en and em dashes, U+2015 and a free-standing `--`.<br>**Kept:**<br>- glyphs ◆ ◇ ★ → ➜ ▶ ♦ ❖ ☐ ☑ ✘ ⬤ ◉ and emoji bullets (✅ 🔹);<br>- markers `a)`, `i.`, `①`;<br>- a `1.` alone on its line when it isn't the text's first token, and an unspaced `1)Mix`.<br>**Dash forms:** small em dash U+FE58, two-em dash U+2E3A and three-em dash U+2E3B are kept and glue words. Figure dash U+2012 becomes a word hyphen, the D15 behaviour. |
| 3c | A formatting or character trick can't turn Likely AI into Likely human | **unverifiable in part 1. Gaps found (R12-4)** | No 4d model exists yet.<br>**Silent tricks:** replacing every space with a punctuation-class separator (U+30FB, U+FF65, U+2E31, U+16EB, U+0701, U+10101, U+205A, U+2E30) leaves the view as **one glued token string**, with no abstention and no note (residual 0).<br>**Band shift:** numbered markers add policy words (77 → 83) while the view drops them, so a text can change word band and model without changing what the model reads.<br>**Covered:** 41 other tricks are read back or abstain (`c3_tricks.py`). |
| 4a | Every variant record has variant, seed salt, style, options, model digest and licence | **met** (humanized records carry no style, by design) | All 11,434 4d records (`c4_gen.py`):<br>- **Always present:** variant, seed_salt, options, a digest and a licence line. There is one digest per generator.<br>- **Consistent with the code:** variant matches the folder, generator matches `pick()`, seed matches `seed_of()`, style matches `style_of()`, and temperature and system prompt match the style.<br>- **Humanized:** the 234 records have no style or target_words, because they use the fixed rewrite prompt. |
| 4b | Counts match the pre-registration table | **met** | Items answered:<br>- stackexchange train4d 2,500 and short4d 2,500;<br>- dolly short4d 1,000; oasst1 short4d 1,000;<br>- stackexchange_heldout test4d 2,100 and short4d 2,100;<br>- humanized rewrites 234 ("about 250").<br>Short targets are all within 25–99 and standard targets within 30–400. Temperatures are within 0.6–1.1. Personas split 25/25/25/25%, and the plain flag covers 29.9%. |
| 4c | The default variant keeps v3's pools; held-out generators never answer training items | **met** | `DEFAULT_POOLS` equals v3's pools, and the `generated/` files are dated before 4d began. No held-out generator appears outside stackexchange_heldout in any 4d variant. v3's Phi and Granite answers on the eight v3 sites are excluded from v4 (0 test_unseen rows in stackexchange; the audit passed). |
| 4d | No artifacts (refusals, meta-commentary, word-count echoes, preambles, wrong language) | **not met in full (R12-3, R12-13)** | After the Amendment 4, 8 and 9 filters (`c4_artifacts.py`):<br>- **Refusals:** 29 in total ("I can't provide veterinary advice…").<br>- **Word-count mentions mid-text:** 37, which the start/end filter misses.<br>- **CJK code-switching:** 8.<br>- **Emoji:** Gemma and Ministral use them in 8–20% of 4d answers.<br>- **Openers:** "Great question!" and "Absolutely!" remain.<br>- **Humanized rewrites:** keep "Here's the casual rewrite:" preambles. |
| 5 | Training reads only train and validation; quarter-epoch checks; best check kept; LoRA merged; digests | **code checked; artifacts in part 2** | **Data:** `bakeoff/data.py:29-31` filters and asserts the split.<br>**Checks and stopping:** 4 checks per epoch, patience 2, at most 3 epochs (`train.py:42-44`). The best check is kept across a resume (`train.py:189-191`).<br>**Merge:** into `merged-<best>` (`train.py:224-226`), and calibration loads that path.<br>**Settings:** fullbatch is 16 × 1 with checkpointing, lr 2e-4, r=16, alpha 32, fp16.<br>**Running log:** checks 0.1 / 0.2 / 0.3 at micro-batches ~3,005 / 6,011 / 9,016 of 12,022.<br>**Gaps:** the committed record has no validation-ids digest (R12-18). |
| 6 | Calibration targets; routing; precision in registry and detector; seal coverage | **code checked; artifacts in part 2** | **Routing:** the 1-point rule is at `calibrate.py:103-107,208`.<br>**Precision:** copied to both models (`calibrate.py:220,253`) and used per model (`registry.py:241-242`). The GPU uses fp16 weights, the CPU fp32. A record without a precision stays closed (`registry.py:182-186`).<br>**Seal:** covers everything but the run fields, a scoring-code list that includes `text.py`, `obfuscation.py`, `calibration.py` and `detectors/base.py`, and the report.<br>**Gaps:**<br>- The targets depend on CLI flags; the defaults are 1% and 10% (R12-18).<br>- Non-finite scores don't stop a locked run (R12-2). |
| 7 | Optional steps decided on validation only | **code checked; artifacts in part 2** | **Stack:** halves split by a parent-id hash, fitted on one half and measured on the other; adoption needs 2 points and the GPU limits (`stack.py:39-42,72-74`).<br>**Register thresholds:** cells under 600 humans keep the pooled bar (`registers.py:103-109`); adoption needs 2 points and every cell within target (`registers.py:157`).<br>**No auto-adoption:** neither step wires itself into the service, so each needs its amendment. |
| 8a | Today's models go through `models/incumbent_registry.json` (format2, bf16, their bands and excerpt view), refused if the v3 seal breaks | **met (R12-14)** | **The record:** equals `git show 6221d3b:models/registry.json` after CRLF normalisation, and its seal verifies.<br>**Unchanged scoring:** the format2 fingerprint `f3289b1f…` is unchanged. There is no precision key, so fp32 weights run under bf16 autocast (`registry.py:31,241-242`, `detector.py:98-109`). Bands come from the record, and the excerpt view is the record's own (`benchmark.py:161`).<br>**Refusal:** tested.<br>**Not checked:** the incumbent's scoring-code digest now differs (`04e3d354` recorded, `97e527e8` current). |
| 8b | Same rows; shared ground, gated slices and the outcome table match the pre-registration and amendments | **met in code (R12-15, R12-16)** | **Same rows:** one frame for both passes; `improvement()` refuses unpaired rows (`headtohead.py:85`).<br>**Shared ground:**<br>- existing and frontier test rows;<br>- stackexchange_heldout test rows excluded (Amendment 2 item 4);<br>- unseen-generator rows only from shared-ground sources (Amendment 4).<br>**Rule and outcome:** gated slices need at least 100 AI rows, the unseen split can't block, and the outcome table includes Amendment 3 item 4.<br>**Pooled AI rows by source (8,878):** lede 4,996 (56.3%), stackexchange_heldout unseen 1,670, mage 921 + 30 unseen, wasitai2026 746, hc3 402, mirage 102, raid 11.<br>**Leakage:** none from maga, dactyl, arena140k, openturingbench or stackexchange. |
| 8c | Recorded before any locked text is scored; a second run needs a reason | **met** | The lock hash is checked and the `started` entry saved (`headtohead.py:158-177`) before any service is built. The history entry, the benchmark block or an existing report each force `--rerun-reason` (`:152-154`). |
| 8d | Each model set loaded only for its own pass; no GPU out-of-memory after recording | **met for loading. Not safe on failure (R12-2)** | **Loading:** `del`, `gc.collect()` and `empty_cache()` run after each pass (`:202-205`), and a weakref test checks the release. The excerpt service reuses the loaded main model (`benchmark.py:137-142`).<br>**Memory:** two fp16 DeBERTa-v3-large models come to about 1.75 GB of weights, against a 1.59 GB scoring peak measured for one (Amendment 12).<br>**Failure:** an out-of-memory error or a NaN doesn't stop the run; it is recorded as Inconclusive and sealed (R12-2). |
| 10 | Every amendment committed before the results it could affect; pass criteria unchanged since v1 | **met, with process notes (R12-12)** | **Timeline (IST):**<br>- Pre-registration 09-28 11:39:51; Amendments 1–5 by 13:29:40.<br>- Amendment 6 14:45:03. The first 4d generation record is 14:45:24.<br>- Amendment 7 18:08; Amendment 8 18:44 (during generation, applied at load).<br>- Amendment 9 09-29 17:38:40. The humanized rewrites ran 17:32–17:57, all from Amendment 9-trimmed text.<br>- v4 build `built_at` 18:42:33. Amendment 10 19:01:34.<br>- Screening-settings training started 19:25:18 and stopped before any check. Amendment 11 was committed 19:25:36.<br>- Amendment 12 19:56:52. The fullbatch run started 19:57:05, and its first check was 21:32.<br>**Pre-registration body:** edited only by Amendment 1, before generation; the rest is append-only.<br>**Gate:** constants `0.01 / 0.10 / 0.40 / 300 / 100` unchanged in the range (`benchmark.py:42-43`), and the text is identical to 4b and 4c.<br>**In-house bound:** the v1 roadmap's in-house rater bound was already report-only before this range (sponsor decision), so this range didn't move it. |
| 11 | The suite passes with no GPU and no model files | **met** | 430 passed, 0 skipped, with CUDA hidden. `tests/conftest.py` points the registry at a missing file, the tests build tiny local encoders, and no test path reads `D:/`. |

## Findings

### R12-1: MAGA and DACTYL are not length-matched: the per-domain cap reuses the length matcher's hash order and puts the length gap back

- **Severity:** High
- **Category:** data integrity (shortcut)
- **Location:**
  - `src/corpus/sources/_common.py:100-112` (`take_per_stratum`) and `:119-139` (`length_matched`). Both order rows by `sha256(key)`.
  - `src/corpus/sources/maga.py:59-60` and `src/corpus/sources/dactyl.py:42-43`: `length_matched(...)` then `take_per_stratum(..., key="id")`.
- **Mechanism:**
  - Within a band where one label is the majority, `length_matched` keeps that label's rows with the *smallest* hashes. The minority label keeps all its rows, spread over the whole hash range.
  - The cap then takes the smallest hashes per (domain, label) across all bands. It over-selects each label's majority bands: AI keeps its long bands and humans their short ones.
- **Repro:**
  1. Synthetic, with the project's functions: one domain, humans log-normal around 80 words, AI around 220, pools larger than the cap.
  2. Real data: stream `corpus.parquet` (maga and dactyl rows, word counts only) and count rows per domain, band and label.
- **Expected vs actual:** the pre-registration says "Length-matched per domain, 1,500 per domain and label" (MAGA) and "2,000 per domain and label" (DACTYL). Actual:
  ```
  synthetic, rows per band (<25, 25-49, 50-99, 100-199, 200-399, 400+)
  after length_matched          ai  1  85 3256 6051 1197 66 | human 1 85 3256 6051 1197 66
  after take_per_stratum 1500   ai  0   0   88  624  722 66 | human 1 85  764  543  105  2
  same cap, independent shuffle ai  -  12  449  854  172 13 | human - 13  438  862  178  9
  ```
  ```
  v4 corpus, rows AI / human (all splits, word counts only)
  maga amazon_reviews   50-99: 103 / 744    100-199: 719 / 403    200-399: 501 / 209
  maga reddit           50-99: 109 / 537    200-399: 629 / 225    400+: 304 / 84
  maga yahoo_answers    <25:   22 / 160     200-399: 323 /  78
  dactyl reviews        <25:  153 / 818     50-99: 1026 / 391
  dactyl tweets         25-49: 379 / 64
  dactyl abstracts      200-399: 24 / 370
  ```
- **Evidence:**
  - On train and validation (format3 view), word count alone separates MAGA at AUC **0.645**.
  - MAGA's band shares:

    | Band | AI | Human |
    |---|---|---|
    | 50–99 words | 12.7% | 30.3% |
    | 200–399 words | 34.0% | 19.6% |

  - The shortcut probe's `log_words` feature is diluted in the shared probe; MAGA's own probe is 0.827.
  - The 4d main model is training on this data now. Bands under 200 words sit fully inside its window.
- **Fix:** draw the cap with an order independent of the matcher's, or cap within each (domain, band). Then rebuild. A rebuild moves the lock, and the model trained on the current corpus has learned the gap.
- **Confidence:** high

### R12-2: A detector exception during the locked run (NaN guard, out-of-memory, load error) turns a whole 4,000-text chunk Inconclusive, and the run is still sealed

- **Severity:** High
- **Category:** reliability / data integrity. Found independently by both sub-reviewers; I reran the repro.
- **Location:**
  - `src/detection/service.py:60-66` and `:148-149`: any exception from `detect_batch` becomes `DETECTOR_ERROR` for every pending text in the call.
  - `src/authorship/benchmark.py:47` and `:82-89`: `CHUNK = 4000` texts per call.
  - `src/authorship/benchmark.py:44-45`: `DETECTOR_ERROR` isn't in `UNSUPPORTED`, so these rows count as supported and Inconclusive.
  - `src/authorship/detector.py:130-133`: the new non-finite guard raises `FloatingPointError`.
  - `src/authorship/detector.py:150`: loading happens inside `detect_batch`.
  - `src/authorship/headtohead.py:187-232` and `benchmark.gate`: neither looks for `DETECTOR_ERROR`, and the run is sealed.
- **Repro:** `PYTHONPATH=. python scratchpad/r12/fork-a/nonfinite.py`. A fake detector raises `FloatingPointError` when any text in its batch is poisoned.
- **Expected vs actual:** Amendment 5 item 4 says "Non-finite scores stop scoring with an error." Claim 8 asks that the run can't run out of GPU memory after it is recorded. Actual:
  ```
  decision      reason
  inconclusive  DETECTOR_ERROR    7
  {'ais': 7, 'detected_ai': 0, 'supported': 7, 'coverage_supported': 0.0, 'reasons': {'DETECTOR_ERROR': 7}}
  without poison: {'likely_ai_generated': 6}
  ```
- **Evidence:**
  - One bad text removes every decision in its 4,000-text chunk.
  - Coverage and detection fall, and the gate, the improvement rule and the outcome are computed anyway. `registry.seal` then seals the record. The only trace is a per-slice `reasons` count.
  - fp16 DeBERTa-v3 is exactly the case the guard was added for, and `torch.OutOfMemoryError` takes the same path.
  - Calibration calls the detector directly, so a NaN on validation would crash loudly. The locked texts are different texts.
- **Fix:** abort the run, or refuse to seal it, on any `DETECTOR_ERROR`, `DETECTOR_INVALID_RESPONSE` or `DETECTOR_REPORTED_ERROR` in the locked splits.
- **Confidence:** high on the behaviour, medium on how likely it is.

### R12-3: Assistant preambles and generation templates remain in training AI text, the class Amendments 8 and 9 trimmed, and no probe sees them

- **Severity:** Medium
- **Category:** data integrity (shortcut)
- **Location:**
  - Loaders `src/corpus/sources/maga.py`, `mirage.py` and `openturingbench.py`; `generate.load_generated`, which trims only greetings and "Okay,".
  - The probes read length and punctuation features only (`src/corpus/confound.py:27-31`).
- **Repro:** stream the train and validation rows, apply `format3_view`, and match the openers below (`scratchpad/r12/c2_surface.py` and the inline script in this review).
- **Expected vs actual:** claim 2 asks for no surface cue that separates the labels, including preambles. Actual, on train and validation:

  | Source | "Here's / Here is …" opener | Meta preamble ("here's a … version / body / article …:") | News template (Date: / Category: / HEADLINE:) |
  |---|---|---|---|
  | MAGA AI, aligned | 13.3% (643) | 13.9% (672) | 0.2% |
  | MAGA AI, plain | 12.8% (768) | 9.3% (560) | 0.1% |
  | MAGA human | 0.04% (4) | 0.01% | 0.02% |
  | MIRAGE AI | 5.4% (1,596) | 5.3% (1,548) | 0.0% |
  | MIRAGE human | 0.12% | 0% | 0.02% |
  | OpenTuringBench AI | 0.2% | 0% | **26.8% (2,060)** |
  | Stack Exchange humanized rewrites | 2.6% (6) | 1.7% (4) | 0% |

- **Evidence:**
  - Top MAGA AI openers in train: "Here is the body …" (175), "Certainly! Here's an improved …" (114), "Here is a potential …" (97), "Okay, here's the body …" (44).
  - These texts announce themselves, and MAGA's "aligned" rows are the ones built to read as human.
  - OpenTuringBench rows carry the prompt's metadata template ("Date: November 26, 2013 Category: WEDDINGS"). The sources of its human pool don't: raid, dactyl and maga humans are at 0–0.02%. The AI-only probe missed it (0.846).
  - Amendment 9 trimmed "Okay," because a 38.5% vs 0.21% gap "would teach" a shortcut, and rater comments are the product's main input. The same reasoning applies here at 13% vs 0.04%.
  - The risk runs both ways. It inflates detection on MAGA, MIRAGE and OpenTuringBench, and a rater comment that opens "Here's my rewrite:" leans AI.
  - Offsetting: human oasst1 and no_robots answers open "Here's" 5–8% of the time, which limits the cue in the assistant register.
- **Fix:** trim leading self-announcing preambles and metadata template lines at load, as Amendments 8 and 9 did, or record a decision.
- **Confidence:** high on the counts, medium on the effect on the trained model.

### R12-4: Character and format tricks that are neither read back nor abstained on

- **Severity:** Medium
- **Category:** robustness / evasion (the R07-1 "normalize or abstain" bar)
- **Location:**
  - `src/detection/obfuscation.py:195-202` counts only S* and Nd characters touching ASCII letters, not P* (punctuation).
  - `src/corpus/text.py:68-70`: `_BULLETS3`, `_NUMBERED3` and `_DASHES3` are closed lists.
  - `src/detection/policy.py:235-237` counts words on the deobfuscated text, not on the model's view.
- **Repro:** `PYTHONIOENCODING=utf-8 python scratchpad/r12/c3_tricks.py`, plus the separator loop in this review (a 21-word sentence with each space replaced).
- **Expected vs actual:** a trick should be read back or abstain with a note. Actual:
  ```
  U+30FB KATAKANA MIDDLE DOT       abstain=False residual=0 policy_words=21 view_words=1
  U+FF65 HALFWIDTH KATAKANA MIDDLE abstain=False residual=0 policy_words=21 view_words=1
  U+2E31 WORD SEPARATOR MIDDLE DOT abstain=False residual=0 policy_words=21 view_words=1
  U+16EB RUNIC SINGLE PUNCTUATION  abstain=False residual=0 policy_words=21 view_words=1
  U+205A / U+2E30 / U+0701 / U+10101: same
  small em dash U+FE58 joins       view_words 71 of 77 (words glued); two-em dash U+2E3A the same
  bullets ◆ / → / ✅ / 🔹           kept in the view (view_words 83 vs 77)
  numbered "1." / "(1)" / "1)"     view identical to the original, but policy_words 77 -> 83
  ```
- **Evidence:**
  - With a dot separator in place of every space, the model reads one glued string. The service gives no note (`changed=False`) and no abstention.
  - Numbered markers raise the word count used for the band and for routing while the model's view drops them. A 95-word text with markers can move from the 50–99 band (short model, its thresholds) to 100+ (main model, other thresholds) with identical model input.
  - 41 other tricks were read back or abstained correctly: homoglyphs, fullwidth and math letters, dot below, zero-width, soft hyphen, NBSP, ideographic, en quad, U+2028, tabs, hair spaces, variation selectors, Hangul fillers, U+180E and the U+2800 braille blank.
  - No 4d model exists, so whether any trick flips Likely AI to Likely human can't be measured in part 1. R11-1 showed that this kind of silent gap did flip v5.
- **Fix:**
  - Count non-ASCII punctuation between ASCII letters as residual.
  - Treat other-script word separators as spaces.
  - Extend the bullet and dash lists by class (Unicode Pd; So bullets at line or sentence start).
  - Count policy words on the model's view.
  - Part 2 should score these tricks on the 4d model.
- **Confidence:** high on the behaviour, low on the flip rate.

### R12-5: Amendment 10 keeps Stack Exchange past the probe line on a per-site safety net the gate doesn't have

- **Severity:** Medium
- **Category:** process integrity
- **Location:**
  - `increment-4d-preregistration.md`, Amendment 10: "shows as false AI on the 8 gated held-out Stack Exchange sites …, each held to the ≤1% bound".
  - `src/authorship/benchmark.py:108-115`: `gate()` groups by `source_dataset`.
- **Repro:** `python -c "from src.authorship.benchmark import upper_bound; ..."`: find the largest k with `upper_bound(k, n) ≤ 0.01`.
- **Expected vs actual:** each site should be gated. Instead stackexchange_heldout is **one** slice of 2,991 humans:
  ```
  2991 max false AI passing: 20     374 max false AI passing: 0
  ```
- **Evidence:**
  - All 20 passing false-AI texts could fall on one site (about 5% false AI on, say, academia), and the slice still passes.
  - The "30 gated test slices" in the amendment are audit cells (source × domain), not gate slices.
  - The code follows the unchanged pass criteria, as v3 did. But the amendment's reason for keeping a source at 0.891 (and not fixing or excluding it, the options the pre-registration named) rests on a stronger protection than exists.
- **Fix:** amend the ruling's rationale, or report per-site false AI beside the gate, before the lock is read.
- **Confidence:** high

### R12-6: The head-to-head spends the lock without checking the 4d record's `verify()` preconditions

- **Severity:** Medium
- **Category:** reliability (one-shot run)
- **Location:**
  - `src/authorship/headtohead.py:162-170` calls only `registry.verify_files`.
  - `registry.py:172-188` holds the checks it skips: the scoring-code digest, the view fingerprint and the precision.
- **Repro:** read the code path. A record whose `scoring_code_sha256` no longer matches (for example after an optional step is wired into `service.py`) still passes `verify_files`. The run is recorded and scored, and the record is sealed.
- **Expected vs actual:** a run that can only end in a record `verify()` rejects ("scoring code changed since calibration") should be refused before the `started` entry is written. Actual: the lock is spent, and a rerun then needs a reason.
- **Evidence:** `run()` saves the `started` entry at `:176-177` after `verify_files` alone.
- **Fix:** run `verify` without the benchmark fields before recording.
- **Confidence:** medium

### R12-7: 99.5% of MIRAGE's test humans share a group with v1–v3 stress rows; the audit's group check covers only fitted and tested splits

- **Severity:** Medium
- **Category:** data integrity (freshness claim)
- **Location:**
  - `src/corpus/build.py:67-77` and `src/corpus/audit.py:96-101`: retired groups come only from `SEEN_SPLITS`.
  - `src/corpus/sources/mirage.py:47`: `skip_used` now takes the originals of rewrites that earlier versions scored as stress.
- **Repro:** `python scratchpad/r12/c1_ids.py` (manifests only) and `c1_mirage_sim.py` (raw MIRAGE files).
- **Expected vs actual:** claim 1 says no tested row matches an earlier row "by … group … in any split, report-only ones included". Actual:
  ```
  tested group in earlier any split: 7749  seen splits: 0
  mirage test human 7746, raid test ai 3
  earlier rows sharing those groups: v1/v2/v3 stress mirage llm_polish 6968, llm_rewrite 7642 (each version)
  ```
- **Evidence:**
  - These are 7,746 of MIRAGE's 7,787 human test rows, and 37% of all v4 test humans.
  - Each is the human original of AI polishes and rewrites that v1–v3 scored and reported as stress.
  - Amendment 2 bars only id and near-duplicate matches against stress rows, and those hold. The paraphrases are not near-duplicates: 5-gram containment has a median of 5–7% and a 99th percentile of 37–45% (14,433 pairs).
  - No model ever fitted these texts or their paraphrases, so the head-to-head stays fair.
  - The wording of the claim, and the audit's silence about it, overstate how fresh this slice is.
- **Fix:** state the rule as Amendment 2 wrote it (id and near-duplicate for report-only rows), or retire these groups. Retiring them would leave MIRAGE 41 test humans.
- **Confidence:** high

### R12-8: An independent near-duplicate check finds short prefixes and long shared passages the build's clustering can't see

- **Severity:** Low
- **Category:** data integrity (leakage, small)
- **Location:**
  - `src/corpus/dedup.py:134`: `opening_pairs` needs at least 30 words.
  - `dedup.py:87,122`: the containment check needs 3 sampled 13-grams (1 in 8), which a text under about 40 words rarely has.
  - `audit.earlier_matches` and the build use the same function, so they can't disagree.
- **Repro:**
  1. `python scratchpad/r12/c1_neardup.py` indexes every word 8-gram of the 41,399 tested rows, with no sampling and a boilerplate cut only at more than 50 texts.
  2. It then streams every v1–v3 text (deduplicated by id) and every v4 train and validation text, and flags pairs sharing at least 3 8-grams covering at least 50% of the smaller text.
  3. `c1_neardup2.py` records the matching row, common prefix and longest common word run for each flagged pair.
  4. No text is printed.
- **Expected vs actual:** claim 1 says no tested row matches earlier text as a near-duplicate, and the audit re-checks it. Actual: 210 tested rows (0.5%) are flagged. Of those:

  | Kind | Rows | Detail |
  |---|---|---|
  | **Shared verbatim run of ≥ 30 words** | 13 | Against earlier text: 6 MIRAGE test humans against their own v1 stress polishes (runs of 39–76 words), and 1 MAGE human. Against v4 train and validation: 3 MAGE humans, 2 MAGA humans and 1 LEDE article (a 41-word run from a DACTYL news human). |
  | **Shared run of ≥ 20 words** | 65 | |
  | **Short text (under 30 words) wholly inside another text** | 122 | Mostly MAGE: a 23–29-word text equal to the first words of a longer MAGE text. MAGE continues human prompts, and the 24-word opening rule exists for exactly this (D3). |

- **Evidence:**
  - Of MAGE's human test rows, 13 share a run of at least 20 words, and 16 of their matches are AI-labelled training or earlier rows.
  - The counts are small, and all of them are below the project's own thresholds, so the audit's pass is consistent with its definition.
  - The gap is structural: prefixes under 30 words, and long shared passages inside rewritten text.
- **Confidence:** high on the counts. The significance is low.

### R12-9: Amendment 2's per-source expectations are wrong for HC3, which is now a gated slice where one false AI fails the gate

- **Severity:** Low
- **Category:** process integrity
- **Location:**
  - Amendment 2 item 1: "hc3 and beemo took every row in v3 … the source has no fresh test rows"; "This includes RAID and HC3".
  - `data/benchmark/v4/build_record.json`, `counts.split_source`.
- **Repro:** read `split_source` and the v1–v3 manifests (`c1_ids.py` and the fresh-share script).
- **Expected vs actual:** Amendment 2 expected HC3 to have no fresh test rows and be reported, not gated. Actual: `test|hc3` has 402 AI and 358 human rows (finance 33 and reddit_eli5 325 humans). None is in v1–v3 by id or source record, and 42–45% of HC3's v4 rows are fresh.
- **Evidence:**
  - HC3 now gates both model sets.
  - With 358 humans, one false AI gives an upper bound of 1.32% and fails the gate (`upper_bound(0, 358) = 0.0083`, `upper_bound(1, 358) = 0.0132`).
  - The rule is the unchanged ≥300-human criterion and applies to both sides equally, but no amendment corrects the stated expectation.
- **Confidence:** high

### R12-10: Stress and held-out slices aren't quite whole: 421 v3 stress rows dropped, and held-out rows excluded for a reason other than an earlier sighting

- **Severity:** Low
- **Category:** data integrity
- **Location:**
  - `src/corpus/split.py:105`: a stress row in a retired, forced-evaluation component is dropped.
  - `split.py:93-95`: a held-out row sharing a component with another source is dropped.
- **Repro:** `c1_stress.py`, and a reload of the three held-out sources in this review (loader plus `pii.redact` plus `sample_id` against the manifests; `c1_seheld.py` for Stack Exchange).
- **Evidence:**
  - **Stress:** 421 of v3's 41,538 stress rows (MIRAGE polish 324, rewrite 97) are absent from v4 with their whole group. Only 1 is in an earlier fitted or tested group. Their fresh human parents were retired as near-duplicates of the earlier stress text itself (`near_duplicate_retired_by_source.mirage = 99`), and the component rule then removed the stress rows, which were never tested.
  - **LEDE:** 3 of 4,999 rows are missing although none is in v1–v3 by id or group and none is duplicated.
  - **stackexchange_heldout:** its 9 missing rows are exact duplicates within the source. That is fine.
- **Expected vs actual:** claim 1 says held-out sources are tested whole "except rows an earlier version saw", and that stress slices kept their rows.
- **Confidence:** high on the counts. The cause is inferred from the code, not re-clustered.

### R12-11: Persona and short-variant cues separate the labels in the Q&A register

- **Severity:** Low
- **Category:** data integrity (shortcut)
- **Location:**
  - `src/corpus/generate.py:66-71` (personas) and `:127-129` (short targets drawn independently of the human length).
  - No matching on the Stack Exchange, Dolly or OpenAssistant human side.
- **Repro:** `c2_surface.py` and the band and AUC script (train and validation, format3 view).
- **Evidence:**
  - **Length:**
    - Stack Exchange: 52.5% of AI answers are under 100 words, against 24.2% of human answers. "Shorter means AI" scores AUC 0.666.
    - OpenAssistant: 63.2% against 40.8%, AUC 0.628.
  - **Emoji:**
    - Stack Exchange: Gemma 10.4% and Ministral 8.3% of AI answers, against 0.0% of humans.
    - Overall: AI 2.95% against human 0.29%.
  - **Exclamation marks:** Stack Exchange AI 0.64 per 100 words, against 0.10 for humans.
  - **Casing:** Llama's casual answers are all-lowercase 15.8% of the time, against Stack Exchange humans at 0.1%.
  - These are partly real style of 2025 chat models, and per-band thresholds soften the length cue. Still, the held-out Stack Exchange test has the same structure, and short, casual, emoji-bearing text is what rater comments look like.
  - Amendment 10 attributes the 0.891 probe score to mean word length alone.
- **Confidence:** high on the measurements

### R12-12: Process: two post-build amendments aren't bug fixes; small timing and record inaccuracies

- **Severity:** Low
- **Category:** process integrity
- **Location:** `increment-4d-preregistration.md`: the opening paragraph and Amendments 9–12.
- **Repro:** `git log --date=iso` for the range; `train-4d-main.log` lines 1 and 33; `created_at` in `raw/*/generated_*/*.jsonl`.
- **Evidence:**
  - **The rule:** the opening says that after the test set is built, "nothing below changes except to fix a bug".
  - **Amendment 10** (after the build) replaces "fixed or excluded" with "kept".
  - **Amendment 12** (after the build) changes the training recipe for speed. The loss changes from the mean of two 8-row weighted means to one 16-row weighted mean.
  - **Timing:** both came before any training check or test scoring, so no result could have informed them.
  - **Amendment 11** was committed at 19:25:36, 18 s after the run it governs started (19:25:18). That run stopped before any check.
  - **Amendment 9** says "after generation". The humanized rewrites ran 17:32–17:57, around its 17:38:40 commit. Every rewrite used Amendment 9-trimmed source text (`source_text_sha256` matches `trim_opener`).
  - **AI-only probe:** the pre-registration's dry run put OpenTuringBench at 0.863, "expected to be excluded". The build measured 0.8459 (one sample of 4,000 per side, `random_state=0`), and Amendment 10 reports it without explaining the drop across the line.
- **Confidence:** high

### R12-13: Generation artifacts the filters miss

- **Severity:** Low
- **Category:** data quality
- **Location:** `src/corpus/generate.py:179-187`: `artifact()` looks only at the first line's first 120 characters and the last 80 characters.
- **Repro:** `python scratchpad/r12/c4_artifacts.py` (the usable answers of every variant, after the filters and trims).
- **Evidence:**
  - **Refusals:** 29 in total, mostly Llama: "I can't provide veterinary advice…", "I cannot provide information on the preservation and cleaning of animal heads."
  - **Word-count mentions mid-text:** 37.
  - **CJK code-switching:** 8 ("an温带 fruit").
  - **Preambles:** "Great question!" and "Absolutely!" (Gemma).
  - **Plain-text echo:** "Here's the rundown in plain text:".
  - **Humanized records** have no `style` or `target_words`. That fits the fixed rewrite prompt, but it isn't the record shape claim 4 describes.
  - All counts are small, under 1% except emoji and preambles.
- **Confidence:** high

### R12-14: Today's record is accepted on its seal; its scoring-code digest no longer matches

- **Severity:** Low
- **Category:** gate integrity
- **Location:** `src/authorship/headtohead.py:35-47`.
- **Evidence:**
  - The recorded digest is `04e3d354` and the current one `97e527e8`. `detector.py`, `excerpts.py`, `registry.py` and `text.py` changed.
  - Reading the diffs, the old behaviour is kept for a record with no precision: bf16 autocast, an explicit fp32 load, format2 unchanged and an explicit excerpt view. The one exception is the new non-finite guard.
  - The explicit fp32 load equals the sealed path only if the v5 checkpoint was saved in fp32. The v5 training code saves fp32, but its config is inside the models folder, which wasn't read.
  - So "same scoring path as sealed" rests on review, not on a check.
- **Confidence:** medium

### R12-15: The shared ground is mostly LEDE, and MIRAGE's gated slice is not neutral

- **Severity:** Low
- **Category:** design observation
- **Location:** `headtohead.shared_ground`; pre-registration, "Improvement rule".
- **Evidence:**
  - **LEDE's share:** LEDE is 4,996 of the 8,878 pooled AI rows (56.3%). Fresh rows from the existing sources are 1,466 (16.5%). A gain of about 8.9 points on LEDE alone meets the 5-point rule.
  - **MIRAGE:** all 102 of MIRAGE's test AI rows are Moonshot.
    - v3 held Moonshot out (`6221d3b:src/corpus/split.py:20`), so v5 never fitted it.
    - 4d fits 1,376 MIRAGE Moonshot rows and 286 Arena Kimi rows.
  - The pre-registration defines these slices as ground "both trained on".
- **Confidence:** high

### R12-16: Unseen-generator rows of excluded sources aren't reported by source (Amendment 4 item 1)

- **Severity:** Low
- **Category:** functional
- **Location:** `src/authorship/headtohead.py:101-107` builds `by_source` from the masked frame, and `benchmark.py:162-164` reports `by_family` only.
- **Evidence:** OpenTuringBench's 216 unseen Microsoft rows appear only inside the Microsoft family figure, mixed with stackexchange_heldout's 480 Phi rows.
- **Confidence:** high

### R12-17: Nothing checks that the registry holds a v4 4d record before the head-to-head

- **Severity:** Low
- **Category:** reliability
- **Location:** `src/authorship/headtohead.py:147-153`.
- **Evidence:**
  - Before calibration, `models/registry.json` is the v5 record.
  - `run(rerun_reason=...)` would score v5 against itself and overwrite v5's sealed benchmark block.
  - This needs an operator error plus a rerun reason.
- **Confidence:** high

### R12-18: Training and calibration code: small gaps

- **Severity:** Low
- **Category:** technical soundness / auditability
- **Location and evidence:**
  - **Calibration defaults:** `calibrate.py:173,280-281` default to the release bounds (1% / 10%), not the pre-registered 0.8% / 8%. They are right only when `--max-false-ai 0.008 --max-false-human 0.08` is passed (Plan B does pass them).
  - **Training settings:** `train.py:151,273` default `--settings` to "fast", which Amendment 11 rejected. Plan B's short-model command (`plans/2026-09-28-increment-4d-training.md:1503`) omits the flag.
  - **Training record:** `train.py:246-248` commits a train-ids digest but no validation-ids digest (it is only in `run.json` under the models folder). `locked_split_reads: 0` is a constant, not a measurement (`train.py:256`, `calibrate.py:231`).
  - **Source weights:** `neural.py:353` normalises source weights per batch, which dilutes the balance on the real train split (192,339 rows):
    - MIRAGE human mass is 13% under target and MIRAGE AI 9% over.
    - Arena is 17% over and OpenTuringBench 22% over.
    - Sources with one label get half their mass, although the docstring says the total mass is kept.
  - **Scoring-code digest:** `registry.py:121-124` still omits `src/detection/__init__.py` (recommended in R11-2) and `src/pipeline.py`, the app's cell and row joining.
  - **v5 on this branch:** the committed `models/registry.json` (v5) can't pass `verify` ("no recorded inference precision"). If the outcome is "keep today's models", the branch must not be merged as is. Plan B notes this.
- **Confidence:** high

## Triage (2026-09-30)

Each finding's disposition is in pre-registration Amendment 13 (fixes, commits `e145a0f` to `cb465aa`, with corpus v4 rebuilt) and Amendment 14. The findings recorded but not changed are in the Debt Ledger: R12-4's word-count ruling is D18, R12-8 is D19, R12-18's source weights are D20. The part 2 review, `13-increment-4d-part2.md`, re-checked each finding.

