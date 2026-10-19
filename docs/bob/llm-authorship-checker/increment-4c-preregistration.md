# Increment 4c pre-registration — 2026-09-27

Written and committed **before** corpus v3, its generated data, or its locked test exist. After the test set is built, nothing below changes except to fix a bug, and any fix is recorded with its reason and reported next to the earlier result. The sponsor allows reruns, but only for bugs, never to tune.

## Why

On the v2 locked test, `distilroberta-authorship-v3` failed on human text written in an assistant's voice: OpenAssistant 19.7% false AI, and Dolly's upper bound at 1.22%. Those human answers had no AI answers to the same prompts, so the model could only learn "assistant voice = AI". Increment 4c pairs them: local open models answer the same prompts. The Ralph R07 fixes also change the system under test (see *System under test*).

## Generated data (local, Ollama, 4-bit)

| Role | Models (family) |
|---|---|
| Training generators | `llama3.2:3b` (meta), `qwen3.5:4b` (alibaba), `gemma3:4b` (google), `ministral-3:3b` (mistral) |
| Held-out generators, locked test only | `phi4-mini` (microsoft), `granite4:3b` (ibm), added to `HOLDOUT_FAMILIES` |

- **Prompt:** system *"You are a helpful assistant."*; user *"{prompt}\n\nAnswer in about {N} words."*, where N is the paired human answer's word count, clipped to 30–400.
- **Sampling:** temperature 0.8, top-p 0.95, seed from the item id, thinking off, at most `2N + 60` new tokens.
- **Assignment:** each prompt gets one answer from one generator, chosen by a hash of its id. Dolly and OpenAssistant use the four training generators; Stack Exchange uses all six.
- **Records:** every generation stores the model, its Ollama digest, the options, the seed, the target length and a timestamp.
- **Formatting:** markdown emphasis and heading markers (`**`, `__`, leading `#`) are stripped from both human and AI text in the paired sources, so formatting can't stand in for authorship.

## Corpus v3

| Source | Role | Human side | AI side |
|---|---|---|---|
| hc3, mage, mirage, raid, beemo, no_robots, aita2026, detector_bias, legacy_inhouse | as in v2 | as in v2 | as in v2 |
| **dolly** | now fitted (its v2 test rows are retired) | answers of 25+ words, 500 per category | local answers to the same prompts |
| **oasst1** | now fitted | first-turn English assistant replies of 25+ words (reviewed, not deleted, not synthetic). Replies with AI-voice giveaways ("as an AI language model", "I am an AI", naming OpenAI or ChatGPT) are removed as unverifiable. | local answers to the same prompts |
| **stackexchange** (new) | **locked test only, never fitted** | accepted or top-voted answers (score ≥ 2) to questions dated 2021 or earlier, so before ChatGPT, from 8 non-code sites (cooking, gardening, money, parenting, pets, outdoors, fitness, interpersonal). No code blocks, at most 3 links, 25–600 words. | local answers to the same questions, including the two held-out families |
| rasbt50k | diagnostic only | — | — |

**Split rules:**
- The locked tests of **v1 and v2 are both retired**. Their components may be fitted, never tested again, and retired components that stress variants would force into evaluation leave whole (R07-3).
- Buckets are re-drawn with salt `v3`.
- Stack Exchange is tested whole.
- **Test share is 35% for aita2026, no_robots and mirage**, so each gated slice has about 1,000 human texts. With 300–600, even one or two errors put the 95% upper bound over 1%. Other sources keep 15%, and validation stays at 15%.
- **Containment dedup (R07-6):** texts sharing enough sampled word 13-grams join one component, so a text mostly contained in another can't span fitting and test.
- **Shortcut probe (R07-7):** it runs on whitespace-collapsed text, shared and per source, fitted on train and scored on validation. A source flagged over 0.85 is investigated and fixed or excluded **before** training, and the decision is recorded here as an amendment.

## Models and routing

- **Main model** `distilroberta-authorship-v4` and **short-text specialist** `distilroberta-authorship-short-v2`: the 4b recipes unchanged.
- **Routing:** the 4b rule unchanged. The specialist serves a band under 100 words only if it beats the main model's validation coverage by at least 1 point.
- **Calibration targets:** 0.8% false AI and 8% missed AI.

## System under test

The R07 fixes are part of the tested system:
- Disguise normalization (R07-1).
- Whole-text windows where later windows can only veto (R07-2).
- A cell flagged Likely AI flags its row (R07-5).
- A file-verified benchmark (R07-4 and R07-8).
- Started-run records (R07-9).

## Pass criteria (unchanged since v1)

These apply on the v3 `test` split, through the full service:
- Every source slice with ≥300 human texts: false-AI 95% upper bound ≤ 1%.
- Every source slice with ≥100 AI texts: false-human 95% upper bound ≤ 10%.
- Decisive coverage ≥ 40% of supported text.

## Reported, not gating

- **Short-text drop-off:** excerpts of test texts, routed system against the main model alone.
- Results by length band.
- The unseen-generator split, now including Phi-4-mini and Granite 4 answers to Stack Exchange questions.
- The stress sets, in-house rater text (flag-only, unverified), and the rasbt50k diagnostic.

## Amendment 1 — before any generation or build (2026-09-27, 14:10)

On this laptop Ollama generates about 130 tokens/s with batched requests (measured on Llama 3.2 3B, 8 concurrent requests on a dedicated local server bound to 127.0.0.1:11435). Answering every prompt would take about 4 hours, so generation covers a fixed subset: the first **2,000 Dolly, 2,000 OpenAssistant and 1,500 Stack Exchange prompts** in a hash order of their ids (`GENERATION_LIMITS`). Every human answer stays in the corpus. No data or result informed this change.

## Amendment 2 — during generation, before any build (2026-09-27, 15:25)

`qwen3.5:4b` doesn't fit the 4 GB GPU. Only 25 of 33 layers loaded, and it ran at 5 answers/min batched and 13 tokens/s alone. **Qwen is replaced by `qwen3.5:2b`** (same family), and its 54 answers from the 4B model are set aside (`raw/dolly/superseded/`).

With the context window cut from 3072 to 2048 tokens (still room for a 450-word prompt and an 860-token answer), every remaining model fits the GPU batched, or misses by one or two layers:

| Model | Layers on GPU |
|---|---|
| phi4-mini | 33/33 |
| gemma3:4b | 34/35 |
| qwen3.5:2b | 24/25 |
| ministral-3:3b | 25/27 |

All of them run batched. The first 1,201 Llama answers used 3072, and each record keeps its own options. This hardware limit was measured on generation speed and GPU fit only; no data or result informed it.

## Amendment 3 — during generation, before any build (2026-09-27, 15:40)

The one or two layers that don't fit are the large-vocabulary output layers. Batching multiplies their cost on the CPU: qwen3.5:2b managed 12 tokens/s batched against 36 alone, and ministral-3:3b about 25 against 47. **qwen3.5:2b, ministral-3:3b and gemma3:4b therefore run one request at a time** on a second local server (127.0.0.1:11436); phi4-mini and granite4:3b fit fully and stay batched. At these speeds the remaining generation would take about 3.7 hours, so the subset is cut to **1,500 Dolly, 1,500 OpenAssistant and 1,200 Stack Exchange prompts** (the same hash order). Answers already written beyond the new cut are kept. No data or result informed this change.

## Amendment 4 — after generation, before any build (2026-09-28)

Generation finished on 2026-09-27 at 22:30 with 4,487 answers. Quality checks on the generated answers and the Ollama logs then found artifacts of our prompt and two provenance bugs. No corpus v3, split or result existed when this was written, and nothing but those checks informed it.

**A. Three filters on generated answers** (`generate.artifact`, applied in `load_generated`, so every paired source is covered, including the Stack Exchange test). The raw files keep every answer.

1. **Cut off at the token cap** (`done_reason = "length"`). These end mid-sentence, which is a shortcut. phi4-mini ignored the length target: its answers ran a median 1.4× the target, and 22.5% were cut off.
2. **The opening line restates the requested length**, for example *"Okay, here's an explanation of home automation, aiming for around 149 words:"*. This is about 10% of gemma3:4b's Dolly and OpenAssistant answers.
3. **The last 80 characters mention a word count**, for example *"(Word count: ~180)"*. This is about 2% of ministral-3:3b's answers.

Rules 2 and 3 use one pattern: a number followed by "word(s)", or "word count", or "word limit", case-insensitive. They apply to the AI side only, because the artifact comes from our instruction, which the human writers never saw.

| Source | Generated | Cut off | Opening note | Ending note | Dropped | Kept |
|---|---|---|---|---|---|---|
| dolly | 1,637 | 28 | 41 | 7 | 75 | 1,562 |
| oasst1 | 1,602 | 58 | 38 | 11 | 106 | 1,496 |
| stackexchange | 1,248 | 55 | 0 | 4 | 59 | 1,189 |
| **Total** | **4,487** | 141 | 79 | 22 | **240 (5.3%)** | **4,247** |

An answer can trip more than one rule, so the rule columns can sum to more than "Dropped". By model, Llama loses 11, Qwen 15, Granite 0, Gemma 90, Ministral 73 and Phi 51.

**Effect on the test:**
- The gated Stack Exchange slice keeps 825 AI answers from the four training families.
- The report-only unseen-generator split keeps 167 phi4-mini and 197 granite4:3b answers.

**B. Provenance fixes** (the text of every answer is unchanged):

4. **Digest.** `model_info` read `digest` from `/api/show`, which has no such field, so every record stored `None`. It now reads it from `/api/tags` (the SHA-256 of the model's manifest). `generate --backfill-provenance` filled it into all 4,487 records and tagged each one with `provenance_backfill`. It refuses any model modified after its first answer. Every model's manifest predates its first answer: the earliest gap is Llama's, at about 7 minutes. The originals are in `generated/backup/`.
5. **Licence line.** Records stored the first line of the licence text. phi4-mini's MIT text opens "Microsoft.", so it was classed as unknown. The line that names the licence is now used, and phi4-mini is recorded as `MIT License` (permissive).

**C. Corrections to the record** (nothing changes):

6. qwen3.5:2b is 8-bit (`Q8_0`). The *Generated data* heading says 4-bit, and the other five models are `Q4_K_M`. Each record stores its own quantization.
7. Batched with 8 parallel requests (a KV cache of 8 × 2048 tokens), phi4-mini loaded 30 of its 33 layers on the GPU, not 33/33 as Amendment 2 says. granite4:3b loaded 41/41. With one request at a time, qwen3.5:2b loaded 24/25, ministral-3:3b 26/27 and gemma3:4b 34/35. This affected speed only, and each answer's options are recorded.

## Amendment 5 — after the first v3 build failed its audit, before any lock or training (2026-09-28, 00:40)

The first v3 build (00:09–00:28) was refused by its own audit, so no test lock was written. Its train/val shortcut probe also flagged two sources. Everything below comes from that audit and that probe. No model has been trained on v3, and no test result exists.

**A. Split bugs found by the audit:**

1. **Retirement by group.**
   - The build retired v1 and v2 test components by `sample_id`, but the rule is that a tested group is never tested again, and the audit checks by `group_key`.
   - 289 OpenAssistant rows (185 human, 104 AI) landed in test from message trees v2 had tested. 284 were replies re-selected in v3, or new AI answers in those trees; 5 were v2 test replies whose text changed under markdown stripping, giving them new ids.
   - `assign_splits` now also retires by `retired_groups`.
2. **Settled splits.**
   - Three MIRAGE components linked distinct Enron emails through a shared footer, but only when the audit reclustered the kept rows. With the excluded rows present, the footer's word 13-grams occurred in more than 20 texts, so the build had treated them as boilerplate.
   - `settle_splits` now reclusters the kept rows as the audit does and merges any link that spans partitions, repeating until none does.

**B. Shortcut probe (the pre-registered rule: fix or exclude before training).**

The first build's per-source probes flagged **dolly at 0.859** and **mirage at 0.867**. Attribution on val (dropping one feature at a time, plus single-feature AUC):

| Source | What drove it |
|---|---|
| Dolly | More punctuation. Non-ASCII characters (curly quotes and em dashes, about 6× the human rate). Human answers lacking final punctuation (17% vs 5%). Longer words. By generator, ministral-3:3b separated at 0.946 and gemma3:4b at 0.921. |
| MIRAGE | Human texts with more digits and capitals. Human texts ending mid-sentence (23% vs 2%). Redaction placeholders. The email domain alone separated at 0.973: its human emails carry glued words where line breaks were lost, document tags and headers. |

3. **The model reads typography as ASCII.**
   - Curly quotes, dashes, ellipses and non-breaking spaces become their ASCII forms in the model's view: at training, calibration and inference, and in the probe.
   - This is recorded as `text_normalization = "typography+collapse_whitespace"`. A model keeps the view it was trained with, and an unknown view fails closed.
   - Typography comes from the typing device or editor, and smart quotes from Word or a phone must not push human text toward AI.
   - Simulated on the first build's train/val: dolly 0.859 → 0.846, mirage 0.867 → 0.857, and every source lower.
4. **MIRAGE's email domain is excluded** from every task. Simulated MIRAGE without email, in the typography view: 0.842.

**Considered and not adopted:** cutting a trailing sentence fragment from the model's view. It lowers MAGE (0.64 → 0.57) and RAID (0.74 → 0.64), which points to truncation artifacts there too, but alone it leaves MIRAGE at 0.852. At inference it would also cut words from short rater comments that lack a final full stop. It is recorded as follow-up work.

The rebuilt corpus's own probe is the result of record.

**C. Observations (no change):**
- Retiring the v2 test with R07-3's rule drops the stress variants of previously tested parents: RAID stress goes from 31,451 to 17, and MIRAGE stress from 61,038 to 42,745. Stress results are report-only.
- Every Dolly group was in v2's test, so Dolly is fitted only and has no v3 test rows.
- OpenAssistant test rows come only from message trees v2 never tested.
- Human assistant-style text is therefore tested through Stack Exchange, No Robots, and the fresh OpenAssistant trees.

## Amendment 6 — after the Ralph R08 review, before any test result (2026-09-28, 01:46)

The Ralph R08 review (`ralph/08-increment-4c.md`, a subagent, 01:05–01:34) returned FAIL. Training of `distilroberta-authorship-v4` had started at 01:04 on the typography view. Its first-epoch validation result (AUROC 0.965, 68% of AI caught at 1% false positives) was seen.

The run was stopped at 01:37 because of R08-2, not because of that result. Its model directory, including the resume state, was deleted so the retrain starts fresh. No model has scored the v3 test.

**Fixed before any test result:**

- **R08-2, formatting.** Markdown survived into the model's view.
  - In train and validation, asterisks appeared in 17.0% of Dolly's AI answers against 1.2% of its human ones. The figures were 26.6% against 6.3% for OpenAssistant, 10.8% against 0.9% for MIRAGE, and 11.5% against 1.6% for Beemo.
  - Stack Exchange's converter had already flattened its human lists, so the test couldn't catch this.
  - The model now reads text in the view `format+typography+collapse_whitespace`. It drops line-leading headings, quote marks and list markers (`-`, `*`, `+`, `•`, `1.`, `1)`), rule lines, emphasis asterisks and underscores, and code ticks. Underscores inside words, as in `snake_case`, stay.
  - The build's probe reads the same view.
  - Adding formatting features (asterisks, ticks, inline list markers, hashes) to the probe lifts dolly to 0.858 and mirage to 0.860 on the old view. On the new view they are 0.841 and 0.842, and every source is under 0.85.
- **R08-1, disguise.**
  - Deobfuscation also removes default-ignorable marks outside Cf: U+034F, variation selectors, Mongolian free variation selectors and Hangul fillers.
  - It maps Latin-script lookalikes (small capitals, dotless i and j, script g, Latin alpha, long s, Latin iota) to plain letters. Ordinary accented letters stay.
  - Live re-check with the v3 model, on CPU, on 40 validation texts it labels Likely AI: 15% lookalikes, U+034F in 30% of words and U+FE0F in 30% of words each leave 40/40 Likely AI, all with the obfuscation note. Ralph had found 28 of 31 turned Likely human by the lookalikes.
- **R08-4, gate binding.**
  - The benchmark seals its result with a SHA-256 of the registry entry it scored: everything except the benchmark's own bookkeeping.
  - `verify()` closes the gate if the entry changed afterwards, if the text view is missing, or if either model lacks its config and tokenizer hash.
- **R08-5, lock.** Each locked row's line now also carries its source and the SHA-256 of its exact text. The v3 lock is rewritten with `--relock`; the rows and splits don't change.
- **R08-10, retirement.**
  - Group retirement is read from the earlier manifests unless it is passed explicitly, however `build` is called.
  - The audit CLI stops when an earlier manifest is missing, rather than shrinking the check.

**Acknowledged, no change:**

- **R08-3.** Dolly has no v3 test rows, and OpenAssistant's fresh trees give 188 human texts, under the 300 a gated slice needs. Their v2 tests are retired into training, with their new AI pairs.
  - The v3 gate tests assistant-style human text through Stack Exchange (2,973 human texts) and No Robots (1,003).
  - OpenAssistant's false-AI rate is reported beside the gate.
  - The pass criteria don't change.
- **R08-6.** 18 fitted answers restate the length past the first line's 120 characters or near the end. None is in the test. Changing Amendment 4's rule now would move fitted rows and could move components, so it stays as written.
- **R08-7 and R08-8.** Token-dense texts leave words unscored, and 16 short MAGE test texts sit under the containment link rules. Both are recorded in the debt ledger.

**Corrections to the record (R08-9):**

- Amendment 2 was edited in place on 2026-09-27; its first text said Gemma would run one request at a time.
- 177 ministral-3:3b answers also used the 3072-token context. Each record stores its own options.
- The gated Stack Exchange slice has 824 AI answers, not 825: one pair left through a component shared with another source.
- The failed first build wrote its build record before Amendment 5 existed. The committed record comes from a later build.
- Amendment 5's diagnosis script printed short snippets of first-build rows, some from test (MIRAGE emails). The read was model-free and served only to trace the clustering.

## Amendment 7 — after the first v3 run passed and Ralph R09 found a bug in it (2026-09-28, 04:22)

**The first locked run on v3 passed** at 03:49 (commit `f9d143a`, reported in `increment-4c-report-2026-09-28.md`):
- False AI: 8 of 11,319.
- AI labelled human: at most 3.1% per slice.
- Coverage: 55.2%.

The Ralph R09 review of that system (`ralph/09-increment-4c-final.md`) confirmed that the pass arithmetic, the seal, the relock and the order of events all hold. It then found a bug in Amendment 6's formatting fix, so the run is repeated once, as this document allows for bugs. Thresholds, criteria and data don't change, and both runs are reported.

- **R09-1 (the bug).** Amendment 6 says the model reads text without list markers. The view removed them only at line starts.
  - `excerpt()` collapsed whitespace first, so every list inside an excerpt kept its markers. Excerpts are 81,929 of the short model's 111,607 training rows and 18,253 of its 24,531 calibration rows.
  - The short model learned them. On validation, removing the markers from RAID AI excerpts cut the mean score from 0.922 to 0.474, and rewriting AI texts as a one-line dash list turned 22 Likely AI into 0.
  - **Fix:** excerpts are cut from the model's view of the whole text, and the view also drops bullets anywhere and free-standing dashes. The new view is `format2+typography+collapse_whitespace`; each earlier name keeps its exact behaviour. Both models are retrained on it as `distilroberta-authorship-v5` and `distilroberta-authorship-short-v3`, so the first run's v4 and short-v2 files stay intact. They are then recalibrated, and the locked run is repeated with its reason recorded.
- **R09-3 (fix).** The seal bound the view's name, not its code. A behaviour fingerprint of the view (`text_view_fingerprint`) is now recorded by training and calibration, and `verify()` closes the gate if it no longer matches.
- **R09-4 (fix).** The seal now also covers the run's own result (passed, failures, time, report path and report SHA-256), so a failed run can't be edited into a pass.
- **R09-2 (fix).** Deobfuscation also handles:
  - ligatures and leader dots
  - thin spaces inside words
  - the rest of plane 14's default-ignorables
  - C0 and C1 controls
  - the Latin-named lookalikes ǀ and ɾ
  - the Armenian lookalikes ո, հ, ց and զ

  A word mixing Latin with another script now raises the disguise note even when it is left unmapped.
- **R09-6 (fix).** The build's probe gains formatting features (asterisks, ticks, spaced dashes, inline numbering, bullets), read on the model's view. Ralph found a spaced hyphen lifting HC3 to 0.858 on the old view. The rebuilt corpus's probe is the result of record.
- **R09-8 (fix).** The disguise note no longer counts U+180E twice.
- **R09-5 (acknowledged).** Detection is low outside AITA and MIRAGE. The first report says so, and so will the second.
- **R09-7 (corrected).**
  - The roadmap had marked 4c done, and the report said "Ralph R09 reviewed the pass", before that review returned. Both are corrected.
  - Amendment 6's heading is corrected to its commit time, 01:46.

## Amendment 8 — after the recorded rerun passed and Ralph R10 found a disguise bug (2026-09-28, 07:24)

**The recorded rerun passed** at 06:31 (v5 with short-v3):
- False AI: 18 of 11,319.
- AI labelled human: at most 3.7% per slice.
- Coverage: 66.9%.

Ralph R10 (`ralph/10-increment-4c-rerun.md`) confirmed four things:
- the rerun's legitimacy (lock and manifest unchanged, only Amendment 7's code changed, the reason recorded before scoring);
- the gate arithmetic;
- that R09-1 is fixed (list markers carry no signal in the training excerpts);
- the R09 disguise cases.

It found one high bug in the disguise defence, so the run is repeated once more. **The models, thresholds, criteria and data are unchanged; only the service's input handling changes.**

- **R10-1 (the bug).** New cheap disguises turned up to a third of confident AI into Likely human with no note: small roman numerals, unassigned and default-ignorable characters, runs of thin spaces, a math space, and i + dot above. This was the fourth round of per-character fixes (R07-1, R08-1, R09-2), so the defence now closes the class:
  - **Remove:** every format, unassigned, private-use, surrogate, default-ignorable and control character.
  - **Remove inside words:** runs of spacing characters between two letters.
  - **Remove after composition:** combining marks left on ASCII letters.
  - **Fold:** compatibility forms whose NFKC form is ASCII.
  - **Map:** Cyrillic, Greek, Armenian and Latin-script lookalikes to Latin.
  - **Abstain:** when a word still holds unusual Latin letters or another script alongside ASCII letters, the service abstains (`DISGUISE_SUSPECTED`) instead of scoring.
  - **Live re-check** (v5 on CPU, 80 validation texts it labels Likely AI): each of R10's disguises, plus U+2065, leaves 77–80 Likely AI with the note. A v-with-hook swap abstains on 80 of 80. None becomes Likely human.
  - **Ordinary text:** on train/val, 247 of 141,281 texts (0.17%) would abstain. They are words glued to CJK text, rare phonetic letters and similar. The note fires on 489 (R10 counted 1,339 before this fix).
- **R10-4 (fix).** Whitespace controls become spaces, and C1 bytes are read as cp1252 punctuation, so words no longer glue. An ellipsis, a narrow no-break space or thin digit groups raise no note.
- **R10-3 (fix).** The registry records a digest of the scoring code: disguise handling, the policy and service, windowing, the language check, the registry and the text views. `verify()` closes the gate when that code changes. Any later edit to these files, even to a comment, needs a new locked run.
- **R10-5 (fix).**
  - The seal now covers every locked run on record.
  - Report digests read CRLF as LF, so a Windows checkout matches the committed file.
  - `verify()` compares the digest with the report file when the file is present.
- **R10-2 and R10-7 (debt D14, D15).** Other list glyphs and `(1)` markers still reach the model. The view isn't idempotent on `**1.**`. Unspaced em dashes survive as hyphens, which lifts MIRAGE to 0.8506 with a hyphen-rate probe feature that isn't pre-registered. Fixing any of these changes the view and needs retraining, so they go to the next increment.
- **R10-6 (disclosed).** The human-side flags rose between runs: OpenAssistant from 0 to 6 of 188, Stack Exchange from 0 to 5, and in-house cells from 5 to 50 of 3,770. All are within the gate or report-only. The report puts them first among its caveats.
- **R10-8 (corrected).**
  - The format2 probe was run with the same function on the same train/val data at 04:26, before training started at 04:27, and flagged nothing. The build record of it was written later (05:51–06:14).
  - Amendment 7's heading is corrected to 04:22.
  - The status docs mark 4c done only after this run and its review.

## Amendment 9 — after run 3 passed and Ralph R11 found a disguise bug (2026-09-28; the commit time is the time of record)

**Run 3, the second recorded rerun, passed:**
- False AI: 18 of 11,319.
- AI labelled human: at most 3.7% per slice.
- Coverage: 66.7%.

Ralph R11 (`ralph/11-increment-4c-final.md`) confirmed the gate arithmetic, the seal and run 3's legitimacy (calibration identical, lock unchanged, only Amendment 8's code changed). It then found one high bug in the disguise defence, so the run is repeated once more. **Models, thresholds, criteria and data are unchanged; only the service's input handling and the seal's coverage change.**

- **R11-1 (the bug).** Amendment 8 judged accented Latin letters by a list of "ordinary" ones. Four holes followed:
  - A mark that composes with its letter (a + dot below becomes ạ) passed as ordinary.
  - A mark after a space was kept.
  - Letter-shaped symbols (⍺, ℮) and non-Latin digits inside words were never examined.
  - The effect: a dot below on 15% of vowels turned 55 of 149 confident AI texts into Likely human, silently.

  **Fix: accents are read back, not judged.**
  - Every Latin letter is decomposed and its combining marks dropped, and marks with no letter under them are removed.
  - Stroked letters (ø, ł, đ, ħ, ŀ, ŧ) and letter-shaped symbols map to plain letters.
  - Symbols or non-Latin digits touching ASCII letters count as unreadable.
  - **The service abstains only when unreadable content is dense:** three or more such words or symbols, or 2% of the words. One "Aβ" or "5 μg" no longer blocks a text (R11-3), and μ and β now map to u and b.
  - **Live re-check** (v5, 80 validation texts it labels Likely AI): dot below, macron below, a mark after a hair space, APL alpha and the estimated sign, e with a dot above, and R10's roman numerals and double hair spaces each leave 76–80 Likely AI. The Devanagari-zero swap abstains on 80 of 80. None becomes Likely human.
  - **Ordinary train/val text:** 62 of 141,281 texts (0.044%) abstain, where Amendment 8 had 247. 679 raise the note, and 2 lose more than two words.
- **R11-3 (fix).** The narrow no-break space between letters stays a space and no longer glues words. Only near-invisible spaces (six-per-em, punctuation, thin, hair, math) inside a word are removed.
- **R11-2 (fix).** The scoring-code digest now also covers `src/authorship/calibration.py` (the Platt map) and `src/detectors/base.py`. A new test changes a copied file and checks that the digest moves.
- **R11-4 (fix, partly).**
  - A registry entry whose benchmark names a report now needs that report present and matching.
  - `benchmark.history` stays informational; the sealed record is `locked_benchmark_history`.
  - The seal is a plain SHA-256 anyone can recompute, so git history remains the anchor against deliberate forgery. This is recorded as a limit.
- **R11-5 (corrected).** Amendment headings now give the date only; the commit time is the time of record.

## Known risks

- Small local models write differently from frontier models. That is why Stack Exchange carries held-out-family answers, and why the rasbt50k frontier diagnostic stays in the report.
- Stack Exchange answers are human by date and site policy. An answer edited after November 2022 can't be ruled out, but accepted answers to questions from 2021 or earlier are overwhelmingly earlier.
