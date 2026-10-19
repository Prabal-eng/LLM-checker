# Increment 4d pre-registration — 2026-09-28

Written and committed **before** corpus v4, its generated data and its locked test exist. After the test
set is built, nothing below changes except to fix a bug; any fix is a numbered amendment, recorded with
its reason, committed before the results it could affect, and reported next to the earlier result.
Reruns are allowed only for bugs, never to tune. Design: `docs/superpowers/specs/2026-09-28-increment-4d-detection-design.md`.

## Why

Run 4 of Increment 4c passed the v3 locked benchmark, but the strict bars catch only 65% of AI text. The
weak spots:

| Slice | AI detected |
|---|---|
| Texts of 25–49 words | 13% |
| Texts of 50–99 words | 8% |
| Stack Exchange | 15% |
| OpenAssistant | 15% |
| MAGE | 31% |
| RAID | 42% |
| HC3 | 47% |
| Unseen Cohere | 12% |

The sponsor set three equal priorities: short texts, overall detection and 2025–26 frontier models.
Increment 4d adds public data with recent generators and native short answers, fits Stack Exchange,
screens larger encoders, and builds a fresh test that no earlier model has seen.

## What ran before this document

Model screening (`python -m src.bakeoff.screen`) started on 2026-09-28 on the **v3 train and validation
splits only**. It reads no locked split and no v4 data. Its results choose the model family (see
*Models*); the choice is recorded as an amendment before any v4 training.

The build's AI-only probe (see *Corpus v4*) was dry-run on the raw training sources on 2026-09-28:
Arena 0.832 and OpenTuringBench 0.863, on the model's view and window. By the rule below, OpenTuringBench
is expected to be excluded. The window choice was made after that dry run; it changes neither outcome
(full text: 0.844 and 0.879).

## Generated data (local, Ollama, 4-bit)

| Role | Models (family) |
|---|---|
| Training generators | `llama3.2:3b` (meta), `qwen3.5:2b` (alibaba), `gemma3:4b` (google), `ministral-3:3b` (mistral) |
| Held-out generators | `phi4-mini` (microsoft), `granite4:3b` (ibm), `falcon3:3b` (tii; TII Falcon License, pulled 2026-09-28) |

v3's records stay as they are. New answers go into their own variant folders, and every record stores
the model, its digest, the options, the seed, the style and the target length.

- **Styles.** Each item's style is drawn from a hash of the variant and item id:
  - one of four personas (system prompts):
    - *neutral*: "You are a helpful assistant."
    - *casual*: "You are chatting casually with a friend. Write the way people type, not like a textbook."
    - *forum*: "You are a regular user answering a question on a Q&A forum."
    - *terse*: "You answer briefly and directly."
  - a *plain* flag on 30% of items, which appends "Write plain text: no headings, lists, bold or
    markdown.";
  - a temperature uniform in 0.6–1.1 (top-p 0.95, thinking off).
- **Short variant (`short4d`):** the target length is drawn uniformly from 25–99 words. The standard
  variant (`train4d`, and `test4d` for the held-out sites; Amendment 1) targets the paired human answer's length, clipped
  to 30–400.
- **Humanized rewrite:** 10% of the `train4d` answers, chosen by hash, are rewritten by the next training
  generator in pool order with: "Rewrite the text below so it sounds casual and natural, like a person
  typed it. Keep the meaning and roughly the same length. Reply with the rewrite only." Label AI,
  transform `llm_rewrite_casual`.
- **Amendment 4 filters still apply:** answers cut off at the token cap, or that mention a word count,
  are dropped.

| Source (variant) | Items answered | Generators |
|---|---|---|
| stackexchange, v3's eight sites (`train4d`) | 2,500 | training |
| stackexchange (`short4d`) | 2,500 | training |
| dolly (`short4d`) | 1,000 | training |
| oasst1 (`short4d`) | 1,000 | training |
| stackexchange_heldout (`""`) | 2,100 | training and held-out (seven) |
| stackexchange_heldout (`short4d`) | 2,100 | training and held-out (seven) |
| humanized rewrite of stackexchange `train4d` | about 250 | the next training generator |

The held-out Stack Exchange sites are answered by all seven generators, as v3 was. Its test slice then
measures detection on a register no model fitted, and the unseen-generator slice measures families no
model fitted.

## Corpus v4

| Source | Role in v4 | Notes |
|---|---|---|
| hc3, mage, mirage, raid, beemo, no_robots, aita2026, detector_bias, legacy_inhouse, rasbt50k | as in v3 | only rows no earlier version fitted or tested can be evaluated (below) |
| dolly, oasst1 | fitted, as in v3 | plus short answers |
| **stackexchange** (v3's eight sites) | **now fitted** | v3's test is spent. Phi and Granite answers stay held-out-family rows |
| **stackexchange_heldout** (new) | **locked test only** | travel, diy, workplace, academia, bicycles, mechanics, photo, writers; same selection as v3 (2021 or earlier, score ≥ 2, no code, ≤ 3 links, 25–600 words), 375 per site |
| **maga** (new, MIT) | fitted | MGB (plain) and MAGA (aligned, transform `maga_aligned`) answers with their shared human texts. wikiHow dropped. Length-matched per domain, 1,500 per domain and label. Test share 20% |
| **dactyl** (new, MIT / CC-BY-4.0) | fitted | tweets, reviews, abstracts, student essays, news. WritingPrompts dropped. Length-matched per domain, 2,000 per domain and label. Test share 20% |
| **arena140k** (new; outputs under provider terms) | fitted, AI only | first English, non-code answer of each side, 400 per model. Held-out families dropped |
| **openturingbench** (new, CC-BY-NC-4.0) | fitted, AI only, if the probe allows | in-domain validation and test files, 1,500 per model |
| **wasitai2026** (new, mixed per row) | **locked test only** | English rows whose text is in the file (389 human, 746 AI). 421 humanized rows form a stress slice |
| **lede** (new, CC-BY-NC-4.0) | **locked test only**, AI only | frontier models (Claude Opus 4.6, Sonnet 4.5–4.6, GPT-5 / 5.4 family, Gemini 3.1), strategies sc, ib and ng, 500 per model |

**Split rules:**
- **Retirement widens.** Every row that v1, v2 or v3 put in train, validation, test or
  test_unseen_generator is retired by sample id and group key. It may be fitted, but never evaluated.
  Stress and in-house rows are report-only and stay available.
- **Near-duplicates are retired too.** Rows that could still be evaluated are clustered (containment
  dedup) with the earlier versions' fitted and tested text, and any that share a cluster are retired. The
  audit re-checks every locked-test row against that text independently.
- **Held-out families:** bigscience, microsoft, ibm, tii. Cohere and Moonshot are now fitted.
- **Held-out sources**, tested whole: stackexchange_heldout, wasitai2026, lede.
- **Test shares:** 35% for aita2026, no_robots and mirage; 20% for maga and dactyl; 15% for the rest.
  Validation is 15%. Buckets are re-drawn with salt `v4`.
- **Cap:** 3,000 per stratum, as in v3.
- **Model view `format3`** (debts D14, D15):
  - every bullet glyph and `(1)` / `1)` marker is dropped;
  - every en or em dash, and a free-standing `--`, becomes a space;
  - the cleanup repeats until nothing changes, so it is idempotent.
- **Shortcut probe:** the shared and per-source probe runs on the model view. A source over **0.85** is
  investigated and fixed or excluded before training, and the decision is recorded as an amendment.
- **AI-only probe:** each AI-only source is probed with 5-fold cross-validation against the human rows
  it is balanced against, on the model's view and window (first 200 words):
  - Arena against dolly, oasst1 and no_robots;
  - OpenTuringBench against raid, dactyl and maga news.

  A source over **0.85** is excluded before splitting.

## Models

**Screening** runs on v3 train and validation.
- **Run:** each candidate trains one epoch on 24,000 label-balanced texts with source-balanced weights and
  length-grouped batches.
- **Tracks:** main (256 tokens) and short (25–99 words and excerpts, 160 tokens).

| Candidate | Training |
|---|---|
| DistilRoBERTa | full, bf16 |
| RoBERTa-base | full, bf16, gradient checkpointing |
| ModernBERT-base | full, bf16, 8-bit AdamW, checkpointing |
| DeBERTa-v3-base | full, fp16 with loss scaling, 8-bit AdamW, checkpointing |
| ModernBERT-large | LoRA r=16, bf16, checkpointing |
| DeBERTa-v3-large | LoRA r=16, fp16, checkpointing |

**Selection rule:**
- The winner has the highest mean of main-track and short-track detection at 1% false AI on validation.
- Within 1 point, the faster model wins.
- The top two train fully on v4 if they are within 2 points of each other.
- A candidate that breaks a GPU limit at inference is dropped.

**Full training on v4:**
- Validation every quarter epoch; stop after two checks without improvement, at most three epochs, keep
  the best check.
- Length-grouped batches.
- LoRA adapters are merged before saving.
- The inference precision is recorded in the registry and read by the detector.

**Routing, unchanged from 4b:** the main model serves 100+ words. The short specialist serves a band
under 100 words only if it beats the main model's validation coverage there by at least 1 point.

**Calibration targets:** 0.8% false AI and 8% missed AI, per word band, as in 4c.

**Optional steps.** Each is adopted only on validation, and each adoption is recorded as an amendment
before the lock is read:
- **Perplexity stack:** a logistic stack of the classifier's logit with the Increment 3 zero-shot scores
  (Qwen2.5-0.5B, Binoculars and Fast-DetectGPT). Fitted on half of validation and measured on the other
  half. Adopted only if it adds at least 2 points of detection at 1% false AI, and the GPU limits hold.
- **Style-aware thresholds:**
  - A TF-IDF logistic register classifier (Q&A and forum, assistant, news and expository, essays and
    academic, short social and reviews), trained on source-to-register labels.
  - A bar per length band × register cell, set on that cell's validation humans at the same targets.
  - A cell with fewer than 600 validation humans uses the pooled band bar.
  - Adopted only if it adds at least 2 points of validation detection, with every cell within target.

## System under test

The full service, as in 4c, with:
- the `format3` view for the 4d models;
- the inference precision read from the registry;
- the register classifier, if adopted.

Today's models (`distilroberta-authorship-v5` and `-short-v3`) are scored through their own registry
records, including their `format2` view.

**Speed and memory limits:**
- **GPU, unchanged:** peak inference memory ≤ 3.5 GB; single text ≤ 500 ms at p95; a 500 × 6 sheet
  ≤ 120 s.
- **CPU (build brief Revision 3):** single text ≤ 10 s; a sheet ≤ 30 minutes.

## Pass criteria (unchanged since v1)

These apply on the v4 `test` split, through the full service:
- Every source slice with ≥300 human texts: false-AI 95% upper bound ≤ 1%.
- Every source slice with ≥100 AI texts: false-human 95% upper bound ≤ 10%.
- Decisive coverage ≥ 40% of supported text.

## Improvement rule (head to head)

Today's models and the 4d models are scored on the same v4 locked test. **Shared-ground slices** are
where neither side has the in-distribution advantage:
- fresh rows of hc3, mage, mirage, raid, beemo, no_robots, aita2026, dolly and oasst1, which both
  trained on;
- wasitai2026, lede and the unseen-generator split, which neither trained on;
- stackexchange_heldout, whose sites, questions and held-out generators are new to both.

Test portions of maga and dactyl are reported but excluded from the rule, because they share a
distribution with the 4d training data.

| Outcome | What happens |
|---|---|
| 4d passes the gate, detects ≥ 5 points more AI on the shared-ground slices together, and no gated shared-ground slice loses more than 5 points | 4d is registered and replaces today's models |
| 4d fails, today's models pass | today's models stay registered |
| both fail | the gate closes until fixed, as in 4b |

## Reported, not gating

**Goals:**
- 75% AI detected overall;
- 25% on 25–99-word texts;
- 70% on the frontier sets (wasitai2026 and lede together).

**Also reported:**
- results by source, word band and generator family, for both model sets;
- short-text excerpts, as in 4c;
- stress sets, including wasitai2026's humanized rows;
- in-house rater text (flag-only, unverified);
- the rasbt50k diagnostic.

## Review before the lock is read

Before the locked run, an independent Ralph review covers:
- the v4 corpus and audit;
- the `format3` view and disguise handling;
- the validation results of the chosen models.

Its findings are fixed as numbered amendments, and only then is the locked test run, once.

## Amendment 1 — before any generation or build (2026-09-28)

The held-out Stack Exchange sites' standard variant is named `test4d`, not `""`. In the generator, `""`
means v3's settings (neutral persona, temperature 0.8, no plain-text instruction), and v3's records must
stay reproducible. The held-out sites should get the same style variety as the training answers. The
table's `stackexchange_heldout (`""`)` row therefore reads `stackexchange_heldout (`test4d`)`; its count
(2,100) and generators (all seven) are unchanged.

## Amendment 2 — before any generation or build (2026-09-28), from the Plan A final review

1. **Fresh rows from existing sources.**
   - v1–v3 all picked the same rows (the first 3,000 per stratum by a fixed hash). Re-running the loaders
     would pick only rows an earlier version used, so they would give no fresh test rows.
   - hc3, mage, raid, mirage, no_robots and aita2026 now pass over the rows every earlier version took and
     take the next ones in hash order. They fall back to used rows only when a stratum has too few, and a
     used row that comes back is still retired.
   - dolly, oasst1 and stackexchange keep v3's items, because their AI answers were generated for those
     items; they give no fresh test rows. Stress selections are unchanged.
   - Where a pool is used up (hc3 and beemo took every row in v3), the source has no fresh test rows.
   - A source whose fresh test rows fall under 300 humans or 100 AI is reported, not gated, like any small
     slice, and the report names it. This includes RAID and HC3, today's weakest slices.
   - The build record lists test rows per source before the lock is read.
2. **The leakage rule covers every earlier split,** as the spec says.
   - A v4 test row may not match any v1–v3 text, by id or as a near-duplicate, including stress and
     diagnostic rows.
   - A core copy of a text that an earlier version only reported is dropped before duplicate resolution,
     so the reported copy keeps its stress slice. For example, WasItAI's TOEFL essays were reported in
     v1–v3 as detector_bias stress rows.
   - Stress texts are matched only against fitted and tested text, because they recur in every version by
     design.
3. **Held-out sources are retired too.** A stackexchange_heldout, wasitai2026 or lede row that matches
   earlier text is excluded, because it can't be moved to training in place of a test row.
4. **Improvement rule on Stack Exchange.**
   - The 4d models train on Stack Exchange answers written by the four training generators with the 4d
     personas, so those generators' answers on the held-out sites are not new ground.
   - stackexchange_heldout's training-generator answers are gated and reported, but leave the shared
     ground.
   - Its held-out-generator answers (Phi, Granite, Falcon 3) still count, through the unseen-generator
     split.
5. **format3** repeats the whole view until nothing changes, as this document already requires. Nested
   markers needed more passes than the first implementation allowed.

## Amendment 3 — head-to-head definitions (2026-09-28, before any v4 build or training)

1. **One run.** The v4 locked run is one head-to-head run (`python -m src.authorship.headtohead`).
   - Both model sets score the same locked rows through the full service, each through its own registry
     record.
   - Today's record is `models/incumbent_registry.json`, frozen at commit `6221d3b`. It is refused unless
     its v3 seal still verifies.
2. **Pooled detection.** The pooled shared-ground figure counts every AI row of the shared-ground slices:
   test rows of the listed existing and frontier sources, and every unseen-generator row.
3. **Gated shared-ground slice.** A gated shared-ground slice is a test-split source slice with at least
   100 AI rows. The unseen-generator split is report-only in the pass criteria, so it counts in the
   pooled figure but can't block the rule by losing points.
4. **The open case in the outcome table.** 4d passes the gate but not the improvement rule, and today's
   models fail the gate. Then 4d is registered, because it is the only model set that meets the contract.
   The improvement rule governs replacing a model set that passes.
5. **Speed and memory limits** are measured on validation text with `python -m src.authorship.speed`
   (on the GPU, then with the GPU hidden for the CPU), before the locked run. A model set that breaks a
   limit can't be registered.

## Amendment 4 — from the Plan C review (2026-09-28, before any v4 build or training)

1. **Unseen-generator rows on shared ground.** Amendment 3 item 2 said every unseen-generator row counts.
   That conflicts with this document's rule that test portions of the new training sources are excluded.
   - Unseen-generator rows count only when their source is on the shared ground: the existing sources
     listed above, wasitai2026, lede and stackexchange_heldout.
   - Unseen-generator rows of maga, dactyl, arena140k, openturingbench and stackexchange come from sources
     the 4d models trained on. They are reported by source, not counted.
2. **The sheet limit is timed the way the app scores a sheet.** The app scores each cell on its own, then
   each row's cells joined, all as rater comments (`src/pipeline.py`).
   - `python -m src.authorship.speed` times that path for a 500 × 6 sheet of validation text. On the CPU
     it times 20 rows and scales to 500.
   - The batched figure the v1 bake-off used is recorded beside it for comparison. The limit applies to
     the app's path.
3. **Each model set is loaded only for its own pass** in the head-to-head, so the two never share the
   GPU. Peak inference memory is recorded per model set.

## Amendment 5 — GPU-first limits (2026-09-28, before any v4 build or training)

**Why.** Screening's main track put DeBERTa-v3-large (LoRA) well ahead: 64.8% detection at 1% false AI,
against 55.5% for ModernBERT-base and 47.1% for today's architecture. The short track was still running.
- A CPU probe then timed it at about 11 s per 256-token text and 2.9 s per short one. The probe ran beside
  screening, so these times are somewhat high.
- At that speed a 500-row sheet takes hours on the CPU.
- The sponsor chose the most accurate model and made the CPU a slow fallback (build brief Revision 4).

**Changes:**
1. **CPU limits.** A single text must take ≤ 20 s at p95 on the CPU. The CPU sheet time is reported, not
   gated. The GPU limits are unchanged.
2. **The app scores a sheet in batched calls** (`QAPipeline.prefetch_sheet`): every cell, then each row's
   joined cells, as rater comments. Decisions are identical to scoring one cell at a time. The speed check
   times this path from a fresh service, so model loading counts (Amendment 4 item 2).
3. **GPU weights load in the recorded precision.** A registry record that states its inference precision
   loads its weights in that precision on the GPU, so a large main and short model fit in 4 GB. The CPU
   runs fp32. Today's records carry no precision and keep the fp32 weights they were calibrated with.
4. **Non-finite scores stop scoring with an error.** A decision can't rest on a NaN or infinite score.

The screening winner is recorded in a separate amendment when the short track finishes.

## Amendment 6 — screening result (2026-09-28, 14:44, before any v4 build or training)

Screening ran on v3 train and validation, as pre-registered. The results are in `data/benchmark/v4/screening/`.
Figures are detection at 1% false AI on validation.

| Candidate | Main track | Short track | Mean |
|---|---|---|---|
| **DeBERTa-v3-large (LoRA)** | **64.8%** | **48.4%** | **56.6%** |
| ModernBERT-base | 55.5% | 36.3% | 45.9% |
| DeBERTa-v3-base | 53.9% | 37.3% | 45.6% |
| RoBERTa-base | 46.5% | 37.5% | 42.0% |
| DistilRoBERTa | 47.1% | 34.6% | 40.8% |
| ModernBERT-large (LoRA) | 12.4% | 49.5% | 31.0% |

- **Winner: DeBERTa-v3-large with LoRA** (r=16, alpha 32, fp16 with loss scaling, gradient checkpointing,
  learning rate 2e-4). It trains both the main and the short model.
- **Only the winner trains.** The runner-up is 10.7 points behind, outside the 2-point rule.
- **ModernBERT-large's main-track run didn't learn** under its pre-registered recipe, though its short track
  reached 49.5%. It is recorded as screened, and no recipe was changed.
- **Inference precision:** the registry records fp16, and the GPU weights load in fp16 (Amendment 5).
  Screening's peak training memory for this candidate was 2.47 GB. The speed check measures inference
  memory with both models loaded before the lock.
- **Learning-rate schedule.** Full training runs the linear schedule over at most three epochs, with
  quarter-epoch checks and early stopping. Screening decayed within one epoch, so the best check comes at a
  different point in the schedule than screening's end.
- **Screening ran from the main checkout**, with its format2 view on v3 data. The 4d models train and
  calibrate on format3.

## Amendment 7 — faster full-training settings (2026-09-28, before any v4 build or training)

The sponsor approved training settings that save time without removing anything.
- **Half-precision frozen base.** Under LoRA, the frozen DeBERTa-v3-large base sits in half precision on
  the GPU (fp16, matching autocast). The weights that train stay in fp32. The arithmetic is unchanged, and
  about 0.9 GB of GPU memory is freed.
- **No gradient checkpointing.** That memory lets full training drop gradient checkpointing, which
  recomputes every layer's forward pass during the backward pass.
- **Same effective batch.** Micro-batches of 4 texts are accumulated 4 times, the same effective batch of
  16 as screening (8 × 2). The data, model, learning rate, schedule and number of optimizer steps are
  unchanged.
- **Expected saving:** training about 25–35% faster.
- **Fallback.** The GPU resume smoke test checks these settings first. If they run out of memory or peak
  above 3.6 GB, training uses the screening settings (`--settings screening`) instead.
- **Recorded:** each training record states which settings it used.

## Amendment 8 — greeting openers in generated answers (2026-09-28, during generation, before any build)

A check on the first generated answers found that the casual persona often opens with a greeting such as
"Hey," "Hi there!" or "Hey man,". Real answers almost never do:

| Source | Human answers opening with a greeting | Generated answers (Qwen, Ministral so far) |
|---|---|---|
| Stack Exchange | 0.03% | 3.7–7.7% (up to 28% of casual-persona answers) |
| Dolly | 0.65% | 2.6–5.2% |
| OpenAssistant | 1.7% | 3.2% |

- **Why it matters.** The model would learn "a greeting means AI". That would flag humans who greet, and
  inflate the Stack Exchange test, whose AI answers use the same personas.
- **Change.** When generated answers are loaded, a leading greeting is trimmed (`generate.trim_greeting`).
  This covers a greeting word, optionally followed by a name or a word like "there" or "man", then
  punctuation. The next word is capitalised.
  - This applies to training and test answers alike.
  - The raw files keep the original text.
- **After the trim,** greeting openers fall to 0–0.16% of generated answers, in line with the human rates.
- **Filler openers stay** ("Okay, so", "Honestly,"). Humans use them too (OpenAssistant's human answers
  7.5%), so trimming them from the AI side alone would turn them into a "human" signal.

No corpus, split or result existed when this was written.

## Amendment 9 — "Okay," openers in generated answers (2026-09-29, after generation, before any build)

The quality check after generation found that Gemma opens a third of its answers with "Okay," ("Okay, so
like, …", "Okay, wow, …"), which Amendment 8 left in place. Measured on the v3 corpus's human rows and on
every generated answer:

| Opener | Human texts (v3, 99,775 rows) | Gemma | Other generators |
|---|---|---|---|
| "Okay" / "OK" | 0.21% (Stack Exchange 0.10%, AITA 2.7%) | 38.5% | 0–0.8% |
| "Sure" / "Certainly" / "Of course" | 0.58% (OpenAssistant 7.8%) | 1.5% | 0–6.7% |

- **Why it matters.** Amendment 8's reason for keeping fillers was that humans use them too. That holds
  for "Sure," but not for "Okay,": a 38.5% against 0.21% gap would teach "starts with Okay means AI", and
  rater comments (the product's main input) often start with "Ok so".
- **Change.** `generate.trim_opener` replaces `trim_greeting` at load. It trims a leading "Okay"/"OK"
  followed by punctuation (optionally with a name, "Okay Marta,"), and greetings as before, in any order
  ("Hey! Okay, so …" → "So …"). The greeting's address words gain "bro", "sis", "bro/sis" and "what up",
  which the check found in held-out Granite answers.
  - Training and test answers alike; raw files keep the original text.
  - "Okay so" without punctuation stays (part of the sentence), as do "Sure," and "Honestly,".
- **After the trim,** "Okay" openers fall to 0–0.31% of every generator's answers and greeting openers to
  0–0.25%, in line with the human rates. Gemma's answers change 39.1%, the others' 0.5–13.6%.

No corpus, split or result existed when this was written: the v4 build has not run.

## Amendment 10 — Stack Exchange over the shortcut-probe line (2026-09-29, after the v4 build, before any training)

The v4 build passed its audit, and so did the separate `audit --version v4` run: 368,278 rows, 30 gated test
slices (8 of them held-out Stack Exchange sites, about 374 humans each), 28 report-only slices, 26,001
near-duplicates of earlier text retired (25,724 from MAGE). The AI-only probe stayed under the line (Arena
0.842, OpenTuringBench 0.846). One source crossed it: Stack Exchange's own shortcut probe scored **0.891**
on validation (shared probe 0.761). The rule says such a source is "investigated and fixed or excluded
before training".

- **Investigation.** One feature carries it: mean word length. AI answers average 5.27 characters a word
  and humans 4.58; that feature alone scores 0.84, and the character and word counts rebuild it.
  - Every layout and markdown feature is zero on the model's view, so no formatting shortcut is involved.
  - The human text is clean prose; the low-word-length answers read as ordinary sentences.
  - Humans use more short function words: 20.2% of their tokens have two characters or fewer, against
    15.6% for AI, and "I" is about 7 times more frequent.
  - The same gap is in sources under the line: Dolly 0.51, MIRAGE 0.53, OpenAssistant 0.44, MAGA 0.41
    (Stack Exchange 0.57).
  - The humanized rewrites average 4.5–4.7 characters a word, close to human.
- **Ruling: kept, not fixed or excluded.** No artifact of collection, conversion or formatting was found
  to fix; the gap is lexical style, which is authorship signal.
  - Excluding Stack Exchange would drop the source 4d exists to improve. It would not remove the cue
    either, since Dolly and MIRAGE carry it.
  - The safety net is the gate: if the model learns "long words means AI", it shows as false AI on the 8
    gated held-out Stack Exchange sites (other sites, other generators), each held to the ≤1% bound.
- **Cost if wrong:** humans who write in long words (academic or formal answers) are flagged more
  often. The gate catches this, but a failed Stack Exchange slice would then fail the run.

No model had been trained on v4, and no test text had been scored, when this was written.

## Amendment 11 — full training uses the screening settings (2026-09-29, before full training's first check)

Plan B's resume smoke test ran on 2,000 v4 training rows, one uninterrupted run and one killed after its
first check and resumed.

| Settings | Peak training memory | Uninterrupted run | Resumed run |
|---|---|---|---|
| fast (Amendment 7: micro-batches of 4 × 4, no gradient checkpointing) | 4.29 GB, the whole card | 618 s | stopped, not needed |
| screening (8 × 2, gradient checkpointing) | 1.66 GB | 332 s | resumed from check 0.1; peak 1.65 GB; record written; pass |

- **Why.** Without gradient checkpointing, DeBERTa-v3-large's activations fill the 4 GB card and spill into
  shared memory, which makes the "fast" settings about twice as slow, not faster. Amendment 7 named the
  screening settings as the fallback if the peak exceeded 3.6 GB.
- **Change.** Full training runs with `--settings screening`: exactly screening's recipe (effective batch
  16, gradient checkpointing), with the frozen base kept in half precision (Amendment 7), which lowers its
  peak from screening's 2.47 GB to 1.66 GB. The training record names the settings.

## Amendment 12 — one micro-batch of 16 (2026-09-29, before full training's first check)

Full training started with the screening settings at 19:25 and was stopped 7 minutes in, before its first
check, to measure whether the unused memory could buy speed. Each configuration trained one epoch on the
same 2,400 v4 training rows, in its own process:

| Micro-batch × steps | Gradient checkpointing | Rows per second | Peak training memory |
|---|---|---|---|
| 8 × 2 (screening) | yes | 7.84 | 1.58 GB |
| 2 × 8 | no | 5.36 | 2.69 GB |
| 16 × 1 | yes | **8.71** | 2.12 GB |

- **Change.** Full training uses `--settings fullbatch`: screening's recipe with the effective batch of 16
  taken as one micro-batch instead of two. Learning rate, LoRA, precision and checkpointing are unchanged.
  Scoring peaks at 1.59 GB in every configuration.
- **Why.** 11% faster at a peak far under the 3.6 GB line. Small micro-batches without checkpointing lose
  more to GPU underuse than they save in recomputation.
- **Side effect.** Each step's loss is now the source-weighted mean over all 16 rows, instead of the average
  of two 8-row weighted means. That is the intended quantity.

## Amendment 13 — Ralph R12 part 1 fixes; corpus v4 rebuilt (2026-09-30, before any locked text is read)

Ralph review R12 ran in two parts. Part 1 (code, corpus v4, generation, the head-to-head;
`ralph/12-increment-4d.md`) returned **FAIL**: 2 High, 5 Medium and 11 Low findings. The High finding
R12-1 means corpus v4 doesn't meet this document's own design (MAGA and DACTYL "length-matched per
domain"), so the fixes below are bug fixes. They are committed before the rebuilt corpus exists, before
any training check is used, and before any locked text is read. The 2026-09-29 build and its lock were
never read, and the rebuild replaces them.

**The stopped training run.** Main training on the 2026-09-29 corpus (Amendment 12) was stopped after
check 0.4, the end of its first epoch. Its checks were on validation only. None is used, its folder is kept
aside for audit, and the 4d models train from scratch on the rebuilt corpus.

Fixed (code test-first; commits `e145a0f` to `be1bc3a`):

1. **R12-1: length matching.** `take_per_stratum` capped each label with the same hash order the length
   matcher used, which kept each label's majority bands and put the gap back (MAGA word-count AUC 0.645).
   The matcher now shares the cap over word-count bands, so both labels keep equal counts per band. The
   bands are about 25% wide (10, 13, 16, 20, 25, 32 … 2,000 words), because bands a factor of two wide left
   one MAGA domain separable at 0.83 within bands. On the raw data:

   | Source | Within-domain word-count AUC (weighted) | Worst domain | Rows per label |
   |---|---|---|---|
   | MAGA | 0.509 | 0.556 | 13,031 (was 13,500) |
   | DACTYL | 0.489 | 0.507 | 10,000 |

2. **R12-3: self-announcing preambles and news templates.** These are trimmed at load, the class
   Amendments 8 and 9 trimmed.
   - "Here is the body of the Reddit post:" and "Certainly! Here's an improved version of your abstract:"
     opened 13% of MAGA's AI rows and 0.04% of its humans. The rate is 5% for MIRAGE.
   - The "Date: …" and "Category: WEDDINGS" lines are on 27% of OpenTuringBench rows, all of them AI.
   - The trims apply to core rows of MAGA, MIRAGE and OpenTuringBench, whatever their label, and to the
     generated answers.
   - A preamble must name its output ("version", "article", "body", …) before its colon, so openers like
     "Here's the thing: …" stay.
   - MIRAGE's stress rows keep their text.
   - The first rebuild (03:53) still left "Category: Parenting"-style lines on 3.4% of OpenTuringBench rows,
     because the trim wanted upper-case values. Title-case categories go too, which leaves 25 of 10,500
     rows with an odd date format. The corpus was rebuilt again before anything read it.
3. **R12-4: character and format tricks.**
   - **A new model view, `format4`.** It treats every Unicode dash except a word hyphen as a space. It
     drops symbol bullets that open a line (◆ → ✅), circled numbers, and letter markers ("a)", "(b)").
     Emoji inside a sentence stay.
   - **Training and calibration.** The 4d models train and calibrate on `format4`. `format3` keeps its
     behaviour by name.
   - **The service** reads other scripts' word separators back as spaces: katakana middle dots, the
     word-separator dot, and runic, Ethiopic, Syriac and Aegean dividers. Other non-ASCII punctuation glued
     between Latin letters counts toward abstaining.
   - **Ruling: word counts are unchanged.** The policy still counts words on the read-back text, not on
     the view. That count sets today's sealed bands too, and calibration counts the same way.
     Ralph part 2 scores marker tricks on the 4d model.
4. **R12-13: generation artifacts.** 38 of 15,921 generated records are now dropped:
   - 23 refusals;
   - 7 bracketed word counts inside the answer;
   - 8 switches into Chinese, Japanese or Korean script.
5. **R12-2: detector errors.** Any detector error during the locked run stops it before the gate is
   computed. The run stays recorded, unsealed, so a rerun needs a reason.
6. **R12-6 and R12-17: checks before the lock is spent.** The registry must hold a v4 4d record. That
   record must pass `verify()` without its benchmark fields: view, precision, scoring code and files.
   Today's record is held to its v3 seal and its files, as before.
7. **R12-5: the reason Amendment 10 kept Stack Exchange.** Amendment 10 said each held-out site is held
   to the ≤1% bound. In fact the gate holds `stackexchange_heldout` as one slice: in the 2026-09-29 build,
   2,991 humans, so up to 20 false AI pass wherever they fall.
   - The decision to keep Stack Exchange stands, on the per-source bound the pass criteria set.
   - The head-to-head now reports each held-out site's false AI beside the gate. This is report only.
8. **R12-16.** Unseen-generator rows of sources outside the shared ground are reported by source, and
   not counted.
9. **R12-18.**
   - The training record binds its validation ids.
   - Training defaults to the Amendment 12 settings, and calibration to the 0.8% / 8% targets.
   - The scoring-code digest covers `src/detection/__init__.py`.
   - **Ruling:** `src/pipeline.py` stays out of the digest. The locked run scores through the service,
     so calibration never validated the app's sheet joining.
   - **Ruling:** the per-batch source-weight normalisation stays. It is the recipe screening measured and
     today's models trained with.

Recorded, not changed:

- **R12-7.** Amendment 2's rule for report-only rows is id and near-duplicate matches, and it holds.
  - MIRAGE's test humans share groups with v1–v3 stress rewrites, which are heavy paraphrases (median
    5-gram containment 5–7%).
  - No model ever fitted those rewrites.
  - Retiring the groups would leave MIRAGE about 40 test humans.
- **R12-8.** An independent all-8-gram check flagged 210 tested rows. 13 share a verbatim run of 30 or
  more words with earlier or training text. 122 are short MAGE prefixes wholly inside a longer text. All
  are below the build's near-duplicate thresholds, which the audit applies the same way.
- **R12-9.** HC3 has fresh test rows (358 humans and 402 AI in the 2026-09-29 build), so it is a gated slice
  for both model sets. Amendment 2 expected none. The rebuilt counts are in `build_record.json`.
- **R12-10.** 421 v3 MIRAGE stress rows left with their retired parents. 3 LEDE rows were dropped for a
  cross-source component.
- **R12-11.** Stack Exchange's generated answers run shorter than its humans' (52% against 24% under 100
  words), and some carry emoji or all-lowercase casual text.
  - The short variant is pre-registered, and so is its uniform 25–99-word target.
  - Calibration sets each word band's threshold on that band's validation humans.
  - Emoji stay in the view as style.
- **R12-12.** Process:
  - Amendments 10 and 12 changed things after the build without fixing a bug. Both came before any check
    or locked read.
  - Amendment 11 was committed 18 s after the run it governs started. That run stopped before any check.
  - The humanized rewrites ran around Amendment 9's commit, all on Amendment 9-trimmed text.
  - OpenTuringBench's AI-only probe scored 0.863 in the dry run and 0.846 in the build, one sample of
    4,000 per side each.
- **R12-14.** Today's record is accepted on its v3 seal and its files. Its scoring-code digest predates the
  4d branch, and the old behaviour is kept for records without a precision. This is carried to the report.
- **R12-15.** LEDE is about 56% of the pooled shared ground. MIRAGE's gated test AI rows are all Moonshot,
  which v3 held out and 4d fits. The head-to-head reports the pooled gain without LEDE beside the rule's
  figure (report only), and the report names MIRAGE's generator. The rule is unchanged.

**Next.** Corpus v4 is rebuilt with these loaders, and its build record and lock are committed. The 4d main
and short models train from scratch (`--settings fullbatch`, `format4`), then calibration and the optional
steps run. Ralph R12 part 2 then checks the run artifacts (claims 5–7 and 9) and re-checks part 1's fixes
before the lock is read. Nothing in the pass criteria or the improvement rule changes.

## Amendment 14 — full training, calibration and the optional steps (2026-09-30, before any locked text is read)

All figures below are on validation. The locked splits were not read.

**Main model** (DeBERTa-v3-large with LoRA, `--settings fullbatch`, `format4`, 191,785 training rows):

| Check | AUROC | Detection at 1% false AI |
|---|---|---|
| **0.1** | **0.9714** | **69.7%** |
| 0.2 | 0.9663 | 64.3% |
| 0.3 | 0.9709 | 61.1% |

It stopped early after two checks without a gain, as pre-registered. Check 0.1 was kept and its LoRA merged
(`merged-0.1`, weights stored in fp32; fp16 is the GPU inference precision, R13-9).

**Short model** (the same recipe at 160 tokens, 184,329 training rows):

| Check | 0.1 | 0.2 | 0.3 | 0.4 | **1.1** | 1.2 | 1.3 |
|---|---|---|---|---|---|---|---|
| Detection at 1% false AI | 56.1% | 55.8% | 69.5% | 64.3% | **73.1%** | 73.0% | 71.7% |

Check 1.1 was kept (AUROC 0.9772, `merged-1.1`, stored in fp32 and served in fp16 on the GPU).

**Calibration** targeted 0.8% false AI and 8% missed AI:
- All four word bands are supported.
- The short model serves the 25–49 and 50–99-word bands. Its decisive coverage there is 35.0% and 50.3%,
  against the main model's 25.2% and 33.7%.
- The main model serves the 100–199 band (decisive coverage 71.3%) and the 200+ band (47.8%).
- Peak inference memory is 3.07 GB, against the 3.5 GB limit.
- The registry holds a v4 record with fp16 for both models and no benchmark yet. It passes `verify()`
  without its benchmark fields.

**Optional step 1: the perplexity stack. Not adopted; the gain was not measured.**
- **Why it fails.** The pre-registration adopts the stack only if it adds 2 points **and** the GPU limits
  hold with every model loaded. They can't hold:
  - **Memory.** The two DeBERTa-v3-large classifiers (0.87 GB each in fp16) and the Qwen2.5-0.5B pair
    (0.99 GB each) come to 3.72 GB of weights before any activation, against the 3.5 GB peak limit.
  - **Sheet time.** The pair scored 3.76 texts a second on this GPU in the Increment 3 bake-off (9,807
    validation texts in 43.5 min). That puts a 3,000-text sheet at about 800 s for the pair alone, against
    the 120 s limit.
- **Why the gain wasn't measured.**
  - The first run ended in an unclean shutdown at 16:37 (Kernel-Power 41, no bug check).
  - The second held the classifiers and the pair on the GPU together. It spilled 1.64 GB into shared
    memory and was stopped. The code now scores with each classifier alone (commit `5d96ae3`, test-first).
  - The third ended in another unclean shutdown at 22:36, during the pair's scoring (1.13 GB spilled, GPU
    86 °C).
  - None wrote a result. The user chose not to run it again. The gain would only be reported, since the
    limits already decide the verdict.
- **The record.** `stack.json` records this decision and its basis, not a measurement.

**Optional step 2: register-aware thresholds. Not adopted.**
- **The gain.** Pooled detection would rise from 40.5% to 49.6% (+9.1 points).
- **Why not adopted.** 9 of 20 register cells go over their false-AI target. The worst is `qa_forum` at
  100–199 words: 36 false AI among 997 validation humans, upper bound 4.7%. Several sources go over too.
  The rule needs every cell within target, so the pooled thresholds stay.

Neither step is adopted, so there is no Plan B2 and the scoring code stays as calibrated.

**Ralph R12 part 2** (`ralph/13-increment-4d-part2.md`): a provisional PASS, with no Critical or High findings.
- **Fixed.** R13-7: every detector failure reason now stops the locked run, and the report-only diagnostic
  rows are read before the run is recorded (`bba3ae1`).
- **Recorded, not changed:**
  - **R13-1.** Under the unchanged pass criteria, No Robots (553 test humans) allows at most 1 false AI, and
    WasItAI none. On validation the 4d system made 3 false AI among No Robots' 584 humans. This is a known
    risk to the gate, not a reason to move a threshold after seeing validation.
  - **R13-3, R13-4, R13-5.** Residual cues and artifacts. They are listed in the Debt Ledger (D16, D17) for
    the next corpus rebuild. Today's models trained on the same BEEMO and RAID text, so the comparison isn't
    tilted.
  - **R13-6.** Debt Ledger entries were added, and the R12 report now carries a triage section.
- **Open before the lock.** Claim 7 (these verdicts), claim 9 (speed) and R13-2, which needs the disguise
  tricks scored on the 4d model. If a trick silently turns Likely AI into Likely human, it is fixed before
  the lock, with a new calibration if the fix touches scoring code.

## Amendment 15 — the speed check fails the GPU limits; the locked run doesn't start (2026-10-01, before any locked text is read)

Plan C Task 7 Step 2 measured the served 4d system, both DeBERTa-v3-large models behind the full service, on
validation texts (`data/benchmark/v4/model/speed.json`):

| Pass | Limit | Measured | Result |
|---|---|---|---|
| GPU | peak inference memory ≤ 3.5 GB | 3.94 GB | **fail** |
| GPU | single-text p95 ≤ 500 ms | 197 ms | pass |
| GPU | 500 × 6 sheet on the app's path ≤ 120 s | 566 s (plain batched cells: 243 s) | **fail** |
| CPU | single-text p95 ≤ 20 s | 3.7 s | pass (sheet about 7,460 s, reported only) |

- **Why memory fails.** Both models (0.87 GB of fp16 weights each), the out-of-distribution banks and batched
  activations overflow the 3.5 GB line. On this 4 GB card, 0.57 GB also spilled into shared memory.
  Calibration's 3.07 GB peak had one model loaded at a time.
- **Why sheet time fails.** The pre-registered sheet is 3,000 validation texts as cells plus 500 rows joined
  from six cells. Long texts are scored as several sentence windows, so the served system runs at about 12
  texts a second on the cells alone. The limit needs about 29, so a large model fails this even without the
  spill.
- **Consequence, as Plan C pre-registered.** "A failed limit stops here: it is recorded as an amendment and
  the plan returns to Plan B." The locked head-to-head does not run. The v4 lock stays unread, and nothing in
  the pass criteria, the improvement rule or the limits changes.
- **Ralph R12 (both parts) passed for reading the lock.** That verdict covers correctness, not the limits;
  claim 9 was the open item, and this is its answer.
- **Next steps are the user's decision** (recorded in the next amendment before any of them runs). The
  options are:
  - a faster served path, a scoring-code change followed by a fresh calibration and speed check;
  - a smaller or single model, which means retraining;
  - a product decision on the limit, which would be recorded as a change made after measuring.

## Amendment 16 — the GPU sheet limit is 900 s; a smaller inference batch for the memory limit (2026-10-01, before any locked text is read)

**The user's decision after Amendment 15.** The GPU sheet limit changes from 120 s to **900 s** (15 minutes)
for a 500 × 6 sheet on the app's path.
- **This limit was changed after measuring.** It is a speed limit, changed after the speed check, before any
  locked accuracy was seen. It does not touch the pass criteria, the improvement rule or the other two GPU
  limits (peak memory ≤ 3.5 GB, single-text p95 ≤ 500 ms), nor the CPU limit.
- **The report states it next to the original 120 s**, with the 566 s measured under it.

**The memory limit is met by a smaller inference batch.** The rule, fixed before measuring:
1. The batch size is the largest of 16, 8 and 4 whose peak inference memory on the full service stays at
   or under 3.5 GB. This is measured on validation texts only, on a 100-row sample of the pre-registered
   sheet, in a fresh process per batch size.
2. The chosen batch becomes the detector's inference batch, test-first. `detector.py` is scoring code, so
   calibration then runs again with the same targets.
3. The speed check then runs again, on both the GPU and the CPU, against the limits above.
4. Only when every limit passes does the locked head-to-head run, once.

If no batch size meets 3.5 GB, the plan stops again and returns to the user.

## Known risks

- **MAGA's aligned answers** are built to read as human and may be noisy for training. Validation shows
  it; they can be dropped by amendment before the lock.
- **Arena** is AI only and its answers are long. Length matters only within the model's window, and the
  AI-only probe checks it.
- **WasItAI is small:** 389 humans, so its gated upper bound is wide.
- **Large encoders trained with LoRA** may not beat fully trained base models; screening measures this.
- **Generation (about 1.8 million tokens)** is the longest step. It runs one model at a time on the 4 GB
  GPU, after screening.
