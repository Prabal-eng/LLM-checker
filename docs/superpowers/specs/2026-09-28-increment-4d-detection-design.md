# Increment 4d: raise AI detection at the same false-AI contract

Date: 2026-09-28. Status: design approved by the sponsor in conversation (goal, data, locked test, model
size); this spec records it. The binding version of the test plan is the 4d pre-registration, written
from this spec and committed before corpus v4 is locked.

## Why

Run 4 of Increment 4c passed the v3 locked benchmark, but the strict bars catch only 65% of AI text. The
weak spots on the locked test:

| Slice | AI detected |
|---|---|
| Short texts, 25–49 words | 13% |
| Short texts, 50–99 words | 8% |
| Stack Exchange | 15% |
| OpenAssistant | 15% |
| MAGE | 31% |
| RAID | 42% |
| HC3 | 47% |
| Unseen Cohere | 12% |
| Unseen BigScience and IBM | 28% |
| AITA and MIRAGE | 86–87% |

On validation, the model catches about 63% of AI text at 1% false AI (short model: 55%). The sponsor
set three equal priorities: short texts, overall detection and 2025–26 frontier models.

## Success criteria

- **Gate, unchanged since v1:**
  - every slice with at least 300 humans has a false-AI 95% upper bound of at most 1%;
  - every slice with at least 100 AI texts has a false-human upper bound of at most 10%;
  - coverage is at least 40%.
- **Improvement rule:** today's models (DistilRoBERTa v5 and short-v3) and the 4d models are scored
  head to head on the same v4 locked test. The 4d models replace today's only if all three hold:
  - they pass the gate;
  - they detect at least 5 points more AI on the shared-ground slices together;
  - no gated shared-ground slice loses more than 5 points.
- **Goals, reported but not gating:**
  - 75% AI detected overall;
  - 25% on 25–99-word texts;
  - 70% on the held-out 2025–26 frontier sets.

| Outcome on the v4 locked test | What happens |
|---|---|
| 4d passes the gate and meets the improvement rule | 4d is registered and replaces today's models |
| 4d fails, today's pass | Today's models stay registered |
| Both fail | The gate closes until fixed, as in 4b |

## Constraints

- **GPU limits, unchanged:**
  - peak inference memory of at most 3.5 GB;
  - single text of at most 500 ms at p95;
  - a sheet of 500 rows × 6 columns in at most 120 s.
- **CPU limits relaxed** (build brief Revision 3, sponsor decision 2026-09-28): the CPU is the fallback
  mode, so model size is set by the GPU.
  - Single text: at most 10 s at p95, previously 2 s.
  - A 500-row sheet: at most 30 minutes, previously 600 s.
- **Unchanged rules:**
  - local only, English only;
  - rater comments are flag-only;
  - no training on in-house sheets;
  - amendments come before results;
  - one locked run, with reruns only for bugs and each reason recorded.
- **Training hardware:** RTX 3050 Laptop, 4 GB, bf16-capable. `peft` 0.21 and `bitsandbytes` 0.50.2
  are installed for LoRA and 8-bit optimisers.

## Training data (corpus v4)

### New public sources

Each source's licence is recorded per row.

| Source | Licence | Use |
|---|---|---|
| MAGA-Bench, English (`anyangsong/MAGA`) | MIT | Q&A (Yahoo Answers, Natural Questions), Reddit, reviews, news; 2025 generators; human and AI paired by title; includes short texts |
| DACTYL (`ShantanuT01/DACTYL`) | MIT (card) / CC-BY-4.0 (paper) | Tweets, reviews, abstracts, news; few-shot generation. WritingPrompts rows are dropped because they duplicate MAGE |
| Arena Human Preference 140k (`lmarena-ai/arena-human-preference-140k`) | Prompts CC-BY-4.0; outputs under provider terms, recorded as such | AI replies only: English, first turn. 2025 frontier models, including Claude 4, Gemini 2.5, o3, Grok, DeepSeek, Llama 4 and Cohere Command A |
| OpenTuringBench (`MLNTeam-Unical/OpenTuringBench`) | CC-BY-NC-4.0 | AI news from 2024 open models. Its human side must be fetched from HuffPost URLs, so only the AI texts are used |

- **AI-only sources:** Arena and OpenTuringBench are balanced against human text of the same register.
  - Arena against the human assistant-style answers: Dolly, OpenAssistant, No Robots.
  - OpenTuringBench against the human news already in the corpus.
  - Source-balanced weights apply as today.
  - The shortcut probe (the 19 layout and length features) keeps its v3 warning threshold of AUC 0.85,
    measured after the model view. For an AI-only source, the probe compares it with the human texts
    it is balanced against. A source above 0.85 is excluded before any training.
- **Rejected:**

  | Source | Reason |
  |---|---|
  | Kaggle "multi-model" | Simulated with a text-faking library, not model output |
  | M4GT | Old generators; its ELI5 part overlaps HC3 and MAGE; licence unclear |
  | EditLens | Access-restricted; its AI-edited class needs a labelling policy |
  | COLING'25 GenAI Task 1 and `acmc/multi_domain_ai_human_text` | Built from HC3, RAID and MAGE |
  | PAN 2025/26 | No redistribution |

### Stack Exchange joins training

The v3 locked test is spent, so its 8 sites join training and validation:
- sites: cooking, gardening, money, parenting, pets, outdoors, fitness and interpersonal;
- their human answers go in, with new AI answers from the four training generators;
- the Phi and Granite answers generated for the v3 test stay out of every split, so those two families
  remain unseen.

### Native short answers and more varied generation

- **Generators:** Llama 3.2 3B, Qwen 3.5 4B, Gemma 3 4B and Ministral 3 3B, each run one at a time
  (4-bit, via Ollama).
- **Short answers:** the generators are asked for 25–99 words. The answers are paired with short human
  texts: 25–99-word Stack Exchange answers, Reddit comments, and the short rows of MAGA and DACTYL.
  Excerpts cut from long answers stay, but no longer dominate the short model (today 82k of its 111k
  training texts are excerpts).
- **Variety:**
  - a share of prompts ask for plain text with no formatting;
  - casual and forum-user personas;
  - a temperature range of 0.6–1.1;
  - 10% of AI answers are rewritten by a different training generator to sound casual. The label stays
    AI.
- **Filters:** the Amendment 4 filters still apply. Answers cut off by the length limit, or that mention
  a word count, are dropped.

### Size

Each source and label pair is capped, so no source dominates. The main model trains on about 200k
texts, up from 115k.

### Text view `format3` (closes debts D14 and D15)

- all bullet glyphs (such as the middle dot) and `(1)`-style markers are dropped;
- the view gives the same result when applied twice, including on `**1.**`;
- spaced and unspaced dashes follow one rule;
- earlier view names keep their behaviour, and `text_view_fingerprint` changes for `format3` only.

## The v4 locked test

### Parts

1. **New Stack Exchange sites.** Eight sites never used before: travel, diy, workplace, academia,
   bicycles, mechanics, photo and writers.
   - Questions are dated 2021 or earlier, so the human answers predate ChatGPT.
   - The AI answers come from three families no model trains on: Phi, Granite, and Falcon 3 (TII). The
     third family is confirmed available before the pre-registration is committed.
   - It includes 25–99-word answers, so short texts get a fresh gated slice.
2. **Frontier sets, held out entirely:**
   - **WasItAIGenerated 2026-09:** only its full-text rows (about 1,600), covering 2026 models such as
     GPT-6, Claude Opus 5.5 and Gemini 3.x.
     - Its humans are probably fewer than 300, so its false-AI rate is reported, not gated.
     - Its 421 humanized rows form a report-only stress slice.
     - Rows given only as URLs are not re-fetched.
   - **LEDE, English frontier part:** Claude 4.5–4.6, GPT-5.x and Gemini 3.1 news. It is AI only, so it
     measures detection.
3. **Fresh rows from existing sources.** MAGE, RAID, MIRAGE and HC3 still have rows no version ever used,
   since each version took at most 3,000 per stratum. These give continuity with past results.
4. **Test portions of the new training sources** (MAGA, DACTYL and OpenTuringBench), split by group.

### Leakage rule

- No v4 test text, or any member of its near-duplicate group, may appear in any split of v1–v3. This
  guarantees today's models never saw it.
- It may not appear in v4 training or validation either.
- Containment dedup runs across every source.
- The audit re-checks all three rules independently, and fails the build on any violation.

### Shared-ground slices (for the improvement rule)

The 4d models train on MAGA, DACTYL, Arena, OpenTuringBench and Stack Exchange; today's models did not.
The improvement rule therefore counts only these slices:
- fresh rows of the existing sources, which both model sets trained on;
- the frontier sets and unseen-generator slices, which neither trained on;
- the new Stack Exchange sites, whose sites, questions and generators are new to both.

Test portions of the new training sources are reported, but excluded from the rule, because they share a
distribution with the 4d training data.

## Models

### Screening

Screening runs now, on v3 train and validation. It never reads a locked split.

- **Command:** `python -m src.bakeoff.screen`.
- **Run:** each candidate trains one epoch on 24k label-balanced training texts, with source-balanced
  weights and length-grouped batches. It is scored on the validation split.
- **Tracks:** the main track (256 tokens) and the short track (25–99-word texts and excerpts, 160 tokens).
- **Recorded:**
  - AUROC;
  - detection at 1% false AI, overall and per word band;
  - training time and peak training memory;
  - scoring speed.

| Candidate | Size | Training |
|---|---|---|
| DistilRoBERTa (today's baseline) | 82M | Full, bf16 |
| RoBERTa-base | 125M | Full, bf16, gradient checkpointing |
| ModernBERT-base | 149M | Full, bf16, 8-bit AdamW, checkpointing |
| DeBERTa-v3-base | 184M | Full, fp16 with loss scaling, 8-bit AdamW, checkpointing |
| ModernBERT-large | 395M | LoRA r=16 on all linear layers, bf16, checkpointing |
| DeBERTa-v3-large | 435M | LoRA r=16, fp16 with loss scaling, checkpointing |

DeBERTa runs in fp16 because DeBERTa-v3-small gave NaN scores under bf16 in Increment 3. A non-finite
loss stops that candidate, and the failure is recorded.

**Selection rule:**
- The winner has the highest mean of the main-track and short-track detection at 1% false AI.
- Within 1 point, the faster model wins.
- The top two go on to full training on corpus v4 if they are within 2 points of each other; otherwise
  only the winner does.
- A candidate is dropped if it breaks a GPU limit at inference.

### Full training on corpus v4

- **Checkpoints:** validation every quarter epoch. Training stops after two checks without improvement,
  at most three epochs, and the best check is kept. This replaces fixed two-epoch runs; in 4c both
  models peaked after epoch 1.
- **Length-grouped batches,** as in screening.
- **Two models, as today:** a main model for 100+ words and a short model for 25–99 words.
- **LoRA models** have their adapters merged into the weights before saving, so the detector loads a
  plain model.
- **Precision:** the detector reads its inference precision from the registry record. Today it is
  hard-coded to bf16, which would break DeBERTa. This changes sealed scoring code, which is expected
  because v4 needs a fresh locked run anyway.
- **Sealed code on the 4d branch.** `format3` (text.py) and the excerpt view (excerpts.py) are sealed too.
  On the 4d branch, today's registry record therefore fails its seal ("scoring code changed since
  calibration").
  - Nothing serves today's models from this branch; the main checkout keeps serving them.
  - The head-to-head scores today's models through their own records, with the gate overridden, as the
    locked benchmark already does for an unvalidated candidate.
  - Each model's excerpts are cut from its own recorded view.

### Perplexity signal

This step comes after the classifier is chosen.
- The Increment 3 zero-shot scorer (`ZeroShotScorer`: Qwen2.5-0.5B as observer, its instruct version as
  performer) scores each validation text: the Binoculars ratio and the Fast-DetectGPT curvature.
- A 1.5B model would take about 3 GB of GPU memory in fp16 and would not fit beside the classifier
  within 3.5 GB, so none is screened.
- A logistic stack combines the classifier's logit with these features. It is fitted on one half of
  validation and measured on the other.
- It is kept only if it adds at least 2 points of detection at 1% false AI, and the GPU limits still hold
  with both models loaded.

### Style-aware thresholds

- **Registers:** a small style classifier assigns each text to one of five registers:
  - Q&A and forum answers;
  - assistant replies;
  - news and expository text;
  - essays and academic text;
  - short social posts and reviews.
- **The classifier:** TF-IDF plus logistic regression on source-to-register labels. It runs on CPU in
  under a millisecond and becomes sealed scoring code.
- **Thresholds:** calibration sets ai_min per length band × register cell, on that cell's own validation
  humans, at the same 0.8% target. A cell with fewer than 600 validation humans uses the band's pooled
  bar. A cell's bar may be lower than the pooled bar only where its own humans support it.
- **When it applies:** at check time the register is predicted from the text, never taken from the
  source.
- **Adoption test:** it is adopted only if it raises validation detection by at least 2 points, with every
  cell within target. Otherwise the pooled per-band bars stay.

## Process

1. **Screening** on v3 data (running since 2026-09-28).
2. **Downloads** of the new sources and the new Stack Exchange sites (running since 2026-09-28).
3. **Loaders** for MAGA, DACTYL, Arena, OpenTuringBench, WasItAI and LEDE, the new Stack Exchange sites,
   and the `format3` view. All test-first.
4. **Pre-registration** `increment-4d-preregistration.md`, committed before any v4 generation or build.
   Later changes are numbered amendments, each committed before the results it could affect.
5. **Generation:**
   - training answers from the four training generators, including short answers and the variety
     settings;
   - test answers from Phi, Granite and Falcon 3 on the new Stack Exchange sites;
   - one Ollama model at a time, run detached.
6. **Corpus v4 build and independent audit.** The audit checks leakage against v1–v3, the shortcut probe,
   PII and licences, and the lock.
7. **Full training** of the screening winner, then calibration, then the perplexity and style-threshold
   steps, each adopted or dropped on validation.
8. **Ralph review before the locked run.** The review covers the corpus, the view, disguises and the
   validation results. Findings are fixed as amendments before the lock is read. This is the step that
   cost four locked runs in 4c.
9. **One locked run,** head to head. The report records the outcome table above.
10. **Debt:** D14 and D15 close with `format3`.

## Error handling

- **Screening:** a failing candidate is recorded and the others still run, and a rerun retries only
  failed or missing candidates.
- **Downloads** resume from partial files.
- **Generation:** the Amendment 4 artifact filters still drop cut-off answers and word-count echoes.
- **Registry:** it stays fail-closed.
  - A missing field, a hash mismatch or a report that doesn't match the seal closes the gate.
  - The new precision field and the register classifier are covered by the seal.

## Testing

- **Unit tests** for:
  - each new loader (fixtures, licences, pairing, dropped duplicates);
  - `format3` (idempotence, bullets, `(1)` markers, dashes);
  - the register classifier;
  - the logistic stack;
  - the v4 split rules (no v1–v3 row or group in the v4 test);
  - head-to-head scoring and the outcome table;
  - LoRA merging;
  - the precision field.
- **Screening options:** already covered in `tests/test_screening.py`.
- **Without a GPU:** the full suite still passes on a machine with no GPU or model files.

## Risks

- **MAGA's "aligned" AI text** is built to look human (role play, self-refinement, adversarial training),
  so it may be noisy for training. The loader keeps a config flag, and validation decides.
- **Arena is AI only.** If the shortcut probe flags it, it is excluded.
- **WasItAI is small,** and its humans are probably under 300, so its false-AI rate is reported, not gated.
- **Large models with LoRA may not beat fully trained base models.** Screening measures this.
- **Generation time dominates the schedule.** Public data replaces much of what 4c generated locally.
