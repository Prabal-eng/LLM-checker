# Increment 4d report — 2026-10-01

**Outcome: `close_gate`.** On the fresh v4 locked test, neither model set passes the pre-registered gate.
The 4d models (DeBERTa-v3-large with LoRA, a main and a short-text model) detect far more AI than today's
models (DistilRoBERTa v5), with about half the false AI. But they fail two gated slices, and the
improvement rule fails on two older-generation sources. As pre-registered, the outcome table closes the
served gate when neither model set passes; that action waits for the user's go-ahead.

- **The run:** one locked run, 2026-10-01 00:41–03:29, `data/benchmark/v4/model/head_to_head.json`. The
  registry's 4d record is sealed with `passed: false`.
- **The rules:** the pre-registration is `increment-4d-preregistration.md`, Amendments 1–16. Nothing in the
  pass criteria or the improvement rule moved since v1. The one limit changed after measuring is the GPU
  sheet time (Amendment 16, below).

## Headline

All figures are on the v4 test split: 20,730 humans and 18,512 AI texts. Detection means AI decided Likely AI.

| | 4d | Today's models (v5) |
|---|---|---|
| AI detected | **37.8%** (6,995) | 24.5% (4,532) |
| False AI | **35** (0.17%; upper bound 0.22%) | 67 (0.32%; upper bound 0.40%) |
| Decisive coverage (supported texts) | **62.6%** | 47.9% |
| 25–99-word texts, AI detected (goal 25%) | **40.6%** ✓ | 10.4% |
| Frontier sets (WasItAI, LEDE; goal 70%) | 40.5% | 23.1% |
| Overall (goal 75%) | 37.8% | 24.5% |
| Shared-ground pooled detection (the improvement rule) | 39.4% | 26.5% |
| Gate | **fail** (2 slices) | **fail** (4 slices) |

## The gate

A gated slice needs ≥ 300 humans for false AI (95% upper bound ≤ 1%) and ≥ 100 AI texts for false human
(upper bound ≤ 10%), plus 40% supported coverage.

| Slice | 4d | Today's models |
|---|---|---|
| **no_robots** false AI | **5 / 553, upper bound 1.89%** ✗ | **3 / 553, upper bound 1.40%** ✗ |
| **wasitai2026** false human (missed AI) | **upper bound 19.7%** ✗ | **upper bound 33.2%** ✗ |
| **wasitai2026** false AI | 0 / 389 ✓ | **1 / 389, upper bound 1.21%** ✗ |
| **maga** false AI | 6 / 2,541 (0.47%) ✓ | **48 / 2,541, upper bound 2.40%** ✗ |

- **No Robots was the known risk** (Ralph R13-1). With 553 humans, one false AI is the most the bound
  allows. The 4d system made 3 of 584 on validation, and 5 on the test.
- **WasItAI** is the frontier set of 2026 models. Both systems miss much of its AI. 4d misses less, but
  still more than 10% at the upper bound.

## The improvement rule

The rule needs ≥ 5 points pooled on the shared ground, and no gated shared-ground slice losing more than
5 points.

| Shared-ground slice | AI texts | 4d | Today's | Change (points) | Gated |
|---|---|---|---|---|---|
| lede | 4,996 | 43.9% | 24.9% | **+19.0** | yes |
| wasitai2026 | 746 | 17.8% | 11.0% | +6.8 | yes |
| mirage | 102 | 94.1% | 81.4% | +12.8 | yes |
| **hc3** | 402 | 10.2% | 41.5% | **−31.3** ✗ | yes |
| **mage** | 921 | 8.0% | 21.8% | **−13.8** ✗ | yes |
| raid | 11 | 54.5% | 27.3% | +27.3 | no (too few) |
| unseen generators of shared sources | 1,696 | 56.1% | 33.7% | +22.4 | report only |
| **Pooled** | 8,874 | **39.4%** | **26.5%** | **+12.9** ✓ | |
| Pooled without LEDE (report only, R12-15) | 3,878 | 33.6% | 28.6% | +4.98 (just under 5) | |

- **The rule fails.** The pooled gain passes easily, but HC3 and MAGE each lose more than 5 points.
- **Where 4d falls behind.** Both are older-generation sources: HC3's ChatGPT-era answers, and MAGE's older
  models. The OpenAI family shows the same weakness: 17.8% for 4d against 15.5% for today's models, on
  4,727 AI texts, mostly MAGE and HC3.
- **Where 4d pulls ahead.** Current-generation text: LEDE, Arena, held-out Stack Exchange and WasItAI.

## Results by source (test split)

| Source | Humans | 4d false AI | Today's false AI | AI texts | 4d detected | Today's detected |
|---|---|---|---|---|---|---|
| arena140k | 0 | – | – | 2,993 | **32.9%** | 2.7% |
| dactyl | 1,975 | 0 | 6 | 2,057 | 34.7% | 32.5% |
| hc3 | 359 | 0 | 0 | 402 | 10.2% | **41.5%** |
| lede | 0 | – | – | 4,996 | **43.9%** | 24.9% |
| maga | 2,541 | 6 | 48 | 2,565 | **45.7%** | 30.2% |
| mage | 4,135 | 2 | 1 | 921 | 8.0% | **21.8%** |
| mirage | 7,787 | 22 | 7 | 102 | 94.1% | 81.4% |
| no_robots | 553 | 5 | 3 | 0 | – | – |
| openturingbench | 0 | – | – | 1,315 | 11.8% | **68.5%** |
| stackexchange_heldout | 2,991 | 0 | 1 | 2,398 | **59.3%** | 13.6% |
| wasitai2026 | 389 | 0 | 1 | 746 | 17.8% | 11.0% |

- **The OpenTuringBench anomaly.** OpenTuringBench is a source 4d trained on, and one it now barely
  detects. It is decisive on only 11.8% of those texts and detects 11.8% of the AI, against today's 68.5%.
  Its unseen-generator rows (not shared ground) show the same: 6.0% against 74.9%.
- **A likely cause** is Amendment 13's template trim. Today's model may be reading the kept "Date:/Category:"
  style cues, while 4d abstains on long news texts. This is the first thing to investigate next.
- **MIRAGE's 22 false AI** among 7,787 humans are within its bound (upper bound 0.40%).

**By word band:**

| Band | AI texts | 4d detected | Today's detected | Humans | 4d false AI | Today's false AI |
|---|---|---|---|---|---|---|
| under 25 | 625 | 0% (abstains) | 0% | 778 | 0 | 0 |
| 25–49 | 998 | **35.5%** | 17.9% | 1,364 | 0 | 14 |
| 50–99 | 2,390 | **42.7%** | 7.2% | 2,570 | 2 | 11 |
| 100–199 | 3,184 | **54.6%** | 26.5% | 11,035 | 30 | 21 |
| 200+ | 11,315 | 34.3% | 29.5% | 4,983 | 3 | 21 |

The short-text specialist is 4d's clearest gain: under 100 words it detects about four times more AI than
today's routed system, with fewer false AI. The long band is its weakest; OpenTuringBench's long news
texts sit there.

**Unseen generator families** (held out of training for both; report only):

| Family | AI texts | 4d | Today's |
|---|---|---|---|
| TII (Falcon 3) | 589 | 57.2% | 28.2% |
| IBM (Granite) | 600 | 42.0% | 21.8% |
| Microsoft (Phi) | 712 | 51.5% | 62.2% |
| **All unseen** | 1,931 | **50.0%** | 38.7% |

**Held-out Stack Exchange, site by site** (Amendment 13, R12-5, report only):
- **4d:** zero false AI on every one of the 8 sites. AI detected ranges from 49.0% (travel) to 72.9%
  (academia).
- **Today's models:** one false AI (diy). AI detected ranges from 4% to 27% per site.

## Short-text excerpts, stress and diagnostics (report only)

**Excerpts of test texts under 100 words** (cut on each model's own view):

| | 4d routed | 4d main model only | Today's routed |
|---|---|---|---|
| 25–49 words, AI detected | **34.0%** (of 2,394) | 17.8% | 31.4% |
| 50–99 words, AI detected | **35.6%** (of 11,254) | 10.3% | 7.3% |
| False AI (25–49 / 50–99) | 3 / 6 | 9 / 0 | 24 / 15 |

**Stress:**
- LLM-polished and rewritten MIRAGE texts:
  - polished: 4d 49.6% against today's 26.8%;
  - rewritten: 4d 41.9% against today's 22.1%.
- Human text from native and non-native writers (detector-bias set): no false AI from either system.

**Diagnostic** (rasbt50k answers from 2026 models, outside the lock), AI detected:

| Generator | 4d | Today's |
|---|---|---|
| Alibaba | 61.1% | 45.5% |
| Google | 52.1% | 34.3% |
| Moonshot | 42.1% | 18.5% |
| DeepSeek | 39.4% | 14.6% |
| OpenAI | 25.2% | 7.3% |

**In-house rater sheets** (unverified labels, so flags may be real AI use; no rater is named):

| | 4d | Today's |
|---|---|---|
| Cells never viewed before, flagged Likely AI | 3.9% (63 of 1,621) | 2.3% |
| Previewed cells, flagged Likely AI | 0.84% | 0.51% |

## Limits

| Check | Limit | Measured |
|---|---|---|
| GPU peak inference memory | 3.5 GB | 3.94 GB at batch 32 ✗ → **2.85 GB at batch 16** ✓ |
| GPU single-text p95 | 500 ms | 197 ms ✓ / 196 ms ✓ |
| GPU 500 × 6 sheet (the app's path) | **900 s** (was 120 s) | 566 s ✗ → **280 s** ✓ |
| CPU single-text p95 | 20 s | 3.7 s ✓ / 3.8 s ✓ |

- **The sheet limit was changed after measuring.** The first speed check failed memory and sheet time
  (Amendment 15). The user then set the GPU sheet limit to 900 s (Amendment 16).
- **The same numbers under the original limit.** At batch 16 the sheet takes 280 s, which still fails the
  original 120 s, so it passes only under the revised limit.
- **Memory** passes with inference batch 16, which triggered a fresh calibration with unchanged thresholds.
- **During the locked run** the 4d system peaked at 2.85 GB.

## What happened along the way

- **Ralph R12 part 1 failed** (2 High findings).
  - **The length-matching bug:** MAGA and DACTYL weren't length-matched, a shortcut at AUC 0.645.
  - **Detector errors:** they could silently seal a locked run.
  - **Amendment 13** fixed both High and all five Medium findings, test-first. It added load-time trims of
    AI preambles and templates, a new model view (`format4`), and generation-artifact filters.
  - Corpus v4 was rebuilt twice, and its first lock was never read. The first training run, on the flawed
    corpus, was stopped and not used.
- **Training and calibration** (Amendment 14):
  - the main model stopped early at check 0.1, validation TPR 69.7% at 1% false AI;
  - the short model at check 1.1, validation TPR 73.1%;
  - calibration targeted 0.8% false AI and 8% missed AI.
- **Optional steps:** neither was adopted.
  - **The perplexity stack:** its GPU limits can't hold, and its gain wasn't measured after two unclean
    machine shutdowns during its runs (user decision).
  - **Register thresholds:** +9.1 points, but 9 of 20 cells went over target.
- **Ralph R12 part 2 passed** the lock. 352 disguise-trick scorings on the 4d models turned no Likely AI into
  Likely human. Its fixes were:
  - detector failure codes;
  - reading the diagnostic rows before the run is recorded;
  - the CPU speed pass that didn't hide the GPU.
- **Infrastructure:**
  - The machine shut down uncleanly twice (16:37 and 22:36 on 2026-09-30), both times during the stack's
    heavy GPU work with shared-memory spill.
  - The stack's code was fixed to load one model at a time.
  - CI was failing since 2026-09-28 because 12 tests reached torch, which CI doesn't install. They now skip
    there, and CI is green (`8defe3a`).
- **The Debt Ledger** has D16–D20 for recorded findings, and D14–D15 are repaid.

## What the outcome means

- **The pre-registered action** for `close_gate` is to record the failed v4 run in the served registry, the
  main checkout's `models/registry.json`. The run is appended to `locked_benchmark_history`, and `benchmark`
  is re-sealed with `passed: false`, so the served gate closes.
- **Why.** Today's served models no longer meet the ≤ 1% false-AI contract on a fresh test: MAGA 2.4%, No
  Robots 1.4%, WasItAI 1.2% at the upper bound.
- **What it means for the app.** With the gate closed, the app stops giving decisive Likely AI / Likely human
  labels until a model passes a locked test.
- **The action leaves this worktree, so it waits for the user.**

**For a next increment, from this run:**
1. **Why 4d abstains on OpenTuringBench's long news texts** (11.8% decisive), and whether the template trim
   removed what today's model relies on.
2. **Old-generation text** (HC3, MAGE, OpenAI-era answers), where the new training mix underweights what
   today's model learned.
3. **The small gated slices.** No Robots (553 humans) and WasItAI (389 humans) allow one false AI or none.
   Their size, not only the model, decides the gate.
4. **Missed AI on WasItAI**, which is 2026-model text, even for the large model.
