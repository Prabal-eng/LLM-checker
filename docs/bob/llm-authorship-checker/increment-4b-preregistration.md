# Increment 4b pre-registration — 2026-09-27

Written and committed **before** corpus v2 and its locked test set exist. Nothing below may be changed after the test set is built, except to fix a bug. Any fix is recorded with its reason and reported next to the earlier result.

## Why

`distilroberta-authorship-v2` failed the v1 locked benchmark on Beemo alone. Human-written, assistant-style answers were flagged as AI 10% of the time (limit 1%). No training text was human-written in an assistant's voice. The sponsor also wants **less accuracy loss on texts under 100 words**, and a separate short-text model is acceptable.

## Corpus v2

| Source | Role | What it adds | Licence (tagged per row) |
|---|---|---|---|
| hc3, mage, mirage, raid, detector_bias, legacy_inhouse | as in v1 | — | as in v1 |
| beemo | **now fitted** (train/val) | human and model answers to the same No Robots prompts | MIT |
| no_robots | fitted, human only | more human assistant-style answers (Beemo's prompts excluded) | CC-BY-NC-4.0 |
| rasbt50k | fitted | human text from openly licensed collections (StackExchange, Wikipedia, arXiv, PLOS, PMC, Gutenberg, OpenStax, blogs), plus 2026-model texts from the same documents | per row (CC0, CC BY, CC BY-SA, public domain); AI side "other" |
| aita2026 | fitted | r/AmItheAsshole posts rebuilt from AI2 Scruples, plus same-title posts by seven 2026 models | Apache-2.0; human side is Reddit content |
| **dolly** | **locked test only, never fitted** | human assistant-style answers of 25+ words | CC-BY-SA-3.0 |
| **oasst1** | **locked test only, never fitted** | human assistant replies of 25+ words (English, reviewed, not deleted, not synthetic) | Apache-2.0 |

Split rules:
- Held-out sources (`dolly`, `oasst1`) go to the test split whole. Any row that shares a group or near-duplicate cluster with another source is dropped instead.
- **v1's locked test is retired.** Components containing a v1 test sample may be trained on but never tested again. A retired component that stress variants would force into evaluation is dropped.
- Buckets are re-drawn with salt `v2` (70/15/15 by component). The fresh test from the v1 sources is therefore drawn from rows v1 and v2 models trained on. The new model is trained from scratch without them.
- Beemo's stress variants follow their parent. They stay in the stress split when the parent is in evaluation and are dropped when the parent is fitted. All of Beemo was v1 test, so all of it is fitted, and its edited variants leave the stress split.
- Held-out families stay `cohere`, `moonshot`, `bigscience`. rasbt50k's Kimi K3 texts are moonshot, so they are held out as an unseen 2026 generator.

## Models

1. **Main model** `distilroberta-authorship-v3`: the v2 recipe unchanged (DistilRoBERTa, collapsed whitespace, source-balanced loss, 256 tokens, 2 epochs, best epoch by validation TPR at 1% FPR).
2. **Short-text specialist** `distilroberta-authorship-short-v1`: the same recipe on 25–99-word texts at 160 tokens.
   - Training data: natural 25–99-word train texts, plus one sentence-aligned excerpt of 25–99 words from every longer train text.
   - Excerpts come from a deterministic function of the sample id and are applied identically to both labels.

## Routing rule (validation only)

Both models are calibrated on validation: natural validation texts plus excerpts made the same way. For each band under 100 words (25–49, 50–99), the specialist serves the band only if **its decisive coverage under the bounded thresholds beats the main model's by at least 1 point**. Otherwise the main model serves it. Bands of 100+ words always use the main model. Calibration targets stay at 0.8% false AI and 8% missed AI, a margin under the release bounds.

## Pass criteria (unchanged from v1)

These are computed on the v2 `test` split through the full service, with the language gate, OOD gate, length bands and routing:
- Every source slice with ≥300 human texts: false-AI 95% upper bound ≤ 1%.
- Every source slice with ≥100 AI texts: false-human 95% upper bound ≤ 10%.
- Decisive coverage ≥ 40% of supported text.

## Reported, not gating

- **Short-text falloff:** 25–49 and 50–99-word sentence-aligned excerpts of test texts. Per source: false-AI, false-human, detection and coverage, for the routed system and for the main model alone, set against natural texts of 100+ words.
- Test results by length band.
- The unseen-generator split (now including Kimi K3), the stress sets, and in-house rater text (unverified labels, flag-only). v1-previewed in-house IDs stay reported apart.

## Reruns

The sponsor allows more than one locked run. A rerun happens only to fix a code bug, never to change a model, threshold or rule after seeing test results. Every run is appended to `locked_benchmark_history` with its reason.

## Amendment 1 — before any model was trained (2026-09-27, 00:58)

The first v2 build's shortcut probe (a surface-feature classifier fitted on train, scored on test) flagged **rasbt50k at AUC 0.98**. Inspection found three artifacts:
- Its human rows are fixed-length chunks cut mid-sentence: 23% end a sentence, against 99.7% of AI rows.
- Code is stripped from its StackExchange answers, leaving broken sentences.
- Outside StackExchange, the AI text answers a question about the source document instead of matching its genre, which is the "assistant-style = AI" confound behind v1's Beemo failure.

**rasbt50k becomes diagnostic only:** never fitted, never gating. The benchmark reports detection on its 2026-model texts (Kimi K3, DeepSeek V4, Gemini 3.6, Qwen 3.6, OpenAI) and, with this caveat, its human rows. Kimi K3 therefore no longer enters the unseen-generator split. No model output or test result informed this change. The corpus was rebuilt and relocked with the reason recorded.

## Known label risk

Dolly and oasst1 contributors were asked not to use model output. That can't be verified, so a false AI flag on those slices may occasionally be real AI text. The count is reported, not excused.
