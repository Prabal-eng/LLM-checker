# Increment 4b report — 2026-09-27

## Result

**`distilroberta-authorship-v3` with the short-text specialist did not pass the fresh v2 locked test, so the validation gate stays closed.** Every check still returns `Inconclusive` with `MODEL_NOT_VALIDATED`.

- **Passes:** every trained-on public source (HC3, MAGE, MIRAGE, RAID) stays within bounds: 6 of 7,201 human texts are flagged as AI, and every slice's upper bound is ≤0.52%. Missed AI stays ≤4.3% per slice (95% upper bound). Coverage is 68.6%, and 2 of 3,770 in-house texts are flagged.
- **Fails:** four slices exceed the 1% false-AI bound.

| Failing slice | Human texts flagged as AI | 95% upper bound | Note |
|---|---|---|---|
| **OpenAssistant oasst1 (never trained on)** | **592 / 2,999 (19.7%)** | 21.0% | Volunteers role-played the assistant. 2.5% of replies literally say "as an AI language model" (Dolly: 0%), so some may be pasted ChatGPT output. The labels can't be verified. |
| Dolly (never trained on) | 37 / 3,998 (0.93%) | 1.22% | Near miss. The rate itself is under 1%, but 3,998 texts can't prove ≤1%. |
| No Robots | 3 / 532 | 1.45% | The slice is too small to prove ≤1% with 3 errors. |
| AITA (2026 models) | 2 / 417 | 1.50% | Same. |

Assistant-voiced human text improved but isn't solved. v2 flagged 10% of Beemo's human answers. v3, with Beemo and No Robots in training, flags 0.9% of Dolly's, but 20% of OpenAssistant's, which read the most like an AI assistant.

## Short-text drop-off (the sponsor's goal)

The specialist handles 25–99-word texts. It won both short bands on validation under the pre-registered rule: 46% against 29% decisive at 25–49 words, and 58% against 50% at 50–99. On excerpts of locked-test texts (report only):

| Excerpt band | AI detected: routed | AI detected: main model alone | Decisive: routed | Decisive: main alone | Human texts flagged as AI: routed (excl. OpenAssistant) |
|---|---|---|---|---|---|
| 25–49 words | **52%** (927 / 1,768) | 12% (212 / 1,768) | 47% | 20% | 10 / 1,238 (0.8%) |
| 50–99 words | **54%** (4,108 / 7,539) | 42% (3,157 / 7,539) | 57% | 48% | 14 / 5,099 (0.3%) |
| 100+ words (natural test, main model) | 80% (7,880 / 9,798) | — | 81% | — | — |

The drop-off under 100 words shrank. Detection at 25–49 words went from 12% to 52%, and false AI stays near 1% everywhere except OpenAssistant (41 of 305 and 207 of 1,365 excerpts).

## Also reported (not gating)

- **Unseen generators:** Moonshot 80% detected, Cohere 31% (15% labelled human), BLOOM 33%.
- **Newest 2026 models (rasbt50k, diagnostic only):**

  | Family | Detected | Labelled human |
  |---|---|---|
  | Alibaba (Qwen 3.6) | 57% | 1.8% |
  | Google (Gemini 3.6) | 33% | 10% |
  | Moonshot (Kimi K3) | 26% | 21% |
  | DeepSeek V4 | 22% | 16% |
  | OpenAI (codename models) | **10%** | **39%** |

  These are answers about documents, a register the model never trained on. Still, current frontier models are the largest detection gap. Human rows there are mid-sentence chunks, so they aren't comparable.
- **Stress sets:**
  - LLM polish and rewrite: 27–31% labelled human.
  - RAID paraphrase: 15% labelled human.
  - Word-level attacks: 3–7% labelled human.
  - Homoglyph and zero-width attacks: always `Inconclusive`.
- **Human essays:** 1 of 90 non-native essays flagged, 0 of 88 native, 34 of 215 in the unlabelled group.
- **In-house rater text (flag-only, unverified):** 1 of 1,621 never-viewed and 1 of 2,149 previewed texts flagged; 0 of 642 six-rationale rows flagged.
- **Peak inference VRAM:** 1.47 GB with both models loaded (NFR ≤3.5 GB ✅).

## What was built

| Part | Where |
|---|---|
| Corpus v2 | `src/corpus/sources/{no_robots,aita2026,dolly,oasst1,rasbt50k}.py`, `split.py`. v1's locked test is retired (41k ids). Buckets are re-drawn with salt `v2`. Dolly and oasst1 are held out whole. Beemo is fitted, and its stress variants follow their parent. |
| Short-text specialist | `src/authorship/excerpts.py`, `train.py --variant short`. Trains on 25–99-word texts plus deterministic sentence-aligned excerpts at 160 tokens. |
| Validation routing | `calibrate.py`: both models are calibrated on the same rows, and a short band is routed only when the specialist wins coverage by ≥1 point. |
| Routed serving | `LengthRoutedDetector` and `registry.short_parts`. Both models' weights and banks are hash-verified. |
| Benchmark | Results by length band, the short-excerpt drop-off (routed vs main only), the rasbt50k diagnostic, and a one-shot guard scoped to the corpus version. |

The test suite has 212 tests and needs neither a GPU nor model files.

## What went wrong, and what changed

1. **rasbt50k had shortcut artifacts** (probe AUC 0.98). It was made diagnostic-only before any model was trained, and this is recorded as Amendment 1 in the pre-registration.
2. **Calibration bug (found on validation):** an AI threshold could not sit just above the top human score. That dropped v3's 25–49 and 100–199-word bands until it was fixed (`d3dc2c0`). Routing was then recomputed on validation.
3. **The first benchmark attempt was cut off** by a system power loss (Kernel-Power 41, no bugcheck) at 28,000 of 121,034 texts. Nothing was written or viewed. The rerun records the reason.
4. **Background training was killed** by Claude Code under memory pressure at 01:23. It was restarted in the morning, once the D: drive was reconnected.

## Review caveats (Ralph R07, 2026-09-27)

The Ralph review of Increments 4 and 4b (`ralph/07-increment-4.md`) failed with two high and four medium findings. It changes how some numbers above should be read:
- **Short texts and long documents were decided from their opening 256 tokens only** (R07-2), and light homoglyph or zero-width edits flipped Likely AI to Likely human (R07-1). Both are now fixed. The v2 benchmark measured the system without these fixes.
- **The stress split reused 49,691 attacked variants** of retired v1-test texts (R07-3). The stress numbers are therefore not fresh.
- **549 MAGE test texts are at least half word-for-word in train or validation** (R07-6). The MAGE slice partly measures memorized text.
- **The build's raw-text shortcut probe flagged RAID** (AUC 0.924) and was not reported here. On the model's whitespace-collapsed view, RAID is 0.61. A per-source probe on that view separates MIRAGE at 0.866 (R07-7).

## Next (options, sponsor to decide)

1. **Treat assistant-voiced human text as out of scope.** Add an "assistant register" support check that abstains instead of labelling. That is honest, but it removes coverage exactly where rater comments may sound like an assistant.
2. **Get verified labels for the hard register:** sponsor-verified human and AI rater rationales (D6), or paired AI answers to Dolly and OpenAssistant prompts generated locally (D8). Then build corpus v3 with a fresh test.
3. **Close the 2026-model gap:** add clean, same-genre text from current frontier models (like AITA) across more registers, since rasbt50k's genre mismatch rules it out.
4. **Small-slice bounds:** No Robots and AITA fail with 2–3 errors because 400–530 texts can't prove ≤1%. A larger test for those sources would settle them either way.

## Reproduction

```bash
export HF_HOME=D:/llm-authorship-corpus/hf-cache
python -m src.corpus.build --relock "..."        # corpus v2 (already locked; see test.lock.json history)
python -m src.authorship.train --epochs 2
python -m src.authorship.train --epochs 2 --variant short
python -m src.authorship.calibrate --max-false-ai 0.008 --max-false-human 0.08
python -m src.authorship.benchmark --rerun-reason "..."   # v2 test already run once
```

Records (no text): `data/benchmark/v2/` holds the build record, lock, training records for both models, calibration and benchmark. `models/registry.json` holds the registry and all three locked runs.
