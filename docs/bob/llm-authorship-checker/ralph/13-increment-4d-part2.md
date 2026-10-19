# wreck-it-ralph report: R12 part 2 (fixes, run artifacts, tricks on the 4d model)

Verdict: **PASS** for reading the lock. There are no Critical or High findings. **Claim 9 (speed) is still pending**, and it must run with the GPU hidden by `CUDA_VISIBLE_DEVICES=-1` (R13-8).
- **Claim 7** is met (addendum).
- **Tricks on the 4d model:** scored. None of the 15 tricks turned Likely AI into Likely human in 352 tricked CPU scorings (382 with the originals). The details are in the addendum, and R13-2 is downgraded to Low.

Summary: tests **511/511** passed with the GPU actually hidden (`CUDA_VISIBLE_DEVICES=-1`, `torch.cuda.is_available()` asserted false, 42.9 s). The first run in this review, 507/507, didn't hide the GPU (R13-8). No end-to-end examples were given. Findings **C0/H0/M2/L7**, with R13-7 fixed since.

This report was first written at `b158631` (the section up to "Findings"). The addendum at the end covers `bba3ae1`, `5d96ae3` and `95e047d`, and the trick scoring.

**Range.** Part 1's HEAD was `8740ef7`. The range here is `8740ef7..b158631`:
- the Amendment 13 fixes, `e145a0f` to `cb465aa`;
- the corpus v4 rebuild, `f4e7f4e`, lock `549b3f10`;
- the training records and calibration, `b158631`.

HEAD didn't move during the review. The working tree stayed clean apart from this report.

**How the rules were kept.**
- **Files:** nothing in the repo was edited or committed except this report. Scratch scripts and outputs are in `scratchpad/r13/`.
- **Commands not run:** headtohead, benchmark, training, calibration, the corpus build, generation and the speed check.
- **The running GPU job:** its process, log and outputs were not touched. The log was only read (tail).
- **GPU:** every Python process ran with `CUDA_VISIBLE_DEVICES=` set. **Correction (addendum):** on this Windows machine an empty value doesn't hide the GPU.
  - Of this review's processes, only the first test run imported torch. Its tiny-encoder tests could reach CUDA while the optional-step GPU job was running, for about 100 s around 20:45.
  - The trick scoring and the second test run used `CUDA_VISIBLE_DEVICES=-1`, and asserted that CUDA was unavailable.
- **Files not opened:** `x`, `inspect_wb.py`, the root `.xlsx` sheets and `dataset_full.csv`. No rater is named.
- **Model folders:** read only. I listed them, hashed them through `registry.verify_calibrated`, and read `run.json`, `checks.json`, `config.json` and the two `calibration_scores_*.npz` caches (validation scores only). Nothing was moved or written.
- **Locked text.** No locked text was read, printed or scored.
  - Test-split facts come from `manifest.parquet` (no text) and `build_record.json`.
  - Every text-level check streamed `corpus.parquet` with a buffered reader (`buffer_size=1 MiB`, `pre_buffer=False`, 2,000–4,000-row batches, column selection) and kept only `train` and `val` rows.
  - Peak memory per streaming process was 185–700 MB.
- **RAM.** Available memory was 0.9–3.4 GB for the whole review, because the GPU job holds about 10.7 GB of private memory.
  - The test suite and the OOD-cache check (the two largest processes) ran under a guard (`scratchpad/r13/guard.py`) that stops a process below 1.5 GB available. Neither tripped.
  - The test suite peaked at 1.05 GB RSS, with 1.79 GB lowest available.
  - **The trick scoring was not run.** The 4d service on the CPU needs fp32 weights (1.74 GB), the main OOD bank (0.25 GB) and torch, about 2.5 GB in all. With at most 3.4 GB available, it would have taken free memory below 1.5 GB.
  - The model-free half of that step was run (R13-2). The scoring script is ready: `scratchpad/r13/t_model.py` with `pool.parquet`. It needs about 4 GB available.

## Status of the R12 findings

| R12 | Sev | Fix or ruling (Amendment 13) | Status | Evidence |
|---|---|---|---|---|
| 1 | High | The matcher shares the cap over word-count bands 25% wide (`_common.py:122-162`). MAGA and DACTYL call it with `cap=` (`e145a0f`). | **closed** | Rebuilt corpus, train+val (`c2_surface.py`):<br>- **MAGA:** within-domain word-count AUC 0.511 weighted (worst: wikipedia 0.567).<br>- **Coarse bands, AI / human:** amazon_reviews 50–99 words 292/338, 200–399 words 280/255 (part 1, all splits: 103/744 and 501/209).<br>- **DACTYL:** 0.45–0.51 per domain.<br>- **DACTYL tweets:** 0.394 on the policy word count, but 0.508 on whitespace words and 0.509 on the view. The gap comes from URLs, which the policy count splits into several words (R13-4).<br>- **Probes:** MAGA's per-source probe 0.706 in the shared probe, 0.792 on its own (part 1: 0.827). |
| 2 | High | The run stops unsealed on `DETECTOR_ERROR`, `DETECTOR_INVALID_RESPONSE` or `DETECTOR_REPORTED_ERROR` (`headtohead.py:210-218`). | **closed** | `test_a_detector_error_stops_the_run_unsealed` passes, and the check runs before `gate()`. Small gap: `DETECTOR_INVALID_EVIDENCE` isn't in the list (R13-7). |
| 3 | Medium | `trims.py`: preambles and news headers are trimmed in MAGA, MIRAGE, OpenTuringBench and the generated answers. | **closed for the named sources.** The same class remains elsewhere (R13-3). | Train+val, format4 view:<br>- **"Here's the … version/body/article:" preamble:** MAGA AI 0.06% plain and 0.02% aligned (part 1: 13.9% and 9.3%). MIRAGE AI 0.00% (part 1: 5.3%).<br>- **OpenTuringBench:** date/category lines gone. Labelled lines ("By: …") are left on 0.70% of rows (part 1: news template 26.8%). |
| 4 | Medium | New `format4` view; other scripts' word separators are read back; other punctuation glued between Latin letters counts toward abstaining. Ruling: word counts stay unchanged. | **partly closed. Model scoring still open** (R13-2) | Model-free, current code (`t_free.py`):<br>- **Now closed:** small and two-em dashes, ◆ → ✅ 🔹 bullets, katakana, word-separator, runic and two-dot separators (read back, with a note), line-start "a)" markers. U+3001, U+0589, U+2E33, U+10FB, U+1802, U+2044, U+A789 and U+22C5 separators abstain.<br>- **Still silent:** hyphen-class and ASCII punctuation separators, attached `--`, and circled numbers, which become digits before the view sees them. |
| 5 | Medium | Amendment 10's rationale corrected; the held-out sites are reported one by one. | **closed** | `benchmark.py:158-161`, and a test. Test humans per site are 372–375. |
| 6 | Medium | `verify_calibrated` runs before the `started` entry is saved (`headtohead.py:177-192`). | **closed** | On the committed record: `verify_calibrated` gives `(True, 'files verified')`, and `verify` gives `(False, 'locked benchmark not passed')`, as expected. |
| 7 | Medium | Ruling: the rule restated as Amendment 2 wrote it (part 1's first fix option). | **closed by ruling** | The rebuild gives the same counts: 7,746 MIRAGE test humans and 3 RAID AI rows share a group with v1–v3 stress rows. The id and source-record matches are 0 (`c1_ids.py`). |
| 8 | Low | Recorded | accepted | — |
| 9 | Low | Recorded | accepted | HC3 test: 402 AI, 359 humans. It is gated, and allows 0 false AI (`upper_bound(1,359)=0.0131`). |
| 10 | Low | Recorded | accepted | Rebuild:<br>- v3 stress 41,117 of 41,538 kept; the 421 missing rows are MIRAGE polish and rewrite rows.<br>- LEDE 4,996 of 4,999.<br>- stackexchange_heldout 7,055 of 7,064 loaded. |
| 11 | Low | Recorded | accepted | Stack Exchange, AI vs human:<br>- under 100 words: 53.6% vs 24.2%;<br>- emoji: 5.0% vs 0.0%;<br>- all lower case: 4.9% vs 0.1%. |
| 12 | Low | Recorded | accepted | — |
| 13 | Low | Refusals, bracketed word counts and CJK switches are dropped (`generate.py:65-73,198-203`). | **mostly closed.** A few remain (R13-5). | Build log:<br>- refusals 23, bracketed counts 7, CJK 8. That matches the amendment.<br>- About 8 train/val rows remain. |
| 14 | Low | Recorded | accepted | — |
| 15 | Low | Pooled figure without LEDE (`headtohead.py:114-117`) | closed | A test. |
| 16 | Low | `excluded_by_source` (`headtohead.py:103-113`) | closed | A test. |
| 17 | Low | A v4 4d record is required (`headtohead.py:160-162`) | closed | A test. The committed record is `corpus_version: v4`. |
| 18 | Low | Validation-ids digest; defaults 0.8% / 8% and `fullbatch`; `detection/__init__.py` added to the digest. Rulings on `pipeline.py` and the weight normalisation. | **closed, except the last bullet** (R13-6) | Both digests are present and recomputed (claim 5). The bullet about keeping today's models isn't addressed. |

## Claims checked

| # | Claim | Result | Evidence |
|---|---|---|---|
| 1 | The v4 locked test is fresh (rebuilt corpus) | **met** (R12-7 is a ruling) | 41,173 tested rows (`c1_ids.py`, manifests only):<br>- **Earlier-split matches:** sample id 0; source record 0; group 0 in fitted or tested splits (7,749 in stress, R12-7).<br>- **Stress:** no stress row in an earlier fitted or tested group; no core row whose id was report-only.<br>- **Held-out sources:** no row in train or val.<br>- **Audit:** `passed: true`, 0 errors, `retired_by_audit_check: 0`.<br>- **Build timing:** it ran on committed code. `cb465aa` at 03:56:19, the build started 03:56:30 with `--relock` and a recorded reason, and ended 04:45:55. Main training started 04:47:12. |
| 1c | Test rows per source | **met** (known differences recorded) | **Test, AI / human:**<br>- arena140k 2,993 / 0; dactyl 2,057 / 1,975; hc3 402 / 359; lede 4,996 / 0; maga 2,565 / 2,541; mage 921 / 4,135; mirage 102 / 7,787;<br>- no_robots 0 / 553; openturingbench 1,315 / 0; raid 11 / 0; stackexchange 6 / 0; stackexchange_heldout 2,398 / 2,991; wasitai2026 746 / 389.<br>**Unseen:** mage 30, openturingbench 235, stackexchange_heldout 1,666.<br>**Gated false AI:** dactyl, hc3, maga, mage, mirage, no_robots, stackexchange_heldout, wasitai2026.<br>**Stack Exchange's 6 test rows** are AI answers in 3 new groups with no human answer, too few to matter. |
| 2 | No shortcut left | **met for the probes and length matching. Small residual cues (R13-3, R13-4)** | **Shortcut probe:** only Stack Exchange is over the line (own 0.8915; Amendment 10 ruling); overall 0.684.<br>**AI-only probe:** Arena 0.837, OpenTuringBench 0.830.<br>**Length matching:** see R12-1 above.<br>**Surface scan:** `c2_surface.out` and `c2_openers.out`. |
| 3a | `format4` is idempotent and leaves prose alone | **met, with minor prose edits** | 0 non-idempotent cases in 40,000 fuzz strings, 20,000 of them with format4's markers (`c3_format4.py`).<br>**Prose edits:** "Options (a) and (b)" → "Options and"; a line-start "©", "±" or emoji is dropped; "3 − 2" → "3 2". |
| 3b | Every listed bullet, marker and dash form goes | **met for the listed forms** | See R12-4 above. By design, inline ★ → ◇ and an inline "b)" stay. |
| 3c | A formatting or character trick can't turn Likely AI into Likely human | **met on the scored sample (addendum).** Silent view changes remain (R13-2, Low) | Scored in the addendum: 0 of 352 tricked texts became Likely human. Originally model-free only. Scoring on the model is pending. |
| 5 | Training read only train and validation; checks, early stopping, best check, merge, digests | **met** | **Digests:** all four recomputed from the corpus match the records (`c5_digests.py`):<br>- main train 191,785 and validation 23,609;<br>- short train 184,329 (139,121 excerpts) and validation 22,555.<br>**Checks and stopping:**<br>- Main: checks 0.1 / 0.2 / 0.3 (TPR 0.697 / 0.643 / 0.611). It stopped after 2 stale checks and kept 0.1.<br>- Short: best 0.3, then 1.1 (0.731). It stopped after 1.3 and kept 1.1.<br>- `checks.json` and `run.json` in the model folders equal the records.<br>**Merge:** `merged-0.1` and `merged-1.1` hold fp32 weights (1,740,304,440 bytes; `"dtype": "float32"`), and their hashes equal the records.<br>**Calibration cache:** it is keyed to the merged artifacts' hashes (`f2a9396eacc8`, `b9455479b255`).<br>**Observation:** the chosen main model saw a quarter epoch, and detection fell 8.5 points over the next two checks. The rule was followed as pre-registered. |
| 6 | Calibration targets, routing, precision, seal | **met** | **Recomputed from the cached validation scores** (`c6_recalc.py`; 69,998 rows, 28,593 of them excerpts; ids in the recorded order):<br>- both Platt maps are identical;<br>- all 4 main and 2 short band thresholds are bit-identical;<br>- routing: main coverage 0.2518 / 0.337, short 0.3498 / 0.5025, so the specialist serves 25–99 words;<br>- the registry's policy bands equal the recomputed ones.<br>**Registry:** precision fp16 for both models; view fingerprint current; scoring-code digest `f845abba` current; `verify_calibrated` true; peak inference 3.07 GB. **Risk: R13-1.** |
| 7 | Optional steps decided on validation only | **met (addendum)**. At first writing: pending, artifact not yet written | No `stack.json` or `registers.json` yet. The GPU job is at "main: scoring 30320 validation texts". The code was checked in part 1. |
| 9 | Speed and memory | **pending: artifact not yet written** | No `speed.json` yet. |
| 10 | Amendments came before the results they could affect | **met** | **Amendment 13:** committed 02:25:34, before the rebuild (03:56) and training (04:47). Its one later edit (`cb465aa`, 03:56:19) records the first rebuild and came before the final build.<br>**Training and calibration results:** none is recorded as an amendment yet, only in `b158631`'s message. The optional-step verdicts are still to come. |
| 11 | The suite passes with no GPU and no model files | **met (addendum)** | 511 passed with `CUDA_VISIBLE_DEVICES=-1` and CUDA asserted unavailable. The first run (507 passed) didn't hide the GPU (R13-8). |

## Findings

### R13-1: The gate's false-AI bound on No Robots allows one false AI in 553 humans; the 4d system makes 3 in 584 on validation, where no threshold ever constrains No Robots

- **Severity:** Medium
- **Category:** design risk to the one-shot outcome. This isn't a code defect.
- **Location:**
  - `src/authorship/calibration.py:108-116`: a slice constrains a band only with ≥300 humans *in that band*.
  - `src/authorship/benchmark.py:108-118`: the gate bounds each *source* over all bands, from 300 humans.
  - The v4 `test|no_robots` slice has 553 humans.
- **Repro:** `python scratchpad/r13/c6_recalc.py`, then `c6_ood.py`.
  - These apply the registry's routed policy (short model for 25–99 words, main for 100+) to the cached calibration scores of the natural validation rows.
  - `upper_bound(k, n)` gives the passing counts.
- **Expected vs actual:** calibration holds each constraining slice to 0.8% false AI. No Robots is never a constraining slice: it is in `unverified_slices` in all four bands. Actual:
  ```
  routed system, validation natural rows   humans  false_ai  upper95
  no_robots                                   584         3   0.0132   (bands 100: 2, 200: 1; all in-distribution)
  oasst1 (no test rows)                       518         7   0.0252
  beemo  (no test rows)                       374         6   0.0314
  test slice sizes: no_robots 553 -> max passing false AI 1; wasitai2026 389 -> 0; hc3 359 -> 0
  Poisson P(<= 1 false AI in 553 | 3/584) = 0.224
  ```
- **Evidence:**
  - At the validation rate, the 4d set fails the gate on No Robots about 3 times in 4.
  - WasItAI (389 humans, no validation rows) must have zero false AI. That is a pre-registered known risk.
  - HC3 must have zero too, but its validation rate is 0 in 1,535.
  - The other gated sources have room: mage 2 of 4,712, maga 3 of 1,933, dactyl 1 of 1,478 and mirage 1 of 763 on validation, against allowances of 11–50.
  - Today's models face the same rows. v5 scored 3 of 1,003 on v3's No Robots, which on 553 rows would give about a 50% chance of more than one. The outcome table could therefore turn on one or two No Robots texts.
  - Nothing in the pre-registration was broken: the mismatch between band-level calibration and source-level gating dates from v1. The sponsor should see it before the lock is spent.
- **Confidence:** medium. It rests on the validation rate carrying over to fresh test rows of the same source and draw.

### R13-2: Character and format tricks the service still passes silently; the tricks aren't yet scored on the 4d model

- **Severity:** Low. It was Medium, provisional, at first writing; scoring on the 4d models found no flip to Likely human (addendum).
- **Category:** robustness / evasion (R12-4 residual)
- **Location:**
  - `src/detection/obfuscation.py:156-158`: only the listed separators are read back.
  - `obfuscation.py:208-217`: ASCII punctuation, and the dash and quote classes Pd, Pi and Pf, are exempt from the residual count.
  - `src/corpus/text.py:104-105`: `_DASHES4` keeps word hyphens, including those NFKC folds to `-`.
  - `text.py:106`: `_CIRCLED4`.
- **Repro:** `PYTHONIOENCODING=utf-8 python scratchpad/r13/t_free.py`. It applies 68 tricks to part 1's 77-word text and compares `deobfuscate` plus the `format4` view with the original.
- **Expected vs actual:** each trick should be read back, or abstain, or at least carry a note. Actual, with `changed=False` and `abstain=False` in every row:
  ```
  hyphen U+2010 / U+2011 / U+FE63 / U+FF0D / ASCII "-" for every space   view_words=1   'Regular-exercise-offers-many-...'
  U+2010 or ASCII "-" for every other space                               view_words=39
  ASCII "." "/" "+" "," and fullwidth ，．  for every space                view_words=1
  ", " -> "--" (attached double hyphen)                                   view_words=71 of 77
  circled numbers per sentence (①②…)                                     view '1 Regular exercise ... 2 It strengthens ...'
  inline "1." / "b)" markers, line-start "i."                            kept in the view (82-83 words)
  numbered (1)/1) markers                                                  view identical, policy_words 77 -> 83 (band ruling)
  ```
- **Evidence:**
  - **Circled numbers:** Amendment 13 says `format4` drops them, and it does on raw text, which is what training saw. In the service, `deobfuscate` folds ① to "1" first (NFKC, not counted as a mapping), so the model reads digit markers it never saw in training. That is a train/serve skew on a class the fix meant to remove.
  - **Separator glue:** hyphen-class and ASCII punctuation separators give the model one glued token string, with no note. The part 1 middle-dot class now reads back or abstains, but these neighbours don't.
  - **Readability:** replacing every space is conspicuous to a reader. The attached `--` for a comma or dash is not: it is how people type an em dash, and it glues 6 of 77 words here.
  - **Open question:** whether any of these flips Likely AI to Likely human on the 4d models is what part 2 was asked to measure. It wasn't run (see "How the rules were kept").
  - **Ready to run:** `t_model.py` scores 30 hash-picked validation AI texts from `pool.parquet` under 15 representative tricks. It skips any variant whose model input and band equal the original's.
    - 20 texts of 100–170 words go to the main model, and 10 of 88–99 words to the short model. All were pre-selected as Likely AI from the calibration cache.
    - The numbered-marker tricks push the short texts past 100 words, so they reach the main model.
    - That is about 390 scorings, one model loaded at a time.
- **Confidence:** high on the model-free behaviour. The flip rate is unknown.

### R13-3: The self-announcing and opener cues Amendments 8, 9 and 13 trimmed remain in sources the trims don't cover, and the "Okay, so" trim leaves a "So" opener

- **Severity:** Low
- **Category:** data integrity (shortcut)
- **Location:**
  - `trims.py` is applied only by `maga.py`, `mirage.py`, `openturingbench.py` and `generate.trim_opener`.
  - `dactyl.py`, `raid.py` and `beemo.py` don't trim, and `trim_header` isn't applied to MAGA.
  - `generate._OKAY` trims "Okay," and leaves the following "so".
- **Repro:** `python scratchpad/r13/c2_openers.py`, and the first-word count in this review. Train+val only.
- **Expected vs actual:** claim 2 asks for no preamble or opener cue separating the labels. Actual, AI vs human rows of the same source:

  | Cue | AI | Human |
  |---|---|---|
  | "Here's a … :" preamble, DACTYL ("Here's a tweet in the style of the dataset:") | 1.24% | 0.13% |
  | Same, RAID ("Sure! Here is the body of a plot summary …:") | 2.42% | 0.00% |
  | Same, BEEMO (v3 source, no test rows) | 18.3% | 0.87% |
  | Labelled "Title:" / "Headline:" lines, MAGA plain / aligned | 0.94% / 1.49% | 0.00% |
  | Same, DACTYL | 0.78% | 0.00% |
  | First word "Okay", MAGA plain / aligned | 3.84% / 9.39% | 0.07% |
  | First word "Okay", Arena (AI only) | 6.48% | pooled human 0.14% |
  | First word "So", Stack Exchange / Dolly / OpenAssistant generated answers | 2.25% / 3.43% / 2.21% | 0.10% / 0.08% / 0.20% |
  | First word "So", Stack Exchange humanized rewrites | 44.6% | 0.10% |

- **Evidence:**
  - Pooled, "Okay" opens 1.23% of AI rows and 0.14% of human rows. That is the gap Amendment 9 called a shortcut ("rater comments … often start with 'Ok so'"), now carried by MAGA and Arena rather than the generated answers.
  - The "So" openers match Amendment 9's own example ("Hey! Okay, so …" → "So …").
  - Each cue is small in share, but nearly label-pure. The DACTYL and RAID cues can inflate detection on their gated test slices by about 1–2 points.
- **Confidence:** high on the counts, low on the effect on the model.

### R13-4: DACTYL tweets separate by URLs and hashtags

- **Severity:** Low
- **Category:** data integrity (shortcut)
- **Location:** `src/corpus/sources/dactyl.py`: the tweets domain, length-matched on whitespace words only.
- **Repro:** `c2_openers.py` (train+val DACTYL tweets, 3,203 rows).
- **Evidence:**
  ```
            url    mention  hashtag
  ai     0.2230   0.1398   0.8907
  human  0.6981   0.1224   0.4589
  word-count AUC (AI longer): whitespace 0.508, view 0.509, policy count 0.394
  ```
  - URLs and hashtags stay in the view.
  - The policy count differs because it splits URLs into several words. A tweet with a URL therefore reads longer to the router and the band than to the length matcher.
  - The cue is confined to one domain of a source outside the shared ground, and rater comments rarely carry hashtags.
- **Confidence:** high

### R13-5: A few generation artifacts get past the Amendment 13 filters

- **Severity:** Low
- **Category:** data quality (R12-13 residual)
- **Location:**
  - `src/corpus/generate.py:68-69`: the refusal list needs a verb such as help, assist or provide right after "I'm not able to".
  - `generate.py:194`: the opening length note is looked for only in the first 120 characters.
- **Repro:** `python scratchpad/r13/c4_artifacts.py` (train+val AI rows of Stack Exchange, Dolly and OpenAssistant).
- **Evidence:**
  - **Refusals:** 3 OpenAssistant ones remain ("I'm not able to browse the internet in real-time …", "I cannot provide the specific fixture …").
  - **Length echoes past character 120:** at least 5 ("… lands in the Intermountain West, aiming for approximately 200 words: The vast …").
  - **Totals:** 19 rows mention a word count, several legitimately ("a vocabulary of 50 words").
  - All counts are under 0.5% of those sources.
- **Confidence:** high

### R13-6: Triage isn't recorded where the Ralph procedure puts it, and the "keep today's models" path still leaves the app closed

- **Severity:** Low
- **Category:** process integrity
- **Location:**
  - `ralph/12-increment-4d.md` has no triage lines.
  - `debt-ledger.md` has no R12 entries; it was last changed in `fadd8c3`.
  - `models/registry.json` holds the unsealed 4d record.
  - `headtohead.run` seals that record with `passed=False` on any outcome other than `register_4d`.
- **Evidence:**
  - Amendment 13 serves as the triage, but `wreck-it-ralph.md` asks for a triage line under each finding. Its Medium "recorded" items (R12-7) went in as rulings, not debt.
  - R12-18's last bullet isn't answered. If the outcome is "keep today's models" or "both fail", the registry ends up with a failed 4d record, and the app abstains on every text until v5's record is restored by hand.
- **Confidence:** high

### R13-7: Small robustness gaps in the locked run (fixed in `bba3ae1`; see the addendum)

- **Severity:** Low
- **Category:** reliability
- **Location:**
  - `src/authorship/headtohead.py:62`: `DETECTOR_FAILURES` omits `DETECTOR_INVALID_EVIDENCE` (`service.py:160`).
  - `headtohead.py:210`: only the general frame is checked. Excerpt, in-house and diagnostic decisions aren't; they are report-only.
  - `benchmark.py:174` and `headtohead.py:169`: the corpus is read with `pd.read_parquet(..., filters=...)`. The v4 corpus is one row group, so each read decodes the whole text column. The diagnostic read happens after the `started` entry, with the model loaded.
- **Evidence:**
  - Reading only the val split this way (`pyarrow.dataset`, five columns) peaked at **1.96 GB** RSS. The same rows through a buffered `iter_batches` reader peaked at **0.21 GB**.
  - This machine rebooted after an unclean shutdown today, and has run at under 3.5 GB available all evening. A memory failure after the `started` entry spends the lock, and a rerun then needs a reason.
- **Confidence:** high on the measurements, low on the likelihood.

## Addendum (2026-09-30, after the optional-step GPU job ended)

**Scope.** HEAD moved to `95e047d`. The addendum covers:
- `bba3ae1`: the R13-7 fix, and the commit of this report's first version;
- `5d96ae3`: the stack's scoring order;
- `95e047d`: `registers.json`, `stack.json`, Amendment 14, Debt Ledger D14–D20, and the R12 triage section;
- the trick scoring on the 4d models;
- a second run of the test suite.

The working tree holds only this report's edits. Nothing was committed.

**Rules for this part.** No GPU, with `CUDA_VISIBLE_DEVICES=-1` and CUDA asserted unavailable. One model at a time. Validation texts only. Free memory was kept above 1 GB: the guard's floor was 1.0 GB, and the lowest point was 1.08 GB.

### Tricks scored on the 4d models (R12-4's open question)

**Setup.**
- **Service:** built with `registry.service_from_record(record, validated=True)` from the committed 4d record, as the app does. It ran on the CPU in fp32; the script printed `model device cpu torch.float32`.
- **One model at a time:** the unused sub-detector was replaced by a stub that raises, and it was never called.
- **Texts:** 30 validation AI texts, in hash order with seed `r13:` and at most 3 per source.
  - 20 of 100–170 words, scored by the main model.
  - 10 of 88–99 words, scored by the short model.
  - All 30 originals scored Likely AI.
- **Tricks:** 15, one for each group of tricks that gives the same model input.
  - A variant whose view, band and route equal the original's is skipped, because identical input gives an identical decision. There were 98 such variants.
- **Volume:** 352 tricked scorings plus 30 originals, taking 65 s and 190 s.
- **Memory:** peak RSS 3.90 GB.
- **Files:** script `scratchpad/r13/t_model.py`; results `scored_*.parquet` and `trick_results.csv`.

| Trick | Scored | Stayed Likely AI | To Likely human | To Inconclusive (score in the middle band, no note) | Abstained (out of distribution) | Disguise note | Band or route changed |
|---|---|---|---|---|---|---|---|
| ", " → em dash (the view reads it as a space) | 30 | 16 | **0** | 14 | 0 | 0 | 0 |
| Circled numbers per sentence | 30 | 20 | **0** | 10 | 0 | 0 | 5 |
| Inline "a)" "b)" markers | 30 | 28 | **0** | 2 | 0 | 0 | 5 |
| Inline "1." markers | 30 | 29 | **0** | 1 | 0 | 0 | 5 |
| Line-start "i." "ii." markers | 30 | 23 | **0** | 7 | 0 | 0 | 5 |
| All lower case | 30 | 27 | **0** | 3 | 0 | 0 | 0 |
| U+2010 hyphen for every space | 30 | 24 | **0** | 4 | 2 | 0 | 0 |
| U+2010 hyphen for every other space | 30 | 20 | **0** | 7 | 3 | 0 | 0 |
| ASCII "/" for every other space | 30 | 22 | **0** | 5 | 3 | 0 | 0 |
| ASCII "." for every space | 30 | 15 | **0** | 3 | 12 | 0 | 0 |
| ", " → attached "--" | 30 | 29 | **0** | 1 | 0 | 0 | 0 |
| Line-start "1." markers | 6 (24 same input) | 6 | **0** | 0 | 0 | 0 | 5 |
| Line-start "a)" markers | 6 (24 same input) | 6 | **0** | 0 | 0 | 0 | 5 |
| Inline "(1)" markers | 5 (25 same input) | 5 | **0** | 0 | 0 | 0 | 5 |
| Inline "1)" markers | 5 (25 same input) | 5 | **0** | 0 | 0 | 0 | 5 |
| **Total** | **352** | **275** | **0** | **57** | **20** | **0** | **40** |

**Results.**
- **No trick turned Likely AI into Likely human,** with or without a note.
  - The zero-count upper bound is 9.5% per trick (0 of 30) and 0.85% pooled (0 of 352).
  - Claim 3c is met on this sample, and R13-2 is downgraded to Low.
- **Silent de-labelling remains.** All 57 moves to Inconclusive carry no disguise note:
  - **Circled numbers, 10 of 30.** This is the train/serve skew: the service folds ① to "1" before `format4` can drop it.
  - **Line-start roman numerals, 7 of 30.**
  - **U+2010 in every other space, 7 of 30.**
  - **Commas turned into em dashes, 14 of 30.** The view reads the dashes as spaces, so this is a comma-removal style edit.
  - Every out-of-distribution abstention carries the `OUT_OF_DISTRIBUTION` evidence.
- **Band and route changes.** On the 5 short-pool texts of 94–99 words, the marker tricks raise the policy count to 100–107, which moves them from the short model (50–99 band) to the main model (100–199 band).
  - Of those 40 scorings, 38 stay Likely AI.
  - 2 become Inconclusive: scores of 0.934 and 0.926, against the main band's 0.9406.
  - None becomes Likely human. Amendment 13's word-count ruling (D18) holds on this sample.

### Claim 7: the optional steps were decided on validation only. **Met**

**`registers.json`: measured, not adopted.**
- **Data:** the register classifier fit on 99,894 train rows. The cell bars and the measurement use validation only (`locked_split_reads: 0`).
- **Cell rule:** no cell under 600 humans has its own bar. Five cells with at least 600 humans kept the pooled bar because their own calibration wasn't supported (`registers.py:103-109`).
- **Result:** gain 9.12 points (40.52% → 49.64%).
  - 9 of 20 cells are over target. The worst is `qa_forum` at 100–199 words, with 36 false AI in 997 humans, upper bound 4.74%.
  - 5 of 47 band × source slices are also over target.
- **Verdict:** the rule needs every cell within target, so not adopting is correct. The figures match Amendment 14.

**`stack.json`: a decision record, not a measurement.**
- **Basis checked:**
  - **Memory:** fp16 weights come to 0.87 + 0.87 + 0.99 + 0.99 = 3.72 GB, over the 3.5 GB peak limit before any activations.
  - **Sheet time:** `bakeoff-gpu.log` shows the pair scoring 9,807 validation texts from 20:08:05 to 20:51:34 (43.5 min), which is 3.76 texts a second. A 3,000-text sheet then takes about 798 s against 120 s, 6.6 times the limit.
- **Why that is enough:** the pre-registration adopts the stack only if it adds 2 points *and* the limits hold. The sheet limit fails by a wide margin whatever the gain. Leaving the gain unmeasured costs a reported figure, not the verdict.
- **Caveat:** the rate comes from Increment 3's texts, not this validation set.

**`5d96ae3`.** It changes only the order of work. Each classifier's raw scores come from the same `raw_and_embeddings` call, and the classifier is then freed before the zero-shot pair loads. `gpu_limits` still loads every model at once. By construction the scores are unchanged, and a test was added.

**Timing.** Amendment 14 was committed at 23:28, before any locked text was read. The verdicts are recorded there.

### Earlier findings after these commits
- **R13-7: closed** (`bba3ae1`).
  - `DETECTOR_INVALID_EVIDENCE` now stops the run.
  - The diagnostic rows are read before the `started` entry is saved.
  - Both corpus reads still decode the whole row group, but they now happen before the run is recorded, so a memory failure there no longer spends the lock.
- **R13-6: partly closed.**
  - The R12 report has a triage section, and D16–D20 are in the Debt Ledger.
  - The "keep today's models" registry path is still unaddressed: that outcome leaves a failed 4d record in `models/registry.json`, and the app stays closed.
- **R13-1, R13-3, R13-4, R13-5:** recorded in Amendment 14 or the Debt Ledger, unchanged.

### R13-8: `CUDA_VISIBLE_DEVICES=` (empty) doesn't hide the GPU on this machine, so the documented CPU speed check would time the GPU again

- **Severity:** Medium
- **Category:** reliability / process (claim 9 and claim 11)
- **Location:**
  - `src/authorship/speed.py:4`: the documented CPU command is `CUDA_VISIBLE_DEVICES= python -m src.authorship.speed`.
  - `speed.py:88`: `device = "cuda" if torch.cuda.is_available() else "cpu"`.
  - `speed.py`'s `run()` merges the results into `speed.json` under that device.
  - `docs/superpowers/plans/2026-09-28-increment-4d-head-to-head.md:902` gives the PowerShell form, `$env:CUDA_VISIBLE_DEVICES=''`.
  - The Ralph prompts' test command uses the same empty form.
- **Repro:**
  ```
  CUDA_VISIBLE_DEVICES= python -c "import os,torch; print(repr(os.environ.get('CUDA_VISIBLE_DEVICES')), torch.cuda.is_available())"
  -> '' True
  CUDA_VISIBLE_DEVICES=-1 python -c "..."          -> '-1' False
  subprocess with env CUDA_VISIBLE_DEVICES=''       -> '' True
  PowerShell: $env:CUDA_VISIBLE_DEVICES=''; python -c "import os; print('CUDA_VISIBLE_DEVICES' in os.environ)"   -> False (removed)
  ```
- **Expected vs actual:** the CPU run should time the CPU and write `speed.json`'s `cpu` entry. Actually, run as documented on this machine, it sees CUDA and times the GPU again.
  - It overwrites the `cuda` entry and writes no `cpu` entry.
  - The pre-registered CPU limit (single text ≤ 20 s at p95; a model set that breaks a limit can't be registered) would go unmeasured.
  - The plan expects `cpu.passed`, so the gap would probably be noticed, but only after the run.
- **Evidence:**
  - The same empty form was used for the "suite passes with no GPU" claim. This review's first test run (507 passed) had CUDA visible: the tiny-encoder tests pick `cuda` when it is available (`neural.py:19`), and that run took 104 s against 42.9 s for the hidden run. Part 1's 430-test run likely had the same exposure.
  - The rerun with `CUDA_VISIBLE_DEVICES=-1` and CUDA asserted unavailable gives **511 passed**, so claim 11 now holds as stated.
- **Fix:**
  - Run the speed check's CPU pass with `CUDA_VISIBLE_DEVICES=-1`, or give `speed.py` a `--device cpu` flag that asserts CUDA is unused.
  - Change the documented commands to match.
- **Confidence:** high

### R13-9: Amendment 14 calls the merged artifacts fp16

- **Severity:** Low
- **Category:** record accuracy
- **Location:** Amendment 14: "Check 0.1 was kept and its LoRA merged (`merged-0.1`, fp16)", and the same for `merged-1.1`.
- **Evidence:**
  - Both merged artifacts are fp32 on disk: `model.safetensors` is 1,740,304,440 bytes, and `config.json` has `"dtype": "float32"`.
  - fp16 is the recorded *inference precision* for the GPU. The CPU runs these fp32 weights.
  - The registry and the detector behave correctly. Only the amendment's wording is off.
- **Confidence:** high
