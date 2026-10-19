# wreck-it-ralph report: the R09 fixes, v5/short-v3 retraining and the recorded v3 rerun

Verdict: **FAIL**

Summary: tests **262/262**, run three times: as is, with the GPU hidden (`CUDA_VISIBLE_DEVICES=-1`), and with an empty `CORPUS_DIR`. No end-to-end examples were given. Findings **C0/H1/M2/L5**.

Range: `4823d02^..HEAD`. HEAD moved during the review, from `fa7aa30` to `5a1cf4d` (a docs commit at 06:34:54 that rewrites the 4c report, roadmap and README around run 2). That commit is in scope too.

How the rules were kept:
- Nothing in the repo was edited or committed except this report. Scratch scripts and outputs are in the session scratchpad, under `scratchpad/r10/`.
- The benchmark, training, calibration and the corpus build were not run. No file named `x` or `inspect_wb.py`, no root `.xlsx` and no `dataset_full.csv` was opened.
- `corpus.parquet` was read only through `load_fit_data` (column selection and a train/val split filter). No locked text was read, hashed or scored. Locked-split facts come from `manifest.parquet` (no text column), `test.lock.json` and the two benchmark reports.
- **Model use:** v5 and short-v3, loaded through the committed registry, which verifies. All scored text was v3 **train/validation**:
  - 2,099 validation human and 687 validation AI texts, each in 7 list formats;
  - 18 validation excerpts, with and without their list markers;
  - 149 validation AI texts that the service labels Likely AI, each under 15 edits.
- **Disclosure:** my first formatting pass used `CUDA_VISIBLE_DEVICES=""`. On Windows an empty value doesn't hide the GPU (`torch.cuda.is_available()` was `True`), so that pass scored validation text on the GPU. Its output was discarded. Every figure below comes from the CPU reruns, where each script asserts `torch.cuda.is_available() is False`.
- **Builder files:** I read the builder's task outputs (for the timeline) and compared against `scratchpad/manifest_v3_before_relock.parquet`.

## Claims checked

| # | Claim | Result | Evidence |
|---|---|---|---|
| 1 | R09-1: excerpts are cut from the format2 view; markers no longer reach short-model training or calibration; bullets and free-standing dashes go, word hyphens stay; the view is idempotent | **met for the training shortcut and the R09 formats. Residuals: other glyphs (R10-2). Idempotence not met** | **Training data:** rebuilt from train/val with the committed code, short-v3's frame reproduces its record exactly: 111,487 rows, 81,809 excerpts, 13,436 validation rows, `ecc144af…`. Main train ids give `213465df…`.<br>**Markers in short-v3 training excerpts**, two or more inline `N.` markers, AI vs human: 0.32% vs 0.46% overall, RAID 0.1 vs 0.3. R09 had RAID 8.7 vs 1.0.<br>**Reliance:** only 18 validation excerpts still carry two or more markers (R09: 275). Removing them moves AI from 0.889 to 0.845 and changes no AI decision (0/0/8 both ways).<br>**One-line and per-line lists, CPU, full service:** a dash list, one line or per line, gives the same view as prose (2,928/3,000). Decisions are identical: short-band AI 42/12 vs 41/12, human 2/425 vs 2/425.<br>**Numbered, one line:**<br>- Short band: AI Likely AI 42 → 38; human Likely AI 2 → 0.<br>- Main band: AI 114 → 105; human Likely AI stays 2, Likely human 459 → 358.<br>**Hyphens:** `well-known`, `state-of-the-art`, `2-3` and `-5` are kept; `10 - 20` becomes `10 20`.<br>**Idempotence:** false. `**1.** First` → `1. First` → `First`, and `## 1. Heading` behaves the same way. 118 of 92,248 real excerpts change when the view is applied again, which `_prep` does at training and calibration. |
| 2 | R09-3: the registry records `text_view_sha256`, and `verify()` closes the gate if the view's behaviour changes; earlier view names keep their behaviour | **met for gross changes; partial (R10-3)** | **Committed state:** `verify` gives `(True, 'validated')`, and the fingerprint equals the recorded `f3289b1f…`.<br>**Monkeypatched in memory:**<br>- identity view: `(False, 'text view changed since calibration')`.<br>- lower-casing view: closed.<br>- also strip inline `N.` markers: `(True, 'validated')`.<br>- glue `U+2009` instead of spacing it: `(True, 'validated')`.<br>- `deobfuscate` replaced by identity: `(True, 'validated')`.<br>**Old names:** `collapse_whitespace`, `typography+…` and `format+…` against the `4823d02^` code, over all 141,281 train/val texts: **0** differ. |
| 3 | R09-4: flipping any sealed field closes the gate; `report_sha256` matches the committed `benchmark.json` | **met for the seal. The digest matches only the Windows checkout (R10-5)** | **Flipped** (passed, failures, ran_at, report, report_sha256): every flip closes the gate. Dropping any of them, or `record_sha256`, closes it too.<br>**Digest:** `sha256sum` of the working-tree file is `88e05185…`, which equals `report_sha256`. The git blob (`git show HEAD:…`, LF endings) is `23918542…`; the working copy has CRLF (`core.autocrlf=true`).<br>**Not sealed:** clearing `benchmark.history`, deleting both v3 runs from `locked_benchmark_history`, or editing run 1's `passed` or the rerun reason each keep `validated`. |
| 4 | R09-2/R09-8: the listed disguises are undone or noted; U+180E counts once | **met for the list; new cheap evasions remain (R10-1)** | **The R09 list, live (149 validation AI texts the service labels Likely AI):** ligatures with U+2024, hair space in 30% of words, ǀ/ɾ, Armenian ո/հ, U+E0080 and U+0090 each give **149/149 Likely AI, with the note on all 149**. Model-free, they also give the same token ids as clean text.<br>**U+180E:** `format_removed=1`.<br>**Still open:** see R10-1. |
| 5 | The rerun is legitimate under the pre-registration | **met**, with process caveats (R10-8) | **Lock:** `test.lock.json` is unchanged since `bc00087`; its `15d9b9a4…` and counts (11,319 / 13,219) match both reports. `benchmark.py:146` refuses on a mismatch.<br>**Data:** the post-training rebuild's `manifest.parquet` equals the pre-relock copy in all 17 columns for all 260,337 rows (0 cells differ). The build-record diff is only `built_at` and the probe block.<br>**Calibration:** targets are 0.008/0.08 in both runs (`--max-false-ai 0.008 --max-false-human 0.08`). The routing rule, `policy_version` v2.2.0, max_tokens 256/160, OOD k=10 and `torch.manual_seed(0)` are unchanged. The calibration rows reproduce (44,380 = 26,195 + 18,185 excerpts; short 24,463).<br>**Code:** from `72188b4` (run 1) to `d5c1bcf` (run 2), `src/` changes only in Amendment 7's files.<br>**Order:** the rerun reason goes into the `started` entry before scoring (`benchmark.py:159-163` runs before `:174`). Timeline:<br>- 04:22:58: `4823d02` (Amendment 7).<br>- 04:27:07–05:50:07: training and calibration.<br>- 05:51:09–06:13:59: build.<br>- 06:14:34–06:31:11: benchmark (`started_at` 00:44:50Z).<br>**Reads:** training and calibration read only train/val (`load_fit_data`, `calibrate.load_split`). |
| 6 | Gate arithmetic | **met** | **Bounds:** one-sided Clopper-Pearson, recomputed, matches every slice to 1e-9 in both reports.<br>**Gated false AI:** No Robots 3/1,003 (0.0077), AITA 2/854 (0.0074), MIRAGE 0.0045, Stack Exchange 5/2,973 (0.0035), HC3 0.0032, RAID 0.0024, MAGE 0.0015. All are ≤ 0.01.<br>**Gated false human:** the worst is MAGE 48/1,653 (0.0368), ≤ 0.10.<br>**Ungated:** OpenAssistant (188 human / 80 AI).<br>**Coverage:** 0.6688, equal to the slice-weighted sum over 23,353 supported texts.<br>**Margins:** AITA fails at 4 false AI (it has 2); No Robots, MIRAGE and HC3 at 5 (3, 1 and 0). |
| 7 | Tests | **met**, with a test gap (R10-2) | `262 passed` three times (23.6 s, 33.8 s, and 37.0 s with an empty `CORPUS_DIR`). The idempotence test uses three inputs, none of which exposes the failure. |

## Findings

### R10-1: New cheap disguises turn up to a third of confident AI into Likely human, with no disguise note

- **Severity:** High
- **Category:** robustness / evasion (R07-1's "normalize or abstain" bar, the R08-1 class)
- **Location:**
  - `src/detection/obfuscation.py:99-100`: a character with `isalpha() == False` is copied as is. Small roman numerals (Nl: ⅰ ⅼ ⅽ ⅾ ⅿ ⅴ ⅹ, and capital Ⅰ) are never NFKC-folded, though `_compat_latin` would fold them.
  - `obfuscation.py:41-42`: `_INVISIBLE` still omits the unassigned default-ignorables that R09-2 named: U+2065 and U+FFF0–FFF8.
  - `obfuscation.py:44`: `_SPLIT_SPACE` removes one thin space between two word characters. Two in a row (`  `) survive, as do U+205F and U+2005. `collapse_whitespace` then splits the word.
  - `obfuscation.py:79-80,103-104`: every other LATIN-named letter is kept with no note, including ĸ, Ɪ (U+A7AE, whose name doesn't match `_SMALL_CAPITAL`), ʋ, ɯ and ƅ. Combining marks (for example i + U+0307) are kept too.
- **Repro steps:**
  1. Take 492 hash-sampled v3 **validation** AI texts of 50–199 words.
  2. Keep the 149 that the registered service labels Likely AI (`verify` = validated).
  3. Apply each edit: the character inserted mid-word in 30% of words over 3 letters, or letters swapped with p=0.15.
  4. Run `check_batch` on CPU. The script is `scratchpad/r10/obf_live.py`.
- **Expected vs actual:** Disguise should be undone, or the text should abstain with a note. Instead, no row below has a single `OBFUSCATION_NORMALIZED` note:

  | Edit (149 texts, all Likely AI unedited) | Likely AI | Likely human | Inconclusive | Note |
  |---|---|---|---|---|
  | small roman numerals for i l c d m v x, p=0.15 (looks identical) | 8 | **27** | 114 | 0 |
  | small roman numerals, every such letter | 0 | 0 | 149 (142 OOD) | 0 |
  | U+FFF0, 30% of words | 1 | **27** | 121 | 0 |
  | U+2065, 30% of words | 3 | 14 | 132 | 0 |
  | two U+200A hair spaces, 10% of words | 51 | 5 | 93 | 0 |
  | two U+200A hair spaces, 30% of words | 5 | **24** | 120 | 0 |
  | U+205F medium math space, 30% of words | 5 | **24** | 120 | 0 |
  | i + U+0307 for every i (a faint extra dot in some fonts) | 1 | **46** | 102 | 0 |

- **Evidence:**
  - Model-free on one sentence: roman numerals take it from 25 to 127 tokens with `changed=False` and `english=True`. U+2065 gives 71 tokens, U+FFF0 86 and a double hair space 48, all with `changed=False` and passing the English gate.
  - ĸ, Ɪ, ʋ, ɯ and ƅ each give `changed=False, mixed=0`.
  - For comparison, the same 149 texts under the R09 list stay 149/149 Likely AI. The fix works for what it lists, but each new list leaves the next character class open.
  - A general rule would close the class: NFKC-fold every character whose fold is an ASCII letter, remove the whole Default_Ignorable property, and remove any run of spaces narrower than a normal space inside a word.
- **Confidence:** high

### R10-2: Other list glyphs and parenthesised numbers survive the view and still move decisions both ways; the view is not idempotent

- **Severity:** Medium
- **Category:** robustness / validity (R09-1 residual)
- **Location:**
  - `src/corpus/text.py:55`: `_INLINE_MARKUP` covers only • ‣ ⁃ ▪ ◦ and free-standing dashes. `·`, `●`, `○`, `■`, `►`, `✓`, `→` and `➢`, emoji bullets, `+`, inline `#`/`>`, `(1)`, `a)` and inline `1.` are kept.
  - `text.py:43`: `_LINE_MARKUP` strips a heading or an emphasis marker first. In `### 1. Title` or `**1.** Step`, the number is then left mid-line, where nothing strips it, so it reaches excerpts.
  - `tests/test_authorship_support.py`: the idempotence test uses three inputs.
- **Repro steps:**
  1. Run `text_view("format2+typography+collapse_whitespace")` on the inputs below.
  2. Hash-sample v3 **validation** texts: 1,065 human and 396 AI at 30–85 words (short model); 1,034 human and 291 AI at 110–180 words (main model).
  3. Split each into sentences and re-join them in each format.
  4. Run `check_batch` on CPU. The script is `scratchpad/r10/fmt_live.py`.
- **Expected vs actual:** Amendment 7 says formatting can't stand in for authorship. Instead, the view gives:
  ```
  'Tips: ● Mix. · Add. ■ Bake. ► Serve.'        -> unchanged
  'Tips: (1) Mix the flour. (2) Add water.'       -> unchanged
  'Intro.\n### 1. Mix the flour\nText.'           -> 'Intro. 1. Mix the flour Text.'
  excerpt of a '### N.' list                      -> '7. Step number 7 This step ... 8. Step number 8 ...'
  '**1.** First step here.'  v -> '1. First step here.'   v(v) -> 'First step here.'
  ```

  Results on validation text (Likely AI / Likely human / inconclusive):

  | Format | Short AI (396) | Short human (1,065) | Main AI (291) | Main human (1,034) |
  |---|---|---|---|---|
  | plain prose | 42 / 12 / 342 | 2 / 425 / 638 | 114 / 0 / 177 | 2 / 459 / 573 |
  | `· ` items, one line | **20 / 27** / 349 | 0 / 504 / 561 | 95 / 0 / 196 | 1 / 369 / 664 |
  | `● ` items, one line | **23 / 24** / 349 | 0 / 455 / 610 | 96 / 0 / 195 | 1 / 320 / 713 |
  | `(1) (2)`, one line | 40 / 8 / 348 | **5** / 312 / 748 | 111 / 0 / 180 | **4** / 233 / 797 |
  | `**N.**` per line | 36 / 15 / 345 | 0 / 408 / 657 | 99 / 0 / 192 | 1 / 353 / 680 |
  | numbered, one line | 38 / 15 / 343 | 0 / 418 / 647 | 105 / 0 / 186 | 2 / 358 / 674 |

- **Evidence:**
  - A one-line `·` or `●` list halves Likely AI in the short band and doubles Likely human among AI texts. This is the R09-1 dash-list evasion with another glyph.
  - `(1) (2)` markers pull human text toward AI. Likely AI rises from 4 to 9 of 2,099 validation humans (upper bound 0.44% to 0.75%): 4 new flags in the short band (mage, dolly, beemo and oasst1 at 40–44 words) and 2 in the main band (beemo and no_robots at 145–146 words). Likely human falls from 884 to 545.
  - Rater comments often enumerate issues as "(1) … (2) …".
  - These glyphs are rare and label-balanced in training (≥2 inline numbers in excerpts: AI 0.32% vs human 0.46%; other bullets 0.04% vs 0.03%). So this is sensitivity to unseen formatting, not a learned shortcut.
- **Confidence:** high on the view behaviour and the shifts. Medium on how often production text uses these forms.

### R10-3: The view fingerprint samples three inputs, and the rest of the scoring code is unbound, so behaviour can change after the pass and the gate stays open

- **Severity:** Medium
- **Category:** gate integrity (R09-3 residual)
- **Location:**
  - `src/corpus/text.py:76-87`: `_FINGERPRINT_CASES` holds three fixed strings.
  - `src/authorship/registry.py:148-153`: `verify` compares only that fingerprint.
  - `src/detection/obfuscation.py`, `policy.py` and `detector.py` windowing and routing are in no digest.
- **Repro steps:** In memory, with the committed registry:
  1. Replace the view with `f(t)` plus inline `N.` stripping, and run `verify`.
  2. Replace it with `f(t.replace(" ", ""))`, and run `verify`.
  3. Set `service.deobfuscate` to identity, and run `verify`.
- **Expected vs actual:** The gate should open only for the system the locked test measured. Output:
  ```
  view also strips inline 'N.' markers; 'Do this: 1. Mix the flour. 2. Add water. 3) Bake.' -> 'Do this: Mix the flour. Add water. Bake.'
     verify: (True, 'validated')
  view glues U+2009 instead of spacing it: '10 km' -> '10km' (True, 'validated')
  deobfuscate replaced by identity; verify: (True, 'validated')
  ```
- **Evidence:**
  - An identity or lower-casing view is caught.
  - The obvious fix for R10-2 (strip inline numbers or more glyphs) would silently change v5's inputs while the registry says validated. That is R09-3's own scenario.
  - `deobfuscate` did change between run 1 and run 2, and neither registry entry records it.
- **Confidence:** high

### R10-4: Deobfuscation now puts the disguise note on ordinary text, and deletes whitespace controls and cp1252 punctuation, gluing words

- **Severity:** Low
- **Category:** functional bug (evidence contract)
- **Location:**
  - `src/detection/obfuscation.py:46,88-90`: U+2026 (…) is a "compat form" and counts as a mapped letter.
  - `obfuscation.py:44`: U+202F and U+2009 between word characters are deleted.
  - `obfuscation.py:42`: the removed controls include VT, FF, U+001C–001F and NEL, which are whitespace, and the cp1252 punctuation bytes U+0080–009F.
- **Repro steps:**
  1. Run `deobfuscate` on the inputs below.
  2. Count `changed` over all 141,281 train/val texts, with the `4823d02^` code and with the current code.
- **Expected vs actual:** The note says invisible characters were removed and lookalike letters were read as Latin. It should appear only for disguise, and word boundaries should survive. Instead:
  ```
  'Well… I guess that works'            -> changed=True mapped=1
  'at 10:30 AM' (macOS/iOS time)   -> '10:30AM'  changed=True
  '1 000 kg'                         -> '1000 kg'  changed=True
  'Point one\x0cPoint two'               -> 'Point onePoint two'   (4 -> 3 words)
  'Line one\x85Line two'                 -> 'Line oneLine two'
  "don\x92t \x93quote\x94"               -> 'dont quote'
  ```
- **Evidence:**
  - Texts that get the note: 240 before, **1,339** now (human 83 → 343, AI 157 → 996). Ellipsis alone accounts for 167 human and 673 AI texts.
  - 14 train/val texts lose words in the model view. They are mostly RAID human text with misdecoded cp1252 `\x85` (…), which the model saw as bytes in training.
  - The decision is unchanged, but the note is shown to reviewers as a sign of disguise.
- **Confidence:** high

### R10-5: `report_sha256` matches only the Windows working copy and is never checked; the run history is unsealed

- **Severity:** Low
- **Category:** gate integrity
- **Location:**
  - `src/authorship/benchmark.py:231,237`: `write_text` writes CRLF on Windows, and the file is hashed as written.
  - `src/authorship/registry.py:137-155`: `verify` never compares `report_sha256` with the file.
  - `registry.py:116`: `locked_benchmark_history` is outside the digest.
- **Repro steps:**
  1. `sha256sum data/benchmark/v3/model/benchmark.json`
  2. `git show HEAD:data/benchmark/v3/model/benchmark.json | sha256sum`
  3. In memory, delete the two v3 entries from `locked_benchmark_history` and run `verify`.
- **Expected vs actual:** A report digest should identify the committed report on any checkout, and editing the run history should close the gate. Instead:
  - `88e05185…` (working tree, equals `report_sha256`) vs `23918542…` (the committed blob). A Linux or `autocrlf=false` checkout won't match.
  - Deleting both runs gives `(True, 'validated')`. So do editing run 1's `passed` and clearing the `rerun_reason`.
- **Evidence:** as above.
- **Confidence:** high

### R10-6: The rerun doubles false AI, with the rise on report-only assistant-style and in-house text

- **Severity:** Low
- **Category:** validity (disclosed)
- **Location:** `data/benchmark/v3/model/benchmark.json` against `benchmark-run1.json`.
- **Repro steps:** Compare `test.by_source` and `inhouse_unverified` in the two reports.
- **Expected vs actual:** The R09-1 fix was expected to remove a shortcut, not to raise human-side flags. Instead:
  - False AI goes from 8 to **18** of 11,319 (run 1 → run 2).
  - OpenAssistant goes from 0 to **6/188** (3.2%, upper bound 6.2%). This is the failure 4c was built to fix, and R08-3 noted it can't fail the gate.
  - Stack Exchange goes from 0 to 5/2,973.
  - In-house cells go from 5 to **50** of 3,770; never-viewed cells go from 3 to 38/1,621 (2.3%, upper bound 3.1%).
  - All are within the gate, or report-only.
- **Evidence:**
  - The report and README state these and flag in-house and OpenAssistant for sponsor review.
  - In-house labels are unverified, so the in-house rise can't be read as false AI. It is still the slice closest to production.
  - Some in-house change could come from R10-4's control-character removal. This review did not read in-house text to check.
- **Confidence:** high on the numbers. Low on the cause.

### R10-7: Unspaced em dashes survive as word hyphens, while spaced dashes are now deleted; one hyphen-rate feature lifts MIRAGE's probe over 0.85

- **Severity:** Low
- **Category:** validity / probe coverage (R09-6 residual)
- **Location:**
  - `src/corpus/text.py:30`: `—` becomes `-`, so AI's `word—word` becomes `word-word`.
  - `text.py:55`: a spaced `-` is deleted.
  - `src/corpus/confound.py:26-30`: the probe has no hyphen feature.
- **Repro steps:**
  1. Fit the build's own-source probe (core rows, train to val, model view). The committed AUCs reproduce exactly (dolly 0.8383, hc3 0.8239, mirage 0.8393).
  2. Add one feature, the rate of `\w-\w` per word.
- **Expected vs actual:** Amendment 7 says every source is under 0.85. Instead:
  - R09's formatting indicators no longer cross it: hc3 0.8283, dolly 0.8446, mirage 0.8414.
  - The hyphen-rate feature alone gives **MIRAGE 0.8506**.
  - In MIRAGE's raw train text, unspaced em or en dashes appear in 22.5% of AI texts and 1.3% of human texts (Enron-era ASCII email).
- **Evidence:** as above. Whether an em dash is authorial style or editor typography is a judgement call, and Amendment 5 classed it as typography. The margin is small and depends on the feature.
- **Confidence:** medium

### R10-8: Process: the probe of record came after training, Amendment 7 postdates its commit, and the docs call 4c done before this review

- **Severity:** Low
- **Category:** process integrity (R08-9/R09-7 class)
- **Location:**
  - `increment-4c-preregistration.md:38` ("fixed or excluded **before** training") and `:210` (Amendment 7 headed 04:25).
  - `roadmap.md:21,39` and `README.md:8`, as of `5a1cf4d`.
- **Repro steps:**
  1. Read the builder's task outputs: training ran 04:27:07–05:50:07 and `python -m src.corpus.build` 05:51:09–06:13:59.
  2. `git log --date=iso`: `4823d02` is at 04:22:58 and `8818eff` (which edits Amendment 7) at 04:24:16.
- **Expected vs actual:** The pre-registered order is probe, then training. Instead:
  - The format2 probe of record (`d5c1bcf`) was computed after v5 and short-v3 were trained and calibrated. It flagged nothing, so nothing changed.
  - Amendment 7's heading time is later than both commits that wrote it.
  - The roadmap says "done (passed twice …; Ralph R10 pending)" and "the gate is open" while this review was running.
- **Evidence:** as listed. The data, the models and the gate arithmetic are unaffected. The corpus rebuild happened between calibration and the run, but it reproduces the training digests and the manifest exactly.
- **Confidence:** high


## Triage (build chat, 2026-09-28)

R10-1 is a bug in the disguise defence, so the service's input handling is replaced and the locked run is repeated once, with its reason recorded. The models, thresholds, criteria and data are unchanged. Details are in Amendment 8. The suite is now 269 tests, all passing.

| ID | Triage | What changed | Evidence after the fix |
|---|---|---|---|
| R10-1 | **Fix, rerun** | Disguise is closed as a class: every format, unassigned, private-use, surrogate, default-ignorable and control character is removed; spacing runs inside words, and combining marks on ASCII letters, are removed; ASCII-bound NFKC forms are folded; lookalikes are mapped; residual unusual-Latin or mixed-script words make the service abstain (`DISGUISE_SUSPECTED`). | v5 on CPU, 80 validation texts labelled Likely AI: roman numerals, U+FFF0, double hair spaces, U+205F, i + U+0307 and U+2065 leave 77–80 Likely AI, all noted. v-with-hook abstains on 80/80. None becomes Likely human. |
| R10-2 | **Debt D14** | Other list glyphs, `(1)` markers and the view's idempotence need a view change and retraining. | — |
| R10-3 | **Fix** | `scoring_code_sha256` over the scoring modules is recorded at calibration and checked by `verify()`. | New registry test. |
| R10-4 | **Fix** | Whitespace controls become spaces; C1 bytes are read as cp1252; ordinary typography raises no note. | Train/val note rate 489 of 141,281 (was 1,339). |
| R10-5 | **Fix** | The seal covers every locked run; report digests are line-ending independent and checked against the file. | New registry tests. |
| R10-6 | **Disclosed** | Human-side flags are reported first among the caveats. | — |
| R10-7 | **Debt D15** | Unspaced em dashes as hyphens: a view change. | — |
| R10-8 | **Corrected** | The pre-training probe run is recorded, Amendment 7's time is fixed, and status waits for this run. | — |
