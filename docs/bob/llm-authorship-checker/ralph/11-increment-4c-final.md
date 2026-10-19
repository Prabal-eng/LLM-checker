# wreck-it-ralph report: the R10 fixes (Amendment 8) and the second recorded v3 rerun

Verdict: **FAIL**

Summary: tests **269/269** (`python -m pytest -q`, 23.6 s). No end-to-end examples were given. Findings **C0/H1/M1/L3**.

Range: `fadd8c3^..HEAD` (`fadd8c3` at 07:24:48, `ff39735` at 07:44:14). HEAD didn't move during the review.

How the rules were kept:
- Nothing in the repo was edited or committed except this report. Scratch scripts and outputs are in the session scratchpad, under `scratchpad/r11/`.
- The benchmark, training, calibration and the corpus build were not run. No file named `x` or `inspect_wb.py`, no root `.xlsx` and no `dataset_full.csv` was opened.
- **Text used:** all train/val. It comes from `scratchpad/r10/fit.parquet` (141,281 rows), which R10 extracted with `load_fit_data`. I asserted its `split` column holds only `train`/`val`. Amendment 8 didn't change the data, and the lock is unchanged. `corpus.parquet` wasn't opened.
- **Locked text:** none was read, hashed or scored. Locked-split facts come from the two committed reports and `test.lock.json`.
- **Model use:** v5 and short-v3, loaded through the committed registry (`verify` = validated), with `CUDA_VISIBLE_DEVICES=-1`. Each script asserts `torch.cuda.is_available() is False` and prints the model device, which was `cpu`.
- **What was scored:** only the 149 **validation** AI texts (50–199 words) that the service labels Likely AI. They are R10's hash-sampled pool, re-derived with the same seed. Each was scored under 21 edits.

## Claims checked

| # | Claim | Result | Evidence |
|---|---|---|---|
| 1a | R10-1's listed disguises are undone or noted | **met** | Same 149 texts, R10's own script against the current code. Roman numerals (p=.15 and p=1), U+2065, U+FFF0, double hair spaces (10% and 30%), U+205F, i + U+0307, and the R09 list each give **149/149 Likely AI, with the note on 149/149**. |
| 1b | Disguise is closed as a class: no cheap character-level edit turns Likely AI into Likely human without an abstention or a note | **not met (R11-1)** | A dot below on 10% of vowels gives 4 Likely AI and **38 Likely human, 37 of them with no note and no abstention**. Macron below, hair space + U+0307, symbol lookalikes and non-Latin digit zeros also flip texts silently. |
| 1c | Ordinary train/val text: abstention and note rates; nothing damaged | **rates met as stated. Minor damage and false abstentions remain (R11-3)** | Of 141,281 texts:<br>- **Note:** 489 (157 human, 332 AI).<br>- **Abstain** (`residual > 0`): 247 (69 human, 178 AI).<br>Both match Amendment 8.<br>**Model-view word count changes in 38 texts.** 34 are benign: C1 `\x85` now reads as `…`, and stand-alone U+200B is dropped. The exceptions are in R11-3: a human No Robots text glued from 67 to 51 words, and Greek `μg` or `Aβ` triggering an abstention. |
| 2a | `verify` closes the gate when any listed scoring module changes | **met for the listed files. The list misses the score map (R11-2)** | I copied the 9 files into a temp repo root (`registry._REPO`) and appended a comment to each. Every one gives `(False, 'scoring code changed since calibration')`.<br>Changing `calibration.probability` (the Platt map used at scoring) keeps `(True, 'validated')`. |
| 2b | Editing or deleting any locked run closes the gate | **met** | For each of runs 0–5, flipping `passed`, deleting the run and editing `rerun_reason` all give `(False, 'registry changed after the locked benchmark')`. Clearing the history or appending a run does the same.<br>`benchmark.history` isn't sealed (R11-4). |
| 2c | Report digest is line-ending independent and checked against the file | **met while the file is present (R11-4)** | `text_sha256` of the CRLF working copy = the LF git blob's sha256 = `report_sha256` = `9d8412a4…`. An edited report gives `(False, 'benchmark report changed after the run')`. An LF copy verifies. A deleted report verifies too. |
| 2d | Committed registry verifies | **met** | `(True, 'validated')`. `scoring_code_sha256` matches the current files, the HEAD blobs and the `fadd8c3` blobs. |
| 3a | Reason recorded before scoring | **met** | `benchmark.py:158-163` saves the `started` entry before `decide_all` (`:174`).<br>Timeline (UTC):<br>- `fadd8c3` committed 01:54:48.<br>- `calibration.json` `finished_at` 01:55:52.<br>- Run `started_at` 01:56:12, `ran_at` 02:13:15.<br>- `ff39735` committed 02:14:14.<br>The reason names Amendment 8 / R10-1. |
| 3b | Calibration identical to the previous registry except the scoring-code digest | **met** | A key-by-key diff of `git show fadd8c3:models/registry.json` against HEAD shows:<br>- `scoring_code_sha256` added;<br>- the `benchmark` block;<br>- one run appended to `locked_benchmark_history` (runs 0–4 byte-identical).<br>`calibration`, both `platt` maps, all bands, both OOD blocks, artifacts, view and fingerprint are identical. `calibration.json` differs only in `finished_at`. |
| 3c | Lock `15d9b9a4` unchanged | **met** | `test.lock.json` was last changed in `bc00087`, and its `sha256` is `15d9b9a4…`. The new run entry records the same `lock_sha256`. The counts are 11,319 human and 13,219 AI, as in run 2. |
| 3d | Code diff since the previous run covers only Amendment 8's files | **met** | `git diff --stat d5c1bcf HEAD -- src/`, where `d5c1bcf` is run 2's code: `benchmark.py`, `calibrate.py`, `registry.py`, `obfuscation.py`, `service.py`. All five are in `fadd8c3`. There is no `src/` change after `fadd8c3`. `benchmark-run2.json` is byte-identical to `fadd8c3:benchmark.json` (`23918542…`). |
| 3e | Gate arithmetic | **met** | One-sided Clopper-Pearson bounds, recomputed with scipy, match every slice exactly (max diff 0).<br>**Gated false AI:** the worst is No Robots 3/1,003 (0.0077), then AITA 2/854 (0.0074). All are ≤ 0.01.<br>**Gated false human:** the worst is MAGE 48/1,653 (0.0368), ≤ 0.10.<br>**Ungated:** OpenAssistant (188/80).<br>**Coverage:** 0.6671, equal to the slice-weighted sum over 23,353 supported texts, ≥ 0.40.<br>**Against run 2:** false AI 18 → 18, false human 90 → 90, detected AI 8,617 → 8,585 (MIRAGE −32). `DISGUISE_SUSPECTED` takes 58 test texts (48 AI) and 300 stress texts (MIRAGE `llm_polish`/`llm_rewrite`). |
| 4 | Tests | **met, with a test gap (R11-2)** | `269 passed`. The new scoring-code test overwrites the recorded digest. It never changes a file. |

## Findings

### R11-1: Combining marks that compose, precomposed "ordinary" letters, symbol lookalikes and non-Latin digits pass silently; a dot below on 10% of vowels turns a quarter of confident AI into Likely human

- **Severity:** High
- **Category:** robustness / evasion (R10-1 not closed as a class; the R07-1 "normalize or abstain" bar)
- **Location:**
  - `src/detection/obfuscation.py:111,115`: NFC runs first. Only a mark still uncomposed on an ASCII letter is removed. `a`+U+0323 composes to `ạ` (U+1EA1), so the rule never sees a mark. **369** ASCII letter + combining-mark pairs, over 26 marks in U+0300–036F, compose into a letter in `_ORDINARY`. That includes U+0307 on `e`, `c`, `g` and `z`: `e`+U+0307 gives `ė` silently. The R10 test used `i`, which has no precomposed form.
  - `obfuscation.py:52-53`: `_ORDINARY` treats all of Latin-1, Latin Extended-A and Latin Extended Additional (U+1E00–1EFF) as ordinary. Any of these letters inside an English word is kept with no count, whether typed precomposed or built with a mark.
  - `obfuscation.py:42,136`: `\w+` never matches symbols (So/Sm). So `⍺` (U+237A) or `℮` (U+212E) inside a word is never examined. A non-ASCII digit such as `०` (U+0966) or `١` (U+0661) has `isalpha() == False` and is copied without a count.
  - `obfuscation.py:115`: a mark after a space or a hair space is kept. `wo` + U+200A + U+0307 + `rd` splits the word, silently.
- **Repro steps:**
  1. Take R10's pool: 492 hash-sampled v3 **validation** AI texts of 50–199 words, seed `r10obf:`.
  2. Keep the 149 that the registered service labels Likely AI.
  3. Apply each edit, with letters swapped with probability p.
  4. Run `check_batch` on CPU. The scripts are `scratchpad/r11/obf_live.py` and `obf_live2.py`; output is in `obf_live*.out`.
- **Expected vs actual:** Disguise should be undone, or the text should abstain with a note. Instead (no row has a `DISGUISE_SUSPECTED` abstention; "no note" excludes the one pool text whose original already carries a note):

  | Edit (149 texts, all Likely AI unedited) | Likely AI | Likely human | Human with no note | Inconclusive |
  |---|---|---|---|---|
  | vowel + U+0323 (dot below), p=.05 | 30 | 6 | 6 | 113 |
  | vowel + U+0323, p=.10 | 4 | **38** | **37** | 107 |
  | precomposed ạ ẹ ị ọ ụ, p=.10 | 4 | 38 | 37 | 107 |
  | vowel + U+0323, p=.15 | **0** | **55** | **54** | 94 |
  | b d k l n r t z + U+0331 (macron below), p=.15 | 2 | **44** | 43 | 103 |
  | hair space + U+0307 in 30% of words | 2 | **26** | 26 | 121 |
  | a → ⍺ (U+237A), e → ℮ (U+212E), p=.15 | 18 | **15** | 15 | 116 |
  | o → ० (U+0966 Devanagari zero), p=.3 | 33 | 4 | 3 | 112 |
  | o → ० , p=.15 | 58 | 0 | 0 | 91 |
  | l → ١ (U+0661), p=.15 | 84 | 0 | 0 | 65 |
  | i → í, p=.15 | 65 | 2 | 1 | 82 |
  | ASCII `I` for `l`, p=.15 | 86 | 0 | 0 | 63 |

- **Evidence:**
  - **Model-free** (`scratchpad/r11/obf_free.py`), `changed=False residual=0 english=True` for every case, for example:
    ```
    a -> a + U+0323 (composes)    changed=False residual=0 ... view='The model is ạlso ạ good choice ...'
    o -> U+0966 devanagari zero   changed=False residual=0 ... view='The m०del is als० a g००d ch०ice ...'
    a -> U+237A APL alpha (So)    changed=False residual=0 ... view='The model is ⍺lso ⍺ good choice ...'
    hair space + U+0307 mid-word  changed=False residual=0 ... view='The mo ̇del is also a good cho ̇ice ...'
    deobfuscate('ėvery case') -> 'ėvery case' changed False residual 0
    ```
  - **Size of the effect:** a dot below on 10% of vowels is barely visible. It removes the AI label from 145 of 149 texts, and 37 of those become Likely human with no reviewer-visible sign. That is worse than R10-1's worst row (46 of 149 with i + U+0307).
  - **Silent de-labelling:** `०` at p=.15 flips none to human, but it silently moves 91 of 149 from Likely AI to Inconclusive.
  - **Scope of the fix:** the R10 cases are all fixed (claim 1a). But the rule is still keyed to what reviewers found. Marks that compose, the "ordinary" accented ranges, symbols and non-Latin digits are the same class under other code points.
  - **A class-level rule would be:**
    - Skeleton every word with UTS #39 confusables.
    - Count any Mn, So/Sm or non-ASCII Nd attached to ASCII letters as residual.
    - Count diacritics inside words whose base letters are ASCII. Note or abstain above a small per-text rate: ordinary English carries a few accents (café, naïve, names), not one per ten vowels.
- **Confidence:** high

### R11-2: The scoring-code digest omits the Platt score map (and `src/detectors/base.py`), so a change to how scores are computed keeps the gate open; the test never changes a file

- **Severity:** Medium
- **Category:** gate integrity (R10-3 residual) / test gap
- **Location:**
  - `src/authorship/registry.py:117-119`: `_SCORING_CODE` lists 9 modules.
  - `src/authorship/detector.py:20,139`: every probability comes from `calibration.probability(raw, *platt)` in `src/authorship/calibration.py`, which isn't listed.
  - `src/detectors/base.py` (`DetectionResult`, `BaseDetector`) is also unlisted.
  - `tests/test_authorship_registry.py`, `test_changed_scoring_code_closes_the_gate`: sets the recorded digest to zeros instead of changing a module.
- **Repro steps:**
  1. Copy the listed files, `calibration.py`, `base.py` and the report to a temp dir, and set `registry._REPO` to it.
  2. Edit each file, then run `verify` on the committed record. The script is `scratchpad/r11/reg_checks.py`.
- **Expected vs actual:** The registry comment says the list covers "everything besides the model that decides an outcome". Amendment 8 says "any later edit to these files ... needs a new locked run". Output:
  ```
  edit src/detection/obfuscation.py: (False, 'scoring code changed since calibration')   [same for all 9 listed]
  edit src/authorship/calibration.py (not listed): digest unchanged=True (True, 'validated')
      (probability() shifted by -2 logits: every score moves)
  edit src/detectors/base.py (not listed): digest unchanged=True (True, 'validated')
  ```
- **Evidence:** As above. The recorded `platt` coefficients are sealed, but the function that applies them isn't. The fix is to add `src/authorship/calibration.py`, `src/detectors/base.py` and `src/detection/__init__.py`, and to derive or check the list from the modules the service imports. A test should change a file in a temp root, not the recorded digest.
- **Confidence:** high

### R11-3: Ordinary text: Greek μ/β units and IPA abstain as disguise, U+202F word separators glue a human text, and in the locked run the new abstention falls mostly on AI text

- **Severity:** Low
- **Category:** functional bug (false abstention, damage)
- **Location:**
  - `obfuscation.py:147-151`: any non-Latin letter in a word with ASCII letters counts as mixed-script.
  - `obfuscation.py:48`: `_SPLIT_SPACE` removes U+202F between letters.
- **Repro steps:**
  1. Run `deobfuscate` on the inputs below.
  2. Run `scratchpad/r11/ordinary.py` over the 141,281 train/val texts.
- **Expected vs actual:** Ordinary English should neither abstain as disguise nor lose word boundaries. Instead:
  ```
  'Give 5 μg of drug'             -> changed True residual 1   (DISGUISE_SUSPECTED)
  'Aggregates of Aβ in plaque'    -> changed True residual 1
  'It returned from the lab' -> 'It returnedfrom the lab'  changed True
  ```
- **Evidence:**
  - **Top residual characters on train/val:** ˈ 93, ə 53, ʿ 45, ʾ 42, ː 37, ɪ 31, Greek μ 23, β 19. Scientific units and transliterations, not only "rare phonetic letters".
  - **U+202F glue:** the one No Robots **human** validation text that separates words with U+202F goes from 67 to 51 words ("returnedfroma holiday tofindmold"). It also gets the disguise note.
  - **Locked run** (report only): `DISGUISE_SUSPECTED` takes 58 test texts. 48 are AI, 47 of them MIRAGE, and MIRAGE detected AI fell 5,953 → 5,921. It also takes 300 stress texts (MIRAGE `llm_polish` 136, `llm_rewrite` 163).
  - The abstention is conservative, and all rates are small. It mostly removes correct AI labels from LLM-rewritten text, not human flags.
- **Confidence:** high on the behaviour. Medium on the production frequency.

### R11-4: The report check is skipped when the file is missing, `benchmark.history` is unsealed and reset at each calibration, and anyone can re-seal an edited registry

- **Severity:** Low
- **Category:** gate integrity
- **Location:**
  - `registry.py:178`: `report.is_file()` gates the check.
  - `registry.py:136`: `_RUN_FIELDS` excludes `benchmark`, so `benchmark.history` is outside the digest.
  - `calibrate.py:265`: sets `"benchmark": None`, so `history` restarts with each calibration.
  - `registry.py:143-155`: the seal is an unkeyed SHA-256.
- **Repro steps:** Run `scratchpad/r11/reg_checks.py`.
- **Expected vs actual:** Output:
  ```
  delete report file: (True, 'validated')
  edit benchmark.history reason: (True, 'validated')
  clear benchmark.history: (True, 'validated')
  drop runs 0-4 then re-seal with the public function: (True, 'validated')
  ```
- **Evidence:**
  - The committed `benchmark.history` holds only run 3. Run 2's entry was replaced when calibration reset the block.
  - `locked_benchmark_history` does keep all six runs, sealed.
  - Amendment 8 discloses "when the file is present".
  - The seal detects accidental edits, not deliberate ones. Git history is the real anchor.
- **Confidence:** high

### R11-5: Process: Amendment 8's heading gave a time after its own commit and after the run started

- **Severity:** Low
- **Category:** process integrity (R10-8 class)
- **Location:** `increment-4c-preregistration.md:241`.
- **Repro steps:** `git diff fadd8c3 ff39735 -- docs/`
- **Expected vs actual:** The amendment's time should precede the run. Instead:
  - At `fadd8c3` (committed 07:24:48 IST), the heading reads "(2026-09-28, 07:40)".
  - That is later than the run's start (07:26:12 IST).
  - The result commit `ff39735` rewrote it to 07:24.
- **Evidence:** The amendment text at `fadd8c3` contains no run-3 result: it cites run 2 and validation checks only. It was committed before the run started. So this is a wrong timestamp, not a post-hoc amendment. It is the third heading-time correction in a row.
- **Confidence:** high


## Triage (build chat, 2026-09-28)

R11-1 is a bug in the disguise defence, so the service's input handling changes and the locked run is repeated once, with its reason recorded. Models, thresholds, criteria and data are unchanged. Details are in Amendment 9. The suite is now 275 tests, all passing.

| ID | Triage | What changed | Evidence after the fix |
|---|---|---|---|
| R11-1 | **Fix, rerun** | Accents are read back: Latin letters are decomposed and their marks dropped, orphan marks are removed, and stroked letters and letter-shaped symbols are mapped. Symbols and non-Latin digits touching ASCII letters count as unreadable; the service abstains when that reaches 3 or 2% of words. | v5 on 80 validation texts labelled Likely AI: dot below, macron below, hair space + U+0307, ⍺/℮, e + U+0307, roman numerals and double hair spaces leave 76–80 Likely AI. Devanagari zero abstains on 80/80. None becomes Likely human. |
| R11-2 | **Fix** | `calibration.py` and `detectors/base.py` are in the scoring-code digest; the new test edits a copied file. | Test passes. |
| R11-3 | **Fix** | Density-based abstention; μ and β are mapped; the narrow no-break space between letters stays a space. | Train/val abstention 62 of 141,281 (was 247). |
| R11-4 | **Fix, partly** | The named report must exist and match. The seal is recomputable by anyone with write access; git history is the anchor. | New test. |
| R11-5 | **Corrected** | Amendment headings are date-only. | — |
