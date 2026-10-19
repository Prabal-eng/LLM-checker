# wreck-it-ralph report: the R08 fixes, the relock, v4 training and the v3 locked pass

Verdict: **FAIL**

Summary: tests **255/255** (no GPU, run twice: once as is, once with an empty `CORPUS_DIR`). No end-to-end examples were given. Findings **C0/H1/M2/L5**.

Range: `061bda0^..HEAD`. HEAD moved during the review, from `f9d143a` to `bb5cf48` (the Increment 4c report, committed at 03:54), so the report is in scope too.

How the rules were kept:
- Nothing in the repo was edited or committed except this report. Scratch scripts are in the session scratchpad.
- The benchmark, training and calibration were not run. No file named `x` or `inspect_wb.py`, no root `.xlsx` and no `dataset_full.csv` was opened.
- `corpus.parquet` was read only with column selection and a split filter. It has one row group, so pyarrow decodes the whole text column before filtering. No text was kept beyond the filtered rows.
- **Model use:** CPU only (`CUDA_VISIBLE_DEVICES=""`, 4 threads). The models were v4 and short-v2, loaded through the committed registry, which verifies. All scored text was v3 **validation**:
  - 422 validation AI texts, of which 84 are Likely AI, each scored under 15 edits;
  - 275 validation excerpts;
  - 710 human and 653 AI validation texts, each reformatted 4 ways.
- **Locked text** was read once, only to recompute the lock hash with `lock.locked_hash` (71,357 rows). It was not printed and not scored. Test results come from `benchmark.json` totals.
- **Builder files:** I read the builder's scratch files and task logs to check the order of events. These were `compare_manifests.py`, `manifest_v3_before_relock.parquet` (columns only), `summarize_benchmark.py`, `recheck_r08_1.py`, `probe_format.py`, `amend6.py` and the task outputs.

## Claims checked

| # | Claim | Result | Evidence |
|---|---|---|---|
| 1 | R08-1: obfuscation removes invisible marks and maps Latin lookalikes, keeps accents, and runs and is noted before scoring | **met for R08's cases; cheap evasions remain (R09-2)** | **R08's repro, run on v4 (84 validation texts it labels Likely AI):**<br>- U+FE0F in 30% of words: 83/84 still Likely AI, with the note on all 84.<br>- U+034F in 30% of words: the same.<br>- ı ɑ ɡ ᴏ ᴄ ꜱ ʏ at p=0.15: the same.<br>- No edit: 83/84.<br>**Service:** `check_batch` deobfuscates before the language gate and the detector, and adds `OBFUSCATION_NORMALIZED` (`service.py:54-76`).<br>**Accents:** the test keeps café, naïve, Ångström and résumé. Over all 141,281 train/val texts, deobfuscation changes 240. The changes are mostly the Turkish dotless ı (316 words, in non-English text) and Greek letters. |
| 2 | R08-2: the `format+typography+collapse_whitespace` view is applied the same way at training, calibration and inference; records and probe use it; nothing is flagged | **partly met (R09-1, R09-6)** | **Natural text:**<br>- Training: `train.py:89` → `neural._prep`.<br>- Calibration: `calibrate._detector` → `detector._prep`.<br>- Inference: `detector.py:128`. Windows are cut from the prepared text.<br>**Records:** both training records and the registry carry the view.<br>**Probe:** it reads the view and flags nothing. My reconstruction of its own-probe AUCs matches exactly: dolly 0.8386, mirage 0.8393, hc3 0.8287.<br>**Not met:** the strip is anchored to line starts. One-line text, and every excerpt (built after whitespace collapse), keeps its list, heading and rule markers. That covers 81,929 of short-v2's 111,607 training rows and 18,253 of the 44,448 calibration rows. |
| 3 | R08-4: `registry.seal` binds the result to a digest of the scored entry; `verify()` rejects edits, a missing view and missing hashes | **met** (residuals R09-3, R09-4) | **Committed registry:** `verify` gives `(True, 'validated')`. The recomputed `record_sha256` equals the stored `b374eff5…`.<br>**17 in-memory edits** each give `(False, 'registry changed after the locked benchmark')`: a policy band, a main band, Platt, the OOD threshold, the short model's Platt, ranges, OOD and max_tokens, max_tokens, the view changed, the view removed, `files_sha256` removed (each model), `policy_version`, the short model dropped, the seal removed and `corpus_version`.<br>**Not bound:** the benchmark's own fields, and the code. |
| 4 | R08-5 and the relock: the reason is recorded, rows and splits are unchanged since `010f700`, and the benchmark checks the new line format | **met** | **History:** `test.lock.json` history holds the relock reason, and `previous` = `7ba11a78…`.<br>**Old format:** the current manifest in the old `sample_id\|split\|label` format hashes to `7ba11a786f19…`, the `010f700` lock.<br>**Before vs after:** the builder's pre-relock manifest copy also hashes to `7ba11a78…`. It equals the current manifest in all 17 columns for all 260,337 rows, with 0 changes.<br>**New format:** hashed over locked text, it gives `15d9b9a4…` = the lock. The counts match.<br>**Benchmark:** `benchmark.py:143-147` hashes with source and text. |
| 5 | Order and independence | **met** | **Timeline (IST):**<br>- 01:46:17: `061bda0`.<br>- 01:46:35–02:08:52: relock build. Its lock entry is stamped 02:03:19.<br>- 02:09:33: `bc00087`.<br>- 02:09:38–03:27:21: training.<br>- 03:27:48–03:32:13: calibration.<br>- 03:32:51: `72188b4`.<br>- 03:33:04–03:49:14: benchmark (`started_at` 22:03:10Z).<br>- 03:51:23: `f9d143a`.<br>**Main record:** 115,086 train rows, and `train_ids_sha256` `213465df…` is the digest of the manifest's train ids. The 13,844 validation rows are `selection_data`'s sample, which I rebuilt.<br>**Short record:** 111,607 rows, 81,929 excerpts, 13,460 validation rows and `950e54c7…`, all reproduced exactly.<br>**Separation:** 0 locked ids, groups or clusters are in train/val. Calibration scored 44,448 validation texts (26,195 + 18,253 excerpts) and a train bank.<br>**One run:** a single v3 run, with `rerun_reason` null. No `src/` change after `061bda0`. No builder script between the lock and the run reads a locked split. |
| 6 | Gate arithmetic | **met** (caveat R09-5) | **Bounds:** Clopper-Pearson, recomputed, matches every slice to 1e-9.<br>**Gated false AI:** aita 0.0074, no_robots 0.0077, mirage 0.0060, hc3 0.0032, raid 0.0024, mage 0.0015, stackexchange 0.0010. All are ≤ 0.01.<br>**Gated false human:** the worst is mage at 0.0314, ≤ 0.10.<br>**Ungated:** oasst1 (188 human / 80 AI).<br>**Coverage:** 12,928 / 23,415 = 0.5521, ≥ 0.40. The pass is correct.<br>**Margins:** AITA would fail at 4 false AI (it has 2); No Robots, MIRAGE and HC3 at 5 (3, 2 and 0); RAID at 7. |
| 7 | Tests | **met**, with a test gap (R09-1) | `255 passed in 24.00s`. With an empty `CORPUS_DIR`: `255 passed in 21.10s`. |

## Findings

### R09-1: The format view only strips markers at line starts, so the short-text specialist learns list formatting from excerpts, and one-line lists still move decisions

- **Severity:** High
- **Category:** validity / shortcut (R08-2 not fully closed)
- **Location:**
  - `src/corpus/text.py:43-44`: `_LINE_MARKUP` and `_RULE` are anchored with `^`/`$` under `re.M`, so they only match at line starts.
  - `src/authorship/excerpts.py:29`: `excerpt()` runs `collapse_whitespace` first, so an excerpt has no lines left when the view runs.
  - `src/authorship/train.py:83-86`: short-v2 trains on `short_frame`.
  - `src/authorship/calibrate.py:181,199`: the main and short models are calibrated on validation plus excerpts.
  - `tests/test_authorship_support.py:91`: the view test covers multi-line input only.
- **Repro steps:**
  1. `v = text_view("format+typography+collapse_whitespace")`
  2. Compare `v(multi)` with `v(multi.replace("\n", " "))` for a small list.
  3. Run `excerpt()` on a numbered list, then `v`.
  4. On v3 **validation**, build `short_frame` excerpts. Count the markers left in the view, by label.
  5. Score the 275 excerpts with two or more inline `N. ` markers with short-v2 (CPU), with and without the markers.
  6. Rewrite validation texts as numbered and dash lists, one item per line or all on one line, and run them through the full service.
- **Expected vs actual:** Amendment 6 says the view drops list markers "so formatting can't stand in for authorship in any source". The report says the view is "used in training, calibration, inference". Instead:
  - The same list gives two model inputs:
    ```
    multi-line  -> 'Here is how to do it: Preheat the oven to 200 degrees. Mix the flour and water. Add salt. Add pepper. Notes Serve warm.'
    one line    -> 'Here is how to do it: 1. Preheat the oven to 200 degrees. 2. Mix the flour and water. - Add salt. - Add pepper. ## Notes Serve warm.'
    excerpt of a numbered list -> 'Step number 11 tells you to do a thing carefully. 12. Step number 12 tells you ... 13. Step number 13 ... 14'
    ```
  - A real validation excerpt reads: *"…in 3-space: 1. Find the Direction Vector: The direction vector…"* (OpenAssistant AI).
  - Inline numbered markers in excerpts (AI% vs human%): RAID 8.7 vs 1.0, Beemo 14.5 vs 9.0, OpenAssistant 21.5 vs 16.5, Dolly 10.7 vs 9.1. On natural text, which the probe sees, the figures are RAID 1.7 vs 3.4 and OpenAssistant 2.0 vs 2.0.
  - **Short-v2 relies on them.** 187 AI excerpts scored with markers, then without:
    - The mean probability falls from 0.790 to 0.524.
    - RAID (128 texts) falls from 0.922 to 0.474.
    - Decisions go from 21 Likely AI and 2 Likely human to 2 Likely AI and 36 Likely human.
  - Same validation texts, full service, 25–99 words (short model):

    | Format | Human texts (n=310): AI / human / inconclusive | AI texts (n=321): AI / human / inconclusive |
    |---|---|---|
    | plain prose | 0 / 114 / 196 | 22 / 6 / 293 |
    | numbered, one item per line | 0 / 119 / 191 | 17 / 6 / 298 |
    | numbered, all on one line | 0 / **74** / 236 | **7** / 3 / 311 |
    | dash items, one per line | 0 / 114 / 196 | 22 / 6 / 293 |
    | dash items, all on one line | 0 / 197 / 113 | **0 / 36** / 285 |

  - For the main model (100–199 words, AI n=332), a one-line dash list takes Likely AI from 92 to 75.
- **Evidence:**
  - 73% of short-v2's training rows (81,929 of 111,607) and 74% of its calibration rows (18,253 of 24,531) are excerpts.
  - Short-v2 serves every 25–99-word text, in production and in the benchmark.
  - Spreadsheet cells and pasted text often put a list on one line. Reformatting AI text as a one-line dash list is a cheap way to get Likely human: 36 of 321 texts.
  - A one-line numbered list pulls human text toward AI: 114 Likely human becomes 74. None crossed into Likely AI in 310 texts.
  - The build's probe reads natural text only, and it has no marker features, so it cannot see this.
- **Confidence:** high on the mechanism and the measured shifts. Medium on how often production inputs trigger it.

### R09-2: Cheap character disguises outside the new lists still suppress detection, with no disguise note

- **Severity:** Medium
- **Category:** robustness / evasion (R08-1 residual)
- **Location:**
  - `src/detection/obfuscation.py:40-41`: `_INVISIBLE` lists only assigned default-ignorables. The unassigned ones (U+2065, U+FFF0–FFF8, U+E0000–E0FFF outside the tags) are also Default_Ignorable_Code_Point and are meant to render as nothing.
  - `src/detection/obfuscation.py:42-47,72-73`: any letter named LATIN… that is not in the short list is kept. That includes ligatures ﬁ ﬂ ﬀ ﬃ (NFKC compatibility forms, so the module's docstring promises them), `ǀ` U+01C0 and `ɾ` U+027E.
  - `src/detection/obfuscation.py:100`: one unmapped non-Latin letter (Armenian `հ`, `ո`) returns the whole word as written, mapped lookalikes included.
  - Spacing and punctuation lookalikes are never touched: U+200A hair space (Zs, so it splits the word) and U+2024 one-dot leader for `.`.
- **Repro steps:**
  1. Take the 84 v3 **validation** AI texts (50–199 words) that the registered service (`verify` = validated) labels Likely AI. They come from 422 hash-sampled texts across 8 sources.
  2. Apply each edit below.
  3. Run `check_batch`.
- **Expected vs actual:** Disguise should be undone, or at least reported. Instead, each row below had no `OBFUSCATION_NORMALIZED` note, except the shield row (a note in all 84):

  | Edit (84 texts, 83 Likely AI unedited) | Likely AI | Likely human | Inconclusive |
  |---|---|---|---|
  | ligatures ﬁ ﬂ ﬀ ﬃ ﬄ everywhere (looks identical) | 67 | 0 | 17 |
  | ligatures + U+2024 for every `.` (looks identical) | **15** | 0 | 69 |
  | U+200A hair space mid-word, 10% of words | 29 | 1 | 54 |
  | U+200A hair space mid-word, 30% of words | 6 | **4** | 74 |
  | `l`→`ǀ`, `r`→`ɾ` (named LATIN), p=0.15 | 31 | 0 | 53 |
  | Armenian `n`→`ո`, `h`→`հ`, p=0.15 | 20 | 0 | 64 |
  | shield: Cyrillic о е а p=0.15 + Armenian `հ` | 14 | 0 | 70 (note in all 84) |
  | unassigned default-ignorable U+E0080, 30% of words | 0 | 2 | 82 |
  | C1 control U+0090, 30% of words | 3 | 3 | 78 |

- **Evidence:**
  - Model-free, each edit changes the token ids with `changed_by_deobf=False` and passes the English gate. For example, ligatures take one sentence from 24 to 54 tokens, `latin=1.0`, `english=True`.
  - Most outcomes are abstentions (abstention band or OOD). That meets R08-1's "normalized or abstain" bar, but for the pipeline it is still a lost flag, and no reviewer is told about the disguise.
  - 1–4 of 84 become Likely human.
- **Confidence:** high

### R09-3: The seal binds the text view's name, not its code, so the scoring code can change after the pass and the gate stays open

- **Severity:** Medium
- **Category:** gate integrity (R08-4 class)
- **Location:**
  - `src/authorship/registry.py:118-141`: `record_sha256` covers registry fields only.
  - `src/corpus/text.py:52-57`: views are looked up by name.
  - `src/authorship/benchmark.py:159-161,169-171`: neither the started entry nor the report records a commit or a code digest.
- **Repro steps:** In memory:
  1. `from src.corpus import text as T; T._TEXT_VIEWS["format+typography+collapse_whitespace"] = lambda t: t`
  2. `registry.verify(registry.load())`
  3. `registry.service_from_record(...)`
- **Expected vs actual:** The gate should open only for the system the locked test measured. Output:
  ```
  view code replaced by identity; verify: (True, 'validated')
  service validated: True | main view of '**Bold** - x': '**Bold**\n- x'
  ```
- **Evidence:**
  - `grep -c "commit\|git_sha\|code_sha"` on `benchmark.json`, `registry.json` and `training_record.json` gives 0, 0 and 0.
  - The same holds for `obfuscation.py`, `policy.py`, windowing and the language gate.
  - The obvious fix for R09-1 (strip markers anywhere) would silently change v4's inputs while the registry still says validated.
- **Confidence:** high

### R09-4: The benchmark block itself is not sealed

- **Severity:** Low
- **Category:** gate integrity
- **Location:**
  - `src/authorship/registry.py:115`: `_RUN_FIELDS` leaves `benchmark` out of the digest.
  - `src/authorship/registry.py:134-135`: `verify` trusts `benchmark.passed`.
- **Repro steps:**
  1. Seal a copy of the committed entry with `passed=False` and a failure.
  2. Set `passed=True`.
  3. Run `verify`.
- **Expected vs actual:** A result edited after the run should not open the gate. Instead:
  ```
  failed run as sealed: (False, 'locked benchmark not passed')
  same entry, passed flipped to True (failures still listed): (True, 'validated')
  ```
- **Evidence:**
  - Clearing `failures`, `history` or `report` also keeps `validated`.
  - Nothing ties the entry to `benchmark.json` (no digest). An unkeyed digest can't stop a deliberate re-seal either, which also verifies.
- **Confidence:** high

### R09-5: The gate passes Stack Exchange with 4.6% of its AI detected

- **Severity:** Low
- **Category:** validity (disclosed)
- **Location:**
  - `src/authorship/benchmark.py:106-117`: false human counts only confident mislabels, and coverage is pooled.
  - `data/benchmark/v3/model/benchmark.json`
- **Repro steps:** Read `test.by_source`.
- **Expected vs actual:** The new gated slice was built to test assistant-voice text in both directions. Instead:
  - Stack Exchange AI gives 38 detected, 13 labelled human and about 773 inconclusive out of 824, so its false-human bound (0.025) passes.
  - Outside MIRAGE and AITA, 959 of 5,468 test AI texts get a decisive label (17.5%).
  - Coverage without MIRAGE is still 49.9%, carried by Likely-human calls on human text.
- **Evidence:**
  - These are the pre-registered criteria, and the commit message, README and report state the 4.6% plainly.
  - The pass supports the false-AI contract. It says little about detection on assistant-style text.
- **Confidence:** high

### R09-6: The build's probe still can't see formatting, and one typing cue lifts HC3 over 0.85

- **Severity:** Low
- **Category:** validity / probe coverage
- **Location:**
  - `src/corpus/confound.py:26-28`: no marker or dash features.
  - Amendment 6: "every source is under 0.85".
- **Repro steps:**
  1. Fit the build's own-source probe (train → val, core rows, model view).
  2. Fit it again with formatting indicators added: `*`, `N.`, ` - `, `---`, backtick, `#`, `|`, `[Title]`, the colon rate, and the item rates.
- **Expected vs actual:** Amendment 6 says every source stays under 0.85. Instead:
  - Own-probe AUCs: hc3 **0.8287 → 0.8584**, dolly 0.8386 → 0.8489, mirage 0.8393 → 0.8447.
  - Dropping the ` - ` indicator returns hc3 to 0.8488. HC3 train human text has ` - ` in 17.3% of texts, against 1.8% for AI.
- **Evidence:**
  - The typography map turns an AI "word—word" into "word-word", while a typed "word - word" keeps its spaces.
  - The builder's `probe_format.py` used rate features and got under 0.85. The result depends on how the features are chosen.
  - The same cue probably drives the one-line dash-list evasion in R09-1.
- **Confidence:** medium

### R09-7: The status docs call 4c done before this review returned, and Amendment 6 postdates its commit

- **Severity:** Low
- **Category:** process integrity
- **Location:**
  - `roadmap.md:21` ("done (passed; gate open)") and `roadmap.md:39` ("Ralph R09 reviewed the pass").
  - `increment-4c-report-2026-09-28.md:98`.
  - `increment-4c-preregistration.md:167`.
- **Repro steps:**
  1. `git log --format='%h %ad %s' --date=iso -3`: `bb5cf48` is at 03:54:06, before this report existed.
  2. `git show 061bda0`.
- **Expected vs actual:** Under the Ralph method, a phase gate can't pass while a Critical or High finding is untriaged. Instead, the roadmap and README mark 4c done and cite R09 in advance. Separately, Amendment 6 is headed "01:50", but its commit is 01:46:17 (`amend6.py` was written at 01:45:26). This is the R08-9 class of error again.
- **Evidence:** as listed. The data and the benchmark are unaffected.
- **Confidence:** high

### R09-8: `format_removed` counts U+180E twice

- **Severity:** Low
- **Category:** technical soundness
- **Location:** `src/detection/obfuscation.py:77`. U+180E is category Cf and also lies in `_INVISIBLE`'s `᠋-᠏`.
- **Repro steps:** `deobfuscate("bene᠎fit")`
- **Expected vs actual:** The count should be 1. The result is `format_removed=2` for one character, so the evidence value `format_characters_removed` is inflated.
- **Evidence:** from the model-free run: `Mongolian U+180E (Cf) changed_by_deobf=True removed=2`.
- **Confidence:** high


## Triage (build chat, 2026-09-28)

R09-1 is a bug in a pre-registered fix, so the models are retrained on the corrected view and the locked run is repeated once, with its reason recorded. Thresholds, criteria and data are unchanged. Details are in Amendment 7. The suite is now 262 tests, all passing.

| ID | Triage | What changed |
|---|---|---|
| R09-1 | **Fix, retrain, rerun** | `excerpt()` cuts from the model's view of the whole text. New view `format2+typography+collapse_whitespace` also drops bullets anywhere and free-standing dashes; earlier view names keep their behaviour. New tests cover excerpts of list text, inline markers, and idempotence. |
| R09-2 | **Fix** | Ligatures and leader dots are NFKC-folded. Thin spaces inside words are removed, as are plane-14 default-ignorables and C0/C1 controls. ǀ, ɾ, ո, հ, ց and զ are mapped. Mixed-script words raise the note (`mixed_script_words`). |
| R09-3 | **Fix** | `text_view_fingerprint` is recorded in the training records and registry, and `verify()` rejects a changed view. |
| R09-4 | **Fix** | The seal covers passed, failures, ran_at, report and `report_sha256`. |
| R09-5 | **Acknowledged** | Detection limits are reported. |
| R09-6 | **Fix** | The probe has formatting features on the model's view; the rebuilt probe is of record. |
| R09-7 | **Corrected** | Roadmap and report wording, and Amendment 6's time. |
| R09-8 | **Fix** | Removal is counted by length. |
