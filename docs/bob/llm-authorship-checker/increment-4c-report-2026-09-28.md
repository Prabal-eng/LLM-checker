# Increment 4c report — 2026-09-28

## Result

**`distilroberta-authorship-v5`, with the short-text specialist `distilroberta-authorship-short-v3`, passes the fresh v3 locked test. The validation gate is open.**

- **This is the fourth run on v3, the third recorded rerun (run 4, 08:27–08:47). All four runs passed.**
  - Run 1 (v4 with short-v2) was followed by Ralph R09, which found list markers inside short-model excerpts. Amendment 7 fixed them and retrained the models as v5 and short-v3 for run 2.
  - Ralph R10 and R11 then found cheap character disguises that turned Likely AI into Likely human. Amendments 8 and 9 replaced the per-character approach: accents are read back, what can be read is normalized, and the service abstains only on dense unreadable text. That gave runs 3 and 4, with the same models and identical thresholds.
  - Across all four runs, thresholds, criteria and data are unchanged.
- Run 4 reproduces run 2's figures: 3 test texts abstain as `DISGUISE_SUSPECTED` (run 3: 58), coverage is 66.9% and AI detected is 65.2%. The tables below apply to run 4.
- The pass criteria are unchanged since v1.
- The registry's pass is sealed to the exact entry it scored, the view's behaviour fingerprint, the result and the report's digest. `registry.verify` returns `validated`.

| Gate (v3 `test`, full service) | Bound | Run 4, the result of record (run 2 identical) | Run 1 |
|---|---|---|---|
| False AI, every source slice with ≥300 human texts | 95% upper ≤ 1% | ✅ **18 / 11,319**; worst No Robots 3/1,003 (0.77%) and AITA 2/854 (0.74%) | ✅ 8 / 11,319; worst 0.77% |
| AI labelled human, every source slice with ≥100 AI texts | 95% upper ≤ 10% | ✅ worst MAGE 48/1,653 (3.7%) | ✅ worst 3.1% |
| Decisive coverage of supported text | ≥ 40% | ✅ **66.9%** | ✅ 55.2% |

**By source (run 2; run 1 in brackets):**

| Source | Human flagged AI | 95% upper | AI labelled human | AI detected | Decisive |
|---|---|---|---|---|---|
| Stack Exchange, pre-ChatGPT answers (new, never fitted) | 5 / 2,973 (0) | 0.35% | 15 / 824 | **14.6%** (4.6%) | 60.3% |
| No Robots, human assistant-style answers | 3 / 1,003 (3) | 0.77% | — | — | 30.5% |
| AITA, same-title posts by 2026 models | 2 / 854 (2) | 0.74% | 0 / 854 | 87.0% (88.2%) | 87.0% |
| MIRAGE | 1 / 1,044 (2) | 0.45% | 1 / 6,897 | **86.3%** (66.2%) | 83.8% |
| HC3 | 0 / 937 (0) | 0.32% | 2 / 1,150 | **46.7%** (17.9%) | 62.9% |
| MAGE | 1 / 3,075 (1) | 0.15% | 48 / 1,653 | **30.9%** (18.4%) | 52.2% |
| RAID | 0 / 1,245 (0) | 0.24% | 21 / 1,761 | **42.1%** (18.6%) | 56.9% |
| OpenAssistant, fresh trees (report only: 188 human texts, under the 300 a gated slice needs) | **6 / 188** (0) | 6.2% | 3 / 80 | 15.0% | 23.7% |

- **AI detection overall is 65.2%**, up from 46.9% in run 1 (v2's model reached 71%).
- Removing the list-marker shortcut raised detection almost everywhere.
- **OpenAssistant moved the other way:** 6 of its 188 fresh human replies are flagged. That is far below v2's 19.7%, but not zero. Too few texts exist to gate it, so it stays the slice to watch.

## Why there are four runs

1. **Run 1** (03:33–03:49): v4 with short-v2 on the Amendment 6 view. It passed.
2. **Ralph R09** (`ralph/09-increment-4c-final.md`) confirmed the pass arithmetic, the seal and the ordering. It found one high bug:
   - The view removed list markers only at line starts, while excerpts were cut from whitespace-collapsed text.
   - The short model learned inline list markers from the RAID and OpenAssistant excerpts.
   - On validation, a one-line dash list turned 22 Likely AI into 0.
3. **Amendment 7** fixed the view (`format2+typography+collapse_whitespace`) and cut excerpts from it. It also fingerprinted the view in the registry, sealed the result itself, and extended disguise handling.
4. The models were retrained as v5 and short-v3, so v4's files stay intact, then recalibrated.
5. **Run 2** (06:14–06:31) was the recorded rerun. Its reason is in the registry history, and it passed.

6. **Ralph R10** (`ralph/10-increment-4c-rerun.md`) confirmed the rerun's legitimacy and that R09-1 is fixed. It found cheap disguises: small roman numerals, unassigned characters, thin-space runs and i + dot above turned up to a third of confident AI into Likely human.
7. **Amendment 8** closed disguise as a class:
   - it normalizes what it can read back and abstains on what it can't;
   - it bound the scoring code, every locked run and the report digest to the seal.

   Recalibration from cached scores reproduced every threshold.
8. **Run 3** (07:26–07:43) was the second recorded rerun, and it passed.
9. **Ralph R11** (`ralph/11-increment-4c-final.md`) confirmed run 3. It found that marks composing with their letter (a dot below), letter-shaped symbols and foreign digits still flipped labels silently.
10. **Amendment 9** changed accents from judged to read back. It made abstention density-based (at least 3, or 2% of words) and added the Platt map to the code digest.
11. **Run 4** (08:27–08:47) was the third recorded rerun, and it passed. Only 3 test texts abstain as disguised.

The pre-registration allows reruns only for bugs. No fix was chosen by looking at a run's test results. All runs are reported. Runs 1–3's reports are `benchmark-run1.json`, `benchmark-run2.json` and `benchmark-run3.json` in `data/benchmark/v3/model/`.

## What the pass does not mean

- **AI detection is still uneven.**
  - Stack Exchange AI answers are 14.6% detected. They are mostly `SCORE_IN_ABSTENTION_BAND` (1,493) or `CALIBRATED_LOW_SCORE`; only 25 were out of domain.
  - Local Meta and Mistral answers reach 44% and 37%, against 86–95% for DeepSeek and Anthropic.
  - The thresholds are strict: a text needs a score of about 0.97 to be called AI in the long bands.
  - An `Inconclusive` answer is not evidence of human writing.
- **Short text.**
  - Natural test texts of 25–49 words: 13.2% of AI detected (run 1: 1.6%). At 50–99 words: 8.0%.
  - On excerpts of test texts (report only), the routed system detects **57.0%** at 25–49 words, with 11 of 1,469 human excerpts flagged (0.75%), and 34.0% at 50–99, with 4 of 6,152 flagged. The main model alone manages 23.2% and 23.9%.
  - v2's routed system detected 52% and 54%, with 0.8% and 0.3% false AI.
- **Unseen generators (report only).** 663 of 1,511 detected and 58 labelled human.
  - Moonshot 84%, Microsoft Phi 46%, IBM Granite 28%, BLOOM 28%, Cohere 12%.
  - Cohere has 48 of 423 labelled human (upper 14.2%).
- **Newest 2026 models (rasbt50k, diagnostic only).** Alibaba 46%, Google 34%, Moonshot 19%, DeepSeek 15%, OpenAI 7%.
- **In-house rater text (unverified labels, report only).**
  - 50 of 3,770 cells are flagged: 38 of 1,621 never viewed and 12 of 2,149 previewed. Run 1 flagged 5. No combined six-rationale row is flagged (0 of 642).
  - A flag may be real AI use or a false alarm on exactly the text this tool is for. Without verified rater labels (debt D6), that can't be told apart. The sponsor should review these flags before the checker is used on rater comments.
- **Stress (report only).**
  - LLM polish: 5,262 of 19,819 detected, 714 labelled human.
  - LLM rewrite: 4,697 of 21,221 detected, 701 labelled human.
  - The RAID attack sets are nearly gone (D13).
- **Also reported.**
  - Human essays: 0 of 393 flagged.
  - rasbt50k human rows: 20 of 21,857.
  - Peak inference VRAM: 1.47 GB.

## What was built

| Part | Where |
|---|---|
| Local paired generation | `src/corpus/generate.py`: 4,487 answers from six local models (Ollama). Every record stores its model digest, options, seed and licence. |
| Stack Exchange test source | `src/corpus/sources/stackexchange.py`: answers from 2021 or earlier, 8 non-code sites, test only. Phi-4-mini and Granite 4 are held out. |
| Artifact filters (Amendment 4) | `generate.artifact`: 240 answers dropped (5.3%). Plus a provenance backfill. |
| Split fixes (Amendment 5) | Retirement by group, and splits settled against the audit's reclustering. |
| Model text view (Amendments 5–7) | `src/corpus/text.py`: `format2+typography+collapse_whitespace` drops markdown, list markers (line-leading and inline), bullets, free-standing dashes, curly quotes and dashes. Earlier names keep their behaviour, and `text_view_fingerprint` is checked by the registry. Excerpts are cut from the view. |
| Disguise (R08-1, R09-2, R10-1, R11-1) | `src/detection/obfuscation.py` reads text back. It removes every format, unassigned, private-use, surrogate, default-ignorable and control character. It drops accents from Latin letters and removes orphan marks and near-invisible spaces inside words. It folds ASCII-bound NFKC forms and maps lookalike letters, stroked letters and letter-shaped symbols. The service abstains (`DISGUISE_SUSPECTED`) only when unreadable content is dense: 0.044% of ordinary train/val text and 3 test texts. |
| Gate binding (R08-4, R09-3, R09-4, R10-3, R10-5) | `registry.seal` and `verify()`: bound to the scored entry, the result, every locked run, the report digest (independent of line endings, and checked against the file), the view fingerprint and a digest of the scoring code. Config and tokenizer hashes are required. |
| Lock (R08-5) | Lines carry each row's source and exact-text hash. v3 was relocked once, with its reason. |
| Corpus v3 | 260,337 samples. The test has 11,319 human and 13,219 AI texts, plus 1,511 unseen-generator texts. The probe, with 19 features on the model's view, flags nothing (dolly 0.838, hc3 0.824, mirage 0.839). |

The test suite has 275 tests and needs neither a GPU nor model files.

## What went wrong, and what changed

1. **The generation chain died once** when its visible window was closed. The 4 GB GPU forced Qwen down to 2B and one request at a time for the large-vocabulary models (Amendments 2 and 3).
2. **Prompt artifacts:** Phi cut off 22.5% of its answers, Gemma restated lengths, and Ministral appended word counts. All were filtered (Amendment 4).
3. **The first build was refused by its audit** (retirement by id; boilerplate links seen only by the audit). The probe flagged dolly and mirage (typography, and MIRAGE's damaged emails). These were fixed in Amendment 5.
4. **Ralph R08 failed the build** (2 high: formatting, and disguise). Training was stopped at epoch 0 and the models retrained (Amendment 6).
5. **Ralph R09 found list markers inside excerpts** in the passing system. The fix, retrain and recorded rerun are in Amendment 7.
6. **Ralph R10 found cheap disguises** after three rounds of per-character fixes. Amendment 8 replaced the approach with a class-level one (normalize or abstain) and made a second recorded rerun.
7. **Ralph R11 found diacritics that composed into 'ordinary' letters.** Amendment 9 reads accents back instead of judging them, and made a third recorded rerun.

## Review caveats

- **Ralph R08** (`ralph/08-increment-4c.md`): FAIL, triaged. R08-3 stands: the sources that motivated 4c can't fail the gate, so OpenAssistant's figure is report-only.
- **Ralph R09** (`ralph/09-increment-4c-final.md`): FAIL, 1 high. Triaged in Amendment 7.
- **Ralph R10** (`ralph/10-increment-4c-rerun.md`): FAIL, 1 high. Triaged in Amendment 8; R10-2 and R10-7 become debt D14 and D15.
- **Ralph R11** (`ralph/11-increment-4c-final.md`): FAIL, 1 high. Triaged in Amendment 9. Run 4 was not re-reviewed: character disguise has been an open-ended search, and further cases belong to the next increment.
- **Validation was seen during training.** Epoch validation results were seen during every training run. No threshold or criterion was set from test data.
- **Test text was seen during diagnosis.** Diagnosing the first build printed snippets of some first-build test rows. The read was model-free (Amendment 6).
- **Not comparable with v2.** v2 numbers aren't directly comparable: the R07–R09 fixes changed the system, and part of v2's MAGE result measured memorised text.

## Next (options, sponsor to decide)

1. **Review the in-house flags and get verified rater labels (D6)** before relying on the checker for rater comments.
2. **Raise AI detection at the same false-AI contract** (proposed Increment 4d), with a fresh locked test. Options: a larger model within 4 GB, more varied generator answers across registers, and register-aware thresholds.
3. **Frontier coverage.** Add current frontier-model text across registers. This conflicts with local-only, and the providers' terms need checking.
4. **Rebuild adversarial stress sets from never-tested parents (D13).** Also: token-dense windows (D10), short MAGE containment (D11), a trailing-fragment trim (D12), more list glyphs and `(1)` markers (D14), and em-dash hyphens (D15). D14 and D15 need the next retrain.
5. **Increment 5:** the web checker, a three-way workbook report and operational hardening.

## Reproduction

```bash
export HF_HOME=D:/llm-authorship-corpus/hf-cache
python -m src.corpus.generate                      # resumable; OLLAMA_URL selects the server
python -m src.corpus.generate --backfill-provenance
python -m src.corpus.build                         # v3 is locked (test.lock.json history has the relock)
python -m src.authorship.train --epochs 2
python -m src.authorship.train --variant short
python -m src.authorship.calibrate --max-false-ai 0.008 --max-false-human 0.08
python -m src.authorship.benchmark --rerun-reason "..."   # v3 has run four times; see the registry history
```

Records, none containing text:
- `data/benchmark/v3/` holds the build record, the lock, both models' training records, calibration, and all four runs' reports.
- `models/registry.json` holds the sealed entry and every locked run.
- The pre-registration and Amendments 1–9 are in `increment-4c-preregistration.md`.
