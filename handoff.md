# Handoff: excel-ai-qa-pipeline

**Updated:** 2026-10-02. **Read first:** this file, then the [README](README.md), then the
[Increment 4d report](docs/bob/llm-authorship-checker/increment-4d-report-2026-10-01.md).

## In one paragraph

No authorship model is validated right now. On 2026-10-01, the Increment 4d locked head-to-head on corpus v4
ended `close_gate`: neither the new DeBERTa-v3-large models nor the served DistilRoBERTa v5 passed the gate.
The served gate is closed, so the default provider returns `Inconclusive` (`MODEL_NOT_VALIDATED`). For review
work, the app can show either model's verdicts as a marked `UNVALIDATED PREVIEW`. The audit workbook was
rebuilt the same day (v2): one verdict tier per row, the bar beside every score, trigger sentences, an Agents
sheet and a Technical sheet. On 2026-10-02 it gained trigger words and a sheet-wide Trigger Words table. All of this is on branch `increment-4d`, which is pushed but not merged.

## Where things are

**Branches** (github.com/inv-google/excel-ai-qa-pipeline, private):

| Branch | What it holds |
|---|---|
| `main` | The initial commit only |
| `llm-authorship-checker` | The checker through Increment 4c. Its served registry was re-sealed `passed: false` after 4d (`ab62956`). It also has the audit-by-locale feature (`ae3d46c`). |
| `increment-4d` | **This branch.** Everything on `llm-authorship-checker` up to 4c, plus corpus v4, the 4d models' records, the head-to-head, the unvalidated previews and audit workbook v2. Its own copies of the locale and CI-skip commits are `491005f` and `8defe3a`. |
| `llm-qa` | A separate workstream, the LLM QA critic. Read `handoff-llm-qa.md` on that branch. Its standalone export is github.com/inv-google/auto-qa. |

**On the author's machine:**
- **Main checkout:** `excel-ai-qa-pipeline`, on `llm-authorship-checker`.
- **Worktree `../excel-ai-qa-pipeline-4d`:** on `increment-4d`. The web app runs from here.
- **Worktree `../excel-ai-qa-pipeline-llm-qa`:** on `llm-qa`.
- **Corpus text and model weights:** `D:\llm-authorship-corpus\` (`CORPUS_DIR`), on an external USB drive. Ollama models are in `D:\ollama\models`.
- **Hardware:** a 4 GB GPU and tight RAM. GPU memory spilling into shared memory came before two hard shutdowns, so run one GPU job at a time. Set `CUDA_VISIBLE_DEVICES=-1` to hide the GPU.

## State

### Authorship model: gate closed

| | 4d (DeBERTa-v3-large LoRA, main + short) | v5 (DistilRoBERTa, served until 2026-10-01) |
|---|---|---|
| AI detected (v4 test) | **37.8%** | 24.5% |
| False AI | **35 of 20,730** (upper bound 0.22%) | 67 (0.40%) |
| 25–99 words, AI detected | **40.6%** | 10.4% |
| Gate | fail: No Robots false AI 5 of 553; WasItAI missed AI (upper bound 19.7%) | fail: MAGA, No Robots, WasItAI |
| Improvement rule | fail: +12.9 points pooled, but HC3 −31 and MAGE −14 | – |

- **Registries:**
  - On `increment-4d`, `models/registry.json` holds the 4d record, sealed `passed: false`.
  - `models/incumbent_registry.json` is v5, frozen as the head-to-head incumbent.
  - On `llm-authorship-checker`, the v5 record is re-sealed `passed: false`.
- **The v4 locked test is spent.** Any new attempt needs a corpus v5 with a fresh locked test.

### The audit app (on `increment-4d`)

- **Providers:**
  - `trained`: the registered model, gate closed, so always `Inconclusive`.
  - `preview-unvalidated`: the 4d model, the dashboard default.
  - `preview-v5`.
  - `gptzero` and `originality`: paid and unvalidated.

  Preview runs carry an `UNVALIDATED PREVIEW` banner that names the failed slices.
- **Audit workbook v2:**
  - **Design and plan:** the [spec](docs/superpowers/specs/2026-10-01-audit-workbook-v2-design.md) and the [plan](docs/superpowers/plans/2026-10-01-audit-workbook-v2.md).
  - **Code:** `src/report/` holds `tiers.py`, `tier_rates.py`, `triggers.py`, `rows.py` and `excel_writer.py`.
  - **No decision, score or threshold changed.** Tiers only rank rows below the calibrated bar.
- **Trigger words (2026-10-02):**
  - Shares are taken of the AI signal (log-odds), so they still move near 99%.
  - A sentence counts at 20% of the signal; words at 5%, from the 3 sentences that matter most.
  - The Trigger Words sheet counts how many rows contain each phrase, to spot wording the model over-reacts to.
  - Feasibility check on 51 public validation texts scoring 75%+: 47 got trigger words, and only 2 had no word above 3%. The new measure named a sentence in 33 rows, against 18 under the old 10-point rule.
  - On high-scoring human texts, one word often takes away more than all of the signal. The model's human false positives can hinge on single words.
- **Tier rates:**
  - Stored in `data/report/tier_rates.json`, measured on the 4d model's validation scores (public writing, not rater text).
  - Rebuild with `python -m src.report.tier_rates` after any recalibration or band change.
- **Real run.** This was the fr_FR tab of a Personality Autorater Meta-Eval workbook, with the human and AR rationale columns. Counts only:
  - 117 Likely AI, 88 High AI score, 27 Low, 2 not checked.
  - With trigger words: 240 s in total, 175 s of it for triggers; peak GPU memory 2.8 GB. Before words, the audit took 84 s, 17 s of it for triggers.
  - Of the 205 rows scored High AI or above, 128 got a trigger sentence (58 under the old rule), 197 got trigger words, and 5 got neither.
  - 5 trigger phrases appear in half or more of all rows. The most frequent one triggered in 17 rows and appears in 74% of rows, which looks like shared wording rather than agent behaviour.
  - With the same columns as an earlier audit, Likely AI matched it exactly (71 = 71).
- **Tests:** `python -m pytest`, 603 pass.
- **Start the app:** from the 4d worktree, `python -m uvicorn src.api:app --port 8000`, then open `http://localhost:8000`. It is stopped now.

## Decide next (the user's calls)

1. **Integrating `increment-4d`.** A merge into `llm-authorship-checker` brings all of 4d, not only the app work.
   - **What comes with it:** corpus v4 code, the head-to-head guards, the 4d registry record and the previews.
   - **Expect conflicts:** the two branches have diverged by 3 commits: the registry re-seal, plus two that 4d holds as its own copies.
   - **Untracked `handoff.md`:** the main checkout has one, and it would block a merge that adds this file. Move it aside first.
   - **The narrower option:** cherry-pick the preview commits (`6aaf253`, `001e212`, `dda1c0a`) and workbook v2 (`5fbbbf9` through `377e2c4`).
2. **The next increment.** From the 4d report:
   - why 4d abstains on OpenTuringBench's long news texts;
   - old-generation text (HC3, MAGE, OpenAI-era answers);
   - the small gated slices (No Robots 553 humans, WasItAI 389), where one false AI decides the gate;
   - missed AI on WasItAI's 2026 models.
3. **Rater text.** Rates and bars are measured on public writing. Missed AI on rater text is unmeasured, which is why sheet rows are flag-only. In-house rater cells flagged by earlier models should be reviewed before anyone relies on rater-comment checks.
4. **Offered, not yet accepted:**
   - Make the column picker skip `ar_` columns. Those hold the autorater's AI output, and checking them flags every row.
   - Delete old uploads in `data/uploads`.
   - Delete the old corpus copies on D:.

## Known issues

**Deferred from the workbook v2 review:**
- Numbered-list markers and abbreviations can count as their own sentences.
- "Each Column Alone" lists empty cells.
- The trigger text rounds to whole percents.
- The Technical sheet shows a bar for texts that were not checked.
- Trigger sentences keep retrying after a CUDA error, with no stop after repeated failures.
- Trigger words add about 0.85 s per high row on the GPU; lower `TriggerConfig.max_words` if audits get slow.
- API tests write into the real `data/uploads` and `data/output`.

**Older issues:**
- Column discovery doesn't recognise a column named just "Agent".
- Quality and proxies read only the first text column.
- Control characters in a cell crash the workbook write (`IllegalCharacterError`).
- Uploaded copies are kept.
- The event loop blocks while an audit runs.
- `#N/A` agents show as Excel error cells.
- The dashboard's meta line builds HTML from sheet and column names (`innerHTML`).
- `requirements-ml.txt` lacks `peft`, which LoRA training needs.

**Debt ledger:** D10–D13 and D16–D20 are open; see `docs/bob/llm-authorship-checker/debt-ledger.md`.

## Working rules

- **Locked tests:** reruns are only for bugs, never for tuning. Each needs a recorded reason, with amendments written before results. Every retry needs a fresh locked test; never tune on a seen one.
- **Sealed scoring files:** `src/detection/*.py`, `src/authorship/{detector,excerpts,language,registry,calibration}.py`, `src/corpus/text.py` and `src/detectors/base.py` are sealed by hash in the registry. Change them only in a new increment. Report code goes in `src/report/`.
- **In-house sheets:** never train on the in-house rater sheets (GemPix, Personality); they are report-only. Read rater workbooks for structure and counts only, never cell text, and never name raters.
- **Machine:** one GPU job and one Ollama model at a time. Run long jobs detached.
- **Paid detectors:** never run trigger sentences or words on them.
- **Git:** commit and push at checkpoints.
- **Local files:** leave `x` and `inspect_wb.py` in the main checkout alone; never open or commit them.
