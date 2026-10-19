# Audit workbook v2: verdict tiers, trigger sentences, a cleaner workbook

Date: 2026-10-01. Branch: `increment-4d` (the web app runs from this worktree). Status: approved in chat
2026-10-01; the user asked for execution straight after the spec.

## Why

Today's ten audits (Autorater Meta-Eval, 2,499 rows) gave 381 Likely AI and 2,118 Inconclusive. Of the
Inconclusive rows, 529 scored 90% or more, 404 scored 75-90%, 778 scored under 30%, and 80 had no score
(67 not English). Two existing decisions cause this, and both stand:

1. The Likely AI bar keeps false AI at or under 0.8% of human writing. For 25-100 words (most rationales)
   the 4d model's bar is 98.8-99.1%; for 100-200 words it is 94.1%, for 200+ it is 96.0%.
2. Rater rationales are flag-only (D6): they are never called Likely human, because missed AI on rater
   text is unmeasured. Everything under the bar lands in one Inconclusive bucket, from 2% to 98%.

The workbook makes this worse: the review tag says "Most likely AI" from 75% beside an Inconclusive
decision, the bar is never shown, the tag's printed rates were measured on v5, the AI Review cuts each
rationale at 120 characters, and the Annotated Master gains 30 columns.

The user also wants to see, for a flagged agent, which sentence triggered the flag.

## Goals

- One verdict per row that never contradicts the calibrated decision, with the bar shown beside the score.
- Measured rates for each tier from the served model's validation data.
- The sentence(s) that drove a high score, highlighted in the rationale.
- A workbook a reviewer can act on: per-agent view, full rationales, technical detail moved aside.
- A dashboard that lists flagged agents.

## Non-goals and constraints

- No change to any decision, score or threshold. Likely AI means exactly what it means today.
- No edit to a sealed scoring file (`registry._SCORING_CODE`: `src/detection/*.py`,
  `src/authorship/{detector,excerpts,language,registry,calibration}.py`, `src/corpus/text.py`,
  `src/detectors/base.py`). New code lives in `src/report/`; it may import from sealed modules.
- `/api/v2/check-text` and the detection contract are unchanged.
- Rates come from validation data only; never the locked test, never the in-house sheets.
- Trigger sentences never run on paid API detectors (GPTZero, Originality): they would cost money.

## Design

### 1. Verdict tiers (`src/report/tiers.py`, replaces `src/report/review_tags.py`)

Each row gets one tier from its row result (decision, score) in this order:

| Tier | Rule |
|---|---|
| Likely AI | decision is Likely AI (cleared the calibrated bar) |
| Likely human | decision is Likely human (never happens for flag-only rater text; kept for other contexts) |
| Not checked | no score; the reason is shown (see below) |
| High AI score – review | score >= 75%, under the bar |
| Moderate AI score | 50% <= score < 75% |
| Needs a second look | 30% <= score < 50% |
| Low AI score | score < 30% |

The 75/50/30 bands are the sponsor's bands from 2026-09-28, renamed so only cleared rows say "Likely
AI". They live in `TierConfig(high_min=0.75, moderate_min=0.50, second_look_min=0.30)`, which replaces
`ReviewTagConfig` (config key `tiers`).

Not-checked reasons, from the row's reason codes: UNSUPPORTED_LANGUAGE "not English";
INSUFFICIENT_TEXT "too short"; UNSUPPORTED_INPUT_LENGTH "too long"; DISGUISE_SUSPECTED "disguised
characters"; MODEL_NOT_VALIDATED "model not validated"; UNSUPPORTED_LENGTH "no bar for this length";
UNSUPPORTED_DOMAIN "unlike calibrated text"; SCORE_UNAVAILABLE and DETECTOR_* "detector unavailable";
any other code is shown as-is.

A row whose score clears the bar but whose decision is Inconclusive (mixed signals across long-text
windows) is tiered by its score, so it lands in High AI score – review, and the Technical sheet keeps
the reason.

**The bar.** `bar_for(policy, text)`: the band for the deobfuscated text's word count
(`policy.band_for(policy.word_count(text)).ai_min_score`), else `policy.ai_min_score` when the policy has
no bands, else None. For a row decided on its combined text, the bar is the combined text's.

**Measured rates** (`src/report/tier_rates.py`, CLI `python -m src.report.tier_rates`). Uses the
validation scores saved at calibration (`calibration_scores_<sha12>.npz` beside each model, keys
`val_raw`, `val_ids`) and the validation frame calibration built (`load_split("val")` plus
`with_excerpts`), so no GPU run is needed. Each validation text is routed as the service routes it (the
short model for its recorded word ranges, else the main model), mapped through that model's Platt
parameters, and tiered with the record's bands and the default tier bands. Output, per tier: the share of
human and of AI validation texts in it, with counts, for all lengths with a band and for 25-100 words.
Written to `data/report/tier_rates.json` keyed by model_id, with the source files' digests and the date.
It fails rather than writing partial rates if any id is missing from a cache. Computed for the
registered 4d model; a model without an entry (e.g. the v5 preview) shows tiers without rates and says
the rates were not measured for it. Caveats printed with the rates: public validation writing, not rater
rationales; later-window vetoes and disguise handling are not replayed.

### 2. Trigger sentences (`src/report/triggers.py`)

Runs for rows in Likely AI or High AI score – review, on the text that decided the row:

- the combined text (cells joined with a blank line, as the pipeline joins them) when the row was
  decided on it;
- the single highest-scoring flagged cell when a cell flagged the row on its own (CELL_FLAGGED_AI);
- the only filled cell otherwise.

Steps:

1. Split the decided text into sentences with character spans that map back to the original cells.
   Boundaries: line breaks, and `.`, `!`, `?` or `…` (plus closing quotes or brackets) followed by
   whitespace and an uppercase letter, digit or opening quote.
2. Candidates: sentences starting within the first 160 words (`authorship.detector.WINDOW_WORDS`),
   because text past the model's window never reached the score; at most 12.
3. Build one variant per candidate with that sentence removed; deobfuscate each variant and the full text
   as the service does.
4. Score the full text and every variant with the same model that scored the row: for the length-routed
   detector, its short model when `serves_short(deobfuscated full text)` else its main model, via
   `raw_and_embeddings` (one truncated pass per text, as calibration scored) and that model's Platt map;
   for any other local detector, `detect_batch(...)[i].ai_score`.
5. Drop = full-text score minus variant score. Trigger sentences: drop >= 10 points, the top 2 by drop.
   None: "spread across the text". One sentence only: the whole text is the trigger.

Result per row: `TriggerResult(kind, sentences=[(column, start, end, drop)], baseline, all_drops)`, where
kind is "sentences", "spread", "single" or "skipped" (with a reason: not run, paid API, error). Errors
never fail an audit: the row reads "trigger not found (error)" and the audit continues.

Shown as "Indicative: the model scores the whole text; this is the sentence whose removal lowers the
score most." On by default; `TriggerConfig(enabled=True, min_drop=0.10, max_sentences=12, top=2)`, a
dashboard checkbox and CLI `--no-triggers` switch it off. Expected cost on today's worst sheet (fr_FR,
long rationales): about 1,500-3,000 extra truncated passes; measured during execution and reported.

### 3. Row report (`src/report/rows.py`)

The pipeline builds one `RowReport` per row: position, task id (if discovered), agent, row result,
tier, not-checked reason, score, bar, decided columns, trigger result. The workbook, the summary
counts, the agent table and the API all read these, instead of parallel lists.

### 4. Workbook (`src/report/excel_writer.py`)

Sheets, in order:

- **Summary Dashboard** (name unchanged): title; model line (detector version) and the red banner for previews; locale, sheet,
  agent column, columns checked; one tier table (tier, rows, meaning, and the four rate columns: human
  and AI texts, 25-100 words and all lengths); "Agents with any Likely AI: N of M"; rows checked; average
  justification quality (unchanged); structural errors and warnings only when there are any; notes on the
  bar, flag-only rater text, the rate caveats, and indicative triggers. The second tag table, the
  always-zero Likely human row and the rater breakdown (moved to Agents) go.
- **Agents** (new): one row per agent: rows, Likely AI, High AI score, Moderate, Not checked, highest
  score, share Likely AI. Sorted by Likely AI, then High AI score, then highest score, descending.
  Filters on, header frozen.
- **AI Review**: one row per task, strongest first (tier order, then score): Row # (row in Annotated
  Master), Task ID (when discovered), Agent, Verdict, AI score, Bar, Decided on (column names, or "all
  columns together"), Rationale (full text of the decided cell(s), each prefixed by its column name when
  there are several, trigger sentences in bold red via rich text), Trigger (e.g. "Without it: 97% -> 41%",
  "spread across the text", "not run"), Each Column Alone (every filled column's own tier and score, when there are several),
  Human Reviewer Decision (dropdown, unchanged values), Reviewer Notes. Wrapped text, filters, frozen
  header and first columns.
- **Annotated Master**: the original columns, then QA Verdict (filled by tier), QA AI Score, QA Bar, QA
  Decided On, QA Trigger Sentence, QA Quality Score, QA Quality Flags; QA Structural Issues only when the
  sheet has any; QA Truncated Columns only when a cell was cut at Excel's 32,767-character limit.
- **Technical** (new, last): one row per scored text (each cell, and each row's combined text): Row #,
  Agent, Text (column name or "all columns together"), Decision, Score, Bar, Words, Reason codes,
  Evidence, Detector status, Detector version, Policy version, Proxy warnings (the row's proxy signals, which leave the AI Review), Sentence drops (every candidate's drop).
- **Structural Issues**: created only when there are issues.

Formula safety is unchanged: every value from the upload stays inert text (rich-text cells included).

### 5. API, CLI, dashboard

- `run()` summary gains `tier_counts` (tier -> rows), `agents` (the Agents rows, as dicts),
  `tier_rates` (the served model's rates or None) and `triggers` (`{"enabled", "rows", "seconds"}`).
  `review_tag_counts` is removed. Existing counts (`ai_flagged_count`, etc.) are unchanged.
- `POST /api/audit` accepts `explain_triggers` (form, default true). The CLI gains `--no-triggers`.
- Dashboard: stat cards Likely AI, High AI score – review, Not checked (beside the existing ones); a
  "Flagged agents" table under the totals (agent, rows, Likely AI, High AI score, highest score), listing
  agents with any Likely AI or High AI score, most-flagged first; a "Find trigger sentences" checkbox
  (checked) in the audit form.

## Error handling

- Missing tier rates: tiers still shown; the rate cells read "not measured for this model".
- Trigger failure on a row: logged, that row reads "trigger not found (error)", the audit completes.
- Paid API detector: triggers skipped with "not run (paid API)".
- Gated engine (MODEL_NOT_VALIDATED): every row is Not checked "model not validated"; no triggers.

## Testing

- Tiers: each tier from decision and score; each not-checked reason; mixed-signal row tiered by score;
  the bar for banded, unbanded and unsupported-length policies.
- Tier rates: a synthetic cache and frame (routing, Platt, bands, both length slices); a missing id fails.
- Triggers: sentence spans map back to cells (combined text across two cells); with a deterministic fake
  detector, the planted sentence is found with its drop; "spread", "single", the 12-candidate and
  160-word limits; the same sub-model is used as the row's; a paid provider is skipped; an exception
  yields "error" without failing the audit.
- Workbook: sheet order and headers; one verdict column; the trigger in bold red within the rationale;
  full rationale text; agents sorted; Technical rows per scored text; Structural Issues only with issues;
  a formula-like rationale stays text.
- API, CLI and dashboard: the new summary fields; `explain_triggers=false` and `--no-triggers`; the
  dashboard has the agents table, the checkbox and the new cards.
- Existing tests that assert the old columns (`tests/test_review_tags.py`, `tests/test_multi_column.py`,
  `tests/test_pipeline.py`, `tests/test_preview.py`) move to the new layout; the 32,767-character
  truncation is still reported.
- Real run: re-audit one of today's sheets with the 4d preview; compare tier counts with the old file;
  time the triggers.
