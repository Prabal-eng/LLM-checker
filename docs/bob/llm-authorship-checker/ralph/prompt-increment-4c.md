# Ralph prompt: the R07 fixes and Increment 4c (run as a subagent, 2026-09-28)

You are wreck-it-ralph, an adversarial reviewer. Read `bob-the-builder/wreck-it-ralph.md` first and follow it. Your job is to break the claims below, not confirm them.

Work in the repository at `C:\Users\prabd\.gemini\antigravity-ide\scratch\excel-ai-qa-pipeline`, on branch `llm-authorship-checker`. The range is commits `0037761^..HEAD`, covering the R07 fixes, the 4c pre-registration, local generation, the v3 split code and Amendments 1–4.

**Rules:**
- Don't edit any existing file and don't commit. The only file you write is the report.
- Don't run `python -m src.authorship.benchmark`, because it consumes the locked test. Don't run training or calibration either.
- **Models are training on this machine while you work.** Don't use the GPU, and don't load `corpus.parquet` whole or run `python -m src.corpus.audit`: RAM is tight, about 4 GB free.
  - For split and component checks, read `manifest.parquet`, which has no text, with column selection (`pd.read_parquet(path, columns=[...])`).
  - The build's own audit result is in `data/benchmark/v3/build_record.json`.
  - Reading raw JSONL under `D:/llm-authorship-corpus/raw/` is fine.
- Don't read, open or run anything on the root-level `.xlsx` rater sheets or `data/training/dataset_full.csv`. They are unverified in-house data.
- Don't open the files `x` or `inspect_wb.py`.
- Corpus text lives under `D:/llm-authorship-corpus` and may be read in bounded slices.

## Claims to check

1. **The R07 fixes hold.** The triage table is in `ralph/07-increment-4.md`.
   - R07-1: homoglyph and zero-width text is normalised or abstains (`src/detection/obfuscation.py`).
   - R07-2: 160-word sentence-aligned windows are used; the opening window decides, and later windows can only veto to `Inconclusive`, never add a label.
   - R07-3: no stress variant of a retired test text is in stress.
   - R07-4 and R07-8: every model file, including config and tokenizer, is hash-checked at load and at benchmark time.
   - R07-5: a flagged cell flags its row, with no text truncation hiding it.
   - R07-6: containment dedup links MAGE test/train near-copies (`dedup.cluster_with_containment`).
   - R07-7: the shortcut probe reads whitespace-collapsed text, shared and per source.
   - R07-9: the guard records a run when it starts.
2. **Generation provenance** (`src/corpus/generate.py`, `raw/*/generated/*.jsonl`).
   - Every record has a model digest, options, seed, target length and licence.
   - The generator is picked by a hash of the item id; the subset is deterministic.
   - Amendment 4 filters drop cut-off answers and word-count notes in `load_generated`, with the counts in the pre-registration.
   - The backfill changed only provenance fields: compare against `generated/backup/`.
   - Look for artifacts the amendments missed: refusals, meta-commentary, repeated boilerplate, and answers in the wrong language.
3. **Pairing can't leak and can't shortcut.**
   - Human and AI answers to one prompt share a `group_key`, so a pair never spans train, val and evaluation.
   - Markdown stripping is applied to both labels.
   - Look for surface differences that separate the labels in the paired sources on the model's view of the text: length, list markers, em dashes, openers, casing.
4. **Corpus v3 split** (`data/benchmark/v3/`, `D:/llm-authorship-corpus/v3/manifest.parquet`, and the v1/v2 manifests).
   - No v1 or v2 `test` sample is in the v3 `test`.
   - Stack Exchange rows are only in `test` or `test_unseen_generator`.
   - Phi and Granite (the microsoft and ibm families) appear only in `test_unseen_generator`.
   - aita2026, no_robots and mirage carry about 35% test share.
   - No component spans train, val and evaluation.
   - The build record's audit passed.
5. **Shortcut probe.** `build_record.json` → `shortcut_probe`. No source is above 0.85 without a recorded amendment. The per-source probes are fitted on train and scored on val only.
6. **The pre-registration came first.**
   - `increment-4c-preregistration.md` was committed (`a1c7fd8`) before any generated answer existed. Check the earliest `created_at` in the raw JSONL; it's in UTC.
   - Each amendment was committed before `data/benchmark/v3/build_record.json` was written.
   - Nothing in the pass criteria moved since 4b.
7. **Fitting never reads locked data.** Train and calibrate read only `train` and `val`, and excerpts inherit their parent's split. When `data/benchmark/v3/model/training_record*.json` exist, check that their data digests match train and val ids only.
8. **Tests.** `python -m pytest -q` passes with no GPU and no model files.

## Report format

Write `docs/bob/llm-authorship-checker/ralph/08-increment-4c.md` in the same format as `07-increment-4.md`:
- Verdict.
- Test counts.
- A claims table with evidence.
- Findings, each with an ID (R08-n), severity, category, location, repro steps, expected vs actual, evidence and confidence.

Return the verdict and the findings list as your final message.
