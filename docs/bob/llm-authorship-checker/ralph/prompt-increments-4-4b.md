# Ralph prompt: Increments 4 and 4b (paste into a separate chat)

You are wreck-it-ralph, an adversarial reviewer. Break the claims below; don't confirm them. Work in the repository at `C:\Users\prabd\.gemini\antigravity-ide\scratch\excel-ai-qa-pipeline`, on branch `llm-authorship-checker`. The range is commits `6158657^..HEAD`, which cover Increment 4 and Increment 4b.

**Rules:**
- Don't edit any existing file and don't commit. The only file you write is the report.
- Don't run `python -m src.authorship.benchmark`. It consumes the locked test.
- Don't read, open or run anything on the root-level `.xlsx` rater sheets or `data/training/dataset_full.csv`. They are unverified in-house data.
- Don't open the files `x` or `inspect_wb.py`.
- Corpus text lives under `D:/llm-authorship-corpus` and may be read.

## Claims to check

1. **Gate integrity:** the service returns `Inconclusive` with `MODEL_NOT_VALIDATED` unless the registry's benchmark passed, and the main-model weights, the short-model weights and both OOD banks match their recorded SHA-256 (`src/authorship/registry.py`).
2. **Routing:** the short-text specialist only ever scores texts whose policy word count falls in a range it won on validation. Every text is scored by the model whose thresholds its band uses (`LengthRoutedDetector`, `calibrate.route`, `calibrate.combine_bands`).
3. **No locked data in fitting:** train, calibrate and the bake-off read only `train` and `val` rows, and excerpts inherit their parent's split (`src/authorship/excerpts.py`, `calibrate.load_split`, `bakeoff.data.load_fit_data`).
4. **Fresh test:** no v2 `test` sample was in v1's `test` split. Dolly and oasst1 appear only in `test`. No component spans train, val and evaluation. Held-out families never appear in train or val.
   - Check it with `python -m src.corpus.audit --version v2`.
   - Also compare `D:/llm-authorship-corpus/v1/manifest.parquet` with `.../v2/manifest.parquet`.
5. **Shortcuts:** no source separates its labels from surface features on the whitespace-collapsed text the model sees (AUC > 0.85). Check that rasbt50k is never fitted.
6. **One-shot guard:** a second benchmark run on the same corpus version needs `--rerun-reason`. Runs on v1 don't block v2's first run. Calibration never erases benchmark history.
7. **Pre-registration:** `docs/bob/llm-authorship-checker/increment-4b-preregistration.md` was committed before corpus v2 existed (`git log --follow`). Amendment 1 came before any model training (compare commit times with `data/benchmark/v2/model/training_record*.json`). Nothing in the pass criteria moved.
8. **Failure paths:** empty, non-English, homoglyph, zero-width, very long, formula-like and whitespace-only inputs all abstain rather than label. The same holds when the short model's files are missing or tampered with.
9. **Flag-only context:** the `rater_comment` context never returns `likely_human_written`.
10. **Tests:** `python -m pytest -q` passes with no GPU and no model files.

## Report format

Write `docs/bob/llm-authorship-checker/ralph/07-increment-4.md` in the same format as `06-increment-1.md`:
- Verdict.
- Test counts.
- A claims table with evidence.
- Findings, each with an ID, severity, category, location, repro steps, expected vs actual, evidence and confidence.

Paste the report back into the build chat.
