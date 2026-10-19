# wreck-it-ralph report: increment 1

Verdict: **FAIL**

Summary: tests **53/53**; e2e examples **2/3**; findings **C0/H3/M1/L0**.

## Claims checked

| # | Requirement / done criterion | Result | Evidence |
|---|---|---|---|
| 1 | FR1: standalone text has exactly three outcomes with reason codes | met | `DetectionDecision` defines three values; threshold probes returned the expected three outcomes. |
| 2 | FR2: every selected spreadsheet cell receives the same decision contract | not met | `RowAuthorshipResult` retained only per-column decision and score; validity, status, reasons, evidence, and versions were discarded. |
| 3 | FR3: insufficient text, unsupported input, detector failure, and abstention-band scores return Inconclusive | not met | Empty/unsupported requests returned 422 or raised `TypeError`; a detector-reported error with a raw response returned `likely_ai_generated`. |
| 4 | FR4: every result includes evidence, detector version, policy version, and score validity | not met | API responses had the fields, but workbook rows omitted score validity and per-column results omitted most of the contract. |
| 5 | No unhandled detector failure is presented as decisive | not met | A detector result with `analysis={"error":"upstream failed"}` and a raw payload returned `likely_ai_generated`. |
| 6 | Three-way policy works through standalone and workbook flows | met for valid inputs | API gate returned `inconclusive/MODEL_NOT_VALIDATED`; injected workbook run produced AI=1, human=1, inconclusive=1. |
| 7 | Data integrity | not met | Long selected column names produced duplicate audit headers. |
| 8 | Ralph review performed | met | Docker boot, full tests, all e2e flows, boundary inputs, failure paths, Unicode, and workbook output were independently exercised. |

## Findings

### R01-1: Detector-reported failure can become a decisive AI label

- Severity: High
- Category: reliability
- Location: `src/detection/service.py:64`
- Repro steps: Return `DetectionResult(ai_score=.99, analysis={"error":"upstream failed"}, raw_response={"status":500})` from a validated detector.
- Expected vs actual: A detector-reported failure must return Inconclusive. It returned Likely AI-generated because error handling only triggered when `raw_response` was false.
- Evidence: `decision='likely_ai_generated', score=0.99, reason_codes=['CALIBRATED_HIGH_SCORE']`.
- Confidence: high

Triage: **Fix** — error detection now ignores raw-payload truthiness. Regression `test_detector_reported_error_with_raw_payload_still_abstains` passes; full suite: `62 passed`.

### R01-2: Empty and unsupported standalone inputs do not return Inconclusive

- Severity: High
- Category: functional bug
- Location: `src/detection/contracts.py:37`, `src/detection/service.py:34`, `src/detection/policy.py:129`
- Repro steps: Post `{"text":""}`, post `{"text":123}`, or call the service with `None`/`123`.
- Expected vs actual: FR3 requires Inconclusive. The API returned 422 and direct service calls raised `TypeError`.
- Evidence: `{"text":123} 422`; `None RAISED TypeError`; `123 RAISED TypeError`.
- Confidence: high

Triage: **Fix** — the policy now abstains with `INSUFFICIENT_TEXT`, `UNSUPPORTED_INPUT_TYPE`, or `UNSUPPORTED_INPUT_LENGTH` before detector execution. API and service regressions pass; full suite: `62 passed`.

### R01-3: Workbook output drops required result-contract fields

- Severity: High
- Category: data integrity
- Location: `src/detection/contracts.py:75-80`, `src/report/excel_writer.py:271-280`, `src/report/excel_writer.py:333-343`
- Repro steps: Run a multi-column pipeline and inspect `Annotated Master`.
- Expected vs actual: Every cell result should retain score validity, reasons, evidence, detector status/version, and policy version. These fields were absent or flattened.
- Evidence: No `QA_Score_Valid` existed and per-column output contained only decision and score.
- Confidence: high

Triage: **Fix** — `RowAuthorshipResult.column_results` now retains each full contract and validates its projections. The workbook emits all contract fields for the row and every selected cell. Regression tests pass; full suite: `62 passed`.

### R01-4: Truncated column-result headers can collide

- Severity: Medium
- Category: data integrity
- Location: `src/report/excel_writer.py:279`
- Repro steps: Select `first_text_column_long_alpha` and `first_text_column_long_beta` and inspect output headers.
- Expected vs actual: Headers must be unambiguous. Fixed-length slices produced duplicate decision and score headers.
- Evidence: Both columns produced `QA_Decision_first_text_column_` and `QA_Score_first_text_column_lon`.
- Confidence: high

Triage: **Fix** — headers now include a stable ordinal and the full source column name. The workbook regression asserts 16 distinct contract headers for two selected cells; full suite: `62 passed`.
