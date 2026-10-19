# wreck-it-ralph report: increment 1

Verdict: **FAIL**

Summary: tests **62/62**; e2e examples **3/3**; findings **C0/H3/M0/L0**.

## Claims checked

| # | Requirement / done criterion | Result | Evidence |
|---|---|---|---|
| 1 | FR1: classify standalone text into exactly three outcomes with stable reason codes | met | The enum has exactly three values and policy boundary probes passed. |
| 2 | FR2: apply the same decision contract to every selected spreadsheet text cell | met | `column_results` preserved complete cell results and the workbook emitted eight distinct contract fields per selected column. |
| 3 | FR3: return Inconclusive for insufficient, unsupported, failed, or mid-band inputs | not met | Invalid detector evidence raised an uncaught validation error. |
| 4 | FR4: attach evidence and version/validity metadata to every result | not met | A mid-band score could produce an empty evidence list. |
| 5 | Unhandled detector failures presented as decisive = 0 | met | A reported detector error with a raw payload returned Inconclusive. |
| 6 | Three-way policy through standalone and workbook flows | met | Default rows abstained; validated fixture produced one of each outcome. |
| 7 | Data integrity | not met | A supported 40,000-character CSV cell was silently truncated to 32,767 characters in Excel. |
| 8 | Ralph review | met | Build, boot, 62 tests, all e2e cases, edge inputs, failure paths, and workbook output were exercised. |

## Findings

### R01-1: Evidence normalization failure escapes instead of abstaining

- Severity: High
- Category: reliability
- Location: `src/detection/service.py:74-76`
- Repro steps: Return a 501-character evidence string from a validated detector.
- Expected vs actual: Invalid detector output must return Inconclusive; `DecisionEvidence` raised an uncaught validation error.
- Evidence: `long-evidence RAISED ValidationError`.
- Confidence: high

Triage: **Fix** — evidence normalization is now inside a failure boundary and returns `DETECTOR_INVALID_EVIDENCE`. Regression passes; full suite: `64 passed`.

### R01-2: Abstention-band results can omit observed evidence

- Severity: High
- Category: functional bug
- Location: `src/detection/policy.py:145-153`
- Repro steps: Use a validated detector with score `0.5` and no detector-supplied evidence.
- Expected vs actual: The observed score should be evidence; the result had `evidence: []`.
- Evidence: `decision='inconclusive', score=0.5, reason_codes=['SCORE_IN_ABSTENTION_BAND'], evidence=[]`.
- Confidence: high

Triage: **Fix** — the policy now adds a `SCORE_IN_ABSTENTION_BAND` evidence item carrying the observed score. Boundary regression passes; full suite: `64 passed`.

### R01-3: Audited workbook silently truncates valid CSV text

- Severity: High
- Category: data integrity
- Location: `src/report/excel_writer.py:297-300`
- Repro steps: Audit a CSV whose selected text cell contains 40,000 characters and reload `Annotated Master`.
- Expected vs actual: The input must be preserved or explicitly rejected; Excel silently reduced it to 32,767 characters.
- Evidence: `input_chars 40000 output_chars 32767 equal False`, while decision reason was `MODEL_NOT_VALIDATED`.
- Confidence: high

Triage: **Fix** — the shared support ceiling now matches Excel's 32,767-character cell limit. Longer text returns `UNSUPPORTED_INPUT_LENGTH`, and the workbook records its source column in `QA_Truncated_Source_Cols`. Regression passes; full suite: `64 passed`.
