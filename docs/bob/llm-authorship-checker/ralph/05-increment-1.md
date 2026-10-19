# wreck-it-ralph report: increment 1

Verdict: **FAIL**

Summary: tests **64/64**; e2e examples **3/3**; findings **C0/H2/M0/L0**.

## Claims checked

| # | Requirement / done criterion | Result | Evidence |
|---|---|---|---|
| 1 | FR1: exactly three outcomes with stable reasons | met | Enum and boundary/contract tests passed. |
| 2 | FR2: same complete contract for each spreadsheet cell | met | Complete `column_results` and eight distinct workbook fields per cell. |
| 3 | FR3: unreliable inputs and detector failures abstain | not met | Malformed successful external-provider responses defaulted to score zero and became Likely human-written. |
| 4 | FR4: evidence, versions, and score validity attached | met | Standalone, row, and cell outputs carried the fields. |
| 5 | No detector failures presented as decisive | not met | Empty GPTZero and Originality JSON responses became decisive human labels. |
| 6 | Three-way standalone and workbook flows | met | All supplied e2e cases passed. |
| 7 | Data integrity | not met | CSV values beginning with `=` became active workbook formulas. |
| 8 | Ralph review | met | Build, boot, 64 tests, e2e, malformed provider responses, boundaries, and workbook serialization were exercised. |

## Findings

### R01-1: Malformed provider responses become decisive human labels

- Severity: High
- Category: reliability
- Location: `src/detectors/gptzero.py:64-66`, `src/detectors/originality.py:64-66`
- Repro steps: Mock either provider to return HTTP 200 with `{}` and use it through a validated service.
- Expected vs actual: A missing score must abstain; both adapters defaulted it to `0.0` and returned Likely human-written.
- Evidence: Both results had `score=0.0`, `score_valid=True`, and `reason_codes=['CALIBRATED_LOW_SCORE']`.
- Confidence: high

Triage: **Fix** — both adapters now validate required response structure and return an error-bearing detector result for missing scores, which the service converts to `DETECTOR_REPORTED_ERROR`. Parameterized regression passes.

### R01-2: CSV content is emitted as active Excel formulas

- Severity: High
- Category: security
- Location: `src/report/excel_writer.py:297-300`
- Repro steps: Audit CSV text `=HYPERLINK("http://example.com","click")` and reload the output with formulas visible.
- Expected vs actual: Source content must remain inert; openpyxl serialized it as formula data type `f`.
- Evidence: `value =HYPERLINK("http://example.com","click") data_type f`.
- Confidence: high

Triage: **Fix** — imported source strings are explicitly serialized with Excel string type while preserving the exact value. Regression verifies the reloaded value and `data_type == 's'`.
