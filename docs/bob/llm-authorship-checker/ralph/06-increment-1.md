# wreck-it-ralph report: increment 1

Verdict: **FAIL**

Summary: tests **67/67**; e2e examples **3/3**; findings **C0/H3/M0/L0**.

## Claims checked

| # | Requirement / done criterion | Result | Evidence |
|---|---|---|---|
| 1 | FR1: exactly three outcomes with stable reasons | not met | Detector exception codes depended on Python exception class names. |
| 2 | FR2: same complete contract for spreadsheet cells | met | Sixteen contract fields were emitted for two selected cells. |
| 3 | FR3: unreliable inputs and detector failures abstain | not met | A detector returning `None` raised; provider boolean scores produced decisive labels. |
| 4 | FR4: evidence and version/validity fields attached | met | Standalone, row, and cell outputs contained the fields. |
| 5 | No detector failures presented as decisive | not met | Boolean provider scores became decisive AI or human labels. |
| 6 | Three-way standalone and workbook flows | met | Default and validated e2e flows produced expected outcomes. |
| 7 | Data integrity | met | Unique headers, explicit length rejection, and inert formula-like values passed. |
| 8 | Ralph review | met | Build, boot, 67 tests, e2e, concurrency, malformed responses, and invalid detector outputs were exercised. |

## Findings

### R01-1: Boolean provider scores become decisive authorship labels

- Severity: High
- Category: reliability
- Location: `src/detectors/gptzero.py:74-85`, `src/detectors/originality.py:64-74`
- Repro steps: Return JSON boolean `true` or `false` in either provider score field.
- Expected vs actual: Booleans are malformed probabilities and must abstain; `float()` converted them to 1.0 or 0.0.
- Evidence: Both providers returned decisive AI for true and decisive human for false.
- Confidence: high

Triage: **Fix** — adapters explicitly reject boolean score values and return error-bearing results. Four provider/value regressions pass.

### R01-2: Missing detector result escapes the Inconclusive failure contract

- Severity: High
- Category: reliability
- Location: `src/detection/service.py:52-64`
- Repro steps: Use a detector whose `detect()` returns `None`.
- Expected vs actual: It must abstain; dereferencing `raw.analysis` raised `AttributeError`.
- Evidence: `RAISED AttributeError 'NoneType' object has no attribute 'analysis'`.
- Confidence: high

Triage: **Fix** — the service validates the returned object as `DetectionResult` before dereferencing it and returns `DETECTOR_INVALID_RESPONSE`. Regression passes.

### R01-3: Detector failure reason codes depend on exception implementation details

- Severity: High
- Category: functional bug
- Location: `src/detection/service.py:54-61`
- Repro steps: Raise `TimeoutError`, `ConnectionError`, and `RuntimeError` from equivalent detectors.
- Expected vs actual: One stable failure category was required; three class-derived public reason codes were returned.
- Evidence: `DETECTOR_ERROR_TIMEOUTERROR`, `DETECTOR_ERROR_CONNECTIONERROR`, `DETECTOR_ERROR_RUNTIMEERROR`.
- Confidence: high

Triage: **Fix** — all thrown detector exceptions now return stable public reason `DETECTOR_ERROR`; the exception class is logged. Parameterized regression passes.
