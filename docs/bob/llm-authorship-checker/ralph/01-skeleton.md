# wreck-it-ralph report: skeleton

Verdict: PASS  
Summary: tests 35/35; e2e examples 4/4; findings C/H/M/L = 0/0/1/2.

## Claims checked

| # | Requirement / done criterion | Result | Evidence |
|---|---|---|---|
| 1 | FR1: standalone contract exposes exactly three outcomes with a stable reason code | Met for skeleton | `src/detection/contracts.py` declares exactly three outcomes; live POST returned the stable skeleton reason code. |
| 2 | FR3: safely return Inconclusive for unreliable cases | Met for skeleton; real detector paths deferred | Normal, whitespace-only, Unicode, and 100,000-character requests abstained; invalid shapes and 100,001 characters returned 422. |
| 3 | FR4: every result carries evidence, versions, and score validity | Met, with R01-1 | Live response carried every field. |
| 4 | Endpoint accepts text and safely abstains | Met | Independently booted endpoint returned the expected response. |
| 5 | Host carries one contract end to end | Met | Health, text check, and metrics worked through Uvicorn. |
| 6 | CI green; e2e passes; health and metrics visible | Hosted CI unverifiable; local equivalent met | Compile passed; 35 tests passed with 87% coverage; CI contains both commands. |

## Findings

### R01-1: The public response model accepts contradictory score-validity states

- Severity: Medium
- Category: data integrity
- Location: `src/detection/contracts.py`
- Evidence: models with `score_valid=true` and no score, unavailable detectors with scores, and empty version identifiers validated successfully.
- Confidence: high
- Triage: **Fix** — added cross-field validation and non-empty version constraints; regression cases are in `test_v2_contract_rejects_contradictory_results`.

### R01-2: The skeleton test does not assert the complete response schema

- Severity: Low
- Category: test gap
- Location: `tests/test_api.py`
- Evidence: the test did not assert `evidence`, the complete key set, or enum cardinality.
- Confidence: high
- Triage: **Fix** — the end-to-end contract test now asserts all three.

### R01-3: Unused request context enlarges the public v2 contract

- Severity: Low
- Category: unnecessary complexity
- Location: `src/detection/contracts.py`
- Evidence: the handler never consumed `context`.
- Confidence: high
- Triage: **Fix** — removed `context` until a tested requirement needs it.

## Operational notes

Ralph booted independently on port 8766 because port 8765 was occupied. A 100-request, 20-thread probe returned 100 successful responses and exact counter increments. No repository files were modified by the reviewer.
