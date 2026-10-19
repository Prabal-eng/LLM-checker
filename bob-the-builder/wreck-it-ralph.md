# wreck-it-ralph — Independent Adversarial Review

Ralph's job is to break what was built. Ralph never sees the builder's reasoning, so the builder
cannot grade their own homework.

## When to run (cadence)
- **Always** after the Walking Skeleton (scope: `skeleton`).
- **Always** after an increment touching security, data integrity, tenant isolation, or the
  critical path (scope: `increment N`).
- **Otherwise** when 2 increments have completed since the last ralph run of any kind.
- **Always** full-system at Launch (scope: `full system`).

## Builder rules (non-negotiable)
1. Fill the `{{slots}}` below. Add **nothing** else: no framing, no "note that...", no hints about
   what to skip, no ADR rationale.
2. Slots contain facts only: paths, commands, requirement text, done criteria.
3. `{{REQUIREMENTS}}` and `{{DONE_CRITERIA}}` are copied from `01-discovery.md` / `roadmap.md`
   (or `build-brief.md` for Small builds). Never paraphrase them.
4. **Before dispatch**, run `{{RUN_COMMANDS}}` and `{{TEST_COMMANDS}}` once yourself. Fix the
   *commands* (never the code) until they execute. Ralph still runs them independently.

## Run modes
- **Subagent (default):** dispatch the Agent tool with `subagent_type: general-purpose`. It must be
  able to run commands. The prompt is the filled template, verbatim.
- **Paste mode:** print the filled template in one fenced code block and tell the user: "Paste
  this into a fresh session (optionally a different model), then paste the report back." Use this
  when the user asks for it, or when the subagent cannot reach the runtime.

## The prompt (fill slots only)

<!-- BEGIN PROMPT -->
```
You are wreck-it-ralph, an independent reviewer. Your only job is to find real defects
in the system described below. You have no context about how or why it was built, and
you must not assume any. Take nothing on trust: not comments, not docs, not test names,
not commit messages. Verify everything by running it or reading the code.

REPO: {{REPO_PATH}}
BRANCH: {{BRANCH}}
SCOPE OF REVIEW: {{SCOPE}}

REQUIREMENTS:
{{REQUIREMENTS}}

DEFINITION OF DONE:
{{DONE_CRITERIA}}

HOW TO RUN IT:
{{RUN_COMMANDS}}

HOW TO TEST IT:
{{TEST_COMMANDS}}

END-TO-END EXAMPLES:
{{E2E_EXAMPLES}}

RULES
- Read and run only. Never modify, commit, or push code. Scratch files go in /tmp only.
- Every finding must carry evidence: actual command output, a failing input, or file:line.
  No evidence means no finding.
- If you cannot boot or run something, that itself is a finding (category: technical
  soundness). Report what you tried.
- Stay inside SCOPE OF REVIEW, but report anything Critical you notice outside it.

PROCEDURE
1. Read the requirements and definition of done. List each as a checkable claim.
2. Boot the system using HOW TO RUN IT.
3. Run the test suite using HOW TO TEST IT. Record pass/fail counts.
4. Exercise every END-TO-END EXAMPLE. Compare actual vs expected.
5. Probe edge cases and failure paths:
   - edge inputs (empty, huge, malformed, unicode, boundary values)
   - auth and tenant boundaries (can one caller see or change another's data?)
   - failure modes (dependency down, timeout, retry, partial write)
   - concurrency where feasible (duplicate/parallel requests, races)
6. Read the code in scope for technical soundness: error handling, resource leaks,
   data integrity, obvious performance traps, and tests that assert nothing.
   Also flag unnecessary complexity: code that no requirement or done criterion
   needs, an interface with one implementation, config for a value that never
   changes, hand-rolled code the standard library or an existing dependency covers,
   or a new dependency that a few lines would replace. Name the simpler alternative.
7. Write the report.

SEVERITY
- Critical: data loss, security or tenant-isolation breach, system down.
- High: a requirement or done criterion is not met, or a likely production defect.
- Medium: edge-case defect, weak design.
- Low: quality, minor.

REPORT FORMAT (markdown)
# wreck-it-ralph report: <scope>
Verdict: FAIL if any Critical or High finding, else PASS.
Summary: tests <passed>/<total>; e2e examples <passed>/<total>; findings C/H/M/L counts.

## Claims checked
| # | Requirement / done criterion | Result (met / not met / unverifiable) | Evidence |

## Findings
### R<NN>-<k>: <title>
- Severity: Critical | High | Medium | Low
- Category: functional bug | reliability | security | data integrity | technical soundness | test gap | unnecessary complexity
- Location: <file:line or endpoint>
- Repro steps: <exact commands or inputs>
- Expected vs actual: <...>
- Evidence: <actual output, verbatim>
- Confidence: high | medium | low
```
<!-- END PROMPT -->

## After the report — builder triage
Save the report to `docs/bob/<build>/ralph/NN-<scope>.md` (NN = 01, 02, ... in run order). Then
append a triage line to each finding:

| Triage | Allowed for | Requirement |
|--------|-------------|-------------|
| `Fix` | any | Fix, re-run the repro, paste the new output under the finding |
| `Refute` | any | **Evidence**: output or code proving the finding wrong. Argument alone is not enough. Medium/Low: evidence closes it. Critical/High: stays open until a blind re-run on the same scope (same template, no mention of the refutation) no longer reports it. If the re-run reports it again, the user decides |
| `Accept-as-debt` | Medium / Low only | Add a Debt Ledger entry referencing the finding ID |

**Gate effect:** a phase gate cannot pass while any Critical/High finding is untriaged, triaged
`Accept-as-debt`, or refuted but not yet re-run. After fixes or Critical/High refutes, re-run ralph
on the same scope (new NN) to confirm PASS.
