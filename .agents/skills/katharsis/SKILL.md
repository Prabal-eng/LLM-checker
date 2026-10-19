---
name: katharsis
description: >-
  Katharsis v2.0 - A systematic methodology for assessing, improving, and
  purifying existing software systems. Use this skill when the codebase has
  accumulated technical debt, needs architectural improvements, or requires a
  structured plan to improve performance, reliability, or velocity. Provides
  a complete 10-section, 5-phase framework with templates, scoring rubrics,
  and checklists.
---

# KATHARSIS: Systematic Purification of Existing Systems

**Version 2.0 (Updated January 2026)**

## When to Use

- The codebase/pipeline has operational problems that need improvement
- Technical debt has accumulated and needs systematic elimination
- Performance, reliability, or velocity have degraded
- Architecture needs evaluation and potential restructuring

## Do NOT Use When

- Building a new system from scratch (use Episteme instead)
- System is fundamentally broken beyond repair (rebuild instead)
- No capacity exists for improvement work

---

## Framework Overview: 5 Phases, 10 Sections

```
PHASE I:   ASSESSMENT  (Sections 0-1)  - Scoping, completeness, baseline measurement
PHASE II:  ANALYSIS    (Sections 2-4)  - Debt cataloging, constraint id, pattern comparison
PHASE III: PLANNING    (Section 5)     - Feasibility validation and roadmap synthesis
PHASE IV:  EXECUTION   (Sections 6-9) - Four-phase implementation
PHASE V:   VALIDATION  (Section 10)   - Transformation measurement and lessons learned
```

---

## Domain Adaptation for the Excel AI QA Pipeline

| Generic Term  | This Project Equivalent                                  |
|---------------|----------------------------------------------------------|
| Latency       | Pipeline processing time per row / per batch             |
| Throughput    | Rows/minute; files processed per run                     |
| Reliability   | Detection accuracy, false-positive/negative rate         |
| Error rate    | Mis-flagged cells, crashes, API failures                 |
| Deployment    | Script execution, environment setup, config change       |
| User journey  | Excel file in => audited report out                      |
| External API  | Gemini API, OpenAI API, or other LLM providers           |

---

## PHASE I: ASSESSMENT

### Section 0 - Scoping and Completeness

**0A: System Scale Classification**

Rate as **Small** (1-10 scripts/modules):
- Skip subsystem analysis; treat as one unit
- Compress timelines (hours not days)
- Target: 80% completion in 2-4 weeks

```
SYSTEM: Excel AI QA Pipeline
SCALE: Small
COMPONENTS: [Count scripts + modules]
TEAM SIZE: [Engineers/annotators available]
KATHARSIS TIMELINE: 2-4 weeks
```

**0B: Completeness Inventory**

Infrastructure:
- [ ] Automated pipeline execution (no manual step-by-step)
- [ ] Environment management (requirements.txt, venv, .env)
- [ ] Test framework running
- [ ] Rollback capability (git, backups of outputs)
- [ ] Observability (logging per file/row processed)

Application:
- [ ] Core AI detection logic implemented
- [ ] Edge cases handled (empty cells, non-text, merged cells)
- [ ] Error states have fallbacks (API timeout, rate limit)
- [ ] Performance acceptable for expected file sizes
- [ ] Output format consistent and parseable

Operational:
- [ ] Architecture/flow documented
- [ ] Setup guide works for new users
- [ ] Troubleshooting notes for common failures

Scoring: 80-100% focus on optimization | 60-80% address gaps | <60% build missing first

**0C: Capacity and Timeline Estimation**

```
Engineers available: [N]
Hours/week per person: [H]
x 0.8 (meetings) x 0.85 (context switching) x 0.75 (interrupts)
= AVAILABLE hours/week

Small system effort: 170-270 engineer-hours total
TIMELINE = Total Effort / Available Capacity + 25% buffer
```

---

### Section 1 - Architecture Mapping + Baseline Measurement

**1A: Architecture Documentation**

```
Component: [e.g., detect_ai.py]
Type: [Script / Module / Config / Data file]
Language/Tech: Python / openpyxl / Gemini API
Lines of Code: ~[N]
Dependencies: [What it calls]
Criticality: [Critical / Important / Supporting]
```

Critical path:
1. Excel file loaded (openpyxl / pandas)
2. Cell extraction: iterate rows, extract text
3. AI detection: send to LLM API per row/batch
4. Score assignment: interpret API response
5. Output: annotated Excel / report generated

**1B: Performance Baselines**

```
LATENCY:
- Time to process 1 row: [X ms]
- Time to process 100 rows: [X s]
- Time to process full file (~500 rows): [X min]
- p95 processing time: [X]

THROUGHPUT:
- Rows/minute: [X]
- Files/hour: [X]

API COSTS:
- Tokens/row average: [X]
- Cost per file: [$X]
- Cost per 1000 rows: [$X]
```

**1C: Reliability Baselines**

```
Error rates (last N runs):
- API timeout errors: [X%]
- Rate limit hits: [X%]
- Parse errors (malformed cells): [X%]
- Crashes / unhandled exceptions: [X%]

Accuracy metrics:
- True positive rate (AI detected correctly): [X%]
- False positive rate (human flagged as AI): [X%]
- False negative rate (AI missed): [X%]
```

**1D: Velocity Baselines**

```
- Time from new Excel file to finished report: [X min/hours]
- Time to add new detection heuristic: [X hours]
- Time to fix a bug in detection logic: [X hours]
- Test coverage: [X%]
```

**1E: Code Quality Baselines** (run these)

```bash
python -m radon cc . -a
python -m pylint src/
```

---

## PHASE II: ANALYSIS

### Section 2 - Technical Debt Cataloging

**Automated Debt Discovery**

```bash
python -m radon cc src/ -a        # Complexity
python -m pylint src/             # Style / smells
python -m bandit -r src/          # Security
pip list --outdated               # Outdated deps
python -m mypy src/               # Type safety
grep -r "api_key" . --include="*.py"  # Hardcoded secrets
```

**Debt Taxonomy (10 Categories)**

1. Code Quality - Complex/unreadable, duplicated, inconsistent style
2. Architectural - Tight coupling, wrong abstraction, God scripts
3. Testing - Missing unit tests, low coverage on AI detection logic
4. Documentation - Missing docstrings, no README, undocumented configs
5. Performance - Slow per-row API calls, missing batching/caching
6. Security - Hardcoded API keys, no secrets management
7. Infrastructure - Manual run steps, no CI, no environment management
8. Dependency - Outdated packages, unneeded imports
9. Data - Missing input validation, no schema checks on Excel input
10. Operational - No logging, no retry logic, no alerting

**Debt Item Template**

```
DEBT ITEM #[N]
Title: [Short name]
Category: [From 10 above]
Location: [File/function]
Description: [What is the debt?]
Evidence: [How do we know?]
Impact:
  Velocity: [How does this slow work?]
  Reliability: [Has this caused failures?]
  Performance: [Does this affect speed?]
Effort to Fix: [X-Y hours]
Priority Score: [Calculated below]
```

**Priority Scoring**

```
IMPACT SCORE (0-10):
+ Has caused a bad run/failure: +7-10
+ Affects critical path (AI detection): +5
+ Slows every file processed: +4
+ Security risk (exposed API key): +10
+ Performance impact >5s per file: +3

EFFORT SCORE (0-10):
<2h:1 | 1d:2 | 3d:4 | 1w:6 | 2w:8 | >2w:10
x complexity: Simple x1.0, Moderate x1.5, Architectural x2.0
x risk: Low x1.0, Medium x1.3, High x1.8

PRIORITY = Impact / Effort
High: >2.0 | Medium: 1.0-2.0 | Low: <1.0
```

---

### Section 3 - Constraint Identification and Breakability

Relevant constraints for this pipeline:
- Technical: API rate limits, token limits per request, Excel row limits
- Architectural: Synchronous row-by-row calls (cannot parallelize easily)
- Operational: Manual trigger to run the pipeline (no scheduling)
- Data: No validation that Excel file matches expected schema
- External: LLM API behavior changes, model deprecations

```
CONSTRAINT #[N]
Name: [Short name]
Type: Hard / Soft
Description: [What does this prevent?]
Breakability (1-10): [1=impossible, 10=trivial]
Breaking Strategy: [What changes are needed?]
Effort: [Weeks]
Benefits: [Quantified if possible]
```

Breakability Priority = Impact / (Effort x Risk^0.5)
Prioritize constraints with score > 1.0

---

### Section 4 - Pattern Analysis and Innovation Mining

**Pattern Compliance Checklist**

Resilience Patterns:
- [ ] Retry logic with exponential backoff on API errors
- [ ] Timeout on all API calls
- [ ] Fallback behavior when API is down
- [ ] Rate limit handling (sleep + retry)

Data Patterns:
- [ ] Input schema validation before processing
- [ ] Idempotency (safe to re-run on same file)
- [ ] Checkpointing (resume from where it stopped)
- [ ] Results caching (do not re-process identical text)

Observability Patterns:
- [ ] Structured logging (file + row + result per entry)
- [ ] Progress tracking (rows done / total)
- [ ] Cost tracking (tokens used per run)
- [ ] Error aggregation (summary at end of run)

Deployment Patterns:
- [ ] Configuration externalization (no hardcoded values)
- [ ] Secrets management (.env or secrets manager)
- [ ] Reproducible environment (requirements.txt, venv)

Anti-Patterns to Eliminate:
- [ ] Hardcoded API keys in source files
- [ ] N+1 API call problem (1 call per cell instead of batching)
- [ ] No error handling (bare except:)
- [ ] Configuration in code (magic strings/numbers)
- [ ] Manual file path changes between runs

**Key Innovation Opportunities for AI Pipelines**

```
BATCHING: Send multiple rows per API call -> 5-10x speed improvement
CACHING:  Cache results for identical text -> avoid redundant cost
ASYNC:    Parallel API calls -> 3-5x throughput improvement
PROMPT VERSIONING: Track prompt changes for reproducibility
OUTPUT SCHEMA VALIDATION: Assert AI response has expected format
```

---

## PHASE III: PLANNING

### Section 5 - Feasibility Validation and Roadmap Synthesis

**5A: Feasibility Assessment (per improvement)**

```
IMPROVEMENT: [Name]
Technical Feasibility (1-10): [Score + justification]
Effort Confidence (1-10): [Estimate + range]
Technical Risks:
  [Risk]: Probability [L/M/H], Impact [L/M/H]
  Mitigation: [How to handle]
Dependencies: [What must be done first]
Decision: [Proceed / Require POC / Defer / Reject]
```

Require POC if:
- Technical Feasibility < 7 AND Effort > 40 hours
- New approach team has not used before
- Multiple high-probability, high-impact risks

**5B: Phased Roadmap (Small System, 2-4 weeks)**

```
Phase 0 (if Completeness < 60%): Foundation (1-2 weeks)
  Build missing observability, environment setup, basic tests

Phase 1: Critical Debt + Enabling (Weeks 1-2)
  Priority score > 8, OR caused bad run, OR unblocks >=5 others
  Fix failures, add safety nets, enable future work

Phase 2: Quick Wins (Weeks 2-3)
  Priority 5-8, high confidence, <= 3 days each
  Sprint-based: 5-7 improvements per sprint

Phase 3: Structural Improvements (Weeks 3-4)
  Architectural changes: batching, async, caching
  Incremental migration: build new -> dual-run -> cutover -> cleanup

Phase 4: Optimization (Week 4+)
  Profile -> Bottleneck -> Hypothesis -> Implement -> Measure -> Iterate
  Stop when: targets achieved, OR <5% gain possible, OR timeline exhausted
```

**5C: Success Metrics**

```
BASELINE (Section 1) -> TARGET (after roadmap):
Processing time p95: [X] -> [Y] ([Z%] faster)
Error rate: [X%] -> [Y%] ([Z%] reduction)
Accuracy (false positive rate): [X%] -> [Y%]
Test coverage: [X%] -> [Y%]
Cost per 1000 rows: [$X] -> [$Y]

MINIMUM VIABLE TARGET = 80% of above targets
```

---

## PHASE IV: EXECUTION

### Sections 6-9 - Implementation Protocol

**Pre-Implementation Checklist**

```
[ ] Detailed design documented
[ ] Technical approach reviewed
[ ] Edge cases identified
[ ] Test plan defined
[ ] Rollback procedure documented
[ ] Acceptance criteria defined and measurable
```

**Weekly Tracking**

```
Week [X] - Improvement: [Name]
Status: [Not Started / In Progress / In Review / Testing / Deployed]
Work Completed: [Specific accomplishments]
Tests Added: [Unit/integration count]
Effort: Estimated [X]h | Actual [Y]h | Remaining [Z]h
Status: [On Track / At Risk / Blocked]
Blockers: [Description + owner + ETA]
```

**Post-Deployment Validation**

```
Immediate (1 hour):
[ ] No error rate increase
[ ] No processing time regression
[ ] Key files produce correct output

24-Hour:
[ ] Error rates stable or improved
[ ] No new failure modes observed

7-Day:
[ ] All success criteria met
[ ] No regressions detected - mark COMPLETE
```

**Phase 3 - Structural Migration Pattern**

```
For each major change (e.g., async processing):
Phase A: Build new implementation alongside old
Phase B: Dual-run validation (compare outputs)
Phase C: Gradual shift (test files -> full usage)
Phase D: Complete cutover
Phase E: Remove old code
```

**Phase 4 - Optimization Cycle**

```
1. Profile pipeline under realistic load (real Excel files)
2. Identify bottleneck (what % of time is it?)
3. Root cause: API call latency? Parsing? Token count?
4. Hypothesis: "Batching 10 rows per call will reduce time by 60%"
5. Implement
6. Measure actual vs expected
7. Iterate on next bottleneck
```

---

## PHASE V: VALIDATION

### Section 10 - Transformation Report

**10A: Final Metrics**

```
TRANSFORMATION ACHIEVED:
Performance:
  Processing time p95: [X] -> [Y] ([Z%] faster)
  Throughput (rows/min): [X] -> [Y]
  Cost per 1000 rows: [$X] -> [$Y]

Reliability:
  Error rate: [X%] -> [Y%]
  False positive rate: [X%] -> [Y%]
  Crash rate: [X%] -> [Y%]

Velocity:
  Time from file to report: [X] -> [Y]
  Test coverage: [X%] -> [Y%]

Quality:
  Tech debt items: [X] -> [Y]
  Pattern compliance: [X%] -> [Y%]
```

**10B: ROI Calculation**

```
INVESTMENT:
  Engineering hours: [X] x [$Y/hr] = $[Z]

RETURNS (annual):
  API cost savings (efficiency): [$X/year]
  Time saved per file run: [X min x N files/year x $Y/hr]
  Reduced debugging time: [X hrs/month x 12 x $Y/hr]
  Fewer re-runs (reliability): [X runs/year x $Y/run]

Payback period = Investment / Annual Returns = [X months]
```

**10C: Retrospective**

```
WHAT WENT WELL: [List with evidence]
WHAT WAS HARDER THAN EXPECTED: [List with root cause]
WHAT WE WOULD DO DIFFERENTLY: [List]

SUSTAINABILITY:
[ ] All improved metrics have monitoring
[ ] New patterns are documented
[ ] CI/CD prevents debt reintroduction
[ ] Next Katharsis cycle planned for: [Date in 6-12 months]
```

---

## Quality Rubric

Rate execution 1-5 per dimension:

| Dimension           | Criteria                                           | Score |
|---------------------|----------------------------------------------------|-------|
| Assessment (0-4)    | Comprehensive baselines, honest debt list          | /5    |
| Planning (5)        | Justified priorities, realistic timeline           | /5    |
| Implementation (6-9)| Systematic protocol, quality gates honored         | /5    |
| Validation (10)     | Rigorous measurement, honest assessment            | /5    |
| **TOTAL**           |                                                    | /20   |

- 18-20: Exceptional - gold standard
- 15-17: Strong - objectives achieved
- 12-14: Adequate - some gaps
- <12: Needs improvement

---

## Quick-Start Commands for This Pipeline

Run these before starting Katharsis:

```bash
# Code quality assessment
python -m radon cc . -a
python -m pylint src/ scripts/

# Security scan
python -m bandit -r .

# Dependency check
pip list --outdated

# Test coverage
pytest --cov=src

# Hardcoded secrets scan
grep -rn "api_key\s*=" . --include="*.py"
grep -rn "sk-\|AIza" . --include="*.py"
```

---

## Core Principles

- **Measure before improving** - baselines must exist before optimization
- **Prioritize ruthlessly** - Impact/Effort scoring, not gut feel
- **Incremental migration** - never big-bang cutovers
- **Systematic optimization** - profile -> bottleneck -> hypothesis -> measure -> iterate
- **Cyclical** - run every 6-12 months to prevent entropy accumulation

---

## References

Based on **Katharsis v2.0** (January 2026) — a complete methodology for transforming
technical complexity into clarity. Companion: **Episteme** (for new systems).
