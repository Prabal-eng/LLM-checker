# bob-the-builder — Templates

## Contents
- T1 Framing Doc → `00-framing.md`
- T2 Requirements & Unknowns Register → `01-discovery.md`
- T3 Design Doc → `02-design.md`
- T4 ADR (light) / T5 ADR (full) → `adr/NNN-<decision>.md`
- T6 Build Roadmap → `roadmap.md`
- T7 Debt Ledger → `debt-ledger.md`
- T8 Launch Readiness Scorecard → `launch-readiness.md`
- T9 30/60/90 Review → `review-30-60-90.md`
- T10 Build Brief (Small scale) → `build-brief.md`

Copy into `docs/bob/<build-name>/`. Replace `<...>`; delete rows that don't apply.

---

## T1 Framing Doc
File: `00-framing.md`

**Problem statement:** <one paragraph: what is missing, who feels it, why now>
**Sponsor:** <name / role>  **Hard deadlines:** <date + reason, or none>
**Build type:** <component-in-platform | from-zero>

| Scale | Rationale |
|-------|-----------|
| <Small / Medium / Large / Very Large> | <why> |

### Inherited context
Mark each area Inherit / Extend / Build. From-zero builds mark everything Build.

| Area | Host provides | Decision (Inherit / Extend / Build) | Notes |
|------|---------------|-------------------------------------|-------|
| CI/CD | <...> | <...> | <...> |
| Deployment | <...> | <...> | <...> |
| Observability | <...> | <...> | <...> |
| Testing | <...> | <...> | <...> |
| Security/auth | <...> | <...> | <...> |
| Data layer | <...> | <...> | <...> |
| Conventions | <...> | <...> | <...> |

### Capacity
```
Available h/week = <engineers> × 40 × 0.8 × 0.85 × 0.75 = <n>
Timeline (weeks) = <effort h> ÷ <n> × 1.25 = <w>
```

Approval: <sponsor/user> — <date>

---

## T2 Requirements & Unknowns Register
File: `01-discovery.md`

**Users / problem:** <who, what job they need done>

### Functional requirements
| ID | Requirement | Tier (MVT/Target/Stretch) | Test (how it's verified) |
|----|-------------|---------------------------|--------------------------|
| FR1 | <...> | <...> | <...> |

### Non-functional requirements (domain metrics)
MVT = 80% of Target; Stretch = 120% of Target (invert for lower-is-better metrics).

| Metric | MVT | Target | Stretch |
|--------|-----|--------|---------|
| <e.g. p99 latency> | <...> | <...> | <...> |

### Constraints
| Constraint | Hard/Soft | Source | Design response |
|------------|-----------|--------|-----------------|
| <...> | <...> | <...> | <design around + document (hard) / may challenge (soft)> |

### Unknowns
impact-if-wrong: 5 = invalidates architecture · 3 = invalidates a component · 1 = minor

| ID | Unknown | impact-if-wrong (5/3/1) | Resolution route (research / POC / increment) | Status (open/confirmed/refuted) |
|----|---------|-------------------------|-----------------------------------------------|---------------------------------|
| U1 | <...> | <...> | <...> | open |

### POC (impact-5 unknowns that can't wait for an increment)
| Question (one, measurable) | Time-box (1–5 days) | Success criterion | Outcome (Go / No-Go / Redesign) |
|----------------------------|---------------------|-------------------|---------------------------------|
| <...> | <...> | <...> | <...> |

---

## T3 Design Doc
File: `02-design.md`

### Components
| Component | Responsibility | Owns data? | New / Inherited |
|-----------|----------------|------------|-----------------|
| <...> | <...> | <...> | <...> |

### Interfaces
<API/RPC/event contracts between components; versioning>

### Data flow
<request/event path, sync vs async, where state lives>

### Critical path
<the chain whose failure means the system is down>

### Pattern checklist
Which do we need at this scale — not scored. Families and their 8 patterns each: Katharsis §4A.

| Family | Needed? | Which patterns | Why |
|--------|---------|----------------|-----|
| Service Architecture | <...> | <...> | <...> |
| Data | <...> | <...> | <...> |
| Observability | <...> | <...> | <...> |
| Resilience | <...> | <...> | <...> |
| Deployment | <...> | <...> | <...> |

### Risk register
| Risk | Probability | Impact | Indicators | Mitigation |
|------|-------------|--------|------------|------------|
| <...> | <H/M/L> | <H/M/L> | <...> | <...> |

### ADR index
| NNN | Title | Door | Status |
|-----|-------|------|--------|
| 001 | <...> | <two-way / one-way> | <...> |

---

## T4 ADR (light)
File: `adr/NNN-<decision>.md`. For a two-way door only (reversal costs < one increment).

```
Decision: <...>
Why (default/inherited): <...>
Revisit if: <...>
```

New dependency: "Why" names why build-ladder rungs 2–5 failed. If the dependency is a one-way door
(hard to rip out later), use T5 instead.

---

## T5 ADR (full)
File: `adr/NNN-<decision>.md`. One-way door.

**Title:** <...>  **Status:** <proposed / accepted / superseded>  **Date:** <...>
**Context:** <forces, requirements, constraints>
**Reversibility:** one-way door — <why reversal costs more than one increment>
**Options (≥2 real):** A: <...> · B: <...>

Score 1–5, higher = better (5 = lowest risk/cost). Anchors: `reference.md` → Formula 2.

| Criterion | Weight | Option A | Option B |
|-----------|--------|----------|----------|
| Requirement fit | 0.30 | <1–5> | <1–5> |
| Risk | 0.20 | <1–5> | <1–5> |
| Operational cost/complexity | 0.20 | <1–5> | <1–5> |
| Reversibility | 0.15 | <1–5> | <1–5> |
| Team familiarity | 0.15 | <1–5> | <1–5> |
| **WEIGHTED SCORE** | | <Σ> | <Σ> |

**Tie check:** top two within 10%? → pick the more reversible. <result>
**Inherit-by-default check:** host platform choice Requirement fit ≥ 3? → host choice wins. <result>
**Decision:** <...>
**Consequences:** <what gets easier, what gets harder, follow-up work>

---

## T6 Build Roadmap
File: `roadmap.md` (also the resume state)

### Skeleton
- Layers touched: <...>
- e2e test: <name>
- Rollback required? <yes — shares blast radius with <live thing> / no — isolated, zero users>

### Increments
`PRIORITY = (Value + 2 × Risk Retired) / Effort Score`, dependencies first, then descending.
Timeline = Σ Est h ÷ capacity × 1.25 (never sum Effort Scores).

| # | Slice | Depends on | Value | Risk Retired | Est h | Effort Score | PRIORITY | Floor-rule? | Ralph trigger? | Status |
|---|-------|------------|-------|--------------|-------|--------------|----------|-------------|----------------|--------|
| 1 | <...> | <...> | <0–10> | <0–5> | <h> | <...> | <...> | <Y/N/N/A + reason> | <Y/N> | <todo/doing/done/deferred> |

### Increment done (one block per increment)
| Behavior | Assumption tested | Number that proves it | Result |
|----------|-------------------|-----------------------|--------|
| <...> | <U#> | <metric + threshold> | <actual; confirmed/refuted> |

### Weekly tracking (Medium+ only)
| Week | Increment | Est h | Actual h | Status |
|------|-----------|-------|----------|--------|
| <...> | <...> | <...> | <...> | <...> |

Resume here: <next action>

---

## T7 Debt Ledger
File: `debt-ledger.md`

| ID | Shortcut taken | Why | Cost to fix (h) | Trigger to repay | Source (increment # or ralph R<NN>-<k>) | Code marker |
|----|----------------|-----|-----------------|------------------|-----------------------------------------|-------------|
| D1 | <...> | <...> | <...> | <...> | <...> | <file:line of `ponytail: D1 ...`> |

Handed to Katharsis §2 as seed catalog at Launch.

---

## T8 Launch Readiness Scorecard
File: `launch-readiness.md`

| Category | Max | Score | Evidence |
|----------|-----|-------|----------|
| Functionality — MVT met, verified vs requirements | 20 | __/20 | <...> |
| Reliability — SLOs measured under realistic load, error paths handled | 15 | __/15 | <...> |
| Observability — logs, metrics, alerts, dashboards | 15 | __/15 | <...> |
| Security — authn/z, secrets, clean dependency scan, tenant isolation verified | 15 | __/15 | <...> |
| Deployability — automated deploy, tested rollback/roll-forward | 15 | __/15 | <...> |
| Testing — unit/integration/e2e + full-system wreck-it-ralph PASS | 10 | __/10 | <...> |
| Operability — runbook, ownership, on-call, Debt Ledger handed off | 10 | __/10 | <...> |
| **Total** | 100 | __/100 | |

≥85 Launch · 70–84 Conditional (dated mitigations) · <70 No launch
N/A items (never whole categories): <item — reason>

**Hard blockers** (any checked = no launch regardless of score):
- [ ] Open critical/high wreck-it-ralph finding
- [ ] No tested rollback
- [ ] MVT not met
- [ ] Critical vulnerability

**Go-live method:** <canary / feature flag / blue-green / direct> — <one-line reason>

### Mitigations (Conditional only)
| Gap | Mitigation | Due date |
|-----|------------|----------|
| <...> | <...> | <...> |

---

## T9 30/60/90 Review
File: `review-30-60-90.md`

| Metric | MVT | Target | Stretch | Day 30 | Day 60 | Day 90 | Tier achieved |
|--------|-----|--------|---------|--------|--------|--------|---------------|
| <...> | <...> | <...> | <...> | <...> | <...> | <...> | <MVT/Target/Stretch/none> |

### Katharsis §1 baseline (day 90; record *how* each was measured so later cycles repeat it)
| Area | Metric | Value | Method |
|------|--------|-------|--------|
| Performance | p50 / p95 / p99 / max latency (critical path) | <...> | <tool, load profile> |
| Performance | Throughput (peak / avg); cost per request | <...> | <...> |
| Reliability | Error rate; availability %; incidents P0–P3 | <...> | <...> |
| Reliability | MTBF / MTTD / MTTR | <...> | <...> |
| Velocity | Deploy frequency; lead time (commit → deploy); rollback rate | <...> | <...> |
| Quality | Test coverage; avg / max cyclomatic complexity; open debt items | <...> | <...> |

Debt Ledger hand-off: <done / pending — owner>
Recommended first Katharsis cycle: <date, 6–12 months post-launch>

---

## T10 Build Brief (Small scale)
File: `build-brief.md` (replaces 00–02 for Small builds)

**Problem:** <one paragraph>  **Scale:** Small  **Build type:** <component-in-platform | from-zero>
**Sponsor:** <name / role>  **Hard deadlines:** <date + reason, or none>

**Inherit / Extend / Build:** CI/CD <..> · Deployment <..> · Observability <..> · Testing <..> ·
Security/auth <..> · Data layer <..> · Conventions <..>

**Capacity:** `Available h/week = <engineers> × 40 × 0.8 × 0.85 × 0.75 = <n>` ·
`Timeline (weeks) = Σ Est h ÷ <n> × 1.25 = <w>`

**Requirements**
| ID | Requirement | Tier (MVT/Target/Stretch) | Test |
|----|-------------|---------------------------|------|
| FR1 | <...> | <...> | <...> |

**Non-functional requirements** (MVT = 80% of Target; Stretch = 120%)
| Metric | MVT | Target | Stretch |
|--------|-----|--------|---------|
| <e.g. p99 latency> | <...> | <...> | <...> |

**Unknowns**
| ID | Unknown | impact-if-wrong (5/3/1) | Route |
|----|---------|-------------------------|-------|
| U1 | <...> | <...> | <...> |

**Design sketch:** <components touched, interface, data flow in 3–6 lines>

**ADRs (light, inline)**
- ADR-001 — Decision: <...> · Why: <...> · Revisit if: <...>

**Roadmap**
| # | Slice | Depends on | Value | Risk Retired | Est h | Effort Score | PRIORITY | Floor-rule? | Ralph trigger? | Status |
|---|-------|------------|-------|--------------|-------|--------------|----------|-------------|----------------|--------|
| 0 | Skeleton (rollback required? <Y/N — blast radius>) | — | — | — | <h> | <...> | — | — | Y | <...> |
| 1 | <...> | <...> | <...> | <...> | <h> | <...> | <...> | <Y/N/N/A + reason> | <Y/N> | <...> |

Resume here: <next action>

Approval: <sponsor/user> — <date> (covers the Phase 0–II gates)
