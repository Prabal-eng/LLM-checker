# bob-the-builder — Complete Reference

## Contents
- Phase 0 — Framing (scale, inherited context, capacity)
- Phase I — Discovery (requirements, NFRs, constraints, unknowns, POC)
- Phase II — Design (architecture, ADRs, patterns, risk register, roadmap)
- Phase III — Walking Skeleton (required foundations, rollback rule)
- Phase IV — Increments (vertical slices, per-increment loop, build ladder, exit to Launch)
- Phase V — Launch (LRS scoring guidance, go-live, 30/60/90)
- Formulas (F1 increment priority, F2 ADR scoring + anchors, F3 LRS + N/A rule)
- Scale Adaptation · Domain Adaptation · Handoff to Katharsis

Greenfield build methodology. Companion to **Katharsis** (which purifies existing systems). Bob
builds; once the system has real dependents, hand off to Katharsis.

Templates for every deliverable are in `templates.md` (T1–T10). The adversarial review prompt and
its rules are in `wreck-it-ralph.md`. Use them as written.

**6 phases, each ending in a GATE:**

| Phase | Name | Core question | Deliverable (gate) |
|-------|------|---------------|--------------------|
| 0 | Framing | How big, what do we inherit, do we have capacity? | Framing Doc + sponsor approval |
| I | Discovery | What problem, for whom, what's success, what don't we know? | Requirements & Unknowns Register + MVT/Target/Stretch |
| II | Design | What shape, and why this shape over alternatives? | Architecture + ADR set + Build Roadmap |
| III | Walking Skeleton | Can the thinnest e2e slice deploy, be observed, be tested? | Skeleton live in prod-like env + wreck-it-ralph PASS |
| IV | Increments | Add value slice by slice, riskiest first | Validated increments + wreck-it-ralph when triggered |
| V | Launch | Ready for production, and did it deliver? | Launch Readiness Score + full-system wreck-it-ralph + 30/60/90 review + baseline handoff to Katharsis |

A gate passes only when (1) the deliverable exists in `docs/bob/<build-name>/` and (2) every
self-validation item is checked. If an item can't be checked, the phase is not done — fix it or
record why it doesn't apply.

---

## Phase 0 — Framing

**Core question:** How big is this, what do we inherit, and do we have the capacity?

### Tasks
1. **Write the problem statement.** One paragraph: what is missing, who feels it, why now. Name
   the sponsor (or the user, for solo work) and any hard deadlines.
2. **Classify scale.** Scale sets the calendar and how heavy each phase is.

   | Scale | Typical build | Calendar | Phase effort |
   |-------|---------------|----------|--------------|
   | Small | New module / endpoint | days – 2 weeks | Hours |
   | Medium | New service | 3 – 8 weeks | Days |
   | Large | New multi-service system | 2 – 4 months | Days to weeks |
   | Very Large | New product | 4+ months | Each phase is a project; split into subsystems |

   If the build falls between two rows, choose the larger one. Very Large builds are split into
   subsystems, and Bob is run per subsystem (see Scale Adaptation).
3. **Inherited-context inventory** (component-in-platform builds). For each area — CI/CD,
   deployment, observability, testing, security/auth, data layer, conventions — record what the
   host provides and mark it:
   - **Inherit:** use as-is. This is the default.
   - **Extend:** use it, adding something small (a new dashboard, a new pipeline stage).
   - **Build:** the host has nothing usable, so build it. Justify every Build.

   From-zero builds mark every area Build. Each Build line adds skeleton work in Phase III.
4. **Capacity math** (reused from Katharsis):
   - Available h/week = `engineers × 40 × 0.8 × 0.85 × 0.75`
     (0.8 = meetings, 0.85 = context switching, 0.75 = interrupt budget; as in Katharsis §0C)
   - Timeline (weeks) = `effort ÷ capacity × 1.25` (the 1.25 is the unknowns buffer)
   - **Worked example:** 2 engineers → 2 × 40 × 0.8 × 0.85 × 0.75 = **40.8 h/week**. An effort of
     300 h → 300 ÷ 40.8 × 1.25 ≈ **9.2 weeks**.
   - A rough effort estimate is fine at this stage. It gets re-derived from the scored roadmap in
     Phase II.
5. **Get approval.** Show the sponsor the scale, inherit table and timeline.

### Deliverable
T1 Framing Doc → `00-framing.md` (Small: T10 `build-brief.md`).

### Self-validation
- [ ] The problem statement is one paragraph and names who feels the problem
- [ ] Scale chosen, with a stated rationale
- [ ] Every inherited-context area is marked Inherit / Extend / Build, and every Build is justified
- [ ] Capacity and timeline computed with the formula, not guessed
- [ ] Timeline compared to hard deadlines; any conflict is surfaced to the sponsor

**Gate:** Framing Doc complete + sponsor (or user, for solo work) approval.

---

## Phase I — Discovery

**Core question:** What problem, for whom, what does success look like, and what don't we know?

### Tasks
1. **Problem and users.** Who uses it, what job they need done, and how they do it today.
2. **Functional requirements, tiered:**
   - **MVT (Minimum Viable Target):** without it, launching isn't worth it.
   - **Target:** what we plan to deliver.
   - **Stretch:** nice to have if capacity allows.

   Each requirement needs a **Test** column saying how it's verified. If you can't write the test,
   the requirement isn't clear enough yet.
3. **Non-functional requirements** in **domain metrics** (see Domain Adaptation), each with
   MVT / Target / Stretch values. **MVT = 80% of Target; Stretch = 120% of Target** (as in
   Katharsis §5C). For "lower is better" metrics, apply the ratio in the worse/better direction:
   a Target p99 of 200 ms gives MVT 250 ms and Stretch ≈ 167 ms.
4. **Constraints**, hard vs soft:
   - **Hard:** can't be broken without fundamental redesign, e.g. laws of physics, regulation,
     contracts, host-platform invariants such as tenant isolation. Design around them and
     document them.
   - **Soft:** preferences, conventions, "we've always done it this way". These may be challenged,
     and the challenge is recorded in an ADR.
5. **Unknowns Register.** List everything that, if wrong, changes the build. Rate each one's
   impact-if-wrong:
   - **5** = invalidates the architecture
   - **3** = invalidates a component
   - **1** = minor

   Give each unknown a resolution route: **research** (read docs or code, answer in hours),
   **POC**, or **increment** (the increment that exercises it answers it). The impact feeds
   Formula 1's Risk Retired, so an increment that tests an impact-5 unknown sorts to the top.
6. **POC protocol** (Katharsis §5A-POC). Use a POC for impact-5 unknowns that can't wait for an
   increment, e.g. when the answer decides the architecture itself:
   - One measurable question ("Can Atlas vector search return top-10 in <100 ms at 1M docs?")
   - Time-boxed to 1–5 days, with the success criterion written before starting
   - Outcome is **Go** (proceed), **No-Go** (drop the option), or **Redesign** (change the approach)
   - POC code is thrown away. Only the answer carries forward.
7. **Success metrics.** MVT / Target / Stretch for the outcome the sponsor cares about. These are
   what the 30/60/90 review measures against.

### Deliverable
T2 Requirements & Unknowns Register → `01-discovery.md`.

### Self-validation
- [ ] Every requirement has a tier and a concrete Test
- [ ] NFRs use domain metrics with MVT / Target / Stretch values
- [ ] Every constraint is classified hard or soft, with a source
- [ ] Every unknown has impact-if-wrong and a resolution route
- [ ] Every impact-5 unknown is either resolved (POC/research) or mapped to an early increment

**Gate:** every requirement is testable; every impact-5 unknown has a resolution route.

---

## Phase II — Design

**Core question:** What shape does the system take, and why this shape over the alternatives?

### Tasks
1. **Architecture.** Components (responsibility, data ownership, new vs inherited), interfaces
   (contracts, versioning), data flow (sync/async, where state lives), and the **critical path**
   (the chain whose failure means the system is down).
2. **ADRs for every significant decision** (Formula 2):
   1. **Classify the door first.**
      - **Two-way door:** reversal costs less than one increment. Write a light ADR (T4, 3 lines),
        take the sensible or inherited default, and move on.
      - **One-way door:** datastore, public API shape, tenancy model, anything with migration cost.
        Write a full ADR (T5) with ≥2 real options and a decision matrix.
   2. Score options with the weighted matrix.
   3. Apply the tie rule and inherit-by-default.
   4. Record the consequences, including the follow-up work they create.
3. **Pattern selection.** Katharsis §4A pattern families (Service Architecture, Data, Observability, Resilience, Deployment)
   are used as a *"which do we need at this scale"* checklist. **Not scored /120**, since there's
   no existing system to score. For each family, name the patterns needed and why (Katharsis §4A
   lists 8 per family). Observability and Deployment patterns usually land in the skeleton; the
   rest become part of an increment's scope.
4. **Risk register** (Katharsis §5D shape): risk, probability, impact, leading indicators,
   mitigation. Unknowns with impact ≥3 that aren't resolved yet appear here too.
5. **Build Roadmap:**
   - **Skeleton definition:** the layers it touches, its one e2e test, and whether rollback is
     required (the blast-radius rule, Phase III).
   - **Candidate increments:** vertical slices, each scored with Formula 1.
   - **Sort:** dependencies first (topological), then PRIORITY descending.
   - **Floor-rule** increments (security, tenant isolation, data integrity) are flagged. They're
     mandatory before Launch whatever their score.
   - Flag which increments trigger wreck-it-ralph (see Phase IV).
   - Re-derive the Phase 0 timeline: Σ Est h of all increments to Launch (incl. skeleton) ÷
     capacity × 1.25. Effort Scores are unitless rubric values, so never sum them as hours. If
     the result differs from Phase 0 by over 25%, tell the sponsor.
6. **Initialize the Debt Ledger** (T7). It starts empty; every intentional shortcut from here on
   goes into it.

### Deliverable
T3 Design Doc → `02-design.md`; T4/T5 ADRs → `adr/NNN-<decision>.md`; T6 Build Roadmap →
`roadmap.md`; T7 Debt Ledger → `debt-ledger.md`.

### Self-validation
- [ ] Components, interfaces, data flow and critical path are documented
- [ ] Every significant decision has an ADR, and every ADR has its door classified
- [ ] Every one-way door has a full ADR with ≥2 real options and a scored matrix
- [ ] Inherit-by-default applied: any rejected host choice shows Requirement fit < 3
- [ ] The roadmap is scored, sorted, floor-rule flagged, and ralph triggers are marked
- [ ] The Debt Ledger file exists

**Gate:** architecture documented, all one-way-door decisions have full ADRs, roadmap sorted.

---

## Phase III — Walking Skeleton

**Core question:** Can the thinnest end-to-end slice deploy, be observed, and be tested?

The skeleton is the **thinnest end-to-end path through every layer, with trivial behavior**. For
example: one RPC that accepts a request, touches the datastore, and returns a constant-shaped
response. It proves the foundations work, not the features.

### Required
- **CI:** builds and runs tests on every push.
- **Automated deploy** to a prod-like environment. No manual steps.
- **Observability:** logs, metrics and a health check are visible where on-call would look.
- **≥1 e2e test** that exercises the full path and runs in CI or post-deploy.
- **Inherit / Extend** items from Phase 0 are wired in. Build items are built here.

### Rollback rule (D5)
- Rollback is required **only if the skeleton shares blast radius with anything live**: deployed
  inside a running host platform, sharing a database or queue with live traffic, or sharing
  ingress/routing. In that case, prove rollback (or roll-forward) works before the gate.
- Rollback is **not required** for an isolated system with zero users. Rolling back something
  nobody depends on is ceremony.
- A tested rollback/roll-forward is **always required at Launch**, whatever was done here.

### Tasks
1. Implement the skeleton path with trivial behavior.
2. Wire CI, deploy, observability and the e2e test.
3. Deploy, and confirm health, logs and metrics are visible.
4. If the blast-radius rule applies, deploy, roll back, and re-deploy.
5. **Run wreck-it-ralph** (scope `skeleton`) per `wreck-it-ralph.md`. Triage every finding.

### Deliverable
The skeleton is live in a prod-like env, and `ralph/01-skeleton.md` exists with its verdict.

### Self-validation
- [ ] CI is green on the skeleton commit
- [ ] Deploy is automated end to end
- [ ] Health, logs and metrics have been seen in the target environment (link or screenshot)
- [ ] The e2e test passes against the deployed skeleton
- [ ] Rollback proven, if the blast-radius rule applies
- [ ] wreck-it-ralph verdict PASS, or every Critical/High finding is fixed and re-run to PASS

**Gate:** deployed + observable + tested + wreck-it-ralph has no open critical/high findings.

**No feature work before this gate passes.** A feature built on unproven foundations is rework.

---

## Phase IV — Increments

**Core question:** Add value slice by slice, riskiest first. Does each slice hold?

An increment is a **vertical slice**: it passes through every layer and is independently built,
deployed and validated.
- Horizontal (wrong): "build all the DB tables, then all the APIs, then the UI".
- Vertical (right): "one query type, end-to-end: API → service → datastore → response → metric".

Each increment should be deployable on its own and leave the system working.

### Per-increment loop
1. **Define "done":** the behavior, the assumption tested (an unknown ID), and the number that
   proves it.
2. **Build lean, with tests.** Climb the build ladder (below) for every piece of the slice. Tests
   ship with the slice, not later.
3. **Deploy** through the skeleton-proven pipeline.
4. **Validate:** compare the metric to its target, and mark the tested unknown confirmed or
   refuted. A refuted impact-5 unknown sends you back to Phase II for the affected ADRs.
5. **Log intentional debt** to the Debt Ledger, and mark it in the code at the shortcut:
   `ponytail: D<n> <ceiling>, <upgrade path>`, e.g. `// ponytail: D3 global lock, per-account
   locks if throughput matters`. The ledger's Code marker column points back at the file:line.
6. **Re-sort the roadmap.** Re-score if what you learned changed Value, Risk Retired or Effort.
7. **Run wreck-it-ralph if triggered.**

### Build ladder
Inherit / Extend / Build decides *infrastructure* in Phase 0. The ladder decides *code* inside
each slice. Stop at the first rung that holds:

1. **Does it need to exist?** No requirement or done criterion needs it → don't build it.
2. **Already in this codebase?** Reuse the helper, type or pattern that's there.
3. **Standard library does it?** Use it.
4. **Platform feature covers it?** DB constraint over app code, native control over a library.
5. **Already-installed dependency solves it?** Use it.
6. **Can it be one line?** One line.
7. **Only then:** the minimum code that works. No interface with one implementation, no config for
   a value that never changes, no scaffolding "for later".

**A new dependency needs a light ADR** (T4) whose "Why" names why rungs 2–5 failed and why the
minimum code is more than a few lines.

**Never climb past:** validation at trust boundaries, security, tenant isolation, and error
handling that prevents data loss. These are floor-rule territory; lean never means removing them.

### wreck-it-ralph cadence
Defined once, in `wreck-it-ralph.md` → "When to run". Mark each increment's trigger in the roadmap.

### Tracking
Medium+ only: a weekly row per active increment (est h vs actual h, status) in `roadmap.md`. If
actuals run more than 25% over estimates for 2 consecutive weeks, re-derive the timeline with the
capacity formula and tell the sponsor.

### Exit to Launch
You exit to Launch when the **MVT is met and all floor-rule increments are done**. Any remaining
increments either continue or are deferred post-launch; record the decision in `roadmap.md`.

### Self-validation (per increment)
- [ ] The done block is filled in, including the actual number
- [ ] Tests merged with the slice
- [ ] Every new dependency has a light ADR; no code without a requirement behind it
- [ ] Deployed via the pipeline, not by hand
- [ ] Tested unknown marked confirmed or refuted
- [ ] Shortcuts logged to the Debt Ledger, each with a `ponytail: D<n>` marker in code
- [ ] Roadmap re-sorted; ralph run and PASS if triggered

**Gate:** done-criteria number met, tests shipped, deployed, roadmap re-sorted, ralph PASS if
triggered.

---

## Phase V — Launch

**Core question:** Is it ready for production, and did it deliver?

### Tasks
1. **Run full-system wreck-it-ralph** (scope `full system`) and triage every finding.
2. **Score the Launch Readiness Score** (Formula 3) with evidence for every category.
3. **Check the hard blockers.** If any applies, there's no launch regardless of score.
4. **Choose the go-live method:**
   - **Canary:** high-traffic or high-risk changes; ramp by percentage while watching metrics.
   - **Feature flag:** behavior must be toggleable inside a host platform.
   - **Blue-green:** stateless services that need instant cutback.
   - **Direct:** only for internal tools with zero dependents.
5. **Go live**, and watch the metrics during the first hours.
6. **Hold the 30/60/90 review.** Compare actuals to MVT/Target/Stretch at days 30, 60 and 90.
7. **Hand off to Katharsis** (see Handoff to Katharsis).

### LRS scoring guidance
Points, thresholds, hard blockers and the N/A rule are in **Formula 3**; this table only says how
to award the points. Give full points only when the evidence exists, not when it's promised.

| Category | Full points | Half points | Zero |
|----------|-------------|-------------|------|
| Functionality | Every MVT requirement verified by its Test; Target status recorded | MVT met but ≥1 verified only manually | Any MVT requirement unverified |
| Reliability | SLOs measured under realistic load; error paths tested | Measured at low load only, or error paths partly tested | Not measured |
| Observability | Logs, metrics, alerts routed to on-call, dashboard | Logs + metrics but no alerts or dashboard | Logs only, or none |
| Security | AuthN/Z enforced, secrets managed, clean dependency scan, tenant isolation tested | One item missing, non-critical | Tenant isolation or authZ untested |
| Deployability | Automated deploy + rollback/roll-forward tested in prod-like env | Automated deploy, rollback documented but not tested | Manual deploy |
| Testing | Unit/integration/e2e in CI + full-system ralph PASS | Ralph PASS but a test layer missing | Ralph not run or FAIL |
| Operability | Runbook, named owner, on-call rota, Debt Ledger handed off | Runbook + owner only | None |

### Deliverable
T8 → `launch-readiness.md`; `ralph/NN-full-system.md`; T9 → `review-30-60-90.md`.

### Self-validation
- [ ] Full-system ralph PASS
- [ ] Every LRS category scored with linked evidence
- [ ] No hard blocker applies
- [ ] Rollback/roll-forward tested in a prod-like env
- [ ] The go-live method is chosen and justified
- [ ] 30/60/90 dates are scheduled

**Gate:** LRS ≥ 85 (or 70–84 Conditional with dated mitigations) and no hard blockers.

---

## Formulas

### Formula 1 — Increment priority
**PRIORITY = (Value + 2 × Risk Retired) / Effort Score**

- **Value** (0–10, capped at 10). Add up the points that apply:
  - MVT requirement: +5
  - Target requirement: +3
  - Stretch-only: +1
  - Unblocks ≥2 other increments: +2
- **Risk Retired** (0–5):
  - Tests an impact-5 unknown: 5
  - Impact-3: 3
  - Impact-1: 1
  - None: 0
  - Several unknowns tested: use the highest (don't sum)
- **Effort Score** (the Katharsis rubric):
  - Hours base: <2h=1, 2–8h=2, 8–24h=4, 24–40h=6, 40–80h=8, >80h=10
  - Multiply by complexity (1.0/1.5/2.0), risk (1.0/1.3/1.8) and dependencies (1.0/1.5/2.0)
  - Record the hours estimate itself as **Est h**. Timelines use Est h, never Effort Scores.
- **Ordering:** dependencies first (topological), then PRIORITY descending.
- **Floor rule:** security, tenant-isolation and data-integrity increments are mandatory before
  Launch regardless of score. A floor category that can't apply (e.g. tenant isolation in a
  single-tenant system) is marked **N/A** in the roadmap with a one-line reason.
- **Examples:**
  - Atlas vector exact-match (MVT +5, tests impact-5 unknown, Effort 4): (5 + 2×5) / 4 = **3.75**
  - Pagination (Target +3, no unknown, Effort 2): (3 + 0) / 2 = **1.5**

  Exact-match goes first even though it costs more, because it retires the architecture risk.

### Formula 2 — ADR option scoring
1. **Classify reversibility.**
   - **Two-way door** (reversal costs < one increment): light ADR, 3 lines, sensible default.
   - **One-way door:** full ADR, ≥2 real options, decision matrix.
2. **WEIGHTED SCORE = Σ(criterion 1–5 × weight)**:

   | Criterion | Weight |
   |-----------|--------|
   | Requirement fit | 0.30 |
   | Risk | 0.20 |
   | Operational cost/complexity | 0.20 |
   | Reversibility | 0.15 |
   | Team familiarity | 0.15 |

3. **Score each criterion 1–5, where higher is always better.** For Risk and Operational cost,
   5 means the *lowest* risk or cost. Use these anchors; 2 and 4 are in-between:

   | Criterion | 5 | 3 | 1 |
   |-----------|---|---|---|
   | Requirement fit | Meets all MVT + Target requirements as-is | Meets MVT; Target needs workarounds | Misses an MVT requirement |
   | Risk | Proven at our scale; no open impact-5 unknowns | Proven elsewhere; open impact-3 unknowns | Unproven; an open impact-5 unknown depends on it |
   | Operational cost/complexity | Nothing new to run (managed or inherited) | One new component with runbook-able ops | New infra with its own scaling, upgrades and on-call |
   | Reversibility | Swap in < one increment (behind an interface) | Swap needs a planned migration | Data or contract lock-in; swap is a rebuild |
   | Team familiarity | Team runs it in production and on-call today | Used in POCs or side projects, never in production | Nobody on the team has used it |

   Team familiarity covers the builders **and** the operators. An unfamiliar choice costs a
   learning curve, subtle bugs, harder on-call and a single-person dependency. It is weighted low
   on purpose: it tips close calls but can't override a real Requirement-fit gap.

- **Tie rule:** if the top two are within 10%, pick the more reversible one.
- **Inherit-by-default** (component-in-platform): the host platform's existing choice wins unless
  its Requirement fit < 3.
- **ADR fields:** title, status, context, reversibility, options, matrix, decision, consequences,
  date.

### Formula 3 — Launch Readiness Score (100 pts)
| Category | Pts |
|----------|-----|
| Functionality — MVT met, verified vs requirements | 20 |
| Reliability — SLOs measured under realistic load, error paths handled | 15 |
| Observability — logs, metrics, alerts, dashboards | 15 |
| Security — authn/z, secrets, clean dependency scan, tenant isolation verified | 15 |
| Deployability — automated deploy, tested rollback/roll-forward | 15 |
| Testing — unit/integration/e2e + full-system wreck-it-ralph PASS | 10 |
| Operability — runbook, ownership, on-call, Debt Ledger handed off | 10 |

- ≥85 Launch · 70–84 Conditional (dated mitigations) · <70 No launch.
- **Hard blockers regardless of score:** open critical/high wreck-it-ralph finding; no tested
  rollback; MVT not met; critical vulnerability.
- **N/A rule:** an *item* inside a category that can't apply (e.g. tenant isolation in a
  single-tenant system) is marked N/A with a one-line reason, and the category is scored on its
  remaining items. A whole category is never N/A: all seven apply to any production launch.

### Reused from Katharsis unchanged
Capacity math; Effort Score rubric; MVT/Target/Stretch; POC protocol; hard/soft constraint
classification; risk register shape.

---

## Scale Adaptation

| Scale | Execution |
|-------|-----------|
| Small | Single session. Phases 0–II collapse into one `build-brief.md` (ADRs inline). wreck-it-ralph via subagent. LRS still scored. |
| Medium | Full artifact set across sessions; resume from `roadmap.md`. |
| Large / Very Large | Each phase its own project; split into subsystems, run Bob per subsystem. |

- **Small:** fill T10 in one pass. The Phase 0–II gates are checked together against the brief.
  The skeleton and increments may land in the same session. Wreck-it-ralph still runs as a
  subagent after the skeleton, and the LRS is still scored before shipping.
- **Medium:** one artifact per phase. Expect several sessions.
- **Large / Very Large:** decompose in Phase II into subsystems with explicit interfaces. Each
  subsystem gets its own `docs/bob/<build-name>-<subsystem>/`. The parent keeps the framing,
  cross-subsystem ADRs and an integration roadmap.
- **Resume rule:** open `roadmap.md` and read its "Resume here" line. Every session ends by
  updating it.

---

## Domain Adaptation

Substitute domain-specific metrics and patterns before Phase I.

### Backend service / gRPC API
- **Target metrics:** p50/p99 latency, error rate, throughput.
- **Typical skeleton:** one RPC/endpoint → service → datastore round-trip, with a health check and
  a trace.
- **Key ADRs:** API style/versioning, datastore, auth propagation, multi-tenancy model.
- **Must-have patterns:** timeouts + retries with backoff, idempotent writes, tenant filtering at
  the data layer.

### Web application
- **Target metrics:** Core Web Vitals (LCP/INP/CLS), error rate.
- **Typical skeleton:** one route rendering server data, deployed behind a CDN.
- **Key ADRs:** rendering model, state management, auth.
- **Must-have patterns:** error boundaries, input validation at the server.

### Data pipeline
- **Target metrics:** freshness/lag, completeness, throughput, cost per run.
- **Typical skeleton:** one source → one transform → one sink, scheduled, with a row-count check.
- **Key ADRs:** batch vs stream, orchestration, idempotency/replay.
- **Must-have patterns:** idempotent sinks, dead-letter, schema validation.

### Mobile app
- **Target metrics:** crash-free sessions, cold start, app size.
- **Typical skeleton:** one screen calling one API, built via CI to an internal track.
- **Key ADRs:** native vs cross-platform, offline strategy, release channel.
- **Must-have patterns:** graceful offline, forced-upgrade path.

---

## Handoff to Katharsis

- **Baseline:** launch actuals (the 30/60/90 numbers) become the Katharsis §1 baseline. They must
  be measured the same way so later cycles can prove improvement.
- **Debt:** the Debt Ledger becomes the Katharsis §2 seed catalog. Each entry already has a cost
  and a repay trigger.
- **Timing:** run the first Katharsis cycle 6–12 months after launch, or earlier if velocity or
  reliability degrade.
- **Boundary:** once the system has real dependents, improvement work uses Katharsis, not Bob. New
  capabilities added to it later can still use Bob as component-in-platform builds.
