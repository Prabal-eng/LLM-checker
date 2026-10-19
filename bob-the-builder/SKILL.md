---
name: bob-the-builder
description: Gated, risk-first methodology for designing and building new software, with scored ADRs, an independent adversarial reviewer (wreck-it-ralph), and a scored launch gate. Use when building a new system, service, module, or endpoint from scratch — greenfield work, "design and build X", standing up a new service, or adding a new capability or subsystem to an existing platform. For assessing or improving an existing system, use Katharsis instead.
---

# bob-the-builder

## Overview
A gated methodology for **building new systems**, from framing to a measured launch. Companion to
**Katharsis** (which improves existing systems). Core principle: **prove the foundations first,
retire the biggest risk next, and let someone else try to break it.**

Follow it faithfully. Before each phase, **read `reference.md`**: it has the tasks, rubrics and
gate checklists. Write every artifact from **`templates.md`**. Read **`wreck-it-ralph.md`** before
any review.

**Do NOT use for:** improving an existing system (Katharsis), bug fixes, or throwaway spikes.

## The 6 Phases

| Phase | Name | Gate to proceed |
|-------|------|-----------------|
| 0 | Framing | Scale, Inherit/Extend/Build, capacity; sponsor approval |
| I | Discovery | Every requirement testable (MVT/Target/Stretch); every impact-5 unknown has a route |
| II | Design | One-way doors have full ADRs; roadmap scored and sorted |
| III | Walking Skeleton | Deployed, observable, e2e-tested; wreck-it-ralph PASS |
| IV | Increments | Each slice meets its done-number; ralph PASS when triggered |
| V | Launch | LRS ≥ 85 (70–84 Conditional); no hard blockers; hand off to Katharsis |

**Every phase ends with a GATE: its deliverable exists and its checklist passes.** No feature work
before the skeleton gate.

## The Three Formulas (never improvise)

| Where | Formula |
|-------|---------|
| Increments | `PRIORITY = (Value + 2 × Risk Retired) / Effort Score`. Dependencies first; security, tenant-isolation and data-integrity slices are mandatory |
| ADRs | `Σ(criterion 1–5 × weight)`: fit 0.30, risk 0.20, ops cost 0.20, reversibility 0.15, familiarity 0.15 |
| Launch | Launch Readiness Score /100: ≥85 Launch, 70–84 Conditional, <70 No launch |

Exact rubrics are in `reference.md`.

## Key rules
These hold in every phase, even if you only skim `reference.md`.

- **Scale:** Small builds collapse Phases 0–II into one `build-brief.md` (T10) and still pass
  every gate. Very Large builds split into subsystems, with Bob run per subsystem.
- **Inherit before building.** Inside an existing platform, the host's choice wins unless its
  Requirement fit is below 3.
- **ADR door:** reversal costs less than one increment → light ADR (3 lines). Otherwise full ADR
  with ≥2 real options and the weighted matrix.
- **Walking Skeleton:** CI, automated deploy, logs/metrics/health, ≥1 e2e test. Rollback is proven
  only if the skeleton shares blast radius with anything live; always at Launch.
- **Increments are vertical slices** through every layer, each deployable alone. Record **Est h**
  per slice. Timelines use Σ Est h ÷ capacity × 1.25, never summed Effort Scores.
- **Build ladder** for code inside a slice (stop at the first rung that holds):
  1. Needed at all?
  2. Already in this codebase?
  3. Standard library?
  4. Platform feature?
  5. Installed dependency?
  6. One line?
  7. Minimum code.

  A **new dependency needs a light ADR** saying why rungs 2–5 failed. Never simplify away
  validation, security, tenant isolation or data-loss handling.
- **Debt:** every shortcut goes in the Debt Ledger *and* gets a `ponytail: D<n> <ceiling>,
  <upgrade path>` marker in the code.
- **wreck-it-ralph** runs:
  - after the skeleton
  - after any increment touching security, data integrity, tenant isolation or the critical path
  - otherwise every 2 increments
  - full-system at Launch

  Fill the template's slots only. Run the commands yourself first. Critical/High findings block
  gates, and a Critical/High `Refute` closes only after a **blind re-run** stops reporting it.
- **Launch hard blockers** (override any score): an open Critical/High ralph finding, no tested
  rollback, MVT not met, a critical vulnerability. N/A applies to single items, never to whole
  categories.
- **Handoff:** the 30/60/90 actuals become Katharsis's §1 baseline (T9) and the Debt Ledger becomes
  its seed catalog. First Katharsis cycle 6–12 months after launch.

## Artifacts
`docs/bob/<build-name>/` in the target repo. Resume from `roadmap.md` ("Resume here").

## Common Mistakes
- Starting features before the skeleton gate passes.
- Horizontal layers instead of vertical slices.
- Easy-first ordering instead of the priority formula.
- Summing Effort Scores as hours (they're unitless; use Est h).
- Writing a custom ralph prompt instead of filling its slots.
- Closing a Critical/High ralph finding by argument, or without a blind re-run.
- A full ADR for a two-way door, or a light ADR for a one-way door.
- Rebuilding what the host platform already provides, or skipping the build ladder: abstractions
  with one implementation, or a new dependency with no light ADR.
- Shortcuts not logged in the Debt Ledger or not marked `ponytail: D<n>` in code.

## Maintaining this skill
After editing any file in this directory, run `sh check.sh`. It must print
`OK: all invariants hold`. It guards the formulas, the template IDs, the Katharsis alignment and
past fixes.
