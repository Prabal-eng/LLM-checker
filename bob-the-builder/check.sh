#!/bin/sh
# Invariant check for bob-the-builder. Usage: sh check.sh [skill-dir]. Prints failures; exit 1 if any.
d=${1:-$(cd "$(dirname "$0")" && pwd)}
k=${KATHARSIS:-$HOME/.claude/skills/katharsis/reference.md}
fails=0
fail() { echo "FAIL: $1"; fails=$((fails+1)); }
has() { grep -qF -- "$2" "$d/$1" || fail "$1 missing: $2"; }
lacks() { grep -qiE -- "$2" "$d/$1" && fail "$1 must not contain: $2"; }

# Files present
for f in SKILL.md reference.md templates.md wreck-it-ralph.md; do [ -f "$d/$f" ] || fail "missing $f"; done

# Pattern families match Katharsis §4A (drift guard)
for fam in "Service Architecture" Data Observability Resilience Deployment; do
  grep -qF "**$fam:**" "$k" || fail "katharsis no longer has family: $fam (update this check)"
  grep 'pattern families' "$d/reference.md" | grep -qF "$fam" || fail "reference.md pattern-selection line missing family: $fam"
  awk '/^## T3 /,/^## T4 /' "$d/templates.md" | grep -qF "| $fam |" || fail "templates.md T3 missing family row: $fam"
done
awk '/^## T3 /,/^## T4 /' "$d/templates.md" | grep -qE '^\| (Architecture|Integration|Operational) \|' && fail "T3 has non-Katharsis family row"

# Template IDs referenced exist
for t in $(grep -ohE '\bT[0-9]+\b' "$d/reference.md" "$d/SKILL.md" | sort -u); do
  grep -qE "^## $t " "$d/templates.md" || fail "template $t referenced but not defined"
done

# LRS sums to 100 wherever the point table appears
for f in reference.md templates.md; do
  s=$(grep -oE '^\| [A-Z][a-z]+ (—|\()[^|]*\| ?[0-9]+ ?\|' "$d/$f" | grep -oE '[0-9]+ ?\|$' | tr -d ' |' | paste -sd+ - | bc)
  [ "$s" = 100 ] || fail "$f LRS points sum to ${s:-none}, expected 100"
done

# Rejected / out-of-scope features and attribution
for f in SKILL.md reference.md templates.md wreck-it-ralph.md; do
  lacks "$f" 'pre-mortem|kill/pivot|falsification|STRIDE|error budget|re-entry'
  lacks "$f" 'co-authored|generated with|anthropic'
done

# Description: what it is + when to use (official best practices), but no step list (writing-skills SDO)
desc=$(grep '^description:' "$d/SKILL.md")
echo "$desc" | grep -qiE 'produces|walking skeleton|launch readiness' && fail "SKILL.md description lists workflow steps"
echo "$desc" | grep -qi 'methodology' || fail "SKILL.md description doesn't say what the skill is"
echo "$desc" | grep -q 'Use when' || fail "SKILL.md description missing 'Use when' triggers"
[ ${#desc} -le 1037 ] || fail "SKILL.md description over 1,024 chars"

# Ralph template slots
for s in REPO_PATH BRANCH SCOPE REQUIREMENTS DONE_CRITERIA RUN_COMMANDS TEST_COMMANDS E2E_EXAMPLES; do has wreck-it-ralph.md "{{$s}}"; done

# Katharsis cycle-1 fixes stay fixed
lacks reference.md 'sum of Effort Scores'                                     # D2
grep -q '^| # | Slice .*| Est h |' "$d/templates.md" || fail "T6/T10 increments table missing Est h column"  # D2
has reference.md "0.85 = context switching"                                   # D3
awk '/^## T10/,0' "$d/templates.md" | grep -q 'Available h/week' || fail "T10 missing capacity"   # D4
awk '/^## T10/,0' "$d/templates.md" | grep -q 'Approval:' || fail "T10 missing approval"          # D4
awk '/^## T10/,0' "$d/templates.md" | grep -qi 'non-functional' || fail "T10 missing NFRs"        # D4
has reference.md "Stretch = 120% of Target"                                   # D5
lacks reference.md 'Katharsis §3B way'                                        # D11
has reference.md "N/A"                                                        # D12
grep '^| `Refute`' "$d/wreck-it-ralph.md" | grep -q 're-run' || fail "Refute of Critical/High not tied to a ralph re-run"  # D7
has wreck-it-ralph.md "Before dispatch"                                       # D8
for m in p99 MTTR "deploy frequency" coverage; do grep -qiF "$m" "$d/templates.md" || fail "T9 missing §1 metric: $m"; done  # D6
# Official skill best practices
b=$(awk 'f>=2; /^---$/{f++}' "$d/SKILL.md" | wc -l); [ "$b" -lt 500 ] || fail "SKILL.md body is $b lines (<500)"
for f in reference.md templates.md; do
  head -15 "$d/$f" | grep -q '^## Contents' || fail "$f (>100 lines) missing table of contents at top"
done
has SKILL.md "## Key rules"
has SKILL.md "check.sh"

# Lean-build integration (ponytail)
has reference.md "### Build ladder"                                           # 1 ladder
grep -q '^2\. \*\*Build lean' "$d/reference.md" || fail "increment loop step 2 not tied to the ladder"
has reference.md "new dependency needs a light ADR"                           # 2 dependency ADR
has templates.md "rungs 2–5"                                                  # 2 T4 hint
has reference.md "ponytail: D"                                                # 3 comment ↔ ledger
awk '/^## T7 /,/^## T8 /' "$d/templates.md" | grep -q 'Code marker' || fail "T7 missing Code marker column"
grep 'Category:' "$d/wreck-it-ralph.md" | grep -q 'unnecessary complexity' || fail "ralph category list missing unnecessary complexity"  # 4
grep -q 'no requirement' "$d/wreck-it-ralph.md" || fail "ralph procedure missing requirement-traceability probe"

[ $fails -eq 0 ] && echo "OK: all invariants hold" || { echo "$fails failure(s)"; exit 1; }
