---
step: phase-16.40
title: Doc-reconciliation sweep -- codify 3rd-CONDITIONAL auto-FAIL clause (#26)
cycle_date: 2026-04-25
harness_required: true
forward_cycle: true
parent_phase: phase-16
deliverables:
  - docs/runbooks/per-step-protocol.md (+~14 lines: new subsection under EVALUATE)
  - CLAUDE.md (extend F1 bullet)
  - .claude/agents/qa.md (extend Constraints section)
---

# Sprint Contract -- phase-16.40

## Research-gate summary

`handoff/current/phase-16.40-research-brief.md`. tier=simple, 6 in-full,
14 URLs, recency scan present, gate_passed=true. Source of the
informal rule: `handoff/archive/phase-16.21/evaluator_critique.md:97-100`.
External validation: Google SRE three-tier escalation +
mergeshield.dev "shared-evaluator bias" warning + Anthropic
harness-design hard-threshold model.

## Scope

Codify the 3rd-CONDITIONAL auto-FAIL rule in 3 durable doc locations
so it survives session restarts and informs every future Q/A
invocation. No code changes; doc-only sweep.

## Concrete plan

1. **`docs/runbooks/per-step-protocol.md`** — add new subsection
   `#### CONDITIONAL escalation clause (3rd-CONDITIONAL auto-FAIL)`
   under §4 EVALUATE (~14 lines verbatim from research brief).

2. **`CLAUDE.md`** — extend the F1 bullet at lines 269-271 to include
   the 3rd-CONDITIONAL clause as a sub-paragraph under the existing
   F1 retry-loop discipline.

3. **`.claude/agents/qa.md`** — add to Constraints section after
   line 165 (the existing verdict-shopping rule) as a
   `**3rd-CONDITIONAL auto-FAIL**` bullet.

## The rule

A single masterplan step-id accumulating 3+ consecutive CONDITIONAL
verdicts without an intervening PASS or FAIL forces the next Q/A
verdict to be FAIL.

- **Why:** prevents the harness from logging instead of correcting
  (mergeshield.dev shared-evaluator-bias forcing function).
- **Counter source:** `handoff/harness_log.md` grep for current step-id.
- **Reset:** on PASS, FAIL, or new step-id (structurally distinct).
- **Scoping:** per-step-id (NOT global; CONDITIONALs across different
  step-ids are independent problems).

## Success Criteria (verbatim, immutable)

```
grep -l "3rd-CONDITIONAL\|3-consecutive-CONDITIONAL\|third consecutive CONDITIONAL\|3rd consecutive" \
  CLAUDE.md docs/runbooks/per-step-protocol.md .claude/agents/qa.md \
  .claude/rules/*.md 2>/dev/null | wc -l | grep -q '^[3-9]'
```

(Passes when AT LEAST 3 of the 4 candidate file groups contain the
text -- i.e., CLAUDE.md + per-step-protocol.md + qa.md.)

Plus:
- `verbatim_text_added`: each of the 3 doc locations contains the
  rule with the same key phrasing ("3rd-CONDITIONAL", "auto-FAIL",
  "harness_log.md grep", "reset on PASS, FAIL, or new step-id").
- `no_code_changes`: no `*.py`, `*.ts`, `*.tsx`, `*.js`, `*.json`
  files modified outside handoff/.
- `existing_text_preserved`: F1 bullet's existing FAIL-counter
  language preserved (the new clause is additive).

## What Q/A must audit

1. Verification command exits 0 (>= 3 file matches).
2. The 3 doc locations all contain the rule with consistent phrasing
   (no contradictory variants).
3. Counter source is `handoff/harness_log.md` (not a new state file).
4. Reset criteria match across all 3 locations (PASS, FAIL, new step-id).
5. No code touched.
6. Existing F1 FAIL-counter language preserved in CLAUDE.md.
7. The Q/A subagent's snapshot now includes the rule (qa.md
   Constraints section).
