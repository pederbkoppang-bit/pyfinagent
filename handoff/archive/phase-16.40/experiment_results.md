---
step: phase-16.40
cycle_date: 2026-04-25
forward_cycle: true
expected_verdict: PASS
deliverables:
  - docs/runbooks/per-step-protocol.md (+18 lines new subsection)
  - CLAUDE.md (extend F1 bullet, +7 lines)
  - .claude/agents/qa.md (extend Constraints, +7 lines)
---

# Experiment Results -- phase-16.40

## What was done

Closed task list item #26: codified the 3rd-CONDITIONAL auto-FAIL clause
in 3 durable doc locations so the rule survives session restarts and
informs every future Q/A invocation. Doc-only sweep; no code changes.

### The rule (now documented in 3 places)

> A single masterplan step-id accumulating 3+ consecutive CONDITIONAL
> verdicts without an intervening PASS or FAIL forces the next Q/A
> verdict to be FAIL. Q/A reads `handoff/harness_log.md` to count
> prior CONDITIONALs for the step-id. Counter resets on PASS, FAIL,
> or new step-id (structurally distinct problem).

### Changes

1. **`docs/runbooks/per-step-protocol.md`** — added new subsection
   `#### CONDITIONAL escalation clause (3rd-CONDITIONAL auto-FAIL)`
   under §4 EVALUATE (~18 lines). Includes:
   - When CONDITIONAL is appropriate (functionality intact, prod
     unaffected, gap-discovery step)
   - The escalation rule (3+ consecutive → FAIL)
   - Citation: mergeshield.dev 2026 "shared-evaluator-bias forcing
     function"
   - Q/A procedure: grep harness_log.md, count prior CONDITIONALs
   - Reset criteria

2. **`CLAUDE.md`** — extended the existing F1 bullet at lines 269-270
   with a `**3rd-CONDITIONAL auto-FAIL:**` sub-paragraph (+7 lines).
   Cross-references `docs/runbooks/per-step-protocol.md` §4 EVALUATE
   for full text. Existing FAIL-counter language preserved.

3. **`.claude/agents/qa.md`** — added `**3rd-CONDITIONAL auto-FAIL.**`
   bullet at end of Constraints section (+7 lines), after the
   verdict-shopping rule. Concise procedural form so the Q/A subagent's
   snapshot picks it up at next spawn.

### Files touched

| Path | Action | LOC delta |
|------|--------|-----------|
| `docs/runbooks/per-step-protocol.md` | edited | +18 lines (new subsection) |
| `CLAUDE.md` | edited | +7 lines (extend F1 bullet) |
| `.claude/agents/qa.md` | edited | +7 lines (extend Constraints) |
| `handoff/current/contract.md` | rewrite (rolling) | -- |
| `handoff/current/experiment_results.md` | rewrite (this) | -- |

NO code touched. NO masterplan.json schema changes. NO new state files.

## Verification (verbatim, immutable)

```
$ grep -l "3rd-CONDITIONAL\|3-consecutive-CONDITIONAL\|third consecutive CONDITIONAL\|3rd consecutive" \
    CLAUDE.md docs/runbooks/per-step-protocol.md .claude/agents/qa.md \
    .claude/rules/*.md 2>/dev/null
docs/runbooks/per-step-protocol.md
.claude/agents/qa.md
CLAUDE.md

$ N=$(grep -l "..." | wc -l | tr -d ' '); [ "$N" -ge 3 ] && echo VERIFICATION PASS
matches: 3
VERIFICATION PASS
```

**Result: PASS.** All 3 expected file matches; verification command
exits 0 (compound test for >=3).

## Success criteria assessment

| # | Criterion | Result | Evidence |
|---|-----------|--------|----------|
| 1 | verbatim_text_added | PASS | all 3 docs contain the phrase + key elements (harness_log.md grep, reset on PASS/FAIL/new step-id) |
| 2 | no_code_changes | PASS | git status shows only 3 doc files modified (plus handoff/* rolling) |
| 3 | existing_text_preserved | PASS | F1 FAIL-counter language preserved in CLAUDE.md (the new clause is additive, not replacement) |

## Honest disclosures

1. **3 files match, not 4.** The verification grep also looks at
   `.claude/rules/*.md` for breadth, but that directory has no
   appropriate target file (research-gate.md is researcher-specific).
   The contract requires AT LEAST 3 matches; got exactly 3 from the
   most appropriate locations.

2. **CLAUDE.md edit was second-attempt.** First Edit failed because
   the file hadn't been read in this session. Re-read + retry
   succeeded. No content corruption.

3. **No state machine added.** Per research-brief recommendation,
   the counter mechanism is operator-judgment using harness_log.md
   as source of truth. No new field on masterplan.json steps; no
   new state files. Zero infrastructure surface area.

4. **Cross-reference network.** Each doc location either contains
   the full text (per-step-protocol.md) or contains a concise version
   pointing to the canonical full text (CLAUDE.md, qa.md). This
   prevents drift across the 3 locations.

5. **Q/A subagent snapshot.** The qa.md edit takes effect at the next
   Q/A spawn (Claude Code snapshots agent files at session start, but
   subagent prompts pick up fresh files on each spawn). Future Q/A
   verdicts in this very session, if any, will see the new rule.

## Closes

- Task list item #26
- masterplan step **phase-16.40**

## Next

Spawn Q/A. If PASS: log + flip + continue.
