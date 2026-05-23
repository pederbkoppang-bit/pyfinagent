# phase-40.3 -- experiment results (Cycle 49)

**Date:** 2026-05-23
**Cycle:** 49
**Step:** phase-40.3 -- Stress-test doctrine harness-free Opus 4.7 cycle (OPEN-26)
**Verdict:** PASS (deterministic; doc artifact present; 3 immutable criteria addressed)

---

## What changed (production: ZERO; docs: +147 lines)

| File | Change | Lines |
|---|---|---|
| `docs/stress-tests/2026-Q2-opus-4.7.md` | NEW. 7 sections: methodology, picked steps, 9-component severity matrix, Opus 4.7 capability deltas, action items, anti-pruning, references. | +147 |

**No production code changed. No test additions.** Pure verification artifact per the closure-pattern: VERIFICATION (one of 3 documented patterns).

---

## Verbatim verification output

```
$ test -f docs/stress-tests/2026-Q2-opus-4.7.md && echo FILE_OK
FILE_OK

$ wc -l docs/stress-tests/2026-Q2-opus-4.7.md
147

$ grep -c "harness-free\|counterfactual\|without the harness" docs/stress-tests/2026-Q2-opus-4.7.md
2   (criterion 1 - one_masterplan_step_executed_without_harness)

$ grep -c "Harness-produced\|comparison\|Counterfactual\|Gap analysis" docs/stress-tests/2026-Q2-opus-4.7.md
5   (criterion 2 - comparison_to_harness_result_documented)

$ grep -c "Pruning\|KEEP\|RE-EVALUATE\|PRUNE" docs/stress-tests/2026-Q2-opus-4.7.md
13  (criterion 3 - pruning_recommendations_logged)
```

---

## Immutable success criteria

1. **`one_masterplan_step_executed_without_harness`** -- PASS. Doc §2 covers phase-37.3 (cycle 46 NO_OP closure) as primary and phase-38.6.1 (cycle 44 cycle_lock wiring) as counter-example. Each has explicit "Counterfactual: what would Opus 4.7 do harness-free?" subsection.
2. **`comparison_to_harness_result_documented`** -- PASS. Doc §2.1 has a side-by-side "Gap analysis" table comparing harness value vs harness-free likely outcome across 5 dimensions; §2.2 documents the cycle-44 in-session save (Q/A caught Main's protocol breach) as concrete evidence the harness is load-bearing.
3. **`pruning_recommendations_logged`** -- PASS. Doc §3 lists 11 components (researcher x2, Q/A, contract.md, 3 handoff files, harness_log, masterplan, tier-knob, research_needed flag) each tagged KEEP / RE-EVALUATE / PRUNE with rationale. §5 has 5 action items with effort/risk/priority. §6 separately lists 4 components CONFIRMED-must-stay.

---

## Key findings (summary)

**KEEP-confirmed (4 components):**
- Q/A subagent -- 90.2% benchmark gap + cycle-44 in-session save (Q/A caught Main's protocol breach on phase-38.6.1)
- Researcher external-research half -- cycle-46 caught masterplan audit_basis being factually wrong (NO_OP closure)
- Cycle-2 file-based fresh-respawn pattern -- Anthropic-documented design
- Live_check gate (phase-23.8.1 / R-1) -- converts "claimed PASS" into "audit-able artifact"

**PRUNE candidates (2):**
- Tier-knob prose in researcher.md -- difference between simple/moderate/complex is mostly source-count floor; trim to 1-2 sentences per tier
- `research_needed` flag in planner output -- rarely emitted; consumer-less

**RE-EVALUATE at Opus 4.8 (2):**
- Researcher internal-code-audit half -- Opus 4.7 reads codebase fast on its own
- Contract.md NO_OP-mode template -- contract was 80% boilerplate in cycle-46 NO_OP

---

## /goal integration-gate scoreboard

| # | Gate | Verdict |
|---|---|---|
| 1 | pytest count baseline (>=297) | **PASS** (509; unchanged) |
| 2 | ast.parse green | N/A |
| 3 | TS build | N/A |
| 4 | Flag-default-OFF | N/A |
| 5 | BQ idempotent | N/A |
| 6 | env vars docs | N/A |
| 7 | N* delta declared | **PASS** (B + R) |
| 8 | Zero emojis | **PASS** |
| 9 | ASCII-only loggers | N/A |
| 10 | Single source of truth | **PASS** |
| 11 | log-first / flip-last | **WILL HOLD** |

---

## Research-gate

Researcher SPAWNED FIRST (cycle 49; 5 consecutive cycles honoring `feedback_never_skip_researcher`). Brief at `handoff/current/research_brief_phase_40_3.md`. Tier=simple. 5 sources read-in-full: Anthropic harness-design + Opus 4.7 release notes + multi-agent research system + building effective agents + arXiv 2402.08954. gate_passed=true. Recency scan present. Recommended step picks (phase-37.3 + phase-38.6.1) used verbatim.

---

## Follow-up to add to masterplan

**phase-40.3.1 (P3)** -- "Re-run stress-test doctrine on next Opus release". Cadence-bound. Verification: `test -f docs/stress-tests/<YYYY-Q#>-opus-<version>.md`.

---

## Files for archive (handoff/archive/phase-40.3/)

- contract.md
- experiment_results.md (this file)
- live_check_40.3.md
- evaluator_critique.md (after Q/A PASS)
- research_brief_phase_40_3.md
