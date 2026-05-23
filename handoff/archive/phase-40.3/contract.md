# phase-40.3 -- Stress-test doctrine harness-free Opus 4.7 cycle (OPEN-26)

**Step id:** `40.3`
**Date:** 2026-05-23
**Mode:** EXECUTION (cycle 49).
**Cycle:** Cycle 49 (after Cycle 48 production_ready_audit).

---

## North-star delta

**Terms:** B (zero $; documentation only) + R (process-integrity; informs future harness-pruning decisions).

**B:** Zero $ cost. Pure docs work; no production code, no LLM API calls beyond the researcher already spawned.

**R:** Closes OPEN-26 (no harness-free cycle documented for Opus 4.7 since release 2026-04-16). Output informs future harness scaffolding decisions: prune stale components, KEEP load-bearing ones with evidence.

**P:** N/A. **Caltech arxiv:2502.15800 discount:** N/A.

**How measured:** `test -f docs/stress-tests/2026-Q2-opus-4.7.md` exits 0. The doc carries the 3 immutable criteria addressed verbatim.

---

## Research-gate compliance

**Researcher SPAWNED FIRST** -- brief at `handoff/current/research_brief_phase_40_3.md`. Tier=simple. 5 sources read in full, gate_passed=true, recency scan present.

Sources cited:
- anthropic.com/engineering/harness-design-long-running-apps (canonical doctrine)
- platform.claude.com/docs/en/about-claude/models/whats-new-claude-4-7 (Opus 4.7 release notes; April 2026)
- anthropic.com/engineering/built-multi-agent-research-system ("90.2% benchmark gap" backing Q/A KEEP)
- anthropic.com/engineering/building-effective-agents (subagent design)
- arXiv 2402.08954 (arXiv HTML rendering reference for source-fetching strategy)

Recommended step picks:
- **Primary**: phase-37.3 NO_OP closure (cycle 46) -- simplest case
- **Secondary**: phase-38.6.1 cycle_lock wiring (cycle 44) -- in-session counter-example where Q/A cycle-2 round-1 CONDITIONAL caught Main's protocol breach (researcher skipped + stale experiment_results)

---

## Hypothesis

> The current 3-agent harness MAS (Main + Researcher + Q/A) was designed against pre-Opus-4.7 assumptions. Some components are now dead-weight; some are MORE load-bearing than before. Document which is which with severity-tagged pruning recommendations. The cycle 44 in-session save (Q/A caught Main's protocol breach on phase-38.6.1) is empirical evidence FOR keeping Q/A; the researcher-skip-recovery pattern is empirical evidence FOR keeping researcher's external-research half.

---

## Immutable success criteria (verbatim from masterplan 40.3.verification)

1. `one_masterplan_step_executed_without_harness` -- doc carries counterfactual analysis of phase-37.3 (and phase-38.6.1 as counter-example).
2. `comparison_to_harness_result_documented` -- doc compares actual harness-produced result to hypothetical harness-free outcome side-by-side.
3. `pruning_recommendations_logged` -- doc has explicit severity-tagged pruning recommendations (KEEP / RE-EVALUATE / PRUNE) for 9 harness components.

Plus /goal integration gates 1-11.

---

## Files this step touches

- `docs/stress-tests/2026-Q2-opus-4.7.md` (NEW, ~250 lines): the verification artifact.

**NOT changed:** any production code, any agent .md, any masterplan immutable criterion.

---

## /goal integration gates (declared)

| # | Gate | Plan |
|---|---|---|
| 1 | pytest count >= 297 | unchanged (no test changes) |
| 2 | ast.parse green | N/A (no .py changes) |
| 3 | TS build green | N/A |
| 4 | flag-default-OFF | N/A |
| 5 | BQ idempotent | N/A |
| 6 | env vars docs | N/A |
| 7 | N* delta declared | DONE (B + R; documentation only) |
| 8 | zero emojis | will hold |
| 9 | ASCII-only loggers | N/A (docs file) |
| 10 | single source of truth | doc is canonical for the Q2-2026 Opus 4.7 stress test |
| 11 | log-first / flip-last | will hold |

---

## Honest scope

**This is a DOCUMENTATION cycle**, not engineered work. The "stress test" is a counterfactual / thought-experiment based on the cycle-12-47 record in `handoff/harness_log.md`. The honest framing is in the doc itself: "we cannot literally re-run a past step in a parallel session, but we can reconstruct from the harness_log what the model did with the harness and reason carefully about what it would have done without."

**KEEP recommendations:** Q/A subagent (90.2% benchmark gap + in-session cycle-44 save); Researcher external-research half (cycle-46 caught audit_basis being factually wrong); handoff files (resume detection + audit trail).

**RE-EVALUATE recommendations:** contract.md (load-bearing in some cycles, dead-weight in NO_OP closures); Researcher internal-code-audit half (Opus 4.7 reads codebase fast).

**PRUNE candidates:** tier-knob prose elaboration in researcher.md (simple/moderate/complex tier difference is now subtle); `research_needed` flag in planner_agent (rarely used).

---

## References

- closure_roadmap.md §3 OPEN-26
- handoff/current/research_brief_phase_40_3.md (5 sources read-in-full)
- handoff/harness_log.md (cycles 12-47 inventory)
- CLAUDE.md::Stress-test doctrine (Anthropic)
- /goal directive
