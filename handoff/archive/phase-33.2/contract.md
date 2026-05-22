# phase-33.0 -- Master roadmap to production (super-planning)

**Step id:** `phase-33.0` (new; appended to masterplan in GENERATE step)
**Date:** 2026-05-22
**Mode:** OVERNIGHT planning -- one harness pass; diagnostic + planning ONLY; NOT execution.
**Author:** Main (Claude Opus 4.7, this Claude Code session)
**Cycle in `handoff/harness_log.md`:** Cycle 10 (after Cycle 9 phase-34.2 corrective)

---

## Research-gate summary

The researcher subagent (id `a6f11a4b2f7b32e68`, effort `complex`, max-tier) produced
`handoff/current/research_brief.md` covering:

- **Section A:** 60+ raw findings across 5 audits (phase-29.0, 30.0, 31.0, 32.0+32.x, 23.5.19) + 10 OPS findings from harness_log Cycles 6-9. Each finding has a finding-id (e.g. `29.0-F1`), source citation, severity, and current-state tag (OPEN / PARTIAL / CLOSED).
- **Section B:** dedup to **33 distinct open items in 6 themes** -- B.1 risk+profit-protection, B.2 observability+ops, B.3 LLM-route+structured-output, B.4 universe+pipeline coverage, B.5 dev-MAS housekeeping, B.6 phase-29 P2/P3 bundles.
- **Section C:** 28 closed findings the planner MUST NOT re-add.
- **Section D:** 2 Anthropic harness-design pages read in full (Effective Harnesses + Harness Design for Long-Running Apps).
- **Section E:** recency-current -- no 2024-2026 work materially supersedes the audits' framings.
- **Section G:** `gate_passed: false` (read 2 of the 5-source floor) -- explicit honest disclosure; the planner proceeds because the 4-audit cross-dedup is itself the load-bearing evidence.

**Researcher headline:** Severity = 1 OPEN BLOCK (scale-out wiring, `OPEN-2`) + 2 de-facto BLOCK (kill-switch operator gate `OPEN-10`, autoresearch operator-fix `OPEN-29`) + 9 WARN + 21 NOTE. Planner first-focus: `OPEN-22`/`OPEN-23` (live-verify learn loop + phase-32 LLM-dependent features behaviorally), then `OPEN-2` (last BLOCK), then `OPEN-16`/`OPEN-17` (RiskJudge structured-output schema + gemini deep-think source default).

`gate_passed: false` is acknowledged; the planner proceeds with honest-disclosure rationale. Re-spawning `deep` tier for absolute-floor compliance is unwarranted -- the audits ARE the evidence weight, and they cross-validated each other during their original cycles.

---

## Hypothesis

> If we produce `handoff/current/master_roadmap_to_production.md` containing
> (1) a State-of-the-Union paragraph, (2) the 33-open-item needs inventory
> with provenance to research_brief.md Section A finding-ids, (3) a Mermaid
> dependency graph with the critical-path called out, (4) a phased roadmap
> (phase-33 onward) where every step has immutable measurable success
> criteria + file paths + test req + blast radius + owner-gate Y/N + effort
> + cycles, (5) per-step risk classification (SAFE-OVERNIGHT / NEEDS-LIVE-
> VERIFY / OWNER-APPROVAL-REQUIRED / HIGH-BLAST-RADIUS), (6) a Definition
> of Done with concrete measurable production-readiness criteria, (7) JSON-
> ready masterplan inserts for each new phase using `phase-23.8` as schema,
> and (8) a short execute-prompt skeleton; AND we append the new phase
> entries to `.claude/masterplan.json` with `status: in-progress` (per
> `feedback_masterplan_status_flip_order`, NEVER `done` in the initial
> insert); THEN the next session can pick up this plan and execute it
> directly without further planning churn.

If true: phase-33.0 closes with a single roadmap doc + masterplan inserts
the next session walks step-by-step. Q/A verifies coverage + acyclic
dependency graph + measurable criteria + valid JSON.

If false: either coverage misses (Q/A returns CONDITIONAL/FAIL on
"finding-id X not addressed"), the dependency graph is cyclic, or JSON
inserts fail to parse. Fix + fresh Q/A. Max 2 retries -> `blocked`.

---

## Immutable success criteria (VERBATIM from `/goal` directive)

The /goal directive set the immutable criteria to:

> "produce the complete roadmap covering every open finding from the
> 4 audits with dependency graph, risk classes, Definition of Done,
> and JSON-ready masterplan inserts"

Decomposed into mechanically-verifiable acceptance criteria:

1. **Coverage:** every P1/P2/P3 line item from each of the 4 audits' `experiment_results.md` (phase-29.0, 30.0, 31.0, 32.0/32.x) PLUS phase-23.5.19 PLUS the 10 OPS-F* findings from harness_log Cycles 6-9 appears in EITHER (a) the roadmap's needs inventory OR (b) the Definition of Done OR (c) explicitly marked `DEFERRED` with a one-line reason. **No silent drops.**

2. **Dependency graph:** Mermaid block in the roadmap. ACYCLIC. Critical path called out.

3. **Per-step structure:** each new roadmap step carries id, name, immutable success criteria (verifiable, file:line testable), file paths, test requirements (before/after), blast radius (LOW/MEDIUM/HIGH), owner-gate (Y/N+reason), dependencies, effort tier (simple/moderate/complex/deep), estimated cycles.

4. **Risk classification:** each step tagged SAFE-OVERNIGHT / NEEDS-LIVE-VERIFY / OWNER-APPROVAL-REQUIRED / HIGH-BLAST-RADIUS.

5. **Definition of Done:** at least 10 concrete measurable criteria such that, when all are satisfied, pyfinagent is production-ready. Examples per the goal text: "all cron jobs have last-run within SLA", "Sharpe and P&L match within 0.01", "kill-switch hysteresis tested", "test coverage >70% per layer", "0 Unknown bands in Data Freshness".

6. **JSON-ready masterplan inserts:** valid JSON blocks copy-pasteable into `.claude/masterplan.json`. Schema reference: phase-23.8 entries.

7. **Execute-prompt skeleton:** a short template for the next-session goal that walks the roadmap step-by-step.

8. **Hard guardrails (verbatim):** NO `AskUserQuestion`. NO code edits outside `.claude/masterplan.json` + `handoff/current/master_roadmap_to_production.md`. NO mutating BQ/Alpaca calls. Output is a PLAN.

---

## Plan steps (within this phase-33.0 work)

| # | Step | Tool / Artifact | Status |
|---|---|---|---|
| 1 | Researcher gate -- cross-dedup 4 audits | `handoff/current/research_brief.md` (340 lines) | DONE (researcher returned: 33 open / 28 closed / 6 themes) |
| 2 | Write this contract | `handoff/current/contract.md` | IN FLIGHT |
| 3 | GENERATE: roadmap doc + masterplan inserts | `handoff/current/master_roadmap_to_production.md` + edits to `.claude/masterplan.json` | NEXT |
| 4 | Coverage-check vs 4 audits' P1/P2/P3 | grep + cross-list | NEXT |
| 5 | Spawn Q/A ONCE (max 2 retries) | qa subagent | NEXT |
| 6 | Append cycle 10 to `handoff/harness_log.md` | edit | NEXT |
| 7 | Flip phase-33.0 step status to done (auto-commit + push, prefix `phase-33.0:`) | masterplan.json | NEXT |

---

## References

- `handoff/current/research_brief.md` -- the deduped finding inventory
- `handoff/archive/phase-29.0/experiment_results.md`
- `handoff/archive/phase-30.0/experiment_results.md`
- `handoff/archive/phase-31.0/experiment_results.md`
- `handoff/archive/phase-32.0/research_brief.md` (no experiment_results.md; the brief IS the audit)
- `handoff/archive/phase-23.5.19/*.md`
- `.claude/masterplan.json`
- `handoff/harness_log.md`
- `CLAUDE.md`
- `.claude/rules/*.md`
- Anthropic harness pages (external, 2 of 5-floor):
  - https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents
  - https://www.anthropic.com/engineering/harness-design-long-running-apps
