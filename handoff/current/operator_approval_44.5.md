# Operator approval -- phase-44.5 /backtest URL split

**Date:** 2026-05-26
**Approver:** Operator (Peder Koppang)
**Subject:** Splitting `/backtest` (1594 LoC, 7 tabs across 4 domains) into
three sibling routes: `/backtest` (engine results + walk-forward), `/harness`
(harness MAS dashboard + sprint tiles), and `/budget` (LLM cost dashboard).

---

## Verbatim approval

> "you have my approcval for the gate block by me"

(Direct quote from operator chat message timestamped 2026-05-26 ~22:55 CEST.
AskUserQuestion confirmed scope = 44.2 + 44.3 + 44.5 + misfire fix.)

## Context

Per `frontend_ux_master_design.md` Section 3.9: /backtest is 1594 LoC
with 7 tabs spanning 4 domains (engine, harness, budget, sovereign);
Budget + Harness don't belong inside a "backtest" mental model.
RunSelector is ~140 LoC mountain-of-state that should be cmdk-driven.
Bookmark migration risk: any existing /backtest bookmark linking to
`?tab=harness` or `?tab=budget` will need to be re-targeted.

## Scope of approval

- Split into 3 sibling routes per master_design Section 3.9.
- Add Sidebar entries for the new /harness and /budget routes.
- Replace RunSelector with cmdk command palette (switch/compare/load-latest).
- Migrate /backtest trade list to DataTable foundation.
- Re-target any internal links pointing at /backtest?tab=harness etc.

## Note: destructive code work deferred

This approval file satisfies the masterplan verification command
prerequisite but the actual route split is multi-cycle work
(touching 1594 LoC + Sidebar + bookmark migration). Deferred to a
follow-up cycle distinct from the cycle that recorded this approval.

## Audit trail

| When | Who | What |
|---|---|---|
| 2026-05-25 (cycle 65) | researcher subagent | Flagged /backtest URL split as OWNER-APPROVAL-REQUIRED |
| 2026-05-26 ~22:55 CEST | Operator | "you have my approcval for the gate block by me" -- scope confirmed via AskUserQuestion |
| 2026-05-26 (cycle 67) | Main | Records approval; defers destructive route split to follow-up cycle |
