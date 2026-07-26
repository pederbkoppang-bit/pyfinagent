---
name: kill-switch-36-12-traps
description: Kill-switch 36.12 design traps measured 2026-07-26 -- pausing on disarmed wedges against the /resume 409; blanket measure-before-mutate breaks the SOD daily roll; no CWE covers a guard that mutates its own datum; execute_buy has no kill-switch gate at all
metadata:
  type: project
---

Measured at source 2026-07-26 while running the 36.12 research gate
(brief: `handoff/current/research_brief_36.12.md`).

**Three traps a fix for `check_and_enforce_kill_switch` will fall into:**

1. **PAUSE-ON-DISARMED IS A CIRCULAR WEDGE.** `paper_trading.py:593` 409s
   `/resume` while `armed` is False. If the trading path calls `state.pause()`
   on a disarmed switch, resume needs armed baselines and the anchor that
   produces them lives on the path that just refused. Use a NON-LATCHING
   per-cycle block instead. Same wedge family as 36.9 finding (3).
2. **BLANKET "measure before mutate" IS ACTIVELY DANGEROUS.** The SOD daily
   roll (`paper_trader.py:1089-1090`) is a *legitimate* pre-measurement
   mutation -- a daily-loss limit is measured from today's open. Evaluating
   before the roll computes a multi-day move as a same-day loss; 36.9 measured
   that as exactly 4.0% on the live book, i.e. a false `flatten_all`. Only the
   `None -> anchor` branch is the defect; the ratchet and the date-roll are not.
3. **BREACH BRANCH MUST STAY AHEAD OF THE DISARMED BRANCH.** With one baseline
   present and one missing, a real breach on the surviving leg must still
   flatten. `test_dod4_tier1_coverage_investment.py:968` is exactly that state
   and catches the wrong order.

**Externally:** CWE-367 TOCTOU explicitly does NOT cover a program that
modifies state it is about to check (MITRE's own note). There is no named
anti-pattern for a guard that mutates its own datum. The defensible framing is
CWE-424 (alternate path) + CWE-223 (omitted audit info) + Saltzer-Schroeder
fail-safe defaults. 15c3-5 never mandates blocking an unevaluable control.
ADVERSARIAL: NYSE Pillar v4.7 p.34 documents failing OPEN on a missing price
reference ("no NBO ... default to $0 ... i.e., no check") -- "everyone fails
closed" is false.

**Scope-adjacent, unfixed:** `execute_buy` has NO kill-switch gate;
`backend/agents/mcp_servers/signals_server.py:444` calls it directly, so the
MCP path bypasses the switch entirely. `is_paused()` is consulted in exactly
two places repo-wide (`paper_trader.py:1097`, `autonomous_loop.py:1287`).

**Why:** these were all found by reading, not by running -- an executor
following the step text literally would ship 1 or 2.
**How to apply:** cite these before any contract touching
`check_and_enforce_kill_switch`; re-verify line numbers first (36.8/36.9 both
intend to change the same lines).

Related: [[project_fabricated_safe_80_36]],
[[feedback_mutation_test_guards_and_fixtures]]
