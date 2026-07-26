---
name: kill-switch-36-9-armed-semantics
description: Step 36.9 kill-switch correctness research -- `armed` asserts diagnostic coverage never exercised; three definitions of "absent" for sod_nav; 5 test fixtures pin a PAST sod_date and expect ARMED; IEC 61511 frames disarming as a BYPASS needing a duration limit
metadata:
  type: project
---

Researched 2026-07-26 for masterplan step 36.9 (P0 kill-switch correctness).
Brief: `handoff/current/research_brief_36.9.md`. Companions:
[[kill-switch-archive-merge-36-8]], [[kill-switch-36-12-traps]].

**The single unifying frame.** All three 36.9 defects are ONE fault class:
`armed` is an assertion of diagnostic coverage that was never exercised. In
IEC 61508 terms that is a dangerous-UNDETECTED fault (`DC = λdd / λd`; a fault
the on-line diagnostic cannot see moves from the DD column to the DU column).
Framing the contract this way beats enumerating three unrelated bugs.

**Three definitions of "absent" for ONE field (the root cause of defect 3).**
Measured, not inferred:
- WRITER `kill_switch.py:525` -- `self._sod_nav = float(nav)`, NO guard, so
  `0.0` latches as a real baseline plus an audit row at `:527`.
- READER `kill_switch.py:745` -- `not (sod is not None and sod > 0)`.
- RE-ANCHOR `paper_trader.py:1142` -- `snap.get("sod_nav") is None or
  snap.get("sod_date") != today`. Tests `is None`, NOT `<= 0`, so a latched
  `0.0` with today's date never re-anchors -> the `/resume` 409 promise at
  `paper_trading.py:609-615` is FALSE for the re-anchor half (the BLOCK half
  IS true; 36.12's block at `paper_trader.py:1195` does fire).

**Non-obvious blast radius -- the biggest regression risk in this step.**
FIVE existing fixtures set `_sod_date` to a PAST date and then expect the
daily leg to EVALUATE: `test_phase_36_7_kill_switch_rotation_rearm.py:246`
(`2026-07-24`) and `test_phase_23_2_5_kill_switch_no_false_fires.py:130/:152/
:174/:248` (`2026-05-22`). Any date-aware disarm turns them RED. They are
genuine guards -- fix the fixtures (or inject a clock), never weaken the fix.

**Also non-obvious:** `tests/verify_phase_23_2_19.py:47-50` is a SOURCE-SCAN
verifier asserting the literal strings `'state.update_sod_nav(nav,
date=today)'` and `'snap.get("sod_date")'` exist in `paper_trader.py` -- so
editing the `:1142-1143` predicate text can break a test that has nothing to
do with the step. And `tests/services/test_sod_daily_roll.py:80/:100/:156`
RE-IMPLEMENT the `:1142` predicate INLINE, so they drift silently if it
changes. Export the predicate as a shared helper to kill both problems.

**Consumer enumeration (grep-derived, not sampled).** 6 backend consumers of
`evaluate_breach`; only ONE re-anchors first (`paper_trader.py:1154`, after
`update_peak:1133` + the SOD roll `:1140-1143`). The five that do NOT:
`paper_trader.py:1096` (deliberate 36.12 pre-measure), `paper_trading.py:517`
(UI badge), `paper_trading.py:580` (`/resume`),
`backend/agents/mcp_servers/risk_server.py:80` (MCP tool),
`kill_switch.py:859` (`check_auto_resume`). So `armed` is on THREE
operator-facing surfaces plus one control path.

**Do NOT encode a new state as an absent key.** Both frontends discriminate
with an explicit `=== false` (`KillSwitchPanel.tsx:137`,
`OpsStatusBar.tsx:318`) and both backend gates use `.get("armed", True)`
(`paper_trading.py:598`, `kill_switch.py:873`). An `armed: undefined` third
state renders ACTIVE and resumes freely. Add a named reason field beside the
boolean instead.

**`_log_disarmed_once` is reason-blind** (`kill_switch.py:792-810`): a
process-lifetime one-shot that prints only `sod_nav`/`peak_nav`. Add a second
disarm reason and the first one encountered is the only one ever logged, and
the message names neither.

**External verdict, incl. the adversarial check.** Fail-loud-and-conservative
is backed on all three: RuntimeAI 2026-05-12 ("If the policy plane is
unreachable, the answer is no. Not 'best-effort.' Not 'log and pass.'
Closed.") for the nav_invalid case; SEC market-wide circuit breakers whose
triggers are "calculated daily based on the prior day's closing price" for the
stale-anchor case; the sentinel-object/semipredicate pattern for the `0.0`
case. **The adversarial source does not oppose it, it CONSTRAINS it:**
IEC 61511 Cl. 16.2.4 treats operating with a safety function bypassed as
permitted only with compensating measures, Cl. 16.2.3 adds duration limits,
Cl. 16.2.6/16.2.7 authorization + indication + a bypass log, Cl. 11.7.3.2
requires manual shutdown stay enabled. pyfinagent already HAS the compensating
measure (36.12's per-cycle order BLOCK, and `BLOCK, NOT PAUSE` is what
preserves Cl. 11.7.3.2). **Defect 3 is the violation: a bypass with no exit.**
Keeping a stale anchor is the WORST quadrant -- it loses coverage AND biases
toward a spurious flatten (a 2-day move read as a 1-day loss), so it is a
nuisance trip and a coverage loss simultaneously.

**Unverified, do not repeat as fact:** the "4.0% measured on this book" figure
lives in a CODE COMMENT at `paper_trader.py:1150-1152` attributing the
measurement to step 36.9 itself. I did not reproduce it. Re-measure before
citing it as evidence.
