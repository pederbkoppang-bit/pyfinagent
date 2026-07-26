Drain the pyfinagent masterplan. Full harness cycle per step: researcher -> contract -> GENERATE ->
ONE fresh Q/A -> harness_log append -> status flip. No self-eval. Read CLAUDE.md +
.claude/rules/research-gate.md first.

STATE (measured 2026-07-26 evening, HEAD ffefe814, pushed):
- 80.40 CLOSED (Q/A PASS cycle 5, harness_log Cycle 172). 36.7 and 36.12 still `pending`.
- Backend NOT restarted since this morning: launchd pid 76381, armed:true, sod_nav 23838.19,
  sod_date 2026-07-24, peak_nav 24666.57, trailing_dd 3.36%. 36.7+80.40 code IS live. 36.12's code
  is COMMITTED (cfb56572) but NOT live -- deliberately: its Q/A returned FAIL. Do not restart to
  pick it up until it passes. `baseline_provenance` absent from GET /kill-switch proves it is not
  loaded.
- kill_switch_audit.jsonl md5 ce8fb93348bb9a3bbe26f2d91b1bc05e (8 lines). VERIFY THIS FIRST.

START HERE, in order:
1. 36.7 -- owes CYCLE 6. Cycle 5 met all seven criteria by the evaluator's own execution but is
   recorded INADMISSIBLE: its mutation harness appended 54 peak_reset rows to the live audit file
   (module-level `_state = KillSwitchState()` builds at import, before any fixture redirect). I
   restored + verified. Per per-step-protocol.md an unauthored tree change is never partially
   trusted. Spawn a fresh Q/A on a quiesced tree. Its one non-self-inflicted finding: it could not
   take its own :3100 UI capture because none was running -- STARTING THAT RIG IS MAIN'S JOB, do it
   before spawning (see item 3).
2. 36.12 -- FAILED cycle 3 with TWO named blockers, both open:
   (a) QA-Z1: deleting `return summary` from run_daily_cycle's halt block survives every suite; a
       halted cycle then falls into Step 5.6 and decide/execute and TRADES. This is the THIRD
       relocation of one hole (inline scan -> neutered predicate -> branch body). Do NOT extend the
       AST guard a fourth time. Drive run_daily_cycle with check_and_enforce_kill_switch stubbed to
       {"triggered":False,"blocked":True} and assert summary["halted"] and that decide/execute never
       ran. The existing loop tests re-implement the sequence instead of importing the module -- that
       is the pattern to break, not copy.
   (b) qa.md 1c live capture for the two changed KillSwitchPanel tooltips. Both strings sit inside
       `disarmed ?` branches and CANNOT render on the live armed book, so it needs a stubbed rig.
3. THE :3100 RIG IS THE UNLOCK FOR BOTH. My scratchpad rig only stubbed the paper-trading router, so
   the page rendered "Cannot reach backend". Build one that serves the whole cockpit surface, then
   leave it running for the evaluator. USE `PLAYWRIGHT_DIST_DIR` (next.config.js:9) -- I used a
   made-up `NEXT_DIST_DIR`, my :3100 shared the operator's `.next`, and :3000/login went 404 until I
   killed it. Always probe /login, never just / (which stayed 302 throughout).
4. Then 36.13 (P0, execute_buy has NO kill-switch gate -- the MCP signals path bypasses it entirely),
   36.15 (P1, peak_reset replay lacks the None-guard its sibling has), 36.8/36.9, 36.14, then the
   phase-80 tail.

DO NO HARM (hard):
- Paper only. No .env edits, no flag flips. historical_macro FROZEN, no optimizer runs. Kill-switch
  limits, stops, sector caps, DSR>=0.95, PBO<=0.5 byte-untouched. NO peak reset (owed token, 79.6).
- Never drive :3000. Never start a 2nd next dev without PLAYWRIGHT_DIST_DIR.
- kill_switch_audit.jsonl is tracked LIVE safety state. Any harness that imports kill_switch can
  write it via the import-time singleton -- redirect _AUDIT_PATH BEFORE exec, and md5-check after.
- `git add -An` before EVERY flip. Commit each step's work separately BEFORE flipping, so the hook's
  `git add -A` stages only masterplan + harness_log.
- TESTS PAGE YOU: alerting.py:167 posts to real Slack with no test guard and the deduper is
  per-process. 17 false P1s reached #ford-approvals today, plus ~5 per full kill-switch test run
  from the pre-existing pause() path. Patch raise_cron_alert_sync in any new harness. Step 36.14.

LESSONS THAT COST REAL CYCLES:
- Derive a gate's scope from git diff / ls, never type it. A hand-typed sweep is what let 36.7's
  count survive its own "class fix" into a fifth failure.
- Re-measure counts every cycle; a growing suite silently invalidates them.
- A guard on the SHAPE of code is not a guard on its BEHAVIOUR. Three cycles proved it.
- Verify your own "I did it" claims in the same turn you write them.

OWED CLEANUP: `rm -rf frontend/.next-audit-36-12` (gitignored; the permission layer refused me).

END OF SESSION, always: write the next goal prompt in chat (under 4000 chars) so I can paste it.
