# Goal — masterplan drain, 2026-08-08 (cycles continue at 181)

Work the masterplan via the full Layer-3 harness (researcher gate → contract → generate →
qa-verdict Workflow → log → flip), ONE step at a time. Never ask mid-run: a step needing an
operator decision gets a row in `handoff/current/operator_ask_2026-08-07.md` (19 rows) and is
SKIPPED.

STARTUP: `git checkout main && git pull origin main`. READ FIRST, all BINDING: this file,
CLAUDE.md, `.claude/rules/research-gate.md`, auto-memory MEMORY.md, the ask list.

## What 2026-08-07 closed (cycles 178-180)
62.1 PASS+flipped (bot was 9 days stale; three already-fixed defects were dark in it).
61.3 and 68.1 deferred-with-reason: code+tests+verdicts complete, status deliberately
`pending`. All verdicts verbatim in `handoff/current/evaluator_critique_*.md`.

## WAVE 1 — the money engine is DOWN. These lead.
- **85.4 (P0)** — no cycle has COMPLETED since 2026-07-31; five consecutive failures, no trade
  in 7 days. Measured: cc_rail median 88s, p90 129s, 17.2% timing out at the 150s cap. Its
  audit_basis carries a **correction** — per-cycle P1 paging and terminal rows ALREADY work;
  do not rebuild them. The gap is aggregate ("nothing has completed in N days"). Re-measure
  the analysis phase before touching the 7200s timeout.
- **85.5 (P0)** — a LIVE cycle's lock is judged stale at 90min (`is_stale` OR's age with
  `pid_alive`), and the stale path unlink+recreates the file, defeating the flock. Proven: two
  cycles can run at once. Fix must use TWO REAL processes and keep a dead holder recoverable.

## WAVE 2 — executor P0s
62.2 (unblocked — 62.1 done, `backend/slack_bot/` freeze LIFTED), 63.3, 63.4, 65.2, 68.5.

## WAVE 3 — defect tail
62.1.1 + 62.1.2 (P1 SECURITY, plaintext tokens). Do 62.1.1's malformed-token check FIRST — it
may be a root cause of 85.4. Then 85.3.1, 85.3.3, 61.3.1/.2/.3, 62.1.3, 36.28-36.30, 84.1.1,
83.x tail.

**Deferred, evidence complete, flip only on an operator answer:** 61.3 (#14/#15), 68.1 (#19 —
a 3rd CONDITIONAL auto-FAILs; cycle 3 only AFTER a disposition exists), 61.2 (#10), 72.0.2 (#13).

## Non-negotiable
- $0 metered. Nothing moves the live book. `historical_macro` FROZEN. Immutable gates
  byte-untouched. No flag promotions, no `backend/.env` writes.
- **Before restarting any service:** read `handoff/.autonomous_loop.lock` (pid, age, fresh log
  writes). `last_result` reports the last COMPLETED run and CANNOT prove idle — that reasoning
  error nearly killed a live trading cycle.
- `kickstart -k` does NOT re-read plist edits; `bootout` is hook-blocked. Use `launchctl setenv`
  and REVERT it, or you leave env permanently outranking .env.
- Sequence restarts clear of other steps' evidence windows (digests 14:00/23:00 CEST; Sat/Sun
  skip — a missed window can cost days).
- Contract BEFORE generate. If you breach a rule, disclose it AND name which automated check is
  blind to it. Never absolve your own process breach — defer it.
- Lint gate (ruff F821,F401,F811, git-derived scope, non-empty asserted) before every Q/A.
  `git add -An` before flips. Masterplan edits via python bypass the auto-commit hook — commit
  manually and verify `git log -1` in the same turn.
- Q/A: LEAN prompts; persist every verdict the turn it lands; errored/empty = NO VERDICT.
  Mutation-test every guard, and expect the evaluator to find the mutant you missed.
- Every out-of-scope defect gets its OWN research-gated step, never a prose mention.

## Done-definition
≥2 of {85.4, 85.5} PASS or deferred-with-recorded-reason; ≥3 of wave 2/3 closed; ask list
current; harness_log appended per cycle.

SOFT STOP at 12 cycles or when only operator-gated work remains → regenerate this goal with
measured state, commit+push, summarise.
HARD STOP: any live-book move, safety-gate loosening, metered spend, Fable repin, or 2
consecutive infrastructure failures.
