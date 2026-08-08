# Goal — masterplan drain, NEXT session (cycles continue at 183)

Regenerated 2026-08-08 with MEASURED state at the end of cycles 181–182.
Work the masterplan via the full Layer-3 harness (researcher gate → contract →
generate → qa-verdict → log → flip), ONE step at a time. A step needing an
operator decision gets a row in `handoff/current/operator_ask_2026-08-07.md`
and is SKIPPED — never ask mid-run.

STARTUP: `git checkout main && git pull origin main`. READ FIRST, all BINDING:
this file, CLAUDE.md, `.claude/rules/research-gate.md`, auto-memory MEMORY.md,
the ask list.

## What 2026-08-08 closed (cycles 181–182)

- **85.5 PASS + pushed** (`4c17f06a`). Cycle-lock split-brain closed: liveness is
  the sole staleness authority, stale-reacquire branch deleted, verify-after-lock
  added, release never unlinks, TTL derived from the live cycle budget. 6/6
  mutants. Cost **four Q/A evaluations for three verdicts** — both rails dropped.
- **85.4 research gate DONE + contract WRITTEN. GENERATE NOT STARTED.**
  `contract_85.4.md` is ready; the next session starts at GENERATE.
- Queued: **85.5.1** (kill-switch daily-anchor disarm makes a book-safety test red).
- Asks filed: **#20** malformed OAuth token (same token in two plists; explicitly
  DE-ESCALATED as an 85.4 root cause), **#21** my test runs overwrote live
  kill-switch pause provenance.
- Ask **#18 corrected**: it wrongly said no operator action was needed.

## START HERE — 85.4 GENERATE

The contract is written and the gate refuted TWO of the step's premises. Build
against `contract_85.4.md`, not the audit basis:
- Terminal rows ARE written (`autonomous_loop.py:1746`); the defect is status
  **fidelity** — the `:1327` kill-switch halt leaks the `:362` placeholder
  `"running"` into the terminal row.
- The failure was NOT invisible; the P1 fired and was buried under ~24 hourly
  freshness P1s.
- Root cause is **(a) legitimate slowness**: ~7,500–8,100s needed vs a 7,200s
  budget; successes truncate at 145s against a 150s cap, wasting ~26% of rail time.
- The real C4 gap is `cycle_health.py:193` — the heartbeat alarm skips only
  `started` rows, so a `timeout` row resets age and it can never see
  "nothing COMPLETED in 8 days".
- C5 binds: config changes are behavioural → dark/operator-gated, never inline.

## THEN — two P0s that 85.4 cannot fix

1. **File and work the kill-switch-latched-paused step.** Paused since
   2026-08-04T11:43:31Z. Even a perfect 85.4 ships **zero trades** until resumed.
   Not yet queued — file it with a research gate.
2. **Widen 36.28** (or file a sibling): tests WRITE to the live kill-switch audit
   journal, not just read it (ask #21).

## THEN — wave 2/3 (unchanged)

62.2, 63.3, 63.4, 65.2, 68.5. Then 85.3.1, 85.3.3, 61.3.1/.2/.3, 62.1.3,
36.28–36.30, 84.1.1, 83.x tail. `62.1.1` is now **ask-gated** (#20) — its
diagnostic is done; the fix needs the correct credential value.

Deferred, evidence complete, flip only on an operator answer:
61.3 (#14/#15), 68.1 (#19), 61.2 (#10), 72.0.2 (#13).

## Non-negotiable

- $0 metered. Nothing moves the live book. No flag promotions, no `backend/.env`
  writes. `historical_macro` untouched.
- **A backend restart is OWED** (ask #18 correction): the running backend
  predates the 85.5 fix and holds the pre-fix module, so that P0 is committed but
  NOT in force. Read `handoff/.autonomous_loop.lock` — never `last_result` —
  before restarting.
- **`pytest backend/tests` mutates live kill-switch audit state on this machine**
  (ask #21). Expect it; disclose it if it happens again.
- **zsh does not word-split unquoted `$VAR`.** This cost two silent non-runs
  today that printed success. Use arrays; make every harness assert its own
  precondition held before it reports a number.
- **Do not restore with `git checkout --`** while holding uncommitted work — it
  silently reverted my edits mid-run; the PreToolUse danger hook blocks it for
  this reason.
- Q/A: **LEAN prompts.** Four evaluations produced three verdicts today; the two
  drops both had heavy prompts, the clean PASS took 8 tool calls. On an empty
  return, SHORTEN and re-spawn on the same rail before switching rails. On an
  Agent-tool drop, `SendMessage` the idle agent for its verdict before re-running
  — that recovered a full CONDITIONAL today.
- Contract BEFORE generate. Disclose any breach AND name which automated check is
  blind to it.

## Done-definition

85.4 PASS or deferred-with-recorded-reason; the kill-switch-paused step filed;
≥3 of wave 2/3 closed; ask list current; harness_log appended per cycle.

SOFT STOP at 12 cycles or when only operator-gated work remains → regenerate
this goal with measured state, commit+push, summarise.
HARD STOP: any live-book move, safety-gate loosening, metered spend, or 3
consecutive infrastructure failures.
