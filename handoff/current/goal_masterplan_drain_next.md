# Goal — masterplan drain, next session (cycles continue at 183)

Regenerated 2026-08-08 after cycles 181-182 with MEASURED state.
Full Layer-3 harness (researcher gate -> contract -> generate -> qa-verdict ->
log -> flip), ONE step at a time. Operator-decision steps get an ask row and are
SKIPPED. Never ask mid-run.

STARTUP: `git checkout main && git pull origin main`. READ FIRST, all BINDING:
this file, CLAUDE.md, `.claude/rules/research-gate.md`, MEMORY.md, the ask list,
`handoff/current/contract_85.4.md`.

## State (measured — do not re-assume)

- **85.5 PASS + pushed.** Backend **RESTARTED** (pid 143, 18:54:45), so the P0 is
  now genuinely IN FORCE — proven by the startup log emitting `state=released`,
  a field only the new `cycle_lock` writes. Slack bot restarted (pid 708).
- **85.4**: research gate DONE, contract WRITTEN, **GENERATE NOT STARTED**.
- **85.6 NEW P0 filed**: the book cannot be un-paused (deadlock, below).
- 85.5.1 queued. Asks #20/#21/#22 filed; #18 corrected.

## START HERE — 85.4 GENERATE, then 85.6

Build against `contract_85.4.md`, NOT the audit basis — the gate REFUTED two
premises. Terminal rows ARE written (`autonomous_loop.py:1746`); the defect is
status FIDELITY (the `:1327` kill-switch halt leaks the `:362` placeholder
`"running"`). The failure was NOT invisible; the P1 fired but was buried under
~24 hourly freshness P1s. Root cause is **(a) legitimate slowness**:
~7,500-8,100s needed vs a 7,200s budget, successes truncating at 145s against a
150s cap (~26% of rail time wasted). Real C4 gap: `cycle_health.py:193` skips
only `started` rows, so a timeout row resets age and it can never see "nothing
COMPLETED in N days". C5 binds: config changes are behavioural -> dark /
operator-gated, never inline. Do NOT lower `paper_analyze_top_n`. Do NOT migrate
gathers to `TaskGroup`.

**85.6 is the reason none of this ships trades yet.** The kill switch is paused
AND disarmed; resume 409s on a stale daily anchor; the anchor rolls only at
`paper_trader.py:1298` (mark/trade region), which timing-out cycles never reach.
The refusal message claims the roll is "at the top of the next cycle" and that it
"clears itself" — both false. Fixing 85.4 may clear it as a side effect; 85.6
must **TEST** that, not assume it. Do NOT hand-write a `sod_snapshot` audit row
to force the anchor — that masks the defect (ask #21 anti-pattern).

## THEN

Widen 36.28 (or file a sibling): tests WRITE to the live kill-switch audit
journal, not just read it (ask #21). Then wave 2/3: 62.2, 63.3, 63.4, 65.2,
68.5; then 85.3.1, 85.3.3, 61.3.1/.2/.3, 62.1.3, 36.28-36.30, 84.1.1, 83.x.
62.1.1 is ASK-GATED (#20) — diagnostic done, fix needs the correct credential.

Deferred, flip only on an operator answer: 61.3 (#14/#15), 68.1 (#19),
61.2 (#10), 72.0.2 (#13).

## Non-negotiable

- $0 metered. Nothing moves the live book. No flag promotions, no `backend/.env`
  writes. `historical_macro` untouched.
- The backend restart is **DONE** — do not repeat it reflexively. Before ANY
  restart read `handoff/.autonomous_loop.lock`, never `last_result`.
- `pytest backend/tests` **MUTATES live kill-switch audit state** on this machine
  (ask #21). Expect it; disclose it if it happens again. Live `paused_at` is
  unreliable — the real operator pause is 2026-08-04T11:43:31Z.
- **zsh does NOT word-split unquoted `$VAR`** — this caused two silent non-runs
  that printed success. Use arrays; make every harness assert its own
  precondition held before reporting a number.
- **Never restore a single tracked file with a file-level git checkout while
  holding uncommitted work** — it silently reverted edits mid-run; the PreToolUse
  danger hook blocks it for exactly this reason. Back up and restore explicitly.
- Q/A: **LEAN prompts.** On an empty return, SHORTEN and re-spawn on the SAME
  rail before switching. On an Agent-tool drop, `SendMessage` the idle agent for
  its verdict before re-running — that recovered a full verdict.
- Contract BEFORE generate. Disclose any breach AND name which automated check is
  blind to it.

## Done-definition

85.4 PASS or deferred-with-recorded-reason; 85.6 PASS or deferred; >=3 of
wave 2/3 closed; ask list current; harness_log appended per cycle.

SOFT STOP at 12 cycles or when only operator-gated work remains -> regenerate
with measured state, commit+push, summarise.
HARD STOP: any live-book move, safety-gate loosening, metered spend, or 3
consecutive infrastructure failures.
