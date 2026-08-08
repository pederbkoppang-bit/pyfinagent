# Goal — OVERNIGHT unattended drain, 2026-08-08 → 08-09 (cycles continue at 183)

Run through the night. Fix the defects enumerated below, in priority order,
each through the full Layer-3 harness. Do not stop for the night at the first
PASS — keep going until a stop condition fires.

STARTUP: `git checkout main && git pull origin main`. Confirm
`.claude/settings.json` has `defaultMode: bypassPermissions` — an unattended run
blocks forever without it. READ FIRST, all BINDING: this file, CLAUDE.md,
`.claude/rules/research-gate.md`, auto-memory MEMORY.md, the operator ask list,
`handoff/current/contract_85.4.md`.

**Only ONE session flips masterplan steps.** Check `ListAgents` at startup; if a
peer is active, coordinate or work read-only. `git add -A` in the auto-commit
hook will otherwise sweep a peer's in-flight files under your step's name.

---

## The defect list — "fix all errors found" means exactly these

Work them in this order. Each gets the full loop: researcher gate → contract →
generate → qa-verdict → harness_log → flip. One step at a time.

### 1. 85.4 (P0) — the engine never completes. START HERE.
Research gate PASSED, contract WRITTEN (`contract_85.4.md`). **GENERATE is the
next action — do not re-run the gate or rewrite the contract.**
Build against the contract, NOT the audit basis: the gate **refuted two of the
step's own premises**. Terminal rows ARE written (`autonomous_loop.py:1746`);
the defect is status FIDELITY — the `:1327` kill-switch halt leaks the `:362`
placeholder `"running"`. The failure was NOT invisible; the P1 fired but was
buried under ~24 hourly freshness P1s. Root cause is **(a) legitimate
slowness**: ~7,500–8,100s needed against a 7,200s budget, with successes
truncating at 145s against a 150s cap (~26% of rail time wasted). The real C4
gap is `cycle_health.py:193` — it skips only `started` rows, so a `timeout` row
resets age and it can never observe "nothing COMPLETED in N days".
Constraints: C5 binds — config changes are behavioural, so dark/operator-gated,
never inline. Do NOT lower `paper_analyze_top_n` (it narrows the trade funnel).
Do NOT migrate the gathers to `TaskGroup` (it cancels siblings on first failure).

### 2. 85.6 (P0) — the book cannot be un-paused. THIS IS WHY NOTHING TRADES.
The deadlock: cycle dies in `analyzing` → daily anchor never rolls (the roll is
`update_sod_nav` at `paper_trader.py:1298`, inside `check_and_enforce_kill_switch()`,
which runs in the mark/trade region `autonomous_loop.py:1271+`) → anchor stale →
switch disarmed → `POST /resume` returns 409 → switch stays paused → even a
completing cycle logs `kill-switch active (paused) -- skipping decide/execute`.
The 409 message claims the roll is "at the top of the next cycle" and that the
refusal "clears itself" — **both false**, and a defect in their own right.
Fixing 85.4 *may* clear this as a side effect. **TEST that; do not assume it.**
**Do NOT hand-write a `sod_snapshot` row into `handoff/kill_switch_audit.jsonl`
to force the anchor.** That masks the defect and is the ask-#21 anti-pattern.
The unblock must come from code.

### 3. 85.5.1 (P1 BOOK SAFETY) — a real-breach test is RED.
`test_book_safety_69.py::test_valid_nav_still_breaches` fails at HEAD: the kill
switch DISARMS on a stale/absent daily anchor instead of firing on a real 20%
breach. The surface reading is an incomplete mock (exactly ONE snapshot mock in
the tree, `test_book_safety_69.py:80`, omitting `sod_date` that the real
`_snapshot_locked()` returns at `kill_switch.py:444`). **Do not stop there** —
the question that sets severity is whether `sod_date` can be `None` in
**production** (at startup before the first anchor, or across a date rollover),
because then a real drawdown does not fire the switch. Answer with a measurement.

### 4. Widen 36.28 (P1) — tests WRITE to the live kill-switch audit journal.
Measured: `pytest backend/tests` wrote **12** `{"event":"pause","trigger":"manual"}`
rows to the real `handoff/kill_switch_audit.jsonl` on 2026-08-08 between
07:26:28Z and 08:35:16Z, and the live `paused_at` now reads a test's timestamp
instead of the operator's real 2026-08-04T11:43:31Z pause. All 12 were pauses,
zero resumes — fail-safe direction, but the audit trail is corrupted. 36.28
currently covers tests *reading* live state; widen it (or file a sibling) to
cover *writes*. Fix by injecting state in the fixture or pointing `_AUDIT_PATH`
at tmp during tests. **Expect your own test runs to keep doing this until it is
fixed — disclose it if it happens again.**

### 5. Triage the 26 full-suite failures — QUEUE, do not blind-fix.
`python -m pytest backend/tests -q --timeout=120 --tb=no` → **26 failed, 2985
passed** at HEAD (all 26 pre-existing; proven by reverting all three phase-85.5
production files and replaying — identical set). Group them by root cause and
**file each cause as its own research-gated masterplan step**, per the standing
rule. Fix only those that are money-path or safety-relevant this session; the
rest are queued work, not overnight work. Do NOT mass-edit tests to green them.

### BLOCKED — cannot be fixed without the operator
**62.1.1 / 85.3.3 (P1 SECURITY)**: the `CLAUDE_CODE_OAUTH_TOKEN` in
`com.pyfinagent.backend.plist` is malformed (length 123, `sk-ant-oat01-` prefix
twice, embedded newline) and byte-identical in
`com.pyfinagent.away-watchdog.plist` — one credential, one fix. Repairing it
needs the correct value. **Do not guess by slicing the malformed one.** Ask #20
is filed; leave it.

---

## Verification spend — bounded authorization

85.6 criterion 4 requires a resume proven live, and **no cycle runs on weekends**
(measured: last 40 cycle_history rows are weekdays only). So a manual cycle
trigger is the only way to roll the anchor and prove the fix before Monday.

**AUTHORIZED: up to TWO manually-triggered verification cycles.** Measured cost
≈ **$0.3 typical, ≤$1.4 worst case each**, billed to Vertex AI on the GCP project
via ADC (per ask #13's costing). This is a deliberate, bounded exception to the
standing `$0 metered` rule — it exists solely to prove 85.4/85.6 end-to-end.

- Read `handoff/.autonomous_loop.lock` — **never `last_result`** — before
  triggering; abort if a cycle holds the lock.
- Log each trigger and its measured cost in the step's live_check.
- If both are consumed without a clean proof, **defer the live leg with a
  recorded reason**; do not spend a third.

---

## Non-negotiable

- **Nothing moves real money.** Paper trading only. Never loosen a safety gate,
  never widen a risk threshold, never disable a guard to make a test pass.
- Resuming the kill switch is IN SCOPE for 85.6 (it is the step's own criterion
  4) — but only via the real code path, after the fix, with the 409 gone.
- No flag promotions. No `backend/.env` writes. `historical_macro` untouched.
- The backend restart is **DONE** (pid 143, 18:54:45; 85.5's P0 is in force,
  proven by the startup log emitting `state=released`). Do not repeat it
  reflexively. Before ANY restart, read the lockfile.
- **Contract BEFORE generate.** If you breach a rule, disclose it AND name which
  automated check is blind to it. Never absolve your own process breach.

## Harness traps that cost real time on 2026-08-08 — do not rediscover these

- **zsh does NOT word-split unquoted `$VAR`.** `$PY $ARGS` passes ONE argument.
  This produced two runs that never executed yet printed "0 failures" and a
  meaningless "IDENTICAL". Use arrays (`"${ARR[@]}"`), and make every harness
  assert its own precondition held before it reports a number.
- **Never restore a single tracked file with a file-level git checkout while
  holding uncommitted work** — it silently reverted edits mid-run. The PreToolUse
  danger hook blocks it; that hook is right. Back up and restore explicitly.
- **Q/A rails drop.** Four evaluations produced three verdicts on 85.5. The two
  drops both had heavy prompts; the clean PASS took 8 tool calls. On an empty
  return, **SHORTEN and re-spawn on the SAME rail** before switching rails. On an
  Agent-tool drop, **`SendMessage` the idle agent for its verdict before
  re-running** — that recovered a full CONDITIONAL carrying a blocker no other
  cycle found.
- An errored/empty Q/A return is **NO VERDICT — never PASS, and not a
  CONDITIONAL either** (it must not count toward the 3rd-CONDITIONAL auto-FAIL).
- Commit promptly with an explicit pathspec. Uncommitted work is not protected.

## Operator-gated work — ask row and SKIP, never ask mid-run

61.3 (#14/#15), 68.1 (#19), 61.2 (#10), 72.0.2 (#13), 62.1.1 (#20).
New decisions get a new numbered row in `handoff/current/operator_ask_2026-08-07.md`.

## Done-definition

85.4 PASS; 85.6 PASS (or its live leg deferred with a recorded reason after two
verification cycles); 85.5.1 PASS; 36.28 widened or a sibling filed; the 26
failures triaged into queued steps; ask list current; `harness_log.md` appended
per cycle; tree committed and pushed clean.

## Stop conditions

- **SOFT STOP** at 12 cycles, or when only operator-gated work remains →
  regenerate this goal with measured state, commit + push, and write
  `handoff/current/overnight_report_2026-08-09.md`: what closed, what did not
  and why, every operator decision owed, and every claim you could not verify.
- **HARD STOP** — stop immediately and write the report: any real-money action,
  any safety-gate loosening, metered spend beyond the two authorized cycles, or
  **3 consecutive infrastructure failures** (rail drops, restart failures).

## Morning deliverable

`handoff/current/overnight_report_2026-08-09.md`, committed and pushed. Lead with
**whether the book can trade on Monday, yes or no**, and if no, exactly what
blocks it. Be honest about what you could not verify — a defect reported as fixed
without live proof is worse than one reported as open.
