# Goal — masterplan drain, 2026-08-10 (cycles continue at 188)

Every number below was measured at ~13:10 CEST on 2026-08-09, not recalled —
**re-derive anything you intend to rely on.**

STARTUP: `git checkout main && git pull origin main`. Confirm
`.claude/settings.json` has `defaultMode: bypassPermissions`. Run `ListAgents` —
**only ONE session flips masterplan steps**. READ FIRST, all BINDING: this file,
CLAUDE.md, `.claude/rules/research-gate.md`, auto-memory MEMORY.md,
`handoff/current/operator_ask_2026-08-07.md`,
`handoff/current/day_report_2026-08-09.md`, and
`handoff/current/killswitch_cluster_reconciliation_2026-08-09.md`.

---

## 1. Measured state

| | measured 2026-08-09 ~13:10 CEST |
|---|---|
| masterplan | **333 pending** (18 P0, 84 P1, 154 P2, 58 P3, 8 P4), 814 done |
| kill switch | `paused: false`, `sod_date: 2026-08-08`, `armed: false`, `daily_baseline_stale: true`, trailing DD **3.3755% / 10%**, NAV 23833.94 |
| token (ask #26) | **STILL MALFORMED** — len 123, `sha256[:12]=9f8c63a185d8`, plist mtime 2026-07-08 |
| `kill_switch_audit.jsonl` | 62 lines, `sha256 90e0303130fc…` |
| lockfile | present but `state: released` — **a leftover from a TEST run, not a cycle** (see §5) |

**CHECK ASK #26 FIRST.** If the token was replaced, **interrupt the queue** and
prove the rail with one cycle — that outranks every step below. If not, nothing
makes the book trade, and everything below is still the right work.

`armed: false` is **case C** — correct, self-clearing at the next cycle's Step-0
roll. **Do not "fix" it.** 85.6's roll is already live-proven
(`backend.log`, `2026-08-08 22:58:29`, the only occurrence). No cycle runs at the
weekend, so there is nothing new to observe until Monday.

---

## 2. START HERE: `86.1` — its research gate is ALREADY PASSED

**Do not re-run the gate.** `handoff/current/research_brief_86.1.md` is on disk
(36,998 bytes) and cleared the enforced rail (8 sources, 44 URLs, 8/8
cross-checked). Go straight to the contract.

**The brief found four things the step text does not know — verify each at
source, then write the contract:**

1. **The isolation asymmetry is INVERTED.** The flag-**ON** arm (`:195-207`) IS
   isolated; the **OFF** arm is not. The step text implies the opposite.
2. **A SECOND landmine.** With the flag ON, `assert out is None` at `:191` goes
   **RED** — suite greenness is coupled to operator configuration.
3. **The `get_state` patch at `:188` is vacuous BY IDENTITY** (`st` is bound at
   `:187` before the patch), and module functions read `_state` directly
   (`:793/:995/:1033/:1047/:1053`).
4. **Redirect-only is a HALF fix** — `:697` corrupts the in-memory singleton
   too. `_audit_archive_dir` is derived (`:89-91`), and `__init__` replays, so
   **redirect BEFORE construction.**

Re-derived line numbers (the step text's `~694` is stale): `reset_peak` at
`kill_switch.py:670` (dark return `:693-694`, assign `:697`, audit `:698-700`),
`_AUDIT_PATH :48`, `_BASELINE_EVENTS :709`, `_append_audit :432-443`,
`_apply_authoritative_peak :397-430`, `settings.py:39`.

**Severity, measured:** the live journal holds **ZERO peak rows** (62 = 44 pause
+ 10 resume + 8 sod). All 20 `peak_update` rows and the 24666.57 max live in
`handoff/audit/` archives; `peak_reset` has never fired. So a row written today
**wins the `ts` merge-sort outright** and destroys 24666.57 permanently —
**trip point 22199.9 → 11110.5.**

**Also in scope:** the stale docstring near `:99` says the old mock "omitted
three" keys; `_snapshot_locked` returns **nine**, so it omitted **seven**.

**79.6 (KS-PEAK-RESET) is APPROVED but NOT APPLIED. It must not be applied until
this lands.**

## 3. Then, each through the full harness loop

Research gate → contract → generate → qa-verdict → harness_log → flip. One step
at a time.

1. **`86.1`** (above) — the landmine armed by an already-approved token.
2. **`86.2`** — one oversized JSON int aborts the whole audit replay and strands
   **both** legs. The only measured path to a *total* disarm. Reproduce with the
   malformed row **FIRST in ts order** — placed last it strands nothing.
3. **`36.17`** — a halted cycle returns before Step 5.6, so stop-losses stop
   being enforced exactly when the book is judged unsafe. Genuine money-path hole.
4. **`86.6`** — the channels a conftest guard cannot reach (filesystem + subprocess).
   **Now carries a live instance** (§5) and 9 criteria.
5. **`86.5`** — the 26-failure triage. Node ids already recorded in
   `live_check_85.4.md` §5; **the baseline is confounded**, see §4.
6. **`36.26` → then `36.20`** (ordering established in the reconciliation), then
   the `36.15` re-scope, `36.10`, `86.4`, remaining `36.x` P1s.

## 4. Traps — do not rediscover these

- **The 26-failure baseline is CONFOUNDED.** It was captured on 2026-08-08 while
  the book was PAUSED. Eleven of its failures are step 36.28's live-pause
  coupling and flip to passing when the book is unpaused — **measured** by
  forcing the singleton back to `paused=True`. Re-baseline under a **known**
  pause state, or land 36.28 first. Compare **live-to-live**, never against a
  worktree (which gives 19-20 failures because it lacks gitignored files).
- **A guard built from the instance you hit is not a guard against the class.**
  This cost two findings in one day: a census that grepped Python method calls
  and could not match an HTTP POST, and an import guard with **1-of-6** recall
  that missed the double-quoted spelling of the very construct it was written
  for. Enumerate the member set, then test recall against all of it.
- **An isolation claim must name every CHANNEL.** Filesystem / HTTP /
  **subprocess** / BigQuery / module-singleton. A worktree relocates file paths
  but not a socket; a conftest guard covers the parent process but not a child.
  Report a measured **delta**, never "never touched".
- **Green in isolation ≠ green in the suite.** A `TestClient` context manager
  runs the app lifespan and killed the whole run at 13% after passing every
  file-scoped run.
- **`node --check` is not evidence a workflow script runs.** It passes on
  forbidden `import` statements and trailing `export` lists that the Workflow
  runtime rejects outright. Only a live spawn catches it.
- **`journal.jsonl` holds PER-AGENT returns; the script's return value lives in
  `workflows/<run>.json`.** Comparing against the wrong one reports a spurious
  mismatch.
- **The 3rd-CONDITIONAL counter is blind on any step still in flight** — it
  greps `harness_log.md`, which is empty until the step closes. **Tell the Q/A
  the real count** from `evaluator_critique_<id>.md`.
- **A "verbatim" block must be re-emitted, never hand-edited.** Generate it with
  `json.dumps(indent=2)` from the stored result.
- **The live_check hold `exit 0`s BEFORE `git add -A`** — a missing
  `live_check_<id>.md` skips the commit AND changelog AND push, not just the push.
- **A newly added workflow is not dispatchable by NAME until session restart.**
  Use `{scriptPath: ...}` in-session. **Verify the name `research-gate` resolves
  this session** and record the result.

## 5. The lockfile is not what it looks like

`handoff/.autonomous_loop.lock` exists with `state: released`,
`cycle_id cycle-1786267675` (= 09:27:55Z), `released_at 09:27:57Z` — a
**two-second** lifetime, pid dead. **That is a test run, not a trading cycle**,
written by the full `backend/tests` suite. Six test files reference
`cycle_lock` / `_LOCK_PATH`. It briefly looks like a real cycle ran on a Sunday.
Folded into **86.6**, which now requires measuring a before/after sha256 over the
**whole** live-state set, not just the kill-switch journal.

## 6. Non-negotiable

- **Nothing moves real money.** Paper trading only. Never loosen a safety gate,
  never weaken an assertion to get green.
- **The full `backend/tests` suite no longer POSTs to the live book** (86.3
  shipped the guard, measured 62 → 62). But it still writes the **filesystem**
  channel (§5), so take before/after digests of the whole live-state set.
- **Search the masterplan before filing a step.**
- **Contract BEFORE generate.** Disclose any breach AND name which automated
  check is blind to it. Never absolve your own breach.
- No flag promotions, no `backend/.env` writes, `historical_macro` untouched.
- Operator-gated work (`79.x`, `62.1.1`, `61.3`, `68.1`, `61.2`, `72.0.2`, asks
  #26 and #27) gets a numbered ask row and is **SKIPPED**.

## 7. Verification spend

**ONE authorized cycle remains — it was NOT spent on 2026-08-09**, deliberately:
with the rail dead it would only re-measure "6/6 degraded, 0 trades". Read
`handoff/.autonomous_loop.lock` — **never `last_result`** — before triggering,
and note §5: a released lock is not a running cycle. Spend it on a
post-token-fix rail verification. If the token is still unfixed, **do not spend
it**; record the reason and defer.

## 8. Stop conditions

- **SOFT STOP** at 20 cycles, when only operator-gated work remains, or at a
  natural end of day → regenerate this goal from measured state, commit + push,
  and write `handoff/current/day_report_2026-08-10.md` leading with **whether the
  book can trade, yes or no**, and if no, exactly what blocks it.
- **HARD STOP** immediately on: any real-money action, any safety-gate
  loosening, metered spend beyond the one authorized cycle, or **3 consecutive
  infrastructure failures**.

Be honest about what you could not verify. **A defect reported as fixed without
proof is worse than one reported as open** — and a claim that is true but
narrower than it sounds is the one that will mislead an operator.
