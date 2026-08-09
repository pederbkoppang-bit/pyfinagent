# Goal — FULL-DAY masterplan drain, 2026-08-09 (cycles continue at 186)

Work the whole day. Every number below was measured at ~04:15 CEST on 2026-08-09,
not recalled — re-derive anything you intend to rely on.

STARTUP: `git checkout main && git pull origin main`. Confirm
`.claude/settings.json` has `defaultMode: bypassPermissions`. Run `ListAgents` —
**only ONE session flips masterplan steps**; if a peer is active, coordinate or
work read-only. READ FIRST, all BINDING: this file, CLAUDE.md,
`.claude/rules/research-gate.md`, auto-memory MEMORY.md,
`handoff/current/operator_ask_2026-08-07.md`, and
`handoff/current/overnight_report_2026-08-09.md`.

---

## 1. Measured state, and the one fact that shapes the day

| | measured 2026-08-09 ~04:15 CEST |
|---|---|
| masterplan | **348 pending** (19 P0, 85 P1, 160 P2, 61 P3, 8 P4), 816 done |
| kill switch | `paused: false`, `sod_date: 2026-08-08`, `sod_nav: 23830.46`, **`armed: false`** |
| token (ask #26) | **STILL MALFORMED** — len 123, prefix ×2, embedded newline, `sha256[:12]=9f8c63a185d8` |
| last cycle | `c67b3b15`, **completed** in 342s, **0 trades**, 6/6 analyses degraded |
| lockfile | absent — no cycle running |

**`armed: false` is EXPECTED and is not a new defect.** It is case C from
phase-85.5.1's own measurement: the UTC date rolled past `sod_date`, so the daily
leg is legitimately unevaluable and correctly disarms. The trailing leg is
date-independent and still fires, so exposure is bounded to `[4%, 10%)`. Phase-85.6's
Step-0 roll re-anchors it at the start of the next cycle. **Verify that happens —
it is the first live confirmation of last night's fix — but do not "fix" it.**

**THE BLOCKING FACT:** the analysis rail is dead. 20 of 20 `claude_code_invoke`
calls exit `code=1` with `duration_api_ms: 0` and zero tokens. **Check ask #26
before anything else.** If the operator has replaced the token, verify the rail
and let a cycle prove it. If not, no engineering makes the book trade today — and
everything below is still the right work.

---

## 2. START HERE: de-duplicate the kill-switch cluster. Do NOT skip this.

**There are 19 pending `36.x` steps plus the 5 `86.x` I filed last night, and they
overlap. Executing them blind will burn the day re-solving solved problems.**
Two overlaps are already confirmed by reading, not guessed:

- **`36.21` ≡ `86.3`.** 36.21 ("THE TEST SUITE WRITES REAL ROWS INTO LIVE
  KILL-SWITCH SAFETY STATE", filed 2026-07-26) describes the *same mechanism* as
  the 86.3 I filed last night, down to "two pause/resume pairs with
  trigger=manual". **I filed a duplicate because I did not search the masterplan
  before filing. That is my error and you should not inherit it.** 86.3 does add
  something real — the measured finding that a git worktree relocates file paths
  but **not** the HTTP channel, so worktree isolation does not contain it. Fold
  that into 36.21 and mark 86.3 `superseded`, or the reverse — but not both.
- **`36.26` may already be CLOSED by `85.6`.** 36.26 is "AN OPERATOR CANNOT UNDO
  THEIR OWN MANUAL PAUSE WHILE THE DAILY ANCHOR IS STALE", hit live 2026-07-27 —
  which is precisely the deadlock 85.6 broke last night (`POST /resume` → 200,
  proven live). **Verify against its own criteria and close it if they are met.**
  Do not re-implement it.

Also inspect for overlap before executing: `36.28` (tests reading live state —
same family as 36.21/86.3), `36.15` (malformed `peak_reset` row discards the true
high-water mark — adjacent to **86.2**'s malformed-row abort and to **86.1**'s
`reset_peak` landmine), and `36.10`/`36.20` (both about the `armed` flag, which
is `false` right now — check whether either is really a live defect or the
correct behaviour you just read about above).

**Deliverable of this first block:** a short written reconciliation
(`handoff/current/killswitch_cluster_reconciliation_2026-08-09.md`) listing every
`36.x` and `86.x` kill-switch step, its true status after inspection, and which
are duplicates / already-closed / genuinely open. Flip the resolved ones with
evidence. **This is the highest-value hour of the day.**

## 3. Then work in this order

Each step gets the full Layer-3 loop: **researcher gate → contract → generate →
qa-verdict → harness_log → flip.** One step at a time.

1. **`86.3` / `36.21`** (whichever survives dedup) — until the suite stops POSTing
   pause/resume to the live book, **every other step's baseline measurement
   corrupts live safety state.** Fixing it first makes the whole day cheaper and
   more honest.
2. **`86.1`** — a landmine armed by an operator decision: the day KS-PEAK-RESET
   (`79.6`, already APPROVED) is applied, running the suite drops the live
   trailing peak from ~24666 to 12345, replayed on every boot. **Fix before that
   token is applied, not after.** Check 79.6's state early.
3. **`86.2`** — one oversized JSON int aborts the whole audit replay and strands
   **both** legs. The only measured path to a *total* disarm.
4. **`36.17`** — a halted cycle returns before Step 5.6, so **stop-losses stop
   being enforced exactly when the book is judged unsafe.** I touched that return
   path in 85.4 (status fidelity) and did **not** fix this. Genuine money-path hole.
5. **`86.5`** — the 26-failure triage. Needs #1 done first to measure safely; the
   node ids are already recorded so you need not re-run the suite to start.
6. **`86.4`** (P3) and the remaining `36.x` P1s by the dedup's own ordering.

If the token is fixed during the day, **interrupt the queue** to verify the rail
end-to-end and let one cycle run — that evidence is worth more than any step.

## 4. Non-negotiable

- **Nothing moves real money.** Paper trading only. Never loosen a safety gate,
  never widen a threshold, never disable a guard to make a test pass. A green
  suite bought by weakening an assertion is worse than an honest red one.
- **Do not casually run the full `backend/tests` suite** — that is defect 86.3/36.21;
  it POSTs a real pause/resume cycle to the live armed book (measured: 8 rows).
  If you must, use a detached worktree **AND** contain the HTTP channel, and
  **prove both**. A worktree relocates `Path(__file__).parents[N]` constants; it
  does **not** relocate a TCP connection to `localhost:8000`.
- **Search the masterplan before filing a new step.** I filed a duplicate last
  night by not doing this.
- **Contract BEFORE generate.** If you breach a rule, disclose it AND name which
  automated check is blind to it. Never absolve your own breach.
- No flag promotions, no `backend/.env` writes, `historical_macro` untouched.
- Operator-gated work (`79.x`, `62.1.1`, `61.3`, `68.1`, `61.2`, `72.0.2`,
  ask #26) gets a numbered ask row and is **SKIPPED** — never ask mid-run.

## 5. Verification spend

**ONE authorized verification cycle remains** (one of two was used at
2026-08-08T20:58Z, measured **$0.60**). It is Sunday — **no scheduled cycle will
run**, so a manual trigger is the only way to prove anything live.

- Read `handoff/.autonomous_loop.lock` — **never `last_result`** — before triggering.
- Spend it on the highest-value proof available, most likely a post-token-fix rail
  verification. If the token is not fixed, **do not spend it** to watch 6/6
  analyses degrade again; that is already measured.
- If more live proof is needed, record the reason and defer rather than overspend.

## 6. Traps from the last two sessions — do not rediscover them

- **An isolation claim must cover every CHANNEL**, not just file paths. I asserted
  worktree isolation and paused the operator's live armed book four times.
  Enumerate filesystem / HTTP / BigQuery / module-singleton, and report a measured
  **delta** ("54 → 62, written by X"), never an absolute ("never touched").
- **Mutate the STUB too.** Three guards last night passed against fakes that
  mirrored the very thing under test. If your fake implements the behaviour, your
  test cannot see production losing it.
- **A source scan cannot tell a live branch from `if False and ...`** — every
  symbol a grep looks for survives. Extract a callable seam and drive it.
- **Ordering matters in reproductions.** A malformed audit row placed *last*
  strands nothing; the run printed the opposite of the claim written above it.
- **zsh does not word-split unquoted `$VAR`.** It bit again. Use arrays and assert
  the derived scope is non-empty before trusting a green result.
- **A masterplan edited via a script does NOT fire the auto-commit hook** (it
  matches Write/Edit only). Commit and push manually, and verify with `git log -1`.
- **The verdict gate reads `evaluator_critique_<id>.json`** — it must hold the
  step's FINAL verdict. Keep earlier passes as `_pass1/_pass2`.
- **Q/A rails drop on heavy prompts.** Keep evidence lean and point at files. An
  empty return is **NO VERDICT** — never PASS, and not a CONDITIONAL either.

## 7. Done-definition

The kill-switch cluster reconciled and its resolved steps flipped with evidence;
`86.3`/`36.21`, `86.1`, `86.2` and `36.17` closed or explicitly deferred with a
recorded reason; `86.5` triage executed if #1 landed early enough; every operator
decision owed captured as a numbered ask; `harness_log.md` appended per cycle;
tree committed and pushed clean.

## 8. Stop conditions

- **SOFT STOP** at **20 cycles**, or when only operator-gated work remains, or at
  a natural end-of-day → regenerate this goal from measured state, commit + push,
  and write `handoff/current/day_report_2026-08-09.md` leading with **whether the
  book can trade on Monday, yes or no**, and if no, exactly what blocks it.
- **HARD STOP** — stop immediately and write the report on: any real-money action,
  any safety-gate loosening, metered spend beyond the one remaining authorized
  cycle, or **3 consecutive infrastructure failures** (rail drops, restart failures).

Be honest about what you could not verify. A defect reported as fixed without
proof is worse than one reported as open.
