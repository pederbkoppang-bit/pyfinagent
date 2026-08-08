# Overnight report — 2026-08-08 → 08-09

## Can the book trade on Monday? **No.**

It can now be *un-paused* — that was last night's deadlock and it is fixed and
proven live. But a cycle that completes still produces **zero trades**, because
the analysis rail is dead.

**What blocks it, exactly one thing:**

> The `CLAUDE_CODE_OAUTH_TOKEN` in `com.pyfinagent.backend.plist` is malformed
> (length 123, `sk-ant-oat01-` prefix twice, embedded newline), and every Claude
> Code rail call now fails instantly. Measured on the verification cycle I ran at
> 20:58Z: **20 of 20** `claude_code_invoke` calls exited `code=1` with
> `duration_api_ms: 0`, `input_tokens: 0`, `total_cost_usd: 0`; the rail breaker
> opened after 20 consecutive failures; the cycle logged
> `Degraded-scoring guard fired: 6/6 analyses scored 0/degraded` and traded
> nothing. `duration_api_ms: 0` with zero tokens means the CLI never reached the
> API — that is authentication, not latency.

**The one action that unblocks Monday: re-issue that token, set it once for both
plists, restart the backend.** It needs the correct value, which only you have.
I did not guess by slicing the malformed one. Filed as **ask #26**.

No credential value appears in any artifact, commit or log from this session —
only lengths, counts and a truncated hash.

---

## What closed

| Step | Result | Cycles |
|---|---|---|
| **85.4** — P0 engine health: the cycle never completes | **PASS**, pushed `8aa3f52e` | 183, 2 EVALUATE passes |
| **85.6** — P0 deadlock: the book cannot be un-paused | **PASS**, pushed `cb34a7c0` | 184, 3 EVALUATE passes |

### 85.4 — why cycles never completed

Measured, not estimated, with a re-runnable harness
(`scripts/diagnostics/measure_analysis_phase.py`) over the loop's own logs:

- 6 tickers, semaphore 3, mean **2310–2320s per ticker** → the 08-06 and 08-07
  cycles project to **8554s and 8529s against a 7200s budget**.
- **7200s is too short. The phase does not hang.**
- The one completing cycle in the window (08-05) fit *only* because two of its
  six analyses failed fast into degraded placeholders. At its own healthy mean it
  projects to 7934s — also over. **The completion rate on a healthy book was 0 of 3.**
- A structural amplifier the step never named: `autonomous_loop.py:1157`/`:1164`
  awaited **two sequential gathers over one semaphore**, so a freed slot idled
  **1923s** on 08-07 while the last ticker waited — larger than the 1329s overrun.

Shipped: status fidelity (a kill-switch halt no longer writes a terminal row
saying `"running"`), alerts that name the phase they died in, and a **second
health clock** measuring *when a cycle last WORKED* rather than *when one last
ENDED* — the first clock read ~24h every day while the book had not completed a
cycle in 7 days.

### 85.6 — why the book could not be un-paused

The daily anchor rolled at exactly one place, behind the analysis phase. Cycles
died there, so the anchor never rolled, the daily leg disarmed, `/resume` 409'd,
and the switch stayed paused. The refusal message told the operator *"NO operator
action is required … this refusal clears itself."* **Both halves were false.**

Fixed by moving the roll to **Step 0**, before anything that can fail.

**Proven live**, not by inspection:

```
20:58:27Z  POST /run-now                      -> 200
20:58:29Z  anchor rolled  '2026-08-05' -> 2026-08-08   (t+2s, real sod_snapshot row)
20:58:43Z  POST /resume                       -> HTTP 200
           {'paused': False, 'sod_date': '2026-08-08'} {'armed': True, 'daily_baseline_stale': False}
```

**The book is live right now.** Kill switch resumed, armed, anchor fresh.

---

## Where I was wrong, and who caught it

The Q/A returned **CONDITIONAL three times across the two steps, and was right
every time.** These are the findings I would want read first.

1. **85.4 — I never ran a required gate.** `ruff` was RED on a dead import in a
   file I created, and no lint run appeared anywhere in my evidence. I had run
   pytest and a mutation matrix and treated those as "the gates".
2. **85.6 pass 1 — my safety claim was directionally half-wrong.** Anchoring from
   the last stored mark and stamping *today's* date on it converts a designed
   disarm into a **spurious flatten** when that mark is several sessions old:
   measured with the production `evaluate_breach`, 5.16% on a 4% limit where the
   old code read 0.00%. It was live-reachable that night — the cycle passed by
   **margin (−0.0146%), not by design**. My contract had disclosed that direction
   and argued it away with no test and no bound.
3. **85.6 pass 2 — my fix was one seam short.** I made the marker an in-memory
   flag on `PaperTrader`, and the loop rebuilds that every cycle, so it did not
   survive the *cycle* — not merely the process, as I had written. **Both of my
   residual disclosures were materially false.** Fixed by persisting the marker
   into the audit row and replaying it on boot.

My own mutation matrices caught three more defects in my own tests: an ordering
guard that timed the wrong event, a scope guard defeated by a monotonic stub, and
two durability guards blind because my fakes mirrored the thing under test. The
"mutate the stub too" lesson arrived three times in one session.

---

## Disclosures against interest

- **A spend accident.** One test run reached the real Claude CLI, because
  emptying `screen_universe` does not empty the funnel (`rank_candidates` was
  also stubbed). Killed on discovery; `llm_call_log` showed **0 rows** in the
  preceding 30 minutes, so **no metered spend**. The fixture now installs a
  **spend guard** that fails loudly if any test reaches the rail.
- **Tests wrote to live operator state**, again — one `pause` row into
  `handoff/kill_switch_audit.jsonl` and an overwrite of
  `handoff/.cycle_heartbeat.json` (the heartbeat is *not* fail-safe: refreshing
  it makes a dead emitter look alive). Pre-existing phase-36.28 class; my own new
  tests are isolated and the Q/A independently confirmed they touch nothing.
- **A pre-existing test was deliberately inverted.** phase-36.9 asserted the 409
  *contains* `"NO operator action is required"` — the phrase 85.6's criterion 2
  requires removed. Intent preserved; the Q/A judged the supersession legitimate.
- **I restarted the backend twice**, both times justified (to put a committed fix
  in force) and both times after reading the lockfile. I cannot rule out that one
  of today's restarts is what put the malformed token into effect — both load the
  same plist and the artifacts cannot separate them. Stated in ask #26.

## Spend

- **One** of the two authorized verification cycles used. Measured **$0.60**.
  One remains unspent.
- No other metered spend.

---

## What I could not verify

- **That Monday's cycle completes.** 85.4's remedies are operator-gated by its
  own criterion 5 (asks #23/#24/#25), so the 7200s budget, the 150s rail timeout
  and the merged-dispatch flag are all unchanged. With a *working* rail the
  budget is still arithmetically short.
- **That the durable provisional marker behaves in production.** It is proven by
  tests against the real `KillSwitchState` and the backend now runs post-marker
  code (pid 36970, started 00:03:29, after the commit), but no cycle has run
  under it. `GET /kill-switch` does not expose the field, so there is no live
  read to show you.
- **Which restart broke the rail** (see ask #26).
- **The 26 pre-existing suite failures as a *set*.** I verified the count is
  unchanged (26 → 26) and that passes rose by exactly my new test count, and I
  recorded the current 26 node ids so the next session can diff sets. I did not
  independently re-derive the prior set; that claim is inherited from cycle 182.

## Operator decisions owed

| # | What | Why it matters |
|---|---|---|
| **26** | **Re-issue `CLAUDE_CODE_OAUTH_TOKEN`, set once for both plists, restart** | **The only thing standing between the book and trading.** |
| 23 | `paper_cycle_max_seconds` 7200 → 10800 | Measured: cycles need ~8500s. Without it they time out even with a working rail. |
| 24 | Rail `timeout_s` 150 → 210 | ~26% of rail time was spent on calls timing out 5s short of succeeding. |
| 25 | Flip `paper_merged_analysis_dispatch_enabled` ON | Dark. Would have saved 1923s on 08-07 — more than that cycle's overrun. |
| 20 | Superseded by #26 (same credential) | — |
| 14/15, 19, 10, 13 | Pre-existing, untouched this session | — |

---

## Not done, and why

The goal's defect list had five items. Two closed; three remain, all still open
and none blocked:

- **85.5.1** (P1 book safety, the RED `test_valid_nav_still_breaches`). The 85.6
  research gate answered its central question as a side effect and I recorded the
  scope boundary in `contract_85.6.md` §7: the RED test *is* a fixture defect
  (the mock omits `sod_date`), **but** `sod_date=None` with a positive `sod_nav`
  **is** production-reachable via `_load_from_audit:285-295`, and the trailing leg
  still fires — so exposure is bounded to drawdowns in
  `[daily_limit, trailing_limit)`. It self-heals except inside 85.6's deadlock,
  which is now fixed. **The next session starts here; the analysis is already done.**
- **Widening 36.28** (tests write to the live kill-switch journal and heartbeat).
  Two fresh instances observed tonight, both recorded.
- **Triaging the 26 failures** into research-gated steps.

I stopped short of these because two P0s consumed the night at five EVALUATE
passes, and because starting a third step at 01:00 after three CONDITIONALs on
safety-critical code is how the defects I spent the night fixing get written.

## State of the tree

Committed and pushed clean. `origin/main` at `129c3eda`. Both masterplan steps
flipped to `done` by the hook, which pushed each after its verdict gate passed.
