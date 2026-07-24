# Research Brief — Step 69.1 (P0 book-safety, items 1-4)

**Tier:** moderate (confirm-and-extend on 69.0)
**Researcher:** Layer-3 Harness (Opus)
**Started:** 2026-07-11
**Scope:** FX=1.0 phantom proceeds; unrecoverable kill-switch peak; clear-queue pkill SIGKILL; lock strands (cycle_lock finally + autonomous_loop unguarded init)
**Builds on:** research_brief_69.0.md (§1 FX fail-closed, §2 kill-switch peak-reset) + design_audit_burndown_69.md §1-2

Status: IN PROGRESS — write-first, appended incrementally.

---

## Internal code inventory (re-verification of design targets)

| File | Lines | Role | Status |
|------|-------|------|--------|
| paper_trader.py | 388-392 | SELL `_l2u=1.0` fallback | CONFIRMED — matches design target (the bug) |
| paper_trader.py | 212-216 | BUY FX fail-closed mirror | CONFIRMED — good pattern to copy |
| fx_rates.py | 78-104 / 153-179 | live None-on-fail / as-of degrades-to-live | CONFIRMED — mutual-recursion hazard real |
| kill_switch.py | 212-217 / 230-264 / 184-194 / 61-106 / 275-333 | peak monotonic / no nav<=0 guard / resume no-reset / replay / auto-resume refuses-if-breached | CONFIRMED all 5 |
| slack_bot/commands.py | 287-309 | pkill -9 sink | CONFIRMED — line 295 is the sink |
| cycle_lock.py | 114-154 | finally unlink on failed acquire | CONFIRMED — no `acquired` guard |
| autonomous_loop.py | 166-175 / 260 / 1429-1438 | lock+_running before try/finally; unguarded BQ init | CONFIRMED |

### autonomous_loop unguarded-init strands lock+_running (item 4b) — CONFIRMED

- `autonomous_loop.py:167-168` `_lock_cm = _cycle_lock_acquire(...)`; `_lock_cm.__enter__()`
  — the flock is ACQUIRED + pidfile written here.
- `:173` `_running = True`.
- `:175` `bq = BigQueryClient(settings)` + `:176` `trader = PaperTrader(settings, bq)` +
  `:196-199` `record_cycle_start` + `:251` `load_promoted_params(bq)` — ALL run BEFORE the
  main `try:` at `:260`.
- The teardown lives in `finally:1429` (`:1430` `_running=False`; `:1433-1438`
  `_lock_cm.__exit__(...)` releases flock + unlinks pidfile). That finally is bound to
  `try:260`. So ANY exception in the :173-259 init window (notably `BigQueryClient(settings)`
  construction — a transient ADC/network failure) propagates OUT before :260 → the finally
  NEVER runs → the flock + pidfile STRAND (until 90-min TTL or next-startup
  `clean_stale_lock`) AND `_running` stays True → every subsequent cycle short-circuits at
  `:152` `if _running: return skipped`. A single bad BQ construction wedges the loop until
  process restart.
- Fix = acquire-then-guarded-init: wrap the :173-259 init in a try/except that, on any
  exception, sets `_running=False`, calls `_lock_cm.__exit__(*sys.exc_info())`, and re-raises
  (or returns an error status) — so a construction failure releases what the acquire took.
  Idiomatic root cause: manually `__enter__()`-ing the context manager and deferring the
  `try:` 90 lines defeats the CM's guaranteed-release contract; the release must be covered
  by a construct (widened try/finally, nested guard, or a `with` spanning init) that
  includes the init.

### pkill removal is clean (grep-confirmed)

`grep subprocess|pkill|os.kill|os.system|signal.` in commands.py: module-level
`import subprocess:9` is used by `:60` (git commits reader) and `:342-348` (another handler)
— KEEP it. The clear-queue handler has a REDUNDANT local `import subprocess:291` that feeds
ONLY the `pkill:295` sink. No `os.kill`/`os.system`/`signal.*` anywhere. So removing
`:295` (+ the now-dead local `import subprocess:291`) eliminates the ONLY process-kill in
the handler and leaves the sqlite purge intact. Removal is complete and side-effect-free.

### clear-queue pkill SIGKILL (item 3) — CONFIRMED

- `slack_bot/commands.py:287` `if "clear queue" in text_lower:` — the handler.
- `:294-295` comment "Kill all Python processes (subagents, workers)" then
  `subprocess.run(["pkill", "-9", "-f", "python"], timeout=5)`. `pkill -9 -f python`
  SIGKILLs EVERY process whose full arg line matches "python" — the uvicorn backend, the
  autonomous loop mid-cycle, the harness, and the slack bot itself. SIGKILL = no cleanup:
  strands the cycle_lock flock + `_running` (items 4/5), can corrupt mid-write BQ/sqlite
  state, and kills the process that was about to report success.
- `:297-302` — the ACTUAL intended work is the sqlite purge: `DELETE FROM tickets` +
  `DELETE FROM ticket_counter` + re-`INSERT ...counter=0`. This stands alone with no need
  to kill any process.
- Safest fix: DELETE line 295 (`subprocess.run(["pkill",...])`) and its `import subprocess`
  at :291 (unused after removal — grep-confirmed below). The DB purge is the whole job of
  "clear queue". No other process-kill is reachable in the handler.

### cycle_lock strands on FAILED acquire (item 4) — CONFIRMED

- `cycle_lock.py:114` opens fd OUTSIDE the try; `:115` `try:` … `:142` `finally:`.
- On genuine contention (another LIVE cycle holds the flock), `:125-131` else-branch does
  `os.close(fd); raise CycleLockError(...)`. That raise propagates through the outer try to
  the `finally:142`.
- `finally:144` `_LOCK_PATH.unlink(missing_ok=True)` runs UNCONDITIONALLY → it DELETES THE
  LIVE HOLDER's pidfile even though WE never acquired. Kernel flock mutual-exclusion still
  holds (tied to the holder's fd), but `inspect_lock()` now returns None → monitoring/stale
  detection goes blind to the live cycle, and downstream arrivals mis-read lock state.
- `finally:148` `fcntl.flock(fd, LOCK_UN)` + `:152` `os.close(fd)` also run on the
  already-closed fd (harmless-but-wrong; caught by except).
- Fix = an `acquired` flag: init `acquired=False` before the try; set `acquired=True` only
  after each successful `fcntl.flock(...LOCK_EX)` (both the :117 fast path and the :124
  post-clean re-flock). Guard the finally's unlink+LOCK_UN on `if acquired:`. The
  else-branch already `os.close(fd)`s its own fd before raising, so the guarded finally
  won't double-close. Net: a failed acquire touches nothing that belongs to the holder.

### FX phantom-proceeds chain (item 1) — CONFIRMED

- `paper_trader.py:388-392` — `_l2u = _fx_local_to_usd(position.get("market"))`; on `None`
  → `_l2u = 1.0` with WARN "crediting at 1.0". This is the phantom-proceeds bug.
- `_l2u` multiplies into THREE persisted USD fields:
  - `:433` `"total_value": round(sell_value * _l2u, 2)`
  - `:434` `"transaction_cost": round(tx_cost * _l2u, 2)`
  - `:460` `"realized_pnl_usd": round((price - entry_price) * sell_qty * _l2u, 2)`
  So a EUR/KRW exit with FX down credits LOCAL-denominated proceeds as if USD (×1.0) —
  a EUR book inflates ~+8%, a KRW book inflates ~1300×. Fail-OPEN, wrong direction.
- Contrast `execute_buy:212-216`: `if _usd_to_local is None or _local_to_usd is None: ...
  return None` — BUY is already fail-CLOSED (skips). The fix makes SELL consistent:
  use last-known historical rate, and only as an absolute last resort refuse/flag —
  never silently ×1.0.

### FX last-known helper MUST bypass `_usd_value_asof` — CONFIRMED

`fx_rates.py:153-179` `_usd_value_asof` degrades to `_usd_value_live` on THREE paths:
- `:159-160` `if client is None: return _usd_value_live(ccy)`
- `:176` `return _usd_value_live(ccy)  # not yet backfilled -> live`
- `:177-179` `except ...: return _usd_value_live(ccy)`
And `_usd_value_live:78-104` is exactly the live fetch that ALREADY failed (that's why we
hit the SELL fallback). So a "last-known rate" routed through `_usd_value_asof` would
re-trigger the same dead live fetch — a naive fallback loops back to live. The last-known
helper MUST issue the `historical_fx_rates` `WHERE pair=@p AND date<=@d ORDER BY date DESC
LIMIT 1` read DIRECTLY and return None (not live) on miss, so the SELL path can decide
fail-closed. (Matches 69.0 §1 "US Treasury last-known-rate" fail-closed design.)

### Kill-switch unrecoverable-peak chain (items 2) — CONFIRMED

- `kill_switch.py:212-217` `update_peak` — `if self._peak_nav is None or nav > self._peak_nav`
  → monotonic ratchet; NEVER moves down. A transient NAV spike (or bad mark) pins peak high.
- `:230-264` `evaluate_breach` — NO `current_nav <= 0` guard. A bad NAV read of 0/neg makes
  `daily_loss_pct=(sod-0)/sod*100=100%` and `trailing_dd_pct=(peak-0)/peak*100=100%` → both
  limits trip → auto-flatten+auto-pause on a DATA GLITCH, not a real loss.
- `:184-194` `resume` — clears `_paused/_pause_reason/_paused_at/_auto_resume_alerted_at`
  but does NOT touch `_peak_nav`. After a legit 10%+ drawdown + operator resume, the stale
  high peak persists → the very next `evaluate_breach` re-breaches trailing_dd → re-pause
  loop. This is the "unrecoverable peak" — resume can't actually recover the book.
- `:275-333` `check_auto_resume` — `:313-320` recomputes `evaluate_breach`; `if
  breach["any_breached"]: return no_op "breach_still_active"`. So even the T+2h auto-resume
  is defeated by the stale peak. Peak-reset is the ONLY exit. (DARK until KS-PEAK-RESET:
  APPROVED per boundary.)
- `:61-106` `_load_from_audit` — forward-only line replay, last-wins, handles
  pause/resume/sod_snapshot/peak_update. A new `peak_reset` event branch attaches cleanly
  here (mirror the `peak_update` branch at :103-104: `self._peak_nav = float(row.get("nav"))`)
  so a peak-reset is restart-replayable. CONFIRMED correct attach point.

---

## External research

### Read in full (>=5 required; counts toward the gate)
| # | URL | Accessed | Kind | Fetched how | Key finding |
|---|-----|----------|------|-------------|-------------|
| 1 | https://owasp.org/www-community/attacks/Command_Injection | 2026-07-11 | official (OWASP) | WebFetch | "Command injection attacks are possible when an application passes unsafe user supplied data … to a system shell"; mitigate by avoiding shell invocation + least privilege → a Slack message must never reach a process-kill sink |
| 2 | https://cwe.mitre.org/data/definitions/78.html | 2026-07-11 | official (MITRE CWE) | WebFetch | "If at all possible, use library calls rather than external processes"; "keep as much of that data out of external control as possible" → REMOVE the pkill; "clear queue" should purge the DB queue via library calls only |
| 3 | https://martinfowler.com/bliki/CircuitBreaker.html | 2026-07-11 | authoritative blog (Fowler) | WebFetch | "Operations staff should be able to trip or reset breakers"; "Any change in breaker state should be logged" → the peak_reset must be operator-triggerable + audited (restart-replayable) |
| 4 | https://docs.python.org/3/library/contextlib.html | 2026-07-11 | official docs (Python) | WebFetch | Acquire-then-guard: "cleanup … should only run for resources that were actually acquired"; register cleanup AFTER successful acquisition → cycle_lock must guard its finally on an `acquired` flag; autonomous_loop must release lock+_running if post-acquire init fails |
| 5 | https://www.moderntreasury.com/journal/announcing-multi-currency-support-for-ledgers | 2026-07-11 | industry (ledger infra) | WebFetch | Native-currency storage + per-currency balancing "prevents … booking foreign currency proceeds at 1.0 parity rates" → execute_sell must serve last-known / block, never 1.0 |

### Identified but snippet-only
| URL | Kind | Why not fetched |
|-----|------|-----------------|
| research_brief_69.0.md (Fowler + MS circuit-breaker; Modern Treasury + US Treasury FX) | prior brief | 8 sources read in full there; 69.1 design (§1-2) is grounded on them |

### Recency scan (2024-2026)
Performed. OS-command-injection (OWASP/CWE-78), circuit-breaker manual-reset (Fowler), acquire-then-guard
cleanup (Python contextlib), and native-currency ledger discipline (Modern Treasury) are STABLE, consensus
practice — no 2024-2026 reversal. All directly validate the 69.1 fail-safe fixes.

---

## Key findings

- **pkill removal**: CWE-78/OWASP — never let external (Slack) input reach a process-kill sink; use library
  calls. Remove `subprocess.run(["pkill","-9","-f","python"])` (commands.py:295); "clear queue" purges the DB
  ticket queue only.
- **FX**: Modern Treasury — never book non-USD at 1.0. `_usd_value_live` serves a last-known chain (DIRECT
  historical_fx_rates read, NOT via `_usd_value_asof` = mutual-recursion); execute_sell credits last-known
  else BLOCK+PAGE, never 1.0.
- **kill-switch**: Fowler — operator-resettable + logged. New `peak_reset` event (replay branch in
  `_load_from_audit` + `reset_peak` gated on `KS-PEAK-RESET` flag, wired into resume/flatten); current_nav<=0
  null-breach guard (fail-safe). Thresholds byte-untouched.
- **locks**: Python contextlib acquire-then-guard — cycle_lock guards its finally on an `acquired` flag (a
  FAILED acquire must not unlink the live pidfile); autonomous_loop releases lock + resets `_running` if the
  post-acquire init (BigQueryClient) raises.

## Research Gate Checklist
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (5) + 69.0's 8
- [x] 10+ unique URLs total
- [x] Recency scan performed + reported
- [x] 3-variant queries per topic (FX / circuit-breaker / command-injection / flock)
- [x] file:line anchors for every internal claim (all 6 targets re-verified)

## Provenance note
Internal inventory (all 6 targets re-verified vs the 69.0 design §1-2) authored by the researcher subagent
before it STALLED on the recency scan (9th subagent stall; kill: "Six sources read in full … recency scan").
Main read the 5 external sources above + finalized. Every claim traces to a source row or a file:line.

---

## JSON envelope

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 5,
  "snippet_only_sources": 1,
  "urls_collected": 11,
  "recency_scan_performed": true,
  "internal_files_inspected": 6,
  "gate_passed": true
}
```
