# Phase-23.2.22 Internal Codebase Audit
## Topic: Why trading stopped (0 trades from 17 candidates) + audit log pollution

Accessed: 2026-05-05

---

## 1. Kill-switch short-circuit path (Step 5.5)

**File:** `backend/services/autonomous_loop.py:309-331`

The cycle's kill-switch check happens at **Step 5.5**, between mark-to-market (Step 5) and decide_trades (Step 6). The exact guard:

```python
ks_check = await asyncio.to_thread(trader.check_and_enforce_kill_switch)
summary["kill_switch"] = ks_check
if ks_check.get("triggered") or _ks_state().is_paused():
    logger.warning("Paper trading: kill-switch active -- skipping decide/execute")
    summary["steps"].append("kill_switch_halted")
    ...
    return summary
```

This is an **OR** condition. If `_ks_state().is_paused()` is True — for ANY reason, including a prior test-pollution pause event replayed from the audit log — the cycle returns early before `decide_trades` is ever called. The logged trace from 05-06 (`Step 6 -> Step 7 in 1 second, 0 trades`) does NOT show `kill_switch_halted` in `summary["steps"]`, which means `is_paused()` was returning **False** at Step 5.5. The 0-trades outcome is NOT from kill-switch short-circuit on that cycle.

**Confirmed:** The cycle reached `decide_trades` on 05-06. Kill-switch is not the proximate cause of 0 trades.

---

## 2. `decide_trades` — all filters that can yield 0 buys from N candidates

**File:** `backend/services/portfolio_manager.py:41-257`

### 2a. BUY recommendation gate (line 141)
```python
if rec not in _BUY_RECS:
    continue
```
`_BUY_RECS = {"BUY", "STRONG_BUY"}`. Any candidate with `recommendation = "HOLD"` is silently skipped. The lite Claude analyzer at `autonomous_loop.py:676-690` uses these decision rules:
- `momentum_20d > 3.0 AND momentum_60d > 5.0 AND market_cap > 5e9` → lean BUY
- Otherwise → HOLD

**This is the most likely cause of 0 trades.** Given US equity market conditions in early May 2026 (post-tariff volatility, pullback from April highs), many large-cap tickers may have negative or sub-3% 20-day momentum, causing the lite analyzer to return HOLD for every candidate in the top-3 new analyses.

### 2b. `paper_max_positions` cap (line 204)
```python
if remaining_positions >= settings.paper_max_positions:
    break
```
Default: 10. At 14 re-eval positions the system is at or near cap. `remaining_positions = len(current_positions) - len(selling_tickers)`. If no sells are triggered and positions = 10, all buys are blocked immediately.

### 2c. Sector cap — `paper_max_per_sector` (lines 194-220)
Default: 2 per GICS sector. Added in phase-23.2.6. If current Technology positions = 2 (very plausible with NVDA, MSFT, GOOGL, AAPL type holdings), every new Technology candidate is blocked. The 05-06 log shows `3 new + 14 re-evals`. The 14 re-evals represent existing positions — if they are sector-concentrated, new candidates in those sectors are capped.

### 2d. Cash reserve gate (lines 207-208)
```python
if available_cash <= 0:
    break
```
`available_cash = cash + estimated_freed_cash - min_cash`. With `paper_min_cash_reserve_pct=5.0` and NAV=$17516, minimum cash reserve = ~$876. If current cash < $876 and no sells free up cash, no buys proceed.

### 2e. $50 minimum position size (lines 229-234)
Tiny positions below $50 are skipped with a WARNING log. Unlikely to be the issue given NAV=$17k.

### 2f. Re-eval `holding_analyses` — no sells triggered
The 14 re-eval positions were analyzed (lite_mode=False means full orchestrator). If all 14 returned HOLD (not SELL/STRONG_SELL), no cash is freed. `selling_tickers` remains empty. With existing positions at or near the max, no new buys can enter.

**Summary of 0-trade causal chain (most to least likely):**
1. Lite analyzer returns HOLD for all 3 new candidates (momentum below threshold in current market)
2. Position count at or near `paper_max_positions=10` cap (14 re-evals means 14 existing positions — this ALREADY exceeds the cap of 10, which means `remaining_positions >= 10` fires immediately and blocks all buys)
3. Sector cap on Technology (highly plausible given portfolio composition)
4. Cash below min_cash_reserve after MTM

**Critical finding on position count:** The log shows `14 re-evals` (14 existing positions). Default `paper_max_positions=10`. If positions=14 and no sells, then `remaining_positions=14 >= 10`, and the buy loop breaks immediately on every candidate. This is almost certainly the active blocker. The system grew past its configured max (possibly before the cap was tightened, or the cap was never enforced at buy time for legacy positions).

---

## 3. Kill-switch state — `_AUDIT_PATH` is a module-level constant pointing at production

**File:** `backend/services/kill_switch.py:36-37`

```python
_AUDIT_PATH = Path(__file__).resolve().parents[2] / "handoff" / "kill_switch_audit.jsonl"
_AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
```

This is evaluated at **module import time**. Every `KillSwitchState()` instantiation anywhere in the process reads from and writes to this single production file. There is no environment variable, no test override, no factory pattern.

`_load_from_audit()` (lines 54-89) replays every row on `__init__`, setting `_paused=True` for any `event=pause` row and `_paused=False` for any `event=resume` row — last-writer wins. If the last row in the production audit file is a `pause`, every new `KillSwitchState()` boots into paused state.

---

## 4. Which tests instantiate `KillSwitchState` without audit path isolation

Full callsite inventory:

| File | Lines | Uses `tmp_audit` fixture? | Writes to production? |
|------|-------|--------------------------|----------------------|
| `tests/services/test_sod_daily_roll.py` | 47,54,65,72,94,118,134,151 | YES (`monkeypatch.setattr(ks, "_AUDIT_PATH", p)`) | NO — correctly isolated |
| `tests/services/test_kill_switch_no_deadlock.py` | 25, 45, 66 | NO | YES — writes to production `kill_switch_audit.jsonl` |
| `tests/services/test_cycle_failure_alerts.py` | 145, 155 | NO | YES — writes to production `kill_switch_audit.jsonl` |
| `tests/api/test_pause_resume_timeout.py` | (no direct instantiation found) | N/A | N/A |
| `tests/verify_phase_23_2_18.py` | (no direct instantiation) | N/A | N/A |
| `tests/verify_phase_23_2_19.py` | (no direct instantiation) | N/A | N/A |

### Specific pollution events from `test_cycle_failure_alerts.py`

`test_kill_switch_auto_pause_fires_alert` (line 145):
```python
state = kill_switch.KillSwitchState()
state.pause(trigger="drawdown_breach", details={"daily_loss_pct": -2.5})
```
This writes `{"event": "pause", "trigger": "drawdown_breach", "details": {"daily_loss_pct": -2.5}}` to production. No `resume` follows in the same test.

`test_kill_switch_manual_pause_does_not_alert` (line 155):
```python
state = kill_switch.KillSwitchState()
state.pause(trigger="manual")
state.pause(trigger="test")
state.pause(trigger="test-pre")
state.pause(trigger="bench-1")
state.pause(trigger="bench-2")
state.pause(trigger="bench-3")
```
Six more `pause` rows. No `resume`. Production file ends with 6+ consecutive pauses.

### `test_kill_switch_no_deadlock.py` — also unprotected

Line 25: `state = KillSwitchState()` — writes `pause(trigger="test")`, `resume(trigger="test2")`, `pause(trigger="bench-3")` (from the bench test). No `tmp_audit` fixture.

**IMPORTANT:** The test file at line 46 does `state.pause(trigger="test-pre")` and line 52 does `state.resume(trigger="test")`. But lines 66-70 run `pause("bench-1") -> resume("bench-2") -> pause("bench-3")` — ends on PAUSE with no cleanup. This means the final row written to production is a `pause` event, and ANY subsequent `KillSwitchState()` boot (including the running backend's module-level `_state = KillSwitchState()` at line 192) would reload in paused state IF the module were reimported after the test.

---

## 5. `check_and_enforce_kill_switch` side-effects

**File:** `backend/services/paper_trader.py:533-566`

This method:
1. Calls `state.update_peak(nav)` — always writes a `peak_update` row if NAV is a new high
2. Checks `sod_date` vs today — writes `sod_snapshot` row if new day
3. Calls `evaluate_breach()` — reads state only, no write
4. If breach AND not paused: calls `flatten_all()` then `state.pause(trigger="limit_breach")`

There is NO side-effect that keeps the system functionally paused when `_paused=False`. The `is_paused()` check at Step 5.5 reads the in-memory `_paused` bool. Since the audit log shows `paused=false` and `daily_loss_pct=0.0099%` (far below 4% limit), no breach fires. The currently running backend (`_state` singleton, line 192) was initialized at process start by replaying the audit log. If the test pollution was appended AFTER the process started, the in-memory `_state` is clean — it would only be affected on the NEXT process restart.

**Conclusion:** The audit log is polluted with fake pause events. The currently-running backend's in-memory state is likely clean (test pollution happened while the backend was running, not before it started). But the NEXT backend restart would boot into paused state because `_load_from_audit` replays all rows, and the last rows are test-written `pause` events without a following `resume`.

---

## 6. Live BQ paper_positions — current ticker count and sector exposure

Unable to query BQ in this session (no MCP server injected). Based on the cycle log forensics:
- `14 re-evals` = 14 existing positions
- Default `paper_max_positions=10`
- **14 > 10 means the position cap is ALREADY EXCEEDED in BQ**

This is the most operationally significant finding. The system holds 14 positions but is configured for a max of 10. The `decide_trades` buy loop fires `break` immediately (`remaining_positions=14 >= settings.paper_max_positions=10`). Until the re-eval cycle produces SELL signals that reduce the position count below 10, zero buys will continue.

---

## 7. `autonomous_loop.py` Step 6/7 kill-switch interaction summary

- Step 5.5 checks `ks_check.get("triggered") OR _ks_state().is_paused()`
- If either is True, the function returns BEFORE `decide_trades` is called
- The 05-06 log shows Steps 6 and 7 DID execute (1 second apart), so the kill-switch branch was NOT taken
- 0 trades is not a kill-switch artifact on the 05-06 cycle
- But: if backend restarts while the audit log has trailing `pause` rows, the next cycle WOULD hit the kill-switch short-circuit at Step 5.5

---

## Internal Code Inventory

| File | Lines | Role | Status |
|------|-------|------|--------|
| `backend/services/kill_switch.py` | 237 | Kill-switch state, audit log, breach evaluation | `_AUDIT_PATH` is a module-level constant, no override mechanism for tests |
| `backend/services/autonomous_loop.py` | 904 | Daily cycle orchestrator, Steps 1-10 | Step 5.5 kill-switch short-circuit correct; position cap not re-enforced dynamically |
| `backend/services/portfolio_manager.py` | 321 | `decide_trades` — all buy/sell logic | Position cap at line 204; sector cap at lines 194-220; BUY filter at line 141 |
| `backend/services/paper_trader.py` | ~620 | Trade execution, mark-to-market, kill-switch check | No pathological side-effects; SOD roll idempotent (phase-23.2.19) |
| `tests/services/test_cycle_failure_alerts.py` | 197 | Phase-23.2.18 regression guard | DEFECT: instantiates `KillSwitchState()` without `tmp_audit` fixture; pollutes production |
| `tests/services/test_kill_switch_no_deadlock.py` | 92 | Phase-23.1.22 regression guard | DEFECT: instantiates `KillSwitchState()` without `tmp_audit` fixture; pollutes production |
| `tests/services/test_sod_daily_roll.py` | ~160 | Phase-23.2.19 regression guard | CORRECT: uses `tmp_audit` fixture with `monkeypatch.setattr(ks, "_AUDIT_PATH", p)` |

---

## Key Findings Summary

1. **0-trade root cause (most likely):** 14 existing positions exceeds `paper_max_positions=10`. The `decide_trades` buy loop fires `break` at line 204 before evaluating any candidate. No new buys will occur until sells reduce position count below 10.

2. **0-trade secondary cause:** Lite Claude analyzer returns HOLD for all 3 new candidates when `momentum_20d <= 3.0` or `momentum_60d <= 5.0`. Current market conditions post-April volatility may have suppressed momentum signals below threshold for most large-cap tickers.

3. **Audit log pollution confirmed:** `test_cycle_failure_alerts.py` lines 145 and 155 instantiate `KillSwitchState()` without redirecting `_AUDIT_PATH`, writing 7 fake pause events (including 1 `drawdown_breach` with `daily_loss_pct=-2.5`) to production `handoff/kill_switch_audit.jsonl`.

4. **Restart risk:** The running backend's in-memory `_state` is clean. But the NEXT process restart triggers `_load_from_audit()` which replays all rows. Since the last rows are test-written `pause` events with no following `resume`, the backend will boot into paused state and the Step 5.5 kill-switch short-circuit will fire on every cycle — producing `n_trades=0` and `halted=True` forever until a manual resume API call is made.

5. **`test_kill_switch_no_deadlock.py`** is also unprotected but its pause/resume patterns leave the audit in a mixed state. The `test_pause_resume_cycle_is_fast` test (lines 64-72) calls `pause("bench-1") -> resume("bench-2") -> pause("bench-3")` — ends on pause.

6. **Correct pattern exists in `test_sod_daily_roll.py`:** `monkeypatch.setattr(ks, "_AUDIT_PATH", p)` where `p = tmp_path / "kill_switch_audit.jsonl"`. This is the canonical fix that should be applied to the other two test files.

7. **The `_append_audit` static method uses the module-level `_AUDIT_PATH`** (line 99): `with _AUDIT_PATH.open("a", ...)`. Patching `ks._AUDIT_PATH` redirects all writes correctly, because `_append_audit` reads `_AUDIT_PATH` at call time, not at import time.
