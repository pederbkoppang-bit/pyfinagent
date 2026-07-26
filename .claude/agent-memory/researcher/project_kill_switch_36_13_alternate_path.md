---
name: kill-switch-36-13-alternate-path
description: phase-36.13 execute_buy kill-switch bypass — pause IS restart-durable so drills really do break; both "deliberate" callers use stub BQ; kill_switch_test.py tests the WRONG control; only 2 paper_trades producers exist
metadata:
  type: project
---

Measured 2026-07-26 during the 36.13 research gate (P0: `execute_buy` has no
kill-switch gate; the switch is enforced only on the autonomous-cycle path).

**Pause is restart-durable — the drill breakage is REAL, not hypothetical.**
`kill_switch.py:675` is a module-level singleton `_state = KillSwitchState()`;
`__init__` (:167-189) ends with `_load_from_audit()`, and the replay at :254-268
handles `event == "pause"` by setting `self._paused = True`. So ANY new Python
process that imports `kill_switch` inherits a live pause from
`handoff/kill_switch_audit.jsonl` + rotated archives. A naive
`if get_state().is_paused(): return None` inside `execute_buy` therefore breaks
`zero_orders_drill.py` and `smoketest_stages_5_through_13.py` — but only on days
the book is paused (intermittent, state-dependent, red exactly during a fire).

**Both "deliberate" callers are already stubbed, so no bypass API is needed.**
`zero_orders_drill.py:64,92` uses an in-file `StubBQ` (:44-59);
`smoketest_stages_5_through_13.py:174,181` uses `MagicMock()` and even patches
`ExecutionRouter`. They need to *supply state*, not bypass a control — which
`PaperTrader.__init__` (:94-107, already injects `bq_client` + `trade_notifier`)
supports. In-repo idiom for the seam: `ExecutionRouter.__init__(mode=None) ->
mode or _current_mode()` at `execution_router.py:268-269` (docstring cites Fowler
"ops toggle"). The smoketest passes a 6-attribute `SimpleNamespace` as settings,
so any new gate MUST use `getattr(self.settings, ..., default)`.

**`kill_switch_test.py` does NOT test the kill switch.** Its scenarios
(:72-129) drive `signals_server`'s `drawdown_circuit_breaker` (:950), whose peak
is **in-memory** (:88-89, resets every restart) — a duplicate, weaker policy that
cannot see a pause or a disarm. `signals_server.py` has ZERO references to
`kill_switch`/`is_paused`/`paused`. That duplicate is why the gap looked guarded.

**Only TWO `paper_trades` producers exist:** `execute_buy:324` and
`execute_sell:510` (both via `_safe_save_trade:1359`). `portfolio_manager.py`
emits `TradeOrder` dataclasses and never writes a trade. So a last-mile gate in
`execute_buy` achieves *complete* BUY mediation. `execute_sell` must stay
un-gated: `check_and_enforce_kill_switch` flattens via `execute_sell` (:1029,
:736, :764), so gating sells would deadlock the pause.

**Counting traps.** `grep -rn ... --include=*.py` FAILS under zsh (`no matches
found`) — quote the glob. `execute_buy` has 4 non-test call sites but **15 test
call sites across 8 files** (19 total), so a rename is a 19-site diff.
`is_paused()` has exactly 2 non-test callers, but `risk_server.py:91,181` is a
THIRD reader via the snapshot dict key `"paused"` — invisible to an
`is_paused()` grep. Step-text line numbers drifted: live lines are
`paper_trader.py:1094`/`:1197` and `autonomous_loop.py:1316`.

**Observability hook to reuse for a refusal:** `self.buy_rejections.append(...)`
(`paper_trader.py:107` init, `:222-226` only producer, consumed
`autonomous_loop.py:1574-1582`). The other four `execute_buy` refusals are
log-only. P1 precedent: `paper_trader.py:1244-1259`, `source="kill_switch"`.

**Literature verdict (see [[research-gate-discipline]] for gate mechanics):**
CWE-424's parent is CWE-638 "Not Using Complete Mediation"; CWE-638 prescribes
*"a single interface that performs the access checks"*, CWE-288 *"funnel all
access through a single choke point"*, arXiv:2603.10092 *"non-bypassable
invariants at the last mile"*. No source recommends per-caller checks. The
counter-thread is arXiv:2605.18991 (layered heterogeneous enforcement +
"test modes that disable security become the attack surface", Cursor YOLO mode)
and Knight Capital's $460M dead-but-reachable path.
