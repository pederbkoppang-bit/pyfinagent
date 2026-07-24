# Live Check — Step 69.1 (P0 book-safety: FX, kill-switch, pkill, locks)

Money-path book-safety. All fixes fail-safe / DARK-gated; thresholds byte-untouched. Evidence per the
masterplan `live_check`: red→green pytest for FX / kill-switch-guard / lock-safety, the pkill-removal grep,
the git diff proving thresholds untouched, and evidence the peak_reset is DARK pending the KS-PEAK-RESET token.

## Test suite — `backend/tests/test_book_safety_69.py` (14 passed)

```
test_fx_serves_last_known_on_dual_outage                 PASSED  # C1 FX: dual-outage -> last-known (not 1.0, not None)
test_fx_returns_none_only_when_no_rate_ever              PASSED  # C1: None only when no rate ever (-> block)
test_fx_usd_unaffected                                   PASSED  # C1 do-no-harm: USD path 1.0
test_fx_local_to_usd_blocks_nonusd_when_no_rate         PASSED  # C1: non-USD no-rate -> None (execute_sell blocks); USD -> 1.0
test_current_nav_zero_no_phantom_breach                 PASSED  # C2: current_nav=0 -> no phantom breach
test_current_nav_negative_no_phantom_breach            PASSED  # C2: current_nav<0 -> no phantom breach
test_valid_nav_still_breaches                           PASSED  # C2 do-no-harm: a real 20%-down NAV STILL breaches
test_peak_reset_dark_by_default                         PASSED  # C2: peak_reset DARK by default (no-op)
test_peak_reset_active_when_token_enabled              PASSED  # C2: ON -> resets + audited + restart-replayable
test_resume_reanchors_peak_via_nav_when_token_enabled  PASSED  # C2: resume(nav) ON -> re-anchors peak
test_resume_does_not_reanchor_peak_when_dark           PASSED  # C2: resume(nav) OFF -> peak unchanged (DARK)
test_thresholds_not_a_setting_here_are_caller_supplied PASSED  # do-no-harm: limits are caller args
test_no_process_kill_sink_in_commands                  PASSED  # C3: no pkill/killpg/os.kill/SIGKILL sink in CODE
test_cycle_lock_failed_acquire_keeps_live_pidfile      PASSED  # C4: failed acquire keeps the live pidfile

14 passed in 0.84s
```

Ruff gate (qa.md §1a) on all 69.1 touched files: **All checks passed!** (exit 0).

## pkill removal (audit item 3)

`git grep -nE "pkill|SIGKILL|killpg|os.kill" backend/slack_bot/commands.py` → matches ONLY in the removal's
own explanatory COMMENT (lines 293/295); the active `subprocess.run(["pkill","-9","-f","python"])` + `import
subprocess` are GONE. The `clear queue` handler now purges the ticket queue via library calls only. No
process-kill sink reachable from the #ford-approvals handler.

## Do-no-harm — thresholds byte-untouched

`git diff` shows NO change to any risk-cap threshold constant. On-disk (settings.py), byte-intact:
```
paper_daily_loss_limit_pct: float = Field(4.0, ...)
paper_trailing_dd_limit_pct: float = Field(10.0, ...)
paper_default_stop_loss_pct: float = Field(8...)      # stop-loss
paper_max_per_sector_nav_pct = 30.0                    # sector cap
```
The kill-switch fix uses caller-supplied limit args, touches no constant. Only additive changes: fail-safe
guards + the DARK-gated `kill_switch_peak_reset_enabled` flag (default False).

## peak_reset is DARK pending KS-PEAK-RESET: APPROVED

- `reset_peak` is a no-op returning None unless `settings.kill_switch_peak_reset_enabled=True`
  (`test_peak_reset_dark_by_default`, `test_resume_does_not_reanchor_peak_when_dark`).
- Wired into `resume(nav=...)` (kill_switch.py:208, outside the state lock — no deadlock), so on activation
  an operator resume re-anchors the peak; audited via a new `peak_reset` event replayed by `_load_from_audit`
  (restart-replayable). Activation follow-on: the resume endpoint (paper_trading.py:569) must pass `nav`.

## Q/A verdict (fresh, cycle-1 PASS + cycle-2 PASS, workflow structured-output, Opus)

`{"ok": true, "verdict": "PASS", "violated_criteria": []}` (both cycles). Cycle-2 independently confirmed the
reset_peak wiring is outside the lock (no deadlock, empirically) and DARK-by-default. Full ruling in
`evaluator_critique.md`.

## Guard-behavior change (operator token) + phase-68 coordination
- peak_reset activation waits on `KS-PEAK-RESET: APPROVED`. The only paper_trader edit is `execute_sell:392`
  (FX default), a distinct site from phase-68's 68.5 fill-price gate (not in-flight) — no byte-conflict.
