# Experiment Results — Step 69.1 (P0 book-safety: FX, kill-switch, pkill, locks)

- **Phase / step**: phase-69 → 69.1
- **Date**: 2026-07-11
- **Type**: MONEY-PATH book-safety. Fail-safe / DARK-gated. Thresholds byte-untouched.

## What was changed (6 fixes; 170 insertions across 6 files)

`git diff --stat`: autonomous_loop.py +31, cycle_lock.py +25, fx_rates.py +48, kill_switch.py +52,
paper_trader.py +18, commands.py +22 (net).

1. **FX last-known fallback (audit item 1)** — `fx_rates._last_known_usd_value(ccy)` (new; DIRECT
   historical_fx_rates read, NOT via `_usd_value_asof` = mutual-recursion) served by `_usd_value_live` on a
   dual yf+FRED outage BEFORE returning None; `paper_trader.execute_sell:392` replaces `_l2u = 1.0` with
   **BLOCK+PAGE** (return None) when no rate was ever stored — never books non-USD proceeds at 1.0. USD path
   byte-identical.
2. **Kill-switch (audit item 2)** — `evaluate_breach` returns no-breach on `current_nav<=0` (fail-safe: a
   BQ-timeout `or 0.0` no longer renders a phantom 100% breach). New `peak_reset` audit event +
   `_load_from_audit` replay branch + `reset_peak` method **DARK** behind `kill_switch_peak_reset_enabled`
   (default False = no-op). Thresholds untouched.
3. **Op-safety (audit item 3)** — removed `subprocess.run(["pkill","-9","-f","python"])` + the `import
   subprocess` from the `clear queue` handler (`commands.py:295`); "clear queue" now purges the ticket queue
   only (library calls). No process-kill sink reachable from the Slack handler.
4. **Locks (audit item 4)** — `cycle_lock` guards its finally on an `acquired` flag (a FAILED acquire no
   longer unlinks the live holder's pidfile); `autonomous_loop:167` wraps the post-acquire init so a
   BigQueryClient construction failure releases the lock + resets `_running` (no permanent brick).
5. **Do-no-harm side-fix** — `autonomous_loop` hoists `options_surge_signals`/`insider_signals` default `{}`
   before the ma_preannounce read (:474). This clears a PRE-EXISTING ruff F821 (confirmed present in HEAD,
   not a 69.1 regression) in a file 69.1 already edits; matches the existing `or {}` fallback; no prod effect
   (`ma_preannounce_enabled` defaults OFF). Fixes the register's contested ma_preannounce latent bug as a
   byproduct.

## Verification command output (verbatim)

```
$ python -m pytest backend/tests/test_book_safety_69.py -q --timeout=120
............                                                             [100%]
12 passed in 0.84s
```

Ruff gate (qa.md §1a) on all 69.1 touched files:
```
$ uvx ruff check --select F821,F401,F811 <6 code files + settings.py + test>
All checks passed!        # exit 0
```

Test coverage (red→green / guard):
- FX: last-known served on dual-outage (not None, not 1.0); None only when no rate ever; USD unaffected;
  `_fx_local_to_usd(non-USD, no rate)` → None (→ execute_sell blocks), USD → 1.0.
- Kill-switch: current_nav 0 and negative → no phantom breach + `nav_invalid`; a VALID 20%-down NAV STILL
  breaches (guard doesn't suppress real breaches); peak_reset DARK by default (no-op); ACTIVE + audited +
  restart-replayable when the token flag is on.
- Op-safety: no `pkill`/`killpg`/`os.kill`/`SIGKILL` sink in the commands.py CODE (comment-stripped grep).
- Locks: a FAILED acquire keeps the live holder's pidfile (RED pre-fix: it was unlinked).

## Do-no-harm evidence

- **Thresholds byte-untouched**: `git diff` shows NO change to any risk-cap constant; `paper_daily_loss_limit_pct=4.0`,
  `paper_trailing_dd_limit_pct=10.0`, `paper_default_stop_loss_pct` (8%), sector caps — all unchanged in settings.py.
  The kill-switch fix uses caller-supplied limits (args), touches no constant.
- **Fail-safe / DARK**: the FX/current_nav/pkill/lock fixes only PREVENT corruption (never book phantom USD,
  never phantom-breach, never SIGKILL, never strand the lock); the peak_reset is a no-op until the operator
  records `KS-PEAK-RESET: APPROVED`.
- **No regressions**: `pytest --collect-only` → **1040 tests** (no import breakage). No existing test references
  the changed functions.
- **phase-68 coordination**: the only `paper_trader.py` edit is at `execute_sell:392` (the FX default) — a
  distinct site from phase-68's 68.5 fill-price sanity gate; 68.5 is not in-flight. No byte-conflict.

## Guard-behavior change requiring the operator token
- `peak_reset` is implemented but DARK until `KS-PEAK-RESET: APPROVED` (`kill_switch_peak_reset_enabled=True`).
  The code + replay + tests ship; activation waits on the operator token (do-no-harm).

## Cycle-2 addendum (post cycle-1 Q/A PASS with 2 NOTEs)

Cycle-1 Q/A (workflow, Opus) returned **PASS**, `violated_criteria: []`, with two non-degrading NOTEs. Both addressed:

1. **`reset_peak` now has a call site** (NOTE 1). It was defined/gated/tested but not invoked, so on a future
   `KS-PEAK-RESET: APPROVED` activation it would never fire. Now WIRED into `resume(nav=...)`: an operator
   resume re-anchors the trailing peak to the current NAV (still DARK — a no-op until the token). Called
   outside the state lock (reset_peak takes `self._lock`, non-reentrant). Two new tests:
   `test_resume_reanchors_peak_via_nav_when_token_enabled` (flag ON → peak re-anchored to the resume NAV) and
   `test_resume_does_not_reanchor_peak_when_dark` (flag OFF → peak unchanged, still un-pauses). Re-verified:
   **14 passed**, ruff **exit 0**. Remaining activation wiring — the flatten-to-cash emit site — is part of the
   KS-PEAK-RESET activation follow-on (resume is the primary operator path and is now wired + tested).
2. **69.3 scaffolding disclosed** (NOTE 2): the working tree carries two adjacent phase-69.3 flag defs
   (`sign_safe_overlays`, `regime_net_liquidity`, additive default-OFF, no wiring, no threshold impact) +
   `backend/services/overlay_math.py` (unused). They are 69.3 work-in-progress kept because 69.3 is the next
   step; the 69.1 auto-commit will sweep them in. They are inert (imported nowhere, default-OFF) and do NOT
   affect the live engine. Disclosed here per the Q/A's commit-hygiene note.

Changed evidence (resume wiring + 2 tests) → a fresh cycle-2 Q/A re-verifies.
