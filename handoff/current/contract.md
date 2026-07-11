# Contract — Step 69.1 (P0 book-safety: FX, kill-switch, pkill, locks)

- **Phase / step**: phase-69 → 69.1
- **Date**: 2026-07-11
- **Type**: MONEY-PATH book-safety. All fixes fail-safe (FX / current_nav-guard / pkill-removal / locks ship active as corrections) or DARK-until-token (peak_reset). Thresholds 4/10/8/30 byte-untouched.
- **Boundaries**: $0 metered, paper-only; do-no-harm (fail-safe + ledger-math only; thresholds byte-untouched); peak_reset DARK until `KS-PEAK-RESET: APPROVED`; money-path edits byte-coordinated with phase-68 (68.5 not in-flight; the execute_sell:392 edit is a distinct site from 68.5's fill-price gate).

## Research-gate summary

Brief: `handoff/current/research_brief_69.1.md` — **gate_passed: true**, 5 external sources read in full
(OWASP Command Injection + CWE-78 → pkill removal; Fowler CircuitBreaker → operator-resettable+logged
peak_reset; Python contextlib acquire-then-guard → lock fixes; Modern Treasury → FX never-1.0) + 69.0's 8.
All 6 internal targets re-verified vs the 69.0 design §1-2 (internal map by the researcher before the 9th
subagent stall; external floor by Main). Full fix design in `design_audit_burndown_69.md` §1-2.

## Hypothesis

The three ways the engine self-destructs (FX=1.0 phantom proceeds, unrecoverable kill-switch, Slack pkill)
plus the lock strands can be corrected with fail-safe / DARK-gated changes — never touching the 4/10/8/30
thresholds — each with a red→green reproduction test, so the live book can no longer book non-USD proceeds
at 1.0, brick itself post-flatten, or be SIGKILLed from Slack.

## Immutable success criteria (verbatim from `.claude/masterplan.json` phase-69 → 69.1)

1. FX correctness (red->green): a test reproduces a KR/EU SELL under a monkeypatched dual-FX outage crediting local-notional-as-USD at 1.0 (RED), and after the fix fx_rates serves the last-known fallback chain; the non-USD exit is BLOCKED (never credited at 1.0) only when NO rate has ever been stored, mirroring execute_buy. paper_trader.py:392 + fx_rates.py:93.
2. Kill-switch no-data guard (red->green): a current_nav<=0 (BQ-timeout `or 0.0`) input no longer renders a phantom 100% daily+trailing breach. kill_switch.py:246. The audited restart-replayable peak_reset is implemented but DARK: no peak reset fires until KS-PEAK-RESET:APPROVED is recorded (test asserts dark-by-default); the 4%/10%/8%/30% thresholds are byte-untouched.
3. Op-safety: the 'clear queue' pkill -9 -f python sink is removed; a grep + a test prove the #ford-approvals handler can no longer reach any process-kill sink. commands.py:295.
4. Lock safety (red->green): a FAILED acquire no longer unlinks the live pidfile (cycle_lock.py:144); the unguarded init is wrapped so a startup exception cannot strand the flock forever (autonomous_loop.py:167).
5. Do-no-harm + coordination: risk-cap / stop / kill-switch thresholds byte-untouched over the step's commit range (git diff evidence); all changes are fail-safe additions + ledger-math only; money-path edits are byte-coordinated with any in-flight phase-68 fill work (no conflict with 68.5's fill-price sanity gate). Fresh Q/A PASS with the 67.1 gates.

## Plan (GENERATE)

1. **FX** (`fx_rates.py`): new `_last_known_usd_value(ccy)` (DIRECT historical_fx_rates read, no `_usd_value_asof` recursion); `_usd_value_live` serves it on dual yf+FRED failure before returning None. `paper_trader.execute_sell:392`: replace `_l2u = 1.0` with credit-at-last-known-else-BLOCK+PAGE (never 1.0).
2. **Kill-switch** (`kill_switch.py`): `current_nav<=0`→null-breach guard in `evaluate_breach` (fail-safe, active); new `peak_reset` audit event + `_load_from_audit` replay branch + `reset_peak` gated on `kill_switch_peak_reset_enabled` (default False = DARK), wired into `resume()`; thresholds byte-untouched.
3. **Op-safety** (`commands.py:295`): remove the `subprocess.run(["pkill",...])` + `import subprocess`; "clear queue" purges the DB ticket queue only.
4. **Locks**: `cycle_lock.py` — `acquired` flag guarding the finally (no unlink/release on FAILED acquire); `autonomous_loop.py:167` — release lock + reset `_running` if post-acquire init raises.
5. Tests `backend/tests/test_book_safety_69.py`: FX red→green (dual-outage KR sell → last-known, not 1.0; block when no rate ever); current_nav<=0 → no phantom breach; peak_reset dark-by-default; pkill sink unreachable (grep + test); cycle_lock failed-acquire keeps the live pidfile; guarded-init releases on failure. Then experiment_results.md, a git-diff proving thresholds untouched, Workflow Q/A.

## References
- `handoff/current/research_brief_69.1.md` (5 sources + 6 re-verified targets) + `research_brief_69.0.md`.
- `handoff/current/design_audit_burndown_69.md` §1 (FX chain) + §2 (peak_reset state machine).
- OWASP/CWE-78 (command injection), Fowler CircuitBreaker, Python contextlib, Modern Treasury.
