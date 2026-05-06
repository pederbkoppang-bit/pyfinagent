---
step: phase-23.2.22
title: Test-audit-log isolation + position-cap diagnostic logging
cycle_date: 2026-05-07
harness_required: true
verification: 'source .venv/bin/activate && PYTHONPATH=. python tests/verify_phase_23_2_22.py'
research_brief: handoff/current/phase-23.2.22-external-research.md (also see phase-23.2.22-internal-codebase-audit.md)
---

# Contract — phase-23.2.22

## Hypothesis

User screenshot: OpsStatusBar Cycle segment shows `paper_trades: red`,
heartbeat green, paper_snapshots green. User asked (A) why trading
stopped, (B) is the red dot supposed to recover when they resume
trading.

**Answer to B (no code change required):** YES — the band logic in
`backend/services/cycle_health.py:_band` is purely a function of
`age_sec / cycle_interval_sec`. As soon as a single new row lands in
`paper_trades` with a recent `created_at`, the ratio drops to ~0 and
the band flips back to green on the next freshness poll. The dot is
a symptom indicator, not a latch. This will be re-stated in
experiment_results.md and then the user-facing reply.

**Answer to A (real bugs):** Forensic — last paper_trade was
2026-05-01T18:02:39 (FIX BUY, 6 days ago). Cycles on 05-05 and 05-06
ran successfully (heartbeat=end, all 8 steps logged), but Step 7
showed `Executing 0 trades` from 17 candidates (3 new + 14 re-evals).
Researcher (ab745941eaa332650) found two distinct issues:

**Issue A — position-cap blocks all buys without telling anyone.**
`paper_max_positions=10`; live `paper_positions` count = 14;
`backend/services/portfolio_manager.py:204` `break`s the buy loop
immediately on every cycle because `14 >= 10`. SELL signals require
`recommendation in _SELL_RECS` or `signal_downgrade` — with no
catalyst, HOLD recommendations don't free slots. The behavior is
working-as-designed but **silent**: no log line says "blocked by
position cap (14 >= 10)" so future 0-trade cycles look like a bug.

**Issue B — pytest writes real events to production audit log.**
`backend/services/kill_switch.py:21-26` defines `_AUDIT_PATH` as a
module-level constant pointing at production
`handoff/kill_switch_audit.jsonl`. Two test files instantiate
`KillSwitchState()` and call `.pause(...)` without redirecting the
path: `tests/services/test_cycle_failure_alerts.py:139,154` and
`tests/services/test_kill_switch_no_deadlock.py:25,45,66`. Forensic:
`handoff/kill_switch_audit.jsonl` already contains 7 spurious pause
events from the 2026-05-05 20:07:52 pytest run, including a
`drawdown_breach details={"daily_loss_pct":-2.5}` row that NEVER
HAPPENED in production. **Latent restart risk**: the audit log ends
with test pauses + no following resume. Next backend restart would
boot `_paused=True` and `autonomous_loop.py` would short-circuit
forever, with no alert.

## Research-gate summary

Researcher (ab745941eaa332650) returned `gate_passed: true`:
- 7 sources read in full via WebFetch (pytest tmp_path docs, pytest
  monkeypatch docs, pytest fixture autouse pattern, append-only
  audit-log isolation references, Kelly-criterion / position-cap
  literature, recency scan articles)
- 17 URLs collected; 10 in snippet-only
- Recency scan 2024-2026 with 3-variant query discipline
- 10 internal files inspected; concrete fix shape for each issue
- Reference: `tests/services/test_sod_daily_roll.py:31-35` already
  uses the canonical `monkeypatch.setattr(ks, "_AUDIT_PATH", tmp)`
  pattern — the offending tests just need the same fixture.

## Immutable success criteria (verbatim — DO NOT EDIT)

1. `tests/services/test_cycle_failure_alerts.py` adds an autouse
   module-scope `tmp_audit` fixture that monkeypatches
   `backend.services.kill_switch._AUDIT_PATH` to `tmp_path /
   "kill_switch_audit.jsonl"` for every test in that file.
2. `tests/services/test_kill_switch_no_deadlock.py` adds the same
   autouse `tmp_audit` fixture.
3. After running the full suite, `handoff/kill_switch_audit.jsonl`
   gets ZERO new rows from those two files. Verifier asserts this
   by snapshotting size before / after a `pytest` run.
4. The 2026-05-05 20:07:52 audit pollution gets a CLEAN-UP marker:
   a `cleanup` event row appended to `handoff/kill_switch_audit.jsonl`
   noting that the prior 7 pause events from that timestamp were
   pytest leakage and should be ignored on boot replay. Plus an
   explicit `resume` event so the next backend restart does NOT
   boot paused.
5. `backend/services/cycle_health.py` is unchanged (the red-dot
   recovery is by design; we are NOT touching the band logic).
6. `backend/services/portfolio_manager.py` adds a single diagnostic
   `logger.info` line at the position-cap break point: `"Position
   cap reached: %d held >= %d max — skipping all BUY candidates"`.
   No behavior change. No threshold change. No setting change.
7. Regression test `tests/services/test_position_cap_logging.py` (new):
   confirms the log line fires with the expected substrings when
   `decide_trades` is invoked with positions ≥ cap.
8. `python tests/verify_phase_23_2_22.py` exits 0.
9. `python -c "import ast; ast.parse(open(P).read())"` passes for
   every modified .py file.

## Plan steps

1. `tests/services/test_cycle_failure_alerts.py`:
   - Add `tmp_audit` autouse fixture at module scope copying the
     pattern from `test_sod_daily_roll.py:31-35`.
2. `tests/services/test_kill_switch_no_deadlock.py`:
   - Same autouse fixture; verify all 4 existing tests still pass
     (they instantiate `KillSwitchState()` 5 times).
3. Append two cleanup rows to production
   `handoff/kill_switch_audit.jsonl`:
   - One `cleanup` event documenting the 2026-05-05 20:07:52 pytest
     leakage so future audits see the disclosure.
   - One `resume` event with `trigger="manual_post_test_cleanup"` so
     boot replay terminates in the unpaused state.
4. `backend/services/portfolio_manager.py:204`:
   - Add `logger.info("Position cap reached: %d held >= %d max -- skipping all BUY candidates", remaining_positions, settings.paper_max_positions)` BEFORE the existing `break`.
5. New `tests/services/test_position_cap_logging.py`:
   - Mock `decide_trades` deps (settings, candidates, current_positions=15
     items, etc) and `caplog.at_level(logging.INFO)`; assert the
     INFO log line fires with both `15` and `10` substrings present.
6. New `tests/verify_phase_23_2_22.py`:
   - AST-parse modified files
   - assert `_AUDIT_PATH` monkeypatch is referenced in both test files
   - assert position-cap log message exists in `portfolio_manager.py`
   - assert audit-log cleanup `resume` event is present
7. Run prior-phase regression suite + new tests; verify
   `handoff/kill_switch_audit.jsonl` did NOT grow during pytest.
8. Append `harness_log.md` AFTER Q/A PASS, BEFORE any masterplan flip.

## Out of scope

- Raising `paper_max_positions` from 10 to 15 (operator decision; not
  a bug fix).
- Time-based forced-sell of positions held > N days (research phase).
- Tightening re-eval sell trigger to include HOLD on winners (research
  phase).
- Changing the band thresholds in `_band` (the red-dot recovery is
  already correct).
- Frontend changes (the red→green explanation goes in the user reply,
  not in code).

## Backwards compatibility

- `tmp_audit` fixture is purely additive in test files; no production
  code changes for the test-isolation fix.
- `logger.info` line is purely additive; no behavior change to
  `decide_trades`.
- Audit-log cleanup `resume` event is consistent with the existing
  schema (just another line in the JSONL); the `cleanup` event is a
  new event type but `_load_from_audit` only acts on `pause`,
  `resume`, `sod_snapshot`, `peak_update` — unknown event types are
  silently ignored.

## References

- Researcher: `handoff/current/phase-23.2.22-external-research.md`,
  `handoff/current/phase-23.2.22-internal-codebase-audit.md`
- `backend/services/kill_switch.py:21-26` (the `_AUDIT_PATH` constant)
- `backend/services/portfolio_manager.py:204` (the position-cap break)
- `tests/services/test_sod_daily_roll.py:31-35` (canonical fixture
  pattern to copy)
- pytest official docs: `tmp_path`, `monkeypatch`, `autouse=True`
