# Sprint Contract — phase-10.6 (Monthly Champion/Challenger Sortino gate, HITL)

**Step id:** 10.6 **Date:** 2026-04-20 **Tier:** complex **Harness-required:** true

## Why

`sprint_calendar.yaml` monthly_anchor mandates a monthly champion/challenger evaluation on the last trading Friday, HITL-gated, min 20 challenger days. This step ships the gate logic + HITL state machine.

## Research-gate summary

Fresh researcher (complex tier): `handoff/current/phase-10.6-research-brief.md` — 8 sources in full, 18 URLs, three-variant queries, recency scan, gate_passed=true.

Key grounding:
- **NYSE calendar:** `exchange_calendars` already wired at `backend/backtest/markets.py:12`; fall-through when uninstalled (local-only deployment may have it)
- **Thresholds:** `sortino_delta >= 0.3`, `pbo < 0.2`, `dd_ratio <= 1.2` are project-calibrated (not canonical literature); documented as such
- **HITL 48h expiry:** state machine `pending → approved / rejected / expired` persisted in `handoff/logs/monthly_approval_state.json` (no BQ table yet — phase-10.7 carry-forward)
- **Slack:** extend existing `reaction_added` handler in `commands.py:275-303`; do NOT register a duplicate
- **Ledger:** monthly gate runs on the same `week_iso` as Friday; do NOT clobber Friday's notes — use the separate JSON state file
- **Sortino frequency:** daily returns with `periods_per_year=252` (20 monthly points would be statistically meaningless)
- **`actual_replacement` hard-coded False:** paper-only promotion until real-capital wiring is explicitly added (SR 11-7 compliance)

## Immutable success criteria (masterplan-verbatim)

Test command: `python scripts/harness/phase10_monthly_sortino_test.py`

1. `fires_on_last_trading_friday_of_month` — `is_last_trading_friday(date)` helper returns True for the last Friday NYSE session of each month, False otherwise
2. `reuses_friday_slot_zero_new_slots` — monthly gate does NOT call `weekly_ledger.append_row` (would consume a slot); state lives in `handoff/logs/monthly_approval_state.json`
3. `requires_sortino_delta_ge_0_3` — gate rejects when `sortino(challenger) - sortino(champion) < 0.3`
4. `requires_pbo_lt_0_2` — gate rejects when `challenger_pbo >= 0.2`
5. `requires_dd_ratio_le_1_2` — gate rejects when `challenger_max_dd / champion_max_dd > 1.2`
6. `peder_slack_approval_with_48h_expiry` — approval state: `created_at + 48h = expires_at`; state-machine transitions pending→approved/rejected/expired; stale state becomes `expired` on next call
7. `no_auto_replacement_of_real_capital_champion` — returns `actual_replacement=False` unconditionally; hardcoded, not a configurable flag

## Plan

1. Create `backend/autoresearch/monthly_champion_challenger.py`:
   - Public `run_monthly_sortino_gate(eval_date, *, champion_returns, challenger_returns, champion_max_dd, challenger_max_dd, challenger_pbo, challenger_min_days=20, sortino_delta_threshold=0.3, pbo_threshold=0.2, dd_ratio_threshold=1.2, slack_fn=None, state_path=None, now=None) -> dict`
   - Returns `{fired, gate_pass, approval_pending, approved, expired, reason, actual_replacement, sortino_delta, dd_ratio, pbo}`
   - `actual_replacement` always `False`
   - `is_last_trading_friday(date)` helper using `exchange_calendars.XNYS` when available; pure-Python fallback otherwise
   - Date check: if not last trading Friday → returns `{fired: False, reason: "not_last_trading_friday"}`
   - Quality gates (in order): `len(challenger_returns) >= min_days` → sortino delta → pbo → dd ratio — first failure short-circuits
   - If all gates pass: load state file; if prior state for this month is `pending` and not expired → return pending state; if expired → transition to `expired`; else create new `pending` state with `expires_at = now + 48h`; call `slack_fn(message, metadata)` if provided
   - Approval state schema: `{"month": "2026-04", "created_at_iso": "...", "expires_at_iso": "...", "status": "pending|approved|rejected|expired", "sortino_delta": ..., "challenger_id": "..."}`
   - `actual_replacement` remains `False` even on `approved` status
   - Injectable `now` for deterministic tests
   - ASCII-only logger messages
2. Create `scripts/harness/phase10_monthly_sortino_test.py`:
   - 7 cases matching the 7 success_criteria verbatim
   - `tempfile.TemporaryDirectory()` per case for the state file
   - Uses injectable `now` and `slack_fn=stub` to avoid real Slack / wall clock
3. Create `tests/autoresearch/test_monthly_champion_challenger.py` with ≥8 pytest cases (7 CLI + edge: challenger with < 20 days is rejected).
4. Verify: ast + immutable CLI + pytest new file + neighbor suites (autoresearch + slack_bot + metrics).
5. Spawn fresh Q/A. Cycle-2 flow if CONDITIONAL/FAIL.
6. Log, flip masterplan, close task.

## References

- `handoff/current/phase-10.6-research-brief.md` (8 in full, 18 URLs, gate_passed=true)
- `backend/backtest/markets.py:12,81-107` (exchange_calendars already wired)
- `backend/autoresearch/sprint_calendar.yaml` (monthly_anchor schema)
- `backend/autoresearch/friday_promotion.py` (10.4 sibling; do not clobber its notes)
- `backend/metrics/sortino.py` (10.5 canonical Sortino)
- `backend/autoresearch/gate.py` (8.5.5 PBO threshold source)
- `backend/slack_bot/commands.py:275-303` (existing reaction_added handler — carry-forward wiring point)

## Carry-forwards (out of scope — explicit)

- **Slack posting of the approval message + reaction-based approval**: `slack_fn=None` in this phase; wire to real Slack in a phase-10.6.1 follow-up (needs Bolt app-home integration)
- **BQ `pyfinagent_pms.champion_state` table**: state lives in local JSON for now; move to BQ in phase-10.8 (slot accounting)
- **Real-capital auto-promotion**: hardcoded False; requires SR 11-7 compliance review before ever being enabled
- **Reject reason propagation to weekly ledger `notes`**: deferred; tangles with Friday's notes
