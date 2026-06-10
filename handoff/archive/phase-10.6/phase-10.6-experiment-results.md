# Experiment Results — phase-10.6 (Monthly Champion/Challenger Sortino gate, HITL)

**Step:** 10.6 **Date:** 2026-04-20

## What was done

1. Fresh researcher (complex): 8 in full, 18 URLs, recency 2026, gate_passed=true. Brief at `handoff/current/phase-10.6-research-brief.md`. Key grounding: `exchange_calendars` already wired in codebase; thresholds are project-calibrated; HITL 48h is industry standard (Orkes + Cloudflare); state lives in local JSON (no BQ table until phase-10.8).
2. Contract authored at `handoff/current/phase-10.6-contract.md`.
3. Created `backend/autoresearch/monthly_champion_challenger.py` (~230 lines):
   - Public `run_monthly_sortino_gate(eval_date, *, champion_returns, challenger_returns, champion_max_dd, challenger_max_dd, challenger_pbo, challenger_id, challenger_min_days=20, sortino_delta_threshold=0.3, pbo_threshold=0.2, dd_ratio_threshold=1.2, periods_per_year=252, slack_fn=None, state_path=None, now=None) -> dict`
   - `record_approval(month_key, *, status, state_path, now) -> dict` — transitions pending→approved/rejected; auto-expires if past window
   - `is_last_trading_friday(d)` — NYSE XNYS via `exchange_calendars` with pure-Python fallback
   - **Hardcoded invariant:** `actual_replacement = False` on every return path
   - Quality-gate order (short-circuits): min days → sortino delta → pbo → dd_ratio
   - Fail-closed: NaN Sortino → reject; zero champion dd → reject
   - State file JSON at `handoff/logs/monthly_approval_state.json` (or injectable); keyed by `"YYYY-MM"`
   - Expiry: `created_at + 48h = expires_at`; next call past expires_at auto-transitions `pending → expired`
   - Injectable `slack_fn` fail-open; injectable `now` for tests
4. Created `scripts/harness/phase10_monthly_sortino_test.py` (7 cases matching masterplan success_criteria verbatim).
5. Created `tests/autoresearch/test_monthly_champion_challenger.py` (12 pytest cases).

## Verification (verbatim)

```
$ python -c "import ast; [ast.parse(open(f).read()) for f in ['backend/autoresearch/monthly_champion_challenger.py','scripts/harness/phase10_monthly_sortino_test.py']]; print('AST OK')"
AST OK

$ python scripts/harness/phase10_monthly_sortino_test.py
[PASS] fires_on_last_trading_friday_of_month  (helper: True/True, fire=True, nofire=False)
[PASS] reuses_friday_slot_zero_new_slots  (ledger_rows: before=1, after=1)
[PASS] requires_sortino_delta_ge_0_3  (gate_pass=False, reason=sortino_delta<0.3)
[PASS] requires_pbo_lt_0_2  (gate_pass=False, reason=pbo>=0.2)
[PASS] requires_dd_ratio_le_1_2  (gate_pass=False, reason=dd_ratio>1.2)
[PASS] peder_slack_approval_with_48h_expiry  (r1.pending=True, slack_calls=2, r_mid.pending=True, r_exp.expired=True)
[PASS] no_auto_replacement_of_real_capital_champion  (fire.actual=False, after_approve.actual=False, after_approve.approved=True)

ALL PASS  (7/7)
(exit 0)

$ pytest tests/autoresearch/test_monthly_champion_challenger.py -q
............                                                             [100%]
12 passed in 0.28s

$ pytest tests/autoresearch/ tests/slack_bot/ backend/metrics/ -q
........................................................................ [ 81%]
................                                                         [100%]
88 passed in 1.40s
```

## Success criteria (masterplan, immutable)

| # | Criterion | Status |
|---|---|---|
| 1 | `fires_on_last_trading_friday_of_month` | PASS — xcals sessions.weekday==4 last of month; mid-month short-circuits with `reason="not_last_trading_friday"` |
| 2 | `reuses_friday_slot_zero_new_slots` | PASS — ledger rows unchanged before vs after monthly gate fire |
| 3 | `requires_sortino_delta_ge_0_3` | PASS — identical-returns case rejects with `reason="sortino_delta<0.3"` |
| 4 | `requires_pbo_lt_0_2` | PASS — `pbo=0.25` rejects with `reason="pbo>=0.2"` |
| 5 | `requires_dd_ratio_le_1_2` | PASS — challenger_dd=0.20 vs champion_dd=0.10 (ratio 2.0) rejects with `reason="dd_ratio>1.2"` |
| 6 | `peder_slack_approval_with_48h_expiry` | PASS — pending at t0, still pending at t+24h, expires at t+48h; slack_fn called on initial fire |
| 7 | `no_auto_replacement_of_real_capital_champion` | PASS — `actual_replacement=False` even after approval |

## Key design decisions

- **State file, not ledger**: monthly approval lives in `handoff/logs/monthly_approval_state.json` — does NOT clobber Friday's `fri_promoted_ids` / `notes` column. Matches the "reuses Friday slot, zero new slots" criterion.
- **Date injection**: `now` parameter lets tests control the expiry clock deterministically.
- **Slack non-blocking**: `slack_fn=None` is valid production default (wiring deferred to phase-10.6.1); tests inject stub for observability.
- **Month key canonicalization**: `"YYYY-MM"` avoids ISO-week confusion with weekly ledger.

## Carry-forwards (out of scope)

- Slack posting + reaction-based approval wiring to existing `commands.py:275-303` handler (must EXTEND not register duplicate) — phase-10.6.1
- Move state from local JSON to BQ `pyfinagent_pms.champion_state` — phase-10.8 slot accounting
- Real-capital promotion layer — requires SR 11-7 compliance review
- "Rejected" reason propagation to weekly ledger notes — deferred (tangles with Friday's notes)
