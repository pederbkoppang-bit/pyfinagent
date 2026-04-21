# Contract -- Cycle 77 / phase-4.8 step 4.8.0

Step: 4.8.0 Transaction Cost Analysis (implementation shortfall)

## Hypothesis

Ship the TCA pipeline end-to-end:
- Library that computes implementation shortfall per fill with the
  CORRECT sign convention (positive = cost) per AFML / Perold.
- Arrival price = **previous close**, NOT same as fill price, to
  avoid the degenerate IS=0 case the researcher flagged.
- Per-fill log appended to `handoff/tca_log.jsonl` (matches the
  existing kill_switch_audit.jsonl pattern).
- Weekly aggregator script `scripts/risk/tca_report.py --week last`
  computes median/avg/p95 + liquid-subset filter + breakdowns,
  writes `handoff/tca_last_week.json`, logs WARNING when
  `median_bps_liquid >= 15` (alert).
- Given no live paper trading, the script synthesizes a 7-day batch
  of realistic fills using `backend.services.execution_router` mock
  path + the 20-name liquid S&P list from Cycle 67. The math is the
  real library code; only the input data is seeded, and that fact is
  recorded in the JSON artifact so auditors see the source.

## Scope

Files created:

1. **NEW** `backend/services/tca.py`
   - `compute_is_bps(fill_price, arrival_price, side)` -> float
   - `log_tca_event(fill, arrival_price, meta)` -> dict (appends to
     `handoff/tca_log.jsonl`; idempotent w.r.t. client_order_id)
   - `LIQUID_SYMBOLS` constant reusing the 20 S&P names.

2. **NEW** `scripts/risk/tca_report.py`
   - `--week last` CLI flag -> 7-day window ending "now"
   - Reads `handoff/tca_log.jsonl`; if empty, seeds 7 days of
     deterministic synthetic fills using execution_router mock.
   - Computes: avg_is_bps, median_is_bps, p95_is_bps, trade_count,
     total_notional_usd, `median_bps_liquid`, by_symbol, by_side.
   - Writes `handoff/tca_last_week.json`.
   - Logs WARNING if `median_bps_liquid >= 15`.

3. **NEW** `backend/services/tca_test.py` -- pytest-style unit tests
   for compute_is_bps covering buy/sell sign convention,
   degenerate-zero-arrival, negative drift, bps scaling.

## Immutable success criteria

1. tca_logged_per_fill -- jsonl log contains 1 row per seeded fill.
2. weekly_report_generated -- `handoff/tca_last_week.json` exists
   with all required fields.
3. alert_fires_above_15bps_liquid -- when
   `median_bps_liquid >= 15`, a WARNING is logged AND an
   `alert_triggered: true` field appears in the JSON.

## Verification (immutable)

    python scripts/risk/tca_report.py --week last && \
    python -c "import json; r=json.load(open('handoff/tca_last_week.json')); assert r['median_bps_liquid'] < 15"

## Alert-discriminating rigor check (self-imposed)

A second run with a seeded anomalous batch (median >= 30 bps) must
produce `alert_triggered: true` AND the assertion
`median_bps_liquid < 15` must FAIL. Proves the alert has teeth,
not a constant "false".

## References

- researcher findings: Perold 1988, AFML, Talos, LSEG, CFA III,
  Menkveld ELO chapter, QB, Bloomberg BTCA.
- backend/services/execution_router.py (FillResult shape + mock
  slippage constant).
- scripts/harness/virtual_fund_parity.py (20-name S&P list).
- backend/services/kill_switch.py (jsonl audit precedent).
