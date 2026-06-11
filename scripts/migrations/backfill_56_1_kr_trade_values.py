"""phase-56.1 backfill (55.1 finding F-2): restate the 7 KR trade-ledger rows.

OPERATOR-GATED. DRY-RUN BY DEFAULT -- nothing is written without --execute,
and --execute should only be run after the operator's explicit approval
(goal-post-away-review constraint 4: historical-row backfill is operator-gated).

WHAT / WHEN / WHY (GIPS-style disclosure, persisted here as the audit trail):
- WHAT: `financial_reports.paper_trades.total_value` on 7 rows (4 BUY + 3 SELL)
  and `transaction_cost` on the 3 SELL rows were persisted in LOCAL currency
  (KRW magnitudes, ~1,500x USD) by the pre-56.1 code (paper_trader.py:265 BUY,
  :413-414 SELL; root-caused in 55.1 sec 2.1, fixed in phase-56.1 on
  2026-06-10). This script restates those fields to the USD values re-derived
  as quantity x price x fx_asof(trade date) from `historical_fx_rates`
  (55.1 sec 2.1 / live_check_55.1 B3 -- the same derivation that closed the
  daily cash-ledger reconciliation to $0.01).
- WHEN: corrupted rows span 2026-06-01T19:33Z .. 2026-06-10T18:39Z (the full
  KR trading history; no EU trades exist). The 06-10 pair landed because the
  running backend predated the fix (restart pending in the operator window).
- WHY: ledger consumers (turnover in perf_metrics, fee/TCA reports, the trades
  UI) assume USD; the mixed-currency rows inflate turnover ~1,500x for KR.
- MATERIALITY (GIPS error-correction framing, 55.1 sec 2.2): IMMATERIAL to
  composite returns (NAV/cash/positions/round-trips were always correct) but
  MATERIAL to any consumer of the trade-ledger value/fee fields -> tier-3/4:
  correct WITH disclosure. This docstring + the harness_log entry + the
  56.1 live_check are the disclosure records.
- IF THE OPERATOR DECLINES: rows stay flagged-not-fixed; the caveat lives in
  frontend/src/components/paper-trading/trades-columns.tsx (header comment)
  and handoff/current/live_check_56.1.md. Do NOT delete this script.

Idempotent: each UPDATE pins the trade_id AND the exact corrupted value, so a
second --execute run matches zero rows (state asserted before/after; safe to
re-run). Uses explicit per-row literals (no recomputation drift).

Run:
    python scripts/migrations/backfill_56_1_kr_trade_values.py            # dry-run (default)
    python scripts/migrations/backfill_56_1_kr_trade_values.py --execute  # OPERATOR-APPROVED ONLY
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.config.settings import get_settings

# Re-derived USD values (55.1 sec 2.1; fx = historical_fx_rates KRWUSD as-of).
# expect_tv / expect_fee pin the CURRENT corrupted values so the UPDATE is a
# no-op if the row was already restated (idempotency) or diverges (safety).
ROWS = [
    # trade_id, ticker, action, fx_asof, expect_tv(local), new_tv_usd, expect_fee, new_fee_usd
    ("6019c11b-46e3-4160-8446-4ffa3518ab78", "000660.KS", "BUY",  0.0006608949, 738196.09,  487.87, None,    None),
    ("d9260ae7-22e9-40af-acae-915361e5a4e9", "000660.KS", "SELL", 0.0006593131, 737259.28,  486.08, 737.26,  0.49),
    ("bf699ae3-abf9-4dc6-8c3f-5c4a5b5ddffb", "000660.KS", "BUY",  0.0006524562, 752280.44,  490.83, None,    None),
    ("f3f5c24c-a821-444d-b981-35d0207f857d", "005930.KS", "BUY",  0.0006524562, 1128428.32, 736.25, None,    None),
    ("c21925a7-8714-4a4d-99b9-2106cc554138", "000660.KS", "SELL", 0.0006410298, 677641.41,  434.39, 677.64,  0.43),
    ("1cc6ed96-c4a0-4f63-8b92-5b501202c3c8", "005930.KS", "SELL", 0.0006410298, 1056195.94, 677.05, 1056.20, 0.68),
    ("a72a164e-3a21-4b20-8c0b-7eb8eb81ecb5", "066570.KS", "BUY",  0.0006546302, 364175.06,  238.40, None,    None),
    # 2026-06-10 18:39Z cycle: the backend process was still running the
    # pre-56.1 code (fix shipped to main 2026-06-10 ~17:00Z; restart is an
    # operator-window action), so two more corrupted rows landed. fx as-of
    # 2026-06-10 = 0.0006578515 (historical_fx_rates KRWUSD).
    ("dd23d746-32b0-4967-8e48-c4caa56141e6", "066570.KS", "SELL", 0.0006578515, 328932.35,  216.39, 328.93,  0.22),
    ("ab054088-918f-4f46-a8a4-c30b3fb5cb31", "000660.KS", "BUY",  0.0006578515, 724479.65,  476.60, None,    None),
]
# NOTE: BUY transaction_cost was ALREADY USD pre-fix (computed on amount_usd) -- untouched.
# Row count: 7 away-week rows (2026-06-01..06-09) + 2 from the 2026-06-10 pre-restart
# cycle = 9. Any KR trade in a FURTHER pre-restart cycle must be appended here before
# --execute (verify with: SELECT ... WHERE created_at >= '2026-06-10' AND ENDS_WITH(ticker,'.KS')).


def build_statements(project: str, dataset: str) -> list[str]:
    table = f"`{project}.{dataset}.paper_trades`"
    stmts = []
    for trade_id, _tk, _act, _fx, old_tv, new_tv, old_fee, new_fee in ROWS:
        sets = [f"total_value = {new_tv}"]
        wheres = [f"trade_id = '{trade_id}'", f"ABS(total_value - {old_tv}) < 0.02"]
        if old_fee is not None:
            sets.append(f"transaction_cost = {new_fee}")
            wheres.append(f"ABS(transaction_cost - {old_fee}) < 0.02")
        stmts.append(
            f"UPDATE {table}\nSET {', '.join(sets)}\nWHERE {' AND '.join(wheres)};"
        )
    return stmts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--execute", action="store_true",
                    help="apply the UPDATEs (OPERATOR-APPROVED runs only; default dry-run)")
    args = ap.parse_args()

    settings = get_settings()
    project = settings.gcp_project_id
    dataset = settings.bq_dataset_reports  # paper tables live in financial_reports
    stmts = build_statements(project, dataset)

    print(f"-- backfill_56_1_kr_trade_values: {len(stmts)} UPDATEs against "
          f"{project}.{dataset}.paper_trades ({'EXECUTE' if args.execute else 'DRY-RUN'})\n")
    for s in stmts:
        print(s + "\n")

    if not args.execute:
        print("-- dry-run only. Re-run with --execute AFTER operator approval.")
        return 0

    from google.cloud import bigquery  # deferred: dry-run needs no GCP dep
    client = bigquery.Client(project=project)
    total = 0
    for s in stmts:
        job = client.query(s)
        job.result(timeout=30)
        n = job.num_dml_affected_rows or 0
        total += n
        print(f"-- affected {n} row(s)")
    print(f"-- DONE: {total} row-updates applied (expected {len(ROWS)} on first run, 0 on re-run).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
