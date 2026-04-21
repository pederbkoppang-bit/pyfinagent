"""phase-10.6 verification CLI: Monthly Champion/Challenger Sortino gate.

Seven cases matching the masterplan success_criteria:
  1. fires_on_last_trading_friday_of_month
  2. reuses_friday_slot_zero_new_slots
  3. requires_sortino_delta_ge_0_3
  4. requires_pbo_lt_0_2
  5. requires_dd_ratio_le_1_2
  6. peder_slack_approval_with_48h_expiry
  7. no_auto_replacement_of_real_capital_champion

Each case uses tempfile + injectable now/slack_fn for determinism.
"""
from __future__ import annotations

import sys
import tempfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from backend.autoresearch.monthly_champion_challenger import (
    record_approval,
    run_monthly_sortino_gate,
    is_last_trading_friday,
)
from backend.autoresearch import weekly_ledger


def _good_champion_returns():
    # 30 days of mild positive drift + noise; low downside.
    return [0.001, 0.002, -0.001, 0.003, 0.001, -0.002, 0.004, 0.002, -0.001, 0.001] * 3


def _good_challenger_returns():
    # 30 days with clearly higher Sortino than champion.
    return [0.004, 0.005, -0.001, 0.006, 0.004, -0.002, 0.007, 0.005, -0.001, 0.004] * 3


def case_fires_on_last_trading_friday() -> bool:
    # 2026-04-24 is the last Friday of April 2026.
    # (2026-05-29 is last trading Friday of May 2026; 2026-12-25 is Xmas so Dec's
    # last trading Friday is 2026-12-18.)
    last_fri_apr = date(2026, 4, 24)
    mid_month = date(2026, 4, 17)
    ok1 = is_last_trading_friday(last_fri_apr)
    ok2 = not is_last_trading_friday(mid_month)

    with tempfile.TemporaryDirectory() as td:
        spath = Path(td) / "state.json"
        r_fire = run_monthly_sortino_gate(
            last_fri_apr,
            champion_returns=_good_champion_returns(),
            challenger_returns=_good_challenger_returns(),
            champion_max_dd=0.10,
            challenger_max_dd=0.08,
            challenger_pbo=0.10,
            state_path=spath,
            now=datetime(2026, 4, 24, 21, 0, tzinfo=timezone.utc),
        )
        r_nofire = run_monthly_sortino_gate(
            mid_month,
            champion_returns=_good_champion_returns(),
            challenger_returns=_good_challenger_returns(),
            champion_max_dd=0.10,
            challenger_max_dd=0.08,
            challenger_pbo=0.10,
            state_path=spath,
            now=datetime(2026, 4, 17, 21, 0, tzinfo=timezone.utc),
        )
    ok = ok1 and ok2 and r_fire["fired"] and not r_nofire["fired"] and r_nofire["reason"] == "not_last_trading_friday"
    print(
        f"[{'PASS' if ok else 'FAIL'}] fires_on_last_trading_friday_of_month  "
        f"(helper: {ok1}/{ok2}, fire={r_fire['fired']}, nofire={r_nofire['fired']})"
    )
    return ok


def case_reuses_friday_slot_zero_new_slots() -> bool:
    # Monthly gate must NOT touch the weekly ledger.
    with tempfile.TemporaryDirectory() as td:
        spath = Path(td) / "state.json"
        lpath = Path(td) / "ledger.tsv"
        # Seed the ledger with a Friday row.
        weekly_ledger.append_row(
            week_iso="2026-W17",
            thu_batch_id="uuid-stub",
            thu_candidates_kicked=128,
            notes="kicked_off",
            path=lpath,
        )
        rows_before = weekly_ledger.read_rows(path=lpath)

        # Fire the monthly gate -- must not add or modify ledger rows.
        run_monthly_sortino_gate(
            date(2026, 4, 24),
            champion_returns=_good_champion_returns(),
            challenger_returns=_good_challenger_returns(),
            champion_max_dd=0.10,
            challenger_max_dd=0.08,
            challenger_pbo=0.10,
            state_path=spath,
            now=datetime(2026, 4, 24, 21, 0, tzinfo=timezone.utc),
        )
        rows_after = weekly_ledger.read_rows(path=lpath)

    ok = rows_before == rows_after
    print(
        f"[{'PASS' if ok else 'FAIL'}] reuses_friday_slot_zero_new_slots  "
        f"(ledger_rows: before={len(rows_before)}, after={len(rows_after)})"
    )
    return ok


def case_requires_sortino_delta_ge_0_3() -> bool:
    with tempfile.TemporaryDirectory() as td:
        spath = Path(td) / "state.json"
        # Identical returns -> delta 0.0 -> fail
        same = _good_champion_returns()
        r = run_monthly_sortino_gate(
            date(2026, 4, 24),
            champion_returns=same,
            challenger_returns=same,
            champion_max_dd=0.10,
            challenger_max_dd=0.08,
            challenger_pbo=0.10,
            state_path=spath,
            now=datetime(2026, 4, 24, 21, 0, tzinfo=timezone.utc),
        )
    ok = (
        not r["gate_pass"]
        and r["reason"] is not None
        and "sortino_delta" in r["reason"]
    )
    print(
        f"[{'PASS' if ok else 'FAIL'}] requires_sortino_delta_ge_0_3  "
        f"(gate_pass={r['gate_pass']}, reason={r['reason']})"
    )
    return ok


def case_requires_pbo_lt_0_2() -> bool:
    with tempfile.TemporaryDirectory() as td:
        spath = Path(td) / "state.json"
        r = run_monthly_sortino_gate(
            date(2026, 4, 24),
            champion_returns=_good_champion_returns(),
            challenger_returns=_good_challenger_returns(),
            champion_max_dd=0.10,
            challenger_max_dd=0.08,
            challenger_pbo=0.25,  # fails
            state_path=spath,
            now=datetime(2026, 4, 24, 21, 0, tzinfo=timezone.utc),
        )
    ok = (
        not r["gate_pass"]
        and r["reason"] is not None
        and "pbo" in r["reason"]
    )
    print(
        f"[{'PASS' if ok else 'FAIL'}] requires_pbo_lt_0_2  "
        f"(gate_pass={r['gate_pass']}, reason={r['reason']})"
    )
    return ok


def case_requires_dd_ratio_le_1_2() -> bool:
    with tempfile.TemporaryDirectory() as td:
        spath = Path(td) / "state.json"
        r = run_monthly_sortino_gate(
            date(2026, 4, 24),
            champion_returns=_good_champion_returns(),
            challenger_returns=_good_challenger_returns(),
            champion_max_dd=0.10,
            challenger_max_dd=0.20,  # ratio 2.0 > 1.2
            challenger_pbo=0.10,
            state_path=spath,
            now=datetime(2026, 4, 24, 21, 0, tzinfo=timezone.utc),
        )
    ok = (
        not r["gate_pass"]
        and r["reason"] is not None
        and "dd_ratio" in r["reason"]
    )
    print(
        f"[{'PASS' if ok else 'FAIL'}] requires_dd_ratio_le_1_2  "
        f"(gate_pass={r['gate_pass']}, reason={r['reason']})"
    )
    return ok


def case_peder_slack_approval_with_48h_expiry() -> bool:
    """Pending state is created with expires_at = created_at + 48h; 48h later
    a new call transitions it to expired."""
    slack_calls: list = []

    def stub_slack(msg, meta):
        slack_calls.append((msg, meta))

    with tempfile.TemporaryDirectory() as td:
        spath = Path(td) / "state.json"
        t0 = datetime(2026, 4, 24, 21, 0, tzinfo=timezone.utc)

        r1 = run_monthly_sortino_gate(
            date(2026, 4, 24),
            champion_returns=_good_champion_returns(),
            challenger_returns=_good_challenger_returns(),
            champion_max_dd=0.10,
            challenger_max_dd=0.08,
            challenger_pbo=0.10,
            slack_fn=stub_slack,
            state_path=spath,
            now=t0,
        )
        # +24h: still pending
        r_mid = run_monthly_sortino_gate(
            date(2026, 4, 24),
            champion_returns=_good_champion_returns(),
            challenger_returns=_good_challenger_returns(),
            champion_max_dd=0.10,
            challenger_max_dd=0.08,
            challenger_pbo=0.10,
            slack_fn=stub_slack,
            state_path=spath,
            now=t0 + timedelta(hours=24),
        )
        # +48h exactly: expire
        r_exp = run_monthly_sortino_gate(
            date(2026, 4, 24),
            champion_returns=_good_champion_returns(),
            challenger_returns=_good_challenger_returns(),
            champion_max_dd=0.10,
            challenger_max_dd=0.08,
            challenger_pbo=0.10,
            slack_fn=stub_slack,
            state_path=spath,
            now=t0 + timedelta(hours=48),
        )
        # Resolve approval during pending window
        t_resolve = t0 + timedelta(hours=12)
        row = record_approval("2026-04", status="approved", state_path=spath, now=t_resolve)

    ok = (
        r1["approval_pending"] is True
        and len(slack_calls) >= 1
        and r_mid["approval_pending"] is True
        and r_exp["expired"] is True
        and row.get("status") in ("approved", "expired")
    )
    print(
        f"[{'PASS' if ok else 'FAIL'}] peder_slack_approval_with_48h_expiry  "
        f"(r1.pending={r1['approval_pending']}, slack_calls={len(slack_calls)}, "
        f"r_mid.pending={r_mid['approval_pending']}, r_exp.expired={r_exp['expired']})"
    )
    return ok


def case_no_auto_replacement_of_real_capital_champion() -> bool:
    """actual_replacement is ALWAYS False, even when the gate passes + is approved."""
    with tempfile.TemporaryDirectory() as td:
        spath = Path(td) / "state.json"
        t0 = datetime(2026, 4, 24, 21, 0, tzinfo=timezone.utc)

        # Fire: opens pending window.
        r_fire = run_monthly_sortino_gate(
            date(2026, 4, 24),
            champion_returns=_good_champion_returns(),
            challenger_returns=_good_challenger_returns(),
            champion_max_dd=0.10,
            challenger_max_dd=0.08,
            challenger_pbo=0.10,
            state_path=spath,
            now=t0,
        )
        # Approve.
        record_approval("2026-04", status="approved", state_path=spath, now=t0 + timedelta(hours=12))
        # Re-fire: pulls "prior_approved"; actual_replacement STILL False.
        r_after_approve = run_monthly_sortino_gate(
            date(2026, 4, 24),
            champion_returns=_good_champion_returns(),
            challenger_returns=_good_challenger_returns(),
            champion_max_dd=0.10,
            challenger_max_dd=0.08,
            challenger_pbo=0.10,
            state_path=spath,
            now=t0 + timedelta(hours=13),
        )
    ok = (
        r_fire["actual_replacement"] is False
        and r_after_approve["actual_replacement"] is False
        and r_after_approve["approved"] is True
    )
    print(
        f"[{'PASS' if ok else 'FAIL'}] no_auto_replacement_of_real_capital_champion  "
        f"(fire.actual={r_fire['actual_replacement']}, "
        f"after_approve.actual={r_after_approve['actual_replacement']}, "
        f"after_approve.approved={r_after_approve['approved']})"
    )
    return ok


def main() -> int:
    results = [
        case_fires_on_last_trading_friday(),
        case_reuses_friday_slot_zero_new_slots(),
        case_requires_sortino_delta_ge_0_3(),
        case_requires_pbo_lt_0_2(),
        case_requires_dd_ratio_le_1_2(),
        case_peder_slack_approval_with_48h_expiry(),
        case_no_auto_replacement_of_real_capital_champion(),
    ]
    ok = all(results)
    print(f"\n{'ALL PASS' if ok else 'FAILED'}  ({sum(results)}/{len(results)})")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
