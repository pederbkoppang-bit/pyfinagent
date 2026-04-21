"""phase-10.6 pytest companion to phase10_monthly_sortino_test.py."""
from __future__ import annotations

import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from backend.autoresearch.monthly_champion_challenger import (
    is_last_trading_friday,
    record_approval,
    run_monthly_sortino_gate,
)
from backend.autoresearch import weekly_ledger


CHAMP_RETURNS = [0.001, 0.002, -0.001, 0.003, 0.001, -0.002, 0.004, 0.002, -0.001, 0.001] * 3
CHALL_RETURNS = [0.004, 0.005, -0.001, 0.006, 0.004, -0.002, 0.007, 0.005, -0.001, 0.004] * 3
T_FIRE = datetime(2026, 4, 24, 21, 0, tzinfo=timezone.utc)
LAST_FRI_APR = date(2026, 4, 24)


def _run(state_path, **overrides):
    base = dict(
        champion_returns=CHAMP_RETURNS,
        challenger_returns=CHALL_RETURNS,
        champion_max_dd=0.10,
        challenger_max_dd=0.08,
        challenger_pbo=0.10,
        state_path=state_path,
        now=T_FIRE,
    )
    base.update(overrides)
    return run_monthly_sortino_gate(LAST_FRI_APR, **base)


def test_fires_on_last_trading_friday(tmp_path):
    assert is_last_trading_friday(date(2026, 4, 24)) is True
    assert is_last_trading_friday(date(2026, 4, 17)) is False
    r = _run(tmp_path / "s.json")
    assert r["fired"] is True
    assert r["gate_pass"] is True


def test_not_last_friday_short_circuits(tmp_path):
    r = run_monthly_sortino_gate(
        date(2026, 4, 17),
        champion_returns=CHAMP_RETURNS,
        challenger_returns=CHALL_RETURNS,
        champion_max_dd=0.10,
        challenger_max_dd=0.08,
        challenger_pbo=0.10,
        state_path=tmp_path / "s.json",
        now=T_FIRE,
    )
    assert r["fired"] is False
    assert r["reason"] == "not_last_trading_friday"


def test_reuses_friday_slot_zero_new_slots(tmp_path):
    lpath = tmp_path / "ledger.tsv"
    weekly_ledger.append_row(
        week_iso="2026-W17",
        thu_batch_id="uuid-stub",
        thu_candidates_kicked=128,
        notes="kicked_off",
        path=lpath,
    )
    before = weekly_ledger.read_rows(path=lpath)
    _run(tmp_path / "state.json")
    after = weekly_ledger.read_rows(path=lpath)
    assert before == after


def test_requires_sortino_delta_ge_0_3(tmp_path):
    r = _run(tmp_path / "s.json", challenger_returns=CHAMP_RETURNS)
    assert r["gate_pass"] is False
    assert "sortino_delta" in r["reason"]


def test_requires_pbo_lt_0_2(tmp_path):
    r = _run(tmp_path / "s.json", challenger_pbo=0.25)
    assert r["gate_pass"] is False
    assert "pbo" in r["reason"]


def test_requires_dd_ratio_le_1_2(tmp_path):
    r = _run(tmp_path / "s.json", challenger_max_dd=0.20)
    assert r["gate_pass"] is False
    assert "dd_ratio" in r["reason"]


def test_challenger_min_days_floor(tmp_path):
    r = _run(tmp_path / "s.json", challenger_returns=CHALL_RETURNS[:10])
    assert r["gate_pass"] is False
    assert "days" in r["reason"]


def test_approval_window_48h_expiry(tmp_path):
    spath = tmp_path / "state.json"
    r1 = _run(spath)
    assert r1["approval_pending"] is True

    r_mid = _run(spath, now=T_FIRE + timedelta(hours=24))
    assert r_mid["approval_pending"] is True
    assert r_mid["expired"] is False

    r_exp = _run(spath, now=T_FIRE + timedelta(hours=48))
    assert r_exp["expired"] is True


def test_record_approval_transitions(tmp_path):
    spath = tmp_path / "state.json"
    _run(spath)
    row = record_approval("2026-04", status="approved", state_path=spath, now=T_FIRE + timedelta(hours=12))
    assert row["status"] == "approved"

    # Re-running after approval surfaces prior_approved
    r_after = _run(spath, now=T_FIRE + timedelta(hours=13))
    assert r_after["approved"] is True
    assert r_after["actual_replacement"] is False


def test_no_auto_replacement_hard_coded(tmp_path):
    spath = tmp_path / "state.json"
    r = _run(spath)
    # Approved or pending or expired -- actual_replacement is ALWAYS False.
    assert r["actual_replacement"] is False


def test_slack_fn_called_when_gate_passes(tmp_path):
    calls = []
    _run(tmp_path / "s.json", slack_fn=lambda msg, meta: calls.append((msg, meta)))
    assert len(calls) == 1
    assert "Monthly Champion/Challenger" in calls[0][0]


def test_slack_fn_fail_open(tmp_path):
    def boom(msg, meta):
        raise RuntimeError("slack down")
    r = _run(tmp_path / "s.json", slack_fn=boom)
    # Gate state is still persisted even if slack_fn raises.
    assert r["approval_pending"] is True
