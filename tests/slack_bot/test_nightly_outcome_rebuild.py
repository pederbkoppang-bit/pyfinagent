"""phase-9.6 tests."""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from backend.slack_bot.jobs.nightly_outcome_rebuild import run, _compute_outcomes
from backend.slack_bot.job_runtime import IdempotencyStore


def test_outcomes_win_loss_classification():
    rows = [{"trade_id": 1, "ticker": "A", "pnl": 5.0}, {"trade_id": 2, "ticker": "B", "pnl": -3.0}]
    out = _compute_outcomes(rows)
    assert out[0]["outcome"] == "win"
    assert out[1]["outcome"] == "loss"


def test_idempotent_rebuild():
    store = IdempotencyStore()
    fetches = []
    for _ in range(2):
        run(
            ledger_fetch_fn=lambda: (fetches.append(1), [{"trade_id": 1, "pnl": 1}])[1],
            outcome_write_fn=lambda o: len(o),
            store=store,
            day="2026-04-20",
        )
    assert len(fetches) == 1  # idempotent -> second run skipped


def test_bq_write_fail_open():
    """write_fn raising must not crash run()."""
    store = IdempotencyStore()

    def bad_write(outcomes):
        raise RuntimeError("BQ exploded")

    out = run(
        ledger_fetch_fn=lambda: [{"trade_id": 1, "pnl": 1}],
        outcome_write_fn=bad_write,
        store=store,
        day="2026-04-20",
    )
    assert out["rebuilt"] == 0  # fail-open
