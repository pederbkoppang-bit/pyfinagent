"""phase-9.4 tests."""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from backend.slack_bot.jobs.nightly_mda_retrain import run
from backend.slack_bot.job_runtime import IdempotencyStore


def test_good_model_promotes_and_commits():
    committed: list = []
    store = IdempotencyStore()
    out = run(
        train_fn=lambda: {"trial_id": "good", "dsr": 0.99, "pbo": 0.10},
        commit_fn=lambda m: committed.append(m),
        store=store,
        day="2026-04-20",
    )
    assert out["promoted"] is True
    assert len(committed) == 1


def test_rejected_model_does_not_commit():
    committed: list = []
    store = IdempotencyStore()
    out = run(
        train_fn=lambda: {"trial_id": "bad", "dsr": 0.80, "pbo": 0.50},  # fails DSR
        commit_fn=lambda m: committed.append(m),
        store=store,
        day="2026-04-20",
    )
    assert out["promoted"] is False
    assert len(committed) == 0


def test_idempotent_same_day():
    called = []
    store = IdempotencyStore()
    for _ in range(2):
        run(
            train_fn=lambda: (called.append(1), {"dsr": 0.99, "pbo": 0.10})[1],
            commit_fn=lambda m: None,
            store=store,
            day="2026-04-20",
        )
    # Second call should be skipped, so train_fn called once
    assert len(called) == 1
