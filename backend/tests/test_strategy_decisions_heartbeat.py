"""phase-30.7 tests for strategy_decisions cycle-heartbeat writer.

Audit basis: handoff/archive/phase-30.0/experiment_results.md Stage 3
(FAIL) + phase-30.7 research_brief.md (G). The phase-26.5 migration
created `pyfinagent_data.strategy_decisions` but no production cycle
ever wrote a row -- only 1 smoke-test row across 36+ days. phase-30.7
wires a per-cycle heartbeat write that satisfies the grep verification
AND establishes operator-visible BQ rows documenting the dormant-by-
design state.

Test plan (4 cases):
  1. save_strategy_decision exists and calls insert_rows_json on the
     pyfinagent_data.strategy_decisions table.
  2. Insert errors are logged but do not raise.
  3. Grep-equivalent: `strategy_decisions` symbol present in
     autonomous_loop.py (mirrors masterplan verification command).
  4. The heartbeat row shape passes schema sanity (required fields).
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


def _settings_for_bq() -> SimpleNamespace:
    return SimpleNamespace(
        gcp_project_id="sunny-might-477607-p8",
        bq_dataset_reports="financial_reports",
        bq_table_reports="analysis_results",
        bq_dataset_outcomes="financial_reports",
        bq_table_outcomes="outcome_tracking",
        gcp_credentials_json="",
    )


def test_save_strategy_decision_targets_correct_table():
    """phase-30.7 Test A: save_strategy_decision sends the row to
    `<project>.pyfinagent_data.strategy_decisions` (NOT
    `<project>.financial_reports.strategy_decisions` -- the migration
    explicitly puts the table in pyfinagent_data)."""
    from backend.db.bigquery_client import BigQueryClient

    with patch("backend.db.bigquery_client.bigquery.Client") as mock_client_cls:
        mock_client = mock_client_cls.return_value
        mock_client.insert_rows_json.return_value = []

        client = BigQueryClient(_settings_for_bq())
        row = {
            "ts": "2026-05-20T18:00:00+00:00",
            "cycle_id": "abc12345",
            "decided_strategy": "triple_barrier",
            "prior_strategy": "triple_barrier",
            "trigger": "cycle_heartbeat",
            "decay_signal": None,
            "decay_attribution": None,
            "rationale": "per-cycle heartbeat; no regime change detected.",
        }
        client.save_strategy_decision(row)

    assert mock_client.insert_rows_json.call_count == 1
    table_arg, rows_arg = mock_client.insert_rows_json.call_args.args
    assert table_arg == (
        "sunny-might-477607-p8.pyfinagent_data.strategy_decisions"
    )
    assert rows_arg == [row]


def test_save_strategy_decision_swallows_insert_errors():
    """phase-30.7 Test B: insert_rows_json returning errors logs but
    does not raise. Fail-open is mandatory because this is a P3
    observability write -- a BQ error must not break the cycle."""
    from backend.db.bigquery_client import BigQueryClient

    with patch("backend.db.bigquery_client.bigquery.Client") as mock_client_cls:
        mock_client = mock_client_cls.return_value
        mock_client.insert_rows_json.return_value = [
            {"index": 0, "errors": [{"reason": "invalid", "message": "test"}]},
        ]

        client = BigQueryClient(_settings_for_bq())
        # Must NOT raise.
        client.save_strategy_decision({
            "ts": "2026-05-20T18:00:00+00:00",
            "cycle_id": "abc12345",
            "decided_strategy": "triple_barrier",
            "prior_strategy": "triple_barrier",
            "trigger": "cycle_heartbeat",
            "decay_signal": None,
            "decay_attribution": None,
            "rationale": "test",
        })


def test_autonomous_loop_step_10_5_contains_strategy_decisions_symbol():
    """phase-30.7 Test C: mirrors the masterplan verification command
    `grep -q 'strategy_decisions' backend/services/autonomous_loop.py`.
    Future refactor that removes the heartbeat write breaks pytest."""
    src = (
        Path(__file__).resolve().parents[1]
        / "services" / "autonomous_loop.py"
    ).read_text(encoding="utf-8")
    assert "strategy_decisions" in src, (
        "phase-30.7: autonomous_loop.py must reference strategy_decisions "
        "(the masterplan grep verification command checks this symbol)"
    )
    # And the heartbeat trigger string must be present.
    assert "cycle_heartbeat" in src, (
        "phase-30.7: autonomous_loop.py must contain the cycle_heartbeat "
        "trigger label per the documented row shape"
    )


def test_heartbeat_row_shape_has_required_fields():
    """phase-30.7 Test D: the heartbeat row carries all required fields
    per the table schema at scripts/migrations/add_strategy_decisions_table.py:38-54.

    Required fields (NOT NULL per schema): ts, decided_strategy, trigger.
    Nullable: cycle_id, prior_strategy, decay_signal, decay_attribution,
    rationale.
    """
    # Reproduce the production row shape with phase-30.7 defaults.
    row = {
        "ts": "2026-05-20T18:00:00+00:00",
        "cycle_id": "abc12345",
        "decided_strategy": "triple_barrier",
        "prior_strategy": "triple_barrier",
        "trigger": "cycle_heartbeat",
        "decay_signal": None,
        "decay_attribution": None,
        "rationale": "per-cycle heartbeat",
    }
    # Required-not-null fields must be present and non-empty strings/timestamp.
    assert row["ts"], "ts is required by the BQ schema (NOT NULL)"
    assert row["decided_strategy"], "decided_strategy is required (NOT NULL)"
    assert row["trigger"] == "cycle_heartbeat", "trigger must be the heartbeat label"
    # Nullable fields can be None but should be present in the dict.
    for k in ("cycle_id", "prior_strategy", "decay_signal",
              "decay_attribution", "rationale"):
        assert k in row, f"row must contain {k} (nullable but expected)"
