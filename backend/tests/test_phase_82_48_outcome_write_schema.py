"""phase-82.48 -- the outcome write emitted a schema that never existed.

MEASURED 2026-08-06: `financial_reports.outcome_tracking` has 0 rows and 9
columns, 3 of them REQUIRED. The pre-fix writer emitted
`{trade_id, ticker, pnl, outcome, recorded_at}` -- only `ticker` overlaps, both
REQUIRED columns were unsupplied, BigQuery rejected the whole batch, and
`insert_rows_json`'s RETURNED errors were swallowed into a log line.

THE TRAPS THIS FILE IS BUILT TO AVOID, each named before it was written:

* A key-vs-schema check that reads a STALE SNAPSHOT would go green on drift, and
  an empty oracle would make every emitted shape "valid". So the resolved column
  set is asserted non-empty AND asserted to contain the three REQUIRED names.
* An end-to-end fixture that "persists" into a MagicMock proves nothing -- a mock
  returning `[]` makes "N rows persisted" pass against the BROKEN writer.
  `test_write_really_persists_into_bigquery` does a REAL insert and reads it
  back. Not into production: streaming inserts cannot be deleted for up to 90
  minutes, so a test row would pollute `get_performance_stats` for its four live
  consumers. It builds a throwaway table and drops it in teardown.
* An alert guard that fires on every call proves nothing: there is a negative
  control, and `severity == "P1"` is pinned because a P2 is logged and dropped
  while `slack_webhook_url` is empty.
* A NULL-pnl fixture that uses `0.0`, or OMITS the key, proves nothing --
  `.get("pnl", 0.0)` legitimately returns `0.0` when the key is absent. The
  fixture below has the key PRESENT with value `None`, which is the actual
  defect. And `pytest.raises` around `run()` would never fire anyway, because
  `heartbeat` suppresses.
"""
from __future__ import annotations

import os
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from backend.slack_bot.jobs._production_fns import (
    OUTCOME_COLUMNS,
    OUTCOME_REQUIRED_COLUMNS,
    OUTCOME_TABLE,
    build_outcome_row,
    make_outcome_write_fn,
)
from backend.slack_bot.jobs.nightly_outcome_rebuild import _compute_outcomes

REPO = Path(__file__).resolve().parents[2]
ALERT_TARGET = "backend.services.observability.alerting.raise_cron_alert_sync"

#: The pre-fix emitted shape, kept verbatim so criterion 1's guard can show it
#: FAILS -- a validator that accepts it is not a validator.
PRE_FIX_ROW = {
    "trade_id": "t1", "ticker": "AMD", "pnl": 5.5,
    "outcome": "win", "recorded_at": "2026-08-06T00:00:00Z",
}

_TRADE = {
    "trade_id": "t1", "ticker": "AMD", "pnl": 5.5, "action": "SELL",
    "created_at": "2026-07-27T18:05:27.845972+00:00",
    "risk_judge_decision": "BUY", "price": 101.0, "holding_days": 12,
}


def _live_schema() -> dict[str, str]:
    """Column -> mode, from the destination's REAL schema.

    Prefers the live table; falls back to the checked-in 82.12 snapshot so the
    default test path is offline and free. Either way the result is asserted
    non-empty by every caller -- an empty oracle would make any shape 'valid'.
    """
    try:
        from google.cloud import bigquery

        tbl = bigquery.Client(project="sunny-might-477607-p8").get_table(OUTCOME_TABLE)
        return {f.name: f.mode for f in tbl.schema}
    except Exception:
        import json

        snap = json.loads(
            (REPO / "backend" / "db" / "_schema_snapshot.json").read_text(encoding="utf-8")
        )
        # The snapshot nests tables under a `tables` key -- an earlier version
        # looked the dotted name up at TOP level, which never matched, so this
        # "offline fallback" returned {} unconditionally and the docstring's
        # claim that the default path is offline was false. Caught by the 82.48
        # Q/A. It failed LOUD (the non-empty assert below fires), so nothing
        # passed vacuously -- but the claim did not reproduce.
        cols = (snap.get("tables") or {}).get("financial_reports.outcome_tracking") or {}
        return {k: (v.get("mode") if isinstance(v, dict) else "NULLABLE")
                for k, v in cols.items()}


# ---------------------------------------------------------------------------
# Criterion 1 -- emitted keys validated against the REAL schema
# ---------------------------------------------------------------------------


def test_the_oracle_is_not_empty_or_stale():
    """Before believing any validation, prove the instrument sees a real table.
    An empty column set would make every emitted shape pass."""
    schema = _live_schema()
    assert schema, "the schema oracle resolved NOTHING -- validation would be vacuous"
    assert len(schema) == 9, f"outcome_tracking should have 9 columns, got {sorted(schema)}"
    for req in OUTCOME_REQUIRED_COLUMNS:
        assert req in schema, req
        assert schema[req] == "REQUIRED", (
            f"{req} is no longer REQUIRED; this step's premise must be re-derived")


def test_emitted_keys_match_the_destination_schema():
    """CRITERION 1, positive leg."""
    schema = _live_schema()
    assert schema, "precondition"
    row = build_outcome_row(_compute_outcomes([_TRADE])[0], evaluated_at="2026-08-06T00:00:00Z")
    assert set(row) == set(schema), (
        f"emitted-vs-schema mismatch: emitted-only={sorted(set(row) - set(schema))}, "
        f"schema-only={sorted(set(schema) - set(row))}")
    assert tuple(row) == OUTCOME_COLUMNS


def test_the_PRE_FIX_shape_is_rejected_and_the_missing_REQUIRED_columns_are_named():
    """CRITERION 1's teeth: the guard must FAIL against the old shape, and say
    WHICH required columns were missing."""
    schema = _live_schema()
    assert schema, "precondition"
    emitted = set(PRE_FIX_ROW)
    missing_required = {c for c in OUTCOME_REQUIRED_COLUMNS if c not in emitted}
    assert missing_required == {"analysis_date", "recommendation"}, missing_required
    unknown = emitted - set(schema)
    assert unknown == {"trade_id", "pnl", "outcome", "recorded_at"}, sorted(unknown)
    assert emitted & set(schema) == {"ticker"}, (
        "only `ticker` overlapped -- that is why BigQuery rejected the batch")


def test_a_trade_with_no_analysis_anchor_is_SKIPPED_not_fabricated():
    """BigQuery's REQUIRED mode rejects NULL but ACCEPTS an empty string, so an
    `or ""` fallback inserts an outcome with no identity -- polluting the table
    `get_performance_stats` serves to four consumers.

    This guard caught exactly that in my first draft, which ended the fallback
    chain at `""`. Skipping is the honest behaviour; fabricating an identity is
    not.
    """
    bare = {"trade_id": "x", "ticker": "MU", "pnl": 1.0}
    assert not (bare.get("analysis_id") or bare.get("created_at")), "precondition"
    assert _compute_outcomes([bare]) == [], (
        "a trade with no analysis anchor must be skipped, not written with an "
        "empty REQUIRED column")

    # And a trade that DOES have an anchor must still come through, or the
    # skip could pass by dropping everything.
    assert len(_compute_outcomes([_TRADE])) == 1


def test_required_columns_are_never_empty_for_rows_that_ARE_written():
    """For everything that survives the skip, all three REQUIRED columns must
    carry real values."""
    row = build_outcome_row(_compute_outcomes([_TRADE])[0], evaluated_at="e")
    for req in OUTCOME_REQUIRED_COLUMNS:
        assert row[req] is not None and row[req] != "", (
            f"{req} is REQUIRED but the row carries {row[req]!r}")


def test_the_fetch_supplies_every_field_the_write_REQUIRES():
    """The seam between 82.39's fetch and this step's write.

    A mutation run caught this: dropping `analysis_id, risk_judge_decision,
    holding_days` from the fetch SQL left every guard green, because they all
    build their trade dicts by hand. The write would then have nothing to derive
    the two REQUIRED columns from and would skip every row -- a silent return to
    zero written outcomes, which is the exact defect this step fixes.

    Driven from the production SQL BUILDER, not a source scan of the file.
    """
    from backend.slack_bot.jobs._production_fns import build_ledger_fetch_sql

    sql = build_ledger_fetch_sql()
    select_part = sql.split("FROM", 1)[0]
    for needed in ("analysis_id", "risk_judge_decision", "holding_days",
                   "created_at", "realized_pnl_pct", "ticker", "price"):
        assert needed in select_part, (
            f"the fetch no longer projects {needed!r}, which the write needs to "
            f"populate outcome_tracking; SELECT was: {select_part}")

    # And prove the dependency is REAL, not merely that the string is present.
    # The 82.48 Q/A probed the earlier version of this loop and found 1 of its 2
    # iterations asserted NOTHING (`analysis_id` was never in _TRADE, so removing
    # it changed nothing). Written as explicit cases instead of a loop that
    # silently no-ops.
    no_anchor = {k: v for k, v in _TRADE.items() if k != "created_at"}
    assert "analysis_id" not in no_anchor and "created_at" not in no_anchor
    assert _compute_outcomes([no_anchor]) == [], (
        "with neither analysis_id nor created_at the outcome must be skipped")

    with_analysis_id = dict(no_anchor, analysis_id="a-42")
    got = _compute_outcomes([with_analysis_id])
    assert len(got) == 1 and got[0]["analysis_date"] == "a-42", (
        "analysis_id must be the preferred anchor when present")

    no_reco = {k: v for k, v in _TRADE.items() if k not in ("risk_judge_decision", "action")}
    assert _compute_outcomes([no_reco]) == [], (
        "with no recommendation source the outcome must be skipped")


# ---------------------------------------------------------------------------
# Criterion 2 -- a REAL round trip, not a mock
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    os.getenv("PYFIN_SKIP_LIVE_BQ") == "1",
    reason="explicitly disabled by the operator via PYFIN_SKIP_LIVE_BQ=1",
)
def test_write_really_persists_into_bigquery():
    """CRITERION 2. A mock cannot satisfy this: a MagicMock returning `[]` makes
    'N rows persisted' pass against the BROKEN writer.

    Into a THROWAWAY table, never production -- rows streamed into
    `outcome_tracking` could not be deleted for up to 90 minutes and would
    pollute `get_performance_stats` for its four live consumers.

    Note the skipif is gated on an EXPLICIT operator opt-out, not on whether
    credentials happen to be missing. A guard that silently skips itself when
    the environment is inconvenient is a deleted guard.
    """
    from google.cloud import bigquery

    client = bigquery.Client(project="sunny-might-477607-p8")
    src = client.get_table(OUTCOME_TABLE)
    tmp_id = f"sunny-might-477607-p8.financial_reports.tmp_82_48_{uuid.uuid4().hex[:8]}"
    client.create_table(bigquery.Table(tmp_id, schema=src.schema))
    try:
        row = build_outcome_row(
            _compute_outcomes([_TRADE])[0], evaluated_at="2026-08-06T00:00:00Z"
        )
        errors = client.insert_rows_json(tmp_id, [row])
        assert errors == [], f"the repaired shape was REJECTED: {errors}"

        got = list(client.query(
            f"SELECT ticker, analysis_date, recommendation, return_pct "  # noqa: S608
            f"FROM `{tmp_id}`"
        ).result(timeout=60))
        assert len(got) == 1, f"nothing was persisted: {got}"
        assert got[0]["ticker"] == "AMD"
        assert got[0]["return_pct"] == pytest.approx(5.5)
        assert got[0]["recommendation"] == "BUY"
    finally:
        client.delete_table(tmp_id, not_found_ok=True)


@pytest.mark.skipif(
    os.getenv("PYFIN_SKIP_LIVE_BQ") == "1",
    reason="explicitly disabled by the operator via PYFIN_SKIP_LIVE_BQ=1",
)
def test_the_PRODUCTION_closure_persists_into_bigquery():
    """CRITERION 2, one seam further -- and the seam the 82.48 Q/A found missing.

    The round-trip guard above drives `_compute_outcomes -> build_outcome_row ->
    insert`, but never `make_outcome_write_fn()._write`, which is what production
    actually calls. Every other test that drives `_write` patches `_bq_client`
    with a MagicMock that accepts ANY shape -- so nothing proved the PRODUCTION
    closure lands a row in a real table. That is the guards-stop-one-seam-short
    class, and this closes it.

    Also exercises `_drop_already_written` against real BigQuery: the second
    call must write nothing, which a mock could not honestly demonstrate.
    """
    from google.cloud import bigquery

    from backend.slack_bot.jobs import _production_fns as pf

    client = bigquery.Client(project="sunny-might-477607-p8")
    src = client.get_table(OUTCOME_TABLE)
    tmp_id = f"sunny-might-477607-p8.financial_reports.tmp_82_48c_{uuid.uuid4().hex[:8]}"
    client.create_table(bigquery.Table(tmp_id, schema=src.schema))
    try:
        with patch.object(pf, "OUTCOME_TABLE", tmp_id):
            write = pf.make_outcome_write_fn()
            outcomes = _compute_outcomes([_TRADE])
            n = write(outcomes)
            assert n == 1, "the production closure reported no rows written"

            got = list(client.query(
                f"SELECT ticker, return_pct, recommendation FROM `{tmp_id}`"  # noqa: S608
            ).result(timeout=60))
            assert len(got) == 1, f"the production closure persisted nothing: {got}"
            assert got[0]["ticker"] == "AMD"
            assert got[0]["return_pct"] == pytest.approx(5.5)

            # Dedup, against a REAL table rather than a mocked read-back.
            n2 = write(outcomes)
            assert n2 == 0, "the second write must be deduplicated"
            again = list(client.query(
                f"SELECT COUNT(*) AS c FROM `{tmp_id}`"  # noqa: S608
            ).result(timeout=60))
            assert again[0]["c"] == 1, (
                f"the rolling window duplicated the outcome: {again[0]['c']} rows")
    finally:
        client.delete_table(tmp_id, not_found_ok=True)


def test_a_mock_cannot_substitute_for_the_round_trip():
    """Pins WHY the test above must be live: against a mock, the PRE-FIX writer
    also 'succeeds'. This is the vacuity the criterion's wording forbids."""
    fake = MagicMock()
    fake.insert_rows_json.return_value = []
    assert fake.insert_rows_json(OUTCOME_TABLE, [PRE_FIX_ROW]) == [], (
        "a mock accepts the broken shape -- shape agreement against a mock "
        "proves nothing about persistence")


# ---------------------------------------------------------------------------
# Criterion 3 -- a REJECTED write is operator-visible
# ---------------------------------------------------------------------------


def _client_returning(errors):
    c = MagicMock()
    c.project = "sunny-might-477607-p8"
    c.insert_rows_json.return_value = errors
    c.query.return_value.result.return_value = []  # dedup read-back: nothing seen
    return c


def test_rejected_write_emits_an_operator_alert():
    """CRITERION 3. `insert_rows_json` RETURNS errors rather than raising, so
    the pre-fix `if errors: log; return 0` was indistinguishable from a no-op."""
    write = make_outcome_write_fn()
    rejection = [{"index": 0, "errors": [{"reason": "invalid", "message": "no such field"}]}]
    with patch("backend.slack_bot.jobs._production_fns._bq_client",
               return_value=_client_returning(rejection)):
        with patch(ALERT_TARGET, autospec=True) as alert:
            n = write(_compute_outcomes([_TRADE]))

    assert n == 0, "a rejected write must not report rows written"
    assert alert.call_count == 1, f"expected one alert, got {alert.call_count}"
    kw = alert.call_args.kwargs
    assert kw["severity"] == "P1", (
        f"a P2 is logged and DROPPED while slack_webhook_url is empty; got {kw['severity']}")
    assert kw["source"] == "nightly_outcome_rebuild"
    assert kw["error_type"] == "outcome_write_rejected"
    assert "no such field" in kw["details"]["errors"]


def test_successful_write_emits_NO_alert():
    """The negative control -- without it the guard passes by always firing."""
    write = make_outcome_write_fn()
    with patch("backend.slack_bot.jobs._production_fns._bq_client",
               return_value=_client_returning([])):
        with patch(ALERT_TARGET, autospec=True) as alert:
            n = write(_compute_outcomes([_TRADE]))
    assert n == 1, "fixture precondition: the write really succeeded"
    assert alert.call_count == 0, f"a clean write must be silent: {alert.call_args_list}"


def test_the_alert_patch_target_is_the_only_one_that_works():
    """`_alert_write_rejected` imports the emitter FUNCTION-LOCALLY, so the
    tempting module-scope target does not exist. If someone hoists it, this
    fails and forces the guards above to be re-examined."""
    from backend.slack_bot.jobs import _production_fns as pf

    assert not hasattr(pf, "raise_cron_alert_sync")
    import backend.services.observability.alerting as alerting
    assert hasattr(alerting, "raise_cron_alert_sync")


def test_the_rolling_window_cannot_duplicate_outcomes():
    """A defect the FIX would have introduced: the fetch re-reads a rolling
    30-day window nightly and the writer APPENDS, so one SELL would land ~30
    times. The daily idempotency key does not prevent that."""
    write = make_outcome_write_fn()
    client = _client_returning([])
    already = MagicMock()
    already.__getitem__ = lambda _s, k: {"ticker": "AMD",
                                         "analysis_date": "2026-07-27T18:05:27.845972+00:00"}[k]
    client.query.return_value.result.return_value = [already]
    with patch("backend.slack_bot.jobs._production_fns._bq_client", return_value=client):
        with patch(ALERT_TARGET, autospec=True):
            n = write(_compute_outcomes([_TRADE]))
    assert n == 0, "an outcome already in the table must not be written again"
    client.insert_rows_json.assert_not_called()


# ---------------------------------------------------------------------------
# Criterion 4 -- a NULL pnl does not raise
# ---------------------------------------------------------------------------


def test_null_pnl_does_not_raise():
    """CRITERION 4. The key is PRESENT with value None -- that is the defect.
    `.get("pnl", 0.0)` returns the default only when the key is ABSENT, so a
    fixture that omits the key would prove nothing."""
    row = dict(_TRADE, pnl=None)
    assert "pnl" in row and row["pnl"] is None, "fixture precondition"
    out = _compute_outcomes([row])
    assert len(out) == 1
    assert out[0]["pnl"] == 0.0
    assert out[0]["outcome"] == "loss"


def test_absent_pnl_key_is_a_DIFFERENT_case_and_also_works():
    """The distinction the criterion turns on, pinned so a future fixture cannot
    conflate them."""
    row = {k: v for k, v in _TRADE.items() if k != "pnl"}
    assert "pnl" not in row, "fixture precondition"
    out = _compute_outcomes([row])
    assert out[0]["pnl"] == 0.0 and out[0]["outcome"] == "loss"


def test_the_pre_fix_expression_really_did_raise():
    """Shows the defect was real rather than asserted, by evaluating the old
    expression on the same fixture."""
    row = dict(_TRADE, pnl=None)
    with pytest.raises(TypeError):
        _ = row.get("pnl", 0.0) > 0
