"""phase-82.39 -- the nightly_outcome_rebuild fetch selected columns that do not exist.

MEASURED 2026-08-06 against the live schema: `financial_reports.paper_trades`
has 18 columns; `timestamp` present=False and `realized_pnl` present=False,
while the real columns are `created_at` (STRING) and `realized_pnl_pct` (FLOAT).
BigQuery answered 400 `Unrecognized name: timestamp`, a fail-open `except`
returned [], and the job produced an outcome rebuild over an empty set that
looked like a successful no-op. Dead since the file's first commit.

WHY THESE GUARDS CAN FAIL.

* Criterion 1's guard dry-runs the SQL obtained from the PRODUCTION builder
  (`build_ledger_fetch_sql`), not a copy pasted into this file -- a copied
  string proves nothing about what `_fetch` issues. It also dry-runs the
  PRE-FIX query and requires a 400 naming `timestamp`, which proves the dry run
  is a real validator rather than a rubber stamp that passes anything.
* Criterion 2's guard runs the query for real over a FIXED window. A fixture
  pinned to the production rolling 30-day window would pass today (3 rows) and
  silently return 0 after 2026-08-26 -- measured: SELLs carrying a P&L are
  2026-05 x8, 2026-06 x20, 2026-07 x4.
* Criterion 3's guard drives the real `_fetch` closure with the BQ client forced
  to raise, and captures the alert PAYLOAD. It patches
  `backend.services.observability.alerting.raise_cron_alert_sync`, because
  `_alert_fetch_failure` imports that symbol FUNCTION-LOCALLY -- patching a
  module-scope name in `_production_fns` would patch nothing and pass vacuously.
* Criterion 4's guard asserts the sweep's INPUT surface is non-empty as well as
  its output, and records the instrument's measured recall limit, because "found
  nothing" and "sees nothing" are indistinguishable otherwise.

NO `pytest.skip` ANYWHERE. Criteria 1 and 2 need live BigQuery by construction
(a dry run IS a BigQuery call). If credentials are missing these tests FAIL --
they do not quietly delete themselves. This is a local-only deployment where ADC
is present; a skip here would be the trapdoor class this project has been bitten
by before.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from backend.slack_bot.jobs._production_fns import (
    LEDGER_TABLE,
    build_ledger_fetch_sql,
    make_ledger_fetch_fn,
)

REPO = Path(__file__).resolve().parents[2]
ALERT_TARGET = "backend.services.observability.alerting.raise_cron_alert_sync"

#: The query as it stood before this step. Kept verbatim so criterion 1's guard
#: can prove the dry run REJECTS it -- a validator that accepts everything is
#: not a validator.
PRE_FIX_SQL = """
    SELECT trade_id, ticker, action, price, quantity, timestamp,
           SAFE_CAST(realized_pnl AS FLOAT64) AS pnl
    FROM `sunny-might-477607-p8.financial_reports.paper_trades`
    WHERE TIMESTAMP_TRUNC(timestamp, DAY) >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 DAY)
      AND realized_pnl IS NOT NULL
    LIMIT 1000
"""

#: A window with a stable, measured row count. NOT the rolling window -- see the
#: module docstring.
FIXED_START, FIXED_END = "2026-06-01", "2026-07-01"
FIXED_WINDOW_ROWS = 20


def _bq():
    """A live BigQuery client. Asserts its own precondition and FAILS -- never
    skips -- so the guard cannot delete itself when credentials are absent."""
    from google.cloud import bigquery

    try:
        return bigquery.Client(project="sunny-might-477607-p8")
    except Exception as exc:  # pragma: no cover -- credential failure path
        pytest.fail(
            "criteria 1 and 2 require live BigQuery (a dry run IS a BQ call) and "
            f"the client could not be constructed: {exc!r}. This test FAILS rather "
            "than skipping on purpose -- a skipped guard is a deleted guard."
        )


def _dry_run(sql: str) -> tuple[bool, str]:
    """(valid, message). A dry run is free: Google's documentation states
    'Dry runs don't use query slots, and you are not charged'."""
    from google.cloud import bigquery

    cfg = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)
    try:
        job = _bq().query(sql, job_config=cfg, location="us-central1")
        return True, f"valid, bytes={job.total_bytes_processed}"
    except Exception as exc:
        return False, str(exc)


# ---------------------------------------------------------------------------
# Criterion 1 -- dry-run validation of the PRODUCTION SQL
# ---------------------------------------------------------------------------


def test_production_sql_dry_runs_valid():
    """CRITERION 1, positive leg. Validates the string the production closure
    actually builds."""
    sql = build_ledger_fetch_sql()
    assert LEDGER_TABLE in sql, "fixture precondition: the builder targets the real table"
    valid, msg = _dry_run(sql)
    assert valid, f"the production query must dry-run valid; BigQuery said: {msg}"


def test_the_pre_fix_query_dry_runs_INVALID():
    """CRITERION 1's teeth. Without this leg, a dry run that rubber-stamped
    anything would satisfy the criterion. This proves it rejects the very defect
    the step exists to fix."""
    valid, msg = _dry_run(PRE_FIX_SQL)
    assert not valid, "the pre-fix query must be REJECTED, or the validator is vacuous"
    assert "Unrecognized name" in msg and "timestamp" in msg, msg


def test_the_phantom_columns_are_gone_from_the_production_sql():
    """A source-level companion to the dry run: the two phantom identifiers must
    not reappear, and the two real ones must be present."""
    sql = build_ledger_fetch_sql()
    assert not re.search(r"\brealized_pnl\b(?!_pct)", sql), (
        "`realized_pnl` does not exist on paper_trades")

    # Two INDEPENDENT assertions, deliberately not one disjunction.
    # An earlier revision read:
    #     assert not re.search(r"\btimestamp\b", sql, re.I) or "SAFE.TIMESTAMP" in sql
    # whose right-hand side is unconditionally True while the repaired query
    # uses SAFE.TIMESTAMP -- so the whole line could never fail, and restoring
    # the phantom `timestamp` column to the SELECT list still left it passing.
    # Measured by the 82.39 Q/A, which executed that counterfactual. That is the
    # `A or B` escape hatch where one side masks the other.
    assert not re.search(r"\btimestamp\b(?!\()", sql, re.I), (
        "the bare identifier `timestamp` must not appear -- it is not a column "
        "on paper_trades; only the SAFE.TIMESTAMP(...) function call is allowed")
    assert "SAFE.TIMESTAMP(created_at)" in sql, (
        "the STRING column must be parsed before comparison")
    assert "created_at" in sql and "realized_pnl_pct" in sql


def test_the_production_sql_stays_visible_to_the_sweep():
    """A near-miss worth pinning: extracting the seam almost blinded the sweep.

    `schema_oracle.extract_sql_literals` reassembles an f-string from its
    CONSTANT parts only. My first version of `build_ledger_fetch_sql` was an
    f-string interpolating both the table name and the WHERE predicate --
    measured, that dropped `tables_resolved` from 1 to 0 and left `scope` empty,
    i.e. it would have made this file invisible to the very instrument that
    found its defect, while every other guard in this file stayed green.

    So the production query is a fully PLAIN literal. This guard fails if anyone
    turns it back into an f-string.
    """
    import ast

    from backend.db import schema_oracle as so

    src = Path("backend/slack_bot/jobs/_production_fns.py")
    literals = so.extract_sql_literals(REPO / src)
    assert literals, "the sweep must find at least one SQL literal in this file"
    hits = [t for _ln, t in literals if "paper_trades" in t and "created_at" in t]
    assert hits, (
        "the production ledger query is no longer visible to extract_sql_literals "
        "-- it has probably become an f-string with an interpolated table name")

    # Structural: the constant itself must be a plain Constant, not a JoinedStr.
    tree = ast.parse((REPO / src).read_text(encoding="utf-8"))
    assigned = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "LEDGER_FETCH_SQL" for t in n.targets)
    ]
    assert len(assigned) == 1, "LEDGER_FETCH_SQL must be assigned exactly once"
    assert isinstance(assigned[0].value, ast.Constant), (
        "LEDGER_FETCH_SQL must be a plain string literal; an f-string hides its "
        "interpolated parts from the unknown-column sweep")


def test_windowed_variant_cannot_silently_be_a_no_op():
    """The fixed-window build is a `str.replace`. A replace that matches nothing
    returns the original unchanged and looks exactly like success -- so the
    target is asserted, and criterion 2's row count would otherwise be measured
    over the rolling window without anyone noticing."""
    from backend.slack_bot.jobs import _production_fns as pf

    windowed = build_ledger_fetch_sql(start=FIXED_START, end=FIXED_END)
    assert windowed != pf.LEDGER_FETCH_SQL, "the variant must actually differ"
    assert f"TIMESTAMP('{FIXED_START}')" in windowed
    assert "CURRENT_TIMESTAMP()" not in windowed, "the rolling predicate must be gone"

    with patch.object(pf, "LEDGER_FETCH_SQL", "SELECT 1"):
        with pytest.raises(AssertionError, match="no-op"):
            pf.build_ledger_fetch_sql(start=FIXED_START, end=FIXED_END)


def test_created_at_uses_safe_timestamp_not_timestamp_trunc():
    """`created_at` is a STRING column, so it must be PARSED before comparison.
    `TIMESTAMP_TRUNC` on a STRING is exactly what 400'd."""
    sql = build_ledger_fetch_sql()
    assert "SAFE.TIMESTAMP(created_at)" in sql
    assert "TIMESTAMP_TRUNC(created_at" not in sql


# ---------------------------------------------------------------------------
# Criterion 2 -- the repaired query actually SELECTS DATA
# ---------------------------------------------------------------------------


def test_repaired_query_returns_rows_for_a_period_with_trades():
    """CRITERION 2. Syntactic validity is not enough -- run it and count.

    The window is FIXED, not rolling: the production 30-day window returns 3
    rows today and 0 after 2026-08-26, so a rolling fixture would rot silently.
    """
    sql = build_ledger_fetch_sql(start=FIXED_START, end=FIXED_END)
    rows = list(_bq().query(sql, location="us-central1").result(timeout=30))
    assert len(rows) == FIXED_WINDOW_ROWS, (
        f"expected {FIXED_WINDOW_ROWS} rows in {FIXED_START}..{FIXED_END}, got "
        f"{len(rows)} -- if paper_trades changed, RE-MEASURE and update the "
        "constant rather than loosening this assertion")
    r = dict(rows[0])
    for key in ("trade_id", "ticker", "action", "created_at", "pnl"):
        assert key in r, f"the projection must carry {key}: {sorted(r)}"
    assert r["pnl"] is not None, "the IS NOT NULL predicate must hold"
    assert all(dict(x)["action"] == "SELL" for x in rows), (
        "only SELL legs carry realized_pnl_pct -- measured 32/32 SELLs, 0/33 BUYs")


def test_the_pre_fix_query_could_not_have_returned_anything():
    """The counterfactual, so 'it returns rows now' means something: the old
    query cannot even be executed."""
    with pytest.raises(Exception) as exc:
        list(_bq().query(PRE_FIX_SQL, location="us-central1").result(timeout=30))
    assert "Unrecognized name" in str(exc.value)


# ---------------------------------------------------------------------------
# Criterion 3 -- a failed fetch is operator-visible
# ---------------------------------------------------------------------------


def test_failed_fetch_emits_an_operator_alert():
    """CRITERION 3. Drives the REAL `_fetch` closure with the BQ client raising,
    and captures the emitted payload -- not a log line. A `caplog` assertion
    would already pass on the pre-fix tree, because the warning was always
    there; the warning is precisely what was NOT enough."""
    fetch = make_ledger_fetch_fn()
    boom = RuntimeError("400 Unrecognized name: nope")
    with patch("backend.slack_bot.jobs._production_fns._bq_client", side_effect=boom):
        with patch(ALERT_TARGET, autospec=True) as alert:
            out = fetch()

    assert out == [], "the fetch must stay fail-open and return an empty list"
    assert alert.call_count == 1, (
        f"exactly one operator alert expected, got {alert.call_count}")
    kw = alert.call_args.kwargs
    assert kw["severity"] == "P1", (
        "P2 is logged and DROPPED while slack_webhook_url is empty, so a P2 "
        f"would be invisible; got {kw['severity']}")
    assert kw["source"] == "nightly_outcome_rebuild"
    assert kw["error_type"] == "ledger_fetch_failed"
    assert "Unrecognized name" in kw["details"]["error"]
    assert kw["details"]["table"] == LEDGER_TABLE


def test_successful_fetch_emits_NO_alert():
    """So the guard cannot pass by always firing."""
    fetch = make_ledger_fetch_fn()
    client = MagicMock()
    client.query.return_value.result.return_value = [{"trade_id": "t1", "pnl": 1.0}]
    with patch("backend.slack_bot.jobs._production_fns._bq_client", return_value=client):
        with patch(ALERT_TARGET, autospec=True) as alert:
            out = fetch()
    assert out, "fixture precondition: the fetch really did return rows"
    assert alert.call_count == 0, f"a successful fetch must be silent: {alert.call_args_list}"


def test_alert_dispatch_failure_cannot_break_the_job():
    """A notification problem must not turn a tolerated fetch failure into an
    exception escaping into the scheduler."""
    fetch = make_ledger_fetch_fn()
    with patch("backend.slack_bot.jobs._production_fns._bq_client",
               side_effect=RuntimeError("bq down")):
        with patch(ALERT_TARGET, autospec=True, side_effect=RuntimeError("slack down")):
            assert fetch() == []


def test_the_alert_patch_target_is_the_only_one_that_works():
    """`_alert_fetch_failure` imports the emitter FUNCTION-LOCALLY, so the
    tempting module-scope patch target does not exist. If someone hoists that
    import, this fails and forces the guards above to be re-examined -- otherwise
    a module-scope patch would silence the real emitter while every assertion
    above still passed."""
    from backend.slack_bot.jobs import _production_fns as pf

    assert not hasattr(pf, "raise_cron_alert_sync"), (
        "_production_fns must not expose raise_cron_alert_sync at module scope")
    import backend.services.observability.alerting as alerting
    assert hasattr(alerting, "raise_cron_alert_sync")


# ---------------------------------------------------------------------------
# Criterion 4 -- the unknown-column sweep, and an honest account of its recall
# ---------------------------------------------------------------------------


def _sweep():
    from backend.db import schema_oracle as so

    return so, so.derive_scope(so.load_snapshot())


def test_sweep_input_surface_is_non_empty():
    """Before believing any output, prove the instrument LOOKED at something.
    'Found nothing' and 'sees nothing' are otherwise identical."""
    _so, r = _sweep()
    assert r["files_scanned"] > 0, r
    assert r["sql_literals"] > 0, r
    assert r["columns_in_oracle"] > 0, r
    assert r["tables_resolved"] > 0, r


def test_derived_scope_is_non_empty():
    """CRITERION 4, first half.

    NOTE this became satisfiable BECAUSE of this step's fix, and the mechanism is
    worth recording. `scope` is "(table, column) where the oracle declares STRING
    and a consumer applies date/number semantics". It was EMPTY before the repair
    because the only resolved table's query applied `TIMESTAMP_TRUNC` to a column
    that did not exist. The repaired query applies `SAFE.TIMESTAMP` to
    `created_at`, which IS a STRING column in the oracle -- so the fix is what
    puts a member in `scope`.
    """
    _so, r = _sweep()
    assert r["scope"], (
        "the derived scope must be non-empty; if this is red the repaired query "
        "no longer applies date semantics to a STRING column")
    assert any(
        s["column"] == "created_at" and s["semantics"] == "date" for s in r["scope"]
    ), r["scope"]


def test_every_unknown_column_is_fixed_or_queued():
    """CRITERION 4, second half. Each remaining member must be named by an OPEN
    masterplan step -- structurally, over the step objects, not a substring scan
    of the file (a substring scan passes on tokens that occur in unrelated
    steps)."""
    _so, r = _sweep()
    unknown = r["unknown_columns"]

    # The two members this step FIXED must be gone.
    flagged = {(u["identifier"], u["file"]) for u in unknown}
    assert ("timestamp", "backend/slack_bot/jobs/_production_fns.py") not in flagged, (
        f"this step's own defect is still flagged: {flagged}")
    assert ("realized_pnl", "backend/slack_bot/jobs/_production_fns.py") not in flagged

    if not unknown:
        return
    mp = json.loads((REPO / ".claude/masterplan.json").read_text(encoding="utf-8"))
    steps = [s for ph in mp["phases"] for s in ph.get("steps", [])]
    assert steps, "no masterplan steps parsed -- the check would be vacuous"
    open_steps = [s for s in steps if s.get("status") not in {"done", "dropped", "superseded"}]
    for u in unknown:
        named = [
            s for s in open_steps
            if u["identifier"] in s.get("name", "") and Path(u["file"]).name in s.get("name", "")
        ]
        assert named, (
            f"unknown column {u['identifier']} in {u['file']} is neither fixed nor "
            "named by an open step")


def test_the_sweeps_recall_limit_is_recorded_not_assumed():
    """The instrument's clean report is FALSE ASSURANCE and the repo must say so.

    Measured during this step: `derive_scope` resolves 1 table of 33 in the
    oracle, because ~20 backend files build table references with f-strings that
    the SQL-literal extractor cannot resolve, and `scripts/` is never scanned.
    The proof this matters is step 82.54 -- a live phantom-column defect in
    cost_budget_api.py that this sweep does NOT flag.

    This guard fails if that limitation stops being disclosed, so a future reader
    cannot mistake a clean sweep for a clean repo.
    """
    _so, r = _sweep()
    n_oracle_tables = len(_so.load_snapshot())
    assert r["tables_resolved"] < n_oracle_tables, (
        "if the sweep now resolves every table, this recall caveat is stale -- "
        "re-measure and update step 82.55 rather than deleting this guard")

    mp = json.loads((REPO / ".claude/masterplan.json").read_text(encoding="utf-8"))
    steps = [s for ph in mp["phases"] for s in ph.get("steps", [])]
    open_steps = [s for s in steps if s.get("status") not in {"done", "dropped", "superseded"}]

    recall = [s for s in open_steps
              if "tables_resolved" in s.get("name", "") and "derive_scope" in s.get("name", "")]
    assert recall, (
        "an open step must record that the sweep's recall is narrow, or a future "
        "criterion-4 'clean' report is false assurance")

    # Match on OWNERSHIP, not mention. A mutation run caught this: matching
    # `"cost_budget_api" in name` alone was satisfied by the sweep-recall step,
    # which merely CITES the defect as motivating evidence -- so closing the step
    # that actually repairs it left this guard green. The discriminator is the
    # step's own CRITERIA: only the owning step commits to repairing the
    # `input_tokens/output_tokens` projection.
    invisible = [
        s for s in open_steps
        if "cost_budget_api" in s.get("name", "")
        and any("input_tokens/output_tokens" in c
                for c in s.get("verification", {}).get("criteria", []))
    ]
    assert invisible, (
        "no OPEN step OWNS the live phantom-column defect the sweep cannot see "
        "(a step that merely mentions it does not count -- its criteria must "
        "commit to repairing the input_tokens/output_tokens projection)")
    assert all(s.get("verification", {}).get("criteria") for s in recall + invisible), (
        "a queued step with no criteria could never be closed")
