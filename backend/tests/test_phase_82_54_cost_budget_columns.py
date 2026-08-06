"""phase-82.54 -- a phantom-column defect the schema sweep was blind to.

`cost_budget_api` selected `input_tokens` / `output_tokens` from
`pyfinagent_data.llm_call_log`, whose real columns are `input_tok` /
`output_tok`. Every call 400'd and the fail-open `except` returned
`(None, None)`.

WHY THE SWEEP MISSED IT: the table reference was an f-string, and
`extract_sql_literals` reassembles an f-string from its CONSTANT parts only, so
the interpolated project collapsed the fully-qualified name and the file
resolved ZERO literals. `test_the_sweep_can_now_see_this_file` pins the fix.

CRITERION 2 IS VACUOUS AS WRITTEN, AND THIS FILE EXCEEDS IT. The query is an
aggregate with no GROUP BY, so it always returns exactly one row, and COALESCE
makes it non-NULL -- measured, a day with ZERO calls still returns
`tokens=0, calls=0`. "Returns a non-null total" therefore cannot fail. The guard
below asserts a POSITIVE total over a FIXED window with a `calls > 0`
precondition.

THE FIX IS NOT A RENAME, and a guard that only checked the two names would have
blessed a 26x undercount: `llm_call_log` has FOUR token columns and the cache
pair dominates (measured 2026-08-05: 353,896 vs 9,159,745).
"""
from __future__ import annotations

import ast
import os
import re
from pathlib import Path
from unittest.mock import patch

import pytest

from backend.api.cost_budget_api import LLM_TOKENS_TODAY_SQL, _fetch_llm_tokens_today

REPO = Path(__file__).resolve().parents[2]
API_FILE = REPO / "backend" / "api" / "cost_budget_api.py"
ALERT_TARGET = "backend.services.observability.alerting.raise_cron_alert_sync"
TABLE = "sunny-might-477607-p8.pyfinagent_data.llm_call_log"

#: A day with measured traffic. FIXED, not "yesterday" -- the research gate
#: measured that 2026-07-26 has 1 call and 0 tokens, so a relative window can
#: silently become a zero day.
BUSY_DAY = "2026-08-05"

_LIVE = os.getenv("PYFIN_SKIP_LIVE_BQ") != "1"


def _bq():
    from google.cloud import bigquery

    return bigquery.Client(project="sunny-might-477607-p8")


def _dry_run(sql: str) -> tuple[bool, str]:
    from google.cloud import bigquery

    cfg = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)
    try:
        job = _bq().query(sql, job_config=cfg)
        return True, f"valid, bytes={job.total_bytes_processed}"
    except Exception as exc:
        return False, str(exc)


#: The pre-fix projection, kept verbatim so criterion 1 can show it REJECTED.
PRE_FIX_SQL = LLM_TOKENS_TODAY_SQL.replace(
    "SUM(input_tok)", "SUM(input_tokens)"
).replace("SUM(output_tok)", "SUM(output_tokens)")


# ---------------------------------------------------------------------------
# Criterion 1 -- dry-run validation of the PRODUCTION string
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _LIVE, reason="operator opt-out via PYFIN_SKIP_LIVE_BQ=1")
def test_production_sql_dry_runs_valid():
    """CRITERION 1, positive leg -- validates the constant production USES."""
    valid, msg = _dry_run(LLM_TOKENS_TODAY_SQL)
    assert valid, f"the production query must dry-run valid; BigQuery said: {msg}"
    assert "bytes=0" in msg, (
        f"the partition filter must prune to 0 bytes; got {msg}")


@pytest.mark.skipif(not _LIVE, reason="operator opt-out via PYFIN_SKIP_LIVE_BQ=1")
def test_the_PRE_FIX_projection_dry_runs_INVALID():
    """CRITERION 1's teeth. Without this a rubber-stamp validator would pass."""
    valid, msg = _dry_run(PRE_FIX_SQL)
    assert not valid, "the pre-fix projection must be REJECTED"
    assert "Unrecognized name" in msg and "input_tokens" in msg, msg


def test_the_phantom_identifiers_are_gone_and_all_four_real_ones_present():
    """Two INDEPENDENT assertions, not one disjunction -- the `A or B` shape
    where one side masks the other has bitten this project this week."""
    assert not re.search(r"\binput_tokens\b(?!\s+AS)", LLM_TOKENS_TODAY_SQL.split("AS")[0]), \
        "the phantom identifier must not be selected"
    for real in ("input_tok", "output_tok", "cache_creation_tok", "cache_read_tok"):
        assert re.search(rf"SUM\({real}\)", LLM_TOKENS_TODAY_SQL), (
            f"{real} must be summed -- omitting the cache pair under-reports 26x")


def test_the_sweep_can_now_see_this_file():
    """The blindness itself, pinned. An f-string here resolves ZERO literals and
    makes the file invisible to the unknown-column sweep -- which is how this
    defect survived while 82.39's twin was caught."""
    from backend.db import schema_oracle as so

    literals = so.extract_sql_literals(API_FILE)
    assert literals, (
        "extract_sql_literals resolved NOTHING for this file -- the SQL has "
        "probably become an f-string again, hiding it from the sweep")
    assert any(TABLE in text for _ln, text in literals), (
        "the fully-qualified table must survive into the resolved literal")

    tree = ast.parse(API_FILE.read_text(encoding="utf-8"))
    assigned = [n for n in ast.walk(tree) if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "LLM_TOKENS_TODAY_SQL"
                        for t in n.targets)]
    assert len(assigned) == 1 and isinstance(assigned[0].value, ast.Constant), (
        "LLM_TOKENS_TODAY_SQL must be a plain literal, not a JoinedStr")


# ---------------------------------------------------------------------------
# Criterion 2 -- and why "non-null" is not enough
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _LIVE, reason="operator opt-out via PYFIN_SKIP_LIVE_BQ=1")
def test_repaired_query_returns_a_POSITIVE_total_on_a_day_with_traffic():
    """CRITERION 2, exceeded deliberately.

    The criterion asks for a NON-NULL total, which this query cannot fail to
    produce: no GROUP BY plus COALESCE means one always-non-null row. So this
    asserts a POSITIVE total with a `calls > 0` precondition, on a FIXED day.
    """
    sql = LLM_TOKENS_TODAY_SQL.replace("DATE(ts) = CURRENT_DATE()", f"DATE(ts) = '{BUSY_DAY}'")
    assert sql != LLM_TOKENS_TODAY_SQL, "the window substitution must not be a no-op"
    row = next(iter(_bq().query(sql).result(timeout=60)))

    assert row["calls"] > 0, (
        f"fixture precondition: {BUSY_DAY} must have traffic, got {row['calls']} "
        "calls -- re-measure and move BUSY_DAY rather than loosening this")
    assert row["tokens"] > 0, "a day with calls must report a positive token total"
    assert row["input_tokens"] > 0 and row["output_tokens"] > 0

    # The 26x finding, pinned: the cache pair must be INCLUDED in the total.
    naive = row["input_tokens"] + row["output_tokens"]
    assert row["tokens"] > naive, (
        "the total must include cache tokens; if these are equal the projection "
        "has silently reverted to input+output and under-reports")
    assert row["tokens"] == (naive + row["cache_creation_tokens"]
                             + row["cache_read_tokens"])


@pytest.mark.skipif(not _LIVE, reason="operator opt-out via PYFIN_SKIP_LIVE_BQ=1")
def test_a_zero_traffic_day_still_returns_non_null_which_is_why_that_is_no_evidence():
    """Pins the vacuity the criterion's wording would have accepted."""
    sql = LLM_TOKENS_TODAY_SQL.replace("DATE(ts) = CURRENT_DATE()", "DATE(ts) = '1999-01-01'")
    row = next(iter(_bq().query(sql).result(timeout=60)))
    assert row["calls"] == 0, "fixture precondition: a day with no traffic"
    assert row["tokens"] == 0 and row["tokens"] is not None, (
        "non-null on an EMPTY day -- which is why 'non-null' proves nothing")


# ---------------------------------------------------------------------------
# Criterion 3 -- every selected identifier, DERIVED and checked
# ---------------------------------------------------------------------------


def _selected_identifiers(sql: str) -> set[str]:
    """Column identifiers a query SELECTS, excluding aliases it invents.

    Aliases are the classic false positive -- the 82.54 research gate's own
    regex sweep produced NINE false positives out of ten hits, and 82.39's
    sweep was defeated by `AS pnl`. So `AS <name>` is stripped first.
    """
    select_part = re.split(r"\bFROM\b", sql, flags=re.I)[0]
    # Remove the `AS <alias>` TOKENS, then read what is left. Do NOT also filter
    # by an alias NAME set: a name can be both a read and an alias -- in the
    # pre-fix query `COALESCE(SUM(input_tokens), 0) AS input_tokens` is exactly
    # that -- and a name-set filter erases the genuine read along with the
    # alias, making the derivation blind to the very defect it exists to find.
    # My own recall test caught that; precision bought with recall is not
    # precision. This is the same substitution `schema_oracle` uses.
    stripped = re.sub(r"\bAS\s+[A-Za-z_][A-Za-z0-9_]*", " ", select_part, flags=re.I)
    # {1,} not {2,}: the 82.54 Q/A found the 3-character floor blind to `ok` and
    # `ts`, BOTH real columns of this very table. No such name is in the current
    # SELECT list, so the criterion held -- but the docstring's universal claim
    # would have over-reached by exactly one character class.
    idents = set(re.findall(r"\b([a-z_][a-z0-9_]{1,})\b", stripped))
    return {i for i in idents if i.upper() not in {"SELECT", "SUM", "COUNT", "COALESCE", "AS", "AND", "OR"}}


def test_every_selected_identifier_exists_in_the_live_schema():
    """CRITERION 3. The derived set is asserted NON-EMPTY -- an empty derivation
    would make every projection 'valid'."""
    derived = _selected_identifiers(LLM_TOKENS_TODAY_SQL)
    assert derived, "the identifier derivation resolved NOTHING -- vacuous"
    assert derived == {"input_tok", "output_tok", "cache_creation_tok", "cache_read_tok"}, (
        f"derived {sorted(derived)}")

    if _LIVE:
        live = {f.name for f in _bq().get_table(TABLE).schema}
        assert live, "the live schema resolved nothing"
        missing = derived - live
        assert not missing, f"identifiers absent from the live schema: {sorted(missing)}"


def test_the_derivation_excludes_aliases_a_RECALL_TEST():
    """The recall test the gate demanded: a method that flagged aliases would
    report false defects, and one that flagged nothing would report false
    health. Both directions checked."""
    derived = _selected_identifiers(LLM_TOKENS_TODAY_SQL)
    for alias in ("tokens", "calls", "input_tokens", "output_tokens"):
        assert alias not in derived, (
            f"{alias} is an ALIAS this query invents, not a column it reads")
    # ...and the method must still SEE a real phantom column when one exists.
    phantom = _selected_identifiers(PRE_FIX_SQL)
    assert "input_tokens" in phantom, (
        "on the PRE-FIX query, input_tokens is a genuine read and must be seen "
        "-- if this fails the derivation is blind, not precise")


# ---------------------------------------------------------------------------
# Criterion 4 -- a failed call is operator-visible
# ---------------------------------------------------------------------------


def test_failed_fetch_emits_an_operator_alert():
    """CRITERION 4, driving the production function with the client raising."""
    boom = RuntimeError("400 Unrecognized name: nope")
    with patch("google.cloud.bigquery.Client", side_effect=boom):
        with patch(ALERT_TARGET, autospec=True) as alert:
            out = _fetch_llm_tokens_today()

    assert out == (None, None, None), "the fetch must stay fail-open"
    assert alert.call_count == 1, f"expected one alert, got {alert.call_count}"
    kw = alert.call_args.kwargs
    assert kw["severity"] == "P1", (
        "only P0/P1 reach the bot-token fallback while slack_webhook_url is "
        f"empty; a {kw['severity']} would be logged and dropped")
    assert kw["source"] == "cost_budget_api"
    assert kw["error_type"] == "llm_tokens_fetch_failed"
    assert "Unrecognized name" in kw["details"]["error"]


@pytest.mark.skipif(not _LIVE, reason="operator opt-out via PYFIN_SKIP_LIVE_BQ=1")
def test_a_successful_fetch_emits_NO_alert():
    """The negative control -- without it the guard passes by always firing."""
    with patch(ALERT_TARGET, autospec=True) as alert:
        tokens, calls, parts = _fetch_llm_tokens_today()
    assert tokens is not None and calls is not None, (
        "fixture precondition: the live fetch really succeeded")
    # The breakdown must actually be RETURNED, not merely computed in SQL -- an
    # earlier revision claimed it was exposed while the endpoint returned one
    # conflated number.
    assert parts is not None and set(parts) == {
        "input", "output", "cache_creation", "cache_read"}, parts
    assert tokens == sum(parts.values()), (
        f"the total must equal the sum of its exposed components: "
        f"{tokens} vs {sum(parts.values())}")
    assert alert.call_count == 0, f"a clean fetch must be silent: {alert.call_args_list}"


def test_the_ENDPOINT_returns_the_breakdown_VALUES():
    """CLOSES TWO HOLES THE CYCLE-2 Q/A FOUND, and it is the only test in this
    file that calls the endpoint at all.

    Hole 1: the guard I wrote to close cycle-1's F2 asserted only that the four
    names exist in `CostBudgetToday.model_fields` -- a CLASS-LEVEL SCHEMA FACT,
    true of every possible response. Keeping the field while ceasing to populate
    it survived. The guard I added to fix a stop-one-seam-short finding stopped
    one seam short.

    Hole 2: the HTTP 500 that shipped -- a second `tokens, calls = await ...`
    unpack my `str.replace(..., 1)` missed -- had NO automated coverage
    whatsoever. Measured by the Q/A: zero tests anywhere referenced
    `get_cost_budget_today`, so unpack arity, kwarg wiring and breakdown
    population were all live-check-only.

    This drives the real endpoint with the fetch patched, and asserts VALUES.
    Reverting the unpack to two targets, or dropping any component from the
    construction, both fail here.
    """
    import asyncio

    from backend.api import cost_budget_api as api
    from backend.services.api_cache import get_api_cache

    def _fresh():
        # The endpoint memoises for 60s; a cached hit would make this guard
        # assert nothing about the code under test.
        get_api_cache().invalidate(api._CACHE_KEY)

    parts = {"input": 11, "output": 22, "cache_creation": 33, "cache_read": 44}
    with patch.object(api, "_fetch_llm_tokens_today", return_value=(110, 5, parts)):
        _fresh()
        resp = asyncio.run(api.get_cost_budget_today())

    assert resp.llm_tokens_today == 110
    assert resp.llm_input_tokens_today == 11, (
        "the component is on the model but not populated -- schema membership is "
        "not value flow")
    assert resp.llm_output_tokens_today == 22
    assert resp.llm_cache_creation_tokens_today == 33
    assert resp.llm_cache_read_tokens_today == 44
    assert (resp.llm_input_tokens_today + resp.llm_output_tokens_today
            + resp.llm_cache_creation_tokens_today
            + resp.llm_cache_read_tokens_today) == resp.llm_tokens_today

    # The fail-open shape must also survive the endpoint, since that is the path
    # the pre-fix module actually served.
    with patch.object(api, "_fetch_llm_tokens_today", return_value=(None, None, None)):
        _fresh()
        degraded = asyncio.run(api.get_cost_budget_today())
    assert degraded.llm_tokens_today is None
    assert degraded.llm_input_tokens_today is None


def test_the_live_bq_gate_is_not_silently_empty():
    """5 of this file's tests are env-conditional on PYFIN_SKIP_LIVE_BQ, and
    they include BOTH criterion-1 dry runs and BOTH criterion-2 fixtures. With
    the opt-out set, the immutable verification command exits 0 with those
    criteria never executed -- the skip-trapdoor class this project has been bitten
    by before. This test is NOT skippable, so the trapdoor is at least visible.
    """
    assert _LIVE, (
        "PYFIN_SKIP_LIVE_BQ=1 is set: criteria 1 and 2 are NOT being executed by "
        "this run, so a green verification command is not evidence for them")


def test_the_alert_patch_target_is_the_only_one_that_works():
    """`_alert_llm_tokens_failed` imports the emitter FUNCTION-LOCALLY, so the
    tempting module-scope target does not exist."""
    import backend.api.cost_budget_api as api

    assert not hasattr(api, "raise_cron_alert_sync")
    import backend.services.observability.alerting as alerting
    assert hasattr(alerting, "raise_cron_alert_sync")


def test_the_spend_py_alert_defect_is_QUEUED_not_silently_inherited():
    """A THIRD live defect the gate found: `spend.py` calls the emitter with
    `detail=` against a `details=` signature, so it raises TypeError into a
    swallowing except and has NEVER fired. Out of scope here -- but it must not
    be inherited silently, so this guard fails if it is neither fixed nor
    queued."""
    import json

    spend = (REPO / "backend" / "services" / "observability" / "spend.py").read_text(
        encoding="utf-8")
    still_broken = re.search(r"raise_cron_alert_sync\([^)]*\bdetail=", spend, re.S)
    if not still_broken:
        return  # fixed -- nothing to queue

    mp = json.loads((REPO / ".claude/masterplan.json").read_text(encoding="utf-8"))
    steps = [s for ph in mp["phases"] for s in ph.get("steps", [])]
    assert steps, "no masterplan steps parsed -- the check would be vacuous"
    open_steps = [s for s in steps if s.get("status") not in {"done", "dropped", "superseded"}]
    owner = [s for s in open_steps
             if "spend.py" in s.get("name", "")
             and "detail" in s.get("name", "")
             and s.get("verification", {}).get("criteria")]
    assert owner, (
        "spend.py still passes `detail=` to raise_cron_alert_sync and NO open "
        "step with criteria owns it -- a P1-class alert that can never fire")
