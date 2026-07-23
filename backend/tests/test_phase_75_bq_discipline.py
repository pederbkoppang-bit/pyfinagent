"""phase-75.9: BigQuery fail-closed dedup, parameterization, 30s-timeout
sweep, cost guard. Offline (no live BQ) -- all queries are mocked.

Six immutable criteria (verbatim from .claude/masterplan.json step 75.9),
one test group per criterion:

1. ingest_prices fail-closed dedup: a mocked client.query() that raises
   must produce ZERO insert_rows_json calls, the error must surface
   (raise), and it must be logged. The ingest fixture is deliberately
   NON-EMPTY (real-shaped OHLCV rows) -- an empty fixture would make "zero
   inserts" vacuously true even under the OLD fail-open bug, since there
   would be nothing to insert either way (see feedback_measure_dont_
   assert_claims / feedback_mutation_test_guards_and_fixtures). A
   companion test proves the fix does not regress the empty-*result*
   (first-run / cold table) path: that must still insert.
2. Parameterization: source-scan for the ScalarQueryParameter/
   ArrayQueryParameter constructs AND the explicit absence of the old
   f-string interpolation patterns (a present param + a surviving f-string
   would be a dead param -- vacuous pass).
3. Timeout sweep: an AST scan (not a loose regex) over bigquery_client.py,
   12 external files, and 13 migration files, requiring every `.result(`
   call to carry an explicit `timeout=` keyword, with a hard failure (not
   a silent skip) if any enumerated path does not exist on disk.
4. Cost-guard factory: the constant value AND a behavioral adoption check
   (job_config.maximum_bytes_billed reaches a real bigquery_client method).
5. skill_optimizer degraded-not-crashed (no bare except:pass, AST-checked)
   + slot_accounting module-level client reuse with timeout=30.
6. get_bq_client() lru-cached singleton identity + call-site (not just
   import) scan of the three api files.
"""
from __future__ import annotations

import ast
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


# ─────────────────────────────────────────────────────────────────────
# Shared fixtures / helpers
# ─────────────────────────────────────────────────────────────────────

class _FakeSettings:
    """Minimal settings double -- only the attributes DataIngestionService
    / BigQueryClient methods under test actually read."""

    def __init__(self):
        self.gcp_project_id = "proj"
        self.bq_dataset_reports = "ds"
        self.bq_table_reports = "analysis_results"
        self.bq_dataset_outcomes = "ds"
        self.bq_table_outcomes = "outcome_tracking"
        self.gcp_credentials_json = ""


def _make_yf_multiindex_frame(ticker: str) -> pd.DataFrame:
    """A NON-EMPTY yfinance-shaped OHLCV frame (group_by='ticker' MultiIndex
    columns). Load-bearing for crit-1: a blank/empty frame would make
    ingest_prices short-circuit at `if data is None or data.empty: continue`
    BEFORE the dedup query is ever called -- see mutation #2 below, which
    exercises exactly that trap."""
    idx = pd.date_range("2024-01-02", periods=3, freq="B")
    cols = pd.MultiIndex.from_tuples([
        (ticker, "Open"), (ticker, "High"), (ticker, "Low"),
        (ticker, "Close"), (ticker, "Volume"),
    ])
    data = [
        [100.0, 105.0, 99.0, 104.0, 1000],
        [101.0, 106.0, 100.0, 105.0, 1100],
        [102.0, 107.0, 101.0, 106.0, 1200],
    ]
    return pd.DataFrame(data, index=idx, columns=cols)


# ─────────────────────────────────────────────────────────────────────
# Criterion 1: fail-closed dedup
# ─────────────────────────────────────────────────────────────────────

def test_crit1_ingest_prices_dedup_failure_raises_and_blocks_insert(caplog):
    """The core fix: a dedup query failure must surface AND must not have
    inserted anything (fail-closed, not fail-open-and-duplicate)."""
    from backend.backtest.data_ingestion import DataIngestionService

    bq_client = MagicMock()
    bq_client.query.side_effect = RuntimeError("BQ 500: simulated dedup query failure")
    svc = DataIngestionService(bq_client, _FakeSettings())

    with patch(
        "backend.backtest.data_ingestion.yf.download",
        return_value=_make_yf_multiindex_frame("AAPL"),
    ):
        with caplog.at_level(logging.ERROR, logger="backend.backtest.data_ingestion"):
            with pytest.raises(RuntimeError):
                svc.ingest_prices(["AAPL"], "2024-01-01", "2024-01-10")

    bq_client.insert_rows_json.assert_not_called()
    assert any(
        "dedup" in rec.message.lower() and rec.levelno == logging.ERROR
        for rec in caplog.records
    ), "the dedup failure must be logged at ERROR (fail-closed, not silent)"


def test_crit1_ingest_fundamentals_dedup_failure_raises_and_blocks_insert():
    """Same fail-closed contract for the fundamentals dedup helper. The
    dedup call happens before any yf.Ticker() call, so no yfinance mock is
    needed here -- the exception must surface before any download starts."""
    from backend.backtest.data_ingestion import DataIngestionService

    bq_client = MagicMock()
    bq_client.query.side_effect = RuntimeError("BQ 500: simulated dedup query failure")
    svc = DataIngestionService(bq_client, _FakeSettings())

    with pytest.raises(RuntimeError):
        svc.ingest_fundamentals(["AAPL"])

    bq_client.insert_rows_json.assert_not_called()


def test_crit1_ingest_prices_success_empty_dedup_still_inserts():
    """Companion test: a SUCCESSFUL query returning zero existing rows
    (first-run / cold table) is NOT the failure path -- it must still
    insert every row. This is the exact distinction the fix must preserve
    (see data_ingestion.py's _get_existing_price_dates docstring)."""
    from backend.backtest.data_ingestion import DataIngestionService

    bq_client = MagicMock()
    bq_client.query.return_value.result.return_value = []  # dedup query succeeds, 0 existing rows
    bq_client.insert_rows_json.return_value = []  # no BQ insert errors

    svc = DataIngestionService(bq_client, _FakeSettings())
    with patch(
        "backend.backtest.data_ingestion.yf.download",
        return_value=_make_yf_multiindex_frame("AAPL"),
    ):
        total = svc.ingest_prices(["AAPL"], "2024-01-01", "2024-01-10")

    assert total == 3
    bq_client.insert_rows_json.assert_called_once()


def test_crit1_stub_mutation_blank_fixture_makes_test_vacuous(caplog):
    """MUTATION #2 (mutate the STUB, not the source): if the yfinance
    fixture were blank/empty, ingest_prices short-circuits at
    `if data is None or data.empty: continue` BEFORE `_get_existing_
    price_dates` is ever called -- so even with client.query.side_effect
    set to raise, NO exception would surface and NO insert would happen
    either, "passing" crit-1's assertions for the wrong reason. This test
    proves that trap is real by reproducing it directly, which is exactly
    why the non-empty fixture above is load-bearing rather than
    decorative."""
    from backend.backtest.data_ingestion import DataIngestionService

    bq_client = MagicMock()
    bq_client.query.side_effect = RuntimeError("BQ 500: simulated dedup query failure")
    svc = DataIngestionService(bq_client, _FakeSettings())

    empty_frame = pd.DataFrame()
    with patch("backend.backtest.data_ingestion.yf.download", return_value=empty_frame):
        # With a blank fixture, the RuntimeError from client.query is NEVER
        # reached -- ingest_prices returns 0 silently instead of raising.
        total = svc.ingest_prices(["AAPL"], "2024-01-01", "2024-01-10")

    assert total == 0
    bq_client.insert_rows_json.assert_not_called()
    bq_client.query.assert_not_called()  # the dedup query is never even attempted


# ─────────────────────────────────────────────────────────────────────
# Criterion 2: parameterization (present + absent, symmetric)
# ─────────────────────────────────────────────────────────────────────

_BQ_CLIENT_SRC = (REPO_ROOT / "backend/db/bigquery_client.py").read_text(encoding="utf-8")
_DATA_INGESTION_SRC = (REPO_ROOT / "backend/backtest/data_ingestion.py").read_text(encoding="utf-8")


def test_crit2_get_agent_memories_uses_parameters_not_fstrings():
    assert "def get_agent_memories" in _BQ_CLIENT_SRC
    # Present: parameterized construction.
    assert 'bigquery.ScalarQueryParameter("agent_type"' in _BQ_CLIENT_SRC
    assert 'bigquery.ScalarQueryParameter("limit", "INT64", int(limit))' in _BQ_CLIENT_SRC
    # Absent: the old f-string interpolation of agent_type/limit VALUES.
    # (A present param + a surviving f-string would be a dead param --
    # this is the "symmetric present+absent" guard from the brief.)
    assert "WHERE agent_type = '{agent_type}'" not in _BQ_CLIENT_SRC
    assert "LIMIT {int(limit)}" not in _BQ_CLIENT_SRC


def test_crit2_data_ingestion_ticker_lists_use_array_parameter_unnest():
    # Present: ArrayQueryParameter + IN UNNEST(@tickers), mirroring cache.py.
    assert _DATA_INGESTION_SRC.count('bigquery.ArrayQueryParameter("tickers", "STRING"') == 2
    assert _DATA_INGESTION_SRC.count("IN UNNEST(@tickers)") == 2
    # Absent: the old manual-quote-join f-string pattern for both dedup helpers.
    assert '", ".join(f"\'{t}\'"' not in _DATA_INGESTION_SRC
    assert "IN ({ticker_list})" not in _DATA_INGESTION_SRC


# ─────────────────────────────────────────────────────────────────────
# Criterion 3: timeout sweep (AST scan, hard-fails on missing paths)
# ─────────────────────────────────────────────────────────────────────

BIGQUERY_CLIENT_FILE = "backend/db/bigquery_client.py"

# 12 external files (phase-75.9 research_brief.md D1-D9 drift corrections).
# cost_budget_watcher.py is a documented PHANTOM -- it has zero .result(
# sites, stays in the scan list, and is expected to pass vacuously at 0.
EXTERNAL_FILES = [
    "backend/services/paper_trader.py",
    "backend/services/cycle_health.py",
    "backend/metrics/sortino.py",
    "backend/api/paper_trading.py",
    "backend/api/performance_api.py",
    "backend/services/pead_signal.py",
    "backend/services/sector_calendars.py",
    "backend/agents/skill_optimizer.py",
    "backend/slack_bot/jobs/cost_budget_watcher.py",  # phantom, 0 sites expected
    "backend/autoresearch/slot_accounting.py",
    "backend/api/harness_autoresearch.py",
    "backend/api/monthly_approval_api.py",
]

# 13 migration files (D1: measured reality, not the masterplan's stale "12").
MIGRATION_FILES = [
    "scripts/migrations/add_efficiency_snapshots.py",
    "scripts/migrations/add_external_flow_today_column.py",
    "scripts/migrations/add_round_trip_schema.py",
    "scripts/migrations/add_session_budget_to_llm_call_log.py",
    "scripts/migrations/add_ticker_to_llm_call_log.py",
    "scripts/migrations/create_alpha_velocity_table.py",
    "scripts/migrations/create_data_source_events_table.py",
    "scripts/migrations/create_directive_versions_table.py",
    "scripts/migrations/create_historical_fx_rates_table.py",
    "scripts/migrations/create_options_snapshots_table.py",
    "scripts/migrations/create_promoted_strategies_table.py",
    "scripts/migrations/create_strategy_deployments_view.py",
    "scripts/migrations/phase_32_1_add_stop_advanced_at_R.py",
]

ALL_SCANNED_FILES = [BIGQUERY_CLIENT_FILE] + EXTERNAL_FILES + MIGRATION_FILES


def _find_untimed_results(relpath: str) -> list[int]:
    """Return line numbers of `.result(` calls in `relpath` missing an
    explicit `timeout=` keyword argument.

    Hard-fails (raises) if the path does not exist -- a moved/renamed file
    must turn this scan red, never silently shrink the denominator.

    Excludes ThreadPoolExecutor `future.result()` calls: this project's
    established convention names those futures `future` (see
    api/paper_trading.py's `for future in as_completed(futures): ...
    info = future.result()`) -- they are not BigQuery jobs and are
    explicitly out of scope (see contract_75.9.md "Explicitly NOT in
    scope"). The exclusion is identifier-based (not line-number-based) so
    it survives line drift.
    """
    path = REPO_ROOT / relpath
    if not path.exists():
        raise FileNotFoundError(f"phase-75.9 timeout scan: expected file missing: {relpath}")
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src, filename=relpath)
    missing: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == "result"):
            continue
        if isinstance(func.value, ast.Name) and func.value.id == "future":
            continue  # ThreadPoolExecutor future.result() -- not a BQ job
        has_timeout = any(kw.arg == "timeout" for kw in node.keywords)
        if not has_timeout:
            missing.append(node.lineno)
    return missing


def test_crit3_scanned_file_list_is_non_empty():
    """Guards the `all([])` trap: an empty file list would make the loop
    below vacuously pass every file (there being none to fail on)."""
    assert len(ALL_SCANNED_FILES) == 26
    assert all(ALL_SCANNED_FILES)


def test_crit3_scan_hard_fails_on_missing_path():
    """Mutation #5: pointing the scan at a nonexistent path must ERROR,
    never silently skip-green."""
    with pytest.raises(FileNotFoundError):
        _find_untimed_results("backend/db/DOES_NOT_EXIST_phase_75_9.py")


@pytest.mark.parametrize("relpath", ALL_SCANNED_FILES)
def test_crit3_every_result_call_has_timeout(relpath):
    missing = _find_untimed_results(relpath)
    assert missing == [], f"{relpath}: untimed .result( call(s) at line(s) {missing}"


def test_crit3_phantom_cost_budget_watcher_has_zero_result_sites():
    """Documented phantom (D5): no .result()/.query() at all in this file.
    Scanning it is a vacuous-but-intentional pass, not an oversight."""
    src = (REPO_ROOT / "backend/slack_bot/jobs/cost_budget_watcher.py").read_text(encoding="utf-8")
    assert ".result(" not in src


# ─────────────────────────────────────────────────────────────────────
# Criterion 4: cost-guard factory (value + behavioral adoption)
# ─────────────────────────────────────────────────────────────────────

def test_crit4_factory_default_value():
    from backend.db.bigquery_client import MAX_BYTES_BILLED_DEFAULT
    assert MAX_BYTES_BILLED_DEFAULT == 5368709120  # 5 GiB = 5 * 1024**3


def test_crit4_factory_produces_job_config_with_cap():
    bq = _bq_with_mock_client()
    job_config = bq._job_config()
    assert job_config.maximum_bytes_billed == 5368709120


def test_crit4_factory_adoption_behavioral():
    """A real bigquery_client method (get_recent_reports) must pass a
    job_config through to client.query() carrying the cap -- not just a
    constant sitting unused (the brief's "tautology assertion" trap)."""
    bq = _bq_with_mock_client()
    bq.reports_table = "proj.ds.reports"
    bq.client.query.return_value.result.return_value = []

    bq.get_recent_reports(limit=5)

    _args, kwargs = bq.client.query.call_args
    job_config = kwargs["job_config"]
    assert job_config.maximum_bytes_billed == 5368709120


def _bq_with_mock_client():
    from backend.db.bigquery_client import BigQueryClient
    bq = BigQueryClient.__new__(BigQueryClient)
    bq.client = MagicMock()
    bq.settings = _FakeSettings()
    return bq


# ─────────────────────────────────────────────────────────────────────
# Criterion 5: skill_optimizer degraded (no bare pass) + slot_accounting
# module-level client reuse
# ─────────────────────────────────────────────────────────────────────

def test_crit5_skill_optimizer_no_bare_except_pass():
    src = (REPO_ROOT / "backend/agents/skill_optimizer.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    bare_pass_lines = [
        node.lineno for node in ast.walk(tree)
        if isinstance(node, ast.ExceptHandler)
        and len(node.body) == 1
        and isinstance(node.body[0], ast.Pass)
    ]
    assert bare_pass_lines == [], f"bare except:pass survives at line(s) {bare_pass_lines}"


def test_crit5_skill_optimizer_outcomes_failure_logs_and_degrades(caplog):
    from backend.agents.skill_optimizer import SkillOptimizer

    opt = SkillOptimizer.__new__(SkillOptimizer)
    opt.bq = MagicMock()
    opt.bq.reports_table = "proj.ds.reports"
    opt.bq.outcomes_table = "proj.ds.outcomes"
    # First query (reports) succeeds with an empty list; second query
    # (outcomes) raises -- exercises exactly the block that used to be a
    # bare `except: pass`.
    ok_result = MagicMock()
    ok_result.result.return_value = []
    fail_result = MagicMock()
    fail_result.result.side_effect = RuntimeError("simulated outcomes query failure")
    opt.bq.client.query.side_effect = [ok_result, fail_result]

    with caplog.at_level(logging.WARNING, logger="backend.agents.skill_optimizer"):
        results = opt.analyze_agent_performance()

    assert any("outcomes" in rec.message.lower() for rec in caplog.records), (
        "the outcomes-query failure must be logged (was a silent bare pass)"
    )
    # Degraded, not crashed: every agent falls back to the neutral default.
    assert results, "should still return a (degraded) per-agent result list"
    assert all(r["accuracy"] == 0.5 and r["sample_size"] == 0 for r in results)


def test_crit5_slot_accounting_reuses_module_level_client(monkeypatch):
    from backend.autoresearch import slot_accounting

    monkeypatch.setattr(slot_accounting, "_module_client", None)

    class _FakeJob:
        def __init__(self, rows):
            self._rows = rows
            self.result_calls: list[dict] = []

        def result(self, **kwargs):
            self.result_calls.append(kwargs)
            return self._rows

    class _FakeClient:
        instances: list["_FakeClient"] = []

        def __init__(self, project=None):
            _FakeClient.instances.append(self)
            self.jobs: list[_FakeJob] = []

        def insert_rows_json(self, table, rows):
            return []

        def query(self, sql, job_config=None):
            job = _FakeJob([[7]])
            self.jobs.append(job)
            return job

    _FakeClient.instances = []
    monkeypatch.setattr("google.cloud.bigquery.Client", _FakeClient)

    slot_accounting._default_bq_insert("t", [{"a": 1}])
    count = slot_accounting._default_bq_query_count("SELECT 1", {"week_iso": "2026-W01"})

    assert len(_FakeClient.instances) == 1, (
        "bigquery.Client must be constructed exactly once and reused across "
        "_default_bq_insert AND _default_bq_query_count"
    )
    assert count == 7
    last_job = _FakeClient.instances[0].jobs[-1]
    assert last_job.result_calls == [{"timeout": 30}]


# ─────────────────────────────────────────────────────────────────────
# Criterion 6: get_bq_client() lru-cached singleton + call-site scan
# ─────────────────────────────────────────────────────────────────────

def test_crit6_get_bq_client_is_singleton():
    from backend.db.bigquery_client import get_bq_client

    with patch("backend.db.bigquery_client.bigquery.Client") as mock_cls, \
         patch("backend.db.bigquery_client.get_settings", return_value=_FakeSettings()):
        mock_cls.return_value = MagicMock()
        get_bq_client.cache_clear()
        try:
            a = get_bq_client()
            b = get_bq_client()
            assert a is b
            assert mock_cls.call_count == 1
        finally:
            get_bq_client.cache_clear()


@pytest.mark.parametrize("relpath", [
    "backend/api/paper_trading.py",
    "backend/api/performance_api.py",
    "backend/api/reports.py",
])
def test_crit6_call_site_actually_calls_get_bq_client(relpath):
    """Import-only would falsely pass a naive `"get_bq_client" in src` scan
    (e.g. an unused import). Require an actual CALL -- `get_bq_client(`
    with the open paren only matches call sites, never the bare-name
    import statement."""
    src = (REPO_ROOT / relpath).read_text(encoding="utf-8")
    assert "import" in src and "get_bq_client" in src, f"{relpath} does not import get_bq_client"
    call_count = src.count("get_bq_client(")
    assert call_count >= 1, f"{relpath} imports get_bq_client but never calls it"
