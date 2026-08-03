"""phase-82.0 -- macro ingestion durable repair.

Guards the six immutable criteria for step 82.0. Two of these tests are written
so they FAIL against the pre-fix tree (there was no scheduled caller at all, and
the staleness gate took a global max across series); a guard that cannot fail
does not count as a guard.
"""
from __future__ import annotations

import json
from datetime import date, timedelta

import pytest

from backend.backtest import cache as cache_mod
from backend.backtest.data_ingestion import DataIngestionService


# ── stubs ────────────────────────────────────────────────────────────

class _StubScheduler:
    """Minimal APScheduler surface: records add_job calls."""

    def __init__(self):
        self.jobs = []

    def add_job(self, func, **kwargs):
        self.jobs.append({"func": func, **kwargs})
        return kwargs.get("id")


class _RaisingQueryClient:
    """BQ client whose query() always raises -- for the fail-closed test."""

    def query(self, *a, **k):
        raise RuntimeError("simulated BigQuery outage")


class _RowsClient:
    """BQ client returning a fixed row list from query().result()."""

    def __init__(self, rows):
        self._rows = rows

    def query(self, *a, **k):
        rows = self._rows

        class _Job:
            def result(self, *a, **k):
                return rows

        return _Job()


class _Settings:
    """Only the attributes the ingestion service reads."""

    def __init__(self, **kw):
        self.backtest_start_date = "2018-01-01"
        self.backtest_end_date = "2025-12-31"
        self.macro_ingest_end_date = ""
        self.gcp_project_id = "test-project"
        self.bq_dataset_reports = "financial_reports"
        for k, v in kw.items():
            setattr(self, k, v)


def _svc(client=None, settings=None, receipts_dir=None):
    svc = DataIngestionService(client or _RowsClient([]), settings or _Settings())
    if receipts_dir is not None:
        # phase-82.0 cycle-3: never let a test write into the real operational
        # receipts ledger. Forged "ok" records there would erode the exact
        # distinguishability that criterion 5 exists to create.
        svc._receipts_dir_override = receipts_dir
    return svc


@pytest.fixture(autouse=True)
def _isolate_receipts(tmp_path, monkeypatch):
    """Belt-and-braces: any service built WITHOUT an explicit receipts_dir in
    this module still writes to a tmp dir, not handoff/logs/."""
    monkeypatch.setattr(
        DataIngestionService, "_receipts_dir_override", tmp_path / "receipts"
    )


# ── criterion 1: macro end date severed from backtest_end_date ───────

def test_macro_end_date_is_severed_from_backtest_end_date():
    """The defect: backtest_end_date leaked into the FRED observation_end param,
    so every ingest asked for data ending 2025-12-31 and inserted zero rows."""
    settings = _Settings(backtest_end_date="2025-12-31")
    resolved = _svc(settings=settings)._resolve_macro_end_date()

    assert resolved > "2025-12-31", (
        f"macro observation_end resolved to {resolved!r}, which is not later than "
        "the pinned backtest_end_date -- the coupling that froze the table is back"
    )
    assert resolved == date.today().isoformat()


def test_pinned_macro_end_date_is_honoured():
    """An explicit pin must still win, for reproducible backfills."""
    settings = _Settings(macro_ingest_end_date="2026-01-15")
    assert _svc(settings=settings)._resolve_macro_end_date() == "2026-01-15"


def test_run_full_ingestion_does_not_forward_backtest_end_date_to_macro(monkeypatch):
    """End-to-end guard on the actual call site, not just the helper."""
    svc = _svc(settings=_Settings(backtest_end_date="2025-12-31"))
    seen = {}

    def _fake_ingest_macro(start_date, end_date=None, fred_api_key=""):
        seen["end_date"] = end_date
        return 0

    monkeypatch.setattr(svc, "ingest_macro", _fake_ingest_macro)
    monkeypatch.setattr(svc, "ingest_prices", lambda *a, **k: 0)
    monkeypatch.setattr(svc, "ingest_fundamentals", lambda *a, **k: 0)
    # unrelated to this assertion: pulls in a top-level migrate_* module
    monkeypatch.setattr(svc, "_ensure_tables_exist", lambda: None)

    svc.run_full_ingestion(["AAPL"], "2018-01-01", "2025-12-31", "key")

    assert seen["end_date"] != "2025-12-31", (
        "run_full_ingestion forwarded backtest_end_date into ingest_macro -- "
        "this is the exact coupling that made every ingest a zero-row no-op"
    )


# ── criterion 2: a scheduled caller exists ───────────────────────────

def test_macro_ingest_cron_is_registered():
    """Pre-fix this fails at import: no macro cron module existed anywhere."""
    from backend.backtest.macro_cron import JOB_ID, register_macro_ingest_cron

    sched = _StubScheduler()
    returned = register_macro_ingest_cron(sched)

    assert returned == JOB_ID
    assert len(sched.jobs) == 1
    job = sched.jobs[0]
    assert job["id"] == JOB_ID
    assert job["trigger"] == "cron"
    assert job["replace_existing"] is True, (
        "replace_existing must be True or every restart duplicates the job "
        "in a persistent jobstore"
    )


def test_app_startup_registers_the_macro_cron():
    """The module existing is not enough -- main.py must actually wire it."""
    src = (
        __import__("pathlib").Path(__file__).resolve().parents[1] / "main.py"
    ).read_text(encoding="utf-8")
    assert "register_macro_ingest_cron" in src, (
        "backend/main.py does not register the macro ingest cron; the job would "
        "never run in the live app"
    )


# ── criterion 3: dedupe fails CLOSED ─────────────────────────────────

def test_get_existing_macro_fails_closed():
    """Fail-open here silently duplicates the whole table on a transient error:
    an empty dedupe set makes every fetched observation look new."""
    svc = _svc(client=_RaisingQueryClient())
    with pytest.raises(Exception) as exc:
        svc._get_existing_macro()
    assert "simulated BigQuery outage" in str(exc.value)


def test_ingest_macro_aborts_when_dedupe_fails(monkeypatch):
    """The fail-closed contract must propagate through the caller."""
    svc = _svc(client=_RaisingQueryClient())
    with pytest.raises(Exception):
        svc.ingest_macro("2018-01-01", None, "fake-key")


# ── criterion 4: per-series staleness SLA ────────────────────────────

def _macro_row(series_id, d):
    """Build a row in the PRODUCTION shape.

    phase-82.0 cycle-2: `historical_macro.date` is a STRING column
    (BQ schema ('date','STRING','REQUIRED')); live rows come back as
    e.g. '2023-07-03'. The cycle-1 fixture passed `datetime.date`, a type the
    production query NEVER returns, which made the criterion-4 guard green for
    every possible production state -- including a fully dead table. Fixtures
    must emit the type the real query emits or they cannot represent the
    failure they claim to catch.
    """
    return {"series_id": series_id, "value": 1.0, "date": d.isoformat()}


def test_per_series_sla_catches_dead_gdp_behind_a_live_daily_series(monkeypatch):
    """THE headline defect. A global MAX(date) across series let a live daily
    series (DGS10, updated every business day) satisfy the freshness gate on
    behalf of a GDP series that had been dead for years."""
    today = date.today()
    rows = [
        _macro_row("DGS10", today - timedelta(days=1)),      # fresh, SLA 5d
        _macro_row("T10Y2Y", today - timedelta(days=1)),     # fresh, SLA 5d
        _macro_row("GDP", today - timedelta(days=3000)),     # DEAD, SLA 225d
    ]
    assert all(isinstance(r["date"], str) for r in rows), (
        "fixture must use the production STRING date type"
    )
    monkeypatch.setattr(cache_mod, "_bq_client", _RowsClient(rows))
    monkeypatch.setattr(cache_mod, "_macro_full", {})

    loaded = cache_mod.preload_macro()

    assert loaded == 0, (
        "preload_macro cached a table containing a GDP series ~8 years stale, "
        "because a live DGS10 masked it under a global MAX(date)"
    )
    assert cache_mod._macro_full == {}


def test_per_series_sla_accepts_a_healthy_table(monkeypatch):
    """Mirror guard: the SLA must not condemn a CURRENT table. FRED dates
    monthly series to month-start and quarterly to quarter-start, so a healthy
    GDP row is routinely ~200 days old -- the old flat 35-day bound would have
    rejected a perfectly good table."""
    today = date.today()
    rows = [
        _macro_row("DGS10", today - timedelta(days=2)),
        _macro_row("T10Y2Y", today - timedelta(days=2)),
        _macro_row("FEDFUNDS", today - timedelta(days=60)),
        _macro_row("CPIAUCSL", today - timedelta(days=70)),
        _macro_row("UNRATE", today - timedelta(days=65)),
        _macro_row("UMCSENT", today - timedelta(days=60)),
        _macro_row("GDP", today - timedelta(days=200)),
    ]
    monkeypatch.setattr(cache_mod, "_bq_client", _RowsClient(rows))
    monkeypatch.setattr(cache_mod, "_macro_full", {})

    loaded = cache_mod.preload_macro()

    assert loaded == len(rows), (
        "a healthy table was refused -- check that the per-series SLA accounts "
        "for FRED start-of-period dating (GDP ~200d is normal, not stale)"
    )
    assert set(cache_mod._macro_full) == {r["series_id"] for r in rows}


def test_gate_is_not_vacuous_on_the_production_date_type(monkeypatch, caplog):
    """Regression pin for the cycle-1 FAIL.

    Q/A cycle-2 finding 4: asserting only `preload_macro() == 0` does NOT pin
    the vacuity, because the fail-closed branch added in the same diff returns
    0 too -- so a re-introduced isinstance bug would still be "killed" by the
    wrong mechanism and the test would pass for the wrong reason. Pin the
    DISCRIMINATING behaviour instead: the gate must have actually EVALUATED the
    series and reported it stale by name. A vacuous gate cannot produce that
    message; it would either cache, or refuse via the unparseable-date path
    whose message is different.
    """
    today = date.today()
    rows = [_macro_row("GDP", today - timedelta(days=3000))]
    monkeypatch.setattr(cache_mod, "_bq_client", _RowsClient(rows))
    monkeypatch.setattr(cache_mod, "_macro_full", {})

    with caplog.at_level("WARNING"):
        loaded = cache_mod.preload_macro()

    assert loaded == 0, (
        "a table whose only series is ~8 years stale was cached -- the freshness "
        "gate is vacuous on the production date type"
    )
    msg = " ".join(r.getMessage() for r in caplog.records)
    assert "past their per-series SLA" in msg and "GDP(newest=" in msg, (
        "refusal did not come from the per-series SLA evaluation -- the gate "
        f"either short-circuited or fell into the fail-closed path. Log: {msg!r}"
    )


def test_unparseable_dates_fail_closed(monkeypatch):
    """If the date column changes shape again, refuse rather than silently
    disabling the gate -- which is exactly how this defect survived for months."""
    rows = [{"series_id": "GDP", "value": 1.0, "date": "not-a-date"}]
    monkeypatch.setattr(cache_mod, "_bq_client", _RowsClient(rows))
    monkeypatch.setattr(cache_mod, "_macro_full", {})

    assert cache_mod.preload_macro() == 0
    assert cache_mod._macro_full == {}


def test_sla_table_covers_every_ingested_series():
    """A series with no SLA silently falls back to the flat 35-day bound, which
    is unsatisfiable for monthly/quarterly data -- it would permanently refuse."""
    from backend.backtest.data_ingestion import FRED_SERIES

    missing = set(FRED_SERIES) - set(cache_mod.MACRO_SERIES_MAX_AGE_DAYS)
    assert not missing, f"series ingested but absent from the SLA table: {sorted(missing)}"


# ── criterion 5: run receipt ─────────────────────────────────────────

def _receipt_lines(d):
    p = d / "macro_ingest_receipts.jsonl"
    if not p.exists():
        return []
    return [ln for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]


def test_receipt_written_on_zero_row_run(tmp_path):
    """MAX(ingested_at) only advances when rows are inserted, so a healthy
    no-op run is otherwise indistinguishable from a job that never ran.

    Deliberately does NOT monkeypatch _write_macro_receipt: asserting that a
    stub was called proves only that the caller calls the stub. This asserts a
    real file gains a well-formed record -- written to an ISOLATED directory,
    never the operational ledger.
    """
    d = tmp_path / "receipts"
    svc = _svc(receipts_dir=d)
    assert _receipt_lines(d) == []

    svc.ingest_macro("2018-01-01", None, "")  # no API key -> zero-row path

    lines = _receipt_lines(d)
    assert len(lines) == 1, "no run receipt was appended for a zero-row run"
    rec = json.loads(lines[0])
    assert rec["rows_inserted"] == 0
    assert rec["job"] == "macro_ingest"
    assert rec["outcome"] == "skipped_no_api_key"


def test_tests_do_not_write_to_the_operational_receipts_ledger():
    """Regression pin for Q/A cycle-2 finding 2: the suite previously appended
    forged 'ok' records to the real handoff/logs ledger (13 -> 37 lines in one
    evaluation). A ledger any pytest run can write into is not evidence."""
    import backend.backtest.data_ingestion as di
    from pathlib import Path as _P

    real = _P(di.__file__).resolve().parents[2] / "handoff" / "logs"
    before = len(_receipt_lines(real))

    svc = _svc(receipts_dir=None)  # autouse fixture still redirects it
    svc._write_macro_receipt(1, "ok", "2026-08-03")

    assert len(_receipt_lines(real)) == before, (
        "a test wrote into the operational receipts ledger at handoff/logs/"
    )


def test_receipt_is_valid_jsonl(tmp_path):
    d = tmp_path / "receipts"
    svc = _svc(receipts_dir=d)

    svc._write_macro_receipt(0, "ok", "2026-08-03")

    lines = _receipt_lines(d)
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["job"] == "macro_ingest"
    assert rec["observation_end"] == "2026-08-03"
    assert "rows_inserted" in rec and "outcome" in rec and "ts" in rec


# ── criterion 6: vintage stamp on written rows ───────────────────────

def test_ingested_rows_carry_a_vintage(monkeypatch):
    """Rows written without realtime_start can NEVER be retro-attributed --
    the information about when we first saw a value is gone permanently."""
    import backend.backtest.data_ingestion as di

    captured = {}

    class _Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"observations": [{"date": "2026-07-01", "value": "3.14"}]}

    class _Client:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def get(self, url):
            captured["url"] = url
            return _Resp()

    monkeypatch.setattr(di.httpx, "Client", lambda **k: _Client())
    monkeypatch.setattr(di, "FRED_SERIES", ["GDP"])

    inserted_rows = []

    class _InsertClient(_RowsClient):
        def insert_rows_json(self, table, rows):
            inserted_rows.extend(rows)
            return []

    svc = _svc(client=_InsertClient([]))
    svc.ingest_macro("2018-01-01", None, "fake-key")

    assert inserted_rows, "no rows were built"
    for r in inserted_rows:
        assert r.get("realtime_start"), f"row missing vintage stamp: {r}"
        assert r["realtime_start"] == date.today().isoformat()

    # and the URL must not carry the backtest cap
    assert "observation_end=2025-12-31" not in captured["url"]
