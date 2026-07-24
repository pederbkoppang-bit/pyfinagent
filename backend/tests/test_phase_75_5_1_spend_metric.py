"""phase-75.5.1: the $25/day budget guard gets an LLM-spend metric (arm a).

Before this step the breaker (`llm_client._check_cost_budget`) consumed
`fetch_spend()` = BigQuery bytes-billed x $6.25/TiB while every name around it
("Daily LLM-spend cap", CLAUDE.md's "$25/day LLM circuit breaker") promised LLM
spend. This suite pins the new `fetch_llm_spend()` (llm_call_log RAW tokens x
the LIVE MODEL_PRICING table, cache-aware, METERED rows only) and the flag
routing (`cost_budget_use_llm_spend_enabled`, default OFF = byte-identical).

Design notes for the guards (qa.md 4c):
  - Expected prices are re-derived INLINE from the imported MODEL_PRICING
    numbers -- never by calling the production pricing helper (that would be
    circular). A production formula change without a test update must fail.
  - The fake BQ client EVALUATES the filter predicates FROM THE SQL TEXT it
    receives (ok / provider != 'claude-code' / NOT LIKE 'cc_rail:%'). That
    makes the SQL load-bearing: deleting the CC-rail exclusion from the
    production query (the phantom-free-token crux) is a killable mutation,
    not a string the fake silently re-applies. The fake's discriminating
    power is itself self-tested (test_fake_client_honors_filter_absence),
    so neutering the fake is ALSO a killable mutation.
  - CC-rail fixtures cover BOTH row shapes (provider='claude-code' AND
    provider='anthropic' + agent='cc_rail:...') -- a fixture that cannot
    represent both categories does not count (75.2.1 lesson).
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path
from typing import ClassVar

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.agents.cost_tracker import MODEL_PRICING
from backend.services.observability import spend

# A fixed mid-month "now" makes the daily/monthly split deterministic (a
# wall-clock 1st-of-month would make month-start rows also be "today"). The
# fake evaluates the query's CURRENT_DATE() against this instant.
FIXED_NOW = dt.datetime(2026, 7, 24, 12, 0, 0, tzinfo=dt.timezone.utc)
TODAY = FIXED_NOW
EARLIER_IN_MONTH = FIXED_NOW - dt.timedelta(days=3)


def _row(model, provider="anthropic", agent="synthesis", ok=True, ts=TODAY,
         in_tok=0, out_tok=0, cw_tok=0, cr_tok=0):
    return {
        "model": model, "provider": provider, "agent": agent, "ok": ok,
        "ts": ts, "input_tok": in_tok, "output_tok": out_tok,
        "cache_creation_tok": cw_tok, "cache_read_tok": cr_tok,
    }


class FakeBQClient:
    """SQL-semantics-aware fake: applies ONLY the filter predicates that are
    present in the SQL text it receives, then reproduces the query's
    GROUP BY model + daily/monthly aggregation shape over the fixture rows."""

    rows: ClassVar[list[dict]] = []
    last_sql: ClassVar[str] = ""
    raise_exc: ClassVar[Exception | None] = None

    def __init__(self, project=None):
        self.project = project
        self._result_rows: list[dict] = []

    def query(self, sql, timeout=None):
        if FakeBQClient.raise_exc is not None:
            raise FakeBQClient.raise_exc
        FakeBQClient.last_sql = sql
        month_start = FIXED_NOW.replace(day=1, hour=0, minute=0, second=0,
                                        microsecond=0)
        agg: dict[str, dict] = {}
        for r in FakeBQClient.rows:
            if r["ts"] < month_start:
                continue  # the month window is structural to the query shape
            if "AND ok" in sql and not r["ok"]:
                continue
            if "provider != 'claude-code'" in sql and r["provider"] == "claude-code":
                continue
            if "NOT LIKE 'cc_rail:%'" in sql and r["agent"] is not None \
                    and str(r["agent"]).startswith("cc_rail:"):
                continue
            a = agg.setdefault(r["model"], {
                "model": r["model"], "d_in": 0, "d_out": 0, "d_cw": 0,
                "d_cr": 0, "m_in": 0, "m_out": 0, "m_cw": 0, "m_cr": 0,
            })
            is_today = r["ts"].date() == FIXED_NOW.date()
            for key, col in (("in", "input_tok"), ("out", "output_tok"),
                             ("cw", "cache_creation_tok"),
                             ("cr", "cache_read_tok")):
                a["m_" + key] += r[col]
                if is_today:
                    a["d_" + key] += r[col]
        self._result_rows = list(agg.values())
        return self

    def result(self):
        return self._result_rows


@pytest.fixture(autouse=True)
def _fake_bq(monkeypatch):
    FakeBQClient.rows = []
    FakeBQClient.last_sql = ""
    FakeBQClient.raise_exc = None
    spend.reset_spend_guard_status()
    import google.cloud.bigquery as bq
    monkeypatch.setattr(bq, "Client", FakeBQClient)
    yield
    spend.reset_spend_guard_status()


def _expected_usd(model, in_tok=0, out_tok=0, cw_tok=0, cr_tok=0):
    """Expected price re-derived INLINE from the REAL pricing table (cache
    read 0.1x input rate, cache write 2.0x) -- deliberately NOT a call into
    the production helper, so formula drift there fails here."""
    rin, rout = MODEL_PRICING[model]
    return (cr_tok * rin * 0.1 + cw_tok * rin * 2.0
            + in_tok * rin + out_tok * rout) / 1_000_000


# -- pricing against the REAL table (criterion 1) --------------------------

def test_metered_rows_priced_against_real_pricing_table():
    FakeBQClient.rows = [
        _row("claude-opus-4-8", in_tok=1_000_000, out_tok=200_000,
             cw_tok=50_000, cr_tok=400_000),
        _row("gemini-2.5-flash", provider="vertex", agent="enrichment",
             in_tok=3_000_000, out_tok=500_000),
    ]
    daily, monthly = spend.fetch_llm_spend()
    expected = (_expected_usd("claude-opus-4-8", 1_000_000, 200_000, 50_000, 400_000)
                + _expected_usd("gemini-2.5-flash", 3_000_000, 500_000))
    assert daily == pytest.approx(expected, rel=1e-9)
    assert monthly == pytest.approx(expected, rel=1e-9)
    assert expected > 0


def test_cache_tokens_are_priced_not_ignored():
    # The sovereign_api anti-pattern ignores cache columns; the guard must not.
    FakeBQClient.rows = [_row("claude-opus-4-8", cr_tok=10_000_000)]
    daily, _ = spend.fetch_llm_spend()
    rin, _rout = MODEL_PRICING["claude-opus-4-8"]
    assert daily == pytest.approx(10_000_000 * rin * 0.1 / 1_000_000, rel=1e-9)
    assert daily > 0


# -- the metered-only crux (phantom free tokens must not trip the breaker) --

def test_cc_rail_rows_contribute_zero_both_shapes():
    metered = _row("gemini-2.5-flash", provider="vertex", in_tok=100_000)
    FakeBQClient.rows = [
        metered,
        _row("claude-opus-4-8", provider="claude-code", agent="mas_main",
             in_tok=500_000_000, out_tok=100_000_000),        # shape 1: flat-fee rail
        _row("claude-opus-4-8", provider="anthropic", agent="cc_rail:synthesis",
             in_tok=500_000_000, out_tok=100_000_000),        # shape 2: rail via SDK tag
    ]
    daily, monthly = spend.fetch_llm_spend()
    only_metered = _expected_usd("gemini-2.5-flash", 100_000)
    assert daily == pytest.approx(only_metered, rel=1e-9), (
        "flat-fee CC-rail tokens were priced at API rates -- phantom spend "
        "would falsely trip the $25 breaker and halt trading")
    assert monthly == pytest.approx(only_metered, rel=1e-9)


def test_failed_calls_are_excluded():
    FakeBQClient.rows = [
        _row("claude-opus-4-8", ok=False, in_tok=1_000_000, out_tok=1_000_000),
    ]
    daily, monthly = spend.fetch_llm_spend()
    assert (daily, monthly) == (0.0, 0.0)


def test_agent_none_rows_are_included():
    FakeBQClient.rows = [_row("gemini-2.5-flash", provider="vertex", agent=None,
                              in_tok=100_000)]
    daily, _ = spend.fetch_llm_spend()
    assert daily == pytest.approx(_expected_usd("gemini-2.5-flash", 100_000), rel=1e-9)


# -- window split -----------------------------------------------------------

def test_earlier_in_month_counts_monthly_not_daily():
    FakeBQClient.rows = [
        _row("gemini-2.5-flash", provider="vertex", ts=EARLIER_IN_MONTH,
             in_tok=1_000_000),
    ]
    daily, monthly = spend.fetch_llm_spend()
    assert daily == 0.0
    assert monthly == pytest.approx(_expected_usd("gemini-2.5-flash", 1_000_000),
                                    rel=1e-9)


# -- fake self-test: the stub CAN represent the failure (4c) ----------------

def test_fake_client_honors_filter_absence():
    """If the production SQL LOST its exclusions, the fake must let CC-rail /
    failed rows through -- otherwise the crux mutation would be undetectable
    and every exclusion test above would be vacuous."""
    FakeBQClient.rows = [
        _row("claude-opus-4-8", provider="claude-code", in_tok=1_000_000),
        _row("claude-opus-4-8", provider="anthropic", agent="cc_rail:x",
             in_tok=1_000_000),
        _row("claude-opus-4-8", ok=False, in_tok=1_000_000),
    ]
    client = FakeBQClient()
    rows = client.query("SELECT model FROM t GROUP BY model").result()
    total_in = sum(r["m_in"] for r in rows)
    assert total_in == 3_000_000, (
        "fake applied exclusions that the SQL text did not ask for -- the "
        "stub can no longer represent a filter-less production query")


# -- fail-open + arch-04 seam (criterion 3) ---------------------------------

def test_fail_open_returns_zero_and_fires_degradation_seam():
    FakeBQClient.raise_exc = RuntimeError("bq outage 75.5.1")
    daily, monthly = spend.fetch_llm_spend()
    assert (daily, monthly) == (0.0, 0.0)
    status = spend.spend_guard_status()
    assert status["degraded_count"] == 1
    assert status["alerted"] is True
    assert "bq outage 75.5.1" in status["last_error"]


# -- flag routing: OFF is byte-identical, ON reads LLM spend (criterion 2) --

def _run_breaker(monkeypatch, flag_on, bq_spend, llm_spend):
    """Drive llm_client._check_cost_budget with sentinel sources; return
    whether it tripped. Cache reset between runs; env escape hatch cleared."""
    from backend.agents import llm_client
    from backend.config.settings import get_settings

    monkeypatch.delenv("COST_BUDGET_HARD_BLOCK_DISABLED", raising=False)
    settings = get_settings()
    monkeypatch.setattr(settings, "cost_budget_use_llm_spend_enabled", flag_on,
                        raising=False)
    import backend.services.observability as obs
    monkeypatch.setattr(obs, "fetch_spend", lambda: bq_spend)
    monkeypatch.setattr(obs, "fetch_llm_spend", lambda: llm_spend)
    llm_client.reset_cost_budget_cache()
    try:
        llm_client._check_cost_budget()
        return False
    except llm_client.BudgetBreachError:
        return True
    finally:
        llm_client.reset_cost_budget_cache()


def test_flag_off_is_byte_identical_to_bq_source(monkeypatch):
    # OFF: trips iff the BQ metric says so -- the LLM metric is IGNORED even
    # when it screams. This is the ON-vs-OFF $0 trip-point diff.
    assert _run_breaker(monkeypatch, False, (9999.0, 9999.0), (0.0, 0.0)) is True
    assert _run_breaker(monkeypatch, False, (0.0, 0.0), (9999.0, 9999.0)) is False


def test_flag_on_reads_the_llm_metric(monkeypatch):
    assert _run_breaker(monkeypatch, True, (0.0, 0.0), (9999.0, 9999.0)) is True
    assert _run_breaker(monkeypatch, True, (9999.0, 9999.0), (0.0, 0.0)) is False


def test_flag_default_is_off():
    from backend.config.settings import Settings
    assert Settings.model_fields["cost_budget_use_llm_spend_enabled"].default is False
