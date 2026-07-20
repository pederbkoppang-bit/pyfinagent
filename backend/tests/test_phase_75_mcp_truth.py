"""phase-75.3: MCP servers must not fabricate state.

Every finding in this step is the same bug in different clothes -- a failure
path that returns a confident, plausible value instead of refusing. These
tests pin the refusals.

MOCKING DISCIPLINE IS LOAD-BEARING. The pre-existing tests/test_mcp_servers.py
uses no mocks and asserts envelope shape rather than outcome (`assert "status"
in result` passes when status == "ERROR"), which is *why* all of this shipped
undetected. A bare Mock() auto-creates any attribute you touch, so it would
happily serve `paper_trader.get_portfolio()` and let gap4-01 regress. These
tests use create_autospec(PaperTrader, instance=True) so calling a method the
real class does not have raises AttributeError, and a real BacktestResult
instance rather than a dict-like stand-in.

All offline: no BQ, no network.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import create_autospec

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.agents.mcp_servers import signals_server as ss  # noqa: E402
from backend.agents.mcp_servers.signals_server import SignalsServer  # noqa: E402
from backend.services.paper_trader import PaperTrader  # noqa: E402

SIGNALS_SRC = Path(ss.__file__).read_text(encoding="utf-8")


# ── helpers ──────────────────────────────────────────────────────────

def _server(paper_trader=None, monkeypatch=None):
    """A SignalsServer with the backend marked available and a stub trader."""
    srv = SignalsServer.__new__(SignalsServer)
    srv.paper_trader = paper_trader
    srv.settings = None
    srv._recent_responses = ss.OrderedDict()
    srv._recent_responses_limit = 50
    srv._peak_equity = 0.0
    return srv


def _autospec_trader(nav=25_000.0, cash=5_000.0, positions=None):
    """create_autospec: calling a method PaperTrader lacks raises AttributeError."""
    trader = create_autospec(PaperTrader, instance=True)
    trader.get_or_create_portfolio.return_value = {
        "total_nav": nav,
        "current_cash": cash,
        "last_updated": "2026-07-20T00:00:00Z",
    }
    trader.get_positions.return_value = positions if positions is not None else []
    return trader


# ── criterion 1: no fabricated portfolio ─────────────────────────────

def test_source_no_longer_calls_nonexistent_paper_trader_get_portfolio():
    assert "paper_trader.get_portfolio(" not in SIGNALS_SRC


def test_paper_trader_really_has_no_get_portfolio():
    """The premise of gap4-01: the method the old code called does not exist."""
    assert not hasattr(PaperTrader, "get_portfolio")
    assert hasattr(PaperTrader, "get_or_create_portfolio")
    assert hasattr(PaperTrader, "get_positions")


def test_get_portfolio_returns_real_nav_and_positions(monkeypatch):
    monkeypatch.setattr(ss, "_SIGNALS_AVAILABLE", True)
    trader = _autospec_trader(
        nav=25_000.0, cash=5_000.0,
        positions=[{"ticker": "AAPL", "shares": 10, "price": 190.0}],
    )
    out = _server(trader).get_portfolio()

    assert out["total_value"] == 25_000.0      # from total_nav, not a $10k default
    assert out["cash"] == 5_000.0              # from current_cash
    assert "AAPL" in out["positions"]          # list -> dict keyed by ticker
    assert out["positions"]["AAPL"]["shares"] == 10
    assert not out.get("stub")


def test_degraded_portfolio_is_marked_stub_and_zeroed(monkeypatch):
    """The exception path must not hand back a plausible book."""
    monkeypatch.setattr(ss, "_SIGNALS_AVAILABLE", True)
    trader = create_autospec(PaperTrader, instance=True)
    trader.get_or_create_portfolio.side_effect = RuntimeError("BQ down")

    out = _server(trader).get_portfolio()

    assert out["stub"] is True
    assert out["total_value"] == 0.0
    assert out["cash"] == 0.0
    assert out["positions"] == {}


def test_no_backend_path_is_marked_stub_and_zeroed(monkeypatch):
    monkeypatch.setattr(ss, "_SIGNALS_AVAILABLE", True)
    out = _server(None).get_portfolio()
    assert out["stub"] is True
    assert out["total_value"] == 0.0


def test_publish_signal_refuses_to_trade_on_degraded_snapshot(monkeypatch):
    monkeypatch.setattr(ss, "_SIGNALS_AVAILABLE", True)
    trader = create_autospec(PaperTrader, instance=True)
    trader.get_or_create_portfolio.side_effect = RuntimeError("BQ down")

    srv = _server(trader)
    resp = srv.publish_signal({
        "ticker": "AAPL", "signal": "BUY", "confidence": 0.9,
        "date": "2026-07-20", "factors": ["momentum"], "price": 190.0,
    })

    assert resp["published"] is False
    assert "degraded" in resp["reason"]
    trader.execute_buy.assert_not_called()   # no trade was booked


# ── criterion 2: risk gates ──────────────────────────────────────────

def test_buy_with_unresolved_price_is_rejected_unknown_price(monkeypatch):
    monkeypatch.setattr(ss, "_SIGNALS_AVAILABLE", True)
    srv = _server(_autospec_trader())
    portfolio = {"total_value": 25_000.0, "cash": 20_000.0, "positions": {}}

    risk = srv.risk_check(portfolio, {
        "ticker": "AAPL", "action": "BUY", "shares": 10, "price": None,
    })

    assert risk["allowed"] is False
    assert "unknown_price" in (risk.get("conflicts") or [])


def test_sell_with_unresolved_price_is_not_trapped_by_unknown_price(monkeypatch):
    """Exits must not be blocked by a missing mark -- they reduce exposure."""
    monkeypatch.setattr(ss, "_SIGNALS_AVAILABLE", True)
    srv = _server(_autospec_trader())
    portfolio = {
        "total_value": 25_000.0, "cash": 20_000.0,
        "positions": {"AAPL": {"ticker": "AAPL", "shares": 50, "price": 0.0}},
    }
    risk = srv.risk_check(portfolio, {
        "ticker": "AAPL", "action": "SELL", "shares": 10, "price": None,
    })
    assert "unknown_price" not in (risk.get("conflicts") or [])


def test_drawdown_breaker_blocks_buys(monkeypatch):
    monkeypatch.setattr(ss, "_SIGNALS_AVAILABLE", True)
    srv = _server(_autospec_trader())
    portfolio = {
        "total_value": 25_000.0, "cash": 20_000.0, "positions": {},
        "current_drawdown_pct": -16.0,
    }
    risk = srv.risk_check(portfolio, {
        "ticker": "AAPL", "action": "BUY", "shares": 1, "price": 100.0,
    })
    assert risk["allowed"] is False


def test_thresholds_are_unchanged(monkeypatch):
    """BOUNDARY pin: this step touches plumbing, never a threshold."""
    monkeypatch.setattr(ss, "_SIGNALS_AVAILABLE", True)
    limits = _server(_autospec_trader()).get_risk_constraints()
    assert float(limits["max_position_pct"]) == 5.0
    assert float(limits["max_position_usd"]) == 1000.0
    assert float(limits["max_drawdown_pct"]) == -15.0
    assert float(limits["drawdown_warning_pct"]) == -5.0
    assert float(limits["drawdown_derisk_pct"]) == -10.0


def test_explicit_size_usd_is_clamped_to_the_hard_cap(monkeypatch):
    monkeypatch.setattr(ss, "_SIGNALS_AVAILABLE", True)
    srv = _server(_autospec_trader())
    portfolio = {"total_value": 100_000.0, "cash": 100_000.0, "positions": {}}

    sized = srv.size_position(
        {"ticker": "AAPL", "signal": "BUY", "size_usd": 999_999.0}, portfolio
    )

    limits = srv.get_risk_constraints()
    cap = min(100_000.0 * float(limits["max_position_pct"]) / 100.0,
              float(limits["max_position_usd"]))
    assert sized == cap
    assert sized < 999_999.0


# ── criterion 3: dedup never fabricates success ──────────────────────

def test_evicted_rejection_never_replays_as_published(monkeypatch):
    """The gap4-06 inversion: seen-but-evicted used to synthesize published."""
    monkeypatch.setattr(ss, "_SIGNALS_AVAILABLE", True)
    srv = _server(_autospec_trader())
    srv._recent_responses_limit = 2

    rejected = {"published": False, "reason": "risk_rejected:cash", "signal_id": "sig-1"}
    srv._remember("sig-1", rejected)
    for i in range(5):                       # force eviction of sig-1
        srv._remember(f"filler-{i}", {"published": True, "reason": "ok"})

    assert "sig-1" not in srv._recent_responses      # evicted
    replay = srv._recent_responses.get("sig-1")
    assert replay is None                            # nothing to synthesize from


def test_remembered_rejection_replays_the_true_outcome(monkeypatch):
    monkeypatch.setattr(ss, "_SIGNALS_AVAILABLE", True)
    srv = _server(_autospec_trader())
    srv._remember("sig-2", {"published": False, "reason": "risk_rejected:cash"})

    cached = srv._recent_responses["sig-2"]
    assert cached["published"] is False              # not inverted to True


def test_seen_and_outcome_evict_together(monkeypatch):
    """One OrderedDict = no set/dict asymmetry, and no unbounded set."""
    monkeypatch.setattr(ss, "_SIGNALS_AVAILABLE", True)
    srv = _server(_autospec_trader())
    srv._recent_responses_limit = 3
    for i in range(10):
        srv._remember(f"s-{i}", {"published": True, "reason": "ok"})
    assert len(srv._recent_responses) == 3
    assert not hasattr(srv, "_seen_signal_ids") or not srv._seen_signal_ids


def test_no_synthesized_published_true_in_source():
    """The dedup miss branch that set published=True must be gone."""
    assert "resp[\"published\"] = True" not in SIGNALS_SRC


# ── criterion 4: stub provenance ─────────────────────────────────────

def test_emit_candidates_source_is_stub_marked_and_keeps_consumer_contract():
    """phase-3.7 A/B harness asserts >=5 candidates each carrying `dsr`.
    stub/reason are ADDITIVE -- the count and the dsr key must survive."""
    import re
    block = SIGNALS_SRC.split("def emit_candidates")[1].split("logger.info(\"Signals server created")[0]
    assert '"stub": True' in block
    assert "PENDING_IMPLEMENTATION" in block
    assert '"dsr": dsr' in block                 # consumer contract preserved
    assert re.search(r"n = max\(5, int\(n\)\)", block)


def test_compute_dsr_real_dead_branch_is_gone():
    assert "compute_dsr_real" not in SIGNALS_SRC


def test_publish_signal_rejects_stub_provenance(monkeypatch):
    monkeypatch.setattr(ss, "_SIGNALS_AVAILABLE", True)
    srv = _server(_autospec_trader())
    resp = srv.publish_signal({
        "ticker": "AAPL", "signal": "BUY", "confidence": 0.9,
        "date": "2026-07-20", "factors": ["momentum"], "price": 190.0,
        "stub": True,
    })
    assert resp["published"] is False
    assert resp["reason"] == "stub_provenance"


# ── criterion 5: data_server ─────────────────────────────────────────

def test_data_server_has_no_hardcoded_cutoff_literal():
    from backend.agents.mcp_servers import data_server as ds
    assert "2025-12-31" not in Path(ds.__file__).read_text(encoding="utf-8")


def test_get_macro_iterates_the_dict_and_filters_the_series(monkeypatch):
    from backend.agents.mcp_servers import data_server as ds

    class _FakeCache:
        @staticmethod
        def cached_macro(cutoff):
            # Real shape: dict keyed by series_id -> as-of entry.
            return {
                "VIXCLS": {"date": "2026-07-19", "value": 14.5},
                "DGS10":  {"date": "2026-07-19", "value": 4.1},
            }

    monkeypatch.setattr(ds, "_CACHE_AVAILABLE", True)
    monkeypatch.setattr(ds, "cache", _FakeCache)

    srv = ds.DataServer.__new__(ds.DataServer)
    out = srv.get_macro("VIXCLS")

    assert out["data"], "dict iteration bug would yield an empty list here"
    assert len(out["data"]) == 1
    assert out["data"][0]["value"] == 14.5
    assert out["data"][0]["date"] == "2026-07-19"


def test_prices_and_fundamentals_use_today_derived_cutoffs(monkeypatch):
    from datetime import date
    from backend.agents.mcp_servers import data_server as ds

    seen = {}

    class _FakeCache:
        @staticmethod
        def cached_prices(ticker, start, end):
            seen["prices_end"] = end
            return None

        @staticmethod
        def cached_fundamentals(ticker, cutoff):
            seen["fund_cutoff"] = cutoff
            return None

    monkeypatch.setattr(ds, "_CACHE_AVAILABLE", True)
    monkeypatch.setattr(ds, "cache", _FakeCache)

    srv = ds.DataServer.__new__(ds.DataServer)
    srv.get_prices("AAPL")
    srv.get_fundamentals("AAPL")

    today = date.today().isoformat()
    assert seen["prices_end"] == today
    assert seen["fund_cutoff"] == today


# ── criterion 6: backtest_server + SecretStr ─────────────────────────

def test_backtest_server_reads_dataclass_fields_without_attributeerror(monkeypatch):
    from backend.agents.mcp_servers import backtest_server as bs
    from backend.backtest.backtest_engine import BacktestResult

    result = BacktestResult(
        aggregate_sharpe=1.23,
        aggregate_return_pct=15.5,
        aggregate_max_drawdown_pct=-7.5,
        total_trades=42,
    )

    class _FakeEngine:
        def __init__(self, *a, **k):
            pass

        def run_backtest(self):
            return result

    monkeypatch.setattr(bs, "_BACKTEST_AVAILABLE", True)
    monkeypatch.setattr(bs, "BacktestEngine", _FakeEngine)

    srv = bs.BacktestServer.__new__(bs.BacktestServer)
    srv.bq_client = object()

    class _S:
        gcp_project_id = "p"
        bq_dataset_reports = "d"

    srv.settings = _S()

    out = srv.run_backtest({})

    assert out["status"] == "PASS"          # was always ERROR before
    assert out["sharpe"] == 1.23
    assert out["return_pct"] == 15.5
    assert out["max_drawdown_pct"] == -7.5
    assert out["total_trades"] == 42
    # DSR is not produced by the engine -- better absent than fabricated 0.0.
    assert "dsr" not in out


def test_backtest_result_lacks_the_fields_the_spec_named():
    """Guards the spec correction: following it literally would AttributeError."""
    from backend.backtest.backtest_engine import BacktestResult
    r = BacktestResult()
    for absent in ("dsr", "return_pct", "max_drawdown_pct", "num_trades"):
        assert not hasattr(r, absent), f"{absent} unexpectedly exists"


def test_secretstr_unwrap_helper_semantics_and_call_site_present():
    from pydantic import SecretStr
    from backend.agents.llm_client import unwrap_secret

    secret = SecretStr("xoxb-real-token")
    assert bool(secret) is True                    # truthy: the old guard missed it
    assert str(secret) == "**********"             # str() silently masks
    assert unwrap_secret(secret) == "xoxb-real-token"
    assert "unwrap_secret" in SIGNALS_SRC


# ── behavioral guards (added after Q/A wf_fcf4f363-339 CONDITIONAL) ───
#
# The two tests below replace proxy assertions that a realistic regression
# would have evaded. This step's whole thesis is that the pre-existing suite
# "asserts envelope shape rather than outcome, which is why all of this
# shipped" -- so shipping proxy guards for its own criteria would have been
# the same mistake one layer up.
#
#   - criterion 6 was guarded by `"unwrap_secret" in SIGNALS_SRC`. That string
#     appears on BOTH the import line and the call site, so reverting only the
#     call site left the test green while security-05 fully regressed.
#   - criterion 3's two behavioral halves ("re-fired after eviction reports
#     published=false", "a freed-up rejection can be retried") were never
#     driven through publish_signal; they rested on an exact-string source
#     scan that a reworded regression evades.


def _funded_signal(ticker="AAPL"):
    return {
        "ticker": ticker, "signal": "BUY", "confidence": 0.9,
        "date": "2026-07-20", "factors": ["momentum"], "price": 100.0,
    }


def test_evicted_refire_of_a_rejection_reports_published_false_end_to_end(monkeypatch):
    """criterion 3, first half -- driven through publish_signal, not the cache dict.

    A rejected signal whose cache entry has been evicted must be RE-EVALUATED
    and reported honestly. It must never come back as a synthesized success.
    """
    monkeypatch.setattr(ss, "_SIGNALS_AVAILABLE", True)
    # A book with no cash: the BUY is rejected by the cash gate.
    trader = _autospec_trader(nav=1_000.0, cash=0.0)
    srv = _server(trader)
    srv._recent_responses_limit = 2

    first = srv.publish_signal(_funded_signal())
    assert first["published"] is False

    # Force eviction of the remembered rejection.
    for i in range(5):
        srv._remember(f"filler-{i}", {"published": True, "reason": "ok"})

    refire = srv.publish_signal(_funded_signal())
    assert refire["published"] is False, "an evicted rejection must not replay as success"
    trader.execute_buy.assert_not_called()


def test_a_freed_up_rejection_can_be_retried_end_to_end(monkeypatch):
    """criterion 3, second half -- rejections stay retryable once unblocked."""
    monkeypatch.setattr(ss, "_SIGNALS_AVAILABLE", True)
    trader = _autospec_trader(nav=1_000.0, cash=0.0)
    srv = _server(trader)

    blocked = srv.publish_signal(_funded_signal())
    assert blocked["published"] is False

    # The blocking condition clears: fund the book and retry the same signal.
    trader.get_or_create_portfolio.return_value = {
        "total_nav": 100_000.0, "current_cash": 100_000.0,
        "last_updated": "2026-07-20T00:00:00Z",
    }
    trader.execute_buy.return_value = {
        "ticker": "AAPL", "shares": 5, "price": 100.0, "action": "BUY",
    }
    srv._recent_responses.clear()   # simulate the entry ageing out

    retried = srv.publish_signal(_funded_signal())
    assert retried["published"] is True, "a freed-up rejection must be retryable"
    trader.execute_buy.assert_called_once()


def test_secretstr_token_reaches_webclient_unwrapped_end_to_end(monkeypatch):
    """criterion 6 -- observe what WebClient actually receives.

    Fails if the unwrap at the SDK boundary is removed, which the old
    source-scan assertion did not.
    """
    from pydantic import SecretStr

    captured = {}

    class _FakeWebClient:
        def __init__(self, token=None, **kwargs):
            captured["token"] = token

        def chat_postMessage(self, **kwargs):
            return {"ok": True, "ts": "123.456", "channel": "C1"}

    import slack_sdk
    monkeypatch.setattr(slack_sdk, "WebClient", _FakeWebClient)
    monkeypatch.setattr(ss, "_SIGNALS_AVAILABLE", True)

    class _Settings:
        slack_bot_token = SecretStr("xoxb-real-token")
        slack_channel_id = "C1"

    trader = _autospec_trader(nav=100_000.0, cash=100_000.0)
    trader.execute_buy.return_value = {
        "ticker": "AAPL", "shares": 5, "price": 100.0, "action": "BUY",
    }
    srv = _server(trader)
    srv.settings = _Settings()

    srv.publish_signal(_funded_signal())

    assert captured.get("token") == "xoxb-real-token"
    assert captured["token"] != "**********"       # the str(SecretStr) mask
    assert not isinstance(captured["token"], SecretStr)


def test_emit_candidates_really_emits_stub_marked_candidates():
    """criterion 4, first clause -- assert over the EMITTED payload.

    The source-scan test above is evadable: keeping the `"stub": True` literal
    in the dict while popping the key before return leaves it green (proven by
    Q/A wf_a66a87f0-756). This drives the actual tool and inspects what callers
    receive. Offline: create_signals_server + in-process fastmcp Client, no BQ,
    no network.
    """
    import asyncio

    from fastmcp import Client

    async def _pull():
        mcp = ss.create_signals_server()
        async with Client(mcp) as client:
            result = await client.call_tool(
                "emit_candidates", {"ticker": "AAPL", "n": 5}
            )
            data = result.data if hasattr(result, "data") else result
            return (data or {}).get("candidates") or []

    candidates = asyncio.run(_pull())

    assert len(candidates) >= 5                              # consumer contract
    assert all("dsr" in c for c in candidates)               # consumer contract
    assert all(c.get("stub") is True for c in candidates), (
        "every emitted candidate must carry stub:true"
    )
    assert all(c.get("reason") == "PENDING_IMPLEMENTATION" for c in candidates)
