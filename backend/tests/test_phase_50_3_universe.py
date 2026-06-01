"""phase-50.3: international universe + suffix mapper + paper_markets routing.

Offline (no network/BQ): tests the suffix mapper, market derivation, curated
lists, the byte-identical paper_markets default, and the TradeOrder.market field.
"""
from backend.backtest import markets
from backend.backtest.universe_lists import DAX40, KOSPI200, INTL_UNIVERSE
from backend.config.settings import get_settings
from backend.services.portfolio_manager import TradeOrder


# ── criterion #1: suffix mapper round-trips ─────────────────────────
def test_to_yfinance_symbol_roundtrips():
    assert markets.to_yfinance_symbol("US:AAPL") == "AAPL"
    assert markets.to_yfinance_symbol("EU:SAP") == "SAP.DE"
    assert markets.to_yfinance_symbol("KR:005930") == "005930.KS"
    assert markets.to_yfinance_symbol("AAPL") == "AAPL"        # bare -> unchanged
    assert markets.to_yfinance_symbol("SAP.DE") == "SAP.DE"    # already suffixed -> unchanged


def test_market_for_symbol_derives_from_suffix():
    assert markets.market_for_symbol("AAPL") == "US"
    assert markets.market_for_symbol("SAP.DE") == "EU"
    assert markets.market_for_symbol("AIR.PA") == "EU"          # Paris-listed DAX member
    assert markets.market_for_symbol("005930.KS") == "KR"
    assert markets.market_for_symbol("035420.KQ") == "KR"       # KOSDAQ


# ── criterion #2: get_universe_tickers EU/KR non-empty + suffixed ───
def test_intl_universes_nonempty_and_suffixed():
    assert len(DAX40) >= 30 and all("." in t for t in DAX40)
    assert "SAP.DE" in DAX40 and "AIR.PA" in DAX40              # incl. the Paris exception
    assert len(KOSPI200) >= 20 and all(t.endswith(".KS") for t in KOSPI200)
    assert "005930.KS" in KOSPI200                             # Samsung
    assert INTL_UNIVERSE["EU"] is DAX40 and INTL_UNIVERSE["KR"] is KOSPI200


def test_kr_codes_keep_leading_zeros():
    # KR codes are 6-digit STRINGS; leading zeros are significant -> never int()
    for t in KOSPI200:
        code = t.split(".")[0]
        assert len(code) == 6 and code.isdigit(), t


# ── criterion #3: paper_markets default byte-identical ──────────────
def test_paper_markets_default_is_us_only():
    # phase-52.2: assert the CODE DEFAULT, not get_settings() -- the operator's 2026-06-01
    # go-live flip set PAPER_MARKETS=['US','EU','KR'] in backend/.env (a deliberate deployment
    # opt-in). The byte-identity invariant is that the code DEFAULT is US-only, so multi-market
    # requires an explicit opt-in; get_settings() correctly reflects the live .env override.
    from backend.config.settings import Settings
    assert Settings.model_fields["paper_markets"].default_factory() == ["US"]


def test_trade_order_market_defaults_us():
    o = TradeOrder(ticker="AAPL", action="BUY")
    assert o.market == "US"                                     # byte-identical default
    o2 = TradeOrder(ticker="SAP.DE", action="BUY",
                    market=markets.market_for_symbol("SAP.DE"))
    assert o2.market == "EU"
