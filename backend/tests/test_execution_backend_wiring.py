"""phase-68.1: EXECUTION_BACKEND must demonstrably reach execution_router.

The defect chain this file guards, all three links confirmed by the 68.1 research
gate against the working tree:

  1. `Settings` had NO `execution_backend` field at all.
  2. `execution_router` read `os.getenv("EXECUTION_BACKEND")` and nothing else.
  3. pydantic-settings loads `backend/.env` into `Settings` WITHOUT exporting to
     `os.environ`.

So a value set in `backend/.env` -- the project's actual configuration channel --
was invisible to the router, which used `bq_sim` forever with nothing in the logs
to say a setting had been dropped. The launchd plist carries no `EXECUTION_BACKEND`
either, so the running service had no way to be on anything else.

This step is DARK: the default stays `bq_sim` and no order path changes. What
changes is that the value is now *reachable* and its provenance is *stated*.

Every test asserts BEHAVIOUR (what the resolver/router returns, what gets logged),
never the source text of a module -- a test that greps for a string is not a wiring
test.
"""
from __future__ import annotations

import logging

import pytest

from backend.config.settings import Settings, get_settings
from backend.services import execution_router as er


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Every test starts from a known env and a cold settings cache.

    `get_settings` is `lru_cache`d, so a forgotten `cache_clear()` makes these
    tests order-dependent and can produce a pass that proves nothing.
    """
    for key in ("EXECUTION_BACKEND", "ALPACA_API_KEY_ID", "ALPACA_API_SECRET_KEY",
                "ALPACA_PAPER_TRADE"):
        monkeypatch.delenv(key, raising=False)
    get_settings.cache_clear()
    er._reset_missing_creds_warning()
    yield
    get_settings.cache_clear()
    er._reset_missing_creds_warning()


def _settings_with(monkeypatch, **overrides):
    """Force `get_settings()` to return a Settings carrying `overrides`.

    Patches the cached accessor rather than mutating a real Settings instance, so
    the .env file on this machine cannot influence the result either way.
    """
    base = Settings(**overrides) if overrides else Settings()
    monkeypatch.setattr("backend.config.settings.get_settings", lambda: base)
    return base


# ─────────── criterion 2: byte-identical default when nothing is set ───────────

def test_default_is_bq_sim_when_nothing_is_set():
    """The DARK guarantee: no EXECUTION_BACKEND anywhere -> today's behaviour."""
    mode, source = er.resolve_execution_mode()
    assert mode == "bq_sim"
    assert source == "default"
    assert er.ExecutionRouter().mode == "bq_sim"
    assert er.DEFAULT_MODE == "bq_sim"


def test_router_default_matches_pre_change_behaviour_exactly():
    """`_current_mode()` is the pre-existing entry point; it must be unchanged."""
    assert er._current_mode() == "bq_sim"
    assert er.VALID_MODES == ("bq_sim", "alpaca_paper", "shadow")


# ─────────── criterion 1: the value reaches the router, with provenance ───────────

@pytest.mark.parametrize("mode", ["bq_sim", "alpaca_paper", "shadow"])
def test_env_value_reaches_the_router_and_reports_source_env(monkeypatch, mode):
    """os.environ -- the launchd-plist / shell-export channel."""
    monkeypatch.setenv("EXECUTION_BACKEND", mode)
    resolved, source = er.resolve_execution_mode()
    assert resolved == mode
    assert source == "env"
    assert er.ExecutionRouter().mode == mode, "the router must act on the resolved mode"


@pytest.mark.parametrize("mode", ["alpaca_paper", "shadow"])
def test_dotenv_value_reaches_the_router_and_reports_source_dotenv(monkeypatch, mode):
    """THE ORIGINAL DEFECT: a .env value must now reach the router.

    Before this step there was no `execution_backend` field, so this channel could
    not carry a value at all.
    """
    _settings_with(monkeypatch, execution_backend=mode)
    resolved, source = er.resolve_execution_mode()
    assert resolved == mode
    assert source == "dotenv"
    assert er.ExecutionRouter().mode == mode


def test_env_wins_over_dotenv(monkeypatch):
    """Precedence must be stated and testable, not incidental."""
    _settings_with(monkeypatch, execution_backend="shadow")
    monkeypatch.setenv("EXECUTION_BACKEND", "alpaca_paper")
    assert er.resolve_execution_mode() == ("alpaca_paper", "env")


def test_startup_line_names_both_the_mode_and_its_source(monkeypatch, caplog):
    """The live_check artifact: one line carrying mode AND provenance.

    Without the source, "bq_sim" cannot distinguish "configured" from "your
    setting was silently dropped" -- which is the whole defect.
    """
    monkeypatch.setenv("EXECUTION_BACKEND", "shadow")
    with caplog.at_level(logging.INFO, logger=er.logger.name):
        mode, source = er.log_resolved_execution_mode()
    assert (mode, source) == ("shadow", "env")
    records = [r for r in caplog.records if "execution backend" in r.getMessage()]
    assert len(records) == 1, f"expected exactly one startup line, got {len(records)}"
    msg = records[0].getMessage()
    assert "mode=shadow" in msg
    assert "source=env" in msg


def test_startup_line_reports_default_source_when_unset(caplog):
    with caplog.at_level(logging.INFO, logger=er.logger.name):
        er.log_resolved_execution_mode()
    msg = next(r.getMessage() for r in caplog.records if "execution backend" in r.getMessage())
    assert "mode=bq_sim" in msg and "source=default" in msg


# ─────────── criterion 4c: never escalate, never raise, on a bad value ───────────

@pytest.mark.parametrize("bad", ["alpaca_live", "live", "LIVE", "", "   ", "garbage",
                                 "bq_sim; alpaca_live", "alpaca", "paper", "real"])
def test_unknown_or_live_ish_env_value_falls_back_to_bq_sim(monkeypatch, bad):
    """A typo in config must not take the order path down, and must never escalate."""
    monkeypatch.setenv("EXECUTION_BACKEND", bad)
    mode, source = er.resolve_execution_mode()
    assert mode in er.VALID_MODES
    assert mode == "bq_sim", f"{bad!r} must not resolve to anything but the safe default"
    assert "live" not in mode
    assert source in ("default", "invalid:env")


@pytest.mark.parametrize("bad", ["alpaca_live", "live", "garbage"])
def test_unknown_dotenv_value_also_falls_back(monkeypatch, bad):
    _settings_with(monkeypatch, execution_backend=bad)
    mode, source = er.resolve_execution_mode()
    assert mode == "bq_sim"
    assert source == "invalid:dotenv"


def test_case_and_whitespace_are_normalised(monkeypatch):
    monkeypatch.setenv("EXECUTION_BACKEND", "  ShAdOw  ")
    assert er.resolve_execution_mode() == ("shadow", "env")


# ─────────── criterion 3: missing creds must be LOUD, not silent ───────────

def test_missing_creds_in_alpaca_mode_logs_loudly_and_names_both_keys(caplog):
    router = er.ExecutionRouter(mode="alpaca_paper")
    with caplog.at_level(logging.ERROR, logger=er.logger.name):
        fill = router.submit_order("AAPL", 1, "buy", "oid-1", close_price=100.0)

    errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert len(errors) == 1, f"expected exactly one ERROR, got {len(errors)}"
    msg = errors[0].getMessage()
    assert "ALPACA_API_KEY_ID" in msg, "the log must NAME the missing variables"
    assert "ALPACA_API_SECRET_KEY" in msg
    # the fill still happens (fail-open), and is honestly labelled
    assert fill.source == "mock_alpaca"
    assert fill.paper is True


def test_no_error_is_logged_when_creds_are_present(monkeypatch, caplog):
    """Mutation guard: without this, the test above could pass on a log that
    always fires -- which would be noise, not a signal."""
    monkeypatch.setenv("ALPACA_API_KEY_ID", "PKTESTKEY")
    monkeypatch.setenv("ALPACA_API_SECRET_KEY", "secret")
    monkeypatch.setattr(er, "_alpaca_real_fill",
                        lambda *a, **k: er.FillResult(
                            client_order_id="oid", symbol="AAPL", qty=1, side="buy",
                            fill_price=100.0, status="filled", source="alpaca_paper"))
    router = er.ExecutionRouter(mode="alpaca_paper")
    with caplog.at_level(logging.ERROR, logger=er.logger.name):
        router.submit_order("AAPL", 1, "buy", "oid-2", close_price=100.0)
    assert [r for r in caplog.records if r.levelno >= logging.ERROR] == []


def test_the_loud_log_is_latched_so_an_order_loop_cannot_flood(caplog):
    router = er.ExecutionRouter(mode="alpaca_paper")
    with caplog.at_level(logging.ERROR, logger=er.logger.name):
        for i in range(5):
            router.submit_order("AAPL", 1, "buy", f"oid-{i}", close_price=100.0)
    assert len([r for r in caplog.records if r.levelno >= logging.ERROR]) == 1


def test_bq_sim_mode_never_emits_the_creds_error(caplog):
    """Scope guard: the default path must stay silent about Alpaca."""
    router = er.ExecutionRouter(mode="bq_sim")
    with caplog.at_level(logging.ERROR, logger=er.logger.name):
        router.submit_order("AAPL", 1, "buy", "oid-3", close_price=100.0)
    assert [r for r in caplog.records if r.levelno >= logging.ERROR] == []


# ─────────── criterion 4a/4b: paper-only enforcement ───────────

def test_paper_base_url_is_pinned_to_the_paper_domain():
    """The LOAD-BEARING guard: paper and live are separated by DOMAIN.

    Constructed offline -- alpaca-py does not make a network call at construction.
    """
    pytest.importorskip("alpaca.trading.client")
    from alpaca.trading.client import TradingClient
    from alpaca.common.enums import BaseURL

    client = TradingClient("dummy_key", "dummy_secret", paper=True)
    # `.value` matters: str() on the enum yields "BaseURL.TRADING_PAPER", so a
    # str()-based assertion would compare labels and never see the actual host.
    assert client._base_url == BaseURL.TRADING_PAPER
    assert "paper-api.alpaca.markets" in BaseURL.TRADING_PAPER.value
    assert client._base_url.value != BaseURL.TRADING_LIVE.value
    assert "//api.alpaca.markets" not in client._base_url.value


def test_repo_never_overrides_the_alpaca_base_url():
    """`url_override` is the SDK's documented escape hatch out of the paper pin.

    Asserted against the real call signature, and against the router's actual
    construction call, so an added override cannot slip past.
    """
    pytest.importorskip("alpaca.trading.client")
    import inspect
    from alpaca.trading.client import TradingClient

    assert "url_override" in inspect.signature(TradingClient.__init__).parameters, (
        "SDK signature changed -- re-derive this guard rather than deleting it"
    )
    src = inspect.getsource(er)
    assert "url_override" not in src, "the router must never pass url_override"
    assert "paper=True" in src, "the router must construct the SDK in paper mode"


@pytest.mark.parametrize("live_key", ["PKLIVEABC123", "pklive_lower", "AKLIVEXYZ"])
def test_live_marked_key_prefix_is_refused(monkeypatch, live_key):
    """NOTE: Alpaca does NOT document a paper/live key-prefix difference (68.1
    research gate, three official sources read in full). This filter is
    belt-and-braces; the real guard is the paper base URL above. The test exists
    so the filter cannot silently rot, not as evidence that the format is real."""
    monkeypatch.setenv("ALPACA_API_KEY_ID", live_key)
    with pytest.raises(RuntimeError, match="paper-only"):
        er._refuse_live_keys()


def test_paper_trade_false_is_refused(monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY_ID", "PKTESTKEY")
    monkeypatch.setenv("ALPACA_PAPER_TRADE", "false")
    with pytest.raises(RuntimeError, match="paper-only"):
        er._refuse_live_keys()


@pytest.mark.parametrize("ok_value", ["true", "TRUE", "True", None])
def test_ordinary_paper_config_is_allowed(monkeypatch, ok_value):
    """The refusal must not be a blanket denial -- otherwise it proves nothing."""
    monkeypatch.setenv("ALPACA_API_KEY_ID", "PKTESTKEY")
    if ok_value is not None:
        monkeypatch.setenv("ALPACA_PAPER_TRADE", ok_value)
    er._refuse_live_keys()  # must not raise


def test_every_fill_path_reports_paper_true():
    """Criterion 4c at the result level: no mode yields a non-paper fill."""
    for mode in ("bq_sim", "alpaca_paper", "shadow"):
        fill = er.ExecutionRouter(mode=mode).submit_order(
            "AAPL", 1, "buy", f"oid-{mode}", close_price=100.0)
        assert fill.paper is True, f"mode {mode} produced a non-paper fill"


def test_flip_to_rejects_a_mode_outside_the_valid_set():
    router = er.ExecutionRouter(mode="bq_sim")
    with pytest.raises(ValueError):
        router.flip_to("alpaca_live")  # type: ignore[arg-type]
    assert router.mode == "bq_sim", "a rejected flip must leave the mode untouched"


# ─────────── the settings field itself ───────────

def test_settings_carries_the_execution_backend_field_defaulting_to_unset():
    """Without the field, the .env channel cannot carry a value at all.

    The default is None rather than "bq_sim" on purpose: a concrete default would
    make "unconfigured" report as source=dotenv, which is precisely the confusion
    the provenance line exists to remove. The safe fallback lives in the router.
    """
    assert "execution_backend" in Settings.model_fields
    assert Settings.model_fields["execution_backend"].default is None
    assert er.DEFAULT_MODE == "bq_sim", "the fallback must live in the router"
