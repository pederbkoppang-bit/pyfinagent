"""phase-61.2 -- decision-input integrity regression suite.

Covers the six masterplan criteria (dark build; both behavior flags default
OFF). File name matches the immutable -k expression via '61_2'.

Criterion map:
  1  synthesis-error never persists as 0.0/HOLD  -> TestSynthesisIntegrity,
     TestDegradedPersistence
  2  claude_code timeout >= 150s configurable    -> TestTimeoutPlumbing
  3  company_name quant fallback                  -> TestCompanyNameFallback
  4  meta-scorer rank-normalized fallback + streak -> TestMetaScorerFallback,
     TestConvictionStreak
  5  positions persist analysis recommendation    -> TestSignalDowngrade
  6  RiskJudge advisory context                   -> TestRiskJudgeAdvisoryCtx
"""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import backend.services.autonomous_loop as al
import backend.services.meta_scorer as ms
from backend.services.portfolio_manager import TradeOrder, decide_trades


# ── helpers ────────────────────────────────────────────────────────────────

def _settings(**over):
    base = dict(
        lite_mode=False,
        gemini_model="gemini-2.5-flash",
        paper_synthesis_integrity_enabled=False,
        paper_position_recommendation_fix_enabled=False,
        paper_risk_judge_reject_binding=False,
        claude_code_empty_retry_max=2,
        claude_code_timeout_s=150,
        # decide_trades / paper_trader surface
        paper_starting_capital=10000.0,
        paper_min_cash_reserve_pct=5.0,
        paper_max_positions=10,
        paper_max_per_sector=2,
        paper_max_per_sector_nav_pct=30.0,
        paper_max_factor_corr=0.0,
        paper_swap_enabled=False,
        paper_default_stop_loss_pct=8.0,
        paper_transaction_cost_pct=0.1,
    )
    base.update(over)
    return SimpleNamespace(**base)


ERROR_SYNTHESIS_REPORT = {
    "final_synthesis": {"error": "Failed to parse final report.", "synthesis_iterations": 2},
    "quant": {"company_name": "Quant Name Co"},
    "cost_summary": {"total_cost_usd": 0.2},
}

HEALTHY_REPORT = {
    "final_synthesis": {
        "scoring_matrix": {"corporate": 8},
        "recommendation": {"action": "BUY"},
        "final_weighted_score": 7.5,
        "risk_assessment": {"reason": "ok"},
    },
    "quant": {"company_name": "Quant Name Co", "yf_data": {}},
    "cost_summary": {"total_cost_usd": 0.2},
}


class _StubOrch:
    """Stands in for AnalysisOrchestrator; returns a canned report."""

    report = ERROR_SYNTHESIS_REPORT

    def __init__(self, settings):
        pass

    async def run_full_analysis(self, ticker):
        return type(self).report


def _lite_ok(ticker, settings, portfolio_context=""):
    async def _run():
        return {
            "ticker": ticker,
            "recommendation": "BUY",
            "final_score": 6.0,
            "_path": "lite",
            "risk_assessment": {},
            "total_cost_usd": 0.01,
            "full_report": {},
        }
    return _run()


def _lite_fail(ticker, settings, portfolio_context=""):
    async def _run():
        raise RuntimeError("lite dead too")
    return _run()


# ── criterion 1: synthesis-error routing ──────────────────────────────────

class TestSynthesisIntegrity:
    def _run(self, settings, lite):
        # _run_single_analysis REBINDS settings via a function-local
        # `from backend.config.settings import get_settings` (phase-38.13
        # rail attribution) -- the flag must come through THAT import.
        def _fresh():
            return settings
        _fresh.cache_clear = lambda: None
        with patch.object(al, "AnalysisOrchestrator", _StubOrch), \
             patch.object(al, "_select_lite_analyzer", lambda model: lite), \
             patch("backend.config.settings.get_settings", _fresh):
            return asyncio.run(al._run_single_analysis("TST", settings))

    def test_flag_on_error_synthesis_routes_to_lite(self):
        _StubOrch.report = ERROR_SYNTHESIS_REPORT
        out = self._run(_settings(paper_synthesis_integrity_enabled=True), _lite_ok)
        assert out["_path"] == "lite"
        assert out["recommendation"] == "BUY"  # REAL scored row, not HOLD/0.0
        assert str(out.get("_fallback_reason", "")).startswith("SynthesisDegradedError")

    def test_flag_on_missing_scoring_matrix_routes_to_lite(self):
        _StubOrch.report = {
            "final_synthesis": {"recommendation": {"action": "BUY"}},
            "quant": {}, "cost_summary": {},
        }
        out = self._run(_settings(paper_synthesis_integrity_enabled=True), _lite_ok)
        assert out["_path"] == "lite"

    def test_flag_on_both_fail_returns_degraded_marker(self):
        _StubOrch.report = ERROR_SYNTHESIS_REPORT
        out = self._run(_settings(paper_synthesis_integrity_enabled=True), _lite_fail)
        assert out is not None and out["_degraded"] is True
        assert out["final_score"] is None and out["recommendation"] is None
        assert out["_path"] == "degraded"
        assert "both_paths_failed" in out["_degraded_reason"]

    def test_flag_off_legacy_fabrication_unchanged(self):
        # Byte-identical legacy: error dict assembles into HOLD / 0.0.
        _StubOrch.report = ERROR_SYNTHESIS_REPORT
        out = self._run(_settings(), _lite_ok)
        assert out["recommendation"] == "HOLD"
        assert out["final_score"] == 0
        assert out["_path"] == "full"

    def test_flag_on_healthy_report_untouched(self):
        _StubOrch.report = HEALTHY_REPORT
        out = self._run(_settings(paper_synthesis_integrity_enabled=True), _lite_fail)
        assert out["recommendation"] == "BUY" and out["final_score"] == 7.5


class TestDegradedPersistence:
    def _persist(self, analysis):
        bq = MagicMock()
        asyncio.run(al._persist_analysis(analysis, bq))
        assert bq.save_report.called, "save_report not called"
        return bq.save_report.call_args.kwargs

    def test_degraded_marker_persists_nulls_never_hold(self):
        kw = self._persist({
            "ticker": "TST", "_degraded": True, "_path": "degraded",
            "_degraded_reason": "both_paths_failed: x",
            "recommendation": None, "final_score": None,
            "risk_assessment": {}, "total_cost_usd": 0.0, "full_report": {},
        })
        assert kw["final_score"] is None
        assert kw["recommendation"] is None
        assert kw["summary"].startswith("DEGRADED:")
        assert kw["full_report"]["_degraded"] is True

    def test_normal_row_still_coerced_legacy(self):
        kw = self._persist({
            "ticker": "TST", "_path": "full", "recommendation": "BUY",
            "final_score": 7.5, "risk_assessment": {}, "total_cost_usd": 0.1,
            "full_report": {"quant": {}},
        })
        assert kw["final_score"] == 7.5 and kw["recommendation"] == "BUY"

    def test_degraded_marker_never_enters_analyses(self):
        # The marker's consumer contract: _run_and_persist_one returns None
        # for _degraded dicts. Source-contract assertion (the wiring lives
        # inside the cycle closure and is not directly invocable).
        import inspect
        src = inspect.getsource(al)
        assert 'if analysis.get("_degraded"):' in src
        assert '"lite", "full", "degraded"' in src


# ── criterion 1 supporting leg: retry-on-empty ─────────────────────────────

class _StubModel:
    model_name = "claude-sonnet-4-6"
    recommended_step_timeout = 150
    supports_thinking = False

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0

    def generate_content(self, prompt, **kw):
        self.calls += 1
        return self._responses.pop(0)


def _mini_orchestrator(settings):
    from backend.agents.orchestrator import AnalysisOrchestrator
    o = object.__new__(AnalysisOrchestrator)
    o.settings = settings
    o.enable_thinking = False
    o.thinking_budgets = {}
    o._cost_tracker = None
    return o


def _resp(text, thoughts=""):
    return SimpleNamespace(text=text, thoughts=thoughts, usage_metadata=None)


class TestRetryOnEmpty:
    def _gen(self, settings, responses):
        o = _mini_orchestrator(settings)
        m = _StubModel(responses)
        with patch("backend.agents.orchestrator.time.sleep"):
            out = o._generate_with_retry(m, "p", "TestAgent")
        return m.calls, out

    def test_errored_empty_retried_then_success(self):
        calls, out = self._gen(
            _settings(paper_synthesis_integrity_enabled=True),
            [_resp("", "errored: boom"), _resp("ok")],
        )
        assert calls == 2 and out.text == "ok"

    def test_rail_guard_skip_never_retried(self):
        calls, out = self._gen(
            _settings(paper_synthesis_integrity_enabled=True),
            [_resp("", "rail_guard_skipped: breaker_open")],
        )
        assert calls == 1 and out.text == ""

    def test_flag_off_no_retry(self):
        calls, out = self._gen(
            _settings(),
            [_resp("", "errored: boom")],
        )
        assert calls == 1 and out.text == ""

    def test_budget_bounds_attempts(self):
        # 1 + 2 extra attempts max (max_retries=3 loop default).
        calls, out = self._gen(
            _settings(paper_synthesis_integrity_enabled=True, claude_code_empty_retry_max=2),
            [_resp("", "errored: a"), _resp("", "errored: b"), _resp("", "errored: c")],
        )
        assert calls == 3 and out.text == ""  # legacy empty returned, never None


# ── criterion 2: timeout plumbing ──────────────────────────────────────────

class TestTimeoutPlumbing:
    def test_configured_timeout_and_step_budget(self):
        from backend.agents.claude_code_client import ClaudeCodeClient
        c = ClaudeCodeClient(model_name="claude-sonnet-4-6", timeout_s=200)
        assert c._timeout_s == 200
        assert c.recommended_step_timeout == 230  # timeout_s + 30

    def test_default_is_150(self):
        from backend.agents.claude_code_client import ClaudeCodeClient
        c = ClaudeCodeClient(model_name="claude-sonnet-4-6")
        assert c._timeout_s == 150 and c.recommended_step_timeout == 180

    def test_settings_field_defaults(self):
        from backend.config.settings import Settings
        f = Settings.model_fields
        assert f["claude_code_timeout_s"].default == 150
        assert f["claude_code_empty_retry_max"].default == 2
        assert f["paper_synthesis_integrity_enabled"].default is False
        assert f["paper_position_recommendation_fix_enabled"].default is False


# ── criterion 3: company_name fallback ─────────────────────────────────────

class TestCompanyNameFallback:
    def _persist(self, full_report):
        bq = MagicMock()
        asyncio.run(al._persist_analysis({
            "ticker": "TST", "_path": "full", "recommendation": "BUY",
            "final_score": 7.0, "risk_assessment": {}, "total_cost_usd": 0.1,
            "full_report": full_report,
        }, bq))
        return bq.save_report.call_args.kwargs

    def test_quant_fallback_when_market_data_absent(self):
        kw = self._persist({"quant": {"company_name": "Quant Name Co"}})
        assert kw["company_name"] == "Quant Name Co"

    def test_market_data_name_still_wins(self):
        kw = self._persist({
            "market_data": {"name": "MD Name"},
            "quant": {"company_name": "Quant Name Co"},
        })
        assert kw["company_name"] == "MD Name"

    def test_both_absent_stays_null(self):
        kw = self._persist({})
        assert kw["company_name"] is None


# ── criterion 4: meta-scorer fallback ──────────────────────────────────────

def _cands(scores):
    return [{"ticker": f"T{i}", "composite_score": s} for i, s in enumerate(scores)]


class TestMetaScorerFallback:
    def test_rank_normalized_spread_not_constant(self):
        convs = ms._rank_normalized_convictions(_cands([78, 92, 105, 121, 140, 163]))
        assert len(set(convs)) > 1, "must not be a constant"
        assert convs == sorted(convs), "monotone with composite order"
        assert convs[0] == 1 and convs[-1] == 10
        assert all(1 <= c <= 10 for c in convs)

    def test_legacy_clamp_saturates_flag_off(self):
        with patch("backend.config.settings.get_settings", return_value=_settings()):
            convs = ms._fallback_convictions(_cands([78, 92, 163]))
        assert convs == [10, 10, 10]  # the criterion-4 defect, preserved OFF

    def test_flag_on_dispatches_rank_normalization(self):
        with patch(
            "backend.config.settings.get_settings",
            return_value=_settings(paper_synthesis_integrity_enabled=True),
        ):
            convs = ms._fallback_convictions(_cands([78, 92, 163]))
        assert len(set(convs)) > 1

    def test_single_candidate_mid_scale(self):
        assert ms._rank_normalized_convictions(_cands([120])) == [5]

    def test_ties_share_rank(self):
        convs = ms._rank_normalized_convictions(_cands([100, 100, 100, 160]))
        assert convs[0] == convs[1] == convs[2]
        assert convs[3] == 10


class TestConvictionStreak:
    def test_streak_increments_and_resets(self, tmp_path, monkeypatch):
        monkeypatch.setattr(al, "_CONVICTION_STREAK_PATH", tmp_path / "streak.json")
        assert al._bump_conviction_fallback_streak(1) == 1
        assert al._bump_conviction_fallback_streak(1) == 2
        assert al._bump_conviction_fallback_streak(0) == 0
        assert al._bump_conviction_fallback_streak(1) == 1

    def test_survives_corrupt_file(self, tmp_path, monkeypatch):
        p = tmp_path / "streak.json"
        p.write_text("not json", encoding="utf-8")
        monkeypatch.setattr(al, "_CONVICTION_STREAK_PATH", p)
        assert al._bump_conviction_fallback_streak(1) == 1

    def test_streak_warn_wired_at_two(self):
        import inspect
        src = inspect.getsource(al)
        assert "conviction_fallback_streak" in src
        assert "_streak >= 2" in src


# ── criterion 5: signal_downgrade revival ──────────────────────────────────

def _pos(rec):
    return {
        "ticker": "AAA", "recommendation": rec, "quantity": 10.0,
        "avg_entry_price": 100.0, "cost_basis": 1000.0, "current_price": 100.0,
        "market_value": 1000.0, "stop_loss_price": 80.0, "sector": "Tech",
    }


def _hold_reeval():
    return {
        "ticker": "AAA", "recommendation": "HOLD", "final_score": 4.0,
        "analysis_date": "2026-07-08", "risk_assessment": {},
        "price_at_analysis": 100.0,
    }


class TestSignalDowngrade:
    def test_rule_matches_when_position_carries_verdict(self):
        orders = decide_trades([_pos("BUY")], [], [_hold_reeval()],
                               {"nav": 10000.0, "cash": 9000.0, "position_count": 1},
                               _settings())
        assert any(o.action == "SELL" and o.reason == "signal_downgrade" for o in orders)

    def test_rule_dead_on_legacy_reason_string(self):
        orders = decide_trades([_pos("new_buy_signal")], [], [_hold_reeval()],
                               {"nav": 10000.0, "cash": 9000.0, "position_count": 1},
                               _settings())
        assert not any(o.reason == "signal_downgrade" for o in orders)

    def test_buy_order_carries_analysis_recommendation(self):
        a = {"ticker": "BBB", "recommendation": "BUY", "final_score": 8.0,
             "price_at_analysis": 50.0, "analysis_date": "x", "risk_assessment": {}}
        orders = decide_trades([], [a], [],
                               {"nav": 10000.0, "cash": 10000.0, "position_count": 0},
                               _settings())
        buys = [o for o in orders if o.action == "BUY"]
        assert buys and buys[0].analysis_recommendation == "BUY"

    def test_unsafe_combination_warns(self, caplog):
        import logging
        with caplog.at_level(logging.WARNING, logger="backend.services.portfolio_manager"):
            decide_trades([], [], [],
                          {"nav": 10000.0, "cash": 10000.0, "position_count": 0},
                          _settings(paper_position_recommendation_fix_enabled=True))
        assert any("interaction hazard" in r.message for r in caplog.records)

    def _execute_buy(self, settings):
        from backend.services.paper_trader import PaperTrader
        bq = MagicMock()
        bq.get_paper_portfolio.return_value = {
            "portfolio_id": "default", "current_cash": 10000.0,
            "total_nav": 10000.0, "starting_capital": 10000.0,
            "total_pnl_pct": 0.0, "benchmark_return_pct": 0.0,
            "inception_date": "2026-01-01T00:00:00+00:00",
        }
        bq.get_paper_positions.return_value = []
        bq.get_paper_trades_for_ticker_since.return_value = []
        t = PaperTrader(settings, bq)
        trade = t.execute_buy(
            ticker="BBB", amount_usd=500.0, price=50.0,
            reason="new_buy_signal", analysis_recommendation="BUY",
        )
        assert trade is not None
        assert bq.save_paper_position.called
        return bq.save_paper_position.call_args.args[0]

    def test_pos_row_stores_verdict_flag_on(self):
        row = self._execute_buy(_settings(paper_position_recommendation_fix_enabled=True))
        assert row["recommendation"] == "BUY"

    def test_pos_row_stores_reason_flag_off(self):
        row = self._execute_buy(_settings())
        assert row["recommendation"] == "new_buy_signal"


# ── criterion 6: RiskJudge advisory context ────────────────────────────────

class TestRiskJudgeAdvisoryCtx:
    def test_ctx_gate_references_both_flags(self):
        # The gate lives inside the cycle closure (not directly invocable);
        # source-contract assertion mirrors TestDegradedPersistence's approach.
        import inspect
        src = inspect.getsource(al)
        i = src.find("_rj_portfolio_ctx = \"\"")
        assert i > 0
        window = src[i - 600:i + 400]
        assert "paper_risk_judge_reject_binding" in window
        assert "paper_synthesis_integrity_enabled" in window
