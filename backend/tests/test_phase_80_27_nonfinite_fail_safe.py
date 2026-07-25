"""phase-80.27 -- a data outage must never become a trading verdict.

The same NaN that 500'd the Signals page (80.1) does NOT 500 the analysis
pipeline. There it was silently converted into a confident NEUTRAL trading
verdict that reached the paper-trading loop, while the data-quality gate
that exists to catch exactly this reported 100%.

Two halves, deliberately shipped differently:

  HALF A -- the DETECTOR (`info_gap._assess_source_status`) ships ON.
    It is a pure tightening: it can only move a source toward MISSING,
    i.e. more gating and fewer trades. It is also what the step's
    immutable verification command exercises.

  HALF B -- the TOOL VERDICTS ship DARK behind
    `tools_nonfinite_fail_safe_enabled` (default False). OFF must be
    byte-identical legacy behaviour; ON returns the documented
    signal: 'ERROR' from .claude/rules/backend-tools.md.

FAIL-SAFE ONLY: every assertion below checks that a non-finite input
produces LESS actionable output, never more.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from backend.agents.info_gap import (
    _SOURCE_CRITICALITY,
    _assess_source_status,
    _has_non_finite,
)

NAN = float("nan")


def _count_non_finite(obj, depth: int = 0) -> int:
    if isinstance(obj, float):
        return 0 if math.isfinite(obj) else 1
    if isinstance(obj, dict):
        return sum(_count_non_finite(v, depth + 1) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(_count_non_finite(v, depth + 1) for v in obj)
    return 0


# ===========================================================================
# HALF A -- the detector. Criterion 1.
# ===========================================================================


def test_the_immutable_verification_payload_is_not_sufficient():
    """The step's own verification command, as a test.

    Pre-fix this returned SUFFICIENT -- a HIGH-criticality source whose every
    number was NaN was reported as "Data available and complete".
    """
    p = {
        "signal": "NEUTRAL",
        "summary": "3M return: +nan% vs sector +nan%",
        "stock_returns": {"1mo": NAN},
    }
    assert _assess_source_status("sector", p) != "SUFFICIENT"
    assert _assess_source_status("sector", p) == "MISSING"


def test_missing_is_the_status_that_reaches_the_retry_loop():
    """Criterion 1 says the EXISTING retry loop must fire.

    Only ERROR/NO_DATA -> MISSING routes into critical_gaps. PARTIAL would
    also be "not SUFFICIENT" but would silently fail the criterion, so this
    pins the exact string rather than merely 'not SUFFICIENT'.
    """
    poisoned = {"signal": "NEUTRAL", "summary": "ok", "data": {"score": NAN}}
    assert _assess_source_status("sector", poisoned) == "MISSING"


def test_nan_only_in_the_payload_is_caught():
    """No 'nan' in the prose at all -- the numeric scan must still fire."""
    p = {"signal": "BULLISH", "summary": "Everything looks great.", "data": {"x": NAN}}
    assert _assess_source_status("sector", p) == "MISSING"


def test_infinities_are_caught_too():
    for bad in (float("inf"), float("-inf")):
        p = {"signal": "BULLISH", "summary": "fine", "data": {"x": bad}}
        assert _assess_source_status("sector", p) == "MISSING", bad


def test_nan_in_the_prose_is_caught():
    p = {"signal": "NEUTRAL", "summary": "3M return: +nan% vs sector +nan%.", "data": {}}
    assert _assess_source_status("sector", p) == "MISSING"


def test_ordinary_words_containing_nan_are_NOT_flagged():
    """THE FALSE-POSITIVE TRAP.

    'financial', 'governance', 'covenant', 'tenant' and 'nanotech' all
    contain the substring 'nan'. A naive `"nan" in summary.lower()` would
    mark nearly every real summary MISSING and cause a self-inflicted
    analysis outage -- strictly worse than the bug being fixed. The token
    regex must use non-letter boundaries.
    """
    for word in ("financial", "governance", "covenant", "tenant", "nanotech", "finance"):
        p = {"signal": "BULLISH", "summary": f"Strong {word} position.", "data": {"x": 1.0}}
        assert _assess_source_status("sector", p) == "SUFFICIENT", word


def test_no_data_verdict_is_now_handled():
    """backend/tools/alt_data.py already returned NO_DATA, and the detector
    had no case for it -- so an explicit failure was classified SUFFICIENT.
    Same defect shape as the NaN one."""
    assert _assess_source_status("alt_data", {"signal": "NO_DATA", "summary": "rate limited"}) == "MISSING"


def test_healthy_payload_is_still_sufficient():
    """Fail-safe must not mean fail-always."""
    p = {"signal": "BULLISH", "summary": "3M return: +12.4% vs sector +3.1%.",
         "data": {"score": 0.42, "features": {"a": 1.0, "b": -2.5}}}
    assert _assess_source_status("sector", p) == "SUFFICIENT"


def test_existing_statuses_are_unchanged():
    assert _assess_source_status("sector", {"signal": "ERROR", "summary": "x"}) == "MISSING"
    assert _assess_source_status("sector", {"signal": "SKIPPED", "summary": ""}) == "SKIPPED"
    assert _assess_source_status("sector", {}) == "MISSING"
    assert _assess_source_status("sector", None) == "MISSING"
    assert _assess_source_status("sector", {"signal": "N/A", "summary": ""}) == "PARTIAL"


def test_quant_model_is_assessed_at_all():
    """It was ABSENT from _SOURCE_CRITICALITY, so detect_info_gaps -- which
    iterates that dict -- never assessed it, never counted it, and it could
    never enter critical_gaps. Flipping it to ERROR would have done nothing.
    """
    assert "quant_model" in _SOURCE_CRITICALITY
    assert _SOURCE_CRITICALITY["quant_model"] == "HIGH"


def test_booleans_are_not_scanned_as_floats():
    assert _has_non_finite({"flag": True, "n": 3, "s": "nan"}) is False


def test_scan_recurses_into_lists_and_nested_dicts():
    assert _has_non_finite({"a": {"b": [{"c": NAN}]}}) is True
    assert _has_non_finite({"a": {"b": [{"c": 1.0}]}}) is False


# ===========================================================================
# HALF B -- the tool verdicts, behind the flag. Criteria 2, 3, 6.
# ===========================================================================

_POISONED_RETURNS = {"1mo": NAN, "3mo": NAN, "6mo": NAN, "1y": NAN}


def _sector_with_flag(monkeypatch, enabled: bool, returns=None):
    """Drive get_sector_analysis with a deterministic non-finite frame."""
    import pandas as pd

    from backend.tools import sector_analysis

    monkeypatch.setattr(sector_analysis, "_nonfinite_fail_safe_enabled", lambda: enabled)

    vals = returns if returns is not None else _POISONED_RETURNS

    def fake_compute(ticker_obj, period):
        return vals.get(period, NAN)

    monkeypatch.setattr(sector_analysis, "_compute_return", fake_compute)

    class _T:
        def __init__(self, *a, **k):
            pass

        @property
        def info(self):
            return {"longName": "Apple Inc.", "sector": "Technology",
                    "industry": "Consumer Electronics"}

        def history(self, *a, **k):
            return pd.DataFrame({"Close": [1.0, 2.0]})

    monkeypatch.setattr(sector_analysis.yf, "Ticker", _T)
    return sector_analysis.get_sector_analysis("AAPL")


def test_sector_returns_ERROR_not_NEUTRAL_on_nonfinite(monkeypatch):
    """Criterion 2 -- the documented failure shape, never a fabricated verdict."""
    out = _sector_with_flag(monkeypatch, enabled=True)
    assert out["signal"] == "ERROR", out.get("signal")
    assert out["signal"] != "NEUTRAL"


def test_sector_flag_OFF_is_byte_identical_legacy(monkeypatch):
    """Dark-launch contract: OFF reproduces today's (wrong) behaviour exactly,
    so enabling is the operator's decision and shipping is inert."""
    out = _sector_with_flag(monkeypatch, enabled=False)
    assert out["signal"] == "NEUTRAL"
    assert "nan" in out["summary"].lower()


def test_sector_error_payload_carries_no_nan_prose_or_numbers(monkeypatch):
    """Criterion 3 -- assert on BOTH the rendered summary and the payload."""
    out = _sector_with_flag(monkeypatch, enabled=True)
    assert "nan" not in out["summary"].lower()
    assert _count_non_finite(out) == 0
    import json
    assert "NaN" not in json.dumps(out)


def test_sector_error_payload_is_flagged_missing_by_info_gap(monkeypatch):
    """End-to-end: the tool's ERROR must reach critical_gaps."""
    out = _sector_with_flag(monkeypatch, enabled=True)
    assert _assess_source_status("sector", out) == "MISSING"


def test_sector_still_produces_a_real_verdict_on_finite_data(monkeypatch):
    """FAIL-SAFE MUST NOT MEAN FAIL-ALWAYS.

    With finite inputs and the flag ON, the tool must still classify
    normally -- otherwise the guard is a denial-of-service on itself.
    """
    finite = {"1mo": 5.0, "3mo": 12.0, "6mo": 20.0, "1y": 30.0}
    out = _sector_with_flag(monkeypatch, enabled=True, returns=finite)
    assert out["signal"] != "ERROR"
    assert out["signal"] in {"DOUBLE_TAILWIND", "SECTOR_TAILWIND",
                             "STOCK_OUTPERFORMING", "LAGGING", "NEUTRAL"}


def _quant_with_flag(monkeypatch, enabled: bool, features=None):
    from backend.tools import quant_model

    monkeypatch.setattr(quant_model, "_nonfinite_fail_safe_enabled", lambda: enabled)
    feats = features if features is not None else {
        "momentum_1m": NAN, "momentum_3m": NAN, "annualized_volatility": NAN,
    }
    monkeypatch.setattr(quant_model, "_build_live_features", lambda t: dict(feats))
    monkeypatch.setattr(quant_model, "get_latest_mda", lambda: {"momentum_1m": 1.0})
    return quant_model.get_quant_model_signal("AAPL")


def test_quant_returns_ERROR_not_NEUTRAL_on_nonfinite(monkeypatch):
    out = _quant_with_flag(monkeypatch, enabled=True)
    assert out["signal"] == "ERROR"
    assert out["signal"] != "NEUTRAL"


def test_quant_does_not_claim_backtest_mda_over_nonfinite_factors(monkeypatch):
    """Criterion 6 -- the payload must not advertise that real walk-forward
    MDA weights were applied when every factor is garbage."""
    out = _quant_with_flag(monkeypatch, enabled=True)
    assert out["mda_source"] != "backtest"
    assert out["mda_source"] == "non_finite_inputs"


def test_quant_flag_OFF_is_byte_identical_legacy(monkeypatch):
    out = _quant_with_flag(monkeypatch, enabled=False)
    assert out["signal"] == "NEUTRAL"
    assert out["mda_source"] == "backtest"
    assert not math.isfinite(out["score"])


def test_quant_error_payload_is_clean_and_flagged(monkeypatch):
    out = _quant_with_flag(monkeypatch, enabled=True)
    assert _count_non_finite(out) == 0
    assert "nan" not in out["summary"].lower()
    assert out["top_factors"] == []
    assert _assess_source_status("quant_model", out) == "MISSING"


def test_quant_still_scores_on_finite_features(monkeypatch):
    """Fail-safe must not mean fail-always."""
    out = _quant_with_flag(monkeypatch, enabled=True,
                           features={"momentum_1m": 0.2, "momentum_3m": 0.1})
    assert out["signal"] != "ERROR"
    assert math.isfinite(out["score"])


# ===========================================================================
# Fixture pins -- bind to the SUBJECT, never assert a library fact.
# (feedback_mutation_test_guards_and_fixtures; 80.1 cycle 1 shipped exactly
#  that mistake and the Q/A caught it.)
# ===========================================================================


# ===========================================================================
# Cycle 2 -- the three holes Q/A found. Each of these existed because a
# guard's ACTIVATION path, or its OPERAND COMPLETENESS, was asserted in a
# comment instead of pinned by a test.
# ===========================================================================


def test_partial_non_finite_sector_still_yields_no_nan_in_the_payload(monkeypatch):
    """D1 -- THE LEAK Q/A DEMONSTRATED.

    The ladder guard only inspects the three 3mo operands. A bad bar at the
    START of the 1mo window poisons that horizon alone (`_compute_return` is
    Close[-1]/Close[0]), so `{"1mo": NaN, "3mo": 5.0, ...}` produced a
    genuine-looking NEUTRAL whose payload still carried NaN -- and
    `json.dumps` at backend/config/prompts.py:537 emits a bare `NaN` token
    into the sector agent's prompt.

    Criterion 3 is about the payload the agent RECEIVES. This pins it.
    """
    import json

    partial = {"1mo": NAN, "3mo": 5.0, "6mo": 6.0, "1y": 7.0}
    out = _sector_with_flag(monkeypatch, enabled=True, returns=partial)

    assert out["signal"] == "ERROR", (
        f"partial non-finite returned {out['signal']!r} -- a partially "
        "fictional verdict"
    )
    assert _count_non_finite(out) == 0
    assert "NaN" not in json.dumps(out)
    assert "nan" not in out["summary"].lower()


def test_non_finite_in_sector_performance_only_still_trips_the_guard(monkeypatch):
    """N-A -- pin the guard's SCOPE, not just its existence.

    Q/A cycle 2: narrowing `_count_non_finite(payload)` to
    `payload["stock_returns"]` left the suite green at 29/29 while
    behaviourally reopening the leak for inputs whose only non-finite value
    sits in `sector_performance` or `peers`. Production code was correct; the
    PIN was narrow -- the same "the test doesn't cover the seam" shape as D1
    and D2.

    Here every *return* is finite, so the ladder guard cannot fire; only a
    whole-payload scan catches it.
    """
    import json

    import pandas as pd

    from backend.tools import sector_analysis

    monkeypatch.setattr(sector_analysis, "_nonfinite_fail_safe_enabled", lambda: True)

    class _T:
        def __init__(self, sym="AAPL", *a, **k):
            self.sym = sym

        @property
        def info(self):
            return {"longName": "Apple Inc.", "sector": "Technology",
                    "industry": "Consumer Electronics"}

        def history(self, *a, **k):
            return pd.DataFrame({"Close": [1.0, 2.0]})

    monkeypatch.setattr(sector_analysis.yf, "Ticker", _T)

    # sector_performance is built from `_compute_return(yf.Ticker(etf), "3mo")`
    # for all 11 SPDR ETFs, so the poison has to be TICKER-SPECIFIC. AAPL maps
    # to Technology/XLK, so poisoning XLE leaves stock_returns, sector_returns
    # (XLK) and spy_returns entirely finite -- the ladder guard cannot fire and
    # only a whole-payload scan can catch this.
    def fake_compute(ticker_obj, period):
        if getattr(ticker_obj, "sym", "") == "XLE":
            return NAN
        return {"1mo": 1.0, "3mo": 5.0, "6mo": 6.0, "1y": 7.0}.get(period, 1.0)

    monkeypatch.setattr(sector_analysis, "_compute_return", fake_compute)

    out = sector_analysis.get_sector_analysis("AAPL")
    blob = json.dumps(out)
    assert "NaN" not in blob, f"NaN leaked into the agent payload: {blob[:200]}"
    assert _count_non_finite(out) == 0
    assert out["signal"] == "ERROR"


def test_partial_non_finite_quant_feature_trips_the_guard(monkeypatch):
    """D3 -- one non-finite factor among finite ones must still fail safe."""
    out = _quant_with_flag(
        monkeypatch, enabled=True,
        features={"momentum_1m": 0.2, "momentum_3m": NAN, "roe": 0.15},
    )
    assert out["signal"] == "ERROR"
    assert out["mda_source"] == "non_finite_inputs"


def test_the_real_flag_read_path_executes(monkeypatch):
    """D2 -- kill the dead-flag-read hole.

    Every other Half-B test monkeypatches `_nonfinite_fail_safe_enabled`
    itself, so the PRODUCTION flag-read executed in zero tests: Q/A mutated
    the helper to `return False` (and to a misspelled settings attribute)
    and the whole 24-test suite still passed. A guard whose activation path
    cannot fail when broken does not count.

    This drives the REAL helper against a stubbed settings object.
    """
    import backend.config.settings as settings_mod
    from backend.tools import quant_model, sector_analysis

    class _Stub:
        tools_nonfinite_fail_safe_enabled = True

    monkeypatch.setattr(settings_mod, "get_settings", lambda: _Stub())

    # the real helpers, not a monkeypatched stand-in
    assert sector_analysis._nonfinite_fail_safe_enabled() is True
    assert quant_model._nonfinite_fail_safe_enabled() is True

    class _StubOff:
        tools_nonfinite_fail_safe_enabled = False

    monkeypatch.setattr(settings_mod, "get_settings", lambda: _StubOff())
    assert sector_analysis._nonfinite_fail_safe_enabled() is False
    assert quant_model._nonfinite_fail_safe_enabled() is False


def test_flag_read_fails_open_to_legacy_when_settings_explode(monkeypatch):
    """The gate must never be the reason an analysis crashes."""
    import backend.config.settings as settings_mod
    from backend.tools import sector_analysis

    def _boom():
        raise RuntimeError("settings unavailable")

    monkeypatch.setattr(settings_mod, "get_settings", _boom)
    assert sector_analysis._nonfinite_fail_safe_enabled() is False


def test_infinity_in_PROSE_is_caught(monkeypatch):
    """D4 -- the `inf` branch of the token regex was only ever exercised via
    the payload scan, never through prose."""
    p = {"signal": "BULLISH", "summary": "P/E ratio is inf after the writedown.",
         "data": {"x": 1.0}}
    assert _assess_source_status("sector", p) == "MISSING"


def test_fixture_poison_is_actually_non_finite():
    assert all(not math.isfinite(v) for v in _POISONED_RETURNS.values()), (
        "the sector fixture is finite -- every Half-B assertion above would "
        "pass against an implementation that does nothing"
    )


def test_flag_defaults_off_in_settings():
    """The dark-launch contract itself. If this ever defaults True, the step
    has silently changed live trading behaviour without an operator token."""
    from backend.config.settings import get_settings

    assert getattr(get_settings(), "tools_nonfinite_fail_safe_enabled", None) is False
