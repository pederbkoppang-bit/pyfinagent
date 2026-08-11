"""phase-86.41 -- the quant sub-agent must be isolated at the DEPENDENCY.

`backend/agents/orchestrator.py` had exactly one unguarded `await self.<sub-agent>`
call inside `run_full_analysis` (the quant step). Its three siblings -- RAG,
ingestion and the phase-32.3 call -- already fail open. When SEC.gov 429s the CIK
map, the REMOTE quant Cloud Function's fetch returns None and raises NoneType at
`get_cik` in `/workspace/main.py` (cite the FUNCTION, not a line -- the
retained logs carry SIX distinct addresses for it across deployments, of
which :89 is the LEAST common at 14x); that exception propagated out of
`run_quant_agent`, aborted the whole ticker, and demoted the entire analysis to
the lite path at `autonomous_loop.py:2201`.

These tests DRIVE THE REAL `run_full_analysis` rather than reading its source.
A source-level assertion cannot distinguish "the guard is present" from "the guard
works", and this file exists because the previous three rounds of this class of
work were caught asserting the former. The pipeline is halted deliberately from
the `on_step` callback once the quant step reports, so the test observes the
orchestrator's own progress channel and never runs the remaining ~640 lines.

MUTATION: delete the try/except at the quant call and `test_quant_failure_*`
fails with `_Boom` escaping instead of `_StopPipeline`. The guard is the only
thing standing between those two outcomes, so the cell discriminates.
"""

import asyncio
import types

from backend.agents import orchestrator as orch_mod
from backend.agents.orchestrator import AnalysisOrchestrator


class _Boom(RuntimeError):
    """Stands in for the remote NoneType raised inside the quant Cloud Function."""


class _StopPipeline(Exception):
    """Raised from on_step to halt the run once the quant step has reported."""


def _settings():
    # The only four settings attributes run_full_analysis reads before the
    # quant step -- derived by grep, not by guess.
    return types.SimpleNamespace(
        gemini_model="gemini-2.5-flash",
        lite_mode=False,
        max_debate_rounds=1,
        max_risk_debate_rounds=1,
    )


def _orchestrator(monkeypatch, *, quant_raises):
    # __init__ is bypassed ON PURPOSE: it constructs live Gemini client bundles
    # from GCP credentials, none of which the code under test touches. Between
    # the top of run_full_analysis and the quant guard the orchestrator reads
    # only `self.settings` (four attributes, enumerated by grep), the three
    # sub-agent coroutines stubbed below, and the `_is_sec_covered` classmethod
    # -- all of which are supplied here. `self._cost_tracker` is assigned by
    # run_full_analysis itself. test_sec_covered_precondition below asserts the
    # branch this setup depends on is actually the one taken.
    o = object.__new__(AnalysisOrchestrator)
    o.settings = _settings()

    async def _market_intel(ticker):
        return {"sentiment_summary": [], "source": "yfinance"}

    async def _ingestion(ticker):
        return None

    async def _quant(ticker):
        if quant_raises:
            raise _Boom("'NoneType' object has no attribute 'get'")
        return {"company_name": "Apple Inc.", "cik": "0000320193"}

    o.fetch_market_intel = _market_intel
    o.run_ingestion_agent = _ingestion
    o.run_quant_agent = _quant

    # Replace the name in the orchestrator module only -- do not mutate the
    # shared yfinance tool module for the rest of the session.
    #
    # The call COUNTER is the load-bearing part. Reading the progress message
    # alone cannot distinguish "the guard stayed out of the way" from "the guard
    # fired, ran the fallback, and overwrote a good quant report" -- the first
    # 'quant/completed' event says "Financial data collected" either way. A
    # yfinance call on the healthy path is the unambiguous signature of the
    # guard firing when it must not. (Mutation cell M3 survived without this.)
    o._yf_calls = []

    def _yf(ticker):
        o._yf_calls.append(ticker)
        return {
            "company_name": "Apple Inc.",
            "sector": "Technology",
            "valuation": {"Current Price": 231.4},
        }

    monkeypatch.setattr(
        orch_mod,
        "yfinance_tool",
        types.SimpleNamespace(get_comprehensive_financials=_yf),
    )
    return o


def _run_until_quant(o):
    """Run the real pipeline, stopping the moment the quant step reports."""
    events = []

    def on_step(name, status, msg=""):
        events.append((name, status, msg))
        if name == "quant" and status == "completed":
            raise _StopPipeline

    escaped = None
    try:
        asyncio.run(o.run_full_analysis("AAPL", on_step=on_step))
    except _StopPipeline:
        pass
    except BaseException as exc:  # noqa: BLE001 -- we assert on what escaped
        escaped = exc
    return events, escaped


def _quant_msg(events):
    for name, status, msg in events:
        if name == "quant" and status == "completed":
            return msg
    return None


# ── precondition: without this the whole file passes vacuously ───────────────


def test_sec_covered_precondition():
    """The guard lives in the `if _sec_covered:` branch.

    If AAPL were not SEC-covered, every test below would exercise the non-US
    `else` path instead and pass while proving nothing about the guard.
    """
    assert AnalysisOrchestrator._is_sec_covered("AAPL") is True, (
        "test ticker is not SEC-covered -- these tests would exercise the "
        "non-US branch and never reach the guard"
    )


# ── the guard ────────────────────────────────────────────────────────────────


def test_quant_failure_does_not_abort_the_ticker(monkeypatch):
    """A raising quant sub-agent must NOT propagate out of run_full_analysis."""
    o = _orchestrator(monkeypatch, quant_raises=True)
    events, escaped = _run_until_quant(o)

    assert not isinstance(escaped, _Boom), (
        "the quant exception escaped run_full_analysis -- the whole ticker would "
        "abort and autonomous_loop.py:2201 would demote it to the lite path"
    )
    assert escaped is None, f"unexpected exception escaped: {escaped!r}"
    assert _quant_msg(events) is not None, (
        "the quant step never reported 'completed' -- the pipeline did not "
        "continue past the failure"
    )
    # POSITIVE CONTROL for the `o._yf_calls == []` assertion in the healthy
    # test. Without this, an always-empty counter (a stub that never records)
    # would make that assertion pass vacuously forever.
    assert o._yf_calls == ["AAPL"], (
        "the degraded path did not call the yfinance fallback exactly once "
        f"(calls={o._yf_calls}) -- the counter cannot detect a fallback, so the "
        "healthy-path assertion that it stays empty proves nothing"
    )


def test_quant_failure_is_reported_as_degraded_and_names_the_cause(monkeypatch):
    """The progress channel must say it degraded, and name the exception type."""
    o = _orchestrator(monkeypatch, quant_raises=True)
    events, _ = _run_until_quant(o)
    msg = _quant_msg(events)

    assert "degraded" in msg.lower(), f"quant step did not report degradation: {msg!r}"
    assert "_Boom" in msg, (
        f"the exception type is not named in the progress message: {msg!r} -- an "
        "operator cannot tell a 429-driven degradation from any other"
    )


def test_degraded_reason_does_not_impersonate_the_non_us_branch(monkeypatch):
    """The whole step exists because one string collapsed two distinct causes.

    phase-60.1's yfinance fallback exists for NON-US LISTINGS. Reusing its
    'non-US listing' reason for a SEC rate-limit would relabel a transient
    upstream failure as a permanent listing-coverage fact -- exactly the
    wrapper-string collapse that made the 86.38 census read 6 code defects
    where there were 0.
    """
    o = _orchestrator(monkeypatch, quant_raises=True)
    events, _ = _run_until_quant(o)
    msg = _quant_msg(events)

    assert "non-US" not in msg, (
        f"degraded quant claims a non-US listing, which is false: {msg!r}"
    )


# ── the positive control: the guard must not fire when quant succeeds ─────────


def test_healthy_quant_is_untouched_by_the_guard(monkeypatch):
    """Without this, every assertion above would also pass on a no-op guard."""
    o = _orchestrator(monkeypatch, quant_raises=False)
    events, escaped = _run_until_quant(o)
    msg = _quant_msg(events)

    assert escaped is None, f"healthy path raised: {escaped!r}"
    assert msg == "Financial data collected", (
        f"the healthy quant path changed message: {msg!r}"
    )
    assert "degraded" not in msg.lower()
    # The load-bearing assertion. The message above is identical whether the
    # guard stayed out of the way or fired and overwrote a good report, so it
    # cannot tell those apart. A yfinance call on the healthy path can only
    # mean the fallback ran.
    assert o._yf_calls == [], (
        "the guard fired on a HEALTHY quant -- the yfinance fallback ran and "
        f"overwrote a good quant report (calls={o._yf_calls})"
    )


# ── the fallback builder's cause string ──────────────────────────────────────


def test_default_reason_is_byte_identical_to_phase_60_1(monkeypatch):
    """Adding the `reason` param must not move the phase-60.1 non-US string."""
    from backend.agents.orchestrator import _quant_from_yfinance

    got = _quant_from_yfinance("005930.KS", None)["part_1_financials"]["source"]
    assert got == (
        "yfinance only -- SEC EDGAR skipped "
        "(non-US listing, phase-60.1 KR-aware skip)"
    ), f"phase-60.1 default string regressed: {got!r}"


def test_degraded_reason_is_distinct_but_keeps_the_pinned_substring():
    from backend.agents.orchestrator import _quant_from_yfinance

    deg = _quant_from_yfinance(
        "AAPL", None, reason="quant agent failed: AttributeError"
    )["part_1_financials"]["source"]

    assert "non-US" not in deg
    assert "AttributeError" in deg
    # test_phase_60_1_deep_pipeline.py:180 pins this substring on the shape.
    assert "SEC EDGAR skipped" in deg
