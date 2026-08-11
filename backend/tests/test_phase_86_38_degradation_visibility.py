"""phase-86.38 -- a degraded cycle that does not PAGE must still be RECORDED.

THE DEFECT, measured on the 2026-08-10 book cycle. Three of six analyses fell
back from the 28-agent orchestrator to the lite Claude analyser after Vertex
returned `429 RESOURCE_EXHAUSTED`. `_fallback_rate_check` fires only when the
fallback fraction STRICTLY exceeds its threshold, and `3/6 = 0.500` does not
exceed `0.500`, so the alarm correctly stayed quiet. But every degradation field
was set INSIDE the `if _fb_fire:` branch -- so the cycle left no durable trace at
all, and the only evidence the book had traded off the thin fallback was a
`grep` of `backend.log`.

WHAT THIS MODULE PINS, and what it deliberately does NOT:

  * It pins that the RATE IS RECORDED regardless of firing, and that the
    recorded value carries whether the alarm fired.
  * It pins that PAGING BEHAVIOUR IS UNCHANGED -- `_fallback_rate_check` is the
    same pure predicate with the same strict `>`, verified here against the
    boundary that produced the real incident.
  * It does NOT re-tune the threshold. `> vs >=` is pinned by
    `test_phase_60_1_deep_pipeline.py::test_fallback_alarm_threshold_is_strictly_greater_than`
    and changing an alarm's sensitivity is an operator decision.
"""
from __future__ import annotations

import inspect

import pytest

from backend.services.autonomous_loop import _fallback_rate_check
from backend.services import autonomous_loop as al
from backend.services.cycle_health import CycleHealthLog


def _mk(ticker: str, reason: str | None = None) -> dict:
    d: dict = {"ticker": ticker}
    if reason:
        d["_fallback_reason"] = reason
    return d


# ---------------------------------------------------------------------------
# 1. The real incident, reproduced at the boundary.
# ---------------------------------------------------------------------------

def test_the_2026_08_10_boundary_does_not_page():
    """3 of 6 is EXACTLY 0.5 and must not fire -- this is the measured case."""
    analyses = [
        _mk("HPE", "ClientError: 429 RESOURCE_EXHAUSTED"),
        _mk("CRWD", "ClientError: 429 RESOURCE_EXHAUSTED"),
        _mk("HUM", "ClientError: 429 RESOURCE_EXHAUSTED"),
        _mk("DELL"), _mk("NTAP"), _mk("PANW"),
    ]
    fire, n_fb, n_total, reasons = _fallback_rate_check(analyses, 0.5)
    assert (n_fb, n_total) == (3, 6)
    assert n_fb / n_total == 0.5
    assert fire is False, "paging behaviour must be UNCHANGED by phase-86.38"
    # The reasons are available even though nothing pages -- that is the whole
    # point: the data existed, it simply had nowhere to go.
    assert set(reasons) == {"HPE", "CRWD", "HUM"}
    assert all("429" in r for r in reasons.values())


def test_one_more_degraded_ticker_would_have_paged():
    """The boundary is real, not an artefact: 4/6 fires."""
    analyses = [_mk(t, "429") for t in ("HPE", "CRWD", "HUM", "DELL")] + [
        _mk("NTAP"), _mk("PANW"),
    ]
    fire, n_fb, n_total, _ = _fallback_rate_check(analyses, 0.5)
    assert (n_fb, n_total) == (4, 6) and fire is True


# ---------------------------------------------------------------------------
# 2. Record-always: the source change, asserted against the SOURCE of the
#    branch rather than by re-running a whole trading cycle.
#
#    This is a source assertion and it is labelled as one. It is NOT the only
#    coverage: `_fallback_rate_check` above is driven for real, and the
#    persistence leg below is driven for real. What cannot be driven here is
#    `run_cycle` itself, which needs BQ, a broker and ~80 minutes.
# ---------------------------------------------------------------------------

def test_degradation_fields_are_set_outside_the_fire_branch():
    src = inspect.getsource(al)
    i_record = src.find('summary["fallback_rate"]')
    i_fire = src.find("if _fb_fire:", src.find("_fb_fire, _n_fb, _n_fb_total"))
    assert i_record != -1, "the fallback_rate record site vanished"
    assert i_fire != -1, "the paging branch vanished"
    assert i_record < i_fire, (
        "fallback_rate is set INSIDE/AFTER the `if _fb_fire:` branch again -- "
        "a sub-threshold degraded cycle would leave no durable trace, which is "
        "the exact defect phase-86.38 removed"
    )
    assert 'summary["fallback_alarm_fired"]' in src, (
        "the recorded rate must say whether it paged, else a reader cannot tell "
        "a quiet-because-fine cycle from a quiet-because-below-threshold one"
    )


def test_intended_path_is_gone_and_not_merely_unused():
    """`_intended_path` was write-only AND redundant; it was removed."""
    src = inspect.getsource(al)
    code_lines = [
        ln for ln in src.splitlines()
        if "_intended_path" in ln and not ln.strip().startswith("#")
    ]
    assert code_lines == [], f"_intended_path is live code again: {code_lines}"
    # Positive control: the field it was redundant WITH must still be written,
    # otherwise this test would pass by having deleted the wrong thing.
    assert '_lite["_fallback_reason"] = _fb_reason' in src


# ---------------------------------------------------------------------------
# 3. Persistence: driven for real against a temp history file.
# ---------------------------------------------------------------------------

@pytest.fixture()
def health(tmp_path, monkeypatch):
    import backend.services.cycle_health as ch
    monkeypatch.setattr(ch, "_HISTORY_PATH", tmp_path / "cycle_history.jsonl")
    monkeypatch.setattr(ch, "_HEARTBEAT_PATH", tmp_path / "hb.json")
    return CycleHealthLog()


def test_degradation_is_persisted_on_a_quiet_cycle(health):
    """A cycle that degraded below the paging threshold is still on the record."""
    health.record_cycle_start("c1")
    health.record_cycle_end(
        cycle_id="c1", started_at="2026-08-10T18:00:00Z", status="completed",
        degradation={"fallback_rate": "3/6", "fallback_alarm_fired": False,
                     "fallback_reasons": {"HPE": "429 RESOURCE_EXHAUSTED"}},
    )
    rows = health.last_cycles(5)
    assert rows, "no cycle row was written at all"
    deg = rows[0].get("degradation")
    assert deg, "degradation is absent -- the quiet degraded cycle is invisible again"
    assert deg["fallback_rate"] == "3/6"
    assert deg["fallback_alarm_fired"] is False


def test_degradation_defaults_empty_and_breaks_no_existing_caller(health):
    """Every pre-86.38 call site omits `degradation`; that must still work."""
    health.record_cycle_start("c2")
    health.record_cycle_end(
        cycle_id="c2", started_at="2026-08-10T18:00:00Z", status="completed",
        funnel={"universe_size": 10},
    )
    row = health.last_cycles(5)[0]
    assert row["degradation"] == {}
    assert row["funnel"] == {"universe_size": 10}, (
        "the 66.2 funnel must be untouched -- degradation is a SEPARATE key, "
        "not a widening of the funnel"
    )
