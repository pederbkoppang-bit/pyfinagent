"""phase-86.38 -- a degraded cycle that does not PAGE must still be RECORDED.

THE DEFECT, measured on the 2026-08-10 book cycle. Three of six analyses fell
back from the 28-agent orchestrator to the lite Claude analyser after Vertex
returned `429 RESOURCE_EXHAUSTED`. `_fallback_rate_check` fires only when the
fallback fraction STRICTLY exceeds its threshold, and the alarm did not fire.
(The alarm's own denominator is `len(candidate_analyses)+len(holding_analyses)`
and was NOT measured; `3/6` below is the TICKER ratio, and it is NOT claimed
that the alarm missed by exactly one ticker.) But every degradation field
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

from backend.services.autonomous_loop import (
    _degradation_summary_fields,
    _fallback_rate_check,
)
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
    """3 of 6 is EXACTLY 0.5 and must not fire.

    The SIX TICKERS are measured (3 full-pipeline Critic verdicts, 3 fallbacks).
    The alarm's real denominator was NOT measured, so this pins the PREDICATE at
    the boundary -- it does not assert that the live alarm saw exactly 3/6.
    """
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

def test_the_recorded_fields_are_produced_by_a_seam_that_can_be_EXECUTED():
    """BEHAVIOURAL, not source-order.

    The first version of this guard asserted only that `summary["fallback_rate"]`
    appeared BEFORE `if _fb_fire:` in the source. Mutation cell M1 disabled the
    recording entirely (`if _n_fb_total:` -> `if False:`), which does not move any
    text, and the guard SURVIVED. The logic was therefore extracted into
    `_degradation_summary_fields` so it can be driven for real.
    """
    # the measured incident: degraded, below threshold, must still be reported
    got = _degradation_summary_fields(False, 3, 6, {"HPE": "429"})
    assert got["fallback_rate"] == "3/6"
    assert got["fallback_alarm_fired"] is False
    assert got["fallback_reasons"] == {"HPE": "429"}

    # a paging cycle reports the same shape, with the flag flipped
    loud = _degradation_summary_fields(True, 4, 6, {"HPE": "429"})
    assert loud["fallback_rate"] == "4/6" and loud["fallback_alarm_fired"] is True

    # a HEALTHY cycle still reports -- absence of a rate must mean "no analyses",
    # never "nothing went wrong", or the two are indistinguishable downstream
    fine = _degradation_summary_fields(False, 0, 6, {})
    assert fine["fallback_rate"] == "0/6"
    assert fine["fallback_alarm_fired"] is False
    assert "fallback_reasons" not in fine

    # the ONLY empty case is a cycle that analysed nothing
    assert _degradation_summary_fields(False, 0, 0, {}) == {}


def test_the_seam_is_actually_wired_into_the_cycle():
    """A seam nothing calls is a seam that guards nothing."""
    src = inspect.getsource(al)
    assert "summary.update(_degradation_summary_fields(" in src, (
        "the extracted seam is no longer called from the cycle -- the fields "
        "would never reach the summary regardless of how well the seam behaves"
    )
    i_call = src.find("summary.update(_degradation_summary_fields(")
    i_fire = src.find("if _fb_fire:", src.find("_fb_fire, _n_fb, _n_fb_total"))
    assert i_call < i_fire, (
        "the recording call moved INSIDE/AFTER the paging branch again -- a "
        "sub-threshold degraded cycle would leave no durable trace"
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


# ---------------------------------------------------------------------------
# 4. THE SECOND SEAM -- summary -> record_cycle_end. Added in cycle 2 because
#    the cycle-1 Q/A deleted `degradation=_degradation,` from the call and the
#    whole suite stayed GREEN. Guarding one end of a two-ended wire is not
#    guarding the wire.
# ---------------------------------------------------------------------------

def test_the_degradation_record_carries_every_key_that_makes_a_cycle_legible():
    """BEHAVIOURAL: driven, not asserted from source."""
    from backend.services.autonomous_loop import (
        DEGRADATION_RECORD_KEYS,
        _degradation_record,
    )
    summary = {
        "fallback_rate": "3/6", "fallback_alarm_fired": False,
        "fallback_reasons": {"HPE": "429"}, "degraded": True,
        "degraded_analyses": "3/6", "meta_scorer_degraded": False,
        "unrelated_key": "must not be carried",
    }
    got = _degradation_record(summary)

    # A LITERAL EXPECTED SET, not the module constant. The first version asserted
    # `set(got) == set(DEGRADATION_RECORD_KEYS)` -- and `_degradation_record`
    # DERIVES `got` by iterating that same tuple, so both sides shrink together
    # and dropping a key passed. MEASURED by the cycle-1 Q/A and reproduced by me:
    # removing "fallback_reasons" from the tuple leaves all 9 tests green.
    # Four of the six keys could vanish from the persisted record in silence,
    # including the one carrying the 429 causes.
    #
    # The expected set is therefore written out HERE, where changing it is a
    # visible edit to a test rather than a side effect of editing production.
    EXPECTED = {
        "fallback_rate", "fallback_alarm_fired", "fallback_reasons",
        "degraded", "degraded_analyses", "meta_scorer_degraded",
    }
    assert set(DEGRADATION_RECORD_KEYS) == EXPECTED, (
        "the production key tuple no longer matches the set this step committed "
        f"to persisting. Production: {sorted(DEGRADATION_RECORD_KEYS)}. "
        f"Expected: {sorted(EXPECTED)}. Adding a key is fine but must be a "
        "deliberate edit here too; DROPPING one silently blinds every downstream "
        "reader to that fact."
    )
    assert set(got) == EXPECTED, (
        "the persisted degradation record dropped or gained a key -- a key "
        "silently missing here is invisible to every downstream reader"
    )
    assert "unrelated_key" not in got
    assert got["fallback_rate"] == "3/6"
    assert got["fallback_alarm_fired"] is False

    # absent facts are omitted, not persisted as None
    assert _degradation_record({}) == {}
    assert _degradation_record({"fallback_rate": "0/6"}) == {"fallback_rate": "0/6"}


def test_the_degradation_record_is_actually_passed_to_record_cycle_end():
    """SOURCE-LEVEL, and labelled as such.

    `run_daily_cycle` needs BigQuery, a broker and ~80 minutes, so the call
    itself cannot be executed here. What CAN be pinned is that the keyword is
    present and fed by the seam -- which is precisely the mutation
    (`degradation=_degradation,` deleted) that survived the entire suite in
    cycle 1.
    """
    src = inspect.getsource(al)
    import ast

    # AST, NOT A SUBSTRING SEARCH -- and that is the whole point of this test.
    # The first version asserted `"degradation=_degradation," in src`. Mutation
    # cell MX deleted the real argument and the test STILL PASSED, because the
    # docstring of `_degradation_record` QUOTES that literal while explaining
    # this very defect. **My own explanatory prose satisfied my own guard.**
    # A grep cannot tell a call site from a sentence about a call site; an AST
    # can, because comments and docstrings are not Call nodes.
    tree = ast.parse(inspect.getsource(al))
    calls = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "record_cycle_end"
    ]
    assert calls, "the record_cycle_end call vanished entirely"
    kwargs_with_degradation = [
        c for c in calls
        if any(k.arg == "degradation" for k in c.keywords)
    ]
    assert kwargs_with_degradation, (
        "record_cycle_end no longer receives a `degradation=` argument -- every "
        "cycle would persist `degradation: {}` and the sub-threshold degraded "
        "cycle would again leave no durable trace. (Checked via AST so that "
        "prose mentioning the argument cannot satisfy this assertion.)"
    )
    # and the value passed must be the seam's output, not some other name
    fed_by_seam = False
    for c in kwargs_with_degradation:
        for k in c.keywords:
            if k.arg == "degradation" and isinstance(k.value, ast.Name):
                fed_by_seam = k.value.id == "_degradation"
    assert fed_by_seam, "the degradation kwarg is no longer fed by `_degradation`"

    assert "_degradation = _degradation_record(summary)" in src, (
        "the degradation record is no longer built from the summary by the seam"
    )

    # AND THE ARGUMENTS, not just the call. The cycle-1 Q/A neutered the OTHER
    # seam's args (`_degradation_summary_fields(_fb_fire, 0, 0, _fb_reasons)`),
    # which makes it return {} on every cycle -- the exact defect restored -- and
    # the suite stayed green because the guard pinned the call substring and its
    # source ORDER, never the payload. Criterion 6 names "reverting the fix at
    # the call site" explicitly, and that IS the call site.
    calls_seam = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        and n.func.id == "_degradation_summary_fields"
    ]
    assert calls_seam, "the record-always seam is never called"
    args_ok = False
    for c in calls_seam:
        names = [a.id for a in c.args if isinstance(a, ast.Name)]
        if names == ["_fb_fire", "_n_fb", "_n_fb_total", "_fb_reasons"]:
            args_ok = True
    assert args_ok, (
        "the record-always seam is called with something other than the live "
        "(_fb_fire, _n_fb, _n_fb_total, _fb_reasons) -- passing constants makes "
        "it return {} for every cycle while this suite stays green"
    )

    # and the record must not be blanked AFTER the seam builds it: pin the VALUE
    # reaching record_cycle_end, not merely the name.
    assigns = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Assign)
        and any(isinstance(tg, ast.Name) and tg.id == "_degradation" for tg in n.targets)
    ]
    assert len(assigns) == 1, (
        f"_degradation is assigned {len(assigns)} times; a second assignment can "
        "blank the record after the seam built it, and the AST call-site guard "
        "pins only the NAME"
    )
    assert isinstance(assigns[0].value, ast.Call), (
        "_degradation is no longer assigned from a call -- a literal here (e.g. "
        "`_degradation = {}`) restores the defect with the name intact"
    )

    # THE BINDING IS NOT THE VALUE. The guard above pins how `_degradation` is
    # BOUND; it cannot see the dict being emptied in place afterwards. The
    # cycle-2 Q/A proved that with three survivors -- `_degradation.clear()`,
    # `.pop("fallback_reasons")`, `.update({k: None ...})` -- of which the first
    # FULLY RESTORES the pre-86.38 defect (`degradation={}` on every cycle)
    # behind a completely green suite. An in-place mutation is a Call on an
    # Attribute, not an Assign, so nothing above could ever fail on it.
    mutators = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        and isinstance(n.func.value, ast.Name) and n.func.value.id == "_degradation"
    ]
    assert not mutators, (
        "_degradation is mutated in place after it is built "
        f"({[m.func.attr for m in mutators]}) -- `.clear()` alone restores the "
        "exact defect this step removed, and the assignment guard above cannot "
        "see it because a method call is not an Assign"
    )

    # AND WHICH CALL SITE. The cycle-1 Q/A's fourth survivor was a DEAD DECOY
    # helper carrying a correct-looking `record_cycle_end(..., degradation=...)`
    # while the real kwarg was deleted -- the AST walk found *a* qualifying call
    # and passed. I claimed all four survivors were fixed and verified; that
    # claim did not reproduce for this one, because I named three remedies for
    # four findings. Pinning the COUNT closes it: a decoy adds a second call.
    assert len(calls) == 1, (
        f"expected exactly ONE record_cycle_end call site, found {len(calls)}. "
        "A second call -- including an unreachable decoy -- lets the real one "
        "lose its degradation argument while this guard still finds a match"
    )
