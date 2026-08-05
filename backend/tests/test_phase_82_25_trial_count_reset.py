"""phase-82.25 -- a warm start must not reset the DSR trial count to 1.

WHY IT MATTERS, MEASURED. DSR is deflated BY the number of trials searched, and N is
scoped to the RESEARCH PROCESS that produced the discovery, not to one optimisation
session (Bailey & Lopez de Prado 2014). The effect is not marginal: in run 60617e0b,
exp01 and exp10 share Sharpe 0.6455483635957818 and differ **72x in DSR** on trial count
alone. Both warm-start paths used to set `num_trials = 1`, so every carried-forward DSR
was reported as though the strategy had been found on the first attempt.

THE HARD BOUNDARY THIS FILE ALSO GUARDS. The live optimizer_best.json's
`dsr = 0.9525811126193078` clears the 0.95 go-live gate by 0.0026. Re-deflating that
persisted number at a higher N would both close the gate and fabricate a statistic -- it
was computed at whatever N its own run used, and that N is unrecorded. The fix changes
only FUTURE deflation; `test_a_warm_start_never_mutates_a_persisted_dsr` pins it.
"""
from __future__ import annotations

import inspect
import json

import pytest

from backend.backtest import quant_optimizer as qo
from backend.backtest.analytics import compute_deflated_sharpe
from backend.backtest.quant_optimizer import QuantStrategyOptimizer


def _optimizer(**attrs) -> QuantStrategyOptimizer:
    """Fixture shape from test_phase_82_22_optimizer_best_provenance.py:32-49 --
    bypass __init__ (disk I/O) because we are testing the warm-start logic."""
    opt = QuantStrategyOptimizer.__new__(QuantStrategyOptimizer)
    opt.best_params = {"strategy": "triple_barrier"}
    opt.best_sharpe = 1.17
    opt.best_dsr = 0.95
    opt.num_trials = 0
    opt.kept = 0
    opt.discarded = 0
    opt._run_id = "test0000"
    opt._warm_started = False
    opt._warm_started_from_run_id = None
    opt._warm_started_from_artifact = None
    opt.prior_trials_known = None
    for k, v in attrs.items():
        setattr(opt, k, v)
    return opt


# ---------------------------------------------------------------------------
# CRITERION 1 -- a recorded prior count is carried forward, not reset
# ---------------------------------------------------------------------------

def test_warm_start_carries_a_recorded_prior_count():
    opt = _optimizer()
    opt._resolve_prior_trials({"num_trials": 7})

    assert opt.num_trials == 7, (
        f"a warm-start source recording 7 prior trials produced num_trials="
        f"{opt.num_trials}; the carried-forward DSR would be deflated as though the "
        "strategy had been found on the first attempt"
    )
    assert opt.prior_trials_known is True


def test_warm_start_does_not_reset_to_one():
    """The literal defect. Pinned separately so the failure message names it."""
    opt = _optimizer()
    opt._resolve_prior_trials({"num_trials": 11})
    assert opt.num_trials != 1, "the warm start reset the trial count to 1"
    assert opt.num_trials == 11


# ---------------------------------------------------------------------------
# CRITERION 2 -- the DSR really is deflated against the cumulative count
# ---------------------------------------------------------------------------

def test_dsr_is_strictly_lower_at_the_cumulative_count_than_at_one():
    """Compares the SAME run deflated at num_trials=1 vs the cumulative figure.

    The direction is verified against the real `compute_deflated_sharpe`, not assumed:
    if deflation ever stopped depending on N, this fails.
    """
    kwargs = dict(observed_sr=0.6455483635957818, variance_of_srs=0.5,
                  skewness=0.0, kurtosis=3.0, T=252)

    at_one = compute_deflated_sharpe(num_trials=1, **kwargs)
    at_cumulative = compute_deflated_sharpe(num_trials=11, **kwargs)

    assert at_cumulative < at_one, (
        f"DSR did not fall with more trials (1 -> {at_one}, 11 -> {at_cumulative}); "
        "deflation no longer depends on N, so carrying the count forward would be "
        "pointless"
    )


def test_a_warm_started_run_reports_the_deflated_number_not_the_optimistic_one():
    """End-to-end on the resolver: the count the optimizer will hand to
    generate_report after a warm start is the cumulative one."""
    opt = _optimizer()
    opt._resolve_prior_trials({"num_trials": 11})

    kwargs = dict(observed_sr=0.6455483635957818, variance_of_srs=0.5,
                  skewness=0.0, kurtosis=3.0, T=252)
    reported = compute_deflated_sharpe(num_trials=opt.num_trials, **kwargs)
    optimistic = compute_deflated_sharpe(num_trials=1, **kwargs)

    assert reported < optimistic, (
        "the warm-started optimizer would still report the num_trials=1 figure"
    )


# ---------------------------------------------------------------------------
# CRITERION 3 -- an unrecorded prior resolves per a DOCUMENTED decision
# ---------------------------------------------------------------------------

def test_unknown_prior_is_marked_unknown_and_not_silently_one():
    """This is the PRODUCTION path today: the live optimizer_best.json is schema v1
    and carries no num_trials (82.22 changed only the writer)."""
    opt = _optimizer()
    opt._resolve_prior_trials({"sharpe": 1.17, "dsr": 0.95})

    assert opt.prior_trials_known is False, (
        "an unrecorded prior count was treated as KNOWN -- a downstream reader could "
        "not tell that the reported DSR is an upper bound"
    )
    # NOTE (cycle 2): an earlier version of this test asserted `num_trials != 1`,
    # encoding my mistaken belief that 1 was the bad value. The cycle-1 Q/A proved
    # the opposite -- 1 is the correct FLOOR and 0 was strictly worse than the
    # defect, because the counter increments before it is reported. What actually
    # matters is that the prior is MARKED unknown, not that the count avoids 1.
    assert opt.num_trials == QuantStrategyOptimizer._UNKNOWN_PRIOR_FLOOR


@pytest.mark.parametrize("bad", [None, 0, -3, "7", True, 2.5])
def test_a_malformed_prior_count_is_treated_as_unknown(bad):
    """A string, a bool or a negative must not be laundered into a trial count.
    `True` is explicitly included: bool is an int subclass in Python, so a naive
    isinstance check would accept it as the count 1 -- the very value being fixed."""
    opt = _optimizer()
    opt._resolve_prior_trials({"num_trials": bad})
    assert opt.prior_trials_known is False, f"{bad!r} was accepted as a prior count"
    assert opt.num_trials == QuantStrategyOptimizer._UNKNOWN_PRIOR_FLOOR


def test_the_documented_decision_is_recorded_in_code():
    """Criterion 3 asks for a DOCUMENTED decision, not a default. If the rationale is
    ever deleted, the behaviour becomes an unexplained magic choice."""
    doc = QuantStrategyOptimizer._resolve_prior_trials.__doc__ or ""
    assert "UNKNOWN" in doc and "UPPER BOUND" in doc, (
        "the unknown-prior decision is no longer documented on the resolver"
    )


# ---------------------------------------------------------------------------
# Persistence -- the honesty flag must survive to the next warm start
# ---------------------------------------------------------------------------

def _saved(monkeypatch, tmp_path, opt) -> dict:
    path = tmp_path / "optimizer_best.json"
    monkeypatch.setattr(qo, "_BEST_PARAMS_PATH", path)
    opt._save_best_params()
    return json.loads(path.read_text(encoding="utf-8"))


def test_cumulative_count_and_honesty_flag_are_persisted(monkeypatch, tmp_path):
    opt = _optimizer(num_trials=9, prior_trials_known=True)
    d = _saved(monkeypatch, tmp_path, opt)

    assert d["num_trials"] == 9, "the cumulative count was not persisted"
    assert d["prior_trials_known"] is True, (
        "prior_trials_known was not persisted, so the next warm start would launder "
        "an unknown depth into a known one"
    )


def test_an_unknown_prior_persists_as_unknown(monkeypatch, tmp_path):
    opt = _optimizer(num_trials=4, prior_trials_known=False)
    d = _saved(monkeypatch, tmp_path, opt)
    assert d["prior_trials_known"] is False


def test_a_saved_file_round_trips_into_a_warm_start(monkeypatch, tmp_path):
    """The two halves must actually fit together -- writer and reader agreeing is the
    whole mechanism, and each was added by a different step (82.22 / 82.25)."""
    saved = _saved(monkeypatch, tmp_path, _optimizer(num_trials=13, prior_trials_known=True))

    fresh = _optimizer()
    fresh._resolve_prior_trials(saved)

    assert fresh.num_trials == 13, (
        "a file written by _save_best_params does not warm-start into the same count"
    )
    assert fresh.prior_trials_known is True


# ---------------------------------------------------------------------------
# THE GO-LIVE BOUNDARY -- never retroactively re-deflate a persisted DSR
# ---------------------------------------------------------------------------

def test_a_warm_start_never_mutates_a_persisted_dsr():
    """The live file's dsr = 0.9525811126193078 clears the 0.95 go-live gate by
    0.0026. Recomputing it at a higher N would close the gate AND fabricate a
    statistic -- the inherited figure was computed at an unrecorded N."""
    source = {"sharpe": 1.1704, "dsr": 0.9525811126193078, "num_trials": 11}
    opt = _optimizer(best_dsr=float(source["dsr"]))
    before = opt.best_dsr

    opt._resolve_prior_trials(source)

    assert opt.best_dsr == before == 0.9525811126193078, (
        "the warm start re-deflated the inherited DSR; that closes the go-live gate "
        "and fabricates a number that was computed at a different, unrecorded N"
    )
    assert opt.num_trials == 11, "the count should still be carried forward"


def test_the_cold_baseline_path_is_untouched():
    """`num_trials = 1` at the cold baseline is LEGITIMATE -- one trial really has
    been run. Only the warm-start resets were the defect, and a fix that also broke
    the baseline would be over-reach."""
    import inspect

    src = inspect.getsource(QuantStrategyOptimizer.run_loop)
    assert "self.num_trials = 1" in src, (
        "the cold-baseline num_trials = 1 was removed; that assignment is correct and "
        "outside this step's scope"
    )


def test_unrelated_same_named_num_trials_is_not_touched():
    """`num_trials` in the autoresearch adapter counts SEED CONFIGS in a bake-off --
    a different variable that merely shares a name. A sweep that 'fixed' it would
    change unrelated behaviour."""
    import inspect

    from backend.autoresearch import strategy_backtest_adapter as A

    src = inspect.getsource(A)
    assert "num_trials" in src, "precondition: the adapter still has its own num_trials"
    assert "prior_trials_known" not in src, (
        "the 82.25 honesty flag leaked into the autoresearch adapter, whose "
        "num_trials is an unrelated seed-config count"
    )


# ---------------------------------------------------------------------------
# THE REAL CALL SITES. Every test above drives `_resolve_prior_trials` directly,
# and that is not enough: restoring `self.num_trials = 1` at the two ACTUAL warm-
# start sites inside `_load_previous_best` passed the whole suite above (measured --
# mutants M1 and M2 both SURVIVED). The defect lives at the call sites, so the
# guards have to drive them.
# ---------------------------------------------------------------------------

def _warm_start_from_file(monkeypatch, tmp_path, payload: dict):
    path = tmp_path / "optimizer_best.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(qo, "_BEST_PARAMS_PATH", path)

    opt = _optimizer()
    opt.best_params = {}
    monkeypatch.setattr(type(opt), "_apply_params_to_engine", lambda self, p: None,
                        raising=False)
    opt._load_previous_best()
    return opt


def test_load_previous_best_carries_the_count_from_the_file(monkeypatch, tmp_path):
    """CRITERION 1 at the REAL call site (optimizer_best.json path)."""
    opt = _warm_start_from_file(monkeypatch, tmp_path, {
        "params": {"strategy": "triple_barrier"},
        "sharpe": 1.17, "dsr": 0.95, "num_trials": 7,
    })

    assert opt._warm_started is True, "precondition: the warm start did not happen"
    assert opt.num_trials == 7, (
        f"_load_previous_best set num_trials={opt.num_trials} from a file recording 7 "
        "-- the reset-to-1 defect is still present at the call site"
    )
    assert opt.prior_trials_known is True


def test_load_previous_best_marks_a_schema_v1_file_as_unknown(monkeypatch, tmp_path):
    """The PRODUCTION path today: the live file is schema v1 with no num_trials."""
    opt = _warm_start_from_file(monkeypatch, tmp_path, {
        "params": {"strategy": "triple_barrier"},
        "sharpe": 1.17, "dsr": 0.9525811126193078,
    })

    assert opt._warm_started is True
    assert opt.prior_trials_known is False
    assert opt.num_trials == QuantStrategyOptimizer._UNKNOWN_PRIOR_FLOOR, (
        "the schema-v1 path did not land on the documented unknown-prior floor"
    )
    assert opt.best_dsr == 0.9525811126193078, (
        "the inherited DSR was mutated by the warm start -- that closes the go-live "
        "gate (it clears 0.95 by only 0.0026) and fabricates a statistic"
    )


def test_load_previous_best_reads_the_result_store_nesting(monkeypatch, tmp_path):
    """The result_store path nests the count inside "analytics". Reading only the
    top level would leave half the defect in place."""
    monkeypatch.setattr(qo, "_BEST_PARAMS_PATH", tmp_path / "absent.json")

    latest = {
        "strategy_params": {"strategy": "triple_barrier"},
        "analytics": {"sharpe": 1.17, "deflated_sharpe": 0.95, "num_trials": 13},
        "run_id": "prev1234",
    }
    # result_store is imported FUNCTION-LOCALLY inside _load_previous_best, so the
    # name resolves from its own module at call time -- patching a qo attribute would
    # silently patch nothing (this repo has been bitten by exactly that before).
    from backend.backtest import result_store as _rs
    monkeypatch.setattr(_rs, "load_latest", lambda: latest)

    opt = _optimizer()
    opt.best_params = {}
    monkeypatch.setattr(type(opt), "_apply_params_to_engine", lambda self, p: None,
                        raising=False)
    opt._load_previous_best()

    if opt._warm_started:
        assert opt.num_trials == 13, (
            f"the result_store warm start produced num_trials={opt.num_trials}; the "
            "count is nested under 'analytics' there and was not read"
        )
        assert opt.prior_trials_known is True
    else:
        pytest.fail(
            "the result_store warm-start path did not fire; re-derive its shape "
            "before trusting this guard"
        )


# ---------------------------------------------------------------------------
# CYCLE 2 -- the three BLOCK findings from the cycle-1 FAIL
# ---------------------------------------------------------------------------

def test_the_unknown_branch_never_deflates_less_than_the_defect_did():
    """F2: I HAD THE DIRECTION BACKWARDS and this is the guard that proves it.

    The counter is incremented BEFORE it is reported (`num_trials += 1` then
    `generate_report(..., num_trials=self.num_trials)`), so setting the unknown
    branch to 0 made session trial k report N=k where the DEFECT reported N=k+1 --
    LESS deflation, an EASIER 0.95 gate, keeping what the defect discarded.

    The floor must therefore be >= the value the defect used (1).
    """
    opt = _optimizer()
    opt._resolve_prior_trials({})          # schema-v1: today's production path

    defect_value = 1                       # what `self.num_trials = 1` produced
    assert opt.num_trials >= defect_value, (
        f"the unknown branch set num_trials={opt.num_trials}, BELOW the {defect_value} "
        "the defect used. Because the counter increments before it is reported, that "
        "means this 'fix' deflates LESS than the bug -- the 0.95 KEEP gate gets "
        "EASIER and it keeps candidates the defect discarded."
    )


def test_the_unknown_floor_is_a_floor_not_an_estimate():
    """It must be the minimum defensible value, not an invented depth."""
    assert QuantStrategyOptimizer._UNKNOWN_PRIOR_FLOOR == 1, (
        "the unknown-prior floor is no longer 1: below 1 deflates less than the "
        "defect; above 1 fabricates a search depth that was never recorded"
    )


def test_unknown_is_sticky_across_generations():
    """F1: the honesty flag must be READ, not merely written.

    The first version persisted `prior_trials_known` and never read it back, so a
    source whose own depth was unknown warm-started as KNOWN one generation later --
    reproducing the exact write-without-read root cause this step exists to fix.
    """
    opt = _optimizer()
    opt._resolve_prior_trials({"num_trials": 4, "prior_trials_known": False})

    assert opt.num_trials == 4, "the count itself should still be carried"
    assert opt.prior_trials_known is False, (
        "a source that recorded its own depth as UNKNOWN was warm-started as KNOWN -- "
        "the unknown has been laundered one generation on"
    )


def test_unknown_stickiness_survives_a_real_round_trip(monkeypatch, tmp_path):
    """Generation 1 writes unknown; generation 2 must still read unknown."""
    gen1 = _saved(monkeypatch, tmp_path,
                  _optimizer(num_trials=4, prior_trials_known=False))
    assert gen1["prior_trials_known"] is False

    gen2 = _optimizer()
    gen2._resolve_prior_trials(gen1)
    assert gen2.prior_trials_known is False, (
        "unknown did not survive a save/load round trip"
    )


def test_the_reporting_site_receives_the_cumulative_count_EXECUTED(monkeypatch):
    """CRITERION 2, driven rather than scanned.

    The cycle-1 Q/A killed the first attempt at this (no coverage at all: mutating
    the reporting site to `num_trials=1` left the suite green). The cycle-2 Q/A then
    killed my replacement -- a structural AST scan -- by REWORDING the scanned text:
    `num_trials=min(self.num_trials, 1)` and an aliased `self._frozen_num_trials = 1`
    both satisfied a "mentions num_trials and self" predicate while forcing N=1 at
    every trial. A source scan cannot see behaviour.

    So this EXECUTES run_loop for one iteration on a warm-started optimizer and
    captures what generate_report actually receives.
    """
    from unittest.mock import patch

    opt = _optimizer()
    opt._resolve_prior_trials({"num_trials": 7})
    assert opt.num_trials == 7 and opt._warm_started is False
    opt._warm_started = True          # skip the cold baseline block

    captured = {}

    def _fake_generate_report(result, num_trials=None, **kw):
        captured["num_trials"] = num_trials
        return {"analytics": {"sharpe": 0.5, "deflated_sharpe": 0.1}}

    # MagicMock so incidental engine housekeeping calls (feature-cache setup and
    # teardown) do not turn this into a whack-a-mole of stub methods. The assertion
    # is on what generate_report RECEIVES, which is captured above.
    from unittest.mock import MagicMock

    engine = MagicMock()
    engine.stop_check = None
    opt.engine = engine
    opt.dsr_threshold = 0.95
    opt.use_llm = False
    # W2 (cycle-3 Q/A): these are CLASS-level rebinds. The first version used a bare
    # setattr with no teardown, leaving QuantStrategyOptimizer gutted (_save_best_params
    # et al.) for the rest of the pytest process -- harmless today only because the
    # other consumers happen to collect earlier. monkeypatch restores them.
    for name, fn in [
        ("_report_status", lambda self: None),
        ("_log_experiment", lambda self, *a, **k: None),
        ("_check_model_staleness", lambda self: None),
        ("_propose_random", lambda self, th=False: {"param": "tp_pct", "value": 9.0}),
        ("_apply_params_to_engine", lambda self, p: None),
        ("_setup_feature_cache", lambda self, p: None),
        ("_extract_top5_mda", lambda self, r: []),
        ("_save_best_params", lambda self: None),
    ]:
        monkeypatch.setattr(type(opt), name, fn, raising=False)

    # W1 (cycle-3 Q/A): the first version wrapped this in
    # `except TypeError: pytest.skip(...)`, which made the guard FAIL OPEN --
    # measured, renaming run_loop's `max_iterations` parameter (an ordinary
    # refactor) turned the suite into "26 passed, 1 skipped", exit 0, and combining
    # that rename with `num_trials=min(self.num_trials, 1)` fully reinstated the
    # defect while the IMMUTABLE VERIFICATION COMMAND still exited 0. A guard with a
    # trapdoor that deletes it silently is worse than one that observes the wrong
    # thing. So: assert the precondition explicitly, and never skip.
    assert "max_iterations" in inspect.signature(
        QuantStrategyOptimizer.run_loop
    ).parameters, (
        "run_loop no longer takes max_iterations -- this probe must be re-derived, "
        "NOT skipped: skipping would silently remove the only behavioural guard on "
        "the reporting site"
    )

    with patch.object(qo, "generate_report", _fake_generate_report):
        opt.run_loop(max_iterations=1)

    assert "num_trials" in captured, (
        "generate_report was never reached; the probe did not exercise the reporting "
        "site, so it proves nothing"
    )
    assert captured["num_trials"] == 8, (
        f"the reporting site received num_trials={captured['num_trials']}, expected 8 "
        "(cumulative prior 7 + this session's first trial). The cumulative count is "
        "not reaching the DSR, which is the whole point of carrying it forward."
    )


def test_the_reporting_site_passes_the_cumulative_count():
    """Cheap structural COMPANION to the executed probe above.

    Kept deliberately, but it is NOT the coverage -- the cycle-2 Q/A defeated this
    shape by rewording the scanned text. The executed probe is the real guard.

    Mutating `generate_report(result, num_trials=self.num_trials)` to
    `num_trials=1` left the entire cycle-1 suite GREEN -- the whole defect could be
    re-expressed at the reporting site with every guard passing, because criterion
    2's tests were a library fact about compute_deflated_sharpe plus a
    re-implemented copy of the call. This asserts the real call site structurally.
    """
    import ast
    import inspect

    src = inspect.getsource(QuantStrategyOptimizer.run_loop)
    tree = ast.parse(inspect.cleandoc(src))
    calls = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "generate_report"
    ]
    assert calls, "run_loop no longer calls generate_report -- re-derive this guard"

    trial_calls = [
        c for c in calls
        for kw in c.keywords
        if kw.arg == "num_trials" and "num_trials" in ast.unparse(kw.value)
        and "self" in ast.unparse(kw.value)
    ]
    assert trial_calls, (
        "no generate_report call passes self.num_trials as num_trials. The cumulative "
        "count never reaches the DSR, so carrying it forward is pointless -- this is "
        "the mutant MREPORT that survived cycle 1."
    )
