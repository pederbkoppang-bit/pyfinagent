"""phase-82.27 -- PBO belongs to whatever ran the SWEEP, and its provenance must survive the trip.

Step 82.23 wrote an immutable criterion requiring `generate_report` to emit a PBO.
That function receives ONE `BacktestResult`, and `compute_pbo` returns a false-good
0.0 at N<2 which PASSES the <=0.20 live ceiling -- so satisfying it literally would
have emitted an unconditional PASS on every report. 82.23 is superseded in place;
this step re-anchors on the sweep-level producer.

Bailey, Borwein, Lopez de Prado & Zhu, Algorithm 2.3: "each column n = 1,...,N
represents a vector of profits and losses over t = 1,...,T observations associated
with a particular model configuration tried by the researcher." The (T x N) matrix
is owned by whatever ran the sweep -- which in this repo is
`strategy_backtest_adapter.make_engine_backtest_fn`, never the single-run report.

Every guard here EXECUTES the code under test. `inspect.getsource` token scans were
rejected by the 82.23 Q/A as satisfiable by a comment, and are not used.
"""
from __future__ import annotations

import asyncio
import contextlib
from unittest import mock

import numpy as np
import pytest

from backend.autoresearch.gate import PromotionGate
from backend.autoresearch.strategy_candidate_producer import build_per_strategy_candidates


# ── stubs ────────────────────────────────────────────────────────────

class _Window:
    def __getattr__(self, n):
        return 0.0

    def __init__(self):
        self.window_id = 1
        self.predictions = []
        self.mda_scores = {}
        self.train_start = self.train_end = self.test_start = self.test_end = "2020-01-01"


class _Result:
    """Minimal BacktestResult-shaped stub.

    The fallback is deliberately TYPE-AWARE rather than a blanket 0.0: a new
    dict-consuming field must get a dict, so the stub cannot silently satisfy a
    field it does not really model (guard-vacuity shape: a fixture that cannot
    represent the failure).
    """

    def __getattr__(self, n):
        return {} if ("importance" in n or n.endswith("_scores") or n.endswith("_params")) else 0.0

    def __init__(self, seed: int, n_obs: int = 400):
        rng = np.random.default_rng(seed)
        navs = 100_000 * np.cumprod(1 + rng.normal(0.0005, 0.01, n_obs))
        self.nav_history = [{"date": f"d{i}", "nav": float(v)} for i, v in enumerate(navs)]
        self.windows = [_Window()]
        self.all_trades = []
        self.aggregate_sharpe = 0.9
        self.initial_capital = 100_000.0
        self.final_nav = float(navs[-1])
        self.num_windows = 1


def _run_sweep(n_obs: int = 400, k: int = 12, fail_first: int = 0) -> dict:
    """EXECUTE the real sweep-level producer against a stub engine."""
    from backend.autoresearch import strategy_backtest_adapter as ad

    counter = {"i": 0}

    def factory(variant):
        counter["i"] += 1
        seed = counter["i"]
        boom = seed <= fail_first

        class _Engine:
            def run_backtest(self, skip_cache_clear=False):
                if boom:
                    raise RuntimeError("variant blew up")
                return _Result(seed, n_obs)

        return _Engine()

    fn = ad.make_engine_backtest_fn(
        engine_factory=factory, num_param_variants=k, clear_cache_fn=lambda: None
    )
    return fn({"strategy": "triple_barrier"})


# ── criterion 1: the sweep producer emits PBO with N and diversity ───

def test_sweep_producer_emits_pbo_with_its_trial_count_and_diversity():
    """Executed, not grepped. A bare PBO float cannot be told apart from a
    gate-grade one, so N and the column correlation must travel WITH it."""
    out = _run_sweep()
    assert out.get("pbo") is not None, f"sweep producer emitted no pbo: {sorted(out)}"
    assert 0.0 <= out["pbo"] <= 1.0
    for key in ("pbo_n_trials", "pbo_n_obs", "pbo_column_corr_mean", "pbo_columns_diverse"):
        assert key in out, f"pbo emitted WITHOUT {key}: {sorted(out)}"
    assert out["pbo_n_trials"] == 12
    assert isinstance(out["pbo_column_corr_mean"], float)


def test_sweep_producer_refuses_a_strategy_it_does_not_know():
    """Bailey Algo 2.3 requires columns to be configs of the SAME model. A silent
    fallback to triple_barrier would make the columns a different model than the
    label claims, so the producer must raise."""
    from backend.autoresearch import strategy_backtest_adapter as ad

    fn = ad.make_engine_backtest_fn(
        engine_factory=lambda v: None, num_param_variants=4, clear_cache_fn=lambda: None
    )
    with pytest.raises(ValueError, match="unknown strategy"):
        fn({"strategy": "not_a_real_strategy"})


# ── criterion 2: the single-run path emits NO pbo ────────────────────

def test_single_run_report_emits_no_pbo_anywhere():
    """BEHAVIOURAL, not a signature check: run the REAL generate_report and search
    the whole returned structure.

    Reason this test exists, stated so a future reader does not "fix" it by adding
    PBO here: generate_report receives ONE BacktestResult, so N would always be 1,
    and compute_pbo returns a false-good 0.0 at N<2 -- which passes the <=0.20 live
    ceiling. Emitting PBO here would mean an unconditional PASS on every run.
    """
    from backend.backtest.analytics import generate_report

    report = generate_report(_Result(seed=1), num_trials=4, baselines=None)
    assert "analytics" in report
    assert "deflated_sharpe" in report["analytics"], "sanity: the report did produce metrics"
    assert "pbo" not in str(report).lower(), (
        "generate_report now emits a PBO, but it sees a single run: N=1 makes "
        "compute_pbo return a false-good 0.0 that passes every ceiling"
    )


def test_no_generate_report_invocation_receives_more_than_one_result():
    """A PROPERTY, not a count.

    The 82.27 research gate corrected my "16 call sites" to 15 invocations (the
    16th reference is a monkeypatch.setattr -- a substitution, not a call). A
    hardcoded number drifts the moment a script is added, so assert the structural
    property instead: the function takes a single `result`, with no plural
    collection parameter through which N series could ever arrive.
    """
    import inspect

    from backend.backtest import analytics

    params = list(inspect.signature(analytics.generate_report).parameters)
    assert params[0] == "result", f"first parameter is {params[:1]}, expected a single result"
    assert not any(p in ("results", "pnl_matrix", "matrix", "runs") for p in params), (
        f"generate_report now accepts {params}; if it can see N configurations the "
        "82.27 anchor decision must be revisited"
    )


# ── criterion 3: an under-length matrix refuses, never 0.0 ───────────

def test_under_length_matrix_refuses_rather_than_emitting_zero():
    """T below S*2. The producer must emit NO pbo so the consumer SKIPS, rather
    than a 0.0 that would sail through the ceiling."""
    out = _run_sweep(n_obs=20)          # 19 daily returns, far below S*2 = 32
    assert out.get("pbo") is None, f"expected refusal, got pbo={out.get('pbo')!r}"
    assert "pbo" not in out or out["pbo"] is None


def test_the_producer_itself_reports_why_it_refused():
    """Criterion 3's second clause, on THE PRODUCER.

    Q/A cycle-1 [BLOCK]: the reason previously existed only as a `_log.warning`,
    which no consumer and no test can see -- the returned mapping was exactly
    ['dsr','n_variants','n_windows','pbo_dropped_columns','sharpe'], with nothing
    matching 'reas|refus'. The only reason-asserting test targeted
    `risk_server.pbo_check`, which the contract itself classifies as a CONSUMER,
    and drove it with T=40/N=1 -- the N<2 branch, not the under-length branch the
    criterion names. So the clause was unproven on the object it is about.

    A refusal that cannot be distinguished from "not computed yet" is not a
    refusal.
    """
    out = _run_sweep(n_obs=20)
    assert out.get("pbo") is None
    reason = out.get("pbo_refused")
    assert reason, f"the producer refused but reported no reason: {sorted(out)}"
    assert isinstance(reason, str) and len(reason) > 10
    # The reason must name the ACTUAL cause, not be a generic placeholder.
    assert "undersized" in reason or "rows" in reason, reason


def test_a_successful_sweep_reports_no_refusal_reason():
    """Discriminating. A constant string would satisfy the test above."""
    out = _run_sweep(n_obs=400)
    assert out.get("pbo") is not None
    assert out.get("pbo_refused") is None, (
        f"a successful sweep is reporting a refusal reason: {out.get('pbo_refused')!r}"
    )


def test_refusal_reason_is_reported_on_the_mcp_surface():
    """The other consumer of compute_pbo. It must refuse AND say why -- a silent
    refusal is indistinguishable from 'within bounds' to a caller."""
    from backend.agents.mcp_servers.risk_server import create_risk_server

    tool = asyncio.run(create_risk_server().get_tool("pbo_check"))
    r = tool.fn(pnl_matrix=[[0.1]] * 40, S=16)      # N=1
    assert r["ok"] is False and r["pbo"] is None
    assert "refused" in r["reason"], r
    assert r["isError"] is True, "a refusal must not read as a clean pass"


def test_mcp_surface_still_vetoes_a_genuinely_bad_pbo():
    """Mirror guard: refusing everything would also satisfy the test above."""
    from backend.agents.mcp_servers.risk_server import create_risk_server

    tool = asyncio.run(create_risk_server().get_tool("pbo_check"))
    mat = np.random.default_rng(7).normal(size=(600, 12)).tolist()
    r = tool.fn(pnl_matrix=mat, S=16, threshold=0.5)
    assert r["ok"] is True and r["pbo"] is not None
    assert r["n_trials"] == 12 and r["gate_grade"] is True
    assert "columns_diverse" in r


# ── the wiring defect: provenance must survive the hop ───────────────

def _candidate(**extra) -> dict:
    metrics = {"dsr": 0.99, "pbo": 0.05, "sharpe": 1.2, **extra}
    cfgs = [{"id": "triple_barrier", "params": {"strategy": "triple_barrier"}}]
    return build_per_strategy_candidates(cfgs, lambda p: metrics)[0]


def test_pbo_provenance_survives_the_candidate_producer_hop():
    """THE DEFECT THIS STEP FIXES. build_per_strategy_candidates rebuilt each
    candidate from a hardcoded 5-key whitelist, so `pbo` reached the gate but the
    fields describing how it was computed did not."""
    c = _candidate(pbo_n_trials=3, pbo_n_obs=900, pbo_gate_grade=False,
                   pbo_column_corr_mean=0.02, pbo_columns_diverse=True,
                   pbo_dropped_columns=2)
    for key in ("pbo_n_trials", "pbo_n_obs", "pbo_gate_grade",
                "pbo_column_corr_mean", "pbo_columns_diverse", "pbo_dropped_columns"):
        assert key in c, f"{key} was dropped in transit to the gate: {sorted(c)}"


def test_the_trials_floor_actually_fires_on_the_producer_path():
    """End to end. Before this step the same candidate was PROMOTED, because the
    floor's input never arrived -- a gate term that cannot fail."""
    c = _candidate(pbo_n_trials=3)
    c["trial_id"] = "t"
    v = PromotionGate().evaluate(c)
    assert v["promoted"] is False, v
    assert "pbo_trials_below_min" in v["reason"], v

    stripped = {k: val for k, val in c.items() if k != "pbo_n_trials"}
    assert PromotionGate().evaluate(stripped)["promoted"] is True, (
        "sanity: without the key the gate promotes -- that WAS the live behaviour, "
        "and it is what this step repairs"
    )


def test_a_producer_that_reports_no_trial_count_keeps_legacy_behaviour():
    """Purely additive. A producer that never emitted these keys must behave
    exactly as before, or this step breaks every existing caller."""
    c = _candidate()
    c["trial_id"] = "t"
    assert "pbo_n_trials" not in c
    assert PromotionGate().evaluate(c)["promoted"] is True


def test_a_well_sized_sweep_still_promotes():
    """Mirror guard against a fix that simply refuses everything."""
    c = _candidate(pbo_n_trials=32, pbo_columns_diverse=True)
    c["trial_id"] = "t"
    assert PromotionGate().evaluate(c)["promoted"] is True


# ── the file drawer ──────────────────────────────────────────────────

def test_dropped_columns_are_reported():
    """Bailey et al.: "Hiding trials will lead to an underestimation of the
    overfit." Failed variants are disproportionately the bad configurations, so
    silently dropping them biases PBO DOWN -- toward promoting. Reporting the
    count does not correct the bias; it makes it visible."""
    out = _run_sweep(k=12, fail_first=3)
    assert out.get("pbo_dropped_columns") == 3, out.get("pbo_dropped_columns")
    assert out["pbo_n_trials"] == 9, "the surviving columns are what CSCV actually saw"


def test_dropped_column_count_is_zero_when_nothing_failed():
    """Discriminating: a constant would satisfy the test above."""
    assert _run_sweep(k=12, fail_first=0).get("pbo_dropped_columns") == 0


def test_dropped_count_is_reported_even_when_pbo_is_refused():
    """The refusal path is exactly where a reader most needs to know how many
    configurations went missing."""
    out = _run_sweep(n_obs=20, k=12, fail_first=2)
    assert out.get("pbo") is None
    assert out.get("pbo_dropped_columns") == 2, out


# ── the composite gate must not swallow a refusal ────────────────────

@contextlib.contextmanager
def _kill_switch_not_paused():
    """The composite gate short-circuits on a paused kill switch, and this
    machine's real switch IS paused -- so without this the PBO gate below is
    never reached and both tests would pass/fail for the wrong reason.

    Patches the READ path only. Never flips real state: a test that mutates the
    live kill switch would be a far worse defect than the one being tested.
    """
    import backend.services.kill_switch as ks

    class _Snap:
        @staticmethod
        def snapshot():
            return {"paused": False, "pause_reason": None}

    with mock.patch.object(ks, "get_state", lambda: _Snap()):
        yield


def _evaluate(candidate: dict) -> dict:
    from backend.agents.mcp_servers.risk_server import create_risk_server

    fn = asyncio.run(create_risk_server().get_tool("evaluate_candidate")).fn
    with _kill_switch_not_paused():
        return fn(candidate=candidate)


def test_composite_gate_does_not_read_a_refusal_as_a_clean_pass():
    """Q/A cycle-1 [BLOCK] -- the finding I asked for and got.

    `evaluate_candidate` read ONLY `pbo_result['vetoed']`, ignoring `ok` and
    `isError`. So an undersized matrix -- which `pbo_check` correctly refuses --
    fell through the composite gate as `passed_all_gates`. That made the
    fail-closed property this step claims for `pbo_check` FALSE at the only
    in-repo consumer, thirty lines below its own definition.

    Not a regression: the pre-change raw `compute_pbo` returned a false-good 0.0,
    which also produced `vetoed=False`. But "no worse than before" is not the
    property being advertised.
    """
    out = _evaluate({
        "pnl_matrix": [[0.01, 0.02, 0.03, 0.04]] * 10,   # T=10 << S*2=32
        "sigma_ann_pct": 12.0,
        "sharpe": 1.17,
    })
    assert out["gates"]["pbo"]["ok"] is False, "sanity: pbo_check should have refused"
    assert out["vetoed"] is True, (
        f"the composite gate passed a candidate whose overfitting term could not "
        f"be evaluated: reason={out.get('reason')!r}"
    )
    assert "pbo_unevaluable" in str(out["reason"]), out["reason"]


def test_composite_gate_still_passes_a_genuinely_good_candidate():
    """Mirror guard: vetoing everything would also satisfy the test above."""
    out = _evaluate({"pbo": 0.05, "sigma_ann_pct": 6.0, "sharpe": 2.0})
    assert out["vetoed"] is False, out


def test_composite_gate_still_vetoes_a_pbo_over_the_threshold():
    """And the ordinary veto path must survive the change."""
    out = _evaluate({"pbo": 0.86, "sigma_ann_pct": 6.0, "sharpe": 2.0})
    assert out["vetoed"] is True and out["reason"] == "pbo_exceeds_threshold", out
