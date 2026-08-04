"""phase-82.22 -- optimizer_best.json provenance.

The file is read by 15 consumers including the live daily cycle. Its headline
`sharpe`/`dsr` were being re-stamped with the executing run's id even when they
had been inherited from an earlier run by the warm-start path.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
BEST = REPO / "backend" / "backtest" / "experiments" / "optimizer_best.json"
CHECKER = REPO / "scripts" / "qa" / "check_optimizer_best_provenance.py"


def _checker():
    spec = importlib.util.spec_from_file_location("provcheck", CHECKER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _Engine:
    _strategy_params = {"strategy": "triple_barrier", "tp_pct": 10.0}
    stop_check = None


def _optimizer(**attrs):
    """A QuantStrategyOptimizer with __init__ bypassed -- we are testing the
    save/provenance logic, not the constructor's disk I/O."""
    from backend.backtest.quant_optimizer import QuantStrategyOptimizer
    o = QuantStrategyOptimizer.__new__(QuantStrategyOptimizer)
    o.best_params = {"strategy": "triple_barrier"}
    o.best_sharpe = 1.17
    o.best_dsr = 0.95
    o.num_trials = 3
    o.kept = 0
    o.discarded = 10
    o._run_id = "currentrun"
    o._warm_started = False
    o._warm_started_from_run_id = None
    o._warm_started_from_artifact = None
    for k, v in attrs.items():
        setattr(o, k, v)
    return o


def _saved(monkeypatch, tmp_path, opt) -> dict:
    import backend.backtest.quant_optimizer as qo
    p = tmp_path / "optimizer_best.json"
    monkeypatch.setattr(qo, "_BEST_PARAMS_PATH", p)
    opt._save_best_params()
    return json.loads(p.read_text(encoding="utf-8"))


# ── criterion 1: the checker FAILS on the current on-disk state ──────

def test_checker_fails_on_the_live_mis_attributed_file():
    """The whole point: this must be RED against the repo as it stands.

    optimizer_best.json records run_id=60617e0b / sharpe=1.1704633657934074,
    while that run's ten artifacts produced 0.5384..0.6506 and kept nothing.
    """
    r = _checker().check()
    if r["status"] == "absent":
        pytest.skip("optimizer_best.json not present in this checkout")
    assert r["ok"] is False, (
        f"the checker passes on a file whose provenance is known-bad: {r}"
    )
    assert r["status"] == "mis_attributed"


def test_checker_names_the_true_origin_artifact():
    """A failure that does not say WHERE the number came from is not
    actionable. The checker must locate it by search, not by being told."""
    r = _checker().check()
    if r["status"] == "absent":
        pytest.skip("optimizer_best.json not present")
    assert r.get("origin_artifacts"), "the checker did not locate the true origin"
    assert any("52eb3ffe" in n for n in r["origin_artifacts"]), (
        f"expected the origin to be a 52eb3ffe artifact, got {r['origin_artifacts']}"
    )


def test_checker_passes_when_metrics_reproduce_from_the_named_run(tmp_path, monkeypatch):
    """Mirror guard -- the checker must not simply always fail."""
    mod = _checker()
    results = tmp_path / "results"
    results.mkdir()
    (results / "20260101T000000Z_goodrun-exp01.json").write_text(
        json.dumps({"analytics": {"sharpe": 0.6455483636, "deflated_sharpe": 0.3771}}),
        encoding="utf-8")
    best = tmp_path / "optimizer_best.json"
    # DISTINCT sharpe and dsr values. The first version used 0.5 for both, so
    # it could not tell a sharpe-only check from one that verifies both --
    # exactly the gap the Q/A found (its PROBE A).
    best.write_text(json.dumps({
        "params": {}, "sharpe": 0.6455483636, "dsr": 0.3771,
        "run_id": "goodrun", "metrics_run_id": "goodrun", "schema_version": 2,
    }), encoding="utf-8")
    monkeypatch.setattr(mod, "BEST", best)
    monkeypatch.setattr(mod, "RESULTS", results)
    r = mod.check()
    assert r["ok"] is True and r["status"] == "verified", r


# ── criterion 2: warm-start source persisted as a DISTINCT field ─────

def test_warm_started_best_records_the_source_run_not_the_current_run(tmp_path, monkeypatch):
    """THE DEFECT. A run that inherited its metrics and beat nothing must not
    claim them as its own."""
    o = _optimizer(_warm_started=True, _warm_started_from_run_id="52eb3ffe-exp10",
                   _warm_started_from_artifact="20260328T072722Z_52eb3ffe-exp10.json",
                   kept=0)
    d = _saved(monkeypatch, tmp_path, o)

    assert d["run_id"] == "currentrun", "run_id must still name the WRITER"
    assert d["metrics_run_id"] == "52eb3ffe-exp10", (
        "the persisted metrics belong to the warm-start source, but "
        f"metrics_run_id says {d['metrics_run_id']!r}"
    )
    assert d["metrics_source_artifact"] == "20260328T072722Z_52eb3ffe-exp10.json"
    assert d["warm_started_from"] == "52eb3ffe-exp10"


def test_a_run_that_improved_claims_its_own_metrics(tmp_path, monkeypatch):
    """Symmetry: if the run actually beat the warm-started value (kept > 0),
    the metrics ARE its own and must be attributed to it. Without this the
    fix could pass by always disclaiming."""
    o = _optimizer(_warm_started=True, _warm_started_from_run_id="52eb3ffe-exp10", kept=3)
    d = _saved(monkeypatch, tmp_path, o)
    assert d["metrics_run_id"] == "currentrun", (
        "a run that kept 3 experiments produced its own best; it must claim them"
    )


def test_a_cold_run_attributes_metrics_to_itself(tmp_path, monkeypatch):
    o = _optimizer(_warm_started=False, kept=2)
    d = _saved(monkeypatch, tmp_path, o)
    assert d["metrics_run_id"] == "currentrun"
    assert d["warm_started_from"] is None


# ── criterion 3: kept=0 cannot claim a best it did not produce ───────

def test_kept_zero_with_warm_start_never_self_attributes(tmp_path, monkeypatch):
    """This is the exact live shape: kept=0, warm-started, and the file
    previously claimed the inherited numbers as the current run's."""
    o = _optimizer(_warm_started=True, _warm_started_from_run_id="prior", kept=0)
    d = _saved(monkeypatch, tmp_path, o)
    assert d["kept"] == 0
    assert d["metrics_run_id"] != d["run_id"], (
        "a run that kept nothing is claiming the best as its own"
    )


# ── absence must not read as freshness ───────────────────────────────

def test_absent_metrics_run_id_is_treated_as_undeclared_not_self_attributed(tmp_path, monkeypatch):
    """A pre-schema-v2 file carries no metrics_run_id. That is UNKNOWN
    provenance -- it must never be read as 'run_id produced these'."""
    mod = _checker()
    results = tmp_path / "results"
    results.mkdir()
    (results / "20260101T000000Z_somerun-exp01.json").write_text(
        json.dumps({"analytics": {"sharpe": 0.11, "deflated_sharpe": 0.2}}), encoding="utf-8")
    best = tmp_path / "optimizer_best.json"
    best.write_text(json.dumps({
        "params": {}, "sharpe": 9.99, "dsr": 0.99, "run_id": "somerun",
    }), encoding="utf-8")
    monkeypatch.setattr(mod, "BEST", best)
    monkeypatch.setattr(mod, "RESULTS", results)
    r = mod.check()
    assert r["ok"] is False
    assert "UNDECLARED" in r["detail"], (
        "a file with no metrics_run_id must be reported as undeclared provenance"
    )


# ── additive only: no existing key changes meaning ───────────────────

def test_schema_is_purely_additive(tmp_path, monkeypatch):
    """All 15 consumers are dict.get based, so ADDING keys is safe -- but
    renaming or dropping one breaks named readers."""
    o = _optimizer(kept=1)
    d = _saved(monkeypatch, tmp_path, o)
    for key in ("params", "sharpe", "dsr", "run_id", "kept", "discarded", "saved_at"):
        assert key in d, f"pre-existing key {key!r} was dropped -- consumers break"
    assert d["schema_version"] == 2
    assert "num_trials" in d, "num_trials is required as input to step 82.25"


def test_num_trials_is_persisted_for_the_deflation_fix(tmp_path, monkeypatch):
    o = _optimizer(num_trials=7, kept=1)
    d = _saved(monkeypatch, tmp_path, o)
    assert d["num_trials"] == 7


# ── criterion 4: the deflation math must be UNCHANGED ────────────────

def test_dsr_still_falls_monotonically_as_trials_rise():
    """A provenance fix must not silently alter the statistic it attributes.

    DSR is deflated by the number of trials searched (Bailey/Lopez de Prado):
    the more configurations you tried, the more of your best Sharpe is
    explained by selection. This pins that relationship so a future edit to the
    provenance path cannot quietly change the number it is recording.

    Reproduced from run 60617e0b's own ten artifacts, where DSR falls
    0.6387 (num_trials=2) -> 0.0088 (num_trials=11) WITHIN one run.
    """
    from backend.backtest.analytics import compute_deflated_sharpe

    # PARAMETER CHOICE IS LOAD-BEARING. At observed_sr=0.65 the statistic
    # SATURATES: [1.0, 1.0, 1.0, 0.9999999999997232]. A monotonicity assert
    # there passes on a 3e-13 floating-point crumb and would survive deflation
    # being all but switched off -- a guard that cannot fail meaningfully.
    # observed_sr=0.15 sits on the responsive part of the curve.
    common = dict(observed_sr=0.15, variance_of_srs=0.04, skewness=0.0,
                  kurtosis=3.0, T=1661, periods_per_year=1)
    dsrs = [compute_deflated_sharpe(num_trials=n, **common) for n in (2, 5, 11, 50)]

    assert all(0.0 <= d <= 1.0 for d in dsrs), dsrs
    assert dsrs == sorted(dsrs, reverse=True), (
        f"DSR must fall as trials rise; got {dsrs}. If this reversed, the "
        "deflation math changed and every recorded DSR means something new."
    )
    # MATERIAL, not a crumb: the gradient must be large enough that switching
    # deflation off would break this.
    assert dsrs[0] - dsrs[-1] > 0.5, (
        f"deflation gradient is only {dsrs[0] - dsrs[-1]:.2e} across 2->50 "
        f"trials ({dsrs}); that is too flat to prove the statistic still "
        "responds to the trial count"
    )
    assert dsrs[0] > 0.5 > dsrs[-1], (
        f"expected the gradient to straddle 0.5; got {dsrs}"
    )


def test_the_live_files_own_artifacts_show_the_deflation_gradient():
    """The same property, measured on the REAL artifacts rather than a fixture
    -- so a fixture that drifts from production cannot hide a change."""
    import glob as _glob

    files = sorted(_glob.glob(
        str(REPO / "backend" / "backtest" / "experiments" / "results" / "*60617e0b*.json")))
    if len(files) < 3:
        pytest.skip("run 60617e0b artifacts not present in this checkout")

    pairs = []
    for f in files:
        a = json.loads(Path(f).read_text(encoding="utf-8")).get("analytics") or {}
        if a.get("num_trials") is not None and a.get("deflated_sharpe") is not None:
            pairs.append((a["num_trials"], a["deflated_sharpe"]))
    pairs.sort()
    assert len(pairs) >= 3, pairs
    assert pairs[0][1] > pairs[-1][1], (
        f"DSR did not fall as trials rose across the real run: {pairs}"
    )


# ── criterion 1, the DSR half (Q/A cycle-1 BLOCK) ────────────────────

def _probe(tmp_path, monkeypatch, best: dict, artifact: dict) -> dict:
    mod = _checker()
    results = tmp_path / "results"
    results.mkdir(exist_ok=True)
    (results / "20260101T000000Z_goodrun-exp01.json").write_text(
        json.dumps({"analytics": artifact}), encoding="utf-8")
    b = tmp_path / "optimizer_best.json"
    b.write_text(json.dumps(best), encoding="utf-8")
    monkeypatch.setattr(mod, "BEST", b)
    monkeypatch.setattr(mod, "RESULTS", results)
    return mod.check()


def test_a_fabricated_dsr_is_rejected_even_when_sharpe_reproduces(tmp_path, monkeypatch):
    """The Q/A's PROBE A. The first checker compared sharpe ONLY -- it read
    deflated_sharpe and discarded it -- so a file could carry a real sharpe and
    an invented DSR and still report 'verified'.

    DSR is the money-path half: it is what the promotion gate spends
    (DSR >= 0.95) and the bar every rotation challenger must clear."""
    r = _probe(tmp_path, monkeypatch,
               best={"sharpe": 0.6455483636, "dsr": 0.99, "run_id": "goodrun",
                     "metrics_run_id": "goodrun"},
               artifact={"sharpe": 0.6455483636, "deflated_sharpe": 0.05})
    assert r["ok"] is False, "a fabricated DSR passed provenance"
    assert r["status"] == "dsr_mis_attributed"
    assert "0.99" in r["detail"] and "0.05" in r["detail"]


def test_a_missing_dsr_is_rejected_when_the_artifact_has_one(tmp_path, monkeypatch):
    """The Q/A's PROBE B."""
    r = _probe(tmp_path, monkeypatch,
               best={"sharpe": 0.6455483636, "run_id": "goodrun",
                     "metrics_run_id": "goodrun"},
               artifact={"sharpe": 0.6455483636, "deflated_sharpe": 0.05})
    assert r["ok"] is False and r["status"] == "dsr_mis_attributed"


def test_both_statistics_matching_is_what_verifies(tmp_path, monkeypatch):
    """Mirror: the stricter check must not simply always fail. Uses DISTINCT
    sharpe and dsr so a sharpe-only implementation cannot pass this either."""
    r = _probe(tmp_path, monkeypatch,
               best={"sharpe": 0.6455483636, "dsr": 0.3771, "run_id": "goodrun",
                     "metrics_run_id": "goodrun"},
               artifact={"sharpe": 0.6455483636, "deflated_sharpe": 0.3771})
    assert r["ok"] is True and r["status"] == "verified", r
    assert "dsr" in r["detail"], "the verified message should name both statistics"
