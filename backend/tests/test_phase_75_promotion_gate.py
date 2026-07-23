"""phase-75.8: promotion-gate stub-fabrication refusal + governance divergence.

Covers the immutable criteria of masterplan step 75.8:
- gauntlet.run(dry_run=False) raises NotImplementedError and writes nothing;
  no code path can emit a report labeled dry_run:false from _run_regime_stub
  (behavioral pincer: both run() branches + the _write_report refusal +
  an AST check that report.json is written only by _write_report).
- promotion_gate rejects the stub fingerprint (bt_drawdown == drawdown for
  every non-skipped regime) while realistic divergent reports pass; the
  empty/all-skipped list is NOT fingerprinted (all([]) trap); a REAL
  gauntlet dry-run report is rejected end-to-end (anti-fixture-divorce).
- promotion_gate --dry-run leaves optimizer_best.json byte-identical on
  BOTH write paths (allocation-stage init + gauntlet stamp), while the
  non-dry-run path still writes (guards are not over-broad).
- divergence checker flags daily-loss 4.0-vs-2.0 with current repo values,
  reports trailing-dd as NON-divergent (unit normalization), raises
  nothing, and mutates nothing; lifespan wires it WARNING-only.

Offline-only: the two CLIs are loaded from file via importlib and every
Path constant is monkeypatched into tmp_path -- the real
optimizer_best.json and handoff/ trees are never touched.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import logging
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
GAUNTLET_PY = REPO / "scripts" / "risk" / "gauntlet.py"
PROMOTION_GATE_PY = REPO / "scripts" / "risk" / "promotion_gate.py"
DIVERGENCE_PY = REPO / "backend" / "governance" / "divergence.py"
MAIN_PY = REPO / "backend" / "main.py"
LIMITS_YAML = REPO / "backend" / "governance" / "limits.yaml"
REAL_OPTIMIZER_BEST = (
    REPO / "backend" / "backtest" / "experiments" / "optimizer_best.json"
)


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def gauntlet(tmp_path, monkeypatch):
    mod = _load(GAUNTLET_PY, "gauntlet_75_8")
    monkeypatch.setattr(mod, "OUT_DIR", tmp_path / "gauntlet_out")
    monkeypatch.setattr(mod, "RUNS_LOG", tmp_path / "gauntlet_runs.jsonl")
    return mod


@pytest.fixture()
def pgate(tmp_path, monkeypatch):
    mod = _load(PROMOTION_GATE_PY, "promotion_gate_75_8")
    monkeypatch.setattr(mod, "REPO", tmp_path)
    monkeypatch.setattr(mod, "OPTIMIZER_BEST", tmp_path / "optimizer_best.json")
    monkeypatch.setattr(mod, "OUT", tmp_path / "promotion_gate_output.json")
    monkeypatch.setattr(mod, "GAUNTLET_ROOT", tmp_path / "gauntlet")
    return mod


def _regime(rid: str, dd: float, bt: float, *, skipped: bool = False,
            forced: int = 0) -> dict:
    if skipped:
        return {"regime_id": rid, "name": rid, "skipped": True,
                "reason": "intraday_only"}
    return {"id": rid, "drawdown": dd, "bt_drawdown": bt,
            "forced_exits": forced, "regime_id": rid, "name": rid,
            "skipped": False}


def _mc(p99: float, bt: float, breaches: int = 0) -> dict:
    return {"n_paths": 1000, "n_days": 252, "p99_drawdown": p99,
            "bt_drawdown": bt, "breaches": breaches}


def _report(per_regime: list, mc: dict) -> dict:
    return {"strategy": "baseline", "seed": 42, "ts": "2026-07-23T00:00:00Z",
            "dry_run": False, "per_regime": per_regime, "monte_carlo": mc}


# Mirrors what _run_regime_stub actually produces: bt == dd exactly on
# every non-skipped regime, MC p99 == bt, forced/breaches zero -- passes
# all four evaluator hard gates by construction, so WITHOUT the
# fingerprint check this report promotes (that is what makes the
# rejection test load-bearing, not vacuous).
def _stub_fingerprint_report() -> dict:
    return _report(
        [_regime("covid", 0.12, 0.12), _regime("gfc", 0.31, 0.31),
         _regime("bull", 0.07, 0.07), _regime("flash", 0.0, 0.0, skipped=True)],
        _mc(0.18, 0.18),
    )


def _realistic_report() -> dict:
    return _report(
        [_regime("covid", 0.10, 0.08), _regime("gfc", 0.05, 0.045),
         _regime("bull", 0.02, 0.019), _regime("flash", 0.0, 0.0, skipped=True)],
        _mc(0.12, 0.10),
    )


def _install_gauntlet_fixture(pgate_mod, report: dict) -> None:
    d = pgate_mod.GAUNTLET_ROOT / "baseline"
    d.mkdir(parents=True, exist_ok=True)
    (d / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )


def _run_main(pgate_mod, monkeypatch, capsys, *argv: str) -> tuple[int, str]:
    monkeypatch.setattr(sys, "argv", ["promotion_gate.py", *argv])
    rc = pgate_mod.main()
    return rc, capsys.readouterr().out


# --------------------------------------------------------------------------
# Criterion 1 -- gauntlet refuses to fabricate
# --------------------------------------------------------------------------

def test_gauntlet_live_mode_raises_not_implemented(gauntlet):
    with pytest.raises(NotImplementedError):
        gauntlet.run("baseline", dry_run=False, seed=42)
    # Refusal happens BEFORE any artifact is produced.
    assert not gauntlet.OUT_DIR.exists()
    assert not gauntlet.RUNS_LOG.exists()


def test_gauntlet_dry_run_report_is_labeled_true_and_fingerprinted(gauntlet):
    report = gauntlet.run("baseline", dry_run=True, seed=42)
    assert report["dry_run"] is True
    on_disk = json.loads(
        (gauntlet.OUT_DIR / "baseline" / "report.json").read_text(
            encoding="utf-8")
    )
    assert on_disk["dry_run"] is True
    # Pin the stub's fingerprint to production behavior: every non-skipped
    # regime carries bt_drawdown == drawdown exactly. This is the property
    # the promotion_gate rejection keys on (fixture-divorce guard).
    non_skipped = [r for r in on_disk["per_regime"] if not r.get("skipped")]
    assert non_skipped
    assert all(r["bt_drawdown"] == r["drawdown"] for r in non_skipped)
    assert len(gauntlet.RUNS_LOG.read_text(
        encoding="utf-8").strip().splitlines()) == 1


def test_write_report_refuses_anything_not_labeled_dry_run_true(
        gauntlet, tmp_path):
    out = tmp_path / "refused"
    for bad in ({"dry_run": False}, {}, {"dry_run": 1}, {"dry_run": "true"}):
        with pytest.raises(RuntimeError):
            gauntlet._write_report(bad, out)
    assert not out.exists()
    ok_dir = tmp_path / "accepted"
    gauntlet._write_report({"dry_run": True, "strategy": "x"}, ok_dir)
    assert (ok_dir / "report.json").exists()


def test_gauntlet_report_json_written_only_by_write_report():
    """Structural support for 'no code path can emit a dry_run:false
    report': the only function that names report.json or calls
    write_text is the guarded _write_report."""
    tree = ast.parse(GAUNTLET_PY.read_text(encoding="utf-8"))
    offenders: set[str] = set()
    for fn in [n for n in ast.walk(tree)
               if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
        for node in ast.walk(fn):
            if (isinstance(node, ast.Constant)
                    and isinstance(node.value, str)
                    and "report.json" in node.value):
                offenders.add(fn.name)
            if (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "write_text"):
                offenders.add(fn.name)
    assert offenders == {"_write_report"}


# --------------------------------------------------------------------------
# Criterion 2 -- stub-fingerprint rejection at the promotion gate
# --------------------------------------------------------------------------

def test_promotion_gate_blocks_stub_fingerprint(pgate, monkeypatch, capsys):
    pgate.OPTIMIZER_BEST.write_bytes(REAL_OPTIMIZER_BEST.read_bytes())
    before = pgate.OPTIMIZER_BEST.read_bytes()
    _install_gauntlet_fixture(pgate, _stub_fingerprint_report())
    rc, out = _run_main(pgate, monkeypatch, capsys, "--require-gauntlet")
    assert rc == 1
    assert "stub fingerprint" in out
    # Blocked BEFORE any writer: even without --dry-run the file is intact.
    assert pgate.OPTIMIZER_BEST.read_bytes() == before


def test_promotion_gate_realistic_report_passes_shape_validation(
        pgate, monkeypatch, capsys):
    pgate.OPTIMIZER_BEST.write_bytes(REAL_OPTIMIZER_BEST.read_bytes())
    _install_gauntlet_fixture(pgate, _realistic_report())
    rc, out = _run_main(pgate, monkeypatch, capsys, "--require-gauntlet")
    assert rc == 0
    assert "stub fingerprint" not in out
    # Non-dry-run stamp still writes (the guards are not over-broad).
    final = json.loads(pgate.OPTIMIZER_BEST.read_text(encoding="utf-8"))
    raw = (pgate.GAUNTLET_ROOT / "baseline" / "report.json").read_bytes()
    assert final["gauntlet_report_hash"] == hashlib.sha256(raw).hexdigest()


def test_single_divergent_regime_defeats_the_fingerprint(
        pgate, monkeypatch, capsys):
    """Fixture mutation (M9): one regime with bt != dd means the report is
    NOT stub-shaped and must not be blocked."""
    report = _stub_fingerprint_report()
    report["per_regime"][0] = _regime("covid", 0.10, 0.09)
    pgate.OPTIMIZER_BEST.write_bytes(REAL_OPTIMIZER_BEST.read_bytes())
    _install_gauntlet_fixture(pgate, report)
    rc, out = _run_main(pgate, monkeypatch, capsys, "--require-gauntlet")
    assert rc == 0
    assert "stub fingerprint" not in out


def test_all_skipped_regimes_are_not_fingerprinted(pgate, monkeypatch, capsys):
    """all([]) is True -- an empty non-skipped list must NOT be treated as
    the stub fingerprint."""
    report = _report(
        [_regime("flash", 0.0, 0.0, skipped=True),
         _regime("flash2", 0.0, 0.0, skipped=True)],
        _mc(0.12, 0.10),
    )
    pgate.OPTIMIZER_BEST.write_bytes(REAL_OPTIMIZER_BEST.read_bytes())
    _install_gauntlet_fixture(pgate, report)
    rc, out = _run_main(pgate, monkeypatch, capsys, "--require-gauntlet")
    assert rc == 0
    assert "stub fingerprint" not in out


def test_real_gauntlet_dry_run_report_is_rejected_end_to_end(
        gauntlet, pgate, monkeypatch, capsys):
    """Anti-fixture-divorce: feed the ACTUAL stub output (not a hand-built
    fixture) into the promotion gate -- it must be fingerprint-blocked."""
    gauntlet.run("baseline", dry_run=True, seed=7)
    real_stub = (gauntlet.OUT_DIR / "baseline" / "report.json").read_bytes()
    d = pgate.GAUNTLET_ROOT / "baseline"
    d.mkdir(parents=True, exist_ok=True)
    (d / "report.json").write_bytes(real_stub)
    pgate.OPTIMIZER_BEST.write_bytes(REAL_OPTIMIZER_BEST.read_bytes())
    rc, out = _run_main(pgate, monkeypatch, capsys, "--require-gauntlet")
    assert rc == 1
    assert "stub fingerprint" in out


# --------------------------------------------------------------------------
# Criterion 3 -- --dry-run is strictly no-write on optimizer_best.json
# --------------------------------------------------------------------------

def test_dry_run_byte_identical_when_allocation_missing(
        pgate, monkeypatch, capsys):
    """Exercises the allocation-stage init writer's guard."""
    blob = json.dumps({"params": {"strategy": "triple_barrier"},
                       "sharpe": 1.1}, indent=2) + "\n"
    pgate.OPTIMIZER_BEST.write_text(blob, encoding="utf-8")
    rc, out = _run_main(pgate, monkeypatch, capsys, "--dry-run")
    assert rc == 0
    assert pgate.OPTIMIZER_BEST.read_text(encoding="utf-8") == blob
    assert "would_set" in out and "allocation_pct" in out


def test_dry_run_byte_identical_on_gauntlet_stamp(pgate, monkeypatch, capsys):
    """Exercises the gauntlet-stamp writer's guard (real repo bytes carry
    allocation_pct + a stale hash, so ONLY the stamp path would fire)."""
    pgate.OPTIMIZER_BEST.write_bytes(REAL_OPTIMIZER_BEST.read_bytes())
    before = pgate.OPTIMIZER_BEST.read_bytes()
    _install_gauntlet_fixture(pgate, _realistic_report())
    rc, out = _run_main(pgate, monkeypatch, capsys,
                        "--dry-run", "--require-gauntlet")
    assert rc == 0
    assert pgate.OPTIMIZER_BEST.read_bytes() == before
    assert "would_set" in out and "gauntlet_report_hash" in out


def test_dry_run_fresh_deploy_writes_no_file(pgate, monkeypatch, capsys):
    """No optimizer_best.json at all: --dry-run must not create one (and
    the post-write re-read must not crash on the missing file)."""
    assert not pgate.OPTIMIZER_BEST.exists()
    rc, out = _run_main(pgate, monkeypatch, capsys, "--dry-run")
    assert rc == 0
    assert not pgate.OPTIMIZER_BEST.exists()


def test_non_dry_run_allocation_init_still_writes(pgate, monkeypatch, capsys):
    """The guard is not over-broad: without --dry-run the init writer
    still fires exactly as before."""
    pgate.OPTIMIZER_BEST.write_text(
        json.dumps({"params": {}}, indent=2) + "\n", encoding="utf-8")
    rc, out = _run_main(pgate, monkeypatch, capsys)
    assert rc == 0
    final = json.loads(pgate.OPTIMIZER_BEST.read_text(encoding="utf-8"))
    assert final["allocation_pct"] == pgate.STAGES[0]
    assert final["stage"] == 0


# --------------------------------------------------------------------------
# Criterion 4 -- governance divergence checker (observability only)
# --------------------------------------------------------------------------

def test_divergence_flags_daily_loss_and_clears_trailing_dd():
    from backend.governance.divergence import compute_divergence

    pairs = {p["name"]: p for p in compute_divergence()}
    daily = pairs["daily_loss_kill_switch"]
    assert daily["settings_value_pct"] == 4.0
    assert daily["governed_value_pct"] == 2.0
    assert daily["divergent"] is True
    trailing = pairs["trailing_dd_kill_switch"]
    # Unit normalization: governed 0.10 FRACTION == live 10.0 PERCENT.
    # Without the x100 normalization this pair false-positives.
    assert trailing["settings_value_pct"] == 10.0
    assert trailing["governed_value_pct"] == 10.0
    assert trailing["divergent"] is False


def test_divergence_log_helper_raises_nothing_and_mutates_nothing(caplog):
    from backend.config.settings import get_settings
    from backend.governance.divergence import log_divergence_warnings

    limits_before = hashlib.sha256(LIMITS_YAML.read_bytes()).hexdigest()
    s = get_settings()
    settings_before = (s.paper_daily_loss_limit_pct,
                      s.paper_trailing_dd_limit_pct)
    with caplog.at_level(logging.INFO, logger="backend.governance.divergence"):
        pairs = log_divergence_warnings()
    assert pairs, "helper must return the computed pairs"
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert any("daily_loss_kill_switch" in r.getMessage() for r in warnings)
    assert not any("trailing_dd_kill_switch" in r.getMessage()
                   for r in warnings)
    assert hashlib.sha256(
        LIMITS_YAML.read_bytes()).hexdigest() == limits_before
    assert (s.paper_daily_loss_limit_pct,
            s.paper_trailing_dd_limit_pct) == settings_before


def test_divergence_log_helper_is_fail_open(monkeypatch, caplog):
    import backend.governance.divergence as div

    def _boom():
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(div, "compute_divergence", _boom)
    with caplog.at_level(logging.WARNING,
                         logger="backend.governance.divergence"):
        result = div.log_divergence_warnings()
    assert result == []
    assert any("fail-open" in r.getMessage() for r in caplog.records)


def test_lifespan_invokes_divergence_inside_try_warning_only():
    """The lifespan wiring is fail-open (inside a Try) and calls only the
    never-raising log helper. Behavioral weight is carried by the two
    tests above; this pins the wiring point in main.py."""
    tree = ast.parse(MAIN_PY.read_text(encoding="utf-8"))
    lifespan = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.AsyncFunctionDef) and n.name == "lifespan"
    )
    calls_in_try = [
        node.func.id
        for try_node in ast.walk(lifespan) if isinstance(try_node, ast.Try)
        for node in ast.walk(try_node)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    ]
    assert "log_divergence_warnings" in calls_in_try


# --------------------------------------------------------------------------
# Criterion 6 -- touched scripts remain parseable
# --------------------------------------------------------------------------

@pytest.mark.parametrize("path", [GAUNTLET_PY, PROMOTION_GATE_PY,
                                  DIVERGENCE_PY], ids=lambda p: p.name)
def test_touched_files_parse(path):
    ast.parse(path.read_text(encoding="utf-8"))
