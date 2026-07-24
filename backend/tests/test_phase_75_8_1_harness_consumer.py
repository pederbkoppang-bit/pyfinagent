"""phase-75.8.1: the SECOND gauntlet-report consumer refuses fabricated evidence.

75.8 hardened scripts/risk/promotion_gate.py; backend/autonomous_harness.py::
promote_strategy -- the consumer with the eventual real caller -- still trusted
handoff/gauntlet/<strategy>/report.json blindly: the evaluator is extensional
(checks values, not provenance), so a dry-run stub whose bt_drawdown ==
drawdown on every non-skipped regime passes all four hard gates by
construction. This suite pins the shared predicate
(backend/backtest/gauntlet/report_integrity.py) THROUGH BOTH consumers:

  (a) stub-fingerprint refusal through promote_strategy (raise + blocklist,
      no exception swallowed into a promote);
  (b) the NEW dry_run:true-label refusal through both consumers (75.8's
      inline block never checked the label -- the step-text correction);
  (c) a realistic divergent dry_run:false report still promotes;
  (d) empty / all-skipped per_regime NOT fingerprinted (all([]) trap),
      through both consumers;
  (e) anti-fixture-divorce: the ACTUAL gauntlet.run(dry_run=True) bytes are
      refused through promote_strategy;
  (f) single-implementation proof BY BEHAVIOR: monkeypatching the shared
      predicate flips BOTH consumers (beats any source-scan for C2).

Offline-only: every autonomous_harness path constant is monkeypatched to
tmp_path (the phase4_9_redteam pattern); promotion_gate/gauntlet CLIs are
loaded from file with their Path constants remapped (the 75.8 pattern).
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import backend.autonomous_harness as ah
from backend.backtest.gauntlet import report_integrity

GAUNTLET_PY = REPO / "scripts" / "risk" / "gauntlet.py"
PROMOTION_GATE_PY = REPO / "scripts" / "risk" / "promotion_gate.py"
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
def harness_env(tmp_path, monkeypatch):
    monkeypatch.setattr(ah, "_GAUNTLET_ROOT", tmp_path / "gauntlet")
    monkeypatch.setattr(ah, "_BLOCKLIST_PATH", tmp_path / "blocklist.jsonl")
    monkeypatch.setattr(ah, "_HARNESS_LOG", tmp_path / "harness_log.md")
    return tmp_path


@pytest.fixture()
def pgate(tmp_path, monkeypatch):
    mod = _load(PROMOTION_GATE_PY, "promotion_gate_75_8_1")
    monkeypatch.setattr(mod, "REPO", tmp_path)
    monkeypatch.setattr(mod, "OPTIMIZER_BEST", tmp_path / "optimizer_best.json")
    monkeypatch.setattr(mod, "OUT", tmp_path / "promotion_gate_output.json")
    monkeypatch.setattr(mod, "GAUNTLET_ROOT", tmp_path / "gauntlet_pg")
    return mod


@pytest.fixture()
def gauntlet(tmp_path, monkeypatch):
    mod = _load(GAUNTLET_PY, "gauntlet_75_8_1")
    monkeypatch.setattr(mod, "OUT_DIR", tmp_path / "gauntlet_out")
    monkeypatch.setattr(mod, "RUNS_LOG", tmp_path / "gauntlet_runs.jsonl")
    return mod


def _regime(rid: str, dd: float, bt: float, *, skipped: bool = False) -> dict:
    if skipped:
        return {"regime_id": rid, "name": rid, "skipped": True,
                "reason": "intraday_only"}
    return {"id": rid, "drawdown": dd, "bt_drawdown": bt, "forced_exits": 0,
            "regime_id": rid, "name": rid, "skipped": False}


def _report(per_regime: list, *, dry_run: bool = False) -> dict:
    return {"strategy": "s1", "seed": 42, "ts": "2026-07-24T00:00:00Z",
            "dry_run": dry_run, "per_regime": per_regime,
            "monte_carlo": {"n_paths": 1000, "n_days": 252,
                            "p99_drawdown": 0.12, "bt_drawdown": 0.10,
                            "breaches": 0}}


STUB = _report([_regime("covid", 0.12, 0.12), _regime("gfc", 0.31, 0.31)])
DRY_RUN_DIVERGENT = _report(
    [_regime("covid", 0.10, 0.08), _regime("gfc", 0.05, 0.045)], dry_run=True)
REALISTIC = _report(
    [_regime("covid", 0.10, 0.08), _regime("gfc", 0.05, 0.045)])
ALL_SKIPPED = _report([_regime("flash", 0.0, 0.0, skipped=True),
                       _regime("flash2", 0.0, 0.0, skipped=True)])
EMPTY_REGIMES = _report([])


def _install(root: Path, strategy: str, report: dict) -> None:
    d = root / strategy
    d.mkdir(parents=True, exist_ok=True)
    (d / "report.json").write_text(json.dumps(report), encoding="utf-8")


def _run_pgate(pgate_mod, monkeypatch, capsys, report: dict) -> tuple[int, str]:
    pgate_mod.OPTIMIZER_BEST.write_bytes(REAL_OPTIMIZER_BEST.read_bytes())
    d = pgate_mod.GAUNTLET_ROOT / "baseline"
    d.mkdir(parents=True, exist_ok=True)
    (d / "report.json").write_text(json.dumps(report), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["promotion_gate.py", "--require-gauntlet"])
    rc = pgate_mod.main()
    return rc, capsys.readouterr().out


# -- (a) stub fingerprint through promote_strategy (C1) ---------------------

def test_stub_report_refused_and_blocklisted_through_promote(harness_env):
    _install(ah._GAUNTLET_ROOT, "s1", STUB)
    with pytest.raises(ah.PromotionBlocked, match="stub fingerprint"):
        ah.promote_strategy("s1")
    rows = [json.loads(x) for x in
            ah._BLOCKLIST_PATH.read_text(encoding="utf-8").splitlines()]
    assert any("stub fingerprint" in r.get("reason", "") for r in rows), (
        "refusal must land on the blocklist with the integrity reason")


# -- (b) dry_run label through BOTH consumers (C1 + step-text correction) ---

def test_dry_run_labeled_divergent_report_refused_through_promote(harness_env):
    # Divergent values -> NOT stub-shaped; only the label refusal catches it.
    _install(ah._GAUNTLET_ROOT, "s1", DRY_RUN_DIVERGENT)
    with pytest.raises(ah.PromotionBlocked, match="dry_run:true"):
        ah.promote_strategy("s1")


def test_dry_run_labeled_divergent_report_blocked_through_pgate(
        pgate, monkeypatch, capsys):
    rc, out = _run_pgate(pgate, monkeypatch, capsys, DRY_RUN_DIVERGENT)
    assert rc == 1
    assert "dry_run:true" in out


# -- (c) realistic divergent dry_run:false promotes (C1) --------------------

def test_realistic_divergent_report_promotes(harness_env):
    _install(ah._GAUNTLET_ROOT, "s1", REALISTIC)
    verdict = ah.promote_strategy("s1")
    assert verdict["overall_pass"] is True
    assert not ah._BLOCKLIST_PATH.exists(), "a promote must not blocklist"


# -- (d) empty / all-skipped NOT fingerprinted, through BOTH consumers (C3) --

@pytest.mark.parametrize("report", [ALL_SKIPPED, EMPTY_REGIMES],
                         ids=["all-skipped", "empty"])
def test_vacuous_regime_lists_not_fingerprinted_through_promote(
        harness_env, report):
    _install(ah._GAUNTLET_ROOT, "s1", report)
    # Not fingerprint-refused: either promotes (vacuous evaluator pass) or,
    # if it ever fails, fails for a NON-fingerprint reason. Both reports are
    # dry_run:false so the label leg is not in play either.
    try:
        ah.promote_strategy("s1")
    except ah.PromotionBlocked as e:
        assert "stub fingerprint" not in str(e)


@pytest.mark.parametrize("report", [ALL_SKIPPED, EMPTY_REGIMES],
                         ids=["all-skipped", "empty"])
def test_vacuous_regime_lists_not_fingerprinted_through_pgate(
        pgate, monkeypatch, capsys, report):
    _rc, out = _run_pgate(pgate, monkeypatch, capsys, report)
    assert "stub fingerprint" not in out


# -- (e) anti-fixture-divorce: REAL gauntlet stub bytes through promote -----

def test_real_gauntlet_dry_run_output_refused_through_promote(
        harness_env, gauntlet):
    gauntlet.run("baseline", dry_run=True, seed=7)
    real_bytes = (gauntlet.OUT_DIR / "baseline" / "report.json").read_bytes()
    d = ah._GAUNTLET_ROOT / "s1"
    d.mkdir(parents=True, exist_ok=True)
    (d / "report.json").write_bytes(real_bytes)
    with pytest.raises(ah.PromotionBlocked) as exc:
        ah.promote_strategy("s1")
    # The real stub is BOTH fingerprinted and dry_run:true; fingerprint-first
    # ordering pins which reason surfaces.
    assert "stub fingerprint" in str(exc.value)


# -- (f) single shared implementation, proven by behavior (C2) --------------

def test_monkeypatching_shared_predicate_flips_both_consumers(
        harness_env, pgate, monkeypatch, capsys):
    forced_reason = "forced-by-75.8.1-single-implementation-probe"
    monkeypatch.setattr(report_integrity, "check_report_integrity",
                        lambda report: (False, forced_reason))
    # Consumer 1: promote_strategy refuses a REALISTIC report.
    _install(ah._GAUNTLET_ROOT, "s1", REALISTIC)
    with pytest.raises(ah.PromotionBlocked, match=forced_reason):
        ah.promote_strategy("s1")
    # Consumer 2: promotion_gate blocks the SAME realistic report.
    rc, out = _run_pgate(pgate, monkeypatch, capsys, REALISTIC)
    assert rc == 1
    assert forced_reason in out


def test_shared_predicate_probe_is_not_vacuous(harness_env, pgate,
                                               monkeypatch, capsys):
    """Stub self-test: WITHOUT the forced monkeypatch, the same realistic
    report passes both consumers -- so the flip above is caused by the shared
    predicate, not by the fixtures."""
    _install(ah._GAUNTLET_ROOT, "s1", REALISTIC)
    assert ah.promote_strategy("s1")["overall_pass"] is True
    rc, _out = _run_pgate(pgate, monkeypatch, capsys, REALISTIC)
    assert rc == 0
