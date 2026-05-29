"""phase-47.6: behavioral guards for the dynamic strategy selector.

Verifies the north-star rotation logic: gate-filter (DSR/PBO), DSR-desc ranking,
and the anti-churn hysteresis (switch from the incumbent only on a material
Delta-DSR). Pure-function tests with synthetic per-strategy DSR dicts -- no live
backtest needed. Mirrors the structure of test_friday_promotion.py.
"""
from __future__ import annotations

from backend.autoresearch.strategy_selector import select_best_strategy


def _c(sid: str, dsr: float, pbo: float = 0.10) -> dict:
    return {"strategy": sid, "dsr": dsr, "pbo": pbo, "params": {"k": sid}}


def test_first_selection_picks_top_dsr():
    out = select_best_strategy(
        [_c("A", 0.96), _c("B", 0.99), _c("C", 0.97)], incumbent=None
    )
    assert out["selected_id"] == "B"
    assert out["switched"] is True
    assert out["reason"] == "first_selection"
    assert out["ranked"] == ["B", "C", "A"]  # DSR desc


def test_dsr_gate_veto_excludes_below_min():
    # A has the highest DSR but is below the 0.95 gate -> excluded.
    out = select_best_strategy([_c("A", 0.94), _c("B", 0.96)], incumbent=None)
    assert out["selected_id"] == "B"
    assert out["ranked"] == ["B"]  # A gate-vetoed


def test_pbo_veto_excludes_overfit():
    # A has top DSR but PBO above the 0.20 cap -> excluded by PromotionGate.
    out = select_best_strategy(
        [_c("A", 0.99, pbo=0.50), _c("B", 0.96, pbo=0.05)], incumbent=None
    )
    assert out["selected_id"] == "B"
    assert "A" not in out["ranked"]


def test_anti_churn_below_min_improvement_retains_incumbent():
    # Challenger beats incumbent by only 0.005 DSR (< 0.01 default) -> retain.
    inc = _c("INC", 0.970)
    out = select_best_strategy([_c("CH", 0.975)], incumbent=inc)
    assert out["selected_id"] == "INC"
    assert out["switched"] is False
    assert out["reason"] == "below_min_improvement"


def test_switch_on_sufficient_improvement():
    # Challenger beats incumbent by 0.02 DSR (>= 0.01) -> switch.
    inc = _c("INC", 0.96)
    out = select_best_strategy([_c("CH", 0.98)], incumbent=inc)
    assert out["selected_id"] == "CH"
    assert out["switched"] is True
    assert out["reason"] == "dsr_improvement"
    assert out["delta_dsr"] == 0.02


def test_incumbent_is_top_retained_no_switch():
    inc = _c("S1", 0.98)
    out = select_best_strategy([_c("S1", 0.98), _c("S2", 0.96)], incumbent=inc)
    assert out["selected_id"] == "S1"
    assert out["switched"] is False
    assert out["reason"] == "incumbent_is_top"


def test_no_passer_retains_incumbent_never_cash():
    inc = _c("INC", 0.97)
    out = select_best_strategy([_c("A", 0.90), _c("B", 0.80)], incumbent=inc)
    assert out["selected_id"] == "INC"
    assert out["switched"] is False
    assert out["reason"] == "no_candidate_passed_gate"


def test_weak_incumbent_replaced_by_large_margin():
    # Incumbent DSR collapsed below a strong challenger by a big margin -> switch.
    inc = _c("INC", 0.951)
    out = select_best_strategy([_c("CH", 0.999)], incumbent=inc)
    assert out["selected_id"] == "CH"
    assert out["switched"] is True
    assert out["reason"] == "dsr_improvement"
