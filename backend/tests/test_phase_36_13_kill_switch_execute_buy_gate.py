"""phase-36.13 -- the kill switch must gate the ACT OF BUYING, not just the cycle.

THE BYPASS. Until this step the switch was enforced only on the autonomous-cycle
path (`check_and_enforce_kill_switch`, and the loop's halt branch). `execute_buy`
itself contained no reference to kill_switch / is_paused / paused, and the MCP
`publish_signal` BUY path (`signals_server.py:444`) calls it directly -- so an
order could be placed while the book was PAUSED after a real breach-triggered
flatten, and while its loss baselines were unrestorable. That is CWE-424
(improper protection of alternate path); the remedy named by its parent CWE-638
is "a single interface that performs the access checks". `execute_buy` and
`execute_sell` are the only two `paper_trades` producers, so a gate in
`execute_buy` is COMPLETE mediation for BUYs.

WHY execute_sell IS NOT GATED, deliberately: `check_and_enforce_kill_switch`
performs its flatten THROUGH `execute_sell`, so gating sells on `paused` would
deadlock the pause -- the switch could never close the positions it had just
decided to close. Selling is the safe direction. Do not "fix" this asymmetry.

WHY THE GATE READS `baselines_present` AND NEVER `armed`: phase-36.9 made `armed`
mean "can this leg fire RIGHT NOW", which is False for an anchor merely from
yesterday. Gating BUYs on that would refuse every order each morning until the
daily roll -- exactly the money-path regression 36.9's cycle-1 evaluator caught.
Lost baselines are a durable fault; an overnight anchor is not.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


class _Paused:
    """A kill-switch state that reports the book PAUSED."""
    def is_paused(self):
        return True

    def snapshot(self):
        return {"paused": True, "pause_reason": "daily_loss_breach",
                "sod_nav": 23838.19, "peak_nav": 24666.57}


class _Healthy:
    def is_paused(self):
        return False

    def snapshot(self):
        return {"paused": False, "pause_reason": None,
                "sod_nav": 23838.19, "peak_nav": 24666.57}


class _LostBaselines:
    """Not paused, but the loss baselines are unrestorable -- the DISARMED case."""
    def is_paused(self):
        return False

    def snapshot(self):
        return {"paused": False, "pause_reason": None,
                "sod_nav": None, "peak_nav": None}


class _StaleButPresent:
    """Baselines intact; the daily anchor is merely from yesterday.

    Post-36.9 `evaluate_breach` reports `armed: False` for this state. The BUY gate
    must NOT refuse it -- see the module docstring.
    """
    def is_paused(self):
        return False

    def snapshot(self):
        return {"paused": False, "pause_reason": None,
                "sod_nav": 23838.19, "sod_date": "2026-07-24",
                "peak_nav": 24666.57}


class _Unreadable:
    """Raises on every read -- the fail-closed case (criterion 6)."""
    def is_paused(self):
        raise RuntimeError("audit trail unreadable")

    def snapshot(self):
        raise RuntimeError("audit trail unreadable")


def _trader(ks_state, nav=23800.0):
    """A PaperTrader with BQ stubbed and the kill-switch state injected.

    Injection is the documented drill/test seam (see PaperTrader.__init__): the
    module singleton replays `pause` rows from the live audit trail, so a test
    reading it would depend on whether the real book happens to be paused today.
    """
    from backend.services.paper_trader import PaperTrader

    bq = MagicMock()
    bq.get_paper_portfolio.return_value = {
        "portfolio_id": "default", "total_nav": nav, "current_cash": nav,
        "starting_capital": 20000.0,
    }
    bq.get_paper_position.return_value = None
    # Real settings: the DOWNSTREAM buy body reads many more fields than the gate
    # does, and this helper is used by tests where the order must SUCCEED. The
    # gate's own tolerance of a minimal settings object is pinned separately by
    # test_..._gate_survives_the_smoketest_settings_shape below.
    from backend.config.settings import get_settings
    return PaperTrader(settings=get_settings(), bq_client=bq,
                       kill_switch_state=ks_state)


# ── criterion 1: the PAUSED bypass ──────────────────────────────────────────

def test_phase_36_13_a_paused_book_refuses_a_buy(caplog):
    """THE DEFECT. Pre-fix `execute_buy` had no kill-switch reference at all, so
    this returned a trade dict while the book was paused after a breach."""
    tr = _trader(_Paused())

    with caplog.at_level("ERROR"):
        result = tr.execute_buy(ticker="AMD", amount_usd=1000.0, price=100.0)

    assert result is None, (
        "a paused kill switch must refuse the order -- pre-fix this returned a trade"
    )
    assert tr.bq.insert_paper_trade.call_count == 0, (
        "REFUSAL MUST MEAN NO TRADE ROW: asserting only the return value would pass "
        "even if the order had been persisted and the dict merely dropped"
    )
    assert "REFUSING BUY" in caplog.text


def test_phase_36_13_the_refusal_is_observable_not_a_silent_return():
    """Criterion 3. A bare `return None` is indistinguishable from 'no signal
    today'. The refusal joins the existing `buy_rejections` accumulator that the
    cycle summary already consumes (autonomous_loop.py:1574-1582)."""
    tr = _trader(_Paused())
    tr.execute_buy(ticker="AMD", amount_usd=1000.0, price=100.0)

    assert len(tr.buy_rejections) == 1
    rej = tr.buy_rejections[0]
    assert rej["ticker"] == "AMD"
    assert rej["reason"] == "kill_switch_paused"
    assert "PAUSED" in rej["detail"]


# ── criterion 2: the DISARMED / baselines-lost bypass ───────────────────────

def test_phase_36_13_lost_baselines_refuse_a_buy():
    """Criterion 2: unrestorable baselines. The book is NOT paused here -- the
    refusal must come from the baselines being unknown, not from the pause.

    Drives the REAL predicate (`baselines_present_in`) against the injected
    state's own snapshot. An earlier version monkeypatched `evaluate_breach`,
    which the gate no longer calls -- an inert patch that would have let this pass
    no matter what the gate did.
    """
    tr = _trader(_LostBaselines())

    result = tr.execute_buy(ticker="AMD", amount_usd=1000.0, price=100.0)

    assert result is None
    assert tr.bq.insert_paper_trade.call_count == 0
    assert tr.buy_rejections[0]["reason"] == "kill_switch_baselines_lost"


def test_phase_36_13_a_STALE_anchor_does_NOT_refuse_a_buy():
    """THE 36.9 LESSON, pinned so it cannot be relearned. `evaluate_breach` reports
    `armed: False` for an anchor merely from yesterday. If this gate keyed on that,
    the book would refuse every BUY each morning until the daily roll -- the exact
    money-path regression 36.9's cycle-1 evaluator caught. Presence, not freshness.
    """
    import backend.services.kill_switch as ks

    st = _StaleButPresent()
    # The premise is real, not assumed: this snapshot IS `armed: False` today.
    assert ks._sod_date_is_stale(st.snapshot()["sod_date"],
                                 st.snapshot()["sod_nav"]) is True

    tr = _trader(st)
    result = tr.execute_buy(ticker="AMD", amount_usd=1000.0, price=100.0)

    assert result is not None, (
        "an overnight anchor is repaired by the next cycle's roll -- it is not a "
        "durable fault and must NOT stop the book trading"
    )
    assert tr.buy_rejections == []


# ── criterion 6: fail closed ────────────────────────────────────────────────

def test_phase_36_13_an_unreadable_kill_switch_blocks_the_order(caplog):
    """Criterion 6. An unknown state is not a healthy one."""
    tr = _trader(_Unreadable())

    with caplog.at_level("ERROR"):
        result = tr.execute_buy(ticker="AMD", amount_usd=1000.0, price=100.0)

    assert result is None
    assert tr.bq.insert_paper_trade.call_count == 0
    assert tr.buy_rejections[0]["reason"] == "kill_switch_unreadable"
    assert "not an open one" in caplog.text


# ── criterion 5: the drills still work ──────────────────────────────────────

def test_phase_36_13_an_injected_healthy_state_still_places_the_order(monkeypatch):
    """Criterion 5, the mechanism the drills rely on: injecting a healthy state
    lets the probe order through DETERMINISTICALLY, regardless of whether the real
    book is paused today."""
    import backend.services.kill_switch as ks
    monkeypatch.setattr(ks, "evaluate_breach", lambda **_k: {
        "any_breached": False, "armed": True, "baselines_present": True,
        "daily_baseline_missing": False, "trailing_baseline_missing": False})

    tr = _trader(_Healthy())
    assert tr.execute_buy(ticker="AMD", amount_usd=1000.0, price=100.0) is not None


def test_phase_36_13_injection_is_audible_at_construction(caplog):
    """The strongest objection to the injection seam is that it is a SILENT
    bypass, visible only at the construction site. It must at least be loud."""
    with caplog.at_level("WARNING"):
        _trader(_Healthy())
    assert "INJECTED" in caplog.text
    assert "Production callers must not pass kill_switch_state" in caplog.text


def test_phase_36_13_no_backend_module_injects_kill_switch_state():
    """Source scan: production code must not be able to bypass the gate. Only
    tests and scripts/ (drills, smoketests) may inject.

    This is the second half of the injection mitigation -- the WARN log makes a
    bypass audible at runtime, this makes it impossible to add under backend/
    without a test failing.
    """
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[2] / "backend"
    offenders = [
        f"{p.relative_to(root.parent)}:{i}"
        for p in root.rglob("*.py")
        if "/tests/" not in str(p)
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1)
        if "kill_switch_state=" in line and "def __init__" not in line
    ]
    assert offenders == [], (
        f"backend modules must not inject a kill-switch state: {offenders}"
    )


def test_phase_36_13_execute_sell_is_deliberately_not_gated():
    """The asymmetry is load-bearing: the switch flattens THROUGH execute_sell, so
    gating sells on `paused` would deadlock the pause. Pinned so a future reader
    does not 'complete' the mediation and wedge the flatten."""
    import inspect

    from backend.services.paper_trader import PaperTrader

    src = inspect.getsource(PaperTrader.execute_sell)
    assert "_kill_switch_refusal_for_buy" not in src, (
        "execute_sell must NOT carry the BUY gate -- the kill switch's own flatten "
        "runs through it, and gating it would prevent the switch from closing the "
        "positions it just decided to close"
    )


def test_phase_36_13_gate_survives_the_smoketest_settings_shape():
    """Criterion 5, the trap the research gate surfaced: the stage smoketest
    constructs a SimpleNamespace with only a handful of attributes as `settings`.
    The gate reads config with `getattr(..., default)` precisely so it cannot die
    with AttributeError on a caller this step is required to keep working.

    Scoped to the GATE, not the whole buy path -- the downstream body's own
    settings requirements are pre-existing and outside this step.
    """
    from backend.services.paper_trader import PaperTrader

    bq = MagicMock()
    bq.get_paper_portfolio.return_value = {"total_nav": 23800.0}
    minimal = SimpleNamespace(paper_starting_capital=20000.0)

    tr = PaperTrader(settings=minimal, bq_client=bq, kill_switch_state=_Paused())
    # Must return a refusal dict, NOT raise AttributeError.
    refusal = tr._kill_switch_refusal_for_buy()
    assert refusal["reason"] == "kill_switch_paused"


def test_phase_36_13_the_drills_inject_their_own_kill_switch_state():
    """Criterion 8's other half: 'removing the bypass must fail the drill test'.

    Found by mutation -- removing the drill's injection SURVIVED the first matrix,
    because nothing pinned it. Without injection these scripts read the module
    singleton, whose boot replay inherits a live `pause` row, so they would fail
    their probe order only on days the book happens to be paused: an intermittent,
    schedule-dependent break that a green suite would never show.

    Parsed with `ast`, not grepped: a string match would pass on the word appearing
    in a comment.
    """
    import ast
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[2]
    targets = [
        root / "scripts" / "go_live_drills" / "zero_orders_drill.py",
        root / "scripts" / "smoketest_stages_5_through_13.py",
    ]
    for path in targets:
        assert path.exists(), f"{path} moved -- update this guard, do not delete it"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        constructions = [
            n for n in ast.walk(tree)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name) and n.func.id == "PaperTrader"
        ]
        assert constructions, f"{path.name}: no PaperTrader(...) construction found"
        for call in constructions:
            kwargs = {k.arg for k in call.keywords}
            assert "kill_switch_state" in kwargs, (
                f"{path.name}:{call.lineno} constructs PaperTrader without "
                "kill_switch_state -- it will read the live singleton and refuse "
                "its probe order on any day the real book is paused"
            )


@pytest.mark.parametrize("script", [
    "scripts/go_live_drills/zero_orders_drill.py",
    "scripts/smoketest_stages_5_through_13.py",
])
def test_phase_36_13_drill_stub_state_actually_passes_the_gate(script):
    """The injected stub must satisfy the REAL gate, not merely exist.

    Found the hard way: the first version of these stubs returned only the pause
    fields, so `baselines_present_in()` read them as LOST HISTORY and both scripts
    refused their probe order -- the drill died with "execute_buy returned None"
    and smoketest Stage 8 flipped PASS->FAIL against its HEAD baseline. The AST
    guard above proves the stub is WIRED IN; this proves it actually WORKS.

    Runs the production predicate against the real stub object, imported from the
    script -- not a copy of either.
    """
    import importlib.util
    import pathlib

    from backend.services.kill_switch import baselines_present_in

    path = pathlib.Path(__file__).resolve().parents[2] / script
    spec = importlib.util.spec_from_file_location(f"_drill_{path.stem}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    stub = mod._DrillKillSwitchState()
    assert stub.is_paused() is False
    assert baselines_present_in(stub.snapshot()) is True, (
        f"{script}: the injected stub's snapshot does not satisfy the BUY gate, so "
        "the script's probe order will be refused as lost history"
    )
