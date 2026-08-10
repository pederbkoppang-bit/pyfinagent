"""phase-86.12 -- does the kill switch evaluate drawdown against a STALE nav?

This is an INVESTIGATION step. These tests exist to ANSWER the question with
executable evidence rather than with a reading of the code -- criterion 4 says
"Demonstrate with a test driving evaluate_breach through the real production
path, not by reasoning about the code."

THE ANSWER, established below and stated up front so a reader is not left to
infer it:

  * `current_nav` is a STORED BigQuery figure (`paper_portfolio.total_nav`) on
    every path that RESOLVES it -- four call sites in three functions, derived
    from the AST, not the five the research brief's prose claimed. None marks.
    Two further call sites take it as a caller-supplied parameter.
  * The daily-loss leg therefore measures a drawdown that is exactly as fresh as
    the last `mark_to_market()` write, and no fresher.
  * INSIDE A CYCLE that is fine, and the ordering is what makes it fine:
    `autonomous_loop` marks at Step 5 (`:1368`) and enforces at Step 5.5
    (`:1400`), 32 lines later. The leg CAN fire on a drawdown present at cycle
    time, and these tests demonstrate it firing.
  * BETWEEN CYCLES nothing evaluates at all, and the read-only badge endpoint
    serves the last cycle's NAV. That is a DISPLAY staleness, not a missed
    enforcement -- but the badge does not say so.
  * The real asymmetry: `evaluate_breach` checks whether the BASELINE is stale
    (`_sod_date_is_stale`) but validates `current_nav` only for `None`/`<= 0`.
    Nothing anywhere checks how old `current_nav` is. The research brief's
    JFE-2004 citation is the reason this direction matters: a ratio built from a
    nonsynchronous pair biases measured variance DOWNWARD, so the limit fires
    LATE rather than early.

NO THRESHOLD IS CHANGED HERE and no guard is weakened -- criterion 5 forbids it.
Contract: handoff/current/contract_86.12.md
Research: handoff/current/research_brief_86.12.md
"""
from __future__ import annotations

import ast
from datetime import datetime, timezone
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def _isolated_kill_switch_state(tmp_path, monkeypatch):
    """Never touch the operator's live journal or the real singleton.

    criterion 6 requires `handoff/kill_switch_audit.jsonl` byte-identical across
    the whole investigation. The phase-86.6 preventer would refuse a stray write
    anyway; this fixture means we never get that far.
    """
    import backend.services.kill_switch as ks

    monkeypatch.setattr(ks, "_AUDIT_PATH", tmp_path / "kill_switch_audit.jsonl")
    yield


# ── criterion 1: PROVENANCE, asserted rather than described ────────────────

#: DERIVED from the AST -- every function containing a call to
#: `evaluate_breach`, classified by whether it resolves `current_nav` from the
#: stored portfolio or receives it from its caller.
#:
#: MY FIRST VERSION OF THIS LIST WAS WRONG IN TWO ENTRIES because I transcribed
#: it from the research brief's prose instead of deriving it:
#: `_roll_sod_anchor_if_needed` does not exist under that name, and
#: `risk_server.kill_switch` does NOT read `total_nav` -- it takes
#: `current_nav: float | None = None` as a parameter. The brief said "five
#: producers"; four call sites in THREE functions actually read the stored
#: figure. Re-derive with the AST scan in the experiment results rather than
#: trusting either list.
STORED_NAV_PRODUCERS = [
    ("backend/api/paper_trading.py", "get_kill_switch_state", "badge endpoint"),
    ("backend/api/paper_trading.py", "resume_trading", "resume endpoint"),
    ("backend/services/paper_trader.py", "check_and_enforce_kill_switch",
     "Step 5.5 -- TWO calls: the pre-roll `pre` and the post-roll decision"),
]

#: These receive `current_nav` from their caller, so their freshness is the
#: CALLER's problem. Worth naming: `check_auto_resume` decides whether to
#: un-pause a halted book, and it re-evaluates the breach on whatever NAV it is
#: handed.
CALLER_SUPPLIED = [
    ("backend/agents/mcp_servers/risk_server.py", "kill_switch"),
    ("backend/services/kill_switch.py", "check_auto_resume"),
]


def test_evaluate_breach_NEVER_marks_to_market():
    """The load-bearing provenance fact. If `evaluate_breach` marked, the
    staleness question would not arise at all."""
    src = (REPO / "backend" / "services" / "kill_switch.py").read_text()
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "evaluate_breach")
    called = {
        (getattr(c.func, "attr", None) or getattr(c.func, "id", None))
        for c in ast.walk(fn) if isinstance(c, ast.Call)
    }
    assert "mark_to_market" not in called, (
        "evaluate_breach now marks to market -- the provenance recorded in "
        "phase-86.12 is stale and the investigation's conclusion must be re-run"
    )
    # ...and it takes the NAV as a parameter, i.e. the caller owns freshness.
    assert "current_nav" in [a.arg for a in fn.args.args]


def test_every_producer_reads_the_STORED_total_nav_not_a_live_mark():
    """Criterion 1 as an assertion. Each stored-NAV call site resolves
    `current_nav` from a portfolio dict, never from a pricing call."""
    misses = []
    for rel, func, _label in STORED_NAV_PRODUCERS:
        src = (REPO / rel).read_text()
        tree = ast.parse(src)
        fn = next((n for n in ast.walk(tree)
                   if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                   and n.name == func), None)
        if fn is None:
            misses.append(f"{rel}::{func} NOT FOUND -- provenance trace is stale")
            continue
        body = ast.unparse(fn)
        if "total_nav" not in body:
            misses.append(f"{rel}::{func} no longer reads total_nav")
        if "mark_to_market" in body:
            misses.append(f"{rel}::{func} now marks -- provenance changed")
    assert not misses, "; ".join(misses)


# ── criterion 4: CAN the daily-loss leg fire? Demonstrated. ────────────────


def _state_with(sod_nav, peak_nav, sod_date):
    """A DETACHED state object -- never the module singleton (phase-86.1)."""
    import backend.services.kill_switch as ks

    st = ks.KillSwitchState()
    st._sod_nav = sod_nav
    st._sod_date = sod_date
    st._peak_nav = peak_nav
    return st


def _today():
    return datetime.now(timezone.utc).date().isoformat()


@pytest.mark.parametrize(
    "drawdown_pct,expect_fire",
    # 4.0 is DELIBERATELY ABSENT from this list and gets its own test below.
    # Constructing a NAV exactly 4.0% down yields a raw ratio of
    # 3.9999999999999973, which DISPLAYS as 4.0 after `round(pct, 4)` and
    # compares FALSE against `>= 4.0`. That is a real display-vs-decision
    # disagreement at the boundary, not a test artifact, so it is reported as a
    # finding rather than smuggled into a parametrise list.
    [(0.0, False), (3.9, False), (4.01, True), (4.1, True), (12.0, True)],
)
def test_the_daily_loss_leg_DOES_fire_on_a_same_day_drawdown(
    drawdown_pct, expect_fire, monkeypatch
):
    """CRITERION 4, the affirmative half.

    Driven through the real `evaluate_breach` with a TODAY-dated anchor -- the
    state the book is in at Step 5.5, immediately after `mark_to_market` has
    refreshed `total_nav`. The leg fires at and above the 4.0% limit.
    """
    import backend.services.kill_switch as ks

    sod = 23833.94
    st = _state_with(sod_nav=sod, peak_nav=sod, sod_date=_today())
    monkeypatch.setattr(ks, "_state", st)

    current = sod * (1.0 - drawdown_pct / 100.0)
    breach = ks.evaluate_breach(current_nav=current,
                                daily_loss_limit_pct=4.0,
                                trailing_dd_limit_pct=10.0)

    assert breach["armed"] is True, "a today-dated anchor must be armed"
    assert breach["daily_loss_breached"] is expect_fire, (
        f"drawdown {drawdown_pct}% -> daily_loss_pct "
        f"{breach['daily_loss_pct']}, breached={breach['daily_loss_breached']}, "
        f"expected {expect_fire}"
    )
    assert breach["daily_loss_pct"] == pytest.approx(drawdown_pct, abs=1e-6)


def test_the_leg_CANNOT_fire_while_the_anchor_is_from_a_PREVIOUS_day(monkeypatch):
    """CRITERION 4, the limiting half -- and this is the live state right now.

    With `sod_date` older than today the daily leg is deliberately unevaluable
    (phase-36.9). Measured on the live book during this investigation:
    `daily_baseline_stale: true`, `armed: false`, `daily_loss_pct: 0.0`, with
    `sod_date=2026-08-09` while the clock had rolled to 2026-08-10.

    So a drawdown arriving before the first cycle of a new UTC day is NOT
    caught by the daily leg. That is by design -- a stale anchor would report a
    multi-day move as a same-day loss -- but it means the daily leg has a blind
    window every day between UTC midnight and the first cycle.
    """
    import backend.services.kill_switch as ks

    sod = 23833.94
    st = _state_with(sod_nav=sod, peak_nav=sod, sod_date="2020-01-01")
    monkeypatch.setattr(ks, "_state", st)

    breach = ks.evaluate_breach(current_nav=sod * 0.80,   # a 20% collapse
                                daily_loss_limit_pct=4.0,
                                trailing_dd_limit_pct=10.0)

    assert breach["daily_baseline_stale"] is True
    assert breach["armed"] is False
    assert breach["daily_loss_breached"] is False, (
        "the daily leg fired on a stale anchor -- phase-36.9's protection is gone"
    )
    assert breach["daily_loss_pct"] == 0.0
    # The trailing leg is NOT date-scoped and MUST still fire -- it is the only
    # thing covering that window.
    assert breach["trailing_dd_breached"] is True, (
        "with the daily leg unevaluable, the trailing leg is the sole remaining "
        "cover; if it cannot fire either, the book is unprotected in that window"
    )


def test_the_production_ENFORCEMENT_path_reads_the_stored_nav(monkeypatch, tmp_path):
    """CRITERION 4, through `check_and_enforce_kill_switch` ITSELF.

    THE CYCLE-2 Q/A KILLED THE PREVIOUS VERSION OF THIS TEST AND WAS RIGHT.
    It built a `PaperTrader`, monkeypatched `get_or_create_portfolio`, then used
    NEITHER -- it re-implemented the `total_nav` resolution in the test body and
    called `evaluate_breach` directly. Replacing
    `PaperTrader.check_and_enforce_kill_switch` with a raiser left it PASSING,
    while its docstring claimed it drove that very method. Vacuity shape #7.

    This version CALLS the real method. `flatten_all` is stubbed (a test must
    not place orders) and asserting it was invoked is what proves the breach
    reached the money path rather than merely being computed.
    """
    import backend.services.kill_switch as ks
    from backend.services.paper_trader import PaperTrader
    from backend.config.settings import get_settings

    sod = 20000.0
    st = _state_with(sod_nav=sod, peak_nav=sod, sod_date=_today())
    monkeypatch.setattr(ks, "_state", st)

    trader = PaperTrader.__new__(PaperTrader)
    trader.settings = get_settings()

    # A portfolio whose STORED nav is 6% down -- what mark_to_market would have
    # written at Step 5 had prices fallen 6% since the anchor.
    stored = {"total_nav": sod * 0.94, "starting_capital": sod,
              "current_cash": 0.0, "portfolio_id": "default"}
    monkeypatch.setattr(trader, "get_or_create_portfolio", lambda: stored)

    flattened = []
    monkeypatch.setattr(trader, "flatten_all",
                        lambda reason=None: flattened.append(reason) or {"sold": 0})
    monkeypatch.setattr(trader, "_sod_anchor_provisional", False, raising=False)

    result = trader.check_and_enforce_kill_switch()

    assert result.get("triggered") is True, (
        f"the real enforcement path did not trigger on a 6% stored drawdown: "
        f"{result}"
    )
    assert flattened == ["kill_switch_auto_flatten"], (
        f"the breach was computed but never reached flatten_all: {flattened}"
    )
    assert result["breach"]["daily_loss_pct"] == pytest.approx(6.0, abs=1e-6)


# ── the asymmetry: baseline freshness IS checked, NAV freshness is NOT ─────


def test_evaluate_breach_checks_BASELINE_staleness_but_not_NAV_staleness():
    """The finding, as an executable statement rather than prose.

    `_sod_date_is_stale` gates the daily leg on the BASELINE's age. There is no
    equivalent for `current_nav`: it is validated for None and <= 0 and nothing
    else, so a NAV of any age is accepted without complaint.
    """
    src = (REPO / "backend" / "services" / "kill_switch.py").read_text()
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "evaluate_breach")
    body = ast.unparse(fn)

    assert "_sod_date_is_stale" in body, "baseline staleness check is gone"
    assert "current_nav is None or current_nav <= 0" in body, (
        "the current_nav validity check moved -- re-derive this finding"
    )
    # There is no asof/age/updated_at consideration anywhere in the function.
    for token in ("updated_at", "asof", "as_of", "nav_age", "stale_nav"):
        assert token not in body, (
            f"evaluate_breach now considers {token!r} -- the phase-86.12 finding "
            f"(no freshness check on current_nav) is out of date"
        )


def test_the_asof_IS_available_and_is_discarded_by_every_caller():
    """Why the fix is cheap, recorded so the follow-up step does not re-derive
    it: `mark_to_market` already stamps `updated_at`, `get_paper_portfolio` is
    a `SELECT *`, so the timestamp is sitting in the very dict each caller
    reads `total_nav` out of -- and all of them drop it."""
    pt = (REPO / "backend" / "services" / "paper_trader.py").read_text()
    assert '"updated_at": datetime.now(timezone.utc).isoformat()' in pt, (
        "mark_to_market no longer stamps updated_at -- the cheap-fix note is stale"
    )
    for rel, func, _ in STORED_NAV_PRODUCERS:
        src = (REPO / rel).read_text()
        tree = ast.parse(src)
        fn = next((n for n in ast.walk(tree)
                   if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                   and n.name == func), None)
        if fn is None:
            continue
        body = ast.unparse(fn)
        if "total_nav" in body:
            assert "updated_at" not in body, (
                f"{rel}::{func} now reads updated_at -- a freshness check may "
                f"have landed; re-run the phase-86.12 investigation"
            )


def test_AT_the_limit_the_displayed_pct_and_the_DECISION_disagree(monkeypatch):
    """A finding this investigation produced by accident, and it is worth
    recording precisely rather than rounding away.

    A NAV constructed to be exactly 4.0% below the anchor yields a raw ratio of
    3.9999999999999973. The payload reports `round(pct, 4)` == **4.0**, and the
    decision compares the RAW value against `>= 4.0`, which is **False**. So the
    badge can read "4.0%" beside a limit of "4.0%" and correctly not be
    breached.

    This is not a bug in the arithmetic -- it is the ordinary consequence of
    binary floating point, and any threshold comparison has some such boundary.
    It IS a display-vs-decision disagreement, and an operator staring at two
    equal numbers and an un-tripped switch deserves to have it written down.
    NOT FIXED HERE: criterion 5 forbids touching a threshold during the
    diagnosis, and changing either the rounding or the comparison IS a threshold
    change.
    """
    import backend.services.kill_switch as ks

    sod = 23833.94
    st = _state_with(sod_nav=sod, peak_nav=sod, sod_date=_today())
    monkeypatch.setattr(ks, "_state", st)

    current = sod * (1.0 - 4.0 / 100.0)
    raw = (sod - current) / sod * 100.0
    breach = ks.evaluate_breach(current_nav=current, daily_loss_limit_pct=4.0,
                                trailing_dd_limit_pct=10.0)

    assert raw < 4.0, f"expected the raw ratio just below the limit, got {raw!r}"
    assert breach["daily_loss_pct"] == 4.0, "the DISPLAYED value rounds to the limit"
    assert breach["daily_loss_breached"] is False, (
        "the decision uses the raw value; if this ever becomes True the "
        "rounding or the comparison changed and the finding must be re-derived"
    )
    # One tick further down and it fires -- so the leg is not broken, only
    # knife-edge at exactly the boundary.
    fires = ks.evaluate_breach(current_nav=sod * (1.0 - 4.01 / 100.0),
                               daily_loss_limit_pct=4.0, trailing_dd_limit_pct=10.0)
    assert fires["daily_loss_breached"] is True
