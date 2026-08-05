"""phase-82.10 -- the data-freshness alarm must page without a browser.

WHY THESE GUARDS CAN FAIL (the anti-vacuity argument, which matters here more
than usual). Criteria 2 and 3 are ALREADY satisfied at the `compute_freshness`
level by `tests/verify_phase_25_A7.py` claims 8 and 9. A new test that merely
re-asserted "red fixture -> alert / green fixture -> no alert" against
`compute_freshness` would have passed unchanged on the PRE-FIX tree, i.e. it
could not observe this step's defect at all.

So every guard in this file drives `backend.services.freshness_cron`, which did
not exist before this step: the whole module fails at import against the
pre-fix tree. Criterion 1's guard goes further and executes the callable that
the STUB SCHEDULER actually received, so it proves the scheduler -> evaluator ->
compute_freshness path rather than asserting that a function exists.

Patch-target note (this bit is load-bearing): `_fire_freshness_alarm` imports
`raise_cron_alert_sync` FUNCTION-LOCALLY (backend/services/cycle_health.py), so
patching `backend.services.cycle_health.raise_cron_alert_sync` patches nothing
and would make the criterion-3 guard pass vacuously. All patches target
`backend.services.observability.alerting.raise_cron_alert_sync`, and
`test_wrong_patch_target_does_not_exist` pins that fact.
"""
from __future__ import annotations

import inspect
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from backend.services import freshness_cron
from backend.services.freshness_cron import (
    DEFAULT_INTERVAL_HOURS,
    JOB_ID,
    register_freshness_cron,
    reset_transition_state,
    run_freshness_check,
)

ALERT_TARGET = "backend.services.observability.alerting.raise_cron_alert_sync"

# `_TABLE_MAX_AGE_SEC` SLAs, for building fixtures that really breach.
# historical_macro = 3_024_000s (35d); red requires ratio > CRITICAL_RATIO (2.0).
_MACRO_RED_AGE = 3_024_000.0 * 3.66  # the ratio actually measured in prod
_ALL_GREEN = {
    "paper_trades": 10.0,
    "paper_portfolio_snapshots": 100.0,
    "historical_prices": 1_000.0,
    "historical_fundamentals": 100_000.0,
    "historical_macro": 50_000.0,
    "signals_log": 60.0,
}


class _StubScheduler:
    """Shape copied from backend/tests/test_phase_82_0_macro_ingestion.py."""

    def __init__(self) -> None:
        self.jobs: list[dict] = []

    def add_job(self, func, **kwargs):
        self.jobs.append({"func": func, **kwargs})
        return MagicMock()


def _fake_bq(ages: dict) -> MagicMock:
    """Fake BQ that routes MAX(age) per table (verify_phase_25_A7.py shape)."""
    fake = MagicMock()
    fake._pt_table = lambda name: f"test-proj.test_ds.{name}"

    def _query(sql, *args, **kwargs):
        for table_name, age_val in ages.items():
            if f".{table_name}`" in sql:
                row = MagicMock()
                row.get = (
                    lambda key, default=None, _a=age_val: _a
                    if key == "age"
                    else default
                )
                rs = MagicMock()
                rs.result.return_value = iter([row])
                return rs
        rs = MagicMock()
        rs.result.return_value = iter([])
        return rs

    fake.client.query.side_effect = _query
    return fake


@pytest.fixture(autouse=True)
def _clean_transition_state():
    """The transition gate is module state. Without this reset, the green test
    could pass because the gate suppressed the alert rather than because the
    fixture was healthy -- passing for the wrong reason."""
    reset_transition_state()
    yield
    reset_transition_state()


# --------------------------------------------------------------------------
# Criterion 1: a scheduled evaluator invokes compute_freshness without any
# HTTP request, asserted by a test against a stub scheduler.
# --------------------------------------------------------------------------

def test_registers_exactly_one_job_on_a_stub_scheduler():
    stub = _StubScheduler()
    returned = register_freshness_cron(stub)

    assert returned == JOB_ID
    assert len(stub.jobs) == 1, f"expected 1 job, got {stub.jobs!r}"
    job = stub.jobs[0]
    assert job["id"] == JOB_ID
    assert job["replace_existing"] is True, (
        "replace_existing=True is mandatory per the APScheduler userguide -- "
        "without it a restart duplicates the job"
    )
    assert job["func"] is run_freshness_check
    # RECURRENCE. Cycle-1 Q/A killed the earlier version of this test: it
    # pinned identity (id/func/replace_existing) but not the kwargs that decide
    # whether the job ever runs AGAIN, so trigger="date" (fires once at
    # startup, then gone) and hours=99999 (~11 years) both shipped green --
    # each re-creating the browser-blind blind spot 82.10 exists to close.
    # The sibling crons already assert this (test_phase_82_0_macro_ingestion.py
    # asserts trigger == "cron"; test_phase_71_6_self_audit_cron.py also
    # asserts day_of_week), so this test had fallen below repo convention.
    assert job["trigger"] == "interval", (
        "a non-interval trigger does not recur; 'date' fires once at startup "
        "and the evaluator is then dead for the life of the process"
    )
    assert job["hours"] == DEFAULT_INTERVAL_HOURS


def test_registered_cadence_is_tight_enough_to_catch_the_tightest_sla():
    """Semantic cadence guard -- deliberately NOT a magic-number pin.

    Derives the bound from the data instead of restating `6`: the dbt
    source-freshness rule is to check at >= 2x the tightest SLA, and the
    tightest SLA is read live from `cycle_health._TABLE_MAX_AGE_SEC`. This
    kills an absurd interval (the cycle-1 Q/A's surviving hours=99999 mutant)
    for the RIGHT reason, and stays correct if someone later tunes 6 -> 8.
    """
    from backend.services.cycle_health import _TABLE_MAX_AGE_SEC

    tightest_sla_hours = min(_TABLE_MAX_AGE_SEC.values()) / 3600.0
    max_allowed = tightest_sla_hours / 2.0

    stub = _StubScheduler()
    register_freshness_cron(stub)
    hours = stub.jobs[0]["hours"]

    assert 0 < hours <= max_allowed, (
        f"cadence {hours}h must be >0 and at most {max_allowed:.1f}h "
        f"(half the tightest SLA, {tightest_sla_hours:.1f}h) or a source can "
        "breach and stay unnoticed for longer than its own SLA"
    )


def test_the_hours_parameter_is_actually_forwarded():
    """A knob that is accepted and ignored is worse than no knob."""
    stub = _StubScheduler()
    register_freshness_cron(stub, hours=3)
    assert stub.jobs[0]["hours"] == 3


def test_registration_kwargs_are_pinned_exactly():
    """Pins the FULL kwarg surface, not a chosen subset.

    This is the class-level guard behind the cycle-1 finding: asserting a
    hand-picked subset is what let two behaviour-changing kwargs go
    unobserved. If a future edit adds, drops or renames a kwarg on this
    registration, this test fails and forces a decision.
    """
    stub = _StubScheduler()
    register_freshness_cron(stub)
    keys = set(stub.jobs[0]) - {"func"}
    assert keys == {"trigger", "id", "replace_existing", "hours", "name"}, (
        f"registration kwarg surface changed: {sorted(keys)}"
    )


def test_the_job_the_scheduler_received_actually_reaches_compute_freshness():
    """THE criterion-1 guard. Not 'a function exists' -- we pull the callable
    the stub scheduler was handed and execute it, proving the whole
    scheduler -> evaluator -> compute_freshness path with no HTTP layer."""
    stub = _StubScheduler()
    register_freshness_cron(stub)
    scheduled_callable = stub.jobs[0]["func"]

    seen = {}

    def _spy(bq, cycle_interval_sec, *, emit_alarm=True):
        seen["called"] = True
        seen["cycle_interval_sec"] = cycle_interval_sec
        seen["emit_alarm"] = emit_alarm
        return {"sources": {}, "overall_band": "green"}

    with patch("backend.services.cycle_health.compute_freshness", _spy):
        with patch(ALERT_TARGET) as mock_alert:
            out = scheduled_callable(bq=_fake_bq(_ALL_GREEN), settings=object())

    assert seen.get("called") is True, (
        "the callable handed to the scheduler never reached compute_freshness"
    )
    assert out["ok"] is True
    assert mock_alert.call_count == 0
    # Same interval expression as the three HTTP call sites, so the pager and
    # the dashboard cannot disagree about a band.
    assert seen["cycle_interval_sec"] == 86_400.0


def test_evaluator_has_no_web_dependency():
    """'without any HTTP request' -- the evaluator module must not pull in the
    web stack at all, so the path cannot secretly route through a handler."""
    src = Path(freshness_cron.__file__).read_text(encoding="utf-8")
    for banned in ("fastapi", "starlette", "TestClient", "httpx", "requests"):
        assert banned not in src, (
            f"{banned!r} appears in freshness_cron.py; the scheduled evaluator "
            "must reach compute_freshness directly, not via HTTP"
        )


def test_main_wires_the_cron_at_startup():
    """The module existing is not enough -- main.py must actually CALL it.

    Deliberately an AST check, not a substring scan. A substring scan for
    "register_freshness_cron" is satisfied by the *import* line alone, so
    deleting the call site leaves it green -- measured: that exact mutant
    (M4) SURVIVED the first version of this guard. Requiring an ast.Call node
    is what makes the guard able to observe a dead registration.
    """
    import ast

    src = (Path(__file__).resolve().parents[1] / "main.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(src)
    calls = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and (
            (isinstance(n.func, ast.Name) and n.func.id == "register_freshness_cron")
            or (
                isinstance(n.func, ast.Attribute)
                and n.func.attr == "register_freshness_cron"
            )
        )
    ]
    assert calls, (
        "backend/main.py never CALLS register_freshness_cron (importing it is "
        "not enough); the job would never run and the alarm stays browser-only"
    )

    # ARGUMENTS, not just existence. Cycle-2 Q/A killed the previous version:
    # it asserted only that an ast.Call node exists, so
    # register_freshness_cron(scheduler, hours=99999), hours=0 and
    # replace_existing=False ALL shipped green -- every other test in this file
    # calls register_freshness_cron with DEFAULTS, so nothing constrained what
    # main.py actually passes. A 99999-hour override re-creates the
    # browser-blind blind spot with a green suite.
    from backend.services.cycle_health import _TABLE_MAX_AGE_SEC

    max_allowed = (min(_TABLE_MAX_AGE_SEC.values()) / 3600.0) / 2.0
    for call in calls:
        # Cycle-3 Q/A defeated the first version of this loop three ways, all of
        # which slipped past a keywords-only check:
        #   `**{"hours": 99999}`  -- kw.arg is None for a ** expansion, so the
        #                            band check never ran and the job shipped at
        #                            11.4 years;
        #   a positional 99999    -- hours is keyword-only, so this raises
        #                            TypeError, which main.py's fail-open except
        #                            swallows: ZERO jobs registered, silently;
        #   `register_freshness_cron(None)` -- the helper's own fail-open, again
        #                            zero jobs and no raise.
        # A fail-open registration that registers nothing is exactly the
        # browser-blind blind spot this step exists to close, so the call SHAPE
        # must be pinned, not just its keyword values.
        assert not any(kw.arg is None for kw in call.keywords), (
            "main.py passes **kwargs to register_freshness_cron; the value "
            "cannot be checked statically, so an absurd cadence could ship "
            "unseen -- pass explicit literals instead"
        )
        assert len(call.args) == 1, (
            f"main.py passes {len(call.args)} positional args to "
            "register_freshness_cron (expected exactly 1, the scheduler); "
            "extra positionals raise TypeError, which the surrounding "
            "fail-open except swallows into ZERO registered jobs"
        )
        assert isinstance(call.args[0], ast.Name) and call.args[0].id == "scheduler", (
            "main.py must register on the `scheduler` object; a literal or a "
            "different object silently yields no live job"
        )
        overrides = {kw.arg: kw.value for kw in call.keywords if kw.arg}
        if "hours" in overrides:
            node = overrides["hours"]
            assert isinstance(node, ast.Constant) and isinstance(
                node.value, (int, float)
            ), "main.py overrides hours with a non-literal; pin it explicitly here"
            assert 0 < node.value <= max_allowed, (
                f"main.py passes hours={node.value}, outside the sane band "
                f"(0, {max_allowed:.1f}] derived from the tightest SLA"
            )
        if "replace_existing" in overrides:
            node = overrides["replace_existing"]
            assert isinstance(node, ast.Constant) and node.value is True, (
                "main.py passes replace_existing=False; a restart would then "
                "duplicate the job instead of replacing it"
            )
        # Any OTHER keyword is an unreviewed override of an audited default.
        unknown = set(overrides) - {"hours", "replace_existing"}
        assert not unknown, (
            f"main.py passes unreviewed kwargs {sorted(unknown)} to "
            "register_freshness_cron; add them to this guard deliberately"
        )


# --------------------------------------------------------------------------
# Criterion 2: a fixture in which a source breaches its critical threshold
# produces an outbound alert through the operator notification path.
# --------------------------------------------------------------------------

def test_breaching_source_pages_through_the_real_notification_path():
    """No `notify=` injection: the evaluator resolves `raise_cron_alert_sync`
    itself, so this drives the operator channel the production job uses."""
    ages = dict(_ALL_GREEN, historical_macro=_MACRO_RED_AGE)

    with patch(ALERT_TARGET) as mock_alert:
        out = run_freshness_check(bq=_fake_bq(ages), settings=object())

    # Precondition assertion: prove the fixture actually breached. Without
    # this, a fixture that failed to go red would make the whole guard
    # meaningless in exactly the way it is meant to detect.
    assert out["overall_band"] == "red", (
        f"fixture did not breach: overall_band={out['overall_band']!r}"
    )
    assert "historical_macro" in out["red_sources"]

    assert mock_alert.call_count >= 1, "a red source emitted no alert"
    kwargs = mock_alert.call_args.kwargs
    assert kwargs["severity"] == "P1", (
        "severity must stay P1: with slack_webhook_url empty, only critical "
        "severities reach the bot-token fallback -- a P2 is logged and dropped"
    )
    assert kwargs["details"]["table"] == "historical_macro"
    assert kwargs["source"] == "cycle_health"
    assert kwargs["error_type"] == "freshness_critical_historical_macro"


def test_steady_state_red_does_not_re_page_and_the_inner_emitter_is_suppressed():
    """Two ticks against the SAME permanently-red table must page exactly once.

    This kills two distinct mutants at once:
      * dropping the transition gate  -> second tick pages again;
      * dropping `emit_alarm=False`   -> the level-triggered path inside
        compute_freshness fires on EVERY tick, so tick 1 emits 2 and tick 2
        emits 1.
    Both matter: AlertDeduper does NOT suppress steady state (a P1 re-fires
    every alert_repeat_hours forever), which is ~512 pages over a 128-day
    outage.
    """
    ages = dict(_ALL_GREEN, historical_macro=_MACRO_RED_AGE)
    bq = _fake_bq(ages)

    with patch(ALERT_TARGET) as mock_alert:
        first = run_freshness_check(bq=bq, settings=object())
        first_calls = mock_alert.call_count
        second = run_freshness_check(bq=bq, settings=object())
        second_calls = mock_alert.call_count - first_calls

    assert first["overall_band"] == "red", "fixture did not breach"
    assert first_calls == 1, (
        f"first tick emitted {first_calls} alerts, expected exactly 1 "
        "(2 means emit_alarm=False was dropped and the inner level-triggered "
        "path fired too)"
    )
    assert second_calls == 0, (
        f"second tick emitted {second_calls} alerts for an unchanged red "
        "source -- the state-transition gate is not holding, which is the "
        "page-storm defect this step exists to prevent"
    )
    assert second["newly_red"] == []


def test_a_newly_red_source_pages_even_when_another_is_already_red():
    """The gate must key on the SET of red sources, not a single boolean --
    otherwise a second table going dark while the first is still red is
    silently swallowed."""
    red_prices = 93_600.0 * 4
    with patch(ALERT_TARGET) as mock_alert:
        run_freshness_check(
            bq=_fake_bq(dict(_ALL_GREEN, historical_macro=_MACRO_RED_AGE)),
            settings=object(),
        )
        before = mock_alert.call_count
        out = run_freshness_check(
            bq=_fake_bq(
                dict(
                    _ALL_GREEN,
                    historical_macro=_MACRO_RED_AGE,
                    historical_prices=red_prices,
                )
            ),
            settings=object(),
        )
        new_calls = mock_alert.call_args_list[before:]

    assert out["newly_red"] == ["historical_prices"]
    assert len(new_calls) == 1
    assert new_calls[0].kwargs["details"]["table"] == "historical_prices"


# --------------------------------------------------------------------------
# Criterion 3: a fixture in which all sources are healthy produces NO alert.
# --------------------------------------------------------------------------

def test_all_healthy_emits_no_alert():
    with patch(ALERT_TARGET) as mock_alert:
        out = run_freshness_check(bq=_fake_bq(_ALL_GREEN), settings=object())

    # Precondition assertion, so a zero cannot come from a broken fixture
    # (e.g. a BQ fake whose queries all raise, yielding unknown bands).
    assert out["overall_band"] != "red", (
        f"all-green fixture computed overall_band={out['overall_band']!r}"
    )
    assert out["red_sources"] == []
    assert mock_alert.call_count == 0, (
        f"healthy fixture emitted {mock_alert.call_count} alerts -- the guard "
        "would pass by always firing"
    )


def test_first_run_after_restart_pages_a_red_source():
    """`_last_red_sources is None` means no baseline, not 'nothing is new'.
    On a fresh process the operator SHOULD be told a table is red; this pins
    that choice so it is visible rather than incidental."""
    assert freshness_cron._last_red_sources is None
    ages = dict(_ALL_GREEN, historical_macro=_MACRO_RED_AGE)
    with patch(ALERT_TARGET) as mock_alert:
        out = run_freshness_check(bq=_fake_bq(ages), settings=object())
    assert out["newly_red"] == ["historical_macro"]
    assert mock_alert.call_count == 1


# --------------------------------------------------------------------------
# Anti-vacuity / regression pins
# --------------------------------------------------------------------------

def test_wrong_patch_target_does_not_exist():
    """Pins the trap: `raise_cron_alert_sync` is imported function-locally in
    cycle_health, so patching it at module scope silently patches nothing and
    the criterion-3 guard would pass for free. If a refactor ever hoists that
    import, this test fails and forces this file to be revisited."""
    from backend.services import cycle_health

    assert not hasattr(cycle_health, "raise_cron_alert_sync"), (
        "cycle_health now binds raise_cron_alert_sync at module scope; the "
        "patch targets in this file must be re-reviewed"
    )


def test_compute_freshness_emit_alarm_defaults_true_and_is_keyword_only():
    """The three HTTP call sites pass positionally and must keep paging."""
    from backend.services.cycle_health import compute_freshness

    sig = inspect.signature(compute_freshness)
    param = sig.parameters["emit_alarm"]
    assert param.default is True
    assert param.kind is inspect.Parameter.KEYWORD_ONLY


def test_http_call_sites_were_not_edited_to_pass_emit_alarm():
    """Scope honesty: this step must not change dashboard behaviour."""
    root = Path(__file__).resolve().parents[1]
    for rel in ("api/paper_trading.py", "api/observability_api.py"):
        src = (root / rel).read_text(encoding="utf-8")
        assert "emit_alarm" not in src, (
            f"{rel} passes emit_alarm; the HTTP paths must stay unchanged"
        )


def test_run_freshness_check_is_fail_open():
    """A scheduler job that raises would be retried into the void and could
    disturb neighbouring jobs; this must degrade, not explode."""
    boom = MagicMock()
    boom._pt_table.side_effect = RuntimeError("simulated BQ outage")
    with patch(
        "backend.services.cycle_health.compute_freshness",
        side_effect=RuntimeError("simulated outage"),
    ):
        out = run_freshness_check(bq=boom, settings=object())
    assert out["ok"] is False
    assert "simulated outage" in out["error"]
