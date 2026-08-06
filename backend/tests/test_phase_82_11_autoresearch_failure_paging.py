"""phase-82.11 -- a nightly autoresearch failure must be audible and escalate.

WHY THESE GUARDS CAN FAIL (the anti-vacuity argument).

1. The criterion-1 and criterion-3 guards drive `run_memo._main_async` -- the
   REAL production entry point that launchd invokes -- with only
   `run_research` patched (to raise / to return a body) and `MEMO_DIR`
   redirected at a tmp dir. They do NOT inject `notify=`. So production has to
   resolve `raise_cron_alert_sync` itself; if the wiring in `_main_async` is
   deleted, or the severity is downgraded to a P2 (which
   `alerting.py:219-224` logs and drops), these fail. Against the pre-fix tree
   `backend.services.autoresearch_health` does not exist at all, so the whole
   file fails at import -- these guards cannot have passed before this step.

2. PATCH TARGET (load-bearing). `report_run_outcome` imports
   `raise_cron_alert_sync` FUNCTION-LOCALLY, so patching
   `backend.services.autoresearch_health.raise_cron_alert_sync` would patch
   NOTHING and every alert assertion would pass vacuously. All patches target
   `backend.services.observability.alerting.raise_cron_alert_sync`, and
   `test_wrong_patch_target_does_not_exist` pins that fact. Same trap 82.10
   pinned for `cycle_health`.

3. Every fixture carries a PRECONDITION assertion -- the failing fixtures
   assert their consecutive count really is what the tier ladder needs, and the
   healthy fixture asserts it really did have prior failures on disk. Without
   those, a "no alert" pass could come from a broken fixture rather than from
   the code under test.

4. `autospec=True` on every patch, so a renamed kwarg on
   `raise_cron_alert_sync` fails the test instead of being silently recorded.
"""
from __future__ import annotations

import asyncio
import datetime as dt
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from backend.services import autoresearch_health as ah
from backend.services.autoresearch_health import (
    ALERT_SOURCE,
    CLASS_CREDIT,
    CLASS_GENERIC,
    ERROR_TYPE_TIER1,
    ERROR_TYPE_TIER2,
    ESCALATE_AFTER_N,
    PAGE_AFTER_N,
    REMIND_EVERY_DAYS,
    classify_failure,
    count_consecutive_failures,
    report_run_outcome,
)

REPO = Path(__file__).resolve().parents[2]
RUN_MEMO = REPO / "scripts" / "autoresearch" / "run_memo.py"
CONTRACT = REPO / "handoff" / "current" / "contract_82.11.md"
NIGHTLY = REPO / "scripts" / "autoresearch" / "run_nightly.sh"

#: The ONLY correct patch target -- see the module docstring, point 2.
ALERT_TARGET = "backend.services.observability.alerting.raise_cron_alert_sync"

TODAY = dt.date(2026, 8, 6)

#: The live credit-exhaustion message, copied verbatim from
#: handoff/autoresearch/2026-08-06-ERROR-topic08.md.
CREDIT_MSG = (
    "Error code: 400 - {'type': 'error', 'error': {'type': "
    "'invalid_request_error', 'message': 'Your credit balance is too low to "
    "access the Anthropic API. Please go to Plans & Billing to upgrade or "
    "purchase credits.'}, 'request_id': 'req_011CdkWbkPNL9cW6qaJCfh3G'}"
)


class _BadRequestError(Exception):
    """Stand-in for anthropic.BadRequestError (no SDK import needed)."""


def _write_error_days(memo_dir: Path, n: int, *, end: dt.date) -> None:
    """Write `n` consecutive `*-ERROR-*` memos ending at `end` (inclusive)."""
    memo_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        d = end - dt.timedelta(days=i)
        (memo_dir / f"{d.isoformat()}-ERROR-topic{i % 14:02d}.md").write_text(
            f"# Autoresearch FAILED -- {d.isoformat()}\n\nError: {CREDIT_MSG}\n",
            encoding="utf-8",
        )


def _write_success_day(memo_dir: Path, d: dt.date) -> None:
    memo_dir.mkdir(parents=True, exist_ok=True)
    (memo_dir / f"{d.isoformat()}-topic03-some-research-question.md").write_text(
        f"# Autoresearch memo -- {d.isoformat()}\n\nbody\n", encoding="utf-8"
    )


def _load_run_memo():
    """Import scripts/autoresearch/run_memo.py as a module.

    It is a script, not a package member, so it needs an explicit loader. This
    is what lets the guards drive the PRODUCTION entry point rather than a
    reimplementation of it.
    """
    spec = importlib.util.spec_from_file_location("_rm_82_11", RUN_MEMO)
    assert spec and spec.loader, f"cannot load {RUN_MEMO}"
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_rm_82_11"] = mod
    spec.loader.exec_module(mod)
    return mod


def _drive_main_async(memo_dir: Path, *, raises: BaseException | None):
    """Run the production `_main_async` with `run_research` stubbed.

    Everything else -- the ERROR-file write, the `_report_outcome` call, the
    function-local resolution of the real emitter -- is production code.
    """
    mod = _load_run_memo()
    mod.MEMO_DIR = memo_dir

    async def _fake_research(topic: str) -> str:
        if raises is not None:
            raise raises
        return "a real research body"

    with patch.object(mod, "run_research", _fake_research):
        args = SimpleNamespace(topic="a fixed test topic", topic_index=None)
        return asyncio.run(mod._main_async(args))


# --------------------------------------------------------------------------
# Criterion 1 -- a credit-exhaustion failure emits a captured operator alert
# --------------------------------------------------------------------------


def test_credit_exhaustion_emits_operator_alert(tmp_path):
    """CRITERION 1. Drives the production `_main_async`; no notify= injection."""
    memo_dir = tmp_path / "autoresearch"
    memo_dir.mkdir()

    with patch(ALERT_TARGET, autospec=True) as mock_alert:
        rc = _drive_main_async(memo_dir, raises=_BadRequestError(CREDIT_MSG))

    assert rc == 1, "a failed run must still exit non-zero"

    # Precondition: the production path really did write tonight's ERROR file,
    # so a count of 1 below is real and not an artefact of an empty dir.
    errors = list(memo_dir.glob("*-ERROR-*.md"))
    assert len(errors) == 1, f"expected the production ERROR file, got {errors}"

    assert mock_alert.call_count == 1, (
        f"exactly one operator alert expected, got {mock_alert.call_count}"
    )
    kwargs = mock_alert.call_args.kwargs
    # Severity must be P0/P1: alerting.py:219-224 LOGS AND DROPS a P2 while
    # slack_webhook_url is empty, so a P2 would not reach the operator.
    assert kwargs["severity"] in ("P0", "P1"), kwargs
    assert kwargs["source"] == ALERT_SOURCE, kwargs
    assert kwargs["error_type"] == ERROR_TYPE_TIER1, kwargs
    assert kwargs["details"]["failure_class"] == CLASS_CREDIT, kwargs
    assert kwargs["details"]["consecutive_failures"] == 1, kwargs


def test_credit_exhaustion_is_classified_not_matched_by_accident():
    """The classifier must key on the real message, and stay narrow."""
    assert classify_failure(_BadRequestError(CREDIT_MSG)) == CLASS_CREDIT
    # Wrapped one level down, which is how langchain_anthropic re-raises it.
    try:
        try:
            raise _BadRequestError(CREDIT_MSG)
        except _BadRequestError as inner:
            raise RuntimeError("chat model call failed") from inner
    except RuntimeError as outer:
        assert classify_failure(outer) == CLASS_CREDIT
    # Narrowness: an unrelated failure must NOT be promoted to the
    # never-self-heals class, or every transient error pages on night one.
    assert classify_failure(RuntimeError("connection reset by peer")) == CLASS_GENERIC
    assert classify_failure(None) == CLASS_GENERIC


# --------------------------------------------------------------------------
# Criterion 2 -- N prior failures ESCALATE, they do not repeat identically
# --------------------------------------------------------------------------


def test_prior_failures_escalate_against_a_fixture_directory(tmp_path):
    """CRITERION 2. Fixture directory of prior ERROR files; escalation asserted
    against the tier-1 alert, so 'louder' means a different severity AND a
    different error_type -- not merely a different integer in the sentence."""
    memo_dir = tmp_path / "autoresearch"

    # Tier-1 reference: a single failure of the same class.
    _write_error_days(memo_dir, 1, end=TODAY)
    assert count_consecutive_failures(memo_dir, TODAY) == 1, "fixture precondition"
    with patch(ALERT_TARGET, autospec=True) as first:
        report_run_outcome(
            failed=True, exc=_BadRequestError(CREDIT_MSG),
            memo_dir=memo_dir, today=TODAY,
        )
    assert first.call_count == 1
    first_kwargs = first.call_args.kwargs

    # Now ESCALATE_AFTER_N consecutive ERROR files ending today.
    for p in memo_dir.iterdir():
        p.unlink()
    _write_error_days(memo_dir, ESCALATE_AFTER_N, end=TODAY)
    assert count_consecutive_failures(memo_dir, TODAY) == ESCALATE_AFTER_N, (
        "fixture precondition: the directory must really carry N consecutive "
        "failing dates, or 'escalated' below would be measuring nothing"
    )
    with patch(ALERT_TARGET, autospec=True) as later:
        summary = report_run_outcome(
            failed=True, exc=_BadRequestError(CREDIT_MSG),
            memo_dir=memo_dir, today=TODAY,
        )
    assert later.call_count == 1
    later_kwargs = later.call_args.kwargs

    assert summary["escalated"] is True, summary
    assert later_kwargs["severity"] != first_kwargs["severity"], (
        "escalation must change SEVERITY, not just the message text")
    assert later_kwargs["severity"] == "P0"
    assert later_kwargs["error_type"] == ERROR_TYPE_TIER2
    assert later_kwargs["error_type"] != first_kwargs["error_type"], (
        "a distinct error_type is required so AlertDeduper keys the escalation "
        "separately instead of swallowing it as a repeat")
    assert later_kwargs["details"]["consecutive_failures"] == ESCALATE_AFTER_N


def test_steady_state_does_not_emit_an_identical_notice_each_day(tmp_path):
    """CRITERION 2, the other half: 'rather than emitting an identical
    low-priority notice each day'. Nights strictly between the tier crossings
    must emit NOTHING."""
    memo_dir = tmp_path / "autoresearch"
    emitted: list[int] = []
    for n in range(1, ESCALATE_AFTER_N + 1):
        for p in list(memo_dir.iterdir()) if memo_dir.exists() else []:
            p.unlink()
        _write_error_days(memo_dir, n, end=TODAY)
        assert count_consecutive_failures(memo_dir, TODAY) == n, "precondition"
        with patch(ALERT_TARGET, autospec=True) as m:
            report_run_outcome(
                failed=True, exc=_BadRequestError(CREDIT_MSG),
                memo_dir=memo_dir, today=TODAY,
            )
        emitted.append(m.call_count)

    # Credit-class ladder: fires at n=1 (tier 0->1) and at n=ESCALATE_AFTER_N
    # (tier 1->2). Every night in between is steady state and must be silent.
    assert emitted[0] == 1, emitted
    assert emitted[-1] == 1, emitted
    assert sum(emitted) == 2, (
        f"expected exactly 2 alerts across {ESCALATE_AFTER_N} nights, got "
        f"{emitted} -- a per-night notice is the alert-fatigue anti-pattern "
        "this criterion forbids")


def test_escalated_tier_reminds_on_a_slow_cadence_not_daily(tmp_path):
    """The slow safety net (Alertmanager repeat_interval role): while at the
    escalated tier, re-emit every REMIND_EVERY_DAYS days -- and on no other
    day. Without this the fix would silence a persisting fault forever, which
    SRE Workbook ch.8 explicitly warns against."""
    memo_dir = tmp_path / "autoresearch"
    fired: dict[int, int] = {}
    for n in range(ESCALATE_AFTER_N, ESCALATE_AFTER_N + 2 * REMIND_EVERY_DAYS + 1):
        for p in list(memo_dir.iterdir()) if memo_dir.exists() else []:
            p.unlink()
        _write_error_days(memo_dir, n, end=TODAY)
        with patch(ALERT_TARGET, autospec=True) as m:
            report_run_outcome(
                failed=True, exc=_BadRequestError(CREDIT_MSG),
                memo_dir=memo_dir, today=TODAY,
            )
        fired[n] = m.call_count

    expected = {
        n: 1 if (n - ESCALATE_AFTER_N) % REMIND_EVERY_DAYS == 0 else 0
        for n in fired
    }
    assert fired == expected, f"got {fired}, expected {expected}"
    assert sum(fired.values()) == 3, fired  # n=7, n=14, n=21


def test_generic_failure_waits_for_the_threshold(tmp_path):
    """A generic (possibly transient) failure must NOT page on night one --
    otherwise the config-class carve-out is meaningless and every blip pages."""
    memo_dir = tmp_path / "autoresearch"
    _write_error_days(memo_dir, 1, end=TODAY)
    with patch(ALERT_TARGET, autospec=True) as m:
        report_run_outcome(
            failed=True, exc=RuntimeError("some transient upstream hiccup"),
            memo_dir=memo_dir, today=TODAY,
        )
    assert m.call_count == 0, "a single generic failure is not yet actionable"

    for p in memo_dir.iterdir():
        p.unlink()
    _write_error_days(memo_dir, PAGE_AFTER_N, end=TODAY)
    assert count_consecutive_failures(memo_dir, TODAY) == PAGE_AFTER_N
    with patch(ALERT_TARGET, autospec=True) as m2:
        report_run_outcome(
            failed=True, exc=RuntimeError("some transient upstream hiccup"),
            memo_dir=memo_dir, today=TODAY,
        )
    assert m2.call_count == 1, "crossing PAGE_AFTER_N must page"


def test_a_missed_night_resets_the_counter_and_the_ladder(tmp_path):
    """PINS A REAL LIMITATION rather than a property we do not have.

    An earlier draft of this file claimed the ladder was "gap-safe" -- that a
    skipped night jumping n from 2 to 4 would still fire. That claim was FALSE
    and this test replaces it. `count_consecutive_failures` walks backwards and
    STOPS at the first non-failing date, and a night on which the job did not
    run at all leaves no ERROR file, so it is not a failing date. Therefore n
    can never jump: it either advances by exactly 1 per consecutive failing
    night, or it RESETS.

    The consequence, pinned here so nobody has to rediscover it: a missed night
    silently rewinds the escalation ladder. That is the dead-man's-switch hole
    (a job that never runs also never reports), which is out of scope for this
    step's four criteria and is queued as its own masterplan step.
    """
    memo_dir = tmp_path / "autoresearch"
    # Six failing nights, then a gap (no file of any kind), then one more.
    _write_error_days(memo_dir, 6, end=TODAY - dt.timedelta(days=2))
    _write_error_days(memo_dir, 1, end=TODAY)
    assert count_consecutive_failures(memo_dir, TODAY) == 1, (
        "the gap at TODAY-1 must stop the walk -- this is the limitation")

    with patch(ALERT_TARGET, autospec=True) as m:
        report_run_outcome(
            failed=True, exc=RuntimeError("generic"),
            memo_dir=memo_dir, today=TODAY,
        )
    assert m.call_count == 0, (
        "n reset to 1, so a generic failure is back below PAGE_AFTER_N")

    # And the counterpart: with NO gap the same seven nights do escalate, so
    # the zero above is caused by the gap and not by a broken fixture.
    for p in memo_dir.iterdir():
        p.unlink()
    _write_error_days(memo_dir, 7, end=TODAY)
    assert count_consecutive_failures(memo_dir, TODAY) == 7
    with patch(ALERT_TARGET, autospec=True) as m2:
        report_run_outcome(
            failed=True, exc=RuntimeError("generic"),
            memo_dir=memo_dir, today=TODAY,
        )
    assert m2.call_count == 1


# --------------------------------------------------------------------------
# Criterion 3 -- a successful run emits NOTHING
# --------------------------------------------------------------------------


def test_successful_run_emits_no_alert(tmp_path):
    """CRITERION 3. Drives the production `_main_async` with a body returned.

    The fixture deliberately carries a long prior failure history, so a zero
    here cannot come from an empty directory -- it has to come from the success
    branch itself.
    """
    memo_dir = tmp_path / "autoresearch"
    _write_error_days(memo_dir, ESCALATE_AFTER_N + 2, end=TODAY - dt.timedelta(days=1))
    # Precondition: without today's memo this directory WOULD be deep in the
    # escalated tier -- proving the guard is not passing on an empty fixture.
    assert count_consecutive_failures(
        memo_dir, TODAY - dt.timedelta(days=1)
    ) == ESCALATE_AFTER_N + 2, "fixture precondition"

    with patch(ALERT_TARGET, autospec=True) as mock_alert:
        rc = _drive_main_async(memo_dir, raises=None)

    assert rc == 0, "a successful run must exit 0"
    memos = [p.name for p in memo_dir.glob("*.md") if "-ERROR-" not in p.name]
    assert len(memos) == 1, f"the production success path must write a memo: {memos}"
    assert mock_alert.call_count == 0, (
        f"a successful run must emit NO alert; got {mock_alert.call_args_list}")


def test_success_short_circuits_before_the_ladder(tmp_path):
    """CRITERION 3, the version that actually kills the always-fires mutant.

    `test_successful_run_emits_no_alert` above passes partly for a BENIGN
    reason: once today's success memo exists the consecutive count is 0, so
    the ladder would have been silent regardless. Measured, not guessed --
    a mutation run that replaced the `if not failed:` short-circuit with
    `if False:` left that guard GREEN.

    This guard removes the alibi. The fixture puts an ERROR file dated TODAY
    on disk as well, which is the real 2026-07-24 / 2026-07-25 shape (a failed
    02:00 launchd run plus a manual rerun the same day). The count is then
    non-zero and sitting on a tier crossing, so the ONLY thing that can keep
    the alert silent is the success short-circuit itself.
    """
    memo_dir = tmp_path / "autoresearch"
    _write_error_days(memo_dir, ESCALATE_AFTER_N, end=TODAY)
    assert count_consecutive_failures(memo_dir, TODAY) == ESCALATE_AFTER_N, (
        "fixture precondition: the ladder must be sitting ON a tier crossing, "
        "so silence cannot be explained by a zero count")
    with patch(ALERT_TARGET, autospec=True) as would_fire:
        report_run_outcome(
            failed=True, exc=_BadRequestError(CREDIT_MSG),
            memo_dir=memo_dir, today=TODAY,
        )
    assert would_fire.call_count == 1, (
        "control: this exact directory DOES fire when the run failed")

    with patch(ALERT_TARGET, autospec=True) as mock_alert:
        summary = report_run_outcome(
            failed=False, exc=None, memo_dir=memo_dir, today=TODAY,
        )
    assert mock_alert.call_count == 0, (
        "a successful run must emit nothing even when the directory is deep "
        "in the escalated tier")
    assert summary["alerts_emitted"] == 0
    assert "succeeded" in summary["reason"], summary


# --------------------------------------------------------------------------
# Counting traps (the numbers the research gate measured)
# --------------------------------------------------------------------------


def test_consecutive_is_not_total_and_stops_at_a_success(tmp_path):
    """The live directory has 62 ERROR dates but only 12 CONSECUTIVE failing
    dates, because 2026-07-24 and 2026-07-25 each carry BOTH an ERROR file and
    a success memo. A total-count implementation would report 62."""
    memo_dir = tmp_path / "autoresearch"
    _write_error_days(memo_dir, 20, end=TODAY)          # 2026-07-18..08-06
    _write_success_day(memo_dir, dt.date(2026, 7, 25))  # ERROR *and* success

    total_error_files = len(list(memo_dir.glob("*-ERROR-*.md")))
    assert total_error_files == 20, "fixture precondition"
    assert count_consecutive_failures(memo_dir, TODAY) == 12, (
        "must stop at 2026-07-25 (which has a success memo), giving "
        "2026-07-26..2026-08-06 = 12 -- not the 20 total ERROR files")


def test_empty_and_missing_directories_return_zero(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    assert count_consecutive_failures(empty, TODAY) == 0
    assert count_consecutive_failures(tmp_path / "nope", TODAY) == 0


def test_warn_files_are_not_failures(tmp_path):
    """phase-76.9 tolerates network weather with a WARN memo and rc=0. Counting
    a WARN as a failure would page for a condition the design accepts."""
    memo_dir = tmp_path / "autoresearch"
    memo_dir.mkdir()
    (memo_dir / f"{TODAY.isoformat()}-WARN-topic01.md").write_text("w", encoding="utf-8")
    assert count_consecutive_failures(memo_dir, TODAY) == 0


def test_live_memo_directory_measurement_is_reproducible():
    """Pins the number the contract and the module docstring both cite.

    If the live directory drifts (new nights land), the consecutive count moves
    -- so this asserts the RULE against the real data rather than a frozen
    integer: every date counted must have an ERROR file and no success memo.
    """
    live = REPO / "handoff" / "autoresearch"
    if not live.is_dir():
        pytest.fail(f"the live memo directory is missing: {live}")
    errors, successes = ah._scan_memo_dir(live)
    assert len(errors) >= 62, f"expected >=62 ERROR dates on disk, got {len(errors)}"
    both = sorted(set(errors) & set(successes))
    assert "2026-07-24" in both and "2026-07-25" in both, (
        f"the two both-files dates must still be present: {both}")
    # The rule, re-derived over live data.
    today = max(errors)
    n = count_consecutive_failures(live, dt.date.fromisoformat(today))
    assert n >= 1
    cur = dt.date.fromisoformat(today)
    for _ in range(n):
        key = cur.isoformat()
        assert errors.get(key, 0) > 0 and successes.get(key, 0) == 0, key
        cur -= dt.timedelta(days=1)
    assert not (errors.get(cur.isoformat(), 0) > 0
                and successes.get(cur.isoformat(), 0) == 0), (
        "the walk must stop at the first non-failing date")


# --------------------------------------------------------------------------
# Anti-vacuity pins
# --------------------------------------------------------------------------


def test_wrong_patch_target_does_not_exist():
    """`report_run_outcome` imports the emitter FUNCTION-LOCALLY, so the
    tempting patch target below is a no-op. If someone ever hoists that import
    to module scope, this fails and forces the guards to be re-examined --
    otherwise a module-scope patch would silence the real emitter and every
    alert assertion above would still pass while nothing reached Slack."""
    assert not hasattr(ah, "raise_cron_alert_sync"), (
        "backend.services.autoresearch_health must NOT expose "
        "raise_cron_alert_sync at module scope -- see the module docstring")
    import backend.services.observability.alerting as alerting
    assert hasattr(alerting, "raise_cron_alert_sync"), (
        f"{ALERT_TARGET} must exist, or every patch in this file is a no-op")


def test_production_entry_point_actually_reports_both_outcomes():
    """Guards above stub `run_research`; this one pins that `_main_async`
    reaches `_report_outcome` on BOTH branches, so deleting either call site is
    caught even if a future refactor changes how the guards drive it."""
    src = RUN_MEMO.read_text(encoding="utf-8")
    code = "\n".join(
        line for line in src.splitlines() if not line.lstrip().startswith("#")
    )
    assert code.count("_report_outcome(True, e, topic)") == 1, (
        "the failure branch must report its outcome")
    assert code.count("_report_outcome(False, None, topic)") == 1, (
        "the success branch must report its outcome")


def test_notify_injection_is_not_the_only_path(tmp_path):
    """If `notify=` were the ONLY way the alert is emitted, nothing would prove
    production resolves the real `raise_cron_alert_sync`. This drives the
    no-injection path and asserts the REAL module attribute was called."""
    memo_dir = tmp_path / "autoresearch"
    _write_error_days(memo_dir, 1, end=TODAY)
    with patch(ALERT_TARGET, autospec=True) as mock_alert:
        report_run_outcome(
            failed=True, exc=_BadRequestError(CREDIT_MSG),
            memo_dir=memo_dir, today=TODAY,
        )
    assert mock_alert.call_count == 1, (
        "with no notify= injection the real emitter must be resolved")


def test_reporting_is_fail_open(tmp_path):
    """A notification problem must never change the nightly job's exit code."""
    memo_dir = tmp_path / "autoresearch"
    _write_error_days(memo_dir, 1, end=TODAY)

    def _boom(**kwargs):
        raise RuntimeError("slack is on fire")

    summary = report_run_outcome(
        failed=True, exc=_BadRequestError(CREDIT_MSG),
        memo_dir=memo_dir, today=TODAY, notify=_boom,
    )
    assert summary["ok"] is False
    assert "fail-open" in summary["reason"]

    # And end to end: the production path still returns rc=1, not a traceback.
    with patch(ALERT_TARGET, autospec=True, side_effect=RuntimeError("boom")):
        rc = _drive_main_async(memo_dir, raises=_BadRequestError(CREDIT_MSG))
    assert rc == 1


# --------------------------------------------------------------------------
# Criterion 4 -- the operator's decision is recorded verbatim
# --------------------------------------------------------------------------

#: The operator's instruction, verbatim. Duplicated here on purpose so the
#: guard fails if the artifact is edited, paraphrased, or deleted.
OPERATOR_INSTRUCTION = (
    "82.11 metered rail: $0-metered stands -> move autoresearch OFF it or\n"
    "  disable it. Do NOT buy credits."
)


def test_operator_decision_recorded_verbatim_in_the_artifact():
    """CRITERION 4."""
    assert CONTRACT.is_file(), f"the step artifact is missing: {CONTRACT}"
    text = CONTRACT.read_text(encoding="utf-8")
    assert OPERATOR_INSTRUCTION in text, (
        "the operator's verbatim instruction must appear in the step artifact")
    # The criterion enumerates three options; the artifact must name all three
    # (so an alternative cannot be quietly dropped) AND say which was chosen.
    for token in ("buy credits", "move off the metered rail", "disable"):
        assert token.lower() in text.lower(), (
            f"the artifact must address the option {token!r}")
    assert "DECISION: move off the metered rail." in text, (
        "the artifact must state which of the three options was chosen, not "
        "merely list them")


def test_default_takes_the_max_rail_branch_with_no_env_flag(tmp_path):
    """The decision must hold with NOTHING set -- that is what "default" means.

    Drives the REAL `run_nightly.sh` in a fixture repo whose `.env` contains no
    `AUTORESEARCH_*` key at all (the measured state of the production
    `backend/.env` on 2026-08-06), pointed at a dead bridge port. If the
    default were still 0 the script would skip the rail block entirely and run
    `run_memo` against the metered API -- so `observed_env.json` would exist
    and the exit code would be 0. Instead it must take the loud-fail preflight.

    This complements the source-text pin below: that one can be defeated by
    rewording, this one observes the script's actual behaviour.
    """
    from backend.tests.test_phase_76_9_2_max_bridge import (
        _free_port, _make_nightly_fixture, _run_nightly,
    )

    root = tmp_path / "repo"
    dead_port = _free_port()  # bound then released -> nothing listens
    _make_nightly_fixture(
        root,
        # No AUTORESEARCH_USE_MAX_RAIL line: the default must decide.
        f"AUTORESEARCH_MAX_RAIL_URL=http://127.0.0.1:{dead_port}\n"
        "ANTHROPIC_API_KEY=a-real-metered-key\n",
    )
    env_text = (root / "backend" / ".env").read_text(encoding="utf-8")
    assert "AUTORESEARCH_USE_MAX_RAIL" not in env_text, "fixture precondition"

    r = _run_nightly(root)
    assert r.returncode == 78, (
        f"the DEFAULT must take the max-rail branch and fail loud on a dead "
        f"bridge; got rc={r.returncode}: {r.stderr}")
    assert not (root / "observed_env.json").exists(), (
        "run_memo must NEVER run against the metered API by default")
    log = (root / "handoff" / "autoresearch.log").read_text(encoding="utf-8")
    assert "NOT falling back to metered" in log


def test_rail_decision_is_implemented_in_tracked_code():
    """The decision is only real if the nightly run actually leaves the metered
    rail. `backend/.env` is gitignored, so the default must live here."""
    text = NIGHTLY.read_text(encoding="utf-8")
    code = "\n".join(
        line for line in text.splitlines() if not line.lstrip().startswith("#")
    )
    assert "${AUTORESEARCH_USE_MAX_RAIL:-1}" in code, (
        "the max rail must be the DEFAULT in executed code, not only in a "
        "comment or an untracked .env line")
    assert "NOT falling back to metered API" in code, (
        "the loud-fail path must survive -- a down bridge must never silently "
        "reach the metered API")
