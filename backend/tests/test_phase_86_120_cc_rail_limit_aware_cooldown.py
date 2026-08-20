"""phase-86.120: the CC rail's circuit breaker gains weekly/session/Opus
quota-exhaustion awareness.

CONTEXT (see handoff/current/contract_86.120.md for the full research
basis). The pre-existing rail guard (phase-66.1, claude_code_client.py) is
an IN-MEMORY, PER-CYCLE, generic-failure breaker: rail_guard_reset() wipes
it at the start of every cycle, and it only trips after
claude_rail_breaker_threshold (default 20) CONSECUTIVE failures of any
kind. Once the operator's weekly (or session, or Opus-only) Claude Max
limit is exhausted, every subsequent cycle for the rest of the week
independently re-discovers the same fact the hard way.

This step adds:
  1. classify_limit_failure() -- best-effort classification of a CLI
     failure as one of the three documented quota-exhaustion messages
     (code.claude.com/docs/en/errors), reading the FULL stdout / JSON
     envelope `result` field rather than the existing truncated
     str(exc)[:150] (research_brief_86.120.md M5).
  2. A disk-persisted cooldown (backend/agents/_cache/cc_rail_cooldown.json)
     keyed off that classification, checked at the TOP of
     claude_code_invoke() -- before any subprocess spawn -- so BOTH known
     callers (ClaudeCodeClient.generate_content AND
     backend/services/ticket_queue_processor.py's direct call) are
     protected uniformly.
  3. Azure "accelerated circuit breaking": a classified hit also trips the
     existing in-cycle breaker at N=1 (not N=20), without paging (a P1 for
     an already-classified, already-mitigated condition would be the exact
     alarm-fatigue anti-pattern scripts/away_ops/auth_state.py's module
     docstring warns about).

THE ASSERTIONS MATTER INDIVIDUALLY, same discipline as this project's other
NaN/None-safety suites: a test that only checks "no subprocess spawned"
would also be satisfied by an implementation that permanently disables the
rail. So the suite separately proves: detection is SPECIFIC (a generic
failure does NOT engage the cooldown), the skip is MECHANICAL (mock-level
subprocess assertion, not just an empty response), the cooldown is
BOUNDED (self-clears), and NOTHING here reopens a path to metered
api.anthropic.com spend (the REGRESSION test).
"""
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from backend.agents import claude_code_client as ccc
from backend.agents.claude_code_client import (
    ClaudeCodeError,
    classify_limit_failure,
    claude_code_invoke,
    cooldown_active,
    cooldown_clear_on_success,
    cooldown_record_hit,
    cooldown_status,
    rail_guard_reset,
    rail_guard_status,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_REAL_SESSION_LIMIT_ENVELOPE_PATH = (
    _REPO_ROOT / "handoff" / "away_ops" / "session_pm_20260707T200007Z.json"
)


def _mock_completed(stdout: str, returncode: int = 0, stderr: str = "") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=["claude"], returncode=returncode, stdout=stdout, stderr=stderr)


@pytest.fixture(autouse=True)
def _isolated_cooldown_file(tmp_path, monkeypatch):
    """Every test gets its OWN cooldown state file -- never the real
    backend/agents/_cache/cc_rail_cooldown.json -- so this suite cannot
    leave residue that changes another test's (or the real rail's)
    behaviour, and so tests can run in any order / in parallel."""
    fake_path = tmp_path / "cc_rail_cooldown.json"
    monkeypatch.setattr(ccc, "_COOLDOWN_PATH", fake_path)
    rail_guard_reset("test-cycle")
    yield
    rail_guard_reset("test-cycle-teardown")


# ---------------------------------------------------------------------------
# 1. Classifier -- detection is SPECIFIC (criterion 1)
# ---------------------------------------------------------------------------


def test_classify_real_captured_session_limit_envelope():
    """The ONE real quota-exhaustion envelope in this repo
    (research_brief_86.120.md 'MEASURED' section). Live shape:
    "You've hit your session limit · resets 1am (Europe/Oslo)" --
    note NO minutes (bare hour), unlike the official doc's "3:45pm"
    examples; the regexes must handle both forms."""
    raw = _REAL_SESSION_LIMIT_ENVELOPE_PATH.read_text(encoding="utf-8")
    c = classify_limit_failure(raw)
    assert c is not None
    assert c.kind == "session"
    assert "hit your session limit" in c.raw_message.lower()
    assert c.parsed_retry_at is not None
    assert c.parsed_retry_at > datetime.now(timezone.utc)


def test_classify_weekly_synthesized_from_docs():
    """SYNTHESIZED from code.claude.com/docs/en/errors -- NOT reproduced
    from real captured data (research_brief_86.120.md M6: zero local
    evidence of an actual weekly-limit hit exists)."""
    envelope = json.dumps({
        "type": "result", "subtype": "success", "is_error": True,
        "api_error_status": 429,
        "result": "You've hit your weekly limit - resets Mon 12:00am",
    })
    c = classify_limit_failure(envelope)
    assert c is not None
    assert c.kind == "weekly"
    assert c.parsed_retry_at is not None
    assert c.parsed_retry_at > datetime.now(timezone.utc)
    assert c.parsed_retry_at <= datetime.now(timezone.utc) + timedelta(days=8)


def test_classify_opus_synthesized_from_docs():
    """SYNTHESIZED from code.claude.com/docs/en/errors."""
    envelope = json.dumps({
        "type": "result", "subtype": "success", "is_error": True,
        "api_error_status": 429,
        "result": "You've hit your Opus limit - resets 3:45pm",
    })
    c = classify_limit_failure(envelope)
    assert c is not None
    assert c.kind == "opus"


def test_classify_plaintext_non_json_limit_message():
    """claude_code_client.py's own pre-existing :441-443 comment documents
    that the CLI's failure text is often plain, NOT JSON. The classifier
    must fall back to a raw substring scan in that case."""
    c = classify_limit_failure("You've hit your session limit - resets 3:45pm")
    assert c is not None
    assert c.kind == "session"


def test_classify_returns_none_for_generic_failure():
    """CONTROL: an ordinary transient failure (network blip, auth hiccup)
    must NOT classify as a quota hit -- it has to fall through to the
    existing consecutive-failure breaker unchanged (criterion 7)."""
    assert classify_limit_failure("Error: connection reset by peer") is None
    assert classify_limit_failure('{"type":"result","subtype":"error_max_turns"}') is None
    assert classify_limit_failure("") is None


def test_classify_reads_full_message_not_the_150_char_truncation():
    """research_brief_86.120.md M5: the EXISTING str(exc)[:150] slice
    severs the human sentence at "You've hit y" when enough JSON keys
    precede `result`. Prove the classifier reads the untruncated stdout,
    not a pre-truncated string, by padding many keys before `result`."""
    padding = {f"k{i}": "x" * 20 for i in range(10)}
    envelope = json.dumps({
        **padding,
        "type": "result", "is_error": True, "api_error_status": 429,
        "result": "You've hit your session limit - resets 3:45pm",
    })
    # sanity: the old-style 150-char truncation WOULD have severed this
    assert "session limit" not in envelope[:150]
    c = classify_limit_failure(envelope)
    assert c is not None
    assert c.kind == "session"
    assert "session limit" in c.raw_message


def test_classify_extracts_the_result_field_not_the_raw_envelope():
    """Closes a Q/A cycle-3 finding. test_classify_reads_full_message_...
    above proves the message isn't cut off at 150 chars, but NOT that the
    classifier extracts JSON's `result` field specifically rather than
    falling through to scanning the raw envelope text. Mutation showed
    deleting the `result`-extraction block entirely survives the WHOLE
    suite, with two real consequences: (a) the operator-facing
    cooldown_message degrades from a clean sentence to a raw JSON blob, and
    (b) a FALSE-POSITIVE risk -- a successful envelope whose `result` is a
    normal analysis, but which merely CONTAINS limit-shaped text somewhere
    else (e.g. a debug echo of the prompt), would wrongly classify as a
    quota hit and engage a real cooldown on a call that actually succeeded."""
    # (a) on the real captured envelope, raw_message must be the extracted
    # sentence, not the raw JSON blob.
    real = _REAL_SESSION_LIMIT_ENVELOPE_PATH.read_text(encoding="utf-8")
    expected_sentence = json.loads(real)["result"]
    c = classify_limit_failure(real)
    assert c is not None
    assert not c.raw_message.startswith("{"), (
        f"raw_message looks like the raw JSON envelope, not the extracted "
        f"result sentence: {c.raw_message[:80]!r}"
    )
    assert c.raw_message == expected_sentence

    # (b) FALSE-POSITIVE control: a successful call whose `result` is a
    # normal analysis must not classify as a limit hit just because some
    # OTHER envelope field happens to contain limit-shaped phrasing.
    envelope = json.dumps({
        "type": "result", "subtype": "success", "is_error": False,
        "result": "AAPL: BUY, confidence 0.81",
        "debug_prompt_echo": "the analyst asked: have we hit your session limit today?",
    })
    assert classify_limit_failure(envelope) is None, (
        "a successful call must never classify as a limit hit because unrelated "
        "envelope text mentions limit phrasing outside the result field"
    )


# ---------------------------------------------------------------------------
# 2. Persistence -- survives rail_guard_reset() and a simulated restart
#    (criterion 2)
# ---------------------------------------------------------------------------


def test_cooldown_record_and_status_roundtrip():
    c = classify_limit_failure("You've hit your weekly limit - resets Mon 12:00am")
    cooldown_record_hit(c)
    status = cooldown_status()
    assert status["cooldown_active"] is True
    assert status["cooldown_kind"] == "weekly"
    assert cooldown_active() is True


def test_cooldown_survives_rail_guard_reset():
    """The in-memory _RAIL_GUARD is wiped every cycle by rail_guard_reset()
    (phase-66.1) -- the disk cooldown must NOT be, because it lives in a
    separate persistence layer entirely."""
    c = classify_limit_failure("You've hit your session limit - resets 3:45pm")
    cooldown_record_hit(c)
    assert cooldown_active() is True

    rail_guard_reset("next-cycle-1")  # simulates autonomous_loop.py's cycle-start reset
    assert cooldown_active() is True, "cooldown must survive a per-cycle rail_guard_reset()"

    rail_guard_reset("next-cycle-2")  # a second cycle, still cooling down
    assert cooldown_active() is True


def test_cooldown_state_persists_across_simulated_process_restart():
    """Simulates a backend restart: read the state back purely from the
    JSON file on disk, with no reliance on any in-process Python object
    (cooldown_status() never caches -- it re-reads the file every call)."""
    c = classify_limit_failure("You've hit your Opus limit - resets 3:45pm")
    recorded = cooldown_record_hit(c)

    # "restart" = read the file directly, bypassing any module state.
    on_disk = json.loads(ccc._COOLDOWN_PATH.read_text(encoding="utf-8"))
    assert on_disk["kind"] == "opus"
    assert on_disk["retry_at"] == recorded["retry_at"]

    # and the public API, called fresh, agrees.
    assert cooldown_status()["cooldown_kind"] == "opus"


def test_cooldown_expires_after_retry_at():
    """A cooldown recorded with a retry_at already in the past (i.e. the
    backoff window has elapsed) must read as inactive."""
    past = datetime.now(timezone.utc) - timedelta(hours=1)
    ccc._cooldown_write({
        "kind": "session", "raw_message": "test", "hit_at": past.isoformat(),
        "retry_at": past.isoformat(),
    })
    assert cooldown_active() is False


def test_cooldown_clears_on_success():
    c = classify_limit_failure("You've hit your session limit - resets 3:45pm")
    cooldown_record_hit(c)
    assert cooldown_active() is True
    cooldown_clear_on_success()
    assert cooldown_active() is False
    assert cooldown_status() == {"cooldown_active": False}


def test_cooldown_falls_back_to_default_when_reset_time_is_unparseable():
    """A message that matches a limit KIND but whose reset-time suffix the
    regex cannot parse must still engage a SAFE cooldown (the configured
    default), never silently skip cooldown engagement entirely."""
    c = classify_limit_failure("You've hit your weekly limit - resets sometime soon")
    assert c is not None
    assert c.parsed_retry_at is None  # confirms this IS the unparseable case
    before = datetime.now(timezone.utc)
    cooldown_record_hit(c, now=before)
    status = cooldown_status()
    retry_at = datetime.fromisoformat(status["cooldown_retry_at"])
    assert retry_at > before + timedelta(minutes=1), "must not under-shoot with a near-zero cooldown"
    assert retry_at <= before + timedelta(hours=192)  # settings ceiling


def test_cooldown_never_trusts_a_past_or_implausible_parsed_time():
    """Defensive clamp: even if parsing produced something nonsensical (in
    the past, or absurdly far out), cooldown_record_hit must not honour it
    verbatim -- it must fall back to the safe default."""
    now = datetime.now(timezone.utc)
    from backend.agents.claude_code_client import LimitClassification

    stale = LimitClassification(kind="session", raw_message="test", parsed_retry_at=now - timedelta(hours=1))
    cooldown_record_hit(stale, now=now)
    retry_at = datetime.fromisoformat(cooldown_status()["cooldown_retry_at"])
    assert retry_at > now

    far_future = LimitClassification(kind="weekly", raw_message="test", parsed_retry_at=now + timedelta(days=30))
    cooldown_record_hit(far_future, now=now)
    retry_at = datetime.fromisoformat(cooldown_status()["cooldown_retry_at"])
    assert retry_at <= now + timedelta(days=9)


# ---------------------------------------------------------------------------
# 2b. WIRING-level guards (Q/A cycle-1 fix, wf_a285260f-0dd).
#
#     Every test above builds cooldown state DIRECTLY via
#     cooldown_record_hit(classify_limit_failure(...)) in its own setup --
#     which proves the persistence/expiry/clamp LOGIC works, but never
#     proves claude_code_invoke's PRODUCTION FAILURE PATH actually calls
#     that logic. Q/A's independent in-memory mutation matrix deleted the
#     single production call `cooldown_record_hit(_limit)` from
#     claude_code_invoke and the whole 27-test suite stayed green (mutant
#     M10) -- sole-coverage vacuity on the step's headline behaviour
#     (criterion 2). It found the same gap for cooldown_clear_on_success()
#     (M3: deleting it also survived, because cooldown_active() reads False
#     both when a record is genuinely cleared AND when it merely expired by
#     time -- a probe that cannot discriminate the two is not a guard) and
#     for the tz-fallback bug fixed during GENERATE (M6: reverting it to
#     `now.tzinfo` also survived, because no test pinned the correct
#     behaviour at all). These three tests close exactly those three gaps,
#     each driving the REAL production entry point rather than pre-seeding
#     state, and each independently reproduced as RED against the ORIGINAL
#     mutation before being confirmed GREEN here (see live_check_86.120.md
#     section 10 for the mutation transcripts).
# ---------------------------------------------------------------------------


def test_a_real_classified_failure_through_generate_content_actually_persists_the_cooldown():
    """Closes M10. Drives the failure through ClaudeCodeClient.generate_content
    (the production entry point), NOT cooldown_record_hit() called directly --
    if `cooldown_record_hit(_limit)` is ever deleted from claude_code_invoke's
    failure path, this test must go red on BOTH assertions below."""
    client = ccc.ClaudeCodeClient(model_name="claude-haiku-4-5")
    envelope = json.dumps({
        "type": "result", "subtype": "success", "is_error": True, "api_error_status": 429,
        "result": "You've hit your weekly limit - resets Mon 12:00am",
    })
    with patch("backend.agents.claude_code_client.subprocess.run") as run:
        run.return_value = _mock_completed(stdout=envelope, returncode=1)
        client.generate_content("analyze AAPL", {})  # the ONE real failure
        assert run.call_count == 1

    assert cooldown_active() is True, "the production entry point must have persisted the cooldown"
    assert cooldown_status()["cooldown_kind"] == "weekly"

    # CORRECTED (Q/A cycle 2): a SECOND call within the SAME cycle spawns
    # zero subprocesses, but a differential mutation run (M1 alone vs M1+M4)
    # showed this specific assertion is satisfied by the accelerated
    # in-cycle breaker (_rail_guard_open_for_quota, N=1) as much as by the
    # persisted cooldown -- both trip together on a real classified hit, so
    # this call alone cannot isolate which one is doing the gating. The
    # genuine PERSISTED-STATE claim (survives a fresh rail_guard_reset() and
    # a simulated restart, where the in-cycle breaker is deliberately wiped)
    # is proven separately by
    # test_generate_content_skips_subprocess_across_two_cycles_and_a_restart.
    with patch("backend.agents.claude_code_client.subprocess.run") as run2:
        client.generate_content("analyze MSFT", {})
        run2.assert_not_called()


def test_a_real_success_after_cooldown_removes_the_persisted_state_file():
    """Closes M3. Q/A's own finding: cooldown_active() CANNOT discriminate
    this mutation, because an EXPIRED-by-time cooldown already reads False
    before any success happens -- the control and the mutant give the same
    boolean. The state FILE's existence is the only property that actually
    distinguishes 'cleared by a real success' from 'merely timed out'."""
    past = datetime.now(timezone.utc) - timedelta(minutes=1)
    ccc._cooldown_write({
        "kind": "session", "raw_message": "test", "hit_at": (past - timedelta(hours=6)).isoformat(),
        "retry_at": past.isoformat(),
    })
    assert ccc._COOLDOWN_PATH.exists(), "control: the file exists before the success"
    assert cooldown_active() is False, (
        "control: an EXPIRED cooldown already reads inactive -- this is exactly "
        "why cooldown_active() cannot discriminate M3; the file check below can"
    )

    with patch("backend.agents.claude_code_client.subprocess.run") as run:
        run.return_value = _mock_completed(
            stdout=json.dumps({"subtype": "success", "result": "BUY", "usage": {}}),
        )
        ccc.claude_code_invoke("analyze AAPL")
        run.assert_called_once()

    assert not ccc._COOLDOWN_PATH.exists(), (
        "a real success must remove the state FILE -- if it merely reads as "
        "expired without being cleared, a stale record survives on disk"
    )


def test_bare_reset_time_with_no_explicit_timezone_uses_host_local_zone_not_utc():
    """Closes M6 -- regression pin for the tz-fallback bug found and fixed
    DURING GENERATE (experiment_results_86.120.md): the parser's fallback
    for a message with NO explicit '(Zone/Name)' suffix (the official doc's
    bare "resets 3:45pm" form) must resolve against the HOST's actual local
    timezone. Reverting to `now.tzinfo` -- UTC, since classify_limit_failure's
    default `now` is datetime.now(timezone.utc) -- silently misparses the
    common case, and nothing pinned this before Q/A caught it.

    Deterministic without hardcoding the host's UTC offset (the fixed-offset
    trap this repo has hit before): convert the PARSED retry_at back to the
    host's own local wall-clock and assert it reads 3:45pm exactly. Under the
    reverted bug, the value would be parsed AS 3:45pm UTC, which reads as a
    DIFFERENT local hour whenever the host's offset is non-zero.
    """
    host_offset = datetime.now().astimezone().utcoffset()
    if host_offset == timedelta(0):
        pytest.skip("host is UTC -- this fixture cannot distinguish the bug from the fix at UTC+0")

    c = classify_limit_failure("You've hit your session limit - resets 3:45pm")
    assert c is not None
    assert c.parsed_retry_at is not None

    local_reading = c.parsed_retry_at.astimezone()
    assert (local_reading.hour, local_reading.minute) == (15, 45), (
        f"parsed as {local_reading} local -- expected 3:45pm LOCAL wall-clock; "
        f"if this instead reads as 3:45pm UTC, the tz fallback reverted to now.tzinfo"
    )


def test_a_corrupt_but_parseable_cooldown_record_fails_toward_still_cooling():
    """Closes M17 (Q/A cycle 2). cooldown_status()'s except-branch carries an
    explicit safety comment -- 'a corrupt record fails toward SAFE (still
    cooling down)' -- that Cycle 1's own code-review leg relied on to wave
    through the module's seven broad `except Exception` blocks. Q/A found
    that claim had ZERO test coverage: inverting `active = True` to
    `active = False` left the whole suite green. A record that IS valid
    JSON but has an unparseable `retry_at` (not literal garbage bytes, which
    _cooldown_read's own try/except would swallow before even reaching this
    branch) is what actually exercises the except at claude_code_client.py:469."""
    ccc._COOLDOWN_PATH.write_text(
        json.dumps({
            "kind": "session", "raw_message": "test",
            "hit_at": "not-a-date", "retry_at": "not-a-date-either",
        }),
        encoding="utf-8",
    )
    assert cooldown_active() is True, (
        "a corrupt-but-parseable record must fail toward SAFE (still cooling) -- "
        "never toward resuming subprocess spawns against a possibly-still-exhausted quota"
    )


# ---------------------------------------------------------------------------
# 3. Pre-subprocess skip guard (criterion 3) -- MOCK-LEVEL, not just an
#    empty response. Covers BOTH known callers of claude_code_invoke.
# ---------------------------------------------------------------------------


def test_claude_code_invoke_skips_subprocess_when_cooldown_active():
    c = classify_limit_failure("You've hit your weekly limit - resets Mon 12:00am")
    cooldown_record_hit(c)

    with patch("backend.agents.claude_code_client.subprocess.run") as run:
        with pytest.raises(ClaudeCodeError) as exc_info:
            claude_code_invoke("analyze AAPL")
        run.assert_not_called()
    assert getattr(exc_info.value, "cooldown_blocked", False) is True


def test_ticket_queue_processor_direct_call_pattern_also_blocked():
    """backend/services/ticket_queue_processor.py:241 calls
    claude_code_invoke() DIRECTLY, bypassing ClaudeCodeClient entirely --
    the guard must live in claude_code_invoke itself (not only in
    generate_content's pre-check) or this call site is unprotected."""
    c = classify_limit_failure("You've hit your session limit - resets 3:45pm")
    cooldown_record_hit(c)

    with patch("backend.agents.claude_code_client.subprocess.run") as run:
        with pytest.raises(ClaudeCodeError):
            claude_code_invoke("ticket task text", system="ticket system prompt", timeout_s=60)
        run.assert_not_called()


def test_generate_content_skips_subprocess_across_two_cycles_and_a_restart():
    """Criterion 3, literal: 'across at least two simulated cycles
    including one after a simulated process restart'."""
    ClaudeCodeClient = ccc.ClaudeCodeClient
    client = ClaudeCodeClient(model_name="claude-haiku-4-5")

    c = classify_limit_failure("You've hit your weekly limit - resets Mon 12:00am")
    cooldown_record_hit(c)

    with patch("backend.agents.claude_code_client.subprocess.run") as run:
        # cycle 1
        resp1 = client.generate_content("prompt one", {})
        run.assert_not_called()
        assert resp1.text == ""

        rail_guard_reset("cycle-2")  # cycle 2 (in-memory breaker wiped)
        client.generate_content("prompt two", {})
        run.assert_not_called()

        # simulated restart: fresh client instance, cooldown re-read from disk only
        rail_guard_reset("cycle-3-post-restart")
        fresh_client = ClaudeCodeClient(model_name="claude-haiku-4-5")
        resp3 = fresh_client.generate_content("prompt three", {})
        run.assert_not_called()
        assert resp3.text == ""


def test_cooldown_block_does_not_inflate_the_generic_consecutive_failure_counter():
    """A KNOWN, already-classified-and-handled cooldown skip must not feed
    claude_rail_breaker_threshold's generic counter -- repeatedly hitting a
    cooldown-blocked call must never itself trip a (redundant, noisy) P1
    page for a condition that is already understood and mitigated."""
    client = ccc.ClaudeCodeClient(model_name="claude-haiku-4-5")
    c = classify_limit_failure("You've hit your weekly limit - resets Mon 12:00am")
    cooldown_record_hit(c)

    with patch("backend.agents.claude_code_client.subprocess.run") as run:
        for _ in range(25):  # well past claude_rail_breaker_threshold's default 20
            client.generate_content("prompt", {})
        run.assert_not_called()

    assert rail_guard_status()["consecutive_failures"] == 0


# ---------------------------------------------------------------------------
# 4. A classified hit ALSO trips the in-cycle breaker at N=1 (Azure
#    "accelerated circuit breaking"), without paging.
# ---------------------------------------------------------------------------


def test_a_classified_hit_opens_the_in_cycle_breaker_immediately():
    with patch("backend.agents.claude_code_client.subprocess.run") as run:
        run.return_value = _mock_completed(
            stdout=json.dumps({"result": "You've hit your session limit - resets 3:45pm"}),
            returncode=1,
        )
        with pytest.raises(ClaudeCodeError):
            claude_code_invoke("analyze AAPL")
    status = rail_guard_status()
    assert status["breaker_tripped"] is True
    assert status["consecutive_failures"] == 0, "tripped via the quota path, not the generic counter"


def test_a_classified_hit_does_not_page(monkeypatch):
    """No P1 for an already-classified, already-mitigated condition --
    paging here would be the exact alarm-fatigue anti-pattern
    scripts/away_ops/auth_state.py's module docstring warns about."""
    paged = []
    monkeypatch.setattr(
        "backend.services.observability.alerting.raise_cron_alert_sync",
        lambda **kw: paged.append(kw),
        raising=False,
    )
    with patch("backend.agents.claude_code_client.subprocess.run") as run:
        run.return_value = _mock_completed(
            stdout=json.dumps({"result": "You've hit your weekly limit - resets Mon 12:00am"}),
            returncode=1,
        )
        with pytest.raises(ClaudeCodeError):
            claude_code_invoke("analyze AAPL")
    assert paged == []


# ---------------------------------------------------------------------------
# 5. The pre-existing generic breaker is UNCHANGED for non-limit failures
#    (criterion 7) -- this step ADDS, never REPLACES.
# ---------------------------------------------------------------------------


def test_generic_failure_still_feeds_the_old_breaker_and_never_engages_cooldown():
    """_rail_guard_record_failure is invoked by ClaudeCodeClient.generate_content's
    except-handler, NOT by claude_code_invoke itself -- that was already true
    before this step (claude_code_invoke has never touched the breaker
    counter directly), so this drives the SAME path production code does."""
    client = ccc.ClaudeCodeClient(model_name="claude-haiku-4-5")
    with patch("backend.agents.claude_code_client.subprocess.run") as run:
        run.return_value = _mock_completed(stdout="", returncode=1, stderr="network unreachable")
        for _ in range(5):
            client.generate_content("analyze AAPL", {})
    assert run.call_count == 5, "a generic failure must still spawn the subprocess every time"
    assert cooldown_active() is False, "a non-limit failure must never engage the cooldown"
    assert rail_guard_status()["consecutive_failures"] == 5


def test_generic_breaker_still_trips_at_the_configured_threshold(monkeypatch):
    monkeypatch.setattr(
        "backend.agents.claude_code_client._rail_breaker_threshold", lambda: 3
    )
    client = ccc.ClaudeCodeClient(model_name="claude-haiku-4-5")
    with patch("backend.agents.claude_code_client.subprocess.run") as run:
        run.return_value = _mock_completed(stdout="", returncode=1, stderr="boom")
        for _ in range(3):
            client.generate_content("analyze AAPL", {})
    assert rail_guard_status()["breaker_tripped"] is True


# ---------------------------------------------------------------------------
# 6. Settings field (criterion 4)
# ---------------------------------------------------------------------------


def test_settings_field_exists_with_a_sane_default_and_bounds():
    from backend.config.settings import Settings

    field = Settings.model_fields["claude_rail_cooldown_default_hours"]
    assert field.default == 6.0
    s = Settings(claude_rail_cooldown_default_hours=6.0)
    assert s.claude_rail_cooldown_default_hours == 6.0


# ---------------------------------------------------------------------------
# 7. At least one always-on signal agent respects the cooldown, driven
#    through the SAME shared entry point (make_client -> ClaudeCodeClient)
#    all six of them use (criterion 8). pead_signal.py:300's call is
#    `make_client(getattr(settings, "pead_signal_model", "claude-haiku-4-5"),
#    None, settings, enable_prompt_caching=False)` -- functionally equivalent
#    to what this test drives (the getattr default is inert while the field
#    exists), not byte-for-byte verbatim; re-testing pead_signal.py's own
#    SEC-filing fetch logic is out of this step's scope.
# ---------------------------------------------------------------------------


def test_an_always_on_signal_agent_respects_cooldown_via_shared_entry_point():
    from backend.agents.llm_client import make_client
    from backend.config.settings import Settings

    settings = Settings(paper_use_claude_code_route=True, pead_signal_model="claude-haiku-4-5")
    c = classify_limit_failure("You've hit your weekly limit - resets Mon 12:00am")
    cooldown_record_hit(c)

    client = make_client(settings.pead_signal_model, None, settings, enable_prompt_caching=False)
    with patch("backend.agents.claude_code_client.subprocess.run") as run:
        resp = client.generate_content("pead prompt", {"response_mime_type": "application/json"})
        run.assert_not_called()
    assert resp.text == ""


# ---------------------------------------------------------------------------
# 8. REGRESSION -- make_client()'s existing $0-metered routing-breach guard
#    (criterion 9). This step touches adjacent failure-handling code and
#    must not weaken it.
# ---------------------------------------------------------------------------


def test_make_client_routing_breach_guard_still_fires(monkeypatch):
    from backend.agents.llm_client import make_client
    from backend.config.settings import Settings

    def _broken_import():
        raise ImportError("simulated ClaudeCodeClient import failure")

    monkeypatch.setattr(ccc, "_make_claude_code_client_class", _broken_import)
    # claude_code_client.py's module __getattr__ (PEP 562) caches
    # ClaudeCodeClient into the module's own __dict__ on first successful
    # resolution, so a LATER `from ... import ClaudeCodeClient` never calls
    # __getattr__ (and never sees this monkeypatch) once anything else in
    # the process has already touched ccc.ClaudeCodeClient once -- which
    # other tests in this same file do. Clear the cache so the import is
    # forced through __getattr__ (and therefore through the patched
    # _make_claude_code_client_class) again, regardless of test order.
    monkeypatch.delattr(ccc, "ClaudeCodeClient", raising=False)
    settings = Settings(
        paper_use_claude_code_route=True,
        anthropic_api_key="sk-test-not-real",
    )
    with pytest.raises(ValueError, match="Routing breach"):
        make_client("claude-haiku-4-5", None, settings)


# ---------------------------------------------------------------------------
# 9. Observability (criterion 10)
# ---------------------------------------------------------------------------


def test_cooldown_state_is_readable_via_rail_guard_status():
    c = classify_limit_failure("You've hit your Opus limit - resets 3:45pm")
    cooldown_record_hit(c)
    status = rail_guard_status()
    assert status["cooldown_active"] is True
    assert status["cooldown_kind"] == "opus"
    assert "cooldown_retry_at" in status


# ---------------------------------------------------------------------------
# 10. MUTATION cells named in the immutable criteria (criteria 5 and 6)
# ---------------------------------------------------------------------------


def test_MUTATION_removing_limit_detection_reverts_to_generic_breaker(monkeypatch):
    """MUTATION (criterion 5): neutralize classify_limit_failure to always
    return None -- the SAME effect as deleting the detection code. Proves
    the cooldown depends on the NEW detection, not the old breaker: with
    detection gone, a real limit-shaped failure must (a) NOT engage the
    cooldown and (b) fall through to the old generic-counter behaviour,
    exactly like test_generic_failure_still_feeds_the_old_breaker above."""
    monkeypatch.setattr(ccc, "classify_limit_failure", lambda *a, **k: None)
    client = ccc.ClaudeCodeClient(model_name="claude-haiku-4-5")
    with patch("backend.agents.claude_code_client.subprocess.run") as run:
        run.return_value = _mock_completed(
            stdout=json.dumps({"result": "You've hit your weekly limit - resets Mon 12:00am"}),
            returncode=1,
        )
        for _ in range(3):
            client.generate_content("analyze AAPL", {})
    assert run.call_count == 3, "with detection removed, every cycle re-spawns the subprocess"
    assert cooldown_active() is False, "with detection removed, the cooldown never engages"
    assert rail_guard_status()["consecutive_failures"] == 3, "reverted to the generic counter"


def test_MUTATION_removing_the_pre_subprocess_guard_lets_subprocess_run_again(monkeypatch):
    """MUTATION (criterion 6): neutralize the cooldown pre-check inside
    claude_code_invoke (simulated by forcing cooldown_status() to always
    report inactive at the exact call site the guard reads) -- with the
    guard gone, an ACTIVE cooldown must no longer prevent a subprocess
    spawn, proving the guard (independent of detection) is load-bearing."""
    c = classify_limit_failure("You've hit your session limit - resets 3:45pm")
    cooldown_record_hit(c)
    assert cooldown_active() is True  # control: cooldown genuinely active before the mutation

    monkeypatch.setattr(ccc, "cooldown_status", lambda: {"cooldown_active": False})
    with patch("backend.agents.claude_code_client.subprocess.run") as run:
        run.return_value = _mock_completed(stdout=json.dumps({"subtype": "success", "result": "ok"}))
        claude_code_invoke("analyze AAPL")
    # with the guard neutralized, the subprocess runs despite the active cooldown
    run.assert_called_once()
