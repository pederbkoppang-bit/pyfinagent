"""phase-36.8 -- a fresh anchor must outrank stale archived history.

`_audit_source_paths` merges the archives UNCONDITIONALLY on every boot and the
`peak_update` replay `max()`es across the merged stream (36.7's fix, which is correct
for a ratchet). So a stale archived peak beats a fresh, intentional, LOWER one -- and
with `reset_peak` DARK and `update_peak` ratchet-only, nothing can lower it again:
`trailing_dd_pct` computes against a phantom high-water mark, `any_breached` fires,
and `flatten_all` + `pause` trip on the first cycle after restart, permanently. That
is the engine-stop phase-69.1 exists to prevent, re-opened from the replay side.

FIX SHAPE (research-led, `handoff/current/research_brief_36.8.md`): an IN-STREAM
AUTHORITY BOUNDARY. `update_peak` marks the anchor-from-`None` case on its row; the
replay ASSIGNS at a marked row and RATCHETS everywhere else; unmarked rows keep
36.7's behaviour byte-for-byte. Not live-file precedence -- PostgreSQL explicitly
prefers the archive over the live directory, which refutes that shape.

Contract: handoff/current/contract_36.8.md

ISOLATION: every test redirects `ks._AUDIT_PATH` to tmp. The autouse fixture below is
ported from the 36.7 module -- an evaluator wrote 54 rows into the LIVE audit file
today by constructing the module singleton against the real path, so this brace is
not decorative.
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def _live_audit_file_is_write_protected():
    """Nothing in THIS module may append to the real audit trail."""
    live = REPO_ROOT / "handoff" / "kill_switch_audit.jsonl"
    before = live.read_bytes() if live.exists() else None
    yield
    after = live.read_bytes() if live.exists() else None
    assert after == before, (
        f"phase-36.8: a test in this module wrote to the LIVE audit trail {live}."
    )


@pytest.fixture
def ks_tmp_audit(tmp_path, monkeypatch):
    """Point kill_switch's audit I/O at a tmp tree; the archive dir is DERIVED."""
    import backend.services.kill_switch as ks

    live = tmp_path / "handoff" / "kill_switch_audit.jsonl"
    archive = tmp_path / "handoff" / "audit"
    live.parent.mkdir(parents=True, exist_ok=True)
    archive.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(ks, "_AUDIT_PATH", live)
    assert ks._audit_archive_dir() == archive, "archive dir must follow _AUDIT_PATH"
    yield ks, live, archive


def _row(ts: str, event: str, **fields) -> str:
    return json.dumps({"ts": ts, "event": event, **fields})


def _write(path: Path, *lines: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _detached(ks, complete: bool):
    state = object.__new__(ks.KillSwitchState)
    state._lock = threading.Lock()
    state._paused = False
    state._pause_reason = None
    state._sod_nav = None
    state._sod_date = None
    state._peak_nav = None
    state._paused_at = None
    state._auto_resume_alerted_at = None
    state._history_complete = complete
    return state



# ── criterion 1: a fresh MARKED anchor outranks a higher archived peak ──────


def test_phase_36_8_a_fresh_marked_anchor_beats_a_higher_archived_peak(ks_tmp_audit):
    """THE DEFECT. Archive holds 24666.57; the live file then anchors fresh at
    18000.0 with the anchor MARKED. The restore must honour the marked anchor.

    Pre-fix this returned 24666.57 -- the stale archived peak -- and the book would
    have been flattened and paused permanently against a phantom high-water mark.
    """
    ks, live, archive = ks_tmp_audit
    _write(archive / "kill_switch_audit-v3.jsonl",
           _row("2026-06-03T10:00:00+00:00", "peak_update", nav=24666.57))
    _write(live,
           _row("2026-07-26T10:00:00+00:00", "peak_update", nav=18000.0,
                anchor=True, prior_peak=24666.57))

    assert ks.KillSwitchState().snapshot()["peak_nav"] == 18000.0


def test_phase_36_8_rows_after_a_marked_anchor_ratchet_up_from_it(ks_tmp_audit):
    """The boundary resets the fold; it does not freeze it. Later rows still ratchet."""
    ks, live, archive = ks_tmp_audit
    _write(archive / "kill_switch_audit-v3.jsonl",
           _row("2026-06-03T10:00:00+00:00", "peak_update", nav=24666.57))
    _write(live,
           _row("2026-07-26T10:00:00+00:00", "peak_update", nav=18000.0,
                anchor=True, prior_peak=24666.57),
           _row("2026-07-26T11:00:00+00:00", "peak_update", nav=19000.0),
           _row("2026-07-26T12:00:00+00:00", "peak_update", nav=18500.0))

    assert ks.KillSwitchState().snapshot()["peak_nav"] == 19000.0


def test_phase_36_8_a_marked_anchor_does_not_apply_retroactively(ks_tmp_audit):
    """ts-order is the authority. An anchor EARLIER than a later legitimate ratchet
    must not suppress it -- the boundary only supersedes what precedes it."""
    ks, live, archive = ks_tmp_audit
    _write(archive / "kill_switch_audit-v3.jsonl",
           _row("2026-06-03T10:00:00+00:00", "peak_update", nav=24666.57))
    _write(live,
           _row("2026-05-01T10:00:00+00:00", "peak_update", nav=9000.0,
                anchor=True, prior_peak=24666.57))

    # The anchor is OLDER than the archived ratchet, so the archived row wins.
    assert ks.KillSwitchState().snapshot()["peak_nav"] == 24666.57


@pytest.mark.parametrize(
    "truthy", ["yes", 1, "true", ["x"]],
    ids=["string-yes", "int-1", "string-true", "nonempty-list"],
)
def test_phase_36_8_only_a_literal_true_grants_authority(ks_tmp_audit, truthy):
    """`row.get("anchor") is True`, never a truthiness test. A row carrying some OTHER
    truthy `anchor` value -- a future schema change, a hand-edited trail, a different
    writer -- must NOT acquire the power to lower the high-water mark. Without the
    identity check these rows would assign 18000.0 over the true 24666.57."""
    ks, live, archive = ks_tmp_audit
    _write(archive / "kill_switch_audit-v3.jsonl",
           _row("2026-06-03T10:00:00+00:00", "peak_update", nav=24666.57))
    # A VALID prior_peak, so the ONLY thing that can reject this row is the `is True`
    # identity check. Without it the cycle-5 redesign masked this guard: the rows had
    # no prior_peak, so the naming clause rejected them and the identity clause was
    # never exercised (mutant M3 SURVIVED -- caught by re-running the matrix).
    _write(live,
           _row("2026-07-26T10:00:00+00:00", "peak_update", nav=18000.0,
                anchor=truthy, prior_peak=24666.57))

    assert ks.KillSwitchState().snapshot()["peak_nav"] == 24666.57


@pytest.mark.parametrize(
    "prior", [None, 0, 0.0, -1.0, "n/a", float("inf"), float("nan"), "__absent__"],
    ids=["null", "zero-int", "zero-float", "negative", "unparseable", "inf", "nan", "key-absent"],
)
def test_phase_36_8_an_anchor_that_NAMES_NOTHING_has_no_authority(ks_tmp_audit, prior):
    """THE STRUCTURAL GUARD -- this is what closes all five routes at once.

    Five independent Q/A passes each found a new way for a lost-history anchor to
    claim authority: no gate at all; a gate keyed on the wrong signal; a chmod-000
    dir where `glob` returns empty WITHOUT raising; unparseable lines; and
    present-but-silent sources (a 0-byte file, an ABSENT LIVE FILE, an unglobbed
    name, a nested subdir). Each fix closed one remembered failure mode, and the
    cycle-4 Q/A named the real problem: *"closing hole #5 by hand is the same move
    that produced holes #2-#5."*

    So authority is no longer derived from the ABSENCE of a failure anyone thought
    to check for. It is derived POSITIVELY: a row may lower the high-water mark only
    if it NAMES what it superseded. Every one of the five routes produces an anchor
    over a peak of `None` -- there was nothing to observe, so nothing can be named,
    so none of them can ever be authoritative, no matter how the source failed.
    """
    ks, live, archive = ks_tmp_audit
    _write(archive / "kill_switch_audit-v3.jsonl",
           _row("2026-06-03T10:00:00+00:00", "peak_update", nav=24666.57))
    fields = {"nav": 18000.0, "anchor": True}
    if prior != "__absent__":
        fields["prior_peak"] = prior
    _write(live, _row("2026-07-26T10:00:00+00:00", "peak_update", **fields))

    assert ks.KillSwitchState().snapshot()["peak_nav"] == 24666.57, (
        "an anchor that cannot name what it superseded must never lower the peak"
    )


def test_phase_36_8_no_production_path_can_write_an_authoritative_anchor(ks_tmp_audit):
    """The honest consequence, asserted rather than left implicit: with authority
    requiring a named prior, and `update_peak` only ever anchoring FROM `None`, no
    production writer can emit an authoritative anchor at all. The authorized
    re-anchor is `peak_reset`, which is token-gated and DARK -- exactly what the
    research found industry practice to be. This branch is deliberately
    production-dead today; the replay honours it so an operator-authored or
    future token-gated writer can use it."""
    ks, live, _archive = ks_tmp_audit
    for complete in (True, False):
        state = _detached(ks, complete=complete)
        state.update_peak(18000.0)
    rows = [json.loads(l) for l in live.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert rows, "the writer wrote nothing"
    assert not any(r.get("anchor") is True for r in rows), (
        "update_peak must never emit an authoritative anchor -- it anchors only from "
        "None, which by construction supersedes nothing"
    )


# ── criterion 2: 36.7's true-peak restore is byte-preserved ─────────────────


def test_phase_36_8_unmarked_rows_still_ratchet_exactly_as_phase_36_7(ks_tmp_audit):
    """36.7's defect must STAY fixed: a higher TRUE historical peak in the archives
    beats a later, lower, UNMARKED live row. All 20 real rows are unmarked, so this
    is the behaviour the live book depends on today."""
    ks, live, archive = ks_tmp_audit
    _write(archive / "kill_switch_audit-v3.jsonl",
           _row("2026-06-03T10:00:00+00:00", "peak_update", nav=24666.57))
    _write(archive / "kill_switch_audit-v2.jsonl",
           _row("2026-06-22T10:00:00+00:00", "peak_update", nav=24124.77))
    _write(live, _row("2026-07-24T10:00:00+00:00", "peak_update", nav=23838.19))

    # Assignment-replay would give 23838.19; the ratchet gives the true mark.
    assert ks.KillSwitchState().snapshot()["peak_nav"] == 24666.57


def test_phase_36_8_the_real_corpus_still_restores_the_true_peak(ks_tmp_audit):
    """Criterion 2 against the REAL row shapes, not only synthetic ones: the measured
    corpus has 20 unmarked peak_update rows and its true peak is 24666.57 in the
    OLDEST file. Copy the real rows into the tmp tree (read-only on the originals)."""
    ks, live, archive = ks_tmp_audit
    real_live = REPO_ROOT / "handoff" / "kill_switch_audit.jsonl"
    real_archive_dir = REPO_ROOT / "handoff" / "audit"

    peaks = []
    for src in sorted(real_archive_dir.glob("kill_switch_audit*.jsonl")) + [real_live]:
        if not src.exists():
            continue
        dest = (archive / src.name) if src != real_live else live
        dest.write_bytes(src.read_bytes())
        for line in src.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("event") == "peak_update" and r.get("nav"):
                peaks.append(float(r["nav"]))

    if not peaks:
        pytest.skip("no peak_update rows in the real corpus to assert against")
    assert ks.KillSwitchState().snapshot()["peak_nav"] == pytest.approx(max(peaks))


# ── the guarded helper: BOTH assignment branches ────────────────────────────


@pytest.mark.parametrize(
    "bad", [None, 0, 0.0, -1.0, "n/a", float("inf"), float("nan")],
    ids=["null", "zero-int", "zero-float", "negative", "unparseable", "inf", "nan"],
)
def test_phase_36_8_a_malformed_marked_anchor_is_ignored_not_applied(ks_tmp_audit, bad):
    """A marked anchor carrying a value that cannot be a baseline must be IGNORED,
    never assigned. Without the guard it would null the peak and disarm the trailing
    leg -- 36.8 must not ship a second assignment branch as unguarded as the first."""
    ks, live, archive = ks_tmp_audit
    _write(archive / "kill_switch_audit-v3.jsonl",
           _row("2026-06-03T10:00:00+00:00", "peak_update", nav=24666.57))
    _write(live,
           _row("2026-07-26T10:00:00+00:00", "peak_update", nav=bad,
                anchor=True, prior_peak=24666.57))

    assert ks.KillSwitchState().snapshot()["peak_nav"] == 24666.57


@pytest.mark.parametrize(
    "bad", [None, 0.0, -5.0, "x", float("inf"), float("nan")],
    ids=["null", "zero", "negative", "unparseable", "inf", "nan"],
)
def test_phase_36_8_a_malformed_peak_reset_is_ignored_not_applied(ks_tmp_audit, bad):
    """The SAME guard on the pre-existing `peak_reset` branch. Before this, a single
    malformed row nulled the peak; a later ratchet then healed it DOWNWARD to
    whatever came next, silently discarding the true high-water mark."""
    ks, live, archive = ks_tmp_audit
    _write(archive / "kill_switch_audit-v3.jsonl",
           _row("2026-06-03T10:00:00+00:00", "peak_update", nav=24666.57))
    _write(live,
           _row("2026-07-26T10:00:00+00:00", "peak_reset", old_peak=24666.57,
                new_peak=bad, trigger="malformed"))

    assert ks.KillSwitchState().snapshot()["peak_nav"] == 24666.57


def test_phase_36_8_a_VALID_peak_reset_still_assigns_downward(ks_tmp_audit):
    """Criterion 4's neighbour: phase-69.1's authoritative downward move must keep
    working. The guard rejects malformed values ONLY."""
    ks, live, archive = ks_tmp_audit
    _write(archive / "kill_switch_audit-v3.jsonl",
           _row("2026-06-03T10:00:00+00:00", "peak_update", nav=24666.57))
    _write(live,
           _row("2026-07-26T10:00:00+00:00", "peak_reset", old_peak=24666.57,
                new_peak=20000.0, trigger="resume:manual", operator="peder"))

    assert ks.KillSwitchState().snapshot()["peak_nav"] == 20000.0


# ── the writer marks the anchor ─────────────────────────────────────────────


def test_phase_36_8_update_peak_marks_an_anchor_but_not_a_ratchet(ks_tmp_audit):
    """The writer is the only place the two cases are distinguishable."""
    ks, live, _archive = ks_tmp_audit
    # `complete=True` states the premise explicitly: a replay that saw every source
    # and still found no peak. The class default is False on purpose -- a state built
    # without declaring completeness must NOT stamp authority, and that default is
    # what caught this test constructing one implicitly.
    state = _detached(ks, complete=True)

    state.update_peak(18000.0)   # anchor-from-None
    state.update_peak(19000.0)   # ordinary ratchet
    state.update_peak(17000.0)   # no-op, must write nothing

    rows = [json.loads(l) for l in live.read_text(encoding="utf-8").splitlines() if l.strip()]
    peaks = [r for r in rows if r.get("event") == "peak_update"]
    assert len(peaks) == 2, f"expected 2 rows (anchor + ratchet), got {peaks}"
    # CYCLE-5 REDESIGN: an anchor-from-None cannot NAME what it superseded (there was
    # nothing to observe), so the writer never stamps authority on it -- not even after
    # a complete replay. Five Q/A passes each found a new way for such an anchor to
    # claim authority; this removes the capability instead of the routes.
    assert peaks[0].get("anchor") is not True, (
        "an anchor over a peak of None must never be authoritative"
    )
    assert peaks[0]["nav"] == 18000.0
    assert peaks[1].get("anchor") is not True, "a ratchet must NOT be marked as an anchor"
    assert peaks[1]["nav"] == 19000.0


# ── the WRITER→REPLAY round trip: the route that had zero coverage ─────────


def test_phase_36_8_a_transient_archive_failure_cannot_destroy_the_true_peak(ks_tmp_audit):
    """THE CYCLE-1 SAFETY REGRESSION, now guarded.

    The cycle-1 Q/A executed this exact sequence against the REAL corpus and watched
    the true 24666.57 be destroyed permanently:

      boot A (archives readable)      -> peak 24666.57
      boot B (archive scan FAILS)     -> peak None, cycle calls update_peak(18000.0)
      boot C (archives readable again)-> peak 18000.0   <-- WRONG, and unrecoverable

    A LOWER peak means LESS protection, `reset_peak` is DARK and `update_peak` only
    ratchets up, so nothing could restore it. The marker was being stamped on the
    same `_peak_nav is None` trigger that fires 36.12's ACCIDENT marker -- which is
    verbatim the reason this step's own contract refused to let 36.12's event carry
    authority. The fix gates the stamp on `_history_complete`.
    """
    ks, live, archive = ks_tmp_audit
    _write(archive / "kill_switch_audit-v3.jsonl",
           _row("2026-06-03T10:00:00+00:00", "peak_update", nav=24666.57))

    # boot B: the replay could NOT see the archives, so any anchor it writes is over
    # history it merely failed to read.
    incomplete = _detached(ks, complete=False)
    incomplete.update_peak(18000.0)

    rows = [json.loads(l) for l in live.read_text(encoding="utf-8").splitlines() if l.strip()]
    written = [r for r in rows if r.get("event") == "peak_update"]
    assert written and written[-1].get("anchor") is not True, (
        "an anchor written while the replay was INCOMPLETE must not claim authority"
    )

    # boot C: archives readable again -> the true mark must come back.
    assert ks.KillSwitchState().snapshot()["peak_nav"] == 24666.57


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores chmod, so the probe cannot fail")
def test_phase_36_8_an_UNLISTABLE_archive_dir_is_incomplete_not_empty(ks_tmp_audit):
    """The cycle-2 Q/A's finding, as a guard.

    `Path.glob` SWALLOWS a directory PermissionError and returns empty, so an
    unlistable archive dir looks exactly like an empty one: `is_dir()` stays True and
    a stat-only check reports a COMPLETE history it never read. The Q/A executed
    `chmod 000 handoff/audit/` against the real corpus and watched boot C return
    18000.0 instead of 24666.57 -- the true mark destroyed, unrecoverably.
    """
    ks, live, archive = ks_tmp_audit
    _write(archive / "kill_switch_audit-v3.jsonl",
           _row("2026-06-03T10:00:00+00:00", "peak_update", nav=24666.57))
    _write(live, _row("2026-07-01T10:00:00+00:00", "sod_snapshot", nav=1.0, date="2026-07-01"))

    archive.chmod(0o000)
    try:
        state = ks.KillSwitchState()
        assert state._history_complete is False, (
            "an archive dir that cannot be listed must NOT report a complete history "
            "-- glob() returns empty without raising, which is why is_dir() is not enough"
        )
    finally:
        archive.chmod(0o755)


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores chmod, so the probe cannot fail")
def test_phase_36_8_an_unreadable_archive_FILE_is_incomplete(ks_tmp_audit):
    """The other half of the completeness fix. Mutant MX4 (deleting `complete = False`
    from the per-file read-failure handler) SURVIVED the cycle-2 suite, so the
    unreadable-FILE branch had no killing test even though its differential is the
    same destroyed peak."""
    ks, live, archive = ks_tmp_audit
    bad = archive / "kill_switch_audit-v3.jsonl"
    _write(bad, _row("2026-06-03T10:00:00+00:00", "peak_update", nav=24666.57))
    _write(live, _row("2026-07-01T10:00:00+00:00", "sod_snapshot", nav=1.0, date="2026-07-01"))

    bad.chmod(0o000)
    try:
        assert ks.KillSwitchState()._history_complete is False, (
            "a source file that could not be opened means the replay is partial"
        )
    finally:
        bad.chmod(0o644)


def test_phase_36_8_an_UNPARSEABLE_source_is_incomplete_not_empty(ks_tmp_audit):
    """The FOURTH route, found by the cycle-3 Q/A. A file that OPENS fine but whose
    lines are not JSON is history we cannot see -- identical in consequence to a file
    we cannot read. The per-file handler recorded that; the per-LINE handlers dropped
    the rows silently and left `complete` True, so the anchor claimed authority and
    the true peak was destroyed once the file became parseable again.

    Same invariant as the other three routes: *complete is False whenever this replay
    may be missing history it cannot see.*
    """
    ks, live, archive = ks_tmp_audit
    (archive / "kill_switch_audit-v3.jsonl").write_text(
        "this is not json\n{ broken\n", encoding="utf-8")
    _write(live, _row("2026-07-01T10:00:00+00:00", "sod_snapshot", nav=1.0, date="2026-07-01"))

    assert ks.KillSwitchState()._history_complete is False, (
        "unparseable lines are missing history, not absent history"
    )


def test_phase_36_8_a_non_dict_json_row_is_also_incomplete(ks_tmp_audit):
    """The other half of the same handler: a line that parses but is not an object."""
    ks, live, archive = ks_tmp_audit
    (archive / "kill_switch_audit-v3.jsonl").write_text('["a", "list"]\n42\n', encoding="utf-8")
    _write(live, _row("2026-07-01T10:00:00+00:00", "sod_snapshot", nav=1.0, date="2026-07-01"))

    assert ks.KillSwitchState()._history_complete is False


def test_phase_36_8_an_unparseable_archive_cannot_authorise_an_anchor(ks_tmp_audit):
    """End-to-end, the Q/A's executed differential: boot over an unparseable archive,
    anchor, then boot again once it is readable. The true peak must survive."""
    ks, live, archive = ks_tmp_audit
    bad = archive / "kill_switch_audit-v3.jsonl"
    good = _row("2026-06-03T10:00:00+00:00", "peak_update", nav=24666.57)
    bad.write_text("not json at all\n", encoding="utf-8")

    booted = ks.KillSwitchState()
    assert booted.snapshot()["peak_nav"] is None
    booted.update_peak(18000.0)

    rows = [json.loads(l) for l in live.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert rows[-1].get("anchor") is not True, (
        "an anchor written over unparseable history must not claim authority"
    )

    _write(bad, good)  # the file becomes parseable again
    assert ks.KillSwitchState().snapshot()["peak_nav"] == 24666.57


def test_phase_36_8_the_class_default_for_history_complete_is_conservative():
    """Mutant MX3 (flipping the class default to True) SURVIVED cycle 2, yet the
    artifacts credit this default as load-bearing safety. A state built without a
    replay -- or one whose replay raised -- has verified nothing, so it must not
    stamp authority."""
    import backend.services.kill_switch as ks

    assert ks.KillSwitchState._history_complete is False
    detached = object.__new__(ks.KillSwitchState)
    assert detached._history_complete is False


def test_phase_36_8_an_anchor_after_a_COMPLETE_replay_still_claims_NO_authority(ks_tmp_audit):
    """CYCLE-5 REDESIGN. Previously a complete replay was enough to stamp authority.
    It is not: an anchor from `None` supersedes nothing, so it can name nothing, so it
    gets nothing. Authority now comes only from a row that NAMES a real prior peak."""
    ks, live, archive = ks_tmp_audit
    complete = _detached(ks, complete=True)
    complete.update_peak(18000.0)

    rows = [json.loads(l) for l in live.read_text(encoding="utf-8").splitlines() if l.strip()]
    written = [r for r in rows if r.get("event") == "peak_update"]
    # Even on a COMPLETE replay: nothing was superseded, so nothing is named, so no
    # authority. The completeness flag is now a diagnostic, not the authority gate.
    assert written[-1].get("anchor") is not True


def test_phase_36_8_a_real_replay_sets_history_complete_both_ways(ks_tmp_audit):
    """`_history_complete` must be DERIVED by the replay, not assumed. With the
    archive dir present it is True; with it absent -- indistinguishable from a lost
    mount -- it is False, the conservative reading."""
    ks, live, archive = ks_tmp_audit
    _write(live, _row("2026-07-01T10:00:00+00:00", "peak_update", nav=1000.0))
    assert ks.KillSwitchState()._history_complete is True

    for f in archive.iterdir():
        f.unlink()
    archive.rmdir()
    assert ks.KillSwitchState()._history_complete is False


# ── criterion 4: reset_peak stays DARK ─────────────────────────────────────


def test_phase_36_8_reset_peak_is_still_dark_by_default(ks_tmp_audit):
    """Untouched by this step: no `peak_reset` row may be written while the flag is
    off, and the call must return None."""
    ks, live, _archive = ks_tmp_audit
    state = object.__new__(ks.KillSwitchState)
    state._lock = threading.Lock()
    state._peak_nav = 24666.57
    state._paused = False
    state._pause_reason = None
    state._sod_nav = None
    state._sod_date = None
    state._paused_at = None
    state._auto_resume_alerted_at = None

    assert state.reset_peak(1.0, trigger="test-should-be-dark") is None
    assert state._peak_nav == 24666.57
    rows = [json.loads(l) for l in live.read_text(encoding="utf-8").splitlines()] if live.exists() else []
    assert [r for r in rows if r.get("event") == "peak_reset"] == []


# ── criterion 3: the archives are protected, not capped ────────────────────


def test_phase_36_8_both_housekeeping_scripts_protect_the_audit_archives():
    """Criterion 3's chosen answer. A cap would be actively dangerous -- MEASURED:
    the true peak 24666.57 lives in the OLDEST file, so pruning oldest-first would
    delete the row the switch depends on. So the archives are declared
    do-not-prune in BOTH housekeeping scripts, and the two declarations must agree.
    """
    import ast

    pats = {}
    for name in ("backfill_handoff_archive.py", "verify_handoff_layout.py"):
        src = (REPO_ROOT / "scripts" / "housekeeping" / name).read_text(encoding="utf-8")
        tree = ast.parse(src)
        found = None
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "AUDIT_KEEP_GLOBS" for t in node.targets
            ):
                found = ast.literal_eval(node.value)
        assert found, f"{name} does not declare AUDIT_KEEP_GLOBS"
        pats[name] = set(found)

    a, b = pats.values()
    assert a == b, f"the two housekeeping allowlists disagree: {pats}"
    assert any("kill_switch_audit" in p for p in a), (
        "the audit archives that determine the live trailing-DD baseline are not "
        "declared do-not-prune"
    )
