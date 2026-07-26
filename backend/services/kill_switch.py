"""
kill_switch -- Production-grade pause / resume / flatten-all for paper trading.

Defaults derived from prop-trading practitioner consensus (see RESEARCH.md
Phase 4.5 step 4.5.7):

  - daily_loss_limit_pct = 4% of start-of-day NAV (modal across FTMO 5%,
    FXIFY 4%, Alpha Capital 4%, FundedNext 4%; Van Tharp 2%/trade x 2 = 4%).
  - trailing_dd_limit_pct = 10% EOD from rolling high-water mark (upper-end
    standard for long-only unlevered equity; Maven/Audacity/PropVator).

Breach semantics (FINRA Rule 15c3-5 "hard block" pattern):
  - Pause = halt new entries; existing positions kept.
  - Flatten = close every open position at market; cancel pending orders.
  - Limit breach => auto-flatten + auto-pause + audit log; explicit human
    resume required once both limits read healthy.

Audit trail (mandatory per 3forge / ESMA Supervisory Briefing 2026):
  Every state transition appends a JSON line to handoff/kill_switch_audit.jsonl
  with {timestamp, event, trigger, details}.

  phase-36.7: that file is also this module's ONLY persistence -- the loss
  baselines (`sod_snapshot` / `peak_update`) are replayed from it at every
  process start. Writes still go to the single live file; READS now merge the
  live file with any rotated copies in handoff/audit/ (`kill_switch_audit*.jsonl`)
  in row-`ts` order, because a housekeeping sweep that relocated the live file
  left the switch DISARMED after every restart. Do NOT move
  handoff/kill_switch_audit.jsonl -- both housekeeping scripts allowlist it.
"""

from __future__ import annotations

import json
import math
import os

from backend.utils import json_io
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

_AUDIT_PATH = Path(__file__).resolve().parents[2] / "handoff" / "kill_switch_audit.jsonl"
_AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)

# phase-36.7: ROTATION-AWARE BASELINE RESTORE.
#
# `_load_from_audit` used to read ONLY `_AUDIT_PATH`. The housekeeping sweep
# (scripts/housekeeping/backfill_handoff_archive.py) classifies that LIVE STATE
# FILE as layout dirt and `shutil.move`s it into `handoff/audit/` with a `-vN`
# suffix, so every rotation left the loader with a file holding nothing but
# pause/resume rows -- zero `sod_snapshot`, zero `peak_update`. Measured on
# 2026-07-26: sod_nav=None / peak_nav=None, and a 50% drawdown returned
# any_breached=False. Git shows the rotation already happened twice
# (fa9aaf8e -> -v3, 77bc7db5 -> -v4).
#
# The fix reads the rotated files too. Two traps a naive version falls into:
#   1. FILE ORDER IS NOT TIME ORDER. `-vN` reflects discovery order in
#      `_safe_target`, not chronology. Rows are merged and sorted by `ts`.
#   2. HARDCODING `-v4` EXPIRES AT `-v5`. We glob.
# The root cause itself is fixed in the two housekeeping scripts, which now
# allowlist this file; reading the archives is the belt to that braces, and
# recovers the baselines the two historical rotations already stranded.
# DERIVED from _AUDIT_PATH, never a second independent constant: the sweep moves
# the live file into `<its own dir>/audit/`, and a caller that monkeypatches
# `_AUDIT_PATH` to a tmp file (backend/tests/test_dod4_tier1_coverage_investment.py
# does exactly this) must get FULL isolation, not a tmp live file silently merged
# with the real production archives.
_AUDIT_ARCHIVE_SUBDIR = "audit"
_AUDIT_ARCHIVE_GLOB = "kill_switch_audit*.jsonl"

# One-shot-per-process guard for the disarmed-baseline ERROR log so a polled
# GET (/api/paper-trading/kill-switch) cannot spam the log file.
#
# DELIBERATELY LOCK-FREE. A racing check-and-set can emit the line twice, which
# is entirely benign, and this is the one module in the codebase with a
# documented non-reentrant-lock deadlock (phase-23.1.22, found via a
# faulthandler SIGUSR1 dump on a hung backend) -- adding a fourth lock here to
# de-duplicate a log line is not a trade worth making. It also keeps the
# phase-23.2.14 `threading.Lock()` roster regression-lock byte-untouched.
_disarmed_logged = False


def _audit_archive_dir() -> Path:
    """Where the housekeeping sweep relocates the live state file."""
    return _AUDIT_PATH.parent / _AUDIT_ARCHIVE_SUBDIR


def _audit_source_paths() -> list[Path]:
    """Every file `_load_from_audit` replays, live file LAST.

    Reads `_AUDIT_PATH` at call time (not import time) so a test that
    monkeypatches it gets the archive dir moved with it. Ordering here is only a
    stable tie-break -- the authoritative ordering is the row `ts` sort in
    `_load_from_audit`.
    """
    paths: list[Path] = []
    archive = _audit_archive_dir()
    try:
        if archive.is_dir():
            paths.extend(sorted(archive.glob(_AUDIT_ARCHIVE_GLOB)))
    except Exception as e:  # unreadable dir must never break boot
        logger.warning(f"kill_switch: audit archive scan failed: {e}")
    if _AUDIT_PATH.exists() and _AUDIT_PATH not in paths:
        paths.append(_AUDIT_PATH)
    return paths


def _coerce_nav(value: Any) -> Optional[float]:
    """Parse a NAV field from an audit row. Returns None when the value is
    absent, unparseable, or non-positive.

    Replaces the `float(x or 0.0) or None` idiom, which mapped a legitimate
    0.0 to None by accident and let a negative NAV through as a usable
    baseline. A funded paper book's NAV is never <=0, so None (which now
    raises the loud disarmed state) is the conservative reading.
    """
    if value is None:
        return None
    try:
        nav = float(value)
    except (TypeError, ValueError):
        return None
    # phase-36.7 hardening: `inf > 0` is True in Python, so a non-finite NAV
    # (inf/-inf/nan) previously passed as a "valid" baseline. json.dumps/loads
    # round-trip a bare `Infinity` token, so a corrupted upstream NAV reaching
    # _append_audit would be replayed as a permanent baseline on every future
    # boot -- and the new max()-ratchet in update_peak makes an inf PEAK
    # irreversible (the prior assignment-replay healed on the next sane row;
    # max() cannot heal downward). With peak_nav=inf, trailing_dd_pct is nan
    # and every `nan >= limit` comparison is False, so the trailing leg goes
    # silently, permanently dead while `armed` still reports True. Reject
    # non-finite here so it is treated as an absent baseline instead.
    if not math.isfinite(nav):
        return None
    return nav if nav > 0 else None


class KillSwitchState:
    """Module-level thread-safe state. Persisted across process restarts via
    the audit log: the most recent `pause` or `resume` line sets the resume
    state; if it's `pause` the system re-enters paused on restart.

    phase-36.7: the replay reads the live audit file AND its rotated copies,
    merged in row-`ts` order -- see `_read_audit_rows`. `peak_update` rows
    RATCHET (max), matching `update_peak`'s own invariant; `peak_reset` rows
    remain an authoritative downward move."""

    # phase-36.12: CLASS-LEVEL default, not only an instance attribute. Several
    # test fixtures build a detached state via `object.__new__(KillSwitchState)`
    # and set the known fields by hand, so a new instance-only attribute would
    # AttributeError inside `_snapshot_locked` for every one of them. A class
    # default keeps those constructions working and keeps the key always present
    # in the snapshot contract.
    _baseline_provenance: Optional[str] = None

    # phase-36.8: False until a replay proves it saw every source. Conservative
    # default: an instance built without a replay has NOT verified its history, so
    # it must not stamp an authoritative anchor.
    _history_complete: bool = False

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._paused = False
        self._pause_reason: Optional[str] = None
        self._sod_nav: Optional[float] = None  # start-of-day NAV snapshot
        self._sod_date: Optional[str] = None  # phase-23.2.19: UTC date of the SOD anchor
        self._peak_nav: Optional[float] = None  # trailing high-water mark
        # phase-38.1 (OPEN-10): auto-resume hysteresis. _paused_at carries
        # the UTC ISO timestamp of the most recent `pause` event; cleared
        # on resume. Persisted via audit log so restart-survivable.
        self._paused_at: Optional[str] = None
        # _auto_resume_alerted_at carries the timestamp of the T+1h pager
        # alert (one-shot per pause-cycle) so we don't spam Slack.
        self._auto_resume_alerted_at: Optional[str] = None
        # phase-36.12: provenance of the CURRENT baselines. None = ordinary
        # (anchored on a real first boot, or ratcheted/rolled normally).
        # "lost_history_anchor" = the baselines were unrestorable and got anchored
        # to that day's NAV, so the drawdown they measure starts from a fiction.
        # Latches until a deliberate operator `peak_reset` supersedes it. Display /
        # audit only -- it gates nothing and changes no threshold.
        self._baseline_provenance: Optional[str] = None
        self._load_from_audit()

    @staticmethod
    def _read_audit_rows() -> tuple[list[dict], bool]:
        """Merge every audit source (rotated archives + the live file) into one
        `ts`-ordered row list.

        Sort key is `(ts, source_index, line_index)`: `ts` is ISO-8601 UTC so it
        sorts lexicographically, and the two positional components make the sort
        stable/deterministic for rows sharing a timestamp. Rows with no `ts`
        collate FIRST (empty string) so a timestamp-less row can never override a
        genuinely later one.
        """
        rows: list[dict] = []
        keyed: list[tuple[str, int, int, dict]] = []
        # phase-36.8: `complete` is False whenever this replay may be missing history
        # it cannot see. It is NOT a health flag -- it gates whether a later
        # anchor-from-None may claim AUTHORITY to lower the high-water mark. An
        # absent archive dir counts as incomplete because a brand-new install and a
        # lost mount are indistinguishable from here, and only one of them is safe.
        complete = True
        archive = _audit_archive_dir()
        try:
            if not archive.is_dir():
                complete = False
            else:
                # phase-36.8 (cycle-3): LIST it, do not merely stat it. `Path.glob`
                # swallows a directory PermissionError and returns EMPTY, so an
                # unlistable archive dir is indistinguishable from an empty one --
                # `is_dir()` stays True and the replay would report a COMPLETE
                # history it never actually read. A cycle-2 Q/A executed exactly that
                # (`chmod 000 handoff/audit/`) and watched the true 24666.57 be
                # destroyed. os.listdir raises where glob stays silent.
                os.listdir(archive)
        except Exception as e:
            logger.warning(f"kill_switch: audit archive not listable: {e}")
            complete = False
        for src_idx, path in enumerate(_audit_source_paths()):
            try:
                with path.open(encoding="utf-8") as f:
                    for line_idx, line in enumerate(f):
                        try:
                            row = json_io.parse_json_line(line)
                        except Exception:
                            # phase-36.8 (cycle-4): a line we cannot PARSE is history
                            # we cannot see, exactly like a file we cannot READ. The
                            # sibling handler below records that; this one used to
                            # `continue` silently, so a file that opened fine but held
                            # unparseable lines reported a COMPLETE history and let a
                            # lost-history anchor claim authority -- destroying the
                            # true peak. Same invariant, same treatment.
                            complete = False
                            continue
                        if not isinstance(row, dict):
                            complete = False
                            continue
                        keyed.append((str(row.get("ts") or ""), src_idx, line_idx, row))
            except Exception as e:
                # One unreadable/rotated-away file must not lose the others -- but it
                # DOES mean the restore is partial, so record that.
                logger.warning(f"kill_switch: audit read failed for {path}: {e}")
                complete = False
        keyed.sort(key=lambda t: (t[0], t[1], t[2]))
        rows = [t[3] for t in keyed]
        return rows, complete

    def _load_from_audit(self) -> None:
        try:
            rows, complete = self._read_audit_rows()
            # phase-36.8: remembered so `update_peak` knows whether an
            # anchor-from-None is a GENUINE absence (the replay saw everything
            # and there was nothing) or merely an UNSEEN one (a source was
            # unreadable). Only the first may claim authority to lower the peak.
            self._history_complete = complete
            for row in rows:
                event = row.get("event")
                if event == "pause":
                    self._paused = True
                    self._pause_reason = row.get("trigger")
                    # phase-38.1: capture the pause ts for hysteresis.
                    self._paused_at = row.get("ts")
                    self._auto_resume_alerted_at = None
                elif event == "resume":
                    self._paused = False
                    self._pause_reason = None
                    self._paused_at = None
                    self._auto_resume_alerted_at = None
                elif event == "auto_resume_alert":
                    # phase-38.1: T+1h pager alert went out -- record so
                    # we don't re-fire on the next cycle within the same
                    # pause window.
                    self._auto_resume_alerted_at = row.get("ts")
                elif event == "sod_snapshot":
                    self._sod_nav = _coerce_nav(row.get("nav"))
                    # phase-23.2.19: prefer explicit `date` (rows written
                    # post-fix); fall back to parsing `ts` for legacy
                    # rows that pre-date the schema bump.
                    sod_date = row.get("date")
                    if not sod_date:
                        ts = row.get("ts")
                        if ts:
                            try:
                                sod_date = datetime.fromisoformat(
                                    str(ts).replace("Z", "+00:00")
                                ).astimezone(timezone.utc).date().isoformat()
                            except Exception:
                                sod_date = None
                    self._sod_date = sod_date
                elif event == "peak_update":
                    # phase-36.7: RATCHET, don't assign. `update_peak` (the
                    # writer) enforces "never moves down", but this replay used
                    # a bare assignment -- so the last row in stream order won
                    # even when an EARLIER row held a higher mark. Measured
                    # across the live archives: assignment yields 24124.77
                    # (2026-06-22) while the true high-water mark is 24666.57
                    # (2026-06-03) -- the restore itself destroyed 541.80 of
                    # peak = 2.17pp of trailing-DD headroom. max() makes the
                    # replay agree with the writer's own invariant.
                    # phase-36.8: a MARKED anchor row is an authority boundary -- it
                    # ASSIGNS, superseding everything before it in `ts` order, and
                    # later rows ratchet up FROM it. Unmarked rows (every row written
                    # before 36.8, all 20 of them) keep the pure ratchet above, so
                    # 36.7's true-peak restore is byte-preserved. `is True` and not a
                    # truthiness test: an older row carrying some other `anchor` value
                    # must not accidentally acquire authority.
                    # phase-36.8 (cycle-5 REDESIGN): authority requires the row to
                    # NAME WHAT IT SUPERSEDED -- `prior_peak` must coerce to a positive
                    # finite NAV.
                    #
                    # STATED PRECISELY, because an earlier wording overclaimed: this
                    # validates COERCIBILITY, not TRUTH. The replay cannot verify that
                    # the named prior was ever observed, and a hand-forged row carrying
                    # any positive finite token (even `true`, which floats to 1.0) would
                    # be honoured. That is a SOUND boundary rather than a hole only
                    # because no production writer emits the field at all -- verified
                    # dynamically across every public writer -- and because the tripwire
                    # test above goes red the moment one starts. Tightening it to
                    # compare against the peak observed so far in the replay is the
                    # obvious hardening if a writer is ever added.
                    #
                    # WHY THIS SHAPE. Five independent Q/A passes each found a new way
                    # for a lost-history anchor to claim authority (no gate; gate on
                    # the wrong signal; chmod-000 dir where glob returns empty without
                    # raising; unparseable lines; present-but-silent sources incl. a
                    # 0-byte file and an absent LIVE file). Every one of those closed a
                    # remembered failure mode, and the sixth would too -- the flag was
                    # NEGATIVELY derived, so it was only ever as good as the author's
                    # imagination. This rule is POSITIVE: all five routes produce an
                    # anchor over a peak of `None`, which by construction cannot name a
                    # prior, so none of them can ever be authoritative. It also matches
                    # the research: the authorized re-anchor is `peak_reset`
                    # (token-gated), and no production path writes an intentional lower
                    # anchor -- so this branch is deliberately production-dead today.
                    if (row.get("anchor") is True
                            and _coerce_nav(row.get("prior_peak")) is not None):
                        self._apply_authoritative_peak(row.get("nav"), "peak_update:anchor")
                    else:
                        nav = _coerce_nav(row.get("nav"))
                        if nav is not None and (self._peak_nav is None or nav > self._peak_nav):
                            self._peak_nav = nav
                elif event == "peak_reset":
                    # phase-69.1 (audit item 2): audited peak reset (flatten /
                    # operator-resume). Restart-replayable + idempotent -- the
                    # reset value wins over prior peak_update rows in stream order.
                    # phase-36.7: still an AUTHORITATIVE downward move (assignment,
                    # not max) -- it must be able to lower the peak, and later
                    # peak_update rows ratchet up from the reset value because the
                    # rows are replayed in `ts` order.
                    # phase-36.8: routed through the SAME guard as the new anchor
                    # branch. Previously this assigned `_coerce_nav(...)` unguarded, so
                    # one malformed row nulled the peak -- and a later ratchet then
                    # healed it DOWNWARD to whatever came next, silently discarding the
                    # true high-water mark while `armed` stayed true (measured).
                    self._apply_authoritative_peak(row.get("new_peak"), "peak_reset")
                    # phase-36.12: a deliberate, token-gated operator re-anchor
                    # supersedes a lost-history anchor -- the operator has now
                    # chosen the mark, so the baselines are no longer a fiction
                    # nobody signed off on.
                    self._baseline_provenance = None
                elif event == "baseline_anchor_on_lost_history":
                    # phase-36.12: DOES NOT set any baseline -- `peak_reset` stays
                    # the only assignment-semantics event. It restores the
                    # PROVENANCE flag so a restart cannot launder a lost-history
                    # anchor into a clean-looking ACTIVE badge.
                    self._baseline_provenance = "lost_history_anchor"
        except Exception as e:
            logger.warning(f"kill_switch: audit load failed: {e}")

    def _apply_authoritative_peak(self, raw: Any, source: str) -> None:
        """phase-36.8: the guarded path for every assignment-semantics branch in the
        REPLAY -- NOT for every assignment to `_peak_nav`.

        Measured: there are 5 assignment sites in this module and exactly one is
        inside this helper. The replay's own ratchet, `update_peak` (2 sites) and
        `reset_peak` assign directly. `update_peak` coerces with a bare `float(nav)`,
        so a non-finite NAV reaches `_peak_nav` in memory by that route -- filed as
        masterplan step 36.19, whose reachability the executor must MEASURE rather
        than assume. Do not read this docstring as a claim that that route is guarded.

        Both assignment-semantics branches of the REPLAY route here -- `peak_reset`
        (phase-69.1's token-gated downward move) and 36.8's marked anchor. Such an
        assignment can LOWER
        the peak, so a value that cannot be a baseline must be ignored rather than
        applied: assigning `None` disarms the trailing leg on the next restart, and a
        subsequent ratchet then re-anchors from whatever row comes next, which is
        lower than the true mark and silently under-measures every future drawdown.

        Ignoring is the conservative direction here precisely because the alternative
        is a silent downward move. It logs at ERROR because a malformed row on a
        safety stream is an operator-visible event, not a debug detail.
        """
        value = _coerce_nav(raw)
        if value is None:
            logger.error(
                "kill_switch: IGNORING an authoritative peak assignment from %s -- "
                "value %r does not coerce to a positive finite NAV. The prior peak "
                "(%s) is retained; assigning this would have lowered or nulled the "
                "trailing high-water mark.",
                source, raw, self._peak_nav,
            )
            return
        self._peak_nav = value

    @staticmethod
    def _append_audit(event: str, **fields: Any) -> None:
        row = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": event,
            **fields,
        }
        try:
            with _AUDIT_PATH.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
        except Exception as e:
            logger.warning(f"kill_switch: audit write failed: {e}")

    # ── State getters ──────────────────────────────────────────────
    def is_paused(self) -> bool:
        with self._lock:
            return self._paused

    def _snapshot_locked(self) -> dict:
        """phase-23.1.22: lock-free snapshot helper. Caller MUST already hold
        self._lock. Used by pause()/resume() which re-entered the same
        threading.Lock() via snapshot() and deadlocked the entire process.
        Found via faulthandler SIGUSR1 dump on a live hung backend.

        phase-23.2.19: includes sod_date so callers can decide whether to
        re-anchor SOD on a new UTC calendar day.

        phase-38.1: includes paused_at + auto_resume_alerted_at for
        hysteresis logic in check_auto_resume."""
        return {
            "paused": self._paused,
            "pause_reason": self._pause_reason,
            "sod_nav": self._sod_nav,
            "sod_date": self._sod_date,
            "peak_nav": self._peak_nav,
            "paused_at": self._paused_at,
            "auto_resume_alerted_at": self._auto_resume_alerted_at,
            # phase-36.12: None normally; "lost_history_anchor" when the current
            # baselines were anchored because their history was unrecoverable.
            "baseline_provenance": self._baseline_provenance,
        }

    def snapshot(self) -> dict:
        with self._lock:
            return self._snapshot_locked()

    # ── State transitions ──────────────────────────────────────────
    def pause(self, trigger: str = "manual", details: Optional[dict] = None) -> dict:
        with self._lock:
            self._paused = True
            self._pause_reason = trigger
            # phase-38.1: stamp the pause timestamp for hysteresis.
            self._paused_at = datetime.now(timezone.utc).isoformat()
            self._auto_resume_alerted_at = None
            self._append_audit("pause", trigger=trigger, details=details or {})
            # phase-23.1.22: call _snapshot_locked, NOT snapshot(), to avoid
            # re-acquiring self._lock (threading.Lock is not reentrant).
            snap = self._snapshot_locked()
        # phase-23.2.18: operator alert on auto-pause. Manual / test / bench
        # triggers stay silent. Outside the lock so the webhook path can
        # block briefly without holding kill-switch state.
        _MANUAL_TRIGGERS = {"manual", "test", "test-pre", "bench-1", "bench-2", "bench-3"}
        if trigger not in _MANUAL_TRIGGERS:
            try:
                from backend.services.observability.alerting import raise_cron_alert_sync
                raise_cron_alert_sync(
                    source="kill_switch",
                    error_type=f"auto_pause_{trigger}",
                    severity="P1",
                    title=f"Kill-switch AUTO-PAUSED trading (trigger={trigger})",
                    details={
                        "trigger": trigger,
                        **{str(k): str(v) for k, v in (details or {}).items()},
                    },
                )
            except Exception as _alert_err:
                logger.warning(f"kill_switch pause-alert dispatch failed: {_alert_err}")
        return snap

    def resume(self, trigger: str = "manual", details: Optional[dict] = None,
               nav: Optional[float] = None) -> dict:
        with self._lock:
            self._paused = False
            self._pause_reason = None
            # phase-38.1: clear pause-cycle state on resume.
            self._paused_at = None
            self._auto_resume_alerted_at = None
            self._append_audit("resume", trigger=trigger, details=details or {})
            # phase-23.1.22: call _snapshot_locked, NOT snapshot(), to avoid
            # re-acquiring self._lock (threading.Lock is not reentrant).
            snap = self._snapshot_locked()
        # phase-69.1 (audit item 2): re-anchor the trailing peak to the current
        # NAV on an operator resume, so a monotonic peak can no longer keep the
        # trailing-DD breach alive forever after a flatten (the resume endpoint
        # passes `nav`). DARK: reset_peak is a no-op until KS-PEAK-RESET: APPROVED.
        # Called OUTSIDE the with-block -- reset_peak takes self._lock, which is
        # non-reentrant. If it fired, prefer its post-reset snapshot.
        if nav is not None and nav > 0:
            rp = self.reset_peak(float(nav), trigger=f"resume:{trigger}",
                                 operator=(details or {}).get("operator"))
            if rp is not None:
                snap = rp
        return snap

    def update_sod_nav(self, nav: float, date: Optional[str] = None) -> None:
        """Record start-of-day NAV for daily-loss calculation.

        phase-23.2.19: now stamps the UTC `date` alongside `nav`. Caller
        passes today's UTC ISO date (`datetime.now(timezone.utc).date().isoformat()`)
        when re-anchoring on a new calendar day; default None falls back
        to today. The audit row gets both `nav` and `date` so a future
        boot replay can detect daily-roll boundaries without parsing `ts`.
        """
        if date is None:
            date = datetime.now(timezone.utc).date().isoformat()
        with self._lock:
            self._sod_nav = float(nav)
            self._sod_date = date
            self._append_audit("sod_snapshot", nav=self._sod_nav, date=self._sod_date)

    def update_peak(self, nav: float) -> None:
        """Ratchet the trailing high-water mark upward. Never moves down.

        phase-36.8 (as shipped after the cycle-5 redesign): this method writes a PLAIN
        `peak_update` row in BOTH branches -- it never stamps `anchor`/`prior_peak`.

        An earlier revision of this docstring claimed it stamped an anchor marker on
        the anchor-from-`None` case. It did, briefly, and that was a live SAFETY
        REGRESSION: five independent review passes each found a new way for a
        lost-history anchor to acquire authority and permanently destroy the true
        high-water mark (no gate; a gate on the wrong signal; a chmod-000 dir where
        `glob` returns empty without raising; unparseable lines; and present-but-silent
        sources such as a 0-byte file or an absent live file). The docstring outlived
        the code by one cycle and is corrected here.

        THE RULE NOW: the replay grants authority only to a row that NAMES what it
        superseded (`prior_peak` coercing to a positive finite NAV). An
        anchor-from-`None` supersedes nothing, so it can name nothing, so it is never
        authoritative -- which makes all five routes unreachable by construction rather
        than by enumeration. Consequently NO production writer emits an authoritative
        anchor at all; the authorized re-anchor is `reset_peak`, token-gated and DARK.
        `test_phase_36_8_no_production_path_can_write_an_authoritative_anchor` is the
        live tripwire that goes red if that ever changes.

        See handoff/current/research_brief_36.8.md for why an in-stream boundary (Kafka
        leader epochs, PostgreSQL timelines, Fowler's rejected-events) and NOT file
        recency -- PostgreSQL prefers the archive over the live directory.
        """
        with self._lock:
            if self._peak_nav is None:
                self._peak_nav = float(nav)
                # phase-36.8 (cycle-2 SAFETY FIX): stamp AUTHORITY only when the
                # replay saw every source. Otherwise this is an anchor over
                # history we merely could not read -- exactly 36.12's accident --
                # and marking it would let one transient archive-scan failure
                # permanently destroy the true high-water mark in the UNSAFE
                # direction. Unmarked => ratchet, so a later complete boot
                # restores the real peak.
                # phase-36.8 (cycle-5 REDESIGN): an anchor-from-None CANNOT name what
                # it superseded, because there was nothing to observe. It is therefore
                # never authoritative, regardless of `_history_complete`. That flag is
                # retained as a diagnostic (and is still asserted by tests) but it is no
                # longer what decides authority -- deriving authority from "no failure I
                # remembered to check for" is what produced five successive holes.
                if not self._history_complete:
                    logger.warning(
                        "kill_switch: anchoring peak at %.2f over an INCOMPLETE audit "
                        "replay -- history may exist that this boot could not read.",
                        self._peak_nav,
                    )
                self._append_audit("peak_update", nav=self._peak_nav)
            elif nav > self._peak_nav:
                self._peak_nav = float(nav)
                self._append_audit("peak_update", nav=self._peak_nav)

    def record_lost_history_anchor(
        self,
        prior_sod_nav: Optional[float],
        prior_peak_nav: Optional[float],
        anchored_nav: float,
        blocked_orders: bool,
    ) -> None:
        """phase-36.12 (CWE-223, omission of security-relevant information): record
        that a baseline was anchored because its HISTORY WAS LOST, not because the
        peak ratcheted.

        `update_peak`'s row carries only `nav` (:381), so an anchor-from-None is
        forensically identical to a legitimate upward ratchet -- there is no way,
        from the trail alone, to tell "the peak rose to 24666" from "the peak was
        lost and re-anchored at 12000". The deliberate form of that act
        (`reset_peak`) is gated behind an operator token precisely because it is a
        risk decision; the accidental form was neither gated nor visible.

        INFORMATIONAL ONLY. `_load_from_audit`'s event chain does not consume this
        name, so replay semantics are byte-unchanged and `peak_reset` remains the
        only assignment-semantics event. `test_phase_36_12_the_new_event_is_replay_inert`
        pins that. Step 36.8 (which also wants a distinguishable re-anchor event)
        should consume this row rather than introduce a second one.
        """
        with self._lock:
            self._baseline_provenance = "lost_history_anchor"
            self._append_audit(
                "baseline_anchor_on_lost_history",
                prior_sod_nav=prior_sod_nav,
                prior_peak_nav=prior_peak_nav,
                anchored_nav=float(anchored_nav),
                blocked_orders=bool(blocked_orders),
            )

    def reset_peak(self, new_peak: float, trigger: str,
                   operator: Optional[str] = None) -> Optional[dict]:
        """phase-69.1 (audit item 2): audited, restart-replayable reset of the
        trailing high-water mark.

        The monotonic `update_peak` never moves down and nothing ever reset it,
        so after a >=10% pullback that flattens the book to 100% cash the
        trailing-DD breach persists forever and BOTH resume paths refuse -- the
        engine stops earning permanently. `reset_peak` re-anchors the peak on a
        flatten-to-cash or an operator resume (callers pass the current/post-flatten
        NAV), emitting a `peak_reset` audit row that `_load_from_audit` replays, so
        the reset survives a restart and is idempotent. Thresholds (4/10/8/30) are
        byte-untouched -- this restores the documented "human resume once healthy".

        DARK by default: a no-op returning None unless the operator has recorded the
        token via `settings.kill_switch_peak_reset_enabled=True` (KS-PEAK-RESET:
        APPROVED). This is a guard-behavior change, so it must not fire until then.
        """
        try:
            from backend.config.settings import get_settings
            enabled = bool(getattr(get_settings(), "kill_switch_peak_reset_enabled", False))
        except Exception:
            enabled = False
        if not enabled:
            return None  # DARK -- no peak reset until KS-PEAK-RESET: APPROVED
        with self._lock:
            old_peak = self._peak_nav
            self._peak_nav = float(new_peak)
            self._append_audit("peak_reset", old_peak=old_peak,
                               new_peak=self._peak_nav, trigger=trigger,
                               operator=operator)
            return self._snapshot_locked()


_state = KillSwitchState()

# phase-36.12: the three events that constitute "a baseline has been anchored at
# some point in this book's life". Kept next to the probe that reads it so the two
# cannot drift.
_BASELINE_EVENTS = frozenset({"sod_snapshot", "peak_update", "peak_reset"})


def baseline_history_exists() -> bool:
    """True iff a baseline row has EVER been written, across every audit source.

    phase-36.12, provenance probe D1. `armed: False` on its own cannot tell a
    genuinely NEW book (nothing to restore, must be allowed to trade) apart from a
    book whose history was LOST (everything to restore, and a real drawdown behind
    it). Both present as `sod_nav=None, peak_nav=None`.

    The canonical shape for this is an on-disk provenance marker written in advance
    -- Galera's `grastate.dat` / `safe_to_bootstrap` / `seqno: -1`, where the marker's
    VALUE says whether the state is trustworthy and the node refuses to bootstrap
    rather than guess. This book has no such marker and cannot retroactively acquire
    one, so provenance is derived from the audit stream instead.

    READ-ONLY: never writes, never mutates state. Pair it with the caller's own
    book-provenance signal (nav vs starting_capital); either alone is too weak.
    On a probe failure this returns True -- "assume history exists" resolves the
    ambiguity toward blocking, which is the conservative direction.
    """
    try:
        rows, _complete = KillSwitchState._read_audit_rows()
        for row in rows:
            if row.get("event") in _BASELINE_EVENTS:
                return True
    except Exception as e:  # unreadable audit tree must not silently unblock trading
        logger.warning(f"kill_switch: baseline-history probe failed: {e}")
        return True
    return False


def get_state() -> KillSwitchState:
    return _state


# ── Breach evaluation ──────────────────────────────────────────────


def evaluate_breach(
    current_nav: float,
    daily_loss_limit_pct: float,
    trailing_dd_limit_pct: float,
) -> dict:
    """
    Check both limits against the current NAV. Returns a dict with booleans
    and diagnostic context. Does NOT flip state -- callers (see PaperTrader
    below) decide whether to flatten+pause based on the returned flags.

    phase-36.7 FAIL-LOUD CONTRACT. A missing baseline used to be indistinguishable
    from a healthy reading: both legs were skipped, both percentages stayed at
    their 0.0 initialisers, and `any_breached` was False for ANY current_nav. The
    return now carries three keys so absence can never be read as health:

      * `daily_baseline_missing`   -- the daily-loss leg could not be evaluated
      * `trailing_baseline_missing`-- the trailing-DD leg could not be evaluated
      * `armed`                    -- False iff either leg is unevaluable

    THREE THINGS THIS DELIBERATELY DOES NOT DO:

      1. It does NOT set `any_breached=True` on a missing baseline.
         `paper_trader.check_and_enforce_kill_switch:1097` would immediately
         `flatten_all()` a healthy book on what is merely a housekeeping file
         move -- a new destructive behaviour, not a conservative one. The loud
         signal is the marker; the refusals live on the resume paths.
      2. It does NOT early-return when a baseline is missing. A present-sod /
         absent-peak state still evaluates the daily leg today (measured: the
         `-v4` archive alone, peak_nav=None, correctly fires via the daily leg at
         a 50% drawdown). A wholesale `if not sod or not peak: return disarmed`
         would disable a leg that still works -- strictly LESS likely to pause.
         The markers are therefore PER LEG.
      3. It does NOT discriminate on VALUE. A leg is missing when its baseline
         cannot produce a percentage (None / <=0), never because the resulting
         percentage happens to be 0.0 -- a book sitting exactly at its
         high-water mark is a legitimate healthy reading (the phase-80.36
         "discriminate on presence, never on value" rule).
    """
    # phase-69.1 (audit item 2, kill_switch:246): guard against invalid NAV.
    # A caller's BQ-timeout `or 0.0` fallback yields current_nav<=0, which the
    # (sod - current_nav)/sod math would render as a phantom 100% daily +
    # trailing breach -- flattening the whole book on a transient 5s timeout.
    # A funded paper book's NAV is never <=0, so treat it as no-data (fail-safe:
    # no breach) rather than a real breach. Thresholds are untouched.
    s = _state.snapshot()
    sod = s.get("sod_nav")
    peak = s.get("peak_nav")
    # A leg is evaluable iff its baseline is present AND positive.
    daily_baseline_missing = not (sod is not None and sod > 0)
    trailing_baseline_missing = not (peak is not None and peak > 0)
    armed = not (daily_baseline_missing or trailing_baseline_missing)
    if not armed:
        _log_disarmed_once(sod, peak)

    if current_nav is None or current_nav <= 0:
        return {
            "daily_loss_breached": False, "daily_loss_pct": 0.0,
            "daily_loss_limit_pct": float(daily_loss_limit_pct),
            "trailing_dd_breached": False, "trailing_dd_pct": 0.0,
            "trailing_dd_limit_pct": float(trailing_dd_limit_pct),
            "any_breached": False, "nav_invalid": True,
            # phase-36.7: present in BOTH return shapes so no consumer has to
            # branch on which one it got. `nav_invalid` stays absent from the
            # normal return (its established optional-key discipline).
            "daily_baseline_missing": bool(daily_baseline_missing),
            "trailing_baseline_missing": bool(trailing_baseline_missing),
            "armed": bool(armed),
        }

    daily_loss_breached = False
    daily_loss_pct = 0.0
    if not daily_baseline_missing:
        daily_loss_pct = (sod - current_nav) / sod * 100.0
        daily_loss_breached = daily_loss_pct >= daily_loss_limit_pct

    trailing_dd_breached = False
    trailing_dd_pct = 0.0
    if not trailing_baseline_missing:
        trailing_dd_pct = (peak - current_nav) / peak * 100.0
        trailing_dd_breached = trailing_dd_pct >= trailing_dd_limit_pct

    return {
        "daily_loss_breached": bool(daily_loss_breached),
        "daily_loss_pct": round(daily_loss_pct, 4),
        "daily_loss_limit_pct": float(daily_loss_limit_pct),
        "trailing_dd_breached": bool(trailing_dd_breached),
        "trailing_dd_pct": round(trailing_dd_pct, 4),
        "trailing_dd_limit_pct": float(trailing_dd_limit_pct),
        "any_breached": bool(daily_loss_breached or trailing_dd_breached),
        "daily_baseline_missing": bool(daily_baseline_missing),
        "trailing_baseline_missing": bool(trailing_baseline_missing),
        "armed": bool(armed),
    }


def _log_disarmed_once(sod: Optional[float], peak: Optional[float]) -> None:
    """One-shot-per-process ERROR log for a disarmed kill switch.

    `evaluate_breach` is called by a polled GET, so this must not log on every
    request. Kept free of any network dispatch for the same reason -- the
    operator-visible surfaces are the `armed` flag on the API response, the
    UNKNOWN badge in the UI, the /resume 409, this log line, and (phase-36.12) the
    autonomous cycle's refusal to place new orders, which emits its own P1 alert
    plus a `baseline_anchor_on_lost_history` audit row.
    """
    global _disarmed_logged
    if _disarmed_logged:
        return
    _disarmed_logged = True
    logger.error(
        "kill_switch DISARMED: baseline missing (sod_nav=%r peak_nav=%r) -- "
        "breach legs without a baseline cannot fire. Sources replayed: %s",
        sod, peak, [str(p) for p in _audit_source_paths()],
    )


# phase-38.1 (OPEN-10): kill-switch auto-resume hysteresis.
# Operator-driven resumes created two 3.5h outage windows in 5 days
# (OPS-F10). Auto-resume after 2h of no-breach closes that gap.

AUTO_RESUME_ALERT_AT_SEC: float = 60 * 60       # T+1h: pager alert
AUTO_RESUME_TRIGGER_AT_SEC: float = 2 * 60 * 60  # T+2h: auto-resume fires


def check_auto_resume(
    current_nav: float,
    daily_loss_limit_pct: float,
    trailing_dd_limit_pct: float,
    enabled: bool = False,
) -> dict:
    # phase-38.1 (OPEN-10): evaluate hysteresis. Default-OFF; caller
    # passes `enabled=True` after operator opts in via the
    # `kill_switch_auto_resume_enabled` settings flag.
    #
    # Returns dict with:
    #   action: "no_op" | "alert" | "resume"
    #   reason: human-readable explanation
    #   seconds_paused: time since pause (or None if not paused)
    #
    # No state mutation here except via state.resume() on action="resume"
    # and audit-log append on action="alert". This keeps the function
    # ergonomic to call once per cycle.
    sentinel = {
        "action": "no_op", "reason": "auto_resume_disabled" if not enabled else "not_paused",
        "seconds_paused": None,
    }
    if not enabled:
        return sentinel
    s = _state.snapshot()
    if not s.get("paused"):
        return sentinel
    paused_at_str = s.get("paused_at")
    if not paused_at_str:
        return {"action": "no_op", "reason": "no_paused_at_timestamp", "seconds_paused": None}
    try:
        paused_at = datetime.fromisoformat(str(paused_at_str).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return {"action": "no_op", "reason": "no_paused_at_timestamp", "seconds_paused": None}
    now = datetime.now(timezone.utc)
    seconds_paused = (now - paused_at).total_seconds()

    # If the breach is STILL active, never auto-resume.
    breach = evaluate_breach(current_nav, daily_loss_limit_pct, trailing_dd_limit_pct)
    if breach["any_breached"]:
        return {
            "action": "no_op",
            "reason": "breach_still_active",
            "seconds_paused": round(seconds_paused, 1),
            "breach": breach,
        }
    # phase-36.7: a DISARMED switch's `any_breached=False` is not evidence of
    # health -- it only means we could not measure. Auto-resuming on it would
    # un-pause a book out of a real drawdown (and would violate
    # docs/runbooks/away-ops-rules.md rail 5, "kill-switch stays paused after
    # any breach"). Fail-open on a missing key so an older/partial dict cannot
    # wedge the hysteresis path.
    if not breach.get("armed", True):
        return {
            "action": "no_op",
            "reason": "kill_switch_disarmed_baseline_missing",
            "seconds_paused": round(seconds_paused, 1),
            "breach": breach,
        }

    # T+2h: auto-resume fires.
    if seconds_paused >= AUTO_RESUME_TRIGGER_AT_SEC:
        _state.resume(trigger="auto_resume_hysteresis", details={
            "seconds_paused": round(seconds_paused, 1),
            "current_nav": current_nav,
            "breach": breach,
        })
        return {
            "action": "resume",
            "reason": "no_breach_for_2h",
            "seconds_paused": round(seconds_paused, 1),
        }

    # T+1h: pager alert (one-shot per pause-cycle).
    already_alerted = s.get("auto_resume_alerted_at") is not None
    if seconds_paused >= AUTO_RESUME_ALERT_AT_SEC and not already_alerted:
        _state._append_audit(
            "auto_resume_alert",
            seconds_paused=round(seconds_paused, 1),
            current_nav=current_nav,
        )
        # In-memory state update so subsequent cycles don't re-alert.
        with _state._lock:
            _state._auto_resume_alerted_at = datetime.now(timezone.utc).isoformat()
        # Fail-open Slack dispatch.
        try:
            from backend.services.observability.alerting import raise_cron_alert_sync
            raise_cron_alert_sync(
                source="kill_switch",
                error_type="auto_resume_pending",
                severity="P2",
                title="Kill-switch auto-resume will fire in ~1h (no breach detected)",
                details={
                    "seconds_paused": str(round(seconds_paused, 1)),
                    "auto_resume_at_sec": str(AUTO_RESUME_TRIGGER_AT_SEC),
                    "current_nav": str(current_nav),
                    "breach": str(breach),
                },
            )
        except Exception as exc:
            logger.warning("kill_switch auto_resume_alert dispatch fail-open: %r", exc)
        return {
            "action": "alert",
            "reason": "no_breach_for_1h_pager_fired",
            "seconds_paused": round(seconds_paused, 1),
        }

    return {
        "action": "no_op",
        "reason": "paused_but_under_hysteresis_threshold",
        "seconds_paused": round(seconds_paused, 1),
    }
