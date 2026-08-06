"""phase-82.11: make a failing nightly autoresearch run AUDIBLE, escalating,
and drivable from pytest.

THE DEFECT, AND WHY THE STEP DESCRIPTION IS HALF WRONG. The masterplan text
says the loop "writes a failure file and exits silently, so 59 failures
produced zero operator signal". Measured 2026-08-06, a paging seam DOES exist
and DID run: `scripts/autoresearch/run_nightly.sh:49-69`
(`_record_fail_and_page`, phase-75.11 / 76.9.2) wrote
`handoff/away_ops/autoresearch_fail_state.json` at 02:00:13 with
`{"consecutive_fails": 13}` and passed its `13 >= PAGE_AFTER_N=3` test. What is
actually broken about it:

  * Its curl ends `>/dev/null 2>&1 || true`, so stdout, stderr AND exit status
    are discarded -- and Slack's `chat.postMessage` returns HTTP 200 with
    `{"ok": false}` for `invalid_auth` / `not_in_channel`. "Did the operator
    get paged?" is unanswerable from local state, by construction.
  * It has no escalation ladder: same channel, same severity, same sentence
    every night for 12 consecutive nights; only an integer changes.
  * It is a bash function inside a launchd wrapper, so no pytest can capture
    the emitted alert -- which is exactly what this step's criteria demand.

WHAT THIS MODULE DOES INSTEAD. It emits through the canonical Python seam,
`backend.services.observability.alerting.raise_cron_alert_sync`
(20+ production call sites), at `P1`/`P0` -- never `P2`, because with
`slack_webhook_url` empty on this machine a `P2` is logged and dropped
(`alerting.py:219-224`) while `P0`/`P1` reach `_bot_token_fallback`, which
parses `ok` at `alerting.py:168`.

THE TRAP A NAIVE FIX WALKS INTO, and it is the same one 82.10 documented.
`AlertDeduper` does NOT suppress steady state: a P0/P1 bypasses the
consecutive-occurrence threshold, and the per-(source, error_type) repeat
window is 1h -- irrelevant to a job that runs once a night. So an
alert-on-every-failure implementation re-pages an unchanged known-bad condition
nightly forever. Google SRE Book ch.6: "When pages occur too frequently,
employees second-guess, skim, or even ignore incoming alerts." Alertmanager's
documented default is to NOT repeat absent a state change. Therefore THIS
module owns the transition gate (edge-trigger on a tier change), exactly as
`backend/services/freshness_cron.py` does for the freshness alarm.

BUT NOT SILENCE EITHER. SRE Workbook ch.8 warns against explaining away a
recurring page. So the ladder gets LOUDER with age (a second tier at a distinct
`error_type`, which also gives `AlertDeduper` a fresh key) and keeps a slow
`REMIND_EVERY_DAYS` safety net -- Alertmanager's `repeat_interval` role.

WHY CONFIG-CLASS ERRORS PAGE ON NIGHT ONE. SRE Workbook ch.8: "All alerts
should be immediately actionable." A dead credit balance is actionable
immediately and will never self-heal, unlike a transient provider error. So
`credit_exhausted` / `auth` start at tier 1 at n=1, while a generic error waits
for the `PAGE_AFTER_N` threshold.

State is derived from the memo DIRECTORY, not from
`autoresearch_fail_state.json`. Three reasons, all measured: (a) the JSON is
silently reset to 0 by any manual run (`run_nightly.sh:96`) -- it read 13 while
the directory showed 12 consecutive failing dates; (b) nothing else in the repo
reads it; (c) a directory is what a `tmp_path` fixture can actually be.

ASCII-only logger messages per `.claude/rules/security.md`.
"""
from __future__ import annotations

import datetime as dt
import logging
import re
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

#: Alert source tag. Distinct from any existing caller of `raise_cron_alert_sync`.
ALERT_SOURCE = "autoresearch"

#: Consecutive failures before a GENERIC failure becomes a page. Matches the
#: bash seam's `PAGE_AFTER_N` default so the two cannot disagree about tier 1.
PAGE_AFTER_N = 3

#: Consecutive failures at which any class escalates to the louder tier.
ESCALATE_AFTER_N = 7

#: While at the escalated tier, re-emit once every N days as a slow safety net
#: (Alertmanager `repeat_interval` role). NOT a daily notice -- criterion 2 of
#: this step exists precisely to forbid that.
REMIND_EVERY_DAYS = 7

#: `error_type` values. Distinct per tier on purpose: `AlertDeduper` keys on
#: (source, error_type), so the escalation is never swallowed as a repeat of
#: the tier-1 alert.
ERROR_TYPE_TIER1 = "nightly_run_failing"
ERROR_TYPE_TIER2 = "nightly_run_failing_escalated"

#: Failure classes. `classify_failure` is deliberately NARROW, in the style of
#: `run_memo._is_network_weather` -- it must not widen into a catch-all.
CLASS_CREDIT = "credit_exhausted"
CLASS_AUTH = "auth"
CLASS_GENERIC = "generic"

#: Classes that never self-heal, so they are immediately actionable at n=1.
_CONFIG_CLASSES = (CLASS_CREDIT, CLASS_AUTH)

_CREDIT_TOKENS = (
    "credit balance is too low",
    "purchase credits",
)
_AUTH_TOKENS = (
    "authentication_error",
    "invalid x-api-key",
    "invalid api key",
    "could not resolve authentication method",
)

#: `YYYY-MM-DD-...` memo filenames. ERROR and WARN carry a marker right after
#: the date; anything else on that date is a real memo (a success artifact).
_DATE_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})-(.*)$")


def classify_failure(exc: BaseException | str | None) -> str:
    """Classify a nightly-run failure. Narrow by design -- see module docstring.

    Walks the `__cause__` / `__context__` chain so a wrapped provider error is
    still recognised (`langchain_anthropic` re-raises the SDK error nested).
    """
    if exc is None:
        return CLASS_GENERIC

    def _text(e: Any) -> str:
        return str(e).lower()

    texts: list[str] = []
    if isinstance(exc, str):
        texts.append(exc.lower())
    else:
        seen: set[int] = set()
        cur: BaseException | None = exc  # type: ignore[assignment]
        while cur is not None and id(cur) not in seen:
            seen.add(id(cur))
            texts.append(_text(cur))
            texts.append(type(cur).__name__.lower())
            cur = cur.__cause__ or cur.__context__

    blob = " ".join(texts)
    if any(tok in blob for tok in _CREDIT_TOKENS):
        return CLASS_CREDIT
    if any(tok in blob for tok in _AUTH_TOKENS):
        return CLASS_AUTH
    return CLASS_GENERIC


def _scan_memo_dir(memo_dir: Path) -> tuple[dict[str, int], dict[str, int]]:
    """Return ({date -> n_error_files}, {date -> n_success_memos}).

    A file is a SUCCESS memo iff its name is `<date>-<rest>` and `<rest>` does
    not start with `ERROR-` or `WARN-`. This distinction is load-bearing:
    2026-07-24 and 2026-07-25 each carry BOTH an ERROR file (the failed 02:00
    launchd run) and a success memo (a manual max-rail run), so a naive
    "this date has an ERROR file => that night failed" scan overcounts.
    """
    errors: dict[str, int] = {}
    successes: dict[str, int] = {}
    try:
        names = [p.name for p in memo_dir.iterdir() if p.is_file()]
    except (OSError, AttributeError):
        return errors, successes
    for name in names:
        if not name.endswith(".md"):
            continue
        m = _DATE_RE.match(name)
        if not m:
            continue
        date = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
        rest = m.group(4)
        if rest.startswith("ERROR-"):
            errors[date] = errors.get(date, 0) + 1
        elif rest.startswith("WARN-"):
            continue  # tolerated network weather (phase-76.9); not a failure
        else:
            successes[date] = successes.get(date, 0) + 1
    return errors, successes


def count_consecutive_failures(memo_dir: Path, today: dt.date) -> int:
    """Consecutive failing dates ending at (and including) `today`.

    A date FAILED iff it has at least one `*-ERROR-*` memo AND no success memo.
    Walks backwards from `today`, stopping at the first non-failing date.
    Returns 0 for an empty/absent directory or when `today` did not fail.

    Measured against the live directory on 2026-08-06: 12 (2026-07-26 ..
    2026-08-06), stopping at 2026-07-25 which carries both file kinds. The
    TOTAL ERROR-file count over the same directory is 62 -- do not confuse the
    two; `consecutive` is the one the ladder is keyed on.
    """
    errors, successes = _scan_memo_dir(Path(memo_dir))
    n = 0
    cur = today
    # Bounded so a pathological directory can never spin: the memo dir has one
    # file per night, so 10 years is far past any real history.
    for _ in range(3660):
        key = cur.isoformat()
        if errors.get(key, 0) > 0 and successes.get(key, 0) == 0:
            n += 1
            cur = cur - dt.timedelta(days=1)
        else:
            break
    return n


def last_success_date(memo_dir: Path) -> Optional[str]:
    """ISO date of the most recent success memo, or None."""
    _, successes = _scan_memo_dir(Path(memo_dir))
    return max(successes) if successes else None


def _tier(n: int, failure_class: str) -> int:
    """Ladder position for `n` consecutive failures of `failure_class`.

    0 = silent, 1 = P1, 2 = P0 (escalated). Config-class failures
    (credit exhaustion, auth) enter tier 1 at n=1 because they are immediately
    actionable and never self-heal; a generic failure waits for PAGE_AFTER_N.
    """
    if n <= 0:
        return 0
    if n >= ESCALATE_AFTER_N:
        return 2
    if failure_class in _CONFIG_CLASSES:
        return 1
    return 1 if n >= PAGE_AFTER_N else 0


def report_run_outcome(
    *,
    failed: bool,
    exc: BaseException | str | None = None,
    memo_dir: Path,
    today: Optional[dt.date] = None,
    notify: Optional[Callable[..., Any]] = None,
    topic: str = "",
) -> dict:
    """Emit an operator alert for a nightly autoresearch outcome, if warranted.

    Returns a summary dict:
    `{"ok", "failed", "failure_class", "consecutive_failures", "tier",
      "prev_tier", "alerts_emitted", "escalated", "reason",
      "last_success_date"}`.

    Emission rules (each traceable to the contract's ladder table):
      * `failed=False` -> NEVER emits. Criterion 3 of this step is
        unconditional ("a successful run emits NO alert"), which overrules the
        research brief's recommendation of a recovery notice. Deliberate.
      * Otherwise emit iff the tier INCREASED versus n-1 (edge trigger), OR
        the run is at tier 2 and
        `(n - ESCALATE_AFTER_N) % REMIND_EVERY_DAYS == 0` (the slow safety
        net).

    KNOWN LIMITATION, pinned by
    `test_a_missed_night_resets_the_counter_and_the_ladder`: because the count
    is derived by walking backwards over failing DATES, a night on which the
    job did not run at all is not a failing date and RESETS the count -- so a
    missed night silently rewinds the ladder. n therefore never jumps; it
    advances by exactly 1 or resets. That is the dead-man's-switch hole (a job
    that never runs also never reports), out of scope for this step's criteria
    and queued as its own masterplan step.

    `notify` is injectable for convenience, but the production path resolves
    `raise_cron_alert_sync` itself via a function-local import -- so at least
    one guard can drive the real emitter with no injection at all.

    Fail-open: a notification problem must never change the nightly job's exit
    code. Every failure path logs and returns `ok=False`.
    """
    summary: dict = {
        "ok": False,
        "failed": bool(failed),
        "failure_class": None,
        "consecutive_failures": 0,
        "tier": 0,
        "prev_tier": 0,
        "alerts_emitted": 0,
        "escalated": False,
        "reason": "",
        "last_success_date": None,
    }
    try:
        memo_dir = Path(memo_dir)
        if today is None:
            today = dt.date.today()

        summary["last_success_date"] = last_success_date(memo_dir)

        if not failed:
            summary["ok"] = True
            summary["reason"] = "run succeeded -- no alert by criterion 3"
            return summary

        failure_class = classify_failure(exc)
        n = count_consecutive_failures(memo_dir, today)
        tier = _tier(n, failure_class)
        prev_tier = _tier(n - 1, failure_class)

        summary["failure_class"] = failure_class
        summary["consecutive_failures"] = n
        summary["tier"] = tier
        summary["prev_tier"] = prev_tier

        reminder = (
            tier == 2
            and n >= ESCALATE_AFTER_N
            and (n - ESCALATE_AFTER_N) % REMIND_EVERY_DAYS == 0
        )
        if tier == 0:
            summary["ok"] = True
            summary["reason"] = f"tier 0 (n={n}, class={failure_class}) -- below threshold"
            return summary
        if tier <= prev_tier and not reminder:
            summary["ok"] = True
            summary["reason"] = (
                f"steady state at tier {tier} (n={n}) -- suppressed to avoid "
                "re-paging an unchanged condition"
            )
            return summary

        if notify is None:
            from backend.services.observability.alerting import (
                raise_cron_alert_sync,
            )

            notify = raise_cron_alert_sync

        escalated = tier >= 2
        severity = "P0" if escalated else "P1"
        error_type = ERROR_TYPE_TIER2 if escalated else ERROR_TYPE_TIER1
        detail_text = str(exc) if exc is not None else ""
        title = (
            f"Autoresearch nightly run has failed {n} night(s) in a row "
            f"({failure_class})"
        )
        details = {
            "consecutive_failures": n,
            "failure_class": failure_class,
            "tier": tier,
            "escalated": escalated,
            "reminder": reminder,
            "last_success_date": summary["last_success_date"] or "never",
            "topic": topic[:200],
            "error": detail_text[:600],
            "runbook": "handoff/autoresearch/root_cause.md",
        }

        notify(
            source=ALERT_SOURCE,
            error_type=error_type,
            severity=severity,
            title=title,
            details=details,
        )
        summary["alerts_emitted"] = 1
        summary["escalated"] = escalated
        summary["ok"] = True
        summary["reason"] = (
            f"emitted {severity} at tier {tier} (prev {prev_tier}, n={n}"
            + (", periodic reminder" if reminder else "")
            + ")"
        )
        logger.warning(
            "autoresearch_health: emitted %s tier=%d n=%d class=%s",
            severity, tier, n, failure_class,
        )
        return summary
    except Exception as exc_inner:  # noqa: BLE001 -- fail-open by contract
        logger.warning("autoresearch_health: fail-open: %r", exc_inner)
        summary["reason"] = f"fail-open: {exc_inner!r}"
        return summary


__all__ = [
    "ALERT_SOURCE",
    "CLASS_AUTH",
    "CLASS_CREDIT",
    "CLASS_GENERIC",
    "ERROR_TYPE_TIER1",
    "ERROR_TYPE_TIER2",
    "ESCALATE_AFTER_N",
    "PAGE_AFTER_N",
    "REMIND_EVERY_DAYS",
    "classify_failure",
    "count_consecutive_failures",
    "last_success_date",
    "report_run_outcome",
]
