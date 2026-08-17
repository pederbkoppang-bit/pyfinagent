"""phase-86.108: make an agent's unparseable output a COUNTABLE RECORD.

## The defect this closes

Before this module, every one of the four invalid-JSON emit sites did exactly
one thing: `logger.warning("<label> returned invalid JSON")`. That is a string.
It is not a record, and the census built on it (step 86.108 criterion 1,
`scripts/qa/census_invalid_json_86_108.py`) had to prove four separate
corrections precisely because a log line carries no structure:

  - the corpus is MIXED-FORMAT, so a `"module":`-keyed parser sees 17% and
    looks complete;
  - the agent label is a free-text phrase, so "Analyst" collapsed three
    distinct analysts into one bucket;
  - one failure can emit two lines, so the total counts LINES not EVENTS;
  - **no line carries a rail, provider or model**, so a per-event
    `claude_code`-vs-`gemini` attribution is not derivable at all.

Every one of those is a property of *logging a sentence instead of recording a
fact*. This module records the fact. Each failure becomes one record with a
fixed field set, counted once, attributed to the agent phrase verbatim, and
carrying **the model that served it plus a rail derived from that model**.

**Read the rail's limits before quoting it.** The rail is resolved by
`resolve_rail`, which mirrors the client's real routing predicate rather than
reading the CC-route flag on its own -- see that function's docstring for why
the flag alone silently misattributes, and for the three cases that honestly
resolve to `unknown`. Where an emit site has no model in scope the record says
`unknown` and says why in `rail_basis`; it never substitutes a guess. So the
attribution the census could not recover retrospectively is available
prospectively **wherever a model is in scope**, and is explicitly marked absent
where it is not.

## What it deliberately does NOT do

- **It changes no control flow.** Every call site records and then behaves
  exactly as before. No parse is retried, no default is fabricated, no gate is
  loosened. A recorder that could alter a money path would be a far worse
  defect than the silence it replaces.
- **It never raises into the parse path.** A failure inside the recorder is
  swallowed at the boundary -- but NOT silently: `recorder_errors` counts it
  and it is logged at ERROR. A fail-open guard that hides its own breakage is
  the failure mode this project has hit before; the counter is what makes this
  one observable.
- **It is process-local.** This is an in-memory ledger for the running process,
  which is exactly what a "committed is not in force" check needs. It is not a
  durable store and does not claim to be; `records_seen` is monotonic for the
  life of the process only.

## Counter vs gauge -- read this before quoting a number

`records_seen` and `by_agent_kind` are **counters**: monotonic, never pruned,
and safe to compare against a threshold. `records_retained` is a **gauge**: the
ring buffer evicts, so it can go DOWN. `evicted` is the reconciling counter
(`records_seen == records_retained + evicted` always holds). Quoting the gauge
as a count is the error this docstring exists to prevent.
"""

from __future__ import annotations

import logging
import os
import threading
from collections import Counter, deque
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ── The kind vocabulary ────────────────────────────────────────────
# Deliberately CLOSED. An unrecognised kind is recorded under
# KIND_UNKNOWN rather than silently creating a new bucket, because a
# typo'd kind that invents its own bucket is invisible in every
# aggregate built on this ledger.

KIND_PARSE_FAILED = "parse_failed"
"""Syntactic failure: the text is not JSON at all."""

KIND_TRUNCATED = "truncated"
"""The provider stopped early (stop_reason), so the cause is a budget, not a
malformed reply. Recorded separately because the remedy is different."""

KIND_SCHEMA_VALID_BUT_REJECTED = "schema_valid_but_rejected_downstream"
"""The text parsed as JSON but the shape the caller requires was not met. In
this codebase the concrete instance is `json.loads` returning a non-dict: valid
JSON, rejected by the caller. Kept distinct from `parse_failed` because a
schema/prompt remedy targets one and not the other."""

KIND_UNKNOWN = "unknown_kind"
"""Sentinel for a caller passing a kind outside the closed set."""

KINDS = frozenset({
    KIND_PARSE_FAILED,
    KIND_TRUNCATED,
    KIND_SCHEMA_VALID_BUT_REJECTED,
})

# The machine-greppable prefix. Every record emits exactly one line with this
# prefix, so a future census greps a STABLE token instead of an English
# sentence that phase-75 already rewrote once mid-corpus.
LOG_PREFIX = "PARSE_FAILURE_RECORD"

_MAX_RETAINED = int(os.environ.get("PARSE_FAILURE_LEDGER_MAX", "500") or "500")


class _Ledger:
    """Thread-safe in-process ledger. One instance per process."""

    def __init__(self, maxlen: int = _MAX_RETAINED) -> None:
        self._lock = threading.Lock()
        self._recent: deque[dict[str, Any]] = deque(maxlen=maxlen)
        self._by_agent_kind: Counter[tuple[str, str]] = Counter()
        self._by_site: Counter[str] = Counter()
        self._by_rail: Counter[str] = Counter()
        self._seen = 0
        self._evicted = 0
        self._recorder_errors = 0
        self._maxlen = maxlen

    # ── write ──
    def record(self, rec: dict[str, Any]) -> None:
        with self._lock:
            self._seen += 1
            if len(self._recent) == self._maxlen:
                self._evicted += 1
            self._recent.append(rec)
            self._by_agent_kind[(rec["agent"], rec["kind"])] += 1
            self._by_site[rec["site"]] += 1
            self._by_rail[rec["rail"]] += 1

    def note_recorder_error(self) -> None:
        with self._lock:
            self._recorder_errors += 1

    # ── read ──
    def snapshot(self, limit: int = 50) -> dict[str, Any]:
        with self._lock:
            recent = list(self._recent)[-limit:] if limit > 0 else []
            return {
                "records_seen": self._seen,
                "records_seen_unit": (
                    "monotonic COUNTER of failure events recorded since process "
                    "start; never pruned; safe to compare against a threshold"
                ),
                "records_retained": len(self._recent),
                "records_retained_unit": (
                    "GAUGE -- ring-buffer occupancy, can go DOWN as records are "
                    "evicted. Never quote this as a count of failures."
                ),
                "evicted": self._evicted,
                "reconciles": self._seen == len(self._recent) + self._evicted,
                "recorder_errors": self._recorder_errors,
                "by_agent_kind": {
                    f"{a}|{k}": v for (a, k), v in sorted(self._by_agent_kind.items())
                },
                "by_site": dict(sorted(self._by_site.items())),
                "by_rail": dict(sorted(self._by_rail.items())),
                "recent": recent,
                "recent_limit": limit,
                "process": _process_identity(),
            }

    def reset(self) -> None:
        """Test-only. Never called from production code."""
        with self._lock:
            self._recent.clear()
            self._by_agent_kind.clear()
            self._by_site.clear()
            self._by_rail.clear()
            self._seen = 0
            self._evicted = 0
            self._recorder_errors = 0


_LEDGER = _Ledger()


def _process_identity() -> dict[str, Any]:
    """Which process answered. Without this a reader cannot tell whether the
    numbers came from the process that holds the flag they are checking."""
    return {"pid": os.getpid()}


RAIL_CLAUDE_CODE = "claude_code"
RAIL_DIRECT = "gemini_or_direct"
RAIL_UNKNOWN = "unknown"


def resolve_rail(model_name: Optional[str]) -> tuple[str, str]:
    """Which transport served this call, MEASURED from the model, plus the basis.

    Returns `(rail, basis)`. The basis is recorded beside the rail so a reader
    can tell a measurement from an absence of one -- an attribution with no
    stated basis is the defect this function was rewritten to remove.

    **This deliberately mirrors the client's real routing predicate**
    (`backend/agents/llm_client.py`, the `model_name.startswith("claude-") and
    paper_use_claude_code_route` gate). Reading the flag ALONE is wrong, and
    wrong in a way that silently misattributes: with the flag on, every
    Gemini-served call would be stamped `claude_code`, and the call log shows
    Gemini traffic outnumbering claude-code-tagged traffic by roughly 20x. The
    first revision of this module did exactly that; a Q/A executed a mutant
    that inverted the attribution and the suite stayed green, because the only
    guard was a set-membership assertion. Both holes are closed:
    `test_resolve_rail_disagrees_with_the_flag_only_rule` fails if anyone
    reverts to the flag, and mutation cells M13/M14 fail if the mapping is
    inverted or the model argument is ignored.

    Three honest `unknown`s, none of them a guess:

    - **no model in scope.** Some emit sites are given only text and a label.
      An unknown transport is recorded as unknown, never as the flag's value.
    - **settings unavailable.** The flag cannot be read at all.
    - **fail-forward may have substituted.** With `paper_rail_failforward_enabled`
      on, a Claude model under an enabled CC route can still be served by the
      Vertex-Gemini workhorse when the rail is dead. The flag pair cannot tell
      which happened, so neither can this function.
    """
    if not model_name:
        return RAIL_UNKNOWN, "no_model_in_scope_at_emit_site"
    try:
        from backend.config.settings import get_settings

        settings = get_settings()
        cc_route = bool(getattr(settings, "paper_use_claude_code_route", False))
        failforward = bool(getattr(settings, "paper_rail_failforward_enabled", False))
    except Exception:  # settings must never break a parse
        return RAIL_UNKNOWN, "settings_unavailable"

    if not str(model_name).startswith("claude-"):
        return RAIL_DIRECT, "measured: not a claude- model, so the CC rail gate cannot match"
    if not cc_route:
        return RAIL_DIRECT, "measured: claude- model but paper_use_claude_code_route is off"
    if failforward:
        return RAIL_UNKNOWN, (
            "claude- model with the CC route on, but paper_rail_failforward_enabled "
            "can substitute a Vertex-Gemini client when the rail is dead -- the flag "
            "pair cannot distinguish the two"
        )
    return RAIL_CLAUDE_CODE, "measured: claude- model and paper_use_claude_code_route on"


def record_parse_failure(
    agent: str,
    kind: str,
    *,
    site: str,
    detail: str = "",
    ticker: Optional[str] = None,
    model_name: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    """Record ONE unparseable-output event. Returns the record, or None if the
    recorder itself failed.

    This function is called from inside `except` blocks on the money path. It
    must never raise and must never change what the caller does next.

    Args:
        agent: the agent phrase verbatim, e.g. `"Conservative Analyst"`. Passed
            through unmodified -- collapsing it to a short label is the exact
            error that turned three analysts into one census bucket.
        kind: one of `KINDS`. Anything else is recorded as `unknown_kind`
            rather than creating a silent new bucket.
        site: `module.py:function` of the emit site, so the four sites stay
            distinguishable in aggregate.
        detail: short free text (truncated); never the full model output.
        ticker: when the failure is ticker-scoped.
    """
    try:
        if kind not in KINDS:
            logger.error(
                "%s: unrecognised kind %r from site %s -- recorded as %s. The kind "
                "vocabulary is closed; add the constant rather than passing a new string.",
                LOG_PREFIX, kind, site, KIND_UNKNOWN,
            )
            kind = KIND_UNKNOWN
        rail, rail_basis = resolve_rail(model_name)
        rec = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "agent": str(agent or "(no agent phrase)"),
            "kind": kind,
            "site": str(site),
            "model_name": model_name,
            "rail": rail,
            # The basis travels WITH the rail so a reader can tell a measured
            # attribution from an absent one without consulting this file.
            "rail_basis": rail_basis,
            "detail": str(detail or "")[:200],
            "ticker": ticker,
        }
        _LEDGER.record(rec)
        # One line, stable prefix, fixed field order -- greppable without the
        # four corrections the free-text corpus needed.
        logger.warning(
            "%s agent=%r kind=%s site=%s model=%s rail=%s ticker=%s",
            LOG_PREFIX, rec["agent"], rec["kind"], rec["site"],
            rec["model_name"], rec["rail"], rec["ticker"],
        )
        return rec
    except Exception as exc:  # never propagate into the parse path
        try:
            _LEDGER.note_recorder_error()
            logger.error(
                "%s: the RECORDER itself failed (%r). The parse path is unaffected, "
                "but this ledger is now under-counting -- see recorder_errors.",
                LOG_PREFIX, exc,
            )
        except Exception:
            pass
        return None


def snapshot(limit: int = 50) -> dict[str, Any]:
    """Read-only view of the ledger. Safe to call from a request handler."""
    return _LEDGER.snapshot(limit=limit)


def reset_for_tests() -> None:
    """Test-only helper. Production code must not call this."""
    _LEDGER.reset()
