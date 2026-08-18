"""Test-suite-wide isolation (phase-61.2 register fix, prod-pollution audit
2026-07-08).

The suite previously ran with ZERO BQ isolation under live ADC: the buffered
observability writers (api_call_log.flush / flush_llm, incl. their 60s
time-based auto-flush) leaked 106 unlabeled fixture rows into the REAL
pyfinagent_data.llm_call_log between 2026-05-19 and 2026-07-07 -- fixture
"successes" that masked a 7-week direct-API credit outage
(live_check_66.2.md 5d + money_engine_audit_2026-07-08.md).

Setting the guard at conftest IMPORT time (not in a fixture) means it is
active before test collection imports any module, covering flushes triggered
from module import side effects and mid-suite timer thresholds. The guard is
honored inside flush()/flush_llm() AFTER the buffer drain, so the drain
semantics tests assert are unchanged. Dormant in production (launchd env
never sets it).
"""

import os

os.environ.setdefault("PYFINAGENT_TEST_NO_BQ", "1")


# ---------------------------------------------------------------------------
# phase-82.58: block Slack egress from the test suite.
#
# Until 82.58 the cost-budget degradation alert could never be delivered (it
# raised TypeError into a swallowing except), so tests that drive
# `_record_degradation` -- test_phase_75_5_1_spend_metric.py:294 and
# test_phase_75_llm_rail.py:582 -- were harmless by accident. Repairing that
# path ARMS them: backend/.env carries a live `xoxb-` bot token, the webhook is
# empty so P1 alerts route to `_bot_token_fallback`, and a routine `pytest` run
# would POST to the operator's real channel.
#
# Installed at import time rather than as an autouse fixture, for the same
# reason as the BQ guard above: it must be active before collection imports any
# module. Scoped to slack.com ONLY -- this is not a general network jail, and
# tests that legitimately reach other hosts are unaffected.
#
# `_bot_token_fallback` catches this and logs "fail-open", so the two tests
# above still pass; they simply stop being able to send. A test that WANTS to
# assert on delivery patches `urllib.request.urlopen` itself, which replaces
# this wrapper for the duration of that patch.
# ---------------------------------------------------------------------------
import urllib.request

_REAL_URLOPEN = urllib.request.urlopen


def _no_slack_egress(req, *args, **kwargs):
    url = getattr(req, "full_url", None) or str(req)
    if "slack.com" in url:
        raise RuntimeError(
            "phase-82.58 test guard: refusing a live Slack POST from the test "
            f"suite (url={url!r}). Patch urllib.request.urlopen in your test if "
            "you mean to assert on delivery."
        )
    return _REAL_URLOPEN(req, *args, **kwargs)


urllib.request.urlopen = _no_slack_egress


# ---------------------------------------------------------------------------
# phase-86.110: fail (and REPAIR) any test that mutates a git-tracked handoff
# state file.
#
# Third guard in this file, and the same shape of incident as the two above: a
# PASSING unit test -- test_phase_66_1_rail_guard.py -- monkeypatched
# `cycle_health._HISTORY_PATH` to a tmp_path but not `_HEARTBEAT_PATH`, and
# `record_cycle_end` writes BOTH. So it wrote a synthetic `cycle_id="c2"` into
# the real, git-tracked `handoff/.cycle_heartbeat.json` that the dashboard reads
# as a liveness signal. Measured: `c2` appears 0 times in the 174-row
# `cycle_history.jsonl`, so the dashboard was reporting a fresh heartbeat off a
# value no cycle ever produced.
#
# The two-line fix is in that test. This fixture exists for the CLASS: a module
# that exposes several path constants and writes several of them from one public
# call will keep producing this bug, and the next instance will be a different
# file or a different constant. `scripts/qa/heartbeat_leak_sweep_86_110.py`
# enumerates the population statically; this catches it at runtime.
#
# Deliberately FUNCTION-scoped, not session-scoped: the point is to name the
# test that did it. A session-scoped check reports that something leaked and
# leaves you bisecting.
#
# It RESTORES the file before failing. A guard that only reports leaves the
# pollution behind, which is most of the damage.
#
# Prior art: `tree-is-clean` (assert the working tree is unchanged around a
# run). No pytest plugin ships tracked-file snapshot-diff; ODRepair explicitly
# excludes filesystem state.
# ---------------------------------------------------------------------------
import hashlib as _hashlib
import pathlib as _pathlib

import pytest as _pytest

# Git-tracked files that NO test may modify. Add to this list, do not widen it
# to the whole tree: a broad snapshot would flag legitimate build artefacts and
# would be turned off within a week.
# Each protected file carries the RULE for deciding whether a change was a
# legitimate concurrent write by the live system or a test leak. This matters
# because pyfinagent runs local-only: the real autonomous_loop writes these
# exact files from this machine while the suite runs, and a full suite run
# takes ~8 minutes. A guard that blindly restored a snapshot would DELETE a
# real cycle row and then blame an innocent test -- worse than the leak it was
# built to stop.
#
#   "append_only"  -- a legitimate writer only ever APPENDS. If the new content
#                     starts with the snapshot, it is a real append: leave it,
#                     do not fail. Anything else is a rewrite: restore + fail.
#   "ledger_backed" -- the file names a cycle_id. If that id exists in the real
#                     cycle ledger it is a legitimate write: leave it. A
#                     fixture value like "c1"/"c2" appears in ZERO ledger rows,
#                     which is exactly what made the polluted state a lie.
_PROTECTED_HANDOFF_FILES = {
    "handoff/.cycle_heartbeat.json": "ledger_backed",
    "handoff/cycle_history.jsonl": "append_only",
}
_CYCLE_LEDGER = "handoff/cycle_history.jsonl"

_REPO_ROOT = _pathlib.Path(__file__).resolve().parents[2]


def _snapshot_protected() -> dict[str, tuple[bytes, str] | None]:
    out: dict[str, tuple[bytes, str] | None] = {}
    for rel in _PROTECTED_HANDOFF_FILES:
        p = _REPO_ROOT / rel
        try:
            b = p.read_bytes()
            out[rel] = (b, _hashlib.sha256(b).hexdigest())
        except FileNotFoundError:
            out[rel] = None
    return out


def _is_legitimate_concurrent_write(rel: str, before: bytes, after: bytes) -> bool:
    """Could the live system plausibly have made this change?

    Returns True only for a change a real writer could have produced. When in
    doubt it returns False -- the cost of a false alarm is a re-run, the cost
    of a false clear is an undetected pollution of a tracked file.
    """
    rule = _PROTECTED_HANDOFF_FILES.get(rel)
    if rule == "append_only":
        # A real append leaves the previous bytes intact as a prefix.
        return after.startswith(before) and len(after) > len(before)
    if rule == "ledger_backed":
        try:
            import json as _json

            cid = _json.loads(after.decode("utf-8")).get("cycle_id")
            if not cid:
                return False
            ledger = (_REPO_ROOT / _CYCLE_LEDGER).read_text(encoding="utf-8")
            # Substring is deliberate: the ledger is JSONL and this runs in a
            # teardown on every test, so a full parse per test is not worth it.
            # A fixture id like "c2" that appears nowhere is what we are after.
            return f'"cycle_id": "{cid}"' in ledger or f'"cycle_id":"{cid}"' in ledger
        except Exception:
            return False
    return False


@_pytest.fixture(autouse=True)
def _no_tracked_handoff_writes(request):
    before = _snapshot_protected()
    yield
    dirtied = []
    for rel, snap in before.items():
        p = _REPO_ROOT / rel
        try:
            now = p.read_bytes()
            now_sha = _hashlib.sha256(now).hexdigest()
        except FileNotFoundError:
            now, now_sha = None, None
        if snap is None and now_sha is None:
            continue
        if snap is None or now_sha is None or snap[1] != now_sha:
            if (snap is not None and now is not None
                    and _is_legitimate_concurrent_write(rel, snap[0], now)):
                # The live system wrote this while the test ran. Leave it
                # ALONE -- restoring here would destroy real data.
                continue
            dirtied.append(rel)
            # REPAIR before reporting.
            if snap is None:
                p.unlink(missing_ok=True)
            else:
                p.write_bytes(snap[0])
    if dirtied:
        _pytest.fail(
            f"phase-86.110 test guard: {request.node.nodeid} modified git-tracked "
            f"handoff state: {dirtied}. The file has been RESTORED, so the repo is "
            "clean, but the test must isolate its writes. If it drives "
            "cycle_health, monkeypatch BOTH _HISTORY_PATH and _HEARTBEAT_PATH -- "
            "record_cycle_start and record_cycle_end each write the heartbeat. "
            "See scripts/qa/heartbeat_leak_sweep_86_110.py.",
            pytrace=False,
        )
