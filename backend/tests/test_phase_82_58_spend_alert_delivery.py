"""phase-82.58 -- the cost-budget degradation alert must actually be DELIVERED.

Until this step, `spend.py::_record_degradation` called the emitter with
`detail=` against a `details=` signature. That raised TypeError into a
swallowing `except -> logger.debug`, so the alert had never fired since
2026-07-23. It was also `severity="P2"`, which `AlertDeduper.should_fire`
suppresses BEFORE the webhook is ever read -- so even a kwarg-only repair
would have delivered nothing.

Why that matters more than a typo: `llm_client.py:452` computes
`tripped = daily >= daily_cap`, and the fail-open value is `(0.0, 0.0)`.
`0.0 >= 25.0` is False, so while the spend fetch is degraded the $25/day
hard-block silently stops enforcing -- and `spend_guard_status()` has ZERO
non-test callers, making this alert the only way an operator could find out.

THE GUARDS HERE ASSERT DELIVERY, NOT BRANCH ENTRY. The whole defect is that
the branch WAS entered every time and nothing came out the other end. So
these tests drive the real `raise_cron_alert_sync` end to end and capture the
outbound HTTP POST.

THE MOCK TRAP, stated because it would have produced a false PASS: patching
`raise_cron_alert_sync` with a plain `MagicMock` accepts `detail=` happily and
reports `called == True` against the BROKEN code. Only `create_autospec`
enforces the signature. These tests do not mock the emitter at all.

These tests must be SYNCHRONOUS. Under a running event loop
`raise_cron_alert_sync` (alerting.py:~276) schedules the coroutine and returns
True without awaiting, so an async test could report success while delivery
never happened.
"""

from __future__ import annotations

import ast
import inspect
import io
import json
import pathlib
import urllib.request
from unittest import mock

import pytest

from backend.services.observability import alerting, spend

REPO = pathlib.Path(__file__).resolve().parents[2]

# The alert this step repairs.
SOURCE = "cost_budget_guard"
ERROR_TYPE = "spend_fetch_degraded"


class _FakeSettings:
    """Pins the environment the criteria describe, so the guard does not pass
    or fail by accident of this machine's config.

    `slack_webhook_url` empty is criterion 2's stated condition; the bot token
    is a DUMMY -- delivery is captured at the socket seam, never sent.
    """

    slack_webhook_url = ""
    slack_bot_token = "xoxb-DUMMY-NOT-A-REAL-TOKEN-82-58"
    slack_channel_id = "C_TEST_82_58"


@pytest.fixture
def captured_posts(monkeypatch):
    """Capture outbound Slack POSTs at `urllib.request.urlopen`.

    That is the true delivery seam: `alerting._bot_token_fallback` imports
    `urllib.request` function-locally and resolves `urlopen` at call time, so
    this patch lands even across the `asyncio.to_thread` hop.
    """
    posts: list[dict] = []

    def fake_urlopen(req, *args, **kwargs):
        posts.append(
            {
                "url": getattr(req, "full_url", str(req)),
                "body": json.loads(req.data.decode()) if getattr(req, "data", None) else None,
            }
        )
        return io.BytesIO(b'{"ok": true}')

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(
        "backend.config.settings.get_settings", lambda: _FakeSettings()
    )

    # A stale one-shot latch or a warm deduper would silently make every
    # assertion below vacuous, so both are reset around each case.
    spend.reset_spend_guard_status()
    alerting.reset_default_deduper()
    yield posts
    spend.reset_spend_guard_status()
    alerting.reset_default_deduper()


def _degrade():
    """Drive the PRODUCTION path: fetch_spend() -> except -> _record_degradation.

    Not a direct call to the private helper -- the criterion is about what the
    real entry point does when the fetch fails.
    """
    with mock.patch(
        "google.cloud.bigquery.Client", side_effect=RuntimeError("BQ down (82.58 fixture)")
    ):
        return spend.fetch_spend()


# ---------------------------------------------------------------------------
# criterion 1 -- a degraded fetch DELIVERS, asserted on the emitted payload
# ---------------------------------------------------------------------------


def test_degraded_spend_fetch_delivers_an_alert_payload(captured_posts):
    """FAILS against the shipped `detail=`/P2 code: with `detail=` the emitter
    raises TypeError into the swallowing except, and with P2 the deduper drops
    it before any HTTP call -- either way `captured_posts` stays empty."""
    daily, monthly = _degrade()

    # precondition: the fail-open really happened (else we would be asserting
    # delivery for a call that never degraded)
    assert (daily, monthly) == (0.0, 0.0)
    assert spend.spend_guard_status()["degraded_count"] == 1

    assert captured_posts, (
        "NO alert was delivered. The degradation path ran and produced nothing "
        "on the wire -- this is exactly the 82.58 defect."
    )
    post = captured_posts[0]
    assert "slack.com" in post["url"]

    text = post["body"]["text"]
    assert SOURCE in text
    assert "fail-open" in text.lower()
    # the operational consequence must survive into the delivered message
    assert "hard-block cannot trip" in text
    assert post["body"]["channel"] == _FakeSettings.slack_channel_id


def test_delivery_is_not_merely_a_constructed_call(captured_posts):
    """The payload must carry the ORIGINating exception, proving the delivered
    message is built from the real degradation and not a canned string."""
    _degrade()
    assert captured_posts
    assert "BQ down (82.58 fixture)" in captured_posts[0]["body"]["text"]


# ---------------------------------------------------------------------------
# criterion 2 -- severity must be one that actually reaches the operator
# ---------------------------------------------------------------------------


def test_emitted_severity_is_deliverable_per_the_live_critical_set(captured_posts):
    """Asserted against the imported live `_CRITICAL_SEVERITIES`, never a
    hardcoded "P1" -- if that set is ever narrowed, this test must fail."""
    _degrade()
    assert captured_posts

    text = captured_posts[0]["body"]["text"]
    assert text.startswith("["), f"cannot parse severity from {text!r}"
    severity = text[1 : text.index("]")]

    live = alerting._CRITICAL_SEVERITIES
    assert live, "the live critical-severity set is empty -- check would be vacuous"
    assert severity in live, (
        f"emitted severity {severity!r} is not in the live deliverable set "
        f"{sorted(live)}; with slack_webhook_url empty this alert is dropped"
    )


def test_the_call_sites_severity_literal_is_deliverable_independently_of_delivery():
    """Criterion 2 with a guard that can actually FAIL on severity alone.

    The delivery-based assertion above is real but it is NOT what kills a
    severity regression, and saying otherwise would mis-credit it: with
    `slack_webhook_url` empty, `alerting.py:210-224` routes to
    `_bot_token_fallback` ONLY when the severity is already critical. So any
    captured POST necessarily carries a deliverable severity, and
    `assert severity in live` on a captured post cannot fail -- the delivery
    assertion is what dies. (Caught by the 82.58 Q/A; it is the same
    can't-fail-guard shape this step exists to hunt.)

    This test reads the severity LITERAL out of the production call site by AST
    and checks it against the live set with no POST involved, so it fails on a
    severity regression even if delivery were broken for some other reason.
    """
    src = (REPO / "backend/services/observability/spend.py").read_text(encoding="utf-8")
    tree = ast.parse(src)

    severities = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", None)
        if name not in _ALERT_FNS:
            continue
        for kw in node.keywords:
            if kw.arg == "severity" and isinstance(kw.value, ast.Constant):
                severities.append((node.lineno, kw.value.value))

    assert severities, (
        "no literal severity found at any alert call site in spend.py -- the "
        "instrument is broken, not the code clean"
    )

    live = alerting._CRITICAL_SEVERITIES
    assert live, "live critical-severity set is empty -- check would be vacuous"
    undeliverable = [(ln, s) for ln, s in severities if s not in live]
    assert not undeliverable, (
        f"spend.py alert call site(s) {undeliverable} carry a severity outside "
        f"the live deliverable set {sorted(live)}; with slack_webhook_url empty "
        "these are logged and dropped, never delivered"
    )


def test_a_non_critical_severity_would_not_reach_the_operator():
    """The negative half of criterion 2: proves the severity check above is
    load-bearing rather than incidentally true. P2 is dropped by the same code
    path that delivers P1."""
    assert "P2" not in alerting._CRITICAL_SEVERITIES
    assert "P3" not in alerting._CRITICAL_SEVERITIES


# ---------------------------------------------------------------------------
# criterion 3 -- negative control: a healthy fetch is silent
# ---------------------------------------------------------------------------


def test_healthy_spend_fetch_emits_no_alert(captured_posts):
    """Without this, a guard that fired unconditionally would pass criterion 1."""

    class _Row(dict):
        pass

    fake_client = mock.MagicMock()
    fake_client.query.return_value.result.return_value = [
        _Row(daily_bytes=1e12, monthly_bytes=5e12)
    ]

    with mock.patch("google.cloud.bigquery.Client", return_value=fake_client):
        daily, monthly = spend.fetch_spend()

    assert daily == pytest.approx(6.25)
    assert monthly == pytest.approx(31.25)
    assert spend.spend_guard_status()["degraded_count"] == 0
    assert captured_posts == [], (
        f"a HEALTHY fetch emitted {len(captured_posts)} alert(s) -- the guard "
        "would pass by always firing"
    )


def test_the_alert_fires_once_per_process_not_per_call(captured_posts):
    """The `_ALERTED` latch is what makes P1 acceptable here (alert fatigue is
    nil). If the latch regressed, a flapping fetch would page repeatedly.

    The deduper is reset between degradations ON PURPOSE. Without that reset
    this test passes even with the latch removed, because `AlertDeduper` would
    suppress calls 2 and 3 on its own (P1 takes the critical branch, which
    re-fires only after `repeat_hours`) -- so it would be asserting a property
    the deduper guarantees while crediting the latch. Measured: mutating the
    latch to `if True:` left the un-reset version GREEN. Resetting the deduper
    removes the confound, leaving the latch as the only thing that can suppress.
    """
    for _ in range(3):
        _degrade()
        alerting.reset_default_deduper()

    assert spend.spend_guard_status()["degraded_count"] == 3
    assert len(captured_posts) == 1, (
        f"expected exactly one page across three degradations, got "
        f"{len(captured_posts)} -- the _ALERTED latch is not holding"
    )


# ---------------------------------------------------------------------------
# criterion 4 -- structural sweep of every call site, bound against the signature
# ---------------------------------------------------------------------------

_ALERT_FNS = ("raise_cron_alert_sync", "raise_cron_alert")


def _import_resolved_alert_calls() -> list[dict]:
    """Every call to an alerting emitter, resolved through imports.

    A bare-name AST sweep is NOT good enough here: measured during the research
    gate, matching on the bare attribute name yields 65 candidates for 3 real
    hits, colliding with `csv.writer`, `yfinance.history`, `json.loads` and
    `numpy.percentile`. So a name is only accepted when it was actually
    imported from the alerting module, referenced as `<alias>.<fn>` on that
    module, or defined in the module itself.
    """
    out: list[dict] = []
    for path in sorted(REPO.rglob("*.py")):
        rel = path.relative_to(REPO).as_posix()
        if rel.startswith((".venv/", "node_modules/", "frontend/")):
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue

        direct: set[str] = set()  # local name -> imported emitter
        module_aliases: set[str] = set()  # local name bound to the alerting module
        is_alerting_module = rel.endswith("services/observability/alerting.py")

        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                if mod.endswith("alerting"):
                    for a in node.names:
                        if a.name in _ALERT_FNS:
                            direct.add(a.asname or a.name)
                elif mod.endswith("services.observability"):
                    for a in node.names:
                        if a.name == "alerting":
                            module_aliases.add(a.asname or a.name)
                        elif a.name in _ALERT_FNS:
                            direct.add(a.asname or a.name)
            elif isinstance(node, ast.Import):
                for a in node.names:
                    if a.name.endswith("observability.alerting"):
                        module_aliases.add(a.asname or a.name.split(".")[-1])

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            if isinstance(fn, ast.Attribute):
                if fn.attr not in _ALERT_FNS:
                    continue
                base = fn.value
                base_name = getattr(base, "id", None)
                if base_name not in module_aliases and not is_alerting_module:
                    continue
                resolved = fn.attr
            elif isinstance(fn, ast.Name):
                if fn.id in direct:
                    resolved = fn.id if fn.id in _ALERT_FNS else None
                    # honour aliasing: recover the real emitter name
                    if resolved is None:
                        continue
                elif is_alerting_module and fn.id in _ALERT_FNS:
                    resolved = fn.id
                else:
                    continue
            else:
                continue

            out.append(
                {
                    "file": rel,
                    "line": node.lineno,
                    "fn": resolved,
                    "kwargs": [k.arg for k in node.keywords if k.arg is not None],
                    "starstar": any(k.arg is None for k in node.keywords),
                    "nargs": len(node.args),
                }
            )
    return out


def test_the_call_site_sweep_actually_sees_things():
    """Recall test. 'Found no mismatches' and 'the sweep is broken' look
    identical from the outside, so the instrument is checked against sites
    known to exist before its negative result is trusted."""
    calls = _import_resolved_alert_calls()
    assert calls, "DERIVED SET IS EMPTY -- the sweep is broken, not the repo clean"

    files = {c["file"] for c in calls}
    # known-present anchors, re-derived rather than assumed
    for anchor in (
        "backend/services/observability/spend.py",
        "backend/services/drawdown_alarm.py",
        "backend/services/observability/alerting.py",
    ):
        assert anchor in files, f"sweep failed to find the known call site in {anchor}"
    assert len(calls) >= 20, f"only {len(calls)} call sites found; expected the repo's ~33"


def test_every_alert_call_site_binds_against_the_live_signature():
    """criterion 4. This is the check that would have caught `detail=` on the
    day it shipped, 14 days before an audit found it by hand."""
    sigs = {
        "raise_cron_alert_sync": inspect.signature(alerting.raise_cron_alert_sync),
        "raise_cron_alert": inspect.signature(alerting.raise_cron_alert),
    }
    calls = _import_resolved_alert_calls()
    assert calls, "derived set empty -- refusing to report 'no mismatches'"

    mismatches = []
    for c in calls:
        if c["starstar"]:
            continue  # **kwargs cannot be bound statically
        try:
            sigs[c["fn"]].bind(
                *[None] * c["nargs"], **{k: None for k in c["kwargs"]}
            )
        except TypeError as exc:
            mismatches.append(f"{c['file']}:{c['line']} {c['fn']}(...) -> {exc}")

    assert not mismatches, (
        "alert call sites whose keyword arguments do not bind against the live "
        "signature (each raises TypeError at runtime, and every one of these "
        "call sites is inside a swallowing except):\n  " + "\n  ".join(mismatches)
    )


def test_the_repaired_call_site_passes_details_not_detail():
    """Pins the specific repair so a revert is loud rather than silent."""
    src = (REPO / "backend/services/observability/spend.py").read_text(encoding="utf-8")
    call_ix = src.index("raise_cron_alert_sync(\n")
    call = src[call_ix : src.index(")", src.index("persists."))]
    assert "details=" in call, "the repaired call no longer passes `details=`"
    assert "detail=(" not in call, "`detail=` is back -- the TypeError defect has returned"
    assert 'severity="P1"' in call, "severity regressed off the deliverable set"
