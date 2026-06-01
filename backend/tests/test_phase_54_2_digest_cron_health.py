"""phase-54.2: the morning digest's optional cron-health line for the operator's
remote week.

DO-NO-HARM: with the new `cron_health` kwarg defaulting to None, the digest is
byte-identical to before (no extra block). When the scheduler supplies a line, it
renders as a single section block immediately before the footer divider. The
scheduler's `_compute_cron_health` is fail-open (any error -> None) so a
jobs-endpoint hiccup never blocks the operator's daily lifeline digest.
"""
from __future__ import annotations

import asyncio

import pytest

ENVELOPE = {
    "portfolio": {
        "total_nav": 24098.52,
        "starting_capital": 20000.0,
        "total_pnl_pct": 20.49,
    }
}
REPORTS = [{"ticker": "MU", "final_score": 7.5, "recommendation": "BUY"}]


def test_cron_health_default_is_byte_identical():
    """Omitting the kwarg and passing None both produce the SAME blocks, and NO
    cron-health section is present -- the change is purely additive."""
    from backend.slack_bot.formatters import format_morning_digest

    omitted = format_morning_digest(ENVELOPE, REPORTS)
    explicit_none = format_morning_digest(ENVELOPE, REPORTS, cron_health=None)
    assert omitted == explicit_none
    # no section block carries a cron-health marker
    texts = [b.get("text", {}).get("text", "") for b in omitted if b.get("type") == "section"]
    assert not any("Crons:" in t for t in texts)


def test_cron_health_line_renders_before_footer():
    """When provided, the cron line is a section block, and it sits immediately
    before the footer divider+context (so it reads under portfolio/analyses)."""
    from backend.slack_bot.formatters import format_morning_digest

    line = ":white_check_mark: *Crons:* 19/19 healthy"
    blocks = format_morning_digest(ENVELOPE, REPORTS, cron_health=line)

    # exactly one section carries the cron line
    cron_sections = [
        i for i, b in enumerate(blocks)
        if b.get("type") == "section" and b.get("text", {}).get("text") == line
    ]
    assert len(cron_sections) == 1
    cron_idx = cron_sections[0]
    # the footer is the LAST divider + the context block; the cron section precedes it
    assert blocks[-1]["type"] == "context"
    assert blocks[-2]["type"] == "divider"
    assert cron_idx < len(blocks) - 2


def test_cron_health_adds_exactly_one_block():
    from backend.slack_bot.formatters import format_morning_digest

    base = format_morning_digest(ENVELOPE, REPORTS)
    with_line = format_morning_digest(ENVELOPE, REPORTS, cron_health=":warning: *Crons:* 17/19 healthy -- FAILED: a, b")
    assert len(with_line) == len(base) + 1


# ---- scheduler._compute_cron_health (fail-open) -------------------------------

class _FakeResp:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload


class _FakeClient:
    """Minimal async httpx-like client; .get returns a preset response or raises."""
    def __init__(self, resp=None, raises=False):
        self._resp = resp
        self._raises = raises

    async def get(self, url):
        if self._raises:
            raise RuntimeError("boom")
        return self._resp


def _run(coro):
    return asyncio.run(coro)


def test_compute_cron_health_all_green():
    from backend.slack_bot.scheduler import _compute_cron_health
    jobs = {"jobs": [{"id": "morning_digest", "status": "ok"},
                     {"id": "daily_price_refresh", "status": "scheduled"},
                     {"id": "x", "status": "running"}]}
    line = _run(_compute_cron_health(_FakeClient(_FakeResp(200, jobs))))
    assert line is not None
    assert "3/3 healthy" in line
    assert "FAILED" not in line


def test_compute_cron_health_reports_failures():
    from backend.slack_bot.scheduler import _compute_cron_health
    jobs = {"jobs": [{"id": "ok_job", "status": "ok"},
                     {"id": "bad_job", "status": "failed"}]}
    line = _run(_compute_cron_health(_FakeClient(_FakeResp(200, jobs))))
    assert "FAILED: bad_job" in line
    assert "1/2 healthy" in line


def test_compute_cron_health_fail_open_on_error():
    from backend.slack_bot.scheduler import _compute_cron_health
    assert _run(_compute_cron_health(_FakeClient(raises=True))) is None


def test_compute_cron_health_fail_open_on_non_200():
    from backend.slack_bot.scheduler import _compute_cron_health
    assert _run(_compute_cron_health(_FakeClient(_FakeResp(503, {})))) is None


def test_compute_cron_health_none_on_empty():
    from backend.slack_bot.scheduler import _compute_cron_health
    assert _run(_compute_cron_health(_FakeClient(_FakeResp(200, {"jobs": []})))) is None


# ---- phase-54.2 cycle-2: system_state (kill-switch + go-live-gate) -------------

class _RouteClient:
    """Routes .get by URL substring -> a (status, payload) tuple; raises if marked."""
    def __init__(self, routes: dict, raises=False):
        self._routes = routes
        self._raises = raises

    async def get(self, url):
        if self._raises:
            raise RuntimeError("boom")
        for key, (status, payload) in self._routes.items():
            if key in url:
                return _FakeResp(status, payload)
        return _FakeResp(404, {})


def test_system_state_default_is_byte_identical():
    from backend.slack_bot.formatters import format_morning_digest
    omitted = format_morning_digest(ENVELOPE, REPORTS)
    both_none = format_morning_digest(ENVELOPE, REPORTS, cron_health=None, system_state=None)
    assert omitted == both_none
    texts = [b.get("text", {}).get("text", "") for b in omitted if b.get("type") == "section"]
    assert not any("Kill switch" in t or "Go-live gate" in t for t in texts)


def test_system_state_renders_one_block():
    from backend.slack_bot.formatters import format_morning_digest
    line = ":large_green_circle: *Kill switch:* ACTIVE\n*Go-live gate:* NOT ELIGIBLE (1/5)"
    base = format_morning_digest(ENVELOPE, REPORTS)
    out = format_morning_digest(ENVELOPE, REPORTS, system_state=line)
    assert len(out) == len(base) + 1
    assert any(b.get("type") == "section" and b.get("text", {}).get("text") == line for b in out)


def test_compute_system_state_active_and_gate():
    from backend.slack_bot.scheduler import _compute_system_state
    routes = {
        "kill-switch": (200, {"paused": False, "breach": {
            "any_breached": False, "daily_loss_pct": -1.5, "daily_loss_limit_pct": 4.0,
            "trailing_dd_pct": -0.1, "trailing_dd_limit_pct": 10.0}}),
        "gate": (200, {"promote_eligible": False, "booleans": {
            "a": True, "b": False, "c": False, "d": False, "e": False}}),
    }
    line = _run(_compute_system_state(_RouteClient(routes)))
    assert "Kill switch:* ACTIVE" in line
    assert "daily -1.5%/4%" in line and "trail -0.1%/10%" in line
    assert "Go-live gate:* NOT ELIGIBLE (1/5)" in line


def test_compute_system_state_paused():
    from backend.slack_bot.scheduler import _compute_system_state
    routes = {
        "kill-switch": (200, {"paused": True, "pause_reason": "manual", "breach": {}}),
        "gate": (200, {"promote_eligible": True, "booleans": {"a": True}}),
    }
    line = _run(_compute_system_state(_RouteClient(routes)))
    assert "PAUSED -- manual" in line
    assert "ELIGIBLE (1/1)" in line


def test_compute_system_state_breach():
    from backend.slack_bot.scheduler import _compute_system_state
    routes = {"kill-switch": (200, {"paused": False, "breach": {"any_breached": True}}),
              "gate": (404, {})}
    line = _run(_compute_system_state(_RouteClient(routes)))
    assert "BREACH" in line


def test_compute_system_state_fail_open():
    from backend.slack_bot.scheduler import _compute_system_state
    assert _run(_compute_system_state(_RouteClient({}, raises=True))) is None
