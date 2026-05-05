"""phase-23.2.18: immutable verification.

Asserts the four fix surfaces are in place:
1. backend/services/observability/alerting.py: raise_cron_alert is async,
   raise_cron_alert_sync exists, webhook routing path present.
2. backend/services/autonomous_loop.py: outer asyncio.timeout block;
   asyncio.TimeoutError handler; post-finally raise_cron_alert_sync call.
3. backend/services/kill_switch.py: trigger allowlist + raise_cron_alert_sync
   guard so manual/test/bench triggers stay silent.
4. scripts/launchd/backend_watchdog.sh: SLACK_WEBHOOK_URL extraction +
   curl POST before the launchctl kickstart -k line.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def _parse(rel: str) -> ast.Module:
    return ast.parse(_read(rel))


def check_alerting():
    rel = "backend/services/observability/alerting.py"
    text = _read(rel)
    mod = ast.parse(text)
    funcs = {n.name: n for n in ast.walk(mod) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    assert "raise_cron_alert" in funcs, f"raise_cron_alert missing from {rel}"
    assert isinstance(funcs["raise_cron_alert"], ast.AsyncFunctionDef), \
        "raise_cron_alert must be async (was sync; see phase-23.2.18 contract)"
    assert "raise_cron_alert_sync" in funcs, f"raise_cron_alert_sync missing from {rel}"
    assert "send_notification" in text, "must route via backend.tools.slack.send_notification"
    assert "slack_webhook_url" in text, "must read settings.slack_webhook_url"
    return f"OK {rel}"


def check_autonomous_loop():
    rel = "backend/services/autonomous_loop.py"
    text = _read(rel)
    ast.parse(text)
    assert "asyncio.timeout(" in text, "outer asyncio.timeout(...) ceiling missing"
    assert "paper_cycle_max_seconds" in text, "settings.paper_cycle_max_seconds knob missing"
    assert "except asyncio.TimeoutError" in text, "asyncio.TimeoutError handler missing"
    assert '"timeout"' in text, "status=timeout branch missing"
    assert "raise_cron_alert_sync" in text, "post-finally cycle-failure alert missing"
    assert 'source="autonomous_loop"' in text, "alert source label missing"
    return f"OK {rel}"


def check_kill_switch():
    rel = "backend/services/kill_switch.py"
    text = _read(rel)
    ast.parse(text)
    assert "raise_cron_alert_sync" in text, "kill_switch must call raise_cron_alert_sync on auto-pause"
    assert '"manual"' in text and '"bench-1"' in text, "manual/bench trigger allowlist missing"
    assert "_MANUAL_TRIGGERS" in text or "manual" in text, "trigger allowlist gate missing"
    return f"OK {rel}"


def check_watchdog():
    rel = "scripts/launchd/backend_watchdog.sh"
    text = _read(rel)
    assert "SLACK_WEBHOOK_URL" in text, "watchdog must read SLACK_WEBHOOK_URL from .env"
    # the curl line must precede the kickstart line
    curl_pos = text.find("curl -sS -m 5 -X POST")
    # Find the actual `launchctl kickstart -k "...` execution line, not the
    # description comment near the top of the script.
    m = re.search(r'^launchctl kickstart -k\b', text, re.MULTILINE)
    assert m is not None, "launchctl kickstart execution line missing"
    kick_pos = m.start()
    assert curl_pos > 0, "curl POST line missing"
    assert curl_pos < kick_pos, "Slack alert curl must precede kickstart -k (else SIGKILL hits first)"
    return f"OK {rel}"


def check_test_exists():
    rel = "tests/services/test_cycle_failure_alerts.py"
    text = _read(rel)
    ast.parse(text)
    assert "test_kill_switch_auto_pause_fires_alert" in text
    assert "test_kill_switch_manual_pause_does_not_alert" in text
    assert "test_raise_cron_alert_fires_webhook_on_cycle_error" in text
    return f"OK {rel}"


def main() -> int:
    checks = [
        check_alerting,
        check_autonomous_loop,
        check_kill_switch,
        check_watchdog,
        check_test_exists,
    ]
    failed = 0
    for fn in checks:
        try:
            print(fn())
        except AssertionError as e:
            print(f"FAIL {fn.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"ERROR {fn.__name__}: {e!r}")
            failed += 1
    if failed:
        print(f"\n{failed} verification(s) failed")
        return 1
    print("\nphase-23.2.18 verification: ALL PASS (5/5)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
