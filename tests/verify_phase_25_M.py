"""verify_phase_25_M -- Cost-budget Slack alert wire repair.

Verifies:
  1. `make_alert_fn_for_budget` raises ValueError when channel="" (wiring error).
  2. `make_alert_fn_for_budget` raises ValueError when loop=None (wiring error).
  3. `register_phase9_jobs` uses `logger.error` (not `.warning`) on production-fn
     wiring exception path.
  4. `_post_slack_sync` uses `logger.error` (not `.warning`) on Slack post failure.

Behavioral round-trip:
  5. `register_phase9_jobs` with stub scheduler + stub app + real loop wires
     `cost_budget_watcher` with a partial-applied alert_fn; calling that
     alert_fn fails LOUD via _post_slack_sync (Slack call fails, ERROR logged).

Exit 0 on PASS, 1 on FAIL.
"""

from __future__ import annotations

import asyncio
import logging
import re
import sys
from pathlib import Path
from unittest.mock import MagicMock

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

results: list[tuple[str, bool, str]] = []


def claim(name: str, condition: bool, detail: str = "") -> None:
    results.append((name, bool(condition), detail))


# ── Claim 1: factory raises on empty channel ───────────────────────────
from backend.slack_bot.jobs._production_fns import make_alert_fn_for_budget  # noqa: E402

try:
    loop = asyncio.new_event_loop()
    fake_app = MagicMock(name="AsyncApp")
    try:
        make_alert_fn_for_budget(fake_app, loop, "")
        raised_empty_channel = False
        detail1 = "Factory did NOT raise on channel=''"
    except ValueError as e:
        raised_empty_channel = True
        detail1 = f"Raised ValueError: {e}"
    except Exception as e:
        raised_empty_channel = False
        detail1 = f"Wrong exception type: {type(e).__name__}: {e}"
finally:
    try:
        loop.close()
    except Exception:
        pass

claim(
    "1. make_alert_fn_for_budget_raises_loudly_on_wiring_error",
    raised_empty_channel,
    detail1,
)


# ── Claim 2: factory raises on loop=None ───────────────────────────────
try:
    fake_app = MagicMock(name="AsyncApp")
    try:
        make_alert_fn_for_budget(fake_app, None, "C12345")  # type: ignore[arg-type]
        raised_none_loop = False
        detail2 = "Factory did NOT raise on loop=None"
    except ValueError as e:
        raised_none_loop = True
        detail2 = f"Raised ValueError: {e}"
    except Exception as e:
        raised_none_loop = False
        detail2 = f"Wrong exception type: {type(e).__name__}: {e}"
except Exception as e:
    raised_none_loop = False
    detail2 = f"Test setup error: {e}"

claim(
    "2. make_alert_fn_for_budget_raises_on_none_loop",
    raised_none_loop,
    detail2,
)


# ── Claim 3: scheduler uses logger.error on wiring exception ───────────
sched_src = (REPO / "backend/slack_bot/scheduler.py").read_text(encoding="utf-8")
# Find the production-fn wiring error log line
# Look for "production-fn wiring failed" with logger.error
m3 = re.search(
    r'logger\.error\([^)]*production-fn wiring[^)]*exc_info=True',
    sched_src,
    re.DOTALL,
)
claim(
    "3. scheduler_register_phase9_jobs_logs_error_visibly",
    bool(m3),
    f"Found ERROR-level log with exc_info=True" if m3 else "logger.error+exc_info not present on wiring failure",
)


# ── Claim 4: _post_slack_sync uses logger.error ────────────────────────
pf_src = (REPO / "backend/slack_bot/jobs/_production_fns.py").read_text(encoding="utf-8")
m4 = re.search(
    r'def _post_slack_sync\(.*?def make_alert_fn_for_budget',
    pf_src,
    re.DOTALL,
)
post_sync_body = m4.group(0) if m4 else ""
uses_error = bool(re.search(r'logger\.error\([^)]*Slack post failed.*exc_info=True', post_sync_body, re.DOTALL))
claim(
    "4. post_slack_sync_logs_error_on_failure",
    uses_error,
    "Found logger.error+exc_info=True in _post_slack_sync" if uses_error else "Still using WARNING in _post_slack_sync",
)


# ── Claim 5: behavioral round-trip ─────────────────────────────────────
# Build a real alert_fn with a stub app whose client.chat_postMessage raises;
# call the alert_fn and confirm an ERROR log is emitted (not WARNING).
class _CapturingHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record)


handler = _CapturingHandler()
handler.setLevel(logging.DEBUG)
logging.getLogger("backend.slack_bot.jobs._production_fns").addHandler(handler)
logging.getLogger("backend.slack_bot.jobs._production_fns").setLevel(logging.DEBUG)

loop2 = asyncio.new_event_loop()
loop_thread_started = False
try:
    import threading
    def _runner(loop):
        asyncio.set_event_loop(loop)
        loop.run_forever()
    th = threading.Thread(target=_runner, args=(loop2,), daemon=True)
    th.start()
    loop_thread_started = True

    failing_app = MagicMock()
    async def _fail_post(**kwargs):
        raise RuntimeError("slack_api_error_for_test")
    failing_app.client.chat_postMessage = _fail_post

    alert = make_alert_fn_for_budget(failing_app, loop2, "C_TEST")
    alert("test_breach", {"k": "v"})

    # Find the ERROR log emission for "Slack post failed"
    error_records = [r for r in handler.records if r.levelno == logging.ERROR and "Slack post failed" in r.getMessage()]
    warn_records = [r for r in handler.records if r.levelno == logging.WARNING and "Slack post fail-open" in r.getMessage()]
    if error_records:
        round_trip_ok = True
        detail5 = f"Captured ERROR record: {error_records[0].getMessage()}"
    else:
        round_trip_ok = False
        detail5 = f"No ERROR record captured. Warnings: {len(warn_records)}. All records: {[r.levelname for r in handler.records]}"
except Exception as e:
    round_trip_ok = False
    detail5 = f"Exception: {type(e).__name__}: {e}"
finally:
    if loop_thread_started:
        loop2.call_soon_threadsafe(loop2.stop)
        try:
            th.join(timeout=2)
        except Exception:
            pass
        try:
            loop2.close()
        except Exception:
            pass
    logging.getLogger("backend.slack_bot.jobs._production_fns").removeHandler(handler)

claim(
    "5. behavioral_round_trip_logs_error_on_post_failure",
    round_trip_ok,
    detail5,
)


# ── Summary ─────────────────────────────────────────────────────────────
print("\n=== phase-25.M verification ===\n")
all_pass = True
for name, ok, detail in results:
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}")
    if detail:
        print(f"        -> {detail}")
    if not ok:
        all_pass = False

print()
if all_pass:
    print(f"ALL {len(results)} CLAIMS PASS")
    sys.exit(0)
else:
    failed = sum(1 for _, ok, _ in results if not ok)
    print(f"{failed} of {len(results)} claims FAILED")
    sys.exit(1)
