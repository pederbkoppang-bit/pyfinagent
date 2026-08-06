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
