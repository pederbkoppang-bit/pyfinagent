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
