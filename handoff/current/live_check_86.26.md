# live_check -- phase-86.26

Captured 2026-08-10 05:03:01 CEST. Every block is stdout.

## A. Ruff BEFORE (from the pre-fix commit) and AFTER

```
$ git stash-free check: ruff against the PRE-FIX revision a16fa5a2
backend/agents/bias_detector.py:12:8: F401 [*] `json` imported but unused
backend/agents/conflict_detector.py:8:8: F401 [*] `json` imported but unused
backend/agents/conflict_detector.py:11:20: F401 [*] `typing.Optional` imported but unused
backend/agents/memory.py:12:8: F401 [*] `json` imported but unused
backend/services/outcome_tracker.py:11:32: F401 [*] `datetime.timedelta` imported but unused
backend/services/outcome_tracker.py:12:20: F401 [*] `typing.Optional` imported but unused
backend/services/outcome_tracker.py:17:63: F401 [*] `backend.services.perf_metrics.compute_benchmark_return` imported but unused
Found 7 errors.
[*] 7 fixable with the `--fix` option.

$ AFTER -- the immutable command, full scope:
All checks passed!
exit=0
```

## B. The per-name consumer proof (criterion 2), with the method validated first

```
METHOD VALIDATION -- can the consumer-finder actually find consumers?
  OK    backend.services.recommendation_vocab.is_buy_intent              hits=8   expected=some
  OK    backend.services.recommendation_vocab.canonical_recommendation   hits=9   expected=some
  OK    backend.services.recommendation_vocab.no_such_name_xyzzy         hits=0   expected=none
  OK    backend.does.not.exist.is_buy_intent              hits=0   expected=none
  Method validated in both directions.

phase-86.26 -- 0 F401 finding(s), RE-DERIVED by ruff
scope: 8 files

nothing to verify.
```

## C. The named suspect -- compute_benchmark_return is NOT a re-export

```
$ grep -rn compute_benchmark_return backend --include=*.py
backend/tests/test_dod4_tier1_coverage_investment.py:346:    from backend.services.perf_metrics import compute_benchmark_return
backend/tests/test_dod4_tier1_coverage_investment.py:348:    assert abs(compute_benchmark_return(holding_days=365, annual_rate=0.10) - 10.0) < 0.5
backend/tests/test_dod4_tier1_coverage_investment.py:350:    assert compute_benchmark_return(holding_days=0) == 0.0
backend/tests/test_dod4_tier1_coverage_investment.py:352:    assert compute_benchmark_return(holding_days=-5) == 0.0
backend/services/perf_metrics.py:536:def compute_benchmark_return(holding_days: int, annual_rate: float = 0.10) -> float:
backend/services/perf_metrics.py:560:    return return_pct > compute_benchmark_return(holding_days, annual_rate)

The only non-defining consumer imports it from perf_metrics (the ORIGIN),
not through outcome_tracker. outcome_tracker keeps beat_benchmark, which it
does use.
```

## D. Criterion 3 -- no noqa, no rule-set change
```
$ git diff -U0 -- backend/ | grep -c noqa
0
```

## E. Criterion 5 -- the failure SET, diffed (not counted)
```
before: 14 failed
after : 14 failed, 3303 passed

NEW  (attributable to the removals):
GONE (attributable to the removals):

Both empty -- the failure set is identical member for member.
```

## F. The diff
```
diff --git a/backend/agents/bias_detector.py b/backend/agents/bias_detector.py
index da24049a..19c48b4f 100644
--- a/backend/agents/bias_detector.py
+++ b/backend/agents/bias_detector.py
@@ -9,7 +9,6 @@ Detects:
   - Source diversity issues (narrow evidence base)
 """
 
-import json
 import logging
 from dataclasses import asdict, dataclass, field
 from typing import Optional
diff --git a/backend/agents/conflict_detector.py b/backend/agents/conflict_detector.py
index e43c4dde..d52e740d 100644
--- a/backend/agents/conflict_detector.py
+++ b/backend/agents/conflict_detector.py
@@ -5,10 +5,8 @@ Identifies discrepancies between what the LLM "believes" (from training data) an
 current market data shows. These conflicts highlight where the model's knowledge is stale.
 """
 
-import json
 import logging
 from dataclasses import asdict, dataclass, field
-from typing import Optional
 from backend.services.recommendation_vocab import canonical_recommendation, STRONG_BUY, BUY, SELL  # phase-86.22
 
 logger = logging.getLogger(__name__)
diff --git a/backend/agents/memory.py b/backend/agents/memory.py
index a6bb51a0..0cebe23e 100644
--- a/backend/agents/memory.py
+++ b/backend/agents/memory.py
@@ -9,7 +9,6 @@ Research basis: TradingAgents FinancialSituationMemory — agents learn from
 past mistakes to avoid repeating wrong BUY/SELL/HOLD calls.
 """
 
-import json
 import logging
 import re
 from datetime import datetime, timezone
diff --git a/backend/services/outcome_tracker.py b/backend/services/outcome_tracker.py
index 1507c6ef..44e711ef 100644
--- a/backend/services/outcome_tracker.py
+++ b/backend/services/outcome_tracker.py
@@ -8,13 +8,12 @@ agent_memories BigQuery table for BM25-based retrieval in future analyses.
 
 import json
 import logging
-from datetime import datetime, timedelta, timezone
-from typing import Optional
+from datetime import datetime, timezone
 
 from backend.config.settings import Settings
 from backend.services.recommendation_vocab import is_buy_intent, is_sell_intent  # phase-86.22
 from backend.db.bigquery_client import BigQueryClient
-from backend.services.perf_metrics import compute_return_pct, compute_benchmark_return, beat_benchmark as _beat_benchmark
+from backend.services.perf_metrics import compute_return_pct, beat_benchmark as _beat_benchmark
 from backend.tools.yfinance_tool import get_comprehensive_financials
 
 logger = logging.getLogger(__name__)
```
