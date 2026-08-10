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

### The two sets, ENUMERATED (cycle-2 fix)

The cycle-1 Q/A was right that `NEW: (none) / GONE: (none)` is byte-identical
whether it came from a real set diff or from a count rendered in set language.
The members were never shown, so the claim was unauditable. Here they are.

**BEFORE** -- captured at `a16fa5a2` (contract committed, no removals yet), 14 members:

```
FAILED backend/tests/test_phase_23_2_6_sector_cap_emit.py::test_phase_23_2_6_backend_log_has_skipping_buy_evidence
FAILED backend/tests/test_phase_40_2_claude_code_v2_1_140_features.py::test_phase_40_2_settings_json_still_valid_json_after_edit
FAILED backend/tests/test_phase_57_1_reject_binding.py::test_off_identity_prompts_are_verbatim_constants
FAILED backend/tests/test_phase_57_1_reject_binding.py::test_reject_binding_main_path_off_emits_on_blocks
FAILED backend/tests/test_phase_57_1_reject_binding.py::test_reject_binding_swap_path_off_emits_on_blocks
FAILED backend/tests/test_phase_60_3_data_integrity.py::test_60_3_flag_defaults_off
FAILED backend/tests/test_phase_75_17_verification_paths.py::test_masterplan_diff_touches_only_the_ten_sibling_insertions
FAILED backend/tests/test_phase_75_17_verification_paths.py::test_sweep_shape_census_matches_the_corrected_figures
FAILED backend/tests/test_phase_75_prompt_contracts.py::test_operator_decision_note_exists_with_token
FAILED backend/tests/test_phase_75_sre_ops.py::test_c1_runbook_and_operator_token_drafted
FAILED backend/tests/test_phase_75_sre_ops.py::test_c6_no_launchctl_bootstrap_executed_in_ops_scripts
FAILED backend/tests/test_phase_82_39_outcome_rebuild_query.py::test_the_sweeps_recall_limit_is_recorded_not_assumed
FAILED backend/tests/test_phase_86_2_replay_poison_row.py::test_c1_c2_a_poison_row_first_no_longer_strands_the_replay
FAILED backend/tests/test_portfolio_swap.py::test_swap_framework_fills_zero_buy_gap
```

**AFTER** -- captured at `1ed39ccd` (the 7 removals), 14 members:

```
FAILED backend/tests/test_phase_23_2_6_sector_cap_emit.py::test_phase_23_2_6_backend_log_has_skipping_buy_evidence
FAILED backend/tests/test_phase_40_2_claude_code_v2_1_140_features.py::test_phase_40_2_settings_json_still_valid_json_after_edit
FAILED backend/tests/test_phase_57_1_reject_binding.py::test_off_identity_prompts_are_verbatim_constants
FAILED backend/tests/test_phase_57_1_reject_binding.py::test_reject_binding_main_path_off_emits_on_blocks
FAILED backend/tests/test_phase_57_1_reject_binding.py::test_reject_binding_swap_path_off_emits_on_blocks
FAILED backend/tests/test_phase_60_3_data_integrity.py::test_60_3_flag_defaults_off
FAILED backend/tests/test_phase_75_17_verification_paths.py::test_masterplan_diff_touches_only_the_ten_sibling_insertions
FAILED backend/tests/test_phase_75_17_verification_paths.py::test_sweep_shape_census_matches_the_corrected_figures
FAILED backend/tests/test_phase_75_prompt_contracts.py::test_operator_decision_note_exists_with_token
FAILED backend/tests/test_phase_75_sre_ops.py::test_c1_runbook_and_operator_token_drafted
FAILED backend/tests/test_phase_75_sre_ops.py::test_c6_no_launchctl_bootstrap_executed_in_ops_scripts
FAILED backend/tests/test_phase_82_39_outcome_rebuild_query.py::test_the_sweeps_recall_limit_is_recorded_not_assumed
FAILED backend/tests/test_phase_86_2_replay_poison_row.py::test_c1_c2_a_poison_row_first_no_longer_strands_the_replay
FAILED backend/tests/test_portfolio_swap.py::test_swap_framework_fills_zero_buy_gap
```

`comm -13` (new) and `comm -23` (gone) over the two sorted files are both
EMPTY: the sets are identical member for member, not merely equal in count.

**Disclosed rather than glossed:** the two runs are not the same tree. The
AFTER run also carries phase-86.12's 12 new tests (3291 -> 3303 passed), which
is why the passed counts differ. That makes exhibiting the members more
necessary, not less -- and they are identical, so no failure moved either way.

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
