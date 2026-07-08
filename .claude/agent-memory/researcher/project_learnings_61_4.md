---
name: learnings-61-4-audit
description: phase-61.4 pre-pay audit — STRING-column coercion root cause, second broken swallowed reader in slack bot, sprint-tile chain, >=10-divergences timing dependency
metadata:
  type: project
---

phase-61.4 pre-pay research (brief 2026-07-08, archived under phase-61.4 when the step closes):

- Root-cause mechanism: GoogleSQL coerces STRING *literals* and *query
  parameters* to TIMESTAMP but NEVER STRING *column expressions*
  (conversion_rules coercion table has empty Coerce-to for STRING). So
  `created_at >= TIMESTAMP_SUB(...)` at bigquery_client.py:957 400s while
  the adjacent STRING-param comparisons (:693, :978) are clean.
- SECOND broken swallowed reader (beyond the masterplan audit basis):
  `backend/slack_bot/jobs/_production_fns.py:219-227` nightly_outcome_rebuild
  queries nonexistent `timestamp` + `realized_pnl` columns (paper_trades has
  created_at:STRING, realized_pnl_pct) -> permanent 400 swallowed fail-open
  -> outcome_tracking never rebuilt. Same defect class; fix or clean-bill it
  in 61.4's audit deliverable.
- Timing dependency: the immutable ">=10 divergences in 30d window"
  criterion needs post-reactivation trades — 2026-07-08 live BQ shows only
  14 trades / 8 RT sells in-window (halt since 07-03), and pair_round_trips
  FIFO drops SELLs whose BUY predates the window. Sequence, don't soften.
- Path drift vs masterplan audit_basis: slot_accounting.py is at
  backend/autoresearch/ not backend/services/; learning_schema
  create_learning_log_table default dataset is "trading" (nonexistent) —
  a WIRE decision must override with pyfinagent_data.
- Error-caching trap: a failed _compute_learnings result is cached 300s
  (paper:learnings:*, api_cache.py:132) — the divergences_error fix must
  skip cache.set on error.
- ARRAY_AGG traps for the sparkline: NULL element in result array raises
  (use IGNORE NULLS); ORDER BY analysis_date DESC LIMIT 30 returns
  newest-first — reverse before the sparkline (cf. [[metric-source-paths]]
  DESC-order trap).

**Why:** step 61.4 starts after 61.3; these facts are load-bearing for its
contract and easy to lose to archive rotation.
**How to apply:** when 61.4's contract is written, verify the counts/paths
still hold (trades accrue daily post-reactivation; code may shift).
