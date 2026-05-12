---
step: phase-24.7
cycle: 7
cycle_date: 2026-05-12
result: PASS_PENDING_QA
verification_command: 'source .venv/bin/activate && python3 tests/verify_phase_24_7.py'
title: Data quality + BQ freshness + yfinance fallback audit (P1)
---

# Experiment Results — phase-24.7

**Action:** READ-ONLY. Findings + brief + contract. No code changes.

## Verbatim verifier output

```
=== phase-24.7 (data-quality) verifier ===
  [PASS] findings_md_exists_at_docs_audits_phase_24_2026_05_12_24_7_data_quality_findings_md
  [PASS] research_gate_envelope_present_with_gate_passed_true
  [PASS] external_sources_count_at_least_5
  [PASS] canonical_url_cited_verbatim_bigquery_client_py
  [PASS] recency_scan_2024_2026_section_present
  [PASS] at_least_three_phase_25_candidate_steps_proposed
  [PASS] each_candidate_step_has_files_list_with_absolute_paths
  [PASS] each_candidate_step_has_draft_verification_command
  [FAIL] harness_log_has_phase_24_24_7_cycle_entry
         -> harness_log.md must contain `## Cycle N -- ... phase=24.7 result=...` header
  [PASS] executive_summary_section_present
  [PASS] findings_audits_bq_table_freshness_across_datasets
  [PASS] findings_audits_yfinance_fallback_pattern
  [PASS] findings_audits_signal_freshness
FAIL (12/13) EXIT=1
```

12/13 PASS. Log-last only FAIL.

## Hypothesis verdict
CONFIRMED with major surprise: historical tables routed to `financial_reports` dataset (not `pyfinagent_hdw` per CLAUDE.md). `/freshness` blind to 5 tables. yfinance fallback silent (INFO log, no counter, no Slack). `preload_macro()` no max-age guard. `yfinance_tool.get_price_history()` completely unguarded.

## Phase-25 candidates (6)
- 25.A7 (P1) — Per-table freshness endpoint covering all 5 tables
- 25.B7 (P1) — yfinance-fallback counter + WARNING promotion
- 25.C7 (P1) — Unified `/api/observability/data-freshness`
- 25.D7 (P1) — `preload_macro()` max-age guard
- 25.E7 (P1) — `yfinance_tool.get_price_history()` try/except + counter
- 25.F7 (P2) — Document or migrate `financial_reports` → `pyfinagent_hdw`

## Next: Q/A
