# Sprint Contract — phase-24.7 — Data Quality + BQ Freshness

**Cycle:** phase-24 cycle 7
**Date:** 2026-05-12
**Step ID:** 24.7
**Priority:** P1

## Research-gate
`gate_passed: true` (tier=moderate). 6 sources: Metaplane BQ freshness, OneUptime Dataplex + Python circuit-breakers (2026), Manik Hossain freshness 2026, craakash yfinance 2025, GCP BigQuery best-practices.

```json
{"tier":"moderate","external_sources_read_in_full":6,"snippet_only_sources":10,"urls_collected":16,"recency_scan_performed":true,"internal_files_inspected":9,"gate_passed":true}
```

## Hypothesis
BQ tables have inconsistent freshness windows. `/freshness` endpoint reports single age. yfinance fallback fires silently when BQ stale.

**Researcher verdict: CONFIRMED with major surprise:**
- `/freshness` endpoint (`cycle_health.py:214-228`) only queries `paper_trades` + `paper_portfolio_snapshots` — blind to 5 historical tables
- **Surprise**: `data_ingestion.py:34` stores historical tables in `financial_reports` dataset, NOT `pyfinagent_hdw` as CLAUDE.md implies — dataset routing bug
- `yfinance_tool.py:84-88` `get_price_history()` is completely unguarded (no try/except, no logging)
- `orchestrator.py:1141` yfinance fallback logs at INFO (default WARNING level suppresses it) — silent
- `cache.py:184-228` `preload_macro()` has idempotency guard but no max-age check — stale macro silently reused
- `bigquery_client.py:386-392` `signals_log` writes not monitored by freshness endpoint

## Success criteria (verbatim)
1. findings_md_exists
2. research_gate_envelope_present_with_gate_passed_true
3. external_sources_count_at_least_5
4. canonical_url_cited_verbatim_bigquery_client_py
5. recency_scan_2024_2026_section_present
6. at_least_three_phase_25_candidate_steps_proposed
7. each_candidate_step_has_files_list_with_absolute_paths
8. each_candidate_step_has_draft_verification_command
9. harness_log_has_phase_24_24_7_cycle_entry
10. executive_summary_section_present
11. findings_audits_bq_table_freshness_across_datasets
12. findings_audits_yfinance_fallback_pattern
13. findings_audits_signal_freshness

**Verifier:** `python3 tests/verify_phase_24_7.py`

## Plan
1. Findings doc
2. experiment_results.md
3. Q/A spawn
4. Cycle 48 log append
5. live_check_24.7.md
6. Flip 24.7 to done
