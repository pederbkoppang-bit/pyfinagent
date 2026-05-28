# Contract — cycle 14 / phase-43.0 DoD-5 closure (freshness probe SQL fix)

**Cycle:** 14 | **Date:** 2026-05-28 | **Sub-step of:** phase-43.0 (P1, H) | **Author:** Main

---

## Research-Gate Summary

- Researcher subagent: `a6e333f3743b90f0b`
- Brief: `handoff/current/research_brief_phase_43_0_dod_5_freshness.md`
- `gate_passed: true` — 6 external sources read in full, 13 URLs, recency scan, 3-variant queries, 7 internal files inspected with file:line anchors.
- **Researcher overturned Main's premature hypothesis.** Main initially suspected the bug was `_pt_table()` resolving the historical_* tables to the wrong dataset (`financial_reports` instead of `pyfinagent_data`). Researcher verified via live `bigquery.Client.get_table()` that ALL 4 tables (`historical_prices`, `historical_fundamentals`, `historical_macro`, `signals_log`) ARE in `financial_reports`. The actual bug is the SQL pattern itself.

## Hypothesis (corrected)

**Real root cause:** `backend/services/cycle_health.py:414-451` `_bq_max_event_age()` wraps EVERY `MAX(time_col)` in `SAFE.TIMESTAMP(...)` to coerce STRING/DATE source columns to TIMESTAMP. This wrapper is REQUIRED for the 2 STRING-typed columns (`paper_trades.created_at` RFC3339-string, `paper_portfolio_snapshots.snapshot_date` YYYY-MM-DD-string) but BREAKS for already-TIMESTAMP-typed columns. Per Google BigQuery docs:
1. `TIMESTAMP()` has no `(TIMESTAMP) -> TIMESTAMP` overload — only STRING/DATE/DATETIME inputs.
2. `SAFE.` prefix is not supported with aggregate/window/UDF functions — and `MAX()` is an aggregate.

Effect: `SAFE.TIMESTAMP(MAX(ingested_at))` returns `400 BadRequest: SAFE with function timestamp is not supported` for `historical_prices`/`historical_fundamentals`/`historical_macro` (all `ingested_at` TIMESTAMP-typed) and `signals_log.recorded_at` (TIMESTAMP-typed). The broad `except Exception` at `cycle_health.py:445-452` swallows the 400 error, returns `None`, and `_band(None, ...)` returns `"unknown"`.

Applying the fix will flip 4 of 4 sources from `band: "unknown"` to a real band (green/amber/red) — closing DoD-5.

## Immutable success criteria

DoD-5 from `master_roadmap_to_production.md` §6:
> `GET /api/paper-trading/freshness` returns no `band='Unknown'` rows across all source rows.

Derived cycle-14 success criteria:
1. `curl -sf http://localhost:8000/api/paper-trading/freshness` returns 0 sources with `band == "unknown"` (case-insensitive).
2. `cycle_health.py:_bq_max_event_age()` patched with the type-aware branch per Pattern C.
3. The paper-trades freshness path (`paper_trades.created_at` STRING + `paper_portfolio_snapshots.snapshot_date` STRING) STILL works — no regression on the working path.
4. Live curl shows verbatim band per source matching researcher's predicted outcomes (historical_prices red, historical_fundamentals green, historical_macro amber, signals_log green) — OR documents a deviation with cited evidence.
5. Markdown / `python -c "import ast; ast.parse(...)"` syntax check on the edited file PASSES.

**Verification commands:**
```bash
# (a) syntax check
python3 -c "import ast; ast.parse(open('backend/services/cycle_health.py').read())" && echo OK

# (b) restart backend, wait for ready
launchctl kickstart -k gui/$(id -u)/com.pyfinagent.backend
sleep 5
curl -sf -m 10 http://localhost:8000/api/health

# (c) probe freshness and count Unknown
curl -sf http://localhost:8000/api/paper-trading/freshness | \
  python3 -c "import json,sys; d=json.load(sys.stdin); src=d['sources']; \
    unk=[k for k,v in src.items() if (v.get('band') or '').lower()=='unknown']; \
    print('total_sources:', len(src), 'unknown:', len(unk), 'unknown_keys:', unk)"

# (d) live_check artifact
test -f handoff/current/live_check_43_0_dod_5.md && grep -qE 'PASS|FAIL' handoff/current/live_check_43_0_dod_5.md
```

## Plan Steps

1. Apply Pattern C fix to `backend/services/cycle_health.py:414-451` per researcher's verbatim diff. Add `_STRING_DATE_TIMESTAMP_COLS` constant at module scope; modify `_bq_max_event_age` to branch on column type.
2. Syntax-check the edited file.
3. Restart the backend via launchctl (per `feedback_npm_install_requires_launchctl_kickstart` discipline applies to backend daemon too — pkill races the launchd watchdog).
4. Wait for `/api/health` to come back green.
5. Curl `/api/paper-trading/freshness` and verify Unknown count = 0.
6. Write `handoff/current/live_check_43_0_dod_5.md` with verbatim pre-fix vs post-fix curl output.
7. Write `handoff/current/experiment_results.md` with file diff, verification commands, predicted-vs-actual band table.
8. Spawn Q/A.
9. Append `handoff/harness_log.md` AFTER Q/A PASS, BEFORE any masterplan touch.
10. Commit + push manually (no masterplan status change — phase-43.0 STAYS pending until all 14 DoDs PASS).

## What this cycle will NOT do

- NOT fix `backend/metrics/sortino.py:108` hardcoded `pyfinagent_data.historical_macro` (researcher flagged this as a SEPARATE bug — `historical_macro` lives in `financial_reports`, sortino query is currently 404'ing). File as a follow-up phase-43.x bug.
- NOT investigate why `historical_prices` is 52 days old (real ingestion-pipeline staleness; post-fix band will land RED, surfacing the issue to the operator).
- NOT change band thresholds in `_TABLE_MAX_AGE_SEC` (existing thresholds drive the predicted post-fix bands).
- NOT touch `_pt_table()` or settings (the dataset is correct as-is — Main's premature hypothesis was wrong).
- NOT flip phase-43.0 to `status=done`.

## Stop-condition contribution

Closes DoD-5 of phase-43.0 gate. After this cycle: DoD-1 / DoD-2 / DoD-6 / DoD-7 / DoD-9 remain open (cycle 12 audit). DoD-14 closed cycle 13. Cumulative tally would flip to 11 most-generous / 7 literal of 14 PASS.

## Anti-pattern check

- `feedback_no_emojis` — no emojis in any cycle artifact.
- `feedback_contract_before_generate` — contract written BEFORE generate.
- `feedback_log_last` — harness_log append AFTER Q/A PASS.
- `feedback_qa_harness_compliance_first` — Q/A prompt opens with 5-item harness audit.
- `feedback_harness_rigor` — DoD-5 verdict was FAIL in cycle 12; closing it requires the unknown count to literally drop to 0, not hand-waving "the probe is broken but the data is fresh."
- `feedback_full_codebase_audit_before_changes` — researcher caught my premature dataset-resolution hypothesis; honored.
- `feedback_never_skip_researcher` — researcher spawned + gate passed BEFORE contract.
- `feedback_auto_commit_hook_stalls` — manual commit + push since no masterplan flip.

## References

- `handoff/current/research_brief_phase_43_0_dod_5_freshness.md` (this cycle's research gate)
- `handoff/current/production_ready_audit_2026-05-28.md` (cycle 12 audit; DoD-5 evidence: 4 of 6 bands "unknown")
- `backend/services/cycle_health.py:414-451` (the function to patch)
- `backend/db/bigquery_client.py:512` `_pt_table()` (NOT the bug after all — dataset resolves correctly to `financial_reports`)
- BigQuery TIMESTAMP functions: https://docs.cloud.google.com/bigquery/docs/reference/standard-sql/timestamp_functions
- BigQuery SAFE prefix limitation: https://docs.cloud.google.com/bigquery/docs/reference/standard-sql/conversion_functions
- Metaplane "Stay Fresh: Four Ways to Track Update Times": https://www.metaplane.dev/blog/stay-fresh-four-ways-to-track-update-times-for-bigquery-tables-and-views
