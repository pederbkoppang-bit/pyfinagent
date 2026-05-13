---
step: 25.B7
slug: yfinance-fallback-counter-bq-warning
status: in_progress
cycle_date: 2026-05-13
parent_research_brief: handoff/current/research_brief.md
---

# Contract -- phase-25.B7

## Step ID + masterplan reference

`25.B7` -- "yfinance fallback counter persisted to BQ + WARNING log promotion"
(P2, harness_required, no dep).

## Research-gate summary

Tier=simple. Main authored from internal inspection. JSON envelope
shows `gate_passed=true`. Migration template copied from 25.Q (cycle 84).

## Hypothesis

`backend/agents/orchestrator.py:1162` currently logs the AV->yfinance
fallback at INFO level which is suppressed in default WARNING log
views, and the count is not persisted -- making it impossible to
detect "AV is broken" until it manifests as Sharpe degradation. By
promoting the log to WARNING and writing each event to a new
`data_source_events` BQ table partitioned by date and clustered by
source, an operator can compute
`pct_yfinance_fallback_dominance = COUNTIF(source='yfinance_fallback')/COUNT(*)`
trivially over any window.

## Success criteria (verbatim from masterplan.json)

> `orchestrator_yfinance_fallback_logs_at_warning_level`
>
> `new_bigquery_table_data_source_events_populated`
>
> `counter_aggregable_for_pct_yfinance_fallback_dominance`

## Plan steps

1. **Migration script** -- `scripts/migrations/create_data_source_events_table.py`
   modeled on 25.Q (idempotent CREATE TABLE IF NOT EXISTS, --apply flag).
2. **`bigquery_client.save_data_source_event(...)` method** -- best-effort
   single-row insert with try/except (fail-open per existing convention).
3. **`orchestrator.py:1161-1162`** -- promote the INFO log to WARNING and
   call `self.bq.save_data_source_event(...)` with `source="yfinance_fallback"`,
   `kind="fallback"`, ticker, article_count.
4. **Verifier** -- `tests/verify_phase_25_B7.py` with 4 claims:
   - Claim 1: orchestrator.py contains `logger.warning(` for the fallback (not `.info`).
   - Claim 2: `save_data_source_event` method exists with the required signature.
   - Claim 3: migration script exists with `CREATE TABLE IF NOT EXISTS` + `data_source_events`.
   - Claim 4: behavioral round-trip -- mock `bigquery_client.save_data_source_event`
     and import a stub orchestrator path, confirm the call shape matches what an
     aggregator would consume.

## Files

| File | Action |
|------|--------|
| `backend/agents/orchestrator.py` | Promote INFO->WARNING; invoke save_data_source_event |
| `backend/db/bigquery_client.py` | Add `save_data_source_event` method |
| `scripts/migrations/create_data_source_events_table.py` | NEW migration script |
| `tests/verify_phase_25_B7.py` | NEW verifier |

## Verification command (immutable)

```
source .venv/bin/activate && python3 tests/verify_phase_25_B7.py
```

## Live-check

`BQ data_source_events table grows per cycle with yfinance_fallback rows`.
Will write `handoff/current/live_check_25.B7.md`.

## Risks + mitigations

- **Risk**: WARNING flood when AV is rate-limited for the whole session.
  **Mitigation**: WARNING is the correct level for a degraded-mode signal;
  the log line includes ticker + count so it's not noise.
- **Risk**: `save_data_source_event` blocking the analysis pipeline.
  **Mitigation**: Single-row BQ insert is fast; fail-open via try/except
  matches existing pattern.

## References

- `handoff/current/research_brief.md` (this cycle)
- `handoff/archive/phase-25.Q/contract.md` (migration template parent)
- `backend/agents/orchestrator.py:1141-1162`
- `.claude/masterplan.json::25.B7`
