---
step: phase-25.A3
topic: Write promoted strategies to pyfinagent_data.promoted_strategies BQ table
tier: moderate
date: 2026-05-12
---

## Research: phase-25.A3 -- BQ promoted_strategies table

### Read in full (>=5 required; counts toward the gate)

| URL | Accessed | Kind | Fetched how | Key quote or finding |
|-----|----------|------|-------------|----------------------|
| https://docs.cloud.google.com/bigquery/docs/creating-clustered-tables | 2026-05-12 | Official doc | WebFetch | "You can specify up to four clustering columns." PARTITION BY + CLUSTER BY can combine; STRING is a supported cluster column type. |
| https://oneuptime.com/blog/post/2026-02-17-how-to-implement-idempotent-data-pipelines-in-gcp-to-handle-retry-safe-processing/view | 2026-05-12 | Engineering blog (2026) | WebFetch | "MERGE is inherently idempotent: running it with the same source data twice results in the same target state." Recommended for low-volume periodic upserts over plain INSERT. |
| https://medium.com/@aroraashish1909/json-datatype-vs-string-datatype-for-storing-json-objects-bigquery-2719616c6da4 | 2026-05-12 | Engineering blog | WebFetch | "97% reduction in bytes processed" with JSON vs STRING; 54% logical storage reduction. JSON datatype is strictly superior for any JSON payload column. |
| https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf | 2026-05-12 | Peer-reviewed paper (Bailey & Lopez de Prado 2014) | WebFetch (full PDF) | Minimal registry record for DSR requires: strategy_id, observed SR, number of trials N, sample length T, skewness, kurtosis. DSR is derived from these five inputs. |
| https://medium.com/@riyatripathi.me2011/handling-duplicates-in-bigquery-merge-vs-deduplication-insert-00f3c5f9e95b | 2026-05-12 | Engineering blog | WebFetch | MERGE WHEN NOT MATCHED THEN INSERT is correct "insert-and-skip-duplicate" pattern for BQ; MERGE non-determinism arises only when multiple source rows match one target row -- avoidable by calling save_promoted_strategy one row at a time. |

### Identified but snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|-----|------|------------------------|
| https://docs.cloud.google.com/bigquery/docs/json-data | Official doc | WebFetch returned partial content; key facts subsumed by Arora blog |
| https://datawise.dev/json-datatype-vs-json-like-string-in-bigquery | Blog | Read; primarily covers validation semantics, not performance |
| https://mlflow.org/docs/latest/model-registry/ | Official doc | Fetch returned empty; snippet-only (model lifecycle states: None, Staging, Production, Archived) |
| https://oneuptime.com/blog/post/2026-02-17-how-to-query-json-data-in-bigquery-using-json-functions/view | Blog (2026) | 2026 recency scan hit; content subsumed by Arora source above |
| https://docs.cloud.google.com/bigquery/docs/partitioned-tables | Official doc | Snippet-only; partition semantics covered by clustering doc |
| https://chartdb.io/blog/design-effective-schemas-for-google-bigquery | Blog | Snippet-only; general guidance, no step-specific content |
| https://medium.com/balaena-quant-insights/deflated-sharpe-ratio-dsr-33412c7dd464 | Blog | Snippet-only; subsumed by Bailey & Lopez de Prado primary paper |
| https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551 | Peer-reviewed (SSRN) | Snippet-only; full PDF fetched from davidhbailey.com instead |

### Recency scan (2024-2026)

Searched for 2026 and 2025 literature on: BQ JSON column, idempotent BQ pipelines, quant strategy registry. Result: Two blog posts from February 2026 (OneUptime/GCP) addressed idempotent BQ pipeline patterns and JSON query functions; both were evaluated and one was fetched in full. No 2025/2026 changes to BQ MERGE semantics or JSON-column GA status were found that would contradict canonical guidance. The Bailey & Lopez de Prado DSR paper (2014) remains the canonical reference for DSR field requirements; no 2024-2026 literature supersedes its field definitions.

Queries run:
1. Year-less canonical: "BigQuery schema design strategy experiment registry table partition clustering"
2. 2026 frontier: "BigQuery JSON column type vs STRING encoded JSON hyperparameters 2026"
3. 2025 window: "Bailey Lopez de Prado DSR deflated Sharpe ratio experiment tracking fields 2025"

---

### Key findings

1. **MERGE is the correct idempotency pattern for low-volume weekly upserts.** Natural key is `(week_iso, strategy_id)`. MERGE WHEN MATCHED THEN UPDATE / WHEN NOT MATCHED THEN INSERT is retry-safe and follows existing `save_paper_snapshot` and `save_paper_position` patterns in `backend/db/bigquery_client.py`. (Source: OneUptime 2026 blog; bigquery_client.py:563-602, 700-740)

2. **JSON column for `params` is preferred over STRING.** BQ JSON datatype delivers 97% query-scan reduction and 54% logical storage reduction vs STRING-encoded JSON. Use `params JSON` not `params STRING`. Equality operators are not defined on JSON columns -- do not JOIN or ORDER BY the `params` column; extract keys with `JSON_VALUE`. (Source: Arora Medium)

3. **CLUSTER BY `(strategy_id, week_iso)`** is correct: queries filter by strategy_id, STRING is a supported cluster column type. Partition by `DATE(promoted_at)` provides cost-containment on date-ranged scans. (Source: BQ official docs)

4. **DSR-required fields for a registry row:** `dsr` (float), `pbo` (float), `n_trials` (int), `sr_observed` (float), `skewness` (float), `kurtosis` (float), `sample_len_t` (int). For this step only `dsr` and `pbo` are already on every candidate; the remaining fields should be nullable and populated by downstream steps 25.B3/C3. (Source: Bailey & Lopez de Prado 2014)

5. **Candidate dict shape confirmed.** Candidates entering `run_friday_promotion` carry: `trial_id` (str), `dsr` (float), `pbo` (float). A `params` sub-dict is present on proposer stub diffs (`proposer.py:43-48`) and on candidates from `thursday_batch._sample_candidates` which reads `spec["params"]` keys from candidate_space.yaml (`thursday_batch.py:101-105`). Use `.get("params") or {}` with a fallback.

6. **Dataset routing:** `pyfinagent_data` is correct. It is the value of `settings.bq_dataset_observability` (`config/settings.py:81`) and hosts `harness_learning_log`, `historical_macro`, and observability tables. `pyfinagent_pms` is for portfolio tables. The migration script and BQ method must hardcode `pyfinagent_data` as the dataset (precedent: `slot_accounting.py:26`).

7. **BQ client method pattern:** Mirror `save_paper_snapshot` (bigquery_client.py:700-740): MERGE on natural key, parameterized SQL, drop None values, call `self.client.query(query, job_config=...).result()`. New method: `BigQueryClient.save_promoted_strategy(row: dict)` with MERGE on `(week_iso, strategy_id)`. The `params` column requires special handling: pass as STRING, wrap in `PARSE_JSON(...)` in the SQL (see implementation note below).

8. **Use DML not insert_rows_json.** Existing paper-trading tables use DML INSERT / MERGE to avoid streaming buffer conflicts on SELECT queries. Consistent pattern: `save_paper_trade` at bigquery_client.py:630 uses DML INSERT INTO.

---

### Internal code inventory

| File | Lines | Role | Status |
|------|-------|------|--------|
| `backend/autoresearch/friday_promotion.py` | 170 | Entry point; wire BQ write after ledger append | Active; ledger write block at lines 121-131 |
| `backend/autoresearch/thursday_batch.py` | ~160 | Candidate space sampling; `params` dict structure | Active; `spec["params"]` keys at lines 101-105 |
| `backend/autoresearch/proposer.py` | 70 | Proposer Diff TypedDict; `trial_id` + `params` shape | Active; candidate shape at lines 43-48 |
| `backend/autoresearch/gate.py` | ~40 | PromotionGate.evaluate -- returns `promoted`, `trial_id` | Active; candidate keys `dsr`, `pbo`, `trial_id` at lines 26-39 |
| `backend/autoresearch/weekly_ledger.py` | ~100 | TSV ledger; append_row signature | Active; columns at lines 24-28 |
| `backend/db/bigquery_client.py` | ~800 | BigQueryClient; MERGE patterns | Active; save_paper_position (MERGE) at line 563; save_paper_snapshot (MERGE) at line 700 |
| `backend/config/settings.py` | ~150 | Settings; `bq_dataset_observability = "pyfinagent_data"` | Active; line 81 |
| `scripts/migrations/create_options_snapshots_table.py` | 114 | Canonical migration skeleton | Active; mirror argparse + CREATE_SQL + main() |

---

### Consensus vs debate (external)

All sources agree: MERGE is the correct idempotent pattern for low-volume tables with a natural key. JSON column vs STRING has a clear winner (JSON is faster and cheaper) with one constraint: no equality comparisons on JSON columns -- not needed here since `params` is write-once and read-by-key.

### Pitfalls (from literature)

- **MERGE non-deterministic match:** if the same `(week_iso, strategy_id)` pair appears twice in a single MERGE source, BQ errors. `save_promoted_strategy` receives one row at a time (loop over `top`), so this is not a risk. (Source: BigQuery deduplication docs)
- **JSON column + parameterized DML:** BQ parameterized queries have no native JSON type parameter. Pass `params` as `json.dumps(dict)` STRING, and cast in the SQL with `PARSE_JSON(@v_params)`. This is the correct insertion path.
- **Streaming vs DML:** use DML to avoid streaming buffer conflicts with SELECT queries. Consistent with all existing paper-trading write methods.
- **already_fired guard:** `run_friday_promotion` returns early on `already_fired=True` (lines 78-91), so BQ write is only reached on first-fire. Wrap BQ write in try/except; log warning on failure but do NOT abort the return dict -- ledger is the authoritative source for `already_fired`.

---

### Application to pyfinagent (external findings mapped to file:line)

| Recommendation | Apply where | Anchor |
|----------------|-------------|--------|
| MERGE on (week_iso, strategy_id) | save_promoted_strategy | Mirror bigquery_client.py:700-740 |
| PARTITION BY DATE(promoted_at), CLUSTER BY (strategy_id, week_iso) | CREATE TABLE SQL | Mirror create_options_snapshots_table.py:63-66 |
| JSON column for params, PARSE_JSON(@v_params) in MERGE SQL | params column | New pattern; no existing JSON column in codebase |
| Wire BQ write after ledger append, in try/except | friday_promotion.py after line 131 | friday_promotion.py:131-140 |
| Hardcode "pyfinagent_data" as dataset | Migration + BQ method | slot_accounting.py:26 for precedent |
| DML not insert_rows_json | save_promoted_strategy | bigquery_client.py:630 |

---

## Files to modify

| File | Action | What |
|------|--------|------|
| `scripts/migrations/create_promoted_strategies_table.py` | CREATE NEW | Idempotent migration, mirror create_options_snapshots_table.py skeleton exactly |
| `backend/db/bigquery_client.py` | ADD METHOD | `save_promoted_strategy(row: dict)` -- MERGE on `(week_iso, strategy_id)` |
| `backend/autoresearch/friday_promotion.py` | EDIT | Wire `bq_client.save_promoted_strategy(row)` after ledger append at line 131; accept `bq_client` kwarg (optional, None by default for backward compat) |
| `tests/verify_phase_25_A3.py` | CREATE NEW | Verifier: checks table exists in BQ, schema has required columns, writes and reads back a test row |

---

## Verbatim CREATE TABLE SQL

```sql
CREATE TABLE IF NOT EXISTS `sunny-might-477607-p8.pyfinagent_data.promoted_strategies` (
    strategy_id       STRING    NOT NULL OPTIONS(description="trial_id from the promotion gate"),
    week_iso          STRING    NOT NULL OPTIONS(description="ISO week string, e.g. 2026-W20"),
    params            JSON               OPTIONS(description="Strategy hyperparameters dict (learning_rate, max_depth, etc.)"),
    dsr               FLOAT64   NOT NULL OPTIONS(description="Deflated Sharpe Ratio at promotion time"),
    pbo               FLOAT64   NOT NULL OPTIONS(description="Probability of Backtest Overfitting at promotion time"),
    status            STRING    NOT NULL OPTIONS(description="pending | active | paused | superseded | rolled_back"),
    allocation_pct    FLOAT64            OPTIONS(description="Starting allocation fraction, e.g. 0.05"),
    promoted_at       TIMESTAMP NOT NULL OPTIONS(description="UTC timestamp of promotion fire"),
    sortino_monthly   FLOAT64            OPTIONS(description="Monthly Sortino ratio at promotion time, from ledger row"),
    rejection_reason  STRING             OPTIONS(description="Non-null only for rejected candidates -- reserved for future use")
)
PARTITION BY DATE(promoted_at)
CLUSTER BY strategy_id, week_iso
OPTIONS(description="phase-25.A3 promoted strategy registry; one row per (week_iso, strategy_id) promotion fire");
```

---

## Verbatim BQ row dict (emitted per promoted candidate)

Called once per item in `top` after the ledger append succeeds (inside `run_friday_promotion`):

```python
import json
from datetime import datetime, timezone

for c in top:
    bq_row = {
        "strategy_id":    str(c.get("trial_id", "")),
        "week_iso":        week_iso,
        "params":          json.dumps(c.get("params") or {}),   # serialized STRING; PARSE_JSON in MERGE SQL
        "dsr":             float(c.get("dsr", 0.0)),
        "pbo":             float(c.get("pbo", 1.0)),
        "status":          "pending",
        "allocation_pct":  float(starting_allocation_pct),
        "promoted_at":     datetime.now(timezone.utc).isoformat(),
        "sortino_monthly": float(row.get("sortino_monthly") or 0.0),
    }
    try:
        bq_client.save_promoted_strategy(bq_row)
    except Exception as exc:
        logger.warning(
            "friday_promotion: BQ write failed for strategy_id=%s: %r",
            bq_row["strategy_id"], exc,
        )
```

The `bq_client` parameter should be added to `run_friday_promotion`'s signature as `bq_client: Any | None = None`; when None, skip the BQ write silently (preserves backward compat with existing tests that don't inject a client).

---

## save_promoted_strategy implementation note

Mirror `save_paper_snapshot` (bigquery_client.py:700-740) with these differences:

1. **MERGE key:** `ON T.week_iso = S.week_iso AND T.strategy_id = S.strategy_id`
2. **params column:** pass as STRING parameter (`@v_params`); in the SQL use `PARSE_JSON(@v_params)` in both INSERT and UPDATE SET. Add a special-case branch in the param-type loop: `if k == "params": use STRING type and wrap in PARSE_JSON in the SQL string`.
3. **Target table FQN:** `f"{self.settings.gcp_project_id}.pyfinagent_data.promoted_strategies"` -- do not use any settings attribute for the dataset name; hardcode `pyfinagent_data` (precedent: slot_accounting.py:26).

---

## Cost note

Max 3 rows/week, 52 weeks/year = 156 rows/year. Storage cost is negligible (sub-cent per year). MERGE DML queries on a sub-1000-row table will scan the full table partition but at less than 1 KB; BQ free tier covers this volume many times over. No cost concern for this table.

---

### Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (5 fetched in full)
- [x] 10+ unique URLs total (incl. snippet-only) -- 13 unique URLs collected
- [x] Recency scan (last 2 years) performed + reported
- [x] Full papers/pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every relevant module (friday_promotion, thursday_batch, proposer, gate, bigquery_client, settings, slot_accounting)
- [x] Contradictions/consensus noted (all sources aligned on MERGE; JSON vs STRING unambiguous)
- [x] All claims cited per-claim (not just listed in a footer)

---

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 5,
  "snippet_only_sources": 8,
  "urls_collected": 13,
  "recency_scan_performed": true,
  "internal_files_inspected": 8,
  "report_md": "handoff/current/research_brief.md",
  "gate_passed": true
}
```
