# Contract — phase-53.3 (Data-stack elevation: BQ cost/perf + freshness/lineage)

**Date:** 2026-06-10. **Tier:** complex. **Step:** phase-53.3 (P3). Measure-first
($0 dry-run); correctness-preserving; NO DROP/DELETE/schema mutation (operator-gated).

## N* delta (N* = Profit − Risk − Burn)

**Burn↓ (measured):** column-pruning the two hot `historical_fundamentals` `SELECT *`
queries cuts bytes-scanned **−41%** (655,079 → 385,021, dry-run measured) → lower BQ
cost on every preload/fallback. No P/R delta; results byte-identical.

## Research-gate summary

`researcher` ran FIRST (gate **PASSED**: 6 sources read in full, 18 URLs, recency scan,
10 internal files). Brief: `handoff/current/research_brief.md`. **HEADLINE (measure-first,
$0 dry-runs):** the 3 hot historical tables (`financial_reports.historical_{prices,
fundamentals,macro}`, us-central1) are **NOT partitioned/clustered** — proven: `preload_prices`
scans the SAME 112,351,601 bytes with vs without its `date` filter. So adding date/partition
WHERE filters CANNOT prune this cycle (cargo-cult); the 90-99% lever is partitioning =
table recreation = **OPERATOR-GATED**. The autonomously-landable, results-preserving win is
**column pruning** (Google best-practices-performance-compute: "query only the columns you
need"). The freshness pattern (`cycle_health.compute_freshness`, MAX-ts-vs-now) already
matches dbt/Monte Carlo canon.

## Immutable success criteria — VERBATIM from masterplan phase-53.3 (do NOT edit)

1. the research gate passed (BQ cost/perf + lineage sources cited) and an audit of the hot
   query paths reports per-query bytes-scanned/cost with the partition/cluster-filter gaps
   identified
2. concrete optimizations land with BEFORE/AFTER bytes-scanned (dry-run) + cost evidence,
   and a freshness/lineage check on the signal/price tables is recorded
3. the 30s fallback-query timeout rule is preserved and query RESULTS are unchanged
   (correctness-preserving optimization); NO DROP or unqualified DELETE (operator-gated)
4. live_check_53.3.md records the before/after bytes-scanned + cost delta + the
   freshness/lineage evidence

## Plan steps

1. **Opt-1** — `backend/backtest/cache.py:153` `preload_fundamentals`: replace `SELECT *`
   with the 12 CONSUMED columns (`ticker, report_date, total_revenue, net_income,
   total_debt, total_equity, total_assets, operating_cash_flow, shares_outstanding,
   sector, industry, dividends_per_share`). Drop the 4 never-`.get()`-ed columns
   (`filing_date, ingested_at, market, currency`). −41% bytes (measured).
2. **Opt-2** — `cache.py:342` `cached_fundamentals` fallback: same `SELECT *` → same 12
   columns. Keep the `WHERE ticker=@ticker AND report_date<=@cutoff ORDER BY report_date
   DESC LIMIT 5` + the 30s timeout EXACTLY.
3. **Prove DO-NO-HARM:** re-grep `historical_data.py` + `data_server.py` to confirm the 4
   dropped columns have ZERO `.get()` call sites + consumers use `.get(key, default)`;
   assert the new projection ⊇ the consumed set.
4. **Measure before/after** via $0 dry-run (`QueryJobConfig(dry_run=True, use_query_cache=
   False)` → `total_bytes_processed`): old vs new SQL for both queries; record the bytes
   delta.
5. **Freshness/lineage check (record, don't auto-fix):** `GET /api/paper-trading/freshness`
   per-source bands; DOCUMENT the lineage discrepancy (`sortino.py:108` reads
   `pyfinagent_data.historical_macro` while the writer + other readers use
   `financial_reports`) as an operator follow-up (repointing would change Sortino's MAR
   input = a result change — NOT this cycle).
6. **Operator-gated recommendations (documented, NOT landed):** partition
   `historical_{prices,fundamentals,macro}` by date + cluster by ticker (the 90-99% lever)
   via a `scripts/migrations/*.py` (re-runnable, idempotent) — needs operator approval
   (schema mutation / table recreation).
7. **Verify:** `ast.parse` cache.py; `python -m pytest -k "cache or fundamental"`; the
   before/after dry-run; write `live_check_53.3.md`. **Fresh qa → log → flip → commit.**

## Guardrails / DO-NO-HARM

- Correctness-preserving ONLY: projection change (column list), no WHERE/ORDER BY/LIMIT/
  timeout change → byte-identical RESULTS (the dropped columns are unused; consumers use
  `.get`). The 30s fallback timeout is untouched.
- NO DROP / NO unqualified DELETE / NO schema mutation / NO repartition (operator-gated).
  NO `.env`/secret edit. $0 (dry-run estimation only; no bytes billed; no LLM).
- Do NOT add date/partition WHERE filters to the non-partitioned tables (proven cargo-cult
  — no byte reduction). Do NOT repoint the sortino lineage (result change). No emoji; ASCII.

## References

`handoff/current/research_brief.md`; `backend/backtest/cache.py:153/342`;
`backend/services/cycle_health.py:426/473`; `backend/metrics/sortino.py:108`;
`backend/agents/historical_data.py` + `data_server.py` (consumed-column grep). External:
Google Cloud "Optimize query computation" / "Controlling costs", BQ INFORMATION_SCHEMA,
dbt/Monte Carlo freshness.
