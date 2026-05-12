# Research Brief: phase-25.C3 -- Strategy registry with status field; flip actual_replacement

Tier: **moderate** (stated by caller).

---

## Read in full (>=5 required; counts toward the gate)

| URL | Accessed | Kind | Fetched how | Key quote or finding |
|-----|----------|------|-------------|----------------------|
| https://mlflow.org/docs/2.19.0/model-registry.html | 2026-05-12 | official docs | WebFetch full | "Model Stages are deprecated as of MLflow 2.9.0 ... recommend model aliases as the modern replacement"; valid stages: None, Staging, Production, Archived |
| https://docs.cloud.google.com/bigquery/docs/transactions | 2026-05-12 | official docs | WebFetch full | "A multi-statement transaction lets you perform mutating operations ... and either commit or roll back the changes atomically"; BEGIN TRANSACTION / COMMIT syntax confirmed |
| https://docs.cloud.google.com/bigquery/docs/samples/bigquery-update-with-dml | 2026-05-12 | official docs | WebFetch full | Python DML UPDATE pattern: `client.query(query_text)` + `.result()`; `QueryJobConfig` with `query_parameters=[ScalarQueryParameter(...)]` is the parameterization mechanism |
| https://docs.cloud.google.com/bigquery/docs/parameterized-queries | 2026-05-12 | official docs | WebFetch full | Named parameter syntax `@param_name` supported in WHERE clauses; `ScalarQueryParameter` and `ArrayQueryParameter` are the binding types; same pattern applies to UPDATE statements |
| https://docs.aws.amazon.com/sagemaker/latest/dg/model-registry.html | 2026-05-12 | official docs | WebFetch full | SageMaker uses `PendingManualApproval`, `Approved`, `Rejected` as model version approval states; approval is a HITL gate before deployment; "manage the approval status of a model" is a first-class registry concern |
| https://medium.com/@fraidoonomarzai99/deployment-evaluation-strategies-in-mlops-c208585aa3bd | 2026-05-12 | blog | WebFetch full | Canonical deployment lifecycle: pre-production -> shadow -> canary -> active/production -> superseded; "versioned, reproducible, logged, auditable" for registry events |
| https://academy.pega.com/topic/promoting-shadow-models-active-status/v1 | 2026-05-12 | official docs | WebFetch full | Pega: shadow mode runs model on production data without impacting business outcome; formal promotion via HITL approval workflow |

### Identified but snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|-----|------|------------------------|
| https://www.snowflake.com/en/developers/guides/ml-champion-challenger-model-deployment/ | guide | Fetched; Snowflake uses aliases/tags not status fields -- confirms alias pattern is alternative to status enum |
| https://www.datarobot.com/blog/introducing-mlops-champion-challenger-models/ | blog | Fetched; conceptual only, no schema details |
| https://docs.cloud.google.com/vertex-ai/docs/model-registry/introduction | official docs | Fetched; Vertex AI uses labels/aliases rather than explicit status enum |
| https://validmind.com/blog/sr-11-7-model-risk-management-compliance/ | blog | Fetched; SR 11-7 requires audit trail but does not prescribe specific field schema |
| https://www.modelop.com/ai-governance/ai-regulations-standards/sr-11-7 | industry | Fetched; governance pillars confirmed: development/validation/governance; no field-level spec |
| https://mlflow.org/docs/latest/model-registry/ | official docs | Search snippet; superseded by 2.19.0 full fetch |
| https://github.com/googleapis/python-bigquery/issues/1663 | community | Snippet; confirms multi-statement transaction Python pattern |
| https://hevodata.com/learn/bigquery-parameterized-queries/ | blog | Snippet; confirms @named_param syntax |
| https://www.federalreserve.gov/supervisionreg/srletters/SR2602.pdf | regulatory | Snippet; 2026 SR update referenced but PDF not fetched |
| https://docs.datarobot.com/en/docs/mlops/monitor/challengers.html | official docs | Snippet; champion-challenger challenger tab pattern confirmed |

---

### Recency scan (2024-2026)

Searched: "model registry shadow active superseded MLOps 2026", "BigQuery BEGIN TRANSACTION UPDATE SET status parameterized Python 2025", "feature flag real vs paper trading actual_replacement 2025 2026".

Results: No new findings in 2025-2026 that supersede the canonical sources above. The Snowflake champion-challenger guide (2026) confirms aliases-over-status-fields as the modern cloud-native pattern, but this project already uses an explicit `status` STRING column (committed in 25.A3) so the alias pattern is not applicable. MLflow 2.9+ deprecation of stage transitions confirms that a custom enum-based status field (as used in 25.A3) is the current best practice for projects not using a managed registry. BigQuery multi-statement transaction support (GA, 2024) is confirmed available for the dual-UPDATE pattern in scope.

---

## Key findings

1. **25.A3 schema status values are a strict superset of the masterplan criterion.** The 25.A3 schema uses `status STRING` with documented values `pending|active|paused|superseded|rolled_back`. The masterplan criterion 1 references `active_shadow_retired`. Mapping: "shadow" maps to `pending` (strategy is evaluated but not yet promoted); "active" maps to `active`; "retired" maps to `superseded` or `rolled_back`. No schema change is needed -- the existing enum is the superset. (Source: `scripts/migrations/create_promoted_strategies_table.py` -- not directly read but referenced by verify_phase_25_A3.py line 38; BigQuery client pattern at `backend/db/bigquery_client.py` line 720-760)

2. **Hardcoded `actual_replacement=False` appears in exactly two places.** (a) `backend/autoresearch/monthly_champion_challenger.py:75` -- in the `result` dict of `run_monthly_sortino_gate`. (b) Line 263 -- inside `_emit_deployment_log_row` in the `notes` string. No other files contain this pattern. (Source: grep result above)

3. **`monthly_champion_challenger.py` has no Settings import.** The module imports only: `backend.autoresearch.gate.PromotionGate`, `backend.metrics.sortino.sortino`, stdlib. Adding `actual_replacement` derivation requires either injecting a flag via parameter or importing `get_settings()`. The injected-flag approach (new kwarg `real_capital_enabled: bool = False` with default False) is safer -- it keeps the module pure-library and testable without Settings side effects. (Source: `backend/autoresearch/monthly_champion_challenger.py` lines 21-33)

4. **`record_approval` already has a `bq_fn` injectable.** The function at lines 201-239 accepts `bq_fn: Callable[[dict[str, Any]], None] | None` for audit emission. Adding a second injectable `status_update_fn: Callable[[str, str], None] | None = None` for the BQ status flip follows the same fail-open pattern already established. (Source: `backend/autoresearch/monthly_champion_challenger.py` line 207)

5. **BQ parameterized UPDATE is well-supported.** Google's official samples confirm: `client.query(query, job_config=QueryJobConfig(query_parameters=[ScalarQueryParameter("status_val", "STRING", new_status), ScalarQueryParameter("strategy_id_val", "STRING", strategy_id)]))` with `.result(timeout=30)`. Named parameters in WHERE clauses are safe against injection. (Source: Google BigQuery DML samples doc, parameterized queries doc)

6. **Dual-flip atomicity: sequential UPDATEs sufficient.** The `promoted_strategies` table is append-only-insert with MERGE idempotency guard (25.A3). For 25.C3 scope (flip new row to `active`, optionally flip prior `active` to `superseded`), two sequential parameterized UPDATEs within a single BEGIN/COMMIT transaction are atomic. However: per the scoping analysis in finding 8 below, the supersession of the prior active row is OUT OF SCOPE for 25.C3 -- only the new-row flip is required. A single UPDATE suffices. (Source: BigQuery transactions doc; BQ DML samples)

7. **SR 11-7 audit fields for a production-approval event.** SR 11-7 does not prescribe specific database fields, but the literature consensus (MLflow, SageMaker, Pega) converges on: `strategy_id`, `approved_by` (or automated HITL), `approved_at` timestamp, `prior_status`, `new_status`, `month_key`, `sortino_delta`, `dd_ratio`, `pbo`. The existing `_emit_deployment_log_row` helper captures most of these; the `notes` field currently hard-codes `actual_replacement=False` but should derive it. (Source: SageMaker model registry doc; SR 11-7 literature; `backend/autoresearch/monthly_champion_challenger.py` lines 250-265)

8. **Supersession of prior-active row is OUT OF SCOPE for 25.C3.** Success criterion 3 is: "monthly_approval_flips_status_from_shadow_to_active". The criterion says nothing about flipping the prior champion to `superseded`. Including it would require a `get_latest_promoted_strategy(status_filter=["active"])` call in `record_approval`, which adds complexity and a BQ read in the approval hot-path. Recommend deferring prior-row supersession to a follow-on step. The existing `get_latest_promoted_strategy` in bigquery_client.py (lines 720-760) already defaults to `["pending","active"]` which means the reader will naturally return the new active row after the flip.

---

## Internal code inventory

| File | Lines | Role | Status |
|------|-------|------|--------|
| `backend/autoresearch/monthly_champion_challenger.py` | 349 | HITL gate + approval logic | Active; two hardcoded `actual_replacement=False` sites at lines 75 and 263 |
| `backend/db/bigquery_client.py` | ~830 | BQ read/write client | Active; `save_promoted_strategy` at line 659, `get_latest_promoted_strategy` at line 720; needs new `update_promoted_strategy_status` helper |
| `backend/config/settings.py` | 211 | Pydantic settings | Active; has `paper_trading_enabled: bool` at line 141; no `real_capital_enabled` flag exists -- must be added |
| `tests/verify_phase_25_B3.py` | 341 | 25.B3 verifier template | Active; shows the MagicMock + `BigQueryClient.__new__` injection pattern for testing BQ helpers without GCP |
| `tests/verify_phase_25_A3.py` | ~160 | 25.A3 verifier template | Active; shows regex-based claim checking pattern |
| `scripts/migrations/create_promoted_strategies_table.py` | ~? | BQ schema migration | Active; defines `status STRING` column (confirmed by verify_phase_25_A3.py line 38 grep) |

---

## Consensus vs debate (external)

**Consensus:** Status-field enums (`pending/active/superseded`) are the standard approach for ML model lifecycle tracking in registry systems that don't use managed cloud registries (SageMaker uses `PendingManualApproval/Approved/Rejected`; MLflow deprecated stage transitions in favor of aliases post-2.9; Snowflake uses tags/aliases). The 25.A3 schema choice of a `status STRING` column with documented enum values is consistent with SageMaker's explicit approval-status field.

**Debate:** Whether `actual_replacement` should be a derived flag vs a persisted column. Industry consensus (feature flags literature) favors env-var / settings-file flags for paper-vs-live switching, NOT a column in the strategy row. The `notes` field in the deployment log is the right place to record the derived value for audit purposes.

**No debate:** The BQ UPDATE pattern with parameterized WHERE clause is well-established and injection-safe.

---

## Pitfalls (from literature)

1. **BQ DML UPDATE is not instant.** BQ DML runs as a query job; always call `.result(timeout=30)` per CLAUDE.md. A fire-and-forget UPDATE (no `.result()`) is a silent failure risk.
2. **Multi-statement transaction pitfall in Python BQ client.** The GitHub issue confirms that `BEGIN TRANSACTION` inside a single `client.query()` call works but requires the entire multi-statement block as one string -- you cannot split across two `client.query()` calls without using session IDs. Since 25.C3 only needs a single UPDATE (not two), this is not a risk here.
3. **`actual_replacement` must NEVER be set True from this module.** The docstring at line 15 states this as a hard invariant. The correct implementation replaces the hardcoded `False` with `settings.real_capital_enabled` (defaulting to `False` via env var), but the feature flag itself must default to `False` at the Settings level. Changing the default to `True` would constitute a scope breach.
4. **Status enum drift.** The masterplan criterion uses the phrase "shadow_to_active" but the 25.A3 schema uses `pending` not `shadow`. The verifier must test for the transition `pending -> active`, not `shadow -> active`. The brief section below documents this mapping explicitly.

---

## Application to pyfinagent (mapping external findings to file:line anchors)

### Status name reconciliation

The masterplan criterion "active_shadow_retired" uses informal names. The 25.A3 schema uses the superset enum. Mapping:

| Criterion name | 25.A3 schema value | Notes |
|---|---|---|
| shadow | `pending` | Row written by `save_promoted_strategy` at `friday_promotion.py:162` with `status="pending"` |
| active | `active` | Flip target; written by new `update_promoted_strategy_status` |
| retired | `superseded` or `rolled_back` | Out of scope for 25.C3; deferred |

No schema change needed. The existing enum is a superset.

### Files to modify

| File | Change | Purpose |
|------|--------|---------|
| `backend/db/bigquery_client.py` | Add `update_promoted_strategy_status(strategy_id, new_status, week_iso=None)` after line 760 | Parameterized UPDATE on `promoted_strategies` |
| `backend/autoresearch/monthly_champion_challenger.py` | (1) Add `real_capital_enabled: bool = False` kwarg to `run_monthly_sortino_gate` (2) Replace `"actual_replacement": False` at line 75 with derived value (3) Replace `actual_replacement=False` literal at line 263 with f-string using the derived value (4) Add `status_update_fn: Callable[[str, str], None] | None = None` kwarg to `record_approval` and call it on approval | Remove hardcoding; add BQ status flip hook |
| `backend/config/settings.py` | Add `real_capital_enabled: bool = Field(False, ...)` | Feature flag; default False; SR 11-7 paper-only invariant |
| `tests/verify_phase_25_C3.py` | Create new verifier | Tests 3 success criteria |

### Verbatim Python signature: `update_promoted_strategy_status`

```python
def update_promoted_strategy_status(
    self,
    strategy_id: str,
    new_status: str,
    *,
    week_iso: str | None = None,
) -> None:
    """phase-25.C3: flip the status of a promoted_strategies row.

    If week_iso is provided, the WHERE clause is tightened to
    (strategy_id, week_iso) -- useful when promoting the exact
    row written by the Friday gate for the current week.
    If week_iso is None, updates ALL rows with the given strategy_id
    to new_status (use with caution; intended for testing).
    30s BQ timeout per CLAUDE.md.
    """
    table = f"{self.settings.gcp_project_id}.pyfinagent_data.promoted_strategies"
    if week_iso is not None:
        query = f"""
            UPDATE `{table}`
            SET status = @new_status
            WHERE strategy_id = @strategy_id
              AND week_iso = @week_iso
        """
        params = [
            bigquery.ScalarQueryParameter("new_status", "STRING", new_status),
            bigquery.ScalarQueryParameter("strategy_id", "STRING", strategy_id),
            bigquery.ScalarQueryParameter("week_iso", "STRING", week_iso),
        ]
    else:
        query = f"""
            UPDATE `{table}`
            SET status = @new_status
            WHERE strategy_id = @strategy_id
        """
        params = [
            bigquery.ScalarQueryParameter("new_status", "STRING", new_status),
            bigquery.ScalarQueryParameter("strategy_id", "STRING", strategy_id),
        ]
    job_config = bigquery.QueryJobConfig(query_parameters=params)
    self.client.query(query, job_config=job_config).result(timeout=30)
```

### Verbatim edits to `run_monthly_sortino_gate`

Add new kwarg (after `now: datetime | None = None`):

```python
    real_capital_enabled: bool = False,
```

Replace line 75 (`"actual_replacement": False,`) with:

```python
        "actual_replacement": real_capital_enabled,
```

### Verbatim edits to `record_approval`

Add new kwarg to signature (after `bq_fn: Callable[[dict[str, Any]], None] | None = None`):

```python
    status_update_fn: Callable[[str, str], None] | None = None,
```

In the body of `record_approval`, after the `_emit_deployment_log_row(row, bq_fn, now_dt)` call at line 238, add:

```python
    if status == "approved" and status_update_fn is not None:
        challenger_id = row.get("challenger_id", "")
        week_iso = row.get("week_iso")  # may be None if not stored in state
        try:
            status_update_fn(challenger_id, "active")
        except Exception as exc:
            logger.warning("record_approval: status_update_fn fail-open: %r", exc)
```

And in the expired branch at line 232 (before `return row`), add the same fail-open block for consistency (status="expired", so `status_update_fn` is NOT called -- the status stays pending/expired in BQ, which is correct).

### Verbatim edit to `_emit_deployment_log_row` (line 263)

Replace:

```python
            f"actual_replacement=False"
```

With:

```python
            f"actual_replacement={row.get('actual_replacement', False)}"
```

This requires that `actual_replacement` is stored into the state row before `_emit_deployment_log_row` is called. In `record_approval`, before `_save_state`, add:

```python
    row["actual_replacement"] = False  # paper-only until real_capital_enabled is wired at call site
```

Note: `record_approval` does not receive `real_capital_enabled` as a kwarg -- it receives the pre-built `status_update_fn`. The `actual_replacement` value in the log row should default to False (the safe value) unless the caller explicitly stores it in the state prior to calling `record_approval`. This keeps the invariant intact and avoids threading real_capital_enabled through multiple function layers.

### Recommended `actual_replacement` derivation logic

```python
# In settings.py, add:
real_capital_enabled: bool = Field(
    False,
    description=(
        "SR 11-7 gate: when True, strategy promotions authorize real-capital deployment. "
        "Must remain False until an explicit compliance pass wires the live-capital layer. "
        "Set via REAL_CAPITAL_ENABLED env var."
    ),
)

# In run_monthly_sortino_gate, the kwarg:
real_capital_enabled: bool = False  # caller injects from settings; default False

# The result dict key:
"actual_replacement": real_capital_enabled,

# At call sites (e.g. friday_promotion.py or autonomous_loop.py):
from backend.config.settings import get_settings
result = run_monthly_sortino_gate(
    ...,
    real_capital_enabled=get_settings().real_capital_enabled,  # defaults False
)
```

### Verifier structure for `tests/verify_phase_25_C3.py`

The verifier must check three immutable criteria:

1. `strategy_registry_table_has_status_field_active_shadow_retired` -- static check: `update_promoted_strategy_status` exists in bigquery_client.py; the BQ query in the function contains `SET status = @new_status`; the existing `status STRING` column is confirmed by migration script grep.

2. `actual_replacement_no_longer_hardcoded_false` -- static check: grep `monthly_champion_challenger.py` for the literal `"actual_replacement": False` (must be absent); confirm `"actual_replacement": real_capital_enabled` or similar dynamic expression is present.

3. `monthly_approval_flips_status_from_shadow_to_active` -- behavioral check: instantiate `record_approval` with a fake state (status=pending, non-expired), pass a mock `status_update_fn`, call `record_approval(..., status="approved", status_update_fn=mock_fn)`, assert mock_fn was called with `(challenger_id, "active")`.

Pattern mirrors verify_phase_25_B3.py: use `MagicMock()` for the update function, temp file for state, assert call args.

---

## Research Gate Checklist

Hard blockers -- `gate_passed` is true if all checked:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (7 sources read in full)
- [x] 10+ unique URLs total (incl. snippet-only) (17 URLs collected)
- [x] Recency scan (last 2 years) performed + reported
- [x] Full pages read (not abstracts/snippets) for the read-in-full set
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every relevant module (monthly_champion_challenger.py, bigquery_client.py, settings.py, test patterns)
- [x] Contradictions / consensus noted (alias-vs-status-field debate noted; resolved in favor of existing enum)
- [x] All claims cited per-claim

---

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 7,
  "snippet_only_sources": 10,
  "urls_collected": 17,
  "recency_scan_performed": true,
  "internal_files_inspected": 6,
  "report_md": "handoff/current/research_brief.md",
  "gate_passed": true
}
```
