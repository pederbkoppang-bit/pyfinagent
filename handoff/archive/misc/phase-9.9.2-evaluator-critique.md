# Q/A Critique — phase-9.9.2 (BQ spend swap) — qa_992_v1

**Verdict: PASS**  **Date:** 2026-04-20

## JSON envelope

```json
{
  "ok": true,
  "verdict": "PASS",
  "qa_id": "qa_992_v1",
  "violated_criteria": [],
  "checks_run": ["harness_audit_5item", "ast_parse", "grep_anthropic_admin_absent", "grep_info_schema_and_6_25", "pytest_slack_bot_full_suite", "live_bq_smoke_test", "spot_read_production_code", "spot_read_test_file", "mutation_rate_constant", "contract_before_generate_mtime", "research_gate_envelope", "no_log_append_yet", "llm_judgment"],
  "reason": "All 7 immutable success criteria met. Deterministic: ast OK, grep ANTHROPIC_ADMIN=0, grep INFORMATION_SCHEMA.JOBS_BY_PROJECT + 6.25 both present, pytest 44/44, live BQ returns daily=$0.0005 monthly=$0.3823 (non-negative, matches experiment-results order). All four researcher-forced corrections present in shipped code: $6.25/TiB, total_bytes_billed, region-us qualifier, direct google.cloud.bigquery.Client. Test renamed + hermeticized. Harness audit all 5 PASS (research gate, contract-before-results mtime, verbatim quote, log-last rule, qa_id v1)."
}
```

## Protocol audit (5/5 PASS)

1. Researcher: 7 in full, 14 URLs, three-variant, recency, gate_passed=true.
2. Contract mtime (1776706411) < results mtime (1776706540).
3. Verbatim verification quoted.
4. No harness_log.md append for 9.9.2 yet.
5. qa_id=qa_992_v1 (cycle v1, not shopping).

## Deterministic reproduction

| Check | Result |
|---|---|
| ast.parse | exit 0 |
| `grep -c ANTHROPIC_ADMIN` | 0 |
| `grep INFORMATION_SCHEMA + 6.25` | multiple hits at expected lines |
| pytest full slack_bot | 44/44 |
| **Live BQ smoke** | `daily=$0.0005 monthly=$0.3823` (Q/A independent run, slight daily drift vs experiment-results due to additional same-day queries — expected; monthly stable) |

## Researcher corrections reflected in shipped code

1. `_BQ_USD_PER_TIB = 6.25` at line 26 (not $5/TB). Used at lines 110-111.
2. `total_bytes_billed` at lines 96-98 (not `total_bytes_processed`).
3. `region-us` qualifier at line 99 (matches CLAUDE.md US multi-region for pyfinagent datasets).
4. Direct `google.cloud.bigquery.Client` at line 93 (not the `BigQueryClient` wrapper that needs a Settings object).

## Test quality

- `test_cost_budget_watcher_bq_unreachable_fail_open` (renamed from admin-key) correctly monkeypatches `google.cloud.bigquery.Client`.
- `test_scheduler_wiring_cost_budget_watcher_fires_zero_args` hermeticized by patching `_default_fetch_spend` — fixed a latent test bug where the old `== 0.0` assumption silently broke on any dev machine with ADC.

## Non-blocking observations

- **Mutation gap:** no runtime assertion on `_BQ_USD_PER_TIB == 6.25`; criterion #7's grep is the only guard. Contract scopes this as acceptable.
- **Latent `BigQueryClient` wrapper issue** in `weekly_data_integrity._default_fetch_counts` flagged by researcher — out-of-scope for this step; file as follow-up.

## Legitimate carry-forwards

- Cap tuning ($5/day → ~$1/day under BQ-only) — policy decision, needs Peder input
- Morning digest line + per-dataset attribution — feature adds, phase-10.x
- `weekly_data_integrity` wrapper `.query()` — separate follow-up ticket

Cleared for log append + task close.
