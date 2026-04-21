# Q/A Critique — phase-9.9.1 (wiring fix) — qa_991_v1

**Verdict: PASS**  **Date:** 2026-04-20

## JSON envelope

```json
{
  "ok": true,
  "verdict": "PASS",
  "qa_id": "qa_991_v1",
  "violated_criteria": [],
  "violation_details": [],
  "certified_fallback": false,
  "checks_run": ["syntax_ast_parse", "pytest_targeted_20of20", "pytest_full_slack_bot_44of44", "bare_run_regression_reproduction", "handoff_files_exist", "contract_mtime_before_results", "researcher_gate_passed", "mutation_resistance_check", "codebase_idiom_consistency", "fail_open_semantics_review"],
  "reason": "All 7 immutable success criteria met. Both qa_99_remediation_v1 bugs fixed via Option B (callable injection). 44/44 slack_bot tests pass. Bare cost_budget_watcher.run() call returns OK 0.0 False (no TypeError). Mutation-resistance verified: reverting required-args would fail test_scheduler_wiring_cost_budget_watcher_fires_zero_args. Fail-open semantics match existing alert_fn pattern. Researcher gate_passed=true with 5 sources in full + recency + three-variant queries. Carry-forwards legit out-of-scope infra."
}
```

## Protocol audit (5/5 PASS)

1. Researcher: 5 in full, 15 URLs, three-variant, recency, gate_passed=true, moderate tier.
2. Contract mtime (1776705041) < results mtime (1776705150).
3. Contract verbatim-quotes immutable criteria 1-7.
4. No log append yet.
5. Cycle v1; qa_id=qa_991_v1.

## Deterministic reproduction

| Check | Result |
|---|---|
| ast.parse both files | exit 0 |
| pytest targeted (cost_budget + data_integrity + scheduler_phase9 + wiring_phase991) | 20/20, exit 0 |
| pytest full slack_bot/ | 44/44, exit 0 |
| **Bare `cost_budget_watcher.run()` call** | `OK 0.0 False` — TypeError eliminated |
| Handoff files | all 3 present |
| Mutation: revert L28-29 to required args | `test_scheduler_wiring_cost_budget_watcher_fires_zero_args` FAILS (`TypeError: run() missing 1 required keyword-only argument`) |
| `_default_fetch_spend` fail-open | missing key at L82-84 → `(0.0, 0.0)`; exception at L107-109 → `(0.0, 0.0)` |
| `_save_snapshot` auto-create parent | L109-110 `mkdir(parents=True, exist_ok=True)` |

## LLM judgment

- Both qa_99 bugs addressed: (1) TypeError eliminated via optional params + `fetch_fn` injection; (2) inert-empty-dict fixed via `_default_fetch_counts` + snapshot persistence.
- Fail-open semantics defensible: matches `alert_fn` fail-open precedent in same file; fail-closed would trigger phantom budget alerts on API outage.
- Option B matches codebase idiom exactly (`daily_price_refresh.py:36`, `nightly_mda_retrain.py:37`).
- Carry-forwards legit: `ANTHROPIC_ADMIN_API_KEY` provisioning, BQ IAM, runbook are infra/docs work genuinely out of scope. Fail-open defaults make these non-blocking.
- Scope honesty: experiment openly discloses 0.0/0.0 until admin key provisioned. No overclaim.

Cleared for log append + task completion.
