# Q/A Critique — phase-9.9 (scheduler wiring) — REMEDIATION v1

**Verdict: PASS**  **qa_id:** qa_99_remediation_v1  **Date:** 2026-04-20

## JSON envelope

```json
{
  "ok": true,
  "verdict": "PASS",
  "qa_id": "qa_99_remediation_v1",
  "violated_criteria": [],
  "checks_run": ["5_item_harness_audit", "syntax_ast_parse", "pytest_4_of_4", "file_existence_3_of_3", "contract_mtime_before_results_mtime", "spot_read_scheduler_line_374", "independent_reproduction_cost_budget_watcher_TypeError", "independent_reproduction_weekly_data_integrity_inertness", "research_brief_review", "llm_judgment_disclosure_placement", "llm_judgment_defer_vs_fix", "llm_judgment_antirubber_stamp"],
  "reason": "Immutable criterion passes (ast OK; 4/4 pytest). 5-item audit clean. 8-source moderate-tier brief with recency + three-variant + gate_passed=true. Runtime-bug disclosures prominent in contract + results + brief — not buried. Independent reproduction CONFIRMS both bugs: cost_budget_watcher.run() zero-arg raises TypeError; weekly_data_integrity.run() inert via empty-dict defaults. scheduler.py:374 **kwargs is trigger-config not forwarded to func — matches brief root-cause. Anti-rubber-stamp honest: 'tests verify registration only, not invocation'."
}
```

## Protocol audit (5/5 PASS)

1. Researcher: 8 full, 19 URLs, three-variant, recency, gate_passed=true, moderate tier.
2. Contract mtime (1776704019) < results mtime (1776704035).
3. Verbatim verification output quoted.
4. No 9.9 log append yet.
5. Cycle v1; not a verdict-shop.

## Deterministic reproduction

| Check | Result |
|---|---|
| ast.parse | exit 0 |
| pytest -q | 4/4, exit 0 |
| Handoff files | all 3 present |
| scheduler.py:374 `**kwargs` is trigger-config | confirmed — not forwarded to `func` |
| Independent repro: `cost_budget_watcher.run()` zero-arg | raises `TypeError: run() missing 2 required keyword-only arguments: 'daily_spend_usd' and 'monthly_spend_usd'` |
| Independent repro: `weekly_data_integrity.run()` zero-arg | returns `{"drifts": []}` (functionally inert, heartbeat "ok") |

## Defer-vs-fix decision (LLM judgment)

**DEFENSIBLE but requires hard gate.** Arguments for deferral:
1. Remediation scope is harness-compliance, not hardening — scope discipline is a harness virtue.
2. Fail-open wrapper means TypeError is caught + logged; no cascading failure.
3. Fixing `cost_budget_watcher` correctly requires a design decision (defaults + side-channel fetch vs. pass `kwargs=` at registration) best handled in a dedicated ticket.

Arguments against:
1. A job silently doing nothing is worse than crashing loudly — heartbeat "ok" for a budget watcher that never checks budget is a monitoring-gap bug.
2. One-line fix could have fit this cycle.

## Recommendation to Main

ACCEPT deferral for THIS cycle, but **block phase-9 go-live on a dedicated `phase-9.9.1-cost-budget-watcher-wiring`** with immutable criterion that `cost_budget_watcher.run()` fires successfully under zero-arg APScheduler invocation (integration test, not registration). Operationalize the carry-forward as an actual masterplan entry before flipping 9.9 done.

Cleared for log append.
