---
step: phase-16.22
cycle_date: 2026-04-25
verdict: PASS
reviewer: qa
---

# Q/A Critique -- phase-16.22

## Harness-compliance (5 items)

1. **Research gate**: `phase-16.22-research-brief.md` exists (10950 bytes,
   mtime 07:20). JSON envelope reports `gate_passed: true`,
   `external_sources_read_in_full: 5`, `urls_collected: 12`,
   `recency_scan_performed: true`. 3-variant search-query discipline visible
   (current-year 2026 + last-2-year 2025 + year-less canonical). Spot-check:
   `https://docs.slack.dev/tools/bolt-python/building-an-app/` returns
   HTTP 301 (resolves). PASS.

2. **Contract-before-GENERATE**: contract.md mtime 07:22:52 < experiment_results
   mtime 07:23:37. Contract immutable success criteria copied verbatim from
   masterplan. PASS.

3. **Experiment results**: step=phase-16.22, includes verbatim 5-stage output.
   3 launchd anomalies disclosed (autoresearch=127, gateway=1, backend=-15) with
   classification (carry-forward / transient / expected-from-bounce). Honest
   disclosure section explicit. PASS.

4. **Log-last**: `grep -c "phase-16.22" handoff/harness_log.md` = 0. Log append
   correctly deferred until after Q/A PASS. PASS.

5. **No verdict-shopping**: Prior critique was for 16.21 (different step).
   No prior 16.22 critique to override. PASS.

## Deterministic checks (5/5 verification stages re-run)

- **slack**: `slack_ok` returned. `build_app()` constructs AsyncApp. PASS.
- **scheduler**: `scheduler_active: True | next_run: 2026-04-27T14:00:00-04:00`.
  PASS.
- **launchd**: 7 jobs present (mas-harness, claude-code-proxy, ablation,
  openclaw.gateway, autoresearch, backend, frontend). PASS.
- **freshness**: HTTP 200, heartbeat band=green (ratio 0.474), source bands
  "unknown" (paper portfolio empty pre-Monday — consistent with disclosure).
  PASS.
- **cost_budget_status**: HTTP 200, daily $0.0005 / monthly $1.9136 well under
  $5/$50 caps, tripped=false. PASS.

## Alias purity check

- **build_app**: Single line `build_app = create_app` at app.py:42. Comment
  block lines 39-41 explains alias rationale. Does NOT shadow or redefine
  `create_app` (line 27 unchanged; line 53 still calls `create_app()` from
  `main()`). Pure module-level rebinding. PURE ALIAS.

- **freshness**: `observability_api.py:25-41` registers `/freshness` route
  that imports and delegates to
  `backend.services.cycle_health.compute_freshness` -- the SAME function
  `paper_trading.py::get_freshness` uses. Same payload shape (sources +
  heartbeat + bq_ingest_lag_sec + thresholds + computed_at). No divergence at
  the helper level. PURE DELEGATION.

- **cost_budget_status**: `cost_budget_api.py:98-104` registers `/status` with
  same `response_model=CostBudgetToday` and body `return await
  get_cost_budget_today()`. Single-line delegation. PURE ALIAS.

## Regression

- **pytest_api**: `7 passed, 1 warning in 2.01s`. Matches Main's reported 7/7.
  PASS.

## LLM judgment

- **alias_within_plan_scope**: PASS. Plan says "very few touch code (only
  fixes that surface from verification, and only if blocking Monday)".
  3 aliases / ~24 net lines / pure delegation / no behavior change qualifies.
  Counter-argument that aliases are "new functionality" is technically true at
  the route level but the underlying logic (`compute_freshness`,
  `get_cost_budget_today`, `create_app`) all pre-existed; no new code paths,
  no new business logic. The aliases respect Q/A's 16.21 escalation clause
  ("third structurally-identical CONDITIONAL must FAIL otherwise harness is
  a logger not a corrector"). Main chose corrector -- correct call.

- **dual_source_of_truth_freshness**: ACKNOWLEDGED RISK, NOT BLOCKING.
  Both `/api/observability/freshness` and `/api/paper-trading/freshness`
  delegate to the same `cycle_health.compute_freshness` helper, so today
  there is no truth-divergence -- only route-surface duplication.
  Future-proofing concern: if `paper_trading.py::get_freshness` adds caching,
  auth, or request-shaping wrappers, the observability alias won't pick it up
  unless explicitly mirrored. **Follow-up ticket**: doc-reconcile in 16.23 or
  after to either (a) consolidate to one route + redirect, or (b) document
  both as intentional with a shared helper contract.

- **autoresearch_exit_127_drill**: ACCEPTED AS DISCLOSED. Exit 127 is
  ENOENT-class (binary or working-dir missing). Independent of paper-trading
  flow; autoresearch is the parameter-optimizer loop, fail-open hourly.
  Main's flag-but-don't-drill is appropriate given Monday-readiness focus.
  **Follow-up ticket**: drill autoresearch launchd plist + binary path before
  next optimizer cycle is needed (post-go-live).

- **harness_as_corrector_pattern_health**: HEALTHY THIS CYCLE, BUT
  ESTABLISHES A LIMIT. The 16.18 (TZ env) and 16.22 (3 aliases) fixes are
  legitimate harness-as-corrector behavior -- small, surgical, behavior-
  preserving. However, the recurring failure mode is masterplan verification
  commands written without first checking the code. This is a meta-issue:
  long-term, the doc-reconciliation follow-up from 16.21 should produce a
  pre-flight script that diff-checks every immutable verification command's
  Python imports / route paths against the live codebase before a step opens.
  Without that, the alias-pattern can become a smell. Logging as a
  follow-up, not a blocker.

## Verdict

```json
{
  "ok": true,
  "verdict": "PASS",
  "reason": "All 5 immutable success criteria met (slack_app_builds, scheduler_active_true, launchd_jobs_present, observability_freshness_200, cost_budget_status_200) re-verified independently. 3 aliases confirmed pure-delegation with no shadowing. 7/7 pytest api green. Harness-compliance audit clean (5/5). Plan-scope honored (~24 net lines, no behavior change). Honest disclosure of 3 launchd exit anomalies with appropriate Monday-blocker triage.",
  "violated_criteria": [],
  "violation_details": [],
  "follow_up_tickets": [
    "Doc-reconcile freshness route surface duplication: /api/observability/freshness and /api/paper-trading/freshness both delegate to compute_freshness -- either consolidate or document both as intentional with shared-helper contract (carry from 16.21).",
    "Drill com.pyfinagent.autoresearch launchd exit_status=127 (ENOENT-class) before next optimizer cycle; not a Monday paper-trading blocker.",
    "Pre-flight script: diff-check every masterplan immutable verification command's Python imports / route paths against live codebase BEFORE step opens, to prevent recurrence of the 16.20/16.21/16.22 alias pattern."
  ],
  "checks_run": [
    "research_brief_gate_envelope",
    "research_brief_url_spotcheck",
    "contract_before_generate_mtime",
    "log_last_invariant",
    "no_verdict_shopping",
    "verification_stage_1_slack",
    "verification_stage_2_scheduler",
    "verification_stage_3_launchd",
    "verification_stage_4_freshness",
    "verification_stage_5_cost_budget",
    "alias_purity_build_app",
    "alias_purity_freshness",
    "alias_purity_cost_budget_status",
    "pytest_regression_backend_tests_api",
    "llm_judgment_scope_dual_source_autoresearch_pattern_health"
  ],
  "certified_fallback": false
}
```
