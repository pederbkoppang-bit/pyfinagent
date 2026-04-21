# Contract — Cycle 4.15.11 — Models / pricing / deprecations / tiers / residency

## Research gate
Merged researcher: external pricing + models docs + service-tiers +
data residency + deprecation timeline + admin API; internal
`cost_tracker.MODEL_PRICING`, `model_tiers.py`, `llm_client.py
GITHUB_MODELS_CATALOG`, every Claude model ID string in `backend/`.

## Hypothesis
Confirms phase-4.11 finding: MODEL_PRICING table is stale;
GITHUB_MODELS_CATALOG has retired IDs; autonomous_loop.py:438 still
uses `claude-sonnet-4-20250514`. Plus live-check: does any code
actually check deprecation dates or fail loudly on retired IDs?

## Success criteria
1. every_doc_pattern_status_evidenced
2. qa_runs_live_code_checks_not_review
3. deviations_cite_doc_page

## Scope
`docs/audits/compliance-models-pricing.md`: every documented model
in pricing table × every ID in pyfinagent code. Plus service_tier,
inference_geo, `anthropic-version` header, deprecation schedule.

## References
Phase-4.11 misc_and_admin.md; phase-4.14 MF-1, MF-7, MF-8.
