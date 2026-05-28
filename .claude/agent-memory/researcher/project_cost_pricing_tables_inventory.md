---
name: cost-pricing-tables-inventory
description: There are THREE+ independent LLM pricing tables in backend/ that must all be updated on a model add/reprice; sovereign_api imports the canonical one
metadata:
  type: project
---

When adding a new Claude model or repricing, pyfinagent has **multiple
independent pricing tables** that do NOT share a source — updating only
`cost_tracker.py` leaves the others stale.

- **Canonical:** `backend/agents/cost_tracker.py::MODEL_PRICING` —
  `dict[str, tuple[float, float]]`, **per-1M-token** `(input, output)`,
  bare hyphenated key (`claude-opus-4-8`). Fallthrough
  `_DEFAULT_PRICING=(0.10, 0.40)` silently understates a missing Opus
  key ~50x in / ~62.5x out. `backend/api/sovereign_api.py` IMPORTS this
  dict (so it inherits fixes — do not duplicate there).
- **Separate display list:** `backend/api/settings_api.py` ~line 214 —
  `{"model":..., "input_per_1m":..., "output_per_1m":...}` rows for the
  Settings UI cost table. NOT imported from MODEL_PRICING. Plus a model
  allowlist tuple ~line 31 that may gate UI selectability.
- **Separate rough estimate:** `backend/slack_bot/governance.py` ~line
  84 — hardcoded `input*0.00001 + output*0.00003` (=$10/$30 per 1M),
  model-agnostic, labelled "update for actual pricing". Pre-existing
  inaccuracy.
- **NOT an LLM-$ table:** `backend/slack_bot/jobs/cost_budget_watcher.py`
  only meters BigQuery bytes ($6.25/TiB). pyfinagent is Claude **Max
  flat-fee**, so there is no live LLM-dollar meter; the cost_tracker
  numbers are accounting/north-star ("Compute" term), not a billing gate.

**Why:** phase-47.3 found commit 8ecc9efe bumped `model_tiers.py` 4.7->4.8
but missed `cost_tracker.py` AND `settings_api.py`. Confirmed Opus 4.8 =
$5/$25 (same as 4.7) via Anthropic pricing docs.

**How to apply:** on any model add/reprice, grep
`grep -rln "PRICING\|per_1m\|cost_per" backend/ --include=*.py` and patch
ALL the independent tables, not just cost_tracker. Non-brittle test:
assert the new model's tuple `== ` the same-priced sibling AND `!=`
`_DEFAULT_PRICING`. See [[anthropic-agent-patterns]] context for the 4.8
effort/max_tokens interaction (effort consumes max_tokens; set ~64k at
xhigh/max — small per-agent caps in orchestrator.py are a latent risk if
a reasoning agent is ever routed to Opus 4.8 at xhigh).
