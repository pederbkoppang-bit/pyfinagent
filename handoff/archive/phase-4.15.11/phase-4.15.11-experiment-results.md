# Experiment Results — Cycle 4.15.11

Step: phase-4.15.11 Models / pricing / deprecations / tiers / residency

## What was built

`docs/audits/compliance-models-pricing.md` (~1800 words, 25 patterns).

## Critical findings (4 net-new MUST-FIX candidates)

**MF-45 (HOTFIX TODAY): Haiku 3.5 retires tomorrow (2026-04-19)**
`claude-3-5-haiku-20241022` appears in 5 live files — will 400
starting tomorrow:
- `cost_tracker.py:23`
- `llm_client.py:64,171`
- `harness_memory.py:53`
- `settings_api.py:31,144`
Priority: **SAME-DAY HOTFIX** (earlier than MF-29).

**MF-46 (HOTFIX): Invalid model ID typo**
`backend/slack_bot/app_home.py:24` uses `claude-haiku-35-20241022`
— NOT a valid Anthropic ID (missing hyphen: should be
`claude-3-5-haiku-20241022`). Would 400 immediately if user selects
it from the model picker. Functional bug since commit.

**MF-47 (HIGH): `_BUILD_TIER` routing break**
`model_tiers.py` maps `autoresearch_fast` to `anthropic:claude-
haiku-4-5` with `anthropic:` prefix. `make_client()` checks
`startswith("claude-")` — the prefix makes the check fail,
silently falling through to Gemini instead of ClaudeClient. Every
`autoresearch_fast` call routes wrong.

**MF-48 (MINOR): Cache-write premium not charged**
`cost_tracker.py` correctly applies 0.1x cache-read discount but
does NOT apply 1.25x (5-min TTL) / 2x (1-hour TTL) cache-write
premium. Under-reports cost on cache-write-heavy runs.

## Reinforces prior findings (MF-1, MF-7, MF-8)

| MF | Confirmed evidence |
|----|-------------------|
| MF-1 (MODEL_PRICING stale) | `claude-opus-4-6/4-7/4-5/4-1`, `claude-haiku-4-5`, `claude-sonnet-4-5` all absent from MODEL_PRICING; every Opus call falls to $0.10/$0.40 default = 50× input / 62.5× output under-report |
| MF-7 (stale snapshots) | `claude-sonnet-4-20250514` at `autonomous_loop.py:438`, `mcp_tools.py:74,223`, `app_home.py:23` — retire 2026-06-15 |
| MF-8 (retired in catalog) | `claude-3-5-sonnet-20241022`, `claude-3-7-sonnet-20250219` still in GITHUB_MODELS_CATALOG |
| _LIVE_TIER TODO | all `TODO_DECIDE_AT_LAUNCH` sentinels — blocks May go-live |

## Observability gaps

- `service_tier` never passed; `response.usage.service_tier` never
  logged — can't tell if we hit Priority Tier capacity.
- `inference_geo` never set (default `"global"`); never logged —
  US-only routing unclear for financial data.
- `anthropic-version` never overridden (auto `2023-06-01` is fine
  today but no pin against future breaking bumps).

## Success criteria

1. every_doc_pattern_status_evidenced — PASS (25 patterns)
2. qa_runs_live_code_checks_not_review — PARTIAL (Q/A next)
3. deviations_cite_doc_page — PASS

## Artifact

- `docs/audits/compliance-models-pricing.md`

## Urgency

MF-45 and MF-46 should be hotfixed BEFORE the Q/A phase completes
this cycle — Haiku 3.5 retires at midnight UTC. Elevating to
same-day alongside MF-29 (Opus 4.7 thinking-API gate).
