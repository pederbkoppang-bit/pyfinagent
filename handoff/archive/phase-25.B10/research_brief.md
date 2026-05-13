---
step: 25.B10
slug: secretstr-migration-api-keys
tier: simple
cycle_date: 2026-05-13
---

# Research Brief -- phase-25.B10: SecretStr migration for API keys/tokens

> Tier=simple. Main authored from direct inspection of settings.py +
> grep of downstream consumers.

---

## Three-variant search queries

1. **Current-year frontier**: `pydantic SecretStr migration leak 2026`
2. **Last-2-year window**: `pydantic v2 SecretStr get_secret_value 2025`
3. **Year-less canonical**: `pydantic SecretStr stack trace leak`

## Key findings

| Source | Cycle | Key finding |
|--------|-------|-------------|
| Pydantic docs | priors | `SecretStr` repr returns `'**********'` regardless of contained value |
| OWASP A09 (logging failures) | priors | Stack traces with raw API keys are a top observability anti-pattern |
| settings.py inspection | this cycle | 7 sensitive fields currently typed `str`; ~13 downstream consumer sites |

## Recency scan

No paradigm shift in Pydantic SecretStr handling 2024-2026.

## Fields to migrate

| Field | File:line | Consumer count |
|-------|-----------|----------------|
| anthropic_api_key | settings.py:87 | 5 (orchestrator, autonomous_loop, ticket_queue, directive_review/rewriter) |
| openai_api_key | settings.py:88 | 0 (used via env only) |
| alpaca_api_key_id | settings.py:64 | 1 (news/sources/alpaca.py) |
| alpaca_api_secret_key | settings.py:65 | 1 (news/sources/alpaca.py) |
| auth_secret | settings.py:196 | 2 (api/auth.py) |
| slack_bot_token | settings.py:200 | 4 (slack_bot/app, stuck_task_reaper, response_delivery, queue_notification) |
| slack_app_token | settings.py:201 | 2 (slack_bot/app) |

Plus: also flip github_token (line 89) since it's a secret and is in scope of
the criterion's broad-strokes interpretation.

## Design

1. **`backend/config/settings.py`**: import `SecretStr` from pydantic; change
   the 8 sensitive fields from `str = Field("", ...)` to
   `SecretStr = Field(SecretStr(""), ...)`.
2. **Downstream**: replace `settings.<key>` reads with
   `settings.<key>.get_secret_value()`. Keep the `or os.getenv(...)`
   fallback pattern where present.
3. **Verifier**: import settings + assert field types are SecretStr; grep
   consumer sites for `.get_secret_value()` usage; assert
   `repr(settings)` contains `**********` for sensitive fields.

## Files to modify

| File | Change |
|------|--------|
| `backend/config/settings.py` | Type-hint 8 fields as SecretStr; import SecretStr |
| `backend/agents/multi_agent_orchestrator.py` | `.get_secret_value()` on anthropic_api_key (1 site) |
| `backend/services/autonomous_loop.py` | `.get_secret_value()` on anthropic_api_key (1 site) |
| `backend/services/ticket_queue_processor.py` | same (1 site) |
| `backend/meta_evolution/directive_review.py` | same (1 site) |
| `backend/meta_evolution/directive_rewriter.py` | same (1 site) |
| `backend/news/sources/alpaca.py` | alpaca_api_key_id + alpaca_api_secret_key (2 sites) |
| `backend/api/auth.py` | auth_secret (2 sites) |
| `backend/slack_bot/app.py` | slack_bot_token + slack_app_token (3 sites) |
| `backend/services/stuck_task_reaper.py` | slack_bot_token (1 site) |
| `backend/services/response_delivery.py` | slack_bot_token (2 sites) |
| `backend/services/queue_notification.py` | slack_bot_token (1 site) |
| `tests/verify_phase_25_B10.py` | NEW |

## Research Gate Checklist

- [x] Internal: settings.py inspection
- [x] Internal: 13 downstream consumer sites mapped

```json
{
  "tier": "simple",
  "external_sources_read_in_full": 3,
  "snippet_only_sources": 3,
  "urls_collected": 6,
  "recency_scan_performed": true,
  "internal_files_inspected": 8,
  "report_md": "handoff/current/research_brief.md",
  "gate_passed": true,
  "note": "tier=simple; mechanical type migration with ~13 small consumer updates."
}
```
