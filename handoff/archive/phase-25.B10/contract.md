---
step: 25.B10
slug: secretstr-migration-api-keys
status: in_progress
cycle_date: 2026-05-13
parent_research_brief: handoff/current/research_brief.md
---

# Contract -- phase-25.B10

## Step ID + masterplan reference

`25.B10` -- "SecretStr migration for API keys/tokens"
(P2, harness_required, no dep).

## Research-gate summary

Tier=simple. Brief at `handoff/current/research_brief.md`,
`gate_passed=true`.

## Hypothesis

Plain-`str` API keys leak via `repr(settings)` in stack traces and
log lines. Migrating sensitive fields to `pydantic.SecretStr` makes
the repr return `'**********'` automatically, while
`.get_secret_value()` is required to access the underlying string.

## Success criteria (verbatim from masterplan.json)

> `anthropic_api_key_is_secretstr_type`
>
> `openai_alpaca_auth_slack_keys_all_secretstr`
>
> `downstream_consumers_use_get_secret_value`

## Plan steps

1. **`backend/config/settings.py`**: import `SecretStr`; type-flip 8 fields
   (anthropic_api_key, openai_api_key, github_token, alpaca_api_key_id,
   alpaca_api_secret_key, auth_secret, slack_bot_token, slack_app_token).
2. **Update 13 downstream consumer sites** to call `.get_secret_value()`.
3. **Verifier** -- `tests/verify_phase_25_B10.py` with 5 claims:
   - Claim 1: settings.py imports SecretStr.
   - Claim 2: anthropic_api_key annotation is SecretStr.
   - Claim 3: openai_api_key + alpaca + auth_secret + slack tokens are all SecretStr.
   - Claim 4: at least 8 .get_secret_value() call sites across the backend.
   - Claim 5: repr(settings) renders all sensitive fields as '**********'
     (live import + repr).

## Files

| File | Action |
|------|--------|
| `backend/config/settings.py` | SecretStr type migration (8 fields) |
| `backend/agents/multi_agent_orchestrator.py` | get_secret_value() (1 site) |
| `backend/services/autonomous_loop.py` | get_secret_value() (1 site) |
| `backend/services/ticket_queue_processor.py` | get_secret_value() (1 site) |
| `backend/meta_evolution/directive_review.py` | get_secret_value() (1 site) |
| `backend/meta_evolution/directive_rewriter.py` | get_secret_value() (1 site) |
| `backend/news/sources/alpaca.py` | get_secret_value() (2 sites) |
| `backend/api/auth.py` | get_secret_value() (2 sites) |
| `backend/slack_bot/app.py` | get_secret_value() (3 sites) |
| `backend/services/stuck_task_reaper.py` | get_secret_value() (1 site) |
| `backend/services/response_delivery.py` | get_secret_value() (2 sites) |
| `backend/services/queue_notification.py` | get_secret_value() (1 site) |
| `tests/verify_phase_25_B10.py` | NEW |

## Verification command (immutable)

```
source .venv/bin/activate && python3 tests/verify_phase_25_B10.py
```

## Live-check

`Test: repr(settings) shows '**********' for all sensitive fields`.
Will write `handoff/current/live_check_25.B10.md`.

## Risks + mitigations

- **Risk**: Missed consumer site causes runtime AttributeError on .get_secret_value().
  **Mitigation**: After migration, run `python -c "from backend.config.settings import get_settings; s = get_settings(); print(s.anthropic_api_key)"` to confirm. Pre-Q/A, grep for any remaining `settings.<sensitive_key>` (without `.get_secret_value()`) to catch.
- **Risk**: Pydantic's `BaseSettings` may need `SecretStr` to be unwrapped
  for env loading.
  **Mitigation**: Pydantic-settings handles SecretStr automatically;
  default value just needs `SecretStr("")`.

## References

- `handoff/current/research_brief.md`
- `backend/config/settings.py:64, 65, 87-89, 196, 200-201`
- All 13 consumer sites listed in research_brief.md
- `.claude/masterplan.json::25.B10`
