# Live-check placeholder -- phase-25.B10

**Step:** 25.B10 -- SecretStr migration for API keys/tokens
**Date:** 2026-05-13

## Live-check field (per masterplan)
> "Test: repr(settings) shows '**********' for all sensitive fields"

## Pre-deployment evidence
- 5/5 verifier PASS.
- Claim 5 is a LIVE repr test: injects fake secrets via env, instantiates
  Settings, asserts repr() contains `**********` AND the actual secrets
  do NOT appear in the repr string.
- 16 consumer call sites confirmed to use `.get_secret_value()`.
- AST clean on all 12 touched .py files.

## Post-deployment operator workflow
1. Pull main + restart backend:
   ```
   git pull origin main
   source .venv/bin/activate
   pkill -f "uvicorn backend.main" || true
   python -m uvicorn backend.main:app --reload --port 8000 &
   ```
2. Confirm repr masks the real secret:
   ```
   python -c "
   from backend.config.settings import get_settings
   get_settings.cache_clear()
   s = get_settings()
   r = repr(s)
   assert '**********' in r
   for f in ['anthropic_api_key', 'auth_secret', 'slack_bot_token']:
       leak = getattr(s, f).get_secret_value()
       if leak and leak in r:
           print(f'LEAK in {f}!')
           break
   else:
       print('All sensitive fields masked')
   "
   ```
3. Sanity-check that auth still works post-migration:
   ```
   curl -s http://localhost:8000/api/auth/me  # or any auth-gated endpoint
   ```

## Closes audit basis
bucket 24.10 F-4 RESOLVED. API keys can no longer leak via repr() in
stack traces, log lines, or pydantic-settings debug output.

**Audit anchor for next bucket:** 25.B6 (seed-stability), 25.F3, 25.B10.1 (lesser-secret cleanup).
