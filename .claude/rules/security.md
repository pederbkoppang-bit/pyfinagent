---
paths:
  - "backend/**"
---

# Security — Backend Conventions

## Authentication
- NextAuth.js v5 JWE tokens decrypted via `HKDF(AUTH_SECRET, info=b"Auth.js Generated Encryption Key", salt=b"", length=64)` → A256CBC-HS512/dir
- Email whitelist enforced on both frontend (NextAuth callback) and backend (`ALLOWED_EMAILS`). Empty backend list is fail-open (legacy) unless `AUTH_ENFORCE_ALLOWLIST=true` (phase-75.1 DARK flag: empty list then rejects all authenticated users). Empty list logs a startup WARNING either way.
- Auth middleware skips EXACTLY these prefixes (`_PUBLIC_PATHS`, backend/main.py — phase-75.1; adding one requires an inline justification there AND a row here):
  - `/api/health` — liveness probe (Slack bot, launchd watchdog, away-ops healthcheck)
  - `/api/changelog` — public changelog page, no business data
  - `/api/auth` — session bootstrap, must be reachable pre-auth
  - `/api/jobs/status` — read-only job liveness poll (Slack bot + frontend header)
  - `/api/harness/demotion-audit` — read-only harness telemetry (Harness tab)
  - `/api/harness/weekly-ledger` — read-only harness telemetry (Harness tab)
  - `/api/harness/candidate-space` — read-only harness telemetry (Harness tab)
  - `/api/harness/results-distribution` — read-only harness telemetry (Harness tab)
- `/docs`, `/openapi.json`, `/redoc` are debug-only (`DEBUG=true`); in the prod default they are not mounted at all (`openapi_url=None`).
- Localhost tooling (Slack bot non-public calls, smoke scripts, immutable masterplan curls) relies on the `DEV_LOCALHOST_BYPASS=1` + client-is-127.0.0.1 rail in `backend/api/auth.py::get_current_user` — both conditions required.

## OWASP Headers (all responses)
- `X-Content-Type-Options: nosniff`
- `X-Frame-Options: DENY`
- `X-XSS-Protection: 1; mode=block`
- `Referrer-Policy: strict-origin-when-cross-origin`
- `Cache-Control: no-store`
- `Permissions-Policy` (restricted)

## CORS
- Allows `localhost:*` and Tailscale CGNAT IPs only (`100.64.0.0/10`, second octet 64–127, RFC 6598) via the single module-level `_TAILSCALE_ORIGIN_RE` in `backend/main.py` — shared by CORSMiddleware `allow_origin_regex` AND the manual 401 CORS echo so the two seams cannot drift (phase-75.1)

## Secret Management
- Local: `backend/.env` file, `frontend/.env.local`
- Production: Google Cloud Secret Manager (never hardcode secrets)
- API keys: ALPHAVANTAGE_API_KEY, FRED_API_KEY, API_NINJAS_KEY, GITHUB_TOKEN, ANTHROPIC_API_KEY, OPENAI_API_KEY

## Input Validation
- All API endpoints validate input parameters
- Ticker symbols sanitized (alphanumeric + dots only)
- No raw user input passed directly to LLM prompts without sanitization
- SEC EDGAR requires custom User-Agent (`FirstName LastName email@domain.com`)
- External API calls respect rate limits with automatic retry and exponential backoff

## Logging
- **ASCII-only logger messages**: Never use Unicode characters (arrows `\u2192`, em dashes `\u2014`, etc.) in `logger.*()` calls. Windows cp1252 encoding in uvicorn handlers crashes on non-ASCII. Use `--`, `->`, plain English instead.
- `setup_logging()` in `main.py` clears uvicorn handlers and forces UTF-8 `TextIOWrapper`. Still use ASCII for defense-in-depth.
