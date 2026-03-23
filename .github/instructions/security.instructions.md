---
applyTo: "backend/**"
---

# Security — Backend Conventions

## Authentication
- NextAuth.js v5 JWE tokens decrypted via `HKDF(AUTH_SECRET, info=b"Auth.js Generated Encryption Key", salt=b"", length=64)` → A256CBC-HS512/dir
- Email whitelist enforced on both frontend (NextAuth callback) and backend (`ALLOWED_EMAILS`)
- Auth middleware skips: `/api/health`, `/api/auth`, `/docs`, `/openapi.json`, `/redoc`

## OWASP Headers (all responses)
- `X-Content-Type-Options: nosniff`
- `X-Frame-Options: DENY`
- `X-XSS-Protection: 1; mode=block`
- `Referrer-Policy: strict-origin-when-cross-origin`
- `Cache-Control: no-store`
- `Permissions-Policy` (restricted)

## CORS
- Allows `localhost:*` and Tailscale IPs (`100.x.y.z:*`) via regex pattern

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
