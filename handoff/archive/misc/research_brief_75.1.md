# Research Brief — Step 75.1: Backend auth surface fail-closed

Tier: moderate | Date: 2026-07-20 | Researcher: Layer-3 (merged external + internal)

## Topic

FastAPI auth-surface fail-closed hardening: pruning public-path allowlists on
authenticated APIs; conditional OpenAPI/docs gating (docs_url/openapi_url None
in prod, settings.debug-gated); CORS allow_origin_regex scoping to the Tailscale
CGNAT block 100.64.0.0/10 with credentials; Pydantic v2 Literal enums + FastAPI
Path(pattern=...) for 422-on-invalid-input; fail-open vs fail-closed
email-allowlist auth patterns and safe DARK-flag rollout.

## Queries planned

1. Current-year (2026): "FastAPI security best practices public path allowlist 2026"
2. Last-2-year (2025): "FastAPI disable docs production openapi_url None 2025"
3. Year-less canonical: "FastAPI middleware authentication exclude paths"
4. Year-less canonical: "Tailscale CGNAT 100.64.0.0/10 CORS origin"
5. Year-less canonical: "fail open fail closed authentication design"
6. Current-year: "Pydantic v2 Literal enum validation FastAPI 422 2026"

## Queries actually run (3-variant discipline)

1. Current-year (2026): `FastAPI security best practices public path allowlist authentication middleware 2026`
2. Last-2-year (2025): `FastAPI disable docs production conditional OpenAPI openapi_url None settings 2025`
3. Year-less canonical: `fail open fail closed authentication design allowlist security`

## Source table (read in full) — INCREMENTAL, appended as read

| # | URL | Accessed | Kind | Fetched how | Key finding |
|---|-----|----------|------|-------------|-------------|
| 1 | https://fastapi.tiangolo.com/how-to/conditional-openapi/ | 2026-07-20 | official doc | WebFetch full | `Settings.openapi_url: str = "/openapi.json"` + `FastAPI(openapi_url=settings.openapi_url)`; empty/None value 404s `/openapi.json`, `/docs`, AND `/redoc` together; explicit caveat: hiding docs is "Security through obscurity" — real protection is auth on the path operations (which is exactly what removing them from `_PUBLIC_PATHS` adds) |
| 2 | https://fastapi.tiangolo.com/tutorial/cors/ | 2026-07-20 | official doc | WebFetch full | `allow_origin_regex` = "regex string to match against origins"; CRITICAL: "None of `allow_origins`, `allow_methods` and `allow_headers` can be set to `['*']` if `allow_credentials` is set to `True`. All of them must be explicitly specified." — pyfinagent currently violates this with `allow_methods=["*"], allow_headers=["*"], allow_credentials=True` (Starlette reflects requested values in practice, but the documented contract wants explicit lists) |
| 3 | https://tailscale.com/kb/1015/100.x-addresses | 2026-07-20 | official vendor doc | WebFetch full | Tailscale assigns from `100.64.0.0/10` = `100.64.0.0`–`100.127.255.255` → **second octet 64–127** (RFC 6598 CGNAT shared address space); addresses are stable per node across network moves; the rest of 100.0.0.0/8 is NOT Tailscale (publicly-routable) — confirming security-04's core claim |
| 4 | https://owasp.org/www-community/Fail_securely | 2026-07-20 | canonical standard | WebFetch full | "a failure will follow the same execution path as disallowing the operation"; `isAuthorized()`-class methods "should all return false if there is an exception"; canonical wrong-pattern = default-allow before try (exact analogue of `if settings.allowed_emails:` skipping the check when empty) |
| 5 | https://fastapi.tiangolo.com/reference/parameters/ | 2026-07-20 | official doc | WebFetch full | `Path()` signature includes `pattern: str \| None = None` ("RegEx pattern for strings"); `regex=` is "Deprecated in FastAPI 0.100.0 and Pydantic v2, use `pattern` instead"; Path params always required; string constraints min_length/max_length/pattern all valid on Path |
| 6 | https://pydantic.dev/docs/validation/latest/api/pydantic/standard_library_types/ (redirect from docs.pydantic.dev/latest/api/standard_library_types/) | 2026-07-20 | official doc | WebFetch full | `Literal['apple','pumpkin']` validates exact membership; non-member → ValidationError `type=literal_error`, message "Input should be 'apple' or 'pumpkin'"; strict matching, no coercion (`'1'` rejected for `Literal[1,2]`) → FastAPI surfaces it as HTTP 422 |

## Snippet-only table (evaluated, not fetched in full)

| URL | Kind | Why not fetched |
|-----|------|-----------------|
| https://fastapi.tiangolo.com/tutorial/security/ | official doc | dependency-based auth tutorial; project uses middleware pattern — context only |
| https://github.com/fastapi/fastapi/blob/master/docs/en/docs/how-to/conditional-openapi.md | official mirror | duplicate of source 1 |
| https://github.com/fastapi/fastapi/discussions/7892 + /pull/1260 | community/GH | historical basis for docs_url=None API; superseded by source 1 |
| https://testdriven.io/tips/80107066-795c-4026-b7df-e250cdcd3dac/ | practitioner | same openapi_url=None recipe as source 1 |
| https://codewolfy.com/fastapi-configuration-disable-api-docs-swagger-ui-redoc/ | practitioner | same recipe |
| https://fastapi-utils.davidmontague.xyz/user-guide/basics/api-settings/ | practitioner lib | settings-driven docs gating variant |
| https://www.restack.io/p/fastapi-answer-disable-endpoint | aggregator | low tier |
| https://authzed.com/blog/fail-open | practitioner | fail-open/closed definitions; OWASP (source 4) outranks |
| https://thorteaches.com/glossary/fail-open/, trainingcamp.com, keysight.com, community.cisco.com, help.forcepoint.com, writer.mrmehta.in, linkedin CISSP tip | glossary/community | definition-level dupes of source 4 |
| https://github.com/NousResearch/hermes-agent/issues/10567 | community (2026) | live 2026 example of Tailscale-vs-CORS-regex problem class; snippet sufficed |
| https://www.stackhawk.com/blog/configuring-cors-in-fastapi/ | practitioner | general CORS recipe; source 2 authoritative |
| https://davidmuraya.com/blog/fastapi-cors-configuration/ + /blog/fastapi-security-guide/ | practitioner | general recipes |
| https://github.com/fastapi/fastapi/issues/4885 + /discussions/8486 | community | allow_origin_regex takes ONE regex string (not a list) — relevant limitation, snippet sufficed |
| https://stac-utils.github.io/stac-fastapi/api/stac_fastapi/api/middleware/ | practitioner lib | example only |
| https://oneuptime.com/blog/post/2026-01-25-fastapi-authentication-middleware/view | practitioner (2026) | middleware auth pipeline structure; consistent with our middleware pattern |
| https://escape.tech/blog/how-to-secure-fastapi-api/, workos.com/blog/top-authentication-solutions-fastapi-2026, docs.logto.io/api-protection/python/fastapi, medium.com (x2), github.com/VolkanSah/Securing-FastAPI-Applications, oboe.com, scribd mirror, arxiv 2510.11837, arxiv 2605.30998 | mixed | general security round-ups / off-topic tail from searches |

Unique candidate URLs collected: **40** (6 read in full, 34 snippet-only).

## Recency scan (last 2 years)

Searched 2025+2026-scoped variants (queries 1, 2 above plus `Tailscale CORS allow origin 100.64.0.0/10 regex FastAPI 2025 2026`). Findings:
1. **`regex=` → `pattern=` rename is settled** (FastAPI 0.100.0, mid-2023, with Pydantic v2): all current-window material uses `pattern=`; using `regex=` today emits deprecation warnings. Use `pattern=` only.
2. **Tailscale-behind-CORS is a live 2026 problem class** (NousResearch hermes-agent issue #10567, 2026): projects hardcode localhost regexes and then widen incorrectly; no canonical library solution has emerged — a single correct `allow_origin_regex` remains the documented approach (allow_origin_regex still accepts only ONE regex string, GH #4885/#8486 unchanged).
3. **No changes in the 2024-2026 window** to the conditional-OpenAPI recipe, CORSMiddleware credential semantics, or Pydantic Literal validation that supersede the canonical sources above. No new FastAPI-native "public paths" primitive shipped; middleware allowlists remain hand-rolled.

## Internal code audit

All claims line-anchored; read on 2026-07-20 at commit eed56025 (main).

### backend/main.py — the auth surface
- `main.py:389-394` — `app = FastAPI(title=..., description=..., version="2.0.0", lifespan=lifespan)`. NO `docs_url`/`openapi_url`/`redoc_url` kwargs → all three docs surfaces enabled with defaults (pysvc-08).
- `main.py:397-403` — `app.add_middleware(CORSMiddleware, allow_origin_regex=r"^http://(localhost|100\.\d+\.\d+\.\d+):\d+$", allow_credentials=True, allow_methods=["*"], allow_headers=["*"])`. The `100\.\d+\.\d+\.\d+` alternate matches ALL of 100.0.0.0/8, not just Tailscale CGNAT 100.64.0.0/10 (security-04). Regex is at line **399** exactly.
- `main.py:406-423` — `_PUBLIC_PATHS` tuple, **16 entries**: `/api/health`, `/api/changelog`, `/api/auth`, `/api/cost-budget` (:410), `/api/jobs/status` (:411), `/api/harness/monthly-approval` (:412), `/api/harness/demotion-audit`, `/api/harness/weekly-ledger`, `/api/harness/candidate-space`, `/api/harness/results-distribution`, `/api/signals` (:417), `/api/observability` (:418), `/api/sovereign` (:419), `/docs` (:420), `/openapi.json` (:421), `/redoc` (:422). (Step spec cites 404-421; actual anchors today are 406-423.)
- `main.py:426-437` — `auth_and_security_middleware`; gate at :435: `if request.method != "OPTIONS" and not any(path.startswith(p) for p in _PUBLIC_PATHS):` → `await get_current_user(request)` (:437). Consumption is **middleware prefix-match (startswith)**, not per-route dependency — removing a prefix instantly auths every route under it, all methods.
- `main.py:446-455` — manual 401 CORS echo: `origin.startswith("http://localhost:") or (origin.startswith("http://100.") and origin.count(".") == 3)` (:449-452) — even looser than the middleware regex (any `100.*` origin, any/no port). Sets `Access-Control-Allow-Origin: <origin>` + `Allow-Credentials: true` + `Vary: Origin` (:453-455). Step (c) wants ONE shared predicate for both sites.
- OPTIONS preflights always skip auth (:431-435 comment + condition) — correct per CORS spec (browsers never send credentials on preflight); keep this behavior.

### backend/api/auth.py — get_current_user (the enforcement point)
- `auth.py:135-220` — `async def get_current_user(request) -> Optional[dict]`.
- `auth.py:150-153` — `DEV_LOCALHOST_BYPASS=1` env + client in `(127.0.0.1, ::1, localhost)` → returns `{"email": "dev@localhost", "localhost_bypass": True}` WITHOUT touching tokens. This is the rail that keeps localhost callers (Slack bot, smoke scripts, immutable curl commands) alive after the prune — see Pitfalls.
- `auth.py:161-167` — empty `AUTH_SECRET` → 401 unless `DEV_DISABLE_AUTH=1` (fail-closed since phase-4.6.4).
- `auth.py:175-189` — token candidates: `Authorization: Bearer` header (non-sentinel) OR cookies `__Secure-authjs.session-token` / `authjs.session-token`. Cookie path works cross-port because cookies are per-host (not per-port) + `credentials:"include"`.
- `auth.py:207-213` — **gap2-03 exact site**: `if settings.allowed_emails:` → empty string skips the whole allowlist block → any Google-authenticated email passes. Comparison is `email.lower() not in allowed` with `[e.strip().lower() ...]` — case-insensitive, comma-separated.

### backend/config/settings.py
- `settings.py:20` — `debug: bool = False` (loads from `DEBUG` env; also flips JSON logging per backend-api.md).
- `settings.py:567-570` — `# --- Authentication ---` section: `auth_secret: SecretStr` (:569), `allowed_emails: str = Field("", description="... Empty = allow all authenticated users.")` (:570). New `auth_enforce_allowlist: bool = Field(False, ...)` belongs directly after :570 in this section.
- `settings.py:604-605` — `@lru_cache get_settings()` — singleton accessor; but `Settings()` is ALSO constructed ad-hoc (e.g. `monthly_approval_api.py:122`), so put the one-shot startup WARNING in `main.py::lifespan` (starts :121), not in a model_validator (would fire per instantiation).

### backend/api/monthly_approval_api.py
- `:36-39` — router prefix `/api/harness/monthly-approval`.
- `:42` — `_ALLOWED_ACTIONS = frozenset({"approved", "rejected"})`.
- `:58-59` — `class ApprovalActionBody(BaseModel): action: str` — plain str (api-design-12 target → `Literal["approved","rejected"]`).
- `:167-169` — GET `/status` already validates: `month_key: Optional[str] = Query(None, pattern=r"^\d{4}-\d{2}$")` — the POST should mirror this exact pattern via `Path(pattern=...)`.
- `:184-193` — `POST /{month_key}`: `month_key: str` bare (no Path validation); invalid action returns **HTTP 200** `MonthlyApprovalState(status="rejected", reason=f"invalid_action:...")` — same status vocabulary as a real rejection (api-design-12). After the Literal change this branch becomes dead code (422 fires first) — executor may keep it as belt-and-braces or drop it; keep `_ALLOWED_ACTIONS` in sync either way.
- `:186` — `action = (body.action or "").strip().lower()` — NOTE: `Literal` is exact-match, so " Approved " / "APPROVED" that today normalize to approved will 422 after the change. That is the intended fail-closed tightening; frontend/callers must send lowercase exact values (no repo caller sends mixed case today — the only POST caller found is external/manual).

### Consumers of the five newly-authed prefixes (grep, exhaustive)
- **frontend (all via `apiFetch` → authed)**: `frontend/src/lib/api.ts:191,195,199` (`/api/signals/...`), `:464` (`/api/observability/data-freshness`), `:740,746,750,807` (`/api/sovereign/...`). `apiFetch` sends `Authorization: Bearer <cookie|session-active>` (api.ts:66-77) AND `credentials: "include"` (api.ts:87) → these keep working once authed. Components (sovereign/page.tsx, AltDataPanel, ComputeCostBreakdown, RedLineMonitor) all go through these api.ts helpers — no direct fetch/EventSource to the five prefixes found in frontend/src.
- **`/api/cost-budget` and `/api/harness/monthly-approval`: ZERO frontend callers** (grep `cost-budget|monthly-approval` over frontend/src returns nothing). The step-spec phrase "frontend already sends the NextAuth cookie via apiFetch" is about the apiFetch mechanism being auth-ready; there is no monthly-approval UI today — POST callers are operator curl / Slack-side manual.
- **backend/slack_bot**: hits ONLY `/api/health`, `/api/analysis`, `/api/paper-trading`, `/api/reports`, `/api/backtest`, `/api/jobs/*` (commands.py:22,75,159,166,191,212; direct_responder.py:102,161; scheduler.py:85,90,146-693). **NONE of the five prefixes being pruned.** No `Authorization` header on any localhost call (only slack_bot/mcp_tools.py:83 sends Bearer — that is a Slack token, not backend auth) — confirming docs-07: the bot already relies on a localhost bypass for the authed endpoints it hits today (`/api/paper-trading/*` is NOT in `_PUBLIC_PATHS`), so the prune does not change the bot's situation.
- **scripts/**: `scripts/go_live_drills/smoke_test_4_17_6.py:63,71,81` — unauthenticated `urllib` GETs to `/api/signals/{t}/alt-data`, `/api/signals/macro/indicators`, `/api/sovereign/leaderboard` against `http://localhost:8000` — will 401 after the prune UNLESS `DEV_LOCALHOST_BYPASS=1` is active on the backend process.
- **.claude/masterplan.json (immutable verification commands)**: 18 mentions of the five prefixes; unauthenticated curls include `/api/sovereign/red-line|leaderboard|compute-cost` and `curl -sS http://127.0.0.1:8000/api/observability/freshness && curl -sS http://127.0.0.1:8000/api/cost-budget/status`. These criteria are IMMUTABLE — they only keep re-running green if the localhost bypass is active (they curl 127.0.0.1, which qualifies for auth.py:150-153).

### .claude/rules/security.md (doc drift)
- "Auth middleware skips: `/api/health`, `/api/auth`, `/docs`, `/openapi.json`, `/redoc`" — 5 documented vs 16 actual (security-03). Step (b) rewrites this to the exact post-prune list with per-prefix justification.

### Audit register cross-check (handoff/current/audit_phase75/)
- `register.md:32` — security-01 row (P1). Full entries live in `confirmed_findings.json` (security-01, security-03, security-04, gap2-03, api-design-12, pysvc-08, docs-07 all present and match the anchors above; docs-07 additionally flags `.claude/rules/backend-slack-bot.md:26` "No auth tokens (internal Docker network)" as dead guidance — single-Mac deployment, `backend:8000` hostname vestigial at scheduler.py:79).

## Key findings

1. **Disabling docs is obscurity, not security — do BOTH halves of step (b).** FastAPI's own docs: hiding docs "shouldn't be the way to protect your API... a form of Security through obscurity" (source 1). The step is correct precisely because it pairs `openapi_url=None` (prod) with removing `/docs`,`/openapi.json`,`/redoc` from `_PUBLIC_PATHS` (auth when enabled in debug).
2. **`openapi_url=None` cascades**: it 404s `/docs` and `/redoc` too (source 1). Setting all three kwargs explicitly is still clearer and covers the debug=True case where openapi stays on.
3. **CGNAT block is exactly second-octet 64–127** (source 3, RFC 6598). The step's regex `100\.(6[4-9]|[7-9]\d|1[01]\d|12[0-7])\.\d+\.\d+` was verified locally: octet sweep 0-255 accepts exactly {64..127}; boundary origins 100.63.x/100.128.x rejected, 100.64.0.1/100.127.255.254 accepted (python re test, this session).
4. **Fail-closed doctrine** (source 4): authorization checks "return false if there is an exception"; a missing/empty config that silently disables a check (auth.py:209 `if settings.allowed_emails:`) is the canonical fail-open anti-pattern. The DARK-flag design (default-OFF `auth_enforce_allowlist`) is the sanctioned migration path: byte-identical behavior now, operator flips to fail-closed later.
5. **`Path(pattern=...)` is the current API** (source 5): `pattern: str|None`; `regex=` deprecated since FastAPI 0.100.0. Mirrors the GET's existing `Query(None, pattern=r"^\d{4}-\d{2}$")` at monthly_approval_api.py:169.
6. **`Literal` yields 422 with `type=literal_error`** and exact strict matching, no coercion (source 6) — replacing the HTTP-200-"rejected" degrade branch with a real validation failure.
7. **Wildcards + credentials violates the documented CORS contract** (source 2): with `allow_credentials=True`, methods/headers should be explicit lists, not `["*"]`. Out of step (c)'s strict scope but worth a line in the contract as a noted deviation or cheap add-on.

## Consensus vs debate (external)

Consensus: settings-driven conditional OpenAPI; fail-closed as default posture for auth controls; pattern= over regex=; CORS-with-credentials wants explicit specification. Debate: none material — community glossaries agree with OWASP on fail-open/closed; no source argues for keeping docs public on an authed API.

## Pitfalls (the ones that will actually bite)

1. **Immutable masterplan verification commands curl three of the five prefixes unauthenticated** (`.claude/masterplan.json`, 18 mentions): `/api/sovereign/red-line|leaderboard|compute-cost`, `/api/observability/freshness`, `/api/cost-budget/status` via bare `curl http://127.0.0.1:8000/...`. These criteria are immutable. They only stay re-runnable if the **localhost bypass** (auth.py:150-153, `DEV_LOCALHOST_BYPASS=1` + client 127.0.0.1) is active on the operator's backend. Executor CANNOT edit `.env` (step boundary) — so the experiment_results/live_check MUST record whether the bypass is active (curl an authed endpoint tokenless from localhost and record 200-with-bypass vs 401) and state the consequence. Slack bot already depends on the same bypass for `/api/paper-trading/*` (not public today), so the prune adds no NEW bot breakage — but do not claim "nothing breaks" without the live probe.
2. **`smoke_test_4_17_6.py` will 401 without the bypass** (scripts/go_live_drills/smoke_test_4_17_6.py:63,71,81 — tokenless urllib to `/api/signals/*`, `/api/sovereign/leaderboard`). Same mitigation; flag in results.
3. **`/api/jobs/status` vs `/api/jobs/heartbeat`**: only `/status` is public (main.py:411); scheduler's heartbeat POST (scheduler.py:85) already relies on the bypass. Don't "fix" this in 75.1; it's evidence the bypass is load-bearing today.
4. **Startup WARNING placement**: `Settings` is instantiated outside the lru_cache singleton too (monthly_approval_api.py:122) — a model_validator would multi-fire. Emit the empty-allowlist WARNING once in `main.py::lifespan` (begins :121) via `logging.getLogger(...).warning(...)`, ASCII-only per `.claude/rules/security.md` logging rule.
5. **Literal is strict**: today `" APPROVED ".strip().lower()` would pass; after `Literal["approved","rejected"]` it 422s. No repo caller sends mixed case (grep: no POST callers at all) — state this in the contract so the tightening is a recorded intent, not a surprise.
6. **The 401-echo predicate becomes stricter when shared**: current echo accepts any `http://100.*` origin with any/no port (main.py:449-452); the shared regex requires `:port` and CGNAT range. Real browser origins always carry the explicit port here (frontend :3000), so this is safe — but compile the regex ONCE at module level and use `regex.match(origin)` in both places so the two sites can never drift again.
7. **`\d+\.\d+` tail octets accept >255 values** (e.g. `100.64.999.1`) — harmless for browser-enforced CORS (browsers never emit such origins; non-browser clients ignore CORS entirely), so do not gold-plate the regex beyond the step spec.
8. **Docs gating changes local dev UX**: with `debug: bool = False` default (settings.py:20) and no `.env` edit, `/docs` 404s on the operator's machine after restart. That is the step's intent ("None in prod default") — but say it loudly in experiment_results so the operator knows `DEBUG=true` re-enables (then behind auth via cookie, which works cross-port since cookies are per-host).
9. **`/api/auth` must STAY public** — it is not in the removal set; `get_current_user` depends on nothing under it, but frontend session bootstrap does. Similarly `/api/health` (Slack bot + watchdogs) and `/api/jobs/status`, `/api/changelog`, and the four read-only harness dashboards (`demotion-audit`, `weekly-ledger`, `candidate-space`, `results-distribution`) are NOT in 75.1's removal scope — touching them is scope creep.

## Application to pyfinagent (exact shapes for the executor)

- **(a)+(b) prune**: delete lines for `/api/harness/monthly-approval` (:412), `/api/cost-budget` (:410), `/api/signals` (:417), `/api/observability` (:418), `/api/sovereign` (:419), `/docs` (:420), `/openapi.json` (:421), `/redoc` (:422) from `_PUBLIC_PATHS` (main.py:406-423). Remaining public: `/api/health`, `/api/changelog`, `/api/auth`, `/api/jobs/status`, 4x read-only harness dashboards.
- **(b) docs gating** (main.py:389-394): `_s = get_settings()` then `app = FastAPI(..., docs_url="/docs" if _s.debug else None, redoc_url="/redoc" if _s.debug else None, openapi_url="/openapi.json" if _s.debug else None)` — source-1 pattern, settings-driven.
- **(c) one predicate, two sites**: module-level `_TAILSCALE_ORIGIN_RE = re.compile(r"^http://(localhost|100\.(6[4-9]|[7-9]\d|1[01]\d|12[0-7])\.\d+\.\d+):\d+$")`; pass `_TAILSCALE_ORIGIN_RE.pattern` as `allow_origin_regex` (main.py:399) and replace the startswith checks at :449-452 with `_TAILSCALE_ORIGIN_RE.match(origin)`.
- **(d) DARK flag**: `auth_enforce_allowlist: bool = Field(False, description="phase-75.1: True makes an EMPTY allowed_emails reject-all (fail-closed). Default False preserves fail-open legacy. Operator flips via AUTH_ENFORCE_ALLOWLIST env.")` after settings.py:570. In auth.py replace :209-213 with: parse `allowed` first; `if not allowed and settings.auth_enforce_allowlist: raise HTTPException(401)`; `if allowed and email.lower() not in allowed: raise ...` (keeps today's behavior byte-identical when flag False). Lifespan WARNING when `not allowed_emails`: "ALLOWED_EMAILS is empty -- any Google-authenticated user is admitted (auth_enforce_allowlist=False)".
- **(e) POST validation** (monthly_approval_api.py:184-185): `month_key: str = Path(pattern=r"^\d{4}-\d{2}$")` (use `from fastapi import Path`; Annotated form fine) + `action: Literal["approved", "rejected"]` on ApprovalActionBody (:58-59). GET already has the pattern (:169) — POST mirrors it.
- **(b) doc rewrite**: `.claude/rules/security.md` public list → exact 8-entry post-prune list with one-line justification each (health=probes/watchdog; changelog=public page; auth=session bootstrap; jobs/status=bot poll; 4 harness dashboards=read-only telemetry consumed by Harness tab pre-auth).
- **Consumer fixes**: frontend needs NO changes (all five prefixes' callers already go through authed `apiFetch`; cost-budget + monthly-approval have zero frontend callers). Slack bot hits none of the five. `smoke_test_4_17_6.py` + immutable curls: document the `DEV_LOCALHOST_BYPASS` dependency (pitfall 1-2) rather than editing immutable criteria.

## Research Gate Checklist

- [x] >=5 authoritative external sources READ IN FULL via WebFetch (6; all official docs / OWASP / vendor)
- [x] 10+ unique URLs total (40)
- [x] Recency scan (last 2 years) performed + reported (3 findings)
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim
- [x] Internal exploration covered every relevant module (main.py, auth.py, settings.py, monthly_approval_api.py, api.ts, slack_bot/*, scripts, masterplan, rules, register)
- [x] Contradictions / consensus noted
- [x] All claims cited per-claim

## JSON envelope

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 6,
  "snippet_only_sources": 34,
  "urls_collected": 40,
  "recency_scan_performed": true,
  "internal_files_inspected": 16,
  "coverage": {
    "audit_class": false,
    "rounds": 1,
    "dry_rounds": 0,
    "K_required": 2,
    "new_findings_last_round": 0,
    "dry": false
  },
  "summary": "All six audit findings verified at exact anchors: _PUBLIC_PATHS 16 entries (main.py:406-423), /8-wide CORS regex (:399) + looser 401 echo (:449-455), fail-open empty allowlist (auth.py:209), unvalidated POST month_key/action (monthly_approval_api.py:58,185). Official patterns confirmed: settings-driven conditional OpenAPI (openapi_url=None cascades to /docs+/redoc; obscurity caveat means the _PUBLIC_PATHS prune is the real security), Path(pattern=...) not regex=, Literal -> 422 literal_error, OWASP fail-closed. Proposed CGNAT regex locally verified == {64..127} second octet. Biggest execution risk is NOT code: immutable masterplan curls + smoke script + Slack bot depend on tokenless localhost calls; they survive only via the DEV_LOCALHOST_BYPASS rail (auth.py:150-153), which the executor must live-probe and document, never assume. Frontend needs zero changes (apiFetch sends Bearer + credentials:include; cost-budget/monthly-approval have no frontend callers). DARK flag auth_enforce_allowlist (default False) after settings.py:570; WARNING in lifespan not a validator (Settings() multi-instantiated).",
  "brief_path": "handoff/current/research_brief_75.1.md",
  "gate_passed": true
}
```
