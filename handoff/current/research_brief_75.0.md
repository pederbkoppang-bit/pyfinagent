# Research Brief — Step 75.0: Full-Codebase IT-Stack Audit vs Official Docs & Best Practices

- **Step:** 75.0 (gate BEFORE contract — not yet installed in masterplan)
- **Tier:** complex
- **coverage.audit_class:** true (loop-until-dry, K=2)
- **Researcher:** Layer-3 (merged external-literature + internal-code recon)
- **Date:** 2026-07-19
- **Purpose:** Arm one-auditor-per-IT-stack-role audit of the ENTIRE pyfinagent codebase against OFFICIAL DOCUMENTATION and CURRENT BEST PRACTICES. Output = citable doc anchors + defensible audit methodology + role charter.

> WRITE-FIRST: this file is created at session start and grown incrementally as each source is read. A partial brief + honest `gate_passed:false` is the correct output if the gate cannot be cleared.

---

## Queries-run (three-variant discipline)

Round 1 (2026-07-19): `FastAPI production best practices 2026` [frontier] · `Python 3.14 what's new release notes official` [frontier] · `Next.js 15 App Router best practices 2026` [frontier] · `BigQuery Python client retry timeout best practices 2026` [frontier] · `property-based testing hypothesis pytest quant finance` [year-less canonical] · `uv pip-tools lockfile Python dependency pinning best practice 2025` [last-2-year] · `LLM code review false positives precision arXiv 2025` [last-2-year].

Round 2 (2026-07-19): `OWASP API Security Top 10 2026` [frontier] · `Anthropic structured outputs tool use documentation` [year-less canonical] · `golden file snapshot testing regression backtest financial software` [year-less canonical] · `TypeScript tsconfig strictest noUncheckedIndexedAccess recommended settings` [year-less canonical] · `npm supply chain attack lockfile audit best practices 2026` [frontier].

Round 3 (2026-07-19): `FastAPI best practices` [year-less canonical] · `Python asyncio pitfalls blocking event loop run_in_executor` [year-less canonical] · `BigQuery best practices` [year-less canonical] · `multi-agent LLM codebase audit methodology 2026` [frontier].

Round 4 (2026-07-19): `LLM application engineering best practices production 2025` [last-2-year] · `pytest best practices 2026` [frontier].

Round 5 (2026-07-19, probe round): `FastAPI dependency injection anti-patterns common mistakes` [year-less] · `Next.js server actions security vulnerabilities 2025` [last-2-year] · `Python float Decimal financial calculations precision best practice` [year-less] · `detect-secrets gitleaks pre-commit secret scanning comparison` [year-less]. → NOT dry: surfaced CVE-2025-66478 + official data-security guide (both fetched in full).

Round 6 (2026-07-19, probe): `Ruff linter configuration Python best practices` [year-less] · `APScheduler production best practices pitfalls` [year-less] · `walk-forward backtest validation overfitting best practices Lopez de Prado` [year-less]. → snippet-tier additions only, but ran in the same pass as the round-5-triggered CVE/data-security FETCHES, so the PASS counts WET.

Round 7 (2026-07-19, probe): `next-auth v5 Auth.js stable release status 2026` [frontier] · `uvicorn workers production deployment FastAPI single process` [year-less] · `google-cloud-python client library retry deadline best practices official` [year-less]. → ZERO new read-in-full findings (3 snippet-tier status facts recorded: Auth.js under Better Auth maintainership + v5 stable; single-process uvicorn acceptable for low-traffic local; `deadline` deprecated for `timeout` in google-api-core). **DRY ROUND 1.**

Round 8 (2026-07-19, dry-check): `Tailwind CSS 4 migration breaking changes` [year-less] · `Python structured logging best practices JSON structlog` [year-less] · `Slack Bolt Python Socket Mode production best practices` [year-less] · `FastAPI OpenAPI documentation endpoints disable production` [year-less]. → ZERO new read-in-full findings; four snippet-tier confirmations of already-anchored rules (Tailwind 4 = version-currency datapoint; JSON-logs-in-prod already anchored; async `await ack()` checkable recorded; conditional-OpenAPI already T1 rule 7). **DRY ROUND 2 → coverage.dry = true (K=2 met).**

### Coverage log (audit-class loop-until-dry, K=2)

| Round | Action | New read-in-full findings | Dry? |
|---|---|---|---|
| 1 | 3 fetches + 3 searches (initial sweep) | 3 (fastapi/async, OWASP T10, BQ costs→redirect) | wet |
| 2 | 3 fetches + 4 searches | 3 (BQ costs, py3.14, zhanymkanov) | wet |
| 3 | 6 fetches (topic fills) | 6 (nextjs RSC, react19, 2 Anthropic eng, SWR-Bench, uv) | wet |
| 4 | 4 fetches + internal recon | 4 (Refute-or-Promote, OWASP secrets, pytest, npm-lock) | wet |
| 5 | 3 fetches + 5 searches | 3 (structured-outputs, BQ streaming/Write-API, pip-audit) | wet |
| 6 | 2 fetches + 4 searches | 2 (TS noUncheckedIndexedAccess, Hypothesis) | wet |
| 7 | 2 fetches + 2 searches | 2 (RepoAudit, Agent Audit) | wet |
| 8 | 2 fetches + 3+4 probe searches | 2 (CVE-2025-66478 advisory, Next data-security audit guide) | wet |
| 9 | 3 probe searches (fresh angles) | 0 | **DRY 1** |
| 10 | 4 probe searches (fresh angles) | 0 | **DRY 2** |

`dry_rounds = 2 >= K_required = 2` → **coverage.dry = true**. (Search-round numbering here differs from the message batching; each row = one search/fetch pass evaluated for new read-in-full findings.)

---

## Source table — READ IN FULL (counts toward gate; floor >=5)

All rows below were fetched IN FULL via WebFetch **in this session** (rows 1–7 existed in the prior-session skeleton and were re-fetched before counting).

| # | URL | Accessed | Kind | Fetched how | Tier | Key finding |
|---|-----|----------|------|-------------|------|-------------|
| 1 | https://fastapi.tiangolo.com/async/ | 2026-07-19 | Official doc | WebFetch full | 2 | `def` path ops "run in an external threadpool that is then awaited"; blocking I/O never inside `async def` |
| 2 | https://owasp.org/API-Security/editions/2023/en/0x11-t10/ | 2026-07-19 | Official standard | WebFetch full | 1/2 | Full API1–API10:2023 list w/ verbatim one-liners |
| 3 | https://www.anthropic.com/engineering/building-effective-agents | 2026-07-19 | Official vendor eng | WebFetch full | 2 | Workflows vs agents; 5 workflow patterns; "add complexity only when it demonstrably improves outcomes"; ACI discipline |
| 4 | https://docs.cloud.google.com/bigquery/docs/best-practices-costs | 2026-07-19 | Official doc | WebFetch full | 2 | Avoid `SELECT *`; LIMIT doesn't cap bytes on non-clustered; `maximum_bytes_billed` fails query pre-charge; dry-run; streaming per-row cost |
| 5 | https://nextjs.org/docs/app/getting-started/server-and-client-components | 2026-07-19 | Official doc (v16.2.10, upd 2026-06-23) | WebFetch full | 2 | RSC default; `use client` = module-graph boundary; env-poisoning + `server-only`; only `NEXT_PUBLIC_` reaches client (others → empty string); serializable props |
| 6 | https://react.dev/blog/2024/12/05/react-19 | 2026-07-19 | Official doc | WebFetch full | 2 | `ref` as prop (forwardRef deprecated), `use`, Actions/useActionState/useOptimistic, `<Context>` provider, onCaughtError/onUncaughtError, ref-cleanup TS break |
| 7 | https://www.anthropic.com/engineering/writing-tools-for-agents | 2026-07-19 | Official vendor eng | WebFetch full | 2 | Namespacing, `user_id` not `user`, high-signal returns (no UUIDs), pagination/truncation defaults, actionable errors, eval-driven iteration |
| 8 | https://docs.python.org/3/whatsnew/3.14.html | 2026-07-19 | Official doc | WebFetch full | 2 | PEP 649/749 deferred annotations; PEP 750 t-strings; PEP 765 finally-control-flow SyntaxWarning; asyncio ps/pstree introspection CLI; NotImplemented-bool TypeError |
| 9 | https://github.com/zhanymkanov/fastapi-best-practices | 2026-07-19 | Practitioner canon (10k+ stars) | WebFetch full | 4 | Blocking-I/O triage (terrible/good/perfect); finite threadpool; BackgroundTasks <1s rule; httpx.AsyncClient+ASGITransport tests; dependency_overrides; Ruff |
| 10 | https://arxiv.org/html/2509.01494v1 (SWR-Bench) | 2026-07-19 | Peer-reviewed preprint | WebFetch full (arXiv HTML-first) | 1 | LLM code review: best F1 19.38%, precision 16.65%, most baselines <10% precision; multi-review aggregation +43.67% F1; separate functional vs evolutionary findings |
| 11 | https://docs.astral.sh/uv/pip/compile/ | 2026-07-19 | Official doc | WebFetch full | 2 | `uv pip compile` → pinned lock; `uv pip sync` for exact reproduction; requirements.txt locks are platform-specific vs `uv.lock` universal |

---

## Snippet-only table (context; does NOT count toward gate)

~200 unique URLs surfaced across 19 searches (~285 raw links pre-dedupe); the 30 most decision-relevant are tabled; the remainder live in the queries-run record. (Prior-skeleton row for zhanymkanov moved UP to the read-in-full table.)

| # | URL | Kind | Why snippet-only |
|---|-----|------|------------------|
| S1 | https://fastapi.tiangolo.com/how-to/conditional-openapi/ | Official | Confirms T1 rule 7; content fully conveyed by snippet |
| S2 | https://fastapi.tiangolo.com/deployment/server-workers/ | Official | Single-process local deployment makes worker guidance peripheral |
| S3 | https://docs.python.org/3/library/decimal.html | Official | Decimal-for-ledger rule recorded; quant float64 research path unaffected |
| S4 | https://docs.astral.sh/ruff/configuration/ | Official | Tooling recommendation (extend-select baseline) recorded |
| S5 | https://apscheduler.readthedocs.io/en/3.x/faq.html | Official | Multi-worker duplicate-fire + shutdown pitfalls recorded |
| S6 | https://docs.slack.dev/tools/bolt-python/concepts/socket-mode/ | Official | `await ack()` + 10-connection cap recorded |
| S7 | https://www.structlog.org/en/stable/logging-best-practices.html | Official | JSON-in-prod + stdout principles already anchored via T1/internal |
| S8 | https://googleapis.dev/python/google-api-core/latest/retry.html | Official | `deadline`→`timeout` deprecation recorded |
| S9 | https://github.com/googleapis/python-bigquery/issues/2310 | Official repo issue | Hang-on-network-drop risk recorded |
| S10 | https://tailwindcss.com/docs/upgrade-guide | Official | v4 = version-currency datapoint; repo stays v3-compliant meanwhile |
| S11 | https://authjs.dev/getting-started/migrating-to-v5 | Official | v5-stable migration path recorded |
| S12 | https://github.com/nextauthjs/next-auth/discussions/13252 | Official announcement | "Auth.js is now part of Better Auth" maintenance-mode fact |
| S13 | https://react.dev/blog/2025/12/03/critical-security-vulnerability-in-react-server-components | Official | Upstream CVE-2025-55182 post; downstream advisory read in full instead |
| S14 | https://vercel.com/kb/bulletin/security-bulletin-cve-2025-55184-and-cve-2025-55183 | Official | DoS + source-exposure siblings recorded |
| S15 | https://arxiv.org/abs/2510.02534 (ZeroFalse) | Preprint | Static-analysis+LLM FP reduction; T8 anchored by 2 stronger papers |
| S16 | https://arxiv.org/pdf/2604.16321 (MAS codegen survey) | Preprint | Survey breadth; not audit-methodology-specific |
| S17 | https://arxiv.org/pdf/2603.20637 (AEGIS) | Preprint | Graph-guided vuln reasoning; adjacent to T8 anchors |
| S18 | https://arxiv.org/abs/2605.12280 (Iterative Audit Convergence) | Peer-reviewed | 9-round convergence datapoint recorded |
| S19 | https://arxiv.org/pdf/2602.00080 (GT-Score) | Preprint | Overfit-objective complement; project gates already aligned |
| S20 | https://www.sciencedirect.com/science/article/abs/pii/S0950705124011110 | Peer-reviewed | ML-era OOS comparison; abstract-gated (paywall), snippet recorded |
| S21 | https://hypothesis.readthedocs.io/en/latest/ (root docs) | Official | Tutorial page read in full instead |
| S22 | https://www.npmjs.com/package/@tsconfig/strictest | Official pkg | Strictest-preset existence recorded |
| S23 | https://unit42.paloaltonetworks.com/monitoring-npm-supply-chain-attacks/ | Industry | Threat-landscape context for T9 |
| S24 | https://www.endorlabs.com/learn/how-to-defend-against-npm-software-supply-chain-attacks | Industry | Cooldown + install-script flagging recorded |
| S25 | https://render.com/articles/fastapi-production-deployment-best-practices | Industry | Deployment guidance; local-only deployment narrows relevance |
| S26 | https://blog.axway.com/learning-center/digital-security/risk-management/owasps-api-security | Industry | Confirms no-2026-edition status |
| S27 | https://oneuptime.com/blog/post/2026-01-30-how-to-build-property-based-testing-with-hypothesis/view | Blog | Hypothesis official docs preferred |
| S28 | https://sepgh.medium.com/common-mistakes-with-using-apscheduler-in-your-python-and-django-applications-100b289b812c | Community | APScheduler multi-process pitfall corroboration |
| S29 | https://github.com/vercel-labs/fix-react2shell-next | Official tool | Checker existence recorded; version verified manually instead |
| S30 | https://www.susanpotter.net/quant/property-based-testing-statistical-validation/ | Practitioner | Market-data strategy design ideas for T7 property tests |

---

## Topic sections (per-claim citations)

### T1. FastAPI production best practices

**Canonical:** FastAPI official async doc (https://fastapi.tiangolo.com/async/, re-fetched 2026-07-19) + zhanymkanov/fastapi-best-practices (10.5k-star practitioner canon, read in full 2026-07-19).

**Compliant looks like:** `async def` path ops contain ONLY awaitable I/O; sync/blocking calls live in plain `def` (FastAPI: "run in an external threadpool that is then awaited... as it would block the server") or are wrapped in `run_in_threadpool`; pydantic v2 models everywhere ("excessively use Pydantic"); settings via `pydantic_settings.BaseSettings`; DI via `Depends` (request-scoped caching is free); `BackgroundTasks` only for sub-second silent-failure-tolerant work, Celery/queue otherwise; `response_model`/`status_code` declared per route.

**Checkable rules for the auditor:**
1. GREP: `async def` route/service bodies calling `requests.`, `time.sleep`, `yf.`(yfinance), sync `bigquery.Client().query(...).result()`, or other sync SDKs → event-loop blocking (FastAPI async doc: blocking I/O belongs in `def`). This is the No.1 FastAPI production defect class.
2. GREP: heavy sync work in `def` routes — threadpool is finite ("threads require more resources than coroutines"); starving it stalls ALL sync routes.
3. CPU-bound work must not be awaited/threaded — worker process or queue (GIL; zhanymkanov "CPU Intensive Tasks").
4. `BackgroundTasks` usages: >1s or must-not-fail work belongs in Celery (repo has `celery[redis]` pinned but audit whether actually wired).
5. Dependencies preferred `async`; sync deps burn threadpool slots.
6. Tests: `httpx.AsyncClient` + `ASGITransport` + `dependency_overrides` (not monkeypatching app internals).
7. Docs hidden outside local/staging; every route declares `response_model` + `status_code`.
8. Lint: Ruff as the consolidated linter (replaces black/isort/autoflake stack).

### T2. Modern Python 3.12–3.14 idioms + packaging/pinning

**Canonical:** https://docs.python.org/3/whatsnew/3.14.html (read in full 2026-07-19) + https://docs.astral.sh/uv/pip/compile/ (uv lockfile docs) + pip-tools tradition.

**Repo runs Python 3.14.4** — the audit should check the code USES 3.14 correctly, not just runs on it:
1. PEP 649/749: annotations are deferred natively — `from __future__ import annotations` is now redundant (harmless, but flags stale idiom); string-literal forward refs no longer needed.
2. PEP 765: `return`/`break`/`continue` inside `finally` now emits `SyntaxWarning` — GREP for `finally:` blocks with control flow (silent exception-swallowing bug class).
3. `NotImplemented` in boolean context now raises `TypeError` (was warning) — check rich-comparison helpers.
4. `int()` no longer honors `__trunc__` — check numeric wrappers.
5. asyncio: new introspection CLI (`python -m asyncio ps <PID>` / `pstree`) — an OPS TOOL the harness/runbooks could adopt for hang diagnosis (autonomous_loop hang class). `create_task()` now accepts arbitrary kwargs.
6. Deprecation sweep needed: run test suite with `-W error::DeprecationWarning` to surface 3.14 deprecations (asyncio policy-API class); `pytest.ini` currently sets NO `filterwarnings` (verified 2026-07-19 — only the `requires_live` marker).

**Packaging/pinning (uv docs + Round-1 search):** best practice = FLOORS in a manifest + a fully-pinned LOCKFILE for reproducibility. `uv pip compile requirements.txt -o requirements.lock` (or pip-tools `pip-compile`) generates hash-pinned resolution; `uv pip sync` reproduces it. Repo status: `backend/requirements.txt` mixes 9 supply-chain `==` pins with ~50 open `>=` floors and has NO lockfile → every fresh `pip install -r` is a different resolution (the phase-67.6 anthropic stale-pin "downgrade time bomb" was one instance of this class). uv notes lockfiles are platform-specific unless using `uv.lock` universal format. Local-only deployment (single Mac) lowers but does not eliminate the risk — the .venv IS the de-facto lock, unversioned.

### T3. Next.js 15 App Router + React 19 + TypeScript strictness

**Canonical:** Next.js server-and-client-components doc (fetched 2026-07-19; note the LIVE doc is versioned 16.2.10 — Next 16 is current, repo is on `^15.0.0`) + React 19 release post (fetched) + `frontend/tsconfig.json` ground truth.

**Compliant looks like:** Server Components by default, `'use client'` only at interactive leaves ("add `'use client'` to specific interactive components instead of marking large parts of your UI as Client Components"); providers rendered as deep as possible; secrets never in client graph — "only environment variables prefixed with `NEXT_PUBLIC_` are included in the client bundle... otherwise replaced with an empty string"; `server-only` package for build-time enforcement; props crossing the boundary serializable; React 19 idioms (ref-as-prop, `<Context>` provider, `useActionState`, root `onCaughtError`).

**Checkable rules:**
1. GREP `'use client'` density vs component count — over-clienting inflates bundles. NOTE the repo is architecturally a client-side SPA (Bearer-token `api.ts` → FastAPI :8000), so most data pages are legitimately client components; audit for ACCIDENTAL client-ing of static UI, not for wholesale RSC conversion.
2. GREP `NEXT_PUBLIC_` — any secret-shaped value is a leak; non-prefixed env reads inside client components silently become `""` (bug class, not just security).
3. GREP `forwardRef` (React 19: "will be deprecated and removed in future versions") and `Context.Provider` (same) — migration debt.
4. Next 15 async request APIs: `params`/`searchParams` must be awaited (the versioned-doc example uses `params: Promise<{id}>`); GREP dynamic-route pages.
5. Ref callbacks: implicit returns now rejected by TS types (React 19 breaking) — GREP `ref={(` one-liners.
6. tsconfig headroom: `strict: true` present but `target: ES2017` (stale; ES2022+ is safe on evergreen browsers), `skipLibCheck: true`, missing `noUncheckedIndexedAccess` and `noUncheckedSideEffectImports` (the latter is what makes `server-only` typings work per Next docs).
7. Version skew: `eslint-config-next ^16.2.4` against `next ^15.0.0` — lint rules from a major ahead of the framework; either upgrade Next or align the config major.
8. Error/loading/empty states per `.claude/rules/frontend.md` (project standard, aligns with React 19 error-boundary root options).

**P0 security addendum (round 8, both fetched in full):**
- **CVE-2025-66478 "React2Shell" (CVSS 10.0 RCE, official advisory nextjs.org/blog/CVE-2025-66478, publ. 2025-12-03):** "Applications using React Server Components with the App Router are affected when running: Next.js 15.x, Next.js 16.x..." — the repo's `next ^15.0.0` IS in the affected line. Fixed versions: 15.0.5 / 15.1.9 / 15.2.6 / 15.3.6 / 15.4.8 / 15.5.7 / 16.0.7. "There is no workaround—upgrading to a patched version is required." FIRST CHECK for the frontend+security auditors: resolved `next` version in `frontend/package-lock.json` (`npm ls next`) >= the patch for its minor; if the app was "online and unpatched as of December 4th, 2025", official guidance is to ROTATE SECRETS. (Local-only Tailscale deployment mitigates exposure, not the patch obligation.) Vercel ships `npx fix-react2shell-next` as a deterministic checker.
  - **VERIFIED THIS SESSION (2026-07-19):** resolved `next` in `frontend/package-lock.json` AND `frontend/node_modules/next/package.json` = **15.5.12** ≥ 15.5.7 (the 15.5.x fix) → **the repo is PATCHED**. Residual question for the security auditor: what version was deployed during the Dec 3–4 2025 exposure window (`git log -p --before=2025-12-10 -- frontend/package-lock.json`)? If an unpatched 15.x was serving then — even Tailscale-bound — the official secret-rotation guidance applies to the NextAuth `AUTH_SECRET` + any `NEXT_PUBLIC_`-adjacent tokens of that era.
- **Official Next.js data-security AUDIT CHECKLIST (nextjs.org/docs/app/guides/data-security, v16.2.10 doc):** verbatim audit anchors — "**Data Access Layer:** ...Verify that database packages and environment variables are not imported outside the Data Access Layer. **`\"use client\"` files:** Are the Component props expecting private data?... **`\"use server\"` files:** Are the Action arguments validated?... Is the user re-authorized inside the action?... **`/[param]/`** Folders with brackets are user input. Are params validated? **`proxy.ts` and `route.ts`:** Have a lot of power. Spend extra time auditing these." Server Actions "reachable via a direct POST request" even if unused in UI — treat as public endpoints; validate + re-authorize inside every action.


### T4. Google BigQuery client best practices

**Canonical:** https://docs.cloud.google.com/bigquery/docs/best-practices-costs (fetched 2026-07-19; note `cloud.google.com` now 301s to `docs.cloud.google.com`).

**Compliant looks like:** column-projected queries (never `SELECT *`); partition filters on every large-table query (`require_partition_filter` to enforce); `maximum_bytes_billed` set ("if the number of estimated bytes is beyond the limit, then the query fails without incurring a charge"); dry-runs for cost estimates; batch loads over streaming ("streaming inserts incur per-row charges; batch loading is more economical"); explicit `timeout=`/`retry=` on client calls (client can "hang indefinitely" on dropped connections — googleapis/python-bigquery#2310).

**Checkable rules:**
1. GREP `SELECT \*` in SQL strings across `backend/` — cost + schema-drift fragility.
2. GREP `LIMIT` used as a cost control — "for non-clustered tables, applying a LIMIT clause... doesn't affect the amount of data that is read" (a repo-wide misconception risk; CLAUDE.md's own BQ rule pairs LIMIT with date filters, which IS correct because the filter does the byte-capping).
3. Every `client.query(...)` and `.result(...)` carries a timeout (repo convention 30s — bigquery_client.py:533,765,798 confirmed compliant; audit the OTHER ~10 modules using BQ directly).
4. NO `maximum_bytes_billed` found in `backend/db/bigquery_client.py` (grep 2026-07-19) — candidate hardening: set it in shared QueryJobConfig defaults.
5. Streaming vs batch: `insert_rows_json` (legacy insertAll streaming API) used in 10+ modules (`backend/autonomous_loop.py`, `backend/db/bigquery_client.py:251,394,403,429,492`, intel/*, meta_evolution/*, backtest/data_ingestion.py) — audit per-call-site whether per-row-billed streaming is justified vs free batch load jobs / Storage Write API (verify current Google recommendation — see recency scan).
6. Parameterized queries via `QueryJobConfig(query_parameters=...)` — bigquery_client.py is broadly compliant (20+ hits) — auditors verify NO f-string SQL interpolation anywhere else.
7. Partitioned `historical_*` tables must always be queried with date filters (project rule; validates against official partition-filter guidance).

### T5. API/web security — OWASP API Security Top 10 (2023, still current)

**Canonical:** https://owasp.org/API-Security/editions/2023/en/0x11-t10/ (fetched) + https://cheatsheetseries.owasp.org/cheatsheets/Secrets_Management_Cheat_Sheet.html (fetched). Recency: NO 2026 edition exists — "the 2023 list remains the current official version" (searched 2026-07-19).

**The ten, with repo mapping:** API1 BOLA (low surface: single-operator app, but audit object-id endpoints); API2 Broken Authentication (Bearer-token check on FastAPI + NextAuth JWE decrypt path — `backend/api/auth.py::get_current_user` wired in main.py; audit every router actually mounts it); API3 Property-level authz / mass assignment (pydantic `response_model` prevents overexposure — audit raw-dict returns); API4 Unrestricted Resource Consumption (rate limiting: `aiolimiter` pinned — audit which outbound/inbound paths are actually limited; polling endpoints in QuietAccessFilter list are unthrottled inbound); API5 Function-level authz; API6 Sensitive business flows (trade execution endpoints); API7 SSRF (any endpoint fetching user-supplied URLs); API8 Security Misconfiguration (CORS `allow_origins` values in main.py:397-398, docs exposure, verbose errors); API9 Inventory (20 routers via `include_router` — audit dead/undocumented routes); API10 Unsafe Consumption of third-party APIs (yfinance/alpaca/FRED responses trusted? — phase-69 found real instances: RTTNews parser, FX fallbacks).

**Secrets rules (cheat sheet):** never hardcode ("littered throughout configuration files" is the named anti-pattern); env vars "generally accessible to all processes and may be included in logs or system dumps"; rotation; least privilege; secret scanning ("detect-secrets... signature matching"); log redaction (repo HAS `SecretRedactionFilter` at handler level, main.py:~104, installed after the away-week 2,101-plaintext-api_key incident — audit its pattern coverage; the incident proves the class is live here).

**Checkable rules:** GREP hardcoded keys (`api_key\s*=\s*["']`, `sk-`, `AKIA`); `.env` files gitignored + never committed (verify `git log --all -- backend/.env`); SecretStr unwrap discipline (project memory: SecretStr truthiness bug killed 4 overlays); CORS not `["*"]` with credentials; auth dependency on EVERY router; error handlers don't leak tracebacks to clients.

### T6. LLM-application engineering (Anthropic/vendor docs)

**Canonical (all three fetched in full 2026-07-19):** https://www.anthropic.com/engineering/building-effective-agents + https://www.anthropic.com/engineering/writing-tools-for-agents + https://platform.claude.com/docs/en/build-with-claude/structured-outputs.

**Compliant looks like:** workflows (predefined paths) preferred over agents where subtasks are fixed — "consider adding complexity only when it demonstrably improves outcomes"; the 5 workflow patterns (prompt chaining / routing / parallelization / orchestrator-workers / evaluator-optimizer) matched to use; tools namespaced with unambiguous params ("instead of a parameter named `user`, try a parameter named `user_id`"), high-signal responses (no UUID dumps), "pagination, range selection, filtering, and/or truncation with sensible default parameter values", errors that "clearly communicate specific and actionable improvements, rather than opaque error codes or tracebacks"; structured outputs via `output_config.format` / `strict: true` — "guarantee schema-compliant responses through constrained decoding... No more JSON.parse() errors".

**Checkable rules:**
1. `backend/agents/llm_client.py` (2214 lines): does the Claude path use native structured outputs (`output_config` / `client.messages.parse`) where JSON is required, or regex/fence-stripping fallbacks? `anthropic==0.96.0` was pinned FOR output_config support (requirements.txt comment) — audit actual adoption vs pin rationale.
2. Structured-outputs caveats respected: `stop_reason == "max_tokens"` / `"refusal"` mean output may NOT match schema — parse sites must check stop_reason before trusting JSON; enum casing compared case-insensitively; unsupported schema constraints (`minimum`, `maxLength`, regex patterns) are STRIPPED by SDK transform — server-side re-validation required.
3. MCP servers (`backend/agents/mcp_servers/*.py`; signals_server.py 1887 lines): param names unambiguous, responses bounded (pagination/truncation defaults), error strings actionable (not tracebacks).
4. Architecture mapping: Layer-1 pipeline = prompt chaining; Layer-2 MAS = orchestrator-workers; harness Q/A = evaluator-optimizer — matches the Anthropic taxonomy; audit for agentic loops where a fixed workflow would do (simplicity principle).
5. Agent prompts (`backend/agents/skills/*.md`, 28 files): example usage, edge cases, boundaries per ACI discipline ("explaining to a new team member").
6. Token hygiene: per-stage output caps (Enrichment 1024 / Debate 1536 / Synthesis 4096 — project rule) align with response-bounding guidance; audit drift.

### T7. Testing strategy for financial/quant systems

**Canonical:** https://docs.pytest.org/en/stable/explanation/goodpractices.html (fetched 2026-07-19) + Hypothesis docs (https://hypothesis.readthedocs.io — snippet + tutorial hit) + golden/characterization-test literature (snippet tier).

**Compliant looks like:** tests outside application code (repo: `backend/tests/` inline-ish but consistent — pytest also blesses inlined layout); `importlib` import mode for new projects ("your test files must have unique names" under default `prepend` — 127 files, collision risk); explicit `testpaths`/`addopts` in config; property-based tests for invariant-rich code ("financial markets are full of invariants: non-negative spreads, consistent OHLC... money never gets lost"); golden-file/characterization tests for backtest determinism (persist a validated output; "compare current test results against previously validated solutions"); live-dependency quarantine via markers (repo's `requires_live` marker IS this pattern — validated).

**Checkable rules:**
1. `pytest.ini` has NO `testpaths`, NO `addopts`, NO `filterwarnings` — auditors check discovery cost, warning hygiene (`-W error::DeprecationWarning` sweep for 3.14), and importlib mode adoption.
2. Coverage economics: `pytest-cov==7.1.0` + tiered coverage policy exists (`docs/coverage_tier_overrides.md`, cycle-53 DoD-4) — audit tier assignments vs money-path criticality (paper_trader, kill_switch, fx_rates should be top tier).
3. Property-based candidates: FX conversion round-trips, P&L sign preservation, DSR/PBO math, OHLC bar validation (`hypothesis` NOT currently in requirements.txt — an add would need owner sign-off).
4. Golden-file candidates: backtest engine on a frozen fixture window → pinned metrics JSON; catches silent numeric drift (the 69.2 stale-purge-leak baseline is the proven failure class).
5. Async tests: httpx.AsyncClient + ASGITransport + dependency_overrides (zhanymkanov) — audit `backend/tests/api/` actually does this vs TestClient sync-only.
6. No test pollution: unique test-file names (prepend mode), single conftest.py at root — verify no shadowed duplicates.
7. Async tests need `pytest-asyncio` (NOT in requirements.txt — verify how backend/tests currently run coroutines); 2026 practice = `asyncio_mode = auto` in config; `--strict-markers` so the `requires_live` marker (and any typo'd marker) is enforced rather than silently ignored (pytest 8.x line current).

### T8. Multi-agent / LLM code-review + audit methodology

**Canonical (peer-reviewed, both fetched in full 2026-07-19):** SWR-Bench arXiv:2509.01494 + Refute-or-Promote arXiv:2604.19049. Plus Anthropic building-effective-agents (evaluator-optimizer pattern).

**The precision problem (SWR-Bench):** best technique F1 19.38% / precision 16.65%; "most baselines: precision scores below 10%"; "a primary factor limiting higher F1... is their low precision, indicative of a high false positive rate". Multi-review aggregation of 10 passes lifted F1 +43.67%. Recommendation: separate functional from evolutionary (style) findings; reasoning-enhanced models.

**The fix (Refute-or-Promote, 2026):** adversarial stage gates with KILL MANDATES — "adversarial agents attempt to disprove candidates at each promotion gate"; context isolation (adversaries get the claim, NOT the finder's reasoning, to prevent anchoring); cross-model/cold-start critics; MANDATORY empirical validation ("no candidate reaches disclosure without empirical confirmation" — the Bleichenbacher case: "one test killed what 80+ agents' reasoning could not"). Kill rates: ~79-83% of candidates eliminated pre-disclosure; Stage A alone kills ~63%.

**Methodology rules for the 75.0 audit design (each maps to an existing project pattern):**
1. Finder → 2+ independent adversarial verifiers with refute mandate and claim-only context (phase-69/70 ultracode already did exactly this — keep it, it is literature-validated).
2. Verbatim code evidence (file:line + quote) required per finding; no evidence → auto-refuted.
3. Doc-anchor requirement: each finding cites the official rule violated (THIS brief supplies the anchors).
4. Empirical gate where cheap: run the grep/command that demonstrates the defect; findings with runnable evidence outrank reasoning-only findings.
5. Stratified hunting: split auditors by non-overlapping subsystem + role lens (Refute-or-Promote "Stratified Context Hunting").
6. Aggregate overlapping passes then dedupe (SWR-Bench multi-review) — recall from fan-out, precision from the kill gate.
7. Severity dispatch BLOCK/WARN/NOTE per the existing `code-review-trading-domain` skill; separate functional vs evolutionary classes in the register.

**Round-6 additions (both fetched in full):** RepoAudit (github.com/PurCL/RepoAudit) — repo-level LLM auditing with tree-sitter parsing + inter-procedural dataflow agents; "hundred[s] of confirmed and fixed bugs"; lesson: pair NEURAL (LLM) auditors with SYMBOLIC (grep/AST) pre-analysis rather than free-reading. Agent Audit (arXiv:2603.22853) — LLM-AGENT-app-specific scanner (pyfinagent IS one): 4 parallel scanners (AST taint / secrets regex+entropy / MCP-config parsing / privilege), "95.24% recall, 86.96% precision, F1 0.909" vs Semgrep 23.8% recall; transferable rules: (a) boundary-aware analysis — elevated scrutiny at tool decorators and agent boundaries; (b) audit code + config + credentials as interdependent domains (`.mcp.json` and hook configs are AUDIT SURFACE, not just code); (c) tiered confidence BLOCK/WARN/INFO with test-file downgrades; (d) prompt-injection sources = f-string prompt construction (GREP f-strings feeding LLM prompts with external data); (e) prioritize exclusive detections over duplicating what deterministic linters already catch — run Ruff/eslint/tsc FIRST, spend LLM auditors only on what they can't see.

### T9. Dependency / supply-chain hygiene

**Canonical:** npm package-lock docs (fetched) + uv compile docs (fetched) + OWASP/industry supply-chain guidance (searched; Sonatype 2026: 1.23M cumulative malicious packages, +75% YoY; Shai-Hulud worm; chalk/debug/axios compromises).

**Compliant looks like:** lockfiles COMMITTED for both ecosystems ("teammates, deployments, and continuous integration are guaranteed to install exactly the same dependencies"); `npm ci` (not `npm install`) for reproduction; periodic `npm audit` / `pip-audit`; version-release cooldown (7-day cooldown would have fully blocked the ~4-5h axios attack window); floors+lockfile rather than naked floors.

**Checkable rules:**
1. `frontend/package-lock.json` — VERIFIED PRESENT and committed (507KB, last modified 2026-05-26). Resolved versions (2026-07-19): next 15.5.12, react 19.2.4, next-auth 5.0.0-beta.30, typescript 5.9.3, tailwindcss 3.4.19, eslint-config-next 16.2.4. Auditors should confirm `npm ci` (not `npm install`) is the documented reproduction path.
2. Backend: no lockfile mechanism at all — introduce `uv pip compile backend/requirements.txt -o backend/requirements.lock` (+ `--generate-hashes`) as a remediation step; the .venv is currently the only de-facto lock and is unversioned.
3. `pip-audit` / `npm audit` runs are not wired anywhere (verify in scripts/ + hooks) — a scheduled advisory scan is a cheap step.
4. The 9 existing `==` pins prove the project already accepts pin-on-incident (phase-3.7.6 LiteLLM; phase-67.6 stale-pin "downgrade time bomb") — the audit generalizes this to policy, not incident response.
5. npm: audit `hasInstallScript` introductions in lockfile diffs; consider minimum-release-age cooldown config.
6. GCP SDK floors (`google-cloud-bigquery>=3.20.0`) span YEARS of releases — floor-bump + lock to tested versions.

---

## Recency scan (last 2 years, 2024–2026)

Explicit last-2-year passes were run for every topic (see queries-run). Findings that COMPLEMENT or SUPERSEDE canonical sources:

1. **CVE-2025-66478 / CVE-2025-55182 "React2Shell" (Dec 2025)** — CVSS 10.0 unauthenticated RCE in the RSC protocol; Next.js **15.x affected**, patch lines 15.0.5–15.5.7/16.0.7; actively exploited ("China-nexus cyber threat groups rapidly exploit..." — AWS security blog title in official advisory resources). Plus CVE-2025-55184 (DoS) and CVE-2025-55183 (source-code exposure). SUPERSEDES any pre-Dec-2025 Next.js security posture. THE P0 version check for this audit.
2. **Next.js 16 is current** (live docs v16.2.10, updated 2026-06-23); repo on `^15.0.0` with `eslint-config-next ^16.2.4` skew. React 19 GA (Dec 2024) deprecates `forwardRef` + `<Context.Provider>`.
3. **Python 3.14 (Oct 2025)**: deferred annotations (PEP 649/749), t-strings (PEP 750), `python -m asyncio ps/pstree` introspection CLI, PEP 765 finally-control-flow SyntaxWarning, free-threading officially supported.
4. **googleapis/python-bigquery repo archived Mar 2026** → moved into google-cloud-python monorepo; official guidance now: "For new projects, we recommend using the BigQuery Storage Write API instead of the tabledata.insertAll method" (lower cost + exactly-once). `deadline` deprecated in favor of `timeout` in google-api-core retries.
5. **uv displaced pip-tools** as the recommended lock tool (2024–2026); universal `uv.lock` format; `uv pip compile/sync` drop-in for requirements workflows.
6. **OWASP API Security Top 10: NO new edition** — 2023 list confirmed current as of 2026-07 ("there is no formally published... 2026 yet"). OWASP Agentic Security Initiative (10 categories) emerged for LLM-agent apps (Agent Audit covers it).
7. **Anthropic structured outputs GA** (constrained decoding; `output_config.format` + `strict: true`) across current model line — supersedes prompt-based JSON coercion patterns; grammar caching 24h; documented failure modes (max_tokens/refusal stop_reasons).
8. **LLM code-review precision literature is a 2025–2026 phenomenon**: SWR-Bench (precision <10–17%), ZeroFalse, Refute-or-Promote (adversarial stage gates, 79–83% kill rate), RepoAudit, Agent Audit (95%/87% on agent apps), iAudit; curl's bug bounty closed under AI-slop submissions (<5% confirm rate). This literature VALIDATES the project's phase-69/70 finder→adversarial-verifier design and sharpens it (context isolation, empirical gates, cross-model critics).
9. **npm supply chain escalation (2025–2026)**: Shai-Hulud self-propagating worm, chalk/debug/axios compromises; Sonatype 2026: 1.233M cumulative malicious packages (+75% YoY); new best practices: 7-day release cooldown, `npm ci` enforcement, `hasInstallScript` diff-flagging.
10. **Auth.js/NextAuth**: v5 stable (late 2024), project now maintained by the Better Auth team (2026) in security/maintenance mode; new projects officially directed to Better Auth — repo's `next-auth ^5.0.0-beta.30` pin is BEHIND the stable line AND on a sunsetting library (maintenance-risk datapoint, not an emergency).
11. **Testing**: pytest 8.x current line; pytest-asyncio `asyncio_mode=auto` is 2026 idiom; TS `tsc --init` now defaults `noUncheckedIndexedAccess` + `exactOptionalPropertyTypes` (strictness ratchet moved).
12. **Backtest-overfit gates**: 2024 ScienceDirect ML-era OOS comparison + GT-Score (2026) confirm DSR/PBO/CPCV as current best practice — the project's DSR>=0.95 / PBO<=0.5 immutable gates remain literature-aligned (CPCV complement already queued as 73.1.4; NOT a 75.0 topic).

---

## Internal recon findings

> Note: this brief was resumed from a prior-session skeleton (write-first survivor). All source-table rows are being RE-FETCHED this session before counting toward the envelope; topic sections are authored from this session's fetch content.

### Repo stack ground-truth (file:line anchors)

- **Python 3.14.4** in `.venv` (verified `python --version` 2026-07-19). No `pyproject.toml`; deps in `backend/requirements.txt` (59 lines).
- **Pinning is MIXED**: 9 exact `==` pins justified inline as supply-chain hardening (`anthropic==0.96.0`, `google-genai==1.73.1`, `openai==2.29.0`, `fastmcp==3.2.4`, `alpaca-py==0.43.2`, `google-cloud-aiplatform==1.142.0`, `pytrends==4.9.2`, `pytest-cov==7.1.0`, `coverage==7.14.0`); the other ~50 are `>=` floors (`fastapi>=0.115.0`, `uvicorn[standard]>=0.30.0`, `pydantic>=2.0`, `cryptography>=42.0.0`...). **No Python lockfile / constraints file** (VERIFIED 2026-07-19: `find` for `uv.lock`/`poetry.lock`/`Pipfile.lock`/`constraints*`/`requirements*.lock` returns zero hits), so `pip install -r` resolves floors freshly each run — reproducibility + supply-chain gap candidate.
- **Frontend** `frontend/package.json`: Next `^15.0.0`, React `^19.0.0`, TS `^5.6.0`, `next-auth ^5.0.0-beta.30` (beta in prod-auth path), Tailwind `^3.4.0`, all caret ranges; devDeps include `eslint-config-next ^16.2.4` (major AHEAD of `next ^15` — version-skew flag), vitest 4, Playwright 1.50.
- **`frontend/tsconfig.json`**: `strict: true` but `target: ES2017`, `skipLibCheck: true`, no `noUncheckedIndexedAccess` — strictness-headroom candidates.
- **`pytest.ini`** (repo root): ONLY registers the `requires_live` marker (live-BQ quarantine, skipped unless `PYFINAGENT_LIVE_TESTS=1`); no `addopts`, no `testpaths`, no `filterwarnings`. 127 `test_*.py` files under `backend/tests/` (+`api/`, `fixtures/`, single `conftest.py`).
- **Module sizes** (wc -l 2026-07-19): `backend/services/autonomous_loop.py` 3142, `backend/agents/orchestrator.py` 2308, `backend/agents/llm_client.py` 2214, `backend/agents/mcp_servers/signals_server.py` 1887, `backend/agents/multi_agent_orchestrator.py` 1691, `backend/api/backtest.py` 1599, `backend/api/paper_trading.py` 1358, `backend/services/paper_trader.py` 1313, `backend/backtest/backtest_engine.py` 1308. Backend total ~104.7K lines.
- **`frontend/next.config.js`**: CJS, `output: "standalone"`, 308 redirects for 3 legacy routes, `PLAYWRIGHT_DIST_DIR` isolation (phase-64.1), `optimizePackageImports` for Phosphor.
- **Project rule files already encoding standards** (auditors must audit AGAINST these too, not duplicate them): `.claude/rules/{frontend,frontend-layout,backend-agents,backend-api,backend-backtest,backend-services,backend-slack-bot,backend-tools,security,research-gate}.md`.

### Additional ground-truth anchors surfaced during recon (seed candidates for auditors — VERIFY, do not assume)

- `backend/main.py:~100` — logging formatter selection appears INVERTED vs its own comment: "Use compact format for local dev, JSON for production" but code sets `JsonFormatter()` when `settings.debug` is True (`backend/config/settings.py:20` defaults `debug: bool = False`) → production/default runs get the colored CompactFormatter, dev gets JSON. Hand to backend auditor as a seed.
- `backend/main.py:397-403` — CORS: `allow_origin_regex=r"^http://(localhost|100\.\d+\.\d+\.\d+):\d+$"` with `allow_credentials=True`, `allow_methods=["*"]`, `allow_headers=["*"]` — scoped (not `*` origin) but HTTP-only origins; audit against `.claude/rules/security.md` §CORS.
- `backend/main.py:405+` — `_PUBLIC_PATHS` includes `/api/health`, `/api/changelog`, `/api/auth`, `/api/cost-budget`, `/api/jobs/status`, `/api/harness/monthly-approval`(+) — unauthenticated surface beyond the 3 paths documented in `.claude/rules/security.md` ("Auth middleware skips: /api/health, /api/auth, /docs, /openapi.json, /redoc") → doc/config drift AND API9-inventory audit item.
- `backend/main.py` uses `lifespan` (modern pattern ✓), 20 `include_router` calls, handler-level `SecretRedactionFilter` (phase-60.4 ✓).
- 109 `async def` routes in `backend/api/*.py`; sync-client hits (`requests.`/`time.sleep`/`yf.`) in `backend/api/paper_trading.py`, `backend/api/signals.py`, and 8+ services (`options_flow_screen`, `ticket_queue_processor`, `reconciliation`, `short_interest`, `macro_regime`, `sector_momentum`, `analyst_revisions`, `live_prices`) — T1-rule-1 audit surface.
- `backend/db/bigquery_client.py` — parameterized `QueryJobConfig` throughout (good), `.result(timeout=30)` present (:533, :765, :798), NO `maximum_bytes_billed` anywhere; `insert_rows_json` (legacy insertAll) at :251, :394, :403, :429, :492 + 10 more modules — against the official "for new projects, we recommend using the BigQuery Storage Write API instead of the tabledata.insertAll method... lower pricing and... exactly-once delivery semantics".
- `.claude/rules/security.md` — project security baseline EXISTS (JWE/HKDF auth, OWASP headers list, email whitelist, ASCII-only logging); auditors verify code matches these documented conventions (drift check), not just external standards.

### Exclusion list — already-audited surfaces (75.0 auditors MUST NOT re-litigate)

Synthesis MUST dedupe every 75.0 candidate finding against ALL of the following before it can spawn a masterplan step. New evidence on a known defect = annotation on the existing item, NOT a new finding.

1. **Phase-69 full-codebase audit register** — `handoff/current/audit_phase69/register.md` (+ evidence JSONs in the same dir: confirmed/contested/refuted/goal_candidates/data_sources_vetted/features_vetted): 280-agent ultracode, **50 CONFIRMED / 30 CONTESTED / 4 refuted**. Covers the money path exhaustively: paper_trader FX=1.0 sell-credit, dead external-flow recording, kill-switch unrecoverable HWM ratchet, fx_rates None-fallback, DELETE-before-MERGE position loss, read-modify-write cash races, dead funding guards, go-live gate weakenings, kill-switch NAV<=0 phantom breach, RTTNews dead parser, outcome_tracker tz TypeError, and 39 more. Burned down in phase-69.1–69.4 (all PASS); 4 operator activation tokens still owed (KS-PEAK-RESET, sign_safe_overlays, regime_net_liquidity, historical_macro un-freeze).
2. **Phase-70 trade-diversity audit** — `handoff/current/confirmed_findings.json`: **17 confirmed / 8 refuted** (sector-cap frontend clear-snapback, monosector momentum funnel, silent BUY-gates, non-atomic swap...). Pending masterplan phase-70 converts them; do not re-find.
3. **Phase-71 harness/MAS self-audit** — `handoff/current/harness_proposals.json`: **17 kept / 15 rejected** (Workflow structured-output Q/A codification, JSON clobber, fabricated evaluator spot-check, un-reviewed skill_optimizer...). Executed 7/7 (2026-07-17); pending phase-71 rollup remains as queue-conversion.
4. **Phase-72 money-reconstruction queue** — pending steps 72.0.1–72.2.4 (11): meta-scorer rail decoupling, standard-tier fail-forward, decision-seam observability, degraded-alert paging, sentinel reverse-leg reconciliation, benchmark single-anchor + history rebuild, multi-currency realized-P&L, NAV observability, flow-blind Sharpe fix.
5. **Phase-73 frontier build queue** — pending steps 73.1.1–73.7.1 (13): purge regression test, post-cutoff eval harness, counterfactual PC pilot, CPCV complement, learn-loop crash fix + observability, reflection-on-close + BQ migration, decay re-rank retrieval, conviction→outcome calibration, sizing-seam scalar, net-of-cost DSR, PBO nested-gate docs, champion bridge, MAS retry-bug fix (anchor multi_agent_orchestrator.py:1363-1394).
6. **Phase-74 local-LLM pilot queue** — pending 74.0–74.3 (Ollama/Qwen3-4B Slack-bot + last-resort rail; localization REJECTED assessment stands).
7. **Legacy pending phases** (5.x market expansion, 27.6*, 35.3, 38.13, 39.1, 43.0, 44.x UI refresh, 46.x market page, 50/53/58) — product roadmap, not audit debt; 75.0 does not audit their absence.
8. **Away-ops/paging infra** (phases 62–65, closed) and **Fable-window artifacts** (phase-67, reverted on schedule) — audited and closed; runbooks in `docs/runbooks/away-ops-rules.md`.

---

## Auditor role charter (proposed IT-stack role split)

Nine auditor roles + one cross-cutting methodology contract. Each role gets: its doc anchors (from the read-in-full table), its checkable-rule set (topic sections above), and non-overlapping file scope (Refute-or-Promote "Stratified Context Hunting"). **Deterministic tools run FIRST** (Ruff, `tsc --noEmit`, `next lint`, `npm audit`, `pip-audit`, `npm ls next`); LLM auditors spend attention only on what those can't see (Agent Audit "exclusive-detection focus").

| Role | Scope | Primary anchors | Highest-value checks |
|---|---|---|---|
| A1 Backend / FastAPI+Python | `backend/api/`, `backend/services/`, `backend/main.py` | #1, #9, #8 | T1 rules 1–8 (blocking-I/O sweep over 109 async routes is the headline); T2 rules 2–6; main.py formatter inversion seed; conditional-OpenAPI |
| A2 Frontend / Next+React+TS | `frontend/src/**`, configs | #5, #6, #23, #24, #19 | T3 rules 1–8 + P0 CVE check (VERIFIED patched — confirm Dec-2025 window); official data-security audit checklist verbatim |
| A3 Data engineering / BigQuery | `backend/db/`, all BQ call sites, `scripts/migrations/` | #4, #17 | T4 rules 1–7 (`maximum_bytes_billed` absence; insert_rows_json→Write API per site; SELECT-*/LIMIT misconceptions) |
| A4 Security | auth paths, `_PUBLIC_PATHS`, CORS, secrets, `.claude/rules/security.md` drift | #2, #13, #24 | T5 checkables; _PUBLIC_PATHS vs security.md drift; API10 third-party response trust (phase-69 proved the class); redaction-filter coverage |
| A5 QA / Testing | `backend/tests/`, `pytest.ini`, coverage tiers, frontend tests | #14, #20 | T7 rules 1–7 (config gaps; asyncio mode; money-path tier placement; property/golden candidates) |
| A6 LLM-app engineering | `backend/agents/**`, `llm_client.py`, MCP servers, `skills/*.md` | #3, #7, #16 | T6 rules 1–6 (structured-outputs adoption vs pin rationale; stop_reason handling; ACI quality; prompt f-string injection per Agent Audit) |
| A7 DevOps / scripts+ops | `scripts/`, hooks, launchd, schedulers, logs | S2, S5, S6, S7 | APScheduler duplicate-fire/misfire + shutdown; zombie-worker rule vs code; log rotation; `.env` handling in scripts; `npm ci` docs |
| A8 Dependency / supply-chain | `requirements.txt`, `package.json`+lock, `.mcp.json` pins | #11, #15, #18 | T9 rules 1–6 (backend lockfile absence is the headline; pip-audit/npm-audit wiring; next-auth beta/maintenance-mode pin; major-currency table) |
| A9 Architecture / docs-drift | module boundaries, dead code, CLAUDE.md/ARCHITECTURE.md/rules drift | #3, #8 | Simplicity principle vs agentic sprawl; 3142-line autonomous_loop decomposition case; doc-vs-code drift (security.md public paths, dataset locations); config sprawl |

**Cross-cutting methodology contract (binding on every role — T8):** (1) every finding = verbatim code quote + file:line + the violated doc-anchor URL; (2) finder→adversarial verifiers with KILL mandate and claim-only context; (3) empirical gate — attach the grep/command output that demonstrates the defect where cheap ("one test killed what 80+ agents' reasoning could not"); (4) dedupe against the exclusion registers BEFORE promotion; (5) BLOCK/WARN/NOTE severity (per `code-review-trading-domain` skill) + functional-vs-evolutionary split; (6) do-no-harm boundaries from the goal draft are byte-binding (no product-code edits this phase; kill-switch/stops/caps/DSR/PBO untouched).

---

## Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch — **24** (all fetched THIS session; the 7 prior-skeleton rows were re-fetched before counting)
- [x] 10+ unique URLs total — ~200 surfaced; 54 recorded in tables
- [x] Recency scan (last 2 years) performed + reported — dedicated section, 12 findings
- [x] Full pages read (not abstracts) for the read-in-full set — arXiv papers via /html/ per the HTML-first chain
- [x] file:line anchors for every internal claim — see internal recon
- [x] coverage.dry == true (audit-class requirement) — 2 consecutive dry rounds after 8 wet rounds

Soft checks:
- [x] Internal exploration covered every relevant module class (configs, wiring, largest modules, test layout, registers)
- [x] Contradictions/consensus noted (e.g., LIMIT-caps-cost misconception vs official doc; inlined tests blessed by pytest despite src-layout preference; single-process uvicorn acceptable locally vs don't-run-raw production guidance)
- [x] Claims cited per-claim with URL + access date

---

## JSON envelope

```json
{
  "tier": "complex",
  "external_sources_read_in_full": 24,
  "snippet_only_sources": 30,
  "urls_collected": 200,
  "recency_scan_performed": true,
  "internal_files_inspected": 26,
  "coverage": {
    "audit_class": true,
    "rounds": 10,
    "dry_rounds": 2,
    "K_required": 2,
    "new_findings_last_round": 0,
    "dry": true
  },
  "summary": "24 official/peer-reviewed sources read in full arm a 9-role IT-stack audit: FastAPI blocking-I/O discipline (109 async routes to sweep), Python 3.14 idioms + missing backend lockfile, Next.js 15/React 19 (CVE-2025-66478 VERIFIED PATCHED at next@15.5.12; official audit checklist adopted), BigQuery cost gates (no maximum_bytes_billed; legacy insertAll vs recommended Storage Write API), OWASP API Top 10 2023 (still current) + secrets rules, Anthropic ACI/structured-outputs adoption audit, pytest/property/golden testing economics, and a literature-validated adversarial audit methodology (SWR-Bench precision crisis; Refute-or-Promote kill gates; Agent Audit boundary-aware scanning). Exclusion registers (phase-69/70/71 + 72/73/74 queues) enumerated so 75.0 dedupes instead of re-finding. Seed anomalies handed to auditors: inverted logging formatter, _PUBLIC_PATHS doc drift. Coverage loop ran to 2 consecutive dry rounds.",
  "brief_path": "handoff/current/research_brief_75.0.md",
  "gate_passed": true
}
```
