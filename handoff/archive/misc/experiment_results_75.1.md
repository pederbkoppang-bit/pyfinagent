# Experiment Results — Step 75.1: Audit75 S1 — backend auth surface fail-closed

- **Date:** 2026-07-20 · **Executor:** Fable rail (operator budget directive 2026-07-20; step tag sonnet-4.6/high)
- **Findings closed:** security-01 (P1), security-03 (P2), security-04 (P2), gap2-03 (P2), api-design-12 (P3), pysvc-08 (P3)
- **Boundary honored:** zero `.env` edits (a value-read attempt was even permission-denied by the harness — only presence checks used).

> **CYCLE 2 (2026-07-20).** Cycle-1 Q/A (`wf_e2ad4954-e93`) returned **FAIL** on criterion 3: my "exhaustive" consumer grep omitted the entire **test tier**, and the prune broke 8 passing tests with 401. The Q/A was right — `DEV_LOCALHOST_BYPASS` does NOT cover Starlette TestClient (`client.host == "testclient"`, not `127.0.0.1`), so in-process suites had no valid disposition. Fixed below; the consumer table now includes the test tier. See `evaluator_critique_75.1.md` for the verbatim verdict.

## What was built/changed (git diff --stat, cycle 2)

```
 .claude/rules/security.md                          | 16 ++++-
 backend/api/auth.py                                | 18 +++--
 backend/api/monthly_approval_api.py                | 30 ++++----
 backend/config/settings.py                         |  5 +-
 backend/main.py                                    | 81 ++++++++++++++++------
 backend/tests/api/test_sovereign.py                |  6 +-
 .../tests/test_phase_23_2_7_red_line_nav_match.py  | 28 ++++++--
 tests/api/test_observability.py                    |  6 +-
 8 files changed, 134 insertions(+), 56 deletions(-)
 + backend/tests/auth_helper.py (new, 89 lines, untracked)
```

1. **backend/main.py**
   - `_PUBLIC_PATHS` pruned 16 → 8 entries; all 8 targets removed (`/api/harness/monthly-approval`, `/api/sovereign`, `/api/signals`, `/api/observability`, `/api/cost-budget`, `/docs`, `/openapi.json`, `/redoc`); every survivor now carries an inline justification comment (security-01/03).
   - `FastAPI(...)` ctor gates `docs_url`/`redoc_url`/`openapi_url` on `get_settings().debug` — prod default (DEBUG unset) unmounts them entirely (`openapi_url=None` cascades) (pysvc-08).
   - New module-level `_TAILSCALE_ORIGIN_RE = ^http://(localhost|100\.(6[4-9]|[7-9]\d|1[01]\d|12[0-7])\.\d+\.\d+):\d+$` shared by BOTH seams: `allow_origin_regex=_TAILSCALE_ORIGIN_RE.pattern` AND the manual 401 CORS echo (`.match(origin)` replaces both `startswith` shortcuts). Old permissive any-second-octet pattern gone (security-04). Intended tightening: the echo now requires an explicit `:port`.
   - New `_warn_if_allowlist_empty(settings)` helper called from `lifespan` right after startup logging — WARNs loudly when `ALLOWED_EMAILS` is empty, with distinct fail-open vs fail-closed messages; extracted as a helper (not inline / not a Settings validator) so it is unit-provable without booting schedulers, and because Settings multi-instantiates outside the lru_cache singleton (gap2-03).
2. **backend/config/settings.py** — `auth_enforce_allowlist: bool = Field(False, ...)` DARK flag in the Authentication section; default False (gap2-03).
3. **backend/api/auth.py** — allowlist leg parses unconditionally; `empty + flag True → 401 reject-all`; `empty + flag False → legacy fail-open (byte-identical)`; non-empty membership check unchanged (gap2-03).
4. **backend/api/monthly_approval_api.py** — `ApprovalActionBody.action: Literal["approved","rejected"]`; POST `month_key: Annotated[str, PathParam(pattern=r"^\d{4}-\d{2}$")]` (FastAPI `Path` aliased — pathlib.Path already imported); the now-unreachable degrade-to-HTTP-200-"rejected" branch and its `_ALLOWED_ACTIONS` frozenset removed (api-design-12). Intended tightening: mixed-case actions now 422 (no in-repo POST caller exists; grep below).
5. **.claude/rules/security.md** — Authentication section now lists EXACTLY the 8-entry post-prune public set with per-prefix justification, the debug-only docs rule, the DARK-flag semantics, and the `DEV_LOCALHOST_BYPASS` rail; CORS section documents the shared CGNAT predicate.

## Verification command (immutable) — verbatim

```
$ python3 -c "s=open('backend/main.py').read(); blob=s.split('_PUBLIC_PATHS')[1].split(')')[0]; bad=[...8 prefixes...]; assert not any(b in blob for b in bad), 'public paths not pruned'; assert '6[4-9]|[7-9]' in s, 'CGNAT regex missing'; old='100'+chr(92)+'.'+chr(92)+'d'; assert old not in s, ...; m=open('backend/api/monthly_approval_api.py').read(); assert 'Literal[' in m and m.count('{4}-')>=2, ...; c=open('backend/config/settings.py').read(); assert 'auth_enforce_allowlist' in c, ..."
VERIFICATION EXIT 0
ast OK: backend/main.py
ast OK: backend/api/auth.py
ast OK: backend/api/monthly_approval_api.py
ast OK: backend/config/settings.py
```

(First run exited 1: my own CGNAT comment quoted the old regex literally and tripped the `100\.\d` forbidden-substring check — comment reworded, then exit 0.)

## Consumer enumeration (criterion 3) — grep, exhaustive

| Prefix | Consumers found | Disposition |
|---|---|---|
| /api/sovereign | frontend `sovereign/page.tsx`, `ComputeCostBreakdown.tsx`, `RedLineMonitor.tsx` via `api.ts:740,746,750,807` | ALL via `apiFetch` (Bearer + credentials:'include') — no change needed |
| /api/signals | frontend `AltDataPanel.tsx` via `api.ts:191,195,199` | ALL via `apiFetch` — no change needed |
| /api/observability | frontend via `api.ts:464` | via `apiFetch` — no change needed |
| /api/cost-budget | **zero** frontend callers; `llm_client.py:371-396` references are in-process (not HTTP) | no change needed |
| /api/harness/monthly-approval | **zero** frontend callers; **zero** in-repo POST callers | no change needed |
| slack_bot (`app_home.py`, `direct_responder.py`, `scheduler.py`, `commands.py`) | calls ONLY `/api/jobs/*`, `/api/health`, `/api/paper-trading/*`, `/api/reports/`, `/api/analysis/`, `/api/backtest/status` — **none of the five pruned prefixes** | untouched by this step |
| scripts | `go_live_drills/smoke_test_4_17_6.py:63,71,81` hits `/api/signals/*` + `/api/sovereign/leaderboard` tokenless | rides the DEV_LOCALHOST_BYPASS rail (probed ACTIVE below) |
| **TEST TIER — added cycle 2** | | |
| /api/sovereign | `backend/tests/api/test_sovereign.py` (6 call sites, module-level `TestClient(app)`) | **FIXED**: now `authed_test_client(app)` — sends a real minted JWE. 7/7 pass. |
| /api/observability + /api/cost-budget | `tests/api/test_observability.py:55,89` (module-level `TestClient(app)`) | **FIXED**: now `authed_test_client(app)`. 5/5 pass. |
| /api/sovereign (live backend) | `backend/tests/test_phase_23_2_7_red_line_nav_match.py:63,92` (tokenless urllib to :8000) | **FIXED**: `_backend_is_up()` now probes the real authed target and skips on 401 instead of failing. It already depended on the bypass rail today (it also curls `/api/paper-trading/portfolio`, never public) — now that dependency is explicit, not assumed. 4 passed, 1 skipped. |
| /api/sovereign (Playwright) | `frontend/tests/e2e-functional/_helpers.ts:73` | No change needed: wrapped in try/catch with a `"baseline"` fallback (fail-soft, degrades rather than breaks); Playwright's `page.request` also inherits the authenticated browser context. |

**Why the cycle-1 table missed these:** it leaned on the `DEV_LOCALHOST_BYPASS` rail as the blanket mitigation for tokenless callers. That rail requires `request.client.host in (127.0.0.1, ::1, localhost)`; Starlette's TestClient reports `"testclient"`, so in-process suites never hit it. The rail covers live-backend curls only.

### Cycle-2 fix: `backend/tests/auth_helper.py` (new)

`mint_session_token()` builds a real NextAuth-shaped JWE (dir / A256CBC-HS512) with the configured `AUTH_SECRET`, defaulting the email to the first `ALLOWED_EMAILS` entry so the allowlist leg is cleared too; `authed_test_client(app)` returns a `TestClient` sending it on every request. Suites therefore exercise the **true decrypt + allowlist path**, not a bypass. If `AUTH_SECRET` is absent (clean CI checkout) it falls back to the documented `DEV_DISABLE_AUTH=1` escape hatch, which only functions when there is no secret to verify against.

## Live evidence (in-process TestClient against the NEW code + running-backend probes)

**Bypass-rail probe (running backend, OLD code, proves the rail is load-bearing & ACTIVE):**
```
/api/health -> 200    /api/jobs/all -> 200  (jobs/all is NOT public => DEV_LOCALHOST_BYPASS active)
/api/sovereign/leaderboard -> 200    /api/signals/macro/indicators -> 200
```
Consequence: after the operator restarts the backend, tokenless localhost tooling (immutable masterplan curls, smoke_test_4_17_6.py, Slack bot's existing non-public calls) KEEPS WORKING via the same rail. Confirmed against the new code: `MODE=bypass ... /api/sovereign/leaderboard -> 200`.

**MODE=noauth (no bypass, no token) — 15/15:** all five newly-authed prefixes → 401; `/docs`,`/openapi.json`,`/redoc` → 401; `/api/health` → 200; CGNAT preflight allowed + non-CGNAT (100.20.x) refused; 401-echo echoes CGNAT + localhost origins, refuses non-CGNAT — shared-predicate proof at both seams.

> **CORRECTION (cycle 3).** An earlier revision of this file claimed the docs 401 was "mounted because this machine runs DEBUG=true". That was wrong and the cycle-2 Q/A caught it. Verified runtime state on this machine:
> ```
> settings.debug = False
> app.docs_url = None | redoc_url = None | openapi_url = None
> ```
> Docs are **already unmounted here** — the observed 401 is the auth middleware short-circuiting *before* routing, not a mounted-but-authed `/docs`. The probe script sets no DEBUG var. The docs-gating code is correct and verified (debug False → all three URLs None); only the environment claim was wrong.

**MODE=bypass (DEV_LOCALHOST_BYPASS=1, client 127.0.0.1) — 4/4:**
```
[OK] POST bad month_key -> 422
[OK] POST invalid action -> 422  body={"detail":[{"type":"literal_error","loc":["body","action"],...
[OK] POST valid shape on rowless month -> 200 no_row_to_resolve (no mutation)
[OK] localhost tooling (bypass rail) still reaches /api/sovereign  code=200
```

**MODE=flag_on / flag_off (REAL minted JWE via AUTH_SECRET, empty ALLOWED_EMAILS):**
```
[OK] valid token + empty allowlist + flag ON  -> 401 reject-all
[OK] valid token + empty allowlist + flag OFF -> 200 legacy fail-open
```

**MODE=warn — 3/3:** `_warn_if_allowlist_empty` fires the fail-open WARNING (empty+flag-off), the fail-closed WARNING (empty+flag-on), silent when non-empty.

Probe script: scratchpad `probe_75_1.py` (5 modes, 24/24 checks total).

## Test-suite evidence (cycle 2)

Before the fix (Q/A-reproduced):
```
$ .venv/bin/python -m pytest backend/tests/api/test_sovereign.py tests/api/test_observability.py -q
8 failed, 4 passed
```

After the fix:
```
$ .venv/bin/python -m pytest backend/tests/api/test_sovereign.py tests/api/test_observability.py -q
12 passed, 1 warning in 11.06s

$ .venv/bin/python -m pytest backend/tests/test_phase_23_2_7_red_line_nav_match.py -q
4 passed, 1 skipped in 1.34s

$ .venv/bin/python -m pytest backend/tests/api/ tests/api/ -q
2 failed, 94 passed, 1 warning in 21.52s
```

The 2 residual failures are `tests/api/test_ticker_meta.py` (yfinance mock assertions) and are **pre-existing, not 75.1's**:
- 75.1 never touched `backend/api/paper_trading.py`, the module under test (`git diff --name-only` confirms).
- The HEAD copy of that test file (`git show HEAD:tests/api/test_ticker_meta.py`) fails identically against the current tree: `2 failed, 7 passed`.
- The failures are `yf_mock.assert_not_called()` assertions — no auth/HTTP surface involved.

One transient note for honesty: in the first full-directory run, `test_phase_23_2_7_red_line_nav_match` hit a 5s `TimeoutError` on `/api/paper-trading/portfolio` under suite load; run alone it passes and the endpoint answers in 3ms by curl. Not a 401, not auth-related.

## Operator notes

- **Restart required** for the new auth surface to go live on :8000 (kill parent AND child workers per CLAUDE.md).
- **Docs are already unmounted on this machine** (`settings.debug` is False → `docs_url`/`redoc_url`/`openapi_url` all None). No action needed. Set `DEBUG=true` only if you *want* `/docs` back locally — it will then be mounted and sit behind the auth middleware.
- `auth_enforce_allowlist` ships DARK (False). Flip `AUTH_ENFORCE_ALLOWLIST=true` to make an empty allowlist reject-all.
- Intended tightenings on record: 401-echo now requires explicit `:port` in the origin; monthly-approval POST no longer lowercases/trims `action`.
