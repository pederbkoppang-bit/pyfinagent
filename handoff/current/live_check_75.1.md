# live_check 75.1 — verbatim evidence (2026-07-20)

## Immutable verification command — exit 0

```
$ cd /Users/ford/.openclaw/workspace/pyfinagent && python3 -c "s=open('backend/main.py').read(); blob=s.split('_PUBLIC_PATHS')[1].split(')')[0]; bad=['/api/harness/monthly-approval','/api/sovereign','/api/signals','/api/observability','/api/cost-budget','/docs','/openapi.json','/redoc']; assert not any(b in blob for b in bad), 'public paths not pruned'; assert '6[4-9]|[7-9]' in s, 'CGNAT regex missing'; old='100'+chr(92)+'.'+chr(92)+'d'; assert old not in s, 'old permissive 100.x regex still present'; m=open('backend/api/monthly_approval_api.py').read(); assert 'Literal[' in m and m.count('{4}-')>=2, 'POST validation missing'; c=open('backend/config/settings.py').read(); assert 'auth_enforce_allowlist' in c, 'allowlist enforce flag missing'"
VERIFICATION EXIT 0
```

## git diff --stat (change surface, cycle 2)

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
?? backend/tests/auth_helper.py   (new, 89 lines)
```

## Cycle-2 test evidence (criterion 3 — the cycle-1 FAIL)

```
BEFORE (cycle-1 state, Q/A-reproduced):
$ .venv/bin/python -m pytest backend/tests/api/test_sovereign.py tests/api/test_observability.py -q
8 failed, 4 passed

AFTER (cycle-2 fix: authed_test_client):
$ .venv/bin/python -m pytest backend/tests/api/test_sovereign.py tests/api/test_observability.py -q
12 passed, 1 warning in 11.06s

$ .venv/bin/python -m pytest backend/tests/test_phase_23_2_7_red_line_nav_match.py -q
4 passed, 1 skipped in 1.34s

$ .venv/bin/python -m pytest backend/tests/api/ tests/api/ -q
2 failed, 94 passed, 1 warning in 21.52s
```

Residual 2 failures = `tests/api/test_ticker_meta.py`, PRE-EXISTING (proof): the HEAD copy of that file
(`git show HEAD:tests/api/test_ticker_meta.py`) run against the current tree gives `2 failed, 7 passed`,
and 75.1 never touched `backend/api/paper_trading.py` (the module under test). Failures are
`yf_mock.assert_not_called()` yfinance mock assertions — no auth/HTTP surface.

## Flag ON-vs-OFF proof ($0, in-process, real minted JWE + empty ALLOWED_EMAILS)

```
=== MODE=flag_on  (AUTH_ENFORCE_ALLOWLIST=true)  ===
  [OK] valid token + empty allowlist + flag ON -> 401 reject-all  code=401
=== MODE=flag_off (AUTH_ENFORCE_ALLOWLIST=false) ===
  [OK] valid token + empty allowlist + flag OFF -> 200 legacy fail-open  code=200
```

## curl-level consumer evidence (criterion 3: /api/sovereign + /api/signals)

Running backend (OLD code), tokenless localhost — proves the DEV_LOCALHOST_BYPASS rail is ACTIVE today (jobs/all is not public and returns 200):

```
$ curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8000/api/health                       -> 200
$ curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8000/api/jobs/all                     -> 200
$ curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8000/api/sovereign/leaderboard        -> 200
$ curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8000/api/signals/macro/indicators     -> 200
```

NEW code, in-process (TestClient / httpx ASGITransport):

```
MODE=noauth (no bypass, no token): 15/15
  GET /api/sovereign/leaderboard -> 401      GET /api/signals/macro/indicators -> 401
  GET /api/observability/data-freshness -> 401   GET /api/cost-budget/status -> 401
  GET /api/harness/monthly-approval/status -> 401
  GET /docs -> 401, /openapi.json -> 401, /redoc -> 401
      (CORRECTION cycle 3: docs are ALREADY UNMOUNTED on this machine --
       settings.debug=False, app.docs_url/redoc_url/openapi_url all None.
       The 401 is the auth middleware short-circuiting before routing.)
  GET /api/health -> 200
  preflight Origin http://100.100.1.1:3000 -> allowed (ACAO echoed); http://100.20.1.1:3000 -> refused (no ACAO)
  401-echo: CGNAT + localhost origins echoed; non-CGNAT refused (shared _TAILSCALE_ORIGIN_RE at both seams)

MODE=bypass (DEV_LOCALHOST_BYPASS=1 + client 127.0.0.1): 4/4
  POST /api/harness/monthly-approval/2026-7  -> 422
  POST /api/harness/monthly-approval/2026-07 {"action":"frobnicate"} -> 422 literal_error
  POST /api/harness/monthly-approval/2099-01 {"action":"rejected"}   -> 200 no_row_to_resolve (no mutation)
  GET  /api/sovereign/leaderboard -> 200 (localhost tooling keeps working post-change)

MODE=warn: 3/3 (_warn_if_allowlist_empty: fail-open WARNING / fail-closed WARNING / silent when non-empty)
```

Frontend consumers need zero changes (all via authed `apiFetch`); Slack bot calls none of the pruned prefixes. Full table in experiment_results_75.1.md.
