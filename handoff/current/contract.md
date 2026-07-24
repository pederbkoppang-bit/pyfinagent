# Contract -- Step 75.12: frontend data-plane (SSE/fetch auth transport, login reload-loop, dead charts, circuit breakers, status union)

- **Step id**: 75.12 (phase-75, Audit75 S12) -- P1, executor sonnet-tier
- **Date**: 2026-07-24
- **Author**: Main (contract + review + Playwright capture). **GENERATE delegated to a Sonnet-4.6 executor** (4th delegated step; Main reviews + independently re-measures before Q/A).
- **BOUNDARY**: UI-touching -- the operator's `next dev` on :3000 hot-reloads every edit IMMEDIATELY. NEVER start a second dev server (standing memory); the Playwright capture is read-only against the operator instance (75.6 pattern, no LIGHTHOUSE_SKIP_AUTH). Land order is risk-ordered (below). No emoji; Phosphor icons via @/lib/icons; navy/slate palette; scrollbar-thin.

## Research-gate summary (gate PASSED)

Workflow `wf_edd47d0e-32d` (researcher, opus/max, tier=moderate).
Envelope: `external_sources_read_in_full=6, snippet_only=20, urls=26, recency_scan=true, internal_files=20, gate_passed=true`.
Brief: `handoff/current/research_brief_75.12.md`.

**Step-text corrections adopted (binding):**
1. **THE HEADLINE**: `DEV_LOCALHOST_BYPASS=1` is ACTIVE in the running backend (verified: no-cred curls to the three authed endpoints all 200) and the operator's browser hits localhost -> **frontend-01/02/03 CANNOT be reproduced live on this box**; the criterion-1 Playwright "connected stream" capture is VACUOUS as fix evidence. Discriminating proof = vitest behavioral tests; the capture is still taken (criterion requires it) and DISCLOSED as non-discriminating.
2. The cookie-auth premise is otherwise SOUND end-to-end: EventSource cannot send Authorization headers (spec); `get_current_user` reads the NextAuth cookie cross-origin (auth.py:169-189); CORS `allow_credentials=True` (main.py:488); localhost:3000/:8000 are same-site (port not in cookie scope) -- no SameSite change needed.
3. `apiFetch` ALREADY sends `credentials:"include"` (:87) -- re-routing raw fetches through it is the whole credentials fix. The 401 redirect is at api.ts **:113** (not :114) and fires as a SIDE EFFECT regardless of caller error handling -- why the root-mounted, ungated LivePortfolioProvider (layout.tsx:36, 3 authed polls on mount + 60s) loops /login.
4. **fe-ts-01 is a REAL fresh-install crash**: backend returns `{status:'not_initialized'}` with NO `loop` key (paper_trading.py:134); types.ts:716 types `loop` REQUIRED so tsc is blind to `status?.loop.running` (layout.tsx:217) / `s.loop.running` (:263). Making `loop` optional turns `npx tsc --noEmit` into the mutation test.
5. useEventSource ALREADY caps reconnects (maxFailures 3 default :63; /agents overrides 5 :191) -- the stop-at-5 gap is **useLivePrices.ts:48-53** (interval keeps firing; the doc comment lies). OpsStatusBar `failRef` (:74) is FULLY dead code (per-call `.catch(()=>null)` at :79-82 -> Promise.all never rejects -> the :89 catch never runs AND failRef is never read).
6. The immutable command is a python3 source-scan + `npx tsc --noEmit` ONLY -- every asserted token is trivially writable without correct behavior; the vitest behavioral legs + mutation matrix carry the verification weight.
7. No chart wrapper exists today -- frontend-03 needs a new `getChartData` in api.ts.
8. Live state healthy: :3000/login 200, :3000/ 302, :8000 health ok; Playwright pinned @0.0.76; vitest is the runner with existing @/lib/api + window.location mock precedents.

## Hypothesis

All seven data-plane defects are fixable client-side with zero backend change and zero visible regression on the operator's live UI, with correctness carried by vitest behavioral tests (fake timers for the breakers; navigation/no-navigation asserts for the loop) because the localhost bypass makes live captures non-discriminating.

## Immutable success criteria (copied VERBATIM from .claude/masterplan.json step 75.12)

verification.command (python3 source-scan asserts + tsc; copied verbatim in the masterplan node -- executor must read it there in full):
```
cd /Users/ford/.openclaw/workspace/pyfinagent && python3 -c "..." (source-scan) && cd frontend && npx tsc --noEmit
```

1. "useEventSource opens with withCredentials: true and the /agents stats+dashboard fetches go through apiFetch (or carry credentials:'include'); live evidence: a Playwright capture of /agents against the running app"
2. "apiFetch's 401 branch does not redirect when already on /login, and LivePortfolioProvider does not fire its poll trio on /login (pathname/session gate) -- no reload loop during a logged-out /login visit"
3. "Reports compare flow fetches charts via apiFetch and renders a visible partial-failure notice when price series are unavailable (never a silently empty chart)"
4. "OpsStatusBar: all-four-null results increment failRef and after 5 consecutive failures a stale/error segment state renders and polling stops or backs off -- failRef is no longer dead code; the same stale/backoff pattern applied to the other named pollers"
5. "types.ts declares the not_initialized union for PaperTradingStatus and paper-trading/layout.tsx guards .loop access -- a fresh/reset install renders the section without a TypeError"
6. "getAuthToken caches the session probe in sessionTokenCache with a TTL and invalidates on 401; npx tsc --noEmit passes"

(NOTE: Main re-read the criteria verbatim from the masterplan node; the executor must too -- the list above is Main's faithful transcription of the criterion intents; where wording differs the masterplan text governs.)

verification.live_check: "handoff/current/live_check_75.12.md: verbatim verification command output (exit 0) + git diff --stat; UI-touching -> Playwright capture (read-only, operator instance) with the DEV_LOCALHOST_BYPASS non-discrimination disclosure."

## Plan steps (RISK-ORDERED for hot-reload safety -- research rec)

1. **fe-ts-01 FIRST, atomically**: types.ts `PaperTradingStatus` status literal union + `loop?` optional AND the layout.tsx:217/:263 optional-chaining guards in the SAME edit burst (types-only first would throw a red tsc overlay on the live app).
2. **frontend-09**: module-level `sessionTokenCache={value,ts}` with ~60s TTL in getAuthToken; INVALIDATE (clear) in the 401 branch.
3. **frontend-01**: `{ withCredentials: true }` at useEventSource:91 (as an overridable option defaulting true); re-route agents fetchStats/fetchOpenClaw (:221/:229) through apiFetch. Inert on this box (bypass) -- no stale claims.
4. **frontend-03**: new `getChartData(ticker)` -> apiFetch in api.ts; reports compare flow catches per-ticker and renders a rose partial-failure notice (border-rose-500/30 bg-rose-950/30, IconWarning from @/lib/icons) -- never a silently empty chart.
5. **frontend-05/06**: cron-page failuresRef+stoppedRef template applied to OpsStatusBar (detect ALL-FOUR-null -> failRef++ -> after 5: stale/error segment renders + polling stops/backs off; keep per-call graceful degrade), agents stats/dashboard, observability freshness, HarnessDashboard seed-stability, AutoresearchLeaderboard; useLivePrices gets an explicit circuitOpen that SKIPS tick and CLEARS the interval at exactly 5.
6. **frontend-02 LAST (highest live risk)**: api.ts:113 skip redirect when `window.location.pathname === '/login'`; LivePortfolioProvider gated via usePathname to skip mount+interval polls on /login. Provider-gate vitest test lands with it.
7. **Vitest behavioral tests** (the discriminating evidence): 401-on-/login does NOT navigate; provider fires zero polls on /login (and normal polls elsewhere); useLivePrices stops at EXACTLY 5 (fake timers; 4 fails -> still polling); OpsStatusBar renders stale after 5 all-null rounds and recovers on success; not_initialized payload renders without throw; sessionTokenCache TTL + 401 invalidation.
8. **Mutation matrix**: revert withCredentials; drop the pathname guard; un-gate the provider; remove the notice render; re-deadify failRef; restore useLivePrices' non-stopping interval; make `loop` required again (tsc must go red); stub mutation -- break the fake-timer clock so the breaker test would pass vacuously (test must fail).
9. **Main takes the Playwright capture** (read-only /agents against :3000) after the executor lands -- stored under handoff/current/captures_75.12/ with the non-discrimination disclosure.
10. **live_check_75.12.md** per the spec above.

## Explicitly NOT in scope

- Any backend change (the SSE auth path is already correct server-side)
- Disabling/altering DEV_LOCALHOST_BYPASS (operator env)
- A second dev server or any :3000 restart (hot-reload only)
- The /login console error triage (queued 75.6.2)

## References

- `handoff/current/research_brief_75.12.md` (6 read-in-full: WHATWG/MDN EventSource + fetch credentials + cookie same-site scope, NextAuth v5 session-cookie handling, circuit-breaker patterns, TS discriminated unions)
- `.claude/rules/frontend.md` + `frontend-layout.md`; feedback_second_next_dev_breaks_operator_3000
- `handoff/current/audit_phase75/confirmed_findings.json` (frontend-01/02/03/05/06, fe-ts-01, frontend-09)
