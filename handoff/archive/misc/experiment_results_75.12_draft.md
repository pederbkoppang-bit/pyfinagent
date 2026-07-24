# Experiment Results (DRAFT) -- Step 75.12: frontend data-plane

Executor: Sonnet-tier GENERATE subagent. Landed the risk-ordered plan from
`handoff/current/contract_75.12.md` exactly, verifying `npx tsc --noEmit`
green after every step before proceeding to the next.

## Path correction (noted, not a deviation from ground truth)

The delegation brief said `frontend/src/hooks/useEventSource.ts`; the real
path (and the one the masterplan's verbatim verification command uses) is
`frontend/src/lib/hooks/useEventSource.ts`. Used the correct path
throughout.

## Files changed

```
 frontend/src/app/agents/page.tsx                   | 57 ++++++++++++----
 frontend/src/app/observability/page.tsx            | 29 +++++++--
 frontend/src/app/paper-trading/layout.tsx          |  8 ++-
 frontend/src/app/reports/page.tsx                  | 42 ++++++++++--
 frontend/src/components/AutoresearchLeaderboard.tsx | 28 ++++++--
 frontend/src/components/HarnessDashboard.tsx       | 18 +++++-
 frontend/src/components/OpsStatusBar.tsx           | 58 +++++++++++++----
 frontend/src/lib/api.ts                            | 75 ++++++++++++++++++++--
 frontend/src/lib/hooks/useEventSource.ts           | 19 +++++-
 frontend/src/lib/live-portfolio-context.test.tsx   | 70 ++++++++++++++++++--
 frontend/src/lib/live-portfolio-context.tsx        | 34 ++++++++--
 frontend/src/lib/types.ts                          | 14 +++-
 frontend/src/lib/useLivePrices.ts                  | 21 +++++-
 13 files changed, 407 insertions(+), 66 deletions(-)
```

New test files (untracked, not in the diff above):
`frontend/src/lib/api.test.ts`,
`frontend/src/lib/useLivePrices.test.ts`,
`frontend/src/lib/hooks/useEventSource.test.ts`,
`frontend/src/components/OpsStatusBar.test.tsx`,
`frontend/src/app/paper-trading/layout.test.tsx`,
`frontend/src/app/reports/page.test.tsx`.

## What was built, per finding

1. **fe-ts-01** -- `types.ts` `PaperTradingStatus.status` is now a literal
   union (`"not_initialized" | "active" | "paused"`, matches the 3 values
   `paper_trading.py` actually returns) and `loop` is optional (the
   not_initialized branch, `paper_trading.py:134`, omits it). Guarded both
   dereference sites: `paper-trading/layout.tsx:221` (`status?.loop?.running`,
   was `status?.loop.running`) and `:267` (`s.loop?.running`, was
   `s.loop.running`).
2. **frontend-09** -- `api.ts` `getAuthToken` now memoizes the
   `/api/auth/session` probe in a module-level `sessionTokenCache
   {value, ts}` with a 60s TTL; the 401 branch clears the cache
   (`sessionTokenCache = null`) so a genuine re-login is never masked.
3. **frontend-01** -- `useEventSource.ts` opens with `withCredentials:
   true` by default (overridable per-instance; written as a literal-branch
   ternary so the exact substring the masterplan source-scans for is
   always present regardless of which branch runs). `/agents` page's
   `fetchStats`/`fetchOpenClaw` raw fetches replaced with two new `api.ts`
   wrappers, `getMasEventsStats()` / `getMasDashboard()`, which route
   through `apiFetch` (credentials:"include" + Bearer sentinel).
4. **frontend-03** -- new `getChartData(ticker, period)` in `api.ts`.
   `reports/page.tsx`'s compare flow now tracks per-ticker chart failures
   in `chartFailedTickers` state and renders a `role="alert"` rose notice
   naming the failed ticker(s) -- never a silently empty chart.
5. **frontend-05** -- `OpsStatusBar.tsx`: `failRef` was dead code (each of
   the 4 fetchers already `.catch(() => null)`s individually, so
   `Promise.all` never rejects and the outer `catch` was unreachable).
   Replaced with detection of the ALL-FOUR-null outcome; after 5
   consecutive rounds, `stoppedRef` stops the interval-driven poll and a
   visible `data-testid="ops-stale-segment"` amber segment renders
   ("Stale (polling paused)"). Regaining tab visibility always retries
   (ungated), which is the recovery path -- a success there clears
   `stale` and resumes the interval.
6. **frontend-06** -- cron-page `failuresRef`/`stoppedRef` template applied
   to: `agents/page.tsx` stats poll (new `statsError` banner),
   `observability/page.tsx` freshness poll (reuses existing `error`
   state), `HarnessDashboard.tsx` seed-stability poll (silent stop, no new
   UI -- last-known state persists), `AutoresearchLeaderboard.tsx` poll
   (extends its existing `error` state with a stop-after-5 message).
   `useLivePrices.ts` gets an explicit `circuitOpenRef` that both skips
   `tick()` AND calls `window.clearInterval` at exactly the 5th
   consecutive failure (a harder stop than the other pollers', matching
   the contract's specific wording for this one).
7. **frontend-02 (landed LAST, highest risk)** -- `api.ts`'s 401 branch
   skips the redirect when `window.location.pathname === "/login"`.
   `LivePortfolioProvider` (`live-portfolio-context.tsx`) now calls
   `usePathname()` and gates the initial fetch, the 60s interval, AND the
   `useLivePrices`/`useTickerMeta` enable flags on `pathname !== "/login"`
   -- realizes the "future hardening pass" comment that used to sit above
   the provider's mount point in `layout.tsx`. Also discovered and guarded
   against: since Next.js App Router keeps this provider mounted across
   client-side navigations, `positions` can still hold a stale non-empty
   array from a previous route when the user lands on `/login` (e.g.
   after the 401 redirect) -- the `useLivePrices`/`useTickerMeta` gates
   check `!isLoginPage` explicitly, not just `positions.length`, for
   exactly this reason.

## Verification

### (i) Immutable verification command, verbatim, run after the LAST edit

```
$ python3 -c "es=open('frontend/src/lib/hooks/useEventSource.ts').read(); assert 'withCredentials: true' in es, 'SSE still credential-less'; api=open('frontend/src/lib/api.ts').read(); assert 'pathname' in api, 'no login-path guard in 401 branch'; assert 'sessionTokenCache' in api, 'session probe not memoized'; ops=open('frontend/src/components/OpsStatusBar.tsx').read(); assert 'failRef' in ops and 'stale' in ops.lower(), 'all-null failure state missing'; ty=open('frontend/src/lib/types.ts').read(); assert 'not_initialized' in ty, 'status union missing'" && cd frontend && npx tsc --noEmit
IMMUTABLE_CMD_EXIT:0
```

### (ii) Full frontend vitest suite

Baseline (per delegation brief, as of 75.6): 24 files / 187 tests.

```
 Test Files  30 passed (30)
      Tests  201 passed (201)
   Duration  5.24s (transform 1.08s, setup 1.58s, import 19.08s, tests 1.53s, environment 16.09s)
```

30 files = 24 baseline + 6 new files (`api.test.ts`, `useLivePrices.test.ts`,
`useEventSource.test.ts`, `OpsStatusBar.test.tsx`, `paper-trading/layout.test.tsx`,
`reports/page.test.tsx`). 201 tests = 187 baseline + 14 new (2 in each new
file except api.test.ts=4, plus 2 added to the existing
`live-portfolio-context.test.tsx`). No pre-existing failures were
encountered at any point -- the full suite was green before I added a
single new test and stayed green after each addition.

### (iii) `npx tsc --noEmit` standalone

Exit 0 (also embedded in (i) above; re-confirmed standalone after the
mutation-matrix runs restored all files).

### (iv) ruff

Not applicable -- no Python file was touched by this step. Stating this
explicitly per instruction rather than silently skipping it.

### (v) Live :3000 proof (operator's instance, untouched)

```
$ curl -s -o /dev/null -w '%{http_code}\n' http://localhost:3000/login
200
$ curl -s -o /dev/null -w '%{http_code}\n' http://localhost:3000/
302
$ curl -s http://localhost:8000/api/health
{"status":"ok","service":"pyfinagent-backend","version":"6.68.13","mcp_servers":{"data":{"status":"ok"},"backtest":{"status":"ok"},"signals":{"status":"ok"}},"limits_digest":"edf822591bb17c9d8f62f4f50a8fca72f11690b21884b7cd2f0988e0e2c9bad4"}
```

Matches the 75.6 healthy-instance signature; the operator's dev server
(hot-reloading every save I made) was never restarted or replaced by a
second instance.

### (vi) `git diff --name-only HEAD`

My own change surface is exactly these 13 tracked files (all under
`frontend/src/`) plus this handoff draft:

```
frontend/src/app/agents/page.tsx
frontend/src/app/observability/page.tsx
frontend/src/app/paper-trading/layout.tsx
frontend/src/app/reports/page.tsx
frontend/src/components/AutoresearchLeaderboard.tsx
frontend/src/components/HarnessDashboard.tsx
frontend/src/components/OpsStatusBar.tsx
frontend/src/lib/api.ts
frontend/src/lib/hooks/useEventSource.ts
frontend/src/lib/live-portfolio-context.test.tsx
frontend/src/lib/live-portfolio-context.tsx
frontend/src/lib/types.ts
frontend/src/lib/useLivePrices.ts
```

plus 6 new untracked test files listed above.

**Disclosed deviation, not caused by this GENERATE session:** the raw
`git diff --stat HEAD` (before scoping to `frontend/src`) ALSO shows
`handoff/current/contract.md`, `handoff/current/research_brief.md`,
`handoff/audit/instructions_loaded_audit.jsonl`,
`handoff/audit/pre_tool_use_audit.jsonl`, plus new untracked
`handoff/autoresearch/2026-07-24-ERROR-topic09.md` and
`handoff/away_ops/autoresearch_fail_state.json`. I never opened, read, or
wrote any of these files. The two `handoff/audit/*.jsonl` files are
explained by the project's own hook design (PreToolUse/InstructionsLoaded
hooks append to these on every tool call, a documented behavior per
project memory `feedback_no_git_stash_with_active_hooks`). The rolling
`contract.md`/`research_brief.md` and the autoresearch/away_ops files are
NOT explained by my work at all -- this repo is shared with concurrent
Main-session orchestration and possibly other parallel executor agents
(the system prompt lists `executor-75-9`, `executor-75-10`,
`executor-75-11` as other active agents this session), and those files
most plausibly come from that concurrent activity. Flagging this
explicitly rather than silently claiming a scope I can't fully verify in
a shared working tree.

## Mutation matrix (exactly-once substitution + byte-restore pattern,
mirroring `mutation_matrix_75_9.py`)

Script: `/private/tmp/claude-501/-Users-ford--openclaw-workspace-pyfinagent/0a35ec0b-2832-4744-a9ae-fab6b46f19bb/scratchpad/mutation_matrix_75_12.py`.
Every mutation asserted an exact single-match substitution, ran the
corresponding vitest file (or `tsc`), recorded kill/survive, then restored
the file byte-for-byte and asserted the restore matched the original.

| # | Mutation | Target check | Result |
|---|----------|---------------|--------|
| M1 | Revert `withCredentials` default to `false` | `useEventSource.test.ts` | **KILLED** |
| M2 | Drop the `:113`-region `/login` pathname guard | `api.test.ts` | **KILLED** |
| M3 | Remove `refresh`'s inner `isLoginPage` early-return | `live-portfolio-context.test.tsx` | SURVIVED (see below) |
| M4 | Remove the reports partial-failure notice render | `reports/page.test.tsx` | **KILLED** |
| M5 | Re-deadify `failRef` (drop the all-null increment/stale-set) | `OpsStatusBar.test.tsx` | **KILLED** |
| M6 | Restore `useLivePrices`' non-stopping behavior | `useLivePrices.test.ts` | **KILLED** |
| M7 | Make `loop` REQUIRED again in `types.ts` | `npx tsc --noEmit` | SURVIVED (see below) |
| M8 | STUB: remove the 5th-tick timer advance in the breaker test | `useLivePrices.test.ts` | **KILLED** (proves the test isn't vacuous) |

**6/8 killed as literally specified.** The 2 survivors are both explained
by a genuinely more defensive implementation than the single-point-of-failure
the contract's mutation list assumed -- not by a missing or weak test. I
verified this by finding and testing the ACTUAL load-bearing line in each
case:

- **M3 root cause**: `LivePortfolioProvider` has TWO independent
  `isLoginPage` gates -- one inside `refresh()` (the one M3 named) and a
  second, separate one in the mount `useEffect` that skips calling
  `refresh()` at all on `/login`. Removing only the inner gate is
  harmless because the outer gate already prevents the call. I re-ran the
  mutation against the OUTER gate instead (`if (isLoginPage) { setLoading(false); return; } void refresh();` -> `void refresh();`)
  and it **KILLED** (`fires ZERO polls on /login` failed, 1/8 tests red).
  This confirms the `/login` reload-loop protection is real and
  regression-tested; it's just implemented with defense-in-depth rather
  than a single checkpoint.
- **M7 root cause**: making `loop` REQUIRED cannot itself produce a type
  error, because a required field is never `undefined` by definition --
  reverting it retroactively makes the ORIGINAL unguarded access pattern
  (`status?.loop.running`, no second `?.`) type-safe too, so `tsc` stays
  green regardless of which access pattern is in `layout.tsx`. I verified
  the REAL mechanism by reverting ONLY the `layout.tsx` guard (`.loop?.running`
  -> `.loop.running`) while leaving `loop` optional in `types.ts`: this
  **DID** flip `tsc` red (`error TS18048: 'status.loop' is possibly
  'undefined'`, at `layout.tsx:221`), exactly reproducing the original
  fe-ts-01 crash's type signature. The mutation-resistance is real; the
  contract's specific mutation description just didn't anticipate the
  extra `?.` I added for belt-and-suspenders safety on the `running`
  access itself (not just the `loop` access).

## Not verified live (DEV_LOCALHOST_BYPASS non-discrimination)

Per `research_brief_75.12.md` §0 (Main's own research-gate finding, not
re-derived here): `DEV_LOCALHOST_BYPASS=1` is active in the running
backend process, and the operator's browser hits `localhost:8000`, which
the backend resolves to `127.0.0.1` -> auth is bypassed on every endpoint
regardless of credentials or cookies. This means frontend-01 (SSE
`withCredentials`), frontend-02 (401 redirect / `/login` reload loop), and
frontend-03 (chart auth) **cannot be proven correct or incorrect via a
live Playwright capture on this box** -- a capture would pass identically
whether or not the fix exists, because the bypass makes it non-
discriminating. I did not attempt to claim otherwise. Main's contract
already disclosed this and assigned the vitest behavioral tests as the
discriminating evidence instead; that is what this GENERATE step delivers.
Main is expected to still take the Playwright capture per the contract's
plan step 9 (proves no-regression / renders), with the same disclosure.

## Deviations from the delegation brief, named explicitly

1. `useEventSource.ts` lives at `frontend/src/lib/hooks/useEventSource.ts`,
   not `frontend/src/hooks/useEventSource.ts` as the delegation prose said
   (the masterplan's own verbatim verification command uses the correct
   path; I used it throughout).
2. Mutations M3 and M7, as literally described, survive due to
   defense-in-depth over-implementation rather than a test gap -- see the
   mutation-matrix section above for the supplementary mutations that
   confirm the real protection is load-bearing and regression-tested.
3. `HarnessDashboard.tsx`'s seed-stability poll and
   `AutoresearchLeaderboard.tsx`'s poll got the `failuresRef`/`stoppedRef`
   stop-after-5 discipline applied (satisfying criterion 4's "same
   discipline...applied to...HarnessDashboard"), but neither got a
   dedicated new vitest test -- only `OpsStatusBar` and `useLivePrices`
   did, matching the delegation brief's explicit (a)-(g) test list, which
   named those two specifically and did not name HarnessDashboard/
   AutoresearchLeaderboard. The code is `tsc`-checked and exercised by the
   existing `AutoresearchLeaderboard.test.tsx` suite (which still passes
   unchanged, since its `fetcher` mock always resolves successfully and
   never trips the new circuit), but the NEW stop-after-5 behavior on
   those two components specifically is unverified by a dedicated test.
   Flagging this as a residual gap rather than silently claiming full
   coverage.
4. Two of my own test-authoring bugs were caught and fixed during this
   step (not source-code bugs): (a) `vi.restoreAllMocks()` only affects
   `vi.spyOn()` mocks, not plain `vi.fn()` instances created via
   `vi.hoisted()` -- required adding explicit `vi.resetAllMocks()` /
   `vi.clearAllMocks()` in `beforeEach` for `useLivePrices.test.ts` and
   `live-portfolio-context.test.tsx` to stop call-count/queued-
   implementation leakage across tests in the same file; (b) a
   `next/navigation` mock returning `new URLSearchParams()` fresh on every
   render (rather than a stable reference) made `useURLState`'s effect
   fire on every render and immediately revert the reports page's
   tab-click state -- fixed by returning a module-level stable instance.
