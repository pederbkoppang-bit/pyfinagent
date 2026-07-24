# Experiment results -- Step 75.12 (frontend data-plane)

Date: 2026-07-24. **Execution model: GENERATE delegated to a Sonnet-4.6
executor (4th delegated step); Main wrote the contract, reviewed, took the
Playwright capture, and independently re-measured every headline figure.
Executor draft preserved at
`handoff/current/experiment_results_75.12_draft.md`.**

## What was built (contract plan steps 1-6, landed risk-ordered on the
hot-reloading operator dev server -- :3000 verified healthy after every burst)

- **fe-ts-01** (real fresh-install crash): `PaperTradingStatus` status
  literal union incl. `not_initialized` + `loop?` optional in types.ts,
  atomically with the layout.tsx guards (`status?.loop?.running`,
  `s.loop?.running`). tsc is now the guard (proven: reverting only the
  layout guard flips `tsc` red with TS18048 -- the original crash's type
  signature).
- **frontend-09**: module-level `sessionTokenCache` (~60s TTL) in
  getAuthToken; cleared in the 401 branch.
- **frontend-01**: `withCredentials` (default true, overridable) on the
  EventSource open; /agents stats + dashboard fetches re-routed through
  apiFetch (which already carries `credentials:'include'`). INERT on this
  box (see non-discrimination below).
- **frontend-03**: new `getChartData()` -> apiFetch; reports compare flow
  renders a rose partial-failure notice (IconWarning via @/lib/icons,
  ASCII) naming failed tickers -- never a silently empty chart.
- **frontend-05/06**: cron failuresRef+stoppedRef template applied to
  OpsStatusBar (all-four-null detection -> failRef LIVE -> stale/error
  segments + stop/backoff after 5, recovery on success), agents
  stats/dashboard, observability freshness, HarnessDashboard,
  AutoresearchLeaderboard; useLivePrices now ACTUALLY stops at exactly 5
  (circuitOpen + clearInterval -- its doc comment is finally true).
- **frontend-02** (landed LAST): api.ts 401 branch skips the redirect on
  `/login` (guard at api.ts:146-147); LivePortfolioProvider gated via
  usePathname with defense-in-depth (outer mount/effect gate + inner
  refresh gate).

## Change surface (measured)

`git diff --stat HEAD -- frontend/`: **13 files, +407/-66** + **6 NEW test
files** (suite 24 -> 30 files, 187 -> 201 tests). ZERO backend files, zero
.env, zero masterplan edits by the executor. Also in the tree (runtime,
NOT step changes): tonight's nightly-autoresearch failure artifacts
(`handoff/autoresearch/2026-07-24-ERROR-topic09.md` -- arXiv HTTP 429 --
and `handoff/away_ops/autoresearch_fail_state.json` `consecutive_fails: 1`)
-- notable because the fail-state counter is the **75.11 paging seam
working live on its first night**.

## Verification (ALL figures independently re-measured by Main)

- Immutable command (python3 source-scan + `npx tsc --noEmit`), run
  VERBATIM from the masterplan node: **exit 0**.
- Full vitest suite: **30 files / 201 tests, all passed** (two independent
  Main runs; one transient rc=1 immediately after a mutation restore was
  re-run green -- hot-reload recompile race, disclosed).
- Live operator instance after ALL edits: `/login` 200, `/` 302; Playwright
  (read-only): navigating to **/agents redirects to /login** (auth wall
  intact) and /login renders STABLE across ~30s of interactions (no reload
  thrash); capture at `handoff/current/captures_75.12/agents_authwall_75.12.png`.
  The 1 console error on /login is the pre-existing queued 75.6.2 item.
- Mutation matrix: executor 6/8 killed as-specified + **2 disclosed
  invalid mutants replaced by the real load-bearing mutation, both
  KILLED**: M3's named line was redundant (defense-in-depth double gate;
  the OUTER gate mutation kills), M7's spec was type-theoretically inert
  (a required field cannot be undefined; the layout-guard revert flips
  tsc red instead). M8 stub mutation killed (fake timers load-bearing).
  Main independently spot-checked **M2 (drop the pathname guard) KILLED**.

## NON-DISCRIMINATION DISCLOSURE (research headline, applies to all live evidence)

`DEV_LOCALHOST_BYPASS=1` is active in the running backend and the
operator's browser hits localhost, so the auth-transport defects
(frontend-01/02/03) cannot reproduce live on this box -- authed endpoints
return 200 without credentials here. The Playwright captures therefore
evidence UI health and the auth wall, NOT the fixes; the vitest behavioral
suite (both-directions redirect tests, zero-polls-on-login, stop-at-
exactly-5 fake-timer tests, not_initialized render) is the discriminating
evidence, per the contract.

## Not verified live

- The SSE credentialed connection and the reload-loop fix behave
  identically on this box with or without the fix (bypass); off-localhost
  (Tailscale) verification would discriminate but requires an operator
  session from a non-local origin -- documented, not performed.
- No backend change; no restart needed for this step (frontend hot-reloaded).
