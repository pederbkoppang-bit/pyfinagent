# Live check -- phase-82.5 (exit-quality tiles)

Produced because the 82.5 cycle-1 Q/A [CONDITIONAL] found two things I had not
verified: qa.md 1c requires a LIVE UI capture for a diff that changes what a card
renders, and my `experiment_results` section 5 was headed **"What the tiles now
report"** while the running backend still served the PRE-FIX numbers. It was right on
both counts.

## A. Method (canonical .claude/rules/frontend.md workflow)

- The operator's :3000 instance was **never touched**. A second, isolated dev server
  was started exactly as `frontend/playwright.config.ts:126-141` specifies:
  `LIGHTHOUSE_SKIP_AUTH=1 NEXT_PUBLIC_E2E_TESTING=true PLAYWRIGHT_DIST_DIR=.next-functional npx next dev --port 3100`
  -- `npx next dev` directly, NOT `npm run dev`, because the latter carries a
  `predev: rm -rf .next` that would delete the operator's shared build and 500 the
  running cockpit.
- Captured with the Playwright MCP (`@playwright/mcp@0.0.76` per `.mcp.json`),
  1440x900 viewport. Same code, same backend (:8000), same BigQuery data as :3000.
- **SIDE EFFECT I FAILED TO DISCLOSE ORIGINALLY (found by the cycle-2 Q/A):** starting
  the :3100 server with `PLAYWRIGHT_DIST_DIR=.next-functional` makes Next **rewrite two
  TRACKED files** -- `frontend/next-env.d.ts` and `frontend/tsconfig.json` -- to point at
  the gitignored `.next-functional/` build dir. `playwright.config.ts:40` declares a
  `globalTeardown` whose entire job is to restore them, but starting `next dev` DIRECTLY
  (rather than through Playwright's runner) means that teardown never runs. Both files
  were restored from HEAD and `tsc --noEmit` re-verified against the restored config
  (0 errors). **Anyone repeating this workflow must restore those two files afterwards,
  or run the capture through Playwright so the teardown fires.**
- :3100 was killed after the capture. Verified afterwards: `:3100 -> 000` (down),
  `:3000/login -> 200` (healthy), launchd `com.pyfinagent.frontend` pid 863 intact.

## B. THE BACKEND WAS STALE -- this is the part I got wrong

The Q/A measured backend PID **654, started 2026-07-28 18:39:22, with no `--reload`**.
Every module I changed was written 2026-08-04. So the live service was executing
seven-day-old code and still returning `avg_capture_ratio: -42.0785`. My results file
claimed the new values in the present tense. That is the
`feedback_verify_own_completed_action_claims` class: a claim about a running system
that was never checked against that system.

Restarted via `launchctl kickstart -k gui/$(id -u)/com.pyfinagent.backend` (launchd-
managed; not a bare kill). New pid **62664**. Health 200 after 4s.

### Before restart (what the Q/A saw)
```
edge_ratio        : 86.9218
avg_capture_ratio : -42.0785
new keys present  : NONE
```

### After restart -- LIVE `GET /api/paper-trading/mfe-mae-scatter`
```json
{
  "edge_ratio": 3.09,
  "avg_capture_ratio": 0.6304,
  "aggregation": "median",
  "min_mfe_pct": 1.0,
  "capture_n_defined": 20,
  "capture_n_undefined": 12,
  "edge_n_infinite": 6,
  "n_points": 32,
  "n_leakers": 0
}
```
`points=32`, of which **12 carry `capture_ratio: null`** -- the undefined rows are
now transmitted as null rather than as a fabricated 0.0.

## C. UI capture

`handoff/current/captures_82.5/live_check_82.5_exit_quality_tiles.png`
(Paper Trading -> Exit quality tab, rendered from the restarted backend)

| Tile | Renders | Hint | Was |
|---|---|---|---|
| EDGE RATIO | **3.09** | `median(MFE / \|MAE\|)` | 86.92 |
| AVG CAPTURE | **63%** | `median(realized_pnl / MFE), n=20` | **-4208%** |
| ROUND-TRIPS | 32 | closed only | 32 |
| LEAKERS | 0 | capture < 40% & MFE > P75 | -- |

Three things the capture proves that no unit test could:

1. The tile reads **63%**, not -4208%. The x100 in `MfeMaeScatter.tsx:114` is retained
   and correct -- `capture_ratio` is percent/percent, i.e. dimensionless.
2. Both hints now say **median**, not mean, so the estimator change is visible to the
   operator rather than silently altering a number under an unchanged label.
3. The capture hint carries **n=20** -- the defined-subset count reached the UI, so a
   reader can see the headline is computed over 20 of 32 round-trips rather than
   silently over all of them.

## D. Scope honesty

- LEAKERS reads 0. That is a real consequence of this change, not a bug: the leakage
  rule now requires a DEFINED capture, so a trade with no gradeable exit can no longer
  be flagged as a leaking exit. Previously the fabricated 0.0 satisfied `< 0.4`.
- The operator status bar shows `KILL: PAUSED`. Pre-existing machine state, unrelated
  to 82.5 -- it is also the root cause the Q/A traced for the one pre-existing test
  failure.
- The screenshot was taken through the skip-auth middleware bypass, so it does not
  exercise the NextAuth session path. Standard for this workflow and disclosed.
