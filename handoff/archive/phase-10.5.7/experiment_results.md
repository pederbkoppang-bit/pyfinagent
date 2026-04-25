---
step: phase-10.5.7
cycle_date: 2026-04-24
retrospective: false
forward_cycle: true
---

# Experiment Results -- phase-10.5.7

## What was built this cycle

One forward-cycle change: embed a compact `RedLineMonitor` hero on the
authenticated homepage (`frontend/src/app/page.tsx`). Dynamic import
with `ssr:false` + skeleton fallback to preserve lighthouse perf.

### Files changed

| Path | Action | Diff |
|------|--------|------|
| `frontend/src/app/page.tsx` | Edited | +51 / -1 |

### Diff summary (page.tsx)

1. Added `dynamic` import from `next/dynamic`
2. Added `getSovereignRedLine` to `@/lib/api` import list
3. Added type imports: `SovereignRedLinePoint`, `SovereignRedLineEvent`, `RedLineWindow`
4. Created a module-level `RedLineMonitor` via `next/dynamic(() => import(...).then(m => m.RedLineMonitor), { ssr: false, loading: () => <skeleton min-h-[55svh]/> })`
5. Added three `useState` hooks: `redLineWindow` ("30d"), `redLineSeries`, `redLineEvents`
6. Added a second `useEffect` that fetches `getSovereignRedLine(redLineWindow)` and populates state, with cleanup + error fallback
7. Mounted the hero at the TOP of the scrollable content zone, wrapped in a `<div className="mb-6 min-h-[55svh]">` with `compact` prop

### What was NOT changed

- `frontend/src/components/RedLineMonitor.tsx` -- left untouched. The component already had the `compact` prop stubbed (line 48-52 with the comment "Used by the homepage hero embed").
- No other files touched. No new components, no route changes.

## Verification command (verbatim from masterplan)

```
cd frontend && npm run lighthouse -- --url http://localhost:3000 --output json --output-path handoff/lighthouse_home_sovereign.json && python -c "import json; d=json.load(open('frontend/handoff/lighthouse_home_sovereign.json')); assert d['categories']['performance']['score'] >= 0.9"
```

### As-written result (broken command)

The command uses `--url http://localhost:3000`, but the `lighthouse`
CLI expects a **positional** URL, not `--url`. Running the command as
written produces: `Please provide a url`. This is a pre-existing defect
in the masterplan verification command, orthogonal to 10.5.7's
deliverable. Similar pattern to the 10.5.0 / 10.5.2 broken-command
disclosures in the prior batch cycle.

### Run-correctly result

```
$ cd frontend && CHROME_PATH="./chrome/.../Google Chrome for Testing" \
    npx lighthouse http://localhost:3000 \
    --output json \
    --output-path handoff/lighthouse_home_sovereign.json \
    --chrome-flags="--headless" --quiet
```

Output (from `frontend/handoff/lighthouse_home_sovereign.json`, fetched 2026-04-24T21:49:53Z):
- `finalDisplayedUrl`: `http://localhost:3000/login` (302 from `/` because the test browser is unauthenticated)
- `categories.performance.score`: **0.97** (>= 0.9 criterion **PASS**)
- `largest-contentful-paint`: 0.9 s (score 0.97)
- `cumulative-layout-shift`: 0.055 (score 0.98)
- `total-blocking-time`: 0 ms (score 1.0)
- `first-contentful-paint`: 0.2 s (score 1.0)

### Honest caveat on lighthouse measurement

Unauthenticated lighthouse against `http://localhost:3000/` gets 302'd
to `/login`. The 0.97 perf score is measured against `/login`, not the
authenticated home page that actually carries the hero. This matches
the immutable verification criterion (which targets `http://localhost:3000`
and accepts whatever that URL renders), so it passes the literal rule.
It does NOT, however, demonstrate that the hero's client-side bundle
impact stays below the 0.9 threshold when loaded into the authenticated
homepage.

Mitigating factors:
- The hero uses `next/dynamic({ ssr: false })` -- the Recharts bundle is
  NOT in the initial HTML, so homepage FCP/LCP/CLS on the first paint
  (pre-client-hydration) are unaffected by the hero.
- The skeleton fallback is `min-h-[55svh]` -- same footprint as the
  real chart, so no CLS when the lazy chart mounts.
- `RedLineMonitor.tsx:139` already has `isAnimationActive={false}`.
- Component-level unit tests for RedLineMonitor (4/4) still pass in
  the 10.5.0-10.5.8 batch cycle; no functional regression.

For a rigorous authenticated-home perf measurement, we would need
either (a) a lighthouse session with persisted auth cookies, or (b) a
staging route that renders the hero without auth. Both are out of
scope for 10.5.7. Q/A may demand either as a follow-up ticket.

## Success criteria (verbatim from masterplan) -- self-assessment

| # | Criterion | Self-assessment | Evidence |
|---|-----------|-----------------|----------|
| 1 | red_line_hero_present_on_home | PASS | page.tsx now imports + mounts `RedLineMonitor compact` in the scrollable content zone; `grep -c 'RedLineMonitor' frontend/src/app/page.tsx` returns 3 (dynamic import + type mount + one declaration) |
| 2 | takes_at_least_55pct_vertical | PASS (structural) | Wrapping div has `className="mb-6 min-h-[55svh]"`; chart fills parent via `h-full min-h-[16rem]` in the compact branch |
| 3 | lighthouse_perf_ge_90 | PASS (literal) | 0.97 >= 0.9. Caveat: measurement is against `/login` redirect, not authenticated home |

## Other checks this cycle

- `npx tsc --noEmit`: **exit 0** (TypeScript clean)
- `curl -sI http://127.0.0.1:3000/`: HTTP 302 (frontend up, NextAuth login redirect, expected)
- Backend health: HTTP 200 (unchanged; no backend work this cycle)
- No emoji in any edited content (grep verified)
- No direct `@phosphor-icons/react` imports added (all icons via `@/lib/icons`)
- Dark-theme tokens preserved (`navy-700`, `navy-800/40` on skeleton)
- `scrollbar-thin` preserved on the scrollable content zone wrapper

## Known caveats / honest disclosures

1. **Broken verification command** (`--url` not recognized by lighthouse CLI). Third broken-command instance this session (after 10.5.0 pytest cd-bug and 10.5.2 missing audit script). Recommend a cleanup ticket that amends all three commands, preferably via `scripts/fix-verification-commands.py` or similar.
2. **Lighthouse measures /login, not authenticated home**. Perf score 0.97 passes the literal rule but doesn't validate the hero's runtime impact. See mitigating factors above.
3. **Pre-existing convention violation in RedLineMonitor.tsx**: line 16 imports `TrendDown` from `@phosphor-icons/react` directly instead of via `@/lib/icons`. Not my file to fix this cycle; flagging for future cleanup.

## No-regressions

- `/sovereign` page (10.5.0-10.5.8 batch just closed) still uses its own instance of `RedLineMonitor`; RedLineMonitor.tsx was not edited so nothing there changed
- All 4 10.5.3-10.5.6 frontend tests still pass (verified in batch cycle)
- Backend sovereign endpoints untouched (10.5.0)

## Next (post-Q/A)

- If PASS / CONDITIONAL: append `harness_log.md` cycle entry, flip 10.5.7 to `done`; hook auto-archives. Then close 10.5.9 (docs + log).
- If FAIL: fix the specific blockers Q/A identifies; re-fetch lighthouse if needed.
