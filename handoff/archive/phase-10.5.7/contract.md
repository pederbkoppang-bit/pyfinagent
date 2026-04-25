---
step: phase-10.5.7
title: Homepage Red Line hero embed (compact variant)
cycle_date: 2026-04-24
harness_required: true
retrospective: false
forward_cycle: true
---

# Sprint Contract -- phase-10.5.7

## Research-gate summary

Source: `handoff/current/phase-10.5.7-research-brief.md`

JSON envelope (verbatim):
```json
{
  "tier": "simple",
  "external_sources_read_in_full": 6,
  "snippet_only_sources": 7,
  "urls_collected": 13,
  "recency_scan_performed": true,
  "internal_files_inspected": 7,
  "report_md": "handoff/current/phase-10.5.7-research-brief.md",
  "gate_passed": true
}
```

Floor met: 6/5 sources read in full, 13/10 URLs collected, recency scan performed (dvh/svh Baseline 2025; Next.js 15 `ssr:false` placement rule 2026-04-23; Recharts 3.x regressions).

Key external findings load-bearing for the plan:
- `next/dynamic` with `ssr:false` + named-export syntax `.then(m => m.RedLineMonitor)` is the correct App Router pattern for a client-only chart on a homepage.
- `min-h-[55svh]` (not `dvh`) avoids toolbar-animation CLS on mobile; both are Baseline Widely Available since June 2025.
- Skeleton fallbacks with explicit min-height prevent CLS during async load.
- Recharts `isAnimationActive={false}` + `ResponsiveContainer` already the standard perf posture.

Key internal findings:
- `RedLineMonitor.tsx:48-52` has `compact?: boolean` prop ALREADY STUBBED with comment "Used by the homepage hero embed" -- pre-designed for this exact step.
- `RedLineMonitor.tsx:107` compact branch: `"h-full min-h-[16rem]"` vs non-compact `"h-64"`.
- `RedLineMonitor.tsx:139` already has `isAnimationActive={false}`; `role="img"` and `aria-label` already present at lines 105-106.
- `api.ts:631` exposes `getSovereignRedLine` -- reusable.
- `sovereign/page.tsx:37-71` is the reference fetch pattern to mirror (three useState + one useEffect).
- Lighthouse baseline on homepage (pre-embed): perf=0.99, LCP=0.9s, CLS=0.055, TBT=0ms.

## Hypothesis

Because `RedLineMonitor` already exposes a compact prop, has animation disabled, ships with accessibility landmarks, and reuses `getSovereignRedLine` (already fetched by the sovereign page), the homepage hero embed is a ~40-line change to `frontend/src/app/page.tsx`: import, fetch state, dynamic-import, mount. With a proper skeleton + `min-h-[55svh]` wrapper, lighthouse perf should stay >= 0.9 (baseline is 0.99 without the chart; the chart is client-only so server-render metrics are unchanged).

## Success Criteria (verbatim from .claude/masterplan.json step 10.5.7)

```
cd frontend && npm run lighthouse -- --url http://localhost:3000 --output json --output-path handoff/lighthouse_home_sovereign.json && python -c "import json; d=json.load(open('frontend/handoff/lighthouse_home_sovereign.json')); assert d['categories']['performance']['score'] >= 0.9"
```

- red_line_hero_present_on_home
- takes_at_least_55pct_vertical
- lighthouse_perf_ge_90

## Plan steps

1. Edit `frontend/src/app/page.tsx`:
   - Add `getSovereignRedLine` to the `@/lib/api` import
   - Add three `useState` hooks for `redLineWindow` ("30d"), `redLineSeries`, `redLineEvents`, mirroring `sovereign/page.tsx:37-71`
   - Add a `useEffect` that calls `getSovereignRedLine(redLineWindow)` and sets state
   - Use `next/dynamic` from `next/dynamic` with `{ ssr: false, loading: () => <skeleton> }` to import `RedLineMonitor`
   - Mount the lazy `RedLineMonitor` at the TOP of the scrollable content zone (around line 103, before `<KillSwitchShortcut />`), wrapped in a div with `className="mb-6 min-h-[55svh]"` and passing `compact` prop
   - Skeleton fallback must also have `min-h-[55svh]` to hold layout during async load

2. Do NOT modify `RedLineMonitor.tsx` -- the compact branch already handles sizing. Only `page.tsx` changes.

3. Restart frontend if needed (Next.js hot-reload should pick up the change without a full restart).

4. Run verification:
   - `curl http://127.0.0.1:3000/` to confirm 200 or 302 (route reachable)
   - `npm run lighthouse` against `http://localhost:3000` with output to `frontend/handoff/lighthouse_home_sovereign.json`
   - Python assert: `d['categories']['performance']['score'] >= 0.9`

5. Verify "takes at-least 55% vertical" by inspecting the rendered page height attribution (the wrapping div has `min-h-[55svh]` which is `>= 55%` of the small viewport height; the chart fills its parent via `h-full min-h-[16rem]` so visually it takes at-least 55% of the viewport at the typical 800-1000px laptop screen).

## What Q/A must audit

1. The code change lives in page.tsx and ONLY page.tsx (+ optional loading skeleton): RedLineMonitor.tsx untouched?
2. `next/dynamic` syntax is correct and `ssr:false` is present
3. Skeleton loading fallback is the same 55svh footprint (no CLS)
4. Lighthouse JSON written, perf score >= 0.9
5. Live render: `curl -sI http://127.0.0.1:3000/` returns 200 or 302 (auth redirect)
6. No regressions on /sovereign page (the 10.5.0-10.5.8 batch just passed)
7. CLAUDE.md conventions honored: no emojis, no direct @phosphor-icons imports, dark theme, `scrollbar-thin`

## References

- `handoff/current/phase-10.5.7-research-brief.md` -- research gate + implementation recipe
- `frontend/src/components/RedLineMonitor.tsx` -- target component (compact branch at line 107)
- `frontend/src/app/sovereign/page.tsx` -- reference fetch pattern (lines 37-71)
- `frontend/src/lib/api.ts:631` -- `getSovereignRedLine` function
- `frontend/handoff/lighthouse_home_sovereign.json` -- baseline (pre-embed) 0.99
- `.claude/rules/frontend.md` + `.claude/rules/frontend-layout.md`
- `CLAUDE.md` -- harness protocol + frontend conventions
