# Visual Regression -- Playwright

phase-25.A12 -- closes phase-24.12 F-6 (no baseline images for visual
diff). 8 page-specs cover the major dashboard routes:
`/`, `/paper-trading`, `/performance`, `/backtest`, `/agents`,
`/sovereign`, `/reports`, `/agent-map`.

## First-run operator flow (REQUIRED before this CI gate has teeth)

Baselines MUST be generated on **ubuntu-latest** (Linux). macOS-generated
snapshots fail CI due to font + antialiasing differences -- a documented
Playwright pitfall.

### Preferred path: GitHub workflow_dispatch

1. From GitHub UI -> Actions -> "Visual Regression" -> Run workflow.
2. Set `update_snapshots = true`. The job runs
   `playwright test --update-snapshots` and auto-commits the PNGs under
   `frontend/tests/visual-regression/snapshots/chromium/<page>.spec.ts/`
   via `git-auto-commit-action@v5`.
3. Subsequent push / PR runs compare new screenshots vs the committed
   baselines; CI fails when `maxDiffPixelRatio > 0.015`.

### Local fallback (Linux only)

```
cd frontend && npm ci
npx playwright install --with-deps chromium
npx playwright test --update-snapshots
git add tests/visual-regression/snapshots && git commit
```

## Threshold tuning

- `maxDiffPixelRatio: 0.015` -- canonical for dark dashboards.
- `threshold: 0.2` -- per-pixel YIQ tolerance for subpixel font rendering.
- `animations: 'disabled'` + `reducedMotion: 'reduce'` -- belt-and-suspenders
  against both CSS and JS (motion package) animations.

## Dynamic content masking

`helpers/visual.ts::dynamicMasks` masks timestamps, skeletons, spinners,
pulses, animate-classed elements, and Recharts tick labels. Add to the
list if a future page introduces new dynamic content classes.

## Why `.gitkeep` placeholders?

Each `snapshots/chromium/<page>.spec.ts/` directory ships with a
`.gitkeep` so the snapshots dir is structurally populated before any
PNG baseline exists. Playwright ignores non-PNG files in snapshot dirs.
On the first `--update-snapshots` run, real PNG baselines are placed
alongside the `.gitkeep`s. We deliberately do NOT ship 1x1 placeholder
PNGs because Playwright raises "snapshot size mismatch" rather than
"no baseline exists" on first comparison.
