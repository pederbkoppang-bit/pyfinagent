# Sprint Contract -- phase-25.A12 -- Playwright visual regression CI baseline

**Cycle:** phase-25 cycle 23 (P1 sprint)
**Date:** 2026-05-13
**Step ID:** 25.A12
**Priority:** P1
**Audit basis:** bucket 24.12 F-6 -- `docs/audits/phase-24-2026-05-12/screenshots/` empty; no baseline images

## Research-gate

Researcher spawned this cycle (agent a498736fb95e4c072). Brief at
`handoff/current/research_brief.md`. Gate envelope: 8 sources read in full,
17 URLs, recency scan performed, 14 internal files inspected, gate_passed=true.

Key research conclusions:
- Playwright `toHaveScreenshot()` with `maxDiffPixelRatio: 0.015` + `threshold: 0.2` is canonical for dark-themed dashboards.
- `animations: 'disabled'` is built-in default since Playwright 1.32; specify for clarity.
- **OS-specific snapshot naming pitfall:** Playwright appends `{platform}` to filenames. macOS-generated baselines fail on `ubuntu-latest` CI. Solution: always generate baselines in CI on Linux.
- **`NEXT_PUBLIC_E2E_TESTING=true`** suppresses live polling, dev overlays, and timers via Next.js webServer.env.
- `reducedMotion: 'reduce'` in browser context handles `motion` package JS animations not caught by Playwright's CSS-only animation disabling.
- **`.gitkeep` per page-spec subdirectory** is the correct placeholder strategy (NOT 1x1 PNGs -- those cause "snapshot size mismatch" errors). Playwright ignores non-PNG files in snapshot dirs.
- 8 pages in scope: `/`, `/paper-trading`, `/performance`, `/backtest`, `/agents`, `/sovereign`, `/reports`, `/agent-map`.

## Hypothesis

Adding (a) `@playwright/test` dev-dep, (b) `frontend/playwright.config.ts`
with the research-canonical config, (c) shared mask helper, (d) 8 per-page
spec files exercising `toHaveScreenshot(...)`, (e) `.github/workflows/
visual-regression.yml` with workflow_dispatch update-snapshots input, and
(f) `.gitkeep` placeholders at each canonical snapshot subdir -- closes
phase-24.12 F-6 with structural infrastructure ready for first-CI-run
baseline capture by the operator.

## Success criteria (verbatim from masterplan)

1. `playwright_config_ts_exists`
2. `github_actions_visual_regression_yml_passes`
3. `screenshots_dir_populated_with_per_page_baselines`

Verification command (immutable):
`source .venv/bin/activate && python3 tests/verify_phase_25_A12.py`

Live check (per masterplan):
`CI run produces visual-regression report; baseline images committed`

## Plan

1. **`frontend/package.json`** -- add `@playwright/test: ^1.50.0` to devDependencies (no install in this cycle; CI installs via `npm ci`).
2. **`frontend/playwright.config.ts`** (new) -- the research-canonical config: `testDir: ./tests/visual-regression`, `maxDiffPixelRatio: 0.015`, `threshold: 0.2`, `animations: 'disabled'`, `reducedMotion: 'reduce'`, viewport 1280x800, chromium-only, `webServer` runs `npm run dev` on port 3000 with `NEXT_PUBLIC_E2E_TESTING=true`, snapshotPathTemplate centralizes baselines.
3. **`frontend/tests/visual-regression/helpers/visual.ts`** (new) -- `disableAnimations(page)` CSS-injection helper + `dynamicMasks(page)` returning Locator array for timestamps, skeletons, spinners, recharts ticks.
4. **`frontend/tests/visual-regression/<page>.spec.ts`** (new, 8 files): `home.spec.ts`, `paper-trading.spec.ts`, `performance.spec.ts`, `backtest.spec.ts`, `agents.spec.ts`, `sovereign.spec.ts`, `reports.spec.ts`, `agent-map.spec.ts`. Each: `goto` -> `disableAnimations` -> `waitForLoadState('networkidle')` -> `expect(page).toHaveScreenshot({ fullPage: true, mask: dynamicMasks(page) })`.
5. **`.github/workflows/visual-regression.yml`** (new) -- ubuntu-latest, Node lts, `npx playwright install --with-deps chromium`, conditional `--update-snapshots` path on `workflow_dispatch.inputs.update_snapshots == 'true'`, `git-auto-commit-action@v5` for baseline commits, artifact uploads (report + diffs).
6. **`.gitkeep` placeholders** at `frontend/tests/visual-regression/snapshots/chromium/<page>.spec.ts/.gitkeep` (8 files) so the snapshots dir is populated for criterion 3 BEFORE first CI run.
7. **Verifier** -- `tests/verify_phase_25_A12.py` -- 11+ claims:
   - Claim 1: `frontend/playwright.config.ts` exists.
   - Claim 2: config declares `testDir`, `maxDiffPixelRatio`, `threshold`, `animations`, `webServer`, `NEXT_PUBLIC_E2E_TESTING`, and chromium project.
   - Claim 3: `.github/workflows/visual-regression.yml` exists.
   - Claim 4: workflow uses `actions/setup-node`, `npx playwright install --with-deps chromium`, conditional update-snapshots step, artifact uploads.
   - Claim 5: workflow runs on `ubuntu-latest`.
   - Claim 6: helpers file exists with `disableAnimations` + `dynamicMasks` exports.
   - Claim 7: at least 7 page spec files exist (one per major page). Each `goto`s + calls `toHaveScreenshot`.
   - Claim 8: `frontend/tests/visual-regression/snapshots/chromium/` is populated with at least 7 page-spec subdirs each containing a `.gitkeep`.
   - Claim 9: `@playwright/test` declared in `frontend/package.json` devDependencies.
   - Claim 10: README or comment documents the operator first-run flow (workflow_dispatch with `update_snapshots=true`).
   - Claim 11: dynamicMasks helper covers timestamps + animations + recharts ticks (grep for the locators).
   - Claim 12: workflow uses `if: ${{ github.event.inputs.update_snapshots == 'true' }}` gate on the update step (correct shape).

## Non-goals

- Not installing Playwright browsers in this cycle (operator-side; CI installs via `--with-deps`).
- Not generating real PNG baselines locally (must be Linux/CI-generated per the macOS/Linux divergence finding).
- No frontend code changes outside testing infrastructure.
- No deletion of the empty `docs/audits/phase-24-2026-05-12/screenshots/` directory; the new infra lives under `frontend/tests/visual-regression/snapshots/` (canonical Playwright location).
- Not running the workflow now -- workflow_dispatch operator-triggered for baseline capture.

## References

- `handoff/current/research_brief.md` -- full brief this cycle
- `frontend/package.json:1-50` (deps, scripts)
- `frontend/src/app/*/page.tsx` (8 pages: home, paper-trading, performance, backtest, agents, sovereign, reports, agent-map)
- `.github/workflows/` (5 existing; visual-regression.yml is new)
- `.claude/rules/frontend.md` -- NextAuth middleware, port 3000 conventions
- Ash Connolly + TestDino + Mazzarolo + Bug0 + Playwright + Next.js docs (per brief)
