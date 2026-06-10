# Live-check placeholder -- phase-25.A12

**Step:** 25.A12 -- Playwright visual regression CI baseline
**Date:** 2026-05-13

## Live-check field (per masterplan)
> "CI run produces visual-regression report; baseline images committed"

## Pre-deployment evidence
- 12/12 verifier PASS (`source .venv/bin/activate && python3 tests/verify_phase_25_A12.py`).
- Structural infrastructure in place: `playwright.config.ts` + 8 page specs + `helpers/visual.ts` + 8 `.gitkeep` placeholders + GitHub Actions workflow + README.
- `@playwright/test: ^1.50.0` declared in `frontend/package.json` devDependencies.

## Post-deployment operator workflow

**Step 1 -- merge this commit to main.** Until baselines are captured, the workflow path-filter triggers will run `npx playwright test` on subsequent PRs and FAIL because no real PNG baselines exist yet. To avoid spurious red CI:
- Operator either (a) immediately triggers the workflow_dispatch update path below, OR
- (b) temporarily disables the workflow until ready to capture baselines.

**Step 2 -- capture baselines via workflow_dispatch.**
1. GitHub UI -> Actions -> "Visual Regression" -> Run workflow.
2. Set `update_snapshots = true`. Branch: `main`.
3. The job runs `npx playwright test --update-snapshots` on Linux + auto-commits the PNGs under `frontend/tests/visual-regression/snapshots/chromium/<page>.spec.ts/` via `git-auto-commit-action@v5`.
4. Expected commit subject: `chore: update playwright visual regression baselines [skip ci]` -- the `[skip ci]` token prevents the just-merged baselines from triggering another workflow loop.

**Step 3 -- verify CI gate is now live.** Open a no-op PR; the workflow runs `npx playwright test` (no --update-snapshots) and compares new screenshots vs the just-committed baselines. CI passes when `maxDiffPixelRatio <= 0.015`.

## Verifying the workflow report

After any run, download the `playwright-report-<sha>` artifact from the workflow run page. The report includes per-test pass/fail status + diff thumbnails. On failure, also download `playwright-diffs-<sha>` for the side-by-side `actual.png` / `expected.png` / `diff.png` per failing test.

## Tuning if false positives surface

- Tweak `maxDiffPixelRatio` (default 0.015) in `frontend/playwright.config.ts`.
- Add additional dynamic masks in `frontend/tests/visual-regression/helpers/visual.ts::dynamicMasks` -- e.g. new live-data classes that didn't exist when the baseline shipped.
- For specific pages where a section legitimately differs run-over-run, scope a per-spec `mask` override.

## Closes audit basis
phase-24.12 F-6 RESOLVED. Structural visual-regression infrastructure shipped; baselines committed by operator on first workflow_dispatch run. Subsequent PRs gate visual fidelity automatically.

**Audit anchor for next bucket:** 25.B9 (P1 system prompt cache threshold) OR 25.A10 (P1 Alpaca MCP smoke test) OR 25.B (P2 cosmetic-patch cleanup).
