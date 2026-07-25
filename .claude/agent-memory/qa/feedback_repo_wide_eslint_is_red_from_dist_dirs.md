---
name: repo-wide-eslint-is-red-from-dist-dirs
description: qa.md 1b's `npx eslint .` exits 1 in frontend/ from generated .next-audit-3100 and .next-functional output -- pre-existing, not a regression
metadata:
  type: feedback
---

`cd frontend && npx eslint .` (the qa.md §1b gate command) **exits 1** on this repo:
26 errors, all `@next/next/no-assign-module-variable` inside `.next-audit-3100/` (13)
and `.next-functional/` (13). Zero errors in `src/`. Measured 2026-07-25 during
phase-80.3 cycle 5.

**Why:** `frontend/eslint.config.mjs:11` ignores `.next/**` but not `.next-*/**`, so any
Playwright-rig dist dir left on disk poisons the repo-wide gate. Already queued as item 5
on the discovered-defects list in `handoff/current/cycle_block_summary.md` — it is a
known defect, NOT something the step under evaluation introduced.

**How to apply:** run the gate, but break the errors down by file before grading. Report
`exit=1 with N errors, all in generated dist dirs, 0 in src/` rather than either
rubber-stamping a scoped two-file lint as "the gate" or failing a step for a pre-existing
config bug. `npx eslint . -f json` + a group-by-top-dir is the cheap discriminator.
Related: [[derived-scope-lint-use-xargs]].
