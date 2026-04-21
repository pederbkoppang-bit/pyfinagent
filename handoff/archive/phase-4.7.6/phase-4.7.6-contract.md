# Contract -- Cycle 75 / phase-4.7 step 4.7.6

Step: 4.7.6 WCAG 2.1 AA + keyboard-only kill-switch workflow

## Hypothesis

`@axe-core/cli` against `/login` (the sole unauthenticated route) is
the minimum WCAG 2.1 AA gate we can run today without a Playwright
auth-setup project; the authenticated routes need
`@axe-core/playwright` + storageState, which is tracked as a later
infra step. For this cycle we lock down the public route + the
kill-switch workflow.

Explore findings that we fix in-scope:
- OpsStatusBar buttons lack aria-label and visible focus ring.
- Login page buttons lack focus ring.

Out of scope (queued for follow-up):
- Playwright authenticated-route axe; 41 low-contrast text candidates.

## Scope

Files created / modified:

1. **MODIFY** `frontend/package.json` -- add devDep
   `@axe-core/cli`, npm script `"axe": "axe http://localhost:3000/login --tags wcag2a,wcag2aa,wcag21a,wcag21aa --exit"`.
2. **MODIFY** `frontend/src/components/OpsStatusBar.tsx` -- add
   `aria-label` + `focus-visible:ring-2 focus-visible:ring-sky-400`
   on Pause / Resume / Flatten buttons.
3. **MODIFY** `frontend/src/app/login/page.tsx` -- add
   `focus-visible:` focus ring on Google + Passkey buttons.
4. **NEW** `scripts/audit/keyboard_flatten.py` -- verifies:
   - KillSwitchShortcut component source registers keydown
   - Handler matches Ctrl/Cmd+Shift+H
   - Handler calls postPaperKillSwitchAction with FLATTEN_ALL + PAUSE
   - aria-live region present
   - preventDefault() called
   - Homepage renders <KillSwitchShortcut />
   Emits handoff/keyboard_flatten.json; `--check` exits 1 on failure.

## Immutable success criteria (from masterplan)

1. `wcag_2_1_aa_pass`
2. `keyboard_only_kill_switch_workflow_green`

## Verification (immutable)

    cd frontend && npm run axe && python ../scripts/audit/keyboard_flatten.py

Note: the masterplan says `python scripts/audit/keyboard_flatten.py`
from frontend cwd, which would look inside frontend/. Use `../`
prefix OR run from repo root. Our npm script + Python script
both work from either; documented here.

## Anti-rubber-stamp check

- qa-evaluator must verify axe is REALLY running WCAG 2.1 AA tags
  (not just default 2.0 A+AA). Check the `--tags` arg.
- qa must verify the keyboard_flatten audit would FAIL if the
  KillSwitchShortcut component dropped preventDefault() or the
  keydown handler. Not just a presence check.

## Known limitations

- Authenticated routes (cockpit, paper-trading, etc.) not audited
  this cycle; Playwright storageState project queued.
- Low-contrast text sweep (41 candidates) is a separate cleanup
  cycle; the axe run on /login covers that route's contrast.
