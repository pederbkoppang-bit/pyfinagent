---
name: currency-display-61-3
description: phase-61.3 (reval 2026-08-07) — criterion 1 is HALF built (formula shipped dark by 70.3, but ZERO test drives _advance_stop on a KR position and the two that do are deselected by the immutable -k); resolveCurrency still explicit-first vs base_currency="USD"; no marked_at column; npm run build shares .next with the operator's :3000; pinning a locale does NOT remove ICU-version nondeterminism
metadata:
  type: project
---

Two briefs exist: `handoff/archive/misc/research_brief_61.3.md` (2026-07-08, complex,
7 full reads) and `handoff/current/research_brief_61.3_reval.md` (2026-08-07, moderate
reval, 9 fresh full reads). Durable, non-obvious facts:

- **Criterion 1 is HALF built, and the missing half is invisible to a source grep.**
  phase-70.3 shipped the LOCAL-share-weighted formula behind
  `paper_avg_entry_fx_fix_enabled` (default False). 18 tests match the immutable
  `-k 'addon or avg_entry or currency or 61_3'`, across 6 files — **every one asserts
  only the saved row's `avg_entry_price`**. The criterion also demands the
  *breakeven-advanced stop* be KRW-scale, and the only two files that call
  `_advance_stop` (`test_phase_32_1_breakeven_ratchet.py`,
  `test_phase_32_2_hwm_trailing.py`) are US-only **and are deselected by that same
  `-k` filter**. "Tests exist for this criterion" is true and misleading; derive
  coverage from what each test ASSERTS, not from the file list.
- **`_advance_stop` has two silent no-op guards** — breakeven returns `(None, None)`
  unless `stop_advanced_at_R` is falsy AND `current_stop < entry_price`. A fixture that
  misses either gets a vacuous pass.
- **Anchors MOVED between 2026-07-08 and 2026-08-07** (re-derive, never cite from
  memory): flagged averaging `paper_trader.py:459-467` (was :286-302); flag
  `settings.py:477` (was :455); `base_currency:"USD"` hardcodes `:490`/`:511` (was
  :313/:334/:481); `mark_to_market` `:693-778` with the `updates` dict at `:731-738`;
  `_POSITION_RT_FIELDS` `:1470`; `_advance_stop` `:1394-1460`.
- **Display half unchanged and `resolveLocalCurrency` was never written** —
  `format.ts:161-171` is still explicit-first; grep for the helper returns zero. A
  FOURTH offender site the first brief missed: `trades-columns.tsx:94-103` (LOCAL trade
  price) plus `:102`/`:123` `$`-templates.
- **`marked_at` needs THREE things, not one** — the live `financial_reports.paper_positions`
  schema is 22 columns with no timestamp; `save_paper_position` builds its MERGE from the
  row's keys, so an unmigrated column makes the statement itself invalid, and
  `_safe_save_position`'s retry only prunes `_POSITION_RT_FIELDS` — a new column MUST be
  added to that set. The READ path is free: `SELECT *` + no `response_model` on
  `/api/paper-trading/portfolio`, so a new column reaches the frontend with zero API work.
  Do NOT reuse `bandFromAgeSec` (90s/300s bands) for a once-a-day mark — it renders red
  forever; write a mark-scale band.
- **A new scheduled mark job is trading-behaviour-adjacent** — `mark_to_market` calls
  `_advance_stop` and writes `stop_loss_price`, so "mark-only" is NOT read-only. Also
  phase-70.5 added `reschedule_paper_job`, which re-adds only what
  `_add_scheduler_job` creates: a second job placed anywhere else silently vanishes on
  the first settings PUT.
- **`npm run build` shares `.next` with the operator's :3000 dev server** —
  `next.config.js` only overrides `distDir` when `PLAYWRIGHT_DIST_DIR` is set (the 64.1
  :3100 isolation). The immutable verification command names `npm run build`, so the
  hazard is baked into the criteria. See [[second-next-dev-breaks-operator-3000]].
- **Pinning a locale removes ONE of TWO nondeterminism sources.** vercel/next.js #79397
  read in full: the hydration mismatch happened WITH an explicit `it-IT` locale — Node
  20.11.1 vs 20.19.0/22.14.0 emit different grouping (nodejs/node#48120). NumberFlow's
  own docs confirm the other source: `locales` omitted ⇒ browser default. So fix the
  `locales={undefined}` sites, but write tests with regex/prop assertions rather than
  exact `Intl` output strings.
- **`paper_avg_entry_fx_fix_enabled` is not in `settings_api.py::_FIELD_TO_ENV`** — no UI
  flip; it needs a `backend/.env` line + restart (operator action).
- Criterion 6 (Playwright of a live KR position) is unsatisfiable while the book holds no
  non-US row; the step closes on the 61.2 pattern (built + Q/A + flip HELD + ask row).
  Do not seed a position. See [[decision-input-integrity-61-2]].
