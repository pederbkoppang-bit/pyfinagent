# Experiment results — step 70.1 (S1: make the setting changeable)

**Phase/step:** phase-70 → 70.1 | **Date:** 2026-07-16 | **Type:** frontend + api.ts (UI-touching)

## Files changed (5)

1. **`frontend/src/components/paper-trading/cockpit-helpers.tsx`** — `PaperSettingNum` rewritten to
   string-state-then-coerce: local `useState<string>` bound to `value={text}` ('' now representable → no
   snap-back), coerce to number only for the dirty/save path; a re-seed effect keeps the field in sync with
   `stored` after a save/reload without clobbering an in-progress edit; field-specific range error
   (`Must be between {min} and {max}.` / `Enter a number.`) with rose border + `aria-invalid`; out-of-range
   values are never written into `dirty`; new optional `onValidity(field, error)` prop lifts validity to the
   parent. Added `import { useEffect, useState } from "react"`.
2. **`frontend/src/lib/api.ts`** — new `getRiskLimits` (GET), `setRiskLimit(key,value,reason)` (PUT with the
   required `confirmation:"SET_RISK_LIMIT"` token), `clearRiskLimit(key)` (DELETE) + `RiskLimitEntry` /
   `RiskLimitsResponse` types, wired to the existing `/api/paper-trading/risk-limits` routes.
3. **`frontend/src/components/paper-trading/RiskLimitsPanel.tsx`** (NEW) — surfaces the risk_overrides shadow:
   per ALLOWED_KEY (friendly labels incl. `paper_max_per_sector_nav_pct`) shows configured-vs-effective, an
   "OVERRIDE ACTIVE" badge, a bounded Set input + Set button, a Clear button (disabled unless overridden), an
   amber active-override warning banner (Phosphor `IconWarning`, no emoji), and loading/error/empty states.
   Refetches + calls the passed `onChanged` (cockpit `refresh`) after set/clear.
4. **`frontend/src/app/paper-trading/manage/page.tsx`** — `fieldErrors` state via `onValidity`; Save disabled
   + rose summary banner when any field is out of range; renders `<RiskLimitsPanel onChanged={refresh} />`
   after the Trading-settings card; `onValidity={handleFieldValidity}` on all 10 number fields.
5. **`frontend/src/app/paper-trading/positions/page.tsx`** — corrected the false "editable at /manage" comment
   (nav_pct is a risk-limit ALLOWED_KEY edited via the risk panel; the constant is a display-only fallback).

## Verification command output (verbatim)

```
$ bash -c 'grep -Eqi "getRiskLimits|setRiskLimit|clearRiskLimit|risk-limits" frontend/src/lib/api.ts && grep -Eqi "paper_max_per_sector_nav_pct" frontend/src/app/paper-trading/manage/page.tsx frontend/src/components/paper-trading/*.tsx'
VERIFICATION: PASS (exit 0)
```

Frontend build: `npm run build` → success (route table printed; `/paper-trading/manage` compiled, 7 kB).

## Live Playwright evidence (see live_check_70.1.md; captures_70.1/)

- **Clear-then-type (criterion 1):** before `"2"` → clear → `""` (no snap-back) → type `"5"` → `"5"` (no
  append) → Save `PUT /api/settings/ 200` → GET returns `paper_max_per_sector: 5`. Restored to 2.
- **Out-of-range (criterion 2):** type `"25"` (>20) → inline `"Must be between 0 and 20."` + Save disabled.
- **Risk override roundtrip (criterion 3):** Set NAV%/sector `25` → `PUT /risk-limits 200`, effective 25 vs
  configured 30 + badge + warning banner; Clear → `DELETE 200`, effective back to 30, badge/banner gone.
- **Panel + editor (criterion 4):** risk panel + nav_pct editor render; false comment corrected.
- **Styling (criterion 5):** Phosphor icon, no emoji, loading/error/empty states, navy/slate palette; build green.

## Do-no-harm / scope
UI + api.ts only; no backend route change (routes pre-existed); no live-loop behavior change; no risk-limit
threshold moved (the panel exposes the EXISTING operator-adjustable caps). Operator config left as found
(max_per_sector=2, zero active overrides). Out-of-scope knobs (factor_corr, reject_binding) explicitly
deferred to a backend follow-up rather than given no-op UI controls.
