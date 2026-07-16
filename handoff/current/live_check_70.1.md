# live_check — step 70.1 (S1: make the setting changeable)

## Method (phase-59.2 canonical, frontend.md "Live-UI verification")

Captured against a **skip-auth `:3100`** dev server (`LIGHTHOUSE_SKIP_AUTH=1 npx next dev --port 3100`);
the operator's `:3000` instance was **never touched** and answered `302 → /login` (healthy authed
signature) after teardown. Same code + same backend `:8000` + same BQ data as `:3000`. Playwright MCP
`@playwright/mcp@0.0.76` (pinned in `.mcp.json`). Frontend `npm run build` passed clean before capture.

Captures (`handoff/current/captures_70.1/`), each mapped to the criterion it visually evidences:
- `70.1-manage-fixed-fullpage.png` — criterion 1: MAX POSITIONS PER SECTOR shows the typed `5` with a green
  "Settings saved." banner (the value that pre-fix would have been `25`).
- `70.1-out-of-range-error.png` — criterion 2: the Trading-settings card with the inline rose
  "Must be between 0 and 20." error, the "Fix the highlighted fields before saving." summary, and the
  disabled Save button (element screenshot of the settings card).
- `70.1-risk-panel-override-active.png` — criteria 3 & 4: the "Risk limits (live overrides)" panel with an
  ACTIVE override — amber warning banner, "OVERRIDE ACTIVE" badge, configured `30` vs effective `25` on the
  "Max NAV % per sector" row, and the enabled Clear button (element screenshot of the panel section).
- `70.1-risk-limits-panel.png` — supplementary page-top view.

_(Cycle-2 remediation: the first Q/A returned CONDITIONAL because the risk panel and the out-of-range state
were not visible in the initial captures — the risk panel is below the fold. Element screenshots of the
panel-with-active-override and the out-of-range error were added; no code changed._

_Cycle-3 correction: the cycle-2 Q/A found the out-of-range element screenshot had grabbed the card HEADER
(398×48) instead of the body. `70.1-out-of-range-error.png` was re-captured as an element screenshot of the
FULL Trading-settings card (1109×859) — visually confirmed by Main to show the `25` field with the rose
"Must be between 0 and 20." error, the "Fix the highlighted fields before saving." summary, and the disabled
Save button. Full-page capture does not work on this page because the shell is `h-screen overflow-hidden`
with an inner scroll container, so an element screenshot of the card is the correct tool. No code changed
between cycles 1–3. Override set for the risk-panel capture was cleared afterward — `active_overrides: []`,
`paper_max_per_sector = 2` confirmed.)_

## Criterion 1 — clear-then-type yields the typed value (was the '2'→'25' bug), saves 200

Live keystroke-level repro of the exact human flow on `/paper-trading/manage` "Max positions per sector":

```
before                : "2"
select-all + Backspace : ""      <-- FIXED (pre-fix: snapped back to "2")
type "5"               : "5"      <-- FIXED (pre-fix: appended to "25")
click Save             : PUT /api/settings/ -> 200
GET /api/settings/     : paper_max_per_sector = 5   (persisted)
```

(Config restored to `paper_max_per_sector = 2` after the test — `PUT 200`, confirmed.)

## Criterion 2 — out-of-range prevented client-side with a field-specific message (no silent 422)

```
clear + type "25" (> max 20):
  field value : "25"
  inline error: "Must be between 0 and 20."   (rose, field-specific)
  Save button : disabled = true                (+ "Fix the highlighted fields before saving." summary)
```
The out-of-range value is never written into the dirty payload, so a value the UI rejected can never
reach the server as a generic 422.

## Criterion 3 — risk_overrides shadow surfaced + clearable, api.ts get/set/clear

Live roundtrip on the new "Risk limits (live overrides)" panel, "Max NAV % per sector" row
(`paper_max_per_sector_nav_pct`):

```
Set 25  -> PUT /api/paper-trading/risk-limits 200
   configured=30, effective=25, "OVERRIDE ACTIVE" badge shown, amber warning banner shown
Clear   -> DELETE /api/paper-trading/risk-limits/... 200
   configured=30, effective=30, badge gone, banner gone
```
`api.ts` exposes `getRiskLimits` / `setRiskLimit` / `clearRiskLimit` (immutable verification grep PASS).
Confirmed no override left active after the test (`active_overrides: []`), so with no override the UI-saved
cap is what the engine enforces.

## Criterion 4 — nav_pct editor present; false comment corrected

`nav_pct_editor_present = true` (the "Max NAV % per sector" editor renders in the risk panel — its proper
editor since it is a `risk_overrides.ALLOWED_KEY`, not a `.env` settings-form field). The false
"editable at /manage" comment at `positions/page.tsx:23-29` is corrected to state it is a display-only
fallback and to point at the risk-limits panel / `getRiskLimits` effective_value. Out-of-scope knobs
(`paper_max_factor_corr`, `paper_risk_judge_reject_binding`) deliberately NOT given no-op controls
(neither in SettingsUpdate nor ALLOWED_KEYS) — deferred to a backend follow-up.

## Criterion 5 — no emoji, Phosphor icons, states

No emoji (Phosphor `IconWarning` used in the panel); navy/slate palette per frontend rules; the panel has
loading (spinner), error (rose banner), and empty ("No adjustable risk limits") states; `npm run build`
green.

## Do-no-harm
UI + `api.ts` only. No backend route change (routes pre-existed). No live-loop behavior change. No
risk-limit threshold moved — the panel only exposes the EXISTING operator-adjustable ALLOWED_KEYS caps
(intended operator control). Operator config left as found (max_per_sector=2, zero overrides).
