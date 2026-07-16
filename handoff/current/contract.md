# Contract — step 70.1 (S1: make the setting actually changeable)

**Phase:** phase-70 (Trade diversity + changeable fund) | **Step:** 70.1 | **Priority:** P1 | harness_required: true
**Cycle:** 1 | Date: 2026-07-16 | **Type:** frontend + api.ts (UI-touching → live Playwright Q/A gate)

## Research-gate summary (gate PASSED)

Researcher via Workflow structured-output (Opus 4.8, $0 Max rail, stall-immune — fable-snapshot avoidance).
Envelope: **gate_passed=true**, tier=moderate, **8 external sources read in full**, 16 snippet-only, 40 URLs,
recency scan performed, **9 internal files audited**. Brief: `handoff/current/research_brief_70.1.md`.

**Snap-back mechanism (confirmed on HEAD 3dd7fc53):** `cockpit-helpers.tsx` `PaperSettingNum` binds
`value = (draft ?? stored ?? "")` (:467); onChange maps a cleared field to `undefined` then DELETES the draft
(:478-483) → `value` re-derives via `??` to `stored` (empty is unrepresentable) → snap-back → next keystroke
appends (`2`→`5`→`25`) → 25 > max 20 → generic save-time 422.

**Fix pattern (React docs + MDN + web.dev):** string-state-then-coerce — local `useState<string>` bound to
`value={text}` ('' allowed → no snap-back), coerce to number only for the dirty/save path; field-specific
range validation lifted to a parent `fieldErrors` record that disables Save + shows a rose summary; keep the
422 as a server backstop. Risk-limits: `api.ts` get/set/clear wired to the existing
`GET/PUT/DELETE /api/paper-trading/risk-limits` routes (PUT requires `confirmation:"SET_RISK_LIMIT"`), a
Manage-tab "Risk limits (live overrides)" panel (configured-vs-effective, amber OVERRIDE-ACTIVE badge, Set +
Clear per key, warning banner when any override active). `paper_max_per_sector_nav_pct` is an
`risk_overrides.ALLOWED_KEYS` member so the risk-panel Set IS its editor. Scope guard: `paper_max_factor_corr`
+ `paper_risk_judge_reject_binding` are in NEITHER SettingsUpdate NOR ALLOWED_KEYS → a UI control would no-op
→ OUT OF SCOPE (backend follow-up).

## Hypothesis

The operator's "can't change MAX POSITIONS PER SECTOR" is a controlled-input clear-snapback bug plus missing
risk-override transparency. Replacing the `?? stored` binding with string-state-then-coerce makes clear-then-
type yield the typed value; client-side range validation replaces the silent 422 with a field-specific message;
a risk-limits panel surfaces/clears the shadow and provides the editor for `paper_max_per_sector_nav_pct`.
No backend route change (routes exist); no live-loop behavior change.

## Immutable success criteria (verbatim from masterplan.json 70.1)

1. On /paper-trading/manage a numeric settings field (e.g. Max positions per sector) can be cleared to empty
   and a new in-range value typed that becomes exactly that value (no snap-back, no append) and saves via
   PUT /api/settings/ 200 -- proven by a live Playwright capture of the clear-then-type flow (the exact flow
   that reproduced '2'->'25' pre-fix)
2. Out-of-range numeric input is prevented or clamped client-side with a field-specific message; no silent
   generic 422 for a value the UI accepted
3. The Manage tab shows any active risk_overrides shadow (effective-vs-configured) with a working Clear button
   and a warning banner; api.ts exposes get/set/clear risk-limits; with no override active the UI-saved cap is
   what the engine enforces
4. Every active BUY-gating knob incl. paper_max_per_sector_nav_pct has a working UI editor OR the false
   'editable at /manage' comment is corrected; no UI claim about a control that does not exist
5. No emojis; Phosphor icons only; error/loading/empty states present

Verification command (immutable):
`bash -c 'grep -Eqi "getRiskLimits|setRiskLimit|clearRiskLimit|risk-limits" frontend/src/lib/api.ts && grep -Eqi "paper_max_per_sector_nav_pct" frontend/src/app/paper-trading/manage/page.tsx frontend/src/components/paper-trading/*.tsx'`
Live check: `live_check_70.1.md` referencing a fresh Playwright MCP capture (skip-auth :3100) of the
clear-then-type flow producing the typed value + save 200, plus the risk-override display.

## Plan

1. (DONE) research gate → research_brief_70.1.md.
2. (this contract, before generate.)
3. GENERATE:
   - `cockpit-helpers.tsx` PaperSettingNum: string-state-then-coerce + field-specific range validation +
     `onValidity` lift.
   - `frontend/src/lib/api.ts`: `getRiskLimits` / `setRiskLimit` / `clearRiskLimit` + types.
   - NEW `frontend/src/components/paper-trading/RiskLimitsPanel.tsx`: ALLOWED_KEYS with friendly labels (incl.
     literal `paper_max_per_sector_nav_pct`), configured-vs-effective, amber override badge, Set + Clear,
     warning banner; error/loading/empty states; Phosphor icons; no emoji.
   - `manage/page.tsx`: `fieldErrors` state → disable Save + rose summary; render RiskLimitsPanel; refresh
     after risk-limit change.
   - `positions/page.tsx:23-29`: correct the false "editable at /manage" comment (point at the risk panel;
     note DEFAULT_SECTOR_CAP_PCT=30 is a display fallback that may not reflect an active override).
4. `npm run build` (frontend build check).
5. Live Playwright capture (skip-auth :3100): clear-then-type on Max positions per sector → typed value +
   save 200; out-of-range shows field message + Save disabled; risk-limits panel renders. Move captures to
   `handoff/current/captures_70.1/`; write `live_check_70.1.md`.
6. EVALUATE: fresh Q/A via Workflow structured-output + the BINDING live Playwright gate (qa.md §1c).
7. LOG: append harness_log.md (after PASS). 8. DECIDE: flip 70.1 → done.

## Boundaries (binding)

$0 metered, paper-only; UI + api.ts only — NO backend route change, NO live-loop behavior change, NO risk-limit
threshold moved (the panel only exposes the EXISTING operator-adjustable ALLOWED_KEYS caps, which is intended
operator control, not a new behavior). No emojis; Phosphor icons; navy/slate palette per frontend rules; every
new render has error/loading/empty states. Out-of-scope knobs (factor_corr, reject_binding) explicitly deferred.

## References

- `handoff/current/research_brief_70.1.md` (this step's research gate; 8 sources, React/MDN/web.dev)
- `handoff/current/design_trade_diversity_70.md` (70.0 design pack) + `confirmed_findings.json` (#4/#12/#13/#11)
- Code: cockpit-helpers.tsx:444-498, manage/page.tsx:86-246, api.ts:247-251, positions/page.tsx:23-29,
  paper_trading.py:578-628 (risk-limits routes), risk_overrides.py (ALLOWED_KEYS/describe), settings.py:277
