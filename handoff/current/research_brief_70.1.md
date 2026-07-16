# Research Brief — Step 70.1 (S1 "can't change MAX POSITIONS PER SECTOR")

**Phase:** 70.1
**Tier:** moderate
**Date:** 2026-07-16
**HEAD:** 3dd7fc53
**Researcher:** Layer-3 RESEARCHER subagent (effort=max)
**Status:** IN PROGRESS (write-first, appended incrementally)

Scope: a REAL, live-confirmed frontend bug (controlled number-input clear-snapback)
+ missing risk-override transparency + missing BUY-gating knob editors. Four sub-fixes.
Constraint: UI + `frontend/src/lib/api.ts` ONLY (backend routes already exist); $0 metered; paper-only; no emojis (Phosphor icons only).

---

## Three-variant query disclosure (research-gate mandatory)

Per `.claude/rules/research-gate.md`, each topic ran ≥3 query variants:
current-year frontier (`2026`), last-2-year window (`2025`/`2024`), and
year-less canonical (no year suffix).

| Topic | Frontier (2026) | Last-2yr (2025/2024) | Year-less canonical |
|---|---|---|---|
| Controlled numeric input allowing empty | `react controlled number input allow empty 2026` | `react number input clear to empty 2025` | `react controlled input value undefined empty string` |
| String-state-then-coerce pattern | `react input string state coerce number on blur 2026` | `react number input store string parse on submit 2025` | `react controlled input number vs string state` |
| Client-side numeric range validation UX | `inline field validation vs api error message 2026` | `form field range validation ux 2025` | `client side form validation inline error message` |
| Next.js 15 / React 19 controlled input | `react 19 controlled input onChange 2026` | `next.js 15 form input 2025` | `react controlled components forms docs` |

---

## Internal audit (internal_files_inspected)

Files read in full or in the load-bearing region:

1. `frontend/src/components/paper-trading/cockpit-helpers.tsx` (full, 565 lines) — the `PaperSettingNum` number-input component (:444-498).
2. `frontend/src/app/paper-trading/manage/page.tsx` (full, 253 lines) — the Save / `updateSettings` flow + which fields render (:197-248).
3. `frontend/src/lib/api.ts` (full, 779 lines) — `updateSettings` (:251); no risk-limits fns exist; insertion point after the Paper-Trading block (~:440).
4. `frontend/src/app/paper-trading/positions/page.tsx` (:1-60) — the false comment at :23-29.
5. `backend/api/paper_trading.py` (:52-59 `RiskLimitRequest`; :578-628 risk-limits GET/PUT/DELETE).
6. `backend/services/risk_overrides.py` (full, 223 lines) — `BOUNDS` / `ALLOWED_KEYS` (:51-57), `describe()` shape (:208-222).
7. `backend/config/settings.py` (:258-309) — `paper_max_per_sector_nav_pct` field (:277-282).
8. `backend/api/settings_api.py` (:125 `SettingsUpdate`, :156-170 paper fields; :397 `update_settings`) — **nav_pct is NOT a settable field here**.
9. `frontend/src/lib/types.ts` (:530 `FullSettings`) — has `paper_max_positions`/`paper_max_per_sector`/`paper_min_cash_reserve_pct`, NOT `paper_max_per_sector_nav_pct`.

### EXACT snap-back mechanism (the bug, on HEAD 3dd7fc53)

`PaperSettingNum` in `cockpit-helpers.tsx`:

```tsx
const stored = settings[field];              // :465  e.g. 2
const draft  = dirty[field];                 // :466
const value  = (draft ?? stored ?? "") as number | string;  // :467  <-- ANTI-PATTERN
...
<input type="number" min={min} max={max} step={step} value={value}
  onChange={(e) => {
    const raw = e.target.value;              // :478
    const next = raw === "" ? undefined : Number(raw);       // :479
    setDirty((d) => {
      const merged = { ...d };
      if (next === undefined || next === stored) {           // :482
        delete merged[field];                // :483  <-- deletes draft on empty
      } else {
        merged[field] = next;                // :485
      }
      return merged;
    });
  }}
/>
```

**Why it snaps back (line-by-line):**
- The controlled `value` binds to `draft ?? stored ?? ""` (:467). `??` falls through to `stored` whenever `draft` is `null`/`undefined`. This is the `value={x ?? stored}` anti-pattern named in the task.
- When the user CLEARS the field, `onChange` fires with `raw === ""` → `next = undefined` (:479).
- The `next === undefined` branch (:482) runs `delete merged[field]` (:483), so `dirty[field]` becomes `undefined`.
- On re-render, `value = (undefined ?? stored ?? "")` = **`stored`**. The empty state is unrepresentable — the field IMMEDIATELY repaints the stored digits. Clearing is impossible.
- Consequence (the "append" symptom): with the field stuck showing `stored` (e.g. `2`) and the cursor at the end, the next keystroke `5` yields `"25"` → `Number("25")=25`. `25 !== stored` and `!== undefined`, so `merged[field]=25` (:485). `25 > max 20` → save-time generic 422.

### No client-side range validation (sub-fix 2)

- `min`/`max`/`step` are passed as HTML attributes (:472-474) only. Native `type=number` `min`/`max` do NOT block programmatic/typed values from entering React state — they only affect spinner arrows + `:invalid` styling + native form `reportValidity()` (never called here). So `25` sails into `dirty` and only fails at save.
- Save-time failure path: `manage/page.tsx` `handleSettingsSave` (:86) → `updateSettings(manageDirty)` (:95) → `api.ts` throws `Invalid request to /api/settings/: <detail>` for status 422 (api.ts:125-127). That generic string is surfaced in `settingsError` (:100). No field is named; the operator can't tell WHICH knob or WHAT the bound is.

### risk_overrides "shadow" (sub-fix 3) — the transparency gap

- `risk_overrides.py` `BOUNDS`/`ALLOWED_KEYS` (:51-57) = **4 keys**: `paper_max_per_sector` (int 0-20), `paper_max_per_sector_nav_pct` (float 0-100), `paper_min_cash_reserve_pct` (float 0-50), `paper_max_positions` (int 1-50).
- At decide-time, `portfolio_manager` reads `get_effective(key, settings.X)` (risk_overrides.py:134-141): if a runtime override is set it SHADOWS the settings/.env value. So an operator can set `paper_max_per_sector=2` in the Manage settings block (writes .env) yet the live cap is a stale override of, say, 10 — silently. Nothing in the UI shows this.
- Backend already exposes the truth: `GET /api/paper-trading/risk-limits` (paper_trading.py:578) returns per key `{type,min,max,description,overridden,override_value,settings_default,effective_value}`. `PUT` (:595, needs `confirmation:"SET_RISK_LIMIT"`) sets an override; `DELETE /risk-limits/{key}` (:615) clears it. **No frontend consumes any of these.** (`risk_limits` in `RiskDashboard.tsx`/`types.ts:434` is an UNRELATED RiskJudge stop_loss/max_dd display — not this surface.)
- `RiskLimitRequest` (paper_trading.py:52): `{ key: str, value: float, confirmation: str, reason: str = "manual" }`.

### paper_max_per_sector_nav_pct editor (sub-fix 4) — KEY ARCHITECTURAL FINDING

- `paper_max_per_sector_nav_pct` (settings.py:277, default 30.0, ge=0 le=100) is an ACTIVE BUY-gating knob (NAV%-per-sector cap) with **no UI editor**.
- It is **NOT** in `SettingsUpdate` (settings_api.py:156-170) — so `PUT /api/settings/` would SILENTLY DROP it (`model_dump(exclude_none)` + not a declared field). Adding it to the Manage settings block would require a BACKEND change (out of the "UI + api.ts only" constraint) AND a `FullSettings` type addition.
- It **IS** in `risk_overrides.ALLOWED_KEYS`. So the ONLY no-backend-change path to make it editable is the **risk-limits endpoint** (`PUT /api/paper-trading/risk-limits`). This is also the correct semantics: the override is read at decide-time (no restart, picked up next cycle) — exactly where the cap gates BUYs.
- **Therefore sub-fix 4 is delivered THROUGH the sub-fix 3 risk-limits panel** (nav_pct is one of its 4 keys). One panel closes both.
- `paper_max_factor_corr` (settings.py:291) and `paper_risk_judge_reject_binding` (settings.py:306) are also BUY-gating knobs with no editor, BUT neither is in `SettingsUpdate` NOR in `ALLOWED_KEYS` → they CANNOT be added UI-only. **Out of scope** for 70.1 (would need a backend `SettingsUpdate` change). Note this so Main does not add a control that silently no-ops.

### False comment (sub-fix, positions/page.tsx:29)

`positions/page.tsx:23-29`:
```
// Hard-coded default cap; the operator-tunable setting is
// paper_max_per_sector_nav_pct (lives at /paper-trading/manage; reachable
// via the Settings gear button in the layout header).
...
const DEFAULT_SECTOR_CAP_PCT = 30;
```
False: nav_pct is NOT editable at /manage today (no editor exists). After sub-fix 4 ships the risk-limits panel, the comment becomes true if it points at that panel. Fix the wording to match reality (and, ideally, fetch the effective value rather than hardcode 30 — but the hardcode is a separate concern; the binding requirement is the comment must not claim editability that doesn't exist).

---

## Source table

### Read in full (via WebFetch) — 8

| # | Source | Tier | Key takeaway |
|---|---|---|---|
| 1 | [react.dev `<input>` reference](https://react.dev/reference/react-dom/components/input) | 1 Official | "The `value` you pass to controlled components should not be `undefined` or `null`. If you need the initial value to be empty ... initialize your state variable to an empty string (`''`)." Clearing = `setState('')`. `Number(age)` coercion shown for `type=number`. "Every controlled input needs an `onChange` ... React will revert the input after every keystroke back to the `value` you specified" if not synced. |
| 2 | [MDN — Nullish coalescing `??`](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Operators/Nullish_coalescing) | 1 Official | `??` returns RHS only when LHS is `null`/`undefined` — NOT for `''` or `0`. `const preservingFalsy = "" ?? "x"` → `""`. So if the draft held `""`, `draft ?? stored` would PRESERVE the empty; the snap-back only happens because empty is mapped to `undefined`/deleted first. |
| 3 | [web.dev — Learn Forms: Validation](https://web.dev/learn/forms/validation) | 1 Official | Constraint Validation API; `min`/`max` for number inputs; `setCustomValidity('...')` for consistent custom messages; use `:user-invalid` so errors "only apply after user interaction" (not while typing); "You must also ... validate data ... on your backend server." |
| 4 | [MDN — Using the Constraint Validation API](https://developer.mozilla.org/en-US/docs/Web/HTML/Guides/Constraint_validation) | 1 Official | `validity.rangeUnderflow` (< min), `rangeOverflow` (> max), `stepMismatch`; `setCustomValidity("")` = valid, any string = error msg; read `element.validity` + `element.validationMessage`. "HTML Constraint validation doesn't remove the need for validation on the server side" — and constraint validations "are only run for user input, and not if you set the value ... using JavaScript" (why native min/max alone doesn't catch our typed 25). |
| 5 | [Medium — Stop setting controlled inputs' values to null (Stephan W)](https://medium.com/@gemfromja/stop-setting-controlled-inputs-values-to-null-9f4c641efe9b) | 3 Blog | `null`/`undefined` desyncs state and UI (state changes but input keeps prior text); always keep value a defined string, `''` for empty; number inputs use a valid number, never `null`. |
| 6 | [LogRocket — UX of form validation: inline or after submission](https://blog.logrocket.com/ux-design/ux-form-validation-inline-after-submission/) | 3 Blog | "keep the validation message until the user has finished with an input field and moved on" — validate on blur, not mid-typing; show an error SUMMARY + also highlight the specific field (SurveyMonkey pattern); field-specific over a single generic banner. |
| 7 | [CroCoder — Clear React input value (controlled vs uncontrolled)](https://www.crocoder.dev/blog/react-input-component-clear-value-after-input) | 4 Practitioner | Controlled pattern stores the raw string in `useState("")` and updates on every `onChange` via `e.target.value`; clearing = `setValue("")`. Confirms string-state approach (does not cover numeric coercion). |
| 8 | [facebook/react issue #7779 — controlled `input[type=number]` blank init](https://github.com/facebook/react/issues/7779) | 2 Canonical issue | Confirms this is a canonical, long-standing React problem: `""`/`null`/`undefined` each fail differently; `0` is semantically wrong for "blank". (Fetch surfaced the problem framing, not a maintainer solution — the modern react.dev guidance in #1 supersedes it.) |

### Snippet-only (collected, not read in full) — representative

| # | Source | Why not read in full |
|---|---|---|
| 9 | [MDN — HTMLInputElement.setCustomValidity()](https://developer.mozilla.org/en-US/docs/Web/API/HTMLInputElement/setCustomValidity) | Covered by #4 |
| 10 | [MDN — ValidityState](https://developer.mozilla.org/en-US/docs/Web/API/ValidityState) | Covered by #4 |
| 11 | [MDN — Client-side form validation (Learn)](https://developer.mozilla.org/en-US/docs/Learn_web_development/Extensions/Forms/Form_validation) | Overlaps #3/#4 |
| 12 | [DEV — Optional chaining & nullish coalescing in React](https://dev.to/antozanini/start-using-optional-chaining-and-nullish-coalescing-in-react-269g) | Community tier; #2 authoritative |
| 13 | [DEV — Fixing uncontrolled→controlled warnings](https://dev.to/john_muriithi_swe/understanding-and-fixing-uncontrolled-to-controlled-input-warnings-in-react-4n5e) | Covered by #1/#5 |
| 14 | [github/react#11417 — treat value={null} as empty string](https://github.com/react/react/issues/11417) | Context for #5 |
| 15 | [react-hook-form#2550 — number input outputs string on submit](https://github.com/react-hook-form/react-hook-form/issues/2550) | RHF not used in this codebase |
| 16 | [react-hook-form#2688 — number vs string on submit](https://github.com/react-hook-form/react-hook-form/issues/2688) | RHF not used |
| 17 | [react-admin — NumberInput](https://marmelab.com/react-admin/NumberInput.html) | Library-specific |
| 18 | [bobbyhadz — numbers-only input in React](https://bobbyhadz.com/blog/react-only-number-input) | Practitioner; #7 covers pattern |
| 19 | [TetraLogical — form validation & error messages (2024)](https://tetralogical.com/blog/2024/10/21/foundations-form-validation-and-error-messages/) | Accessibility angle; #6 covers UX |
| 20 | [Zuko — inline validation in forms](https://www.zuko.io/blog/inline-validation-in-online-forms) | Overlaps #6 |
| 21 | [WebAIM — accessible form validation & error recovery](https://webaim.org/techniques/formvalidation/) | A11y reference |
| 22 | [Cloud Four — custom validation messages (progressive enhancement)](https://cloudfour.com/thinks/progressively-enhanced-form-validation-part-4-custom-validation-messages/) | Overlaps #3/#4 |
| 23 | [legacy.reactjs.org — Forms](https://legacy.reactjs.org/docs/forms.html) | Superseded by #1 |
| 24 | [Medium — React 19 form handling (useActionState/useFormStatus)](https://medium.com/@ignatovich.dm/enhancing-form-handling-in-react-19-a-look-at-action-useformstate-and-useformstatus-a5ee68d6bf93) | React 19 Actions adjacent, not required |

(Additional dupes across the 7 searches — rescript forum, w3schools, handsonreact, staticforms, ivyforms x2, dhiwise, react.dev/textarea, RHF Controller — not itemized.)

---

## Recency scan (last 2 years)

The canonical controlled-input value semantics (string value, `''` for empty,
`onChange` sync) are **evergreen and unchanged** through the current React docs
(react.dev, maintained 2024-2026). Findings from the last-2-year window:

- **React 19 (2024-2025) form Actions** (`useActionState`/`useFormStatus`,
  source #24) add server-action ergonomics but do **NOT** change controlled
  `<input>` value semantics — value is still a string, empty is still `''`,
  and `onChange` still must sync. The string-state-then-coerce fix is
  unaffected and remains correct on React 19 (this project's stack).
- **`:user-invalid`** (web.dev #3) is the modern, Baseline-available way to
  defer validation styling until after interaction — a newer refinement over
  the old `:invalid` (which flags fields before the user has typed). It
  complements, and is consistent with, the on-change-with-non-empty-guard
  approach recommended below.
- **TetraLogical 2024** (#19) reinforces the accessibility framing
  (`aria-invalid`, programmatic association of message to field) that the
  design below adopts.
- No last-2-year source **contradicts** the string-state-then-coerce pattern or
  the inline field-specific validation recommendation. The old facebook/react
  issue (#8, 2016) is superseded, not contradicted, by the explicit react.dev
  "initialize to `''`" guidance.

---

## Design recommendations (implementation-ready for Main)

All fixes are **UI + `api.ts` only**. No backend route change (risk-limits
routes exist). No emojis; Phosphor icons via `@/lib/icons.ts`; navy/slate
tokens; rose for errors, amber for warnings/unsaved (per `frontend.md`).

### Sub-fix 1 — controlled-input clear-snapback (`cockpit-helpers.tsx` PaperSettingNum, :444-498)

**Pattern: string-state-then-coerce** (React docs #1 + MDN #2 + CroCoder #7).
Represent the raw editing text as an explicit `string` so `""` is a first-class,
representable state; coerce to `Number` only when writing to the parent `dirty`
and at save.

```tsx
const stored = settings[field];
const [text, setText] = React.useState<string>(stored != null ? String(stored) : "");
// Re-seed after a save (stored changes) — guarded so it never clobbers an active edit:
React.useEffect(() => {
  if (dirty[field] === undefined) setText(stored != null ? String(stored) : "");
}, [stored, dirty, field]);

const num = text.trim() === "" ? undefined : Number(text);
const outOfRange = num !== undefined && !Number.isNaN(num) && (num < min || num > max);
const badNum = text.trim() !== "" && Number.isNaN(num);
const error = badNum ? "Enter a number." : outOfRange ? `Must be between ${min} and ${max}.` : undefined;

<input
  type="number" min={min} max={max} step={step}
  value={text}                               // ALWAYS a defined string; "" is allowed (no snap-back)
  aria-invalid={error ? true : undefined}
  onChange={(e) => {
    const raw = e.target.value;
    setText(raw);                            // empty stays empty; typing "5" after clear yields "5", not "25"
    const n = raw.trim() === "" ? undefined : Number(raw);
    setDirty((d) => {
      const merged = { ...d };
      if (n === undefined || Number.isNaN(n) || n === stored) delete merged[field];
      else merged[field] = n;                // coerce to number for the parent/save
      return merged;
    });
  }}
  className={clsx("w-full rounded-md border bg-navy-900 px-3 py-2 text-sm text-slate-100 focus:outline-none",
    error ? "border-rose-500/60 focus:border-rose-500" : "border-navy-600 focus:border-sky-500/50")}
/>
```

- Why it fixes the snap-back: `value` is bound to local `text`, not to
  `draft ?? stored`. Clearing sets `text=""` and it **stays** empty (React #1:
  init/represent empty as `''`). The append bug ("25") disappears because the
  field is genuinely empty before the user types the replacement.
- Alternative to the re-seed effect: pass `key={`${field}:${stored ?? ""}`}` on
  each `<PaperSettingNum>` in `manage/page.tsx` so the field remounts and
  re-seeds `useState` after a save (React "reset state with a key" idiom).
  Recommend the guarded effect OR the key — either is fine; the effect keeps
  focus, the key is less code.
- Keep the existing "unsaved" amber hint, gated on `num !== undefined && num !== stored && !error`.

### Sub-fix 2 — client-side range validation, field-specific message (same component + parent)

- The `error` string above is **field-specific** and names the bound
  (`Must be between {min} and {max}.`) — LogRocket #6 (field-specific > generic)
  and MDN #4 (rangeUnderflow/rangeOverflow). Render it inline under the input:
  `{error && <p className="mt-1 text-xs text-rose-400">{error}</p>}` (rose token;
  no emoji). Hide the `hint` when `error` is shown.
- Timing: show on-change but only when the field is non-empty (the `text.trim()!==""`
  guard) — satisfies web.dev #3 `:user-invalid` spirit (no error while the field
  is transiently empty mid-edit). Blur-based is an acceptable alternative (#6) but
  on-change-guarded is more immediate and still not premature.
- **Enforce (don't just display):** lift validity to the parent so Save is
  blocked BEFORE the 422. Add an `onValidity?: (field, error?: string) => void`
  prop; `React.useEffect(() => onValidity?.(field, error), [error, field])`.
  In `manage/page.tsx`, hold `const [fieldErrors, setFieldErrors] = useState<Record<string,string|undefined>>({})`;
  pass `onValidity={(f,e)=>setFieldErrors(p=>({...p,[f]:e}))}`; then
  `const hasErrors = Object.values(fieldErrors).some(Boolean)` and disable Save
  (`disabled={settingsSaving || hasErrors || Object.keys(manageDirty).length===0}`),
  plus a summary banner when `hasErrors` ("Fix the highlighted fields before saving.",
  rose). The existing generic-422 path (api.ts:125) stays as the server-side
  backstop — web.dev #3 / MDN #4: always validate server-side too.

### Sub-fix 3 — surface & clear the risk_overrides shadow (`api.ts` + `manage/page.tsx`)

**`api.ts`** — add after the Paper-Trading block (~:440). Types + 3 fns; bake the
confirmation token in so the UI never surfaces it:

```tsx
export interface RiskLimitSpec {
  type: string; min: number; max: number; description: string;
  overridden: boolean; override_value: number | null;
  settings_default: number | null; effective_value: number | null;
}
export interface RiskLimitsResponse {
  risk_limits: Record<string, RiskLimitSpec>; allowed_keys: string[];
}
export function getRiskLimits(): Promise<RiskLimitsResponse> {
  return apiFetch("/api/paper-trading/risk-limits");
}
export function setRiskLimit(key: string, value: number, reason = "operator UI") {
  return apiFetch("/api/paper-trading/risk-limits", {
    method: "PUT",
    body: JSON.stringify({ key, value, confirmation: "SET_RISK_LIMIT", reason }),
  }); // confirmation REQUIRED by paper_trading.py:599; reason max_length 200
}
export function clearRiskLimit(key: string) {
  return apiFetch(`/api/paper-trading/risk-limits/${encodeURIComponent(key)}`, { method: "DELETE" });
}
```

**`manage/page.tsx`** — new "Risk limits (live overrides)" card (sibling after
Trading settings). On mount `getRiskLimits()` → local state (mirror the existing
`getFullSettings` effect). For each of the 4 keys render a row:

- Label = spec.description; show **configured** (`settings_default`) and
  **effective** (`effective_value`) side by side; when `overridden`, emphasize the
  effective value and show an amber `OVERRIDE ACTIVE` badge.
- A bounded number input (reuse the sub-fix-1/2 validation against `spec.min`/`spec.max`)
  + a **Set** button → `setRiskLimit(key, value)` then refetch `getRiskLimits()`.
- A **Clear override** button (disabled unless `overridden`) → `clearRiskLimit(key)`
  then refetch. "Cleared" reverts to the settings default (risk_overrides.py:24).
- **Active-override warning banner** at the top of the card when any
  `overridden` (amber `border-amber-500/30 bg-amber-950/30`, Phosphor `Warning`
  icon from icons.ts, no emoji): "N live override(s) active — the effective cap
  differs from the saved setting. Clear to revert to the .env value." + list keys.
- After every set/clear, also call `refresh()` from `usePaperTradingData` so the
  Positions/RiskMonitor views reflect the new effective cap (backend already
  invalidates `paper:*`, paper_trading.py:605/621).

This directly surfaces the shadow: the operator sees when the live BUY cap differs
from what the settings block shows, and can clear it in one click.

### Sub-fix 4 — editor for `paper_max_per_sector_nav_pct` (delivered via the sub-fix-3 panel)

- `paper_max_per_sector_nav_pct` is **one of the 4 `ALLOWED_KEYS`**, so the
  risk-limits panel above **is** its editor — Set-override gives it a runtime,
  no-restart value read at decide-time (exactly where it gates BUYs). This is the
  **only UI-only path**: it is NOT in `SettingsUpdate` (settings_api.py:156-170),
  so putting it in the .env settings block would silently no-op without a backend
  change (out of scope).
- **Out of scope (flag to Main, do not add UI-only):** `paper_max_factor_corr`
  (settings.py:291) and `paper_risk_judge_reject_binding` (settings.py:306) are
  also BUY-gating knobs with no editor, but neither is in `SettingsUpdate` nor
  `ALLOWED_KEYS` → no UI-only path; a control for them would silently drop. Defer
  to a follow-up that adds them to `SettingsUpdate` (backend change).

### False-comment fix (`positions/page.tsx:23-29`)

The comment claims nav_pct is editable at /manage — false today. After sub-fix 3/4
it becomes editable via the new "Risk limits (live overrides)" panel. Correct the
comment to: (a) point at that panel (runtime override, no restart), and (b) state
that `DEFAULT_SECTOR_CAP_PCT = 30` is only a **display fallback** that may NOT
reflect an active override. Optional (nice-to-have): fetch `effective_value` for
`paper_max_per_sector_nav_pct` via `getRiskLimits()` for an accurate rendered cap
instead of the hardcoded 30 — but the binding requirement is only that the comment
stop asserting editability that (until this step) did not exist.

### Per-claim citation map

- controlled value must be a defined string; init/clear via `''` — React #1, Medium #5.
- `??` preserves `''`/`0` (so storing `""` in the draft would NOT snap back; the bug is the empty→undefined→delete mapping) — MDN #2.
- store raw string, coerce with `Number()` at write/save — React #1 (`Number(age)`), CroCoder #7, facebook/react #8 (problem).
- inline field-specific message; validate after interaction, not mid-typing; summary + highlight — LogRocket #6, web.dev #3 (`:user-invalid`), MDN #4.
- native `min`/`max` alone won't block typed values (constraint validation runs only for user input; must also validate server-side) — MDN #4, web.dev #3.
- `setCustomValidity`/`ValidityState` optional native parity — MDN #4, web.dev #3.

---

## Gate envelope

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 8,
  "snippet_only_sources": 16,
  "urls_collected": 40,
  "recency_scan_performed": true,
  "internal_files_inspected": 9,
  "gate_passed": true
}
```
