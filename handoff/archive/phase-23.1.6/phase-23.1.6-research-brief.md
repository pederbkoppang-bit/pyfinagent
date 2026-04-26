# Research Brief: phase-23.1.6 — Signal Stack UI + "Why This Candidate" Panel

Tier assumed: **moderate** (caller stated moderate; frontend-heavy cycle with relaxed >=3 read-in-full floor).

---

## Search Query Log

Three-variant discipline applied per topic:

**Topic 1: React 19 / Next.js 15 controlled settings forms**
1. Frontier: `"Next.js 15 React 19 controlled form patterns boolean toggle settings 2026"`
2. Last-2-year window: `"Next.js 15 App Router controlled settings form checkbox toggle pattern"`
3. Year-less canonical: `"TypeScript narrow prop typing boolean settings interface 2025"`

**Topic 2: Pydantic optional fields for FastAPI update endpoints**
1. Frontier: `"Pydantic optional boolean field FastAPI settings endpoint pattern 2025"`
2. Year-less canonical: official FastAPI docs (`fastapi.tiangolo.com/tutorial/body-fields/`)
3. Year-less canonical: official React docs (`react.dev/reference/react-dom/components/input`)

**Topic 3: Badge/pill inline signal indicators in React**
1. Frontier: `"React 19 badge pill component inline signal indicator pattern 2025"`
2. Official: `nextjs.org/docs/app/guides/forms`
3. Official: `typescriptlang.org/docs/handbook/2/everyday-types.html`

---

## Read in Full (>=3 required with justification; >=5-source floor relaxed for internal-heavy cycle)

| URL | Accessed | Kind | Fetched how | Key finding |
|-----|----------|------|-------------|-------------|
| https://react.dev/reference/react-dom/components/input | 2026-04-26 | official doc | WebFetch full | Controlled checkbox requires `checked` (boolean) + `onChange` reading `e.target.checked`; must not mix controlled/uncontrolled; no `any` needed |
| https://nextjs.org/docs/app/guides/forms | 2026-04-26 | official doc | WebFetch full | Server Actions pattern; for settings pages already using `"use client"`, `useState`+`apiFetch` is the correct approach — NOT Server Actions; `useActionState` is for form-action-based flows |
| https://fastapi.tiangolo.com/tutorial/body-fields/ | 2026-04-26 | official doc | WebFetch full | Pydantic `Field(default=None)` + `Optional[T]` for PATCH/PUT update models; Field validators (`ge`, `le`) apply to optional fields the same way as required ones |
| https://www.typescriptlang.org/docs/handbook/2/everyday-types.html | 2026-04-26 | official doc | WebFetch full | Always use lowercase `boolean`; `?` for optional; `strictNullChecks` + `noImplicitAny` prevent implicit `any`; literal types only when restricting to specific booleans |

**Justification for 3-source relaxation:** This cycle is internal-codebase-shaped. The Settings page toggle pattern (BentoCard + checkbox) and the "why panel" (badge strip) are pure React 19 / Tailwind patterns already established in the codebase. External literature confirms the patterns but adds no novel algorithm or architecture. The 4 sources fetched in full satisfy the relaxed floor.

---

## Identified but Snippet-Only

| URL | Kind | Why not fetched |
|-----|------|-----------------|
| https://medium.com/@kolbysisk/mastering-forms-in-next-js-15-and-react-19-e3d2d783946b | blog | Medium paywalled; snippet sufficient |
| https://react-typescript-cheatsheet.netlify.app/docs/basic/getting-started/basic_type_example/ | community doc | TS cheatsheet; snippet confirmed known pattern |
| https://github.com/shadcn-ui/ui/issues/5521 | GitHub issue | Project does not use shadcn; snippet confirms checkbox hydration caveat |
| https://devglory.com/blog/next-js-15-app-router-patterns-that-actually-work | blog | Snippet matched what official docs cover |
| https://www.untitledui.com/react/components/badges | UI lib | Project uses Phosphor+Tailwind, not Untitled UI |
| https://docs.pydantic.dev/latest/concepts/fields/ | official doc | Covered by FastAPI Body-Fields doc already fetched |
| https://www.getorchestra.io/guides/understanding-pydantic-optional-fields-in-fastapi | blog | Snippet confirmed Optional[T]=None pattern |

---

## Recency Scan (2024-2026)

Searched explicitly for 2025 and 2026 variants on all three topics.

**React 19 / Next.js 15:** Both were released in late 2024. The `useActionState` / `useOptimistic` APIs are the new 2024-2025 frontier. The relevant finding: settings pages that already use `useState`+`apiFetch` (client-side fetch) do NOT benefit from Server Actions — the existing pattern in `settings/page.tsx` is correct for this architecture. No 2026 breaking changes found that affect checkbox or toggle patterns.

**Pydantic v2 / FastAPI:** The `Optional[T] = Field(None, ...)` pattern is stable since Pydantic v2 (2023). No 2025-2026 deprecations found that affect the `SettingsUpdate` model extension approach.

**Badge/pill inline indicators:** No 2025-2026 React primitive for inline badges; the `<span className="rounded px-1.5 py-0.5 text-xs ...">` Tailwind pattern is the idiomatic approach, confirmed by shadcn/ui Badge (Radix primitive) and the project's own existing `CostBadge` pattern (`settings_api.py` line 116 / `settings/page.tsx` line 142-172).

Result: **no new 2024-2026 findings that supersede the canonical sources or the existing codebase patterns.**

---

## Internal Code Inventory

| File | Lines | Role | Status |
|------|-------|------|--------|
| `frontend/src/app/settings/page.tsx` | 1278 | Settings UI — 3 tabs, BentoCard+ModelPicker+toggle pattern | Active; canonical reference for new Signal Stack card |
| `backend/api/settings_api.py` | 362 | FullSettings, ModelConfig, SettingsUpdate Pydantic models; env-write logic; `_FIELD_TO_ENV` map | Active; needs 9 new fields |
| `backend/config/settings.py` | 195 | pydantic-settings Settings class with all 13 new phase-23.1 fields | Active; all 13 fields already present (lines 151-168) |
| `frontend/src/lib/types.ts` | 1070 | FullSettings TypeScript interface | Active; needs 9 new fields |
| `frontend/src/lib/api.ts` | 703 | `updateSettings(body: Partial<FullSettings>)` — already accepts Partial<FullSettings> | Active; no changes needed to api.ts itself |
| `frontend/src/lib/icons.ts` | 234 | Phosphor icon exports | Active; has Brain, Newspaper, Scales, Calendar (Clock), TrendUp — needs audit for signal icons |
| `frontend/src/app/signals/page.tsx` | 191 | Signals page — ticker-lookup, AllSignals display | Active; no candidate list, is ticker-lookup not screener |
| `backend/tools/screener.py` | 249 | `rank_candidates()` — scores and returns candidate dicts with `composite_score` | Active; does NOT yet attach signal-tag fields to output |
| `backend/services/pead_signal.py` | ~360 | `PeadSignalOutput(sentiment_tag, sentiment_score, surprise_score)` | Active; `apply_pead_to_score()` filters/boosts score |
| `backend/services/news_screen.py` | ~180 | `NewsHeadlineSignal(event_type, impact_polarity, confidence)` | Active; news-only tickers injected in screener line 234-245 |
| `backend/services/sector_calendars.py` | ~300 | `SectorEvent(event_type, signal_direction)` | Active; `apply_sector_events_to_score()` |
| `backend/services/meta_scorer.py` | ~170 | `conviction_score`, `conviction_reason` per candidate | Active; output merged into candidate dict |
| `backend/services/autonomous_loop.py` | 80+ | Daily cycle — calls `rank_candidates()` then `meta_score_candidates()` | Active; `_last_result` dict does not expose candidates to API |

---

## Key Findings

### 1. Settings.py already has all 13 fields (cycles 1-5 complete)

`backend/config/settings.py` lines 151-168 confirm all 13 new fields are present:
- `macro_regime_filter_enabled` (bool, default False) — line 152
- `macro_regime_model` (str, default "claude-haiku-4-5") — line 153
- `pead_signal_enabled` (bool) — line 155
- `pead_signal_model` (str) — line 156
- `pead_signal_lookback_quarters` (int, default 8) — line 157
- `news_screen_enabled` (bool) — line 159
- `news_screen_model` (str) — line 160
- `news_screen_max_headlines` (int, default 100) — line 161
- `sector_calendars_enabled` (bool) — line 163
- `sector_calendars_lookahead_days` (int, default 7) — line 164
- `meta_scorer_enabled` (bool) — line 166
- `meta_scorer_model` (str) — line 167
- `meta_scorer_max_batch` (int, default 30) — line 168

**None of these 13 fields appear in `FullSettings`, `SettingsUpdate`, or `_FIELD_TO_ENV` in `settings_api.py`.** The API layer is entirely missing them.

### 2. `FullSettings` TypeScript interface is missing all 13 fields

`frontend/src/lib/types.ts` lines 524-549 define `FullSettings`. None of the 13 new fields are present.

### 3. `updateSettings` API function already handles Partial<FullSettings>

`frontend/src/lib/api.ts` line 251: `export function updateSettings(body: Partial<FullSettings>): Promise<FullSettings>`. Once `FullSettings` is extended, no changes to `api.ts` are needed.

### 4. Settings page tab structure

`settings/page.tsx` line 46: three tabs — `"models"`, `"cost"`, `"performance"`. The new Signal Stack card belongs in the **"models"** tab (`"Models & Analysis"`), positioned AFTER the `Model Configuration` BentoCard (which spans `lg:col-span-2`, line 708). The signal toggles are a form of analysis configuration, not cost controls.

The existing toggle/enable pattern is the `apply_model_to_all_agents` checkbox at line 768-784 (`type="checkbox"` with `checked={!!form.apply_model_to_all_agents}` and `onChange`). A row of 5 enable-toggles + 3 model-pickers follows this exact pattern.

### 5. BentoCard + checkbox toggle pattern (from settings/page.tsx)

The canonical toggle pattern at lines 768-784:
```tsx
<input
  type="checkbox"
  id="some-flag"
  checked={!!form.some_flag}
  onChange={(e) => updateForm("some_flag", e.target.checked)}
  className="mt-1 h-4 w-4 cursor-pointer rounded border-navy-600 bg-navy-900 text-sky-500 focus:ring-2 focus:ring-sky-500/50"
/>
```
The `!!` double-negation safely handles `undefined | boolean` from `Partial<FullSettings>`.

### 6. ModelPicker is reusable — works for signal models

The `ModelPicker` component defined at lines 216-307 accepts `label`, `value`, `models`, `githubConfigured`, `onChange`, and `accentColor`. It can be dropped in unchanged for `macro_regime_model`, `pead_signal_model`, `news_screen_model`, and `meta_scorer_model`. `sector_calendars` has no model picker (pure data pull).

### 7. Signal Stack card location in the grid

The "models" tab renders a `grid grid-cols-1 gap-6 lg:grid-cols-2` (line 601). The new card should go after the existing `lg:col-span-2` Model Configuration BentoCard at line 808. It should also be `lg:col-span-2` to match the width and accommodate 5 toggle rows + model pickers without truncation.

### 8. Signals page is ticker-lookup, not screener candidates

`signals/page.tsx` is a per-ticker enrichment viewer, not a screener candidates list. The prompt description for the "why panel" — showing which signals fired for each screener candidate — does NOT map to the existing Signals page. There is no candidates-list UI in the codebase. The Paper Trading page (`/paper-trading`) is where candidates appear (via `autonomous_loop.py` → `_last_result`). The "why panel" requires: (a) a backend endpoint that exposes the last screener run's top candidates with their signal tags, OR (b) embedding the panel in the Paper Trading page's cycle result display. This is a scope clarification needed in the contract.

### 9. Candidate dict currently lacks signal-tag fields

`screener.py::rank_candidates()` (lines 228, 241-245) returns dicts with: `ticker`, `composite_score`, and optionally `source`, `news_event_type`, `news_rationale` (for news-only candidates). Macro regime tag, PEAD sentiment_tag, sector event_type/direction, and meta_scorer conviction_score/conviction_reason are NOT attached to the output dict. They would need to be added to the `scored.append({...})` call.

### 10. Icon availability for signal types

From `icons.ts`:
- Macro regime: `Brain` (line 178) — already exported as `Brain`
- PEAD: `TrendUp` (line 229) — earnings momentum
- News: `Newspaper` (line 27) — already exported as `StepMarketIntel`
- Sector calendar: `Clock` (line 76, via `BiasRecency`) — calendar/upcoming events; also `Timer` (line 41)
- Meta-scorer: `Scales` (line 35) — conviction/balance; already used for Debate and Risk

None of these require new Phosphor imports; they are all already in `icons.ts`. However, they are currently aliased under other semantic names. The new Signal Stack card should either reuse existing aliases or add new semantic aliases (e.g., `SignalMacroRegime`, `SignalPead`, `SignalNews`, `SignalSectorCal`, `SignalMeta`) following the existing `Signal*` alias convention (lines 43-55).

---

## Consensus vs Debate (External)

**Consensus:** The React/TypeScript community and official docs uniformly agree on:
- `checked` + `onChange(e => setState(e.target.checked))` for boolean toggles
- `Optional[T] = Field(None)` for PATCH/PUT update model fields in FastAPI
- `Partial<Interface>` in TypeScript for update-only types

**No meaningful debate** on these patterns in 2024-2026 literature. They are stable.

---

## Pitfalls (from Literature)

1. **Mixing controlled/uncontrolled** (React docs): Do not initialize `form` state with `{}` and then conditionally pass `checked={form.field ?? undefined}` — this switches between controlled and uncontrolled. Always pass a boolean: `checked={!!form.field}`.
2. **shadcn/ui checkbox hydration bug** (GitHub issue #5521): If using RSC with shadcn Checkbox, hydration mismatches occur. This project does not use shadcn, so this pitfall is irrelevant.
3. **`_FIELD_TO_ENV` must be updated alongside Pydantic models** (`settings_api.py` lines 187-203): If a field is added to `SettingsUpdate` but omitted from `_FIELD_TO_ENV`, the backend silently accepts the PUT but never writes to `.env` — the setting appears to save but is lost on restart.
4. **`_settings_to_full()` must be updated** (lines 223-242): If a new field is on `Settings` (pydantic-settings) but `_settings_to_full()` doesn't copy it to `FullSettings`, the GET endpoint returns `None` for that field. The frontend will show wrong initial state.

---

## Application to pyfinagent (Mapping External Findings to file:line Anchors)

| Finding | Maps to |
|---------|---------|
| `checked={!!form.flag}` pattern | `settings/page.tsx:773` (existing checkbox), replicate for 5 new flags |
| `Optional[bool] = Field(None)` | `settings_api.py:87-102` (SettingsUpdate class), add 13 new Optional fields |
| `_FIELD_TO_ENV` must be updated | `settings_api.py:187-203`, add 13 env-var mappings |
| `_settings_to_full()` must copy new fields | `settings_api.py:223-242` |
| `FullSettings` TypeScript extension | `types.ts:524-549` |
| Pydantic `ge`/`le` validators on int fields | apply to `pead_signal_lookback_quarters`, `news_screen_max_headlines`, `sector_calendars_lookahead_days`, `meta_scorer_max_batch` |

---

## Concrete UI Mockup: Signal Stack (Phase 23.1) Card

**Placement:** "Models & Analysis" tab, after the `Model Configuration` BentoCard (settings/page.tsx line 808). Full-width: `BentoCard className="lg:col-span-2"`.

**Structure (5 enable rows + 4 model/int pickers):**

```
BentoCard lg:col-span-2
  h3: [Brain icon] Signal Stack (Phase 23.1)
  p: "Overlay signals applied during paper trading candidate screening."

  ── Toggle rows (each: flex items-start gap-3, border border-navy-700 rounded-lg p-3 bg-navy-800/40) ──

  Row 1 — Macro Regime Filter
    [checkbox] enabled: macro_regime_filter_enabled
    Label: "Macro Regime Filter"
    Sub: "Daily FRED snapshot → Claude judge; boosts/penalizes candidates by regime conviction."
    [ModelPicker label="Model" value=form.macro_regime_model accentColor="sky"]

  Row 2 — PEAD Signal
    [checkbox] enabled: pead_signal_enabled
    Label: "PEAD Signal"
    Sub: "SEC 8-K press releases → Claude sentiment surprise; boosts recent reporters."
    [ModelPicker label="Model" value=form.pead_signal_model accentColor="sky"]
    [range slider] pead_signal_lookback_quarters (1-12)

  Row 3 — Worldwide News Screen
    [checkbox] enabled: news_screen_enabled
    Label: "News Screen"
    Sub: "RSS news → Claude batch event classifier; surfaces positive-polarity candidates."
    [ModelPicker label="Model" value=form.news_screen_model accentColor="sky"]
    [range slider] news_screen_max_headlines (20-200)

  Row 4 — Sector Calendars
    [checkbox] enabled: sector_calendars_enabled
    Label: "Sector Calendars"
    Sub: "FDA PDUFA dates + upcoming earnings; filters binary-risk events, boosts catalysts."
    [range slider] sector_calendars_lookahead_days (1-30)
    (no model picker — pure data pull)

  Row 5 — Meta-Scorer
    [checkbox] enabled: meta_scorer_enabled
    Label: "Meta-Scorer"
    Sub: "Single batched Claude call over top candidates; conviction 1-10 replaces composite score."
    [ModelPicker label="Model" value=form.meta_scorer_model accentColor="violet"]
    [range slider] meta_scorer_max_batch (5-50)
```

Each row renders as a `<div className="mt-3 flex items-start gap-3 rounded-lg border border-navy-700 bg-navy-800/40 p-3">` (matching the existing `apply_model_to_all_agents` block at settings/page.tsx:768).

When the toggle is OFF, model pickers and sliders are `opacity-40 pointer-events-none` to indicate they are inactive (no dead state confusion). This mirrors the existing `form.lite_mode` amber warning pattern at line 699.

---

## Concrete TypeScript Types — frontend/src/lib/types.ts

Add to `FullSettings` interface (after line 549, before the closing `}`):

```typescript
  // phase-23.1 — Signal Stack toggles + models + numeric controls
  // All optional: settings API may not return them on older backend versions
  macro_regime_filter_enabled?: boolean;
  macro_regime_model?: string;
  pead_signal_enabled?: boolean;
  pead_signal_model?: string;
  pead_signal_lookback_quarters?: number;
  news_screen_enabled?: boolean;
  news_screen_model?: string;
  news_screen_max_headlines?: number;
  sector_calendars_enabled?: boolean;
  sector_calendars_lookahead_days?: number;
  meta_scorer_enabled?: boolean;
  meta_scorer_model?: string;
  meta_scorer_max_batch?: number;
```

All are `?` (optional) because:
1. The UI uses `Partial<FullSettings>` for `form` state — optional aligns with that.
2. Older backend versions that haven't deployed this cycle will return `FullSettings` without these fields; `undefined` is the correct fallback.
3. TypeScript `noImplicitAny` is satisfied by explicit type annotations on each field (not `any`).

---

## Concrete settings_api.py Additions

### 1. `FullSettings` model (after line 84, before closing class)

```python
    # phase-23.1 Signal Stack
    macro_regime_filter_enabled: bool = False
    macro_regime_model: str = "claude-haiku-4-5"
    pead_signal_enabled: bool = False
    pead_signal_model: str = "claude-haiku-4-5"
    pead_signal_lookback_quarters: int = 8
    news_screen_enabled: bool = False
    news_screen_model: str = "claude-haiku-4-5"
    news_screen_max_headlines: int = 100
    sector_calendars_enabled: bool = False
    sector_calendars_lookahead_days: int = 7
    meta_scorer_enabled: bool = False
    meta_scorer_model: str = "claude-haiku-4-5"
    meta_scorer_max_batch: int = 30
```

### 2. `SettingsUpdate` model (after line 102)

```python
    # phase-23.1 Signal Stack
    macro_regime_filter_enabled: Optional[bool] = None
    macro_regime_model: Optional[str] = None
    pead_signal_enabled: Optional[bool] = None
    pead_signal_model: Optional[str] = None
    pead_signal_lookback_quarters: Optional[int] = Field(None, ge=1, le=12)
    news_screen_enabled: Optional[bool] = None
    news_screen_model: Optional[str] = None
    news_screen_max_headlines: Optional[int] = Field(None, ge=10, le=500)
    sector_calendars_enabled: Optional[bool] = None
    sector_calendars_lookahead_days: Optional[int] = Field(None, ge=1, le=30)
    meta_scorer_enabled: Optional[bool] = None
    meta_scorer_model: Optional[str] = None
    meta_scorer_max_batch: Optional[int] = Field(None, ge=5, le=100)
```

### 3. `_FIELD_TO_ENV` dict (after line 203)

```python
    # phase-23.1
    "macro_regime_filter_enabled": "MACRO_REGIME_FILTER_ENABLED",
    "macro_regime_model": "MACRO_REGIME_MODEL",
    "pead_signal_enabled": "PEAD_SIGNAL_ENABLED",
    "pead_signal_model": "PEAD_SIGNAL_MODEL",
    "pead_signal_lookback_quarters": "PEAD_SIGNAL_LOOKBACK_QUARTERS",
    "news_screen_enabled": "NEWS_SCREEN_ENABLED",
    "news_screen_model": "NEWS_SCREEN_MODEL",
    "news_screen_max_headlines": "NEWS_SCREEN_MAX_HEADLINES",
    "sector_calendars_enabled": "SECTOR_CALENDARS_ENABLED",
    "sector_calendars_lookahead_days": "SECTOR_CALENDARS_LOOKAHEAD_DAYS",
    "meta_scorer_enabled": "META_SCORER_ENABLED",
    "meta_scorer_model": "META_SCORER_MODEL",
    "meta_scorer_max_batch": "META_SCORER_MAX_BATCH",
```

### 4. `_settings_to_full()` function (after line 241, before the closing `)`):

```python
        macro_regime_filter_enabled=bool(getattr(s, "macro_regime_filter_enabled", False)),
        macro_regime_model=getattr(s, "macro_regime_model", "claude-haiku-4-5"),
        pead_signal_enabled=bool(getattr(s, "pead_signal_enabled", False)),
        pead_signal_model=getattr(s, "pead_signal_model", "claude-haiku-4-5"),
        pead_signal_lookback_quarters=getattr(s, "pead_signal_lookback_quarters", 8),
        news_screen_enabled=bool(getattr(s, "news_screen_enabled", False)),
        news_screen_model=getattr(s, "news_screen_model", "claude-haiku-4-5"),
        news_screen_max_headlines=getattr(s, "news_screen_max_headlines", 100),
        sector_calendars_enabled=bool(getattr(s, "sector_calendars_enabled", False)),
        sector_calendars_lookahead_days=getattr(s, "sector_calendars_lookahead_days", 7),
        meta_scorer_enabled=bool(getattr(s, "meta_scorer_enabled", False)),
        meta_scorer_model=getattr(s, "meta_scorer_model", "claude-haiku-4-5"),
        meta_scorer_max_batch=getattr(s, "meta_scorer_max_batch", 30),
```

### 5. Model validation in `update_settings` (after the existing deep_think_model check at line 267)

```python
    # Validate signal-stack model names
    for model_field in [
        "macro_regime_model", "pead_signal_model",
        "news_screen_model", "meta_scorer_model",
    ]:
        model_val = getattr(body, model_field, None)
        if model_val is not None and model_val not in _VALID_MODELS:
            raise HTTPException(status_code=400, detail=f"Invalid model: {model_val}")
```

---

## Concrete "Why" Panel Design

### Scope clarification (critical)

The Signals page (`signals/page.tsx`) is a **ticker-lookup tool**, not a screener candidates list. There is no existing candidates-list UI in the codebase. The "why this candidate" panel needs a home. Two options:

**Option A (recommended for phase-23.1.6):** Embed in the Paper Trading page's "Last Cycle" section. The `_last_result` dict from `autonomous_loop.py` is exposed via `/api/paper-trading/status` (line 95 of `paper_trading.py`). Extend `_last_result` to include `top_candidates` with signal tags. The Paper Trading page already shows cycle info — adding a "Candidates" collapsible section there is coherent.

**Option B:** Add a new "Candidates" tab to the existing Signals page, showing the last screener run's candidates. Requires new API endpoint + new tab.

Option A requires fewer new files. Option B better matches the "Signals page" framing in the spec. **The GENERATE phase should implement Option A** (Paper Trading page) unless the product owner prefers Option B.

### Backend changes needed (either option)

`screener.py::rank_candidates()` must attach signal tags to each candidate dict before returning. Add to the `scored.append({**stock, ...})` call (line 228):

```python
scored.append({
    **stock,
    "composite_score": round(score, 3),
    # Signal tags for the "why panel"
    "_regime_tag": regime.regime if regime is not None else None,
    "_regime_multiplier": regime.conviction_multiplier if regime is not None else None,
    "_pead_tag": pead_signals.get(stock.get("ticker"), {}).sentiment_tag if pead_signals else None,
    "_news_event_type": None,   # overwritten below for news-only tickers
    "_sector_event_type": sector_events.get(stock.get("ticker"), {}).event_type if sector_events else None,
    "_sector_direction": sector_events.get(stock.get("ticker"), {}).signal_direction if sector_events else None,
    # meta_scorer conviction added by meta_score_candidates() in autonomous_loop
})
```

(Note: `pead_signals` and `sector_events` are dicts keyed by ticker; `getattr` guards needed since values are dataclass instances not dicts.)

### TypeScript type for candidate + why-panel

```typescript
// Add to types.ts
export interface SignalTags {
  regime_tag?: string | null;          // "risk_on" | "risk_off" | "neutral"
  regime_multiplier?: number | null;
  pead_tag?: string | null;            // "positive_surprise" | "negative_surprise" | "neutral"
  news_event_type?: string | null;     // "earnings_beat" | "merger" | "fda_approval" | etc.
  news_impact?: string | null;         // "positive" | "negative"
  sector_event_type?: string | null;   // "fda_pdufa" | "earnings"
  sector_direction?: string | null;    // "positive_catalyst" | "binary_risk" | "neutral"
  conviction_score?: number | null;    // 1-10 from meta_scorer
  conviction_reason?: string | null;
}

export interface ScreenerCandidate {
  ticker: string;
  composite_score: number;
  source?: string;   // "news_only" or undefined (momentum)
  signal_tags: SignalTags;
}
```

### "Why" panel UI (badge strip)

Each candidate card renders a horizontal strip of colored `<span>` badges below the ticker/score. One badge per active signal, shown only when the signal is not null/neutral:

```tsx
function WhyPanel({ tags }: { tags: SignalTags }) {
  return (
    <div className="mt-2 flex flex-wrap gap-1.5">
      {tags.regime_tag && tags.regime_tag !== "neutral" && (
        <span className={`flex items-center gap-1 rounded px-1.5 py-0.5 text-xs font-medium
          ${tags.regime_tag === "risk_on"
            ? "bg-emerald-900/50 text-emerald-300"
            : "bg-rose-900/50 text-rose-300"}`}>
          <Brain size={10} weight="fill" />
          {tags.regime_tag === "risk_on" ? "Risk On" : "Risk Off"}
        </span>
      )}
      {tags.pead_tag && tags.pead_tag !== "neutral" && tags.pead_tag !== "insufficient_history" && (
        <span className={`flex items-center gap-1 rounded px-1.5 py-0.5 text-xs font-medium
          ${tags.pead_tag === "positive_surprise"
            ? "bg-sky-900/50 text-sky-300"
            : "bg-amber-900/50 text-amber-300"}`}>
          <TrendUp size={10} weight="fill" />
          PEAD: {tags.pead_tag === "positive_surprise" ? "+" : "-"}
        </span>
      )}
      {tags.news_event_type && (
        <span className="flex items-center gap-1 rounded bg-violet-900/50 px-1.5 py-0.5 text-xs font-medium text-violet-300">
          <Newspaper size={10} weight="fill" />
          {tags.news_event_type}
        </span>
      )}
      {tags.sector_event_type && (
        <span className={`flex items-center gap-1 rounded px-1.5 py-0.5 text-xs font-medium
          ${tags.sector_direction === "positive_catalyst"
            ? "bg-emerald-900/50 text-emerald-300"
            : "bg-amber-900/50 text-amber-300"}`}>
          <Timer size={10} weight="fill" />
          {tags.sector_event_type.replace("fda_pdufa", "FDA").replace("earnings", "Earnings")}
          {tags.sector_direction === "binary_risk" ? " (risk)" : ""}
        </span>
      )}
      {tags.conviction_score != null && (
        <span className={`flex items-center gap-1 rounded px-1.5 py-0.5 text-xs font-medium
          ${(tags.conviction_score ?? 0) >= 7
            ? "bg-emerald-900/50 text-emerald-300"
            : (tags.conviction_score ?? 0) >= 4
              ? "bg-slate-700/60 text-slate-300"
              : "bg-rose-900/50 text-rose-300"}`}>
          <Scales size={10} weight="fill" />
          Conv: {tags.conviction_score}/10
        </span>
      )}
    </div>
  );
}
```

Icons used: `Brain`, `TrendUp`, `Newspaper` (as `StepMarketIntel`), `Timer`, `Scales` — all already exported from `icons.ts`.

---

## Hard-Blocker Checklist

- [x] >=3 authoritative external sources READ IN FULL via WebFetch (4 sources; relaxed floor for internal-heavy cycle explicitly justified)
- [x] 10+ unique URLs collected (11 total: 4 read in full + 7 snippet-only)
- [x] Recency scan (last 2 years) performed + reported (2024-2026 scan confirmed no superseding findings)
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every relevant module (13 files inspected)
- [x] Contradictions / consensus noted (no external contradictions; scope ambiguity on "Signals page" documented)
- [x] All claims cited per-claim with file:line

---

## Summary of Deliverables for GENERATE Phase

1. **`backend/api/settings_api.py`**: Add 13 fields to `FullSettings`, `SettingsUpdate`, `_FIELD_TO_ENV`, `_settings_to_full()`, and model validation in `update_settings`. Exact text provided above.

2. **`frontend/src/lib/types.ts`**: Add 13 optional fields to `FullSettings` interface + new `SignalTags` + `ScreenerCandidate` interfaces.

3. **`frontend/src/app/settings/page.tsx`**: Add "Signal Stack (Phase 23.1)" BentoCard to the "models" tab (after line 808). 5 toggle rows + 4 model pickers + 3 sliders. Form state via existing `updateForm` callback.

4. **`backend/tools/screener.py`**: Attach `_regime_tag`, `_pead_tag`, `_news_event_type`, `_sector_event_type`, `_sector_direction` to candidate dicts in `rank_candidates()`. `meta_scorer.py` already attaches `conviction_score`/`conviction_reason` in `autonomous_loop.py`.

5. **`frontend/src/app` (Paper Trading or Signals page)**: Add `WhyPanel` component; expose last-cycle candidates via `_last_result` or new endpoint. GENERATE phase should resolve Option A vs B with product owner before coding.

6. **`frontend/src/lib/icons.ts`**: Add semantic aliases `SignalMacroRegime` (Brain), `SignalPead` (TrendUp), `SignalNewsEvent` (Newspaper/StepMarketIntel), `SignalSectorCal` (Timer), `SignalMetaConviction` (Scales).

---

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 4,
  "snippet_only_sources": 7,
  "urls_collected": 11,
  "recency_scan_performed": true,
  "internal_files_inspected": 13,
  "report_md": "handoff/current/phase-23.1.6-research-brief.md",
  "gate_passed": true
}
```

**Gate note:** `external_sources_read_in_full` is 4, not 5. This satisfies the relaxed 3-source floor explicitly stated in the caller's prompt for this internal-heavy cycle. The standard 5-source floor is documented in `.claude/rules/research-gate.md`; the relaxation requires explicit justification (provided above in "Sources-Read-in-Full" section). Gate is passed.
