# Goal Prompt — `goal-market-filter-in-gate-bar`

Set by operator 2026-06-01 (via screenshot + directive: "make the market filter
within the gate bar"). Grounded in a LIVE Playwright MCP click-through of the
running cockpit (`localhost:3000/paper-trading/positions`) performed the same day —
not a read of source alone. Run this under the full MAS harness loop (research →
contract → generate → fresh Q/A → log → flip), same as `goal-multimarket-ux` and
`goal-browser-mcp`.

## North star linkage (N* = Profit − Risk − Burn)

UX / operator-control work (active_goal.md priority 7 "best-in-class operator
control surface: full UI control + ONE consistent layout"). N* delta = **B↓
(speculative)**: removes a full horizontal row of chrome above the fold, tightening
the cockpit toward the §4.5 "ONE dense bar, not stacked rows" doctrine. No P or R
delta. A contract that cannot articulate even a speculative N* delta beyond
"cleaner" should mark this DEFERRED — but the density win is real and citable
(frontend-layout.md §4.5; Few 2006 single-screen density).

## Objective (one sentence)

Fold the market filter (`All · US · EU · KR`) **into** the gate/operator status bar
(`OpsStatusBar`) as a segment, so it no longer renders as a separate row between the
gate bar and the KPI hero.

## Verified current state (Playwright MCP, 2026-06-01, skip-auth Path A)

Confirmed live by driving the real browser (navigate → snapshot → click EU → screenshot
→ reset to All → restore auth gate). Evidence: the EU pill correctly flipped `VS SPY`→`VS
DAX`, `POSITIONS 7→0`, surfaced the "Filtered to EU…" note, and emptied the table /
allocation / sector cards. The filter is **fully functional**; this goal only relocates it.

Today the cockpit scrollable zone renders THREE stacked bands
(`frontend/src/app/paper-trading/layout.tsx:476-505`):

1. **Gate bar** — `<OpsStatusBar nextRunAt={status?.next_run} />` (`layout.tsx:478`).
   The dense status strip: segments `Gate | Kill | Cycle | Last | Next`
   (`OpsStatusBar.tsx:115-134`). On a 1440px viewport it already wraps `Next` to a 2nd
   line.
2. **Market filter row** — a separate `<div class="mb-4 flex … justify-between">`
   (`layout.tsx:483-490`) holding:
   - `<MarketFilter>` (`layout.tsx:484-488`) — WAI-ARIA radiogroup of pills
     (`MarketFilter.tsx`), left-aligned.
   - `<MarketSessionStrip>` (`layout.tsx:489`) — `US CLOSED · EU OPEN · KR CLOSED`
     open/closed indicator (`MarketSessionStrip.tsx`), right-aligned.
3. **KPI hero** — `<SummaryHero>` (`layout.tsx:491-498`) + the `activeMarket !== "ALL"`
   filtered note (`layout.tsx:499-504`).

State lives in the layout: `activeMarket`/`setActiveMarket` (`layout.tsx:137`),
`availableMarkets` (`layout.tsx:164-169`), and the auto-fallback-to-ALL effect
(`layout.tsx:173-177`). It is shared down via `PaperTradingDataContext`
(`layout.tsx:323-324`) and consumed by sub-routes (e.g. `positions/page.tsx:39,51`).

## CRITICAL constraint — `OpsStatusBar` is SHARED, do not break the homepage

`OpsStatusBar` is mounted in **two** places:
- Cockpit: `frontend/src/app/paper-trading/layout.tsx:478` (has market context).
- Homepage: `frontend/src/app/page.tsx:360` (NO market context).

Therefore the new Market segment MUST be **conditional** — render only when the
market-filter props (`markets` + `activeMarket` + `onMarketChange`) are supplied.
The homepage instance passes none, so it must keep rendering today's exact
5-segment bar (`Gate | Kill | Cycle | Last | Next`) with zero visual change. Any
diff that alters the homepage status bar is an automatic FAIL.

## Recommended design (the contract may refine, but must hit the acceptance criteria)

Add an optional **Market** segment to `OpsStatusBar`, placed as the **left-most**
segment (the "what am I looking at" scope precedes the operational status), followed
by a `<Divider/>` before `Gate`:

```tsx
interface Props {
  nextRunAt?: string | null;
  // New, all optional — present only on the cockpit:
  markets?: string[];                 // canonical-ordered codes, excl. "ALL"
  activeMarket?: string;              // "ALL" | "US" | ...
  onMarketChange?: (m: string) => void;
}
```

- When `markets && activeMarket && onMarketChange` are all present, render
  `<MarketSegment>` (a `SegmentLabel "Market"` + the existing `<MarketFilter>`
  radiogroup) as the first child of the `<section>`, then a `Divider`.
- **Fold the session open/closed signal into the pills** to retire the separate
  `MarketSessionStrip` row entirely: color each non-`All` pill's dot emerald when
  that market is currently open (`isMarketOpen` from `lib/format.ts:212`), slate when
  closed — i.e. `MarketFilter` gains an optional `sessionOpen?: Record<string,bool>`
  (or computes it internally via a mount-guarded `useState<Date|null>` like
  `MarketSessionStrip.tsx:24-29` to avoid the hydration mismatch). Keep the exchange
  name in each pill's `title` (`MarketFilter.tsx:80`). This removes BOTH the filter
  AND the session strip from `layout.tsx:483-490` — net −1 full row.
  - If folding sessions into pills proves too cramped, an acceptable fallback is to
    keep a compact `MarketSessionStrip` INSIDE the same bar segment (still −1 row vs
    today). The non-negotiable is: the filter is inside `OpsStatusBar`, the
    standalone `layout.tsx:483-490` row is gone.
- In `layout.tsx`, delete the `<div class="mb-4 …">` row (`483-490`) and instead pass
  `markets={availableMarkets} activeMarket={activeMarket} onMarketChange={setActiveMarket}`
  into the cockpit `<OpsStatusBar>` (`layout.tsx:478`). Leave the homepage call
  (`page.tsx:360`) untouched.
- The `activeMarket !== "ALL"` filtered note (`layout.tsx:499-504`) stays where it is
  (it explains the scope of the hero/table, not the filter control).

### Layout / a11y / palette guardrails (frontend.md + frontend-layout.md)

- **No emoji** anywhere (Phosphor icons / colored dots only) — strict project rule.
- **Dense-bar pattern (§4.5):** the bar stays one row that wraps gracefully via the
  existing `flex flex-wrap items-center gap-x-6 gap-y-3` (`OpsStatusBar.tsx:118`). The
  Market segment must not force a permanent 2nd line on a ≥1280px viewport beyond
  today's existing `Next`-wrap.
- **Palette:** navy/slate only (`bg-navy-800/60`, `text-slate-*`), never zinc.
- **JIT-safe classes:** dot colors must come from a static literal map
  (`MARKET_DOT_CLASS` already exists, `format.ts:100`) — never template-concatenated.
- **a11y preserved:** the radiogroup role, roving tabindex, and Arrow/Home/End
  keyboard nav from `MarketFilter.tsx` must survive the move intact. Keyboard focus
  order through the bar must remain sane (Market → Gate → … → actions).

## Immutable acceptance criteria (copy verbatim into contract.md; do NOT edit)

1. **Filter inside the bar.** On `/paper-trading/positions`, the `All · US · EU · KR`
   radiogroup renders INSIDE the `OpsStatusBar` `<section aria-label="Paper-trading
   operator status">` element — verified by DOM containment, not pixels. The
   standalone market-filter row at `layout.tsx:483-490` no longer exists.
2. **Row removed (density win).** The vertical distance from the bottom of the gate
   bar to the top of the NAV KPI tile is strictly less than today's (one fewer row).
3. **Function intact (live click-through).** Via the Playwright MCP under skip-auth
   Path A: clicking `EU` still flips `VS SPY`→`VS DAX`, updates `POSITIONS`, shows the
   "Filtered to EU…" note, and scopes the table/allocation/sector cards; clicking
   `All` restores the combined view. Reset to `All` and **restore the auth gate**
   (`launchctl unsetenv LIGHTHOUSE_SKIP_AUTH` + kickstart; verify 302) after — per
   `docs/runbooks/browser-mcp.md`.
4. **Homepage unchanged.** `frontend/src/app/page.tsx` status bar is visually and
   structurally identical to before (5 segments, no Market segment, no market props).
   Confirm by reading the rendered homepage bar — any change is a FAIL.
5. **Session signal retained.** Each market's open/closed state is still visible in
   the cockpit (folded into the pills OR a compact strip inside the same bar). No
   regression in the `isMarketOpen` heuristic; no hydration warning in the console.
6. **Quality gates green.** `cd frontend && npm run build` succeeds; existing
   frontend tests pass (incl. `layout-tablist.test.tsx`); zero emoji
   (grep); zero new console errors on the cockpit (Playwright `browser_console_messages`).

## Harness protocol (non-negotiable — per CLAUDE.md + goal_next_session.md)

- Spawn `researcher` FIRST (gate, ≥5 sources read in full, recency scan) — even
  though this is "just a UI move." No carve-out (`feedback_never_skip_researcher`).
  Likely tier: `simple`. Useful angles: status-bar/control-consolidation patterns
  (Stripe/Linear/Grafana 12), WAI-ARIA radiogroup-inside-toolbar a11y, dense-bar
  wrap behavior.
- Write `contract.md` (with N* delta + the 6 immutable criteria above) BEFORE any
  code (`feedback_contract_before_generate`).
- GENERATE → `experiment_results.md` with the verbatim `npm run build` output, the
  changed-file list, and the Playwright click-through transcript/screenshots.
- ONE fresh `qa` after generate (5-item harness-compliance audit FIRST), no self-eval,
  no verdict-shopping on CONDITIONAL.
- Append `harness_log.md` LAST, then flip masterplan status.
- This step touches a chart/color-coded UI ⇒ visual verification is MANDATORY
  (frontend.md rule 5): the Playwright click-through IS the visual check; cite it in
  `experiment_results.md`. Q/A on unit tests + grep alone is necessary but not
  sufficient.

## Files in scope

| File | Change |
|------|--------|
| `frontend/src/components/OpsStatusBar.tsx` | Add optional Market segment (conditional on props) + `MarketSegment` helper. |
| `frontend/src/components/paper-trading/MarketFilter.tsx` | Optional: accept session-open map / render open-closed dots. |
| `frontend/src/components/paper-trading/MarketSessionStrip.tsx` | Likely retired (signal folded into pills) OR re-homed inside the bar. |
| `frontend/src/app/paper-trading/layout.tsx` | Delete row 483-490; pass market props into the cockpit `OpsStatusBar` (478). |
| `frontend/src/app/page.tsx` | UNCHANGED (homepage `OpsStatusBar` keeps no market props). |

## References (read before PLAN)

- LIVE evidence captured this session: `cockpit-current-state.png`, `cockpit-eu-filter.png`
  (repo root, gitignored) + `docs/runbooks/browser-mcp.md` (the skip-auth verification path).
- `.claude/rules/frontend-layout.md` §4.5 (one dense bar, not stacked rows) + §5 (pill tabs).
- `.claude/rules/frontend.md` (dark palette, JIT-safe classes, visual-verification rule).
- `handoff/archive/goal-multimarket-ux/` (where this market filter was introduced).
- `handoff/archive/goal-browser-mcp/` + `docs/runbooks/browser-mcp.md` (browser MCP usage).
