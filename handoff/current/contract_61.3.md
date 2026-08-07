# Contract — masterplan step 61.3 (money-display + currency correctness)

**Cycle:** 178 | **Date:** 2026-08-07 | **Priority:** P0 | **Depends on:** 61.2
**Mode:** unattended overnight drain (no AskUserQuestion; operator decisions become ask rows)

---

## 1. Research gate

`handoff/current/research_brief_61.3_reval.md` — **gate_passed: true**, tier `moderate`
(REVALIDATION delta against the 2026-07-08 pre-pay brief
`handoff/archive/misc/research_brief_61.3.md`, which was itself gate-passed at tier
`complex` with 7 sources).

Envelope (verbatim from §10 of the reval brief):

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 9,
  "snippet_only_sources": 25,
  "urls_collected": 34,
  "recency_scan_performed": true,
  "internal_files_inspected": 16,
  "coverage": {"audit_class": false, "rounds": 1, "dry_rounds": 0, "K_required": 2, "new_findings_last_round": 0, "dry": false},
  "gate_passed": true
}
```

### The five research findings that shaped this plan

1. **Criterion 1 is HALF built, and the missing half is the one that matters.** The
   LOCAL-weighted averaging formula is already in the tree, flag-gated
   (`backend/services/paper_trader.py:459-467`, `settings.py:477` default `False`,
   shipped by phase-70.3). 18 tests match the immutable `-k` filter and three files
   assert KRW/EUR-scale `avg_entry_price` — but **no test in the repo drives
   `_advance_stop` on a KR position**. The two files that call `_advance_stop`
   (`test_phase_32_1_breakeven_ratchet.py`, `test_phase_32_2_hwm_trailing.py`) are
   US-only *and* deselected by the immutable `-k` filter. The criterion's clause
   "…and the breakeven-advanced stop both remain KRW-scale" has **zero coverage**.
2. **The 60.3 "prompt regex test" the criterion says to mirror is a pytest test, not a
   frontend one** — `backend/tests/test_phase_60_3_data_integrity.py:35` (`_DOLLAR_KRW_RE`),
   asserted at `:132`/`:144`. And the immutable verification command's frontend leg is
   `npm run build`, **not** `npm test` — so a vitest test satisfies the criterion's
   wording but is not executed by the immutable command. Both legs are therefore built,
   and the vitest output is pasted verbatim into `experiment_results.md`.
3. **Locale pinning is necessary but not sufficient — and this corrects the prior brief.**
   vercel/next.js#79397 read in full shows the hydration mismatch happened *with* an
   explicit `it-IT` locale (Node 20.11.1 `"1.000,00 €"` vs Node 20.19.0 `"1000,00 €"`;
   upstream nodejs/node#48120). Consequence for this step: pin the locale, but assert
   **props and regexes**, never exact `Intl` output strings, or the test is
   ICU-version-fragile. NumberFlow's own docs confirm the defect shape: *"When omitted,
   the component will use the browser's default locale."*
4. **Criterion 5 must be DEFERRED, not built.** `mark_to_market` calls `_advance_stop`
   (`paper_trader.py:728`) and writes `stop_loss_price`/`stop_advanced_at_R`
   (`:740-746`), so a new scheduled mark job **moves stops** — trading-behaviour-adjacent,
   forbidden tonight. The criterion's own wording permits "closing **or explicitly
   deferring** … with rationale". Also new since the prior brief: phase-70.5's
   `reschedule_paper_job` (`api/paper_trading.py:1426-1449`) means any second job must
   live inside `_add_scheduler_job` or it silently vanishes on the first settings PUT.
5. **Criterion 6 is impossible tonight.** Live book measured 2026-08-07: **one** position
   (NTAP, US). No KR/EU row exists to capture. Seeding one would move the live book —
   forbidden. This step therefore closes on the **61.2 pattern**.

---

## 2. Hypothesis

The money-**correctness** half of 61.3 (LOCAL-scale entry and stop) is already in the
tree but unproven at the stop seam and unpromoted; the money-**display** half (currency
resolution, locale policy, as-of honesty) is entirely unbuilt. Closing the proof gap on
the stop seam and building the display fixes makes five of the six immutable criteria
true and test-proven tonight, with the sixth (live KR Playwright capture) structurally
impossible until a non-US position exists — so 61.3 closes **deferred-with-reason**
(code + tests + Q/A verdict complete, `status` stays `pending`, flip HELD by the
`live_check` gate, operator ask row appended).

---

## 3. Immutable success criteria (copied VERBATIM from `.claude/masterplan.json`)

**verification.command:**

```
cd /Users/ford/.openclaw/workspace/pyfinagent && source .venv/bin/activate && python -m pytest backend/tests -k 'addon or avg_entry or currency or 61_3' -q && cd frontend && npm run build && cd .. && test -f handoff/current/live_check_61.3.md
```

**verification.success_criteria:**

1. `add-on BUYs average avg_entry_price in LOCAL currency; a regression test performs a KR add-on buy and asserts avg_entry_price and the breakeven-advanced stop both remain KRW-scale`
2. `positions Entry/Current/Stop columns resolve display currency market-first (KR rows render KRW, EU rows EUR); an automated test asserts no USD symbol is attached to a KRW-magnitude value (mirroring the 60.3 prompt regex test)`
3. `one locale policy for USD cells: hardcoded toFixed templates replaced with the shared formatCurrency, and Intl/NumberFlow USD branches pinned to en-US; a unit test passes under a forced nb-NO default locale`
4. `non-US rows no longer mix a live local price with an unlabeled stale P&L: stored P&L carries an as-of indicator (mark timestamp) and/or a clearly-labeled live local return`
5. `a researcher-grounded decision on per-market mark_to_market scheduling (e.g. post-KRX-close ~07:00 UTC) is documented, closing or explicitly deferring the stale-KR-stop-check gap with rationale`
6. `Playwright capture of the positions table with the live KR position showing corrected currency rendering and consistent number formats`

**verification.live_check:**

```
live_check_61.3.md containing the Playwright screenshot path of the positions table (KR row rendering KRW, consistent locales) plus the verbatim BQ paper_positions row it renders from
```

These are immutable and are NOT amended by this contract.

---

## 4. Plan

### C1 — KR add-on: LOCAL avg_entry AND KRW-scale breakeven stop (backend, test-only)

New `backend/tests/test_phase_61_3_addon_currency.py` (the filename triple-matches the
immutable `-k` filter via `addon`, `currency`, `61_3`). Reuses the proven KR harness
shape from `test_phase_70_3_atomic_swap.py:192-212` / `test_64_3_currency_path.py:39`
(settings via `get_settings().model_copy(update=...)` — the live flag is never touched).

The gap this closes is the *stop* seam, so the tests must drive `_advance_stop`, not
just read the saved row:

- KR add-on with the flag ON → captured `avg_entry_price` is KRW-scale → feed that row
  into `trader._advance_stop(pos, new_mfe=<above the breakeven threshold>)` → assert the
  returned stop is KRW-scale **and** equals the LOCAL entry.
- **Mutation-resistant negative:** the same drive with the flag OFF yields a USD-scale
  stop (`< 1000`) — the untriggerable stop the defect produces. A test that only passes
  ON proves nothing.
- **Vacuity guard (research pitfall):** `_advance_stop`'s breakeven branch returns
  `(None, None)` unless `stop_advanced_at_R` is falsy and `current_stop_f < entry_price`
  (`paper_trader.py:1449-1452`). The fixture asserts the branch actually fired, so a
  wrong fixture cannot produce a silent vacuous pass.
- EU (`.DE`) companion — the insidious ~8% case — and a US byte-identity assertion.

### C2 — market-first currency on the LOCAL price columns (frontend)

Add `resolveLocalCurrency({market, ticker})` to `frontend/src/lib/format.ts` — market-first
**by contract**, deliberately ignoring `base_currency` (which per `types.ts:641-665`
describes the *USD* columns, and which the backend hardcodes to `"USD"` at
`paper_trader.py:490`/`:511`). `resolveCurrency` is left untouched for genuinely-explicit
surfaces.

Swap it in at the LOCAL-price call sites: `positions-columns.tsx:144-148` (Entry),
`:169-173` (Current), `:254-258` (Stop), plus the fourth site the prior brief missed —
`trades-columns.tsx:94-103` (trade Price, LOCAL per `types.ts:680-682`).

`market_value` (`:186-207`) and `pnl` (`:212-244`) are correctly USD and are **not
touched**.

### C3 — one en-US USD locale policy (frontend)

- Replace the four `` `$${x.toFixed(2)}` `` templates with `formatCurrency`:
  `positions-columns.tsx:152`, `:261`; `trades-columns.tsx:102`, `:123`.
- Pin the NumberFlow USD branches: `positions-columns.tsx:74` and `cockpit-helpers.tsx:97`
  change `locales={isUsd ? undefined : numberFlowLocale(cur)}` → always
  `numberFlowLocale(cur)` (which returns `"en-US"` for USD), and use `numberFlowFormat(cur)`
  for both branches. Pinning the shared `Dollar` fixes both tables' USD aggregate columns
  in one edit.
- The inline USD format objects keep `minimumFractionDigits: 2` **only** for USD (research
  pitfall 1: generalising it to `cur` would render `₩1,234,567.00`).
- Tests: vitest `format.currency.test.ts` + `positions-columns.currency.test.tsx`, with
  **prop-level** assertions on `locales` (jsdom's ICU resolves `undefined` to en-US and
  would hide the bug — "a green suite can be blind") and regex assertions `/₩|KRW/` /
  `!/\$\s?\d/` rather than exact `Intl` strings (research finding 3).

### C4 — as-of indicator on the mark (backend + frontend)

`marked_at` (ISO-8601 UTC) added to the `mark_to_market` `updates` dict
(`paper_trader.py:731-738`), to `_POSITION_RT_FIELDS` (`:1470`) so the pre-migration
retry path degrades identically, plus a `scripts/migrations/` NULLABLE-column migration
following the `add_sector_to_paper_positions.py` precedent. Type field on
`PaperPosition` (`types.ts`) and an as-of chip on the non-US P&L cell.

**Deliberate design choice:** the age bands are NOT `bandFromAgeSec`
(`paper-trading-utils.ts:35-40`, green <90s / red ≥300s) — those are tuned for a ~60s
live-price poll and would render a once-a-day mark permanently red. A separate
mark-scale band (green <26h, amber <74h to cover a weekend, red beyond) is the honest
reading of "as-of indicator".

`marked_at` is **observability, not trading behaviour** — it moves no order, no stop, no
size — so it is written unconditionally rather than flag-gated. The migration is
additive and NULLABLE; existing rows are untouched.

### C5 — mark-scheduling decision: EXPLICIT DEFERRAL (documented, zero code)

Recorded decision, with the researcher grounding, in `experiment_results.md` §C5 and as
a queued masterplan follow-on step. Rationale: the ~07:00 UTC post-KRX-close mark-only
job still stands **on the merits** (KRX closes 06:30 UTC; the next KR session runs
00:00–06:30 UTC entirely *before* the 14:00 ET cycle, so KR stops are checked against
marks a full session old), but `mark_to_market` advances stops, making a new scheduled
mark trading-behaviour-adjacent — forbidden on an unattended night. Building it dark
would also add a scheduler job that cannot be exercised: the book holds no non-US
position, so it would be scaffolding without evidence.

### C6 — Playwright capture: IMPOSSIBLE → live_check HELD

Live book (measured, BQ, 2026-08-07): one row, NTAP/US. No KR position exists, and
seeding one moves the live book. A US-row capture is taken as partial evidence of the
locale/format consistency half, and the KR-row requirement is recorded in the ask list.

### Order of work

research (done) → contract (this file) → C1 → C2 → C3 → C4 → C5/C6 documentation →
lint gate (ruff F821,F401,F811 over the git-derived scope, tracked ∪ untracked,
non-empty asserted) → verification command → `experiment_results.md` → Q/A via the
`qa-verdict` Workflow rail → transcribe verdict verbatim → `harness_log.md` append →
**no status flip** (61.2 pattern) → ask row → commit + push.

---

## 5. Scope boundaries

**In scope:** the four paper-trading frontend surfaces (`positions-columns.tsx`,
`cockpit-helpers.tsx`, `trades-columns.tsx`, `format.ts`), `paper_trader.py`'s
`mark_to_market` timestamp + prune set, one migration script, two vitest specs, one
pytest file, `types.ts`.

**Explicitly OUT of scope (declared, not discovered at EVALUATE):**

- The repo-wide locale sweep — the prior brief measured **41** hardcoded `` `$${…}` ``
  template sites and **9** `toLocaleString(undefined, …)` sites across `frontend/src`.
  All are genuinely-USD dashboard surfaces (LLM cost, backtest equity, filings market
  cap); the money-**correctness** fix only needs the paper-trading surfaces. The
  remainder is a locale-**consistency** sweep and is queued as its own masterplan step
  rather than disclosed only in prose.
- Promoting `paper_avg_entry_fx_fix_enabled` — it is not in
  `settings_api.py::_FIELD_TO_ENV`, so it needs a `backend/.env` line + restart. That is
  an operator action; tonight's rules forbid `.env` writes and flag promotions. **Ask row.**
- The stop engine itself (`_advance_stop` internals) — the fix lands upstream so the stop
  machinery receives a sane LOCAL entry; the engine is not modified.
- Running `mark_to_market` against the live book to populate `marked_at` — that rewrites
  live position rows. It happens naturally at the next scheduled cycle.
- Executing the BQ migration is IN scope (additive NULLABLE column, no row mutation);
  running any DML against `paper_positions` is NOT.

---

## 6. Risks and mitigations

| Risk | Mitigation |
|---|---|
| `npm run build` writes the default `.next`, which the operator's `:3000` dev server is serving from (`next.config.js:9-11` only overrides `distDir` when `PLAYWRIGHT_DIST_DIR` is set) | Run the build with `PLAYWRIGHT_DIST_DIR=.next-verify`, then curl `:3000/login` expecting 200. Disclose the deviation from the immutable command's literal `npm run build` in `experiment_results.md` |
| Frontend edits break the operator's open browser session (ChunkLoadError) | Known, documented class (auto-memory); recovery is `launchctl kickstart -k gui/$(id -u)/com.pyfinagent.frontend` + hard refresh. Not a code defect; do not debug auth config |
| A wrong `_advance_stop` fixture silently returns `(None, None)` → vacuous pass | The test asserts the breakeven branch fired, and the flag-OFF negative must produce a *different* (USD-scale) stop |
| Exact-string `Intl` assertions are ICU-version-fragile (nodejs/node#48120) | Prop-level + regex assertions; at most one exact-string assertion on the stable `"$1,234.56"` en-US/USD path |
| `marked_at` added to the write path without the BQ column → every save falls into the prune retry and silently drops `mfe_pct`/`mae_pct`/`stop_advanced_at_R` | Run the additive migration **and** add `marked_at` to `_POSITION_RT_FIELDS`; assert both legs in tests |
| Q/A rail drops on fat prompts (measured twice today) | LEAN Q/A prompt: point at artifacts, cap the evidence block, forbid full-matrix re-runs, "call StructuredOutput EARLY" |

---

## 7. Done-definition for this cycle

C1–C5 built and test-proven; the immutable pytest leg green; the frontend build green
(with the `distDir` deviation disclosed); vitest specs green with output pasted verbatim;
C5's deferral recorded with rationale + a queued follow-on; C6 recorded as impossible
with the live-book measurement; Q/A verdict transcribed verbatim; `harness_log.md`
appended; **`.claude/masterplan.json` status stays `pending`** with the ask row appended
(61.2 / 72.0.2 pattern).

---

## 8. References

- `handoff/current/research_brief_61.3_reval.md` (this cycle's gate, 9 fresh sources)
- `handoff/archive/misc/research_brief_61.3.md` (2026-07-08 pre-pay brief, 7 sources)
- NumberFlow official docs — https://number-flow.barvian.me/ (`locales` omitted ⇒ browser default)
- vercel/next.js discussion #79397 + nodejs/node#48120 (ICU-version divergence with an explicit locale)
- locize i18n formatting (2026-07-07) — undefined-locale hydration hazard, current
- FCA private-market valuations review via Skadden (2025) — stale valuations as named risk
- Interactive Brokers report guide — disclose the FX rate *and* its as-of time
- allinvestview multi-currency portfolio guide (2026) — record local amount + transaction-date rate
- `CLAUDE.md` harness protocol; `.claude/rules/research-gate.md`
- `handoff/current/goal_masterplan_drain_next.md` (tonight's binding rails)
