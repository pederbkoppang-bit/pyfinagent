# Research Brief — phase-61.3 REVALIDATION (money-display + currency correctness)

**Tier:** moderate (REVALIDATION mode — delta against
`handoff/archive/misc/research_brief_61.3.md`, 2026-07-08)
**Date:** 2026-08-07
**Status:** COMPLETE — `gate_passed: true` (envelope in §10).

---

## 0. Method

- Read the 2026-07-08 brief first; this document records the DELTA only.
- Internal delta-audit = primary output (per-criterion remaining-work table
  with fresh `file:line` anchors verified against today's tree).
- External = fresh recency pass; the old brief's 7 sources are re-cited for
  continuity but do NOT count toward this session's >=5 read-in-full floor.

(sections below fill in as work proceeds)

---

## 1. Criterion 1 — LOCAL add-on averaging + KRW-scale breakeven stop

**Verdict: HALF BUILT. The averaging code + avg_entry assertions exist; the
breakeven-stop half of the criterion has ZERO coverage anywhere in the suite.**

### Production code (fresh anchors, 2026-08-07)

`backend/services/paper_trader.py`:
- `:456` `old_cost = existing["cost_basis"] or (old_qty * existing["avg_entry_price"])`
- `:459-467` the flagged fork:
  ```python
  if getattr(self.settings, "paper_avg_entry_fx_fix_enabled", False):
      new_avg = (old_qty * (existing["avg_entry_price"] or 0.0) + quantity * price) / new_qty
  else:
      new_avg = new_cost / new_qty
  ```
- `:474` `"avg_entry_price": round(new_avg, 4)` (add-on row)
- `:490` / `:511` `"base_currency": "USD"` hardcoded on BOTH the add-on and the
  new-position row (the old brief's :313/:334/:481 anchors have MOVED)
- `:498` first-lot `"avg_entry_price": price` (LOCAL — the invariant the fix restores)
- `:1394-1460` `_advance_stop`; `:1409` `entry_price = float(pos.get("avg_entry_price") or 0.0)`;
  breakeven returns `(entry_price, now_iso)` at `:1460`; trailing computes
  `peak_price = entry_price * (1 + mfe/100)` at `:1430` and
  `new_trail = peak_price * (1 - trail_pct/100)` at `:1435`
- `:1470` `_POSITION_RT_FIELDS = {"mfe_pct","mae_pct","stop_advanced_at_R","entry_strategy","company_name"}`
- Flag definition: `backend/config/settings.py:477`, default `False`
  (the old brief / 72.x notes said `:455` — it has MOVED)

So the fix formula is in the tree behind `paper_avg_entry_fx_fix_enabled` (phase-70.3),
default OFF, and the flag is NOT in `backend/api/settings_api.py::_FIELD_TO_ENV`
(no UI flip; needs a `backend/.env` line + restart).

### What the existing tests actually assert

`-k 'addon or avg_entry or currency or 61_3'` collects **18 tests / 8 files**
(measured 2026-08-07, `--collect-only`): `test_64_3_currency_path.py` (4),
`test_64_4_multi_market_e2e.py::test_64_4_multi_market_e2e_currency_invariants` (1),
`test_dod4_tier1_coverage_investment.py::test_paper_trader_execute_buy_average_up_recomputes_avg_entry` (1),
`test_phase_50_2_multicurrency.py` (10), `test_phase_60_3_data_integrity.py::test_60_3_currency_mismatch_blocks` (1),
`test_phase_70_3_atomic_swap.py::test_avg_entry_fx_fix_local_consistent_for_kr` (1).

Every one of them asserts on the **saved position row's `avg_entry_price`** only.
Representative bodies:

- `backend/tests/test_64_3_currency_path.py:73-85` — KR: ON `abs(avg_on-70000.0) < 500.0`;
  OFF `avg_off < 1000.0`. `:88-103` EU: ON `abs(avg_on-150.0) < 2.0`; OFF `avg_off > 155.0`
  and `avg_off > avg_on`. `:106-114` US byte-identity `< 1e-6`. `:117-126` fx-unavailable
  returns None + nothing saved.
- `backend/tests/test_phase_70_3_atomic_swap.py:214-231` — same KR shape
  (`abs(avg-70000.0) < 500.0` ON, `avg_legacy < 1000.0` OFF); `:234-241`
  `test_flags_present_and_default_off` asserts the three flags exist and default `False`.
- `backend/tests/test_64_4_multi_market_e2e.py:129-151` — `_add_on_avg_entry(...)` returns
  `cap["row"]["avg_entry_price"]`; asserts KR ~70000 (<500 tol) and EU ~150 (<2 tol).

**The stop half is missing.** `grep -rn "_advance_stop" backend/tests/` returns hits in
exactly two files — `test_phase_32_1_breakeven_ratchet.py` (5 calls) and
`test_phase_32_2_hwm_trailing.py` (6 calls) — and **neither file matches the immutable
`-k` filter**, so they are deselected by the verification command; more importantly
neither drives a KR/KRW-scale position nor touches
`paper_avg_entry_fx_fix_enabled`. **No test in the repo asserts that the
breakeven-advanced stop is KRW-scale after a KR add-on.** That is the criterion-1
remaining work.

### Remaining work for criterion 1

Write `backend/tests/test_phase_61_3_addon_currency.py` (name matches `addon`,
`currency`, and `61_3` — triple-matches the immutable `-k`). It must, in ONE test,
(a) run the KR add-on through `execute_buy` with the flag ON, (b) take the captured
`avg_entry_price` into a position dict, (c) call `trader._advance_stop(pos,
new_mfe=<above paper_default_stop_loss_pct>)` and assert the returned stop is
KRW-scale (e.g. `stop > 1000` and `abs(stop - avg) < 500`), and (d) assert the
mutation-resistant negative: with the flag OFF the same drive yields a stop `< 1000`
(the untriggerable USD-scale stop). Reuse the `_kr_trader` harness shape at
`test_phase_70_3_atomic_swap.py:192-212`. Note `_advance_stop`'s breakeven branch
needs `stop_advanced_at_R` falsy and `current_stop_f < entry_price` (`:1449-1452`) or
it returns `(None, None)` — a silent vacuous pass if the fixture is wrong.

---

## 2. Criterion 2 — market-first currency on Entry / Current / Stop

**Verdict: UNCHANGED since 2026-07-08. The old brief's `resolveLocalCurrency` design
is still the right seam, and no helper has landed to obviate it.**

`frontend/src/lib/format.ts` (255 lines today) — what CHANGED since the old brief:
the file GREW (Nordics + Canada added to `MARKET_CURRENCY:23-33`, `CURRENCY_LOCALE:86-95`,
`MARKET_HOURS:224-233`; new `marketForSymbol` suffixes at `:118-123`; new
`positionMarketValueUsd:140-157`; new `isMarketOpen:237-254`). What did NOT change:

- `resolveCurrency:161-171` is still **explicit-first**:
  `const explicit = (opts.baseCurrency ?? opts.currency ?? "").trim(); if (explicit) return explicit.toUpperCase();`
- `localeForCurrency:173-175` (default `"en-US"`), `formatCurrency:180-196`,
  `formatUsd:199-201`, `numberFlowFormat:205-211`, `numberFlowLocale:215-217` all unchanged.
- **`grep resolveLocalCurrency` across `frontend/src` returns ZERO hits** — the helper
  the old brief recommended was never written.

Call sites to change (`frontend/src/components/paper-trading/positions-columns.tsx`,
all three pass `baseCurrency: row.original.base_currency`, which the backend hardcodes
to `"USD"` at `paper_trader.py:490`/`:511`):

| Cell | resolveCurrency call | render |
|---|---|---|
| Entry | `:144-148` | `:152-153` |
| Current | `:169-173` (comment at `:168` says "LOCAL currency (phase-50.2)") | `CurrentPriceCell`, `:44-83` |
| Stop Loss | `:254-258` | `:261` |

A **fourth** site exists that the old brief did not list:
`frontend/src/components/paper-trading/trades-columns.tsx:94-103` — the trade **Price**
column (LOCAL per the `types.ts:680-682` comment) also runs `resolveCurrency` with the
row's explicit currency and then `` `$${row.original.price.toFixed(2)}` ``. Same defect
class. Trades rows usually have no `market`/`base_currency` at all (paper_trades has no
market column), so this one degrades to ticker-suffix resolution and is less broken —
but it should be converted in the same pass for consistency.

Do NOT touch `market_value` (`:186-207`) or `pnl` (`:212-244`) — those are correctly USD
and correctly avoid client-side FX for non-US.

---

## 3. Criterion 3 — single en-US USD locale policy

**Verdict: UNCHANGED. All four offenders are still live, and the "60.3 prompt regex
test" the criterion mirrors is a PYTHON test, not a frontend one — this materially
changes where the new test should live.**

### Offender list (paper-trading surfaces, verified 2026-08-07)

| Anchor | Offence |
|---|---|
| `positions-columns.tsx:152` | `` `$${row.original.avg_entry_price.toFixed(2)}` `` |
| `positions-columns.tsx:261` | `` `$${sl.toFixed(2)}` `` |
| `positions-columns.tsx:74` | `locales={isUsd ? undefined : numberFlowLocale(cur)}` (CurrentPriceCell) |
| `positions-columns.tsx:64-73` | inline USD format object forcing `minimumFractionDigits: 2` |
| `cockpit-helpers.tsx:97` | `locales={isUsd ? undefined : numberFlowLocale(cur)}` (shared `Dollar`) |
| `cockpit-helpers.tsx:87-95` | same inline USD format object |
| `trades-columns.tsx:102` | `` `$${row.original.price.toFixed(2)}` `` |
| `trades-columns.tsx:123` | `` `$${row.original.transaction_cost.toFixed(2)}` `` |

`Dollar` is imported by `positions-columns.tsx:16` (used at `:207` for Market Value)
and `trades-columns.tsx:9` (used at `:113` for Total Value) — so pinning `Dollar`'s
locale fixes both tables' USD aggregate columns in one edit.

### The "60.3 prompt regex test" — located

`backend/tests/test_phase_60_3_data_integrity.py:35`:
```python
_DOLLAR_KRW_RE = re.compile(r"\$\s?\d{7,}|\$\s?\d{1,3}(,\d{3}){2,}|\$\s?\d{5,}\.?\d*B")
```
asserted at `:132` and `:144` (`assert not _DOLLAR_KRW_RE.search(lines), lines`). It is a
**pytest** test over generated LLM-prompt text — that is what "mirroring the 60.3 prompt
regex test" refers to. Its `test_60_3_currency_mismatch_blocks` is one of the 18 tests
the immutable `-k` filter already selects.

### The test-runner trap (IMPORTANT for contract design)

The immutable verification command's frontend leg is `npm run build` — **not** `npm test`.
So a vitest test satisfies the criterion's wording but is NOT executed by the immutable
command. Frontend test infrastructure that DOES exist:
`frontend/vitest.config.ts` (jsdom, globals, setupFiles `./vitest.setup.ts`, include
`src/**/*.{test,spec}.{ts,tsx}`), run via `npm test` → `frontend/scripts/run-test.mjs`
(translates `--filter=X` to a vitest positional). 33 test files exist under `src/`;
**`grep -l 'resolveCurrency|formatCurrency|KRW|nb-NO'` across them returns ZERO** — no
frontend test touches currency today.

Recommended split so the criterion is BOTH honest and covered by the immutable command:
1. **vitest** `frontend/src/lib/format.currency.test.ts` +
   `frontend/src/components/paper-trading/positions-columns.currency.test.tsx` — the real
   behavioural assertions (render a KR row carrying `base_currency:"USD"`, assert the
   text matches `/₩|KRW/` and NOT `/\$\s?\d/`; force a nb-NO default locale and assert
   `formatCurrency(1234.56,"USD") === "$1,234.56"`).
2. **pytest** in the same `test_phase_61_3_addon_currency.py` — a guard that RUNS the
   vitest suite is over-engineering; instead assert the backend-side invariant that the
   61.3 fix depends on (KRW-scale avg_entry + KRW-scale stop, §1) so the immutable
   pytest leg is non-vacuous. Paste the verbatim vitest output into
   `experiment_results.md` as the criterion-3 evidence.

**Locale-forcing note for the vitest test:** `formatCurrency` calls
`new Intl.NumberFormat(localeForCurrency(cur), ...)` — it ALWAYS passes an explicit
locale, so it is already deterministic. The nb-NO test therefore only proves a
regression guard for `formatCurrency`; the REAL nondeterminism is the
`locales={undefined}` NumberFlow props, which must be asserted at the **prop level**
(assert the rendered `locales` is `"en-US"`, or that `numberFlowLocale("USD")` is
passed) because jsdom's ICU will happily resolve `undefined` to en-US and hide the bug.
This is the "a green suite can be blind" failure mode — assert the prop, not the output.

---

## 4. Criterion 4 — marked_at / as-of indicator

**Verdict: NOT BUILT. Zero timestamp is persisted per position on the mark path. The
old brief's design (a) is still correct and the plumbing is friendlier than it looked.**

Fresh anchors:
- `backend/services/paper_trader.py:693-778` `mark_to_market`. The `updates` dict at
  `:731-738` carries exactly `current_price, market_value, unrealized_pnl,
  unrealized_pnl_pct, mfe_pct, mae_pct` (+ conditionally `stop_loss_price` at `:740`
  and `stop_advanced_at_R` at `:746`). **No timestamp.** `pos.update(updates)` at `:748`,
  then `self._safe_save_position(pos)` at `:749`.
- Portfolio-level `updated_at` is written at `:772` (`upsert_paper_portfolio`) — the
  only mark-time evidence that exists, and it is portfolio-wide.
- `_safe_save_position:1483-1492` — on a schema error it prunes `_POSITION_RT_FIELDS`
  (`:1470`) and retries. **A new `marked_at` MUST be added to that prune set**, or the
  retry fails identically on a pre-migration table.
- `save_paper_position` (`backend/db/bigquery_client.py:626-665`) builds the MERGE
  column list from the row's keys (`:641-646`) — an unmigrated column makes the MERGE
  itself invalid, which is why the prune set matters.

**Live schema measured 2026-08-07** (`financial_reports.paper_positions`, 22 columns):
`position_id, ticker, quantity, avg_entry_price, cost_basis, current_price,
market_value, unrealized_pnl, unrealized_pnl_pct, entry_date, last_analysis_date,
recommendation, risk_judge_position_pct, stop_loss_price, market, base_currency,
mfe_pct, mae_pct, sector, stop_advanced_at_R, entry_strategy, company_name`.
**No `marked_at`.** Adding a NULLABLE STRING column is a supported BQ ALTER (only
REQUIRED columns cannot be added to an existing table); precedent migration:
`scripts/migrations/add_sector_to_paper_positions.py`.

**Read path is already a passthrough** — `get_paper_positions` (`bigquery_client.py:607-613`)
is `SELECT *`; `PaperTrader.get_positions` (`paper_trader.py:169-170`) returns it
verbatim; the `/api/paper-trading/portfolio` handler (`backend/api/paper_trading.py:181-230`)
has **no `response_model`** (the only one in the file is `StartResponse` at `:86`), so a
new column reaches the frontend with zero API work. Only
`frontend/src/lib/types.ts:641-665` (`PaperPosition`) needs the optional field.

**Frontend as-of idiom to reuse:** `bandFromAgeSec`
(`frontend/src/lib/paper-trading-utils.ts:35-40`; green <90s, amber <300s, red ≥300s,
`"unknown"` for null), consumed via `LiveBadge` at `positions-columns.tsx:15/:60/:167`.
Those thresholds are tuned for a ~60s live-price poll and are WRONG for a once-a-day
mark — a mark 6h old would render red permanently. Add a SEPARATE band function
(e.g. `bandFromMarkAgeSec`: green <26h, amber <74h to cover a weekend, red beyond)
rather than reusing `bandFromAgeSec`; that is the honest reading of "as-of indicator".

---

## 5. Criterion 5 — per-market mark scheduling decision

**Verdict: structurally UNCHANGED; the recommendation still holds, but tonight it must
be DEFERRED-with-rationale, not built.**

Fresh anchors:
- `backend/api/paper_trading.py:1390-1397` `init_scheduler` — gated on
  `settings.paper_trading_enabled`, calls `_add_scheduler_job(settings)`.
- `:1400-1423` `_add_scheduler_job` — ONE cron job: `hour=settings.paper_trading_hour,
  minute=0, day_of_week="mon-fri", timezone=ZoneInfo("America/New_York"),
  id=_scheduler_job_id, replace_existing=True, misfire_grace_time=3600, coalesce=True`.
- `:1426-1449` `reschedule_paper_job` (phase-70.5, NEW since the old brief) — a settings
  PUT that changes the hour re-adds the cron without a restart, guarded so it never
  CREATES the job when paper trading is off. **This is a new constraint on the design:
  any second job must either be added inside `_add_scheduler_job` (so the 70.5 reschedule
  path re-adds it too) or given its own id and its own re-add path; putting it elsewhere
  makes it silently disappear on the first settings PUT.**
- `backend/config/settings.py:390` `paper_trading_hour: int = Field(10, ...)` (ET).
  Researcher sandbox cannot read `backend/.env`; Main must confirm the live value.

Exchange arithmetic is unchanged (KRX close 15:30 KST = 06:30 UTC, no DST; XETRA 17:30
CET/CEST = 16:30 UTC winter / 15:30 UTC summer), so the "~07:00 UTC post-KRX-close
mark-only job" recommendation from the old brief still stands on the merits.

**But it must NOT be built tonight.** `mark_to_market` calls `self._advance_stop(pos,
new_mfe)` at `paper_trader.py:728` and writes `stop_loss_price` /
`stop_advanced_at_R` at `:740-746`. A new scheduled mark therefore **moves stops** —
it is trading-behaviour-adjacent, not a read-only refresh. Under tonight's unattended
rules (no flag promotions, nothing that moves the live book) the only two admissible
options are:

- **(A) DEFER** — record the decision + rationale in the contract/experiment_results
  and add a masterplan follow-on step. Zero code. **RECOMMENDED.**
- (B) Build it dark behind a new default-False flag (e.g. `paper_kr_close_mark_enabled`)
  wired inside `_add_scheduler_job`, gated on `is_trading_day(<KR-local date>, "KR")`
  (`backend/backtest/markets.py`, the phase-50.4 `is_session` fix), copying the
  `misfire_grace_time=3600 / coalesce=True` rationale from `:1413-1422`.

Recommend **(A)**. The criterion's own wording — "closing **or explicitly deferring** the
stale-KR-stop-check gap with rationale" — is satisfied by a documented deferral, and (B)
adds a scheduler job whose only observable effect tonight would be on a book with no
non-US position in it. Building a dark scheduler job that can never be exercised is
scaffolding without evidence.

---

## 6. Execution hazards for tonight (read before GENERATE)

1. **`npm run build` shares `.next` with the operator's :3000 dev server.**
   `frontend/next.config.js:9-11` only overrides `distDir` when `PLAYWRIGHT_DIST_DIR` is
   set (the phase-64.1 :3100 isolation); a plain `next build` writes the default `.next`,
   and `frontend/package.json:6` shows `dev` runs from that same dir (`predev` even
   `rm -rf .next`). The phase-61.3 P0 triage on 2026-07-26 deliberately skipped this leg
   for exactly this reason. Mitigation: run the build with `PLAYWRIGHT_DIST_DIR=.next-verify`
   (or accept the operator-session breakage and follow the
   `launchctl kickstart -k gui/$(id -u)/com.pyfinagent.frontend` + hard-refresh recovery).
   Either way, DISCLOSE which was done — the immutable command names `npm run build`.
2. **The immutable command's third leg is `test -f handoff/current/live_check_61.3.md`.**
   It fails until that file exists, so the command cannot go green before the live_check
   is written. With no KR/EU position in the book (live book = 1 US row, NTAP),
   criterion 6's "live KR position" capture is **impossible**; the step closes on the
   61.2 pattern (built + Q/A + flip HELD on live_check + an operator ask row).
3. **`paper_avg_entry_fx_fix_enabled` is not in `backend/api/settings_api.py::_FIELD_TO_ENV`**,
   so it cannot be flipped from the Settings UI — it needs a `backend/.env` line +
   restart. That is an operator action, explicitly out of scope tonight.
4. **The flag stays OFF tonight**, so the new criterion-1 test must drive the ON branch
   via `get_settings().model_copy(update={...})` (the established harness at
   `test_64_3_currency_path.py:39`), never by touching the live setting.

---

## 7. External research (FRESH — the 2026-07-08 brief's 7 sources are NOT counted here)

### Read in full (counts toward the gate)

| URL | Accessed | Kind | Fetched how | Key quote or finding |
|---|---|---|---|---|
| https://number-flow.barvian.me/ | 2026-08-07 | official docs (NumberFlow) | WebFetch full | `locales` type is `Intl.LocalesArgument`, "The locale(s) for the number"; **"When omitted, the component will use the browser's default locale."** `format` is `Intl.NumberFormatOptions`, applied via native `Intl.NumberFormat`. Warning: **"Non-Latin digits and RTL locales aren't currently supported."** |
| https://www.locize.com/blog/i18n-formatting | 2026-08-07 | industry docs (locize/i18next, dated **2026-07-07**) | WebFetch full | **"No locale argument. `toLocaleString()` or `new Intl.NumberFormat()` without a locale uses the runtime's locale, not your user's language, a classic source of server/client hydration mismatches."** Also: "The currency comes from the transaction, the formatting comes from the user's locale." |
| https://github.com/vercel/next.js/discussions/79397 | 2026-08-07 | community bug report (2025) | WebFetch full | **The mismatch occurred WITH an explicit locale (`it-IT`)**: Node 20.11.1 renders `"1.000,00 €"`, Node 22.14.0 / 20.19.0 render `"1000,00 €"` (grouping separator dropped); browser V8/Nitro render correctly. Upstream: nodejs/node#48120. |
| https://www.skadden.com/insights/publications/2025/03/fca-findings-on-private-market-valuations | 2026-08-07 | authoritative legal analysis of a regulator review (Mar 2025) | WebFetch full | **"infrequent valuation cycles can lead to stale valuations, meaning that an asset's valuation no longer accurately reflects its current market value"**; detrimental "particularly if investor fees, redemption prices, and subscription prices are based on stale valuations"; FCA "encouraged firms to consider establishing a formal process for ad hoc valuations" with defined trigger thresholds. |
| https://next-intl.dev/docs/usage/numbers | 2026-08-07 | official docs (next-intl) | WebFetch full | Locale comes from context via `useFormatter()`, never from the runtime; `format.number(499.9, {style:'currency', currency:'USD'})`; **"To reuse number formats for multiple components, you can configure global formats"** — the centralised-formatter pattern. |
| https://www.allinvestview.com/articles/multi-currency-portfolio-guide/ | 2026-08-07 | industry practitioner (2026) | WebFetch full | **"For each transaction (buy, sell, dividend), record both the local currency amount and the exchange rate on that date."** "Your cost basis must be recorded in your base currency using the exchange rate on the purchase date." Performance split into "local asset returns and currency returns". |
| https://www.ibkrguides.com/reportingreference/reportguide/basecurrencyexchangerate_default.htm | 2026-08-07 | official docs (Interactive Brokers) | WebFetch full | Multi-currency statements carry a dedicated Base Currency Exchange Rate section: **"The currency exchange rate as of the report date. For closing rates, we use the midpoint of the bid/ask as reported by Reuters just prior to 4:00 PM ET."** — the rate AND its as-of time are disclosed, not implied. |
| https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Intl/NumberFormat | 2026-08-07 | official docs (MDN) | WebFetch full | LOW YIELD — the landing page defers option semantics to the constructor page (already cited in the 2026-07-08 brief); `resolvedOptions()` "Returns a new object with properties reflecting the locale and collation options computed during initialization". Recorded honestly rather than dropped. |
| https://www.abrigo.com/resources/fair-value-disclosure-review/ | 2026-08-07 | industry (Q1 2026 quarterly review) | WebFetch full | LOW YIELD for this step — covers exit-price methodology, **not** measurement-date/staleness disclosure. Recorded as a negative result: the 2026 fair-value-disclosure literature does not add an as-of-labelling rule beyond the FCA finding above. |

### Identified but snippet-only (does NOT count toward the gate)

| URL | Kind | Why not fetched in full |
|---|---|---|
| https://github.com/vercel/next.js/discussions/19409 | community | already cited snippet-only in the 2026-07-08 brief; #79397 read in full supersedes it |
| https://github.com/nuxt/nuxt/discussions/17629 | community | same failure class, non-Next framework |
| https://github.com/amannn/next-intl/issues/528 | community | plural (not currency) hydration |
| https://simplelocalize.io/blog/posts/handling-dates-times-numbers-localization/ | industry blog | duplicate of locize coverage |
| https://react-spectrum.adobe.com/react-aria/useNumberFormatter.html | official docs (Adobe) | alternative library; not the pinned dependency |
| https://allshadcn.com/tools/number-flow/ | aggregator | mirrors the official NumberFlow docs |
| https://www.sec.gov/.../valuation-portfolio-securities-other-assets-held-registered-investment-companies-select | regulator bibliography | link list, not substantive text |
| https://dart.deloitte.com/USDART/... (AICPA valuation guide spotlight) | official (Deloitte DART) | login-walled (same wall as the 2026-07-08 attempt) |
| https://www.sbai.org/static/.../Private-Market-Valuations-...pdf | industry standards body | binary PDF; FCA/Skadden covers the same staleness point |
| https://www.interactivebrokers.co.uk/en/general/education/pdfnotes/PDF-Statements.php | official docs (IBKR) | webinar notes duplicate of the report guide read in full |
| https://help.portfolio-performance.info/en/reference/file/currency/ | community docs | consumer tool; lower tier |
| https://en.wikipedia.org/wiki/Multi-currency_pricing | tertiary | encyclopedic |
| ~13 further hits (react-intl-number-format npm + GitHub + netlify, egghead, dev.giuseppeciullo.it, jsguides.dev, Medium Intl tip, thefundcfo substack, Deloitte fair-value roadmap, IBKR clientportal base currency, Oracle base currency, Skadden index, MDN constructor page) | mixed | lower marginal value; counted in `urls_collected` |

### Search queries run (3-variant discipline)

1. **Year-less canonical:** `NumberFlow react locales format prop currency Intl`;
   `multi-currency brokerage statement local currency cost basis versus base currency display`
2. **Current-year frontier (2026):** `Intl.NumberFormat currency locale hydration mismatch Next.js 2026`
3. **Last-2-year window (2025):** `stale price as-of timestamp disclosure portfolio valuation guidance 2025`

### Recency scan (last 2 years, 2024–2026)

Performed. Four findings; **one materially changes a design assumption**.

1. **NEW / CORRECTS THE OLD BRIEF — locale pinning is necessary but NOT sufficient.**
   The 2026-07-08 brief treated vercel/next.js #79397 as snippet evidence that
   `locales: undefined` causes hydration errors. Read in full, the report shows the
   mismatch happened **with an explicit `it-IT` locale**: Node 20.11.1 emits
   `"1.000,00 €"` while Node 20.19.0/22.14.0 emit `"1000,00 €"` (upstream nodejs/node#48120).
   So there are TWO independent nondeterminism sources — (a) undefined locale → runtime
   default, and (b) ICU/Node-version divergence for the SAME explicit locale. Fixing (a)
   is still the right move, but **a unit test that asserts an exact formatted string is
   ICU-version-fragile**; prefer regex/absence assertions (`/₩|KRW/`, `!/\$\s?\d/`) and
   prop-level assertions over exact-string equality. Practical mitigation for the
   en-US/USD path specifically: `"$1,234.56"` is stable across the ICU builds in evidence,
   so a single exact-string assertion there is acceptable if paired with regex assertions
   elsewhere.
2. **2026-07-07 (locize) restates the undefined-locale rule verbatim** — the hazard is
   current, not historical, one month before this step runs.
3. **NumberFlow's own docs (current) confirm the fix shape** — `locales` omitted ⇒
   browser default. This is the authoritative confirmation the old brief inferred from
   MDN; it is now sourced from the pinned dependency's own documentation
   (`@number-flow/react ^0.6.0`, `frontend/package.json:19`).
4. **No 2024–2026 source supersedes the IAS-21-style local-vs-base separation.**
   The FCA/Skadden 2025 review and the 2026 practitioner guidance both reinforce it
   (record the local amount AND the rate on that date; disclose the rate's as-of time).
   The Abrigo Q1-2026 fair-value review is a **negative result**: it adds no
   measurement-date-labelling rule. New supporting angle for criterion 4: IBKR discloses
   both the rate and its as-of time ("as of the report date … just prior to 4:00 PM ET"),
   which argues for persisting the FX rate used at mark time next to `marked_at`, not
   just the timestamp.

### Key findings applied to 61.3

1. **The `locales={isUsd ? undefined : ...}` pattern is defective by the dependency's own
   documentation** — "When omitted, the component will use the browser's default locale"
   (number-flow docs). On the operator's nb-NO browser this renders a NOK/nb-NO-shaped
   USD string beside `positions-columns.tsx:152`'s `` `$${...toFixed(2)}` ``. Fix:
   always pass `numberFlowLocale(cur)` (which returns `"en-US"` for USD via
   `format.ts:173-175`) and always use `numberFlowFormat(cur)`.
2. **Centralise, don't sprinkle** — next-intl's "configure global formats" pattern is what
   `format.ts` already is; the fix is to route the remaining hardcoded templates through
   `formatCurrency`/`formatUsd` rather than to add new inline format objects.
3. **Stale marks are a named regulatory risk, not a cosmetic one** (FCA via Skadden 2025)
   — "an asset's valuation no longer accurately reflects its current market value". This
   is the external grounding for criterion 4's as-of indicator and for criterion 5's
   deferral rationale (the gap is real; the remediation is scheduled, not skipped).
4. **Persist the rate with the timestamp** (IBKR + allinvestview 2026) — a `marked_at`
   alone tells you WHEN; the FX rate used tells you WHAT the USD number means. Optional
   companion column `marked_fx_rate`; note this doubles the migration surface, so it is a
   recommendation, not a requirement.
5. **Two-ledger discipline is unchanged** — "record both the local currency amount and the
   exchange rate on that date" (allinvestview 2026) is the same rule the 2026-07-08 brief
   drew from IAS 21; `avg_entry_price` LOCAL / `cost_basis` USD stays correct, which is
   exactly what `paper_avg_entry_fx_fix_enabled=True` restores.

### Consensus vs debate

**Consensus:** never let a formatter fall back to the runtime locale; centralise
formatter config; keep local and base ledgers separate with the transaction-date rate;
label the as-of time of any value that is not live.

**Debate:** whether pinning the locale is *sufficient* for SSR determinism — the locize/
MDN line says pin it; the vercel #79397 evidence says pinning still leaves ICU-version
divergence, and the reporter's accepted workaround was to change the *locale* rather than
the code. For this project the practical resolution is that only ONE Node runs both
server and client render paths (local dev, one machine), so (b) is a test-brittleness
concern rather than a production hydration risk.

### Pitfalls carried forward (still valid) + new

1. Do not force `minimumFractionDigits: 2` on non-USD — `formatCurrency` avoids it
   (`format.ts:178-179` comment); the inline USD objects at `positions-columns.tsx:67-71`
   and `cockpit-helpers.tsx:90-94` reintroduce it and must not be generalised to `cur`.
2. NumberFlow cannot render non-Latin digits / RTL (official docs) — `ko-KR` and `en-IE`
   are Latin-digit and safe; do not widen `CURRENCY_LOCALE` to `ar`/`fa`/`bn`.
3. **NEW:** exact-string assertions on `Intl` output are ICU-version-fragile (#79397).
4. **NEW:** a `marked_at` column must be added to the `_safe_save_position` prune set
   (`paper_trader.py:1470`) or the pre-migration retry path fails identically.
5. The `-k 'addon or avg_entry or currency or 61_3'` filter is immutable — name the new
   pytest file so it matches (`test_phase_61_3_addon_currency.py` matches three ways).

---

## 8. Per-criterion REMAINING-WORK table (the deliverable)

| # | Criterion (immutable, abridged) | Already in tree | REMAINING work | Fresh anchors |
|---|---|---|---|---|
| 1 | LOCAL add-on averaging + regression test asserting avg_entry AND breakeven-advanced stop both KRW-scale | Formula built + flag-gated (phase-70.3); 18 tests match the `-k` filter, 3 files assert KRW/EUR-scale `avg_entry_price` | **The stop half. No test anywhere drives `_advance_stop` on a KR position under the flag.** Write `test_phase_61_3_addon_currency.py`: KR add-on → capture avg → `_advance_stop(pos, new_mfe>threshold)` → assert KRW-scale stop; mutation-negative with flag OFF | `paper_trader.py:459-467, 474, 498, 1394-1460`; `settings.py:477`; `test_64_3_currency_path.py:73-126`; `test_phase_70_3_atomic_swap.py:214-241`; `test_64_4_multi_market_e2e.py:129-151` |
| 2 | Market-first currency on Entry/Current/Stop | Nothing | Add `resolveLocalCurrency()` to `format.ts` (market-first, ignores `base_currency`); swap it in at 3 sites (+1 bonus site in trades) | `format.ts:161-171` (explicit-first, unchanged); `positions-columns.tsx:144-148, 169-173, 254-258`; `trades-columns.tsx:94-103`; `paper_trader.py:490, 511` (the `"USD"` hardcode) |
| 3 | Single en-US USD locale policy + test under forced nb-NO | Nothing; ZERO frontend tests touch currency | Replace 4 `` `$${x.toFixed(2)}` `` templates; drop `locales={isUsd ? undefined : ...}` at 2 sites; add vitest specs; keep the immutable pytest leg non-vacuous via criterion 1 | `positions-columns.tsx:64-74, 152, 261`; `cockpit-helpers.tsx:87-97`; `trades-columns.tsx:102, 123`; mirror-target `test_phase_60_3_data_integrity.py:35,132,144`; runner `frontend/vitest.config.ts` + `package.json:12` + `scripts/run-test.mjs` |
| 4 | Non-US P&L carries an as-of indicator | Nothing | Add `marked_at` to the `updates` dict + prune set + a BQ NULLABLE-column migration + `PaperPosition` type + an as-of chip with MARK-scale age bands | `paper_trader.py:693-778` (`updates` at `:731-738`), `:1470`, `:1483-1492`; `bigquery_client.py:607-613, 626-665`; `api/paper_trading.py:181-230` (no `response_model`); live schema = 22 cols, no `marked_at`; `types.ts:641-665`; `paper-trading-utils.ts:35-40` |
| 5 | Researcher-grounded per-market mark-scheduling decision (close OR defer with rationale) | The analysis exists in the 2026-07-08 brief but the DECISION is recorded nowhere | **Record the decision.** Recommend explicit DEFERRAL + a masterplan follow-on. Zero code tonight: `mark_to_market` advances stops, so a new mark job is trading-behaviour-adjacent | `api/paper_trading.py:1390-1397, 1400-1423, 1426-1449` (70.5 reschedule — new constraint); `settings.py:390`; `paper_trader.py:728, 740-746` |
| 6 | Playwright capture of the live KR position | n/a | **IMPOSSIBLE tonight** — live book is 1 US row (NTAP), no KR/EU position. Close on the 61.2 pattern: build + Q/A + flip HELD on live_check + operator ask row. Do NOT seed a position | caller-verified live book; `verification.live_check` in `.claude/masterplan.json` |

## 9. Research Gate Checklist

Hard blockers:
- [x] ≥5 authoritative external sources READ IN FULL via WebFetch (**9**, all fresh; the
      2026-07-08 brief's 7 are re-cited for continuity and NOT counted)
- [x] 10+ unique URLs total (**~34** across 4 searches)
- [x] Recency scan (last 2 years) performed + reported (4 findings, 1 of which CORRECTS
      the prior brief's framing)
- [x] Full pages read (not abstracts) for the read-in-full set; 2 low-yield fetches
      recorded honestly rather than dropped
- [x] file:line anchors for every internal claim, re-verified against the working tree
      2026-08-07

Soft checks:
- [x] Internal exploration covered every module the criteria touch (`paper_trader.py`,
      `settings.py`, `bigquery_client.py`, `api/paper_trading.py`, `format.ts`,
      `positions-columns.tsx`, `cockpit-helpers.tsx`, `trades-columns.tsx`,
      `paper-trading-utils.ts`, `types.ts`, `vitest.config.ts`, `next.config.js`,
      `package.json`, the 4 relevant test files, live BQ schema)
- [x] Contradictions / consensus noted (locale-pinning sufficiency debate)
- [x] Claims cited per-claim
- Gaps stated honestly: (a) `backend/.env` is unreadable from this sandbox, so the live
  `PAPER_TRADING_HOUR` and the live flag values must be confirmed by Main; (b) Deloitte
  DART remains login-walled (same as 2026-07-08); (c) no live UI capture was taken — that
  is Main's Playwright step, and it cannot show a KR row tonight.

## 10. JSON envelope

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 9,
  "snippet_only_sources": 25,
  "urls_collected": 34,
  "recency_scan_performed": true,
  "internal_files_inspected": 16,
  "coverage": {
    "audit_class": false,
    "rounds": 1,
    "dry_rounds": 0,
    "K_required": 2,
    "new_findings_last_round": 0,
    "dry": false
  },
  "summary": "REVALIDATION delta for 61.3. Criterion 1 is HALF built: the LOCAL-weighted avg_entry formula is in the tree flag-gated (paper_trader.py:459-467, settings.py:477 default False) and 18 tests match the immutable -k filter, but every one asserts only avg_entry_price -- NO test anywhere drives _advance_stop on a KR position, so the breakeven-stop half of the criterion has zero coverage (the two _advance_stop test files are US-only AND deselected by the -k filter). Criteria 2/3/4 are untouched since 2026-07-08: resolveCurrency is still explicit-first (format.ts:161-171), resolveLocalCurrency was never written, all four $-template/undefined-locale offenders are live (+2 more in trades-columns.tsx), zero frontend tests touch currency, and paper_positions has no marked_at (live schema measured: 22 columns). Read path is SELECT * with no response_model, so a new column needs no API work. Criterion 5 should be DEFERRED with rationale, not built: mark_to_market advances stops (paper_trader.py:728), so a new mark job is trading-behaviour-adjacent and tonight's rules forbid it; phase-70.5 reschedule_paper_job adds a new constraint on where a second job may live. Criterion 6 is impossible (no KR position). Execution hazard: npm run build shares .next with the operator's :3000 dev server. External recency scan CORRECTS the prior brief: vercel #79397 read in full shows the hydration mismatch happened WITH an explicit locale (Node ICU-version divergence), so pin the locale but avoid exact-string Intl assertions in tests.",
  "brief_path": "handoff/current/research_brief_61.3_reval.md",
  "gate_passed": true
}
```
