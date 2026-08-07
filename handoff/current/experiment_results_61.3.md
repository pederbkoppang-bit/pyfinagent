# Experiment Results — masterplan step 61.3 (money-display + currency correctness)

**Cycle 178 | 2026-08-07 | GENERATE phase**
Contract: `handoff/current/contract_61.3.md` · Research gate:
`handoff/current/research_brief_61.3_reval.md` (`gate_passed: true`, 9 fresh sources)

**Headline:** five of six immutable criteria are built and test-proven; the sixth
(Playwright capture of a live KR position) is **structurally impossible tonight** — the
live book holds exactly one position and it is US. This step therefore closes
**deferred-with-reason** on the 61.2 / 72.0.2 pattern: code + tests + Q/A verdict
complete, `.claude/masterplan.json` status stays `pending`, flip held by the
`live_check` gate, operator ask row appended.

---

## 1. What was built

| # | Criterion | Status | Where |
|---|---|---|---|
| 1 | LOCAL add-on averaging + KRW-scale breakeven stop | **BUILT (test gap closed)** | `backend/tests/test_phase_61_3_addon_currency.py` (new, 7 tests) |
| 2 | Market-first currency on Entry/Current/Stop | **BUILT** | `format.ts::resolveLocalCurrency` (new) + 4 call sites |
| 3 | One en-US USD locale policy | **BUILT** | 2 NumberFlow sites pinned, 4 `$`-templates removed, 3 vitest specs |
| 4 | As-of indicator on a stale non-US P&L | **BUILT + migrated** | `marked_at` column, write path, prune set, type, UI chip |
| 5 | Per-market mark-scheduling decision | **DECIDED: explicit deferral** | §C5 below + queued follow-on step |
| 6 | Playwright capture of a live KR position | **IMPOSSIBLE** | §C6 below — measured live book |

### Files changed

```
backend/services/paper_trader.py                                   (+9 -1)
backend/tests/test_phase_61_3_addon_currency.py                    (new, 7 tests)
scripts/migrations/add_marked_at_to_paper_positions.py             (new)
frontend/src/lib/format.ts                                         (+20)
frontend/src/lib/paper-trading-utils.ts                            (+27)
frontend/src/lib/types.ts                                          (+8)
frontend/src/components/paper-trading/positions-columns.tsx        (+96 -21)
frontend/src/components/paper-trading/cockpit-helpers.tsx          (+5 -1)
frontend/src/components/paper-trading/trades-columns.tsx           (+11 -8)
frontend/src/lib/format.currency.test.ts                           (new, 13 tests)
frontend/src/lib/paper-trading-utils.mark.test.ts                  (new, 8 tests)
frontend/src/components/paper-trading/positions-columns.currency.test.tsx (new, 14 tests)
```

---

## 2. C1 — the stop half of criterion 1

The averaging formula already shipped (phase-70.3, flag `paper_avg_entry_fx_fix_enabled`,
`paper_trader.py:459-467`). The research gate measured what the 18 `-k`-selected tests
actually assert: **all of them read the saved row's `avg_entry_price` and none drives
`_advance_stop`**. The two files that do call `_advance_stop` are US-only *and*
deselected by the immutable filter. So the criterion's second clause — "and the
breakeven-advanced stop both remain KRW-scale" — had zero coverage.

That clause is the money-safety one. Under the legacy formula a KR add-on writes a
USD-per-share number into a KRW-scale field, the breakeven ratchet copies it into
`stop_loss_price`, and `check_stop_losses` then asks `70000 <= 46` — which never fires.
Downside protection is silently deleted.

New file, name triple-matches the immutable `-k` filter (`addon`, `currency`, `61_3`):

```
$ source .venv/bin/activate && python -m pytest backend/tests/test_phase_61_3_addon_currency.py -q
.......                                                                  [100%]
7 passed in 1.45s
```

Vacuity is guarded explicitly: `_advance_stop`'s breakeven branch returns `(None, None)`
unless `stop_advanced_at_R` is falsy and the current stop is below entry
(`paper_trader.py:1449-1452`), so
`test_61_3_fixture_reaches_breakeven_branch_not_a_vacuous_pass` asserts the branch fires
AND asserts both non-firing preconditions, in both directions.

### Kill-switch injection (why these tests inject and the old ones do not)

`PaperTrader` gates BUYs on `kill_switch.get_state()`, a module singleton whose
`_load_from_audit` replays real `pause` rows. The live switch is **paused right now**:

```
is_paused: True
{"paused": true, "pause_reason": "manual", "sod_nav": 23830.46,
 "peak_nav": 24666.57, "paused_at": "2026-08-07T13:43:09.567348+00:00"}
```

The new tests therefore construct the trader through its documented injection seam
(`kill_switch_state=`, `paper_trader.py:100-120`). This is not a safety bypass — it is
what makes a PaperTrader test deterministic instead of dependent on whether the book
happens to be paused that day.

---

## 3. C2/C3 — currency resolution and the locale policy

`resolveLocalCurrency({market, ticker})` is market-first **by contract**: it ignores
`base_currency`, which per the `PaperPosition` contract describes the USD columns and
which the backend hardcodes to `"USD"` on every row. Feeding that to the explicit-first
`resolveCurrency` is what rendered a won price as `$70,000.00`.

Swapped in at four LOCAL-price sites: positions Entry / Current / Stop Loss, plus the
trades Price column the earlier brief had missed. `market_value` and `pnl` are correctly
USD and were not touched.

Locale: both NumberFlow USD branches now always pass `numberFlowLocale(cur)` (`"en-US"`
for USD). Per NumberFlow's own docs, an omitted `locales` means *"the component will use
the browser's default locale"* — which is how an nb-NO browser rendered
`70 000,00 USD` beside a sibling cell's `$70000.00`. Pinning the shared `Dollar` fixes
both tables' USD aggregate columns in one edit.

The four `` `$${x.toFixed(2)}` `` templates are gone, routed through
`formatCurrency`/`formatUsd`.

**Assertions are prop-level, not output-level, and that is deliberate.** jsdom's ICU
resolves an omitted locale to en-US, so an output-only assertion would pass just as
happily against the defective code — green and blind. And exact-`Intl`-string assertions
are ICU-version-fragile: vercel/next.js#79397 (upstream nodejs/node#48120) documents the
*same explicit locale* producing `"1.000,00 €"` on Node 20.11.1 and `"1000,00 €"` on
20.19.0. So the specs assert regexes, absences, and props, with one exact-string check
only on the stable en-US/USD `"$1,234.56"` path.

```
$ cd frontend && npx vitest run src/lib/format.currency.test.ts \
    src/lib/paper-trading-utils.mark.test.ts \
    src/components/paper-trading/positions-columns.currency.test.tsx
 Test Files  3 passed (3)
      Tests  35 passed (35)
```

---

## 4. C4 — the as-of indicator

`marked_at` (ISO-8601 UTC) is stamped by `mark_to_market` on every position row, added
to `_POSITION_RT_FIELDS` so the pre-migration retry path degrades identically, exposed
on `PaperPosition`, and rendered as a chip on any P&L that is NOT live-recomputed.

The read path needed no API work: `get_paper_positions` is `SELECT *` and the portfolio
handler has no `response_model`, so the column reaches the frontend as a passthrough.

**The chip's age bands are deliberately NOT `bandFromAgeSec`.** Those thresholds
(green <90s, red ≥300s) are tuned for a ~60s live-price poll; reusing them would paint
every healthy once-a-day mark red within five minutes and the signal would carry no
information. `bandFromMarkAgeSec` uses green <26h (one cycle plus DST slack), amber <74h
(a Friday mark read on Monday is expected, not broken), red beyond.

### Migration executed — additive, NULLABLE, no row rewritten

```
$ python scripts/migrations/add_marked_at_to_paper_positions.py
INFO executing: ALTER TABLE `sunny-might-477607-p8.financial_reports.paper_positions`
    ADD COLUMN IF NOT EXISTS marked_at STRING OPTIONS(description='phase-61.3 as-of indicator: ...')
INFO marked_at column added (or already present) on sunny-might-477607-p8.financial_reports.paper_positions

$ python scripts/migrations/add_marked_at_to_paper_positions.py --verify
INFO verify OK: paper_positions.marked_at exists on sunny-might-477607-p8.financial_reports.paper_positions
```

Live row before (pre-migration read) and after — money columns byte-identical:

```
before: {'ticker': 'NTAP', 'quantity': 5.346643, 'avg_entry_price': 177.8498992919922,
         'cost_basis': 950.9, 'usd_ps': 177.8499, 'current_price': 188.8699951171875,
         'stop_loss_price': 164.62225}
after:  {'ticker': 'NTAP', 'quantity': 5.346643, 'avg_entry_price': 177.8498992919922,
         'cost_basis': 950.9, 'unrealized_pnl': 58.92, 'marked_at': None}
```

`marked_at` is NULL because populating it means running `mark_to_market` against the
live book, which rewrites position rows — deliberately NOT done on an unattended night.
It populates at the next scheduled cycle. `marked_at` moves no order, stop, or size, so
it is written unconditionally rather than flag-gated: it is observability, and the
project's standing rule is "bug fixes live, behaviour changes dark".

---

## 5. C5 — mark-scheduling decision: EXPLICIT DEFERRAL

**Decision: defer the ~07:00 UTC post-KRX-close mark job. Do not build it tonight.**

The gap is real and re-measured. KRX closes 15:30 KST = **06:30 UTC** (no DST). The live
cycle runs at `paper_trading_hour=14` ET (read from the running settings object, not
from the default) = 18:00/19:00 UTC. The next KR session runs 00:00–06:30 UTC, entirely
*before* the next cycle — so KR stop checks run against marks a full KR session old, and
gap risk realised at the KR open is invisible for hours. XETRA (15:30/16:30 UTC close)
is adequately served by the existing cycle.

Why deferred rather than built:

1. **`mark_to_market` moves stops.** It calls `_advance_stop` (`paper_trader.py:728`) and
   writes `stop_loss_price` / `stop_advanced_at_R`. A new scheduled mark is therefore
   trading-behaviour-adjacent, not a read-only refresh — forbidden on an unattended
   night.
2. **It could not be exercised.** The book holds no non-US position, so a dark scheduler
   job would be scaffolding with no evidence behind it.
3. **A new constraint appeared since the original design.** phase-70.5's
   `reschedule_paper_job` (`api/paper_trading.py:1426-1449`) re-adds the cron on a
   settings PUT; a second job added anywhere but inside `_add_scheduler_job` would
   silently vanish on the first settings change. Worth building once, correctly.

The criterion's own wording — "closing **or explicitly deferring** the stale-KR-stop-check
gap with rationale" — is satisfied by this record. Queued as its own masterplan step so
it is executable by someone with no memory of this cycle.

**Scope note, stated rather than discovered later:** the mark-only job would refresh
marks and stop-advancement but would NOT execute stops — that sell loop lives inline in
`autonomous_loop.py`, not in a reusable method. Closing the execution half needs that
extraction and is a separate decision.

---

## 6. C6 — Playwright capture: impossible, not skipped

Live book measured against BigQuery this cycle:

```
$ SELECT ticker, market, base_currency, quantity, avg_entry_price, cost_basis, ...
  FROM financial_reports.paper_positions
{'ticker': 'NTAP', 'market': 'US', 'base_currency': 'USD', 'quantity': 5.346643,
 'avg_entry_price': 177.8498992919922, 'cost_basis': 950.9, 'usd_ps': 177.8499,
 'current_price': 188.8699951171875, 'stop_loss_price': 164.62225}
```

One row, US. The criterion requires "the positions table with the **live KR position**".
Seeding a KR position would move the live book — forbidden. So `live_check_61.3.md` is
NOT written, the immutable command's third leg (`test -f`) stays red by design, and the
flip is held. Ask row appended.

The KR *rendering* is not unverified, it is verified differently: the component spec
renders the real cell renderers against a KR row carrying `base_currency: "USD"` — the
exact shape the backend ships — and asserts won symbols with no dollar sign on any
KRW-magnitude value.

---

## 7. Verification — verbatim

### Immutable command, leg 1 (pytest)

```
$ cd /Users/ford/.openclaw/workspace/pyfinagent && source .venv/bin/activate && \
  python -m pytest backend/tests -k 'addon or avg_entry or currency or 61_3' -q
FAILED backend/tests/test_64_3_currency_path.py::test_64_3_currency_path_kr_avg_entry_stays_krw
FAILED backend/tests/test_64_3_currency_path.py::test_64_3_currency_path_eu_avg_entry_stays_eur
FAILED backend/tests/test_64_3_currency_path.py::test_64_3_currency_path_us_byte_identical
FAILED backend/tests/test_64_4_multi_market_e2e.py::test_64_4_multi_market_e2e_currency_invariants
FAILED backend/tests/test_dod4_tier1_coverage_investment.py::test_paper_trader_execute_buy_average_up_recomputes_avg_entry
FAILED backend/tests/test_phase_70_3_atomic_swap.py::test_avg_entry_fx_fix_local_consistent_for_kr
6 failed, 19 passed, 2948 deselected, 1 warning in 7.06s
```

**This leg is RED, and the redness is pre-existing and out of scope — measured, not
asserted.** All six failures are the live kill-switch pause leaking into uninjected
PaperTrader tests (queued masterplan step **36.28**); the run logs 7 `kill_switch:
REFUSING BUY` lines. Control run on a clean worktree of HEAD (`53fdb54c`), with none of
this cycle's changes present:

```
$ git worktree add --detach <scratch>/wt-head HEAD
HEAD is now at 53fdb54c chore: auto-changelog hook entry for abcadfb8
$ cd <scratch>/wt-head && python -m pytest backend/tests -k 'addon or avg_entry or currency or 61_3' -q
FAILED backend/tests/test_64_3_currency_path.py::test_64_3_currency_path_kr_avg_entry_stays_krw
FAILED backend/tests/test_64_3_currency_path.py::test_64_3_currency_path_eu_avg_entry_stays_eur
FAILED backend/tests/test_64_3_currency_path.py::test_64_3_currency_path_us_byte_identical
FAILED backend/tests/test_64_4_multi_market_e2e.py::test_64_4_multi_market_e2e_currency_invariants
FAILED backend/tests/test_dod4_tier1_coverage_investment.py::test_paper_trader_execute_buy_average_up_recomputes_avg_entry
FAILED backend/tests/test_phase_70_3_atomic_swap.py::test_avg_entry_fx_fix_local_consistent_for_kr
6 failed, 12 passed, 2948 deselected, 1 warning in 3.80s
```

**Measured delta: identical 6 failures, 12 → 19 passed (+7, all mine), 0 new failures.**
Fixing those six means editing other steps' test files, which is 36.28's declared scope,
not this step's.

### Immutable command, leg 2 (frontend build)

```
$ PLAYWRIGHT_DIST_DIR=.next-verify-61-3 npm --prefix frontend run build
BUILD_EXIT=0
 ✓ Compiled successfully in 3.1s
```

**Disclosed deviation:** the immutable command says a bare `npm run build`, which writes
the default `.next` — the very directory the operator's live `:3000` dev server is
serving from. The build was run into an isolated `distDir` instead. Operator instance
verified healthy afterwards:

```
$ curl -s -o /dev/null -w "%{http_code}" http://localhost:3000/login   -> 200
$ curl -s -o /dev/null -w "%{http_code}" http://localhost:3000/        -> 302
$ launchctl list | grep frontend  -> 863  0  com.pyfinagent.frontend
```

**Side effect caught and reverted:** the isolated build rewrote `frontend/next-env.d.ts`
and `frontend/tsconfig.json` to point at the throwaway dist dir. Both were restored to
their HEAD contents by hand (`git diff` on both is now empty) and the throwaway dir is
gitignored (`.gitignore:25`). Had this shipped it would have broken the operator's type
references.

### Immutable command, leg 3 (live_check)

```
$ test -f handoff/current/live_check_61.3.md   -> ABSENT BY DESIGN (see §6)
```

### Typecheck

```
$ cd frontend && npx tsc --noEmit
tsc_exit=0
```

### Full frontend suite (regression check on untouched components)

```
$ cd frontend && npx vitest run
 Test Files  40 passed (40)
      Tests  303 passed (303)
     Errors  4 errors
```

The 4 unhandled rejections reproduce in isolation from `src/lib/api.single-flight.test.ts`
(`Tests 6 passed / Errors 4 errors`), a file this cycle did not touch — pre-existing.

### Lint gate (ruff F821,F401,F811 over the git-derived scope, non-empty asserted)

```
SCOPE:
backend/services/paper_trader.py
backend/tests/test_phase_61_3_addon_currency.py
scripts/migrations/add_marked_at_to_paper_positions.py
All checks passed!
RUFF_EXIT=0
```

---

## 8. Mutation matrix — 11 mutants, 11 killed (one only after a repair)

Each mutant was applied with a uniqueness-asserted `str.replace`, tested, then reverted.

| # | Mutation | Result |
|---|---|---|
| M1 | add-on averaging reverts to the legacy USD mix (flag becomes a no-op) | **killed** (2 tests) |
| M2 | drop `marked_at` from the `updates` dict | **killed** (1) |
| M3 | drop `marked_at` from `_POSITION_RT_FIELDS` | **killed** (1) |
| M4 | breakeven returns a USD-scaled entry instead of the LOCAL one | **killed** (2) |
| F1 | Entry cell reverts to `base_currency`-first | **killed** (3) |
| F2 | restore `locales={isUsd ? undefined : ...}` | **killed** (2) |
| F3 | `resolveLocalCurrency` delegates to `resolveCurrency` (helper becomes a synonym) | **killed** (1) |
| F4 | remove the as-of chip | **killed** (2) |
| F5 | chip also labels LIVE US rows (the honesty inversion) | **killed** (1) |
| F6 | `bandFromMarkAgeSec` collapses to the live-poll thresholds | **SURVIVED → repaired → killed (4)** |
| F7 | `ageSecFromIso` stops clamping a future timestamp | **killed** (1) |

**F6 is reported as a finding, not a footnote.** On the first pass the mark-age
thresholds — which *are* the honesty signal, deciding whether a stale P&L reads as
normal, notable, or broken — were asserted by nothing: the component spec checked only
the chip's text (`"as of 5h"`), which is identical under any banding. The guard stopped
one seam short of the behaviour it was supposed to protect. Repair:
`paper-trading-utils.mark.test.ts` (8 tests) pins the thresholds directly, including the
property that makes the function worth having (`bandFromMarkAgeSec(6h) === "green"` while
`bandFromAgeSec(6h) === "red"`), plus a component assertion on the chip's *class* rather
than its text. Re-run of F6 after the repair: 4 tests fail.

---

## 9. Scope honesty

**Deliberately not done, declared in the contract before GENERATE:**

- **The repo-wide locale sweep.** The 2026-07-08 brief measured 41 hardcoded `` `$${…}` ``
  sites and 9 `toLocaleString(undefined, …)` sites across `frontend/src`. All are
  genuinely-USD dashboard surfaces; the money-correctness fix needs only the
  paper-trading surfaces. The remainder is queued as its own step.
- **Promoting `paper_avg_entry_fx_fix_enabled`.** The formula ships flag-OFF, so the
  add-on defect is still live in production. The flag is absent from
  `settings_api.py::_FIELD_TO_ENV`, so it needs a `backend/.env` line + restart — an
  operator action, and `.env` writes are forbidden tonight. **Ask row.** Nothing in this
  cycle changes the runtime behaviour of the averaging path.
- **The stop engine.** `_advance_stop` is unmodified; the fix lands upstream so the
  engine receives a sane LOCAL entry.
- **Running `mark_to_market` against the live book** to populate `marked_at`.
- **The six red pre-existing tests** (36.28's scope, measured above).

**Discovered this cycle, recorded rather than silently fixed:** the live kill switch has
been paused (`pause_reason: "manual"`) since `2026-08-07T13:43:09Z`, so the book is not
trading. Resuming is a live-book action and belongs to the operator.

---

## 10. Artifact shape

- `handoff/current/contract_61.3.md` — plan, immutable criteria verbatim, risks
- `handoff/current/research_brief_61.3_reval.md` — gate, `gate_passed: true`
- `handoff/current/experiment_results_61.3.md` — this file
- `handoff/current/evaluator_critique_61.3.md` — Q/A verdict, transcribed verbatim
- `handoff/current/live_check_61.3.md` — **deliberately absent** (§6)
