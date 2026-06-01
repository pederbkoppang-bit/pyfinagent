# Q/A Critique — goal-multimarket-ux (Multi-Market UX: US / EU / KR)

**Evaluator:** Q/A (Layer-3 harness MAS, merged qa-evaluator + harness-verifier)
**Date:** 2026-06-01
**Verdict:** PASS (visual_verification: pending-operator)
**Mode:** in-place working-tree read (changes UNCOMMITTED; no worktree, no checkout)

---

## 1. Harness-compliance audit (5 items)

1. **Researcher spawned before contract** — PASS. `research_brief.md` exists,
   8 sources read in full, 18 URLs, recency scan present (4 findings; the
   "radiogroup-in-toolbar breaks selection-follows-focus" nuance is load-bearing
   and is honoured by the standalone-radiogroup `MarketFilter`), `gate_passed: true`.
   Contract "Research-gate summary" cites it with file:line internal anchors.
2. **Contract written before GENERATE** — PASS. `contract.md` has step id
   (`goal-multimarket-ux`), all 7 immutable criteria copied verbatim, a
   dependency-ordered plan A-D, and a references block.
3. **experiment_results present** — PASS. Lists what was built (A+B+C), the
   NEW/MODIFIED file list, verbatim tsc/build/format-proof output, and a
   per-criterion status table.
4. **Log-last discipline** — PASS. `grep -i goal-multimarket handoff/harness_log.md`
   returns only historical phase-5/phase-50 references — NO `goal-multimarket-ux`
   cycle header. Log NOT yet appended; masterplan has NO such step (it is a /goal
   step, not a phase id). Correct: log + status come AFTER this PASS.
5. **No verdict-shopping** — PASS. No prior `goal-multimarket-ux` CONDITIONAL/FAIL
   in `harness_log.md`. This is a FIRST verdict on the full A+B+C evidence (the
   prior partial A+B was never adjudicated by Q/A). Not a reversal-on-unchanged-
   evidence. sycophancy-under-rebuttal / mtime heuristics N/A (no prior cycle).

---

## 2. Deterministic checks (re-run independently — did not trust generator numbers)

### 2.1 TypeScript (binding gate)
```
$ cd frontend && node_modules/.bin/tsc --noEmit -p tsconfig.json
TSC_EXIT=0
```
Reproduced independently → **0**. Matches experiment_results.

### 2.2 Real-module format.ts proof (transpiled the SHIPPED file, then node)
Transpiled `src/lib/format.ts` standalone (`tsc ... --module esnext --target es2022
--moduleResolution bundler --skipLibCheck`, TRANSPILE_EXIT=0; the `import type
{ Format }` was correctly erased so the module loads with NO NumberFlow dependency),
then imported the emitted `format.js` in node and asserted against the REAL exports:

```
PASS formatCurrency EUR 243.1            -> "€243.10"
PASS formatCurrency KRW 71200 (0dp)      -> "₩71,200"      <-- load-bearing: KRW 0-decimal
PASS formatCurrency USD 971.55           -> "$971.55"
PASS marketForSymbol SAP.DE              -> "EU"
PASS marketForSymbol 005930.KS           -> "KR"
PASS resolveMarket {market:"kr"} wins    -> "KR"
PASS MARKET_BENCHMARK_LABEL.EU           -> "DAX"
PASS KRW no-decimal proof (123456.78)    -> "₩123,457"     <-- confirms NOT forcing minFractionDigits
PASS formatUsd 1694                      -> "$1,694.00"
PASS formatCurrency null -> dash         -> "—"
PASS marketForSymbol bare AAPL -> US     -> "US"           <-- do-no-harm: bare ticker => US
PASS resolveCurrency bare -> USD         -> "USD"          <-- do-no-harm: => USD
PASS resolveCurrency SAP.DE -> EUR       -> "EUR"
PASS marketForSymbol EQNR.OL -> NO       -> "NO"
PASS US Sat closed                       -> false
PASS unknown currency no-throw (ZZZ)     -> "ZZZ 100.00"   (catch-fallback, never throws in a cell)
RESULT pass=16 fail=0  NODE_EXIT=0
```
The KRW 0-decimal claim is PROVEN against the shipped module (₩71,200 and ₩123,457,
no `.00`). The do-no-harm US byte-identity path is proven (bare→US→USD).

### 2.3 Flag-emoji + emoji scan (the NO-flag-emoji / no-emoji rule)
```
grep -rnP '[\x{1F1E6}-\x{1F1FF}]' <12 scope files>                 -> none (rc=1)
grep -rnP '[\x{1F300}-\x{1FAFF}\x{2600}-\x{27BF}\x{2B00}-\x{2BFF}\x{1F000}-\x{1F0FF}]' -> none (rc=1)
```
NO flag emoji and NO pictograph/dingbat emoji anywhere in the scope. Market is
conveyed by a colored dot (`aria-hidden`) + the market CODE (WCAG: not colour-alone).

### 2.4 Do-no-harm / no-client-FX guard audit
Every hardcoded `currency:"USD"` and every literal `$` money prefix is inside an
explicit US byte-identity branch:
- `cockpit-helpers.tsx:86-95` — `Dollar`: `isUsd ? {currency:"USD", minFrac:2} : numberFlowFormat(cur)`, `locales={isUsd ? undefined : ...}`. USD = EXACT legacy object → byte-identical.
- `positions-columns.tsx:66-74` — `CurrentPriceCell`: same `isUsd` ternary.
- `positions-columns.tsx:151-153` (Entry), `:261` (Stop Loss) — `cur === "USD" ? \`$${...}\` : formatCurrency(...)`.
- `trades-columns.tsx:97-99` (Price) — `cur === "USD" ? \`$${price}\` : formatCurrency(...)`. Price is LOCAL, correctly branched.
- `trades-columns.tsx:119` (Fee) — hardcoded `$` is CORRECT: a fee/`transaction_cost` is always USD base (per contract #3).
- `LatestTransactionsBox.tsx:47-49` — `cur === "USD" ? \`$${...}\` : formatCurrency(p, cur)`.
- `format.ts:169` — hardcoded USD only in the `catch` fallback for an unknown currency code (safety; never throws in a table cell).

**No client-side FX** (the would-be correctness BLOCK): every `livePrice * quantity`
recompute is gated behind `isUs` (`resolveMarket(...) === "US"`):
- `positions/page.tsx:67-72` `mvUsd`: US → `px * quantity`; non-US → `pos.market_value` (backend USD).
- `positions-columns.tsx:192-197` and `:202-206` (Market Value accessor + cell): `isUs && livePrice!=null ? livePrice*qty : (row.market_value ?? 0)`.
- `positions-columns.tsx:217-227` / `:232-242` (P&L%): US recomputes from live price; non-US uses backend `unrealized_pnl_pct` (USD-consistent).
LOCAL notional is NEVER labelled USD. The donut + sector bar use `mvUsd` (same guard).

### 2.5 ESLint (REQUIRED — diff touches frontend/**; catches Rules-of-Hooks which tsc cannot)
```
$ cd frontend && npx eslint <12 scope files>
3 problems (0 errors, 3 warnings)   ESLINT_EXIT=0
```
**Exit 0 → gate PASS** (errors-only fail semantics; warnings do not fail).
`react-hooks/rules-of-hooks` (severity "error" in eslint.config.mjs:34) fired
ZERO times — no hook-order violation of the phase-23.2.23 class.
The 3 warnings are all `react-hooks/set-state-in-effect` (a perf advisory, NOT
an error):
- `layout.tsx:175` — the active-market "reset to ALL when the market disappears"
  effect (intentional derived-state sync; the contract calls for this fallback).
- `layout.tsx:214` — `refresh()` on mount; PRE-EXISTING (phase-44.2), not this diff.
- `MarketSessionStrip.tsx:26` — `setNow(new Date())` post-mount; this is the
  DOCUMENTED correct pattern to avoid an SSR hydration mismatch on the OPEN/CLOSED
  text (the alternative causes the very bug it avoids). Defensible.

### 2.6 secret-in-diff
`git diff -- frontend/ | grep -iE '(api_key|secret|password|token)...'` → none (rc=1).

### 2.7 Backend markets.py diff
Additive metadata only: Nordic markets (SE/DK/FI/IS) added to `MARKET_CONFIG`,
`YF_SUFFIX`, and `market_for_symbol` suffix branches. The comment correctly notes
trading only happens for codes in `PAPER_MARKETS`, so this does NOT change the live
US/EU/KR loop. The frontend `format.ts` mirror matches these suffixes exactly
(.ST→SE, .CO→DK, .HE→FI, .IC→IS, .OL→NO, .TO→CA, .DE/.PA/.AS/.F→EU, .KS/.KQ→KR).

---

## 3. The 7 immutable criteria — independent verification

1. **Global market filter scopes EVERY table/KPI/donut/sector bar; "All"=USD** — PASS.
   `MarketFilter` (radiogroup) is mounted in `layout.tsx:484-488` as a Tier-4 global
   control; state owned by the layout (`activeMarket`, default "ALL") and published via
   `PaperTradingDataContext`. Verified scope, not just the table:
   - positions table: `positions/page.tsx:52-60` `visiblePositions` filter.
   - trades table: `trades/page.tsx:23-31`.
   - donut: `allocationSlices` (`page.tsx:122-132`) iterates `visiblePositions`; center
     `totalNav` switches to `filteredNavUsd` for a single market (`:168`).
   - sector bar: `sectorItems` (`page.tsx:104-116`) iterates `visiblePositions`; single-
     market denominator = `filteredNavUsd` so sectors sum to ~100% within the market.
   - KPI tile: `SummaryHero` Positions count filters (`cockpit-helpers.tsx:189-195`).
   "All" uses fund USD NAV/cash; Cash donut slice only shown in "All" (`page.tsx:130`).
2. **MARKET column; chip = colored dot + code; NO flag emoji** — PASS.
   `MarketChip` (`cockpit-helpers.tsx:108-135`): `aria-hidden` dot + code (+ optional
   exchange tag). Market column added to positions (`positions-columns.tsx:99-107`) and
   trades (`trades-columns.tsx:56-64`). Colors US sky / EU amber / KR violet
   (`format.ts:100-110`). Emoji scan = clean.
3. **Dual currency: price/entry/stop LOCAL; value/NAV/cost-basis/fee USD; no client FX** — PASS.
   Entry/Current/Stop-Loss consume `resolveCurrency(row)` (LOCAL); Market Value, Value,
   Fee, NAV, Cash use USD `Dollar`/backend `market_value`. No client-side FX (§2.4).
   Mirrors backend (price/current_price/stop = LOCAL per phase-50.2; market_value/
   cost_basis = USD).
4. **Locale-correct Intl; KRW 0dp; no stray hardcoded $/USD on market-dependent paths** — PASS.
   Proven in §2.2 (KRW 0-decimal) + §2.4 (every market-dependent path is USD-guarded;
   the only literal-$ on a market-dependent value branches on `cur === "USD"`).
5. **Dynamic "vs SPY/DAX/KOSPI" benchmark label** — PASS.
   `cockpit-helpers.tsx:198` `benchLabel = vs ${isAll ? "SPY" : MARKET_BENCHMARK_LABEL[...]}`.
   **Honest fallback (scrutinized):** for a specific non-US market the per-market index
   return is NOT exposed by the API, so the value shows that market's USD-consistent
   HOLDINGS return (`sum unrealized_pnl / sum cost_basis`, both USD — `:209-218`) with an
   explanatory `title` tooltip, rather than inventing an FX-converted excess. This is the
   correct honest choice — it does NOT fabricate an FX excess and discloses the limitation.
6. **Market-session strip (open/closed per active market)** — PASS.
   `MarketSessionStrip` (mounted `layout.tsx:489`) renders emerald/slate dot + OPEN/CLOSED
   from `isMarketOpen` (`format.ts:212-229`, weekday + local cash-session window per
   exchange tz). Holiday-blind by design and documented (backend exchange_calendars is the
   authoritative gate; this is a UI hint). SSR-safe (null→Date on mount; refresh 60s).
7. **Cockpit "Latest Transactions" + Reports widgets show market chip + local price** — PASS.
   `LatestTransactionsBox.tsx:135-141` market dot before ticker; `:149` price via
   `fmtPrice(t.price, t.ticker)` → LOCAL (USD byte-identical). Reports History is
   score-based (no price/currency field) so "local price" is legitimately N/A there —
   honestly disclosed in experiment_results, not a dodge.

---

## 4. Code-review heuristics (5 dimensions) — findings

- **Dim 1 Security**: secret-in-diff = none. No LLM/exec/SQL/SSRF sinks (pure UI). No
  dep-pin removal (no package.json change). Clean.
- **Dim 2 Trading-domain correctness**: no kill-switch / stop-loss / perf-metrics /
  execution paths touched (pure presentational UI + one additive backend metadata map).
  The ONE correctness-relevant surface — mislabeling LOCAL notional as USD — is
  explicitly guarded (§2.4); NOT a BLOCK. markets.py is additive config (no NOT NULL
  column add, no live-table migration). Clean.
- **Dim 3 Code quality**: no `print`, no broad-except in TS. New public functions in
  `format.ts` are typed. NOTE only: `MARKET_HOURS` open/close minutes are inline numeric
  literals (e.g. `9*60+30`) — they are self-documenting session windows with comments,
  not financial-formula magic numbers; NOTE, not WARN.
- **Dim 4 Anti-rubber-stamp**: no `perf_metrics.py`/`risk_engine.py`/`backtest_*` change →
  the financial-logic-without-behavioral-test BLOCK does NOT apply. The new financial-
  display logic (`format.ts`) IS exercised by a real-module behavioral proof (§2.2, 16
  assertions against the transpiled shipped code) — the strongest available test for a
  pure formatter. No tautological/over-mocked assertions. Not a rubber-stamp.
- **Dim 5 LLM-evaluator anti-patterns**: this critique cites file:line throughout (not
  <3-sentence). First verdict on this evidence → no sycophancy-under-rebuttal, no
  second-opinion-shopping, no 3rd-conditional escalation due. position/verbosity-bias N/A.

Worst severity across all dimensions: **NOTE** (no BLOCK, no WARN). Verdict not degraded.

---

## 5. LLM judgment — scope honesty + alignment

- **Contract alignment**: all 7 criteria are genuinely wired in code, not just the table
  (criterion #1 scopes donut/sector/KPI verified at file:line; #5 label is dynamic AND the
  non-US fallback is an honest USD holdings-return with disclosure, not a fabricated FX
  excess).
- **Mutation resistance**: the §2.2 real-module proof WOULD catch a regression of the
  KRW-0dp claim (asserts ₩71,200 with no `.00`) and of the US byte-identity path
  (asserts bare→US→USD and $971.55/$1,694.00 exact). The §2.4 guard audit would catch a
  client-FX regression (any un-gated `livePrice*qty` labelled USD). tsc=0 + ESLint
  errors=0 catch type + hook-order regressions.
- **Scope honesty**: the visual-browser caveat is honest and well-grounded — the live
  book is currently all-US (first multi-market cycle Mon 14:00 UTC), so EU/KR € / ₩ row
  rendering cannot be seen live yet; the US-unchanged + filter/session-strip rendering can
  be confirmed now. Marking visual verification "pending operator review" per
  `.claude/rules/frontend.md` rule 5 is the documented, legitimate path for color-coded UI
  — NOT a dodge. The EU/KR rendering is proven deterministically against the real module.
  "DONE" claims map to verifiable code; nothing is overclaimed.

---

## Verdict

**PASS** — all 7 immutable criteria satisfied in code; tsc=0; real-module format proof
16/16 (incl. KRW 0-dp ₩71,200 and US byte-identity); ESLint exit 0 with zero
rules-of-hooks errors; no flag/any emoji; every market-dependent money path USD-guarded
(do-no-harm) and no client-side FX; no secret-in-diff; no BLOCK/WARN heuristic.

`visual_verification: pending-operator` — the browser pass is deferred because the live
portfolio is all-US until Mon 14:00 UTC; this is a documented limitation (frontend rule 5),
not a code defect. EU/KR rendering is proven deterministically.

checks_run: harness_compliance_audit, syntax/tsc, real_module_format_proof, eslint,
emoji_scan, do_no_harm_guard_audit, no_client_fx_audit, secret_scan, backend_diff_review,
code_review_heuristics, evaluator_critique, harness_log_prior_verdict_scan
