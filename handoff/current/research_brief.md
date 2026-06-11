# Research Brief -- phase-60.3: Decision-input integrity for non-USD markets (AW-9, P1)

Tier: MODERATE-COMPLEX (caller-stated). Date: 2026-06-11. Agent: researcher (Layer-3 MAS, merged Explore).
Prior briefs archived at handoff/archive/phase-60.1/ and phase-60.2/.
Disclosed overrun: audit tables push past the 1500-word ceiling; prose kept tight (60.2 precedent).

## 1. Executive summary

- **Leg 1 (currency-honest prompts):** Both lite prompt sites render raw yfinance KRW values behind `$` literals -- autonomous_loop.py:1898 (Claude trader), :2137 (Gemini trader), :1627 (shared Risk-Judge template fed at :1973/:2184). Fix = convert via existing `fx_rates.get_fx_rate(ccy,"USD")` (fx_rates.py:182; 6h-cached; KRW=X inversion already handled) keyed off `markets.market_for_symbol(ticker)` (markets.py:142); on FX failure LABEL native currency, never render `$`. GIPS practice grounds both modes: convert to one base currency AND disclose the currency (gipsstandards.org Q&A 5336). CRITICAL: convert ONLY the prompt string -- `price_at_analysis` + persisted `market_data` stay native (fills/tolerance gate consume native via `_fx_local_to_usd`, paper_trader.py:32,:208).
- **Leg 2 (actionable integrity flags):** Deterministic pre-check AFTER the info fetch (:1839-1857/:2104-2120), acting IN CODE -- literature is unanimous that the LLM's prose flag cannot be the enforcement layer (GuardAgent: "admitted ... if O_l=0 or denied if O_l=1", code-executed; arXiv:2604.01483: intercept "before it reaches the execution environment"; "probabilistic execution without rigid constraints is architecturally and legally untenable"). Checks: USD-converted market cap > ceiling (largest real cap on earth = NVDA $4.854T 2026-06-10 -> $10T ceiling catches the persisted $44.5T/LG and $1.45-quadrillion/SK-hynix corruption with 2x+ headroom over NVDA); P/E==0 on a mega-cap = missing-data artifact of `info.get("trailingPE", 0)` (:1841/:2106); price-currency mismatch (`info.currency` vs `market_for_symbol` suffix). Tag -> EXCLUDE or FLOOR-SIZE via the 57.1 `blocked_out` mirror (portfolio_manager.py:60,:204; autonomous_loop.py:1159-1177) and the sizing hook at portfolio_manager.py:655; add a machine-readable flag to the judge schema (:1630-1637) -- today prose lands only in `reasoning`->`summary` and nothing reads it (BQ rows below prove flag-then-trade).
- **Leg 3 (staleness honesty):** KRX regular session 09:00-15:30 KST (UTC+9) = 00:00-06:30 UTC, no lunch break; observed cycle analyses run 18:03-18:06 UTC (BQ) = ~11.6h after close. Label the as-of (yfinance `regularMarketTime`/last hist bar) instead of "Price:". Calendar access exists: `markets.get_trading_calendar("KR")` (markets.py:168) -> xcals XKRX `session_close`. Recency caveat: KRX adds extended sessions from 2026-06-29 -- label as-of timestamps, don't hardcode "close=06:30 UTC" semantics.
- **Leg 4 (do-no-harm):** US byte-identity test mirrors `test_off_identity_prompts_are_verbatim_constants` (backend/tests/test_phase_57_1_reject_binding.py:187); name new tests `test_phase_60_3_*` (the -k net misses anything else; 59.1 lesson). Default-OFF flag, verbatim-constant OFF path = the 57.1 dark-launch idiom.

## 2. External research

### A. LLM-prompt input validation: deterministic pre-checks vs trusting the model

1. **Enforcement must be code, not prose.** GuardAgent: target-agent actions are "admitted by GuardAgent if O_l=0 or denied if O_l=1" -- the denial is executed code, not a textual remark; the guard "strictly follow[s] the safety guard requests to generate guardrail code"; >98% (EICU-AC) / >83% (Mind2Web-SC) guardrail accuracy with 100% preserved task accuracy (arXiv:2406.09187, read in full via /html, accessed 2026-06-11). A model-based guard failure mode is instructive: it "considerately" granted the access it was supposed to block -- exactly the shape of a Risk Judge that flags corruption in prose while the BUY executes.
2. **Deterministic gate ahead of the execution environment.** "Probabilistic execution without rigid constraints is architecturally and legally untenable" in financial systems; the orchestrator "intercepts this API call before it reaches the execution environment"; execution permitted iff the constraint is proven; design assumption "the LLM is compromised" (Rashie & Rashi 2026, *Type-Checked Compliance: Deterministic Guardrails for Agentic Financial Systems*, arXiv:2604.01483, read in full via /html, accessed 2026-06-11; same source family 57.1 used for the binding-REJECT gate).
3. **Production-guide consensus (2026, snippet tier):** input validation sits BEFORE the model, output filtering after, with deterministic checks (regex/bounds/allowlists) stacked under classifier checks; deterministic rails are "microsecond speed, deterministic behavior, easy auditing" and should be versioned rules acted on in the pipeline (orq.ai 2026 guide; myengineeringpath 2026; wiz.io; rulebricks -- snippet-only, full fetch returned nav header). Maps 1:1 to 60.3: bounds-check the info dict pre-prompt, act on the result in the candidate flow.

### B. Market-data sanity bounds

1. **Market-cap ceiling.** NVIDIA is the world's largest company at "$4.854 Trillion USD" as of June 10, 2026 (companiesmarketcap.com, read in full, accessed 2026-06-11; corroborating snippets: Motley Fool/alpha-sense ~$5.0-5.4T June 2026; first-ever $5T touch Oct 2025). A **$10T USD post-conversion ceiling** is defensible: >2x today's world record, yet 4x below the corrupted LG render ($44.5T) and ~145x below the SK hynix render ($1.45 quadrillion). Apply AFTER conversion -- a KRW-native ceiling is meaningless.
2. **P/E exactly 0 on a mega-cap = artifact.** yfinance omits `trailingPE` for many intl tickers; the code default `info.get("trailingPE", 0)` (:1841/:2106) manufactures 0.0 (a real P/E of 0 requires price 0). All four BQ KR rows persisted `pe_ratio=0.0` for LG Electronics / SK hynix -- both profitable. Even the judge read it correctly: "P/E of 0.0 is a data-quality flag (missing or negative earnings, not a cheap multiple)" (BQ 000660.KS 06-10). Pre-check rule: cap>$10B AND pe==0 -> treat as MISSING, never "cheap"; render "P/E: n/a".
3. **yfinance currency semantics (canonical behavior).** `info.currency` = LISTING/trading currency of prices (KRW for .KS; "GBp" pence for LSE); `financialCurrency` = statements' reporting currency; they legitimately diverge (Toyota TM: currency=USD, financialCurrency=JPY) and yfinance gives no per-field currency guarantee (GitHub issue #2699, read in full, accessed 2026-06-11 -- open, no maintainer resolution). Sibling defect: #2593 shows yfinance mixing INR prices with USD book values for ratio fields (snippet). `marketCap` is price x shares = LISTING currency -> KRW for .KS. **Mismatch detector:** `market_for_symbol(ticker)` currency (markets.py MARKET_CONFIG) vs `info.get("currency")` -- disagreement = FLAG; suffix is deterministic ground truth (markets.py:142-165 docstring already declares "the suffix IS the source of truth").

### C. FX presentation + KRX hours

1. **Convert-to-base AND disclose -- both are required practice.** "The GIPS standards require that firms disclose the currency used to express performance"; multi-currency composites "must convert the individual portfolio values to the composite's base currency"; method is flexible but must be applied consistently (GIPS Q&A 5336, gipsstandards.org, read in full, accessed 2026-06-11). For 60.3: USD-converted figures with an explicit currency note (e.g. "Price: $166.55 (converted from KRW 230,000 @ 1381/USD)") satisfies both halves; label-native-only is the compliant degraded mode when FX is unavailable.
2. **KRX hours.** Regular session 09:00-15:30 local (Asia/Seoul, UTC+9) -> 00:00-06:30 UTC close 06:30 UTC; existing off-hours blocks 07:30-09:00 / 15:40-18:00 (Korea Exchange, Wikipedia, read in full, accessed 2026-06-11; tradinghours.com corroborates "no lunch break" via search snippet -- direct fetch 403). A .KS quote consumed at the observed 18:03 UTC cycle is the 06:30 UTC close, ~11h33m old. **2026-06-29 change (recency):** KRX launches extended trading (pre 07:00-08:00 KST, after-market 16:00-20:00 KST; 24h target 2027) per FSC press release + BigGo (snippets) -- so the staleness label should state the quote's as-of timestamp (yfinance `regularMarketTime` epoch + `exchangeTimezoneName`) rather than a hardcoded close-time constant.

## 3. Recency scan (2024-2026)

Performed; substantive findings:
- arXiv:2604.01483 (Apr 2026): deterministic guardrails specifically for agentic FINANCIAL systems -- on-domain, supersedes generic 2023-era guardrail blogging.
- GuardAgent (2024, arXiv:2406.09187): code-executed guard verdicts; the enforcement pattern 57.1 adopted and 60.3 extends to data integrity.
- 2026 production guardrail guides (orq.ai "Complete 2026 Guide", myengineeringpath "Production LLM Safety Guide (2026)"): input-validation-before-model is now standard practice (snippet tier).
- KRX extended trading hours effective 2026-06-29 (FSC, BigGo): changes "how stale is a 18:03 UTC quote" semantics after that date; design the label around as-of timestamps, not a fixed close constant.
- NVDA $4.854T (2026-06-10, companiesmarketcap): current ceiling anchor; first $5T touch Oct 2025.
- yfinance #2699 (2025, open) + #2593: the currency-incoherence defect family is live and unfixed upstream -- downstream validation is the only defense.

## 4. Search queries run

| # | Query | Variant |
|---|---|---|
| 1 | "LLM guardrails deterministic input validation production pipeline" | year-less canonical (surfaced 2026-dated guides) |
| 2 | "largest company market cap 2026 NVIDIA trillion" | current-year |
| 3 | "yfinance currency financialCurrency international tickers KRW GBp wrong currency issue" | year-less (surfaced 2025 issues) |
| 4 | "KRX Korea Exchange trading hours 09:00 15:30 KST regular session close" | year-less (surfaced 2026-06-29 change) |
| 5 | "GIPS standards presentation currency disclosure multi-currency portfolio returns" | year-less canonical |

Three-variant discipline satisfied via the source-table mix (rule's second prong): current-year hits (companiesmarketcap 2026-06-10, orq.ai 2026, FSC 2026), last-2-year hits (GuardAgent 2024, yfinance #2699 2025), year-less canonical (GIPS Q&A, Wikipedia KRX, Boehmer-era none needed).

## 5. Internal code audit (all anchors verified current 2026-06-11, branch main)

### 5.1 Corrupted prompt sites ($-literal renders)

| Site | File:line | What renders |
|---|---|---|
| Claude lite trader prompt | backend/services/autonomous_loop.py:1894-1914, corrupted line :1898 | `Price: ${current_price:.2f} \| Market Cap: ${market_cap/1e9:.1f}B \| P/E: {pe_ratio:.1f}` |
| Claude BUY rule | :1903 | `market_cap > 5e9` -- USD-intended; 5e9 KRW = ~$3.6M so EVERY KR ticker trivially passes the size leg |
| Gemini lite trader prompt | :2133-2147, corrupted line :2137 | same `$` render |
| Gemini BUY rule | :2142 | same broken 5e9 semantics |
| Risk-Judge shared template | `_LITE_RISK_JUDGE_TEMPLATE` :1625-1638, line :1627 | `Market Cap: ${market_cap_b:.1f}B` |
| Risk-Judge system prompt | `_LITE_RISK_JUDGE_SYSTEM` :1613-1623, axis 3 :1619 | "market cap < $2B (micro-cap)" -- judge instructed in USD while fed KRW |
| Judge format calls | :1968-1978 (Claude; `market_cap_b=(market_cap or 0)/1e9` :1973); :2179-2189 (Gemini; :2184) | KRW cap / 1e9 presented as $B |

Value provenance (`stock.info`): `currentPrice`/`regularMarketPrice` :1839/:2104; `marketCap` :1840/:2105; `trailingPE` defaulted to 0 :1841/:2106; `sector`/`industry`/`shortName` :1842-1844/:2107-2109. Momentum :1847-1857/:2111-2120 is KRW/KRW -- currency-neutral, untouched. Persisted `full_report.market_data` keeps raw native values (Claude ~:2060-2078; Gemini :2243-2256).

### 5.2 Existing FX helpers to REUSE (never reimplement)

- `backend/services/fx_rates.py::get_fx_rate(from_ccy, to_ccy, date=None)` :182-196 -- live = 6h api_cache TTL (:53,:84-104), yfinance `KRW=X` inversion handled (:41), FRED `DEXKOUS` fallback (:46-51), BQ `historical_fx_rates` as-of (:153-179). Returns **None** on genuine failure -> label-native fallback required. `market_currency(market)` :56-58.
- `backend/services/paper_trader.py:32-41` `_fx_local_to_usd(market, date=None)`; consumed at :208 (fill), :371, :515 (MTM). This is why analysis-dict prices must STAY native.
- `backend/backtest/markets.py:142-165` `market_for_symbol` (.KS/.KQ->KR; suffix = source of truth); `MARKET_CONFIG["KR"]` :55-61 (KRW, Asia/Seoul, XKRX); already imported at the screener quality door backend/tools/screener.py:162-164.

### 5.3 Lite Risk Judge prompt + output schema

- Schema (template :1630-1637): `decision | recommended_position_pct | risk_level | reasoning | risk_limits`. **No machine-readable integrity field.** Prose lands in `reasoning` -> aliased `reason` (Gemini :2214-2218; Claude mirror ~:2027-2045) -> persisted ONLY as `summary` via `_persist_analysis` :2297 (`risk_assessment` is NOT inside `full_report_json` -- confirmed live in BQ). Nothing machine-reads it.
- 57.1 flag-gated prompt builders `_build_risk_judge_system/template` :1649-1680: the verbatim-constant-when-OFF idiom to copy.
- Consumption: `decide_trades` reads `decision` portfolio_manager.py:219, `recommended_position_pct` :655 (floor-sizing hook).

### 5.4 The 06-09 066570.KS persisted row (regression fixture) -- pulled live via BQ ADC

`financial_reports.analysis_results` (us-central1): `analysis_date=2026-06-09T18:03:49.454653Z, ticker=066570.KS, recommendation=BUY, final_score=7.0, price_at_analysis=248000.0, market_cap=44540606021632.0, pe_ratio=0.0`; `full_report_json.market_data={industry:"Consumer Electronics", market_cap:44540606021632, momentum_20d:60.93, momentum_60d:111.96, name:"LGELECTRONICS", pe_ratio:0, price:248000, sector:"Technology"}`; summary (judge prose) verbatim: "a $44,540.6B (~$44.5T) market cap is physically impossible for LG Electronics (a KRW/USD unit error on the newly-onboarded KR market), a data-integrity failure... reject and do not chase the trader's BUY regardless of 68 confidence." **The BUY executed anyway** (57.1 binding flag dark) -> stop-out.
Siblings: 000660.KS 06-09 18:06 `market_cap=1572328605483008.0` ("nonsensical units error"); 06-10 18:03 ("implausible as USD (likely unconverted KRW)"); 06-08 18:05 ("~$1.3 quadrillion... impossible"). Note 44.54e12 KRW at ~1380 KRW/USD ~= $32B -- sane for LG once converted.
Verbatim query for Main:
```sql
SELECT analysis_date, ticker, final_score, recommendation, summary, price_at_analysis, market_cap, pe_ratio,
       TO_JSON_STRING(full_report_json) AS fr
FROM `sunny-might-477607-p8.financial_reports.analysis_results`
WHERE ticker IN ('066570.KS','000660.KS') AND DATE(analysis_date) BETWEEN '2026-06-03' AND '2026-06-10'
ORDER BY analysis_date DESC
```
(python client, `location="us-central1"`, 30s timeout.)

### 5.5 Where exclusion/floor-sizing can act IN CODE

- **Pre-LLM (preferred for hard corruption):** inside `_run_claude_analysis`/`_run_gemini_analysis` right after the info fetch (:1839-1857 / :2104-2120) -- the only place the info dict exists. Wrapper alternative `_run_and_persist_one` :847-890 has no info dict.
- **Post-analysis, pre-decision:** between candidate assembly :897/:904 and `decide_trades` :1163.
- **In decide_trades (floor-sizing/exclusion):** portfolio_manager.py:53 already takes `blocked_out` :60 with append pattern :204-205; position-pct consumption :655. Mirror 57.1's out-channel wiring at autonomous_loop.py:1159-1177 -> `summary["data_integrity_blocked"]`.
- **Deterministic-validator idiom to mirror:** `backend/tools/price_quality.py::validate_ohlcv` :48 -- two-tier FLAG/DROP rules R1-R4 (:14-21), US fast-path no-op, returns `(df, report)`; screener door screener.py:162-164. 60.3's pre-check = the info-dict analogue (e.g. `validate_info(info, market, ticker) -> (info, report)`).

### 5.6 Staleness: as-of timestamps + KRX close exposure

- Quote source `stock.info` :1835-1839/:2100-2104; yfinance info carries `regularMarketTime` (epoch) + `exchangeTimezoneName`; `hist.index[-1]` :1837/:2102 = last bar date fallback.
- markets.py has NO close-time helper -- only `is_trading_day` :192-213 (xcals `is_session`, 50.4 rewrite) and `get_trading_calendar` :168-189 (xcals XKRX exposes `session_close(session)` UTC). `MARKET_CONFIG["KR"].timezone="Asia/Seoul"` :58.
- Observed: BQ stamps 18:03-18:06 UTC vs 06:30 UTC KRX close = ~11.6h stale, presented as "Price:" with no qualifier.

### 5.7 Existing test inventory

- `-k 'prompt_fx or lite_prompt or 60_3'` matches NOTHING today (3/725 collected only via the broader `fx` token = fx_rates/50.x tests; 7 pre-existing env-coupled collection errors in tests/ root, none in backend/tests/). New tests MUST be `test_phase_60_3_*`.
- Reusable byte-identity fixtures: backend/tests/test_phase_57_1_reject_binding.py -- `test_off_identity_prompts_are_verbatim_constants` :187, `test_prompt_content_flag_on_real_cap_and_sector_line` :200, off-identity orders :174.
- Adjacent: tests/services/test_risk_judge_lite_path.py (:8,:32,:56), tests/services/test_persist_lite_analysis.py, backend/tests/test_phase_50_3_universe.py:21-26, backend/tests/test_phase_50_4_calendar.py:48-51.

## 6. Risks and gotchas

1. **Do NOT convert `price_at_analysis` or persisted `market_data`** -- execute_buy's tolerance gate (autonomous_loop.py:1204-1238) compares native-vs-native; fills convert at paper_trader.py:208. Convert/label ONLY the prompt string.
2. **Momentum is currency-neutral** (:1847-1857) -- leave untouched.
3. **`info.get("trailingPE", 0)` conflates missing with 0.0**; same for `marketCap` 0. Pre-check treats 0/absent as MISSING.
4. **`market_cap > 5e9` BUY rule** (:1903/:2142) silently broke for KR; conversion restores intent -- but that IS a prompt-behavior change, so flag-gate it (US byte-identity preserved either way since US values are already USD; the test must prove byte-identity, not assume it).
5. **fx_rates.get_fx_rate can return None** -> degraded mode = label-native ("KRW 248,000"), never a silent `$`.
6. **57.1 dark-launch idiom:** default-OFF settings flag; OFF path returns verbatim constants. Note nuance: for US tickers even the ON path must be byte-identical (ccy=="USD" fast-path returns 1.0 at fx_rates.py:190-191 -- format must avoid re-rendering, e.g. only branch when `market != "US"`).
7. **Ceiling applies post-conversion** ($10T USD; NVDA $4.854T anchor). A KRW-native ceiling is meaningless.
8. **Judge-schema change ripple:** adding a structured flag key requires `_LITE_RISK_DEFAULT` :1640-1646 + both parse sites (~:2027, :2200-2225) updated, else fallback drift; `save_report` summary alias `reason` :2218 must survive.
9. **yfinance `info.currency` can itself be wrong/missing** (#2699, #2593) -- the mismatch check FLAGS, it does not auto-convert using `info.currency`; conversion keys off the suffix-derived market (deterministic, no network).
10. **KRX extended hours from 2026-06-29** -- staleness label must render the quote's as-of timestamp, not a hardcoded "close was 06:30 UTC".

## 7. Source table

### Read in full (counts toward gate)

| # | URL | Accessed | Kind | Fetched how | Key finding |
|---|---|---|---|---|---|
| 1 | https://arxiv.org/html/2604.01483 (Rashie & Rashi 2026, Type-Checked Compliance) | 2026-06-11 | preprint (arXiv) | WebFetch /html full render | "probabilistic execution without rigid constraints is architecturally and legally untenable"; intercept "before it reaches the execution environment"; assume "the LLM is compromised" |
| 2 | https://arxiv.org/html/2406.09187 (GuardAgent) | 2026-06-11 | preprint (arXiv, NeurIPS-era 2024) | WebFetch /html full render | verdict ENFORCED by code: "admitted ... if O_l=0 or denied if O_l=1"; >98%/83% accuracy; model-based guard "considerately" granted what it should block |
| 3 | https://companiesmarketcap.com/nvidia/marketcap/ | 2026-06-11 | market-data reference | WebFetch full | NVDA "$4.854 Trillion USD" as of 2026-06-10; "world's most valuable company by market cap" -> $10T ceiling defensible |
| 4 | https://github.com/ranaroussi/yfinance/issues/2699 | 2026-06-11 | upstream issue tracker | WebFetch full | `currency`=listing ccy of PRICES vs `financialCurrency`=statements ccy; Toyota TM USD/JPY divergence; open, unresolved |
| 5 | https://www.gipsstandards.org/qadatabase/5336/ | 2026-06-11 | official standard (CFA Institute GIPS) | WebFetch full | "firms [must] disclose the currency used to express performance"; multi-ccy composites "must convert ... to the composite's base currency"; method consistent |
| 6 | https://en.wikipedia.org/wiki/Korea_Exchange | 2026-06-11 | reference | WebFetch full | KRX regular session 09:00-15:30 local (UTC+9 -> 06:30 UTC close); off-hours 07:30-09:00 / 15:40-18:00 |

### Identified but snippet-only (does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|---|---|---|
| https://rulebricks.com/blog/deterministic-guardrails-for-llms-building-safe-auditable-ai-systems | vendor blog | fetch returned nav header only; quotes via search snippet ("deterministic safety rails ... explainable, versioned") |
| https://www.tradinghours.com/markets/krx | reference | HTTP 403; snippet confirms 09:00-15:30 KST no lunch break |
| https://orq.ai/blog/llm-guardrails | vendor guide (2026) | snippet sufficient; corroborates A3 |
| https://myengineeringpath.dev/genai-engineer/ai-guardrails/ | guide (2026) | snippet; three-layer guardrail architecture |
| https://www.wiz.io/academy/ai-security/llm-guardrails | vendor guide | snippet; input/output/runtime checks |
| https://guardrailsai.com/blog/guardrails-mlflow | vendor blog | snippet; deterministic validators as scorers |
| https://github.com/ranaroussi/yfinance/issues/2593 | issue tracker | snippet; yfinance mixes INR price / USD book value (currency incoherence family) |
| https://github.com/ranaroussi/yfinance/issues/1251 | issue tracker | snippet; info-dict reliability history |
| https://www.fool.com/research/largest-companies-by-market-cap/ | research page | snippet; ~$5T NVDA corroboration June 2026 |
| https://www.alpha-sense.com/largest-companies-by-market-cap/ | research page | snippet; corroboration |
| https://finance.biggo.com/news/LgbmW5wBZk7xib5fNndL | news | snippet; KRX extended hours June 29 2026, 24h by 2027 |
| https://www.fsc.go.kr/eng/pr010101/83967 | official regulator press release | snippet; KRX schedule change |
| https://www.gipsstandards.org/wp-content/uploads/2021/03/2020_gips_standards_firms.pdf | official standard PDF | Q&A 5336 answered the precise question; PDF not needed |
| https://www.market-clock.com/markets/krx/equities/ | reference | snippet; hours corroboration |

(+ remaining unique search-result URLs not tabled: kalviumlabs, medium/ajayverma, qed42, polymarket, tradingkey, statista, stockanalysis, twelvedata, global.krx.co.kr, grokipedia, weex, quora, yahoo-help, stock-data-solutions, afg.asso.fr, ryanoconnellfinance, matsonmoney, nasra, globalexchanges, arxiv 2604.07264 -- collected, evaluated by title/snippet, not load-bearing.)

## 8. Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (6)
- [x] 10+ unique URLs total (40 collected across 5 searches)
- [x] Recency scan (2024-2026) performed + reported (Section 3, substantive)
- [x] Full pages read (not abstracts) for the read-in-full set (arXiv via /html per protocol)
- [x] file:line anchors for every internal claim (Section 5; BQ row pulled live)

Soft checks:
- [x] Internal exploration covered every relevant module (autonomous_loop, fx_rates, paper_trader, markets, portfolio_manager, screener, price_quality, tests)
- [x] Contradictions/consensus noted (GIPS: convert vs label BOTH valid -> recommend convert+disclose; yfinance currency fields unreliable -> suffix is ground truth)
- [x] All claims cited per-claim

```json
{
  "tier": "moderate-complex",
  "external_sources_read_in_full": 6,
  "snippet_only_sources": 14,
  "urls_collected": 40,
  "recency_scan_performed": true,
  "internal_files_inspected": 10,
  "report_md": "handoff/current/research_brief.md",
  "gate_passed": true
}
```
