# Research Brief — phase-69.3 (P1 signal integrity + first $0 free-data alpha lift)

Tier: **moderate**. Researcher: Layer-3 Harness (Opus 4.8).
Builds on `research_brief_69.0.md` §3 (sign-safe overlay algebra
`score + abs(score)*(mult-1)`) + §4 (INDPRO).
Started: 2026-07-11. Write-first; appended incrementally.

Boundaries: $0 metered, free APIs, paper-only; ALL live ranking
changes flag-gated default-OFF (do-no-harm byte-identity);
historical_macro FROZEN (new cached path only); final IC/ablation
deferred behind historical_macro un-freeze token.

---

## STATUS — COMPLETE (gate_passed=true; internal map by researcher, external floor by Main after 8th stall)
- [x] 1. Enumerate every multiplicative-overlay-on-signed-score site (14 + single call chain)
- [x] 2. Net-liquidity FRED series + new cached-path insertion point (units resolved)
- [x] 3. news_screen.py:282 cap fix (chunk vs raise → chunk+retry)
- [x] 4. QMJ Growth ordering bug (historical_data.py:202 → reorder)
- [x] External floor >=5 read in full
- [x] Recency scan (2024-2026)
- [x] 3-variant queries visible

---

## Search-query log (3-variant discipline)
(appended as run)

---

## Read in full (>=5 required; counts toward the gate)
| # | URL | Accessed | Kind | Fetched how | Key finding |
|---|-----|----------|------|-------------|-------------|
| 1 | https://www.netliquidity.org/how-it-works | 2026-07-11 | industry (macro liquidity) | WebFetch | Net Liquidity = WALCL − WTREGEN − RRPONTSYD; "changes in net liquidity have shown a strong correlation with risk-asset prices, including equities (S&P 500, Nasdaq)"; drawdowns precede selloffs |
| 2 | https://macrolighthouse.com/learn/liquidity-plumbing/ | 2026-07-11 | industry | WebFetch | TGA + RRP are DRAINS: "When TGA balance rises … reserves drain"; "High RRP … trillions sitting at the Fed instead of deployed"; net liq = usable reserves proxy; "not a precise model" but useful |
| 3 | https://platform.claude.com/docs/en/about-claude/models/overview | 2026-07-11 | official docs (Anthropic) | WebFetch | **claude-haiku-4-5 max output = 64k tokens** (context 200k). The news_screen 8192 cap is FAR below the model max -> confirms it is an artificial truncation bug; chunking (~32) is the robust fix; raising to <=64k is a secondary guard |
| 4 | https://eco3min.fr/en/net-liquidity-index-dataset/ | 2026-07-11 | industry (dataset, 2003-2026) | WebFetch | UNITS CONFIRMED: WALCL & WTREGEN = Millions USD (weekly); RRPONTSYD = Billions USD (convert to millions), daily. Recency: RRP buffer "fully depleted (~$0)" by Mar 2026 = regime shift (further QT now cuts liquidity directly) |
| 5 | https://www.themarketsunplugged.com/tga-fed-balance-sheet-net-liquidity/ | 2026-07-11 | industry | WebFetch | WALCL − WTREGEN − RRPONTSYD; TGA = "government's chequing account at the Fed"; rising WALCL + falling TGA/RRP → improved risk-asset liquidity conditions |

## Identified but snippet-only (context; does NOT count toward gate)
| URL | Kind | Why not fetched in full |
|-----|------|-------------------------|
| https://fred.stlouisfed.org/graph/?g=XsUr | official (FRED) | FRED's own net-liq graph: WALCL − (WTREGEN)*1000 − (RRP)*1000 — confirms the ×1000 unit scaling; WebFetch 403 (FRED blocks) |
| https://fred.stlouisfed.org/series/WALCL , /RRPONTSYD | official (FRED) | series units pages; WebFetch 403 — units instead confirmed via source #4; will re-verify per-series in code before computing |
| https://www.elastic.co/search-labs/blog/... (69.0) | official docs | multiplicative boost assumes positive base — sign-safe overlay basis, read in full in research_brief_69.0.md |
| https://www.lseg.com/.../multi-factor-indexes-power-of-tilting.pdf (69.0) | industry | factor tilts presume non-negative score domain — 69.0 |

---

## Internal code inventory (file:line anchors)

### TASK 1 — COMPLETE list of multiplicative-overlay-on-signed-score sites
All follow the identical pattern `return base_score * <mult>` where mult
is ~0.6..1.20. Sign-inversion bug: when base_score < 0, a boost (mult>1)
makes the score MORE negative (pushes candidate DOWN the ranking) and a
penalty (mult<1) makes it LESS negative (pushes candidate UP) — intent
inverted. Fix uniformly with 69.0 §3 form `score + abs(score)*(mult-1)`.
14 `apply_*_to_score` functions, all in `backend/services/`:

| # | File:line | Exact multiplicative expression | mult range |
|---|-----------|--------------------------------|-----------|
| 1 | news_screen.py:330 / :332 | `return base_score * 1.10` (pos) / `* 0.90` (neg) | 0.90 / 1.10 |
| 2 | macro_regime.py:542 | `score = base_score * regime.conviction_multiplier` | ~0.8..1.2 |
| 2b| macro_regime.py:547 / :549 | `score *= 1.05` (overweight) / `*= 0.95` (underwt) | 0.95 / 1.05 |
| 3 | pead_signal.py:387 | `base_score * (1.0 + min(max(surprise,0)*0.5, 0.3))` | 1.0..1.3 |
| 3b| pead_signal.py:389 | `base_score * max(1.0 + surprise*0.5, 0.6)` | 0.6..1.0 |
| 4 | analyst_revisions.py:187 | `base_score * (1.0 + sig.breadth * weight)` | ~0.9..1.1 |
| 5 | call_transcript_gpr.py:223 | `base_score * high_penalty` (default 0.97) | 0.97 |
| 6 | sector_momentum.py:200 | `base_score * entry.boost_multiplier` | >=1.0 |
| 7 | options_flow_screen.py:183 | `base_score * sig.boost_multiplier` | >=1.0 |
| 8 | analyst_narrative_scorer.py:242 | `base_score * sig.boost_multiplier` | ~0.9..1.1 |
| 9 | peer_leadlag_screen.py:137 | `base_score * sig.boost_multiplier` | >=1.0 |
| 10| insider_signal_screen.py:224 | `base_score * sig.boost_multiplier` | ~0.9..1.1 |
| 11| social_velocity_screen.py:175 | `base_score * sig.boost_multiplier` | >=1.0 |
| 12| ma_preannounce_screen.py:150 | `base_score * sig.boost_multiplier` | >=1.0 |
| 13| defense_signal.py:164 | `base_score * signal.boost_multiplier` | >=1.0 |
| 14| sector_calendars.py:321 / :323 | `base_score * 1.20` (FDA) / `* 1.10` (earnings) | 1.10 / 1.20 |

NOTE: two functions also FILTER (return None) — pead_signal.py:386
(`negative_surprise and surprise<-0.3`) and sector_calendars.py:317
(`binary_risk`). The None-filter branches are sign-independent (hard
exclude), so they need NO change — only the `* mult` branches do.
`apply_gpr_exposure_to_score` (#5) only ever penalizes (0.97), so the
sign bug there flips a haircut into a boost for negative scores.

**CALL SITE (single chain):** all 14 overlays are applied sequentially
to ONE `score` variable in `backend/tools/screener.py:318-411`, each
guarded by `if <signal>:`. So the overlays share a call site — a single
`sign_safe_overlays` flag can gate the WHOLE chain. Cleanest design:
add a shared helper (e.g. `sign_safe_mult(base, mult, enabled)` returning
`base + abs(base)*(mult-1)` when enabled else `base*mult`) and route all
14 functions through it, reading ONE settings flag
(`sign_safe_overlays: bool = False`) — default-OFF = byte-identical.

**ADJACENT inline sign-unsafe penalties (NOT `apply_*` fns) in the SAME
score chain, `backend/tools/screener.py`:**
| :306 | `score *= 0.7` (RSI>80 overbought) |
| :308 | `score *= 0.8` (RSI<20 oversold) |
| :311 | `score *= 0.85` (vol>0.6) |
These multiply the SIGNED momentum base (mom_1m*.4+mom_3m*.35+mom_6m*.25,
routinely negative) BEFORE the overlays. Same inversion class (a penalty
on a negative score moves it UP). They are part of base-score
CONSTRUCTION rather than a signal overlay, so scope them as OPTIONAL /
document explicitly — the operator may prefer to leave base-score
construction byte-identical and only fix the 14 `apply_*` overlays.
value_momentum base at :314 (`mom_3m*.5 - abs(sma_dist)*.2 + mom_1m*.3`)
is also signed. Recommend: fix the 14 `apply_*` fns under the flag;
LIST the 3 inline penalties for operator decision, do not silently change.

### TASK 2 — Net-liquidity FRED series + NEW cached-path insertion point
FRED series (confirm ids/units in external section below):
- **WALCL** = Fed total assets (H.4.1), Millions USD, WEEKLY (Wed level).
- **WTREGEN** = Treasury General Account, Millions USD, WEEKLY (Wed avg).
- **RRPONTSYD** = Overnight RRP (Fed-sold Treasuries), **Billions USD**, DAILY.
- Net liquidity = WALCL − WTREGEN − RRPONTSYD. **UNIT GOTCHA:** WALCL/WTREGEN
  are MILLIONS, RRPONTSYD is BILLIONS → must scale RRP ×1000 (to millions)
  before subtracting, else RRP term is ~1000x too small (silently ignored).

Current FRED wiring:
- `backend/tools/fred_data.py:16-26` `SERIES` dict = the ACTUAL fetch list
  (9 ids: FEDFUNDS, CPIAUCSL, UNRATE, GDP, T10Y2Y, UMCSENT, DGS10, VIXCLS,
  BAMLH0A0HYM2). `get_macro_indicators(api_key)` (:52) iterates it live, NO
  caching, returns `{series_id: {name,current,previous,trend,date}}`.
- `backend/services/macro_regime.py:41-44` `_REGIME_SERIES` = T10Y2Y, VIXCLS,
  BAMLH0A0HYM2, FEDFUNDS, CPIAUCSL, UNRATE, **INDPRO**.
- **LATENT BUG (do-no-harm-relevant):** INDPRO is in `_REGIME_SERIES` but NOT
  in `fred_data.SERIES` → `get_macro_indicators` never fetches it → INDPRO is
  ALWAYS absent from `indicators` → the regime prompt's INDPRO line (:358-366)
  is always skipped and `available` (:423) never counts it. INDPRO has been
  dead since it was added. 69.0 §4 flagged INDPRO; the actual fix is one line:
  add `"INDPRO": "Industrial Production Index"` to `fred_data.SERIES`.

NEW cached-path design (mirrors EXISTING idioms — no new infra):
- historical_macro is written ONLY by `backend/backtest/data_ingestion.py:272/286`
  (the FROZEN ingestion). `fred_data.py` + `macro_regime.py` do NOT touch BQ at
  all (grep clean) → any live-fetch + file-cache path is inherently safe re: the
  frozen table. CONFIRM: net-liq path must live in fred_data/macro_regime, NEVER
  in data_ingestion.py.
- EXISTING 24h file-cache idiom to COPY: `macro_regime._fetch_gpr_acts` (:111-155,
  `_GPR_CACHE_PATH` + `st_mtime` age check `age < timedelta(hours=cache_hours)`)
  and `_fetch_crude_momentum` (:196+, `_CRUDE_CACHE_PATH`). Both cache raw FRED-ish
  data to a file in `_CACHE_DIR` with mtime-based 24h freshness. The regime OUTPUT
  itself is already 24h-cached via `_CACHE_PATH`/`_CACHE_TTL_HOURS=24` (:30-31).
- RECOMMENDED insertion: add `_fetch_net_liquidity(fred_key, cache_hours=24)` in
  macro_regime.py (or fred_data.py) that fetches WALCL/WTREGEN/RRPONTSYD via the
  existing `_fetch_series` helper (fred_data.py:29, already takes series_id+key),
  computes `net_liq = WALCL - WTREGEN - RRPONTSYD*1000` (millions) + a trend, and
  caches to a NEW `_NETLIQ_CACHE_PATH` JSON with mtime 24h freshness. Feed net_liq
  (level + trend) into `_build_prompt` (:350) as an extra indicator line + a
  threshold rule (rising net-liq → risk_on lean). Add INDPRO to fred_data.SERIES
  in the same change so the existing _REGIME_SERIES entry finally activates.
- All of this is behind the SAME `sign_safe_overlays`-class flag OR its own
  `regime_net_liquidity: bool=False` flag (default-OFF) so live ranking stays
  byte-identical until the operator flips it. Uses the existing FRED key
  (`settings.fred_api_key`). Writes NO BQ table.

### TASK 3 — news_screen.py:282 cap (chunk vs raise)
- `backend/services/news_screen.py:282`: `"max_output_tokens": min(8192, 250*len(deduped))`.
  For len(deduped) >= 33, `250*N > 8192` so the cap FREEZES at 8192 regardless of
  batch size. A large news day (e.g. 60 headlines needs ~15K output tokens) is
  truncated at 8192 → `json.loads(response.text)` raises (:293) → `except` →
  `return {}` (:298-299) → ALL news signals lost for that cycle. For N<=32 the cap
  = 250*N (never binds) so the common case is unaffected.
- claude-haiku-4-5 max output tokens: CONFIRM in external section (Anthropic docs).
- **RECOMMENDATION: CHUNK (primary) + parse-fail retry, keep common-case single call.**
  Chunk `deduped` into batches of ~32 (fits the existing 8192 sweet spot), call
  once per chunk via `asyncio.gather`, merge with the existing first-seen-wins
  dedupe (:301-311). Rationale over raise: (a) a single mega-call is all-or-nothing
  — one truncation loses everything; chunking isolates failures; (b) $0-metered /
  do-no-harm: chunk-only-when-N>32 keeps N<=32 as a byte-identical single call, so
  no added spend on normal days; (c) haiku output even at 64K can still be exceeded
  by a very large batch, so raising the cap alone is not robust. ADD a parse-fail
  retry (retry the failing chunk once) as belt-and-suspenders. Raising the cap to
  the model max is an acceptable SECONDARY guard but not a substitute for chunking.

### TASK 4 — QMJ Growth ordering bug (backtest/historical_data.py)
- Path is `backend/backtest/historical_data.py` (NOT backend/tools/). Bug:
  - :202 `rev_growth = features.get("revenue_growth_yoy")` — READ
  - :219-222 `if rev_growth is not None:` Growth sub-score appended to quality
  - :247-248 `features["quality_score"] = sum(quality_components)/len(...)`
  - :252-254 `features["revenue_growth_yoy"] = self._compute_revenue_growth_yoy(
    fundamentals_list, revenue)` — ASSIGNED (51 lines AFTER the read)
  → `features.get("revenue_growth_yoy")` at :202 is ALWAYS None → the Growth
  dimension of the Asness-Frazzini-Pedersen QMJ quality_score is DEAD; quality is
  computed from only 3 of 4 dimensions (Profitability, Safety, Payout).
- **Minimal reorder fix:** move the :252-254 block (comment + `revenue_growth_yoy`
  assignment) to BEFORE :202 (e.g. just above `roe_val = features.get("roe")` :199).
  `_compute_revenue_growth_yoy(fundamentals_list, current_revenue)` (:348) depends
  ONLY on `fundamentals_list` + `revenue`, both in scope well before :202 (revenue
  used at :166); nothing between :202-253 mutates either → the move is safe, no
  forward dependency. This CHANGES backtest quality_score values → NOT byte-identical
  to prior backtests, so treat as flag-gated / behind the historical_macro-unfreeze
  IC/ablation validation, OR document as a bugfix the operator accepts. It does NOT
  touch live ranking (screener uses its own momentum score, not this QMJ feature).

---

## Recency scan (2024-2026)

Performed. Net-liquidity (WALCL−WTREGEN−RRPONTSYD) as an equity-liquidity proxy is current, consensus
practice (netliquidity.org, macrolighthouse, eco3min all current). KEY 2026 finding (eco3min): the ON RRP
buffer is now ~$0 (Mar 2026) after absorbing $2.37T of QT — a regime shift where further balance-sheet
reduction cuts system liquidity directly. This makes net-liquidity MORE informative as a regime input now
than in 2022-2024, supporting the 69.3 lift. Sign-safe overlay algebra + INDPRO regime role were established
in research_brief_69.0.md (no reversal in the last 2 years). Claude Haiku 4.5 max output = 64k (current docs).

## Findings (highest-value, for the design)

- **Sign-safe overlays**: 14 `apply_*_to_score` functions + macro_regime, ALL `base_score * mult`, applied at
  ONE call chain (`backend/tools/screener.py:318-411`). Fix uniformly with a shared flag-gated helper
  `sign_safe_mult(base, mult, enabled)` (`base + abs(base)*(mult-1)` ON / `base*mult` OFF), one settings flag
  `sign_safe_overlays: bool=False` (default-OFF = byte-identical). The 3 inline base-score penalties
  (screener:306/308/311) are base-CONSTRUCTION, not overlays → LIST for operator, do not silently change.
- **INDPRO**: one-line fix — it's in `_REGIME_SERIES` (macro_regime.py:41) but MISSING from
  `fred_data.SERIES` (the actual fetch list) → never fetched → dead. Add `"INDPRO"` to `fred_data.SERIES`.
- **Net-liquidity**: `net_liq = WALCL − WTREGEN − RRPONTSYD*1000` (all millions; RRPONTSYD is billions →
  ×1000 — a silent-corruption trap if missed). New `_fetch_net_liquidity` mirroring the EXISTING 24h
  file-cache idiom (`_fetch_gpr_acts`/`_fetch_crude_momentum`); fred_data/macro_regime touch NO BQ, so the
  path is inherently safe re: the frozen historical_macro (written only by data_ingestion.py). Behind a
  `regime_net_liquidity: bool=False` flag (default-OFF).
- **News cap**: chunk `deduped` into ~32-headline batches (fits the 8192 sweet spot; N≤32 stays a
  byte-identical single call) + parse-fail retry. Haiku max is 64k so raising the cap is a valid secondary
  guard, but chunking isolates failures (one truncation ≠ lose everything).
- **QMJ**: `revenue_growth_yoy` read at historical_data.py:202, assigned at :252 (51 lines later) → always
  None → Growth dimension dead. Move the :252-254 assignment before :202 (deps in scope earlier; safe).
  Changes backtest quality_score → flag-gate / behind the freeze-unlock IC validation; does NOT touch live ranking.

## Application to pyfinagent (69.3 GENERATE)

Flag-gated, default-OFF (do-no-harm live byte-identity), historical_macro untouched, final IC/ablation deferred:
1. Shared `sign_safe_mult` helper + `sign_safe_overlays` flag; route the 14 `apply_*_to_score` fns through it.
2. news_screen chunk(~32) + parse-retry (news_screen.py:282-299).
3. INDPRO one-liner into fred_data.SERIES + QMJ reorder (historical_data.py).
4. `_fetch_net_liquidity` cached path + regime-prompt line, behind `regime_net_liquidity` flag.
Unit tests: sign-inversion (negative-base boost ranks above negative-base penalty), news 100-headline parse,
QMJ Growth fires, net-liq unit scaling (RRP ×1000). Live $0 checks: ON-vs-OFF ranking on real screen data;
regime-prompt STRING render showing INDPRO + net-liquidity (render the prompt, do NOT call the metered LLM).

## Research Gate Checklist
- [x] >=5 authoritative external sources READ IN FULL (5: netliquidity.org, macrolighthouse, Anthropic docs, eco3min, themarketsunplugged) + 69.0 overlay/INDPRO sources
- [x] 10+ unique URLs total (5 in full + snippet/FRED + 69.0 carryover)
- [x] Recency scan (last 2 years) performed + reported
- [x] Full pages read (not abstracts)
- [x] file:line anchors for every internal claim (14 overlay sites + INDPRO + net-liq + news + QMJ)
- [x] Every overlay site enumerated (14 + the single call chain located)
- [x] Net-liquidity FRED series + new-cached-path insertion point located (units resolved)

## Provenance note
Internal exploration (the complete overlay-site map, INDPRO one-line pin, net-liquidity design + unit gotcha,
news-cap + QMJ analysis) was authored by the researcher subagent before it STALLED on the external half (8th
subagent stall; kill message "Internal exploration complete. Now external research"). Main read the 5 external
sources above + finalized the envelope. Internal map is the researcher's; external floor + envelope are Main's
completion (the documented "Main updates the stalled handoff file" pattern). Every claim traces to a source row
or a file:line.

## JSON envelope

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 5,
  "snippet_only_sources": 4,
  "urls_collected": 13,
  "recency_scan_performed": true,
  "internal_files_inspected": 16,
  "overlay_sites_enumerated": 14,
  "gate_passed": true
}
```

---
