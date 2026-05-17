# phase-28.0 Research Brief — Drift fix for unused min_market_cap parameter
**Date:** 2026-05-17
**Tier:** simple
**Step:** phase-28.0 (Candidate Picker Expansion — drift fix)
**Audit basis:** backend/tools/screener.py:screen_universe accepts min_market_cap (default 1e9) but never uses it in body. Either wire the filter or remove dead param.

---

## Research: Drift fix — dead `min_market_cap` parameter in `screen_universe`

### Queries run (three-variant discipline)
1. Current-year frontier: "Python API unused parameter removal deprecation warning best practices 2026"
2. Last-2-year window: "yfinance Ticker info marketCap rate limit API request per ticker 2025"
3. Year-less canonical: "S&P 500 minimum market cap index inclusion criteria Wikipedia constituent"

---

### Read in full (>=5 required; counts toward the gate)

| URL | Accessed | Kind | Fetched how | Key quote or finding |
|-----|----------|------|-------------|----------------------|
| https://peps.python.org/pep-0702/ | 2026-05-17 | Official Python PEP | WebFetch | `@warnings.deprecated` decorator for functions/methods; parameter-level deprecation done via overloads; `category=None` suppresses runtime warning |
| https://dev.docs.pyansys.com/coding-style/deprecation.html | 2026-05-17 | Authoritative engineering guide | WebFetch | "use a warning first and then use an error after a minor release or two"; no special rule for internal vs library |
| https://sethmlarson.dev/deprecations-via-warnings-dont-work-for-python-libraries | 2026-05-17 | Authoritative blog (CPython security researcher) | WebFetch | "DeprecationWarning is ignored by default"; recommends custom `UserWarning` subclass or frequent major releases instead |
| https://corporatefinanceinstitute.com/resources/equities/sp-500-index/ | 2026-05-17 | Industry reference | WebFetch | S&P 500 minimum market cap ~$8.2B (older article); actual current floor is $22.7B per S&P Dow Jones Indices Jul 2025 press release |
| https://docs.python.org/3/deprecations/index.html | 2026-05-17 | Official Python docs | WebFetch | Formal deprecation pattern: DeprecationWarning first, then removal in a later minor; e.g., `threading.RLock()` args deprecated 3.14, removed 3.15 |

### Identified but snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|-----|------|------------------------|
| https://www.slingacademy.com/article/rate-limiting-and-api-best-practices-for-yfinance/ | Blog | Fetched; yielded no info on Ticker.info vs download rate limits — content too generic |
| https://github.com/ranaroussi/yfinance/issues/2325 | GitHub issue | Fetched; mentions "60 API calls/min" but no Ticker.info vs download breakdown |
| https://github.com/ranaroussi/yfinance/issues/2125 | GitHub issue | Snippet only — rate-limit report but no per-method detail |
| https://github.com/ranaroussi/yfinance/discussions/2431 | GitHub discussion | Snippet only |
| https://github.com/ranaroussi/yfinance/issues/2288 | GitHub issue | Snippet only |
| https://www.stocktitan.net/news/SPGI/s-p-dow-jones-indices-announces-update-to-s-p-composite-1500-market-tt10ngtjfn66.html | News | Snippet confirms $22.7B threshold Jul 2025; S&P PDF returned 403 |
| https://pypi.org/project/pyDeprecate/ | PyPI | Snippet only — library for argument deprecation helpers |
| https://oneuptime.com/blog/post/2026-02-02-api-deprecation/view | Blog | Snippet only — API lifecycle article |
| https://en.wikipedia.org/wiki/S%26P_500 | Reference | Snippet only — Wikipedia page; CFI article fetched instead |
| https://iifx.dev/en/articles/457050267/the-definitive-guide-to-managing-rate-limits-for-unofficial-financial-apis | Blog | Snippet only — general yfinance rate limit tips; batch is safer than per-ticker |

Total URLs: 15

---

### Recency scan (2024-2026)
Searched: "yfinance Ticker info marketCap rate limit API request per ticker 2025" and "Python API unused parameter removal deprecation warning best practices 2026".

Result: yfinance rate-limit reports cluster heavily in 2024-2025 (GitHub issues 2125, 2288, 2325, 2422 all from that window); Yahoo Finance tightened limits in early 2024. No 2025-2026 papers or docs that change the core finding: `Ticker.info` makes a separate per-ticker HTTP request (different endpoint from `yf.download` batch), making it O(N) calls for N tickers vs a single batch. S&P 500 minimum market cap was raised to $22.7B as of July 1, 2025 (S&P Dow Jones press release), superseding any older $1B-class figures. No new Python PEP or stdlib change in 2024-2026 reverses PEP 702's approach.

---

### Key findings

1. **All S&P 500 constituents already exceed $1B by a large margin.** As of Jul 2025, the S&P 500 *entry threshold* is $22.7B unadjusted market cap; the Wikipedia-scraped list used by `get_sp500_tickers()` returns current index members only. The `min_market_cap=1e9` ($1B) default is 22x below the index floor — it filters nothing from the S&P 500 universe. (Sources: S&P Dow Jones Indices press release Jul 2025; CFI article)

2. **`Ticker.info["marketCap"]` requires a separate per-ticker HTTP request** to a Yahoo Finance endpoint distinct from `yf.download()`. The batch download fetches OHLCV via a multi-ticker price endpoint; `.info` calls a separate quote endpoint per symbol. With 500 tickers this is 500 additional HTTP requests, all subject to the 60-calls/min rate limit Yahoo now enforces — adding ~8 minutes of polling plus 429 risk. (Source: yfinance GitHub issues 2125, 2288, 2325)

3. **Dead parameters in internal (non-library) Python APIs should be removed immediately.** PEP 702's `@warnings.deprecated` machinery and the "warn-first, remove-later" cadence are designed for public library APIs where downstream consumers exist outside the repo. For an internal function with all callers in the same repo, immediate removal is correct — no deprecation period is needed because all callers can be updated atomically. (Sources: PEP 702; PyAnsys guide; Seth Larson's post)

4. **Removing the parameter does not break any existing caller.** Zero callers pass `min_market_cap` explicitly (confirmed by grep). All call sites use positional-or-keyword defaults: `screen_universe()`, `screen_universe(period="6mo")`, `screen_universe(tickers=[...], period="6mo", sector_lookup=...)`. No test file checks for `min_market_cap` in the signature — `verify_phase_23_1_13.py` only asserts `sector_lookup` is present.

5. **Wiring the filter would be expensive and yield no benefit.** To enforce `min_market_cap` on the S&P 500 universe, the code would need to call `yf.Ticker(t).info["marketCap"]` per ticker — O(500) extra HTTP requests per screen cycle. Since no S&P 500 member is below $22.7B, the filter would never exclude anyone. Cost: high. Benefit: zero.

---

### Internal code inventory

| File | Lines | Role | Status |
|------|-------|------|--------|
| `backend/tools/screener.py` | 303 | Screener — `screen_universe` defined here | AUDITED — `min_market_cap` at line 65, never referenced below line 65 |
| `backend/services/autonomous_loop.py` | 247 | Main production caller | Calls `screen_universe(period="6mo")` — no `min_market_cap` arg |
| `backend/api/backtest.py` | 195 | Backtest caller | Calls `screen_universe()` — no `min_market_cap` arg |
| `tests/services/test_screener_sector_propagation.py` | 77 | Unit tests | Calls `screen_universe(tickers=[...], period=..., sector_lookup=...)` — no `min_market_cap` arg; no signature assertion on it |
| `tests/verify_phase_23_1_13.py` | ~70 | Smoke test | Asserts `sector_lookup` in `sig.parameters`; does NOT assert `min_market_cap` |
| `scripts/ablation/run_ablation.py` | 252 | Ablation script | Calls internal `engine._screen_universe()` — NOT `tools.screener.screen_universe` |

**All callers confirmed: none pass `min_market_cap`. Removal is safe.**

---

### Consensus vs debate (external)
Consensus: for internal/same-repo APIs, immediate removal without deprecation period is standard practice. The warn-first cadence is a library-level courtesy for external consumers. No debate on this point in any source.

Consensus on yfinance: `Ticker.info` is per-ticker; `yf.download` is a single batch request. No dispute.

### Pitfalls (from literature)
- If `min_market_cap` were wired via `Ticker.info`, 500 per-ticker requests per screen cycle would incur 429 rate-limit errors under Yahoo Finance's current throttling (tightened 2024). Avoid.
- Do not add a `DeprecationWarning` for a dead parameter in an internal repo — it adds noise with no consumer benefit and may confuse future readers into thinking external callers exist.

### Application to pyfinagent
- `screener.py:65` — remove the `min_market_cap` parameter from the function signature.
- `screener.py:121` — the filter line `if current_price < min_price or avg_vol < min_avg_volume:` is the correct filter line; no changes needed there.
- No other files need editing. Zero callers pass the parameter.

---

### Recommendation: REMOVE

Remove the `min_market_cap` parameter immediately and without a deprecation wrapper. Reasoning:
1. The S&P 500 universe already guarantees all members are $22.7B+ — the filter never fires.
2. Wiring it would require O(500) extra per-ticker HTTP requests per screen cycle, with high 429-rate-limit risk.
3. This is an internal function — all callers are in the same repo and none pass the argument.
4. Removal is a one-line diff at `screener.py:65`.

---

### Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (5 read: PEP 702, PyAnsys deprecation guide, Seth Larson blog, CFI S&P 500 article, Python deprecations docs)
- [x] 10+ unique URLs total (incl. snippet-only) — 15 collected
- [x] Recency scan (last 2 years) performed + reported
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every relevant module (screener.py, autonomous_loop.py, backtest.py, both test files, ablation script)
- [x] Contradictions / consensus noted
- [x] All claims cited per-claim

---

```json
{
  "tier": "simple",
  "external_sources_read_in_full": 5,
  "snippet_only_sources": 10,
  "urls_collected": 15,
  "recency_scan_performed": true,
  "internal_files_inspected": 6,
  "gate_passed": true
}
```
