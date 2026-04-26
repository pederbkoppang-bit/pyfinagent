# Research Brief — phase-5.6: Options Integration (Black-Scholes Greeks + options_ingestion --dry-run)

Tier: moderate (assumed — caller did not specify)
Date: 2026-04-26

---

## Search queries run (3-variant discipline)

1. **Current-year frontier**: "Black-Scholes greeks Python implementation 2026"
2. **Last-2-year window**: "options greeks scipy norm cdf Python 2025", "OCC option symbol format 2025"
3. **Year-less canonical**: "Black-Scholes formula delta gamma theta vega derivation", "OCC option symbology", "argparse dry-run pattern Python"

---

## Read in full (>=5 required; counts toward the gate)

| URL | Accessed | Kind | Fetched how | Key quote or finding |
|-----|----------|------|------------|----------------------|
| https://en.wikipedia.org/wiki/Greeks_(finance) | 2026-04-26 | reference doc | WebFetch | Full analytic formulas for delta/gamma/theta/vega for calls and puts under GBM; confirms q=0 convention and sign of theta |
| https://www.investopedia.com/terms/b/blackscholes.asp | 2026-04-26 | reference doc | WebFetch | d1, d2 definitions; call/put price; confirms delta_call = N(d1), delta_put = N(d1)-1 |
| https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html | 2026-04-26 | official docs | WebFetch | norm.cdf(x) and norm.pdf(x) usage; confirms scipy.stats is the correct import path; already in backend/requirements.txt as scipy>=1.12.0 |
| https://www.theocc.com/Company-Information/Documents-and-Archives/Options-Disclosure-Document | 2026-04-26 | official doc | WebFetch | OCC standard option symbol format: 21-char root+date+type+strike |
| https://realpython.com/python-command-line-arguments/#the-argparse-module | 2026-04-26 | authoritative blog/doc | WebFetch | argparse --underlyings nargs='+' pattern, --dry-run store_true; standard dry-run idiom |
| https://www.macroption.com/black-scholes-formula/ | 2026-04-26 | authoritative finance reference | WebFetch | Full worked numeric example confirming ATM call delta ~0.54 for short-dated options; vega per 1% vol move formula |

---

## Identified but snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|-----|------|------------------------|
| https://arxiv.org/abs/1906.04203 | paper (deep hedging) | Not directly relevant to analytic greek computation |
| https://quantlib.org/reference/ | official docs | Full QL is overkill; analytic BS formulas are self-contained |
| https://yfinance.readthedocs.io/en/latest/ | official docs | yfinance already used in options_flow.py — no new pattern needed |
| https://pandas.pydata.org/docs/ | official docs | No pandas-specific greek pattern needed |
| https://stackoverflow.com/q/36762764 | community | scipy norm CDF usage — covered by official scipy docs |

---

## Recency scan (2024-2026)

Searched for 2025 and 2026 literature on: Black-Scholes Python greeks, OCC symbology changes, scipy.stats options pricing.

Result: No substantive new findings in the 2024-2026 window that supersede the canonical Black-Scholes analytic formulas. The OCC 21-character option symbol standard (introduced 2010) remains unchanged. scipy.stats.norm.cdf is stable API since scipy 0.14; scipy>=1.12.0 (already pinned in backend/requirements.txt) is current. One 2025 blog post (macroption.com) confirms the q=0 simplified form is still the industry standard for equity options without dividends.

---

## Key findings

### 1. Black-Scholes formulas — exact (no dividend, q=0)

Let:
```
d1 = (ln(S/K) + (r + 0.5*sigma^2)*T) / (sigma * sqrt(T))
d2 = d1 - sigma * sqrt(T)
```

**Delta:**
- Call: delta_c = N(d1)
- Put:  delta_p = N(d1) - 1  (equivalently: -N(-d1))

**Gamma** (same for call and put):
- gamma = N'(d1) / (S * sigma * sqrt(T))
- where N'(x) = norm.pdf(x) = (1/sqrt(2*pi)) * exp(-x^2/2)

**Theta** (annualized, divide by 365 for per-calendar-day):
- Theta_c = -(S * N'(d1) * sigma) / (2*sqrt(T))  -  r*K*exp(-r*T)*N(d2)
- Theta_p = -(S * N'(d1) * sigma) / (2*sqrt(T))  +  r*K*exp(-r*T)*N(-d2)

**Vega** (same for call and put; expressed per 1% change in sigma):
- vega = S * N'(d1) * sqrt(T) / 100

(Source: Wikipedia Greeks (finance), Investopedia Black-Scholes, macroption.com)

### 2. ATM 30-DTE call delta — verified numeric

Inputs: S=450, K=450, T=30/365=0.08219, r=0.05, sigma=0.20

Computed locally with `scipy.stats.norm` in the project venv:
- d1 = 0.100342
- d2 = 0.043004
- delta_call = N(d1) = **0.5400** (within [0.45, 0.55] as required)
- delta_put  = N(d1)-1 = -0.4600
- gamma = 0.015384
- theta_call per calendar day = -0.2024
- vega per 1% vol move = 0.5121

(Source: computed in-session via `source .venv/bin/activate && python3 -c "from scipy.stats import norm; ..."`)

### 3. scipy.stats.norm usage

```python
from scipy.stats import norm

# d1, d2 computed as above
delta_call = norm.cdf(d1)           # N(d1)
delta_put  = norm.cdf(d1) - 1       # N(d1) - 1
gamma      = norm.pdf(d1) / (S * sigma * math.sqrt(T))
vega       = S * norm.pdf(d1) * math.sqrt(T) / 100
```

scipy is already declared in `backend/requirements.txt` at `scipy>=1.12.0`.
No new dependency needed.

(Source: scipy official docs, backend/requirements.txt line grep)

### 4. OCC option symbol format

Standard 21-character OSI (Options Symbology Initiative) format:
```
<root ticker padded to 6 chars><YY><MM><DD><C|P><strike * 1000 padded to 8 digits>
```
Example:
```
AAPL  260117C00150000
 ^6    ^2^2^2^1 ^8
```
- Root: left-justified, space-padded to 6 chars (AAPL -> "AAPL  ")
- Date: YYMMDD (2-digit year, 2-digit month, 2-digit day)
- Type: C = call, P = put
- Strike: integer(strike * 1000) zero-padded to 8 digits (150.00 -> 00150000)

Full 21-char string: "AAPL  260117C00150000"

(Source: OCC official documentation, theocc.com)

### 5. Ingestion script --dry-run shape

Canonical pattern (from `scripts/migrations/add_news_sentiment_schema.py`, line 86-105):
```python
import argparse, sys
from pathlib import Path

def main(dry_run: bool, underlyings: list[str]) -> int:
    # ... build rows ...
    if dry_run:
        print(f"dry-run: would insert {len(rows)} rows into options_greeks table")
        for row in rows[:3]:
            print(row)
        return 0
    # ... BQ insert ...
    return 0

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Ingest options greeks into BigQuery")
    ap.add_argument("--underlyings", nargs="+", default=["SPY", "QQQ", "AAPL"],
                    help="Ticker symbols to ingest")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print rows without writing to BigQuery")
    args = ap.parse_args()
    raise SystemExit(main(dry_run=args.dry_run, underlyings=args.underlyings))
```

With `--dry-run`, the script must print to stdout and exit 0 without any BQ writes.

(Source: scripts/migrations/add_news_sentiment_schema.py lines 86-105, dry-run pattern)

### 6. Migration script template location

Canonical template location: `/Users/ford/.openclaw/workspace/pyfinagent/scripts/migrations/`

Best reference template for a new BQ table: `scripts/migrations/add_news_sentiment_schema.py`
- Pattern: DDL string with `{project}.{dataset}.{table}`, formatted with settings
- `--dry-run` flag prints DDL without executing
- Uses `backend.config.settings.get_settings()` for project/dataset resolution
- Idempotent: `CREATE TABLE IF NOT EXISTS`
- `bigquery.Client(project=project)` with ADC (no SA credentials file needed locally)

(Source: scripts/migrations/add_news_sentiment_schema.py read in full)

### 7. Test plan (8-12 tests)

```
test_bs_greeks.py (unit tests, no BQ):
  1. test_delta_call_atm_30dte         -- delta_call in [0.45, 0.55] for S=K=450, T=30/365
  2. test_delta_put_atm_30dte          -- delta_put in [-0.55, -0.45]
  3. test_delta_put_call_parity        -- delta_call + abs(delta_put) == 1.0 (within 1e-9)
  4. test_gamma_positive               -- gamma > 0 for any valid inputs
  5. test_vega_positive                -- vega > 0 for any valid inputs
  6. test_theta_negative_call          -- theta_call < 0 (time decay)
  7. test_occ_symbol_format_call       -- AAPL 260117 C 00150000 -> "AAPL  260117C00150000" (21 chars)
  8. test_occ_symbol_format_put        -- same for P
  9. test_ingestion_dry_run_no_bq      -- subprocess.run(["python", "scripts/options_ingestion.py",
                                           "--dry-run", "--underlyings", "SPY"],
                                           check=True) exits 0 with "dry-run" in stdout
  10. test_ingestion_dry_run_output_shape -- stdout from dry-run contains ticker, strike, expiry,
                                             delta, gamma, theta, vega fields
  11. test_deep_itm_call_delta_near_1  -- S=500, K=300, T=1.0 -> delta_call > 0.95
  12. test_deep_otm_call_delta_near_0  -- S=300, K=500, T=0.25 -> delta_call < 0.05
```

---

## Internal code inventory

| File | Lines | Role | Status |
|------|-------|------|--------|
| `backend/tools/options_flow.py` | 113 | yfinance options chain P/C analysis; NO greeks | Active, no BS math |
| `backend/agents/skills/options_agent.md` | 76 | Gemini prompt for options flow signal | Active; no greek computation |
| `backend/requirements.txt` | ~40 | scipy>=1.12.0 declared | scipy confirmed present |
| `scripts/migrations/add_news_sentiment_schema.py` | ~105 | Canonical migration template | Active; best template |
| `scripts/migrations/migrate_bq_schema.py` | 188 | BQ schema migration pattern (older style) | Active; secondary pattern |
| `scripts/debug/debug_ingestion.py` | 60+ | Ticket ingestion debug, not options | Unrelated |
| `functions/ingestion/main.py` | 79 | Cloud Function market data EL — not options | Unrelated |

No existing Black-Scholes implementation found in the codebase. No `options_greeks` table exists. No `options_ingestion.py` script exists. All three must be created from scratch.

No duplicate or dead code for options greeks. `options_flow.py` uses yfinance chain data for P/C signal only — greeks computation is additive, not a replacement.

---

## Consensus vs debate

Consensus: Black-Scholes analytic greeks (delta/gamma/theta/vega) with q=0 are the standard first-order approximation for equity options without dividends. No debate among canonical sources on the formulas themselves. Some debate exists on whether to use calendar days vs trading days for theta — calendar-day convention (divide annualized theta by 365) is the market standard for reporting.

---

## Pitfalls

1. **Theta sign convention**: some textbooks express theta as a positive value (absolute decay per day); the BS formula as written above yields a negative theta_call. Be explicit about the sign in the returned dict.
2. **T=0 division**: when T approaches 0, d1/d2 blow up. Guard with `max(T, 1e-9)`.
3. **Vega reporting**: raw vega from the formula is per unit change in sigma (not per 1%). Divide by 100 to get "vega per 1% vol move" — the trader-standard unit. Mismatching these produces a 100x error.
4. **OCC symbol zero-padding**: strike=150.0 -> integer(150.0*1000)=150000 -> zero-pad to 8 digits -> "00150000". Fractional strikes (e.g. 147.5) work: integer(147.5*1000)=147500 -> "00147500". Always use `int(round(strike * 1000))`.
5. **scipy import**: `from scipy.stats import norm` — not `scipy.special`. norm.cdf and norm.pdf are the correct functions.
6. **dry-run must exit 0**: test #9 calls `check=True` on subprocess; a non-zero exit code will fail the test even if dry-run output looks correct.
7. **OCC root field**: ticker must be left-justified and space-padded to exactly 6 characters. Using `ticker.ljust(6)` is correct; `ticker.rjust(6)` is wrong.

---

## Application to pyfinagent (file:line anchors)

- `backend/tools/options_flow.py:1-13` — import section to reference for style; the new `bs_greeks.py` should live at `backend/tools/bs_greeks.py` alongside it.
- `backend/requirements.txt` — `scipy>=1.12.0` already present; no edit needed.
- `scripts/migrations/add_news_sentiment_schema.py:1-105` — use as line-for-line template for `scripts/migrations/create_options_greeks_schema.py`.
- `scripts/migrations/add_news_sentiment_schema.py:86-105` (argparse + dry-run block) — copy verbatim pattern for `scripts/options_ingestion.py`.
- `backend/agents/skills/options_agent.md:1-8` — new greeks values can be plumbed into this agent's data context in a future step; phase-5.6 scope is computation + ingestion only.

---

## Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (6 sources)
- [x] 10+ unique URLs total (incl. snippet-only) (11 URLs)
- [x] Recency scan (last 2 years) performed + reported
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every relevant module
- [x] No contradictions in canonical sources; consensus noted
- [x] All claims cited per-claim

---

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 6,
  "snippet_only_sources": 5,
  "urls_collected": 11,
  "recency_scan_performed": true,
  "internal_files_inspected": 7,
  "report_md": "handoff/current/phase-5.6-research-brief.md",
  "gate_passed": true
}
```
