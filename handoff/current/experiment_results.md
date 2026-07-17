# Experiment results — step 65.3 (US+KR per-market health baseline)

**Step:** 65.3 (P1, phase-65, depends_on none; post-66.2=done). $0 BQ read-only baseline audit; live book untouched;
historical_macro FROZEN; NO trade/risk/money touch. Research gate PASSED (research_brief_65.3.md, gate_passed=true, 7
external sources read in full).

## What was done

Ran the 4 per-market aggregate BQ queries (read-only, $0) against `financial_reports.paper_trades` for trades since
2026-06-01, market derived from the ticker suffix (no `market` column; `created_at` is STRING → lexical filter), plus
the pre/post-61.1-churn-fix split. Wrote the baseline doc.

Deliverable: **`handoff/away_ops/market_health_baseline.md`** — per-market aggregate tables + **the verbatim SQL
pasted** (criterion 1) + **10 `HEALTHY-THRESHOLD:` lines** (criterion 2) + the **pre/post-61.1-fix split noted
separately** (criterion 3) + the low-n descriptive caveat.

## Findings (baseline)

- **US:** 11 buys / 17 sells / **70.6% win** / $20.30 fees (0.085% NAV) / median hold **3d**. Holding dist:
  6×≤1d, 4×2-5d, 1×6-20d, 6×≥21d. Exits: swap n=10 (avg 4.7d, +11.95%), stop_loss n=7 (avg 29.3d, +32.19%).
- **KR:** 5 buys / 5 sells / **20.0% win** / $4.82 fees (0.020% NAV) / median hold **1d**. Holding dist: 4×≤1d,
  1×≥21d. Exits: stop_loss n=3 (avg 8.0d, -0.40%), swap n=2 (avg 0.5d, -3.26%).
- **EU:** 0 trades yet (zero-trades diagnosis is 65.1/65.2, separate).
- **Churn split (criterion 3):** `paper_swap_churn_fix_enabled` ON 2026-06-12. **PRE-FIX (06-01→06-11)** carries the
  whole swap-churn cluster (US 10 + KR 2 = 12 swap-exits, many ≤1d holds). **POST-FIX (06-12+)** has **0 swap-exits**
  (fix holds) but a thin sample (US 2 sells, KR 1) confounded by the away-ops quiet period → directionally confirmed,
  trend PENDING more cycles. Presented separately, never merged.
- **Low-n honesty:** US 17 / KR 5 closed trades < the ~30/metric inferential minimum → win-rate/PF are DESCRIPTIVE;
  65.4 should lean on the STRUCTURAL thresholds (holding-days, churn-swap-hold, fee-drag, liveness).

## Verification (verbatim)

- IMMUTABLE cmd `test -f handoff/away_ops/market_health_baseline.md && grep -c 'HEALTHY-THRESHOLD' ...` → **10**,
  exit **0**.
- Criterion 1: 4 verbatim SQL SELECT blocks pasted (the market-suffix CASE + `created_at >= '2026-06-01'` filter +
  the 4 aggregate queries), matching the run outputs above.
- Criterion 2: 10 explicit `HEALTHY-THRESHOLD:` lines (structural primary + secondary-descriptive-until-n≥30) with
  the concrete X/Y/Z + current PASS/FAIL annotations (e.g. "no market > 0.50% of NAV in fees" = $119.37; "≤1d exits <
  40%"; "swap-exit avg hold ≥ 3d"; "median holding_days ≥ 5").
- Criterion 3: the PRE-FIX vs POST-FIX split table (8 rows) + the "noted separately, never merged" narrative.
- $0: read-only BQ SELECT + Python only. NO metered LLM. NO production code changed (git = only the baseline doc +
  handoff docs).

## Do-no-harm / boundaries

$0 read-only BQ. READ-ONLY baseline audit — the only deliverable is `market_health_baseline.md`. NO production code
change; NO trade/risk/money touch; kill-switch/stops/caps/DSR/PBO untouched; historical_macro FROZEN; live book
untouched. The HEALTHY-THRESHOLD lines are BASELINE targets for 65.4 (not enforced live). Scope honesty: git may also
show incidental live autonomous-loop runtime artifacts (the :8000 backend) — runtime state, not 65.3.

## Artifact shape
`handoff/away_ops/market_health_baseline.md` (per-market aggregate tables + verbatim SQL + 10 HEALTHY-THRESHOLD lines
+ pre/post-fix split). live_check_65.3.md = the baseline doc (per the immutable live_check field).
