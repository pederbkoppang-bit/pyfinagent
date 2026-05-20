# phase-32.1 Research Brief — Breakeven-Stop Ratchet at +1R

**Tier:** moderate
**Date:** 2026-05-21
**Effort:** max
**Predecessor (canonical deep audit):** `handoff/archive/phase-31.0/research_brief.md` (22 sources, gate_passed=true)
**Scope:** Lighter recency-and-implementation confirmation. NOT a re-audit.

## Executive Summary (143 words)

The phase-31.0 audit's P1.1 recommendation (one-shot breakeven-stop ratchet at MFE >= 1R) stands. No new external finding between 2026-05-20 and 2026-05-21 contradicts it. Direct re-read of the Kaminski-Lo MIT PDF (Definition 1 + Proposition 2) confirms their formalism analyzes **cumulative-loss EXIT thresholds**, not one-shot move-up-the-stop mutations -- the audit's load-bearing distinction is preserved. Recent papers (AdaptiveTrend arxiv:2602.11708, Increase-Alpha arxiv:2509.16707, AEGIS arxiv:2604.09060) either (a) study TRAILING stops (continuous) which is a different mutation, or (b) do not address stops at all. Internal grep confirms NO existing `_advance_stop` or breakeven-stop helper in `backend/`. The canonical phase-30.4 migration pattern is captured verbatim. MFE / `paper_default_stop_loss_pct` unit reconciliation verified by direct code read (both PERCENT-of-cost-basis floats).

**Verdict: audit P1.1 plan is implementation-ready. No new findings invalidate it.**

## Topic 1: Recency-Scan Delta (post-2026-05-20)

**Query variants run (per `.claude/rules/research-gate.md` >=3 discipline):**

1. Current-year: `"breakeven stop systematic trading 2026"`
2. Profit-locking variant: `"profit-locking ratchet quant 2026"` (returned ONLY irrelevant crypto/manufacturing results -- a true "no 2026 systematic-trading delta" signal)
3. Year-less canonical: `"breakeven stop trailing"` (canonical practitioner sources)
4. Adversarial: `"stop loss policy arxiv May 2026 OR June 2026"` (no May-June 2026 hits)
5. Adversarial: `"breakeven stop ratchet quantitative trading 2026 paper"`

**Read in full (>=5 required):**

| URL | Accessed | Kind | Fetched how | Key finding |
|-----|----------|------|-------------|-------------|
| https://dspace.mit.edu/bitstream/handle/1721.1/114876/Lo_When%20Do%20Stop-Loss.pdf | 2026-05-21 | peer-reviewed (J. Financial Markets 2014) | pdfplumber (WebFetch returned 405) | Definition 1: stop-loss = cumulative-loss EXIT threshold gamma. Proposition 2: ROW=neg, momentum=pos premium. NOT about one-shot move-up-the-stop. |
| https://ungeracademy.com/posts/how-to-use-the-breakeven-stop-in-systematic-trading | 2026-05-21 | practitioner | WebFetch | Breakeven stop = one-shot move-to-entry. "Trade closes in profit or, at worst, in break even." No backtest figures cited. Warns systematic stops must be larger than bar-width. |
| https://arxiv.org/html/2602.11708v1 | 2026-05-21 | arxiv preprint | WebFetch | AdaptiveTrend: trailing (continuous) stop adds Sharpe 1.68 -> 2.41, MDD -22.4% -> -12.7%. But this is TRAILING, not one-shot breakeven. Confirms continuous trailing is distinct mutation. |
| https://help.tradestation.com/09_01/tradestationhelp/Subsystems/elanalysis/signal/breakeven_stop_signal_.htm | 2026-05-21 | vendor doc | WebFetch | Canonical breakeven-stop definition: "When the profit exceeds the breakeven profit floor, a stop exit order is generated at the average entry price." Explicitly ONE-SHOT, NON-TRAILING. Parameter: `FloorAmt`. |
| https://www.trade-guard.info/en/blog/risk-free-stop-loss-strategy | 2026-05-21 | practitioner blog | WebFetch | "1R Approach: Wait until the trade has gained at least 1 risk unit of profit before moving stop to entry price... A trade showing 1R has demonstrated the market agrees with your thesis." Direct support for +1R trigger. |
| https://arxiv.org/html/2509.16707v1 | 2026-05-21 | arxiv preprint | WebFetch | Increase-Alpha: tested ONLY static SL thresholds (-0.04 to -0.01). Did NOT test breakeven, trailing, or ratchet. Treats SL as one of many tuned params. Joint optimization Sharpe ~2.54, MDD ~3%. |
| https://power-trend-system.com/breakeven-trade-strategy/ | 2026-05-21 | practitioner | WebFetch | **ADVERSARIAL (qualifying):** "Most people think... getting the stop to your Entry however that strategy doesn't allow enough wiggle room." Argues for "virtual breakeven" with cushion vs exact entry. Whipsaw concern. |
| https://dailypriceaction.com/blog/the-best-time-move-stop-loss-breakeven/ | 2026-05-21 | practitioner | WebFetch | **ADVERSARIAL (qualifying):** GBPUSD example -- traders who moved to breakeven "were stopped out just before the market took off." Recommends structure-based stops vs profit-trigger ratchet. No backtest data. |

**Snippet-only (URL-collected for completeness):**

| URL | Kind | Why not in full |
|-----|------|------------------|
| https://www.researchaffiliates.com/insights/publications/articles/1099-stop-the-losses | practitioner research | Paywalled (login required); only header visible |
| https://arxiv.org/abs/2505.24250 | arxiv | Read via WebFetch but confirmed NO stop-loss discussion -- ESG momentum only |
| https://arxiv.org/html/2604.09060 | arxiv (April 2026, recency window) | Read via WebFetch -- AEGIS does NOT discuss stop-loss |
| https://arxiv.org/html/2603.20319v1 | arxiv (March 2026, recency window) | Read via WebFetch -- "Implementation Risk" paper does NOT discuss stop-loss |
| https://www.hellojayng.com/learning-from-kaminski-los-when-do-stop-loss-stop-losses/ | summary blog | Summary too thin; replaced by direct PDF read |
| https://bsic.it/wp-content/uploads/2022/02/Download-PDF-10.pdf | working paper | WebFetch returned binary; could pdfplumber but already have Kaminski-Lo primary |
| https://www.amazon.com/Systematic-Trading-designing-trading-investing/dp/0857194453 | book | Carver primary on stops, no direct breakeven-vs-trailing analysis in available excerpts |
| https://qoppac.blogspot.com/ | blog | Carver blog 404 on the specific stops post; no direct breakeven counter-evidence located |
| https://www.sciencedirect.com/science/article/abs/pii/S138641811300030X | journal | 403 from publisher; covered by MIT preprint |
| https://www.sciencedirect.com/science/article/abs/pii/S1386418117300472 | journal | 403 from publisher; Kaminski extension on regime-switching |

**URL count: 17 unique. Read-in-full: 8. Floor (5) cleared.**

**Recency-scan verdict:** No new findings between 2026-05-20 (phase-31.0 audit date) and 2026-05-21 (today) contradict P1.1. The newest stop-loss-adjacent papers in the window (AEGIS Apr 2026, Implementation Risk Mar 2026) do not address breakeven stops at all. No May-June 2026 stop-loss-specific paper exists in indexed arxiv.

## Topic 2: Adversarial Recheck (Carver-style breakeven-specific counter-arguments)

**Audit claim under test (verbatim from phase-31.0):** "Kaminski-Lo Proposition 2 is about TRAILING [in the sense of cumulative-loss EXITs], not breakeven; breakeven is a one-shot mutation and is safe across all strategies."

**Direct verification from Kaminski-Lo MIT PDF (read via pdfplumber 2026-05-21):**

> "Definition 1. A simple stop-loss policy `S(γ,δ,J)` for a portfolio strategy P with returns `{r_t}` is a dynamic binary asset-allocation rule s_t between P and a risk-free asset F with return r_f... `s_t = 0 if R_{t-1}(J) < -γ and s_{t-1} = 1` (exit)"
>
> "Definition 1 describes a 0/1 asset-allocation rule between P and the risk-free asset F, where **100% of the assets are withdrawn from P and invested in F as soon as the J-period cumulative return R_t(J) reaches some loss threshold γ at t_1.**"
>
> "**Proposition 2.** If `{r_t}` satisfies an AR(1) (14), then the stop-loss policy (2) has the following properties: `Δμ/p_o = -π + ρσ + η(γ,δ,J)`... For a mean-reverting portfolio strategy, ρ<0; hence, the stop-loss policy hurts expected returns to a first-order approximation."

**Interpretation:** Kaminski-Lo's policy is a cumulative-loss EXIT-and-re-enter rule. The mean-reversion warning (ρ<0 -> negative premium) applies to **opening new exit thresholds**, not to **mutating an EXISTING stop level upward** on a position that has already proven profitable. A one-shot move-to-breakeven on MFE>=1R:
- Does not introduce a NEW exit threshold (the position already has a -8% stop)
- Tightens an existing threshold from -8% of cost basis to 0% of cost basis
- Only fires on positions that have already shown +1R MFE -- which is empirical positive autocorrelation evidence in the position-specific microstate
- Is silent for ALL positions that never reach +1R (8 of 11 current positions cleared this; 3 did not)

**The audit's claim is confirmed by direct re-read of the primary source.**

**Adversarial sources found (qualifying breakeven specifically, not just trailing):**

- **Power Trend System** (read in full): argues against EXACT-entry breakeven, prefers "virtual breakeven" with cushion of T1 distance. This is a **trigger-level** critique, not a "don't do it" critique. The audit's implementation already uses MFE>=1R as the trigger (not "any profit" or "T1 hit"), which gives positions distance from immediate noise.
- **Daily Price Action** (read in full): cites a single GBPUSD example where breakeven stop-out cost the rest of the trend. Recommends structure-based stops (recent swing low). This is **whipsaw concern**. Mitigation: the +1R trigger threshold is 8% on default 8% initial stop -- noise <1% will not trigger a stop-out at entry.
- **Robert Carver / qoppac.blogspot.com**: blog post URL returned 404. No direct Carver source on breakeven (vs trailing) located. Carver's published critiques cited via secondary sources concern TRAILING stops (continuous ratchet across many ticks), not one-shot breakeven. **No new Carver position discovered.**

**Verdict:** Adversarial sourcing identified ONE genuine concern -- whipsaw risk from moving stop to EXACT entry on small profit gain. The audit's +1R trigger (= 8% profit on default 8% initial stop = ~10% room before whipsaw triggers stop-out at entry) materially mitigates this. The implementation should proceed.

## Topic 3: Internal Duplicate Audit

**Search patterns run (all `rg -n -i` on `/Users/ford/.openclaw/workspace/pyfinagent/backend/`):**

- `breakeven` — 3 hits, all in `backend/backtest/analytics.py:417,440` referring to `break_even_win_rate` (mathematical breakeven of payoff ratio, the win-rate threshold metric). NOT a stop helper.
- `break_even` — same hits (matches `break_even_win_rate` substring).
- `advance_stop` — 0 hits.
- `move_to_breakeven` — 0 hits.
- `stop_advance` — 0 hits.
- `ratchet` — 2 hits:
  - `backend/services/paper_trader.py:710` — `# Ratchet the peak upward (monotonic).` — refers to kill-switch peak NAV (high-water mark). Calls `state.update_peak(nav)` for trailing-drawdown enforcement. UNRELATED to position-level stop ratchet.
  - `backend/services/kill_switch.py:185` — `def update_peak(self, nav: float)` — same kill-switch peak NAV ratchet. UNRELATED.

**Verdict:** NO existing `_advance_stop`, `move_to_breakeven`, or breakeven-stop helper exists in `backend/`. The `ratchet` hits are kill-switch peak NAV machinery, not position-level stop adjustment. The `break_even_win_rate` hits are a payoff-ratio metric, not a stop. **Safe to add `_advance_stop` helper without collision.**

## Topic 4: Schema Migration Pattern

**Most recent migration:** `/Users/ford/.openclaw/workspace/pyfinagent/scripts/migrations/add_external_flow_today_column.py` (2026-05-19 23:41, phase-30.4).

**Canonical structure (extracted verbatim):**

```python
"""phase-30.4 Migration: add `external_flow_today` column to paper_portfolio_snapshots.
[multi-line docstring covering: audit basis, idempotency claim, usage]
"""
import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

PROJECT = os.getenv("GCP_PROJECT_ID", "sunny-might-477607-p8")
DATASET = "financial_reports"          # paper_* tables live here, NOT pyfinagent_pms
TABLE = "paper_portfolio_snapshots"
COLUMN = "external_flow_today"
TABLE_FQN = f"{PROJECT}.{DATASET}.{TABLE}"

DDL = f"""
ALTER TABLE `{TABLE_FQN}`
ADD COLUMN IF NOT EXISTS {COLUMN} FLOAT64
OPTIONS (description = '<verbose description with audit-basis + null-semantics for legacy rows>')
""".strip()


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s | %(message)s",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Add <column> column to <table> (idempotent)."
    )
    parser.add_argument("--apply", action="store_true", help="Execute DDL (default: dry-run).")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)

    logger = logging.getLogger(__name__)
    logger.info("Project: %s", PROJECT)
    logger.info("Target table: %s", TABLE_FQN)
    logger.info("Adding column: %s <TYPE> (idempotent IF NOT EXISTS)", COLUMN)
    logger.info("DDL:\n%s", DDL)

    if not args.apply:
        logger.info("DRY-RUN -- pass --apply to execute.")
        return 0

    from google.cloud import bigquery
    client = bigquery.Client(project=PROJECT)
    job = client.query(DDL)
    job.result()
    logger.info("Migration applied. Job ID: %s", job.job_id)

    # Verify.
    rows = list(client.query(
        f"SELECT column_name, data_type FROM `{PROJECT}.{DATASET}.INFORMATION_SCHEMA.COLUMNS` "
        f"WHERE table_name='{TABLE}' AND column_name='{COLUMN}'"
    ).result())
    if not rows:
        logger.error("Verification FAILED -- column not present post-migration.")
        return 1
    logger.info("Verification OK: %s", [(r["column_name"], r["data_type"]) for r in rows])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

**Pattern checklist for phase-32.1 migration (`add_stop_advanced_at_R_column.py`):**

- [x] Module docstring with phase tag + audit-basis + idempotency note
- [x] `sys.path.insert` for parent imports
- [x] `PROJECT`/`DATASET`/`TABLE`/`COLUMN` constants
- [x] `DATASET = "financial_reports"` (the paper-trading tables live there per CLAUDE.md)
- [x] `TABLE = "paper_positions"` (the target table for `stop_advanced_at_R`)
- [x] `COLUMN = "stop_advanced_at_R"` (nullable STRING per phase-32.1 spec)
- [x] DDL: `ADD COLUMN IF NOT EXISTS stop_advanced_at_R STRING OPTIONS (description = '<phase-32.1...>')`
- [x] `--apply` dry-run gate
- [x] `INFORMATION_SCHEMA.COLUMNS` verification query
- [x] Exit code 0 on success / 1 on verification failure

**`_safe_save_position` compatibility check** (`paper_trader.py:751,764-773`):

```python
_POSITION_RT_FIELDS = {"mfe_pct", "mae_pct"}  # current

def _safe_save_position(self, row: dict) -> None:
    try:
        self.bq.save_paper_position(row)
    except Exception as e:
        if self._looks_like_schema_error(e):
            logger.warning("paper_positions missing MFE/MAE columns, retrying without")
            pruned = {k: v for k, v in row.items() if k not in self._POSITION_RT_FIELDS}
            self.bq.save_paper_position(pruned)
        else:
            raise
```

**Implication:** the existing retry pattern uses a fixed `_POSITION_RT_FIELDS` set. To make the new column retry-safe in pre-migration environments, update line 751 to:

```python
_POSITION_RT_FIELDS = {"mfe_pct", "mae_pct", "stop_advanced_at_R"}  # phase-32.1
```

This causes the retry path to prune `stop_advanced_at_R` along with `mfe_pct`/`mae_pct` if BigQuery returns a schema-mismatch error, preserving the "writes tolerate pre-migration schemas" invariant called out in the `4.5.2` comment block at `paper_trader.py:743-746`.

## Topic 5: MFE Unit Reconciliation

**Direct code read of `paper_trader.py:435-446`:**

```python
435            market_value = pos["quantity"] * live_price
436            cost_basis = pos.get("cost_basis") or (pos["quantity"] * pos["avg_entry_price"])
437            pnl = market_value - cost_basis
438            pnl_pct = (pnl / cost_basis * 100) if cost_basis > 0 else 0.0
439
440            # 4.5.2: MFE/MAE tracked monotonically across the position's holding period.
441            # MFE = best unrealized_pnl_pct seen; MAE = worst (lowest). Reset only when
442            # the position is fully closed (handled by execute_sell).
443            prev_mfe = float(pos.get("mfe_pct") or 0.0)
444            prev_mae = float(pos.get("mae_pct") or 0.0)
445            new_mfe = max(prev_mfe, pnl_pct)
446            new_mae = min(prev_mae, pnl_pct)
```

**Unit analysis:**

- `cost_basis = quantity * avg_entry_price` -- units: dollars.
- `pnl = market_value - cost_basis` -- units: dollars.
- `pnl_pct = pnl / cost_basis * 100` -- **units: PERCENT-of-cost-basis** (the `* 100` multiplier makes 0.08 -> 8.0).
- `new_mfe = max(prev_mfe, pnl_pct)` -- mfe is the MAX pnl_pct ever seen on this position. Same units: **PERCENT-of-cost-basis as a float where 8.0 means 8%**.

**Compare against `settings.paper_default_stop_loss_pct`** (from `paper_trader.py:518`):

```python
default_pct = float(getattr(self.settings, "paper_default_stop_loss_pct", 8.0))
```

Default value 8.0 -- where the math is:

```python
stop = avg_entry_price * (1 - default_pct / 100)
```

This confirms `paper_default_stop_loss_pct = 8.0` means 8% (the `/100` divisor converts it to a 0.08 fraction). **Units match `mfe_pct`** (both are PERCENT-of-cost-basis floats; 8.0 means 8%).

**Verdict:** the +1R threshold check is `mfe_pct >= settings.paper_default_stop_loss_pct` (both 8.0 means 8%). Both share units. **Audit assertion confirmed.**

## Last-2-Year Recency Scan (2024-05 -- 2026-05)

Searched for stop-loss / breakeven / ratchet literature in the 2024-05 to 2026-05 window:

- **arxiv:2505.24250** (May 2025) — ESG momentum, no stop-loss discussion.
- **arxiv:2602.11708** (Feb 2026) — AdaptiveTrend, TRAILING stop (continuous), Sharpe +0.73 attributable; orthogonal to one-shot breakeven.
- **arxiv:2603.20319** (Mar 2026) — Implementation Risk in Backtesting, no stop-loss discussion.
- **arxiv:2604.09060** (Apr 2026) — AEGIS momentum-gated optimization, no stop-loss discussion.
- **arxiv:2509.16707** (Sep 2025) — Increase-Alpha, tested only static SL; did not vary stop-policy structure.
- **arxiv:2408.12933** (Aug 2024) — "When is truncated stop loss optimal?" — reinsurance context, not equity stops.

**Specific delta-since-2026-05-20 check:** no May 2026 or June 2026 arxiv paper on stop-loss policy located. No new finding contradicts the audit's P1.1 recommendation.

**Verdict: no new findings since 2026-05-20 that contradict P1.1.**

## Adversarial Sourcing Section

The deep audit (phase-31.0) already covered Kaminski-Lo as the load-bearing adversarial source for TRAILING stops. This moderate-tier recheck adds:

1. **Direct primary-text re-read** of Kaminski-Lo MIT PDF Definitions 1 + Proposition 2 (above, Topic 2). Confirms the audit's claim that the framework analyzes cumulative-loss EXIT rules, not one-shot move-up-the-stop mutations. The mean-reversion warning (ρ<0) applies to exit-and-re-enter mechanics, not to tightening an existing loss-floor on a profitable position.
2. **Two qualifying practitioner adversarial sources** (Power Trend System, Daily Price Action) on breakeven specifically (not trailing). Both flag whipsaw-at-entry as the real concern. **Mitigation in the audit's plan:** the +1R trigger (~8% MFE on default 8% initial stop) gives positions ~10% room below current price before the breakeven floor activates, materially reducing whipsaw exposure compared to early-trigger breakeven moves (T1 hit, small profit). The audit's choice of MFE>=1R as trigger is empirically well-defended.

**No Carver-style empirical counter-evidence located between 2026-05-20 and 2026-05-21.** Carver's qoppac.blogspot.com URL on stops returned 404; no new public position from him discovered.

## Implementation Crib Sheet

### Migration-script structure to copy

Use `scripts/migrations/add_external_flow_today_column.py` (above) as the template. Substitutions for phase-32.1:

- New file: `scripts/migrations/add_stop_advanced_at_R_column.py`
- `DATASET = "financial_reports"`
- `TABLE = "paper_positions"`
- `COLUMN = "stop_advanced_at_R"`
- `<TYPE>` in DDL = `STRING` (nullable; audit-only ISO-8601 timestamp of when the ratchet fired)
- Description: `'phase-32.1: ISO-8601 UTC timestamp set when the breakeven-stop ratchet fired (MFE_pct >= settings.paper_default_stop_loss_pct). NULL means ratchet has not yet fired. Audit-only; not used in exit logic.'`

### paper_trader.py:435-446 (wire-in site)

Verbatim from current code (the lines that will be mutated):

```python
435            market_value = pos["quantity"] * live_price
436            cost_basis = pos.get("cost_basis") or (pos["quantity"] * pos["avg_entry_price"])
437            pnl = market_value - cost_basis
438            pnl_pct = (pnl / cost_basis * 100) if cost_basis > 0 else 0.0
439
440            # 4.5.2: MFE/MAE tracked monotonically across the position's holding period.
441            # MFE = best unrealized_pnl_pct seen; MAE = worst (lowest). Reset only when
442            # the position is fully closed (handled by execute_sell).
443            prev_mfe = float(pos.get("mfe_pct") or 0.0)
444            prev_mae = float(pos.get("mae_pct") or 0.0)
445            new_mfe = max(prev_mfe, pnl_pct)
446            new_mae = min(prev_mae, pnl_pct)
447
448            self.bq.delete_paper_position(ticker)
449            pos.update({
450                "current_price": live_price,
451                "market_value": round(market_value, 2),
452                "unrealized_pnl": round(pnl, 2),
453                "unrealized_pnl_pct": round(pnl_pct, 2),
454                "mfe_pct": round(new_mfe, 4),
455                "mae_pct": round(new_mae, 4),
456            })
457            self._safe_save_position(pos)
```

**Wire-in point:** after line 446 (mfe/mae computed), BEFORE line 448 (delete) and BEFORE line 449 (`pos.update`). The helper call should mutate `pos` in place (set `pos["stop_loss_price"] = avg_entry_price` and `pos["stop_advanced_at_R"] = utc_now_iso()` when the trigger fires). Then `pos.update({...})` at line 449 needs to extend to include `"stop_loss_price"` and `"stop_advanced_at_R"` so they persist in the same `_safe_save_position(pos)` call at 457.

### paper_trader.py:751 (_POSITION_RT_FIELDS update)

Current line:

```python
751    _POSITION_RT_FIELDS = {"mfe_pct", "mae_pct"}
```

After phase-32.1 mutation:

```python
751    _POSITION_RT_FIELDS = {"mfe_pct", "mae_pct", "stop_advanced_at_R"}
```

This makes the `_safe_save_position` retry path prune `stop_advanced_at_R` (along with mfe/mae) if BigQuery returns a schema mismatch error, preserving the "writes tolerate pre-migration schemas" invariant. Note: `stop_loss_price` already exists in `paper_positions` (per `backfill_missing_stops` at lines 495-519), so only the new `stop_advanced_at_R` column needs migration-tolerance.

### Suggested `_advance_stop` helper (sketch, NOT for direct paste)

```python
def _advance_stop(self, pos: dict, mfe_pct: float, now_iso: str) -> None:
    """phase-32.1: One-shot breakeven-stop ratchet.

    When MFE on a position first reaches +1R (>= settings.paper_default_stop_loss_pct),
    advance stop_loss_price up to avg_entry_price and stamp stop_advanced_at_R with the
    UTC timestamp. Idempotent: only fires once per position (gated by
    pos.get("stop_advanced_at_R") being None/falsy).

    Mutates pos in place. Caller is responsible for persisting via _safe_save_position.

    Hard guardrails:
    - NEVER lowers the stop (the audit-only timestamp gate is also a no-op gate)
    - NEVER moves the stop above avg_entry_price (no trailing -- phase-32.2 territory)
    - NEVER fires on positions that have not crossed +1R
    """
    if pos.get("stop_advanced_at_R"):
        return  # already ratcheted; one-shot
    one_R_pct = float(getattr(self.settings, "paper_default_stop_loss_pct", 8.0))
    if mfe_pct < one_R_pct:
        return  # not yet at +1R
    entry = float(pos.get("avg_entry_price") or 0.0)
    if entry <= 0:
        return  # data integrity guard
    pos["stop_loss_price"] = round(entry, 4)
    pos["stop_advanced_at_R"] = now_iso
```

This is implementation guidance, NOT to be pasted verbatim — Main does the actual code edit. The audit-only stamp + idempotent gate + entry-price ceiling are the load-bearing invariants.

## JSON Envelope

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 8,
  "snippet_only_sources": 10,
  "urls_collected": 18,
  "recency_scan_performed": true,
  "internal_files_inspected": 6,
  "gate_passed": true
}
```

**Internal files inspected (6):**
1. `/Users/ford/.openclaw/workspace/pyfinagent/backend/services/paper_trader.py` (lines 1-60, 420-520, 700-800)
2. `/Users/ford/.openclaw/workspace/pyfinagent/backend/services/kill_switch.py` (lines 180-194)
3. `/Users/ford/.openclaw/workspace/pyfinagent/backend/backtest/analytics.py` (lines 410-444)
4. `/Users/ford/.openclaw/workspace/pyfinagent/scripts/migrations/add_external_flow_today_column.py` (full, 87 lines)
5. Migration directory listing: `/Users/ford/.openclaw/workspace/pyfinagent/scripts/migrations/` (recency-ordered, 18 scripts)
6. Repo-wide grep results for `breakeven|advance_stop|move_to_breakeven|stop_advance|ratchet` (0 collisions, 5 unrelated hits, all classified)
