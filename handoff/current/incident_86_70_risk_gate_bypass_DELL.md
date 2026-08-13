# INCIDENT -- the risk judge REJECTED and the book bought anyway (DELL, 2026-08-13)

Raised by the operator from the Agent Rationale UI (DELL card shows 3 signals,
no RiskJudge; NTAP 2026-07-31 shows 4 including `RiskJudge (gate)`).
Investigated 2026-08-13 ~21:40 CEST. **The operator's read is correct**, and the
mechanism is a known, already-approved, never-applied fix.

## What executed

```
21:31:27 [paper_trader] BUY 4.8064 x DELL @ $497.72 (source=bq_sim) = $2392.26 (fee: $2.39)
```

Live API (`/api/paper-trading/portfolio`), read from the running process:

| ticker | risk_judge_position_pct | cost_basis | % of NAV | entry |
|---|---|---|---|---|
| DELL | **null** | 2392.26 | **10.00%** | 2026-08-13 |
| NTAP | 4.0 | 950.90 | 3.98% | 2026-07-31 |

NAV 23,920.63. DELL's stop was also defaulted, not judge-set:
`phase-25.6: no stop_loss_price provided for DELL; defaulting to 457.9024 (8.0% below entry)`.

## The gate ran. Its verdict was discarded.

6 risk debates started today, **6 completed** -- none skipped:

| started | ticker | | completed | verdict |
|---|---|---|---|---|
| 20:29:39 | HPE | | 20:35:54 | APPROVE_HEDGED, HIGH, **4%** |
| 20:30:45 | MRVL | | 20:36:19 | REJECT, EXTREME, **0%** |
| 20:31:26 | **DELL** | | 20:37:59 | **REJECT, HIGH, 0%** |
| 20:55:50 | 009150.KS | | 21:00:36 | REJECT, HIGH, 0% |
| 21:01:34 | HPQ | | 21:07:29 | APPROVE_REDUCED, MODERATE, 3% |
| 21:26:18 | NTAP | | 21:30:56 | REJECT, HIGH, 0% |

**Attribution method (the completion log line carries NO ticker -- itself a
defect under concurrency).** Paired by exact-second match between each
`Risk debate complete` line and the row's `analysis_date` in
`financial_reports.analysis_results` (BQ is UTC, log is CEST = UTC+2). Five pair
to the second (HPE 18:35:54, MRVL 18:36:19, 009150.KS 19:00:36, HPQ 19:07:29,
NTAP 19:30:56); DELL's row is 18:38:03, 4s after the only remaining completion.
So DELL = REJECT/0% is **five exact matches plus elimination -- a strong
inference, not a direct per-ticker record.**

**The finding does not depend on that inference.** The batch contained only
`4%`, `0%`, `0%`. DELL executed at **10.00%**. Whichever verdict was DELL's, the
executed size exceeded it -- by 2.5x at best, unboundedly at worst.

## Mechanism: a falsy-zero check inverts REJECT into maximum size

`backend/services/portfolio_manager.py:939-955`

```python
pct = risk_assessment.get("recommended_position_pct")
if pct:            # <-- 0.0 is FALSY. A REJECT at 0% falls through.
    return float(pct)
pct = analysis.get("risk_judge_position_pct")
if pct:            # <-- same
    return float(pct)
return None        # REJECT/0% is now indistinguishable from "no judge ran"
```

`portfolio_manager.py:507`

```python
position_pct = cand["position_pct"] or 10.0  # Default 10% if Risk Judge didn't specify
```

So `0%` -> `None` -> **`10.0`**. The strongest possible risk signal is converted
into the largest default position. This is the falsy-zero class, the same shape
as the SecretStr truthiness bug that killed four alpha overlays.

## The fix exists, is written, was approved, and is OFF

`paper_risk_judge_shape_fix_enabled` (phase-66.2) already implements both halves:
`portfolio_manager.py:324-330` recovers the explicit 0.0 that `_extract_position_pct`
destroys, and `:504-505` uses `is not None` so 0% means no-buy.

Running process (`GET /api/settings/`): `paper_risk_judge_shape_fix_enabled: null`
-- **OFF**.

Masterplan **79.1** `[OPERATOR ACTION, pending]` -- "PROMOTE-66.2 FLAGS --
approved 2026-07-09, never applied" -- names all three symptoms verbatim:

> full-orchestrator BUYs still size at the 10%-NAV default instead of the
> RiskJudge pct, **a REJECT cannot bind**, and `risk_judge_decision` persists as `''`

All three are confirmed live today:
1. 10%-NAV default -- DELL at exactly 10.00%.
2. A REJECT cannot bind -- DELL bought.
3. `risk_judge_decision` persists as `''` -- **129/129 rows over 25 days**
   (2026-07-20..08-13) have empty `risk_judge_decision` and null
   `recommended_position_pct`, for every ticker. The verdict is never persisted.

## Corroboration independent of the UI

`financial_reports.signals_log.factors_json` for the two screenshotted entries:

- NTAP 2026-07-31 18:47:46 -- agents: Quant/screener, SignalStack/overlay, Trader/decision, **RiskJudge/gate** (1232 chars)
- DELL 2026-08-13 19:31:27 -- agents: Quant/screener, SignalStack/overlay, Trader/decision (517 chars) -- **no RiskJudge**

The missing gate is in the persisted data, not merely the rendering.

## Not done (operator decisions)

- Position NOT sold or resized; kill switch NOT touched.
- Flag NOT promoted -- 79.1 is explicitly an operator action, needs a
  `backend/.env` edit plus a restart, and restarts are batched to session end.
- No masterplan step flipped.

## Open, not established

- Why NTAP got a judge pct on 2026-07-31 while `analysis_results` shows the
  verdict was never persisted on that date either -- there are two write paths
  and only one was traced. Not resolved here.
- Whether any earlier position was opened under the same inversion. Only the two
  current positions were examined; `paper_trades` history was NOT swept.
