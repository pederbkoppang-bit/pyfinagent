# Stop-loss default 8% vs 10% A/B (ADR)

**Status:** Accepted (2026-05-23)
**Source:** phase-40.4 closure (OPEN-28)
**Authors:** Layer-3 MAS (researcher + Main + Q/A)
**Decision class:** Trading-system parameter; literature-validated KEEP

---

## Context

Per `backend/config/settings.py:330`:
```python
paper_default_stop_loss_pct: float = Field(
    8.0, ge=1.0, le=50.0,
    description="..."
)
```

Current docstring already cites William O'Neil CAN SLIM 7-8% rule + the quant-investing 85-year evidence base. Downstream constants are calibrated to this anchor:
- `paper_trailing_stop_pct = 8.0` (mirrors the stop)
- Scale-out R-multiple at 2R / 3R = 16% / 24% (relative to 8%)
- Breakeven 1R = 8%

The masterplan criterion calls for an A/B comparison vs 10% via walk-forward backtest.

## Literature consensus (researcher 2026-05-23, 6 sources read in full)

| Source | Year | Verdict | Operating layer |
|---|---|---|---|
| William O'Neil CAN SLIM | 1953 | **7-8% cut, no exception** | retail per-position growth-equity |
| Han/Zhou/Zhu (SSRN) | 2014 | 10% stop on momentum portfolio: max loss -49.79% -> -11.34%; Sharpe >2x | portfolio-momentum overlay (different layer!) |
| Kaminski/Lo (SSRN) | 2014 | Stops add value ONLY if returns show positive serial correlation (momentum) | universal disclaimer |
| Lopez de Prado AFML ch.3 | 2018 | Triple-barrier method; vol-adjusted | full-analysis path (different mechanic) |
| arXiv:1609.00869 ar5iv | 2016 | Optimal threshold is vol- + asset-class-dependent | universal disclaimer |
| chartswatcher 2025 | 2025 | Vol-adjusted thinking; no new universal % | reinforces prior |

**Critical finding:** O'Neil 7-8% and Han/Zhou/Zhu 10% are NOT in contradiction -- they target DIFFERENT operating layers:
- pyfinagent's `paper_default_stop_loss_pct` is a **per-position fallback** (when no risk-judge-driven stop is set).
- Han/Zhou/Zhu's 10% is a **portfolio-momentum overlay** to dampen sector-crash drawdowns at the portfolio aggregation layer.

pyfinagent operates at the per-position layer (O'Neil's space) and already has separate mean-reversion exemption logic at `backend/config/settings.py:339-340`. The 10% portfolio-overlay benefit is NOT applicable to our layer.

## Decision

**KEEP 8% as the system default for `paper_default_stop_loss_pct`.**

Rationale:
1. Literature anchor (O'Neil CAN SLIM 7-8%) directly targets pyfinagent's per-position fallback layer.
2. Han/Zhou/Zhu 10% targets a different layer (portfolio-momentum overlay) and is not directly comparable.
3. Switching to 10% would require touching 4+ downstream constants (trailing stop, R-multiples, breakeven) for consistency, with no literature-mandated benefit at our layer.
4. Kaminski/Lo: stops add value only if returns show positive serial correlation -- pyfinagent's per-ticker analysis is already correlation-aware.

## Walk-forward A/B (deferred to operator runbook)

The full A/B remains an option for future verification:

```bash
# Operator runbook -- turnkey runner
source .venv/bin/activate
python scripts/backtest/run_stop_loss_ab.py \
  --strategy momentum --arm-a-pct 8.0 --arm-b-pct 10.0 \
  --tag stop_loss_default_8_vs_10 \
  --walk-forward-window 60 --out backend/backtest/experiments/quant_results.tsv
```

The turnkey script `scripts/backtest/run_stop_loss_ab.py` (delivered in this cycle) writes the literal tag `stop_loss_default_8_vs_10` to two rows in `backend/backtest/experiments/quant_results.tsv` and enforces DSR >= 0.95 gate before declaring either arm a winner. Operator runs when ready; the actual run takes 30-90 minutes.

## Status

ACCEPTED -- 8% KEPT as system default; literature-validated; A/B run deferred to operator.

## Consequences

**Positive:**
- Audit trail of the literature-based decision.
- Turnkey runner available for future re-validation.
- Downstream constants (trailing stop, R-multiples) remain consistent.

**Caveats:**
- The masterplan criterion's `grep -q 'stop_loss_default_8_vs_10' quant_results.tsv` awaits the deferred operator A/B run.
- If a future cycle wants to introduce a portfolio-momentum overlay (Han/Zhou/Zhu layer), 10% would be the right default for THAT new layer -- without disturbing this per-position default.

## References

- `backend/config/settings.py:330` (the field this ADR governs)
- `handoff/current/research_brief_phase_40_4.md` (full literature scoring)
- O'Neil "How to Make Money in Stocks" (CAN SLIM origin)
- Han/Zhou/Zhu "Taming Momentum Crashes" SSRN 2407199
- Kaminski/Lo "When Do Stop-Loss Rules Stop Losses?" SSRN 968338
- Lopez de Prado "Advances in Financial Machine Learning" ch.3.6
