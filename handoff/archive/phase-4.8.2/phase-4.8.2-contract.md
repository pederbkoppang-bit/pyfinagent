# Contract -- Cycle 79 / phase-4.8 step 4.8.2

Step: 4.8.2 Portfolio CVaR + factor-exposure gate

## Hypothesis

Ship `backend/services/portfolio_risk.py` with three primitives:
1. `compute_cvar(returns, alpha=0.975)` -- Conditional VaR at 97.5%
   (Rockafellar-Uryasev 2000). Historical-simulation method (no
   distributional assumption).
2. `compute_ff3(portfolio_returns, factor_returns)` -- Fama-French
   3-factor OLS regression returning (alpha, market_beta,
   smb_beta, hml_beta, R^2).
3. `daily_check()` -- reads current paper-trading NAV history and
   returns a dict with cvar_97_5, ff3, and a gate decision
   (new_positions_allowed: bool + blocking reasons).

Thresholds from researcher defaults + contract:
- CVaR_97.5 > 2% daily loss magnitude -> block new positions
- |market_beta| > 1.5 -> block new positions

Where no live NAV history exists, the function synthesizes a 252-day
return series deterministically (seed recorded) so the signature
contract is testable end-to-end without waiting for live trading.
JSON artifact records `data_source: "seeded" | "live"`.

## Scope

Files created:

1. **NEW** `backend/services/portfolio_risk.py`
   - `compute_cvar` (historical method, honest for empirical returns)
   - `compute_ff3` (OLS via numpy.linalg.lstsq, no statsmodels dep)
   - `daily_check()` returning full gate-decision dict
   - Constants: `CVAR_LIMIT_PCT = 0.02`, `BETA_CAP = 1.5`

2. **NEW** `scripts/audit/portfolio_risk_audit.py`
   - Exercises gate by supplying both (a) benign returns that pass
     and (b) anomalous returns (fat left tail) that trigger both
     CVaR and beta limits. Asserts the gate blocks correctly.
   - Writes `handoff/portfolio_risk_audit.json`.

## Immutable success criteria

1. cvar_daily_computed -- `daily_check()['cvar_97_5']` present + float.
2. ff3_weekly_computed -- `daily_check()['ff3']` present with
   market_beta / smb_beta / hml_beta / alpha keys.
3. new_positions_blocked_when_cvar_over_2pct -- audit proves gate
   returns `new_positions_allowed=False` and includes
   "cvar_exceeded" in blocking_reasons when CVaR > 2%.
4. beta_cap_enforced -- same audit proves |market_beta| > 1.5 blocks.

## Verification (immutable)

    python -c "from backend.services.portfolio_risk import daily_check; r=daily_check(); assert 'cvar_97_5' in r and 'ff3' in r"

Plus self-imposed:
    python scripts/audit/portfolio_risk_audit.py --check

## Anti-rubber-stamp

qa must verify:
- CVaR uses empirical (historical) method, not a placeholder that
  returns a constant.
- FF3 regression actually solves via lstsq, not a stub.
- Gate's `blocking_reasons` list is populated from REAL checks,
  not hardcoded strings.
- Audit's anomalous-returns fixture really crosses the 2% CVaR
  threshold (no fixture-cheating).

## References

- Rockafellar & Uryasev 2000 "Optimization of Conditional Value-
  at-Risk" J. Risk 2(3)
- Fama & French 1992/1993 three-factor model
- AFML ch.15 (risk management)
