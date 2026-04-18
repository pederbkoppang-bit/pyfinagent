# Experiment Results -- Cycle 79 / phase-4.8 step 4.8.2

Step: 4.8.2 Portfolio CVaR + factor-exposure gate

## What was generated

1. **NEW** `backend/services/portfolio_risk.py`
   - `compute_cvar(returns, alpha=0.975)` -- Rockafellar-Uryasev
     historical method; returns positive loss magnitude.
   - `compute_ff3(port_returns, factor_returns, rf)` -- OLS via
     `np.linalg.lstsq` on [1, Mkt-Rf, SMB, HML] design matrix;
     returns alpha + 3 betas + R^2 + n_obs.
   - `daily_check()` -- returns gate decision:
     {cvar_97_5, ff3, gate: {new_positions_allowed, reasons}}.
     Seeds deterministic 252-day returns + 3-factor series when
     live NAV history is absent; `data_source: seeded|live` made
     explicit so auditors see the origin.
   - Constants: `CVAR_LIMIT_PCT = 0.02`, `BETA_CAP = 1.5`,
     `CVAR_ALPHA = 0.975`.

2. **NEW** `scripts/audit/portfolio_risk_audit.py`
   - Three fixtures: benign (seeded 0.8% daily vol, gate allows),
     cvar-trip (5% of days overwritten with -5% returns, CVaR 5.14%),
     beta-trip (port = 2.2 * mkt + noise, recovered beta 2.18).
   - Asserts blocking_reasons contain "cvar_exceeded" and
     "beta_cap_exceeded" on the trips, empty on benign.
   - Emits `handoff/portfolio_risk_audit.json`.

## Verification (verbatim, immutable)

    $ python -c "from backend.services.portfolio_risk import daily_check; \
                  r=daily_check(); assert 'cvar_97_5' in r and 'ff3' in r"
    exit=0  (cvar=1.87%, beta=0.09, gate allows)

    $ python scripts/audit/portfolio_risk_audit.py --check
    {"verdict": "PASS", "benign_ok": true,
     "cvar_trip_ok": true, "beta_trip_ok": true}
    exit=0

## Success criteria

| Criterion | Result |
|-----------|--------|
| cvar_daily_computed | PASS (value 0.018749 on seeded benign) |
| ff3_weekly_computed | PASS (market_beta, smb_beta, hml_beta, alpha, r_squared, n_obs) |
| new_positions_blocked_when_cvar_over_2pct | PASS (cvar-trip fixture: 5.14% > 2%, blocked) |
| beta_cap_enforced | PASS (beta-trip fixture: 2.18 > 1.5, blocked) |

## Known limitations (tracked follow-up)

- When live NAV history lands (post go-live), `daily_check` will
  prefer the live series; seeded path becomes a test-only
  convenience. `data_source` field already flags which path ran.
- `daily_check` called without rf defaults to 0; a follow-up step
  will wire in FRED DGS3MO as risk-free source + FF3 factors from a
  persistent factor table.
- Blocking reasons are ADVISORY at this layer; wiring them into the
  order-submission code path lives in the next phase-4.8.x step.
