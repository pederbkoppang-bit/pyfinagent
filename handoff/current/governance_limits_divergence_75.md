# Governance limits vs runtime settings -- divergence report (phase-75.8, gap3-02)

Date: 2026-07-23. Scope: OBSERVABILITY ONLY -- no cap value, kill-switch
line, limits.yaml value, or lint severity was changed in phase-75.8.
Checker: `backend/governance/divergence.py` (startup WARNING in
`main.py` lifespan; fail-open; pure read via the lru-cached
`limits_schema.load()`).

Units: `limits.yaml` stores FRACTIONS (0.02 = 2%); `settings.py` stores
PERCENTS (4.0 = 4%). All values below are normalized to percent. The
checker uses `math.isclose(rel_tol=1e-9)` so float noise cannot
manufacture a divergence.

## The six governed limits vs their runtime counterparts (measured 2026-07-23)

| # | limits.yaml field | Governed | Runtime counterpart | Live value | Status |
|---|---|---|---|---|---|
| 1 | `max_position_notional_pct` | 5% | NO dedicated settings field. De-facto BUY sizing default is the RiskJudge 10%-NAV fallback (`settings.py:314` description; `decide_trades` sizing default) | ~10% per BUY (default path) | **UNMAPPED -- de-facto 2x the governed value.** Not machine-checked (no single settings attr to compare); recorded here for GOV-LIMITS-DECIDE |
| 2 | `max_portfolio_leverage` | 1.5x | NONE -- paper engine is long-only, cash-bounded; no leverage/margin settings field exists (grep of `settings.py` 2026-07-23: zero hits) | structurally 1.0x | UNMAPPED -- structurally satisfied (cannot leverage) |
| 3 | `max_daily_loss_pct` | **2%** | `settings.paper_daily_loss_limit_pct` (`settings.py:529`; enforced by `paper_trader.py::check_and_enforce_kill_switch`; hot-mutable in [0.5, 25.0] via `PUT /api/settings`) | **4%** | **DIVERGENT -- live kill-switch fires at DOUBLE the governed loss.** Flagged by the checker |
| 4 | `max_trailing_dd_pct` | 10% | `settings.paper_trailing_dd_limit_pct` (`settings.py:530`) | 10% | CONVERGENT (checker-verified; naive fraction-vs-percent comparison would false-positive this pair) |
| 5 | `max_gross_exposure_pct` | 100% | NONE -- long-only cash-bounded engine; no gross-exposure settings field | structurally <= 100% | UNMAPPED -- structurally satisfied |
| 6 | `max_sector_weight_pct` | 30% | `settings.paper_max_per_sector_nav_pct` (`settings.py:279`, default 30.0; plus the independent COUNT cap `paper_max_per_sector`) | 30% | CONVERGENT (not yet machine-checked -- see follow-up note) |

Checker scope note: `divergence.py` machine-checks pairs 3 and 4 (the two
kill-switch limits named in the masterplan step). Pairs 1, 2, 5, 6 are
documented here from a repo grep on 2026-07-23; extending the checker to
pair 6 (sector) is a natural follow-up once GOV-LIMITS-DECIDE resolves
which source binds. Pair 1 has no single comparable settings attr today.

Additional divergence-adjacent facts (from the 75.8 research gate):
- The six limits have ZERO runtime enforcement consumers -- `load_once()`
  in `main.py:277-286` discards the parsed values and only logs the file
  digest. `limits.yaml` is integrity-protected (GPG tag + mutation
  watcher kills the process) but its VALUES bind nothing.
- `scripts/governance/lint_limits_usage.py` already marks
  `settings.paper_daily_loss_limit_pct` as a legacy attr that "should be
  migrated to the immutable snapshot (phase-4.9 follow-up)" -- a
  migration that never happened. The lint runs WARN-only today.
- The live kill-switch threshold is hot-mutable at runtime via
  `PUT /api/settings` (`settings_api.py:165`, range [0.5, 25.0]) while the
  governed value requires a GPG-signed git tag -- the two sources have
  opposite change-control regimes.

## DRAFT operator token -- GOV-LIMITS-DECIDE (not active until Peder issues it)

```
GOV-LIMITS-DECIDE: <one of>
  BIND-GOVERNED   -- limits.yaml values become enforcement source of
                     truth: settings.paper_daily_loss_limit_pct snaps to
                     2.0 (and future kill-switch reads route through the
                     governance snapshot). Tightens the daily kill-switch
                     2x; requires confirming the 4% operating history
                     (phase-66+ live trading) tolerates 2%.
  BIND-SETTINGS   -- settings values are authoritative: limits.yaml is
                     re-tagged (GPG limits-rotation-YYYYMMDD) to
                     max_daily_loss_pct: 0.04 so governance matches
                     reality. No behavior change; governance doc honesty.
  HOLD            -- keep the WARNING-only divergence log; decide later.
Optionally append LINT-FAIL to flip scripts/governance/
lint_limits_usage.py from WARN to fail (stays operator-gated per the
masterplan step text either way).
```

Recommendation (advisory only): BIND-SETTINGS for the daily-loss pair --
4% has been the live operating value through the phase-66+ trading
window and the 2% governed value predates that history; re-tagging the
YAML is the honest low-risk close. BIND-GOVERNED is the conservative
alternative if the operator wants the institutional-canon 2%.

Until the token is issued, the only runtime effect of phase-75.8 is one
startup WARNING line per divergent pair.
