# Experiment Results -- Phase 4.4.6.3 First-Week Monitoring

**Date:** 2026-04-21
**Cycle:** 32

## Changes

### 1. `backend/config/settings.py`
- Added `first_week_mode: bool = Field(False, ...)` setting
- Toggled via `FIRST_WEEK_MODE=true` env var at go-live

### 2. `backend/services/sla_monitor.py`
- Added `from backend.config.settings import get_settings` import
- Modified `get_sla_thresholds()` to tighten P3 response from 4h to 1h when `first_week_mode=True`
- P3 resolution also tightens from 24h to 8h in first-week mode
- P0/P1/P2 unchanged (already tight)

### 3. `backend/agents/mcp_servers/signals_server.py`
- Added 2-line override in `track_drawdown()` after reading thresholds from `get_risk_constraints()`
- When `self.settings.first_week_mode` is True, `derisk_pct` overridden to `warn_pct` (-5.0)
- `get_risk_constraints()` NOT modified (4.4.4.4 compliance preserved)
- Kill switch at -15% unchanged in both modes

### 4. `scripts/go_live_drills/first_week_monitoring_test.py` (new)
- 15 scenarios: 8 AST-level checks + 7 runtime checks
- Tests both normal and first-week modes
- Verifies 4.4.4.4 compliance (hardcoded risk limits unchanged)

## Drill output
```
PASS S0 first_week_mode field exists in settings.py
PASS S1 first_week_mode defaults to False
PASS S2 sla_monitor.py imports get_settings
PASS S3 sla_monitor.py has first_week conditional branch
PASS S4 SLA P3 normal response = 4 * 3600 (4h)
PASS S5 SLA P3 first-week response = 60 * 60 (1h)
PASS S6 track_drawdown has first_week_mode override
PASS S7 get_risk_constraints unchanged (4.4.4.4 compliant)
PASS S8 normal mode: -9.5% drawdown -> tier=warning (not derisk)
PASS S9 normal mode: -10.0% drawdown -> tier=derisk
PASS S10 first-week mode: -5.0% drawdown -> tier=derisk (tightened from -10%)
PASS S11 first-week mode: -4.0% drawdown -> tier=ok
PASS S12 first-week mode: -15.0% drawdown -> kill (unchanged)
PASS S13 normal mode: -15.0% drawdown -> kill (baseline)
PASS S14 get_risk_constraints literals unchanged (4.4.4.4 verified)
DRILL PASS: 15/15 first-week monitoring scenarios verified
```

## Activation recipe
```bash
# At go-live (day 1):
export FIRST_WEEK_MODE=true
# Restart backend

# After day 7 (if live Sharpe tracks paper Sharpe):
unset FIRST_WEEK_MODE
# Restart backend
```
