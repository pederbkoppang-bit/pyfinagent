# Experiment Results — phase-8.5.6 (Promotion path)

2 files: `backend/autoresearch/promoter.py` + `scripts/harness/autoresearch_promotion_test.py`. 3 cases PASS, exit 0. Regression 152/1 unchanged.

```
PASS: shadow_5_trading_days_minimum -- shadow_trading_days >= 5 enforced
PASS: position_size_tied_to_realized_dsr -- position_size scales with realized DSR
PASS: kill_switch_auto_triggers_on_dd_breach -- kill_switch auto-triggers on |dd| > 0.1
PASS EXIT=0
```
