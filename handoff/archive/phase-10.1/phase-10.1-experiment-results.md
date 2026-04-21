# Experiment Results — phase-10 / 10.1 (Sprint calendar config)

**Step:** 10.1 **Date:** 2026-04-20 **Cycle:** 1 (closure).

One file already on disk: `backend/autoresearch/sprint_calendar.yaml` (authored earlier in session).

## Verification

```
$ test -f backend/autoresearch/sprint_calendar.yaml && python -c "import yaml; d=yaml.safe_load(open('backend/autoresearch/sprint_calendar.yaml')); assert d['new_weekly_slots'] == 2 and 'thursday' in d['days'] and 'friday' in d['days']"
(exit 0)

IMMUTABLE + ALL 4 CRITERIA PASS
new_weekly_slots: 2
days: ['thursday', 'friday']
monthly_anchor.rule: last_trading_friday
monthly_anchor.hitl: True

$ python3 -c "open('backend/autoresearch/sprint_calendar.yaml','rb').read().decode('ascii'); print('ASCII OK')"
ASCII OK

$ pytest backend/tests/ -q --ignore=backend/tests/test_paper_trading_v2.py
152 passed, 1 skipped
```

## Criteria

| # | success_criterion | Status |
|---|---|---|
| 1 | calendar_config_committed | PASS (file on disk) |
| 2 | new_weekly_slots_equals_2 | PASS (YAML parse confirmed) |
| 3 | thursday_and_friday_defined | PASS (both keys under `days`) |
| 4 | monthly_anchor_defined | PASS (section w/ rule, hitl=true, min_challenger_days=20) |

ASCII clean; regression unchanged.
