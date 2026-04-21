# Sprint Contract — phase-10 / 10.1 (Sprint calendar config)

**Step id:** 10.1 **Cycle:** 1 **Date:** 2026-04-20 **Tier:** simple (closure)

## Research-gate summary

Closure audit at `handoff/current/phase-10.1-research-brief.md`. gate_passed: true. YAML on disk carries `new_weekly_slots: 2`, both `thursday` + `friday`, monthly_anchor with HITL=true, ASCII clean.

## Hypothesis

`backend/autoresearch/sprint_calendar.yaml` already satisfies the immutable + all 4 success_criteria. No new edits.

## Immutable criterion

- `test -f backend/autoresearch/sprint_calendar.yaml && python -c "import yaml; d=yaml.safe_load(open('backend/autoresearch/sprint_calendar.yaml')); assert d['new_weekly_slots'] == 2 and 'thursday' in d['days'] and 'friday' in d['days']"`

## Plan

1. Re-run immutable command. Capture verbatim.
2. Confirm all 4 success_criteria via Python inspection of the YAML.
3. Write `phase-10.1-experiment-results.md`.
4. Spawn Q/A.
5. Log-last + flip.

## References

- `handoff/current/phase-10.1-research-brief.md`
- `backend/autoresearch/sprint_calendar.yaml`
- `.claude/masterplan.json` → phase-10 / 10.1
