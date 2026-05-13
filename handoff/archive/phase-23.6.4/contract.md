---
step: 25.P
slug: weekly-autoresearch-slack-summary
status: in_progress
cycle_date: 2026-05-13
parent_research_brief: handoff/current/research_brief.md
---

# Contract -- phase-25.P

## Step ID + masterplan reference

`25.P` -- "Weekly autoresearch summary Slack notification"
(P2, harness_required, masterplan dep on 25.F3 = planner typo;
treated as satisfied by 25.F done at cycle 91).

## Research-gate summary

Tier=simple. Brief at `handoff/current/research_brief.md`,
`gate_passed=true`. Near-clone of 25.N (cycle 89).

## Hypothesis

`run_meta_evolution_cycle` returns a structured results dict on
Sunday completions but doesn't emit any Slack signal. Adding a P3
summary alert mirrors 25.N's daily-cycle pattern -- operators get a
once-a-week ping confirming the autoresearch run completed.

## Success criteria (verbatim from masterplan.json)

> `format_autoresearch_summary_in_formatters`
>
> `meta_evolution_cron_emits_slack_on_sunday_completion`

## Plan steps

1. **`backend/slack_bot/formatters.py`** -- add
   `format_autoresearch_summary(results: dict) -> list[dict]` (Block Kit).
2. **`backend/meta_evolution/cron.py::run_meta_evolution_cycle`** -- at
   the end of the function, emit a P3 summary alert via
   `raise_cron_alert_sync(source="meta_evolution", error_type="meta_evolution_weekly_summary", severity="P3", ...)` wrapped in try/except (fail-open).
3. **Verifier** -- `tests/verify_phase_25_P.py` with 5 claims:
   - Claim 1: `format_autoresearch_summary` exists with `results` arg + `list[dict]` return.
   - Claim 2: cron.py imports raise_cron_alert_sync.
   - Claim 3: cron.py emits `severity="P3"` + dedup key `meta_evolution_weekly_summary`.
   - Claim 4: behavioral -- import format_autoresearch_summary + call with sample; assert valid Block Kit shape.
   - Claim 5: behavioral -- patch raise_cron_alert_sync; call run_meta_evolution_cycle (with stubs for sub-modules); assert P3 alert fired.

## Files

| File | Action |
|------|--------|
| `backend/slack_bot/formatters.py` | Add format_autoresearch_summary |
| `backend/meta_evolution/cron.py` | Emit P3 alert on completion |
| `tests/verify_phase_25_P.py` | NEW |

## Verification command (immutable)

```
source .venv/bin/activate && python3 tests/verify_phase_25_P.py
```

## Live-check

`After next Sunday cron, Slack post with champion vs challenger, promotions, regressions`.
Will write `handoff/current/live_check_25.P.md`.

## Risks + mitigations

- **Risk**: cron emits on every invocation (incl. manual test runs).
  **Mitigation**: AlertDeduper's `repeat_hours` suppresses dupes within
  the window.

## References

- `handoff/current/research_brief.md`
- `backend/meta_evolution/cron.py:87-152`
- `backend/slack_bot/formatters.py::format_cycle_summary` (25.N template)
- `.claude/masterplan.json::25.P`
