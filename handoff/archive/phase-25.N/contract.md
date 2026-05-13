---
step: 25.N
slug: cycle-completion-slack-summary
status: in_progress
cycle_date: 2026-05-13
parent_research_brief: handoff/current/research_brief.md
---

# Contract -- phase-25.N

## Step ID + masterplan reference

`25.N` -- "Cycle-completion summary Slack notification"
(P2, harness_required, depends on `25.A2` done).

## Research-gate summary

Tier=simple. Brief at `handoff/current/research_brief.md`,
`gate_passed=true`. Pattern mirrors the existing failure-path
`raise_cron_alert_sync` wire at `autonomous_loop.py:614-639` and the
25.R Block Kit formatter.

## Hypothesis

The autonomous_loop currently posts a P1 Slack alert only when a
cycle DOESN'T complete (`status not in ("completed", "skipped")`).
When the cycle completes successfully the operator has NO Slack
signal -- only logs. By emitting a P3 cycle-summary alert with
duration / trades / stops / mode, the operator gets a once-per-cycle
ping confirming the loop is healthy.

## Success criteria (verbatim from masterplan.json)

> `format_cycle_summary_function_in_formatters`
>
> `autonomous_loop_emits_slack_at_cycle_completion`

## Plan steps

1. **`backend/slack_bot/formatters.py`** -- add `format_cycle_summary(summary: dict) -> list[dict]`
   returning Block Kit blocks (header + fields + context). Mirrors 25.R's
   `format_strategy_switch` style.
2. **`backend/services/autonomous_loop.py`** finally block -- after the
   existing `_final_status` check, add a sibling branch:
   `if _final_status == "completed":` emit a P3 summary via
   `raise_cron_alert_sync(source="autonomous_loop", error_type="cycle_completed_summary", severity="P3", title=..., details=...)`.
   The dedup key error_type="cycle_completed_summary" is distinct from the failure-path
   "cycle_<status>" so it never collides.
3. **Verifier** -- `tests/verify_phase_25_N.py` with 5 claims:
   - Claim 1: `format_cycle_summary` exists in formatters with the right signature.
   - Claim 2: formatter returns Block Kit blocks (`list[dict]`) with header + section.
   - Claim 3: autonomous_loop imports `raise_cron_alert_sync`.
   - Claim 4: autonomous_loop has a `cycle_completed_summary` branch on success.
   - Claim 5: behavioral round-trip -- call format_cycle_summary with a sample
     summary and assert the block structure is valid.

## Files

| File | Action |
|------|--------|
| `backend/slack_bot/formatters.py` | Add `format_cycle_summary` |
| `backend/services/autonomous_loop.py` | Emit P3 alert on cycle_completed |
| `tests/verify_phase_25_N.py` | NEW verifier |

## Verification command (immutable)

```
source .venv/bin/activate && python3 tests/verify_phase_25_N.py
```

## Live-check

`Slack post after each completed autonomous cycle with duration, trades, stops, mode`.
Will write `handoff/current/live_check_25.N.md`.

## Risks + mitigations

- **Risk**: Dedup collision with the existing failure-path alert.
  **Mitigation**: New `error_type="cycle_completed_summary"` is a distinct dedup
  key; AlertDeduper uses (source, error_type) tuples.
- **Risk**: One Slack post per cycle = noisy.
  **Mitigation**: dedup with `repeat_hours=24` defaults to the existing pattern;
  but for a per-cycle digest we want to actually post each cycle. Set severity=P3
  so the deduper treats it as low-priority and doesn't suppress.

## References

- `handoff/current/research_brief.md`
- `backend/services/autonomous_loop.py:614-639` (existing P1 path)
- `backend/slack_bot/formatters.py:679-757` (25.R template)
- `.claude/masterplan.json::25.N`
