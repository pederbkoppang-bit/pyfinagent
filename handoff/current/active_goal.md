# Active Goals -- goal-away-ops (primary) + goal-phase61-churn-integrity (in flight)

Refreshed 2026-06-12 by step 62.0. Operator away ~2026-06-15 .. ~2026-07-06.

## Binding rails

docs/runbooks/away-ops-rules.md -- read FIRST in every session. Headline: bug fixes live /
behavior changes dark+token; $0 new metered spend; kill-switch stays paused absent
`KILL SWITCH: RESUME`; masterplan freeze (phases 62-65 + existing backlog only); one step
per AM session; `HALT-DEV` honored before any step work.

## goal-away-ops (phases 62-65; payload + constraints: handoff/current/goal_away_ops.md)

Calendar authority: handoff/away_ops/approved_plan_2026-06-12.md "Calendar" table.
- phase-62 away infra: PRE-DEPARTURE, ends with the 62.7 dress rehearsal (operator watching)
- phase-63 live audit -> defect register -> fixes (week 1 + rolling AM fix slots)
- phase-64 test matrix (weekends)
- phase-65 all-markets proof (65.1/65.2 week 1; 65.4 wall-clock-gated week 3)

## goal-phase61-churn-integrity (phases continue inside the away window)

61.1 PARTIAL: criteria 1-3 done (flags ON + restarts), criterion 4 = first post-flag cycle
BQ evidence (2026-06-12 18:00 UTC cycle), then fresh Q/A -> flip. Then 61.2 -> 61.3 ->
61.4 -> 61.5 (dark; FEE TABLE / TURNOVER LEVERS tokens). Full spec:
handoff/current/goal_phase61_churn_integrity.md.

## Prior goal residue

phase-58.1 ($25 live window, operator-approved 06-11) self-closes on its window evidence;
MUST NOT be disturbed. FRED key rotation = return-day ask (operator deferred).

## Token mechanics (after 62.2 ships)

Operator replies in the bot channel; bot appends to handoff/operator_tokens.jsonl;
sessions apply new tokens FIRST, then advance handoff/away_ops/tokens_cursor (mtime opens
the .env hook gate 6h). Open asks + exact reply strings:
handoff/away_ops/pending_tokens.json.

## Cycle ledger

- 2026-06-12: goal-away-ops installed (66cb8bc1); 62.0 in progress (rules file, 10
  backlog deferrals, hook away-patterns, deny mirrors).
