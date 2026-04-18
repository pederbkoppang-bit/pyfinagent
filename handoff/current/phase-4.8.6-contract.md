# Contract -- Cycle 83 / phase-4.8 step 4.8.6

Step: 4.8.6 Disaster-recovery runbooks + tabletop drills

## Hypothesis

Three disaster scenarios with the highest probability x impact for
pyfinagent live paper trading:
1. **broker_outage** -- Alpaca paper API unreachable or 5xx.
2. **data_feed_outage** -- yfinance / BQ historical_prices stale or
   500-ing; live prices unavailable.
3. **llm_outage** -- Anthropic / Vertex AI / OpenAI 5xx or auth
   failure; agent pipeline stalls.

Each gets a runbook with a required structure:
- ## Scope
- ## Trigger (how do we detect it)
- ## Response Steps (numbered; first step <=1min, last step
  stabilizes the system)
- ## Rollback
- ## RTO Target (minutes)
- ## Last Drill (date + measured RTO + verdict)

Plus `handoff/dr_drill_log.md` appending one tabletop-drill entry
per scenario with (date, participants, scenario-injection, actions
taken, measured RTO, RTO target, pass/fail).

## Scope

Files created:

1. **NEW** `docs/runbooks/broker_outage.md`
2. **NEW** `docs/runbooks/data_feed_outage.md`
3. **NEW** `docs/runbooks/llm_outage.md`
4. **NEW** `handoff/dr_drill_log.md` (initial drill entries for
   today's tabletop sessions)
5. **NEW** `scripts/audit/dr_runbooks_audit.py`
   Verifies each runbook has the six required sections + the drill
   log has 3 scenario entries each with an rto_actual_minutes value.

## Immutable success criteria

1. three_runbooks_landed -- all three files exist under
   docs/runbooks/.
2. three_tabletop_drills_logged -- dr_drill_log.md has 3 entries,
   one per scenario.
3. rto_per_scenario_measured -- each entry has a numeric
   rto_actual_minutes AND a rto_target_minutes.

## Verification (immutable, from masterplan)

    for f in broker_outage data_feed_outage llm_outage; do
      test -f docs/runbooks/$f.md || exit 1
    done && test -f handoff/dr_drill_log.md

Plus: `python scripts/audit/dr_runbooks_audit.py --check` -> PASS.

## Anti-rubber-stamp

qa must verify:
- Runbooks are SUBSTANTIVE (not placeholder 3-line stubs). Each
  Response Steps section has >=4 numbered steps.
- RTO targets are realistic (5-30 min for broker, not 0 or
  unbounded).
- The drill log's measured RTOs are plausible relative to targets
  (not auto-PASS by setting actual=target-1 for everything).
- Rollback sections describe a REAL rollback path (kill-switch,
  flatten-all, EXECUTION_BACKEND flip, etc.) referencing existing
  code.

## References

- Google SRE book ch.14 "Managing Incidents"
- Alpaca status page https://status.alpaca.markets
- backend/services/kill_switch.py + execution_router.py
  (existing rollback primitives)
