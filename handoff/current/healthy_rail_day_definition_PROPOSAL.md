# Healthy-rail trading day -- codified definition (PROPOSAL, awaiting ratification)

Status: PROPOSED 2026-07-08 (Fable burn-down session). Ratification pending:
operator token `HEALTHY-RAIL DEF: RATIFY` (or amendment), or Q/A adoption at
the 66.2 evidence milestone. Until ratified, the 66.2 criterion-1(b) day count
is frozen at DAY 0 (the conservative reading).

## Why this exists

Masterplan 66.2 criterion 1(b) requires ">=5 consecutive healthy-rail trading
days with zero BUYs" before the funnel diagnosis can close the step -- but NO
numeric definition of "healthy-rail" exists anywhere in code or artifacts
(verified by the 2026-07-08 sweep, live_check_66.2.md 5c). The three existing
signals CONFLICT on 2026-07-07:

| Signal | Source | 07-07 verdict |
|---|---|---|
| rail_skipped=false AND breaker_tripped=false | cycle_history funnel fields (66.1) | PASS (max fail streak 12 < breaker 20) |
| any rail failure on an all-HOLD day => "pipeline defect: rail down" | scripts/diagnostics/funnel_report.py verdict logic | FAIL |
| cc_rail ok-rate > 90% | research_brief_66.2.md:102 falsifier | FAIL (58/123 = 47.2%) |

## Proposed definition

A calendar day D counts as a **healthy-rail trading day** iff ALL of:

1. A scheduled trading cycle ran on D and wrote a completed cycle_history row
   (holidays/no-cycle days do not count for OR against -- the clock pauses).
2. `rail_skipped == false` AND `breaker_tripped == false` on that row.
3. Per-cycle cc_rail ok-rate >= 90%:
   `COUNT(ok=true) / COUNT(*) >= 0.90` over
   `pyfinagent_data.llm_call_log WHERE agent LIKE 'cc_rail%' AND cycle_id = <D's cycle>`
   (zero rail calls => NOT healthy; the rail must actually have carried load).

"Consecutive" counts over trading days satisfying #1, i.e. weekends/holidays
do not break a streak; an unhealthy trading day RESETS the streak to 0.

## Rationale

- The booleans alone are blind to interleaved failures: 07-07 had 65 failures
  (52.8%) yet never tripped the consecutive-20 breaker (max streak 12) -- a
  day like that produced synthetic HOLDs at 5/5 scale and cannot honestly
  support a "gates correctly reject" conclusion.
- The funnel tool's any-failure rule is too strict the other way: a single
  transient failure among ~120 calls (99% ok) does not invalidate a day's
  gate evidence; requiring 100% would make the 5-day clock nearly unfinishable.
- 90% matches the research brief's pre-registered falsifier (the only number
  anyone wrote down BEFORE the dispute existed) and tolerates the observed
  benign transient rate on healthy days (06-11: 3 fail / 5 ok was NOT healthy;
  06-12: 45/81 fail NOT healthy -- both correctly excluded by this rule).

## Application

- 2026-07-07: NOT healthy (47.2%) -> clock at DAY 0. First candidate day:
  2026-07-08 (evaluated tonight).
- Each live_check_66.2.md funnel row gains a `healthy_day: yes/no (ok-rate)`
  column entry from tonight onward.
- Implementation note (post-ratification, non-urgent): fold this rule into
  funnel_report.py's verdict logic so the tool and the definition cannot
  drift. Read-only diagnostics change; no trading behavior.

## Non-goals

This defines EVIDENCE accounting only. It does not gate trading, does not
change the breaker (register item: rate-alarm proposal is separate), and does
not modify the immutable criterion text (which says "healthy-rail" without a
number -- this document supplies the number the criterion delegates).
