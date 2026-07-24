# Operator decision note -- 75.14: schemas deliberately NOT extended

Date: 2026-07-24. Required by 75.14 criterion 3.

## The decision taken in 75.14

The four prompt-vs-schema seams were reconciled by ALIGNING THE PROMPTS to
the enforced Pydantic schemas (`RiskAnalystArgument` 3 fields;
`RiskJudgeVerdict` without `unresolved_risks`; `DevilsAdvocateResult`
without `bull/bear_weakness` and with boolean `groupthink_flag`;
`ModeratorConsensus` without `bull_case`/`bear_case`/`winner`). The
schemas, and the `debate.py:327-328` backfill that feeds real bull/bear
text to downstream consumers, are byte-untouched. This is the
behavior-STABILIZING arm: on Gemini the schema already hard-dropped the
extra promised fields; on the live Claude soft-schema path the prompts no
longer fight the schema instruction riding the same request.

## What EXTENDING the schemas instead would change (why it needs an operator call)

1. **Sizing inputs change.** Extending `RiskAnalystArgument` (catalysts,
   tail risks, hedging as structured fields) and `RiskJudgeVerdict`
   (`unresolved_risks`) would put analyst evidence in front of the Risk
   Judge and its verdict consumers as STRUCTURED data. Anything that flows
   into `recommended_position_pct` is a decision-changing input on the
   money path -- exactly what the 75.14 boundary forbids without explicit
   sign-off.
2. **Frontend impact (research finding, cuts BOTH ways).** Three formerly
   promised fields are LIVE frontend-rendered when present:
   `unresolved_risks` (RiskDashboard.tsx:429), `bull_weakness`/`bear_weakness`
   (DebateView.tsx:42-43; types.ts:303-304/435). On the Claude soft-schema
   path they sometimes populated and displayed. AFTER 75.14's alignment
   these UI sections go PERMANENTLY EMPTY (the components tolerate absence
   -- no crash). Extending the schemas would make them light up reliably
   on BOTH provider rails instead.
3. **Token/latency cost**: structured extras are generated on every debate
   for every ticker; the alignment arm is the cheaper one.

## The token

```
SCHEMA-EXTEND-75.14: <one of>
  EXTEND      -- add unresolved_risks to RiskJudgeVerdict and
                 bull_weakness/bear_weakness to DevilsAdvocateResult (UI
                 fields light up on both rails; Judge sees structured
                 analyst evidence -- sizing-input change, needs its own
                 research-gated step + OOS check)
  KEEP-ALIGNED -- accept the 75.14 state; optionally queue a frontend
                 cleanup step to remove the permanently-empty UI sections
  (no token = KEEP-ALIGNED by default; nothing further ships)
```
