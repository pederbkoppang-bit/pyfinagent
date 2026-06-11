# Active Goal -- goal-phase61-churn-integrity (installed 2026-06-11)

Set by operator 2026-06-11 (verbatim install decision: "Install + begin 61.1 now
(Recommended)", AskUserQuestion, local session). Full specification:
handoff/current/goal_phase61_churn_integrity.md (the goal prompt).

Prior goal (goal-post-away-review) is NOT fully closed: phase-58.1 (go-live runway,
$25 live window) remains pending and MUST NOT be disturbed by this goal. It closes on
its own 1-2-week window DoD evidence.

## North star
(verbatim masterplan.json::goal) "Ship an Intelligence Engine trading system that
maximizes Net System Alpha = Profit - (Risk Exposure + Compute Burn) by dynamically
shifting capital to the highest-earning strategy, recursively self-improving under hard
risk caps, within a 15-slot daily Claude-routine budget." THIS GOAL'S LENS: the fee
complaint was mis-attributed -- fees were $17.14/8d (0.07% NAV); the real N* drag is the
churn's realized P&L (-$139.83 on <=2-day holds vs +$1,355.22 on long holds) plus
poisoned decision inputs. Activate the already-built dark fixes first, then integrity,
then measured cost-aware turnover policy.

## Scope -- in order
1. 61.1 -- activate dark fixes + deploy phase-60 code (operator flag tokens 60.2/60.3/
   57.1, .env edits, backend restart -- running process predated ALL phase-60 commits --
   frontend kickstart, first post-flag cycle BQ evidence). P0.
2. 61.2 -- decision-input integrity (no synthetic 0.00/HOLD persists, claude_code
   timeout >=150s, company_name fallback, meta-scorer fallback rank-normalization +
   alert + root-cause, dead signal_downgrade path, RiskJudge portfolio context in
   advisory mode). P0.
3. 61.3 -- money-display + currency correctness (latent add-on-buy USD-into-LOCAL fix,
   market-first currency resolution, en-US USD locale policy, P&L staleness honesty,
   per-market MTM decision). P0.
4. 61.4 -- learnings + reports history (SAFE_CAST divergences fix, error-vs-empty API
   distinction, STRING-vs-TIMESTAMP audit, 30D-trend backend score history, SPRINT
   TILE: WIRE|PRUNE token). P1.
5. 61.5 -- cost-aware turnover policy (per-market fee table default OFF, >=5-trading-day
   churn measurement, 55.3-sanctioned min-holding lever ONLY if churn persists,
   per-market alpha-slope estimation; hysteresis banned absent HYSTERESIS: AUTHORIZE). P1.

## Founding principles (non-negotiable)
- Full harness loop per step: researcher FIRST (>=5 sources read in full, recency
  scan) -> contract.md (criteria verbatim) -> GENERATE -> ONE fresh qa -> harness_log.md
  append -> masterplan flip. No self-evaluation; no verdict-shopping.
- Do-no-harm: behavior changes config-gated default OFF with ON-vs-OFF measurement;
  pure bug fixes exempt but regression-tested.
- No flag flips without the operator's verbatim token. Trailing-stop + long-hold engine
  untouchable except the add-on averaging money-safety fix.
- Playwright MCP capture for every UI claim; BQ rows for every data claim.

## Operator tokens pending
- 61.1: `60.2 FLAG: ON|KEEP OFF`, `60.3 FLAG: ON|KEEP OFF`, `57.1 FLAG: ON|KEEP OFF`
  (audit recommendation: all ON)
- 61.4: `SPRINT TILE: WIRE|PRUNE`
- 61.5: `FEE TABLE: ON|KEEP OFF`, `TURNOVER LEVERS: APPROVE <subset>|DECLINE`,
  optional `HYSTERESIS: AUTHORIZE`

## Cycle ledger
- 2026-06-11: goal installed (phase-61.0 commit); 61.1 started same session.
