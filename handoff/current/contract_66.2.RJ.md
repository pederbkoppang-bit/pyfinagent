# Contract -- 66.2 RJ-shape fix (operator-approved dark build, 2026-07-08 evening)

Supporting fix for 66.2 (not a standalone masterplan step yet -- see deferral
below). Operator authorization: in-session AskUserQuestion 2026-07-08 ~12:40 UTC
-- "Dark-gated fix now" (build the judge-aware fix + zero-falsy guard on Opus as
a full harness step, config-gated DEFAULT OFF).

## Research basis (money-engine audit wf_e26ca01b-6c6, adversarially verified;
   re-confirmed inline on Opus 2026-07-08 evening)
- risk_debate.py:310 returns `{"judge": judge_result, "analysts": [...]}`; the
  full-orchestrator path sets this as final_synthesis.risk_assessment
  (orchestrator.py:2256 -> :2251), so decide_trades receives
  analysis["risk_assessment"] = {"judge": {decision, recommended_position_pct,
  ...}}.
- portfolio_manager._extract_position_pct (:679) + the REJECT gate (:237) +
  the recorded risk_judge_decision (:262) read risk_assessment TOP-LEVEL ->
  full-path BUYs sized at the 10%-NAV default, REJECT unenforceable even with
  reject_binding ON, risk_judge_decision persisted ''.
- api/analysis.py:158 + tasks/analysis.py:162 already resolve
  risk_assessment.get("judge", {}) -- the manual/API consumers are correct;
  decide_trades is the drifted one.
- Zero-falsy: `if pct:` (:683) + `cand["position_pct"] or 10.0` (:372) invert an
  explicit judge 0.0 into the 10% default (audit finding C4).
- Lite path is FLAT (top-level decision/pct) -> unaffected by resolving
  nested-first.

## Immutable-criterion linkage
Directly serves 66.2 criterion 1(a): "first BUY ... with risk_judge_decision
recorded" -- today (2 cycles) any full-path BUY would persist '' and FAIL that
wording. This fix makes a clean 1(a) close POSSIBLE once promoted.

## Change (all flag-gated `paper_risk_judge_shape_fix_enabled`, default OFF)
1. settings.py: new bool Field(False) + settings_api _FIELD_TO_ENV entry.
2. decide_trades: resolve `_rj_view` nested-first when ON; sizing, REJECT gate,
   recorded decision, and the info log all read `_rj_view`; explicit 0.0 pct
   respected (is-not-None) in both _extract sizing and the emit-loop `or 10.0`.
3. OFF path is byte-identical top-level reads (asserted). Lite byte-identical
   across the flag (asserted).

## Scope boundaries
Main emit path only. Swap-path sizing (:642/:667 `or 10.0`) left as a register
item (swap requires an existing position to displace; unreachable on the
current 100%-cash book; touching swap-delta math risks the 53.1/55.3-protected
territory). Trailing-stop engine untouched. No masterplan step added tonight.

## Verification
`python -m pytest backend/tests/test_phase_66_2_risk_judge_shape.py -q` -> 8
passed (OFF legacy 10%/'' asserted; ON judge-sizing + decision-recorded +
0%-no-buy + lite-unaffected + lite-byte-identical asserted). Combined 61.2+66.2
immutable set: 59 passed. dod4 flat-dict _extract_position_pct callers: 7
passed (signature preserved).

## Deferrals (honest)
- Fresh Q/A EVALUATE deferred to NEXT session: this session's qa subagent spawns
  from the session-start Fable roster snapshot, and the operator moved off Fable
  (/model -> Opus 4.8). The Opus-pinned qa takes effect next session
  (verify_qa_roster_live.sh). Deploying dark (flag OFF) tonight is safe without
  the verdict; PROMOTION (flag ON) must wait for the fresh Q/A + operator token.
- Formalize as a masterplan step (66.2.1 or fold into 61.x) next session.
- Pre-existing red test noted: test_portfolio_swap.py::test_swap_framework_
  fills_zero_buy_gap fails on this morning's committed code (07b7d9c3) too --
  NOT a regression from this fix or 61.2; register for separate triage.
