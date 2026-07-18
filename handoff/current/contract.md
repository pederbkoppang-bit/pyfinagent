# Contract — phase-72.1: P1 approved-but-unapplied operator token audit

**Step id:** 72.1 (phase-72, depends_on 72.0 = done/PASS @7b2499e3)
**Session role:** Fable 5 + ultracode, AUDIT + RESEARCH ONLY. No product code, no .env, no flag flips, $0 metered.

## Research-gate summary (gate_passed: true)

Researcher via structured-output Workflow `wf_ce9e1cac-e72` (opus/max, tier=simple — depth knob only; the 5-source floor held: 6 external sources read in full, 19 URLs, recency scan, 14 internal files). Brief: `handoff/current/research_brief_72.1.md`. Structured `token_inventory` returned: **15 rows** covering every `operator_tokens.jsonl` line + every owed/derived token found by grep (harness_log, pending_tokens.json, flag_promotion_brief, masterplan phase-69, runbooks).

Load-bearing findings:
1. **Exactly one approved-and-recorded token gates a NOT-live flag**: PROMOTE SYNTHESIS-INTEGRITY + RJ-SHAPE (operator_tokens.jsonl:1, 2026-07-09) → `paper_synthesis_integrity_enabled` (settings.py:197) + `paper_risk_judge_shape_fix_enabled` (settings.py:311), both default False, **double-blocked** (never written to the agent-locked `.env` AND the backend process pid 98681 started 07-08 23:24, predating the approval — even a correct write is inert until restart).
2. **The 06-11 keystroke batch IS applied and loaded**: 60.2 `paper_swap_churn_fix_enabled` (runtime-corroborated by 70.3 test + 65.3 BQ 0-churn), 60.3 `paper_data_integrity_enabled`, 57.1 `paper_risk_judge_reject_binding` (harness_log:26954, live_check_61.1.md:12-14; predate the 07-08 restart).
3. **Owed-not-approved ≠ deployment gap**: KS-PEAK-RESET, sign_safe_overlays, regime_net_liquidity are correctly dark (tokens never issued). historical_macro un-freeze / KILL SWITCH: RESUME / FABLE PERMANENT are not settings flags (posture / process action / agent-file pin); KILL SWITCH: RESUME is not currently owed (never paused since 06-11).
4. **Structural root of the gap**: `operator_tokens.py:52-61` KNOWN_TOKEN_ENV_MAP contains only 'AWAY DRILL' — no flag key registered, so the bot has no auto-apply path; `sentinel.sh:102-126` reconciles **one-directionally** (catches unauthorized ON, blind to approved-but-unapplied). External GitOps/flag-governance consensus prescribes a bidirectional intent-vs-live reconciliation loop → recommend a report-only reverse leg as a future executor-tagged step.
5. **Necessary-not-sufficient caveat**: promoting the 07-09 flags lets BUYs survive rail hiccups but does not fix the P0 credit exhaustion or the meta-scorer bypass.

## Hypothesis

The token backlog contains exactly one true approval-to-deployment gap (the 07-09 bundle); documenting every line with an applied-verdict and emitting exact one-line `.env` changes converts the gap from tribal knowledge into an operator-actionable sheet, and a reverse-leg reconciliation step prevents recurrence.

## Immutable success criteria (verbatim from .claude/masterplan.json step 72.1)

- "Every line of handoff/operator_tokens.jsonl is reconciled: token verbatim, approval date, gated flag(s), code default, live .env state (or UNCONFIRMED + the grep needed), applied-or-not verdict"
- "Each approved-but-dark lever appears in operator_decision_sheet_72.md as one actionable line with the exact .env change"
- "No flag was flipped and backend/.env was not modified by this session"

verification.command: `bash -c 'test -f handoff/current/operator_decision_sheet_72.md && grep -Eqi "SYNTHESIS.?INTEGRITY" handoff/current/operator_decision_sheet_72.md && grep -Eqi "token" handoff/current/operator_decision_sheet_72.md'`

## Plan

1. GENERATE (Main authorship from the verified inventory): create `handoff/current/operator_decision_sheet_72.md` with the P1 token-reconciliation table (all 15 rows; live state UNCONFIRMED-marked where only documentary/runtime inference exists — the operator `.env` grep remains unprovided) + the actionable-lines block (exact `.env` lines + restart requirement). Update `money_diagnosis_72.md` §P1. Recommend (not install yet — belongs to the sheet) the reverse-leg sentinel reconciliation as a future executor-tagged step; queue it as a masterplan step in this GENERATE since it is remediation surfaced by this audit.
2. `experiment_results.md` with verbatim verification output.
3. EVALUATE via qa-verdict Workflow; transcribe verbatim.
4. LOG (Cycle 113) then flip 72.1 → done.

## References

- `handoff/current/research_brief_72.1.md` (envelope + per-source notes; GitOps/flag-governance sources)
- `handoff/current/money_diagnosis_72.md`, `handoff/current/live_check_72.0.md` (72.0 verified baseline)
- `handoff/operator_tokens.jsonl`, `handoff/away_ops/pending_tokens.json`, `handoff/current/flag_promotion_brief_2026-07-09.md`, `backend/slack_bot/operator_tokens.py:52-61`, `scripts/away_ops/sentinel.sh:102-126`
