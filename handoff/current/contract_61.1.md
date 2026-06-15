# Contract -- phase-61.1: Activate the dark fixes + deploy phase-60 code (criterion-4 closure)

**Cycle:** AM away session 2026-06-15 (Mon, W1). **Goal:** goal-phase61-churn-integrity
(head of the phase-61 money chain) under goal-away-ops rails.
**Step status entering cycle:** `pending` (PARTIAL -- criteria 1-3 COMPLETE since 2026-06-12;
only criterion 4 open). This is the FIRST Q/A spawn for 61.1 (prior CONDITIONAL count = 0).

## Research gate (PASSED)

`handoff/current/research_brief_61.1.md` -- `gate_passed: true`. 7 external sources read in
full (floor 5), 17 URLs, last-2-year recency scan performed, 8 internal files inspected
(file:line). Key findings:

1. **Criterion-4 interpretation verified** against `.claude/masterplan.json` phase-61->61.1.
2. **All three guardrails correctly wired + flag-gated** (`getattr(settings,"<flag>",False)`,
   ON=>block / OFF=>byte-identical advisory passthrough): 57.1 at
   `portfolio_manager.py:194-212` + `autonomous_loop.py` reject sites; 60.2 at
   `portfolio_manager.py:471-507` + `autonomous_loop.py:805`; 60.3 at
   `autonomous_loop.py:1948-1959`/`2228-2239`. **No wiring defect.**
3. **Topology finding (load-bearing for the evidence shape):** there is NO queryable
   "generated-but-blocked decisions" table. A correctly-blocked REJECT surfaces only as an
   in-memory `summary["risk_judge_blocked"]` + a `logger.warning` -- NOT persisted to BQ or
   cycle_history. `paper_trades.risk_judge_decision` is written only for EXECUTED trades. So
   "REJECT that executed" (the pre-flag bug) is queryable; "REJECT correctly blocked" is not.
   The criterion-4 live evidence is therefore *necessarily* absence-in-paper_trades.
4. **Weekend cycles confirmed NOT to run** (scheduler `day_of_week="mon-fri"`). Cycle
   `5f15fdbe` (06-12 Fri) is genuinely the ONLY post-flag cycle; next = today Mon 18:00 UTC.
5. **Vacuousness = textbook vacuous-pass** ("no executed trade is a REJECT" is vacuously true
   when zero trades occurred). Literature remedy = an "interesting witness" (positive
   activation example). The 28 passing activation tests ARE that witness. Recommendation:
   **PASS-with-caveat (vacuous-in-prod + activation-tested)** is the supported call;
   unqualified-PASS-that-hides-the-vacuousness is NOT defensible; CONDITIONAL-pending-Mon-cycle
   is a defensible alternative (but the wait may not resolve vacuousness -- last 2 cycles = 0
   trades). Final verdict is Q/A's.

## Hypothesis

The first post-restart daily cycle (`5f15fdbe`, 2026-06-12 18:00 UTC) ran with the three
fixes live-ON and produced ZERO trades, hence zero swap_for_higher_conviction SELLs lacking
same-cycle analysis (60.2) and zero executed REJECT trades (57.1). Combined with the passing
activation tests (the witness that the guardrails *would* block when triggered) and the
pre-flag contrast (06-09 066570.KS REJECT-that-executed; swap-churn SELLs), criterion 4 is
satisfiable as PASS-with-caveat. No code, .env, or trading-behavior change is made this cycle
-- this is evidence collection only.

## Immutable success criteria (copied VERBATIM from .claude/masterplan.json phase-61 -> 61.1)

Verification command:
```
cd /Users/ford/.openclaw/workspace/pyfinagent && source .venv/bin/activate && python -c "from backend.config.settings import get_settings; s = get_settings(); print('churn_fix', s.paper_swap_churn_fix_enabled, 'data_integrity', s.paper_data_integrity_enabled, 'rj_binding', s.paper_risk_judge_reject_binding)" && test -f handoff/current/live_check_61.1.md
```

success_criteria:
1. "the operator's verbatim flag tokens (60.2 FLAG / 60.3 FLAG / 57.1 FLAG, each ON or KEEP OFF) are recorded in handoff/current/live_check_61.1.md and backend/.env matches them exactly; no flag changed without its token"
2. "post-restart, the running uvicorn process start time is later than the phase-60.4 commit timestamp (ps -o lstart vs git log evidence pasted verbatim), proving phase-60.2/60.3/60.4 code is loaded"
3. "frontend kickstarted via launchctl; Playwright capture shows http://localhost:3000/login loads without ChunkLoadError"
4. "first post-restart daily-cycle evidence in live_check_61.1.md as verbatim BQ rows: if 60.2 FLAG: ON, zero swap_for_higher_conviction SELLs of holdings lacking a same-cycle analysis_results row; if 57.1 FLAG: ON, zero executed trades with risk_judge_decision='REJECT'"
5. "handoff/harness_log.md cycle entry appended before the status flip"

live_check requirement: "live_check_61.1.md containing: verbatim operator flag tokens,
ps -o lstart output post-restart vs commit timestamps, Playwright screenshot path for /login,
and first post-flag cycle BQ rows from financial_reports.paper_trades"

## Status of each criterion entering this cycle

- C1 (flag tokens recorded + .env matches): COMPLETE (live_check_61.1.md section A; live
  settings print churn_fix=True data_integrity=True rj_binding=True). NO away-window .env
  change made -- flags were set by operator keystroke 2026-06-12 pre-departure (rails 1/6
  satisfied: no flag changed without its token).
- C2 (restart loads phase-60 code): COMPLETE (section D; lstart 2026-06-12 08:05:49 > 60.4
  commit b0fe1983 2026-06-11 16:30:22).
- C3 (frontend /login no ChunkLoadError): COMPLETE (section B; Playwright capture
  2026-06-12 04:12 UTC, zero console errors).
- C4 (first post-flag cycle BQ evidence): **OPEN -- this cycle closes it.**
- C5 (harness_log appended before flip): to be done at LOG phase.

## Plan steps (GENERATE)

1. Record into live_check_61.1.md section E the verbatim BQ evidence:
   (a) the first post-flag cycle `5f15fdbe` row from cycle_history.jsonl (n_trades=0,
       completed, meta_scorer_degraded=true disclosed);
   (b) the verbatim 0-row BQ query result for post-flag `paper_trades` (created_at >=
       2026-06-12) -- both negative conditions satisfied;
   (c) the pre-flag contrast rows (06-09 066570.KS REJECT-that-executed + swap-churn SELLs)
       as the antecedent-can-occur baseline;
   (d) the activation-test witness: `28 passed` under the neutralized-.env run + the plain
       run's 4 .env-bleed failures explained (NOT a guardrail regression).
2. Disclose the vacuousness explicitly (n_trades=0 => guardrails not exercised live; live
   evidence is absence-in-paper_trades + the test witness covers would-block behavior).
3. Note the `.env`-bleed test-isolation defect as a phase-63 defect-register candidate
   (NOT in scope for 61.1 criterion 4; NOT a trading-behavior change).
4. Spawn ONE fresh Q/A (Opus, FABLE-HEADLESS override) to rule PASS/CONDITIONAL/FAIL.
5. Append harness_log.md cycle entry, THEN flip 61.1 status (only if Q/A PASS).

## Rails compliance (goal-away-ops)

- Rail 1/6: NO .env or trading-behavior file edited; flags already ON via operator keystroke,
  recorded verbatim. Evidence-collection only.
- Rail 4 ($0 metered): all checks are LLM-free (BQ read-only bounded queries via ADC Python
  client; offline pytest; cycle_history read). No Gemini/Anthropic API call.
- Rail 2/8: one masterplan step; criteria immutable (copied verbatim, not amended).
- Rail 3: main-only, no force-push/history rewrite; only files this session creates/edits.
- Researcher + Q/A pinned fable -> Opus per-spawn override (FABLE-HEADLESS, non-blocking).

## References

- `.claude/masterplan.json` phase-61 -> 61.1 (immutable criteria source)
- `handoff/current/research_brief_61.1.md` (research gate)
- `handoff/current/live_check_61.1.md` sections A-D (criteria 1-3 evidence)
- `handoff/cycle_history.jsonl` (cycle `5f15fdbe`)
- `financial_reports.paper_trades` (BQ; us-central1)
- Beer et al. FMSD 2001 (vacuity / interesting-witness); NIST AI RMF Measure 2.6
