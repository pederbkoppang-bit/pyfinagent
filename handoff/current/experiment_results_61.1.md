# Experiment Results -- Step 61.1 (GENERATE)

**Step:** 61.1 -- Activate the dark fixes + deploy phase-60 code (criterion-4 closure).
**Date:** 2026-06-15 (AM away session, Cycle 66). **State:** complete pending Q/A.
**Change class:** EVIDENCE COLLECTION ONLY -- no code / .env / trading-behavior file edited.

## What was done

Criteria 1-3 were already COMPLETE (live_check_61.1.md A-D, 2026-06-12). This cycle closes
the only open criterion, **C4** (first post-flag-cycle BQ evidence), by recording verbatim
evidence into `live_check_61.1.md` section E.

1. Confirmed the three fixes are live-ON (verification command, below).
2. Identified the first post-flag cycle = `5f15fdbe` (06-12 18:00 UTC, completed, n_trades=0)
   from `handoff/cycle_history.jsonl`. Weekend cycles do not run; it is the only post-flag
   cycle so far.
3. Queried `financial_reports.paper_trades` (ADC Python client -- pinned MCP is US-locked,
   table is us-central1; CLAUDE.md BQ rule 6 fallback) for the two criterion-4 conditions:
   both return 0 rows (zero post-flag swap_for_higher_conviction SELLs; zero post-flag
   executed REJECT trades). Captured the pre-flag contrast (06-08..06-11) showing the audited
   bug (066570.KS REJECT-that-executed + swap churn).
4. Produced the positive activation-test witness: neutralized-.env run = `28 passed`;
   documented + root-caused the 4 plain-run `.env`-bleed failures (test-isolation artifact,
   not a guardrail regression; flagged as a phase-63 defect-register candidate).
5. Disclosed the vacuousness caveat (n_trades=0) explicitly in section E.4.

## Files changed (this cycle)

| File | Change |
|------|--------|
| `handoff/current/research_brief_61.1.md` | created (researcher subagent, research gate) |
| `handoff/current/contract_61.1.md` | created (PLAN) |
| `handoff/current/live_check_61.1.md` | status line + section E updated (PENDING -> COMPLETE) |
| `handoff/current/experiment_results_61.1.md` | this file |

No source file, `backend/.env`, or trading-behavior file touched (verify: `git diff --stat`
shows only `handoff/` paths). Flags were set ON by operator keystroke 2026-06-12 pre-departure
(recorded verbatim in live_check section A) -- no away-window .env change (rails 1/6 clean).

## Verbatim verification command output (immutable command from masterplan)

```
$ python -c "from backend.config.settings import get_settings; s = get_settings(); print('churn_fix', s.paper_swap_churn_fix_enabled, 'data_integrity', s.paper_data_integrity_enabled, 'rj_binding', s.paper_risk_judge_reject_binding)" && test -f handoff/current/live_check_61.1.md
churn_fix True data_integrity True rj_binding True
EXIT_CODE=0
```

=> all three flags ON; live_check_61.1.md exists. Verification command exit 0.

## Criterion-4 evidence (verbatim BQ, summarized; full output in live_check section E)

- 4a (60.2): `WHERE created_at >= '2026-06-12' AND action='SELL' AND reason='swap_for_higher_conviction'` -> **0 rows**.
- 4b (57.1): `WHERE created_at >= '2026-06-12' AND risk_judge_decision='REJECT'` -> **0 rows**.
- All post-flag trades -> **0 rows** (the cycle traded nothing).
- Pre-flag contrast (06-08..06-11) -> 6 rows incl. `2026-06-09 066570.KS BUY swap_buy rj=REJECT` (executed REJECT) + swap-churn SELLs.
- Activation witness: `28 passed` (neutralized env). Plain run: `4 failed, 24 passed` (the 4 = `.env`-bleed, root-caused in live_check E.6).

## Artifact shape

- `live_check_61.1.md`: 7-subsection section E (E.1 cycle ran; E.2 4a; E.3 4b; E.4
  vacuousness; E.5 pre-flag contrast; E.6 positive witness; E.7 verdict basis).
- All five harness files for 61.1 present: research_brief_61.1.md, contract_61.1.md,
  experiment_results_61.1.md, (evaluator_critique_61.1.md to come from Q/A), harness_log
  append (LOG phase).

## Honest limitations for Q/A to weigh

1. **Vacuous pass:** the live BQ evidence is absence-of-violation with n_trades=0, not a
   demonstrated live block. The activation tests (E.6) are the witness that the block fires.
   A defensible alternative verdict is CONDITIONAL pending a non-zero post-flag cycle -- but
   the last 2 cycles were both 0 trades, so waiting may not resolve it.
2. **No live block can ever be queried** (topology: blocks are log-only, not in BQ). So
   "wait for a live block to appear in BQ" is not a satisfiable closure path; the strongest
   live evidence possible is what E.2/E.3 show.
3. **`.env`-bleed test failures** exist in a plain `pytest` run (4 of them). Root-caused as
   test-isolation, not a guardrail regression; phase-63 candidate; not fixed here.
