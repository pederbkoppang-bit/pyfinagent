# Experiment Results -- Step 62.4 (GENERATE; per-step file, rolling slot holds 62.8's closed results)

**Step:** 62.4 -- guardrail/budget sentinel. **Date:** 2026-06-12.

## What was built

1. scripts/away_ops/sentinel.sh: venv-python core; header pins the metered source
   (pyfinagent_data.llm_call_log, SUM(session_cost_usd) @ CURRENT_DATE() UTC -- flat-fee
   claude_code-rail rows log 0 by design so the plain SUM IS the metered figure; LOWER
   BOUND caveat) + baseline_usd=8.00 pinned constant (58.1-ledger daily class; rolling
   mean would false-trip) + test-override rationale. Gates: metered_budget (exit 1),
   flags_match_tokens (exit 1), metered_source_unavailable / flags_reconciliation_
   unavailable (exit 2 -- infra distinct from breach). kill_switch_paused REPORT-ONLY
   (endpoint + audit-replay fallback). Overrides: SENTINEL_TEST_METERED_USD,
   SENTINEL_ENV_FILE, SENTINEL_TEST_BQ_FAIL.
2. scripts/away_ops/flag_baseline.json: grandfather manifest -- 3 phase-61.1
   operator-keystroke flags + 2 pre-away operational flags (PAPER_TRADING_ENABLED,
   PAPER_USE_CLAUDE_CODE_ROUTE -- discovered as false-positives by the sentinel's own
   first live run) + ops-exempt list (AWAY_MODE_ENABLED). Provenance notes inline.
3. run_away_session.sh amendment: AWAY_SESSION_TEST_PREFLIGHT=1 exercises the REAL
   pre-flight (HALT-DEV, sentinel, dirty-tree, prompt selection) with zero git/claude
   side effects (criterion-3 testability).
4. backend/tests/test_phase_62_4_sentinel.py: 9 tests (8 offline + 1 requires_live).

## Verification transcripts (verbatim; criterion 2)

    healthy:        {"metered_llm_usd_today": 0.0, ..., "ok": true}  exit=0
    tamper-metered: SENTINEL_TEST_METERED_USD=99 -> gates_failed=["metered_budget"]  exit=1
    tamper-flag:    SENTINEL_ENV_FILE w/ PAPER_FAKE_UNAUTHORIZED_FLAG=true
                    -> gates_failed=["flags_match_tokens"], warning names the flag  exit=1
    infra:          SENTINEL_TEST_BQ_FAIL=1 -> ["metered_source_unavailable"]  exit=2
    wrapper:        SENTINEL_TEST_METERED_USD=99 AWAY_SESSION_TEST_PREFLIGHT=1
                    -> PREFLIGHT_PROMPT=digest_only; session.log "sentinel FAILED --
                    downgrading to digest-only"
    pytest:         8 passed (offline) + 1 passed (requires_live, real BQ)

## Iterations (honest log)

- First live run: BQ 400 "Unrecognized name: rail" -- the brief's assumed column does
  not exist; real schema is provider/model/session_cost_usd. SQL fixed to the plain
  SUM(session_cost_usd) (flat-fee rows are 0 by design; verified by a 7-day provider
  breakdown: anthropic n=28 usd=0.0, gemini n=11 usd=0.7).
- Same run false-positived PAPER_TRADING_ENABLED + PAPER_USE_CLAUDE_CODE_ROUTE as
  unauthorized -- pre-away operational state; grandfathered with provenance note.
- chmod +x was missing on sentinel.sh -- the wrapper's -x check read it as "sentinel
  missing" (fail-closed worked as designed, but for the wrong reason). Fixed; both
  preflight paths re-proven (breach -> digest_only; healthy -> sentinel-pass -> next
  gate [recovery, because this step's own uncommitted files made the tree dirty -- the
  full healthy->am path is re-proven on the clean tree post-commit]).

## Criterion-2 deviation (pre-declared in the contract)

"a synthetic cost row" implemented as the SENTINEL_TEST_METERED_USD override by
researched necessity: insertAll rows sit in the streaming buffer ~30 min un-DML-able and
would inflate the real metered figure all day (self-DoS). The override exercises the
identical gate logic and can only TRIP gates, never mask a breach.
