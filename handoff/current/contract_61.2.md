# Contract -- 61.2 Decision-input integrity (goal-phase61-churn-integrity, DARK BUILD)

Step: 61.2 | Cycle 74 | 2026-07-08 | Operator present (Fable burn-down session)

SEQUENCING DISCLOSURE (binding context): operator-authorized EARLY START via the
2026-07-08 /goal ("61.2 DARK BUILD -- OPERATOR AUTHORIZED early start (this
goal)") while 66.2's criterion-1 evidence clock runs. Constraints honored:
(1) code COMMITS to main but the backend is NOT restarted today -- PID 24910
(started 08:44 UTC) is tonight's 18:00 UTC cycle host and holds tonight's
evidence environment untouched; deploy = post-cycle or tomorrow. (2) The
66.2 criterion-2 discipline (zero gate/threshold/sizing changes to manufacture
trades) is unaffected: both new behavior flags ship DEFAULT OFF and are
byte-identical-OFF tested; ungated deltas are reliability/observability fixes
(timeout scalar, NULL read-guards, company_name fallback), none of which is a
gate, cap, limit, or entry criterion. Expected first Q/A verdict: CONDITIONAL
(criteria 3 + live_check need a post-fix DEPLOYED cycle; 39.1 scheduled-run
doctrine) -- a designed intermediate state, not a failure.

## Research-gate summary

research_brief_61.2.md: tier complex, gate_passed TRUE -- 8 sources read in
full (Azure retry, Fowler circuit-breaker, Brooker backoff+jitter, Google SRE
overload, Anthropic errors doc, Tenacity, arXiv 2605.08563 fresh-context
retries, Minimal Modeling sentinel-free schemas), 28 URLs, recency scan (AWS
Oct-2025 retry-storm lesson; 2026 result-based-retry consensus), 19 internal
files. Load-bearing findings:
- TWO fabrication sites: autonomous_loop.py:1596/:1607-1609 (synthetic
  HOLD/0.0 assembly) AND _persist_analysis :2462-2463 (re-coerces NULL back to
  0.0/"Hold") -- fixing only one is insufficient.
- The existing lite-fallback + 60.1 fallback-rate-P1 machinery (:1630-1650) is
  UNREACHABLE for synthesis errors (dict return, no raise) -- raising
  SynthesisDegradedError inside the try reuses all of it for free.
- Retry-on-empty belongs at orchestrator level ONLY (one retry layer -- SRE);
  the empty response is classifiable via thoughts prefix: "errored:" =
  retryable, "rail_guard_skipped:" = never retry (open breaker -- Fowler).
  Each attempt re-enters the RailGuard accounting = honest metering + breaker
  acceleration, no RailGuard changes.
- INTERACTION HAZARD: reviving signal_downgrade (criterion 5) makes synthetic
  HOLDs lethal (rail failure on a held ticker -> downgrade SELL of a healthy
  position) -> TWO flags, WARN if downgrade-ON while integrity-OFF.
- Criterion-4 root cause PRE-RESOLVED by Main's BQ archaeology (live_check
  _66.2.md 5d CORRECTION): last GENUINE direct-API success 2026-05-17; all 30
  June-July "ok" rows are test fixtures (1000/50/123.4, writer
  test_observability.py:230); failures never logged; alert site dead until
  66.1. The 06-03..06-10 window is one continuous credit-death span since
  ~05-18. experiment_results.md will document this with the query outputs.

## Immutable success criteria (verbatim from .claude/masterplan.json 61.2)

1. "a synthesis result carrying final_synthesis.error (or missing
   scoring_matrix) is never persisted as a 0.0 final_score with a default
   HOLD: it is either routed to the existing lite fallback or persisted with
   NULL score plus an explicit degraded marker; a regression test simulates
   the timeout and asserts no 0.0/HOLD row is written and the same-cycle
   trade-decision input is not silently neutralized"
2. "claude_code synthesis/critic-class calls run with timeout >= 150s (per
   the file's own recommended_step_timeout) and the value is configurable"
3. "_persist_analysis falls back to the quant company_name when
   market_data.name is absent; live_check shows BQ rows from a post-fix
   autonomous full-path cycle with non-null company_name"
4. "the meta-scorer fallback no longer emits a constant saturated conviction:
   composite scores are rank/percentile-normalized into the 1-10 scale, and a
   WARN-level alert fires after 2 consecutive all-fallback cycles; the root
   cause of the 06-03..06-10 LLM unavailability is diagnosed and documented
   in experiment_results.md"
5. "positions persist the analysis recommendation (not the trade reason) so
   the signal_downgrade rule at portfolio_manager.py:127 can match; covered
   by a unit test"
6. "RiskJudge receives portfolio sector-breakdown context regardless of
   paper_risk_judge_reject_binding"

Verification command (immutable): pytest -k 'synthesis or persist or
downgrade or meta_scorer or 61_2' + test -f live_check_61.2.md.
live_check (immutable): BQ rows from >=1 post-fix autonomous cycle --
non-null company_name on full-path rows, zero new final_score=0.0 AND
final_synthesis.error rows, non-constant convictions in paper_trades.signals.

## Hypothesis

Persisting honest absence (lite-fallback rescue, else NULL + degraded marker)
instead of fabricated 0.0/HOLD, plus a bounded result-based retry on the
stateless rail call, converts transient rail failures from silent BUY
destroyers (5/5 on 0725d2aa; two live BUY/0.62 consensuses lost) into either
recovered analyses or visibly-degraded rows -- without changing any gate.

## Plan (per research_brief_61.2.md Recommended design, adopted in full)

1. settings.py: paper_synthesis_integrity_enabled=False (umbrella: error
   detection -> SynthesisDegradedError -> existing lite fallback; both-fail
   degraded row + return None; _persist_analysis NULL passthrough for marker
   rows; retry-on-empty; RJ advisory ctx; meta-scorer rank-normalization +
   streak WARN), paper_position_recommendation_fix_enabled=False (criterion 5
   + unsafe-combination WARN), claude_code_timeout_s=150 (ungated) +
   recommended_step_timeout = timeout_s+30 instance attr,
   claude_code_empty_retry_max=2. Both bools added to settings_api
   _FIELD_TO_ENV (decision: yes -- operator visibility beats .env-only).
2. Ungated pure fixes: company_name quant fallback (:2461); read-side NULL
   guards (models.py Optional + degraded field, reports.py, types.ts,
   reports-columns.tsx null guard + degraded badge + sparkline null filter,
   formatters.py or-0 guards).
3. Retry-on-empty in _generate_with_retry (result predicate, 1+2 attempts,
   full jitter cap 15s, WARNING/ERROR logging split).
4. Meta-scorer: percentile-rank fallback across head+tail on one scale;
   heartbeat-file streak counter; WARN alert at streak 2.
5. Tests test_phase_61_2_*.py per the brief's 7-item plan (incl. flag-OFF
   byte-identical legacy assertions + criterion-1 immutable regression shape).
6. Run the immutable pytest command + full 60.4/62.4/66.x regression set;
   IMPORT_OK syntax gate; commit (NO restart, NO status flip).
7. experiment_results.md incl. criterion-4 root-cause documentation with BQ
   query outputs; live_check_61.2.md created with the DEPLOY-PENDING evidence
   plan (criteria 1/2/4/5/6 test evidence now; criterion 3 + live_check BQ
   evidence after first post-deploy cycle).
8. ONE fresh Q/A (Fable) on the artifacts -- expected CONDITIONAL pending
   deploy evidence; then harness_log append. NO masterplan flip today.

## Scope boundaries

No gate/threshold/sizing/entry changes (66.2 criterion-2 + 61.2 boundary);
trailing-stop engine untouched; hysteresis untouched; no backend restart
today; no .env writes; flags default OFF; no BQ writes outside tests' own
fixtures (and NOT to prod tables -- use mocks; the test_observability.py:230
prod-pollution pattern is the anti-pattern, register item).

## References

research_brief_61.2.md (8 full sources + envelope); live_check_66.2.md 5b/5d
(+CORRECTION); masterplan 61.2; goal_phase61_churn_integrity.md; Azure retry
pattern; Fowler CircuitBreaker; Brooker exponential-backoff-and-jitter; Google
SRE handling-overload; Anthropic API errors doc; Tenacity retry_if_result;
arXiv 2605.08563; Minimal Modeling sentinel-free schemas.
