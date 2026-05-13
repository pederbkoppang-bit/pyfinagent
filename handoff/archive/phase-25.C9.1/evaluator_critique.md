---
step: 25.C9.1
slug: orchestrator-batchclient-routing
cycle: 103
cycle_date: 2026-05-13
verdict: PASS
agent: qa
---

# Q/A Critique -- phase-25.C9.1

## Harness-compliance audit (5 items)

1. **researcher spawned?** YES. `handoff/current/research_brief.md`
   tier=moderate, 6 sources fetched in full (Anthropic batch docs +
   5 practitioner blogs incl. dotzlaw.com 2026 + Finout 2026 pricing
   guide), recency scan present, `gate_passed=true`.
2. **Contract before generate?** YES. `handoff/current/contract.md`
   step=25.C9.1, immutable criteria copied verbatim, references
   research_brief.md.
3. **experiment_results present?** YES, lists 4 touched files +
   verbatim verification output.
4. **Masterplan status pending?** YES. 25.C9.1 entry inserted with
   `status: pending`; will flip to `done` after Q/A PASS + log append.
5. **No verdict-shopping?** First spawn. No prior CONDITIONAL on this
   step-id in `handoff/harness_log.md`.

## Deterministic checks

| Check | Result |
|-------|--------|
| `python3 tests/verify_phase_25_C9_1.py` | ALL 7 PASS |
| AST parse `backend/agents/orchestrator.py` | OK |
| AST parse `backend/config/settings.py` | OK |
| grep symbols | present at settings.py:76, orchestrator.py:328,336,338,569,625,647 |

Verbatim verification output:

```
[PASS] 1. settings_carries_backtest_batch_mode_flag
[PASS] 2. orchestrator_constructor_accepts_backtest_mode_and_n_tickers
[PASS] 3. orchestrator_run_enrichment_batch_dispatches_via_batchclient
[PASS] 4. gate_evaluates_true_when_backtest_and_n_tickers_above_three
[PASS] 5. gate_false_when_n_tickers_at_or_below_three
[PASS] 6. gate_false_when_settings_flag_is_off
[PASS] 7. run_enrichment_batch_invokes_submit_poll_fetch_with_custom_ids
ALL 7 CLAIMS PASS
```

## Immutable success criteria check

| # | Criterion | Status | Evidence |
|---|-----------|--------|----------|
| 1 | `settings_carries_backtest_batch_mode_flag` | PASS | claim 1; settings.py:76 |
| 2 | `orchestrator_constructor_accepts_backtest_mode_and_n_tickers` | PASS | claim 2; args verified via ast |
| 3 | `orchestrator_run_enrichment_batch_dispatches_via_batchclient` | PASS | claims 3 + 7; mock round-trip submit=1 poll=1 fetch=1 |
| 4 | `gate_evaluates_true_when_backtest_and_n_tickers_above_three` | PASS | claims 4-6 cover all three boundary cases |

## LLM judgment

- **Contract alignment:** all 4 immutable criteria proven; files
  enumerated in contract match what was touched.
- **Mutation resistance:** claim 7 is a real behavioral round-trip
  with `MagicMock` BatchClient asserting submit/poll/fetch each
  invoked exactly once AND custom_id pattern matches
  `{ticker}__{agent_name}` per dotzlaw.com 2026 anti-corruption
  finding. Claims 4-6 cover the three gate boundary conditions
  (positive, n_tickers=3 boundary, flag-off boundary).
- **Scope honesty:** explicit deferral of `run_full_analysis()`
  refactor to 25.C9.2 is disclosed and mirrors the 25.C9 split
  pattern (mechanism shipped, caller adoption deferred). Not
  over-claimed.
- **Caller safety:** both new constructor kwargs default to safe
  values (`backtest_mode=False`, `n_tickers=1`); existing
  `AnalysisOrchestrator(settings)` callers see no behavioral change.
  Gate stays False on the live single-ticker API path.
- **North-star alignment:** directly cuts the cost denominator in
  Net System Alpha (50% batch discount, ~95% effective when
  compounded with 1h prompt cache per Finout). Aligned with
  red-line goal-a (lowest cost) and goal-d (cost observability via
  `is_batch=True` cost-tracker tag).

## Verdict

**PASS.** All 4 immutable criteria proven, deterministic checks
green, mutation-resistant behavioral test included, scope deferral
honest, caller safety intact, research-gate satisfied.

```json
{
  "ok": true,
  "verdict": "PASS",
  "reason": "All 4 immutable criteria proven by 7-claim verifier (ALL PASS). Deterministic: syntax OK, verification cmd exit=0, mock round-trip confirms submit/poll/fetch dispatch + custom_id safety pattern. Research gate satisfied (6 sources in full). Scope deferral (run_full_analysis() caller adoption -> 25.C9.2) is disclosed.",
  "violated_criteria": [],
  "violation_details": [],
  "certified_fallback": false,
  "checks_run": ["syntax", "verification_command", "grep_symbols", "research_gate", "contract_alignment", "mutation_test_review", "scope_honesty", "caller_safety"]
}
```
