---
step: phase-25.C9
cycle: 84
cycle_date: 2026-05-13
agent: qa (merged qa-evaluator + harness-verifier)
verdict: PASS
violated_criteria: []
certified_fallback: false
checks_run: [syntax, verification_command, harness_log_scan, contract_alignment, research_gate_compliance, mutation_resistance, scope_honesty]
---

# Q/A Critique -- phase-25.C9 (Adopt Batch API for non-interactive pipeline steps)

## 5-item harness-compliance audit

1. **Researcher spawn** -- CONFIRM. `handoff/current/research_brief.md` carries `authored_by: researcher-agent`, header `phase-25.C9`, gate envelope: `external_sources_read_in_full=6`, `urls_collected=14`, `recency_scan_performed=true`, `internal_files_inspected=4`, `gate_passed=true`. Three-variant search queries listed. Floor (>=5 in full) cleared at 6.
2. **Contract pre-commit** -- CONFIRM. `handoff/current/contract.md` step phase-25.C9 present, immutable success criteria copied verbatim (the three criteria match the masterplan).
3. **Results captured** -- CONFIRM. `handoff/current/experiment_results.md` contains verbatim 12/12 PASS verifier output.
4. **Log-last** -- CONFIRM. `grep -c "phase-25.C9" handoff/harness_log.md` = 0 (the "Next cycle candidate" mention does not count as a cycle header). Append to be performed by Main AFTER this PASS verdict + BEFORE status flip.
5. **No verdict-shopping** -- CONFIRM. First Q/A spawn for 25.C9 (no prior `evaluator_critique.md` for this step in `handoff/archive/phase-25.C9/`; no prior CONDITIONAL/FAIL for 25.C9 in `handoff/harness_log.md`).

## Deterministic checks

### Verification command (immutable)
```
$ source .venv/bin/activate && python3 tests/verify_phase_25_C9.py
PASS: batchclient_wrapper_implemented_in_llm_client
PASS: batchclient_submit_signature
PASS: batchclient_poll_signature
PASS: batchclient_fetch_signature
PASS: agent_cost_entry_is_batch_field
PASS: cost_tracker_record_accepts_is_batch_kwarg
PASS: cost_tracker_records_is_batch_true_for_50_percent_pricing
PASS: steps_1_through_7_use_batchclient_in_backtest_mode_with_n_greater_than_3_tickers
PASS: behavioral_submit_returns_batch_id
PASS: behavioral_poll_returns_ended
PASS: behavioral_cost_halved_when_is_batch_true
PASS: behavioral_fetch_returns_succeeded_and_errored_rows_honestly

12/12 claims PASS, 0 FAIL
EXIT=0
```

### AST parse
- `backend/agents/llm_client.py` -- OK
- `backend/agents/cost_tracker.py` -- OK
- `tests/verify_phase_25_C9.py` -- OK

### Harness log scan
`phase-25.C9` not yet logged (Main must append the cycle-84 block before flipping masterplan status).

## Per-criterion judgment

### Criterion 1: `batchclient_wrapper_implemented_in_llm_client` -- PASS
Structural claims 1-4 verify class declaration + `submit/poll/fetch` signatures (`requests: list[dict] -> str`, `max_wait_sec`/`initial_delay_sec` kwargs, `dict[str, LLMResponse]` return). Behavioral claims 9 (submit returns batch id via mocked SDK), 10 (poll transitions `in_progress` -> `ended`), 12 (fetch surfaces succeeded + errored rows distinctly) exercise the lifecycle round-trip. The wrapper actually does what its docstring says.

### Criterion 2: `steps_1_through_7_use_batchclient_in_backtest_mode_with_n_greater_than_3_tickers` -- PASS (mechanism-shipped, with honest deferral)
Claim 8 verifies the `BatchClient` class docstring documents the routing rule (`n_tickers > 3 AND backtest_mode`). The orchestrator hot-path wiring is explicitly deferred to 25.C9.1, mirroring the 25.D9 mechanism-vs-adoption split that prior cycles accepted. The deferral is disclosed in three places (contract Non-goals, results Non-goals, results Live-check section). This is the same scope-honesty pattern the project has previously accepted. Accepting at PASS.

### Criterion 3: `cost_tracker_records_is_batch_true_for_50_percent_pricing` -- PASS
Claims 5-7 verify the `AgentCostEntry.is_batch: bool = False` field, the `CostTracker.record(is_batch=...)` kwarg, and that recording with `is_batch=True` yields a persisted entry with the flag set. Claim 11 is the load-bearing behavioral check: records two entries with identical token counts (one `is_batch=False`, one `is_batch=True`) and asserts `ratio == 0.5` exactly. The 50% multiplier is exercised end-to-end through `record()`, not just inspected by AST.

## Anti-rubber-stamp mutation review

Walked the six prompt-listed mutations against the verifier:
- Drop `cost *= 0.5` -- claim 11 fails (ratio != 0.5).
- Drop `is_batch` field on `AgentCostEntry` -- claim 5 (AST/field) fails AND claim 11 fails (kwarg can't persist).
- `BatchClient.fetch` missing -- claim 4 (signature) fails AND claim 12 (behavioral) fails.
- `poll()` never returns `"ended"` -- claim 10 fails (the mock sequence `in_progress -> ended` exercises both states).
- Drop errored-row surfacing -- claim 12 fails (asserts `thoughts.startswith("errored:")` and `text == ""`).
- Routing rule docstring removed -- claim 8 fails.

No non-covered spirit-breaking mutation found. The verifier's behavioral half (claims 9-12) cannot be rubber-stamped by string-search alone.

## Scope honesty

Contract, experiment_results, and planned live_check all consistently disclose that the orchestrator hot-path adoption is deferred to 25.C9.1. The "Phase 1: mechanism shipped / Phase 2: orchestrator integration" framing appears in the live-check section. This matches the previously-accepted 25.D9 pattern (Files API mechanism shipped, caller adoption deferred). Scope is bounded honestly, not overclaimed.

## Research-gate compliance

Contract `## Research-gate` section cites the brief at `handoff/current/research_brief.md`. Brief is researcher-authored, gate_passed=true, three-variant queries visible, recency scan present (with 5 concrete 2024-2026 findings), source quality hierarchy respected (Anthropic official docs + SDK source + authoritative blogs; no community-tier dependence).

## Verdict

**PASS.**

`violated_criteria: []`
`violation_details: []`
`certified_fallback: false`

## Required next actions for Main (in order, before flipping masterplan status)

1. Append cycle-84 block to `handoff/harness_log.md` with `result=PASS`.
2. Flip `.claude/masterplan.json` step 25.C9 `status -> done`.
3. Live check (per masterplan): backtest run with >3 tickers showing `is_batch=True` in cost_tracker_events with ~50% cost reduction. Per the `verification.live_check` gate, create `handoff/current/live_check_25.C9.md` capturing the evidence; otherwise the auto-push hook will hold the push (which is correct: orchestrator integration ships in 25.C9.1, so the live cost reduction is honestly a follow-up artifact).
