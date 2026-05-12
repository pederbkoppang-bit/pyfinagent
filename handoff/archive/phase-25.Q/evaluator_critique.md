---
step: phase-25.Q
cycle: 74
cycle_date: 2026-05-12
qa_run: 1
verdict: PASS
violated_criteria: []
---

# Q/A Critique -- phase-25.Q -- profit_per_llm_dollar (closes red-line goal-d)

## 5-item harness-compliance audit

1. **Researcher spawn for 25.Q** -- CONFIRM. `handoff/current/research_brief.md`
   header reads `step: phase-25.Q`, tier `moderate-complex`, JSON envelope
   `gate_passed: true`, `external_sources_read_in_full: 6`, `urls_collected: 17`,
   `recency_scan_performed: true`, 8 internal files inspected. Three-variant
   search discipline visible (current-year frontier 2026, last-2-year 2024-2025,
   year-less canonical). Exceeds the 5-sources-in-full floor.

2. **Contract pre-commit** -- CONFIRM. `handoff/current/contract.md` step ID
   `25.Q`, audit basis cited, depends_on 25.A9 (done). Success criteria copied
   verbatim from `.claude/masterplan.json`:
   (a) `sovereign_api_compute_cost_returns_non_zero_anthropic_vertex_costs`
   (b) `new_api_sovereign_efficiency_endpoint_returns_profit_per_llm_dollar`
   (c) `metric_persisted_to_bq_for_30d_window`.
   Verification command immutable.

3. **Results captured** -- CONFIRM. `handoff/current/experiment_results.md`
   contains verbatim verifier output (`11/11 claims PASS, 0 FAIL`), AST gate
   results, file inventory, hypothesis verdict, downstream notes.

4. **Log-last discipline** -- CONFIRM. `grep -c "phase-25.Q" handoff/harness_log.md` = 0.
   No premature append. Main will append AFTER this Q/A PASS and BEFORE the
   masterplan status flip.

5. **No verdict-shopping** -- CONFIRM. First Q/A spawn for 25.Q this cycle.
   No prior CONDITIONAL/FAIL entries for 25.Q in `handoff/harness_log.md`.

All five audit items pass.

## Deterministic checks

### Verification command (immutable)

```
$ source .venv/bin/activate && python3 tests/verify_phase_25_Q.py
PASS: migration_declares_efficiency_snapshots_with_required_columns
PASS: migration_idempotent_with_partition_and_cluster
PASS: bq_save_efficiency_snapshot_merge_and_timeout
PASS: fetch_llm_cost_by_provider_helper_exists
PASS: sovereign_api_compute_cost_returns_non_zero_anthropic_vertex_costs
PASS: efficiency_response_pydantic_model_has_profit_per_llm_dollar
PASS: new_api_sovereign_efficiency_endpoint_returns_profit_per_llm_dollar
PASS: behavioral_efficiency_endpoint_returns_correct_ratio
PASS: behavioral_zero_llm_cost_yields_none_ratio_not_inf
PASS: metric_persisted_to_bq_for_30d_window
PASS: provider_mapping_gemini_to_vertex_enforced

11/11 claims PASS, 0 FAIL
EXIT=0
```

### AST parse of touched files

- `backend/api/sovereign_api.py` -- OK
- `backend/db/bigquery_client.py` -- OK
- `scripts/migrations/add_efficiency_snapshots.py` -- OK
- `tests/verify_phase_25_Q.py` -- OK

### Scope check (`git status`)

Modified: `backend/api/sovereign_api.py`, `backend/db/bigquery_client.py`,
contract / experiment_results / research_brief (current).
Untracked: `scripts/migrations/add_efficiency_snapshots.py`,
`tests/verify_phase_25_Q.py`, `handoff/current/live_check_25.Q.md`.
Audit JSONL streams change as a side effect of hook execution.
Scope matches contract Plan items 1-6 exactly. No drift.

### Targeted greps

- `grep -n "anthropic=0.0\|vertex=0.0\|openai=0.0" backend/api/sovereign_api.py`
  -> ZERO matches. Both the per-day `ProviderCostPoint` site (lines 463-471)
  and the `totals` dict (lines 476-479) are sourced from
  `_fetch_llm_cost_by_provider(days)` called once at line 450 -- not a
  separate code path that could silently regress. This is the critical
  structural check for criterion 1.
- `_fetch_llm_cost_by_provider` defined at sovereign_api.py:236; called at
  L450 (compute-cost) and L560 (efficiency).
- `EfficiencyResponse` class at L114 with `profit_per_llm_dollar:
  Optional[float]` at L122.
- Route `@router.get("/efficiency", response_model=EfficiencyResponse)` at L505.
- Zero-denominator guard at L568-570 (`else: profit_per_llm_dollar = None`).
- Persistence call at L600 via `bq.save_efficiency_snapshot({...})`.
- `save_efficiency_snapshot` defined at `bigquery_client.py:804` with MERGE
  on `(snapshot_date, window_days)` and `result(timeout=30)`.

## Per-criterion judgment

### Criterion 1 -- `sovereign_api_compute_cost_returns_non_zero_anthropic_vertex_costs`

PASS. Two redundant evidence layers:
- Structural (claim 5): grep on `anthropic=0.0, vertex=0.0, openai=0.0`
  returns zero matches at the original L386-390 site. Both per-day rows and
  the totals dict are populated from the same `_fetch_llm_cost_by_provider`
  call -- a regression at the totals layer cannot exist without breaking the
  per-day layer simultaneously. They share one source.
- Behavioral (claims 8 + 11): monkey-patched BQ rows with non-zero
  `input_tok/output_tok` flow through pricing and surface as non-zero
  `anthropic` / `vertex` keys, and `provider="gemini"` is mapped to `vertex`.

### Criterion 2 -- `new_api_sovereign_efficiency_endpoint_returns_profit_per_llm_dollar`

PASS. Four evidence layers:
- Structural (claim 6): `EfficiencyResponse` exists with
  `profit_per_llm_dollar: Optional[float]`.
- Structural (claim 7): route `@router.get("/efficiency",
  response_model=EfficiencyResponse)` registered.
- Behavioral happy path (claim 8): pnl=1000, cost=100 -> ratio=10.0.
- Behavioral zero-cost (claim 9): `profit_per_llm_dollar = None` (not inf,
  not 0, no crash); descriptive `note` populated. Principled first-mover
  contract.

### Criterion 3 -- `metric_persisted_to_bq_for_30d_window`

PASS. Behavioral claim 10 asserts:
- `save_efficiency_snapshot.call_count == 1` when invoked with `persist=True`.
- Row dict carries `window_days=30`, pnl, cost, ratio.
- MERGE on `(snapshot_date, window_days)` confirmed at `bigquery_client.py:804`
  with `result(timeout=30)` -- matches CLAUDE.md 30s BQ rule.

## Anti-rubber-stamp mutation table

| Mutation | Caught by | Mechanism |
|---|---|---|
| Restore hardcoded zeros at per-day `ProviderCostPoint(...)` | claim 5 | grep on three literal `anthropic=0.0, vertex=0.0, openai=0.0` patterns |
| Restore hardcoded zeros at totals dict | claim 5 | same grep covers both sites |
| Return `inf` or `0` instead of `None` on zero LLM cost | claim 9 | behavioral `ratio is None` |
| Silently no-op `save_efficiency_snapshot` when `persist=True` | claim 10 | `MagicMock.call_count == 1` + row-shape |
| Drop `gemini -> vertex` mapping | claim 11 | direct equality on helper output |
| Typo on `response_model` (breaking shape) | claim 7 | route registration AST check |
| Migration drops PARTITION/CLUSTER | claim 2 | string grep on migration source |
| `save_efficiency_snapshot` skip `result(timeout=30)` | claim 3 | grep on timeout arg |

I cannot name a spirit-breaking mutation that escapes all 11 claims. The
suite combines structural greps (catch regressions at named call sites) and
behavioral round-trips (catch silent semantic regressions).

## Scope honesty

- First-mover claim is genuine and cited (arxiv 2503.21422, March 2025 survey).
  Recency scan in brief confirms no superseding work in last-2-year window.
- Migration ships dry-run by default with explicit `--apply` gate (CLAUDE.md
  BQ rule honoured; operator-gated MERGE/DDL).
- Per-day even-split in `get_compute_cost` is documented inline at L446-449
  as a first-pass simplification, not a hidden bug. The totals dict (what
  the UI primarily renders) carries real aggregated values.
- Zero-denominator None (not zero, not inf) is documented at the Pydantic
  model docstring (L117-118) and behaviorally enforced (claim 9).
- Live-check evidence file `handoff/current/live_check_25.Q.md` exists for
  the push-gate.

## Research-gate compliance

Contract `## Research-gate` section names the researcher spawn agent id and
cites the brief with the gate envelope (6 sources, 17 URLs, recency scan,
gate_passed=true). Six in-full external sources is above the floor of five.
Three-variant search-query discipline visible in brief.

## Final verdict

```json
{
  "ok": true,
  "verdict": "PASS",
  "reason": "All 3 immutable success criteria met. Verifier 11/11 PASS exit=0. Hardcoded zeros removed from BOTH per-day and totals sites (shared helper -- one regression surface). Mutation-resistant test suite (structural greps + 4 behavioral round-trips). Zero-denominator returns None (principled first-mover contract). Migration dry-run + operator-gated --apply. Research gate cleared (6 sources read in full, 17 URLs, recency scan, gate_passed=true). First-mover claim genuine per arxiv 2503.21422.",
  "violated_criteria": [],
  "violation_details": [],
  "certified_fallback": false,
  "checks_run": [
    "verification_command_exit_0",
    "ast_parse_sovereign_api",
    "ast_parse_bigquery_client",
    "ast_parse_migration",
    "ast_parse_verifier",
    "git_status_scope",
    "grep_hardcoded_zeros_absent",
    "grep_symbols_present",
    "harness_log_no_premature_append",
    "harness_log_no_prior_conditional",
    "research_brief_gate_envelope",
    "contract_success_criteria_verbatim",
    "mutation_table_no_uncovered_spirit_break",
    "scope_honesty_first_mover_claim",
    "zero_denominator_contract_None_not_inf"
  ]
}
```
