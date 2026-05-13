---
step: phase-25.B9
cycle: 80
cycle_date: 2026-05-13
verdict: PASS
agent: qa (merged qa-evaluator + harness-verifier)
---

# Q/A Critique -- phase-25.B9 -- Bump system prompt above 4096-token cache threshold

## 1. 5-item harness-compliance audit

| # | Check | Status | Evidence |
|---|---|---|---|
| 1 | Researcher spawn (gate_passed=true, step=25.B9) | PASS | `handoff/current/research_brief.md` header `step: 25.B9`; JSON envelope `gate_passed: true`, `external_sources_read_in_full: 7`, `urls_collected: 17`, `recency_scan_performed: true`, `internal_files_inspected: 7`. Three-variant query discipline visible (current-year, last-2-year, year-less). |
| 2 | Contract pre-commit + verbatim immutable criteria | PASS | `handoff/current/contract.md` `Step ID: 25.B9`; 3 success criteria listed verbatim matching masterplan; verification command immutable. |
| 3 | Results captured (verbatim verifier output) | PASS | `handoff/current/experiment_results.md` quotes 11/11 PASS verifier output, file list, size measurement, hypothesis verdict. |
| 4 | Log-last (harness_log NOT yet appended for 25.B9) | PASS | `grep -c "phase-25.B9\|phase=25.B9" handoff/harness_log.md` returns 0. Correct order: Q/A first, log + status-flip after PASS. |
| 5 | No verdict-shopping (first Q/A spawn) | PASS | No prior 25.B9 evaluator_critique on file; this is cycle-1 evaluation. |

## 2. Deterministic outputs

```
$ source .venv/bin/activate && python3 tests/verify_phase_25_B9.py
PASS: house_instructions_constant_declared
PASS: system_prompt_consolidates_skill_and_schema_above_4096_tokens
PASS: house_instructions_contains_key_sections
PASS: system_prompt_assignment_uses_house_instructions
PASS: old_short_literal_no_longer_assigned_to_system_prompt
PASS: cache_control_wiring_preserved
PASS: house_instructions_excludes_skills_and_schemas
PASS: behavioral_estimated_tokens_above_threshold
PASS: cache_hit_rate_proxy_increases_to_30_percent_or_higher
PASS: usage_meta_cache_read_input_tokens_grows_post_25_B9
PASS: no_regression_claude_client_constructs_cleanly

11/11 claims PASS, 0 FAIL
EXIT=0
```

```
$ python -c "import ast; ast.parse(open('backend/agents/llm_client.py').read()); print('AST_OK')"
AST_OK
```

```
$ python -c "from backend.agents.llm_client import _HOUSE_INSTRUCTIONS; print(len(_HOUSE_INSTRUCTIONS), 'chars,', round(len(_HOUSE_INSTRUCTIONS)/3.5), 'est tokens')"
19026 chars, 5436 est tokens
above_threshold_chars (>=14336): True
above_threshold_tokens (>=4096): True
```

Headroom: 5436 - 4096 = 1340 tokens (~33% margin on the Opus 4.7 / Haiku 4.5 floor; 2.65x on the Sonnet 4.6 floor of 2048).

Cache wiring (`llm_client.py:1120`, `:1147`):
- `system_prompt = _HOUSE_INSTRUCTIONS` (line 1120)
- Schema-append rides AFTER prefix (line 1124, dynamic)
- `cache_control={"type": "ephemeral", "ttl": "1h"}` preserved (line 1147)

Spirit-mutation safety probes:
- safety_anchor_override_resistance: True
- has_FACT_LEDGER: True
- has_JSON_output_rules: True (literal "JSON output rules" present)
- has_reasoning_framework: True
- has_Safety_anchor: True
- no_skill_marker ("## Prompt Template" absent): True
- no_schema_marker ("model_json_schema" absent): True

## 3. Per-criterion judgment

### Criterion 1: `system_prompt_consolidates_skill_and_schema_above_4096_tokens`
**PASS.** Claims 2 + 8 enforce >= 4096 tokens via chars/3.5 heuristic. Measured 5436 est tokens (33% headroom). Constant `_HOUSE_INSTRUCTIONS` declared at module scope (verified via re.search), assigned to `system_prompt` in `ClaudeClient.generate_content` (claim 4). Old bare literal `"You are a financial analysis AI."` no longer assignment-bound (claim 5). Schema-append remains AFTER cached prefix so dynamic content does not invalidate the cache (verified at `llm_client.py:1124`).

### Criterion 2: `usage_meta_cache_read_input_tokens_grows_post_25_B9`
**PASS.** Claim 10 behaviorally simulates 3 sequential `CostTracker.record()` calls with mocked Anthropic `usage_metadata`. Asserts entry[0].cache_read==0, entries[1+2].cache_read>0. The check passes through the real `cost_tracker.CostTracker.record()` code path -- not just constant inspection. Caveat (correctly disclosed in contract + results): this is a proxy on the cost_tracker capture, not live API traffic; live BQ evidence is deferred to `live_check_25.B9.md`.

### Criterion 3: `cache_hit_rate_proxy_increases_to_30_percent_or_higher`
**PASS.** Claim 9 computes `e2.cache_read / (e2.cache_read + e1.cache_creation)` = 5000/(5000+5000) = 0.5 >= 0.30. Uses the formula sourced from research brief (`startdebugging.net` + arXiv 2601.06007). Scope honesty: this is a proxy from simulated cost_tracker entries, documented as such; the immutable criterion is met against the proxy, with live verification deferred.

## 4. Anti-rubber-stamp coverage

| Mutation | Detected by | Verified |
|---|---|---|
| Revert literal to "You are a financial analysis AI." | Claims 4 + 5 (assignment swap + bare literal regex) | YES |
| Shrink `_HOUSE_INSTRUCTIONS` below 4096 tokens | Claims 2 + 8 (length + chars/3.5) | YES |
| Drop `cache_control` wiring | Claim 6 (regex `"ephemeral".*"1h"`) | YES |
| Inline skill markdown | Claim 7 (`## Prompt Template` marker) | YES |
| Inline Pydantic schema | Claim 7 (`model_json_schema` marker) | YES |
| Drop safety anchor language | Claim 3 (`Safety anchor` required phrase) | YES |
| Drop FACT_LEDGER discipline | Claim 3 (`FACT_LEDGER` required phrase) | YES |
| Change TTL from "1h" to "5m" | Claim 6 regex pins `"ttl":"1h"` | YES |

No spirit-breaking mutation found uncovered. Mutation coverage is strong (8 distinct mutations, 5 claims).

## 5. Scope honesty

The contract and results both correctly:
- Defer Files API integration to step 25.D9 (no scope creep).
- Acknowledge chars/3.5 is an Anthropic-recommended heuristic, not an exact tokenizer; actual count may vary +/- 10-15%. Headroom (5436 vs 4096 = ~33%) absorbs this slack.
- Mark the cache_read/hit-rate checks as proxies driven by simulated `CostTracker` entries, with the live BQ check explicitly deferred to `handoff/current/live_check_25.B9.md` (per masterplan `live_check` field).
- Do not claim cost-savings figures from this step alone; compound savings with 25.D9 are noted as expected, not measured.

## 6. Research-gate compliance

Contract's `## Research-gate` section cites researcher findings (cache thresholds 4096/2048/4096, chars/3.5 heuristic, cost-savings linearity from arXiv 2601.06007, OWASP LLM01 safety-anchor recommendation, hit-rate proxy formula from startdebugging.net). Research brief gate envelope clears all hard blockers.

## 7. Output

```json
{
  "ok": true,
  "verdict": "PASS",
  "reason": "All 3 immutable criteria met: _HOUSE_INSTRUCTIONS measured at 5436 est tokens (>=4096), behavioral cache_read growth verified via simulated CostTracker round-trips, hit-rate proxy 0.5 (>=0.30). Deterministic verifier 11/11 PASS, exit=0. AST OK. Cache wiring (ttl='1h') preserved. Schema-append rides AFTER cached prefix. All 5 harness-compliance audit items PASS. Mutation coverage strong (8 mutations across 5 claims).",
  "violated_criteria": [],
  "violation_details": [],
  "certified_fallback": false,
  "checks_run": [
    "syntax_ast",
    "verification_command_exit0",
    "house_instructions_length_measurement",
    "cache_wiring_grep",
    "schema_append_order_check",
    "spirit_mutation_probes",
    "researcher_gate_envelope",
    "contract_immutable_criteria",
    "harness_log_not_yet_appended"
  ]
}
```

Next step for Main: append cycle entry to `handoff/harness_log.md` with `result=PASS`, then flip `.claude/masterplan.json` step `25.B9` to `status: done`. Live-check evidence file `handoff/current/live_check_25.B9.md` will be required by the auto-push gate before the commit is pushed to `origin/main`.
