---
step: 25.D9.1
slug: caller-side-files-api-adoption
cycle: 104
cycle_date: 2026-05-13
verdict: PASS
---

# Evaluator Critique -- phase-25.D9.1

**Step:** 25.D9.1 -- Caller-side Files API adoption (skill_file_id wiring in run_*_agent calls)
**Cycle:** 104
**Date:** 2026-05-13
**Verdict:** PASS

## Harness-compliance audit (5 items)

1. **Researcher spawned?** YES -- `handoff/current/research_brief.md` tier=moderate,
   5 sources fetched in full (Anthropic Files API + Prompt Caching + 3 practitioner
   blogs), 12 URLs collected, recency scan present (2026-03-06 default-TTL change
   noted as non-blocking). gate_passed=true.
2. **Contract before generate?** YES -- `handoff/current/contract.md` step=25.D9.1
   with immutable success criteria written before code.
3. **experiment_results present?** YES.
4. **Masterplan status pending at qa-time?** YES -- newly inserted between 25.D9
   and 25.E9.
5. **No verdict-shopping?** YES -- first spawn.

## Deterministic checks (run by Q/A)

| Check | Command | Expected | Got |
|-------|---------|----------|-----|
| Verification suite | `python3 tests/verify_phase_25_D9_1.py` | 5/5 PASS | 5/5 PASS |
| AST sanity | `python -c "ast.parse(...)"` | OK | OK |
| Call-site count | `grep -c "generation_config=self._skill_gen_config"` | 11 | 11 |
| Helper definition | `grep -n "_skill_gen_config"` | line 569 | line 569 |

Verbatim verification suite output:

```
[PASS] 1. orchestrator_has_skill_gen_config_helper
        -> _skill_gen_config method present
[PASS] 2. enrichment_agents_pass_generation_config_with_skill_file_id
        -> call_sites=11 (expected >=11)
[PASS] 3. helper_returns_none_when_skill_file_ids_empty_gemini_fallback
        -> got None
[PASS] 4. helper_returns_skill_file_id_dict_for_mapped_stem
        -> got {'skill_file_id': 'file_xyz_123'}
[PASS] 5. helper_returns_none_for_unmapped_stem_no_keyerror
        -> got None

ALL 5 CLAIMS PASS
```

## Immutable success criteria

1. `orchestrator_has_skill_gen_config_helper` -- MET (line 569).
2. `enrichment_agents_pass_generation_config_with_skill_file_id` -- MET
   (11 call sites at lines 866, 873, 880, 887, 894, 910, 917, 924, 931, 938, 945).
3. `helper_returns_none_when_skill_file_ids_empty_gemini_fallback` -- MET
   (helper guards via `getattr(self, "_skill_file_ids", None)` and returns None
   on empty dict or missing stem).

## LLM judgment

- **Contract alignment**: 3 files in contract Files table, all touched. Success
  criteria copied verbatim. No criterion was amended.
- **Mutation-resistance**: claims 3 / 4 / 5 are LIVE behavioral tests on the
  helper (empty dict, mapped stem returns correct dict shape, missing stem
  returns None without KeyError). Claim 2 is a grep that counts the real call
  sites in source. Real coverage, not a syntax-only check.
- **Scope honesty**: `cache_control` on the document block is explicitly deferred
  to 25.D9.2 and called out in experiment_results.md. The 25.D9.1 immutable
  criteria do NOT require cache_control, so this is honest scoping rather than
  a silent dropped item.
- **Caller safety**: Gemini path is untouched. On non-Claude orchestrators
  `_skill_file_ids` stays empty, the helper returns None, and the existing
  inline-skill path is preserved. Downstream `_generate_with_retry` already
  accepts a `generation_config` kwarg (line 511) -- no API break introduced.
- **North-star alignment**: per Anthropic Files API + Prompt Caching docs, this
  cuts the per-call skill input from ~5K-15K tokens to ~8 tokens (the file-id
  reference), ~98-99.5% reduction per Claude-mode enrichment call. Multiplied
  across 11 agents and many tickers, this lands directly in the
  `profit_per_llm_dollar` metric (25.Q).
- **Research-gate compliance**: contract references the research brief and the
  brief cites the Files API + Prompt Caching docs. Gate passed cleanly.

## Verdict JSON

```json
{
  "ok": true,
  "verdict": "PASS",
  "reason": "All 3 immutable criteria met. 5/5 verification claims PASS, AST OK, exactly 11 generation_config call sites wired, helper null-guards via getattr so Gemini path returns None and never raises. Mutation-resistance covered by live behavioral tests (claims 3/4/5).",
  "violated_criteria": [],
  "violation_details": [],
  "certified_fallback": false,
  "checks_run": ["harness_compliance_audit", "verification_command", "ast_syntax", "grep_call_site_count", "helper_definition_inspection", "contract_alignment", "mutation_resistance", "scope_honesty", "caller_safety", "research_gate_compliance"]
}
```
