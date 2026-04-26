---
step: phase-23.1.5
cycle_date: 2026-04-27
verdict: PASS
qa_pass_index: 1
---

# Q/A Critique — phase-23.1.5 (LLM-as-judge meta-scorer)

## 5-item harness-compliance audit

1. Researcher brief on disk: `handoff/current/phase-23.1.5-research-brief.md`
   with `external_sources_read_in_full: 7`, `recency_scan_performed: true`,
   `gate_passed: true`. PASS.
2. Contract front-matter `step: phase-23.1.5` matches; `verification:` field
   contains the immutable command (Python one-liner exercising
   `meta_score_candidates`). PASS.
3. `experiment_results.md` includes the verbatim verification block (`$ source .venv/bin/activate && python -c "..."` followed by `ok n=2 top=AAPL(7) bottom=NVDA(6)` and `exit=0`). PASS.
4. `harness_log.md` not yet appended for `phase=23.1.5`. PASS (Main appends LAST).
5. First Q/A spawn — no second-opinion shopping. PASS.

## Deterministic checks

| ID | Check | Result |
|---|---|---|
| A | Immutable verification command | exit 0; output `ok n=2 top=AAPL(7) bottom=NVDA(6)` matches contract expectation; convictions in [1,10], sorted desc |
| B | `pytest tests/services/ -v` | 81 passed in 0.22s (no regression; 14 new tests in `test_meta_scorer.py`) |
| C | `ast.parse` of meta_scorer.py, autonomous_loop.py, settings.py, test_meta_scorer.py | `all syntax ok` |
| D | Default-OFF safety | `meta_scorer_enabled: bool = Field(False, ...)` in settings.py:166; autonomous_loop.py:169 wraps with `if getattr(settings, "meta_scorer_enabled", False):`; existing rank_candidates flow preserved when flag unset |
| E | Anti-rubber-stamp prompt — all six mitigations confirmed in `_build_meta_prompt`: (1) "First state what could go WRONG" line 124; (2) `risk_off` regime warning line 111+126; (3) `composite_score_pre_meta` field labeling line 101; (4) calibration anchors "Score 9-10 only when ... all align" line 130; (5) `random.Random(0xC0FFEE).shuffle` line 177; (6) "Score each candidate INDEPENDENTLY" line 123 | PASS |
| F | Fallback path correctness — `_fallback_conviction` clamps via `max(1, min(10, ...))` (test `test_fallback_conviction_clamps_to_1_10` passes with 100→10, -5→1, None→5); `test_meta_score_no_anthropic_key_returns_fallback` covers no-API-key path; LLM parse error → `_fallback_all` (line 215); out-of-range LLM output (e.g. 11) clamped to 10 (test_meta_score_clamps_out_of_range_llm_output line 189) | PASS |
| G | Batch cap `_MAX_BATCH = 30` (line 31). Bottom-N candidates use `"below batch cap (composite-score fallback)"` reason (line 235) | PASS |
| H | Schema-strip applied: `_strip_unsupported_schema_keys(MetaScorerBatch.model_json_schema())` called BEFORE `client.generate_content` (line 181) — lesson from cycles 1-3 honored | PASS |
| I | Audit trail preserved: candidate dicts include both original `composite_score` (untouched) AND new `conviction_score` + `conviction_reason` (additive overlay, Option B per brief) | PASS |
| J | Git diff scope — required files all present (meta_scorer.py NEW, autonomous_loop.py M, settings.py M, test_meta_scorer.py NEW, contract.md M, experiment_results.md M). Other modified paths are hook-maintained noise (`.archive-baseline.json`, `cycle_heartbeat.json`, audit JSONLs, perf TSV, frontend tsbuild artifacts) — no scope creep | PASS |

## LLM-judgment leg

- **Phase-23.1.5 accomplishment**: single batched Claude call combines momentum/PEAD/news/regime sub-signals into integer 1-10 convictions with reason strings. Verified live via the immutable verification command (exit 0).
- **Mutation-resistance**: verification calls real Anthropic API; would FAIL if anyone broke the schema-strip (would 400 from Anthropic), the prompt template (would degrade output), or the clamp logic (would let `>10` through and break the `<=10` assertion).
- **Anti-rubber-stamp**: all six mitigations from the brief are present in `_build_meta_prompt`, with reproducible RNG seed `0xC0FFEE` for candidate-order shuffling.
- **Scope honesty**: per-candidate mode, SELL-side judging, BQ persistence, and UI exposure are explicitly deferred — the additive-overlay Option B keeps `composite_score` untouched so the existing pipeline degrades gracefully.
- **Research-gate compliance**: brief on disk with `gate_passed: true`, 7 sources read in full, recency scan present.
- **Default-off**: `meta_scorer_enabled=False`; integration site is a single `if` block at line 169 of autonomous_loop.py — flag-flip surface area is minimal.
- **Cost discipline**: one Claude call/cycle (~$0.025); fallback path zero LLM cost.

## Verdict

```json
{
  "ok": true,
  "verdict": "PASS",
  "violated_criteria": [],
  "violation_details": [],
  "certified_fallback": false,
  "checks_run": [
    "harness_compliance_5item_audit",
    "syntax_ast_parse",
    "verification_command_live_anthropic",
    "pytest_tests_services_81_passed",
    "default_off_flag_check",
    "anti_rubber_stamp_six_mitigations",
    "fallback_path_unit_tests",
    "batch_cap_check",
    "schema_strip_check",
    "additive_overlay_audit_trail",
    "git_diff_scope_review",
    "research_gate_brief_present"
  ]
}
```

PASS. Main may proceed to append `harness_log.md` (LAST), then flip
`phase-23.1.5` status to `done` in `.claude/masterplan.json`.
