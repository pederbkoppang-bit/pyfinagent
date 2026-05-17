# Evaluator Critique — phase-27.0

Q/A subagent: `qa` (abbf2cf5a3317c096), 2026-05-16, single pass, no verdict-shopping.
Evidence under evaluation: `handoff/current/research_brief.md` (24,954 bytes) + `handoff/current/contract.md`.

## Harness-compliance audit (5 items)

| # | Item | Verdict | Note |
|---|---|---|---|
| 1 | Researcher spawned BEFORE contract | PASS | Brief dated 2026-05-16 23:07; contract references the brief in §"Research-gate summary" |
| 2 | contract.md exists + pre-Generate | PASS | 27.0 is gate-only; brief IS the Generate artifact; contract describes the broader phase-27 plan |
| 3 | Results file present | PASS | `research_brief.md` is the artifact under evaluation |
| 4 | log-last discipline | PASS | No 27.0 entry in `harness_log.md` yet — Main appends after this PASS |
| 5 | No verdict-shopping | PASS | First Q/A pass for 27.0 on this evidence |

## Deterministic checks

- Immutable verification command (from masterplan 27.0) → **EXIT=0**: file exists, gate_passed:true found, URL count 22 ≥ 10.
- Sources-read-in-full table rows: **8** (≥5 floor) → PASS.
- "Recency scan (2025-2026)" section: present (lines 36-46) → PASS.
- 4 sub-step sections: C3@83, C1@165, C2@254, C4@326 → all present → PASS.
- file:line anchors into pyfinagent: 15 matches across llm_client.py, autonomous_loop.py, bigquery_client.py, migrations/ → PASS.

## External spot-check

C3 primary URL `https://platform.claude.com/docs/en/docs/build-with-claude/structured-outputs`:
- HTTP 200, redirects to canonical `/build-with-claude/structured-outputs` (real Anthropic page).
- Page text contains `additionalProperties: false` AND `nested` AND `required` — brief's quoted claim verified consistent with source.

## Code-anchor re-read (independent)

- `llm_client.py:1388-1391` — `schema_dict` passed raw to `output_config.format.schema` with NO `additionalProperties` injection. Brief's C3 diagnosis correct.
- `llm_client.py:891-899` — try/except catches `(ValueError, AttributeError)` but NOT a `None` return. Brief's C1 two-mode analysis correct.
- `autonomous_loop.py:764-769` — `_run_claude_analysis` called unconditionally on lite_mode=True. Brief's C2 diagnosis correct.

## LLM-judgment

- Coverage: brief addresses all four bugs with diagnosis + citation + fix pattern + code. No padding.
- Consensus vs Pitfalls: internally consistent (root-only `additionalProperties: false` insufficient; C1 has two failure modes).
- C2 fix plausibility: `_select_lite_analyzer(model_name) → Callable` factory + new `_run_gemini_analysis` is plausible — single call site, existing `create_llm_client()` is the right reuse target. LiteLLM/LangChain correctly rejected as overkill.
- Sub-step thinness: every sub-step has a code block + file:line. None too thin to support GENERATE.
- Three-variant search discipline: current-year + last-2-year + year-less queries visible per topic (lines 14-32).

## Verdict (machine-readable)

```json
{
  "ok": true,
  "verdict": "PASS",
  "violated_criteria": [],
  "violation_details": "Brief satisfies all immutable + implicit gate criteria. C3 primary citation live HTTP 200 with corroborating page content. Code anchors at llm_client.py:1388-1391, 891-899, autonomous_loop.py:764-769 independently re-read and match the brief's diagnoses verbatim.",
  "certified_fallback": null,
  "checks_run": [
    "harness_compliance_audit",
    "verification_command_verbatim",
    "source_count_independent",
    "recency_scan_presence",
    "sub_step_section_presence",
    "file_line_anchor_count",
    "external_url_spot_check_http200",
    "external_url_content_corroboration",
    "code_anchor_independent_reread",
    "code_review_heuristics"
  ]
}
```

Brief is ready to unblock GENERATE on sub-steps 27.1 through 27.6.
