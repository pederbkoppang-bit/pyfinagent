# Evaluator Critique — phase-27.1

Q/A subagent: `qa` (a4fc0971063ed1463), 2026-05-16, single pass, no verdict-shopping.
Evidence under evaluation: `handoff/current/contract.md` (27.1), `handoff/current/experiment_results.md` (27.1), `backend/agents/llm_client.py` (helper @ 312-340, call-site @ 1393-1402).

## Harness-compliance audit (5 items)

| # | Item | Verdict | Note |
|---|---|---|---|
| 1 | Researcher spawned BEFORE contract | PASS | research_brief.md (gate_passed=true, 8 sources, §C3 lines 83-163) predates the 27.1 contract |
| 2 | contract.md exists + step-27.1 focused | PASS | Step id 27.1, immutable criteria copied verbatim from masterplan |
| 3 | experiment_results.md present | PASS | Built artifacts + verbatim verification cmd output + live_check evidence |
| 4 | log-last discipline | PASS | Only 27.0 PASS entry currently in harness_log.md; 27.1 will append after this verdict |
| 5 | No verdict-shopping | PASS | First Q/A pass for 27.1 |

## Deterministic checks

- Immutable verification command from masterplan 27.1 → **EXIT=0**, all 3 nested-object assertions pass.
- **Adversarial probes (Q/A-independent):**
  - (a) 4-level nested object: all four levels carry `additionalProperties: False` → PASS
  - (b) Array of objects: `items.additionalProperties is False` → PASS
  - (c) Pydantic `$defs` (Y referencing X): root + `$defs.X` both flagged → PASS
  - (d) Idempotent on already-normalized schema: re-run no-op, no error → PASS
- **Grep audit:** helper defined `llm_client.py:313`, self-recurses `:333/:336`, single call site `:1421` inside `ClaudeClient.generate_content`. NO other callers set `output_config["format"]["schema"]` directly — no bypass paths.

## Code-context re-read

Call site at `llm_client.py:1413-1429`: helper runs ONLY inside the `if schema is not None and _fmt_eligible` block, gated on Claude model prefixes (`claude-opus-4-7`, `claude-opus-4-6`, `claude-sonnet-4-6`, `claude-haiku-4-5`). Gemini and other providers never enter this branch — regression risk for Gemini = zero. `try/except` swallows any helper error to a debug log — safe failure mode.

## LLM-judgment

- **Structural soundness:** structure-blind recursion (visits every dict value + every list item) covers `properties.*`, `items`, `$defs`, `anyOf`/`oneOf`/`allOf`, AND any future schema keyword. More robust than a narrow keyed walker.
- **Idempotence:** `schema.get("type") == "object"` check is safe to re-run; setting `additionalProperties = False` on an already-False dict is a no-op.
- **Side effects:** mutates in place AND returns. Single caller reassigns; no aliasing risk.
- **Live evidence authenticity:** experiment_results.md captures the pre-fix 400 from `backend.log` 21:06:08 UTC (cycle 756a19c7) and the post-fix LLMResponse with nested `score`/`rationale`/`sub` keys conforming to the Pydantic schema.
- **Scope honesty:** helper + 1 call-site line + comment block. No scope creep into 27.2/27.3/27.4.

## Verdict (machine-readable)

```json
{
  "ok": true,
  "verdict": "PASS",
  "violated_criteria": [],
  "violation_details": [],
  "certified_fallback": null,
  "checks_run": [
    "harness_compliance_audit",
    "syntax_import",
    "verification_command_verbatim",
    "adversarial_helper_probes_4",
    "grep_call_sites",
    "grep_bypass_paths",
    "code_review_heuristics",
    "scope_honesty",
    "regression_isolation"
  ]
}
```

Live-check artifact (per masterplan 27.1 `live_check`): post-fix Anthropic call returning HTTP-200-equivalent LLMResponse with conforming nested JSON, no 400, captured in `experiment_results.md` §"Live check".
