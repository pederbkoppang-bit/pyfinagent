---
name: reference-structured-output-guarantee-classes-86-108
description: Three different guarantee classes ship under the name "structured outputs" - Anthropic API constrains, Claude Code --json-schema validates post-hoc and RE-PROMPTS, Gemini never claims an absolute guarantee
metadata:
  type: reference
---

Verified 2026-08-17 against primary vendor docs (step 86.108). "Structured
output" is three different things; a config written for one does not bind on
another.

| Transport | Mechanism | Doc wording |
|---|---|---|
| **Anthropic API** (`output_config.format`) | constrained decoding | *"guarantee schema-compliant responses through constrained decoding … Always valid: No more `JSON.parse()` errors"* |
| **Claude Code / Agent SDK** (`--json-schema`, `outputFormat`) | **post-hoc validate + RE-PROMPT** | *"validated JSON output … after the agent completes its workflow"*; *"the SDK validates the output against it, re-prompting on mismatch. If validation does not succeed within the retry limit, the result is an error"* |
| **Gemini** (`response_schema`+`response_mime_type`) | not stated | only *"adhere to a provided JSON Schema"* — **never claims an absolute guarantee** |

Key operational facts:

- **`success` with NO `structured_output` must be treated as a FAILURE** —
  Anthropic says so explicitly. This is the mode pyfinagent logs as a rail drop.
  Failure subtype is `error_max_structured_output_retries`.
- **A model fallback can retract an already-completed output mid-stream**; check
  the `errors` list to distinguish that from a schema failure.
- **CC SDK validates against JSON Schema draft-07**; newer drafts are *rejected*
  (Zod defaults to 2020-12 → pass `target: "draft-7"`).
- **Anthropic STRIPS `minimum`/`maximum`/`minLength`/`maxLength`**, caps
  `minItems` at 0-or-1, requires `additionalProperties:false`, and re-validates
  client-side. This is why research-gate floors are JS-enforced, not schema-enforced.
- **Truncation is outside every guarantee.** Anthropic and Google are silent on
  it; **only OpenAI documents detection** (`status === 'incomplete' &&
  incomplete_details.reason === 'max_output_tokens'`).
- **Measured, empirically, in this repo:** the Moderator emitted 359
  invalid-JSON events *while* a Gemini `response_schema` was in force
  (`debate.py:47-51`). Schema present != invalid output unreachable.
- **`--max-tokens` is not a CC CLI flag** (absent from the flag table); it is
  no-op'd at `claude_code_client.py:393`, so Gemini-side `max_output_tokens`
  budgets do not transfer to the CC rail.

**Literature counterweight (do not cite one side alone):** "Let Me Speak
Freely?" (arXiv 2408.02442) reports JSON mode costing 26-63 accuracy points, but
JSONSchemaBench (arXiv 2501.10868) finds constrained decoding *improves*
downstream accuracy up to 4% and can be 50% faster. They measure different
populations (sub-3B/reasoning vs extraction). The reconciling metric is
**wrong-valid-schema rate** (arXiv 2605.26128): *"A valid JSON object can still
encode the wrong decision, so a dashboard that tracks parse success alone can
improve while downstream execution gets worse."*
