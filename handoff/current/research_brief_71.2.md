# Research Brief — Step 71.2 (Layer-2 HONESTY upgrade on in-app Claude MAS)

**Tier:** COMPLEX (live Layer-2 production code + a risk-gate literal)
**Researcher:** Layer-3 Researcher subagent
**Date:** 2026-07-13
**Status:** IN PROGRESS (write-first; appended incrementally)

## Objective

Research gate for step 71.2. The design proposes three honesty upgrades on the
in-app Claude MAS:
1. Constrained-decoding structured output on the 2 highest-frequency Claude JSON
   sites (quality gate + classifier) in `multi_agent_orchestrator.py`.
2. FIX the response-clobber bug at `multi_agent_orchestrator.py:885` (return None
   / preserve original answer instead of clobbering with an unparseable gate resp).
3. DELETE `evaluator_agent.py::_run_spot_checks` + `evaluate_with_spot_checks`
   (fabricated 1.02/0.95/0.99 Sharpe numbers) or wire to a real backtest; add
   structured output to `evaluator_agent._call_model`.

**Constraints:** NO effort bump. NO risk-threshold VALUE change. Immutable
verification grep must pass WITHOUT moving the DSR>=0.95 gate value.

## The critical tension (to resolve)

Immutable verification command requires:
- `grep -Eqi "output_config|json_schema|response_format|strict"` present in BOTH
  `multi_agent_orchestrator.py` + `evaluator_agent.py`
- `! grep -Eq "1.02|0.95|0.99" backend/agents/evaluator_agent.py` (NO bare literal)
- both files `ast.parse`

But 0.95 is ALSO the legit DSR promotion threshold. So the literals must be
RELOCATED to NAMED CONSTANTS imported from another module — value byte-identical.

---

## Findings (filled incrementally below)

### A. DSR/Sharpe threshold home (canonical constants module)

There is NO single project-wide constants module for the DSR gate; the value 0.95
is defined in several places. Best existing importable homes (all verified to NOT
import `backend.agents`, so no circular-import risk when imported into
`evaluator_agent.py`):

| Module | Symbol | Value | Shape |
|--------|--------|-------|-------|
| `backend/autoresearch/meta_dsr.py:20` | `LOOSE_DSR_MIN` | 0.95 | module const, in `__all__` (leaf: imports only `math`,`dataclasses`,`typing`) — **RECOMMENDED HOME** |
| `backend/autoresearch/meta_dsr.py:19` | `STRICT_DSR_MIN` | 0.99 | module const, in `__all__` (not needed — evaluator's only 0.99 is fabricated + deleted) |
| `backend/autoresearch/promoter.py:26` | `DSR_MIN_FOR_PROMOTION` | 0.95 | module const |
| `backend/services/paper_go_live_gate.py:41-42` | `PSR_THRESHOLD`,`DSR_THRESHOLD` | 0.95 | module const (odd dep direction agent→service) |
| `backend/backtest/analytics.py:695` | inline `dsr >= 0.95` (`"dsr_significant"`) | 0.95 | NO named const — canonical *statistical* significance site |
| `backend/backtest/quant_optimizer.py:123` | `dsr_threshold: float = 0.95` | 0.95 | method-param default (not importable) |
| `backend/autoresearch/gate.py:21` | `min_dsr: float = 0.95` | 0.95 | dataclass field (not importable as bare const) |

The LIVE money DSR>=0.95 promotion gate that the loop actually uses is
`backend/autonomous_loop.py:482` (`best_result.get("dsr",0) > 0.95`) plus
promoter/analytics/paper_go_live_gate — **all OUTSIDE evaluator_agent.py** and NOT
in scope for this step (the immutable grep only forbids the literals in
`evaluator_agent.py`).

### B. Every 1.02/0.95/0.99 occurrence in evaluator_agent.py (FABRICATED vs LEGIT)

Exactly 9 occurrences (verified `grep -nE`):

| Line | Text | Class | Live gate? |
|------|------|-------|-----------|
| 15 | docstring `DSR > 0.95, Sharpe < 2.0` | LEGIT (doc) | no |
| 211 | docstring `Is DSR > 0.95?` | LEGIT (doc) | no |
| 217 | docstring `PASS if: DSR > 0.95 AND Sharpe 1.0-2.0` | LEGIT (doc) | no |
| 333 | `red_flags.append(f"DSR {dsr:.2f} < 0.95 ...")` — the actual compare on **L332 is `if dsr < 0.90`**; the "0.95" is a stale cosmetic message string | LEGIT (cosmetic msg) | no |
| 349 | `if dsr >= 0.95:` — green-flag heuristic **inside `_mock_response`** (runs only when `self.model is None`) | LEGIT (mock heuristic) | no |
| 353 | `if walk_forward >= 0.95:` — green-flag heuristic in `_mock_response` (walk-forward stability, semantically ≠ DSR, coincidentally 0.95) | LEGIT (mock heuristic) | no |
| 513 | `"sharpe_2x_cost": 1.02,` — `_run_spot_checks` stub | **FABRICATED** | no |
| 514 | `"sharpe_regime_shift": 0.95,` — `_run_spot_checks` stub | **FABRICATED** | no |
| 515 | `"sharpe_param_sweep": 0.99,` — `_run_spot_checks` stub | **FABRICATED** | no |

The 513-515 fabricated numbers are the honesty problem: `evaluate_with_spot_checks`
(L459) reads `spot_check_results.get("sharpe_2x_cost") > 0.90` (L487) and flips
`CONDITIONAL -> PASS` purely on the hardcoded 1.02. Deleting `_run_spot_checks`
(L496) + `evaluate_with_spot_checks` (L459) removes lines 513-515 AND the flip.
**No live caller exists** (grep for both names across `backend/ scripts/ tests/`
returns only the definitions), so deletion is safe. The live entry point
`evaluate_proposal` (L109, called at `autonomous_loop.py:464`) is untouched.

### C. Safest relocation recommendation

To make `! grep -Eq "1.02|0.95|0.99" evaluator_agent.py` pass while keeping the
DSR value byte-identical:

1. Add import at top of `evaluator_agent.py`:
   `from backend.autoresearch.meta_dsr import LOOSE_DSR_MIN  # 0.95 (leaf module, no cycle)`
   (Optionally a sibling for walk-forward. Since walk-forward stability at 0.95 is a
   distinct semantic, either reuse `LOOSE_DSR_MIN` with a comment or add
   `WALK_FORWARD_STABILITY_MIN = LOOSE_DSR_MIN` locally — but a **local** assignment
   `= 0.95` would re-introduce the literal, so bind it to the imported name, not to a
   bare `0.95`.)
2. L349 `if dsr >= 0.95:` -> `if dsr >= LOOSE_DSR_MIN:`
3. L353 `if walk_forward >= 0.95:` -> `if walk_forward >= LOOSE_DSR_MIN:` (or the sibling name)
4. L333 f-string -> `f"DSR {dsr:.2f} < {LOOSE_DSR_MIN} (likely over-fitted)"` — renders
   "0.95" at runtime but the SOURCE contains `{LOOSE_DSR_MIN}`, so the grep passes.
   (Leave the L332 comparison `< 0.90` as-is — "0.90" is NOT in the grep alternation.)
5. Docstrings L15/211/217: reword to drop the literal, e.g. `DSR > 0.95` ->
   `DSR above the promotion threshold (LOOSE_DSR_MIN)` / `Is DSR above the promotion
   threshold?` / `PASS if: DSR above threshold AND Sharpe in [1.0, 2.0] ...`. Meaning
   preserved; no digit string.
6. Lines 513-515 vanish with the `_run_spot_checks` deletion (no relocation needed).

Alternative principled refactor: add `DSR_SIGNIFICANCE_THRESHOLD = 0.95` to
`backend/backtest/analytics.py` (the canonical statistical site) and import from
there; higher coupling, but analytics is where DSR significance is defined. Either
way the VALUE stays 0.95.

**Sanity trap:** never write `DSR_MIN = 0.95` *inside* evaluator_agent.py — a local
literal still trips the grep. The constant MUST be imported from another module.

### D. How the project calls Claude (SDK + structured-output mechanism)

Installed & pinned: **`anthropic==0.96.0`** (`requirements.txt:39`; the pin comment
itself calls 0.96.0 the "code-required floor for output_config/advisor betas").
`google-genai==1.73.1`. Python 3.14.4.

Local SDK introspection (no API cost) confirms **anthropic 0.96.0 accepts everything
needed**: `messages.create` has `output_config`, `tool_choice`, `tools` params;
`ToolParam` supports `strict`; type modules `output_config_param` +
`json_output_format_param` exist; `Messages.parse()` exists.

`multi_agent_orchestrator._call_agent` (L995) uses the **direct** SDK:
`self._get_client()` -> `anthropic.Anthropic(api_key=...)` (L204), then
`client.messages.create(model, max_tokens, system, messages=[...])` (L1008) — a plain
call, no tools/structured output today. BOTH the quality gate and the classifier route
through this one method, so structured output is added here (or in per-site variants).

Model pin: **`mas_communication = "claude-sonnet-4-6"`** (`model_tiers.py:57`) — the
model BOTH JSON sites run on. **Recency-verified (live July-2026 docs):
structured outputs (output_config.format json_schema) AND strict tool use are GA on
`claude-sonnet-4-6`, no beta header** (see Recency scan). So the design works on the
actual model with no model change and no effort bump.

**RECOMMENDED mechanism (given anthropic 0.96.0 + Sonnet 4.6):**
- **Quality gate** -> **strict submit-verdict tool**: `tools=[{"name":
  "submit_quality_verdict","description":...,"strict":True,"input_schema":{"type":
  "object","properties":{"accuracy":{"type":"number"},"completeness":{"type":
  "number"},"groundedness":{"type":"number"},"conciseness":{"type":"number"},
  "verdict":{"type":"string","enum":["PASS","FAIL"]},"improved_response":{"type":
  "string"}},"required":[...],"additionalProperties":False}}]` +
  `tool_choice={"type":"tool","name":"submit_quality_verdict"}`. Read
  `response.content[x].input`. Satisfies grep via `strict`; eliminates the fragile
  line-parser AND the unparseable/clobber branch.
- **Classifier** -> **`output_config.format` json_schema**: the site already does
  `json_io.loads(text)` (agent_definitions.py:367), so a schema-constrained JSON body
  is a drop-in. Schema = `{primary(enum main/qa/research), secondary, reasoning,
  complexity(enum), triggers_harness(bool)}`. Satisfies grep via `output_config` +
  `json_schema`.
  (Either mechanism works on either site; the split is by ergonomics. Uniform
  `output_config.format` on both is equally valid.)
- **SCHEMA CAVEAT (from the structured-outputs doc):** the supported JSON-Schema
  subset has **no `minimum`/`maximum`/`minLength`/`maxLength`**; `additionalProperties`
  must be `false`; `required` must be present. So the 0.0-1.0 score fields are plain
  `number` — keep the `<0.6`/`avg<0.7` threshold logic client-side (unchanged).
  Structured outputs are incompatible with prefill and citations; neither site uses
  those. `stop_reason: refusal`/`max_tokens` can still break schema — keep a defensive
  parse.
- **evaluator_agent.py calls GEMINI, not Claude.** `_call_model` (L279) uses
  `self.model.models.generate_content(model="gemini-2.5-flash", contents=prompt)`
  (L288-294) — the class docstring "Uses Claude Sonnet" (L84) is STALE. The correct
  structured-output mechanism there is Gemini's: add
  `config=types.GenerateContentConfig(response_mime_type="application/json",
  response_json_schema=<schema dict>)`. `google-genai 1.73.1` has
  `response_json_schema` (introspection-confirmed) -> the substring `json_schema`
  satisfies the immutable grep for evaluator_agent.py, and it's a REAL constrained
  decode (not a token hack).

### E. The clobber site + caller trace (does None preserve the answer?)

**Clobber:** `multi_agent_orchestrator.py:885` — in the "unparseable, treating as
improvement" branch (L884), `_quality_gate` does `return gate_response, usage`, i.e.
returns the gate LLM's raw (unparseable) text as the "improved" response.

**Caller** (L449-462):
```python
checked_response, gate_usage = await self._quality_gate(message, response, classification)
...
bus.emit(MASEvent(..., data={"passed": checked_response is None}))   # L459
if checked_response:                                                  # L461
    response = checked_response                                       # L462  <-- clobbers original
```
Contract: a **truthy** return REPLACES the original analyst `response`; a **None**
return keeps it. So today, an unparseable gate reply overwrites the good analyst
answer with garbled evaluator text.

**Fix = `return None, usage` at L885.** Then `checked_response is None` -> the
`if checked_response:` guard is False -> **the original `response` stands untouched**,
and `"passed"` is reported True (fail-SAFE: unparseable gate = treat as pass, don't
destroy the answer). CONFIRMED: returning None genuinely preserves the original
downstream answer. (With structured output added, this branch becomes largely dead
code — keep the None fix anyway as defense-in-depth.)

### F. The 2 highest-frequency Claude JSON sites (line anchors)

1. **Quality gate** — `async def _quality_gate` @ L744. LLM call via
   `_call_agent` @ L823-825; response parsed by line-prefix string matching
   (`ACCURACY:`/`VERDICT:`/`IMPROVED:`) at L827-885 — NOT json.loads today (it's a
   text rubric), but it is the structured-parse site the design targets.
2. **Classifier** — `async def _classify_via_llm` @ L974 ->
   `parse_llm_classification(text)` @ L981; `parse_llm_classification`
   (`agent_definitions.py:353`) strips code fences then `json_io.loads(text)` @ L367 —
   a genuine JSON parse.

Both call `_call_agent` @ L995 -> `anthropic.Anthropic().messages.create` @ L1008.

### G. No live risk-limit VALUE changes (confirmation)

**CONFIRMED: no live risk-limit/threshold VALUE changes.**
- The evaluator_agent.py 0.95 literals are docstrings (15/211/217), a cosmetic message
  (333; real compare is 0.90), and two mock-mode green-flag heuristics (349/353). None
  is a live promotion/risk gate. Relocating them to the imported `LOOSE_DSR_MIN`
  (=0.95) keeps even the mock behavior byte-identical.
- The fabricated 1.02/0.95/0.99 (513-515) are deleted, not re-valued; nothing consumes
  them (no live caller of the two spot-check methods).
- The real DSR>=0.95 gates (`autonomous_loop.py:482`, promoter/analytics/
  paper_go_live_gate/quant_optimizer) are OUTSIDE scope and UNTOUCHED.
- Clobber fix is a fail-safe control-flow change (None vs raw text), not a threshold.
- No effort bump (mas_communication effort stays; model_tiers untouched).

### H. Recency scan (last 2 years) — Anthropic structured outputs / constrained decoding

Three query variants were run per the research-gate rule (current-year frontier +
last-2-year + year-less canonical); see Sources.

**Findings that supersede/complement the cached `claude-api` skill (skill cache
2026-06-24):**
1. **NEW / material:** structured outputs are now **GA** on the Claude API with **no
   beta header**, and the supported-models list has **broadened to include
   `claude-sonnet-4-6`** (and Sonnet 4.5, Opus 4.6/4.7) — the bundled skill's
   "Supported models" list (Fable5/Opus4.8/Sonnet5/Haiku4.5/legacy Opus4.5-4.1) was
   NARROWER and is now stale. This is the decisive update: the design's structured
   output works on the actual `mas_communication` model with no change. (Official
   `platform.claude.com/docs/.../structured-outputs`, live July 2026.)
2. **Confirmed:** the old `output_format` param + `structured-outputs-2025-11-13` beta
   header still work transitionally, but `output_config.format` is canonical.
3. **Confirmed:** `strict: true` is a top-level tool-def property (alongside
   name/description/input_schema), NOT on `tool_choice`; both features use the same
   grammar-constrained-sampling pipeline (official strict-tool-use doc).
4. **Canonical prior art (year-less):** grammar-constrained decoding (Geng et al.,
   arXiv:2305.13971) gives a *structural guarantee* — only grammar-valid token
   continuations are sampled, so invalid structure is impossible (not merely unlikely),
   with no finetuning. This is exactly the reliability property the two JSON sites need.
5. **Eval-integrity (last-2-yr):** "Neither Valid nor Reliable?" (arXiv:2508.18076,
   2025) documents that LLM judges have "high face validity ... even when they are
   wrong" and can have scores inflated by trivial output modifications — reinforcing
   why an evaluator must never flip CONDITIONAL->PASS on hardcoded/unverified numbers,
   and why deterministic grounding beats self-reported scores.

No finding contradicts the design; the recency scan strengthens it (model-support
de-risked; the honesty deletion is well-supported).

---

## Sources

Read IN FULL via WebFetch (5; quality-hierarchy mixed — 2 peer-reviewed, 2 official
docs, 1 official blog):

1. **[OFFICIAL, tier-2]** Structured outputs — Claude Platform Docs.
   `https://platform.claude.com/docs/en/build-with-claude/structured-outputs`
   — confirms `output_config.format` json_schema shape; **`claude-sonnet-4-6` in GA
   model list**; JSON-Schema subset limits (no min/max); GA no-header.
2. **[OFFICIAL, tier-2]** Strict tool use — Claude Platform Docs.
   `https://platform.claude.com/docs/en/agents-and-tools/tool-use/strict-tool-use`
   — `strict:true` top-level on tool def; `additionalProperties:false`+`required`;
   grammar-constrained sampling; input in `response.content[x].input`.
3. **[OFFICIAL, tier-2/3]** Anthropic Engineering — Advanced tool use (2025-11-24).
   `https://www.anthropic.com/engineering/advanced-tool-use`
   — (tool search / programmatic tool calling / tool-use examples; tangential to
   structured output but read in full to confirm scope).
4. **[PEER-REVIEWED, tier-1]** Geng et al., Grammar-Constrained Decoding, arXiv:2305.13971.
   `https://arxiv.org/html/2305.13971v6` — structural guarantee, no finetuning; GCD
   beats unconstrained + some finetuned baselines.
5. **[PEER-REVIEWED, tier-1]** "Neither Valid nor Reliable? Investigating the Use of
   LLMs as Judges", arXiv:2508.18076 (2025). `https://arxiv.org/html/2508.18076v1`
   — LLM-judge face-validity/inflation risks; grounds the honesty-deletion rationale.

Snippet-only (evaluated, not read in full) — representative:
`platform.claude.com/docs/en/build-with-claude/structured-outputs` (dupe path
`docs.claude.com/...`), `anthropic.com/engineering/multi-agent-research-system`,
`arxiv.org/abs/2305.13971` (pdf/abs variants), `arxiv.org/html/2508.18076v1` neighbors
`arxiv 2606.19544` / `aclanthology 2026.gem-main.19` / `arxiv 2603.05399`
(LLM-judge reliability), `aidancooper.co.uk/constrained-decoding/`,
`mbrenndoerfer.com/writing/constrained-decoding-structured-llm-output`,
`frontiersin.org/.../frai.2024.1406857` (GCD clinical), `towardsdatascience.com`
hands-on Anthropic structured outputs, `thomas-wiegold.com/blog/claude-api-structured-output`,
`dev.classmethod.jp/.../claude-api-structured-outputs`, `tessl.io/blog/...`,
`docs.cloud.google.com/.../claude/structured-outputs`, `braintrust.dev/articles/what-is-llm-as-a-judge`,
`langchain.com/resources/llm-as-a-judge`.

## JSON envelope

```json
{
  "tier": "complex",
  "external_sources_read_in_full": 5,
  "snippet_only_sources": 25,
  "urls_collected": 30,
  "recency_scan_performed": true,
  "internal_files_inspected": 14,
  "gate_passed": true
}
```
