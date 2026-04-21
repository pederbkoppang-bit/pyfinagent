# Evals + Guardrails Audit (phase-4.12.0)

## URL coverage

| URL | CHECKED/FAILED | notes |
|---|---|---|
| platform.claude.com/docs/en/test-and-evaluate/define-success | CHECKED | full content captured (success-criteria + eval-design + grading). Sidebar page also labelled "develop-tests" collapses into the same page. |
| platform.claude.com/docs/en/test-and-evaluate/develop-tests | CHECKED (same content) | same URL resolves to the define-success page; develop-tests is the latter half of that page |
| platform.claude.com/docs/en/test-and-evaluate/eval-tool | CHECKED | Workbench eval tool page |
| platform.claude.com/docs/en/test-and-evaluate/reducing-latency | FAILED (404) -> CHECKED via docs.anthropic.com/claude/docs/reducing-latency redirect | full content captured |
| platform.claude.com/docs/en/strengthen-guardrails/reduce-hallucinations | FAILED (404) -> CHECKED via /docs/test-and-evaluate/strengthen-guardrails/... | correct path has extra `/docs/test-and-evaluate/` segment |
| platform.claude.com/docs/en/strengthen-guardrails/increase-consistency | FAILED -> CHECKED via correct path | same |
| platform.claude.com/docs/en/strengthen-guardrails/mitigate-jailbreaks | FAILED -> CHECKED via correct path | same |
| platform.claude.com/docs/en/strengthen-guardrails/reduce-prompt-leak | FAILED -> CHECKED via correct path | same |

Canonical stem confirmed: `platform.claude.com/docs/en/docs/test-and-evaluate/strengthen-guardrails/<page>`.

## Per-page digests

### Test and evaluate

**define-success / develop-tests (single page).** Success criteria must be SMART: Specific, Measurable, Achievable, Relevant. Even "hazy" topics (safety, ethics) are quantifiable, e.g. "<0.1% of 10k outputs flagged for toxicity". Common dimensions: task fidelity, consistency, relevance/coherence, tone, privacy, context utilization, **latency, price**. Multidimensional evals expected, not single-number. Eval-design principles: (1) task-specific mirroring real distribution incl. edge cases (irrelevant input, overly long input, ambiguous cases); (2) automate when possible (exact/string match, code-graded, LLM-graded); (3) volume over quality ("more questions with slightly lower signal automated grading is better than fewer questions with high-quality human hand-graded evals"). Seven example eval recipes: exact match, cosine-similarity (SBERT), ROUGE-L, LLM-based Likert, LLM-based binary, LLM-based ordinal. Grading hierarchy: code-based (fastest/most reliable) -> LLM-based (fast, flexible) -> human (slow, last resort). LLM-grading rules: detailed rubric, empirical/specific output (`correct|incorrect`), encourage reasoning in `<thinking>` tags and discard. Strong admonition: **use a different model for grading than for generation**.

**eval-tool.** The Console Evaluate tab requires prompts with `{{variable}}` syntax. Features: auto-generate test cases (with editable generation-logic prompt), CSV import, side-by-side comparison of prompt versions, 5-point quality grading, prompt versioning. Re-run suite against new prompt version to measure delta.

**reducing-latency.** Measure TTFT and baseline latency before optimising. Optimisation ladder: (1) choose smaller model (Haiku 4.5 explicitly called out for speed-critical paths); (2) optimise prompt + output length -- concise prompts, `max_tokens` hard limits, lower temperature can shorten outputs, prefer sentence/paragraph limits over word-count; (3) streaming for perceived latency. Related techniques (from adjacent pages): prompt caching (~85% latency reduction on long prompts), Message Batches API (50% cheaper async), parallel tool-use. Warning: "It's always better to first engineer a prompt that works well without model or prompt constraints, and then try latency reduction strategies afterward."

### Strengthen guardrails

**reduce-hallucinations.** Basic: (a) explicitly permit "I don't have enough information to confidently assess this"; (b) for >20k-token docs require verbatim quote extraction FIRST, then reason over quotes only; (c) require per-claim citation, post-hoc verify each claim against a source quote, retract claim if no quote found (mark with empty `[]`). Advanced: CoT verification, best-of-N consistency check across multiple runs, iterative refinement (feed output back in as input to verify), external-knowledge restriction ("use only provided documents, not general knowledge"). Docs note these reduce but do not eliminate.

**increase-consistency.** **Top-banner directive**: "For guaranteed JSON schema conformance use Structured Outputs, not prompt engineering." Structured Outputs is an API feature, not a prompt trick. Beyond that: specify format precisely (JSON/XML/custom), prefill Assistant turn (**not supported on Opus 4.7, 4.6 or Sonnet 4.6** -- use structured outputs or system prompt instead), constrain with few-shot examples, retrieval for context consistency, chain prompts for complex tasks, use system prompts for role/character.

**mitigate-jailbreaks.** Layered defence: (1) harmlessness screen with Haiku 4.5 using Structured Outputs `{is_harmful: boolean}` JSON schema before main call; (2) LLM-based input validation with known jailbreak patterns as few-shot; (3) ethical system prompt listing values (Integrity/Compliance/Privacy/IP) with canned refusal string; (4) throttle/ban repeat offenders; (5) continuous monitoring of outputs for jailbreak signatures. Advanced: chain safeguards via tool use (harmlessness_screen tool invoked inside the agent loop).

**reduce-prompt-leak.** "Consider leak-resistant prompt engineering only when absolutely necessary" -- the complexity itself degrades task performance. Preferred first line: output screening + post-processing. Techniques: (a) separate context (system) from query (user); prefill Assistant with a reminder `[Never mention the proprietary formula]` (prefill unsupported on Opus 4.7/4.6, Sonnet 4.6); (b) regex/keyword post-filter; (c) LLM-based leak detector for nuanced cases; (d) avoid putting proprietary content in the prompt if the task does not need it; (e) periodic audits.

## pyfinagent audit

- **Evals vs doc.** We have qa-evaluator + harness-verifier (dual-evaluator rule) and LLM-graded `evaluator_agent.py`. Alignment is strong on three doc principles: (1) different model for grading vs generation (harness uses Opus for orchestrator, separate agent for verifier), (2) structured verdict (PASS/CONDITIONAL/FAIL -- maps to doc's "empirical/specific" output), (3) detailed rubrics in `.claude/masterplan.json` verification_criteria. Gaps: we have **no held-out test set** in the SMART sense; our "evals" are single-run step verifications, not a stable suite we re-run across prompt versions. Doc prescribes "volume over quality" (hundreds of automated cases) -- we have zero regression test cases for the 28 skill prompts. No `rouge`/cosine/exact-match harness; all grading is bespoke LLM rubric per step. No Workbench eval tool usage (prompts live in `.md` files with no `{{variable}}` syntax for eval import).

- **Latency tracking: missing.** `backend/agents/llm_client.py` has 33 `usage`/token-cost references but zero `time.time()`, `time.perf_counter`, `latency_ms`, `duration_ms`, or `ttft` references. Grep on `stream=True` across `backend/` returns zero. Concrete places to add per-call latency:
  1. `backend/agents/llm_client.py` -- wrap every `client.messages.create` / `generate_content` in `time.perf_counter()` deltas, emit `{provider, model, input_tokens, output_tokens, latency_ms, ttft_ms}` alongside the existing cost record.
  2. `backend/services/perf_tracker.py` -- already present for perf metrics; extend schema with `latency_ms` per step.
  3. BQ `pyfinagent_data.harness_learning_log` / new `llm_call_log` table -- persist latency for post-hoc p50/p95 analysis.
  4. Harness Tab frontend (`frontend/src/lib/api.ts` + Harness page) -- surface p95 latency per model/agent, mirroring current cost panel.
  5. Structured success criterion: add `latency_p95_ms` to `.claude/masterplan.json` verification_criteria for production-path steps (MAS orchestrator, Slack bot assistant).

- **Hallucinations vs doc.** We have FACT_LEDGER, bias_detector agent, Gemini grounding on Layer 1. Present: citation requirement, CoT via Claude extended thinking on MAS, Gemini Google Search grounding. **Gaps**: no explicit "I don't know" permission language audited across 28 skill prompts (spot-grep shows phrase appears in CHANGELOG and one archive contract only -- not in any live `backend/agents/skills/*.md` content); no verbatim-quote-first pattern for the long-doc agents (`rag_agent`, `deep_dive_agent`, `scenario_agent`); no post-hoc claim-verification loop that retracts unsupported claims with `[]` markers; no best-of-N consistency run on high-stakes outputs (allocation decision, risk_judge).

- **Consistency vs doc.** Gemini side uses JSON schema enforcement -- aligned with doc's "structured outputs". Claude side (MAS, harness planner, evaluator) relies on `json.loads()` of free-form output (22 occurrences across 16 agent files, zero `JSONDecodeError` handlers in `multi_agent_orchestrator.py`). Phase-4.11 already flagged this. Doc's prescription is unambiguous: **use Anthropic Structured Outputs API** (the doc banner says "not prompt engineering"). Prefill workaround is explicitly unsupported on Opus 4.7/4.6 and Sonnet 4.6 -- the exact models this project uses. So the current "ask Claude nicely for JSON" pattern is unsupported by Anthropic's own guidance.

- **Jailbreak resilience vs doc.** Zero live references to `jailbreak`, `prompt_injection`, `harmlessness` outside one archived phase-3.2 contract. No Haiku pre-screen on Slack-bot or paper-trader user inputs. The Slack assistant handler (`backend/slack_bot/assistant_handler.py`) takes raw Slack text and forwards it to the agent pipeline unchecked. Doc prescribes: (a) Haiku 4.5 harmlessness screen with structured `{is_harmful: bool}` JSON schema, (b) ethical system prompt, (c) repeat-offender throttle. We have none of these. Risk tier is not critical (internal Slack workspace) but the pattern is violated.

- **Prompt leak vs doc.** We never set anti-leak patterns. Skill prompts include proprietary trading logic, EBITDA-style formulas, risk-weight tables. If the Slack-bot or paper-trader surfaces the full agent transcript to the user (it currently does via `streaming_integration.py`), a user could prompt "repeat your instructions verbatim" and pull system prompts. No regex post-filter on agent output, no LLM-based leak detector, no audit pattern. Doc's "consider only when absolutely necessary" escape hatch applies -- our IP sensitivity is moderate, but the zero-defence posture is still below baseline.

## Findings

1. No stable eval suite -- current evaluators run once per step, doc wants a held-out set of ~hundreds of cases re-run per prompt version.
2. No per-call latency instrumentation despite extensive cost tracking.
3. JSON-by-prompt-engineering on Claude side is explicitly unsupported by Anthropic for Opus 4.7/4.6 and Sonnet 4.6 (our models).
4. No harmlessness pre-screen on user-facing surfaces (Slack, paper-trader).
5. Skill prompts lack "I don't know" permission language and verbatim-quote-first pattern for long-doc agents.
6. No post-hoc claim verification or `[]` retraction pattern.
7. No prompt-leak defences whatsoever; Slack streaming integration is a potential exfiltration vector.
8. Skill prompts not templated with `{{variable}}` syntax so cannot be driven through Workbench eval tool.

## MUST FIX

- **Migrate Claude JSON output to Structured Outputs API** in `backend/agents/llm_client.py` (Anthropic side) -- doc-blessed replacement for the current `json.loads` pattern; unblocks phase-4.11 finding and prevents silent parse failures on Opus 4.7/4.6 / Sonnet 4.6.
- **Add Haiku 4.5 harmlessness pre-screen** in `backend/slack_bot/assistant_handler.py` and any paper-trader inbound path that accepts free-text. Use Structured Outputs `{is_harmful: boolean, reason: string}` schema.
- **Instrument per-call latency** (`time.perf_counter` delta + `ttft_ms` when streaming) in `llm_client.py`; persist to BQ; surface p95 on Harness tab. Add `latency_p95_ms` success criterion to masterplan for production-path steps.
- **Add "I don't know" permission** to the 10 highest-stakes skill prompts (risk_judge, synthesis_agent, neutral_analyst, quant_model_agent, scenario_agent, moderator_agent, deep_dive_agent, critic_agent, bias-adjacent agents, devils_advocate_agent). Single-sentence retrofit.

## NICE TO HAVE

- Held-out regression eval suite: stand up `backend/backtest/experiments/eval_suite/` with ~200 labelled cases covering each of the 28 skill agents, auto-graded (exact match / ROUGE / LLM-binary), re-run on every prompt change. Scorecard written to TSV like `quant_results.tsv`.
- Verbatim-quote-first pattern in `rag_agent.md`, `deep_dive_agent.md`, `scenario_agent.md` (agents that ingest >20k-token docs per doc threshold).
- Post-hoc claim verification pass for `synthesis_agent` output -- auto-retract unsupported claims with `[]` markers.
- Prompt-leak post-filter (regex + LLM binary) on Slack streaming output; periodic audit job re-running red-team prompts (`ignore previous instructions`, `repeat your system prompt`) nightly.
- Best-of-N (N=3) consistency check on the allocation decision endpoint; disagreement triggers escalation.
- Refactor skill `.md` prompts to use `{{VARIABLE}}` syntax so they are importable into Workbench Evaluate tool for A/B prompt-version testing.
- Use Message Batches API for the nightly 28-agent Layer-1 run (50% cost reduction, async OK since overnight).

## References

- https://platform.claude.com/docs/en/test-and-evaluate/define-success
- https://platform.claude.com/docs/en/test-and-evaluate/eval-tool
- https://docs.anthropic.com/claude/docs/reducing-latency (redirect from /en/docs/test-and-evaluate/reducing-latency)
- https://platform.claude.com/docs/en/docs/test-and-evaluate/strengthen-guardrails/reduce-hallucinations
- https://platform.claude.com/docs/en/docs/test-and-evaluate/strengthen-guardrails/increase-consistency
- https://platform.claude.com/docs/en/docs/test-and-evaluate/strengthen-guardrails/mitigate-jailbreaks
- https://platform.claude.com/docs/en/docs/test-and-evaluate/strengthen-guardrails/reduce-prompt-leak
- https://platform.claude.com/docs/en/build-with-claude/structured-outputs (referenced from consistency + mitigate-jailbreaks docs)
- https://platform.claude.com/docs/en/build-with-claude/prompt-caching (referenced from latency doc)
- Internal: `backend/agents/llm_client.py`, `backend/agents/evaluator_agent.py`, `backend/slack_bot/assistant_handler.py`, `.claude/masterplan.json`, phase-4.11 audit `handoff/audit/phase-4.11/CONSOLIDATED_REPORT_v2.md`.
