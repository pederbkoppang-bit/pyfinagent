---
name: local-llm-74-0
description: Phase-74.0 local-Ollama research — the 80% hallucination figure is a GROUNDED-vs-UNGROUNDED artefact (5.7% on Vectara HHEM); a dry round caused by an exhausted search budget is a FALSE dry
metadata:
  type: project
---

Phase-74.0 (Ollama local LLMs on the 16GB M4 mini), researched 2026-08-18. Brief:
`handoff/current/research_brief_74.0.md`. Returned `gate_passed: false` on
`coverage.dry` ONLY — 31 sources read in full against a floor of 5.

**Why:** these are class-level traps that will recur, not facts derivable from code.

**How to apply:** read before any future local-model, quantization, or audit-class
research step.

## 1. The headline risk number was measuring the wrong thing
"Qwen3.5 4B/9B hallucinate 80-82%" (AA-Omniscience) is a **calibration** statistic on
**closed-book adversarial recall**: hallucination rate = incorrect answers as a share of
*non-correct* responses, with **no penalty for refusing**. On that board only three
models have EVER scored above zero and the frontier tops out at 43.
The independent Vectara HHEM board measures **grounded summarization** ("Summarize using
only the information in the given passage") and puts **qwen3-4b at 5.7% hallucination /
94.3% factual consistency — within 1.6 pts of Llama-3.3-70B (4.1%)**.
**A ~14x spread on the same size class, explained entirely by whether the answer is in
the prompt.** Before quoting any hallucination rate, ask *grounded or ungrounded* —
they are different benchmarks and predict different roles.
Corollary: model choice inside a size class dominates size — **Phi-4 3.7% vs
Phi-4-mini 23.5%**.

## 2. A dry round caused by an exhausted search budget is a FALSE dry
WebSearch is **session-shared** and hit 200/200 at round 9. Every later round could only
WebFetch guessable URLs, so round 16's zero came from a 404 plus a content-free page —
**dry by exhaustion of guessable URLs, not by coverage completeness**. Counting that
toward `K_required` would invert the meaning of the audit-class test. If the search
budget dies mid-audit, say so and return `dry: false`; do not let the loop "converge"
by starvation.

## 3. Small-model evidence splits by TASK TYPE, never by an average
Consistent across four independent sources: constraints/small models are **fine on
classification + extraction, bad on multi-step reasoning**. "Let Me Speak Freely?"
(JSON-mode costs up to 63 pts on reasoning but *helps* classification), FinBen
(extraction good, FinQA ~0.00 for open models), FAITH (Qwen-3-8B 30.6% on financial
numerics), 2406.11402 (weakest at comparative/relational reasoning). Never cite a single
average for a small model.

## 4. Vendor docs are silent exactly where the risk is
- Ollama's structured-output docs never state the mechanism (GBNF) or any failure mode.
  A practitioner source-read found the real ones: the grammar enforces **syntactic
  validity only**, truncation mid-JSON still yields invalid JSON, and **Ollama never
  validates the response against the schema**. "Schema-invalid output is impossible" is
  not an achievable success criterion.
- Ollama's `think` doc lists Qwen 3 / GPT-OSS / DeepSeek — **Qwen3.5 is absent**, and
  Qwen states there is no soft switch. "Thinking OFF" is an assumption, not a fact.
- The tool-calling doc documents **no** error handling and does **not** say whether
  `format` can combine with tools.
A documented ABSENCE is a finding; record it as one.

## 5. A closed vendor bug is not a fixed bug CLASS
ollama#14745 (qwen3.5 prints tool calls as text) was genuinely fixed by PR #15022,
merged 2026-03-27, shipped **v0.19.0**. But #16686 — same symptom, same parser family —
opened 2026-06-12 on **0.30.7** and is still open. Hand-written per-family parsers
recur. Also: the internal reassessment's advice to "pin 0.17.5" pins to **before** the
fix — a workaround harvested from an issue goes stale the moment the fix lands.

## 6. Re-derive every anchor; the masterplan's were all stale
All five phase-74 anchors had drifted: the schema skip is `llm_client.py:1249` (not
:1200-1202), `make_client` is `:2072` (not :2030-2039), fail-forward `:2146-2184` (not
:1983-2044), `_DEFAULT_PRICING` `cost_tracker.py:95` (not :83), `mas_communication`
`model_tiers.py:98` (not :57). Two files named in the spawn scope
(`slack_bot/assistant_handler.py`, `slack_bot/mcp_tools.py`) **do not exist**.

## 7. Run the frozen verification commands before trusting them
All four phase-74 commands were executed verbatim: **all honestly RED**, none vacuously
green — but three are bare substring greps (`local|ollama`) that prove a string was
typed, not that a rail works. Separately, 74.2 freezes "*~40 tok/s*", which **exceeds
the theoretical bandwidth ceiling** (120 GB/s / 3.4 GB = 35.3) for the model it pins.

## 8. Write-first collides with itself
My own incremental Write invalidates my earlier Read, so the next Edit fails "modified
since read". This looks exactly like a peer clobbering the file. Re-Read immediately
before each Edit, or batch into one Write — and do not spend turns hunting a phantom
second writer.
