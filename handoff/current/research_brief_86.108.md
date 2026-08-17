# Research Brief — step 86.108

**Topic:** Structured-output reliability across two LLM transports (Anthropic
constrained decoding / Claude Code `--json-schema` vs Google Gemini
`response_schema`), published JSON-failure rates + mitigations, the
"constraint tax" on semantic quality, LOUD-vs-SILENT degradation design, and
read-only runtime-config exposure.

**Tier:** moderate (caller-specified). **Audit-class:** YES (loop-until-dry,
K=2). **Role:** Layer-3 Researcher (external literature + internal code
inventory in one session).

---

## ENVELOPE (born inert — phase-86.37)

```json
{
  "brief_status": "COMPLETE",
  "tier": "moderate",
  "external_sources_read_in_full": 15,
  "snippet_only_sources": 20,
  "urls_collected": 35,
  "recency_scan_performed": true,
  "internal_files_inspected": 21,
  "coverage": {
    "audit_class": true,
    "rounds": 12,
    "dry_rounds": 2,
    "K_required": 2,
    "new_findings_last_round": 0,
    "dry": true
  },
  "gate_passed": true
}
```

*`internal_files_inspected` = 14 code/config/test files + 7 rotated log
archives. `urls_collected` = 15 read-in-full + 20 snippet-only.*

---

## Read in full (>=5 required; counts toward the gate)

| # | URL | Accessed | Kind | Fetched how | Key finding |
|---|-----|----------|------|-------------|-------------|
| 1 | https://arxiv.org/html/2408.02442v1 | 2026-08-17 | paper (EMNLP 2024 Industry, "Let Me Speak Freely?") | WebFetch, arXiv HTML | **The canonical constraint-tax paper.** "we observe a significant decline in LLMs' reasoning abilities under format restrictions." GSM8K text→JSON-mode: GPT-3.5-Turbo 75.99%→49.25% (-26.74pt); Claude-3-Haiku 86.51%→23.44% (-63.07pt); LLaMA-3-8B 74.73%→48.90%; **Gemini-1.5-Flash 89.33%→89.21% (essentially immune)**. Root cause is NOT parsing: "the parsing error rate for the Last Letter task in JSON format is only 0.148%, yet there exists a substantial 38.15% performance gap." Mechanism = **key ordering**: "100% of GPT 3.5 Turbo JSON-mode responses placed the 'answer' key before the 'reason' key, resulting in zero-shot direct answering instead of zero-shot chain-of-thought." Dropping the schema recovered Claude-3-Haiku GSM8K 23.44%→86.99%. Classification tasks IMPROVE (DDXPlus Gemini-1.5-Flash 41.59%→60.36%). Also measures raw parse-failure rates: Claude-3-Haiku **60.07%** parse errors on GSM8K-JSON, LLaMA-3-8B 22.75%, Gemini-Flash and GPT-3.5 near zero. |
| 3 | https://platform.claude.com/docs/en/build-with-claude/structured-outputs | 2026-08-17 | official doc (Anthropic) | WebFetch | **The API path DOES guarantee.** "Structured outputs guarantee schema-compliant responses through constrained decoding: Always valid: No more `JSON.parse()` errors." But the guarantee is **structural only** and has holes the project must not paper over: `minimum`/`maximum`/`multipleOf`/`minLength`/`maxLength` are **STRIPPED** from the wire schema (SDK moves them into `description` and re-validates client-side); `minItems` supported only for 0 or 1; recursive schemas and external `$ref` unsupported; `additionalProperties` **must be `false`**. **No guarantee is stated for `max_tokens` truncation or refusal stop-reasons** — a truncated response is still a broken body. Also: grammars are compiled + cached 24h, first use pays compile latency, and an extra system prompt is injected (input tokens rise, prompt cache invalidated). This corroborates `.claude/rules/research-gate.md`'s own claim that the research floors are not schema-expressible. |
| 4 | https://code.claude.com/docs/en/cli-reference | 2026-08-17 | official doc (Anthropic) | WebFetch | **The CC CLI path does NOT guarantee — it validates after the fact.** Verbatim: `--json-schema` = *"Get validated JSON output matching a JSON Schema **after the agent completes its workflow** (print mode only) … Claude Code exits with an error on an invalid schema and accepts the `format` keyword as an annotation without client-side validation."* "After the agent completes" is **post-hoc validation**, not constrained decoding. Also confirms **`--max-tokens` is not a CLI flag at all** (absent from the flag table), and `--output-format` takes `text` \| `json` \| `stream-json`. |
| 5 | https://arxiv.org/html/2604.06066v1 | 2026-08-17 | paper (preprint) | WebFetch, arXiv HTML | **The constraint tax, mechanised.** Coins **"structure snowballing"**: *"constrained decoding successfully suppresses divergent semantic hallucinations, [but] it forces the model into formatting traps and death loops."* Qwen3-8B / HotpotQA: accuracy **50.0% → 38.0%** under Outlines FSM constraint; 23 samples flipped correct→incorrect vs 11 recovered (McNemar p≈0.059). Degraded samples burned **4,005.5** mean tokens vs **2,850** for stable ones — *"This increasing cost of tokens provides empirical evidence for an 'alignment tax.'"* **96 of 100** first-round diagnoses collapsed to `FORMATTING_MISMATCH`, and **58 samples entered continuous "death loops"** repeating the identical formatting error. Mechanism: *"When the decoding library heavily restricts the output vocabulary, the model shifts its attention weights toward syntactic compliance rather than semantic reasoning."* Proposed mitigation (untested, no measured cost): **temporarily lift the constraint** when an agent repeats a constrained correction without success. |
| 6 | https://arxiv.org/html/2606.14589v1 | 2026-08-17 | paper (preprint) | WebFetch, arXiv HTML | **Direct hit on LOUD-vs-SILENT (sub-question e).** Five-class silent-failure taxonomy for a production LLM agent runtime; **Class C = "Error Swallowing and Dilution"** (exception status vanishes; cause stripped across hops) and **Class D = "Chained Hallucination / fail-plausible"**, called *"The most dangerous class."* Detection is overwhelmingly human: *"Human user-view observation: ~70%"*, while *"Unit tests/preflight: ≈0 for this corpus"*. Silence lasted **13 hours to 60 days**. **"≥28 distinct incidents … in which an error signal existed somewhere but never reached a human in actionable form."** Three-layer root cause = **trigger / amplifier / concealer**, where the concealer is *"a status file lying 'ok'"* or *"a fail-open guard"*. Defence maturation: point fix (recurs within 2 days) → meta-rule (memory-bound) → **mechanised scanner** (recurrence "structurally impossible"), and every guard must be proven by **sabotage validation** — it caught *"67 vacuous checks."* Measured **0% ex-ante prevention but 87% ex-post regression-blocking**. |
| 7 | https://arxiv.org/html/2605.02363v1 | 2026-08-17 | paper (preprint) | WebFetch, arXiv HTML | **Quantifies "valid ≠ usable" and prices the mitigations.** Defines `output accuracy = task_accuracy × json_valid`, arguing *"A response that solves the task but violates the output schema is as unusable as one that is simply wrong."* Shows the gap is bidirectional and brutal: Gemma/GSM8K had **88.4% underlying task accuracy but 0% output accuracy** (markdown wrapping alone). Schema-valid-but-semantically-wrong verified at **1.5–1.8%** of failures in one pool (0–0.97% under CONSTRAINED). **Costs, measured:** grammar-constrained decoding carries **3.6x–8.2x latency overhead** and "in several settings degrades task performance substantially" (Gemma: **52.4% of outputs were exact duplicates**); prompt-level optimisation ran at **0.63x–1.06x** baseline latency for a one-time cost of ~5,000–10,000 calls. Explicitly does **not** test retry-with-repair. |
| 8 | https://developers.openai.com/api/docs/guides/structured-outputs | 2026-08-17 | official doc (OpenAI) | WebFetch | **The third vendor — and the only one that documents how to DETECT the hole.** Guarantee: *"the model will always generate responses that adhere to your supplied JSON Schema."* Crucially it then names the two escape hatches and makes both **programmatically detectable**: a safety refusal *"does not necessarily follow the schema you have supplied"* and arrives in a dedicated `refusal` field; and truncation is caught by checking *"if (response.status === 'incomplete' and response.incomplete_details.reason === 'max_output_tokens')"*. Also: first request with a new schema pays schema-processing latency; `allOf`/`not`/`if`-`then`-`else` and top-level `anyOf` unsupported; **all fields must be `required`** and `additionalProperties:false` mandatory; and *"JSON Mode only guarantees valid JSON syntax; Structured Outputs guarantees schema adherence."* |
| 9 | https://owasp.org/Top10/2021/A05_2021-Security_Misconfiguration/ | 2026-08-17 | official standard (OWASP) | WebFetch | Sub-question (f). Vulnerable-to condition: *"Error handling reveals stack traces or other overly informative error messages to users."* Scenario #3: *"The application server's configuration allows detailed error messages … to be returned to users. This potentially exposes sensitive information or underlying flaws such as component versions that are known to be vulnerable."* Controls: *"A minimal platform without any unnecessary features, components, documentation, and samples"* and *"An automated process to verify the effectiveness of the configurations and settings in all environments."* **Honest limit of this source: it "does not specifically address API endpoints exposing configuration values to authenticated users"** — the pyfinagent case is narrower than A05's scenarios, so A05 constrains *what* may be exposed (no secrets, no versions, no stack traces) but does not forbid an authenticated read-only config view. |
| 10 | https://arxiv.org/html/2501.10868v3 | 2026-08-17 | paper (JSONSchemaBench) | WebFetch, arXiv HTML | **[ADVERSARIAL — contradicts source 1 and source 5.]** 10,000 real-world schemas, 6 engines (Guidance, Outlines, Llamacpp, XGrammar, OpenAI, Gemini). Introduces the three metrics this step needs: **declared coverage** (framework accepts the schema), **empirical coverage** (outputs are actually compliant), **compliance rate** = empirical/declared. The gap between them is the whole story: on GitHub-Hard, Outlines declares **0.47** but empirically covers **0.03** — a **0.06 compliance rate**, i.e. a framework that *accepts* a schema and then fails to honour it 94% of the time. And on quality it reports the **opposite sign** to sources 1/5: *"Constrained decoding, regardless of the framework, achieves higher performance than the unconstrained setting"*, *"consistently improves the performance of downstream tasks up to 4%, even for tasks with minimal structure like GSM8k."* Efficiency also inverts the folklore: *"Constrained decoding can speed up the generation process by 50% compared to unconstrained decoding"* (Guidance TPOT 6.37ms vs 15.40ms unconstrained), though Outlines pays 3.48s grammar compilation. |
| 11 | https://arxiv.org/html/2605.26128v1 | 2026-08-17 | paper (preprint, "The Constraint Tax") | WebFetch, arXiv HTML | **The single most directly applicable source, and it supplies the metric this step should adopt.** Defines `Tax = max(0, Acc(baseline) − Acc(constrained))` and, critically, the **wrong-valid-schema rate**. Aggregate over 15,000 generations on sub-3B models: answer accuracy **19.7% → 11.0%** while schema validity went **61.5% → 100.0%** and **wrong-valid-schema rose 49.5% → 88.9%**. The calendar tool-call analogue isolates semantics (both modes 100% valid): executable accuracy **91.5% → 48.0%**, a **43.5-point** tax, with *"102 of 104 hard-schema failures are single-field duration errors."* Tax persists at 3B (15.3 pts). Mitigation **"Reason free, constrain late"** is measured, not theoretical: delayed constraint scored **40.7% executable at 100% validity** vs 24.5% for prompt-only JSON, and deterministic re-serialisation of a free first stage showed **0.0 executable tax**. Money quote for the design: *"A valid JSON object can still encode the wrong decision, so a dashboard that tracks parse success alone can improve while downstream execution gets worse."* Recommends: *"Track wrong-valid-schema rate as a first-class reliability metric"* and *"Treat schema validity as an interface SLO, not as a task-success metric."* |
| 2 | https://ai.google.dev/gemini-api/docs/structured-output | 2026-08-17 | official doc (Google) | WebFetch | Gemini's own doc is **cautious, not absolute**: responses "adhere to a provided JSON Schema" / output is "syntactically correct JSON" — it never claims a hard guarantee, and the page carries **no** "strong hint" language and **no** discussion of MAX_TOKENS truncation, tuned-model quality loss, schema-token cost, or thinking-mode interaction. Documented limitation verbatim: *"Very large or deeply nested schemas may be rejected."* Supported keywords include `anyOf` and `$ref` (contradicting older secondary sources that said anyOf is unsupported); the page says only "not all JSON Schema features are supported" without enumerating the gaps. |

| 12 | https://www.getunleash.io/blog/feature-flag-security-best-practices | 2026-08-17 | vendor engineering doc (Unleash) | WebFetch | Fills the gap OWASP A05 explicitly leaves. Client-side flag state is **not** a security boundary: *"A knowledgeable user can inspect the network traffic, find the feature flag payload, and modify the JavaScript state to 'enable' a hidden feature in their browser"* — authorization must stay server-side regardless of what a read-only endpoint reports. Least privilege: *"A developer might need write access in testing environments but should only hold read access in production."* Never put *"secret keys or future pricing data"* in exposed payloads. Audit entries must carry identity, timestamp, action, **"before" and "after" configuration states**, and source IP. Production changes should follow a **four-eyes principle**: *"No single individual can initiate and approve a change to a production flag without a secondary review."* |

| 13 | https://arxiv.org/html/2607.18261v1 | 2026-08-17 | paper (preprint, 2026-07) | WebFetch, arXiv HTML | **The most recent source found, and it prescribes the exact degradation policy this step needs.** Schema-constrained ordering agents: *"A JSON Schema can ensure that items is an array and quantity is an integer. It cannot ensure that a requested allergen conflict was rejected."* At **100% schema validity**, semantic success was 83.0% (GPT-OSS-120B), 30.7% (Qwen3-30B), **2.0%** (Gemma-2-2B). Defines **unsafe acceptance** = *"status=accepted for an object that the verifier says must not be sent to execution"* — **41.7%** for Gemma-2-2B *at 100% schema validity*; 16.1% across all 2,400 cases under JSON-schema mode. Proposes a **four-layer stack** (syntactic → schema → semantic/domain → execution) whose execution layer is the direct answer to sub-question (e): **"Fail closed on verifier errors, using clarification rather than auto-repair."** Monitoring rule: *"Log model output and verifier decisions as paired artifacts for regression testing."* |
| 14 | https://arxiv.org/html/2604.25359v1 | 2026-08-17 | paper (Structured Output Benchmark) | WebFetch, arXiv HTML | **The cross-provider number that covers pyfinagent's own stack.** 21 models across OpenAI / Google / Anthropic / open-weight, 7 dimensions. Headline: **"Every model exceeds 84% JSON Pass, yet no model surpasses 80.4% Value Accuracy."** Per-provider JSON-pass vs value-accuracy: GPT-5.4 **99.97% / 79.8%**, Claude-Sonnet-4.6 **97.9% / 77.9%**, **Gemini-2.5-Flash 97.2% / 79.6%**, GLM-4.7 97.2% / 83.0% — a **~14-20 point gap** at every provider. *"Schema compliance is high, but correct field values remain much harder"* and *"schema adherence is not the bottleneck: grounded value extraction is."* Failure archetype: a model emitting `'American country music artist'` where the source supports only `'country music artist'` — *"The JSON is valid. The type is correct, but the data is wrong."* |

| 15 | https://code.claude.com/docs/en/agent-sdk/structured-outputs | 2026-08-17 | official doc (Anthropic) | WebFetch | **Completes the answer to (a) for the CC rail, and lands squarely on this project's own rail-drop problem.** Mechanism is explicit and is **neither** constrained decoding **nor** bare post-hoc validation: *"the SDK validates the output against it, **re-prompting on mismatch**. If validation does not succeed within the retry limit, the result is an error instead of structured data"*, surfacing as subtype **`error_max_structured_output_retries`**. Three findings the project must act on: (i) **"A result can also end with subtype `success` but no `structured_output` value … Treat that case as a failure as well."** — Anthropic's own instruction for exactly the mode pyfinagent logs as a rail drop; (ii) a **model fallback "can retract an already-completed output mid-stream, and if no retry replaces it the run ends with the same error. Check the `errors` list on the result message to tell the two causes apart"** — so a drop is not necessarily schema-related; (iii) the SDK validates against **JSON Schema draft-07** and *"schemas that declare a newer version are rejected"* (Zod defaults to 2020-12 → must pass `target: "draft-7"`). Also a documented SILENT-failure fix: *"Before v2.1.205, an invalid schema was silently ignored and the agent returned unstructured text."* |

## Identified but snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|-----|------|-------------------------|
| https://aclanthology.org/2024.emnlp-industry.91.pdf | paper (PDF) | Same paper as source 1; deliberately read via arXiv HTML instead — project rule forbids WebFetch on PDFs (they fabricate quotes). |
| https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/capabilities/control-generated-output | official doc (Google Cloud) | **Fetch ATTEMPTED and FAILED** — returned nav/TOC only, no body. Known behaviour: `cloud.google.com` reference pages are JS-rendered. This is why the "strong hint" wording could not be corroborated from a primary source (see Pitfalls). |
| https://arxiv.org/abs/2305.13971 | paper | Founding GCD paper (year-less canonical hit); superseded for this step's purposes by sources 10/11 which carry the measurements. |
| https://arxiv.org/abs/2502.05111 | paper | "Flexible and Efficient Grammar-Constrained Decoding" (ICML 2025); engine-efficiency angle already covered by source 10. |
| https://aclanthology.org/2025.acl-industry.34/ | paper | GCD improves logical parsing (+2.7% low-data); corroborates source 10's direction, adds no new mechanism. |
| https://openreview.net/forum?id=wKs9fHYxCV | paper | CRANE (interleaved free/constrained generation) — the mitigation family source 11 measures as "constrain late". |
| https://arxiv.org/html/2601.07525v2 | paper | "Thinking Before Constraining" — same mitigation family. |
| https://arxiv.org/pdf/2606.04056 | paper | Token-budget overrun catalogue; retry-cost adjacent, superseded by source 11's measured tax. |
| https://github.com/googleapis/python-genai/issues/706 | community | Gemini 2.0-vs-2.5 structured-output inconsistency; corroborates source 2's caution, community tier. |
| https://github.com/googleapis/python-genai/issues/460 | community | Same class. |
| https://arxiv.org/abs/2606.14589v1 | paper (abs page) | Abstract page for source 6; the HTML full text was read instead. |
| https://futureagi.substack.com/p/why-do-multi-agent-llm-systems-fail | blog | MAST taxonomy figures (42/37/21%) — community tier, unverified provenance; not relied on. |
| https://www.digitalapplied.com/blog/llm-structured-output-json-reliability-production | blog | Source of the widely-repeated "8–15% broken JSON" figure; traced to a vendor's own 2M-call analysis, not independently verifiable — deliberately NOT used as a load-bearing number. |
| https://tianpan.co/blog/2026-04-16-retry-budget-llm-agent-cost-amplification | blog | Retry-budget cost amplification; community tier. |
| https://docs.getunleash.io/guides/feature-flag-best-practices | vendor doc | Overlaps source 12. |
| https://arxiv.org/pdf/2604.09360 | paper | "LLM-Rosetta" cross-provider API translation; adjacent (this step is not building a translation layer). PDF-only URL — project rule forbids WebFetch on PDFs. |
| https://github.com/googleapis/python-genai/issues/782 | community | Thinking-budget vs `response_schema` unreliability; corroborates the in-repo phase-37.1 guard note. Community tier. |
| https://github.com/anthropics/claude-agent-sdk-python/issues/571 | community | Agent wraps output in `{"output": {...}}` and fails root-level validation. Superseded by source 15's primary account. |
| https://www.requesty.ai/blog/structured-outputs-across-llm-providers-the-compatibility-mess | blog | "244 models tested" portability survey; vendor-authored, not independently verifiable. |
| https://json-schema.org/draft/2020-12 | spec | Draft 2020-12 reference; relevant only via source 15's draft-07 constraint. |

## Recency scan (last 2 years, 2024-2026) — PERFORMED

Searched explicitly across the window using the mandated three-variant
composition (current-year `2026`, last-2-year `2025`, and year-less canonical):

- **Year-less canonical** (`grammar constrained decoding language models`,
  `constrained decoding degrades LLM reasoning format restrictions`) surfaced
  the founding prior art — `arXiv:2305.13971` (GCD without finetuning) and the
  EMNLP-2024 "Let Me Speak Freely?" paper, which a year-locked query buried.
- **Last-2-year / current-year** surfaced sources 5, 6, 7, 10, 11.

**Result: 5 new findings in the window that materially change the picture, and
one that SUPERSEDES the canonical source.**

1. **Source 10 (JSONSchemaBench) partially supersedes source 1.** The 2024 paper's
   headline (JSON mode costs 10-60 accuracy points) is **not** reproduced by the
   10K-schema benchmark, which finds constrained decoding *improves* downstream
   accuracy up to 4% and can be **50% faster**. The reconciliation is in
   "Consensus vs debate" below — this is the highest-value recency finding and
   it is why source 1 must not be quoted alone.
2. **Source 11 supplies `wrong-valid-schema rate`** (2026), a metric that did
   not exist in the 2024 framing and is the correct KPI for this step.
3. **Source 6 (2026)** gives an empirical silent-failure taxonomy with the
   trigger/amplifier/**concealer** structure.
4. **Anthropic structured outputs went GA 2026-02-04** (source 3) — the entire
   Anthropic half of this question is newer than the canonical literature.
5. **Source 5 (2026)** names "structure snowballing" / death loops, a failure
   mode absent from the 2024 work.

## Key findings

1. **"Structured output" names three different guarantee classes, and this
   repo uses two of them.** Anthropic's API guarantees via constrained decoding
   — *"Always valid: No more `JSON.parse()` errors"* (source 3); the **Claude
   Code CLI does not** — `--json-schema` yields *"validated JSON output …
   **after the agent completes its workflow**"* (source 4), i.e. post-hoc
   validation; Gemini's doc claims only that responses *"adhere to a provided
   JSON Schema"* and never states an absolute guarantee (source 2). **A config
   written for one transport does not bind on the other.**
2. **The guarantee, where it exists, is structural only — and every vendor
   leaves the same hole.** Truncation. Anthropic's doc is silent on it
   (source 3); Gemini's is silent on it (source 2); **only OpenAI documents the
   detection** — `status === 'incomplete' && incomplete_details.reason ===
   'max_output_tokens'` (source 8). A `max_output_tokens` cut mid-object
   produces a broken body under *every* one of these regimes.
3. **Schema constraints are enforced by STRIPPING what they cannot express.**
   Anthropic removes `minimum`/`maximum`/`minLength`/`maxLength`, caps
   `minItems` at 0-or-1, and re-validates client-side (source 3) — the exact
   mechanism `.claude/rules/research-gate.md` already relies on for this
   project's own research floors. Expecting a numeric bound to bind server-side
   is a category error.
4. **The constraint tax is real, large, and semantic rather than syntactic.**
   Source 11: schema validity **61.5% → 100%** while answer accuracy fell
   **19.7% → 11.0%** and **wrong-valid-schema rose 49.5% → 88.9%**; the calendar
   analogue lost **43.5 points** of executable accuracy at a constant 100%
   validity. Source 1's mechanism — key ordering forcing the answer before the
   reasoning, *"100% of GPT 3.5 Turbo JSON-mode responses placed the 'answer'
   key before the 'reason' key"* — explains why.
5. **Therefore parse-success is the wrong KPI.** *"A valid JSON object can still
   encode the wrong decision, so a dashboard that tracks parse success alone can
   improve while downstream execution gets worse"* (source 11). The
   recommendation is explicit: *"Treat schema validity as an interface SLO, not
   as a task-success metric."*
6. **The mitigation with the best measured evidence is "reason free, constrain
   late."** Source 11 measures delayed constraint at **40.7% executable / 100%
   valid** vs 24.5% for prompt-only JSON, and **0.0 executable tax** for
   deterministic re-serialisation of a free-form first stage. Source 1's
   NL-to-Format and source 5's "temporarily lift the constraint" are the same
   family; source 5's variant is **unmeasured** and should not be cited as
   costed.
7. **Silent failure is a design defect with a name and a fix path.** Source 6:
   the **concealer** layer — *"a status file lying 'ok'"*, *"a fail-open
   guard"* — is what converts an error into a silence; detection was ~70% human
   observation and **≈0% from unit tests**; remediation matures point fix →
   meta-rule → **mechanised scanner**, and every new guard must be proven by
   **sabotage validation** (which caught *"67 vacuous checks"*).
8. **Exposing config read-only is defensible but has named controls.** OWASP A05
   forbids leaking secrets/versions/stack traces and requires *"an automated
   process to verify the effectiveness of the configurations … in all
   environments"* (source 9), but **does not address authenticated config
   endpoints** — so source 12 supplies the operative rules: read-only in
   production, never ship secrets in the payload, log identity + before/after +
   timestamp, and never treat reported flag state as an authorization boundary.

## Consensus vs debate (external)

**Consensus.** (a) Constrained decoding makes *syntactic* validity essentially
free (sources 3, 8, 10, 11). (b) Validity is not correctness (1, 5, 7, 11).
(c) Vendors solved validity but **not portability** — the same schema is legal
on one provider and rejected on another (3, 8, 2 read together). (d) Truncation
and refusal are outside the guarantee everywhere (3, 8).

**Genuine debate — and it is a sign reversal, not a nuance.**

| Position | Sources | Claim |
|---|---|---|
| Constraints **hurt** | 1, 5, 7, 11 | GSM8K -26 to -63 pts (1); 50.0%→38.0% (5); 43.5-pt executable tax (11) |
| Constraints **help** | 10 | *"consistently improves … up to 4%"*, and **50% faster** |

**Reconciliation — do not average these; they measure different populations.**
Three discriminators, all supported by the sources themselves:
1. **Model scale.** The "hurt" results cluster on **sub-3B** models (source 11
   is explicitly a sub-3B suite; source 5 is Qwen3-8B). Source 11 checked the
   3B boundary and the tax **persisted**, so scale attenuates but does not
   remove it.
2. **Whether reasoning must precede the answer.** Source 1's own control settles
   this: **removing the schema recovered Claude-3-Haiku GSM8K from 23.44% to
   86.99%** — the damage was key ordering, not constraint per se. Where the
   schema permits reasoning first, the tax largely vanishes.
3. **Task type.** Source 1 finds classification *improves* (DDXPlus
   41.59% → 60.36%); source 10's benchmark skews to extraction/parsing, which is
   classification-like. pyfinagent's Synthesis and Critic are **reasoning**
   tasks; its Moderator consensus is closer to classification.

Also note source 1's own Gemini result — **89.33% → 89.21%, i.e. no tax** — is
the most directly transferable single number in the whole external set, since
the Gemini transport is the one carrying this repo's structured configs.

## Pitfalls (from the literature, and one from this session)

1. **Quoting a "failure rate" without its denominator.** The widely-cited
   "8-15% broken JSON" traces to a single vendor's own analysis and is not
   independently verifiable — recorded snippet-only and deliberately unused.
   The same trap is live in-repo: see census correction **C-d**.
2. **Trusting `declared` coverage.** Source 10: Outlines declared 0.47 and
   empirically covered **0.03** on GitHub-Hard. A framework accepting a schema
   is not evidence it honours it — the direct analogue of "committed is not in
   force."
3. **A dashboard that improves while the system degrades** (source 11) — the
   specific risk of instrumenting parse-success without wrong-valid-schema.
4. **Death loops under constraint** (source 5): 58 samples repeated the
   identical formatting error across all permitted trials, and degraded samples
   burned 4,005 vs 2,850 tokens. **Any repair-retry design needs a hard attempt
   cap**, which mirrors this project's own F1b cumulative-budget doctrine.
5. **Fail-open guards are concealers** (source 6). Directly relevant: the
   settings route's `reschedule_paper_job` is explicitly fail-open
   (`settings_api.py:478-479`) and the live_check gate is fail-open by design.
   A new observability path must not inherit that pattern.
6. **Session pitfall, recorded for honesty:** the Google Cloud "strong hint"
   wording appeared in a *search snippet* and could **not** be corroborated —
   the primary page fetched as nav-only JS. It is therefore **not** asserted as
   a finding anywhere in this brief.

## Internal code inventory

| File | Lines | Role | Status |
|---|---|---|---|
| `backend/agents/orchestrator.py` | 129-183 | 6 Gemini structured configs | **2 of 6 DEAD** (`_THINKING_CRITIC_CONFIG`, `_THINKING_MODERATOR_CONFIG` — 0 refs, positive-control verified) |
| `backend/agents/orchestrator.py` | 308-316 | `_parse_json_with_fallback` — returns `None` on failure | LIVE; **silent** (warning only, no counter/record) |
| `backend/agents/debate.py` | 39-51 | `_MODERATOR_GEN_CONFIG` + `_DA_STRUCTURED_CONFIG` + `_MODERATOR_STRUCTURED_CONFIG` | LIVE, schema'd; still produced 359 failures |
| `backend/agents/debate.py` | 127 | emit site 2 | LIVE; falls back to raw text |
| `backend/agents/risk_debate.py` | 123 | emit site 3 | LIVE; falls back to raw text |
| `backend/agents/risk_debate.py` | 127-142, 339 | `_judge_parse_fail_fallback` | LIVE; **fabricates a verdict by default** (see below) |
| `backend/agents/llm_parse.py` | 149 | emit site 4 | LIVE; falls back to raw text |
| `backend/agents/claude_code_client.py` | 300, 379-380, 393 | CC rail: schema passed, `max_tokens` no-op | LIVE; `--json-schema` is post-hoc per source 4 |
| `backend/agents/claude_code_client.py` | 713-731 | schema derivation + **LOUD** fall-through | LIVE; the in-repo model for loud degradation |
| `backend/api/settings_api.py` | 65-127, 350-401, 406-416 | `GET /api/settings/` exposed key set | LIVE; **omits all 5 integrity flags + both diversity flags** |
| `backend/api/settings_api.py` | 281-285 | `_FIELD_TO_ENV` rows for those 5 flags | **UNREACHABLE** (no matching `SettingsUpdate` fields) |
| `backend/api/settings_api.py` | 409-415, 469 | `settings:full` API cache | LIVE; can serve stale config |
| `backend/config/settings.py` | 346-352 | `paper_risk_judge_parse_fail_reject` (DARK, default False) | LIVE; **OFF = silent APPROVE_REDUCED at 3% NAV** |
| `backend/config/settings.py` | 487-488 | `paper_soft_sector_diversity_*` | LIVE; **absent from settings_api.py entirely** |
| `backend/services/autonomous_loop.py` | 2671-2679, 2810, 3166, 3172, 3272-3273 | `_parse_failed` / `_degraded` marked-record idiom | LIVE; **the existing loud-marking mechanism** |
| `backend/api/observability_api.py` | 25-64 | freshness / data-freshness / latency | LIVE; **no config or flag exposure precedent exists** |
| `backend/tests/test_phase_37_4_moderator_schema.py` | 1-40 | Moderator schema regression lock | LIVE; criterion #2 **still unmet** |

## Application to pyfinagent

**A. The rail split (C1) cannot be produced from the logs, and the brief says so
rather than inventing one.** No marker line carries model/provider/rail
(`grep -c '"model"'` → 0). `pyfinagent_data.llm_call_log` **does** carry
`provider STRING NOT NULL` and `model STRING NOT NULL` (plus `agent`, `ts`,
`request_id`, `ok`) per `scripts/migrations/add_llm_call_log.py`, so it is the
only viable source — **but two obstacles must be stated in the contract, not
discovered later**: (i) the warning lines carry **no `request_id`**, so any join
is time-proximity + agent-label, a heuristic; (ii) `ok` is defined as *"true on
2xx, false on exception"*, and an invalid-JSON body **is a 2xx** — so
`llm_call_log` can supply the *rail for a window*, but can never by itself
identify a parse failure. The honest deliverable is a rail split **by era**
(bucketed on `paper_use_claude_code_route`), explicitly labelled as such.

**B. The strongest empirical result already in hand: Gemini's `response_schema`
did not prevent invalid JSON here.** 359 Moderator failures occurred with
`_MODERATOR_STRUCTURED_CONFIG` in force. Externally, source 2 shows Google never
promised an absolute guarantee, and source 10 shows declared≠empirical coverage.
This refutes any plan premised on "the schema makes it unreachable."

**C. Adopt source 11's metric, not a parse-success counter.** The step's instinct
— make failures countable — is right, but counting *parses* is the exact trap
source 11 names. The counter should distinguish at minimum: `parse_failed`
(syntactic), `schema_valid_but_rejected_downstream` (the wrong-valid-schema
proxy), and `truncated` (detectable via finish-reason, per source 8's pattern).

**D. The LOUD mechanism already exists — extend it, don't invent one.**
`_parse_failed` is already computed, persisted, and escalated to `_degraded` at
`autonomous_loop.py:3272-3273`. Three concrete gaps: (i) the escalation is gated
on `paper_synthesis_integrity_enabled`, which is **invisible** in the settings
API — so an operator cannot tell whether marking is active; (ii) the four emit
sites do not feed it (they only `logger.warning`); (iii)
`_judge_parse_fail_fallback` **fabricates a default verdict** — with the flag
OFF, *"a garbled/empty judge response silently becomes APPROVE_REDUCED at 3%
NAV."* That is precisely the "fabricating a default verdict" the step forbids,
it is the **shipped default**, and source 6 would classify it as a Class-D
fail-plausible concealer. Note the correct fix is *marking*, not flipping a risk
default mid-step — flipping it is a behaviour change requiring its own step.

**E. `GET /api/settings/` answers the step's question with a clean NO** — and the
phase-61.2 comment at `:279-280` asserts the opposite of what the code does.
Adding the seven missing keys to `FullSettings` is a small, read-only change that
also makes the five `_FIELD_TO_ENV` rows reachable. Controls from sources 9+12
that should land with it: no secrets in the payload (the route already models
this correctly with `*_key_configured` **booleans** rather than key values —
keep that idiom); read-only in production; and be aware the `settings:full`
cache (`:409-415`) can serve a stale value, which for an observability endpoint
is itself a silent-failure risk (source 6's *"status file lying 'ok'"*).

**F. Any repair-retry must be attempt-capped.** Source 5's 58 death-loop samples
and rising token curve, plus source 11's evidence that constraint-hardening
*raises* wrong-valid-schema, mean an unbounded repair loop can burn budget while
degrading decisions. This aligns with the project's existing F1b doctrine.

## Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch — **15**
- [x] 10+ unique URLs total (incl. snippet-only) — **35** (15 read-in-full + 20 snippet-only)
- [x] Recency scan (last 2 years) performed + reported — yes, 5 findings, 1 supersedes
- [x] Full papers / pages read (not abstracts) — arXiv HTML chain used; no PDF WebFetch; one failed fetch disclosed
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every named module + the census
- [x] Contradictions / consensus noted (sign reversal documented + reconciled)
- [x] All claims cited per-claim
- [~] **Brief exceeds the `moderate` 700-word guidance.** Disclosed deliberately:
  the audit-class loop plus a mandated internal census/inventory are not
  compressible without dropping load-bearing measurements. Depth of *analysis*
  was kept at moderate tier.

## Coverage log (audit-class, K=2)

| Round | Activity | New read-in-full findings | Dry? |
|---|---|---|---|
| 1 | 3 searches (frontier/canonical/vendor) + census start | 3 (src 1,2,3) | no |
| 2 | CC CLI + alignment-tax + llm_call_log schema | 2 (src 4,5) | no |
| 3 | silent-failure taxonomy + valid-vs-usable + config security | 3 (src 6,7,9) | no |
| 4 | cross-vendor OpenAI + in-repo LOUD precedents | 1 (src 8) | no |
| 5 | repair/overhead + OWASP → surfaced JSONSchemaBench + Constraint Tax | 0 (leads only) | no |
| 6 | fetched both leads | 2 (src 10,11) | no |
| 7 | recency + year-less canonical + flag-observability | 1 (src 12) | no |
| 8 | wrong-valid-schema metric / tool-forcing / parse-failure instrumentation | 2 (src 13,14) | no |
| 9 | semantic verifier + fail-closed agent | **0** — only community/vendor-marketing tier (futureagi glossary, forums) | **DRY 1** |
| 10 | OTel GenAI conventions + Claude Code SDK schema behaviour | 1 (src 15) — **dry streak RESET** | no |
| 11 | draft-07 vs 2020-12 portability | **0** — community blogs restating primary docs already read (src 3/8/15) | **DRY 1** |
| 12 | Gemini thinking-vs-schema + multi-provider abstraction | **0** — GitHub issues + vendor blogs; one arXiv (LLM-Rosetta) adjacent, not design-deciding | **DRY 2** |

**Loop-until-dry terminated legitimately: `dry_rounds = 2 = K_required`, so
`coverage.dry = true`.** Note round 10 **reset** a dry streak that had already
reached 1 at round 9 — declaring dryness at round 9 would have missed source 15,
which is the single most operationally relevant source in the set (it names the
`success`-with-no-`structured_output` mode this project logs as a rail drop).
That reset is the audit-class loop doing exactly its job, and it is recorded
here rather than smoothed over.

## Open questions for PLAN (not resolved by this brief)

1. **The rail split is not derivable as specified.** Contract should either
   scope C1 to an **era-bucketed** split (via `paper_use_claude_code_route`) or
   drop the per-event rail claim. A per-event split would require a join key
   that does not exist.
2. **The census is blind after 2026-08-14T15:53Z** (no live uncompressed
   `backend.log`). If a current rate is needed, locating the live log is a
   prerequisite, not an assumption.
3. **Whether to flip `paper_risk_judge_parse_fail_reject`** is a risk-behaviour
   change, out of scope for an observability step; this brief recommends
   *marking* the fabricated-default path, not changing it.
4. **`llm_call_log` liveness was not queried.** The schema is confirmed from
   `scripts/migrations/add_llm_call_log.py`; whether rows exist for the census
   window was deliberately NOT checked, because `mcp__bigquery__execute-query`
   is approval-gated and this is an unattended run. Verify before depending on
   it.

---

## INTERNAL: the census, RE-DERIVED (not inherited)

### Exact method

- **Glob:** `handoff/logs/backend.log.*.gz` — **7 files**, all gzipped, rotated
  `20260612T104931Z` → `20260814T155315Z`. Read with `gzcat`.
- **Match rule:** fixed-string `grep -c "returned invalid JSON"` (case-sensitive,
  counts **LINES**, and a line may match at most once).
- **There is NO live uncompressed `handoff/logs/backend.log`.** The census
  therefore covers only rotated history and is **blind to everything since
  2026-08-14T15:53Z**. Any "current rate" claim from this corpus is
  unsupported.

### Result — the headline total REPRODUCES

| File | Marker lines |
|---|---|
| backend.log.20260612T104931Z.gz | 939 |
| backend.log.20260706T225648Z.gz | 792 |
| backend.log.20260724T064045Z.gz | 640 |
| backend.log.20260729T171222Z.gz | 192 |
| backend.log.20260804T182713Z.gz | 91 |
| backend.log.20260810T064130Z.gz | 146 |
| backend.log.20260814T155315Z.gz | 59 |
| **TOTAL** | **2859** |

The per-agent split also reproduces the step text **exactly**: Analyst 926,
Critic 602, Moderator 359, Advocate 342, Judge 314, Synthesis-Final 264,
Critic-Retry 52 (sum = 2859).

### ...but four corrections make the split mean something different

**C-a. The corpus is MIXED-FORMAT; a single parser sees only 17%.**
488 of the 2,859 lines are `JsonFormatter` records
(`{"timestamp","level","module","message"}`); the other **2,371** are
`CompactFormatter` lines with ANSI colour codes
(`\e[33m19:11:09 W [debate]\e[0m Moderator returned invalid JSON...`). A
`"module":`-keyed parse returns 488 and looks complete. Per
`.claude/rules/backend-api.md`, the formatter is chosen by `DEBUG`, so the
corpus spans both settings.

**C-b. The step's agent labels are a MATCH-RULE ARTIFACT, not agent names.**
Taking the last token before the phrase collapses distinct agents:
"Analyst" 926 is **not one agent** — it is `Conservative Analyst` (259+50=309)
+ `Neutral Analyst` (258+52=310) + `Aggressive Analyst` (258+49=307). Likewise
"Advocate" = `Devil's Advocate` and "Judge" = `Risk Judge`. There is no agent
called "Analyst".

**C-c. The Critic path is DOUBLE-LOGGED, so 2,859 counts lines, not events.**
In the compact corpus, `Critic returned invalid JSON` appears **274** times and
`Critic returned invalid JSON, treating as PASS with draft.` appears **274**
times — exactly equal, i.e. one failure emitting two lines. 274+274+54 = 602 =
the "Critic 602" figure. The second wording survives today **only as a negative
assertion in a test** (`backend/tests/test_phase_75_skill_delivery.py:260`
asserts `"treating as PASS with draft" not in src`), so phase-75 removed it —
the corpus spans **multiple code generations** and a rate over the whole of it
mixes builds.

**C-d. `9.2%` is a COMPOSITION SHARE, not a failure rate.** 264/2859 = **9.23%**
— Synthesis-Final's share of invalid-JSON *log lines*. It is **not** "9.2% of
synthesis calls returned invalid JSON"; no synthesis-attempt denominator exists
in this corpus (candidate markers `Synthesis complete` / `Running Synthesis` /
`Analysis complete` all return **0**). Quoting 9.2% as a rate would be
unreproducible, exactly as the step text warned.

### C1 — RAIL ATTRIBUTION IS IMPOSSIBLE FROM THESE LOGS (explicit finding)

The full field set of a JSON marker record is **`timestamp`, `level`, `module`,
`message`** — nothing else. `grep -c '"model"'` over the newest rotated log
returns **0**. **No log line carries a rail, provider, or model.** All 2,859
events are therefore **unattributable to `claude_code` vs `gemini` from the log
alone**, and any presented "rail split" would be fabricated.

Three candidate sources that *could* supply the split, in descending strength:
1. **`pyfinagent_data.llm_call_log` in BigQuery** — needs a per-call model/
   provider column AND a join key (ticker + timestamp) to these lines; the log
   lines carry a timestamp but **no request id**, so any join is a
   time-proximity heuristic, not a key join.
2. **Config-at-the-time** — `paper_use_claude_code_route` (settings_api.py:122,
   :173, :324) is the rail switch; its value over time would let events be
   bucketed by era, not by call.
3. **Code path** — `debate.py:127` / `risk_debate.py:123` / `llm_parse.py:149` /
   `orchestrator.py:315` are rail-agnostic; they sit above the client, so the
   emit site does not identify the transport.

### The emit surface is FOUR sites, not one

| File:line | Wording | Note |
|---|---|---|
| `backend/agents/orchestrator.py:315` | `f"{agent_name} returned invalid JSON"` | inside `_parse_json_with_fallback` (:308-316); returns `None` |
| `backend/agents/debate.py:127` | `f"{label} returned invalid JSON, using raw text"` | falls back to raw text |
| `backend/agents/risk_debate.py:123` | `f"{label} returned invalid JSON, using raw text"` | falls back to raw text |
| `backend/agents/llm_parse.py:149` | `"%s returned invalid JSON, using raw text"` | falls back to raw text |

`orchestrator.py:308-316` is the SILENT-degradation core:

```python
def _parse_json_with_fallback(json_string: str, agent_name: str) -> Optional[dict]:
    try:
        data = json_io.loads(json_string)
        if isinstance(data, str):
            return json_io.loads(data)
        return data
    except json.JSONDecodeError:
        logger.warning(f"{agent_name} returned invalid JSON")
        return None
```

A `WARNING` to a rotating file is the **only** trace: no counter, no BQ row, no
field on the analysis record. `None` then flows downstream as "absent input" —
indistinguishable from "the agent had nothing to say". This is precisely the
LOUD-vs-SILENT problem the step names, and it is why the census had to be
reconstructed from log archaeology at all.

## INTERNAL: the two transports

### Gemini side — `orchestrator.py:129-183`

Six structured configs, all `response_mime_type: "application/json"` +
`response_schema: <Pydantic model>`, i.e. the **Gemini** contract:

| Const | Line | max_output_tokens | Schema |
|---|---|---|---|
| `_SYNTHESIS_STRUCTURED_CONFIG` | :129 | 4096 | `SynthesisReport` |
| `_CRITIC_STRUCTURED_CONFIG` | :140 | 6144 | `CriticVerdict` |
| `_THINKING_CRITIC_CONFIG` | :153 | 6144 | `CriticVerdict` |
| `_THINKING_MODERATOR_CONFIG` | :160 | 2048 | **none** |
| `_THINKING_RISK_JUDGE_CONFIG` | :165 | 1536 | `RiskJudgeVerdict` |
| `_THINKING_SYNTHESIS_CONFIG` | :177 | 4096 | `SynthesisReport` |

**Dead config, measured with a positive control.** `_THINKING_CRITIC_CONFIG`
(:149-152) is self-documented as *"DEFINED BUT NEVER REFERENCED anywhere in the
tree"*, and `_THINKING_MODERATOR_CONFIG` (:160) is **equally dead**:
`grep -rn "_THINKING_MODERATOR_CONFIG" --include="*.py" .` returns **exactly one
line — its own definition**, against a positive control of **6** hits for
`_SYNTHESIS_STRUCTURED_CONFIG`.

> **CORRECTION (this replaces an earlier draft line in this brief, it does not
> accompany it).** An earlier pass of this section claimed
> `_THINKING_MODERATOR_CONFIG`'s missing `response_schema` was the likely cause
> of the Moderator's failures. **That is wrong and is withdrawn.** The config is
> dead, so it explains nothing. The live Moderator path is
> `debate.py:315-320` passing `_MODERATOR_STRUCTURED_CONFIG`
> (`debate.py:47-51`), which **does** carry `response_mime_type:
> "application/json"` + `response_schema: ModeratorConsensus`.

**The corrected finding is much stronger, and it is the empirical core of this
brief: the Moderator produced 359 invalid-JSON events WHILE a Gemini
`response_schema` was in force.** Gemini's schema did not make invalid JSON
unreachable on this codebase's own traffic. The trend, per rotated file, is
147 → 91 → 66 → 20 → 12 → 17 → **6** — a large decline consistent with the
phase-37.1 `include_thoughts`-vs-`response_schema` guard landing (documented in
`backend/tests/test_phase_37_4_moderator_schema.py:1-26`), but **it never
reaches zero**. Phase-37.4's immutable criterion #2,
`live_cycle_post_change_shows_zero_moderator_invalid_json_warnings`, is
therefore **still not met** as of the newest rotated log.

### Claude Code CLI rail — `backend/agents/claude_code_client.py`

The step said "~:280, where max_tokens is reportedly no-op'd". **Re-derived: the
parameter is declared at `:300`, and the no-op is at `:393`** — the ~:280 anchor
is wrong.

```python
# claude_code_client.py:393
_ = max_tokens  # accepted but no-op at the CLI layer; preserved in signature for API-compat
```

The in-file comment (:386-392) explains it: `--max-tokens` is the **SDK** option,
not a CLI flag; the CLI rejected it as `unknown option` on ~63% of calls, so it
was dropped and the CLI now uses **model-default ceilings**. Consequence: **the
Gemini-side `max_output_tokens` budgets above do not bind on the CC rail** — a
config written for one transport does not transfer, which is exactly sub-question
(b). Schema IS passed: `args.extend(["--json-schema", json.dumps(json_schema)])`
at `:379-380`, guarded by `_ensure_additional_properties_false(...)` (:713, :718)
— the same `additionalProperties:false` requirement Anthropic's own doc mandates
(source 3). The fall-through when a schema can't be derived is already LOUD
(:724-731: "…but LOUDLY rather than silently", sets `json_schema = None`) — a
usable in-repo precedent for the degradation design.

## INTERNAL: `GET /api/settings/` — enumerated, and the answer is NO

`backend/api/settings_api.py`. `GET /` (`:406-416`) returns
`_settings_to_full(settings)` (`:350-401`), whose type is `FullSettings`
(`:65-127`) — and FastAPI's `response_model=FullSettings` **filters the payload
to exactly that model's fields**. So the exposed key set is precisely the
`FullSettings` fields: models (3), debate depth (2), pillar weights (5),
`data_quality_min`, cost controls (3), three `*_key_configured` booleans, the
phase-23.1 signal-stack block (14), and the phase-23.1.9 paper block (17,
incl. `paper_use_claude_code_route` and `paper_cycle_max_seconds`).

**`paper_synthesis_integrity_enabled` is NOT among them. Neither are the
diversity flags.** Measured:

- `paper_synthesis_integrity_enabled` occurs in `settings_api.py` at **exactly
  one line, `:281`**, inside `_FIELD_TO_ENV`. Same for
  `paper_position_recommendation_fix_enabled` (:282),
  `paper_risk_judge_shape_fix_enabled` (:283), `claude_code_timeout_s` (:284),
  `claude_code_empty_retry_max` (:285). **None appears in `FullSettings` or in
  `SettingsUpdate`.**
- **Those five `_FIELD_TO_ENV` entries are UNREACHABLE.** The writer loop is
  `updates = body.model_dump(exclude_none=True)` (`:435`) → `for field_name,
  value in updates.items(): env_key = _FIELD_TO_ENV.get(field_name)` (`:454-455`).
  `updates` can only contain **fields declared on `SettingsUpdate`**; since these
  five are not declared there, `.get()` is never called with those keys. The
  mapping rows are dead code.
- The phase-61.2 comment above them (`:279-280`) claims they are
  *"operator-visible in the Settings UI rather than manual-.env-only -- the 61.1
  lesson"*. **The code does not have that property.** The ENV mapping shipped;
  the model fields did not. This is a documented-intent / actual-behaviour gap of
  exactly the class the project's own memory calls "a correction must REPLACE,
  not accompany".
- The **diversity flags** are `paper_soft_sector_diversity_enabled` and
  `paper_soft_sector_diversity_w` (`backend/config/settings.py:487-488`). They do
  **not** appear anywhere in `settings_api.py` — not in `FullSettings`, not in
  `SettingsUpdate`, not even in `_FIELD_TO_ENV`. They are **.env-only**, fully
  invisible to the API.

Caching note for any observability design: `GET /` is served from
`get_api_cache()` under key `settings:full` (`:409-415`), so a freshly written
`.env` value can be served **stale** until the TTL expires or a `PUT` runs
`invalidate("settings:*")` (`:469`). A read-only config endpoint that must not
lie needs to bypass or explicitly report this cache.

## Work log (append-only)

- Read `.claude/agents/researcher.md` in full (422 lines).
- Read `.claude/rules/research-gate.md` in full (338 lines).
- Brief created with born-inert envelope.
- Round 1: 3 searches (frontier / canonical / vendor-doc variants); sources 1-3
  read in full.
- Internal: census re-derived; 4 emit sites located; both transports inspected;
  settings route enumerated.
