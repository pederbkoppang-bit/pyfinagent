# Research Brief -- Masterplan step 75.4

**Step:** Audit75 S4 -- skill-prompt delivery integrity: loader truncation, undelivered
sections, critic cap
**Tier:** moderate (NOT audit-class -- bounded, enumerated defect list)
**Researcher:** Layer-3 Harness MAS Researcher
**Date:** 2026-07-20
**Status:** COMPLETE

---

## 0. Scope

Two legs:
- **Leg 1** -- external literature (>=5 read in full, >=10 URLs, 3-variant query discipline)
- **Leg 2** -- internal code audit with verbatim file:line evidence; verify/refute each of
  (a)-(f). Standing project rule: *"Measure, don't assert"* -- every claim below was
  **executed**, not inferred.

---

## 1. Search queries run (three-variant discipline)

| # | Query | Variant |
|---|-------|---------|
| 1 | `Anthropic Claude API max_tokens output truncation stop_reason max_tokens structured outputs` | year-less canonical |
| 2 | `prompt regression testing canary string golden file test LLM prompt template silently truncated` | year-less canonical |
| 3 | `mutation testing detect vacuous assertions tests that cannot fail assertion-free test smell` | year-less canonical |
| 4 | `LLM pipeline fail loudly vs silently parse failure retry degraded flag 2026` | current-year frontier |
| 5 | `prompt template single source of truth prompt assembly testing 2025` | last-2-year window |

Recency evidence in the source table: current-year hits (arXiv 2601.22025, JavaCodeGeeks
2026-05, Medium 2026-07), last-2-year hits (arXiv 2509.13656, arXiv 2411.09846), and
year-less canonical hits (arXiv 2301.12284, Anthropic platform docs).

---

## 2. Sources read IN FULL (>=5 required -- gate-counting)

| # | URL | Accessed | Kind | Fetched how | Key finding |
|---|-----|----------|------|-------------|-------------|
| 1 | https://platform.claude.com/docs/en/build-with-claude/handling-stop-reasons | 2026-07-20 | Official doc (Anthropic) | WebFetch | **"Treat `max_tokens` truncation as a retriable error for structured outputs, not as a valid response to parse."** Also: "Do NOT treat truncated responses as success -- don't attempt to parse incomplete JSON... Retry with higher limits instead." Recommends `max_tokens` at **1.5-2x expected output size** for structured outputs. |
| 2 | https://platform.claude.com/docs/en/build-with-claude/structured-outputs | 2026-07-20 | Official doc (Anthropic) | WebFetch | Constrained decoding guarantees schema compliance **only when generation completes normally**. On `stop_reason:"max_tokens"` the response is **HTTP 200 but "may be incomplete and NOT match your schema"**; "Do not attempt to parse partial structured outputs"; "Retry with a higher `max_tokens`". Separately recommends **delta/patch schemas over full-document echo**: "Return deltas/patches rather than full documents", "Minimize schema complexity". |
| 3 | https://towardsdatascience.com/prompt-engineering-fails-quietly-prompt-regression-is-why/ | 2026-07-20 | Authoritative blog (TDS) | WebFetch | Silent-failure doctrine: "A prompt is a stochastic API... you are changing the API contract for every query type it handles, not just the ones you were thinking about." Prescribes **golden query sets with validation signatures**: required schema keys, **"pattern strings that must appear"**, and **"strings that must not appear"** guard checks. Explicitly: guard checks + pattern validation are what "catch when instructions don't reach the model." Directly validates the canary-string test design in criteria 1-2. |
| 4 | https://ar5iv.labs.arxiv.org/html/2301.12284 | 2026-07-20 | Peer-reviewed preprint (2023, canonical) | WebFetch (ar5iv, per research-gate PDF chain) | Weak assertions are "trivial to satisfy and would not trigger any error if the target class had incorrect behaviour" (example tautology: `assert(x>=y \|\| x<=y)`). Mutation-kill criterion, verbatim: *"valid assertions that are also coherent with every mutant's execution of target class C are weak because they represent properties that hold also for buggy versions of C"*; useful assertions "do not hold for at least one mutant of C". No explicit weak/useful ratio is reported -- the paper instead notes SpecFuzzer "reports thousands of constraints... and only a few are invalidated by the test suite." |
| 5 | https://arxiv.org/html/2601.22025v1 | 2026-07-20 | Peer-reviewed preprint (2026) | WebFetch | False Improvement Pattern measured: a generic helpfulness wrapper cut **extraction pass rate 100% -> 90%** and **RAG compliance 93.3% -> 80%**. Prescribes compact golden sets ("50-200 cases", ~20% edge cases) version-controlled with the prompts, and **deterministic decoding** for reproducibility. Explicitly proposes **no** promotion-blocking threshold. |

**PDF-chain note:** `https://arxiv.org/pdf/2301.12284` was attempted FIRST and returned
binary with no extractable text. Per `.claude/rules/research-gate.md` §"PDF and arXiv
paper fetching strategy", the ar5iv fallback was used (paper is 2023-01, pre-Dec-2023, so
`arxiv.org/html/` is not available for it). Recorded here rather than silently skipped.

---

## 3. Snippet-only sources (context; do NOT count toward gate)

| URL | Kind | Why not fetched in full |
|-----|------|------------------------|
| https://www.promptquorum.com/prompt-engineering/prompt-audit-and-regression-risk | Vendor blog | Vendor-tier; superseded by source 3 |
| https://scrolltest.com/llm-regression-testing-promptfoo/ | Vendor blog | Tooling-specific (promptfoo), not doctrine |
| https://www.traceloop.com/blog/automated-prompt-regression-testing-with-llm-as-a-judge-and-ci-cd | Vendor blog | LLM-as-judge CI, out of scope for offline test |
| https://medium.com/@alexrodriguesj/testing-llm-prompts-like-code-regression-evals-in-ci-cd-with-promptfoo-5242b4dcb9be | Community (2026-07) | Community tier; recency datapoint only |
| https://circleci.com/blog/what-is-mutation-testing/ | Vendor blog | Canonical mutation-testing primer, snippet sufficient |
| https://testrigor.com/blog/understanding-mutation-testing-a-comprehensive-guide/ | Vendor blog | Duplicate coverage |
| https://www.javacodegeeks.com/2026/05/mutation-testing-with-pit-in-java-the-coverage-metric-youre-ignoring-that-actually-measures-test-quality.html | Community (2026-05) | Recency datapoint: "high line coverage + low mutation coverage = assertion-free touch tests" |
| https://www.augmentcode.com/guides/mutation-testing-ai-generated-code | Vendor guide | AI-generated-test weak-assertion angle; snippet sufficient |
| https://arxiv.org/pdf/2509.13656 | Preprint (2025) | Adjacent (ML-notebook assertion generation) |
| https://arxiv.org/pdf/2411.09846 | Preprint (2024) | Adjacent (mutant propagation/crossfire) |
| https://arxiv.org/pdf/2509.23674 | Preprint (2025) | Hardware-assertion domain, not transferable |
| https://platform.claude.com/docs/en/build-with-claude/extended-thinking | Official doc | Thinking config is DEAD in this repo (see §5) |
| https://thomas-wiegold.com/blog/claude-api-structured-output/ | Community blog | Superseded by source 2 |
| https://a2a-mcp.org/blog/claude-response-incomplete | Community blog | Superseded by source 1 |
| https://pithycyborg.substack.com/p/the-token-budget-bug-that-makes-claude | Community blog | Anecdotal; corroborates source 1 |

**Total unique URLs collected: 20** (5 read in full + 15 snippet-only).

---

## 4. Recency scan (last 2 years, 2024-2026)

**Performed.** Result: **3 new findings that complement (do not supersede) the canonical
sources.**

1. **arXiv 2601.22025v1 (2026), "When Generic Prompt Improvements Hurt: Evaluation-Driven
   Iteration for LLM Applications"** -- read in full. Empirically documents the *False
   Improvement Pattern*: replacing task-specific constraints with a generic
   helpfulness-emphasising wrapper made **extraction pass rate fall 100% -> 90% and RAG
   compliance 93.3% -> 80%**. Advocates compact **golden sets ("50-200 cases", ~20% edge
   cases)** version-controlled alongside the prompts, and **deterministic decoding** (local
   inference, no API) so measurements carry no stochastic noise. It **does not** propose a
   promotion-blocking threshold -- it frames evaluation as a Define -> Test -> Diagnose ->
   Fix diagnostic loop. Direct read-across: the 75.4 test must be **offline and
   deterministic** (criterion 1 already demands "passes offline"), and per-file canaries
   beat any aggregate "all skills load OK" assertion.

   *Attribution note:* the sharper "aggregate 67.5% while a critical category collapsed
   100% -> 33.3%, block promotion on critical regression" figures belong to **source 3
   (Towards Data Science)**, NOT to this paper. An earlier draft of this brief conflated
   the two; corrected after reading the paper in full.
2. **Mutation-coverage-vs-line-coverage framing (JavaCodeGeeks 2026-05; Augment Code
   guide)** -- "A project with high line coverage and low mutation coverage has a large
   number of assertion-free *touch tests*." This is the exact failure mode this project
   has hit 5x (auto-memory `feedback_mutation_test_guards_and_fixtures`). Complements, does
   not supersede, arXiv 2301.12284.
3. **Anthropic structured-outputs GA doc (current)** -- the delta/patch schema
   recommendation is *new guidance relative to the 2023-era "just ask for JSON" advice* and
   is the decisive input for the (c) design choice (see §7).

No source found in the window that *contradicts* "do not parse truncated structured
output". Consensus is unanimous across Anthropic official docs and all secondary sources.

**arXiv 2301.12284 "Assertion Inferring Mutants" (canonical, year-less query)** --
read in full via ar5iv. Weak assertions are *"trivial to satisfy and would not trigger any
error if the target class had incorrect behaviour"*; the tautology example given is
`assert(x>=y || x<=y)`, *"a valid proposition that cannot be falsified, but... unlikely to
be useful."* The usefulness test, verbatim: *"valid assertions that are also coherent with
every mutant's execution of target class C are weak because they represent properties that
hold also for buggy versions of C"* -- useful assertions *"do not hold for at least one
mutant of C."* The paper reports **no explicit weak/useful proportion**; it characterises
severity qualitatively (SpecFuzzer *"reports thousands of constraints... and only a few are
invalidated by the test suite"*).

**Operational rule for 75.4: every assertion in the new test file must be paired with a
named mutation that kills it.** An assertion no mutant kills is, by this definition,
vacuous and must not be counted as a guard. Suggested mutation matrix for the contract
(each row must flip the test to RED):

| # | Mutation | Assertion it must kill |
|---|---|---|
| M1 | Revert `quant_model_agent.md:78` `### ` -> `## ` | criterion-1 `{{quant_model_data}}` presence |
| M2 | Revert `:81` `### Instructions` -> `## Instructions` | criterion-1 "one line of the former Instructions block" |
| M3 | Move ONE of the 8 Uncertainty-Permission blocks back below the terminator | criterion-2 per-file canary (proves the loop is per-file, not `any()`) |
| M4 | Move ONE of the 3 Code-Execution blocks back | criterion-2 code-exec canary |
| M5 | Restore stem `sector_analysis_agent` -> `sector_agent` at `orchestrator.py:1267` | criterion-3 stem assertion + startup stem-exists assertion |
| M6 | Drop the `format_skill` warning statement | criterion-3 `caplog` assertion |
| M7 | Set `_CRITIC_STRUCTURED_CONFIG["max_output_tokens"]` back to 2048 | criterion-4 budget assertion |
| M8 | Re-insert the literal `"treating as PASS with draft"` | criterion-4 string-absence assertion |
| M9 | Remove `max_output_tokens` from the **no-file-id** return of `_skill_gen_config` | criterion-5 (proves BOTH paths are covered, not just the file-id one) |
| M10 | Replace the real `load_skill` with a string stub returning the raw file | criterion-1 "via the REAL `load_skill()`" -- the anti-stub guard |

M9 and M10 are the two that this project's five prior vacuous-guard incidents suggest are
most likely to be quietly skipped; M3/M4 are what prevent an `any()`-shaped assertion from
passing with 7 of 8 files broken.

---

## 5. Internal audit evidence (all MEASURED, not asserted)

### 5.1 The real loader, executed

Command run:
```
.venv/bin/python -c "from backend.config.prompts import load_skill; t=load_skill('quant_model_agent'); print(len(t)); print(repr(t))"
```
Verbatim output:
```
LEN: 190
'{{fact_ledger_section}}\nYou are a Quantitative Factor Analysis Agent for {{ticker}}.\n\nYour task: Interpret the MDA-weighted quant model factor score and provide investment-relevant analysis.'
```
**190 chars, and `{{quant_model_data}}` is ABSENT.** The agent is asked to "interpret the
MDA-weighted quant model factor score" and is then handed **no score**.

### 5.2 Full delivered-vs-raw table (every file in `backend/agents/skills/`)

29 `.md` files exist. `load_skill()` was invoked on all 29 stems.

| file | raw chars | delivered | % | UncPerm | CodeExec | first H2s after `## Prompt Template` |
|---|---:|---:|---:|---|---|---|
| SKILL_TEMPLATE | 3402 | 262 | 8% | - | - | Experiment Log |
| alpha_decay_agent | 5166 | 856 | 17% | - | - | Experiment Log |
| alt_data_agent | 4347 | 740 | 17% | - | - | Experiment Log |
| anomaly_agent | 5001 | 827 | 17% | - | - | Experiment Log |
| **bias_detector** | 1409 | 411 | 29% | **RAW-only** | - | **Uncertainty Permission**, Empty-bracket |
| competitor_agent | 4202 | 438 | 10% | - | - | Experiment Log |
| **critic_agent** | 8184 | 2136 | 26% | **RAW-only** | - | Experiment Log, Uncertainty Permission, Empty-bracket |
| debate_stance | 4293 | 169 | 4% | - | - | Experiment Log |
| **deep_dive_agent** | 5433 | 595 | 11% | **RAW-only** | - | Experiment Log, Uncertainty Permission, Empty-bracket |
| earnings_tone_agent | 4730 | 797 | 17% | - | - | Experiment Log |
| **enhanced_macro_agent** | 5866 | 967 | 16% | - | **RAW-only** | Experiment Log, Code Execution Tasks |
| info_gap_agent | 5574 | 940 | 17% | - | - | Experiment Log |
| insider_agent | 4742 | 745 | 16% | - | - | Experiment Log |
| market_agent | 6177 | 1916 | 31% | - | - | Experiment Log |
| **moderator_agent** | 8121 | 1624 | 20% | **RAW-only** | - | Experiment Log, Uncertainty Permission, Empty-bracket |
| nlp_sentiment_agent | 4683 | 929 | 20% | - | - | Experiment Log |
| options_agent | 4648 | 718 | 15% | - | - | Experiment Log |
| patent_agent | 4709 | 746 | 16% | - | - | Experiment Log |
| **quant_model_agent** | 7532 | **190** | **3%** | **RAW-only** | **RAW-only** | **Quant Model Data**, **Instructions**, Experiment Log, Uncertainty Permission |
| **quant_strategy** | 19400 | **ValueError** | n/a | - | delivered raw (see 5.6) | *(no `## Prompt Template` section at all)* |
| rag_agent | 4474 | 558 | 12% | - | - | Experiment Log |
| **risk_judge** | 9137 | 1438 | 16% | **RAW-only** | - | Experiment Log, Uncertainty Permission, Empty-bracket |
| risk_stance | 4442 | 470 | 11% | - | - | Experiment Log |
| **scenario_agent** | 6715 | 1002 | 15% | **RAW-only** | **RAW-only** | Experiment Log, Uncertainty Permission, Empty-bracket, Code Execution Tasks |
| sector_analysis_agent | 4746 | 841 | 18% | - | - | Experiment Log |
| sector_catalyst_agent | 4699 | 1042 | 22% | - | - | Experiment Log |
| social_sentiment_agent | 4290 | 702 | 16% | - | - | Experiment Log |
| supply_chain_agent | 4156 | 723 | 17% | - | - | Experiment Log |
| **synthesis_agent** | 13173 | 5792 | 44% | **RAW-only** | - | Experiment Log, Uncertainty Permission, Empty-bracket |

"RAW-only" = the phrase exists in the file but is **absent from the delivered template**.

### 5.3 Truncation-class completeness (is the step's list COMPLETE?)

**Yes for the classes named, with two corrections.**

- **Unintended mid-template H2 (the (a) class): exactly ONE file** --
  `quant_model_agent.md`. Every other file's first H2 after `## Prompt Template` is the
  intended terminator (`## Experiment Log`, or the phase-4.14.26 sections in
  `bias_detector.md`). Verified by enumerating the H2 list after `## Prompt Template` for
  all 29 files (table above). **The step's implicit "1 file" scope for (a) is correct.**
- **`## Uncertainty Permission`: exactly 8 files**, and they are exactly the 8 the step
  names -- measured via `grep -ln "Uncertainty Permission" backend/agents/skills/*.md`:
  bias_detector, critic_agent, deep_dive_agent, moderator_agent, quant_model_agent,
  risk_judge, scenario_agent, synthesis_agent. **CONFIRMED, count exact.**
- **`## Code Execution Tasks`: 4 files, not 3** -- `grep -ln "Code Execution Tasks"`
  returns enhanced_macro_agent, quant_model_agent, scenario_agent **and
  `quant_strategy.md`**. The step names 3. The step is *correct for the load_skill-delivered
  set* (quant_strategy is not loaded via `load_skill`; see 5.6), so criterion 2's "3
  phase-26.3 files" is satisfiable as written -- but the 4th file must not be forgotten
  when someone later greps for the section.
- **NEW DEFECT (not in the step): `quant_strategy.md` has NO `## Prompt Template`
  section**, so `load_skill('quant_strategy')` raises
  `ValueError: No '## Prompt Template' section found`. Not a production bug (5.6), but it
  is a **live landmine for the new test**: any test that iterates `SKILLS_DIR.glob('*.md')`
  and calls `load_skill(p.stem)` will crash. Must be explicitly excluded or the exclusion
  itself asserted.

### 5.4 Exact line anchors in `quant_model_agent.md` (step claimed :78 and :81)

```
72:## Prompt Template
73:{{fact_ledger_section}}
74:You are a Quantitative Factor Analysis Agent for {{ticker}}.
76:Your task: Interpret the MDA-weighted quant model factor score and provide investment-relevant analysis.
78:## Quant Model Data
79:{{quant_model_data}}
81:## Instructions
82:1. Assess the overall factor signal (score direction and magnitude)
...
89:Respond with a concise analysis (200-300 words). Do NOT invent numbers.
91:## Experiment Log
96:## Uncertainty Permission (phase-4.14.26)
110:## Empty-bracket retraction format (phase-4.14.26)
119:## Code Execution Tasks (phase-26.3)
```
**Step's `:78` and `:81` are EXACT.**

### 5.5 Orchestrator anchors (all line numbers re-measured)

| Claim in step | Measured reality | Verdict |
|---|---|---|
| `_CRITIC_STRUCTURED_CONFIG` caps output at 2048 | `backend/agents/orchestrator.py:89-93`; `"max_output_tokens": 2048` at **:90** | CONFIRMED |
| critic must echo a "4096-token report" | **No "4096" or "token" string exists in `critic_agent.md`.** The real basis is `_SYNTHESIS_STRUCTURED_CONFIG` `max_output_tokens: 4096` at **orchestrator.py:85** + `critic_agent.md:108` *"Always include the corrected_report field with the full report JSON"* + `CriticVerdict.corrected_report: Optional[SynthesisReport]` at `backend/agents/schemas.py:66`. | **PARTIAL -- restate** |
| `orchestrator.py:1522-1524` parse-fail branch | `:1521 critic_result = _parse_json_with_fallback(...)`; `:1522 if not critic_result:`; **`:1523 logger.warning("Critic returned invalid JSON, treating as PASS with draft.")`**; `:1524 break` | CONFIRMED, exact |
| `_skill_gen_config` file-id and no-file-id paths | defined `orchestrator.py:908-933`. Returns `{"skill_file_id": fid}` (:933) or `None` (:929, :932). **Carries NO `max_output_tokens` on either path.** | CONFIRMED |
| `'sector_agent'` stem at `orchestrator.py:1267` | `orchestrator.py:1267` -- `generation_config=self._skill_gen_config("sector_agent")`. Only occurrence of `sector_agent` in the entire Python tree. Real file is `sector_analysis_agent.md`; the builder `prompts.py:457` correctly uses `load_skill("sector_analysis_agent")`. | CONFIRMED, exact |

**Extra finding -- dead twin config:** `_THINKING_CRITIC_CONFIG` (`orchestrator.py:97-103`)
also carries `max_output_tokens: 2048` and is **defined but never referenced anywhere**
(grep across the tree returns only the definition line). A fix that raises only
`_CRITIC_STRUCTURED_CONFIG` leaves a misleading 2048 twin behind. Recommend deleting it or
raising it in the same commit, and noting it in `experiment_results.md`.

**Extra finding -- 12 `_skill_gen_config` call sites**, not 11:
orchestrator.py:1216, 1223, 1230, 1237, 1244, 1260, **1267**, 1274, 1281, 1294, 1307, 1339.
Only :1267 uses a non-existent stem. The other 11 stems all resolve to real `.md` files
(verified against the 29-file listing).

### 5.6 `quant_strategy.md` is loaded RAW, not via `load_skill`

`backend/backtest/quant_optimizer.py:486` builds
`guide_path = Path(__file__).parent.parent / "agents" / "skills" / "quant_strategy.md"`
and reads the whole file. So its missing `## Prompt Template` is harmless in production and
its `## Code Execution Tasks` section **is** delivered. Confirms it is out of scope for the
(b) relocation -- **do not "fix" it**; moving its content would change what the optimizer
sees.

### 5.7 `bias_detector.md` is an ORPHAN prompt file

`grep -rn "bias_detector" backend/**/*.py` returns only
`orchestrator.py:38 from backend.agents.bias_detector import detect_biases` and
`api/agent_map.py:85`. **No `load_skill("bias_detector")` call exists anywhere** -- the
production bias detector is deterministic Python (`backend/agents/bias_detector.py`).
Consequence: relocating its Uncertainty Permission section has **zero production effect**,
but criterion 2 is still satisfiable because `load_skill('bias_detector')` works when
called directly by the test. Also note `bias_detector.md` has **no `## Experiment Log`** --
its H2 map is `6:## Prompt Template`, `18:## Uncertainty Permission`,
`28:## Empty-bracket retraction format`. So step (b)'s phrase *"sit AFTER '## Experiment
Log'"* is **inaccurate for bias_detector** (there is no Experiment Log to sit after); the
net effect -- undelivered -- is identical.

### 5.8 The 1024 enrichment cap: where it actually lives

- Documented in `.claude/rules/backend-agents.md`: *"Output token limits: Enrichment 1024,
  Debate 1536, Moderator 2048, Synthesis 4096"*.
- In code: `orchestrator.py:530` --
  `_enrichment_config = {"temperature": 0.0, "top_k": 1, "max_output_tokens": 1024}`,
  consumed only at `orchestrator.py:568` and `:589` (the **batch** enrichment path).
- The **per-agent** enrichment calls at :1216-:1339 pass `self._skill_gen_config(stem)`
  instead, which **never sets `max_output_tokens`**.
- `ClaudeClient` default: `backend/agents/llm_client.py:1348` --
  `max_tokens = config.get("max_output_tokens", 2048)`. **The step's "ClaudeClient default
  2048" claim is CONFIRMED, exact.** (The OpenAI path has the same 2048 default at
  `llm_client.py:1144`.)
- So on the Claude path an enrichment agent silently gets **2048**, double the documented
  cap; on the `None`-return path `_generate_with_retry` skips config entirely.

**Prior art worth reusing for (c):** `llm_client.py:1653-1684` already implements the
Anthropic-recommended pattern -- on `stop_reason == "max_tokens"` with a `tool_use` tail it
**retries once with `max_tokens = min(max_tokens*2, 8192)`** (`:1664`) and logs; on a text
tail it logs `"stop_reason=max_tokens on text; partial output"` (`:1684`). The critic
parse-fail branch should mirror this existing idiom rather than invent a new one.

### 5.9 Existing tests at risk

- `backend/tests/test_phase_75_skill_delivery.py` -- **does not exist** (confirmed via
  `ls`). Clean create.
- Files that reference skill headings / SKILLS_DIR and must be re-run after the heading
  edits: `backend/tests/test_phase_71_4_skill_review.py`,
  `tests/verify_phase_25_B9.py`, `tests/verify_phase_25_D9.py`,
  `tests/verify_phase_25_D9_1.py` (this one AST-parses `_skill_gen_config` and asserts
  `>= 11` call sites + asserts the `None` return for a missing stem -- **it will break if
  (d) changes the no-file-id path from `None` to a dict**; see risks).
- `backend/tests/test_phase_32_3_sector_exposure.py` asserts the Risk Judge prompt renders
  `fact_ledger_section` -- unaffected by heading demotion but re-run it.

---

## 6. Per-item verdicts

| Item | Verdict | Evidence / correction |
|---|---|---|
| **(a)** gap5-01 loader truncation, `quant_model_agent` -> ~190 chars, no `{{quant_model_data}}` | **CONFIRMED (exact)** | Executed loader: `LEN: 190`, placeholder absent. `prompts.py:194-198` regex `^## Prompt Template\s*\n(.*?)(?=^## |\Z)`. Offending headings at `quant_model_agent.md:78` and `:81` -- both line numbers exact. Fix = demote to `### `. |
| **(b)** gap5-02, 8 Uncertainty-Permission files + 3 Code-Execution files never delivered | **CONFIRMED with 2 corrections** | 8 UncPerm files exact (list matches). Code-Execution = **4 files contain it**, 3 in the load_skill set (`quant_strategy.md` is the 4th, loaded raw -- out of scope, do not touch). `bias_detector.md` has **no `## Experiment Log`**, so "sit AFTER `## Experiment Log`" is inaccurate for it. |
| **(c)** critic 4096-vs-2048 + silent "treating as PASS with draft" | **PARTIAL on the premise, CONFIRMED on the defect** | The "4096-token report" is **not stated in `critic_agent.md`** (no "4096"/"token" string). Real premise: synthesis cap 4096 (`orchestrator.py:85`) + `critic_agent.md:108` "Always include the corrected_report field with the full report JSON" + `CriticVerdict.corrected_report: Optional[SynthesisReport]` (`schemas.py:66`). Cap 2048 at `orchestrator.py:90` CONFIRMED. Silent branch at `:1522-1524` CONFIRMED verbatim. **Plus: dead `_THINKING_CRITIC_CONFIG` twin at `:97-103` also 2048.** |
| **(d)** `_skill_gen_config` must merge the 1024 cap; ClaudeClient default 2048 | **CONFIRMED** | `_skill_gen_config` (`:908-933`) returns `{"skill_file_id": ...}` or `None` -- no token key on either path. ClaudeClient default 2048 at `llm_client.py:1348`. Documented 1024 lives in `.claude/rules/backend-agents.md` + `orchestrator.py:530` (batch path only). |
| **(e)** `'sector_agent'` stem at `:1267` + startup stem assertion | **CONFIRMED (exact)** | `orchestrator.py:1267` is the only `sector_agent` occurrence; real file is `sector_analysis_agent.md`. Effect today is silent: `_skill_gen_config` returns `None` for the bogus stem, so the Sector agent **never gets the Files-API token saving** -- a silent perf/cost regression, not a crash. 12 call sites total (step implies 11 via the old `verify_phase_25_D9_1.py` count). |
| **(f)** `format_skill` should warn on unmatched kwarg | **CONFIRMED as a gap; note the inverse gap too** | `prompts.py:207-215` blindly `.replace()`s and returns; no warning. **Blast radius is small: `format_skill` has ZERO callers outside `prompts.py`** (only two doc-string mentions in `test_phase_32_3_sector_exposure.py`). Note the *inverse* failure (placeholder present but kwarg missing) is the one that actually bit this project before -- `prompts.py:976-982` documents the phase-32.3 Risk Judge `{{fact_ledger_section}}` regression. Consider warning on BOTH directions. |

---

## 7. Recommended approach for (c): **raise the cap, and additionally fix the branch**

**Recommendation: take the "raise to >= 6144" arm of criterion 4, NOT the patch-semantics
arm** -- and treat the parse-fail branch fix as non-optional regardless of arm.

External basis:

1. **Anthropic's own remedy for `stop_reason:"max_tokens"` is "retry with a higher
   `max_tokens`"** -- stated in both official docs read in full (sources 1 and 2). The
   structured-outputs doc is explicit that on truncation the output "may be incomplete and
   not match your schema" and "Retry with a higher `max_tokens` value to get the complete
   structured output."
2. **Sizing rule from source 1:** "Request `max_tokens` at least 1.5-2x your expected
   output size." Expected output = a full `SynthesisReport` budgeted at **4096**
   (`orchestrator.py:85`) plus the `verdict` + `issues[]` wrapper. 1.5x of 4096 = 6144.
   **Criterion 4's `>= 6144` is exactly the doc-sanctioned floor** -- this is a
   well-grounded number, not an arbitrary one. Worth stating in the contract.
3. **Why NOT patch semantics, despite Anthropic recommending delta/patch schemas
   (source 2).** The patch recommendation in the structured-outputs doc is motivated by
   *grammar-compilation cost and schema-complexity limits*, not by truncation. Adopting it
   here would mean changing `CriticVerdict.corrected_report` from
   `Optional[SynthesisReport]` (`schemas.py:66`) to a partial-field type, rewriting
   `critic_agent.md:66/71/103/108`, and adding a merge step at
   `orchestrator.py:1535-1538` and `:1544-1547` (two separate `corrected_report` accept
   paths). That is a **schema + prompt + two-call-site contract change** on the live
   synthesis path, versus a one-integer edit. Per source 3's False-Improvement warning, a
   prompt-contract rewrite of the critic is precisely the kind of change that regresses a
   critical category silently. **Patch semantics is the better long-term design and should
   be queued as its own masterplan step; it is the wrong change to bundle into 75.4.**
4. **The branch fix is mandatory either way.** Source 1 is unambiguous: *"Treat max_tokens
   truncation as a retriable error for structured outputs, not as a valid response to
   parse"* and *"Do NOT treat truncated responses as success."* The current
   `orchestrator.py:1522-1524` does the exact opposite -- an unparseable critic response is
   silently upgraded to **PASS**, so a truncation makes the quality gate *disappear* rather
   than fail. This is a fail-OPEN gate on the report-quality path. Mirror the in-repo idiom
   at `llm_client.py:1656-1684`: **retry once with a raised budget, then on second failure
   mark the report degraded** (e.g. set a `critic_degraded: True` field + `logger.warning`)
   and let the report proceed flagged rather than silently blessed.

Concrete recommendation for the contract:
- `_CRITIC_STRUCTURED_CONFIG["max_output_tokens"]: 2048 -> 6144` (`orchestrator.py:90`);
  same for the dead `_THINKING_CRITIC_CONFIG` (`:98`) or delete it.
- Replace `:1522-1524` with: retry-once (fresh `_generate_with_retry`), then on continued
  parse failure log a distinct warning and set an explicit degraded flag; the literal
  string `"treating as PASS with draft"` must disappear (criterion 4).
- Keep `CriticVerdict` unchanged in 75.4. Queue patch-semantics separately.

---

## 8. Consumer list (grep-verified -- change nothing without checking these)

**`load_skill` (34 call sites):**
- `backend/config/prompts.py` -- lines 274, 279, 286, 293, 307, 328, 350, 409, 415, 421,
  427, 433, 439, 451, 457, 478, 560, 628, 669, 733, 803, 874, 940, 975, 1001, 1015, 1021,
  1027, 1033, 1049 (30 builder call sites).
- `backend/agents/skill_optimizer.py:456` (re-load-after-write validation).
- `backend/tests/test_phase_71_4_skill_review.py:143` (monkeypatched stub).
- Definition: `backend/config/prompts.py:176`.

**`format_skill`:** **no callers outside `backend/config/prompts.py`.** Only doc-string
mentions at `backend/tests/test_phase_32_3_sector_exposure.py:115,150`. Adding a warning is
therefore low-risk -- but see risk R3 (log-noise on intentionally-unused kwargs).

**`reload_skills`:** `skill_optimizer.py:451, 460, 479`; `tests/verify_phase_25_D9.py:93-101`
(regex-asserts the signature); `test_phase_71_4_skill_review.py:142` (monkeypatched).

**`_skill_gen_config` (12 call sites):** `orchestrator.py:1216, 1223, 1230, 1237, 1244,
1260, 1267, 1274, 1281, 1294, 1307, 1339`; plus `tests/verify_phase_25_D9_1.py` asserts the
helper's shape by AST.

**`SKILLS_DIR`:** `prompts.py:23, 33, 98, 143, 166` and `skill_optimizer.py:21, 278, 404,
476, 523, 548`; `quant_optimizer.py:486` reaches the directory by its own path
construction (not the constant).

---

## 9. Risks

- **R1 (HIGH) -- SkillOptimizer rewrites these files at runtime.**
  `backend/agents/skill_optimizer.py:445` does
  `skill_path.write_text(new_content, encoding="utf-8")` after an LLM proposes an
  `old_text -> new_text` replacement. Its prompt constrains the LLM to *"You may ONLY
  modify text within the ## Prompt Template section"* (`skill_optimizer.py:334`) but that
  is a **prompt-level instruction, not an enforced invariant** -- `apply_modification`
  only checks that `old_text` occurs exactly once (`:409-421`). A phase-71.4 independent
  review gate exists at `:429-442` but is **flag-gated and DARK**
  (`skill_modification_review_enabled`). So a heading-level fix **can be silently undone or
  re-broken** by an autonomous optimizer run. Mitigation to consider in the contract: the
  new test is the regression net (it will fail loudly), and/or add an invariant check to
  `apply_modification` that `load_skill()` still returns the canaries after the write.
  This is a genuine "the fix can be reverted by a robot" risk, not theoretical.
- **R2 (HIGH -- HARD CONFLICT, measured) -- criterion 5 directly breaks THREE existing
  assertions in `tests/verify_phase_25_D9_1.py`.** Criterion 5 requires
  `_skill_gen_config` to carry `max_output_tokens=1024` on **both** paths, so it can no
  longer return `None` and no longer returns a bare one-key dict. The existing script
  asserts the exact opposite, verbatim:
  - `:71-75` claim 3 -- `out_empty is None` (empty `_skill_file_ids`, Gemini fallback)
  - `:81-85` claim 4 -- `out_mapped == {"skill_file_id": "file_xyz_123"}` (**exact dict
    equality** -- adding a second key breaks it)
  - `:91-95` claim 5 -- `out_missing is None` (unmapped stem)

  All three flip to FAIL the moment (d) is implemented. This is legitimate -- it is a
  deliberate contract change to a prior phase's verification script, not an immutable
  masterplan criterion -- but **the implementer must update `verify_phase_25_D9_1.py` in
  the same commit**, and `experiment_results.md` must disclose the change. Leaving it
  broken would silently red a previously-green verifier. Also note the now-stale docstring
  at `orchestrator.py:919-921` ("The `None` return is the safe fallback") must be rewritten.

- **R2b (MEDIUM -- NEW, not in the step) -- (d) is a live behavior change on the GEMINI
  path, not only the Claude path.** Traced end to end:
  - Claude path is safe: `llm_client.py:1409-1410` does
    `skill_file_id = config.get("skill_file_id")` / `if skill_file_id:`, so a dict without
    that key is simply skipped, and `llm_client.py:1348`
    (`config.get("max_output_tokens", 2048)`) then picks up 1024 as intended. **This is
    exactly the fix (d) wants.**
  - Gemini path changes behavior: `orchestrator.py:823` does
    `gen_kwargs = {"generation_config": final_config} if final_config else {}`. Today
    `_skill_gen_config` returns `None` on Gemini, so **no generation_config is passed at
    all** and the model uses its own default. After (d) it will receive
    `max_output_tokens=1024` for the first time, capping 12 enrichment agents that are
    currently uncapped on that path. That is arguably the documented intent ("Enrichment
    1024"), but it is a **real output-length reduction on a live path** and must be called
    out in `experiment_results.md` rather than shipped as a silent side effect. Cross-check
    against source 1's truncation guidance: if any Gemini enrichment response currently
    exceeds 1024 output tokens, it will now be truncated.
- **R3 (LOW) -- `format_skill` warning noise.** Several builders intentionally pass kwargs
  that may be empty strings for conditional sections (`prompts.py:211` doc-string:
  *"Unmatched placeholders are left as-is (for conditional sections set to empty
  string)"*). Warn on **kwarg-with-no-placeholder** (the (f) direction) rather than
  placeholder-with-no-kwarg, or the log will fill with benign warnings. Criterion 3 asks
  only for the (f) direction -- keep it there.
- **R4 (MEDIUM) -- `quant_strategy.md` will crash a naive all-files test loop.**
  `load_skill('quant_strategy')` raises `ValueError`. Any `for p in SKILLS_DIR.glob('*.md')`
  test must exclude it (and `SKILL_TEMPLATE.md`) *explicitly and assertively*, not by
  try/except swallow.
- **R5 (MEDIUM) -- vacuous-guard relapse.** Criteria 1-3 are canary-substring assertions.
  Per arXiv 2301.12284's criterion, each must be killed by a named mutation. The specific
  trap here: a test that asserts against a **string stub** of the file rather than the real
  loader would pass even with the loader broken -- criterion 1 already forbids this ("via
  the REAL `load_skill()` (not string stubs)"), and the mutation matrix must prove it.
- **R6 (LOW) -- delivered-content growth changes token cost.** Relocating the phase-4.14.26
  + phase-26.3 sections *increases* every affected prompt (e.g. synthesis_agent delivered
  5792 -> larger). That is the intent, but it interacts with the enrichment 1024 cap in (d)
  and with the Files-API path. Worth one line in `experiment_results.md` quantifying the
  new delivered lengths (the table in §5.2 is the before-baseline).
- **R7 (LOW) -- `bias_detector.md` fix is cosmetic.** No production caller (§5.7). Fine to
  do for criterion 2, but do not claim a behavioral win from it.

---

## 9b. Internal doctrine this step must not violate

`handoff/harness_log.md:27750` -- **"THE LESSON, FINALLY GENERALIZED"**, quoted verbatim:

> five findings across 75.3 + 75.2.1, one root cause that kept changing shape -- a guard
> that cannot fail. Source-string scans (x3), then a tautology (`is not None`), then a
> FIXTURE that could not represent the failure, then a guard that CLAIMED to pin the fixture
> while asserting a library fact. Mutation testing caught levels 1, 2 and 4. It missed level
> 3 because I mutated production while leaving the broken stub in place, and level 5 because
> I never mutated the TEST HARNESS itself. Durable rule: mutate the fixture too, not just
> the code under test -- and when you catch yourself defending a guard in a spawn prompt,
> that is the guard to mutate first.

This lands squarely on 75.4, which is a **canary-substring test step** -- the exact shape
that produced the "source-string scan" failures. Two concrete traps to avoid:

- **Trap A (level-1 relapse): a source-string scan masquerading as a delivery test.**
  Asserting `"Uncertainty Permission" in Path(...).read_text()` proves nothing -- the phrase
  is *already* in every one of those 8 files today and always was. Only
  `"Uncertainty Permission" in load_skill(stem)` distinguishes the fixed state from the
  broken state. This is why criterion 1 says "via the REAL `load_skill()` (not string
  stubs)" -- mutation **M10** is what proves it.
- **Trap B (level-3 relapse): mutating production while leaving the stub intact.** The
  `format_skill` caplog test (criterion 3) and the stem-exists assertion are both easy to
  satisfy with a fixture that cannot represent failure. Mutate the **fixture** as well as
  the code (per the durable rule above): M6 must be run with the real logger wiring, and M5
  must be run against the real `SKILLS_DIR` listing rather than a hard-coded stem list.

Corroborating external basis: arXiv 2301.12284's kill-criterion (§4) and the 2026-05
mutation-coverage framing -- "high line coverage + low mutation coverage = assertion-free
touch tests." The §4 mutation matrix (M1-M10) is the concrete discharge of this rule.

---

## 10. Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (5: 2 Anthropic official
      docs, 1 authoritative blog, 2 peer-reviewed preprints)
- [x] 10+ unique URLs total (20)
- [x] Recency scan (last 2 years) performed + reported (§4)
- [x] Full pages read, not abstracts (arXiv PDF-chain fallback documented)
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every relevant module (prompts.py, orchestrator.py,
      llm_client.py, skill_optimizer.py, schemas.py, quant_optimizer.py, all 29 skill files,
      existing test surface)
- [x] Contradictions / consensus noted (unanimous on truncation handling; patch-vs-raise
      tension surfaced and adjudicated in §7)
- [x] Claims cited per-claim

---

## 11. JSON envelope

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 5,
  "snippet_only_sources": 15,
  "urls_collected": 20,
  "recency_scan_performed": true,
  "internal_files_inspected": 44,
  "coverage": {
    "audit_class": false,
    "rounds": 1,
    "dry_rounds": 0,
    "K_required": 2,
    "new_findings_last_round": 0,
    "dry": false
  },
  "brief_path": "handoff/current/research_brief_75.4.md",
  "gate_passed": true
}
```
