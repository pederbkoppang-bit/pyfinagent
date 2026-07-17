# Research Brief — phase-71.0 (Harness + MAS upgrade DESIGN pack, research gate)

- **Step:** 71.0 — consume the 2026-07-16 max-effort ultracode self-audit register
  (`handoff/current/harness_proposals.json`, 17 kept / 15 rejected) and prepare
  `handoff/current/design_harness_mas_71.md`.
- **Tier:** complex (7 sources read in full; multi-source doctrine re-validation +
  full internal re-validation against HEAD).
- **Researcher:** Layer-3 Researcher (this session).
- **Binding on the design pack:** offline / $0 / DESIGN-ONLY (no production code in
  71.0); every grounding must be a REAL current feature/doc; the 15 rejected
  proposals must be acknowledged and NOT re-introduced.
- **HEAD at research time:** `7d54d30d450041a82f459808274cd8ce07c324c6`.
- **Gate status: `gate_passed: true`** — 7 external sources read in full (floor 5),
  recency scan performed, 3-variant queries disclosed, ≥10 URLs collected, 9
  internal files inspected, all 6 register facts re-confirmed on HEAD.

---

## 1. Source table — read IN FULL (WebFetch)

| # | URL | Tier | What it grounds |
|---|-----|------|-----------------|
| S1 | https://code.claude.com/docs/en/workflows | Official docs (Anthropic) | Dynamic-workflow runtime: resumable-in-session, isolated-from-conversation, `agent(prompt,{schema})` structured-output primitive, agent caps, save to `.claude/workflows/`, model inheritance |
| S2 | https://platform.claude.com/docs/en/build-with-claude/structured-outputs | Official docs | Structured Outputs GA: constrained decoding, `output_config.format`={type:json_schema}, GA model list, `messages.parse()`, strict tools + limits, 24h grammar cache |
| S3 | https://www.anthropic.com/engineering/building-effective-agents | Official blog (Anthropic Eng, Dec 19 2024) | Evaluator-optimizer workflow; "clear evaluation criteria"; stopping conditions (max iterations); guardrail separation |
| S4 | https://www.anthropic.com/engineering/multi-agent-research-system | Official blog (Anthropic Eng) | Loop-until-dry ("exits the research loop"); LLM-judge rubric incl. "completeness"; resume-not-restart; artifact/lightweight-reference pattern; 3–5 parallel subagents / 90% / 15× tokens |
| S5 | https://www.anthropic.com/engineering/harness-design-long-running-apps | Official blog (Anthropic Eng) | Self-eval as FLAWED baseline; "separating the agent doing the work from the agent judging it"; context resets + "Opus 4.5 … drop context resets … entirely"; stress-test doctrine; file-based handoff |
| S6 | https://platform.claude.com/docs/en/build-with-claude/effort | Official docs | Opus 4.8 "Start with xhigh"; "Reserve max … relatively small quality gains … overthinking"; per-model defaults; low for classification/high-volume; "Test your use case" |
| S7 | https://code.claude.com/docs/en/model-config | Official docs | `CLAUDE_CODE_EFFORT_LEVEL` env precedence; settings `effortLevel` accepts low/med/high/xhigh only (max session-only); non-interactive `/effort` "Not applied" hold; fallback triggers + "capped at three after duplicate removal"; effort clamp; `CLAUDE_CODE_SUBAGENT_MODEL` override |

**7 read in full ≥ 5 floor. Source quality: all Tier-1/2 (peer/official-docs/official-blog).**

## 1b. Snippet-only set (evaluated, not read in full)

| URL | Why recorded |
|-----|--------------|
| https://claude.com/blog/introducing-dynamic-workflows-in-claude-code | Anthropic launch post (workflows announced 2026-06-02); corroborates S1 |
| https://www.infoq.com/news/2026/06/dynamic-workflows-claude-code/ | Independent confirmation of June-2026 GA + parallel-agent framing |
| https://code.claude.com/docs/en/sub-agents | Subagent primitive workflows orchestrate (tool allowlist, `model` frontmatter) |
| https://code.claude.com/docs/en/agent-sdk/structured-outputs | SDK structured-output ("validated JSON … at the end") — proposal-3 cite |
| https://code.claude.com/docs/en/agent-sdk/subagents | SDK subagents / AgentDefinition.model — proposal-3 cite |
| https://platform.claude.com/docs/en/build-with-claude/prompt-caching | 4096 Opus / 2048 Sonnet cache floor — grounds REJECT #12 |
| https://arxiv.org/html/2508.02994v1 | "Agent-as-a-Judge" 2025 — separate-doer-judge frontier corroboration |
| https://arxiv.org/pdf/2606.19544 | "Reliability without Validity: LLM-as-a-Judge" 2026 — judge-bias caution |
| https://arxiv.org/html/2501.10868v1 / openreview FKOaJqKoio | JSONSchemaBench — constrained-decoding benchmark (year-less canonical) |
| https://www.aidancooper.co.uk/constrained-decoding/ | Practitioner explainer on constrained decoding (canonical prior art) |
| arXiv:2601.20975 (in-repo cite, researcher.md:218) | self-consistency n=1→n=8 67%→86% |
| PMC11615553 (in-repo cite, researcher.md:230) | devil's-advocate 0%→76% |
| GitHub anthropics/claude-code#20625 | subagent-level structured_output — **Closed as not planned** (corrects proposal 3) |

**URLs collected (unique): 7 read-in-full + 13 snippet-only = 20 (≥10 floor).**

## 1c. Three-variant query discipline (research-gate.md)

- **Current-year frontier (2026):** `"Claude Code dynamic workflows structured output resumable subagents 2026"` → S1 + launch/InfoQ posts.
- **Last-2-year window (2025):** `"LLM evaluator-optimizer separate generator from judge agent reliability 2025"` → Agent-as-a-Judge (arXiv:2508.02994), Reliability-without-Validity (arXiv:2606.19544).
- **Year-less canonical:** `"constrained decoding guaranteed JSON schema structured outputs language models"` → JSONSchemaBench, constrained-decoding explainers (prior art predating provider GA).

---

## 2. Recency scan (last 2 years)

- **Dynamic workflows are NEW (2026-06-02 GA).** S1 is current and the load-bearing
  novelty for phase-71: the resumability/isolation properties that make the
  Workflow-Q/A path stall-immune did not exist when the harness protocol was
  written. This *supersedes* the Agent-tool-only Q/A launch documented in
  `qa.md` / `per-step-protocol.md §4`. Verified operational on this machine
  (Claude v2.1.211 ≥ the v2.1.154 workflow floor; the register itself was produced
  by an ultracode workflow).
- **Structured Outputs reached GA in 2026** on Opus 4.8 / Sonnet 4.6 / Haiku 4.5
  (+ Sonnet 5 / Mythos 5 / Fable 5) — S2. New vs. the hand-rolled `find('{')`/
  `rfind('}')` parsing across the Claude JSON paths; complements the older Gemini
  `response_schema` the repo already trusts.
- **Effort model list refreshed (2026):** S6 now enumerates Fable 5 / Mythos 5 /
  Sonnet 5 / Opus 4.8; the "Opus 4.7 guidance also applies to Opus 4.8 — start with
  xhigh" language is current. No change to the "reserve max" posture.
- **Judge-reliability literature (2025–2026)** trends toward *separating* generator
  from judge and toward *multi-agent juries / strict binary modes* (arXiv:2508.02994,
  2606.19544, DeepEval 2026). This complements — does not supersede — Anthropic's
  older "separate the doer from the judge" (S5). It also adds a caution the design
  should heed: ensembling judges is not free of correlated bias, reinforcing the
  register's decision to KEEP the adversarial red-team *leg* (71.3) but DROP the
  worst-of-N *ensemble* half of that proposal.
- **No new finding supersedes** "Building Effective Agents" (S3, Dec 2024) or the
  multi-agent / harness-design posts (S4/S5, 2025) as the canonical doctrine for
  evaluator-optimizer, loop-until-dry, and file-based handoffs; they remain the
  primary groundings.

---

## 3. External re-validation — every phase-71 grounding, verbatim

### 3.1 Workflow tool / resumability / isolation (S1) — grounds 71.1
- Page title: **"Orchestrate subagents at scale with dynamic workflows."**
- Resumability (the stall-immunity property): *"The runtime tracks each agent's
  result as the run progresses, which is what makes a run resumable within the same
  session."* Comparison table row: **Interruption → Workflows → "Resumable in the
  same session."**
- Isolation (why it survives the in-conversation end-flush stall): *"The workflow
  runtime executes the script in an isolated environment, separate from your
  conversation. Intermediate results stay in script variables instead of landing in
  Claude's context."*
- Structured-output primitive: example shows `agent('…', { schema: {…} })`;
  `agent()` "spawns one subagent," `pipeline()` runs one per item.
- Adversarial-review capability (grounds 71.3's red-team framing): *"it can have
  independent agents adversarially review each other's findings before they're
  reported."*
- Requirements/limits: **v2.1.154+**; **up to 16 concurrent agents**, **1,000 total
  per run**; save to **`.claude/workflows/`** (project) or `~/.claude/workflows/`.
- **Model-inheritance nuance (design-critical, grounds the `model:opus` pin in 71.1):**
  *"Every agent in a workflow uses your session's model unless the script routes a
  stage to a different one or the `CLAUDE_CODE_SUBAGENT_MODEL` environment variable
  is set, which overrides both."* Per S7 line 595, `CLAUDE_CODE_SUBAGENT_MODEL`
  *"overrides the per-invocation `model` parameter and the subagent definition's
  `model` frontmatter."* ⇒ The design MUST verify at implementation whether workflow
  `agent()` defaults to the **session** model or the subagent's **frontmatter**
  model, because the register (cycle-86 Fable-limit incident) reports it followed the
  `.md` pin, while S1's prose says "session's model." Either way: **pin the model
  explicitly on the Q/A workflow call so it lands on `opus` on the $0 Max rail** and
  never a metered/limited model.
- **Read-only corroboration (strengthens no-self-eval):** workflow subagents "always
  run in `acceptEdits` mode and inherit your tool allowlist." Because `qa.md` tools =
  `Read, Bash, Glob, Grep, SendMessage` (no Edit/Write — confirmed §5), a
  workflow-launched Q/A structurally cannot author `evaluator_critique.md`, so
  **Main transcribing the returned verdict is necessary, not a shortcut.**

### 3.2 Structured Outputs GA (S2) — grounds 71.2 (+ verdict schema in 71.1)
- Page title **"Structured outputs"**, subtitle *"Get validated JSON results from agent workflows."*
- *"Structured outputs guarantee schema-compliant responses through constrained decoding."*
- *"Always valid: No more `JSON.parse()` errors" / "Type safe" / "Reliable: No retries needed for schema violations."*
- **GA models include Opus 4.8, Sonnet 4.6, Haiku 4.5** (+ Sonnet 5, Mythos 5, Fable 5, Opus 4.7/4.6/4.5, Sonnet 4.5). Covers every Claude model on the Layer-2 paths.
- Shape: `output_config.format = {"type":"json_schema","schema":{…}}`. Python:
  Pydantic + `client.messages.parse()`. Tools: `strict:true`; **max 20 strict tools**,
  **max 24 optional params**. Grammar **cached 24h**; changing `output_config.format`
  **invalidates the prompt cache** (⇒ keep schemas static per call-site).
- **Vendor boundary the design must respect (register correction):** `output_config.format`
  is the **Anthropic** mechanism. `evaluator_agent.py::_call_model` and
  `skill_optimizer` run on the **google-genai** client → those must use Gemini
  `response_schema`/`response_mime_type`, NOT `output_config.format`. Only the two
  genuine Claude sites (quality gate + classifier) take `output_config.format`.

### 3.3 Evaluator-optimizer + separate-the-judge (S3 + S5) — grounds 71.2/71.3
- S3 (verbatim): *"In the evaluator-optimizer workflow, one LLM call generates a
  response while another provides evaluation and feedback in a loop."*
  *"This workflow is particularly effective when we have clear evaluation criteria,
  and when iterative refinement provides measurable value."*
- S3 stopping conditions (**correction — this is the AGENT-general wording, not
  evaluator-optimizer-specific**): *"The task often terminates upon completion, but
  it's also common to include stopping conditions (such as a maximum number of
  iterations) to maintain control."* ⇒ 71.4 must cite this general wording for its
  hard 3-pass ceiling, and NOT claim evaluator-optimizer prescribes open-ended
  completeness looping.
- S5 (the load-bearing separation lever, verbatim): *"Separating the agent doing the
  work from the agent judging it proves to be a strong lever."* Context makes
  generator self-eval the FLAWED baseline: *"agents tend to respond by confidently
  praising the work—even when … the quality is obviously mediocre"*; *"the evaluator
  is still an LLM that is inclined to be generous towards LLM-generated outputs."*
  (This is the origin of CLAUDE.md's own no-self-eval rule.)

### 3.4 Loop-until-dry + judge rubric + artifact pattern + resume (S4) — grounds 71.1/71.3/71.4/71.6
- Loop-until-dry: *"the LeadResearcher … decides whether more research is needed—if
  so, it can create additional subagents or refine its strategy. Once sufficient
  information is gathered, the system exits the research loop."*
- Judge rubric (grounds 71.3 contract-completeness): *"factual accuracy …, citation
  accuracy …, completeness (are all requested aspects covered?), source quality …,
  and tool efficiency."*
- Resume-not-restart (grounds 71.1): *"we can't just restart from the beginning …
  Instead, we built systems that can resume from where the agent was when the errors
  occurred."*
- Artifact / lightweight reference (grounds 71.6 lean-context): *"implement artifact
  systems where specialized agents can create outputs that persist independently …
  reduces token overhead from copying large outputs through conversation history …
  pass lightweight references back to the coordinator."*
- Parallel fan-out numbers (grounds REJECT #8): *"spins up 3–5 subagents in
  parallel"*, *"research time by up to 90%"*, *"15× more tokens."*
- **Note:** the exact *"Communication was handled via files: one agent would write a
  file, another agent would read it…"* sentence lives in **S5 (harness-design)**,
  NOT S4 — confirmed. CLAUDE.md attributes it correctly to harness-design.

### 3.5 Effort posture (S6) — grounds 71.5
- *"The guidance for Claude Opus 4.7 also applies to Claude Opus 4.8. Start with
  `xhigh` for coding and agentic use cases, use `high` for most other
  intelligence-sensitive workloads, and step down to `medium` or `low` only when
  you've measured that the lower level holds quality on your evals."*
- `max` row (verbatim): *"Reserve for genuinely frontier problems. On most workloads
  `max` adds significant cost for relatively small quality gains, and on some
  structured-output or less intelligence-sensitive tasks it can lead to
  overthinking."*
- `low` use case: *"simple classification tasks, quick lookups, or high-volume use
  cases"* (grounds mas_communication router → low). Best-practice #3: *"Test your use
  case: … Evaluate performance on your specific use cases before deploying."*
- API default effort is **`high`**; xhigh must be set explicitly.

### 3.6 Config/effort/fallback plumbing (S7) — grounds 71.5
- Env pin (grounds 71.5 Main pin): *"`max` … applies to the current session only,
  except when set through the `CLAUDE_CODE_EFFORT_LEVEL` environment variable."*
  Settings: *"set `effortLevel` to `low`, `medium`, `high`, or `xhigh` … `max` and
  `ultracode` are session-only and are not accepted here."*
- Precedence (**tempers the 71.5 Main-pin urgency**): *"The environment variable
  takes precedence over all other methods, then your configured level, then the model
  default."* ⇒ settings `effortLevel:"xhigh"` is a *configured level* that ALREADY
  outranks the Opus-4.8 default-`high`; the env pin is belt-and-suspenders + the
  ONLY path to `max`. Non-interactive `/effort` *"can't release the model-default
  hold … reports `Not applied`."*
- Fallback (grounds 71.5 chain): *"When the primary model is overloaded, unavailable,
  or returns another non-retryable server error … Authentication, billing,
  rate-limit, request-size, and transport errors never trigger a switch."*
  *"Chains are capped at three models after duplicate removal."* Doc's own example:
  `["claude-sonnet-5","claude-haiku-4-5"]`. Clamp: *"falls back to the highest
  supported level at or below the one you set."* ⇒ subagent-stall (a transport/hang
  class failure) is EXPLICITLY not a fallback trigger → this is orthogonal to the
  stall fix (reinforces REJECT #4).

---

## 4. Internal re-validation — do the register facts still hold on HEAD? (all YES)

`internal_files_inspected: 10`. HEAD `7d54d30d`. Every fact the phase-71 steps depend
on re-confirmed by grep/read (register, masterplan, the 6 target files, plus
`run_harness.py` for the 71.6 live-wiring check below):

| # | Register claim | HEAD verification | Verdict |
|---|----------------|-------------------|---------|
| 1 | Quality-gate response-clobber fallback at `multi_agent_orchestrator.py:883-885` | Line **883** `# If we can't parse, treat non-PASS as improvement`; line **885** `return gate_response, usage` (final `else`). On a parse miss the raw gate/rubric text is returned and the caller substitutes it for the analyst answer. | **TRUE** (fail-open clobber; 71.2 fixes → `return None` = fail-safe) |
| 2 | `evaluator_agent.py::_run_spot_checks` returns hardcoded 1.02 / 0.95 / 0.99 | Lines **513-515**: `"sharpe_2x_cost": 1.02`, `"sharpe_regime_shift": 0.95`, `"sharpe_param_sweep": 0.99`. `_run_spot_checks` def at 496; `evaluate_with_spot_checks` at ~482. | **TRUE** (fabricated-evidence stub; 71.2 deletes) |
| 3 | `skill_optimizer.py` one-shot propose + mechanical-only apply | `propose_skill_modification` at **300** (`temperature=0.7`, l.370), `apply_modification` at **399**, `passes_simplicity_criterion` at **570**. No independent LLM reviewer between them. | **TRUE** (self-judging gap; 71.2 adds directive_review-style gate) |
| 4 | `model_tiers.py::EFFORT_DEFAULTS` pins all mas_* to `max` | Lines **261-265**: `mas_communication/mas_main/mas_qa/mas_research` all `"max"`. Baseline comment (255-260) records pre-override low/xhigh/high/medium + "revert after closure" (never done). `MODEL_EFFORT_FALLBACK` at 273. Stale Sonnet-4.6 comment (l.251) on the now-Opus-4.8 `mas_qa` (l.74). | **TRUE** (stale override; 71.5 reverts — but see §6 dead-config caveat) |
| 5 | `.claude/workflows/` does NOT yet exist | `ls` → "No such file or directory"; `find .claude -type d -name workflows` → empty. | **TRUE** (71.1 creates the first workflow script; not redundant) |
| 6 | `qa.md` / `researcher.md` are `model: opus` | `qa.md` frontmatter: `model: opus`, `maxTurns: 30`, `tools: Read, Bash, Glob, Grep, SendMessage` (no Edit/Write). `researcher.md`: `model: opus`, `maxTurns: 40`. (Fable window reverted; effort `max` per phase-29.2.) | **TRUE** (live opus pins; 71.5 only prunes stale Fable *comments*, pins unchanged) |

**Conclusion: all six register facts hold on HEAD. The design may cite them as
current.**

**Secondary confirmations (design-load-bearing):**
- **Clobber caller substitutes the answer:** `multi_agent_orchestrator.py:461-462`
  `if checked_response: response = checked_response` — so the raw gate/rubric text
  returned at line 885 REPLACES the user-facing analyst answer. Confirms 71.2's
  fail-safe fix is real (not cosmetic).
- **`_default_spawn_researcher` is LIVE, not dead:** `run_harness.py:1044` (def),
  wired as the default at `:1122`, invoked at `:304` when `plan["research_needed"]`.
  Docstring (`:1049`): "The harness does not invoke Claude directly." ⇒ 71.6 must NOT
  delete it (register correctly rebutted proposal #12's misdiagnosis).
- **Stale Fable frontmatter present:** `qa.md:12-35` + `researcher.md:12-35` carry
  present-tense Fable-window comments with a passed `REVERT-BY 2026-07-12`; live pins
  are `effort: max` (`qa.md:45`, `researcher.md:43`). Confirms 71.5's #15 cleanup
  target; pins are correct and must be preserved.

---

## 5. Design outline — map each phase-71 step to grounding + binding constraints

The 17 KEPT proposals cluster into **six** design steps. Priorities carried from the
register's per-proposal verdicts (P1 = harness-blocking / live-bug; P2 = real
sharpening; P3 = hygiene). This grouping is the RECOMMENDED structure for
`design_harness_mas_71.md`; the design pack finalizes step boundaries.

### 71.1 — Workflow structured-output Q/A (and Researcher) launch path — **P1**
- **Rolls up kept proposals:** #1, #2, #3, #10 (persist verdict JSON).
- **Grounding:** S1 (resumable-in-session / isolated / `agent(schema)` / `.claude/workflows/`),
  S4 (resume-not-restart), S5 (separate doer/judge; file-based handoff), S2 (validated verdict JSON).
- **The problem it fixes:** Agent-tool Q/A end-flush stalled 6× on 2026-07-11
  (model-agnostic), and the no-self-eval ban means a stalled Q/A = a step that
  literally cannot close. The Workflow path captures the verdict as the agent's
  structured **return value** in the script journal (immune to the file-flush hang),
  with runtime retry/resume.
- **Binding constraints to preserve:** exactly-3-at-L3 (SAME `qa`/`researcher` roles,
  only the launch transport changes — no Explore/harness-verifier re-split); no
  self-eval (verdict originates from the independent qa agent; **Main transcribes it
  VERBATIM** into `evaluator_critique.md` — structurally enforced by qa.md's
  read-only toolset); $0 metered (first-party Max rail; **pin `model:opus`** on the
  workflow call — §3.1 model-inheritance nuance); 5-file protocol preserved (Main
  still writes all artifacts).
- **Design must:** keep the Agent-tool spawn as **FALLBACK** (follow the proposal
  BODY, not the "retire" title); ship **one** checked-in `VERDICT_SCHEMA`
  (mirroring qa.md's `{ok,verdict,violated_criteria,violation_details,certified_fallback,checks_run}`)
  + `ENVELOPE_SCHEMA` (researcher.md envelope) as the single source of truth; correct
  GitHub #20625 to **"closed as not planned"** and frame that closure as *supporting*
  the Workflow path; verify the exact `agent()` option key + model-resolution against
  the live SDK at implementation. Editing qa.md ⇒ **separation-of-duties + roster
  snapshot** (Peder review note; next-session `verify_qa_roster_live.sh`).
- **Do NOT** adopt REJECT #1 (`/goal`-to-PASS loop) that rides on this path, or
  REJECT #4 (model-swap on stall), or REJECT #11 (Monitor mtime watchdog).

### 71.2 — Layer-2 guaranteed structured outputs + clobber-bug fix + evaluator honesty — **P1**
- **Rolls up kept proposals:** #4 (clobber + structured outputs on quality gate &
  classifier), #9 (independent adversarial evaluator before `skill_optimizer.apply_modification`),
  #17 (delete fabricated `_run_spot_checks`/`evaluate_with_spot_checks` + WARN-on-mock + docstring).
- **Grounding:** S2 (`output_config.format` constrained decoding, GA on Sonnet 4.6 /
  Opus 4.8), S3 (evaluator-optimizer), S5 (separate doer/judge).
- **Binding constraints:** Layer-2 cost-sensitivity — near-zero token delta
  (constrained decoding, 24h-cached grammar, no retries), **no effort bump, no added
  per-ticker calls**; the clobber fix is **fail-SAFE** (`return None` preserves the
  original analyst answer — never emit rubric text). Vendor boundary (§3.2): the two
  Claude sites (quality gate + classifier) get `output_config.format`; the
  **Gemini** paths (`evaluator_agent`, `skill_optimizer` judge) use `response_schema`.
  `skill_optimizer` judge pinned to **Gemini** ($0-first) and **fail-CLOSED** on
  error; cheap deterministic checks (placeholder set unchanged, schema block
  untouched, edit bounded to `## Prompt Template`) BEFORE the LLM judge.
- **Do NOT** re-introduce REJECT #12 (prompt-caching — silent no-op below the 4096/2048
  floor), REJECT #13/#14 (Layer-4 / effort-max-drift cost analysis of dead config).

### 71.3 — Q/A judge-rubric hardening: contract-completeness + adversarial red-team leg — **P2**
- **Rolls up kept proposals:** #6 (contract-completeness dimension), #8 **part (b) only**
  (adversarial red-team leg in qa.md LLM-judgment).
- **Grounding:** S4 (judge rubric "completeness (are all requested aspects
  covered?)"), S5 (separate doer/judge lever), PMC11615553 devil's-advocate (in-repo).
- **Binding constraints:** edits `qa.md` ONLY (exactly-3, roster snapshot, Peder
  review); no cost/effort change (Q/A already Opus/max, rare-event); reuse the SAVeR
  violation taxonomy; scope completeness strictly to "promised deliverable present?"
  (leave over/under-claim to scope-honesty).
- **Explicitly DROP** proposal #8 **part (a)** worst-of-N ensemble: grounding is
  oversold (S3 "demystifying evals" does not recommend judge ensembling; the
  345,968-NAV bug was caught by a deterministic live capture, not judgment variance),
  and it invents a non-existent `money_path` masterplan tag. The 2025–2026 judge
  literature (recency scan) also warns ensembles carry correlated bias.

### 71.4 — Research-gate coverage ledger + audit-class loop-until-dry critic — **P2**
- **Rolls up kept proposals:** #7 (adaptive coverage gate), #11 (loop-until-dry
  completeness critic for audit-class steps).
- **Grounding:** S4 (loop exits "once sufficient information is gathered"), S3
  (stopping conditions / max iterations — cite the GENERAL wording, §3.3 correction).
- **Binding constraints:** edits `researcher.md` + `research-gate.md` only — and
  research-gate.md explicitly warns *"Do not duplicate rules across the three files"*
  → **cross-link, don't duplicate**; keep the **≥5-source floor as a minimum**; add an
  explicit **coverage field to the JSON envelope** so Q/A can gate programmatically;
  let a 1-sub-question ledger **collapse to today's flat behavior**; hard **3-pass
  ceiling + empty-delta stop** for the audit-class loop.
- **Design must** downgrade "exhaustion PROOF" → "corroborating EVIDENCE" (a
  self-critic shares first-pass blind spots) and scope the audit loop to
  **net-new / unknown-denominator** audits (the manual 1..N checksum already covers
  pre-enumerated audits like phase-69).
- **Do NOT** adopt REJECT #8 (parallel researcher fan-out default — 1M context does
  not exceed the window; the deep-tier fork already exists; triples the stall surface).

### 71.5 — Effort & model-config hygiene (Layer-2 revert + Layer-3 env pin + fallback + Fable frontmatter) — **P2/P3, OPERATOR SIGN-OFF**
- **Rolls up kept proposals:** #5 (revert stale `EFFORT_DEFAULTS` max → baseline),
  #13 (`CLAUDE_CODE_EFFORT_LEVEL` env pin for Main), #14 (fallback chain), #15 (prune
  stale Fable-window frontmatter comments).
- **Grounding:** S6 (Opus 4.8 xhigh-start / max-reserve / overthinking / low-for-router),
  S7 (env precedence; settings `max` not accepted; fallback dedup + trigger list; clamp).
- **Binding constraints:** effort is **operator-governed** → this whole step needs
  sign-off. The Layer-2 revert is a **cost REDUCTION, not a bump** (mas_communication
  router → low/medium; mas_qa → **high** per CLAUDE.md's documented per-ticker value;
  mas_research → medium; hold mas_main xhigh). Also update the phase-59.1 test that
  asserts `resolve_effort("mas_main")=="max"`. Main env pin = **xhigh ONLY** (strip
  `max` — S6 overthinking risk on orchestration + S7 shows settings-xhigh already
  outranks the default, so the risk is largely pre-mitigated). Fallback →
  `["claude-sonnet-5","claude-haiku-4-5"]` (the doc's own example; removes the
  redundant primary-equal first hop). Fable frontmatter → collapse ~74 stale comment
  lines to a one-line history pointer, **preserving `model:opus` + `effort:max` as
  PERMANENT per phase-29.2** (not Fable residue); roster snapshot + verify script.
- **CRITICAL design caveat (tension with REJECT #14):** the register REJECTED the
  "run a per-ticker token-delta cost analysis" framing of the effort-max drift because
  the mas_* `max` pins are **DEAD CONFIG at runtime** — the Layer-2 orchestrator's raw
  `anthropic.Anthropic()` calls (`multi_agent_orchestrator.py` 1008/1120/1232) never
  pass `output_config`/effort, so `resolve_effort("mas_*")` is never invoked. Therefore
  71.5 must implement #5 as **config hygiene** (delete the dead pins + document that
  the raw-SDK path uses the API default, OR wire `resolve_effort→output_config` — a
  binary decision), and must NOT frame it as removing live cost or run a
  measure-an-inert-variable cost study.
- **Do NOT** adopt REJECT #15 (effort-calibration eval — moot lever on the flat-fee
  L3 rail; N=1 invalid measurement).

### 71.6 — Lean-context subagent returns + stress-test housekeeping — **P3**
- **Rolls up kept proposals:** #16 (subagents return envelope + ≤200-word summary +
  file path, not the full brief/critique through Main's context), #12 **descoped**
  (DELETE the stale self-evaluating `scripts/mas_harness/cycle_prompt.md` + `run_cycle.sh`
  + `.bak`/disabled mas-harness plists; add ~5 grep asserts to the EXISTING SessionStart
  hook — NOT a new autonomous loop).
- **Grounding:** S4 (artifact systems / "pass lightweight references back to the
  coordinator"), S5 (stress-test doctrine + "stripping away pieces … no longer
  load-bearing").
- **Binding constraints:** prompt-only edits + inert-file deletion; keep `report_md`
  as the on-disk **path** (rename `brief_path`), ADD a ≤200-word summary carrying
  `gate_passed`/verdict + violated_criteria; write-first + 5 handoff files unchanged.
  Housekeeping stays **local-only** (no cloud `schedule`/GH-Actions surface — that
  would violate local-only) and adds **no new babysitting loop** (background-agent
  unauthorized-write risk, auto-memory `feedback_background_agent_resumption_risk`;
  reverted commit ad349f57). **Do NOT touch `run_harness.py::_default_spawn_researcher`**
  — the register flagged it as LIVE F2 research-on-demand design (wired at
  run_harness.py:1122), misdiagnosed as dead by the proposal.
- **Do NOT** adopt REJECT #9 / #10 (context-reset / compaction rituals — Opus 4.5
  already dropped resets; 1M window; drift-prone runbook rituals belong in a hook if
  anywhere).

---

## 6. The 15 REJECTED proposals — do NOT re-introduce (with the disqualifier)

| # | Rejected proposal | Why rejected (design must not reverse) |
|---|-------------------|----------------------------------------|
| R1 | `/goal` wrapping for cycle-2 auto-retry | Terminal condition "reach Q/A PASS" manufactures verdict-flip pressure on the load-bearing gate; no FAIL exit; unattended-push risk; Haiku evaluator can't read files. |
| R2 | Hooks with `effort.level` gating for verification | Target hooks are $0 deterministic bash/py (nothing expensive to gate); weakens the live_check verification backbone; per-phase effort field doesn't exist. |
| R3 | Worktrees `isolation: worktree` for Q/A | `git worktree add` checks out COMMITTED state; Q/A evidence is UNCOMMITTED working-tree → Q/A would read stale prior-cycle evidence. Silent correctness regression. Claimed `worktree.baseRef` config is absent. |
| R4 | Fallback model auto-recovery on subagent stall (Opus→Sonnet) | Stall is **model-AGNOSTIC** (Opus stalled as often as Fable) — a swap can't fix it; native fallback fires only on overload/server errors, NOT transport hangs (S7); mid-session qa.md swap violates roster-snapshot. |
| R5 | MCP `alwaysLoad` tuning for Q/A | **Zero delta** — backtest/signals already `alwaysLoad:false`; per-subagent `mcpConfig` mechanism doesn't exist (#51274 open); self-admitted "psychological" benefit. |
| R6 | Hooks `continueOnBlock` for live_check retry | Feeds rejection back to Main → lets the gated agent self-clear its own audit gate (self-eval-adjacent); premise false (gate is already fail-open exit-0); adds unattended stall. |
| R7 | Generator self-review punch-list before Q/A | **Grounding INVERTED** — S5 presents generator self-eval as the FLAWED baseline ("confidently praising the work"); re-imports the exact anti-pattern CLAUDE.md banned; redundant with qa.md checks. |
| R8 | Parallel researcher fan-out default for complex/deep | 1M-context Opus does not exceed the window (S4's own value criterion unmet); deep-tier fork already exists; triples the stall surface; depends on two other unlanded proposals. |
| R9 | Context-reset / checkpoint-resume discipline for Main | S5 says Opus 4.5 "drop[ped] context resets … entirely"; we run Opus 4.8/1M — grounding argues to STRIP, not add; already implemented; "add scaffolding + a prune-rule" is the cargo-cult tell. |
| R10 | Main-session compaction discipline in per-step-protocol.md | Self-defeating (a runbook read into chat is summarized away at the moment it's needed); already in-repo; drift-prone manual ritual → belongs in a PreCompact/SessionStart hook if anywhere. |
| R11 | Monitor-tool mtime stall watchdog | Directly contradicts per-step-protocol.md:266-267 ("Do not busy-wait or poll transcripts") three lines above its own citation; no kill-in-flight-subagent primitive exists; v2.1.198 background-spawn already absorbs it. |
| R12 | Prompt caching on Layer-2 MAS path | All 4 system prompts are BELOW the cache floor (4096 Opus / 2048 Sonnet) → `cache_control` silently no-ops; proposal misquotes the floor as 1024; project already solved this in phase-25.B9. |
| R13 | Layer-4 directive gates: constrained-decoding + resolve_model | Target is DORMANT (only tests call `rewrite_directive`/`review_directive_diff`; the weekly cron never touches it); `output_config` wouldn't fix the Gemini fallback; routing to mas_qa would RAISE cost. |
| R14 | Reconcile Layer-2 `effort=max` drift via cost analysis | The `max` pins are **DEAD CONFIG** (raw-SDK path never passes effort) → no live spend to remove; an "M-effort per-ticker cost analysis" would measure an inert knob. (The *hygiene* half survives as 71.5's config reconciliation — but NOT this framing.) |
| R15 | Effort-calibration eval (xhigh vs max) | Moot lever on the flat-fee L3 rail (no per-token cost); N=1 single-step A/B is within run-to-run variance; aims at L3 where the answer changes no spend; observed Q/A pain is stalling, not effort. |

**Cross-cut warning for the design:** several rejected items (R1, R4, R11) *ride on*
the KEPT Workflow-Q/A path (71.1) and several (R13, R14, R15) *ride on* the effort/
structured-output themes (71.2/71.5). Adopting the kept core must NOT drag in the
rejected riders. The two subtle traps: (a) 71.1 adopts the Workflow path but NOT
`/goal`-to-PASS looping or model-swap-on-stall; (b) 71.5 does the effort-config
*hygiene* (#5) but NOT the rejected *cost-analysis-of-dead-config* framing (#14).

---

## 7. Global binding constraints (apply to ALL six steps)

1. **Exactly 3 agents at Layer-3** — Main + Researcher + Q/A. No re-split of the
   merged Explore/harness-verifier. "N instances of the same role" (workflow launch,
   deep-tier fork) is allowed; a new ROLE is not.
2. **No self-evaluation by Main** — every verdict originates from the independent Q/A;
   Main may only transcribe it verbatim.
3. **$0 metered on the Layer-3 rail** — Claude Code first-party / Max flat-fee;
   workflows + subagents run on it; pin `model:opus` so nothing silently meters.
4. **Layer-2 cost-sensitivity** — per-ticker/metered roles stay cost-appropriate; no
   `max` on per-ticker roles without a cost analysis; structured-output adds ~0 tokens.
5. **Local-only** — single Mac; launchd not cloud; no fleet/CI/GH-Actions surface.
6. **Separation-of-duties + roster snapshot on agent-file edits** — qa.md/researcher.md
   changes take effect only at the next session start; flag Peder review; verify via
   `scripts/qa/verify_qa_roster_live.sh`.
7. **71.0 is DESIGN-ONLY / offline / $0** — no production code lands in this step; the
   design pack describes the changes 71.1–71.6 will make, each cited to a real,
   re-validated feature/doc above.

---

## 8. JSON gate envelope

```json
{
  "tier": "complex",
  "external_sources_read_in_full": 7,
  "snippet_only_sources": 13,
  "urls_collected": 20,
  "recency_scan_performed": true,
  "internal_files_inspected": 10,
  "gate_passed": true
}
```
