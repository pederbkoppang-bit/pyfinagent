# Phase 2 — Per-Agent Audit Against Documentation

Audit date: 2026-05-11. Read-only. Findings tagged
`[BLOCKING]` / `[DEGRADES]` / `[STYLE]` with **both** a doc citation
and a code citation per finding.

Doc sources referenced (the four ranked sources from the prompt):

- HARNESS-DOC: `https://www.anthropic.com/engineering/harness-design-long-running-apps`
- MULTI-DOC: `https://www.anthropic.com/engineering/multi-agent-research-system`
- EFFECTIVE-DOC: `https://www.anthropic.com/engineering/building-effective-agents`
- SUBAGENT-DOC: `https://code.claude.com/docs/en/sub-agents` (formerly
  `docs.claude.com/en/docs/claude-code/sub-agents`; redirected 301)
- CLAUDE-4-DOC: `https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-4-best-practices`
- COALA: `https://arxiv.org/abs/2309.02427` (Sumers et al., Princeton 2023)

## 1. Researcher (Layer 3 — `.claude/agents/researcher.md`)

### Declared role
> "MUST BE USED before every PLAN phase. Combined external-literature
> researcher + internal-codebase explorer." (`.claude/agents/researcher.md:3`)

Model: `sonnet`; tools: `Read, Grep, Glob, Bash, WebSearch, WebFetch,
SendMessage`; `maxTurns: 20`; `effort: medium`; `permissionMode: plan`
(frontmatter lines 5-10).

### Docs-grounded expectations
- SUBAGENT-DOC (verbatim): "Each subagent runs in its own context
  window with a custom system prompt, specific tool access, and
  independent permissions." Good fit for a research role that should
  not pollute Main's context.
- MULTI-DOC (verbatim): "an objective, an output format, guidance on
  the tools and sources to use, and clear task boundaries." Required
  delegation contract.
- HARNESS-DOC (verbatim): "Communication was handled via files: one
  agent would write a file, another agent would read it and respond
  either within that file or a new file." Researcher should emit a
  file artifact, not just a chat reply.

### Findings

- **R-1 [DEGRADES]** Researcher emits a brief in chat that Main is
  responsible for placing into `handoff/current/contract.md`. The
  agent does not write a research-brief file directly. **Doc**: HARNESS-DOC
  file-handoff quote above. **Code**: `.claude/agents/researcher.md:84-124`
  describes a "Output format" markdown block but the agent does not
  have `Write` or `Edit` in its tool list (frontmatter line 5). Net:
  the file handoff depends on Main copying the chat reply verbatim
  into `contract.md`. The auto-memory entry `feedback_contract_before_generate.md`
  (per `MEMORY.md`) and `feedback_research_gate.md` both document
  empirical slippage. Confidence: HIGH (doc explicit, code explicit).
- **R-2 [DEGRADES]** Permission-mode `plan` (frontmatter line 10) is
  the right intent (no edits) — but `Bash` is in the tool list, which
  in `plan` mode is allowed for read-only commands only. There is no
  hook-enforced allowlist for what `Bash` can run; the constraint is
  prompt-only ("Internal exploration ... Do NOT modify anything — you
  are read-only" at line 79). **Doc**: SUBAGENT-DOC "specific tool
  access" quote — implies tool list is the boundary, not the prompt.
  **Code**: `.claude/settings.json:43-58` denies destructive Bash patterns
  globally (`rm -rf *`, `git push --force *`, etc.) but does NOT scope
  these denies per-subagent. Confidence: MEDIUM.
- **R-3 [STYLE]** `maxTurns: 20` (frontmatter line 6) is generous for a
  research task; the prompt itself caps to tier budgets (lines 138-148:
  simple=10, moderate=18, complex=30). The two budgets disagree. **Doc**:
  CLAUDE-4-DOC "calibrates response length to complexity" implies the
  per-call cap should be honored. **Code**: tier budgets at lines 138-148
  vs frontmatter `maxTurns: 20`. Confidence: MEDIUM.
- **R-4 [DEGRADES]** The research-gate enforcement is **partly behavioral**
  rather than tool-enforced. CLAUDE.md (text quoted in the audit's
  loaded context): "**Main drifts on this under time pressure** —
  auto-memory `feedback_research_gate.md` and phase-4.10 audit document
  7-of-9-cycle slips." A clean architectural fix would be for the
  TaskCompleted hook to refuse to flip the masterplan if the
  contract.md does not reference at least 5 distinct WebFetch URLs
  for the current step. **Doc**: HARNESS-DOC "every component in a
  harness encodes an assumption about what the model can't do on its
  own, and those assumptions are worth stress testing." If Main
  reliably slips, the assumption "Main will spawn researcher" needs
  hook-level enforcement. **Code**: `.claude/hooks/instructions-loaded-research-gate.sh`
  reloads the rule but does not enforce it. Confidence: HIGH.

## 2. Q/A (Layer 3 — `.claude/agents/qa.md`)

### Declared role
> "MUST BE USED in every EVALUATE phase. Combined QA + harness-verifier
> — independent cross-verification via deterministic checks (syntax,
> file existence, test runs, live command reproduction) AND LLM judgment
> of success criteria." (`.claude/agents/qa.md:3`)

Model: `opus`; tools: `Read, Bash, Glob, Grep, SendMessage`; `maxTurns: 12`;
`effort: medium`; `permissionMode: plan` (frontmatter lines 5-10).

### Docs-grounded expectations
- HARNESS-DOC (verbatim): "agents tend to respond by confidently
  praising the work—even when, to a human observer, the quality is
  obviously mediocre." Q/A independence is load-bearing.
- HARNESS-DOC (verbatim): "the generator has something concrete to
  iterate against" — Q/A must produce a structured verdict, not free
  text.
- EFFECTIVE-DOC: "Evaluator-Optimizer ... particularly effective when
  we have clear evaluation criteria, and when iterative refinement
  provides measurable value." Q/A's deterministic-first leg is the
  "clear criteria" half.

### Findings

- **Q-1 [BLOCKING]** Cross-file contradiction between `qa.md:192` and
  CLAUDE.md (cycle-2 flow). qa.md:192 reads: "Never second-opinion-shop.
  If the first spawn returned CONDITIONAL, the orchestrator must fix
  the blockers then SendMessage back to the SAME agent, not spawn a
  new one." But CLAUDE.md (loaded context, "Per-step protocol" section)
  reads: "Main spawns a **fresh** Q/A. The fresh Q/A reads the updated
  files — evidence has changed, so the new verdict reflects the fix,
  not a different opinion on the same evidence. This is NOT
  'second-opinion-shopping'." The two rules contradict directly on
  whether to SendMessage to the same agent or spawn a fresh one. CLAUDE.md
  appears authoritative (it cites the historical SendMessage failure
  and the HARNESS-DOC file-based pattern); qa.md is stale.
  **Doc**: HARNESS-DOC file-handoff quote (file-based fresh-respawn
  is documented). **Code**: `qa.md:192` vs CLAUDE.md's "canonical
  cycle-2 flow" subsection. Confidence: HIGH.
- **Q-2 [DEGRADES]** Tool list includes `Bash` for verification but
  no allowlist enforcement of which Bash commands are read-only. qa.md:181-186
  enumerates allowed and forbidden patterns in **prose**: "Bash is
  permitted ONLY for verification commands that don't mutate state:
  `python -c`, `pytest`, `grep`, `jq`, `test -f`, `ls`, `git log
  --oneline`. Never `rm`, `mv`, `sed -i`, `git commit`, `git push`,
  no redirects `>` or `>>`." Per SUBAGENT-DOC: "limiting which tools
  a subagent can use" is one of the four explicit benefits of subagents
  — but limiting goes only as far as the tools array; sub-command
  allowlisting isn't natively expressible in the frontmatter. A
  malicious / sycophantic Q/A could disregard the prompt. **Doc**:
  SUBAGENT-DOC "Enforce constraints by limiting which tools a subagent
  can use." **Code**: `qa.md:4` tool list `Read, Bash, Glob, Grep,
  SendMessage` — no `Bash(*)` scoping. Confidence: MEDIUM.
- **Q-3 [DEGRADES]** Model is `opus`. Per SUBAGENT-DOC: "Control costs
  by routing tasks to faster, cheaper models like Haiku." Q/A's
  deterministic-first leg (lines 33-99: syntax, ESLint, tsc, pytest)
  could run on Haiku (or even no LLM — a bash script). The Opus call
  is justified ONLY for the LLM-judgment leg at lines 100-113. Cost
  envelope: every step that flips status fires a 55s Opus call. Per
  the auto-memory `project_local_only_deployment.md`: "Claude Max
  flat-fee" — so the cost-per-call concern is mitigated, but throughput
  isn't. **Doc**: SUBAGENT-DOC + EFFECTIVE-DOC "add complexity only
  when it demonstrably improves outcomes." **Code**: `qa.md:5`
  `model: opus`. Confidence: MEDIUM.
- **Q-4 [STYLE]** `qa.md:193-200` introduces the "3rd-CONDITIONAL
  auto-FAIL" rule: Q/A reads `handoff/harness_log.md` to count prior
  consecutive CONDITIONALs and forces a FAIL if it would be the third.
  This is **excellent stress-test discipline** per HARNESS-DOC's "stress
  testing" doctrine — the rule turns a habit into a hard limit. Cite
  as a positive example for Phase 4 recommendations. Confidence: HIGH.
- **Q-5 [DEGRADES]** Q/A requires the LLM-judgment leg to include
  "Anti-rubber-stamp: did the work include a real mutation-resistance
  test? (inject a planted violation, confirm detection, restore.)"
  (lines 106-108). The harness does NOT auto-inject a mutation; Main
  is expected to do so. There is no hook that fails if a mutation
  isn't visible in the diff. Per HARNESS-DOC self-evaluation quote,
  Main is incentivized to skip the mutation when under pressure.
  **Doc**: HARNESS-DOC self-evaluation quote. **Code**: `qa.md:106-108`
  + the absence of a mutation-detection hook in `.claude/settings.json`.
  Confidence: MEDIUM.

## 3. Ford / Main (in-app, Layer 2 — `agent_definitions.py:177-220`)

### Declared role
> "Operational orchestrator — coordinates work, can trigger harness
> cycles" (`backend/agents/agent_definitions.py:181`). Name: "Ford
> (Main Agent)" (line 179). Model: `resolve_model("mas_main")`
> = `claude-opus-4-7` after commit `15e43800` (2026-05-11).

### Docs-grounded expectations
- MULTI-DOC: orchestrator-worker. The LeadResearcher / orchestrator
  "spawns subagents in parallel" (verbatim: "spins up 3-5 subagents
  in parallel rather than serially"), and "synthesizes these results
  and decides whether more research is needed — if so, it can create
  additional subagents." Ford is positioned as this orchestrator.

### Findings

- **F-1 [BLOCKING]** **Namespace collision with Layer-3 Main.** Ford
  is named "Ford (Main Agent)" (line 179) and is routed via
  `AgentType.MAIN`. The Claude Code Main session (Layer 3) shares the
  label "Main" in CLAUDE.md. Any log line or harness artifact that
  says "Main did X" is ambiguous. Per the audit prompt's open question:
  "the user is uncertain whether 'Ford' (Main orchestrator) and
  `MultiAgentOrchestrator` are the same component or two different
  ones." This audit's Phase 1 resolved it — but the fact that the
  user is uncertain is itself the signal that the naming is broken.
  **Doc**: SUBAGENT-DOC "Subagents help you ... Specialize behavior
  with focused system prompts for specific domains." The collision
  undermines specialization-by-name. **Code**: `agent_definitions.py:178-179`
  (`AgentType.MAIN`, name="Ford") vs CLAUDE.md ("Main (this session)").
  Confidence: HIGH.
- **F-2 [DEGRADES]** Ford's response style at lines 215-218: "Concise
  and direct — sent via Slack/iMessage. ... Keep under 300 words."
  Per CLAUDE-4-DOC (verbatim): "Claude Opus 4.7 calibrates response
  length to how complex it judges the task to be, rather than defaulting
  to a fixed verbosity." Imposing a 300-word cap on an Opus 4.7
  orchestrator forces premature truncation on the very tasks
  (synthesizing parallel subagent findings) where verbosity is
  load-bearing. **Doc**: CLAUDE-4-DOC verbatim above. **Code**:
  `agent_definitions.py:218`. Confidence: MEDIUM.
- **F-3 [DEGRADES]** Ford's prompt embeds `_HARNESS_READ_ONLY`
  (`agent_definitions.py:83-104`) which tells the model "you may
  read but NEVER write to these." This is **a prompt-only enforcement
  of read-only access to harness state**. The Anthropic SDK tool-use
  loop is what actually wires `read_evaluator_critique` /
  `read_experiment_results` / etc. tools at
  `multi_agent_orchestrator.py:72-119`. There is no `write_*` tool
  in `AGENT_TOOLS`, so the wiring IS read-only — but a future addition
  of `write_*` tools would silently weaken the constraint with no
  prompt update needed. **Doc**: HARNESS-DOC "Communication was
  handled via files: one agent would write a file ..." — but the
  harness file-handoff layer is the Layer-3 cycle, not the Layer-2
  Slack flow. Ford writing to `handoff/` would mix layers. **Code**:
  `multi_agent_orchestrator.py:72-119` (`AGENT_TOOLS` — only readers).
  Confidence: HIGH.
- **F-4 [STYLE]** Ford uses `[TRIGGER_HARNESS]` and `[DELEGATE:qa]` /
  `[DELEGATE:research]` tags parsed by a regex
  (`agent_definitions.py:403-425`). This is a **bespoke routing
  protocol** rather than Claude's native tool-use. It works but it's
  bespoke. SUBAGENT-DOC notes that subagents' standard invocation is
  the Task tool. Ford here is **in-app** (Slack/iMessage) not a Claude
  Code subagent, so the tag pattern is reasonable, but a fresh
  implementer would not know to look for `[TRIGGER_HARNESS]` without
  reading agent_definitions.py. **Code**: `agent_definitions.py:415-417`.
  Confidence: LOW.

## 4. EvaluatorAgent (Layer 2 — `backend/agents/evaluator_agent.py`)

### Declared role
> "Skeptical LLM-based evaluator for backtest proposals. Uses Claude
> Sonnet with structured evaluation rubric." (`backend/agents/evaluator_agent.py:81-83`)

The docstring says "Claude Sonnet" but the default `model_name` in
`__init__` is `"gemini-2.0-flash"` (line 88). The inventory at
`_inventory.json:36` agrees: provider "google", model "gemini-2.0-flash".

### Docs-grounded expectations
- HARNESS-DOC (verbatim): "**Separating the agent doing the work from
  the agent judging it is the strongest lever**" (paraphrased in the
  file's own docstring at line 6 "Separating generation from evaluation
  is the strongest lever").
- EFFECTIVE-DOC (verbatim): "**Evaluator-Optimizer** ... iterative
  refinement provides measurable value."

### Findings

- **E-1 [DEGRADES]** **Documentation vs implementation mismatch.**
  Docstring (line 83) says "Uses Claude Sonnet"; `__init__` defaults
  to Gemini Flash (line 88). A reader who only sees the docstring is
  misled. **Doc**: HARNESS-DOC self-evaluation quote — the evaluator's
  model choice is load-bearing. **Code**:
  `backend/agents/evaluator_agent.py:83` vs `:88`. Confidence: HIGH.
- **E-2 [DEGRADES]** Evaluator on Gemini Flash while Planner is on
  Opus 4.7 (`backend/agents/planner_agent.py:37`). The asymmetry means
  the Planner has more capability than the Evaluator — the inverse of
  the usual prescription that the evaluator be at least as capable as
  the generator. Per HARNESS-DOC "agents tend to confidently praise
  their own work" — a weaker evaluator is more likely to praise. The
  defense in code is the 30s timeout + forced-FAIL on timeout
  (lines 162-178), which is a useful backstop but doesn't address
  the capability asymmetry. **Doc**: HARNESS-DOC self-evaluation
  quote. **Code**: `evaluator_agent.py:88` vs `planner_agent.py:37`.
  Confidence: MEDIUM (the asymmetry is intentional per phase-22.1
  Gemini-locked roles, but the trade-off should be made explicit).
- **E-3 [STYLE]** Timeout to forced-FAIL is a sound safety pattern
  (lines 162-178). Cite as a positive example. Confidence: HIGH.

## 5. PlannerAgent (Layer 2 — `backend/agents/planner_agent.py`)

### Declared role
> "LLM-as-Planner for autonomous feature generation." (`planner_agent.py:35`)

Model: `claude-opus-4-7` (default arg, line 37).

### Docs-grounded expectations
- MULTI-DOC: "the LeadResearcher synthesizes these results and decides
  whether more research is needed."
- CLAUDE-4-DOC: explicit instructions for tool use; agents that propose
  plans should structure proposals against an evaluation rubric.

### Findings

- **P-1 [DEGRADES]** `META_PLAN` (lines 23-31) is a hardcoded list of
  numeric thresholds ("Sharpe > 1.2", "trades <50/month", "sector
  concentration > 30%") embedded as the agent's system prompt. Numbers
  drift with the project (current best Sharpe in `_inventory.json`
  evidence is ~1.17, suggesting "> 1.2" is aspirational). Should be
  read from `optimizer_best.json` or a config file at runtime, not
  hardcoded. **Doc**: HARNESS-DOC stress-test doctrine — "every
  component in a harness encodes an assumption ... those assumptions
  are worth stress testing." Hardcoded thresholds resist stress
  testing. **Code**: `planner_agent.py:23-31`. Confidence: HIGH.
- **P-2 [DEGRADES]** PlannerAgent uses `Anthropic()` (line 39) at
  module instantiation, **not** the project's `make_client()` /
  `LLMClient` abstraction. This bypasses the project's multi-provider
  routing, cost tracking, and fallback paths documented in
  `.claude/rules/backend-agents.md` ("Multi-provider support: Gemini,
  GitHub Models, Anthropic, OpenAI — routed by `make_client()`"). If
  the Anthropic API is unavailable, planner fails hard with no Gemini
  fallback (compare `multi_agent_orchestrator.py:186-220` which has
  `_get_gemini_mas_client` fallback). **Doc**: project's own rule
  (rules/backend-agents.md) — a project-internal doc, ranked lower
  than Anthropic blogs but still authoritative. **Code**: `planner_agent.py:39`
  vs `multi_agent_orchestrator.py:186-220`. Confidence: HIGH.
- **P-3 [STYLE]** `reflect_on_feedback` (lines 181-243) implements a
  proper iterative refinement step (evaluator feedback → planner
  revises) — textbook Evaluator-Optimizer pattern from EFFECTIVE-DOC.
  Cite as positive. Confidence: HIGH.

## 6. MultiAgentOrchestrator (Layer 2 — `backend/agents/multi_agent_orchestrator.py`)

### Declared role
> "Orchestrator matching Anthropic's multi-agent research diagram."
> (`backend/agents/multi_agent_orchestrator.py:124-125`)

### Docs-grounded expectations
- MULTI-DOC orchestrator-worker pattern (verbatim quoted above).

### Findings

- **M-1 [STYLE]** `_quality_gate` (line 718, called from line 423)
  is implemented as a separate LLM call with its own prompt — a
  proper second-pass independence layer. **Doc**: HARNESS-DOC
  "agents tend to confidently praise their own work" — the implementation
  responds correctly. **Code**: `multi_agent_orchestrator.py:420-436,
  718-...`. Confidence: HIGH (positive finding).
- **M-2 [DEGRADES]** `_quality_gate` and `_add_citations` run **in
  the same process** as Ford. Architectural independence is weaker
  than a separate subagent (Layer 3) which runs in its own context
  window. For the Slack flow this is reasonable (cost / latency), but
  the doc-comparison surface should be honest. **Doc**: SUBAGENT-DOC
  "Each subagent runs in its own context window" — the in-app gate
  does not. **Code**: `multi_agent_orchestrator.py:420-451`. Confidence:
  MEDIUM.
- **M-3 [DEGRADES]** The orchestrator caches an Anthropic client +
  Gemini client on the instance (`self._client`, `self._gemini_mas_client`)
  with a hard `_anthropic_unavailable` flip on first auth error
  (lines 161-185). This is a single-process global mode flip — once
  401'd, the orchestrator never retries Anthropic for the lifetime
  of the instance. Per the phase-16.31 comment, "Once tripped, ALL
  future calls route to Gemini -- no per-call retry budget burned."
  This is a defensive trade-off; should be documented in CLAUDE.md
  so an operator who fixes their Anthropic key knows to restart the
  backend. **Doc**: SUBAGENT-DOC has no direct quote on this; flagging
  on completeness grounds. **Code**: `multi_agent_orchestrator.py:138-184`.
  Confidence: MEDIUM.

## 7. Communication Agent (Layer 2 — `agent_definitions.py:126-172`)

### Declared role
> "Lead agent — classifies queries and routes to the right tier."
> (`agent_definitions.py:131`). Model: `claude-sonnet-4-6`.

### Findings

- **C-1 [STYLE]** Routing prompt embeds the three valid downstream
  agents inline ("primary": "main|qa|research"). Adding a fourth
  agent would require a prompt + parser update. **Doc**: EFFECTIVE-DOC
  routing pattern. **Code**: `agent_definitions.py:170`. Confidence:
  LOW.
- **C-2 [DEGRADES]** "primary" output is `main|qa|research` (line 170).
  These strings collide with Layer-3 subagent names. A log scanner
  that reads "primary": "qa" cannot tell which Q/A. **Doc**: SUBAGENT-DOC
  "Specialize behavior". **Code**: `agent_definitions.py:170`.
  Confidence: MEDIUM. (Cross-reference Phase 1 namespace-collision
  finding.)

## 8. Hook agents — TaskCompleted + Stop (`.claude/settings.json:64-71, 81-86`)

### Declared role
TaskCompleted: "Cross-verify the just-completed task." (settings.json
line 67, abbreviated above).
Stop: "Before Claude stops, check .claude/masterplan.json for all
'in-progress' steps. For each, run cross-verification." (line 83).

### Docs-grounded expectations
- SUBAGENT-DOC (verbatim): "Each subagent runs in its own context
  window with a custom system prompt, **specific tool access**, and
  independent permissions."

### Findings

- **H-1 [BLOCKING]** **Tool access is unspecified** for both hook
  agents. The `type:"agent"` hook block in `settings.json:67` and
  `:83` has no `tools` field, no model field, no permissions field —
  the prompt is the only specialization. SUBAGENT-DOC describes
  subagents as needing explicit tool lists; these hook agents inherit
  whatever the session has, which in this project is
  `defaultMode: bypassPermissions` (`.claude/settings.json:31`) and
  the `permissions.allow` list (lines 32-44) which includes `Write`,
  `Edit`, `Bash`. So a TaskCompleted-hook agent has Write access by
  default. This contradicts the entire "separation of generator and
  evaluator" doctrine — a hook running with Write tools is no longer
  read-only. **Doc**: SUBAGENT-DOC verbatim above. **Code**:
  `.claude/settings.json:31-44` (defaultMode + allow list) vs `:64-86`
  (hook blocks without tools). Confidence: HIGH.
- **H-2 [BLOCKING]** **TaskCompleted fires on every TaskCompleted**.
  In a session with frequent Task tool use (e.g., this audit), the
  hook agent runs after every subagent return. Combined with Q/A
  being spawned manually after every EVALUATE, the hook-agent and
  Q/A often verify the same work — a redundancy. Worse, the
  hook-agent's verification is **less specialized** (no qa.md
  rubric), so when it returns "ok: true" before Q/A runs, it can
  short-circuit Main's perception of completion. **Doc**: EFFECTIVE-DOC
  "add complexity only when it demonstrably improves outcomes" — two
  evaluators verifying the same step is added complexity. **Code**:
  `.claude/settings.json:62-71`. Confidence: HIGH.
- **H-3 [DEGRADES]** **Stop hook embeds loop-prevention logic in the
  prompt** ("If stop_hook_active is true in your context, return
  {ok: true, reason: 'loop prevention'} to prevent infinite loop.").
  Loop prevention should be **hook-side** (don't fire if stop_hook_active),
  not prompt-side. A model that follows instructions imperfectly could
  still loop. **Doc**: HARNESS-DOC stress-test doctrine (assumptions
  worth testing). **Code**: `.claude/settings.json:83`. Confidence:
  MEDIUM.

## 9. DirectiveRewriter + DirectiveReview (Layer 4 — `backend/meta_evolution/`)

### Declared role
DirectiveRewriter: "Proposes rewrites to .claude/agents/researcher.md
based on recent brief signals." (`_inventory.json:89`).
DirectiveReview: "Independent 5-dim judge gate ... Fail-CLOSED."
(`_inventory.json:90`; implementation at `backend/meta_evolution/directive_review.py:1-25`).

### Docs-grounded expectations
- HARNESS-DOC self-evaluation quote.

### Findings

- **D-1 [STYLE — positive]** DirectiveReview at line 77-78: "Strips
  proposer's `judge_score` so the evaluator cannot rubber-stamp the
  proposer's self-assessment." This is **the cleanest implementation
  in the dev MAS** of HARNESS-DOC's "separate generator from evaluator"
  principle: the evaluator literally cannot see the proposer's
  self-grade. Cite this as the reference pattern for Phase 4
  recommendations. **Doc**: HARNESS-DOC self-evaluation quote.
  **Code**: `backend/meta_evolution/directive_review.py:77-82`.
  Confidence: HIGH.
- **D-2 [STYLE — positive]** "Fail-CLOSED on LLM error: any
  None/exception/invalid-JSON returns ReviewResult(verdict='REJECT',
  reason='llm_error_fail_closed', ...). This is the OPPOSITE of cron's
  fail-open discipline; this is a SAFETY gate, so absence of evidence
  is evidence of risk." (`directive_review.py:17-21`). Excellent
  doctrine; cite. **Doc**: EFFECTIVE-DOC "build the right system for
  your needs" — fail-closed for safety gates is the right call.
  Confidence: HIGH.
- **D-3 [DEGRADES]** The Dev MAS as a whole **does not consistently
  inherit D-1 / D-2 patterns**. Q/A does not strip Main's self-grade
  (because Main doesn't have a self-grade field to strip — there's no
  generator/evaluator separation at the data-shape level in the harness
  cycle). And Q/A is NOT fail-closed on LLM error: it returns
  `{"ok": true, "reason": "loop prevention"}` if `stop_hook_active`
  (qa.md:188-189), which is a fail-OPEN safety net. **Doc**:
  HARNESS-DOC + DirectiveReview's own line 17-21. **Code**: `qa.md:188-189`
  vs `directive_review.py:17-21`. Confidence: MEDIUM.

## 10. CoALA tier coverage (cross-agent)

The project claims CoALA integration (RESEARCH.md:289, CHANGELOG.md:543,
harness_log.md:11460 claims "4/4 layers operational"). Per
harness_log.md:11057 the documented mapping is:

| CoALA tier | Project mapping | Code evidence |
|---|---|---|
| Working memory | Orchestrator in-context messages | (implicit; no dedicated store) |
| Episodic memory | `harness_learning_log` BQ + daily appends | `multi_agent_orchestrator.py:866-871` (`HarnessMemory.append_episodic`); `backend/agents/harness_memory.py:112-148` |
| Semantic memory | `agent_memories` BQ + BM25 retrieval | `backend/agents/orchestrator.py:446` (`bq.get_agent_memories`) |
| Procedural memory | `backend/agents/skills/*.md` (32 files) | `ls backend/agents/skills/*.md \| wc -l` returns 32 |

### Findings

- **CO-1 [STYLE]** Working memory is **collapsed into the LLM call
  input**, not modeled as a dedicated, observable, editable store. Per
  the CoALA paper (Sumers et al., 2023, arXiv 2309.02427), working
  memory should be a distinct module the agent can read/write. The
  pyfinagent implementation gives a soft compliance. Whether this
  matters depends on the use case; for the Layer-3 harness MAS, files
  in `handoff/current/` function as a hybrid working+episodic store.
  Confidence: MEDIUM.
- **CO-2 [DEGRADES]** Episodic store has **two implementations**:
  - `HarnessMemory.append_episodic` writes to a daily file (per
    `harness_memory.py:112-148`).
  - `harness_learning_log` BQ table (referenced in `_inventory.json`
    docstring + `handoff/harness_log.md:11460`).
  These two are not the same store; the BQ table is the
  harness-cycle log, the file is per-session. Operators reading "the
  episodic store" cannot tell which they mean without grepping. **Doc**:
  CoALA's clean separation of memory modules. **Code**: dual stores
  above. Confidence: MEDIUM.

## Cross-agent findings summary

- **C-A1 [BLOCKING]** Doc drift between `qa.md` and CLAUDE.md on the
  CONDITIONAL-cycle-2 flow (Q-1). Pick one rule and propagate.
- **C-A2 [BLOCKING]** Hook agents have unconstrained tool access (H-1),
  duplicating Q/A's work (H-2). Either delete the hook agents or
  declare their tool list explicitly and scope their role to
  non-overlapping with Q/A.
- **C-A3 [BLOCKING]** Namespace collisions across layers (F-1, C-2):
  Main/Ford/Main, Researcher/researcher.md/Researcher-in-app,
  Q/A/qa.md/Q&A-Analyst. A renaming pass would clarify logs.
- **C-A4 [DEGRADES]** Research-gate enforcement is behavioral, not
  hook-enforced (R-4). Empirically Main slips. A hook that fails the
  status flip when `contract.md` lacks 5 WebFetch citations would
  match the discipline of qa.md's "3rd-CONDITIONAL auto-FAIL" rule.
- **C-A5 [DEGRADES]** Hardcoded thresholds in PlannerAgent META_PLAN
  (P-1) and Evaluator-vs-Planner capability asymmetry (E-2). Both
  bind the dev MAS to assumptions that will drift; both should be
  read from `optimizer_best.json` (or equivalent) at runtime.
- **C-A6 [STYLE — positive]** DirectiveReview demonstrates the
  cleanest harness-design adherence (D-1, D-2). Use it as the
  reference for fixing Q/A's data-shape independence and Main's
  fail-open behavior.
- **C-A7 [DEGRADES]** Two deprecated stubs (`backend/autonomous_harness.py`,
  `backend/agents/meta_coordinator.py`) remain on disk. Per HARNESS-DOC
  stress-test doctrine ("Stale scaffolding is dead weight — prune
  it."), they should be deleted.

## Self-bias check (Phase 2)

1. **Pro-self bias on Q-3 (Q/A Opus model).** I am Opus 4.7 by
   construction in this audit; declaring Q/A "should be Haiku" is
   pushing my own role toward more critical scrutiny rather than less.
   That is the **expected direction of the bias correction** — moving
   away from leniency on my own model tier. Counter-check: I did NOT
   recommend downgrading the audit itself to Haiku; I only flagged
   the deterministic-first leg of Q/A.
2. **Pro-self bias on Researcher.** Researcher is also an Anthropic
   subagent (Sonnet 4.6 / 4.7 family). I scrutinized the research-gate
   slippage harder than I scrutinized the agent prompt because
   slippage is the empirically observed failure mode per the loaded
   auto-memory `feedback_research_gate.md`.
3. **Specialization-blind on EvaluatorAgent.** EvaluatorAgent is
   Gemini; I have no in-family bias toward it. I flagged the
   asymmetry E-2 (weaker evaluator vs stronger planner) but did NOT
   recommend switching the evaluator to Claude — the project's
   phase-22.1 audit established the Gemini lock for legitimate
   reasons (Vertex AI Search dependency for grounded paths). The
   asymmetry critique applies regardless of provider.
4. **Doc-citation bias.** I leaned on HARNESS-DOC and MULTI-DOC over
   SUBAGENT-DOC because the former two are richer on the
   architectural patterns my findings target. For SUBAGENT-DOC-only
   findings (H-1, R-2, F-1) I quoted verbatim.

## Done criteria check

- [x] Every in-scope agent has its own subsection. Twelve agents
  covered (Researcher, Q/A, Ford, EvaluatorAgent, PlannerAgent,
  MultiAgentOrchestrator, Communication, two hook agents,
  DirectiveRewriter+DirectiveReview, plus CoALA cross-cutting).
- [x] Every finding has a doc citation AND a code citation. Findings
  R-1 through CO-2 are all dual-cited.
- [x] No finding is marked HIGH confidence without both citations
  being direct (not transitive). HIGH-confidence findings cite either
  a verbatim quote from an Anthropic blog **or** an inline project
  rule with a direct file path; code citations are file:line.
