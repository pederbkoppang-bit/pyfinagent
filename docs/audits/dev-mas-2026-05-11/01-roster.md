# Phase 1 — Roster Reconciliation

Audit date: 2026-05-11. Auditor: Claude Code (Opus 4.7 / 1M context),
read-only, scope = dev MAS only (Layer 2 in-app dev agents + Layer 3
harness MAS + Layer 4 dev-loop services). Layer 1 (28-skill Gemini
analysis pipeline) is **out of scope** per the audit charter; it is
referenced only as evidence for Phase 3 symptom traces.

## Method

1. Enumerated `.claude/agents/*.md` (Claude Code subagents).
2. Read `backend/agents/_inventory.json:1-94` — the canonical, version-3
   agent-topology inventory authored by phase-22.1 (last touched 2026-05-11
   commit `15e43800`). Cross-checked against code by grepping for every
   `id` string.
3. Inspected `.claude/settings.json` for hook-driven agents.
4. Inspected `backend/agents/multi_agent_orchestrator.py` and
   `backend/agents/agent_definitions.py` for in-app MAS members.
5. Inspected `backend/meta_evolution/*` for dev-loop participants.
6. Doc citations: `https://www.anthropic.com/engineering/harness-design-long-running-apps`
   (file-based handoffs, separation of generator and evaluator) and
   `https://www.anthropic.com/engineering/multi-agent-research-system`
   (orchestrator-worker pattern).

## Table 1 — Dev-MAS agents that actually exist in the repo

Last-touched commit SHAs from `git log -1 --format="%h %ad"` per file
(date format YYYY-MM-DD).

### Layer 3 — Harness MAS (Claude Code subagents)

| Name | Location | Model | Declared role (file:line) | Invocation surface | Last touched |
|---|---|---|---|---|---|
| **Main** | (this Claude Code session; **no `.md` file**) | varies (Opus 4.7 this session) | "Coordinates harness cycles. Spawns researcher and qa subagents. Writes contract/results/log files." (`backend/agents/_inventory.json:30`) | The interactive Claude Code session itself; spawns subagents via the Task tool | n/a (orchestrator session) |
| **Researcher** | `.claude/agents/researcher.md` (199 LOC) | `sonnet` (frontmatter line 5) | "MUST BE USED before every PLAN phase. Combined external-literature researcher + internal-codebase explorer." (`.claude/agents/researcher.md:3`) | Auto-delegated by Main via Task tool (description = routing hint) | `22e78958` 2026-04-21 |
| **Q/A** | `.claude/agents/qa.md` (201 LOC) | `opus` (frontmatter line 5) | "MUST BE USED in every EVALUATE phase. Combined QA + harness-verifier — independent cross-verification via deterministic checks ... AND LLM judgment of success criteria." (`.claude/agents/qa.md:3`) | Auto-delegated by Main via Task tool after every GENERATE | `39141ec3` 2026-05-07 |

### Layer 3 — Harness driver (script, not a subagent)

| Name | Location | Model | Role | Invocation | Last touched |
|---|---|---|---|---|---|
| **run_harness.py** (script) | `scripts/harness/run_harness.py:1` | n/a (orchestrator code) | "Three-Agent Harness Loop — Anthropic harness design pattern applied to quant research. Planner (heuristic) -> Generator (QuantStrategyOptimizer) -> Evaluator (independent backtests)" (`scripts/harness/run_harness.py:1-4`) | CLI: `python scripts/harness/run_harness.py [--cycles N --iterations-per-cycle N]` | `ef351405` 2026-04-16 |

### Layer 2 — In-app orchestrator agents (build/evaluate the application)

| Name | Location | Model | Declared role | Invocation | Last touched |
|---|---|---|---|---|---|
| **MultiAgentOrchestrator** (class) | `backend/agents/multi_agent_orchestrator.py:124` | host class; routes to children | "Orchestrator matching Anthropic's multi-agent research diagram. ... PLANNING STEP, ITERATIVE LOOP, QUALITY GATE (Ford never self-evaluates), MEMORY PERSIST" (`backend/agents/multi_agent_orchestrator.py:124-133`) | Constructed in `backend/slack_bot/app.py` and called from `autonomous_loop` | `15e43800` 2026-05-11 |
| **Communication Agent** | `backend/agents/agent_definitions.py:126-172` | `claude-sonnet-4-6` | "Lead agent — classifies queries and routes to the right tier" (line 131) | `_classify_via_llm()` in MAO | `22e78958` 2026-04-21 |
| **Ford / Main agent** | `backend/agents/agent_definitions.py:177-220` | `claude-opus-4-7` (per `15e43800`; previously 4.6) | "Operational orchestrator — coordinates work, can trigger harness cycles" (line 181); name="Ford (Main Agent)" (line 179) | Routed-to by MAO; can emit `[TRIGGER_HARNESS]` tag | `22e78958` 2026-04-21 (config); `15e43800` 2026-05-11 (model bump) |
| **Q&A / Analyst** | `backend/agents/agent_definitions.py:224-266` | `claude-opus-4-7` (per `15e43800`) | "Quantitative reasoning with read-only harness state access" (line 229) | Routed-to by MAO for "why" questions | `22e78958` 2026-04-21 |
| **Researcher (in-app)** | `backend/agents/agent_definitions.py:270-315` | `claude-sonnet-4-6` | "Deep research with literature access and RESEARCH.md integration" (line 275) | Routed-to by MAO | `22e78958` 2026-04-21 |
| **PlannerAgent** | `backend/agents/planner_agent.py:34` | `claude-opus-4-7` (default arg, line 37) | "LLM-as-Planner for autonomous feature generation" (line 35); META_PLAN at line 23-31 | Imported by `run_harness.py` planner step | `15e43800` 2026-05-11 |
| **EvaluatorAgent** | `backend/agents/evaluator_agent.py:80` | `gemini-2.0-flash` (default, line 88) — note: **NOT Claude** | "Skeptical LLM-based evaluator for backtest proposals. ... 'Separating generation from evaluation is the strongest lever.'" (lines 81-83 + line 6) | Imported by harness Plan/Generate/Evaluate cycle | `b23c8c1f` 2026-04-19 |
| **SynthesisAgent / ModeratorAgent** (skills) | `backend/agents/skills/synthesis_agent.md`, `moderator_agent.md` | `gemini-2.0-flash` / `claude-sonnet-4-6` | Layer-1 skill prompts loaded by `orchestrator.py`. Inventory tags them as Layer 2 because they aggregate Layer 1 outputs (`_inventory.json:39-40`). | Loaded by `load_skill()` in `orchestrator.py` | varies |

### Layer 4 — Meta-evolution and dev-loop services

| Name | Location | Model | Role | Invocation | Last touched |
|---|---|---|---|---|---|
| **SkillOptimizer** | `backend/agents/skill_optimizer.py` | `gemini-2.0-flash` | "Proposes edits to skill prompts via LLM reflection on outcomes" (`_inventory.json:81`); "Mirrors Karpathy's autoresearch pattern" per phase-audit-2.10 brief | Cron via `meta_evolution_cron`, also direct | `75f331fa` 2026-04-25 |
| **DirectiveRewriter** | `backend/meta_evolution/directive_rewriter.py` | `claude-sonnet-4-6` | "Proposes rewrites to .claude/agents/researcher.md based on recent brief signals" (`_inventory.json:89`) | Weekly cron Sun 02:00 ET | (not git-checked here) |
| **DirectiveReview** | `backend/meta_evolution/directive_review.py` | `claude-sonnet-4-6` | "Independent 5-dim judge gate (clarity/alignment/safety/proportionality/factuality) on directive proposals. Fail-CLOSED." (`_inventory.json:90`) | Gates DirectiveRewriter output | (see above) |
| **OutcomeTracker** | `backend/services/outcome_tracker.py` | `claude-sonnet-4-6` | "Evaluates past recommendations vs actuals ... LLM reflection -> agent_memories BQ" (`_inventory.json:77`) | Daily, called from `autonomous_loop` | (see above) |
| **MetaCoordinator** | `backend/agents/meta_coordinator.py` | (n/a) | "DEPRECATED — Phase 4 stub. Not part of the active MAS architecture." (file head, lines 1-2 per `handoff/archive/phase-audit-2.10-4.14.20-research-brief.md:49`) | (none — stub) | (not active) |
| **AutonomousHarness (stub)** | `backend/autonomous_harness.py` | (n/a) | "DEPRECATED — Phase 4 stub. Not part of the active MAS architecture. The active harness is run_harness.py" (`backend/autonomous_harness.py:1-7`) | (none — stub) | (not active) |

### Hook-driven agents (also dev MAS)

Two `.claude/settings.json` hook entries spawn agents inline. They have
no `.md` file, no name, no model frontmatter — the prompt is inlined.

| Hook event | Source | Inlined prompt (verbatim, abbreviated) | Timeout |
|---|---|---|---|
| **TaskCompleted** | `.claude/settings.json:64-71` | "Cross-verify the just-completed task. Read .claude/masterplan.json, find the matching step, run verification checks in order: deterministic first (syntax, tests, file existence), then existing evaluator critiques, then harness dry-run if within timeout. Return JSON {ok, reason, violated_criteria, checks_run}." | 60s |
| **Stop** | `.claude/settings.json:81-86` | "Before Claude stops, check .claude/masterplan.json for all 'in-progress' steps. For each, run cross-verification. ... If `stop_hook_active` is true ... return {ok: true, reason: 'loop prevention'} to prevent infinite loop." | 55s |

These are **unnamed agents** that overlap functionally with Q/A. They
are part of the dev MAS by construction (they evaluate the just-finished
masterplan step). See Phase 2 findings on this redundancy.

## Table 2 — User-supplied roster reconciliation

**Omitted — no roster supplied.** The placeholder in the audit prompt
was left as the literal string `<paste your 15-agent list here, one
per line; if blank, omit table 2>`. Per the audit prompt: "If the
user-supplied roster above is blank, write 'omitted — no roster
supplied' and skip this table." Auditor proceeded from repo discovery
alone. Table 1 above is the complete repo-discovered roster.

## Ford ≟ MultiAgentOrchestrator

**Verdict: NOT THE SAME.** Ford is a **role** (an `AgentConfig` of
`AgentType.MAIN`); `MultiAgentOrchestrator` is the **class** that
orchestrates Ford alongside Communication, Q&A/Analyst, and Researcher.

Code citations (≥2 as required):

1. `backend/agents/agent_definitions.py:177-179` — `AgentType.MAIN:
   AgentConfig(agent_type=AgentType.MAIN, name="Ford (Main Agent)",
   model=resolve_model("mas_main"), ...)`. Ford is one of four
   `AGENT_CONFIGS` entries; the others are `COMMUNICATION`, `QA`, and
   `RESEARCH`.
2. `backend/agents/multi_agent_orchestrator.py:124-133` — `class
   MultiAgentOrchestrator: """Orchestrator matching Anthropic's
   multi-agent research diagram. ... QUALITY GATE: Separate agent
   evaluates (Ford never self-evaluates) ..."`. The orchestrator's
   `_think_plan()` (line 478-503) calls `AGENT_CONFIGS[AgentType.MAIN]`
   to get Ford's prompt; Ford is **consumed by**, not **identical to**,
   the orchestrator.
3. Corroborating: `backend/agents/_inventory.json:34` declares
   `multi_agent_orchestrator` with children `["planner_agent",
   "evaluator_agent", "communication_agent", "analyst_agent"]`.
   Critically — and noted in Phase 2 — the inventory does NOT list
   "main" or "ford" among MAO's children even though Ford is the
   default `agent_type` route, indicating a documentation drift
   between `_inventory.json` and `agent_definitions.py`.

## The "3 agents" claim in CLAUDE.md

CLAUDE.md states: "**The MAS is exactly 3 agents: Main (this session)
\+ Researcher + Q/A.**" (CLAUDE.md, "Critical Rules" section, the long
🔴 bullet on the harness loop).

**Verdict: CONFIRMED for the Harness MAS layer specifically; MISLEADING
for the dev MAS as a whole.** The 3-agent statement is true if you
restrict "MAS" to the harness-cycle MAS (Layer 3 in `_inventory.json`'s
layering). But the dev MAS — defined by the audit prompt to include
the in-app orchestrator agents that build/evaluate the application — is
materially larger. Counts:

- **Layer 3 harness MAS: 3 active agents** (Main, Researcher, Q/A).
  Match.
- **Layer 2 in-app agents: 4 named in `agent_definitions.py` + 3
  helper agents in `multi_agent_orchestrator.py`** (Quality Gate,
  CitationAgent inline; planner_agent and evaluator_agent imported)
  = 6-7 functional agents.
- **Layer 4 dev-loop services: 4 LLM-callers** (SkillOptimizer,
  DirectiveRewriter, DirectiveReview, OutcomeTracker).
- **Hook-driven agents: 2 unnamed** (TaskCompleted, Stop).

Sum across the dev-MAS scope of this audit: ~15-16 agents. This matches
the user's working count of "15 agents" referenced in the prompt
context. The CLAUDE.md "exactly 3" line is therefore **technically true
for the cited scope but read out of context it is misleading** — most
operators reading CLAUDE.md would assume the 3-agent line constrains
the whole dev MAS, and it does not.

The "exactly 3" framing also functions normatively: it pushes a "Don't
re-split" rule in CLAUDE.md ("**The MAS is exactly 3 agents** ...
Researcher absorbs the old `Explore` ... Q/A absorbs the old
`harness-verifier`. No separate Explore. No separate harness-verifier.
Don't re-split."). The rule is about a specific consolidation history,
not a count cap on the dev MAS as a whole. The text would be more
accurate as: "**The Harness MAS layer is exactly 3 agents.**"

## Shadow MAS subsection

Agents invoked by code or wired in settings but NOT documented in
CLAUDE.md and NOT in `.claude/agents/*.md`, or vice-versa.

1. **Unnamed hook agent: TaskCompleted** —
   `.claude/settings.json:64-71` registers a `type:"agent"` hook with
   an inlined prompt that performs Q/A-style cross-verification. It is
   not in `.claude/agents/`, not in `_inventory.json`, and not in
   CLAUDE.md. It runs on every TaskCompleted event with a 60s budget.
   **Shadow.**

2. **Unnamed hook agent: Stop** — `.claude/settings.json:81-86`
   registers another inlined-prompt agent that checks in-progress
   masterplan steps before the session stops. 55s budget. Not in any
   roster file. **Shadow.**

3. **In-app "Quality Gate" + "CitationAgent"** —
   `multi_agent_orchestrator.py:420-451` runs a `_quality_gate()` and
   an `_add_citations()` step. Both consume LLM calls, both have
   embedded prompts, neither is in `_inventory.json` nor
   `.claude/agents/`. **Shadow (semi: visible in source).**

4. **DirectiveRewriter + DirectiveReview** —
   `backend/meta_evolution/directive_rewriter.py` proposes edits to
   `.claude/agents/researcher.md`. This is a **self-modifying-dev-MAS**
   pattern: the dev MAS rewrites its own subagent prompts. Listed in
   `_inventory.json:89-90` and `_inventory.json` flags
   `directive_review` as "Fail-CLOSED" — but neither is mentioned in
   CLAUDE.md's "harness protocol" or "agent definition changes
   require session restart" rules. Yet they materially affect the
   live researcher prompt. **Shadow vs CLAUDE.md.**

5. **`autonomous_harness.py` + `meta_coordinator.py`** — both files
   exist on disk under `backend/`, but their first 14 lines explicitly
   declare them deprecated stubs ("Not part of the active MAS
   architecture"). They are NOT shadows; they are documented dead code
   that a fresh reader of `backend/agents/*.py` could mistake for
   active components. Flagged here for symmetry: the inverse of a
   shadow is a "ghost" — present in code, absent from the active
   roster.

6. **In-app Layer-2 names that collide with Layer-3 names** — the
   in-app `AgentType.RESEARCH` / `AgentType.QA` /`AgentType.MAIN`
   share the labels "Researcher", "Q&A", and "Main" with the
   distinct Claude Code subagents `.claude/agents/researcher.md` /
   `.claude/agents/qa.md` / (Main session). This is not a shadow but
   a **namespace collision** with real operational risk: a reader of
   the harness log can't tell from "Researcher said X" which layer
   the agent belongs to. (See Phase 2 finding S-NS-1.)

## Cross-cutting observation

The roster is **multi-layered, but the "exactly 3 agents" rule in
CLAUDE.md flattens the picture in a way that obscures the in-app and
meta-evolution layers**. Subsequent phases treat the dev MAS as the
union of Layer 2 + Layer 3 + Layer 4 (dev-loop only), with Layer 1
deliberately out of scope.

## Self-bias check (Phase 1)

I (Claude Code Opus 4.7) am the **Main** of the Layer-3 harness MAS by
construction in this audit session. Risks of leniency:

1. **Pro-self bias on "exactly 3 agents".** I am one of the three; my
   first instinct was to accept the count as correct. Counter-check: I
   audited against `_inventory.json:30-32` (which lists Main+Researcher+Q/A
   under `layer:3`) AND against the audit prompt scope (which is broader
   than Layer 3). The reframing — "true for Layer 3 specifically,
   misleading for the dev MAS as a whole" — is the harsher reading the
   anti-leniency protocol requires.
2. **Pro-script bias toward `_inventory.json`.** The inventory was
   authored by a prior Researcher run in phase-22.1; I rely on it
   heavily. I cross-checked every cited node by grepping for the `id`
   string in code and confirming the file path exists. Where the
   inventory disagrees with `agent_definitions.py` (no "main" / "ford"
   listed as a child of `multi_agent_orchestrator` despite being
   routed via `AGENT_CONFIGS[AgentType.MAIN]`), I sided with the code,
   not the inventory.
3. **Subagent-blind spot.** I default to thinking of subagents as the
   `.claude/agents/*.md` files. The hook-driven agents in
   `settings.json` are easy to miss because they have no
   self-describing prompt file. I explicitly grepped
   `.claude/settings.json` for `type":"agent"` to catch them. Two were
   found; both surfaced in Table 1.

## Done criteria check

- [x] Every agent in the user-supplied roster is either located or
  marked absent with evidence. **Roster was blank — Table 2 omitted
  with the documented exception.**
- [x] Ford ≟ MultiAgentOrchestrator answered with at least two code
  citations. **Provided three:** `agent_definitions.py:177-179`,
  `multi_agent_orchestrator.py:124-133`, and `_inventory.json:34`.
- [x] The "3 agents" claim in CLAUDE.md is either confirmed (with the
  other 12 scoped out as non-dev-MAS) or contradicted (with evidence).
  **Confirmed for Layer 3 only; flagged as misleading for the dev
  MAS as a whole, with the additional 12-13 agents enumerated above
  per Table 1.**
