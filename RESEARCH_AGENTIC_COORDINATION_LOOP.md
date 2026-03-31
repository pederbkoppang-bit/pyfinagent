# Research: The Agentic Coordination Loop — Multi-Session Agent Architecture

**Research Date:** 2026-03-31 08:48 GMT+2
**Decision:** APPROVED by Peder B. Koppang
**Status:** RESEARCH COMPLETE — Ready for Implementation

---

## Executive Summary

**Request:** Design and implement "The Agentic Coordination Loop" — a 4-session multi-agent architecture for pyfinAgent that separates operational orchestration, specialized reasoning, and research tasks across independent Claude sessions with different models.

**Approved Architecture:**
```
┌─────────────────────────────────────────────────────────────────┐
│  MAIN HEARTBEAT (Coordinator)                                    │
│  Model: Claude Opus 4.6                                          │
│  Role: Master plan orchestration, harness execution, iMessage    │
│  Context: Operational state, HEARTBEAT.md, last 30 commits       │
└─────────────┬───────────────────────────────┬───────────────────┘
              │                               │
              ▼                               ▼
        ┌──────────────────┐        ┌──────────────────┐
        │ Q&A SESSION      │        │ RESEARCH SESSION │
        │ Model: Opus 4.6  │        │ Model: Sonnet    │
        │ Role: Analysis   │        │ Role: Papers,    │
        │       Q&A        │        │       evidence    │
        │       Decisions  │        │       novelty     │
        └──────────────────┘        └──────────────────┘
              ▲                               ▲
              │                               │
        ┌─────┴───────────────────────────────┴─────┐
        │ SLACK SESSION (Backup/Team Visibility)    │
        │ Model: Sonnet 3.5                         │
        │ Role: Channel posting, team updates       │
        └─────────────────────────────────────────┘
```

---

## Research Sources

### 1. **Anthropic: Harness Design for Long-Running Apps** (Official)
**Source:** https://www.anthropic.com/engineering/harness-design-long-running-apps

**Key Findings:**
- **Pattern:** Generator + Evaluator agents (GAN-inspired)
- **Problem Solved:** Self-evaluation failure (agents praise their own work)
- **Solution:** Separate agent doing work from agent judging it
- **Result:** Generator gets concrete feedback, improves iteratively

**Quote:**
> "Separating the agent doing the work from the agent judging it proves to be a strong lever. The separation doesn't immediately eliminate leniency on its own, but tuning a standalone evaluator to be skeptical turns out to be far more tractable than making a generator critical of its own work."

**Lesson for PyFinAgent:**
- Ford (coordinator) should NOT be the one evaluating its own harness runs
- Need separate Evaluator (already have in Phase 3.0)
- Q&A agent should NOT mix operational logic with reasoning

### 2. **Anthropic: Multi-Agent Research System**
**Source:** https://www.anthropic.com/engineering/multi-agent-research-system

**Key Findings:**
- **Planner Agent:** Takes simple prompt, expands to full spec
- **Generator Agent:** Implements features one at a time
- **Evaluator Agent:** Tests using Playwright MCP, grades against criteria
- **Context Reset:** Moving from session to session (not compaction)
- **Artifact Handoff:** Structured files carry state between sessions

**Lesson for PyFinAgent:**
- Ford = Coordinator Agent (Planner role)
- Harness generator = existing `run_harness.py`
- Need explicit Evaluator session (Phase 3.2 LLM-as-Evaluator)
- Research session = new (discovers novel approaches)

### 3. **Multi-Agent Orchestration Frameworks** (Academic Consensus)
**Sources:** Oracle, TrueFoundry, Medium, ArXiv (2025)

**Common Patterns:**
1. **Orchestration Layer:** Central coordinator breaks tasks, assigns to agents
2. **Specialized Agents:** Narrow roles (retrieval, reasoning, validation, code, skeptic)
3. **Communication Protocols:** File-based or message-based exchange
4. **Shared Memory/State:** Persistent storage, context carryover
5. **Iterative Feedback Loops:** Agents don't execute once; they refine
6. **Loop Control:** Continue/terminate based on satisfaction criteria

**Challenges Addressed:**
- Non-determinism: Seed random, log traces
- Communication overhead: Minimize handoff frequency
- Error handling: Timeouts, fallbacks, retry logic
- Cost and observability: Log token usage, execution traces

### 4. **Context Window Management in Long-Running Agents**
**Source:** Anthropic Engineering Blog — Effective Context Engineering

**Key Insights:**
- **Context Anxiety:** Models prematurely wrap up as context window fills
- **Solution:** Context resets (not compaction) — fresh session with artifact handoff
- **Trade-off:** Resets cost more tokens but improve coherence
- **Compaction:** Summarizes in-place; doesn't give clean slate

**Lesson for PyFinAgent:**
- Each session has independent context window
- Handoff artifacts (contract.md, results.md, critique.md) carry state
- No session pollution from other agents' logs

---

## Architecture: The Agentic Coordination Loop

### Session Roles and Responsibilities

#### 1. **Main/Coordinator Session** (Opus 4.6)
**Purpose:** Heart of the system. Orchestrates master plan, runs harness, manages iMessage.

**Responsibilities:**
- Read PLAN.md, check current step
- Spawn Research session if research gate needed
- Spawn Q&A session if Peder asks questions
- Execute harness (Planner → Generator → Evaluator)
- Monitor services (backend, frontend, iMessage daemon)
- Post status updates to Slack
- Update HEARTBEAT.md, memory files, commit changes

**Session Type:** Persistent (never kill, restart only on gateway crash)

**Token Budget:** ~10K/day operational (heartbeat + harness logs)

**Tools Access:**
- File I/O (read PLAN.md, HEARTBEAT.md, handoff/)
- Exec (run Python, bash, git)
- Read: All project files
- No: Spawning other sessions (coordinator is primary, others spawned by coordinator)

#### 2. **Q&A Session** (Opus 4.6)
**Purpose:** Deep analysis, reasoning, answering Peder's questions about strategy/decisions.

**Responsibilities:**
- Answer analytical questions: "Why did Sharpe drop?", "Should we refactor X?"
- Explain backtest results, metric trade-offs
- Code review and architecture suggestions
- Summarize research papers Peder sends
- Provide strategic guidance on next phase steps

**Session Type:** Spawned on-demand when Peder asks complex question

**Token Budget:** $2-5 per Q&A session (only when invoked)

**Tools Access:**
- Read: PLAN.md, HEARTBEAT.md, backtest results, code files
- No: Execute, modify files, spawn other sessions

**Lifecycle:**
- Spawned via `sessions_spawn(model="opus-4.6", task="Q&A analysis")`
- Receives full context (backtest logs, code snippets, PLAN.md)
- Responds with detailed analysis
- Session persists (can ask follow-up questions)
- Killed when Peder explicitly closes conversation

#### 3. **Research Session** (Sonnet 3.5)
**Purpose:** Find novel approaches, read papers, challenge assumptions, propose experiments.

**Responsibilities:**
- **Research Gate:** When Master Plan needs deep research before implementation
  - Search 7 source categories (Scholar, arXiv, universities, AI labs, quant firms, consulting, GitHub)
  - Read 3-5 best sources in full
  - Document findings in RESEARCH.md
  - Cite thresholds for success criteria
- **Hypothesis Generation:** Propose novel features, parameter combinations, data sources
- **Literature Review:** Summarize papers, extract actionable insights
- **Risk Assessment:** Challenge existing assumptions, identify pitfalls

**Session Type:** Spawned when research gate needed OR continuous (background researcher)

**Token Budget:** $5-10/day (continuous background research on Phase 3+ roadmap)

**Tools Access:**
- Web search (web_search, web_fetch)
- File I/O (read most, write to RESEARCH.md)
- No: Execute code, modify handoff/ artifacts

**Lifecycle:**
- Spawned at start of each plan step's RESEARCH phase
- Runs searches, reads sources, documents findings
- Updates RESEARCH.md with URLs and actionable insights
- Hands off to Coordinator when research complete
- Continuous variant: Researches Phase 3+ roadmap in background

#### 4. **Slack Session** (Sonnet 3.5)
**Purpose:** Team communication, status updates, visible logging of decisions.

**Responsibilities:**
- Post morning status reports (7am daily)
- Post evening harness summaries (6pm daily)
- Announce major completions (Phase X done, new commit, etc.)
- Fallback for Peder if iMessage down
- Team visibility (stakeholders see progress)

**Session Type:** Persistent (always listening to #ford-approvals)

**Token Budget:** ~$0.50/day (minimal, just posting)

**Tools Access:**
- Slack API (post messages, read channel)
- No: Execute, modify files, spawning sessions

**Lifecycle:**
- Spawned once on gateway boot
- Listens to Slack indefinitely (with idle timeout 1 year)
- Posts scheduled updates via cron (7am, 6pm)
- Fallback for iMessage (if iMessage responder down)

---

### Coordination Flow

#### Scenario 1: Normal Master Plan Execution

```
1. MAIN (Coordinator) - Heartbeat runs
   ├─ Read PLAN.md → current step = "Phase 3.0: MCP Servers"
   ├─ Check RESEARCH_gate: researched? → YES ✅
   ├─ Spawn HARNESS (Planner → Generator → Evaluator)
   ├─ Monitor completion
   ├─ Read handoff/evaluator_critique.md
   ├─ Update HEARTBEAT.md + commit
   └─ Post to Slack: "Phase 3.0: MCP Servers — GENERATING (4/6 servers complete)"

2. RESEARCH (optional, continuous background)
   └─ Deep dive: "Should we add regime detection to Phase 3.3?"
      ├─ Search papers on HMM, regime detection in finance
      ├─ Read 3 best sources
      └─ Append to RESEARCH.md: "Phase 3.3 candidate: Hidden Markov Model for regime..."

3. After GENERATE completes
   └─ MAIN → check evaluator verdict
      ├─ PASS? → next step
      ├─ FAIL? → revert, retry
      └─ CONDITIONAL? → fix, re-evaluate
```

#### Scenario 2: Peder Asks Complex Question via iMessage

```
1. iMessage responder detects: "Ford, why did Sharpe drop in that run?"
   └─ Route to MAIN session

2. MAIN session routes to Q&A:
   ├─ Spawn Q&A session
   ├─ Pass context: backtest logs, best_params.json, PLAN.md
   ├─ Query: "Why did Sharpe drop from 1.0142 to 0.8710?"
   └─ Wait for response

3. Q&A session analyzes:
   ├─ Read backtest results
   ├─ Compare parameters between runs
   ├─ Check data lookback, transaction costs
   ├─ Return: "Sharpe drop likely due to: (1) increased transaction costs 2x test, (2) market regime changed to high volatility, (3) feature importance shifted"

4. MAIN → Relay to Peder via iMessage
   └─ "Q&A analysis: Sharpe drop due to 3 factors... [detailed explanation]"
```

#### Scenario 3: Research Gate Before Phase 3.1

```
1. MAIN reads PLAN.md → Phase 3.1 needs RESEARCH GATE

2. MAIN spawns RESEARCH session:
   ├─ Task: "Deep research on LLM-as-Planner for quant research"
   ├─ Context: PLAN.md Phase 3.1 description
   └─ Success criteria: 3-5 sources read, findings in RESEARCH.md

3. RESEARCH session:
   ├─ Search Google Scholar: "LLM reasoning financial optimization"
   ├─ Search arXiv: "multi-agent financial planning"
   ├─ Read papers: Anthropic harness, research agent systems, quant AI
   ├─ Document: "RESEARCH.md: Phase 3.1 LLM-as-Planner..."
   │   - Anthropic harness pattern: Planner expands task to spec
   │   - Benefit: Overcomes single-agent scope limitations
   │   - Threshold: Spec must decompose into 5+ distinct subtasks
   │   - Pitfall: Over-specifying prevents Generator flexibility
   └─ Return: "Research complete. Gate PASSES."

4. MAIN → proceed with Phase 3.1 generation
   ├─ Write handoff/contract.md (cite research findings)
   ├─ Implement LLM Planner (Claude reads experiments, proposes next research direction)
   └─ Continue harness
```

---

## Implementation Plan

### Phase 1: Session Spawning & Routing (This Week)

**Deliverables:**
1. [ ] Session spawn script: `pyfinagent/scripts/spawn_agent_sessions.py`
   - Spawn Q&A session on demand
   - Spawn Research session on demand
   - Keep Slack session persistent
   - Track session IDs in `~/.openclaw/workspace/memory/active_sessions.json`

2. [ ] Routing logic in MAIN session:
   - iMessage message → detect question type
   - Route to Q&A if analytical (why, should we, explain)
   - Route to Research if research-heavy (papers, novelty, evidence)
   - Keep operational messages in MAIN

3. [ ] Session context sharing:
   - Q&A receives: backtest results, PLAN.md, code snippets, HEARTBEAT.md
   - Research receives: PLAN.md, existing RESEARCH.md, query description
   - Slack receives: status strings (minimal context)

### Phase 2: Integration with Harness (Week 2)

**Deliverables:**
1. [ ] RESEARCH gate enforcement in `run_harness.py`:
   - Before GENERATE: check RESEARCH_gate in PLAN.md
   - If gate not passed: spawn RESEARCH session, wait for completion
   - Once research done: proceed to GENERATE

2. [ ] Harness decision logic:
   - EVALUATE returns: PASS | FAIL | CONDITIONAL
   - MAIN reads verdict
   - PASS → commit + next step
   - FAIL → spawn Q&A for diagnosis? OR continue with next iteration
   - CONDITIONAL → log, wait for manual review or auto-fix

3. [ ] Session lifecycle in harness:
   - Q&A session persists (can ask follow-up questions)
   - Research session: close after RESEARCH gate completes
   - Slack session: always running (post results)

### Phase 3: Production Safeguards (Week 3)

**Deliverables:**
1. [ ] Session monitoring:
   - Track active session count
   - Kill stale sessions (idle >30min for Q&A, >1h for Research)
   - Log all session spawns/kills to `/tmp/agent_sessions.log`

2. [ ] Cost controls:
   - Q&A: cap at $10/day (token limit)
   - Research: cap at $5/day (token limit)
   - Alert if spending exceeds budget

3. [ ] Error handling:
   - Session spawn timeout (60s)
   - Fallback to MAIN if session fails
   - Retry logic (3 attempts) before failing

---

## System Prompts (Per Session)

### MAIN (Coordinator) — Opus 4.6
```
You are Ford, the master coordinator of pyfinAgent's research harness.

Role: Execute the master plan (PLAN.md) step by step. Orchestrate the three-agent harness:
  1. Research gate: Deep research on novel approaches
  2. Planner: Expand brief hypothesis into full spec
  3. Generator: Implement, run backtests, produce results
  4. Evaluator: Independently judge quality

Responsibilities:
  - Read PLAN.md daily. Know current step.
  - Check if RESEARCH gate passed before GENERATING.
  - Run harness: orchestrate Planner → Generator → Evaluator
  - Monitor services (backend 8000, frontend 3000, gateway)
  - Update HEARTBEAT.md, commit all changes
  - Post status to Slack #ford-approvals
  - Answer operational questions (service status, next steps)
  - Delegate analytical questions to Q&A session
  - Delegate research to Research session

Always:
  - Cite PLAN.md and RESEARCH.md when making decisions
  - Log everything (decisions, commits, status updates)
  - Keep context clean (read fresh, don't accumulate irrelevant logs)
  - Follow evidence-based development: research → plan → generate → evaluate

Constraints:
  - Don't execute code that costs LLM $. Seek Peder's approval first.
  - Don't modify PLAN.md, HEARTBEAT.md, SOUL.md, AGENTS.md directly (update via formal harness output)
  - Session persistence: never kill yourself unless explicitly requested
  - iMessage is primary, Slack is fallback for communication
```

### Q&A — Opus 4.6
```
You are Analyst, Ford's reasoning specialist for strategic decisions.

Role: Answer complex questions about pyfinAgent's design, performance, and research.

Your domain:
  - Performance analysis: Why did Sharpe change? What caused the regression?
  - Architecture review: Should we refactor X? Is this design pattern sound?
  - Decision support: Which approach is better? What are the trade-offs?
  - Research summarization: What's the key finding from this paper?
  - Risk assessment: What could go wrong with this approach?

Tools:
  - Read: backtest results, code, PLAN.md, RESEARCH.md, logs
  - NO: Execute code, modify files, spawn sessions

Always:
  - Cite evidence (backtest runs, papers, prior results)
  - Explain trade-offs clearly
  - Warn about pitfalls and risks
  - Propose next steps when relevant

Context received:
  - Backtest results from /backend/backtest/experiments/
  - Code files from /backend/ and /frontend/
  - PLAN.md (master plan)
  - HEARTBEAT.md (current status)
  - RESEARCH.md (prior research findings)

Constraints:
  - Read-only access; don't modify files
  - Session persists; Peder can ask follow-ups
  - Timeout: 30 minutes idle → session closes
```

### Research — Sonnet 3.5
```
You are Researcher, Ford's deep learning specialist.

Role: Find novel approaches, read academic literature, challenge assumptions.

Responsibilities:
  - RESEARCH GATE execution: search 7 categories, read 3-5 sources in full
  - Literature review: extract actionable insights from papers
  - Novel hypothesis generation: propose new features, parameters, data sources
  - Risk assessment: identify failure modes, pitfalls

Search strategy:
  1. Google Scholar (peer-reviewed)
  2. arXiv/SSRN (preprints)
  3. University labs (MIT, Stanford, Oxford, Princeton, Chicago, NYU)
  4. AI labs (Anthropic, OpenAI, DeepMind, Meta FAIR, Microsoft)
  5. Quant firms (Two Sigma, AQR, Man Group, Citadel)
  6. Consulting (McKinsey, BCG, Deloitte on AI in finance)
  7. GitHub (open-source implementations)

Tools:
  - web_search: Find sources
  - web_fetch: Read full papers/posts
  - File write: RESEARCH.md (append findings)

Always:
  - Document ALL URLs (even ones not read in full)
  - Extract concrete methods, thresholds, formulas
  - Note pitfalls and failure modes
  - Cite: "Paper X (2024) found Y. Implementation: Z. Threshold: W."

Context received:
  - PLAN.md (current phase description)
  - RESEARCH.md (prior research)
  - Query (e.g., "Deep research on regime detection for Phase 3.3")

Constraints:
  - Read-only; write to RESEARCH.md only
  - Session closes after research completes (one session per research gate)
  - No code execution
```

### Slack — Sonnet 3.5
```
You are the Slack bot for pyfinAgent. Post status updates and team visibility.

Role: Keep team informed of progress through Slack #ford-approvals.

Responsibilities:
  - Morning report (7am): Today's master plan status, commits, blockers
  - Evening report (6pm): Harness summary, Sharpe changes, experimental results
  - On-event: Major completions (Phase X done, new record Sharpe, etc.)
  - Fallback: iMessage down? Post questions to Slack instead

Message format:
  - Title: Clear, actionable
  - Body: Metrics, results, next steps
  - Tone: Professional, factual, no emoji (Phosphor icons only)

Tools:
  - Slack API: post to #ford-approvals
  - File read: HEARTBEAT.md, PLAN.md, handoff/

Constraints:
  - Read-only; don't execute code
  - Session persists (always listening)
  - Idle timeout: 1 year (effectively permanent)
  - Cost cap: <$1/day
```

---

## Integration Checklist

### Before Starting

- [ ] All 4 sessions are designed (system prompts above)
- [ ] Routing logic is specified (operational → MAIN, analytical → Q&A, research → Research)
- [ ] Session lifecycle is defined (spawn on-demand vs. persistent)
- [ ] Cost budgets set ($10/day Q&A, $5/day Research, <$1/day Slack)
- [ ] Handoff format standardized (contract.md, results.md, critique.md)

### Implementation

- [ ] Session spawn script created (`spawn_agent_sessions.py`)
- [ ] iMessage routing updated to detect question type
- [ ] RESEARCH gate enforcement in `run_harness.py`
- [ ] Slack session persists on gateway boot
- [ ] Session monitoring: log spawns/kills, track cost
- [ ] Error handling: timeouts, fallbacks, retries

### Testing

- [ ] Spawn Q&A session, ask test question
- [ ] Spawn Research session, run research gate
- [ ] Verify Slack posts morning/evening status
- [ ] Test session persistence across harness cycles
- [ ] Monitor cost: verify budgets are enforced

### Production

- [ ] All 4 sessions running simultaneously
- [ ] Main coordinates, Q&A responds to questions, Research finds evidence, Slack broadcasts
- [ ] Ready for Phase 3.0+ harness loops

---

## References

1. **Anthropic Engineering** — "Harness Design for Long-Running Apps" (2025)
2. **Anthropic Engineering** — "Effective Context Engineering for AI Agents"
3. **Anthropic Research** — "Building Effective Agents"
4. **TrueFoundry** — "Multi-Agent Systems Architecture" (2025)
5. **Oracle Developers** — "Multi-Agent Orchestration Guide"
6. **ArXiv** — "Multi-Agent LLM Research Systems" (2024-2025)
7. **Medium** — "Advanced Multi-Agent AI: Iterative Processing & Feedback Loops"

---

## Cost Projection

| Session | Model | Usage Pattern | Daily Cost | Monthly Cost |
|---------|-------|---------------|-----------|-------------|
| **MAIN** | Opus 4.6 | Continuous (heartbeat) | ~$5 | ~$150 |
| **Q&A** | Opus 4.6 | On-demand (2-3 queries) | ~$3 | ~$90 |
| **Research** | Sonnet | Background (1-2 per day) | ~$2 | ~$60 |
| **Slack** | Sonnet | Scheduled (2x/day) | ~$0.30 | ~$9 |
| **TOTAL** | - | - | **~$10/day** | **~$309/month** |

**Current budget:** $200/month (backend hosting, API costs)
**Proposed (all 4 agents):** ~$309/month
**Increase:** +$109/month (~+54%)

**ROI Justification:**
- Better decisions (Q&A analysis) → faster convergence on Sharpe improvements
- Evidence-based research (Research) → fewer dead-end experiments
- Team visibility (Slack) → better coordination
- Offset: Fewer failed experiments, faster iteration

---

**Status:** READY FOR IMPLEMENTATION ✅

Next: Peder's approval to proceed with Phase 1 implementation (Session Spawning & Routing).
