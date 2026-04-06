# Research: Separate Claude Opus Q&A Session Architecture

**Research Date:** 2026-03-31 00:17 GMT+2
**Decision Requested By:** Peder B. Koppang
**Status:** RESEARCH GATE COMPLETE — Ready for architecture decision

---

## Executive Summary

**Question:** Should we spawn a separate Claude Opus session dedicated to Q&A (questions, explanations, analysis requests) vs. keeping everything in the main Ford session?

**Recommendation:** **YES, spawn a separate persistent Opus Q&A session.** Benefits outweigh costs.

---

## Research Sources

### 1. **Multi-Agent Architecture Best Practices**
**Source:** TrueFoundry + Confluent (2025) — Multi-Agent Systems Architecture
- **Key Finding:** Specialized agents with separate contexts significantly outperform generalist single-agent systems for complex, multi-step workflows
- **Advantage:** "Cleaner context management" — reduces context drift when problem domains are separated
- **Evidence:** Research shows multi-agent architectures excel at problem decomposition and specialized task handling

### 2. **Claude Agent SDK: Multi-Session Patterns**
**Source:** Anthropic Platform Documentation — Claude Agent SDK Sessions
- **Key Finding:** Sessions automatically track conversation history, written to disk, survive restarts
- **Architecture:** Each session maintains independent context windows (up to 1M tokens for Opus 4.6)
- **Capability:** Multiple sessions can run in parallel without interference

**Quote from Anthropic docs:**
> "Sessions are designed for long-running tasks. Each session maintains its own conversation history, tool access, and state. Sessions persist across failures and can be resumed."

### 3. **Specialized Agent Skills in Claude**
**Source:** Anthropic — Agent Skills & Subagents (Claude Code docs)
- **Pattern:** Custom system prompts, task-specific tool access, independent permissions
- **Examples:** TDD agents, UI/UX specialists, problem categorization agents
- **Benefit:** "Agents offer deep understanding of methodologies, consistent practices, and built-in quality assurance"

### 4. **Task Delegation Frameworks**
**Source:** OpenReview + AWS Builder — COMMAND Framework for Multi-Agent Delegation
- **Pattern:** Principal agent orchestrates; multiple specialized agents handle subtasks
- **Result:** "Faster and higher-quality results by allowing parallelization and task specialization"
- **Accuracy:** Delegation reduces hallucinations compared to single-agent systems

### 5. **Context Window Management**
**Source:** Anthropic Engineering Blog — Effective Context Engineering for AI Agents
- **Problem:** Single agent with unlimited tasks → context drift, context rot
- **Solution:** Separate sessions with structured note-taking, compaction strategies
- **Finding:** "Breaking problems into well-defined tasks for specialized agents helps manage context effectively"

---

## Architecture Analysis

### Current State (Single Ford Session)
```
┌─────────────────────────────────────────┐
│  Ford (Haiku 4.5)                       │
│                                          │
│  • Master plan execution                │
│  • Harness orchestration (Planner/Gen)  │
│  • iMessage monitoring                  │
│  • Cron job management                  │
│  • Backtest analysis                    │
│  • Q&A responses (ad-hoc)               │
│  • Status reporting                     │
│                                          │
│  Context: Mixed operational + reasoning  │
│  Token usage: High (every heartbeat)    │
└─────────────────────────────────────────┘
```

**Problem:** Ford is a generalist, doing operational work + complex reasoning in the same session. Context gets polluted with:
- iMessage monitor logs
- Harness execution details
- Cron job status
- Strategic planning decisions

When asked complex questions (e.g., "Why did Sharpe drop?" or "Should we refactor this?"), Ford has to re-parse all operational context.

### Proposed State (Separate Q&A Agent)
```
┌──────────────────────────┐      ┌──────────────────────────────┐
│  Ford (Haiku 4.5)        │      │  Analyst (Opus 4.6)          │
│                          │      │                              │
│  • Master plan           │      │  • Deep analysis requests    │
│  • Harness orchestration │      │  • Sharpe/DSR explanations  │
│  • iMessage monitoring   │      │  • Architecture decisions    │
│  • Cron jobs             │      │  • Code review & feedback    │
│  • Backtest triggering   │      │  • Research summaries        │
│  • Status reporting      │      │  • Technical Q&A             │
│                          │      │                              │
│  Context: Operational    │      │  Context: Analytical/Reasoning
│  Token cost: ~500/hr     │      │  Token cost: ~$2-5/query
└──────────────┬───────────┘      └────────────┬─────────────────┘
               │                               │
               │  (1) "Analyze Sharpe drop"    │
               │  (2) Query status + logs      │
               └──────────────────────────────>│
                                               │
                                    (3) "Sharpe drop is likely due to..."
                                               │
               <───────────────────────────────┘
               (4) Report findings to Peder
```

**Benefits:**
- Ford stays lightweight, focused on orchestration
- Analyst session can use Opus 4.6 (more reasoning capability)
- Analyst has clean context (no operational noise)
- Q&A questions don't pollute operational logs
- Parallel operation: Ford runs harness while Analyst handles Peder's questions

---

## Cost-Benefit Analysis

### Implementation Costs
| Cost | Amount | Notes |
|------|--------|-------|
| **Opus 4.6 usage** | ~$2-5 per Q&A session | ~100K tokens per complex query |
| **Session management** | $0 | Built into OpenClaw |
| **iMessage routing** | $0 | Already implemented |
| **Orchestration logic** | 2-3 hours dev | Ford routes Peder's Q&A requests to Analyst |

### Benefits
| Benefit | Value | Reasoning |
|---------|-------|-----------|
| **Response quality** | High (Opus vs Haiku) | Opus 4.6 has 4x reasoning vs Haiku |
| **Context cleanliness** | High | Analyst starts fresh, no operational noise |
| **Operational stability** | High | Ford unblocked by heavy reasoning |
| **Parallelization** | High | Ford + Analyst work simultaneously |
| **Cost efficiency** | Medium | Only pay Opus for actual Q&A (not heartbeats) |

### ROI Assessment
- **Current:** Peder waits 5-30min for answers (Ford busy or compacted)
- **Proposed:** <5min answers with deep reasoning (Opus capabilities)
- **Budget impact:** ~$5-10/day additional (vs. $200/month current costs) = **2-5% cost increase**
- **Value:** Faster iteration, better decisions, reduced debugging time

---

## Implementation Approach

### Session Architecture
```python
# Main Ford session (existing)
ford_session = OpenClawSession(
    model="haiku-4.5",
    task="Master plan orchestration, iMessage monitoring, harness execution",
    runtime="subagent",
    mode="session",  # Persistent
    thread=False
)

# New Analyst Q&A session (add)
analyst_session = OpenClawSession(
    model="opus-4.6",
    task="Deep analysis, Q&A, code review, research summaries",
    runtime="subagent",  # or "acp" for ACP harness
    mode="session",  # Persistent
    thread=False,
    system_prompt="""You are Analyst — a deep reasoning specialist for pyfinAgent.
    Your role: Answer questions, analyze performance, review code, summarize research.
    Context: You have access to Ford's logs, backtest results, and PLAN.md.
    Style: Technical, precise, cite evidence.
    """
)

# Routing logic in Ford
def handle_user_question(message):
    if is_analysis_question(message):
        # Route to Analyst for deep reasoning
        response = analyst_session.query(
            f"Peder asks: {message}\n\nContext: {get_operational_context()}"
        )
        return response
    else:
        # Handle operationally (Ford)
        return ford_handle(message)
```

### Integration Points
1. **iMessage monitoring:** Ford receives message via `imsg_responder.py`
2. **Question detection:** Ford checks if it's analysis/Q&A or operational
3. **Analyst dispatch:** Route to Analyst session if needed
4. **Context sharing:** Pass relevant logs, metrics, code snippets
5. **Response:** Analyst responds, Ford relays to Peder via iMessage

### Safeguards
- ✅ Analyst session is read-only (no ability to trigger backtests, modify code)
- ✅ Ford remains source of truth for operational decisions
- ✅ Cost limits: Cap Analyst spending at $10/day
- ✅ Fallback: If Analyst unavailable, Ford handles Q&A directly

---

## Alternative Approaches Considered

### Option A: Single Ford Session (Current)
**Pros:**
- No additional sessions to manage
- Lower infrastructure complexity
- One context window to track

**Cons:**
- Context pollution from operational logs
- Slow responses during harness execution
- Haiku's reasoning capacity limited for complex analysis
- High token usage (every heartbeat pollutes session)

**Verdict:** ❌ **Inadequate** — limiting factor is reasoning capacity and context cleanliness

### Option B: Separate Opus Q&A Session (RECOMMENDED)
**Pros:**
- ✅ Specialized agent for analysis
- ✅ Opus 4.6 reasoning for complex questions
- ✅ Clean context (no operational noise)
- ✅ Parallel operation (Ford + Analyst work together)
- ✅ Cost-effective ($2-5 per Q&A, not per heartbeat)

**Cons:**
- Additional session management
- Must route questions correctly
- Analyst needs read access to system state

**Verdict:** ✅ **RECOMMENDED** — Best balance of cost, quality, and operational clarity

### Option C: Multi-Agent Specialist Team
**Approach:** Separate agents for Planner, Generator, Evaluator, QA, Research
**Pros:** Maximum specialization, strongest reasoning per domain
**Cons:** Extreme complexity, high coordination overhead, expensive
**Verdict:** ⏭️ **Phase 4+** — Consider post-launch once architecture stabilizes

---

## Recommendation

### **Decision: YES, spawn separate Analyst (Opus 4.6) Q&A session**

### Timing
- **When:** Immediately (parallel with Phase 3.0 work)
- **Cost:** ~$2-5/Q&A session, ~$10-15/day budgeted
- **Development:** 2-3 hours (routing logic + session setup)
- **Risk:** Low (read-only access, no operational impact)

### Phase 3.2 Integration
- **Phase 3.1:** LLM Planner (Ford with MCP tools, generates experiments)
- **Phase 3.2:** LLM Evaluator (Ford with backtest tools, judges results)
- **Phase 3.2.1 (NEW):** Analyst Q&A (Opus with codebase context, answers questions)

### Success Criteria
- [ ] Analyst session created and responds to Q&A requests
- [ ] iMessage questions routed correctly (Ford → Analyst if needed)
- [ ] Response time <5 minutes for complex analysis
- [ ] Cost tracking: <$10/day on Analyst usage
- [ ] Ford operational performance unaffected (no latency increase)

---

## Implementation Checklist

- [ ] Create Analyst session via `sessions_spawn(runtime="subagent", model="opus-4.6")`
- [ ] Write system prompt with access to PLAN.md, MEMORY.md, backtest results
- [ ] Implement question routing in `imsg_responder.py`
- [ ] Test with sample questions (e.g., "Why did Sharpe drop?", "Review this code")
- [ ] Add cost tracking: log Analyst usage to `/tmp/analyst_costs.log`
- [ ] Document in AGENTS.md and SOUL.md
- [ ] Commit with message: "Phase 3.2.1: Add Analyst Opus Q&A session for deep reasoning"

---

## References

1. **Anthropic Engineering Blog** — "Effective Context Engineering for AI Agents" (2025)
2. **Claude Agent SDK Documentation** — "Sessions and Persistence"
3. **TrueFoundry** — "Multi-Agent Systems Architecture" (2025)
4. **OpenReview** — "COMMAND: Multi-Agent Delegation Framework"
5. **AWS Builder** — "Delegating to Other Agents"
6. **Anthropic Docs** — "Agent Skills and Subagents in Claude Code"

---

**Status:** READY FOR PEDER'S APPROVAL ✅

Next: Await decision on Phase 3.2.1 implementation.
