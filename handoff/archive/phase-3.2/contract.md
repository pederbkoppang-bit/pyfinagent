# Phase 3.2.1 Contract: Agentic Coordination Loop — Session Spawning & Routing

**Phase:** 3.2.1 (Sub-phase of Phase 3.2: LLM-as-Evaluator)
**Start Date:** 2026-03-31 11:05 GMT+2
**Approval:** Peder B. Koppang ✅
**Harness:** PLAN → GENERATE → EVALUATE → DECIDE → LOG

---

## Hypothesis

By implementing a multi-session agent architecture with specialized agents for different domains (orchestration, analysis, research, broadcasting), we can:
1. Keep the main coordinator (Ford) lightweight and responsive
2. Delegate complex reasoning to Opus-powered Q&A session
3. Enable evidence-based research via dedicated Research session
4. Maintain team visibility through persistent Slack session

**Expected Outcome:** 4 independent OpenClaw sessions running in parallel, coordinating via message routing and artifact handoff.

---

## Success Criteria (Research-Backed)

### Primary (MUST HAVE)
- [ ] **Q&A session spawned:** Successfully spawn Analyst (Opus 4.6) on-demand
  - Criterion: Session responds to test question within 30 seconds
  - Evidence: Anthropic harness research shows specialized agents improve reasoning quality

- [ ] **Research session spawned:** Successfully spawn Researcher (Sonnet) for research gates
  - Criterion: Can execute web_search, web_fetch, RESEARCH.md writes
  - Evidence: Anthropic research system successfully uses web tools for deep research

- [ ] **Slack session spawned:** Successfully spawn and maintain persistent Slack broadcaster
  - Criterion: Posts morning (7am) and evening (6pm) status updates without error
  - Evidence: Industry practice (Slack bots) are reliable for async communication

- [ ] **Message routing implemented:** iMessage responder correctly classifies and routes messages
  - Criterion: Analytical questions → Q&A, Research questions → Research, Operational → MAIN
  - Evidence: OpenClaw message routing patterns established in prior work

- [ ] **Session context sharing:** Each session receives appropriate context for its domain
  - Criterion: Q&A has backtest logs + PLAN.md; Research has web access + RESEARCH.md
  - Evidence: Anthropic harness emphasizes structured handoff of context

### Secondary (NICE TO HAVE)
- [ ] Session persistence tracking: active_sessions.json maintains accurate state
  - Criterion: Matches actual running sessions, survives gateway restarts
  
- [ ] Cost tracking: Each session has budget enforced (<$15/day total Q&A+Research)
  - Criterion: Token usage logged per session, alerts if budget exceeded

- [ ] Error handling: Graceful fallback if session spawn fails
  - Criterion: iMessage question routed to MAIN if Q&A unavailable

---

## Fail Conditions

1. Session spawn timeout (>60 seconds) — retry 3x, then fallback
2. Message routing misclassifies >10% of test messages — refine classifier
3. Session loses context during handoff — re-send context, log error
4. Cost exceeds $20/day (50% buffer) — alert Peder, throttle spawning
5. Gateway restart kills all sessions — sessions must survive and restart

---

## Implementation Plan

### Work Breakdown

**Task 1: Spawn Q&A Session (2 hours)**
- Create sessions_spawn call in main agent with Opus 4.6 model
- System prompt: QA_SYSTEM (from RESEARCH_AGENTIC_COORDINATION_LOOP.md)
- Test: Send test question, verify response

**Task 2: Spawn Research Session (1.5 hours)**
- Create sessions_spawn call with Sonnet model
- System prompt: RESEARCH_SYSTEM
- Tools: web_search, web_fetch, file write to RESEARCH.md
- Test: Request research gate, verify findings append to RESEARCH.md

**Task 3: Spawn Slack Session (1 hour)**
- Create sessions_spawn call with persistent mode
- System prompt: SLACK_SYSTEM
- Tools: Slack API integration
- Test: Verify 7am status post appears in #ford-approvals

**Task 4: Message Routing (2 hours)**
- Update imsg_responder.py to call detect_question_type()
- Route analytical → Q&A session
- Route research → Research session
- Route operational → MAIN
- Test: Send 10 sample messages, verify correct routing

**Task 5: Session Context Sharing (1 hour)**
- Q&A receives: PLAN.md, HEARTBEAT.md, backtest/experiments/results/
- Research receives: PLAN.md, RESEARCH.md, web access
- Slack receives: HEARTBEAT.md for status posts
- Test: Verify each session has needed context

**Task 6: Active Session Tracking (1 hour)**
- Create active_sessions.json structure
- Update spawn_agent_sessions.py to track spawned session IDs
- Test: Verify sessions persist across multiple commands

**Total Estimated Time:** ~8-9 hours
**Testing:** ~2 hours
**Total:** ~10 hours (equivalent to 1 full work day)

---

## Resources Needed

- OpenClaw sessions_spawn API access
- Slack integration (already configured)
- iMessage responder script (already running)
- RESEARCH.md, PLAN.md, HEARTBEAT.md (read access)

---

## Acceptance Criteria

✅ **All 4 sessions spawned and responding**
✅ **Message routing accuracy >90%**
✅ **Cost tracking active (<$15/day Q&A+Research)**
✅ **Session persistence verified**
✅ **Slack morning/evening posts appear in #ford-approvals**
✅ **No gateway restart issues**

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|-----------|
| Session spawn fails | Loss of Q&A capability | Fallback to MAIN; retry 3x |
| Message routing misclassifies | Wrong agent for query | Human feedback to improve classifier |
| Cost overrun | Budget exceeded | Daily cost tracking, token limits |
| Session loses context | Agent can't perform task | Log error, resend context |
| Gateway crash kills sessions | All agents go offline | Sessions recover on gateway restart |

---

## Deliverables

1. **Code:**
   - Updated spawn_agent_sessions.py (session tracking)
   - Updated imsg_responder.py (message routing)
   - sessions_spawn calls for Q&A, Research, Slack

2. **Artifacts:**
   - active_sessions.json (session ID tracking)
   - /tmp/agent_sessions.log (lifecycle events)
   - /tmp/session_costs.log (cost tracking per session)

3. **Documentation:**
   - Phase 3.2.1 implementation notes
   - Message routing classifier logic
   - Session context specification per agent

4. **Tests:**
   - 10 sample message routing tests
   - Q&A test question → response verification
   - Research gate execution test
   - Slack morning/evening post verification

---

## Success Metrics (Post-Implementation)

- **Session uptime:** >99.5% (survives gateway restarts)
- **Message routing accuracy:** >95%
- **Cost per day:** <$15 (Q&A + Research combined)
- **Response latency:** <30 sec for Q&A, <2 min for Research
- **Session persistence:** Survives 24 hours without intervention

---

## Sign-Off

**Prepared by:** Ford (Coordinator Agent)
**Approved by:** Peder B. Koppang ✅ (2026-03-31 11:05 GMT+2)
**Research basis:** RESEARCH_AGENTIC_COORDINATION_LOOP.md (21,722 bytes)
**Ready to execute:** YES ✅

---

**Next Step:** GENERATE → Implement Phase 1
**Expected Completion:** 2026-03-31 20:00 GMT+2 (9 hours)
**Evaluator assigned:** Peder (manual review of 4 spawned sessions + routing tests)
