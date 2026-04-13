# Phase 3.2.1 Phase 3 Contract: Inter-Session Communication Testing

**Phase:** 3.2.1 Phase 3 (Verify agents communicate and respond correctly)
**Start Date:** 2026-03-31 17:52 GMT+2
**Approval:** Peder B. Koppang ✅
**Harness:** PLAN → GENERATE → EVALUATE → DECIDE → LOG

---

## Hypothesis

By testing message routing from iMessage to each spawned agent (Q&A, Research, Slack), we can validate that the Agentic Coordination Loop correctly:
1. Classifies incoming messages (operational/analytical/research)
2. Routes to appropriate agent session
3. Receives responses back to user
4. Handles failures gracefully

**Expected Outcome:** 3+ test messages routed successfully, all agents respond within SLA (<30s Q&A, <2min Research, <5s Slack).

---

## Success Criteria

### Primary (MUST HAVE)
- [ ] **Q&A routing test:** Analytical question → Q&A agent → response within 30s
  - Test message: "Why did seed_42 Sharpe drop to 1.0142?"
  - Expected: Analyst responds with evidence-backed analysis
  - Validates: Routing accuracy + Q&A agent responsiveness

- [ ] **Research routing test:** Research question → Research agent → RESEARCH.md update
  - Test message: "Research trending indicators for signal generation"
  - Expected: Research agent searches, reads sources, appends findings
  - Validates: Web tools working, file writes operational

- [ ] **Slack integration test:** Slack status post appears in #ford-approvals
  - Test message: Trigger morning/evening summary post
  - Expected: Formatted update appears in channel
  - Validates: Slack API integration working

- [ ] **Operational message handling:** Operational questions stay with MAIN
  - Test message: "What's the next phase?"
  - Expected: MAIN (Ford) handles directly
  - Validates: Fallback behavior correct

- [ ] **Error recovery:** Message to unavailable session falls back gracefully
  - Test: Query Q&A if session crashes
  - Expected: Fallback to MAIN with "Q&A unavailable" message
  - Validates: Robustness

### Secondary (NICE TO HAVE)
- [ ] Response time tracking (latency <SLA for each agent)
- [ ] Cost per message logged
- [ ] Session ID verification in responses

---

## Fail Conditions

1. Q&A doesn't respond within 30s → investigate session or spawn new
2. Research finds <3 sources or doesn't update RESEARCH.md → incomplete
3. Slack post doesn't appear in #ford-approvals → API integration issue
4. Routing accuracy drops <95% → revise classifier
5. Agent session crashes on message → needs error handling

---

## Implementation Plan

### Task 1: Q&A Communication Test (20 min)
- Send analytical test message via iMessage: "Why did Sharpe vary so much?"
- Wait for Q&A response
- Record response time, quality, evidence citations
- Log results

### Task 2: Research Communication Test (20 min)
- Send research test message: "Research market microstructure for execution"
- Monitor RESEARCH.md for updates
- Verify findings are actionable (concrete methods, not vague)
- Log results

### Task 3: Slack Communication Test (10 min)
- Trigger test status post to #ford-approvals
- Verify formatting and content
- Log results

### Task 4: Fallback Test (15 min)
- Send operational message to Q&A (should route to MAIN)
- Send message to non-existent session (should fallback)
- Verify error handling

### Task 5: Performance Metrics (10 min)
- Collect response latencies
- Verify within SLA (<30s Q&A, <2min Research, <5s Slack)
- Calculate routing accuracy percentage

**Total Estimated Time:** ~75 minutes

---

## Success Metrics

| Test | Target | Metric |
|------|--------|--------|
| Q&A response time | <30s | Record actual latency |
| Q&A response quality | Evidence-backed | Verify citations |
| Research sources | ≥3 | Count found sources |
| Research findings | In RESEARCH.md | Verify file update |
| Slack post | Appears in channel | Screenshot/log |
| Routing accuracy | >95% | Count successes |
| Error recovery | Graceful fallback | Verify MAIN handles |

---

## Sign-Off

**Prepared by:** Ford (Coordinator Agent)
**Approved by:** Peder B. Koppang ✅ (2026-03-31 17:52 GMT+2)
**Ready to execute:** YES ✅

---

**Next Step:** GENERATE → Execute test messages and gather results
**Expected Completion:** 2026-03-31 19:30 GMT+2 (~1.5 hours)
**Evaluator assigned:** Peder (review all test results + agent responses)
