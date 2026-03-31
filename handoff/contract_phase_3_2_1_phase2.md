# Phase 3.2.1 Phase 2 Contract: Agentic Coordination Loop — Session Spawning

**Phase:** 3.2.1 Phase 2 (Spawn Q&A, Research, Slack sessions)
**Start Date:** 2026-03-31 15:49 GMT+2
**Approval:** Peder B. Koppang ✅
**Harness:** PLAN → GENERATE → EVALUATE → DECIDE → LOG

---

## Hypothesis

By spawning three independent Claude sessions (Q&A, Research, Slack) with proper system prompts and context, we can:
1. Enable the main coordinator (MAIN/Ford) to delegate analytical questions to Q&A (Opus 4.6)
2. Enable research gates to spawn Research sessions (Sonnet) for evidence gathering
3. Maintain team visibility through persistent Slack sessions (Sonnet)
4. Achieve <30 second response times for Q&A, <2 minute response times for Research
5. Keep all sessions active in parallel with automatic session recovery

**Expected Outcome:** 3 independent OpenClaw sessions spawned, integrated with iMessage routing, responding to messages from the Agentic Coordination Loop.

---

## Success Criteria (Research-Backed)

### Primary (MUST HAVE)

- [ ] **Q&A session spawned:** Successfully spawn Analyst (Opus 4.6)
  - Criterion: Session responds to analytical question within 30 seconds
  - Example: "Why did Sharpe drop?" → detailed analysis within 30s
  - Evidence: Anthropic harness shows Opus 4.6 maintains quality on reasoning tasks

- [ ] **Research session spawned:** Successfully spawn Researcher (Sonnet)
  - Criterion: Can execute web_search, web_fetch, file writes
  - Example: Research gate → finds 5+ sources, updates RESEARCH.md
  - Evidence: Sonnet model proven for information retrieval in Anthropic research

- [ ] **Slack session spawned:** Successfully spawn persistent Slack broadcaster
  - Criterion: Posts morning (7am) and evening (6pm) status without error
  - Example: "Phase 3.2.1 complete — Message routing 100% accurate"
  - Evidence: Industry standard for async team communication

- [ ] **Message routing integration:** iMessage responder correctly dispatches to spawned sessions
  - Criterion: Analytical questions route to Q&A, research → Research
  - Example: "Research HMM approaches" → Research session receives it
  - Evidence: Phase 1 routing tested at 100% accuracy

- [ ] **Active session tracking:** active_sessions.json updated with session IDs
  - Criterion: Tracks Q&A, Research, Slack session IDs accurately
  - Example: {"qa": "sess-uuid-1", "research": "sess-uuid-2", "slack": "sess-uuid-3"}
  - Evidence: File-based state tracking is reliable and auditable

- [ ] **Session persistence:** Sessions survive 24+ hours without crashing
  - Criterion: Uptime >99% (allow 1 restart per day max)
  - Example: Gateway restart → sessions recover within 5 minutes
  - Evidence: OpenClaw session SDK supports persistence across restarts

### Secondary (NICE TO HAVE)

- [ ] Cost tracking: Per-session token usage logged
  - Criterion: Can identify which session used how many tokens

- [ ] Performance monitoring: Response latencies tracked
  - Criterion: Q&A <30s, Research <2min, Slack <5s

- [ ] Error recovery: Automatic retry on session spawn failure
  - Criterion: Retry 3x before failing over

---

## Fail Conditions

1. Session spawn timeout (>120 seconds) — retry 3x
2. Session unresponsive to test message (>1 minute) — kill and respawn
3. Routing fails to dispatch messages — fallback to MAIN
4. Active sessions JSON corrupted — recover from backup
5. Cost exceeds $25/day (50% buffer above budget) — alert Peder

---

## Implementation Plan

### Task 1: Spawn Q&A Session (1 hour)
- Use `sessions_spawn()` with Opus 4.6 model
- System prompt: QA_SYSTEM from RESEARCH document
- Test question: "Why did Sharpe drop from 1.0142 to 0.8710?"
- Verify response within 30 seconds
- Store session ID in active_sessions.json

### Task 2: Spawn Research Session (1 hour)
- Use `sessions_spawn()` with Sonnet 3.5 model
- System prompt: RESEARCH_SYSTEM from RESEARCH document
- Tools: web_search, web_fetch, file write to RESEARCH.md
- Test request: "Research regime detection approaches"
- Verify RESEARCH.md gets updated
- Store session ID in active_sessions.json

### Task 3: Spawn Slack Session (30 minutes)
- Use `sessions_spawn()` with Sonnet 3.5 model, persistent mode
- System prompt: SLACK_SYSTEM from RESEARCH document
- Tools: Slack API integration
- Test: Schedule test post to #ford-approvals
- Store session ID in active_sessions.json
- Configure cron for 7am/6pm status posts

### Task 4: Integration Testing (1 hour)
- Test iMessage routing → Q&A (analytical question)
- Test iMessage routing → Research (research question)
- Test Slack session posting
- Verify session recovery after mock failure
- Document results in experiment_results.md

### Task 5: Production Hardening (30 minutes)
- Add session monitoring (ping every 5 min)
- Add cost tracking (log token usage)
- Add error recovery (automatic restart if unresponsive)
- Update memory/active_sessions.json atomically

**Total Estimated Time:** ~4 hours
**Testing:** ~1 hour
**Total:** ~5 hours

---

## Resources Needed

- OpenClaw sessions_spawn API
- Slack integration (already configured)
- iMessage responder (already running)
- System prompts (documented in RESEARCH_AGENTIC_COORDINATION_LOOP.md)
- PLAN.md, HEARTBEAT.md, RESEARCH.md (read access for context)

---

## Acceptance Criteria

✅ **All 3 sessions spawned and responding**
✅ **Message routing dispatches to correct session**
✅ **Session IDs persist in active_sessions.json**
✅ **Q&A responds within 30 seconds**
✅ **Research produces findings in RESEARCH.md**
✅ **Slack posts appear in #ford-approvals**
✅ **No crashed sessions or unrecoverable errors**

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|-----------|
| Session spawn fails | Q&A/Research unavailable | Retry 3x, fallback to MAIN |
| Session unresponsive | Message stuck in queue | Timeout + kill + respawn |
| Session ID lost | Routing broken | Atomic file writes, backups |
| Cost overrun | Budget exceeded | Daily tracking, alerts at $20 |
| Gateway restart | Sessions go offline | Recovery via cron watchdog |

---

## Sign-Off

**Prepared by:** Ford (Coordinator Agent)
**Approved by:** Peder B. Koppang ✅ (2026-03-31 15:49 GMT+2)
**Research basis:** RESEARCH_AGENTIC_COORDINATION_LOOP.md
**Ready to execute:** YES ✅

---

**Next Step:** GENERATE → Spawn Q&A, Research, Slack sessions
**Expected Completion:** 2026-03-31 20:00 GMT+2 (4-5 hours)
**Evaluator assigned:** Peder (manual review of spawned sessions + routing tests)
