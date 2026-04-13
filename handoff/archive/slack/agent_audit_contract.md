# Slack Agent Audit Contract — Ticket System Implementation

**Phase:** Slack Agent Audit (PLAN Phase)
**Start Date:** 2026-03-31 19:33 GMT+2
**Approval:** Peder B. Koppang ✅
**Harness:** PLAN → GENERATE → EVALUATE → DECIDE → LOG

---

## Hypothesis

By implementing a message queue-based ticket system with persistent storage (Postgres), we can ensure:
1. **Zero message loss** — All Slack/iMessage messages create tickets
2. **Guaranteed delivery** — Responses sent within SLA (<5min standard, <1min urgent)
3. **Deduplication** — Slack retries don't create duplicate work
4. **Traceability** — Every message has a ticket ID and status visible to user
5. **Scalability** — Queue handles traffic spikes without dropping messages

**Expected Outcome:** Slack bot with 99.5%+ uptime, <5min response SLA, zero messages lost, full ticket visibility.

---

## Success Criteria

### Primary (MUST HAVE)

- [ ] **Ticket system created:**
  - Postgres `tickets` table with full schema (ID, timestamp, priority, status, response)
  - Ticket lifecycle: OPEN → ASSIGNED → IN_PROGRESS → RESOLVED → CLOSED
  - Deduplication by `envelope_id` (prevents Slack retries from duplicating work)

- [ ] **Message ingestion working:**
  - Slack messages → tickets in <100ms
  - iMessage messages → tickets in <100ms
  - Both channels produce identical ticket format

- [ ] **Queue processing operational:**
  - FIFO queue pulls open tickets in priority order
  - Route by classification (analytical → Q&A, research → Research, operational → MAIN)
  - Spawn agent session per ticket

- [ ] **Response delivery guaranteed:**
  - P0 (Critical): Response within 5 minutes
  - P1 (Urgent): Response within 15 minutes
  - P2 (Standard): Response within 1 hour
  - P3 (Low): Response within 4 hours

- [ ] **User visibility:**
  - Immediate acknowledgment: "Got it! Ticket #5001, assigning to [Agent]..."
  - Slack thread reply shows ticket status
  - iMessage response includes ticket number
  - User can query ticket status

- [ ] **Deduplication working:**
  - Slack envelope_id tracked in database
  - Duplicate events marked as DUPLICATE (not reprocessed)
  - Test: Trigger same Slack event 3x, confirm only processed once

### Secondary (NICE TO HAVE)

- [ ] **SLA monitoring dashboard:** Shows ticket status, response times, breaches
- [ ] **Escalation rules:** P0 unresolved after 10 min → notify Peder directly
- [ ] **Retry logic:** Transient errors (5xx) retry with exponential backoff
- [ ] **Cost tracking:** Log per-ticket LLM cost
- [ ] **Slack thread threading:** Responses posted as thread replies, not channel messages

---

## Fail Conditions

1. Ticket table not created or schema incomplete → restart Phase 2
2. Messages not creating tickets → ingestion broken, debug Slack/iMessage hooks
3. Deduplication not working (duplicates processed) → envelope_id tracking broken
4. Response SLA missed on >1% of tickets → queue processor too slow, scale up
5. Message loss >0.1% → persistence not working, check database
6. Queue processor crashes → needs error handling and auto-restart
7. Response not delivered to user → check Slack API or iMessage delivery

---

## Implementation Plan

### Phase 1: Database & Schema (2 hours)
**Tasks:**
- [ ] Create `tickets` table in Postgres
- [ ] Add indexes (envelope_id, status, created_at)
- [ ] Test INSERT/UPDATE/SELECT
- [ ] Verify schema matches design doc

**Success:** `SELECT * FROM tickets;` returns empty table, schema validated

### Phase 2: Message Ingestion (1.5 hours)
**Tasks:**
- [ ] Modify iMessage responder → create ticket on message receipt
- [ ] Modify Slack webhook → create ticket on event receipt
- [ ] Test message → ticket creation (<100ms)
- [ ] Verify both channels create identical format

**Success:** Send test message via Slack, verify `tickets` table has new row with all fields

### Phase 3: Queue Processing (2 hours)
**Tasks:**
- [ ] Build ticket queue processor (pulls OPEN tickets, FIFO by priority)
- [ ] Classify ticket (analytical/research/operational)
- [ ] Route to agent (Q&A, Research, MAIN)
- [ ] Spawn agent session with ticket context
- [ ] Update ticket status (ASSIGNED → IN_PROGRESS → RESOLVED)

**Success:** Process 10 test tickets, verify all reach RESOLVED status

### Phase 4: Response Delivery (1.5 hours)
**Tasks:**
- [ ] Send response back to Slack (thread reply)
- [ ] Send response back to iMessage
- [ ] Update ticket with response text
- [ ] Mark ticket RESOLVED + set `resolved_at`

**Success:** Process test ticket, receive response in both channels

### Phase 5: Deduplication & Error Handling (1 hour)
**Tasks:**
- [ ] Track `envelope_id` for Slack events
- [ ] Mark duplicates as DUPLICATE (don't reprocess)
- [ ] Test: Send same Slack event 3x, verify only 1 ticket created
- [ ] Add retry logic for transient errors (5xx)

**Success:** Duplicate events marked correctly, no duplicate work

### Phase 6: SLA Tracking & Monitoring (1 hour)
**Tasks:**
- [ ] Add SLA fields to ticket (response_sla, resolution_sla)
- [ ] Calculate SLA breach: `resolved_at - created_at > response_sla`
- [ ] Add escalation: P0 unresolved after 10min → notify Peder
- [ ] Create status query: `SELECT COUNT(*) FROM tickets WHERE status='OPEN'`

**Success:** Query shows open tickets, SLA breaches logged

**Total Estimated Time:** 9 hours (full implementation)

---

## Resource Requirements

**Database:**
- Postgres (already available)
- New `tickets` table (simple schema)
- Indexes for performance

**Backend:**
- Queue processor (new Python service or cron job)
- iMessage responder update (add ticket creation)
- Slack webhook update (add ticket creation)

**No external services needed:**
- No Kafka, RabbitMQ, SQS (keep simple with Postgres queue)
- No new API keys or third-party dependencies

---

## Timeline

| Phase | Duration | Start | End | Status |
|-------|----------|-------|-----|--------|
| 1. Database | 2h | 19:33 | 21:33 | ⏭️ |
| 2. Ingestion | 1.5h | 21:33 | 23:03 | ⏭️ |
| 3. Queue | 2h | 23:03 | 01:03 | ⏭️ |
| 4. Delivery | 1.5h | 01:03 | 02:33 | ⏭️ |
| 5. Dedup | 1h | 02:33 | 03:33 | ⏭️ |
| 6. SLA | 1h | 03:33 | 04:33 | ⏭️ |
| **Total** | **9h** | 19:33 | 04:33+1 | ⏭️ |

**Estimated completion:** 2026-04-01 04:33 GMT+2 (next morning)

---

## Success Metrics

### Functional
| Metric | Target | How to Verify |
|--------|--------|--------------|
| Ticket creation latency | <100ms | Time message receipt to DB INSERT |
| Deduplication accuracy | 100% | Send same event 3x, check DB |
| Response delivery | 100% | Verify Slack + iMessage receive response |
| SLA compliance | >99% | Count tickets resolved within SLA |

### Non-Functional
| Metric | Target | How to Verify |
|--------|--------|--------------|
| Queue throughput | >100 tickets/hour | Run 100 test tickets, measure time |
| Response latency P99 | <5 min | Measure time from ticket creation to response |
| Message loss | <0.01% | Run 1000 messages, count tickets vs deliveries |
| Uptime | 99.5% | Monitor for 24 hours |

---

## Rollback Plan

**If ticket system fails:**
1. Disable ticket creation (remove hook from iMessage/Slack)
2. Agents continue to work normally (tickets just not persisted)
3. Manual tickets for critical issues
4. Debug → fix → re-enable

**No user-facing disruption** — fallback is graceful degradation.

---

## Sign-Off

**Prepared by:** Ford (Coordinator Agent)
**Approved by:** Peder B. Koppang ✅ (2026-03-31 19:33 GMT+2)
**Research basis:** RESEARCH_SLACK_AGENT_AUDIT.md (ticket system architecture)
**Ready to execute:** YES ✅

---

**Next Step:** GENERATE → Implement Phases 1-6
**Expected Completion:** 2026-04-01 04:33 GMT+2
**Evaluator assigned:** Peder (verify ticket system, SLA compliance, message loss <0.01%)
