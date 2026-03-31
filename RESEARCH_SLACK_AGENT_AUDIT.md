# Research: Slack Agent Audit — Ticket System & Reliable Communication

**Research Date:** 2026-03-31 18:51 GMT+2
**Request:** Peder B. Koppang
**Status:** RESEARCH GATE COMPLETE

---

## Executive Summary

**Problem:** Current Slack agent (team communication) may not be answering several messages reliably. Need to audit and implement a ticket system with guaranteed message handling.

**Solution:** Implement a message queue-based ticket system with:
- Message deduplication (handle Slack retries)
- Persistent queue (Postgres or in-memory)
- Exponential backoff for rate limits
- Response time SLAs (<5min for team messages)
- Thread-based organization
- Priority routing (P1 urgent, P2 standard, P3 low)

**Architecture:** Message queue (Slack/iMessage inbound) → Ticket store (Postgres) → Agent processor → Response queue → Slack/iMessage outbound

---

## Research Sources

### 1. **Message Queue Architecture & Ticket Systems**
**Source:** Medium, Apps365, AWS Prescriptive Guidance (2023-2025)

**Key Findings:**
- **FIFO guarantees:** Use FIFO queues to preserve message order (important for conversation context)
- **Deduplication essential:** Slack retries failed events up to 3 times; must deduplicate by event ID
- **Decoupling principle:** Immediately acknowledge Slack event (HTTP 200) within 3 seconds, then process asynchronously
- **Buffer for spikes:** Queue absorbs traffic spikes, allowing steady-state processing
- **Producers/Consumers pattern:** Slack/iMessage → Producer → Queue → Consumer (Agent) → Response

**Implementation options:**
- **AWS SQS FIFO:** Managed, durable, deduplication, ordering guarantees
- **Postgres + trigger:** SQL-based queue (simpler, integrated with existing DB)
- **RabbitMQ:** Complex but powerful, good for microservices
- **Redis:** Fast but ephemeral (risky for persistence)

**Recommendation:** Postgres queue with trigger-based processing (already have DB infrastructure)

### 2. **Slack Bot Reliability Best Practices**
**Source:** Slack Dev Blog, Medium, Slack Docs (2024-2025)

**Key Findings:**

**The 3-Second Timeout Problem:**
- Slack expects HTTP 200 response within 3 seconds for webhook events
- If bot doesn't respond in time, Slack retries the event
- Multiple retries → duplicate messages processed → inconsistent state
- Solution: Acknowledge immediately, process later

**Rate Limiting:**
- Slack API enforces Tier 1 rate limits (~60 requests/minute per token)
- 429 response includes `Retry-After` header (respect it!)
- Use exponential backoff with jitter: 1s, 2s, 4s, 8s (+ random 0-1s)
- Never retry 4xx errors (except 429), only 5xx

**Deduplication Strategy:**
- Slack assigns each event a `ack_deadline` and `envelope_id`
- Track processed `envelope_id` in database → ignore duplicates
- FIFO queue helps enforce order per user

**Error Handling:**
- 4xx errors: Log and skip (malformed request)
- 5xx errors: Retry with backoff
- Timeout errors: Add to retry queue
- Network errors: Circuit breaker pattern (fail fast if service down)

**Thread Management:**
- Use message threading to group bot responses
- Prevents channel flooding
- Easier to follow conversations

### 3. **Ticket System Design for Teams**
**Source:** Zendesk, Moveworks, Internal Note (2024-2025)

**Key Patterns:**
- **Ticket lifecycle:** Open → Assigned → In Progress → Resolved → Closed
- **Priority levels:** P0 (critical), P1 (urgent), P2 (standard), P3 (low)
- **SLA tracking:** Response time SLA, resolution time SLA
- **Escalation:** If unresolved after N hours → escalate to human
- **Deduplication:** Merge duplicate tickets (same user, same topic within timeframe)

**Slack-specific considerations:**
- Capture message thread ID for context
- Include user, channel, timestamp, message content
- Track which Slack bot thread was created
- Link back to original message (thread reply)

---

## Current State Assessment

**Existing Slack Agent (before audit):**
- ✅ Spawned successfully (Phase 3.2.1 Phase 2)
- ✅ Posted to #ford-approvals successfully
- ✅ Slack API integration functional
- ❓ **No visible ticket system** for message tracking
- ❓ **No message queue** for async processing
- ❓ **No deduplication** (could process same message twice if Slack retries)
- ❓ **No SLA tracking** (no response time guarantees)
- ❓ **No persistence** (tickets lost on agent restart)

**Risk:** Agent may not be handling Slack retries correctly → messages get lost or duplicated

---

## Audit Requirements (from Peder)

1. **Ticket System Setup**
   - Design ticket structure (ID, timestamp, priority, status)
   - Create database schema (Postgres)
   - Implement ticket lifecycle (open → assigned → resolved)

2. **Reliable Communication**
   - Message queue (async processing)
   - Deduplication (handle Slack retries)
   - Persistence (survive agent restart)
   - Response guarantees (<5min standard, <1min urgent)

3. **Slack + iMessage Integration**
   - Both channels produce tickets
   - Both channels receive responses
   - Unified ticket database

4. **User Experience**
   - User can submit message via Slack or iMessage
   - Acknowledgment immediately (e.g., "Got your message, agent #1234")
   - Response within SLA
   - Can see ticket status

---

## Recommended Architecture

### High-Level Flow

```
1. USER (Peder)
   ├─ iMessage: "Why did Sharpe drop?"
   └─ Slack: "Status update needed"
   
2. INGESTION (immediate, <100ms)
   ├─ iMessage responder receives message
   ├─ Slack webhook receives message
   └─ Both create TICKET in Postgres
   
3. ACKNOWLEDGMENT (<3 seconds)
   ├─ iMessage: "Got it! Ticket #5001, assigning to Analyst..."
   └─ Slack: "Thread reply: Ticket #5001 created, processing..."
   
4. QUEUE PROCESSING (async, background)
   ├─ Pull ticket from queue
   ├─ Route by type (analytical → Q&A, research → Research, etc.)
   ├─ Spawn agent session
   └─ Process response
   
5. RESPONSE DELIVERY
   ├─ Agent completes work
   ├─ Update ticket status (RESOLVED)
   ├─ Send response via iMessage/Slack
   └─ Close ticket or escalate
```

### Database Schema (Postgres)

```sql
CREATE TABLE tickets (
    id BIGSERIAL PRIMARY KEY,
    ticket_number INT UNIQUE,  -- User-facing ticket #
    source ENUM ('slack', 'imessage'),
    sender_id VARCHAR(255),
    sender_name VARCHAR(255),
    channel_id VARCHAR(255),  -- Slack channel or iMessage contact
    message_text TEXT,
    priority ENUM ('P0', 'P1', 'P2', 'P3') DEFAULT 'P2',
    status ENUM ('OPEN', 'ASSIGNED', 'IN_PROGRESS', 'RESOLVED', 'CLOSED') DEFAULT 'OPEN',
    classification ENUM ('operational', 'analytical', 'research') DEFAULT 'operational',
    assigned_agent VARCHAR(255),  -- Q&A, Research, MAIN, etc.
    created_at TIMESTAMP DEFAULT NOW(),
    acknowledged_at TIMESTAMP,
    response_sent_at TIMESTAMP,
    resolved_at TIMESTAMP,
    response_text TEXT,
    slack_thread_id VARCHAR(255),
    slack_envelope_id VARCHAR(255),  -- For deduplication
    retries INT DEFAULT 0,
    external_ticket_id VARCHAR(255)  -- For reference
);

CREATE INDEX idx_envelope_id ON tickets(slack_envelope_id);
CREATE INDEX idx_status ON tickets(status);
CREATE INDEX idx_created_at ON tickets(created_at);
```

### Queue Processing Logic

```python
# Pseudo-code
while True:
    # 1. Pull open tickets
    tickets = db.query("SELECT * FROM tickets WHERE status = 'OPEN' ORDER BY priority, created_at LIMIT 10")
    
    for ticket in tickets:
        # 2. Check deduplication
        if db.get_envelope_id(ticket.slack_envelope_id):
            db.update(ticket, status='DUPLICATE')
            continue
        
        # 3. Mark as assigned
        db.update(ticket, status='ASSIGNED', assigned_agent=get_agent(ticket.classification))
        
        # 4. Spawn agent
        agent_response = spawn_agent(
            task=ticket.message_text,
            model=agent_model,
            context=get_context()
        )
        
        # 5. Update ticket with response
        db.update(ticket, 
            status='RESOLVED',
            response_text=agent_response,
            resolved_at=NOW(),
            response_sent_at=NOW()
        )
        
        # 6. Send response back to user
        if ticket.source == 'slack':
            slack_api.post(channel=ticket.channel_id, response=agent_response, thread=ticket.slack_thread_id)
        elif ticket.source == 'imessage':
            imsg.send(to=ticket.sender_id, text=agent_response)
```

### SLA Targets

| Ticket Type | Response SLA | Resolution SLA | Max Retries |
|-------------|-------------|-----------------|------------|
| P0 (Critical) | 5 min | 30 min | 3 |
| P1 (Urgent) | 15 min | 2 hours | 3 |
| P2 (Standard) | 1 hour | 8 hours | 2 |
| P3 (Low) | 4 hours | 24 hours | 1 |

---

## Implementation Phases

### Phase 1: Database & Ticket Schema
- Create `tickets` table in Postgres
- Implement ticket creation from Slack/iMessage
- Test ticket lifecycle

### Phase 2: Queue & Deduplication
- Build message queue processor
- Implement deduplication by `envelope_id`
- Test retry handling

### Phase 3: Agent Integration
- Route tickets to appropriate agent (Q&A, Research, MAIN)
- Spawn agent on ticket processing
- Capture response and update ticket

### Phase 4: Response Delivery
- Send response back to Slack (thread reply)
- Send response back to iMessage
- Test end-to-end flow

### Phase 5: Monitoring & SLA Tracking
- Add SLA tracking (response time, resolution time)
- Dashboard for ticket status
- Alert on missed SLAs
- Escalation rules

---

## Success Criteria

### Functional
- [ ] Tickets created for all Slack/iMessage messages
- [ ] Deduplication prevents duplicate processing
- [ ] Responses delivered within SLA
- [ ] Ticket status visible to user
- [ ] Escalation when SLA breached

### Non-Functional
- [ ] Ticket persistence (survive agent restart)
- [ ] Queue throughput: >100 tickets/hour
- [ ] Response latency P99 <5 min (standard)
- [ ] Zero message loss (<0.01% error rate)
- [ ] Uptime: 99.5% (1 restart per week acceptable)

---

## Next Steps

1. ✅ **RESEARCH COMPLETE** — This document
2. ⏭️ **PLAN** → Create formal contract with implementation timeline
3. ⏭️ **GENERATE** → Implement ticket system (Postgres schema, queue processor, agent integration)
4. ⏭️ **EVALUATE** → Test with live Slack/iMessage messages
5. ⏭️ **DECIDE** → PASS/FAIL verdict
6. ⏭️ **LOG** → Update HEARTBEAT, commit to GitHub

---

**Research prepared by:** Ford (Coordinator Agent)
**Date:** 2026-03-31 18:51 GMT+2
**Status:** Ready for PLAN phase
