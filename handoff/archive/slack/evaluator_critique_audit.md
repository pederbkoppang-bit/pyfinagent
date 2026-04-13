# Evaluator Critique: Slack Agent Audit
**Date:** 2026-04-01 09:40 CEST  
**Evaluator:** Independent Subagent  
**Verdict:** ⚠️ CONDITIONAL

---

## Executive Summary

The ticket system backend (database, ingestion, queue processor) is **well-built and functional**. The Slack bot process is running. However, there is a **critical integration gap**: the live Slack bot (`commands.py`) does NOT use the ticket system at all. Messages received via Socket Mode are acknowledged with a generic reply but **never persisted as tickets**. The ticket system exists as a parallel, disconnected layer.

---

## Checklist Results

### 1. Bot Process Running — ✅ PASS
- **PID 28735** running: `python -m backend.slack_bot.app`
- Socket Mode via `AsyncSocketModeHandler` (no public URL needed)
- APScheduler started for morning digest

**Evidence:**
```
ford  28735  0.4%  python -m backend.slack_bot.app
```

### 2. Message Ingestion — ❌ FAIL (Critical)
- The Slack bot's `commands.py` catch-all handler (`@app.message("")`) responds to messages in `#ford-approvals` but **does not call `ticket_ingestion.ingest_slack_message()`**
- Zero imports from `backend.services.ticket_ingestion` or `backend.db.tickets_db` in the slack_bot package
- The ticket ingestion service works perfectly when called directly (tested), but it's simply **never wired in**
- The `slack_ticket_webhook.py` handler exists but is a separate FastAPI route — not integrated with the Socket Mode bot

**Evidence:**
```bash
$ grep -r "ticket_ingestion\|ingest_slack\|get_ingestion" backend/slack_bot/
# (no results — zero integration)
```

The only tickets in the DB were created by the test script and this evaluator's manual tests — **none from actual Slack messages**.

### 3. Response Delivery — ⚠️ PARTIAL PASS
- The bot **does** reply to messages in `#ford-approvals` with `"✅ Message received and logged. Timestamp: {ts}"`
- The word "logged" is misleading — nothing is actually logged to the ticket DB
- Slash commands (`/analyze`, `/portfolio`, `/report`) work correctly
- Reaction handlers (✅/❌ for push approval) are functional

### 4. Deduplication — ✅ PASS (DB layer only)
- `is_duplicate_envelope()` correctly detects duplicate envelope IDs
- Tested: creating a ticket with `envelope_id='eval_test_002'`, then re-submitting → returns `None`
- **Caveat**: This only works IF ingestion is actually called (see #2)

**Evidence:**
```python
INGEST: ticket_id=7
DEDUP: ticket_id=None (should be None)  # ✅ Correct
```

### 5. SLA Tracking — ✅ PASS (DB layer only)
- Priority → SLA mapping is correct:
  - P0: 5min response / 30min resolution
  - P1: 15min response / 2hr resolution
  - P2: 1hr response / 8hr resolution
  - P3: 4hr response / 24hr resolution
- `response_sla_seconds` and `resolution_sla_seconds` stored per ticket
- `get_sla_breaches()` query works (detected 2 breaches in test data)

**Evidence:**
```
TEST6 SLA: response_sla=3600s, resolution_sla=28800s  # P2 correct
```

### 6. Database Schema — ✅ PASS
All required columns present with proper constraints:

| Column | Present | Type/Constraint |
|--------|---------|-----------------|
| id | ✅ | INTEGER PRIMARY KEY AUTOINCREMENT |
| ticket_number | ✅ | INTEGER UNIQUE NOT NULL |
| source | ✅ | CHECK(IN 'slack','imessage') |
| sender_id | ✅ | TEXT NOT NULL |
| sender_name | ✅ | TEXT |
| channel_id | ✅ | TEXT |
| message_text | ✅ | TEXT NOT NULL |
| priority | ✅ | CHECK(IN P0-P3), DEFAULT P2 |
| status | ✅ | CHECK(IN 6 statuses), DEFAULT OPEN |
| classification | ✅ | CHECK(IN 3 types) |
| created_at | ✅ | TIMESTAMP DEFAULT CURRENT_TIMESTAMP |
| acknowledged_at | ✅ | TIMESTAMP |
| assigned_at | ✅ | TIMESTAMP |
| in_progress_at | ✅ | TIMESTAMP |
| resolved_at | ✅ | TIMESTAMP |
| closed_at | ✅ | TIMESTAMP |
| response_text | ✅ | TEXT |
| slack_envelope_id | ✅ | TEXT (indexed for dedup) |
| slack_event_ts | ✅ | TEXT |
| response_sla_seconds | ✅ | INTEGER |
| resolution_sla_seconds | ✅ | INTEGER |
| retries | ✅ | INTEGER DEFAULT 0 |
| error_message | ✅ | TEXT |
| metadata | ✅ | TEXT (JSON) |

6 indexes created for performance. Ticket counter table for sequential numbering (starts at 5000).

### 7. Ticket Lifecycle — ✅ PASS
Full lifecycle tested successfully: OPEN → ASSIGNED → IN_PROGRESS → RESOLVED

**Evidence:**
```
TEST4 ASSIGN: status=ASSIGNED, agent=MAIN
TEST5 RESOLVE: status=RESOLVED, response=Test response
```

Queue processor batch processing works:
```
BATCH PROCESSED: 5 tickets
AFTER BATCH: RESOLVED: 6, DUPLICATE: 1
```

### 8. Error Handling — ⚠️ CONDITIONAL
- Empty messages are accepted (created ticket with empty `message_text`) — should be rejected
- Missing event fields (empty `{}` event dict) still creates a ticket — no input validation
- Very long messages (10,000 chars) accepted without truncation — acceptable but worth noting
- Queue processor correctly handles failures: sets status back to OPEN with error_message

**Evidence:**
```
EMPTY TEXT: ticket_id=4 (created - no validation)
MISSING FIELDS: ticket_id=5 (created - no validation)
LONG MSG: ticket_id=6 (created - no size limit)
```

---

## Component Inventory (9 files, not 24)

| File | Status | Purpose |
|------|--------|---------|
| `backend/db/tickets_db.py` | ✅ Working | SQLite ticket database ORM |
| `backend/services/ticket_ingestion.py` | ✅ Working | Message → ticket conversion, classification, priority |
| `backend/services/ticket_queue_processor.py` | ✅ Working | Queue processing with agent routing (simulated agents) |
| `backend/services/slack_ticket_webhook.py` | ✅ Working | FastAPI webhook handler (NOT used by Socket Mode bot) |
| `backend/slack_bot/app.py` | ✅ Running | Socket Mode entry point |
| `backend/slack_bot/commands.py` | ⚠️ **Gap** | Slash commands + message handler — **no ticket integration** |
| `backend/slack_bot/formatters.py` | ✅ Working | Block Kit message formatting |
| `backend/slack_bot/scheduler.py` | ✅ Working | APScheduler for morning digest |
| `test_tickets_db.py` | ✅ Working | Phase 1 validation test |

---

## Critical Fix Required

### The Integration Gap

`commands.py` `handle_any_message()` must be modified to:

1. Import and call `get_ingestion_service().ingest_slack_message()`
2. Use the returned ticket ID to send a proper acknowledgment (with ticket number and priority)
3. Start or integrate with `ticket_queue_processor` for async processing

**Approximate fix** (add to `handle_any_message` in `commands.py`):
```python
from backend.services.ticket_ingestion import get_ingestion_service

# In handle_any_message:
ingestion = get_ingestion_service()
ticket_id = ingestion.ingest_slack_message(
    event=message,
    sender_id=message.get("user"),
    channel_id=channel
)
if ticket_id:
    ack = ingestion.acknowledge_ticket_immediately(ticket_id)
    await say(ack["message"])
```

### Secondary Fixes

1. **Input validation**: Reject empty `message_text` in `create_ticket()`
2. **Queue processor startup**: The processor's `start_processing_loop()` is never called anywhere — needs to be started as a background task in the bot or backend
3. **Agent responses are simulated**: `_simulate_agent_response()` returns canned text. Production needs real agent spawning.
4. **`_get_user_name()` is hardcoded**: Only maps one user ID. Should call Slack Users API.

---

## Verdict: ⚠️ CONDITIONAL

**The backend ticket system is solid** — schema, lifecycle, dedup, SLA, classification, and priority all work correctly. But the live Slack bot doesn't use it. This is a wiring problem, not a design problem.

### Before PASS:
1. **[MUST]** Wire `ticket_ingestion` into `commands.py` `handle_any_message()`
2. **[MUST]** Start the queue processor as a background task
3. **[SHOULD]** Add empty-message validation
4. **[SHOULD]** Replace simulated agent responses with real agent spawning

Estimated effort: ~30 minutes for items 1-2 (the critical path).
