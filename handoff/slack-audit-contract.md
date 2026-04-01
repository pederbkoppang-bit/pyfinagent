# Slack Audit Contract: Responsiveness & Ticket System

**Date:** 2026-03-31 18:54 GMT+2  
**Phase:** PLAN  
**Auditor:** Ford (subagent)

## Success Criteria (Research-Backed)

### 🎯 **Primary Goals**

1. **Message Reply Rate ≥99%** 
   - Industry standard: Modern chatbots achieve 99%+ uptime (Microsoft Bot Framework SLA)
   - Measurement: Successful replies / Total messages requiring response
   - Monitoring: Real-time tracking with alerts <95%

2. **Response Time ≤30 seconds**  
   - Slack UX best practice: Users expect <30s for bot responses
   - Measurement: Time from message received to response sent
   - Acceptable: 95th percentile ≤30s

3. **Zero Lost Issues**
   - Every issue reported gets tracked with unique ID
   - Persistent storage with audit trail
   - Auto-escalation after defined SLAs

### 🔧 **Technical Requirements**

1. **Process Reliability**
   - Slack bot process runs continuously
   - Auto-restart on crash within 30 seconds
   - Health monitoring with proactive alerts

2. **Message Handling Coverage**
   - Direct mentions (@Ford)
   - Natural language queries ("What's the status?")
   - Slash commands (existing `/analyze` etc.)
   - Error graceful degradation

---

## Design: Improved Slack Responsiveness

### **Component 1: Process Supervision**

**Approach:** Cron-based monitoring + auto-restart

```bash
# Every 2 minutes: check if slack bot is running
*/2 * * * * /path/to/check_slack_bot.sh >> /var/log/slack_monitor.log 2>&1
```

**Implementation:**
- `scripts/check_slack_bot.sh` - checks process, restarts if dead
- Logs to dedicated file for debugging
- Slack notification on restart events
- Max restart attempts: 5/hour (prevent flapping)

### **Component 2: Enhanced Message Handling**

**Current:** Only responds to "status" and slash commands
**Proposed:** AI-powered natural language processing

**Message Router Logic:**
```python
@app.event("message")
async def handle_all_messages(message, say):
    """Route all messages to appropriate handlers"""
    
    channel = message.get("channel")
    text = message.get("text", "").lower()
    user = message.get("user")
    
    # Skip bot messages
    if message.get("bot_id"):
        return
        
    # Channel filtering  
    if channel not in MONITORED_CHANNELS:
        return
    
    # Mention detection
    if is_ford_mentioned(text):
        await handle_mention(message, say)
    
    # Intent classification
    elif contains_query_intent(text):
        await handle_query(message, say)
    
    # Ticket keywords
    elif contains_issue_keywords(text):
        await handle_potential_ticket(message, say)
```

**AI Integration:**
- Use lightweight LLM (Haiku 4.5) for intent classification
- Classify: `query`, `ticket`, `casual`, `ignore`
- Cache frequent patterns to reduce API calls

### **Component 3: Comprehensive Monitoring**

**Metrics Dashboard:**
- Messages received/processed/responded (per hour)
- Response time distribution  
- Bot uptime percentage
- Error rate trends
- Active ticket count

**Alerting Thresholds:**
- Response rate <95% → Immediate Slack alert
- Average response time >45s → Warning
- Bot process down >60s → Critical alert
- Unhandled exception → Log + notify

---

## Design: Ticket System

### **Architecture Overview**

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐
│   Slack     │───▶│  Ticket      │───▶│  Storage    │
│  Messages   │    │  Engine      │    │  (SQLite)   │
└─────────────┘    └──────────────┘    └─────────────┘
                           │
                           ▼
                   ┌──────────────┐
                   │ Escalation   │
                   │   Logic      │
                   └──────────────┘
```

### **Data Model**

```sql
CREATE TABLE tickets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticket_id TEXT UNIQUE,           -- T-2026-001, T-2026-002...
    title TEXT NOT NULL,
    description TEXT,
    status TEXT DEFAULT 'open',     -- open, in_progress, closed
    priority TEXT DEFAULT 'medium', -- low, medium, high, critical
    reporter_user_id TEXT,
    reporter_name TEXT,
    channel_id TEXT,
    message_ts TEXT,                -- Original Slack message timestamp
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    due_date TIMESTAMP,
    tags TEXT,                      -- JSON array: ["bug", "backend"]
    metadata TEXT                   -- JSON: {"slack_thread": "...", "related_commits": []}
);

CREATE TABLE ticket_updates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticket_id TEXT REFERENCES tickets(ticket_id),
    update_type TEXT,               -- status_change, comment, assignment
    old_value TEXT,
    new_value TEXT,
    comment TEXT,
    user_id TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### **Ticket Lifecycle**

1. **Creation Triggers**
   - Keywords: "bug", "issue", "broken", "not working", "problem"
   - Direct commands: `/ticket [description]`
   - AI-detected problem reports

2. **Auto-Classification**
   ```python
   def classify_ticket(message_text: str) -> dict:
       """Use LLM to extract ticket metadata"""
       prompt = f"""
       Classify this issue report:
       "{message_text}"
       
       Return JSON:
       {{
         "title": "Brief summary",
         "priority": "low|medium|high|critical", 
         "category": "bug|feature|question|other",
         "urgency": "immediate|same_day|this_week|backlog"
       }}
       """
       # Call Haiku 4.5 for classification
   ```

3. **Status Workflow**
   - `open` → `in_progress` → `closed`
   - Status changes logged with timestamp
   - Slack notifications on major updates

4. **SLA & Escalation**
   ```python
   ESCALATION_RULES = {
       'critical': timedelta(hours=1),    # Page immediately
       'high':     timedelta(hours=4),    # Same business day  
       'medium':   timedelta(days=2),     # This week
       'low':      timedelta(days=7)      # Next sprint
   }
   ```

### **User Interface**

**Slack Commands:**
- `/ticket create [description]` - Create new ticket
- `/ticket list [status]` - Show open tickets  
- `/ticket show T-2026-001` - Show ticket details
- `/ticket close T-2026-001 [reason]` - Close ticket
- `/ticket status` - Show ticket summary

**Example Interaction:**
```
Peder: "The backend keeps crashing when I run backtests"

Ford: 🎫 **Ticket Created: T-2026-015**
📋 **Title:** Backend crashes during backtest execution
⚠️ **Priority:** High  
🕐 **SLA:** Response within 4 hours
🔗 **Track:** Reply with `/ticket show T-2026-015`

Would you like me to investigate the logs immediately?
```

**Ticket Dashboard (Web UI Extension):**
- Add `/tickets` route to existing frontend
- Show open tickets with status/priority
- Click to view details/comments
- Integration with existing auth system

---

## Implementation Plan

### **Phase 1: Process Reliability (Day 1)**
1. Create `scripts/check_slack_bot.sh` monitoring script
2. Set up cron job for auto-restart  
3. Add startup logging and health checks
4. Test crash recovery scenarios

### **Phase 2: Enhanced Responsiveness (Day 2-3)**
1. Implement general message handler for mentions
2. Add AI intent classification (Haiku 4.5)
3. Expand response patterns beyond "status"
4. Add comprehensive error handling

### **Phase 3: Basic Ticket System (Day 4-5)**
1. Create SQLite database schema
2. Implement ticket creation from keywords
3. Add `/ticket` slash commands
4. Basic lifecycle management (open → close)

### **Phase 4: Advanced Features (Day 6-7)**
1. SLA tracking and escalation logic
2. Web dashboard integration
3. Ticket analytics and reporting
4. Performance monitoring

---

## Testing Strategy

### **Unit Tests**
- Message classification accuracy
- Ticket CRUD operations
- SLA calculation logic
- Error handling paths

### **Integration Tests**  
- End-to-end Slack message → ticket creation
- Process restart recovery
- Database consistency
- API reliability

### **Load Testing**
- Message processing under high volume
- Concurrent ticket operations
- Memory/CPU usage over time

---

## Success Metrics

### **Responsiveness**
- [ ] Bot uptime ≥99.5%
- [ ] Message response rate ≥99%  
- [ ] Response time P95 ≤30 seconds
- [ ] Zero messages lost to crashes

### **Ticket System**
- [ ] 100% issue capture (no reports lost)
- [ ] Ticket creation time ≤10 seconds
- [ ] SLA tracking accuracy ≥95%
- [ ] User satisfaction: Can find/track issues easily

### **Operational**
- [ ] Auto-recovery from crashes ≤30 seconds
- [ ] Comprehensive logging and monitoring
- [ ] Integration with existing systems
- [ ] No manual intervention required for normal operation

---

## Risk Mitigation

**Risk:** Bot process still crashes frequently  
**Mitigation:** Multiple restart attempts, fallback to iMessage notifications

**Risk:** AI classification costs too high  
**Mitigation:** Keyword-based fallback, request batching, pattern caching

**Risk:** Ticket system overwhelms users  
**Mitigation:** Smart filtering, auto-close for resolved issues, digest summaries

**Risk:** Database corruption/loss  
**Mitigation:** Regular backups, transaction safety, migration scripts

---

## Definition of Done

- ✅ Slack bot runs continuously with auto-restart
- ✅ Responds to mentions and natural language queries  
- ✅ Creates tickets automatically from issue reports
- ✅ Provides `/ticket` commands for management
- ✅ Tracks SLA and escalates overdue items
- ✅ Integrates with existing dashboard
- ✅ Comprehensive monitoring and alerting
- ✅ Documentation and runbook complete

**Acceptance Test:** Peder reports an issue in Slack → Ford responds within 30 seconds → Issue tracked as ticket with ID → Status updates provided → Issue resolved and ticket closed → Full audit trail available.