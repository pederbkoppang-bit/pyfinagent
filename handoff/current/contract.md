# Phase 2.11: Resilient Background Job Monitoring — Contract

## Hypothesis
Implementing production-grade resilience patterns will prevent session compaction from killing long-running processes and ensure pyfinAgent's autonomous systems (harness, paper trading, Slack bot) survive infrastructure disruptions and maintain operational continuity.

## Success Criteria (Research-Backed)

### Primary Criteria (Based on 2.11 Deep Research)

1. **OpenClaw Session Compaction Survival:**
   - All critical state persisted to `MEMORY.md` and daily memory files (NOT conversation context)
   - Pre-compaction memory flush implemented with `NO_REPLY` pattern
   - Main session remains responsive during long-running operations (no blocking poll loops)
   - Context budgeting: 20k+ token reserve for housekeeping operations
   - ✅ Evidence: Harness/backtest can survive session compaction without losing progress

2. **Background Job Resilience Architecture:**
   - **Heartbeat Monitoring**: All long-running jobs ping healthchecks.io or equivalent on completion
   - **Process Isolation**: Long-running work launched with `nohup` + detached monitoring
   - **Cron Survival**: Critical monitoring (health checks, progress reporting) via cron jobs
   - **Status Persistence**: Job status tracked in Redis/files, not in-memory session state
   - ✅ Evidence: Paper trading + harness continue through gateway restarts, session compaction

3. **Slack Socket Mode Resilience:**
   - **Connection Recovery**: Automatic reconnection with exponential backoff
   - **Event Acknowledgment**: All events `ack()`-ed within 3 seconds (no Slack retries)
   - **Async Processing**: Non-blocking event handlers using `AsyncApp` pattern
   - **Buffer During Disconnects**: Queue events during connection loss, replay on recovery
   - ✅ Evidence: Slack bot maintains responsiveness during high load + network issues

4. **Production Monitoring Patterns:**
   - **Circuit Breakers**: Protect external API calls (BigQuery, Vertex AI) with `pybreaker`
   - **Health Check Layers**: Process health + application health + business metrics
   - **Distributed Lock Pattern**: Prevent duplicate job execution using Redis locks
   - **Graceful Cancellation**: Long-running tasks implement cancellation points
   - ✅ Evidence: System remains available during external service failures

### Secondary Criteria
- [ ] **Memory Management**: Daily memory files rotated, `MEMORY.md` stays under 50KB
- [ ] **Observability**: Prometheus metrics for job duration, failure rates, queue depth
- [ ] **Resource Limits**: Background jobs have CPU/memory limits to prevent resource exhaustion
- [ ] **Incident Response**: Automated escalation path (Slack → iMessage → memory/incidents.md)

## Fail Conditions (Research-Backed Anti-Patterns)
1. **Session Compaction Kills Operations**: Long-running processes die when context compacted
2. **Event Loop Blocking**: Synchronous operations in async code cause hangs/timeouts
3. **Polling Loops in Main Session**: Active monitoring blocks session, dies on compaction
4. **In-Memory State Loss**: Critical status stored in conversation context, lost on reset
5. **Cascade Failures**: Single external service failure brings down entire system
6. **Resource Exhaustion**: Background jobs monopolize CPU/memory, crash host

## Timeline  
- **Research:** 3 hours (COMPLETE ✅) — Deep research into resilience patterns
- **Generation:** 6-8 hours — Implement resilience architecture, monitoring, background job patterns
- **Evaluation:** 2-3 hours — Stress testing, session compaction simulation, failure injection
- **Total:** ~11-14 hours (can span 2-3 calendar days with integration testing)

## Budget Impact
- **External Monitoring:** $5-10/month (healthchecks.io Pro for team features)
- **Redis Instance:** $0 (use existing local Redis for distributed locks)
- **No API Cost Impact:** Resilience patterns reduce external API failures, may save money
- **Infrastructure:** No additional cost (patterns run on existing backend/gateway)

## References (Research Citations)
- [OpenClaw Session Management Deep Dive](https://docs.openclaw.ai/reference/session-management-compaction)
- [Slack Bolt-Python Socket Mode](https://docs.slack.dev/tools/bolt-python/concepts/socket-mode/)
- [ArXiv: Fault-tolerance in Distributed Systems](https://arxiv.org/abs/2106.08545)
- [Healthchecks.io Heartbeat Monitoring](https://healthchecks.io/docs/monitoring_cron_jobs/)
- [Circuit Breaker Pattern Best Practices](https://pypi.org/project/pybreaker/)
- **Microsoft Azure Background Jobs Guide** — Idempotency, distributed locks, resource limits
- **Production Python Async Monitoring Patterns 2024** — asyncio task management, monitoring integration
