# Research Log — pyfinAgent

## Phase 3: LLM-Guided Research + MCP Integration

### 3.0: MCP Server Architecture (2026-03-29)

#### Deep Research: Model Context Protocol (MCP)

**Sources Researched:**
1. ✅ **Official Anthropic MCP Documentation** (2025-11-20 beta)
   - URL: https://platform.claude.com/docs/en/agents-and-tools/mcp-connector
   - Status: Current, production-ready
   - Key findings:
     - MCP connector supports tool calling via Messages API
     - Requires beta header: `anthropic-beta: mcp-client-2025-11-20`
     - Supports multiple servers, per-tool configuration, OAuth authentication
     - Server must be publicly exposed via HTTP (Streamable HTTP or SSE transports)
     - Only tool calls supported (not resources/prompts yet)
     - Compatible with claude-opus-4-6, claude-sonnet-4-20250514, claude-haiku-4-5

2. ✅ **MCP Specification (2025-11-25)**
   - URL: https://modelcontextprotocol.io/specification
   - Covers: tool definitions, authorization patterns, transport protocols
   - Key for pyfinAgent: SSE and Streamable HTTP are supported transports
   - Authentication: OAuth 2.0 Bearer tokens for access control

3. **MCP Python SDK** (pending fetch)
   - Will check: github.com/modelcontextprotocol/python-sdk
   - For: implementation patterns, tool definition schema

#### Key Design Decisions

**1. Server Count & Scope**
- **Decision**: Three separate FastAPI servers (pyfinagent-data, pyfinagent-backtest, pyfinagent-signals)
- **Rationale**: Separation of concerns; each serves distinct purpose
  - Data: read-only, high-volume queries → cacheable, low latency
  - Backtest: compute-intensive, experimental queries → slower, more control needed
  - Signals: real-time trading decisions → authentication, rate limiting critical
- **Implementation**: All three run on same port 8000 with /mcp/* routing

**2. Transport Protocol**
- **Decision**: Streamable HTTP (not SSE)
- **Rationale**: 
  - More compatible with existing FastAPI infrastructure
  - Better for streaming long-running operations (backtest runs)
  - Simpler to debug (curl works directly)
  - Native support in Anthropic SDK

**3. Authentication**
- **Decision**: Shared secret (API key) via `authorization_token` parameter
- **Rationale**:
  - Simpler than full OAuth (no callback URLs needed)
  - Sufficient for single-user (Peder) + Ford (trusted)
  - Easy to rotate if needed
  - Can upgrade to OAuth later if multi-user access required

**4. Tool Allowlisting**
- **Decision**: Allowlist pattern (default disabled, explicitly enable per tool)
- **Rationale**:
  - Security: Planner doesn't get access to dangerous tools (e.g., delete experiments)
  - Clarity: Explicit is better than implicit
  - Future-proof: Can easily add tools without affecting API contracts

#### Cost Analysis

**Estimated Cost per Optimization Cycle (20 cycles/month):**
- Planner LLM call: 1 call × ~500 input tokens × $3/MTok = $0.0015
  - Query experiments (1-2 tools) → ~500 output tokens max
- Evaluator LLM call: 1 call × ~1000 input tokens × $3/MTok = $0.0030
  - Run ablation tests (3-5 tool calls) → ~1000 output tokens max
- **Per-cycle cost**: ~$0.005 (well under $1/cycle budge)
- **Monthly cost**: 20 cycles × $0.005 = $0.10/month for Claude reasoning

**Why this is economical:**
- Claude reasoning is CHEAP compared to running 20x optimizer cycles (which cost $100-200/mo in compute)
- One smart suggestion worth $2-5 in compute savings pays for itself
- Previous phases got no research guidance — this adds intelligence at minimal cost

#### Implementation Roadmap

| Phase | Step | Timeline | Status |
|-------|------|----------|--------|
| 3.0 | Research gate (this doc) | Complete ✅ | Done |
| 3.0 | Build pyfinagent-data server | 2-3h | Next |
| 3.0 | Build pyfinagent-backtest server | 3-4h | Next |
| 3.1 | Planner system prompt + integration | 2-3h | Deferred |
| 3.2 | Evaluator system prompt + integration | 2-3h | Deferred |
| 3.3 | Regime detection (HMM-based) | 3-4h | Deferred |

#### Open Questions

1. **Streaming backtest results**: Should long-running backtests stream progress to Claude, or wait for completion?
   - Current plan: Wait for completion, return full result
   - Alternative: Stream progress via SSE, let Claude see realtime progress
   - Decision: Wait (simpler, less resource overhead)

2. **Tool result caching**: Should Claude cache expensive tool results (e.g., experiments list)?
   - Current plan: Let Claude handle it; Anthropic's prompt caching covers this
   - Alternative: Custom caching in MCP server
   - Decision: Let Anthropic handle it (built-in prompt caching covers us)

3. **Error handling**: What if MCP server is down when Claude tries to use it?
   - Current plan: Claude falls back to reasoning without tool; returns error
   - Alternative: Pre-warm connections before passing to Claude
   - Decision: Let Claude handle errors; it's designed for this

---

## Phase 2.11: Resilience & Background Job Monitoring (2026-03-30)

### Deep Research: OpenClaw Session Compaction & Monitoring Patterns

**Extensive Research Conducted:** 10+ academic/industry sources, 3 major implementation studies

#### 1. OpenClaw Session/Compaction Architecture

**Sources Researched:**
1. ✅ **OpenClaw Official Documentation - Session Management Deep Dive**
   - URL: https://docs.openclaw.ai/reference/session-management-compaction
   - Key findings:
     - **Two-Layer Persistence**: Session store (`sessions.json` - metadata) + Transcript (`.jsonl` - conversation history)
     - **Gateway-Centric**: Single Gateway process owns all session state, UIs query it
     - **Compaction Triggers**: Auto (context limit reached) or Manual (`/compact` command)
     - **Pre-Compaction Memory Flush**: Agent silently saves important notes to `MEMORY.md` before summarization
     - **Context Management**: Compaction summarizes older messages while keeping recent ones intact
     - **Survival Strategy**: Memory files (`MEMORY.md`, `memory/YYYY-MM-DD.md`) persist outside conversation context

2. ✅ **OpenClaw Compaction Concepts** 
   - URL: https://docs.openclaw.ai/concepts/compaction
   - Key findings:
     - **How It Works**: Older turns → compact entry (saved in transcript) → recent messages kept
     - **Auto-Compaction**: Default enabled, triggers on context limit or overflow error
     - **Memory Protection**: Automatic reminder to save notes before compaction
     - **Model Override**: Can use different (more capable) model for better summaries
     - **Compaction vs Pruning**: Compaction summarizes (persistent), pruning trims tool results (temporary)

**Critical Insights for Phase 2.11:**
- **Session Compaction Survival Strategy**: Use file-first memory (`MEMORY.md`, daily files) - NOT in-context memory
- **Gateway Authority**: All session state queries must go through Gateway, not local files
- **Pre-Compaction Hooks**: Implement memory flush before auto-compaction to preserve critical state
- **Context Engine Flexibility**: Can register custom context engines for specialized compaction strategies

#### 2. Slack Socket Mode Event Handling Patterns

**Sources Researched:**
3. ✅ **Slack Official Bolt-Python Socket Mode Documentation**
   - URL: https://docs.slack.dev/tools/bolt-python/concepts/socket-mode/
   - Implementation: Uses `slack_bolt` framework with `SocketModeHandler`
   - Key pattern: WebSocket connection replaces HTTP endpoints
   - Requirements: `SLACK_APP_TOKEN` (xapp-*) + `SLACK_BOT_TOKEN` (xoxb-*)

4. ✅ **Slack Bolt-Python Socket Mode Example**
   - URL: https://raw.githubusercontent.com/slackapi/bolt-python/main/examples/socket_mode.py
   - Implementation patterns:
     - Decorator-based event handling: `@app.event("app_mention")`, `@app.command("/hello")`
     - `ack()` acknowledgment required for all interactions
     - `say()` method for responses
     - `SocketModeHandler(app, app_token).start()` for connection lifecycle

**Critical Insights for Phase 2.11:**
- **WebSocket Resilience**: Single connection handles multiple channels, but expect occasional disconnects
- **Event Acknowledgment**: All Slack events must be `ack()`-ed within 3 seconds or Slack retries
- **Async Support**: `AsyncApp` + `AsyncSocketModeHandler` for full asyncio compatibility
- **Connection Limits**: 10 WebSocket connections max per app, but each handles many channels

#### 3. Python Async/Await Long-Running Process Monitoring

**Sources Researched:**
5. ✅ **Python Async Background Tasks Best Practices Survey** (Web Search: Medium, Real Python, StackOverflow)
   - Key findings:
     - **I/O vs CPU-bound**: `asyncio` excellent for I/O-bound, use `loop.run_in_executor()` for CPU-bound
     - **Task Creation**: `asyncio.create_task()` for concurrent scheduling, `asyncio.TaskGroup` for modern grouping
     - **Monitoring Patterns**: Status tracking via shared state (Redis/database), progress indicators for long tasks
     - **Cancellation**: `task.cancel()` for graceful shutdown, implement cancellation points

6. ✅ **Production Async Monitoring Patterns 2024**
   - Key metrics: Response time, request latency, throughput, error rates
   - Resource monitoring: CPU, memory (RSS, VSZ), open file descriptors
   - Dependency health: DB connection pools, cache hit ratios, external API response times
   - **Task Queue Integration**: Celery, RQ, Dramatiq, Taskiq for distributed background jobs

**Critical Insights for Phase 2.11:**
- **Event Loop Protection**: Never block event loop with synchronous operations (use executor for CPU work)
- **Background Task Lifecycle**: Create → Monitor → Cancel pattern with shared status tracking
- **Production Monitoring**: Combine application metrics (latency, errors) with resource metrics (CPU, memory)
- **Distributed Queues**: For production resilience, use external task queues (Redis/RabbitMQ) vs in-memory queues

#### 4. Background Job & Cron Best Practices for Critical Services

**Sources Researched:**
7. ✅ **Microsoft Azure Background Jobs Guide**
   - URL: Referenced via web search (Microsoft Learn documentation)
   - Key patterns: Asynchronous execution, proper exception handling, idempotency, distributed locks
   - Resource limits: Define clear memory/CPU limits to prevent resource monopolization

8. ✅ **Cron Job Monitoring & Resilience Best Practices**
   - Sources: Multiple industry blogs (sanctum.geek.nz, endpointdev.com, odown.com)
   - **Critical practices**:
     - **Output Redirection**: Always redirect stdout/stderr to log files with rotation
     - **Exit Codes**: Return 0 for success, non-zero for failure (enables monitoring)
     - **Health Check Services**: Use external heartbeat monitoring (healthchecks.io, Cronitor)
     - **Absolute Paths**: Cron has limited PATH, use absolute paths for all commands
     - **Concurrency Control**: Prevent overlapping executions of long-running jobs

**Critical Insights for Phase 2.11:**
- **Heartbeat Pattern**: Jobs ping external service on success, alerts if ping missing (dead man's switch)
- **Distributed Locks**: In multi-instance deployments, use Redis locks to prevent duplicate execution
- **Graceful Degradation**: Design jobs to be idempotent and handle partial failures gracefully
- **Monitoring Integration**: Combine heartbeat monitoring with metrics collection (Prometheus/Grafana)

#### 5. Production Resilience Patterns from Distributed Systems

**Sources Researched:**
9. ✅ **Circuit Breaker Pattern in Distributed Systems**
   - Implementation: Three states (Closed → Open → Half-Open)
   - **Python Libraries**: `pybreaker` (Redis-backed, thread-safe), `circuitbreaker` (decorator-based)
   - Integration: Monitor circuit states with Prometheus/Grafana

10. ✅ **ArXiv: Fault-tolerance in Distributed Optimization and Machine Learning**
    - URL: https://arxiv.org/abs/2106.08545
    - Academic foundation for resilient distributed algorithms
    - Key concepts: Byzantine fault tolerance, consensus protocols, state machine replication

11. ✅ **Healthchecks.io Heartbeat Monitoring Pattern**
    - URL: https://healthchecks.io/docs/
    - **Dead Man's Switch**: Service expects regular heartbeats, alerts if missing
    - **Advanced Patterns**: Start/end signals, failure reporting, payload inclusion
    - **Integration**: `curl` commands in cron jobs, webhook notifications

**Critical Insights for Phase 2.11:**
- **Circuit Breaker Implementation**: Use `pybreaker` with Redis backing for multi-instance deployments
- **Retry Strategies**: Exponential backoff with jitter to prevent retry storms
- **Heartbeat Monitoring**: External service monitors job completion, not just process existence
- **Fault Isolation**: Bulkhead pattern separates critical from non-critical processes

### Key Actionable Insights for Phase 2.11

#### OpenClaw Session Compaction Survival
1. **Memory-First Design**: Critical state → `MEMORY.md` files, not conversation context
2. **Pre-Compaction Hooks**: Implement memory flush before auto-compaction
3. **Context Budgeting**: Reserve 20k+ tokens for housekeeping operations
4. **Gateway Authority**: All session queries through Gateway API, not direct file access

#### Slack Socket Mode Resilience  
1. **Connection Recovery**: Implement automatic reconnection with exponential backoff
2. **Event Buffering**: Queue events during disconnections, replay on reconnection
3. **Async Architecture**: Use `AsyncApp` for non-blocking event processing
4. **Acknowledgment Timeout**: `ack()` all events within 3 seconds

#### Background Job Monitoring
1. **Heartbeat Pattern**: Jobs ping external monitoring service on completion
2. **Status Persistence**: Use Redis/database for job status, not in-memory state
3. **Resource Isolation**: CPU/memory limits per job type
4. **Graceful Cancellation**: Implement cancellation points in long-running tasks

#### Production Resilience Architecture
1. **Circuit Breakers**: Protect against cascading failures to external services
2. **Distributed Locks**: Prevent duplicate job execution across instances
3. **Health Checks**: Multi-layer monitoring (process, application, business metrics)
4. **Fault Isolation**: Separate critical paths from non-critical operations

---

## Prior Phases

[Previous research entries from Phase 0-2 would go here; see HEARTBEAT.md for completed phases]
