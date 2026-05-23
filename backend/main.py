"""
PyFinAgent Backend — FastAPI application entry point.
"""

import asyncio
import importlib.util
import io
import logging
import json
import sys
import time
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware

from backend.api.agent_map import router as agent_map_router
from backend.api.analysis import router as analysis_router
from backend.api.auth import get_current_user
from backend.api.backtest import router as backtest_router
from backend.api.charts import router as charts_router
from backend.api.investigate import router as investigate_router
from backend.api.cron_dashboard_api import (
    router as cron_dashboard_router,
    register_scheduler as _register_cron_scheduler,
)
from backend.api.paper_trading import router as paper_trading_router, init_scheduler
from backend.api.performance_api import router as performance_router
from backend.api.portfolio import router as portfolio_router
from backend.api.reports import router as reports_router
from backend.api.settings_api import router as settings_router
from backend.api.signals import router as signals_router
from backend.api.skills import router as skills_router
from backend.api.mas_events import router as mas_events_router
from backend.config.settings import get_settings
from backend.services.perf_tracker import get_perf_tracker


# ── Logging ─────────────────────────────────────────────────────────

class CompactFormatter(logging.Formatter):
    """One-line colored log format for terminals. Falls back to plain for non-TTY."""
    COLORS = {"DEBUG": "\033[90m", "INFO": "\033[36m", "WARNING": "\033[33m", "ERROR": "\033[31m", "CRITICAL": "\033[1;31m"}
    RESET = "\033[0m"

    def format(self, record):
        ts = self.formatTime(record, "%H:%M:%S")
        lvl = record.levelname[0]  # D/I/W/E/C
        color = self.COLORS.get(record.levelname, "")
        reset = self.RESET if color else ""
        msg = record.getMessage()
        base = f"{color}{ts} {lvl} [{record.module}]{reset} {msg}"
        if record.exc_info:
            base += "\n" + self.formatException(record.exc_info)
        return base


class JsonFormatter(logging.Formatter):
    """Structured JSON logs for production / Cloud Logging."""
    def format(self, record):
        log_record = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "module": record.module,
            "message": record.getMessage(),
        }
        if record.exc_info:
            log_record["stack_trace"] = self.formatException(record.exc_info)
        return json.dumps(log_record)


class QuietAccessFilter(logging.Filter):
    """Suppress uvicorn access-log lines for high-frequency polling endpoints."""
    _QUIET_PREFIXES = ("/api/backtest/status", "/api/optimizer/status", "/api/health",
                       "/api/backtest/ingestion/status", "/api/paper-trading/status",
                       "/api/perf-optimizer/status")

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return not any(p in msg for p in self._QUIET_PREFIXES)


def setup_logging():
    settings = get_settings()
    level = getattr(logging, settings.log_level.upper(), logging.INFO)

    root = logging.getLogger()
    root.setLevel(level)
    if root.hasHandlers():
        root.handlers.clear()

    stream = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    handler = logging.StreamHandler(stream)

    # Use compact format for local dev, JSON for production
    if settings.debug:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(CompactFormatter())

    root.addHandler(handler)

    # Prevent uvicorn from adding its own cp1252 handlers on Windows
    # and suppress polling endpoint noise in access logs
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        uv_logger = logging.getLogger(name)
        uv_logger.handlers.clear()
        uv_logger.propagate = True
    logging.getLogger("uvicorn.access").addFilter(QuietAccessFilter())


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    settings = get_settings()
    logging.info(f"PyFinAgent backend starting (project={settings.gcp_project_id})")

    # phase-31.1: misnamed `settings.gemini_model` field can silently route
    # to Anthropic when set to "claude-*" -> the orchestrator hits the in-app
    # Anthropic SDK (NOT covered by Max plan) and fails on credit-balance
    # exhaustion (phase-31.0.3 Stage 3 Run 1 observation). Emit a clear
    # startup line documenting which provider the standard-tier resolves to
    # AND a WARNING when the field name disagrees with the routed provider.
    # Field rename to `standard_model` is deferred (would require Settings UI
    # + .env migration); this log is the minimum-disruption observability.
    _std_model = (settings.gemini_model or "").strip()
    if _std_model.startswith("gemini-"):
        _std_provider = "Gemini (Vertex AI or direct AI Studio)"
        _std_warning = False
    elif _std_model.startswith("claude-"):
        _std_provider = "Anthropic Claude API (requires ANTHROPIC_API_KEY + funded balance)"
        _std_warning = True
    elif _std_model.startswith(("gpt-", "o1", "o3", "o4")):
        _std_provider = "OpenAI (requires OPENAI_API_KEY + funded balance)"
        _std_warning = True
    else:
        _std_provider = f"unknown (model='{_std_model}')"
        _std_warning = True
    logging.info(
        "phase-31.1 model routing: settings.gemini_model='%s' -> standard-tier provider=%s",
        _std_model, _std_provider,
    )
    if _std_warning:
        logging.warning(
            "phase-31.1: settings.gemini_model is set to a non-Gemini model ('%s'). "
            "The field name is preserved for backward compat but routes via "
            "backend/agents/llm_client.py::make_client. Ensure the API key for "
            "%s is funded; OR switch to a 'gemini-*' model to use Vertex AI / AI "
            "Studio (no credit balance dependency).",
            _std_model, _std_provider,
        )

    # phase-38.3: parallel banner for the deep-think tier (Moderator / Critic /
    # Synthesis / RiskJudge). Closes the observability gap documented in
    # phase-34.1 + closure_roadmap.md §3 OPEN-12: silent regression to
    # claude-opus-4-7 + Anthropic credit-exhaustion was hard to spot because
    # only the standard-tier banner existed. Same provider-detect + warning
    # logic as the standard tier; greppable via `phase-3[18] model routing`.
    _dt_model = (settings.deep_think_model or "").strip()
    if _dt_model.startswith("gemini-"):
        _dt_provider = "Gemini (Vertex AI or direct AI Studio)"
        _dt_warning = False
    elif _dt_model.startswith("claude-"):
        _dt_provider = "Anthropic Claude API (requires ANTHROPIC_API_KEY + funded balance)"
        _dt_warning = True
    elif _dt_model.startswith(("gpt-", "o1", "o3", "o4")):
        _dt_provider = "OpenAI (requires OPENAI_API_KEY + funded balance)"
        _dt_warning = True
    else:
        _dt_provider = f"unknown (model='{_dt_model}')"
        _dt_warning = True
    logging.info(
        "phase-38.3 model routing: settings.deep_think_model='%s' -> deep-think-tier provider=%s",
        _dt_model, _dt_provider,
    )
    if _dt_warning:
        logging.warning(
            "phase-38.3: settings.deep_think_model is set to a non-Gemini model ('%s'). "
            "The deep-think tier (Moderator/Critic/Synthesis/RiskJudge) routes via "
            "backend/agents/llm_client.py::make_client. Ensure the API key for "
            "%s is funded; OR switch to a 'gemini-*' model to use Vertex AI / AI "
            "Studio (no credit balance dependency). phase-34.1e history: the "
            "previous claude-opus-4-7 default caused silent regression to Anthropic "
            "credit-exhaustion on fresh checkout / restart without DEEP_THINK_MODEL "
            "env override.",
            _dt_model, _dt_provider,
        )

    # phase-23.1.21: register faulthandler on SIGUSR1 so a hung process can
    # be diagnosed without a kill. Operators (or the watchdog) send
    # `kill -USR1 <pid>` to dump all thread stacks to stderr (which is
    # tee'd into backend.log via the launchd plist). Then `kickstart -k`
    # for actual restart. Diagnosing a 19-hour silent hang post-mortem
    # is impossible without this.
    try:
        import faulthandler
        import signal as _signal
        # all_threads=True dumps every Python thread; chain=False prevents
        # the previous SIGUSR1 handler (none here) from running afterward.
        faulthandler.register(_signal.SIGUSR1, all_threads=True, chain=False)
        logging.info("faulthandler registered on SIGUSR1 (kill -USR1 PID for stack dump)")
    except Exception:
        logging.warning("faulthandler registration failed", exc_info=True)

    # phase-38.6.1: cycle-lock stale-recovery hook. If the prior backend
    # was SIGKILL'd mid-cycle, handoff/.autonomous_loop.lock still exists
    # with the dead pid; clean_stale_lock detects + unlinks. Fail-open
    # per existing convention (faulthandler block above also fail-opens).
    try:
        from backend.services.cycle_lock import clean_stale_lock as _clean_stale_lock
        _cleaned = _clean_stale_lock(reason="startup_recovery")
        if _cleaned:
            logging.warning(
                "phase-38.6.1: cleaned stale autonomous_loop lock on startup "
                "(prior_pid=%s prior_cycle_id=%s age_sec=%.0f). Prior cycle "
                "did not exit cleanly; recovery complete.",
                _cleaned.get("pid"), _cleaned.get("cycle_id"),
                _cleaned.get("age_sec", 0.0),
            )
    except Exception:
        logging.exception("phase-38.6.1: cycle_lock recovery hook failed (fail-open)")

    # phase-23.1.19: log RLIMIT_NOFILE so FD-exhaustion crashes are easy to
    # diagnose. The launchd plist sets NumberOfFiles=16384; if the soft limit
    # at boot is dramatically lower, FDs run out faster than expected.
    try:
        import resource as _resource
        _soft, _hard = _resource.getrlimit(_resource.RLIMIT_NOFILE)
        logging.info("RLIMIT_NOFILE: soft=%d hard=%d", _soft, _hard)
        if _soft < 4096:
            logging.warning(
                "RLIMIT_NOFILE soft=%d is dangerously low; backend will crash "
                "after a few hours of normal traffic. Run: ulimit -n 65536",
                _soft,
            )
    except Exception:
        logging.warning("could not read RLIMIT_NOFILE", exc_info=True)

    # phase-4.9.2: Immutable risk limits boot-loader. Installs
    # SIGHUP-ignore + on-disk watcher that os._exit(2)s the process
    # if limits.yaml is mutated at runtime. Runs in the main thread
    # before workers fork so signal.signal() is legal.
    try:
        from backend.governance.limits_loader import (
            get_digest as _limits_digest,
            load_once as _load_limits_once,
        )
        _load_limits_once()
        logging.info("governance: immutable limits loaded; digest=%s...",
                      _limits_digest()[:12])
    except Exception:
        logging.exception("governance: limits_loader failed; continuing "
                           "(service will run with default soft limits)")

    # Start paper trading scheduler if enabled
    if settings.paper_trading_enabled:
        try:
            from apscheduler.schedulers.asyncio import AsyncIOScheduler
            scheduler = AsyncIOScheduler()
            init_scheduler(scheduler)
            scheduler.start()
            # phase-23.2.23: register so /api/jobs/all can introspect
            _register_cron_scheduler("main", scheduler)
            logging.info("Paper trading scheduler started")
        except ImportError:
            logging.warning("APScheduler not installed, paper trading scheduler disabled")
        except Exception as e:
            logging.warning(f"Failed to start paper trading scheduler: {e}")

    # Start Slack monitor (Phase 2.11: responsive Slack connection)
    try:
        from backend.slack_monitor import init_slack_monitor
        from slack_sdk import WebClient
        import os
        slack_token = os.getenv("SLACK_BOT_TOKEN")
        if slack_token:
            slack_client = WebClient(token=slack_token)
            await init_slack_monitor(slack_client)
            logging.info("Slack monitor started (Phase 2.11)")
        else:
            logging.warning("SLACK_BOT_TOKEN not set, Slack monitor disabled")
    except Exception as e:
        logging.warning(f"Failed to start Slack monitor: {e}")

    # Start ticket queue processor (Phase 3.2.1: Agent coordination)
    # Use APScheduler to run processor in background
    processor_job = None
    try:
        from apscheduler.schedulers.asyncio import AsyncIOScheduler
        from backend.services.ticket_queue_processor import get_queue_processor
        import asyncio as aio_lib
        
        # Create a separate async scheduler just for the queue processor
        queue_scheduler = AsyncIOScheduler()
        
        async def process_batch():
            """Process one batch of tickets."""
            processor = get_queue_processor()
            try:
                await processor.process_queue_batch(batch_size=10)
            except Exception as e:
                logging.warning(f"Queue processor batch error: {e}")
        
        # Schedule batch processing every 5 seconds.
        # phase-23.3.1: pass explicit id + name so /cron Jobs tab shows
        # human-readable identifiers instead of an APScheduler-generated
        # UUID and `lifespan.<locals>.process_batch` qualname.
        processor_job = queue_scheduler.add_job(
            process_batch, 'interval', seconds=5,
            id="ticket_queue_process_batch",
            name="Ticket queue batch processor",
            replace_existing=True,
        )
        queue_scheduler.start()
        # phase-23.2.23: register so /api/jobs/all can introspect
        _register_cron_scheduler("queue", queue_scheduler)
        logging.info("Ticket queue processor started (Phase 3.2.1)")
    except Exception as e:
        logging.warning(f"Failed to start ticket queue processor: {e}")

    # phase-23.1.16: prewarm ticker-meta cache for current paper_positions
    # tickers so the first user landing on /paper-trading sees populated
    # COMPANY + SECTOR columns within 1-2s instead of 15-20s. Non-blocking;
    # failure is logged non-fatal and backend boots normally.
    async def _prewarm_ticker_meta():
        try:
            import asyncio as _asyncio
            from backend.api.paper_trading import _fetch_ticker_meta
            from backend.config.settings import get_settings as _get_settings
            from backend.db.bigquery_client import BigQueryClient as _BQ
            from backend.services.api_cache import (
                ENDPOINT_TTLS as _TTLS,
                get_api_cache as _get_cache,
            )
            settings = _get_settings()
            bq = _BQ(settings)
            positions = await _asyncio.to_thread(bq.get_paper_positions)
            tickers = sorted({p.get("ticker") for p in positions if p.get("ticker")})
            if not tickers:
                logging.info("Ticker-meta prewarm: no current positions, skipped")
                return
            logging.info("Prewarming ticker-meta cache for %d tickers...", len(tickers))
            result = await _asyncio.to_thread(_fetch_ticker_meta, tickers, settings, bq)
            cache = _get_cache()
            ttl = _TTLS["paper:ticker_meta"]
            for t, info in (result.get("meta") or {}).items():
                cache.set(f"paper:ticker_meta:single:{t}", info, ttl)
            logging.info("Ticker-meta prewarm complete (%d resolved)", len(result.get("meta") or {}))
        except Exception as e:
            logging.warning("Ticker-meta prewarm failed (non-fatal): %s", e)

    asyncio.create_task(_prewarm_ticker_meta())

    try:
        yield
    finally:
        # Shutdown
        logging.info("PyFinAgent backend shutting down")
        
        # Stop queue processor scheduler if it was started
        try:
            if 'queue_scheduler' in locals():
                queue_scheduler.shutdown(wait=False)
        except Exception:
            pass


app = FastAPI(
    title="PyFinAgent API",
    description="Agentic AI financial analyst backend",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS \u2014 allow the Next.js frontend in dev, production, and Tailscale
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^http://(localhost|100\.\d+\.\d+\.\d+):\d+$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths that skip authentication
_PUBLIC_PATHS = (
    "/api/health",
    "/api/changelog",
    "/api/auth",
    "/api/cost-budget",
    "/api/jobs/status",
    "/api/harness/monthly-approval",
    "/api/harness/demotion-audit",
    "/api/harness/weekly-ledger",
    "/api/harness/candidate-space",
    "/api/harness/results-distribution",
    "/api/signals",
    "/api/observability",
    "/api/sovereign",
    "/docs",
    "/openapi.json",
    "/redoc",
)


@app.middleware("http")
async def auth_and_security_middleware(request: Request, call_next):
    """Authentication check + OWASP security headers + latency tracking."""
    path = request.url.path

    # Skip auth for public paths and CORS preflights. Browsers never
    # send credentials on OPTIONS preflights, so rejecting them here
    # produces a 401 before CORSMiddleware can emit the Access-Control-
    # Allow-* headers, and Safari/Chrome surface this as "Load failed".
    if request.method != "OPTIONS" and not any(path.startswith(p) for p in _PUBLIC_PATHS):
        try:
            await get_current_user(request)
        except HTTPException as auth_exc:
            # FastAPI's exception handlers only run inside the route
            # dispatch, so we must translate manually here or Starlette
            # wraps it as 500.
            # Must also emit CORS headers ourselves: this response is
            # returned BEFORE CORSMiddleware wraps the response chain,
            # so browsers surface a missing Access-Control-Allow-Origin
            # as "Load failed" instead of showing the 401.
            from starlette.responses import JSONResponse
            origin = request.headers.get("origin", "")
            headers = {"WWW-Authenticate": "Bearer"}
            if origin and (
                origin.startswith("http://localhost:")
                or (origin.startswith("http://100.") and origin.count(".") == 3)
            ):
                headers["Access-Control-Allow-Origin"] = origin
                headers["Access-Control-Allow-Credentials"] = "true"
                headers["Vary"] = "Origin"
            return JSONResponse(
                status_code=auth_exc.status_code,
                content={"detail": auth_exc.detail},
                headers=headers,
            )

    start = time.perf_counter()
    response: Response = await call_next(request)
    latency_ms = (time.perf_counter() - start) * 1000

    # Latency tracking
    cache_hit = response.headers.get("X-Cache") == "HIT"
    get_perf_tracker().record(
        endpoint=path,
        method=request.method,
        status_code=response.status_code,
        latency_ms=round(latency_ms, 1),
        cache_hit=cache_hit,
    )
    response.headers["X-Response-Time"] = f"{latency_ms:.0f}ms"

    # OWASP security headers
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "0"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Cache-Control"] = "no-store"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"

    return response

# Routes
app.include_router(agent_map_router)
app.include_router(analysis_router)
app.include_router(backtest_router)
app.include_router(charts_router)
app.include_router(investigate_router)
app.include_router(paper_trading_router)
app.include_router(performance_router)
app.include_router(portfolio_router)
app.include_router(reports_router)
app.include_router(settings_router)
app.include_router(signals_router)
app.include_router(skills_router)
app.include_router(mas_events_router)
# phase-23.2.23 cron / logs operator dashboard
app.include_router(cron_dashboard_router)

# phase-10.11 autoresearch sprint-state tile endpoint.
from backend.api.harness_autoresearch import router as harness_autoresearch_router
app.include_router(harness_autoresearch_router)

# phase-15.1 cost-budget watcher
from backend.api.cost_budget_api import router as cost_budget_router
app.include_router(cost_budget_router)
# phase-15.2 Slack job heartbeat
from backend.api.job_status_api import router as job_status_router
app.include_router(job_status_router)
# phase-15.3 Monthly HITL approval
from backend.api.monthly_approval_api import router as monthly_approval_router
app.include_router(monthly_approval_router)
# phase-15.10 Observability latency
from backend.api.observability_api import router as observability_router
app.include_router(observability_router)
# phase-10.5.0 Sovereign UI read endpoints
from backend.api.sovereign_api import router as sovereign_router
app.include_router(sovereign_router)


@app.get("/api/health")
async def health():
    # Read version from CHANGELOG.md first entry, fallback to hardcoded
    try:
        import re as _re
        _cl = (Path(__file__).parent.parent / "CHANGELOG.md").read_text(encoding="utf-8")
        _m = _re.search(r"### v([\d.]+)", _cl)
        _ver = _m.group(1) if _m else "6.0.0"
    except Exception:
        _ver = "6.0.0"
    mcp_servers = {}
    for name, mod_name in [
        ("data", "backend.agents.mcp_servers.data_server"),
        ("backtest", "backend.agents.mcp_servers.backtest_server"),
        ("signals", "backend.agents.mcp_servers.signals_server"),
    ]:
        spec = importlib.util.find_spec(mod_name)
        mcp_servers[name] = {"status": "ok"} if spec else {"status": "error", "detail": "module not found"}

    # phase-4.9.2: expose the immutable-limits boot digest so
    # operators can confirm every live instance is running the
    # same governance state.
    limits_digest: str | None = None
    try:
        from backend.governance.limits_loader import get_digest as _lg
        limits_digest = _lg()
    except Exception:
        limits_digest = None

    return {
        "status": "ok",
        "service": "pyfinagent-backend",
        "version": _ver,
        "mcp_servers": mcp_servers,
        "limits_digest": limits_digest,
    }


@app.get("/api/changelog")
async def get_changelog():
    """Parse CHANGELOG.md into structured entries for the frontend."""
    import re
    changelog_path = Path(__file__).parent.parent / "CHANGELOG.md"
    if not changelog_path.exists():
        return {"entries": []}

    text = changelog_path.read_text(encoding="utf-8")
    entries: list[dict] = []
    # Split on version headers: ### v5.12.10 — Title (Date)
    parts = re.split(r"^### (v[\d.]+)\s*—\s*(.+?)(?:\s*\(([^)]+)\))?\s*$", text, flags=re.MULTILINE)
    # parts[0] is preamble, then groups of 4: version, title, date, body
    i = 1
    while i + 3 <= len(parts):
        version = parts[i].lstrip("v")
        title = parts[i + 1].strip()
        date = (parts[i + 2] or "").strip()
        body = parts[i + 3].strip()

        # Extract human-readable summary: first bold paragraph or numbered items
        # Simplify: take numbered items and strip markdown
        changes: list[str] = []
        for line in body.split("\n"):
            line = line.strip()
            # Numbered items like "1. **Title** — description"
            m = re.match(r"^\d+[a-z]?\.\s+\*\*(.+?)\*\*\s*[-—]?\s*(.*)", line)
            if m:
                item_title = m.group(1).strip()
                item_desc = m.group(2).strip()
                if item_desc:
                    changes.append(f"{item_title} — {item_desc}")
                else:
                    changes.append(item_title)
                continue
            # Bullet items like "- [ ] task" or "- description"
            m2 = re.match(r"^-\s+\[.\]\s+(.*)", line)
            if m2:
                changes.append(m2.group(1).strip())
                continue
            m3 = re.match(r"^-\s+\*\*(.+?)\*\*\s*[-—]?\s*(.*)", line)
            if m3:
                changes.append(f"{m3.group(1).strip()} — {m3.group(2).strip()}" if m3.group(2).strip() else m3.group(1).strip())
                continue
            # First bold paragraph as summary
            if line.startswith("**") and line.endswith("**") and not changes:
                changes.append(re.sub(r"\*\*", "", line))

        # Limit to 8 changes per entry for readability
        entries.append({
            "version": version,
            "title": title,
            "date": date,
            "changes": changes[:8],
        })
        i += 4

    # Auto-generate recent commits from git
    recent_commits: list[dict] = []
    try:
        import subprocess
        result = subprocess.run(
            ["git", "log", "--oneline", "--no-decorate", "-20"],
            capture_output=True, text=True, timeout=5,
            cwd=str(changelog_path.parent)
        )
        if result.returncode == 0:
            for line in result.stdout.strip().split("\n"):
                if line:
                    parts_c = line.split(" ", 1)
                    if len(parts_c) == 2:
                        recent_commits.append({"hash": parts_c[0], "message": parts_c[1]})
    except Exception:
        pass

    return {"entries": entries, "recent_commits": recent_commits}
