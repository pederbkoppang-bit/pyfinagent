"""
PyFinAgent Backend — FastAPI application entry point.
"""

import asyncio
import io
import logging
import json
import sys
import time
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware

from backend.api.analysis import router as analysis_router
from backend.api.auth import get_current_user
from backend.api.backtest import router as backtest_router
from backend.api.charts import router as charts_router
from backend.api.investigate import router as investigate_router
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

    # Start paper trading scheduler if enabled
    if settings.paper_trading_enabled:
        try:
            from apscheduler.schedulers.asyncio import AsyncIOScheduler
            scheduler = AsyncIOScheduler()
            init_scheduler(scheduler)
            scheduler.start()
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
        
        # Schedule batch processing every 5 seconds
        processor_job = queue_scheduler.add_job(process_batch, 'interval', seconds=5)
        queue_scheduler.start()
        logging.info("Ticket queue processor started (Phase 3.2.1)")
    except Exception as e:
        logging.warning(f"Failed to start ticket queue processor: {e}")

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
_PUBLIC_PATHS = ("/api/health", "/api/auth", "/docs", "/openapi.json", "/redoc")


@app.middleware("http")
async def auth_and_security_middleware(request: Request, call_next):
    """Authentication check + OWASP security headers + latency tracking."""
    path = request.url.path

    # Skip auth for public paths
    if not any(path.startswith(p) for p in _PUBLIC_PATHS):
        await get_current_user(request)

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
    return {"status": "ok", "service": "pyfinagent-backend", "version": _ver}


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
