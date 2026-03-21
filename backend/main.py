"""
PyFinAgent Backend — FastAPI application entry point.
"""

import logging
import json
import time
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
from backend.config.settings import get_settings
from backend.services.perf_tracker import get_perf_tracker


class JsonFormatter(logging.Formatter):
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


def setup_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)


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

    yield
    logging.info("PyFinAgent backend shutting down")


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


@app.get("/api/health")
async def health():
    return {"status": "ok", "service": "pyfinagent-backend"}
