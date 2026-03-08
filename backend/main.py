"""
PyFinAgent Backend — FastAPI application entry point.
"""

import logging
import json
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.analysis import router as analysis_router
from backend.api.charts import router as charts_router
from backend.api.investigate import router as investigate_router
from backend.api.reports import router as reports_router
from backend.config.settings import get_settings


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
    yield
    logging.info("PyFinAgent backend shutting down")


app = FastAPI(
    title="PyFinAgent API",
    description="Agentic AI financial analyst backend",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS — allow the Next.js frontend in dev and production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
app.include_router(analysis_router)
app.include_router(charts_router)
app.include_router(investigate_router)
app.include_router(reports_router)


@app.get("/api/health")
async def health():
    return {"status": "ok", "service": "pyfinagent-backend"}
