"""
Investigate API — follow-up RAG-grounded questions about a company.
"""

import asyncio
import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from backend.agents.orchestrator import AnalysisOrchestrator
from backend.config.settings import Settings, get_settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["investigate"])

# Cache the orchestrator so we don't re-initialise Vertex AI on every request.
_orchestrator_cache: dict[str, AnalysisOrchestrator] = {}


def _get_orchestrator(settings: Settings = Depends(get_settings)) -> AnalysisOrchestrator:
    key = settings.gcp_project_id
    if key not in _orchestrator_cache:
        _orchestrator_cache[key] = AnalysisOrchestrator(settings)
    return _orchestrator_cache[key]


class InvestigateRequest(BaseModel):
    ticker: str = Field(..., min_length=1, max_length=10)
    question: str = Field(..., min_length=5, max_length=1000)


class Citation(BaseModel):
    uri: str = ""
    title: str = ""


class InvestigateResponse(BaseModel):
    ticker: str
    question: str
    answer: str
    citations: list[Citation] = []


@router.post("/investigate", response_model=InvestigateResponse)
async def investigate(
    req: InvestigateRequest,
    orchestrator: AnalysisOrchestrator = Depends(_get_orchestrator),
):
    """Ask a follow-up question grounded in 10-K / 10-Q documents via RAG."""
    ticker = req.ticker.upper().strip()
    question = req.question.strip()

    try:
        prompt = (
            f"You are a financial analyst assistant. "
            f"Answer this question about {ticker} using the 10-K and 10-Q filings: "
            f"{question}"
        )

        response = await asyncio.to_thread(
            orchestrator.rag_model.generate_content, prompt
        )

        citations: list[Citation] = []
        gm = getattr(response, "grounding_metadata", None)
        if gm:
            attrs = getattr(gm, "grounding_attributions", None) or []
            for cit in attrs:
                web = getattr(cit, "web", None)
                if web:
                    citations.append(
                        Citation(
                            uri=getattr(web, "uri", ""),
                            title=getattr(web, "title", ""),
                        )
                    )

        # Grounded responses may have multiple content parts;
        # response.text raises ValueError in that case.
        try:
            answer_text = response.text
        except ValueError:
            parts = response.candidates[0].content.parts
            answer_text = "\n".join(p.text for p in parts if hasattr(p, "text") and p.text)

        return InvestigateResponse(
            ticker=ticker,
            question=question,
            answer=answer_text,
            citations=citations,
        )

    except Exception as exc:
        logger.error(
            "Investigate failed for %s: %s", ticker, exc, exc_info=True
        )
        raise HTTPException(
            status_code=500,
            detail=f"Investigation failed: {type(exc).__name__}: {exc}",
        )
