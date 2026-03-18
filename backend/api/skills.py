"""
Skills optimization API endpoints.

POST /api/skills/optimize  — start the autonomous optimization loop
POST /api/skills/stop      — stop the loop gracefully
GET  /api/skills/experiments — view experiment history
GET  /api/skills/status    — current loop status
GET  /api/skills/{agent}   — view an agent's current skills.md
GET  /api/skills/analysis  — experiment analysis summary + chart data
"""

import asyncio
import logging
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, HTTPException

from backend.agents.skill_optimizer import SkillOptimizer, SKILLS_DIR, OPTIMIZABLE_AGENTS
from backend.agents.skills.experiments.analyze_experiments import full_analysis
from backend.config.settings import get_settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/skills", tags=["skills"])

# Singleton optimizer instance (created on first use)
_optimizer: SkillOptimizer | None = None


def _get_optimizer() -> SkillOptimizer:
    global _optimizer
    if _optimizer is None:
        _optimizer = SkillOptimizer(get_settings())
    return _optimizer


def _run_optimization_loop(max_iterations: int = 0):
    """Background task for the optimization loop."""
    optimizer = _get_optimizer()
    try:
        optimizer.establish_baseline()
        optimizer.run_loop(max_iterations=max_iterations)
    except Exception as e:
        logger.error(f"Optimization loop crashed: {e}", exc_info=True)


@router.post("/optimize")
async def start_optimization(background_tasks: BackgroundTasks, max_iterations: int = 0):
    """Start the autonomous skill optimization loop in the background."""
    optimizer = _get_optimizer()
    if optimizer.is_running:
        raise HTTPException(status_code=409, detail="Optimization loop is already running")

    background_tasks.add_task(_run_optimization_loop, max_iterations)
    return {"status": "started", "max_iterations": max_iterations or "unlimited"}


@router.post("/stop")
async def stop_optimization():
    """Signal the optimization loop to stop gracefully."""
    optimizer = _get_optimizer()
    if not optimizer.is_running:
        raise HTTPException(status_code=409, detail="Optimization loop is not running")

    optimizer.stop()
    return {"status": "stopping"}


@router.get("/status")
async def get_status():
    """Get current optimizer status."""
    optimizer = _get_optimizer()
    return optimizer.get_status()


@router.get("/experiments")
async def get_experiments():
    """Get all experiment history from skill_results.tsv."""
    optimizer = _get_optimizer()
    return {"experiments": optimizer.get_all_experiments()}


@router.get("/analysis")
async def get_analysis():
    """Get experiment analysis: summary, keep rates, delta chain, running best, top hits."""
    return full_analysis()


@router.get("/agents")
async def list_agents():
    """List all optimizable agents with their skill file status."""
    agents = []
    for name in OPTIMIZABLE_AGENTS:
        path = SKILLS_DIR / f"{name}.md"
        agents.append({
            "name": name,
            "has_skill_file": path.exists(),
        })
    return {"agents": agents}


@router.get("/{agent_name}")
async def get_agent_skill(agent_name: str):
    """View an agent's current skills.md content."""
    if agent_name not in OPTIMIZABLE_AGENTS:
        raise HTTPException(status_code=404, detail=f"Unknown agent: {agent_name}")

    skill_path = SKILLS_DIR / f"{agent_name}.md"
    if not skill_path.exists():
        raise HTTPException(status_code=404, detail=f"Skill file not found for {agent_name}")

    content = skill_path.read_text(encoding="utf-8")
    return {"agent": agent_name, "content": content}
