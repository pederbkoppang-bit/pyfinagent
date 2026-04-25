"""phase-10.7 Meta-evolution package.

Houses the metrics + machinery that observe how the system improves over
time (alpha velocity, recursive prompt optimization, cron budget
allocation, evaluator review gate).

Distinct from the DEPRECATED `backend/agents/meta_coordinator.py` Phase-4
stub. Do not extend that module; build new work here.
"""
from backend.meta_evolution.archetype_library import (
    ALLOWED_REGIMES,
    ARCHETYPES,
    IMPLEMENTED_STRATEGY_IDS,
    Archetype,
    get_archetype,
)

__all__ = [
    "ALLOWED_REGIMES",
    "ARCHETYPES",
    "IMPLEMENTED_STRATEGY_IDS",
    "Archetype",
    "get_archetype",
]
