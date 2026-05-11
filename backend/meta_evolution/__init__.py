"""phase-10.7 Meta-evolution package.

Houses the metrics + machinery that observe how the system improves over
time (alpha velocity, recursive prompt optimization, cron budget
allocation, evaluator review gate).

Distinct from the legacy `backend/agents/meta_coordinator.py` module
(which remains ACTIVE for `autonomous_loop.py` + `skill_optimizer.py`
but should NOT be extended — see phase-23.8.3 closure of audit R-6).
Build all new dev-loop work here under `backend/meta_evolution/`.
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
