"""
Decision trace logger — Explainable AI (XAI) audit trail.
Every LLM agent call produces a DecisionTrace capturing inputs, outputs,
reasoning steps, and evidence citations for full transparency.

Research basis: Goldman Sachs XAI requirements (ref 16).
"""

import hashlib
import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class DecisionTrace:
    """Immutable record of a single agent decision."""

    agent_name: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    input_data_hash: str = ""
    output_signal: str = ""
    confidence: float = 0.0
    evidence_citations: list[str] = field(default_factory=list)
    reasoning_steps: list[str] = field(default_factory=list)
    raw_output: str = ""
    latency_ms: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def hash_input(data) -> str:
    """Create a deterministic hash of input data for reproducibility tracking."""
    serialized = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode()).hexdigest()[:12]


class TraceCollector:
    """Collects decision traces across the entire analysis pipeline."""

    def __init__(self):
        self._traces: list[DecisionTrace] = []

    def add(self, trace: DecisionTrace) -> None:
        self._traces.append(trace)
        logger.info(
            f"Trace: {trace.agent_name} → {trace.output_signal} "
            f"(confidence={trace.confidence:.2f}, latency={trace.latency_ms:.0f}ms)"
        )

    def all_traces(self) -> list[dict]:
        return [t.to_dict() for t in self._traces]

    def get_agent_trace(self, agent_name: str) -> Optional[dict]:
        for t in self._traces:
            if t.agent_name == agent_name:
                return t.to_dict()
        return None

    def summary(self) -> dict:
        """High-level summary of all agent decisions for the debate framework."""
        return {
            "total_agents": len(self._traces),
            "signals": {
                t.agent_name: {
                    "signal": t.output_signal,
                    "confidence": t.confidence,
                    "top_evidence": t.evidence_citations[:3],
                }
                for t in self._traces
            },
        }
