"""
Decision trace logger — Explainable AI (XAI) audit trail.
Every LLM agent call produces a DecisionTrace capturing inputs, outputs,
reasoning steps, and evidence citations for full transparency.

Also defines AnalysisContext — short-term session memory that accumulates
key findings during a single analysis run.

Research basis: Goldman Sachs XAI requirements (ref 16).
"""

import hashlib
import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

# ── Session Memory ──────────────────────────────────────────────

MAX_FINDINGS = 20
MAX_FINDING_LEN = 100


@dataclass
class AnalysisContext:
    """Short-term session memory that accumulates key findings during a run.

    Passed to later agents (Synthesis, Deep Dive, Critic) so they can build
    on insights from earlier steps. Capped at MAX_FINDINGS entries of
    MAX_FINDING_LEN chars each to avoid prompt bloat.
    """

    key_findings: list[str] = field(default_factory=list)
    contradictions: list[str] = field(default_factory=list)
    signal_consensus: dict[str, str] = field(default_factory=dict)

    def add_finding(self, finding: str) -> None:
        if len(self.key_findings) < MAX_FINDINGS:
            self.key_findings.append(finding[:MAX_FINDING_LEN])

    def add_contradiction(self, contradiction: str) -> None:
        if len(self.contradictions) < MAX_FINDINGS:
            self.contradictions.append(contradiction[:MAX_FINDING_LEN])

    def set_signal(self, source: str, signal: str) -> None:
        self.signal_consensus[source] = signal

    def format_for_prompt(self) -> str:
        """Format accumulated context as a prompt section."""
        if not self.key_findings and not self.contradictions:
            return ""
        parts = ["--- ACCUMULATED ANALYSIS CONTEXT (Session Memory) ---"]
        if self.key_findings:
            parts.append("Key Findings So Far:")
            for i, f in enumerate(self.key_findings, 1):
                parts.append(f"  {i}. {f}")
        if self.contradictions:
            parts.append("Contradictions Detected:")
            for c in self.contradictions:
                parts.append(f"  - {c}")
        if self.signal_consensus:
            bullish = sum(1 for s in self.signal_consensus.values() if "BULL" in s.upper())
            bearish = sum(1 for s in self.signal_consensus.values() if "BEAR" in s.upper())
            parts.append(f"Signal Consensus: {bullish} bullish, {bearish} bearish, {len(self.signal_consensus) - bullish - bearish} neutral")
        parts.append("----------------------------------------------------")
        return "\n".join(parts)


# ── Decision Trace ──────────────────────────────────────────────


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
