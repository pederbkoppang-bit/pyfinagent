"""
Financial Situation Memory — BM25-based learning from past analysis outcomes.

Stores (situation, lesson) tuples and retrieves top-N relevant past lessons
for injection into agent prompts. Uses BM25 (lexical similarity) — no API
calls, no token limits, works offline.

Research basis: TradingAgents FinancialSituationMemory — agents learn from
past mistakes to avoid repeating wrong BUY/SELL/HOLD calls.
"""

import json
import logging
import re
from datetime import datetime
from typing import Optional

from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

# Seed data: common financial archetypes for cold-start
_SEED_MEMORIES = [
    (
        "High inflation with rising interest rates and declining consumer spending in consumer discretionary sector",
        "Be cautious on consumer discretionary stocks during rate hike cycles. "
        "Historically, high P/E names underperform when rates rise above 4%. "
        "Prioritize companies with pricing power and low debt-to-equity ratios.",
    ),
    (
        "Tech sector showing high volatility with increasing institutional selling and insider sales",
        "Cluster insider selling in tech often precedes a 10-15% drawdown within 60 days. "
        "Reduce position size when insider sell/buy ratio exceeds 3:1. "
        "Look for divergence between social sentiment (still bullish) and smart money (bearish).",
    ),
    (
        "Strong earnings beat but stock sold off on forward guidance concerns with elevated put volume",
        "Post-earnings selloffs on strong beats often signal the market was already pricing in perfection. "
        "High put/call ratio after a beat is bearish — institutions are hedging. "
        "Wait 2-3 weeks for dust to settle before re-evaluating.",
    ),
    (
        "Small-cap biotech with patent cliff approaching and competitor gaining FDA approval",
        "Patent cliff risk is often underpriced by 12-18 months before expiry. "
        "Competitor FDA approvals accelerate revenue erosion faster than models predict. "
        "Conservative position sizing (1-2% max) is warranted for binary-outcome names.",
    ),
    (
        "Yield curve inversion with unemployment rising and GDP slowing in industrial sector",
        "Yield curve inversion has preceded every recession since 1970 with 12-18 month lag. "
        "Industrial cyclicals typically decline 20-30% from peak during recessions. "
        "Defensive rotation to utilities, healthcare, and consumer staples outperforms.",
    ),
]


class FinancialSituationMemory:
    """Memory system for storing and retrieving financial situation lessons using BM25."""

    def __init__(self, name: str):
        self.name = name
        self.documents: list[str] = []
        self.lessons: list[str] = []
        self.metadata: list[dict] = []
        self.bm25 = None

        # Seed with archetypes for cold-start
        for situation, lesson in _SEED_MEMORIES:
            self.documents.append(situation)
            self.lessons.append(lesson)
            self.metadata.append({"source": "seed", "timestamp": "2026-01-01"})
        self._rebuild_index()

    def _tokenize(self, text: str) -> list[str]:
        """Simple whitespace + punctuation tokenization with lowercasing."""
        return re.findall(r"\b\w+\b", text.lower())

    def _rebuild_index(self):
        """Rebuild BM25 index after adding documents."""
        if self.documents:
            tokenized = [self._tokenize(doc) for doc in self.documents]
            self.bm25 = BM25Okapi(tokenized)
        else:
            self.bm25 = None

    def add_memory(
        self, situation: str, lesson: str, metadata: Optional[dict] = None
    ):
        """Add a single situation-lesson pair."""
        self.documents.append(situation)
        self.lessons.append(lesson)
        self.metadata.append(metadata or {"timestamp": datetime.utcnow().isoformat()})
        self._rebuild_index()

    def add_memories(self, entries: list[tuple[str, str]]):
        """Add multiple (situation, lesson) tuples."""
        for situation, lesson in entries:
            self.documents.append(situation)
            self.lessons.append(lesson)
            self.metadata.append({"timestamp": datetime.utcnow().isoformat()})
        self._rebuild_index()

    def get_memories(self, current_situation: str, n_matches: int = 2) -> list[dict]:
        """
        Retrieve top-N relevant past lessons for the current situation.

        Returns list of dicts with situation, lesson, and similarity_score.
        """
        if not self.documents or self.bm25 is None:
            return []

        query_tokens = self._tokenize(current_situation)
        scores = self.bm25.get_scores(query_tokens)
        max_score = max(scores) if max(scores) > 0 else 1

        top_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:n_matches]

        results = []
        for idx in top_indices:
            normalized = scores[idx] / max_score if max_score > 0 else 0
            if normalized > 0.1:  # Only return meaningfully similar matches
                results.append({
                    "situation": self.documents[idx],
                    "lesson": self.lessons[idx],
                    "similarity_score": round(normalized, 3),
                })
        return results

    def format_for_prompt(self, current_situation: str, n_matches: int = 2) -> str:
        """Get memories formatted as a string for prompt injection."""
        memories = self.get_memories(current_situation, n_matches)
        if not memories:
            return ""
        parts = []
        for i, mem in enumerate(memories, 1):
            parts.append(
                f"Reflection {i} (relevance: {mem['similarity_score']:.0%}):\n"
                f"  Past situation: {mem['situation'][:200]}\n"
                f"  Lesson learned: {mem['lesson'][:300]}"
            )
        return "\n\n".join(parts)

    def load_from_bq_rows(self, rows: list[dict]):
        """Load memories from BigQuery rows."""
        for row in rows:
            self.documents.append(row.get("situation", ""))
            self.lessons.append(row.get("lesson", ""))
            self.metadata.append({
                "ticker": row.get("ticker", ""),
                "agent_type": row.get("agent_type", ""),
                "timestamp": row.get("created_at", ""),
                "source": "bq",
            })
        self._rebuild_index()

    def clear(self):
        """Clear all stored memories (keeps seed data)."""
        self.documents = []
        self.lessons = []
        self.metadata = []
        for situation, lesson in _SEED_MEMORIES:
            self.documents.append(situation)
            self.lessons.append(lesson)
            self.metadata.append({"source": "seed", "timestamp": "2026-01-01"})
        self._rebuild_index()


def build_situation_description(
    ticker: str,
    sector: str,
    enrichment_signals: dict,
    debate_result: dict | None = None,
) -> str:
    """
    Build a textual description of the current financial situation
    for memory retrieval. Captures key market context.
    """
    parts = [f"Analyzing {ticker} in the {sector or 'unknown'} sector."]

    # Summarize signal directions
    bullish, bearish, neutral_sigs = [], [], []
    for name, sig in enrichment_signals.items():
        signal = str(sig.get("signal", "")).upper()
        if "BULL" in signal or "RISING" in signal or "BREAKOUT" in signal:
            bullish.append(name)
        elif "BEAR" in signal or "DECLINING" in signal or "RISK" in signal:
            bearish.append(name)
        else:
            neutral_sigs.append(name)

    if bullish:
        parts.append(f"Bullish signals from: {', '.join(bullish)}.")
    if bearish:
        parts.append(f"Bearish signals from: {', '.join(bearish)}.")
    if neutral_sigs:
        parts.append(f"Neutral/mixed signals from: {', '.join(neutral_sigs)}.")

    if debate_result:
        consensus = debate_result.get("consensus", "N/A")
        confidence = debate_result.get("consensus_confidence", "N/A")
        parts.append(f"Debate consensus: {consensus} (confidence: {confidence}).")
        contradictions = debate_result.get("contradictions", [])
        if contradictions:
            topics = [c.get("topic", "") for c in contradictions[:3] if c.get("topic")]
            if topics:
                parts.append(f"Key contradictions: {'; '.join(topics)}.")

    return " ".join(parts)


def generate_reflection(
    model,
    agent_type: str,
    ticker: str,
    original_recommendation: str,
    actual_return_pct: float,
    situation: str,
    holding_days: int,
) -> str:
    """
    Generate a reflection/lesson from a past analysis outcome using LLM.

    This is called by the outcome tracker AFTER evaluating actual returns.
    The LLM reflects on what went right/wrong and produces a concise lesson.
    """
    direction_correct = (
        (original_recommendation in ("Strong Buy", "Buy") and actual_return_pct > 0)
        or (original_recommendation in ("Strong Sell", "Sell") and actual_return_pct < 0)
    )

    prompt = (
        f"You are reflecting on a past {agent_type} analysis to learn from it.\n\n"
        f"Ticker: {ticker}\n"
        f"Original recommendation: {original_recommendation}\n"
        f"Actual return after {holding_days} days: {actual_return_pct:.1f}%\n"
        f"Directionally correct: {'YES' if direction_correct else 'NO'}\n"
        f"Market context at the time: {situation[:500]}\n\n"
        "Write a 2-3 sentence lesson learned. Be specific about what signal "
        "or factor was most important in hindsight. If the call was wrong, "
        "identify what was missed. If correct, note the key signal to watch for next time."
    )

    try:
        response = model.generate_content(prompt, generation_config={"temperature": 0.3})
        return response.text.strip()[:500]
    except Exception as e:
        logger.warning(f"Failed to generate reflection for {ticker}: {e}")
        return (
            f"{'Correct' if direction_correct else 'Incorrect'} call on {ticker}. "
            f"Recommended {original_recommendation}, actual return {actual_return_pct:.1f}% "
            f"over {holding_days} days."
        )
