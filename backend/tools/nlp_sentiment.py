"""
NLP Sentiment tool — Transformer-based contextual sentiment via Vertex AI embeddings.
Uses text-embedding-005 to compute semantic similarity between news articles
and a curated financial sentiment corpus, providing nuanced sentiment scoring
beyond keyword-based approaches.

Research basis: Stanford University transformer embeddings (ref 11).
"""

import logging
from typing import Optional

import numpy as np
import vertexai
from vertexai.language_models import TextEmbeddingModel

logger = logging.getLogger(__name__)

# Financial sentiment reference corpus — carefully curated phrases
BULLISH_CORPUS = [
    "strong revenue growth exceeding expectations",
    "significant market share gains in key segments",
    "management raised full-year guidance",
    "expanding operating margins and profitability",
    "breakthrough product launch with strong adoption",
    "institutional investors increasing positions",
    "competitive moat strengthening through innovation",
    "free cash flow generation at record levels",
]

BEARISH_CORPUS = [
    "revenue decline and shrinking market share",
    "management lowered guidance citing headwinds",
    "margin compression from rising costs",
    "regulatory investigation and compliance concerns",
    "key executive departures raising uncertainty",
    "increasing competition eroding pricing power",
    "balance sheet deterioration with rising debt",
    "product recalls and quality control issues",
]


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


async def get_nlp_sentiment(
    ticker: str,
    articles: list[dict],
    project_id: str,
    location: str = "us-central1",
) -> dict:
    """
    Compute transformer-based NLP sentiment for a set of articles.

    Args:
        ticker: Stock ticker symbol
        articles: List of dicts with 'title', 'summary', 'source' keys
        project_id: GCP project ID for Vertex AI
        location: GCP region

    Returns:
        Structured NLP sentiment data with per-article and aggregate scores.
    """
    try:
        if not articles:
            return {
                "ticker": ticker,
                "signal": "NEUTRAL",
                "summary": "No articles available for NLP sentiment analysis.",
                "aggregate_score": 0.0,
                "confidence": 0.0,
            }

        # Initialize embedding model
        model = TextEmbeddingModel.from_pretrained("text-embedding-005")

        # Prepare article texts (use title + summary)
        article_texts = []
        for art in articles[:30]:  # Limit to 30 articles to control API calls
            text = f"{art.get('title', '')}. {art.get('summary', '')}"
            if text.strip(". "):
                article_texts.append(text[:500])  # Truncate individual articles

        if not article_texts:
            return {
                "ticker": ticker,
                "signal": "NEUTRAL",
                "summary": "No valid article text for embedding.",
                "aggregate_score": 0.0,
                "confidence": 0.0,
            }

        # Batch embed — articles + corpus
        all_texts = article_texts + BULLISH_CORPUS + BEARISH_CORPUS
        embeddings = model.get_embeddings(all_texts)

        article_embeddings = [np.array(e.values) for e in embeddings[:len(article_texts)]]
        bullish_embeddings = [
            np.array(e.values)
            for e in embeddings[len(article_texts):len(article_texts) + len(BULLISH_CORPUS)]
        ]
        bearish_embeddings = [
            np.array(e.values)
            for e in embeddings[len(article_texts) + len(BULLISH_CORPUS):]
        ]

        # Score each article
        article_scores = []
        for i, art_emb in enumerate(article_embeddings):
            bull_sim = float(np.mean([_cosine_similarity(art_emb, b) for b in bullish_embeddings]))
            bear_sim = float(np.mean([_cosine_similarity(art_emb, b) for b in bearish_embeddings]))

            # Sentiment score: positive = bullish, negative = bearish
            score = bull_sim - bear_sim
            # Normalize to roughly [-1, 1] range (typical diff is small)
            normalized_score = max(-1.0, min(1.0, score * 20))

            source = articles[i].get("source", "unknown") if i < len(articles) else "unknown"
            # Source reliability weight
            weight = _source_weight(source)

            article_scores.append({
                "index": i,
                "score": round(normalized_score, 3),
                "bull_similarity": round(bull_sim, 4),
                "bear_similarity": round(bear_sim, 4),
                "source": source,
                "weight": weight,
            })

        # Weighted aggregate
        total_weight = sum(a["weight"] for a in article_scores)
        if total_weight > 0:
            aggregate = sum(a["score"] * a["weight"] for a in article_scores) / total_weight
        else:
            aggregate = float(np.mean([a["score"] for a in article_scores]))

        aggregate = round(max(-1.0, min(1.0, aggregate)), 3)

        # Confidence based on article count and score variance
        scores_array = [a["score"] for a in article_scores]
        variance = float(np.var(scores_array)) if len(scores_array) > 1 else 1.0
        count_factor = min(1.0, len(article_scores) / 15)  # More articles = higher confidence
        variance_factor = max(0.2, 1.0 - variance)  # Lower variance = higher confidence
        confidence = round(count_factor * variance_factor, 2)

        # Source breakdown
        source_scores: dict[str, list[float]] = {}
        for a in article_scores:
            source_scores.setdefault(a["source"], []).append(a["score"])
        source_breakdown = {s: round(float(np.mean(v)), 3) for s, v in source_scores.items()}

        # Signal classification
        if aggregate > 0.15:
            signal = "BULLISH"
        elif aggregate < -0.15:
            signal = "BEARISH"
        else:
            signal = "NEUTRAL"

        return {
            "ticker": ticker,
            "signal": signal,
            "summary": (
                f"NLP sentiment: {aggregate:+.3f} (confidence: {confidence:.0%}). "
                f"Analyzed {len(article_scores)} articles via transformer embeddings."
            ),
            "aggregate_score": aggregate,
            "confidence": confidence,
            "article_count": len(article_scores),
            "article_scores": article_scores[:10],  # Top 10 for brevity
            "source_breakdown": source_breakdown,
        }

    except Exception as e:
        logger.error(f"NLP sentiment analysis failed for {ticker}: {e}")
        return {
            "ticker": ticker,
            "signal": "ERROR",
            "summary": f"NLP sentiment analysis failed: {e}",
            "aggregate_score": 0.0,
            "confidence": 0.0,
        }


def _source_weight(source: str) -> float:
    """Assign reliability weight based on source type."""
    source_lower = source.lower() if source else ""
    # SEC filings / official sources
    if any(s in source_lower for s in ["sec", "edgar", "10-k", "10-q", "filing"]):
        return 1.0
    # Financial press
    if any(s in source_lower for s in [
        "reuters", "bloomberg", "wsj", "financial times", "barron",
        "cnbc", "marketwatch", "seeking alpha", "motley fool",
    ]):
        return 0.8
    # Mainstream news
    if any(s in source_lower for s in [
        "nytimes", "washington post", "bbc", "cnn", "associated press", "ap news",
    ]):
        return 0.6
    # Social media / forums
    if any(s in source_lower for s in ["reddit", "twitter", "stocktwits", "x.com"]):
        return 0.3
    # Default
    return 0.5
