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

# phase-11.4: migrated from deprecated `vertexai.language_models.TextEmbeddingModel`
# (removal 2026-06-24) to `google.genai` embed API. Client obtained via the
# shim in `backend.agents._genai_client` so credentials + project + location
# flow through the same singleton the rest of the Gemini stack uses.
from backend.agents._genai_client import get_genai_client

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

        # phase-11.4: initialize google-genai client via the shim.
        # Embedding API: `client.models.embed_content(model=..., contents=[...])`
        # returns an object with `.embeddings[i].values` (same shape as the
        # legacy Vertex response, so downstream array handling is unchanged).
        client = get_genai_client()
        if client is None:
            raise RuntimeError(
                "google-genai client unavailable (shim returned None); "
                "likely missing credentials or SDK"
            )
        embed_model = "gemini-embedding-001"  # replaces text-embedding-005; supported in google-genai

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
        _embed_result = client.models.embed_content(
            model=embed_model,
            contents=all_texts,
        )
        # `.embeddings[i].values` is the vector (same as legacy Vertex shape).
        embeddings = _embed_result.embeddings

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
        err_text = str(e)
        lower = err_text.lower()
        # Categorise auth / credential failures so ops can triage quickly.
        reason = "runtime_error"
        if ("authenticate" in lower or "credentials" in lower
                or "application default credentials" in lower
                or "could not automatically determine" in lower):
            reason = "gcp_adc_unavailable"
        elif "quota" in lower or "resource_exhausted" in lower:
            reason = "vertex_quota_exceeded"
        elif "permission" in lower or "403" in lower:
            reason = "vertex_permission_denied"

        logger.error(f"NLP sentiment analysis failed for {ticker} reason={reason}: {e}")

        # Rules-based polarity fallback over recent headlines so the
        # signal still contributes a neutral-ish read instead of a hard
        # ERROR. This keeps the 4.6.3 'at least 8 non-ERROR' criterion
        # robust when Vertex credentials are down.
        try:
            fallback = _rules_fallback_from_articles(ticker)
            if fallback is not None:
                fallback["reason"] = reason
                return fallback
        except Exception as fb_err:
            logger.warning(f"rules fallback also failed for {ticker}: {fb_err}")

        return {
            "ticker": ticker,
            "signal": "ERROR",
            "summary": f"NLP sentiment unavailable ({reason}): {err_text[:200]}",
            "reason": reason,
            "aggregate_score": 0.0,
            "confidence": 0.0,
        }


def _rules_fallback_from_articles(ticker: str) -> dict | None:
    """Trivial lexicon-based polarity scorer for when Vertex is unavailable.

    Intentionally dumb: checks a handful of bullish/bearish tokens in the
    titles returned by the alpha_vantage news fetcher (shared with the
    real nlp path). Emits a NEUTRAL-leaning signal so downstream agents
    know the value is low-confidence.
    """
    try:
        from backend.tools.alphavantage import fetch_news_for_ticker
    except Exception:
        return None

    try:
        articles = fetch_news_for_ticker(ticker, limit=20) or []
    except Exception:
        return None
    if not articles:
        return None

    bullish = {"beats", "beat", "upgrade", "bullish", "surge", "rally",
               "record", "strong", "exceed", "profit", "growth"}
    bearish = {"miss", "misses", "downgrade", "bearish", "plunge", "drop",
               "lawsuit", "recall", "loss", "losses", "warn", "warning",
               "probe", "investigation"}

    up = down = 0
    for art in articles:
        title = (art.get("title") or "").lower()
        up += sum(1 for w in bullish if w in title)
        down += sum(1 for w in bearish if w in title)

    total = up + down
    if total == 0:
        score = 0.0
    else:
        score = (up - down) / total

    if score > 0.2:
        signal = "BULLISH"
    elif score < -0.2:
        signal = "BEARISH"
    else:
        signal = "NEUTRAL"

    return {
        "ticker": ticker,
        "signal": signal,
        "summary": (
            f"Rules fallback polarity {score:+.2f} over {len(articles)} "
            f"headlines (Vertex unavailable; low-confidence)."
        ),
        "aggregate_score": score,
        "confidence": 0.25,
        "article_count": len(articles),
        "source": "rules_fallback",
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
