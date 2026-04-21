"""
Earnings tone analysis tool — Yahoo Finance earnings transcripts.
Scrapes earnings call transcripts from Yahoo Finance and performs
keyword-based tone analysis (CONFIDENT / CAUTIOUS / EVASIVE).
Full transcripts are cached in GCS so paywalled content remains accessible.
The orchestrator's Gemini model provides a deeper tone assessment.
"""

import base64
import json
import logging
import re

import httpx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# GCS caching — same bucket as 10-K filings
# ---------------------------------------------------------------------------

def _gcs_path(ticker: str, quarter_label: str) -> str:
    """Build GCS blob path: {TICKER}/transcripts/{YEAR}_Q{Q}.json"""
    # quarter_label is e.g. "Q4 2026"
    parts = quarter_label.split()
    if len(parts) == 2:
        q_num = parts[0].replace("Q", "")
        year = parts[1]
        return f"{ticker.upper()}/transcripts/{year}_Q{q_num}.json"
    return f"{ticker.upper()}/transcripts/{quarter_label.replace(' ', '_')}.json"


def _load_from_gcs(bucket_name: str, blob_path: str) -> dict | None:
    """Try loading a cached transcript from GCS. Returns None on miss."""
    if not bucket_name:
        return None
    try:
        from google.cloud import storage
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_path)
        if blob.exists():
            data = json.loads(blob.download_as_text())
            logger.info("GCS cache hit: gs://%s/%s", bucket_name, blob_path)
            return data
    except Exception as e:
        logger.debug("GCS read failed for %s: %s", blob_path, e)
    return None


def _save_to_gcs(bucket_name: str, blob_path: str, payload: dict) -> None:
    """Upload a transcript JSON to GCS (fire-and-forget)."""
    if not bucket_name:
        return
    try:
        from google.cloud import storage
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_path)
        blob.upload_from_string(
            json.dumps(payload, indent=2), content_type="application/json"
        )
        logger.info("Saved transcript to gs://%s/%s", bucket_name, blob_path)
    except Exception as e:
        logger.warning("GCS write failed for %s: %s", blob_path, e)

# ---------------------------------------------------------------------------
# Keyword-based tone scoring
# ---------------------------------------------------------------------------

_CONFIDENT_PHRASES = [
    "strong growth", "exceeded expectations", "well positioned",
    "we're confident", "we are confident", "record revenue",
    "record quarter", "record results", "beat guidance",
    "above guidance", "strong demand", "robust demand",
    "strong momentum", "significant progress", "outperformed",
    "best quarter", "ahead of plan", "very pleased",
    "accelerating growth", "strong execution", "all-time high",
    "raising guidance", "raising our guidance", "increasing guidance",
    "raise our outlook", "strong pipeline", "healthy demand",
    "better than expected", "strong performance", "impressive results",
]

_CAUTIOUS_PHRASES = [
    "uncertain", "challenging environment", "headwinds",
    "cautious outlook", "monitoring closely", "remain cautious",
    "below expectations", "softness in", "macro uncertainty",
    "cautiously optimistic", "mixed results", "volatile market",
    "tempered expectations", "slower than expected", "inventory build",
    "weaker demand", "cost pressures", "margin pressure",
    "prudent approach", "measured approach", "cautious approach",
    "navigating challenges", "near-term challenges", "normalizing demand",
]

_EVASIVE_PHRASES = [
    "can't comment on that", "cannot comment", "not in a position to",
    "we'll get back to you", "let me redirect", "i'd rather not",
    "we don't disclose", "we're not going to get into",
    "competitive reasons", "as i mentioned earlier",
    "i think the question", "that's a good question but",
    "next question", "we'll have more to share",
    "not going to speculate", "we don't guide on that",
]


def _analyze_tone(text: str) -> dict:
    """Score transcript text for confident/cautious/evasive tone.

    Returns dict with signal, confidence_score (1-10), and evidence.
    """
    lower = text.lower()

    confident_hits = [p for p in _CONFIDENT_PHRASES if p in lower]
    cautious_hits = [p for p in _CAUTIOUS_PHRASES if p in lower]
    evasive_hits = [p for p in _EVASIVE_PHRASES if p in lower]

    c_score = len(confident_hits)
    ca_score = len(cautious_hits)
    e_score = len(evasive_hits)
    total = c_score + ca_score + e_score

    if total == 0:
        return {
            "signal": "CAUTIOUS",
            "management_confidence": 5,
            "tone_evidence": {
                "confident_phrases": [],
                "cautious_phrases": [],
                "evasive_phrases": [],
            },
        }

    # Determine signal
    if e_score >= 3 or (e_score > 0 and e_score >= c_score):
        signal = "EVASIVE"
        confidence = max(1, 3 - e_score)
    elif c_score > ca_score + e_score:
        signal = "CONFIDENT"
        confidence = min(10, 5 + c_score)
    elif ca_score > c_score:
        signal = "CAUTIOUS"
        confidence = max(1, 5 - ca_score + c_score)
    else:
        signal = "CAUTIOUS"
        confidence = 5

    return {
        "signal": signal,
        "management_confidence": confidence,
        "tone_evidence": {
            "confident_phrases": confident_hits[:5],
            "cautious_phrases": cautious_hits[:5],
            "evasive_phrases": evasive_hits[:5],
        },
    }

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}


def _accept_consent(client: httpx.Client, url: str) -> httpx.Response:
    """Handle Yahoo GDPR consent redirect if triggered."""
    resp = client.get(url)
    if "consent.yahoo.com" in str(resp.url):
        hidden = re.findall(
            r'<input[^>]*name="([^"]*)"[^>]*value="([^"]*)"', resp.text
        )
        form_data = {n: v for n, v in hidden}
        form_data["agree"] = "agree"
        resp = client.post(str(resp.url), data=form_data, follow_redirects=True)
    return resp


def _extract_transcript(html: str) -> tuple[str | None, bool]:
    """Extract transcript text from Yahoo Finance earnings call page HTML.
    
    Returns (transcript_text, is_paywalled).
    """
    paras = re.findall(r"<p[^>]*>(.*?)</p>", html, re.DOTALL)
    clean = [re.sub(r"<[^>]+>", "", p).strip() for p in paras if len(p) > 30]

    is_paywalled = any(
        "subscription plan is required" in p.lower() for p in clean
    )

    # Find where the transcript begins
    start_idx = None
    for i, p in enumerate(clean):
        lower = p.lower()
        if any(
            kw in lower
            for kw in [
                "welcome to",
                "earnings conference",
                "earnings call",
                "good afternoon",
                "good morning",
                "good evening",
            ]
        ):
            start_idx = i
            break

    if start_idx is None:
        return None, is_paywalled

    transcript_paras = clean[start_idx:]
    # Filter out navigation/UI artefacts
    filtered = [
        p
        for p in transcript_paras
        if len(p) > 20
        and not any(
            s in p.lower()
            for s in ["skip to", "cookie", "sign in", "trending ticker"]
        )
    ]
    return "\n\n".join(filtered) if filtered else None, is_paywalled


async def get_earnings_tone(ticker: str, api_key: str = "", max_transcripts: int = 4, bucket_name: str = "") -> dict:
    """
    Fetch recent earnings transcripts from Yahoo Finance with GCS caching.
    Full transcripts are saved to GCS on first scrape and loaded from cache
    when Yahoo's paywall blocks access on subsequent runs.
    The api_key parameter is kept for backward compatibility but unused.
    """
    try:
        client = httpx.Client(
            timeout=30, headers=_HEADERS, follow_redirects=True
        )

        # Step 1: Accept GDPR consent if needed
        _accept_consent(client, f"https://finance.yahoo.com/quote/{ticker}/")

        # Step 2: Load the earnings page to discover transcript links
        resp = client.get(
            f"https://finance.yahoo.com/quote/{ticker}/earnings/"
        )
        pattern = rf"({re.escape(ticker)}-Q\d-\d{{4}}-earnings_call-\d+\.html)"
        links = list(dict.fromkeys(re.findall(pattern, resp.text)))  # dedupe, preserve order

        if not links:
            return {
                "ticker": ticker,
                "signal": "N/A",
                "available": False,
                "transcript_excerpt": "",
                "transcripts": [],
                "summary": "No earnings transcripts found on Yahoo Finance.",
            }

        # Step 3: Fetch transcripts (most recent first, up to max_transcripts)
        transcripts = []
        # Allocate excerpt budget: most recent gets more, older ones less
        excerpt_limits = [8000] + [3000] * (max_transcripts - 1)

        for i, link in enumerate(links[:max_transcripts]):
            try:
                m = re.match(r"\w+-Q(\d)-(\d{4})-earnings_call-\d+\.html", link)
                quarter = f"Q{m.group(1)} {m.group(2)}" if m else f"Transcript {i+1}"
                blob_path = _gcs_path(ticker, quarter)

                # --- Check GCS cache first ---
                cached = _load_from_gcs(bucket_name, blob_path)
                if cached and cached.get("content"):
                    full_text = cached["content"]
                    transcripts.append({
                        "quarter": quarter,
                        "transcript_length": len(full_text),
                        "transcript_excerpt": full_text[:excerpt_limits[i]],
                        "paywalled": False,
                        "source": "gcs_cache",
                    })
                    continue

                # --- Scrape Yahoo Finance ---
                transcript_url = (
                    f"https://finance.yahoo.com/quote/{ticker}/earnings/{link}"
                )
                resp2 = client.get(transcript_url)
                text, paywalled = _extract_transcript(resp2.text)

                if text and not paywalled:
                    transcripts.append({
                        "quarter": quarter,
                        "transcript_length": len(text),
                        "transcript_excerpt": text[:excerpt_limits[i]],
                        "paywalled": False,
                        "source": "yahoo",
                    })
                    # Save full transcript to GCS for future use
                    _save_to_gcs(bucket_name, blob_path, {
                        "ticker": ticker,
                        "quarter": quarter,
                        "year": m.group(2) if m else "",
                        "quarter_num": m.group(1) if m else "",
                        "content": text,
                        "source": "yahoo_finance",
                        "transcript_length": len(text),
                    })
                elif paywalled:
                    # Try GCS one more time with relaxed check (cached may lack "content")
                    transcripts.append({
                        "quarter": quarter,
                        "transcript_length": 0,
                        "transcript_excerpt": "",
                        "paywalled": True,
                        "source": "yahoo_paywalled",
                    })
                elif text:
                    # Has text but also flagged paywalled (partial)
                    transcripts.append({
                        "quarter": quarter,
                        "transcript_length": len(text),
                        "transcript_excerpt": text[:excerpt_limits[i]],
                        "paywalled": True,
                        "source": "yahoo_partial",
                    })
            except Exception as e:
                logger.warning("Failed to fetch transcript %s: %s", link, e)

        if not transcripts:
            return {
                "ticker": ticker,
                "signal": "N/A",
                "available": False,
                "transcript_excerpt": "",
                "transcripts": [],
                "summary": "Transcript pages loaded but text extraction failed.",
            }

        full = [t for t in transcripts if not t.get("paywalled")]
        paywalled_list = [t for t in transcripts if t.get("paywalled")]
        cached_list = [t for t in full if t.get("source") == "gcs_cache"]
        quarters_full = [t["quarter"] for t in full]
        quarters_locked = [t["quarter"] for t in paywalled_list]
        total_chars = sum(t["transcript_length"] for t in transcripts)

        # Analyze tone from all available full transcripts
        all_text = " ".join(
            t["transcript_excerpt"] for t in full if t["transcript_excerpt"]
        )
        tone = _analyze_tone(all_text) if all_text else {
            "signal": "N/A",
            "management_confidence": 0,
            "tone_evidence": {"confident_phrases": [], "cautious_phrases": [], "evasive_phrases": []},
        }

        parts = []
        if full:
            label = f"{len(full)} full transcript(s) ({', '.join(quarters_full)})"
            if cached_list:
                label += f" [{len(cached_list)} from cache]"
            parts.append(label)
        if quarters_locked:
            parts.append(
                f"{len(quarters_locked)} paywalled ({', '.join(quarters_locked)})"
            )

        return {
            "ticker": ticker,
            "signal": tone["signal"],
            "management_confidence": tone["management_confidence"],
            "tone_evidence": tone["tone_evidence"],
            "quarter": transcripts[0]["quarter"],
            "transcript_length": transcripts[0]["transcript_length"],
            "transcript_excerpt": transcripts[0]["transcript_excerpt"],
            "transcripts": transcripts,
            "available": True,
            "summary": (
                f"Tone: {tone['signal']} (confidence {tone['management_confidence']}/10). "
                f"{'; '.join(parts)} for {ticker}. "
                f"{total_chars} total chars from Yahoo Finance."
            ),
        }

    except Exception as e:
        logger.error("Failed to fetch earnings transcript for %s: %s", ticker, e)
        return {
            "ticker": ticker,
            "signal": "ERROR",
            "available": False,
            "transcript_excerpt": "",
            "summary": f"Error fetching transcript: {e}",
        }


# ── phase-4.14.14 (MF-31): document-block builder for downstream Claude prompts ──
#
# Wraps the earnings-call transcript (or summary when paywalled) as
# an Anthropic document content block with citations.enabled=True.
# Used by callers that want the synthesis model to return cited_text
# attributions pointing back to specific transcript quotes.
#
# Pure data utility -- NO API call. Must not be combined with
# response_schema / output_config.format in the same request (that
# combination 400s on Claude; guarded in ClaudeClient at 4.14.9).

def build_earnings_document_block(ticker: str, result: dict) -> dict:
    """Return a Claude document block wrapping the earnings transcript."""
    transcript = (
        result.get("transcript_excerpt")
        or result.get("summary", "")
    )
    return {
        "type": "document",
        "source": {
            "type": "text",
            "media_type": "text/plain",
            "data": transcript or f"No transcript available for {ticker}.",
        },
        "title": f"Earnings call transcript -- {ticker}",
        "citations": {"enabled": True},
    }


# phase-4.14.17 (MF-34b): PDF-native document block for earnings
# decks. Claude ingests PDF pages directly (charts + tables preserved)
# without text extraction. cache_control:ephemeral with ttl:"1h"
# matches the project-wide convention from llm_client.py.
def build_earnings_pdf_block(ticker: str, pdf_bytes: bytes) -> dict:
    """Return a Claude PDF-native document block for an earnings deck."""
    return {
        "type": "document",
        "source": {
            "type": "base64",
            "media_type": "application/pdf",
            "data": base64.b64encode(pdf_bytes).decode("ascii"),
        },
        "title": f"Earnings deck PDF -- {ticker}",
        "cache_control": {"type": "ephemeral", "ttl": "1h"},
        "citations": {"enabled": True},
    }

