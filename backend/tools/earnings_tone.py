"""
Earnings tone analysis tool — Yahoo Finance earnings transcripts.
Scrapes the latest earnings call transcript from Yahoo Finance.
The orchestrator's Gemini model handles the tone scoring.
"""

import logging
import re

import httpx

logger = logging.getLogger(__name__)

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


async def get_earnings_tone(ticker: str, api_key: str = "", max_transcripts: int = 4) -> dict:
    """
    Fetch recent earnings transcripts from Yahoo Finance.
    Returns the latest plus up to max_transcripts-1 older ones for trend analysis.
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
                transcript_url = (
                    f"https://finance.yahoo.com/quote/{ticker}/earnings/{link}"
                )
                resp2 = client.get(transcript_url)
                text, paywalled = _extract_transcript(resp2.text)

                m = re.match(r"\w+-Q(\d)-(\d{4})-earnings_call-\d+\.html", link)
                quarter = f"Q{m.group(1)} {m.group(2)}" if m else f"Transcript {i+1}"

                if text:
                    transcripts.append({
                        "quarter": quarter,
                        "transcript_length": len(text),
                        "transcript_excerpt": text[:excerpt_limits[i]],
                        "paywalled": paywalled,
                    })
                elif paywalled:
                    # Note the quarter exists but is locked
                    transcripts.append({
                        "quarter": quarter,
                        "transcript_length": 0,
                        "transcript_excerpt": "",
                        "paywalled": True,
                    })
            except Exception as e:
                logger.warning("Failed to fetch transcript %s: %s", link, e)

        if not transcripts:
            return {
                "ticker": ticker,
                "available": False,
                "transcript_excerpt": "",
                "transcripts": [],
                "summary": "Transcript pages loaded but text extraction failed.",
            }

        full = [t for t in transcripts if not t.get("paywalled")]
        paywalled = [t for t in transcripts if t.get("paywalled")]
        quarters_full = [t["quarter"] for t in full]
        quarters_locked = [t["quarter"] for t in paywalled]
        total_chars = sum(t["transcript_length"] for t in transcripts)

        parts = []
        if full:
            parts.append(
                f"{len(full)} full transcript(s) ({', '.join(quarters_full)})"
            )
        if quarters_locked:
            parts.append(
                f"{len(quarters_locked)} paywalled ({', '.join(quarters_locked)})"
            )

        return {
            "ticker": ticker,
            "quarter": transcripts[0]["quarter"],
            "transcript_length": transcripts[0]["transcript_length"],
            "transcript_excerpt": transcripts[0]["transcript_excerpt"],
            "transcripts": transcripts,
            "available": True,
            "summary": (
                f"{'; '.join(parts)} for {ticker}. "
                f"{total_chars} total chars from Yahoo Finance."
            ),
        }

    except Exception as e:
        logger.error("Failed to fetch earnings transcript for %s: %s", ticker, e)
        return {
            "ticker": ticker,
            "available": False,
            "transcript_excerpt": "",
            "summary": f"Error fetching transcript: {e}",
        }
