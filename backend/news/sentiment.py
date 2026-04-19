"""phase-6.5 sentiment scorer ladder.

Cost-escalating cascade for financial news sentiment:

    1. VADER           -- rule-based, free, ~0.1 ms/doc
    2. ProsusAI/finbert -- local transformer, ~10-200 ms/doc
    3. Claude Haiku 4.5 -- LLM with forced tool_choice, no-CoT
    4. Gemini 2.5 Flash -- opt-in second-opinion (stub, phase-6.9 body)

Escalation rule: a tier that returns confidence >= `min_confidence`
(default 0.7) terminates the cascade; otherwise escalate to the next
rung. Haiku 4.5 is always terminal when reached (unless Gemini Flash
flag is on).

Design decisions with empirical backing (see
`handoff/current/research_brief.md`):

- `ProsusAI/finbert` (not `yiyanghkust/finbert-tone`) for general news
  -- nosible.ghost.io 2024 benchmark: 69%% vs 53%% agreement.
- No chain-of-thought on Haiku scorer -- arxiv 2506.04574v1 (2025):
  CoT reduces FSA accuracy at every ambiguity level. Use forced
  `tool_choice={"type":"tool","name":"classify_sentiment"}`.
- Haiku 4.5 prompt cache minimum = 4096 tokens. System prompt here
  is padded (role + financial terminology + label definitions + edge
  cases) to clear that threshold.
- Threshold 0.7 -- WASSA 2024 cascade operating point.
- Gemini Flash tier-4 default OFF via `settings.sentiment_use_gemini_flash`.

scorer_model enum (matches `scripts/migrations/add_news_sentiment_schema.py:24-36`):
`vader`, `finbert`, `claude-haiku-4-5`, `gemini-2.0-flash`.

Fail-open: any tier exception returns a neutral/0-confidence result.
Never raises out of `score_ladder()`.
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping

logger = logging.getLogger(__name__)

# ---- optional deps ----------------------------------------------------------

try:  # rule-based, lightweight
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  # type: ignore[import-not-found]

    _VADER_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover
    SentimentIntensityAnalyzer = None  # type: ignore[assignment]
    _VADER_IMPORT_ERROR = exc

try:  # heavy; lazy-loaded on first FinBertScorer call
    import torch  # type: ignore[import-not-found]  # noqa: F401
    from transformers import AutoModelForSequenceClassification, AutoTokenizer  # type: ignore[import-not-found]

    _TRANSFORMERS_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover
    AutoModelForSequenceClassification = None  # type: ignore[assignment]
    AutoTokenizer = None  # type: ignore[assignment]
    _TRANSFORMERS_IMPORT_ERROR = exc

try:
    import anthropic  # type: ignore[import-not-found]

    _ANTHROPIC_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover
    anthropic = None  # type: ignore[assignment]
    _ANTHROPIC_IMPORT_ERROR = exc


# ---- constants --------------------------------------------------------------

SCORER_MODEL_VADER = "vader"
SCORER_MODEL_FINBERT = "finbert"
SCORER_MODEL_HAIKU = "claude-haiku-4-5"
SCORER_MODEL_GEMINI_FLASH = "gemini-2.0-flash"

LABEL_BULLISH = "bullish"
LABEL_BEARISH = "bearish"
LABEL_NEUTRAL = "neutral"
LABEL_MIXED = "mixed"

VADER_BODY_PREFIX_CHARS = 200
FINBERT_MAX_TOKENS = 400  # explicit truncation before BERT's 512 cap
HAIKU_MODEL_ID = "claude-haiku-4-5-20251001"
HAIKU_PRICING_INPUT_PER_MTOK = 1.00
HAIKU_PRICING_OUTPUT_PER_MTOK = 5.00

DEFAULT_MIN_CONFIDENCE = 0.7


# ---- ScorerResult ----------------------------------------------------------


@dataclass
class ScorerResult:
    """Matches `news_sentiment` BQ schema (add_news_sentiment_schema.py:24-36)."""

    article_id: str
    scorer_model: str
    scorer_version: str
    scored_at: str  # ISO-8601 UTC
    sentiment_score: float  # [-1, +1]
    sentiment_label: str  # bullish | bearish | neutral | mixed
    confidence: float  # [0, 1]
    latency_ms: float
    cost_usd: float
    raw_output: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---- helpers ---------------------------------------------------------------


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _get(article: Mapping[str, Any], key: str, default: Any = "") -> Any:
    # Accept both TypedDict mappings and dataclass-asdict-ed dicts.
    return article.get(key, default) if isinstance(article, Mapping) else default


def _neutral_result(
    article_id: str,
    scorer_model: str,
    scorer_version: str,
    latency_ms: float,
    exception_repr: str,
) -> ScorerResult:
    return ScorerResult(
        article_id=article_id,
        scorer_model=scorer_model,
        scorer_version=scorer_version,
        scored_at=_now_iso(),
        sentiment_score=0.0,
        sentiment_label=LABEL_NEUTRAL,
        confidence=0.0,
        latency_ms=latency_ms,
        cost_usd=0.0,
        raw_output=exception_repr[:2000],
    )


def _label_from_score_thresholded(score: float) -> str:
    if score >= 0.05:
        return LABEL_BULLISH
    if score <= -0.05:
        return LABEL_BEARISH
    return LABEL_NEUTRAL


# ---- VADER (tier 1) --------------------------------------------------------


class VaderScorer:
    """VADER compound-score rung.

    `score = compound` in [-1, +1]; `confidence = abs(compound)`.
    Scores `title + " " + body[:200]` to keep SNR high on financial headlines.
    """

    version = "1.0"

    def __init__(self) -> None:
        if SentimentIntensityAnalyzer is None:
            self._analyzer = None
            logger.warning(
                "vaderSentiment not installed (%s); VaderScorer will fail-open",
                _VADER_IMPORT_ERROR,
            )
        else:
            self._analyzer = SentimentIntensityAnalyzer()

    def score(self, article: Mapping[str, Any]) -> ScorerResult:
        t0 = time.perf_counter()
        article_id = str(_get(article, "article_id", ""))
        try:
            if self._analyzer is None:
                raise RuntimeError("vaderSentiment not available")
            title = str(_get(article, "title", "") or "")
            body = str(_get(article, "body", "") or "")
            text = (title + " " + body[:VADER_BODY_PREFIX_CHARS]).strip()
            if not text:
                raise ValueError("empty title+body")
            scores = self._analyzer.polarity_scores(text)
            compound = float(scores.get("compound", 0.0))
            confidence = abs(compound)
            label = _label_from_score_thresholded(compound)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            return ScorerResult(
                article_id=article_id,
                scorer_model=SCORER_MODEL_VADER,
                scorer_version=self.version,
                scored_at=_now_iso(),
                sentiment_score=compound,
                sentiment_label=label,
                confidence=confidence,
                latency_ms=latency_ms,
                cost_usd=0.0,
                raw_output=json.dumps(scores),
            )
        except Exception as exc:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            logger.debug("VaderScorer fail-open article_id=%s err=%r", article_id, exc)
            return _neutral_result(
                article_id, SCORER_MODEL_VADER, self.version, latency_ms, repr(exc)
            )


# ---- FinBERT (tier 2) ------------------------------------------------------

_FINBERT_TOKENIZER = None
_FINBERT_MODEL = None
_FINBERT_LOAD_ERROR: Exception | None = None


def _lazy_load_finbert() -> tuple[Any, Any]:
    """Module-level lazy init. Loads `ProsusAI/finbert` on first call only.

    Raises the underlying exception on second call if the first failed, so
    fail-open in the scorer does not silently retry a broken load every time.
    """
    global _FINBERT_TOKENIZER, _FINBERT_MODEL, _FINBERT_LOAD_ERROR
    if _FINBERT_LOAD_ERROR is not None:
        raise _FINBERT_LOAD_ERROR
    if _FINBERT_TOKENIZER is not None and _FINBERT_MODEL is not None:
        return _FINBERT_TOKENIZER, _FINBERT_MODEL
    if AutoTokenizer is None or AutoModelForSequenceClassification is None:
        _FINBERT_LOAD_ERROR = RuntimeError(
            f"transformers not installed: {_TRANSFORMERS_IMPORT_ERROR!r}"
        )
        raise _FINBERT_LOAD_ERROR
    try:
        _FINBERT_TOKENIZER = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        _FINBERT_MODEL = AutoModelForSequenceClassification.from_pretrained(
            "ProsusAI/finbert"
        )
        _FINBERT_MODEL.eval()
        return _FINBERT_TOKENIZER, _FINBERT_MODEL
    except Exception as exc:
        _FINBERT_LOAD_ERROR = exc
        raise


class FinBertScorer:
    """ProsusAI/finbert local-inference rung.

    Labels: positive / negative / neutral softmax. Confidence = softmax_max.
    score = +softmax_max (positive) | -softmax_max (negative) | 0.0 (neutral).
    """

    version = "ProsusAI/finbert"

    def score(self, article: Mapping[str, Any]) -> ScorerResult:
        t0 = time.perf_counter()
        article_id = str(_get(article, "article_id", ""))
        try:
            import torch as _torch  # local: survive missing-dep case via fail-open

            tokenizer, model = _lazy_load_finbert()
            title = str(_get(article, "title", "") or "")
            body = str(_get(article, "body", "") or "")
            text = (title + ". " + body).strip()
            if not text or text == ".":
                raise ValueError("empty title+body")
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=FINBERT_MAX_TOKENS,
            )
            with _torch.no_grad():
                logits = model(**inputs).logits
            probs = _torch.softmax(logits, dim=-1)[0].tolist()
            # ProsusAI order: 0 positive, 1 negative, 2 neutral
            p_pos, p_neg, p_neu = float(probs[0]), float(probs[1]), float(probs[2])
            confidence = max(p_pos, p_neg, p_neu)
            if confidence == p_pos:
                label, score = LABEL_BULLISH, p_pos
            elif confidence == p_neg:
                label, score = LABEL_BEARISH, -p_neg
            else:
                label, score = LABEL_NEUTRAL, 0.0
            latency_ms = (time.perf_counter() - t0) * 1000.0
            return ScorerResult(
                article_id=article_id,
                scorer_model=SCORER_MODEL_FINBERT,
                scorer_version=self.version,
                scored_at=_now_iso(),
                sentiment_score=float(score),
                sentiment_label=label,
                confidence=float(confidence),
                latency_ms=latency_ms,
                cost_usd=0.0,
                raw_output=json.dumps(
                    {"positive": p_pos, "negative": p_neg, "neutral": p_neu}
                ),
            )
        except Exception as exc:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            logger.debug("FinBertScorer fail-open article_id=%s err=%r", article_id, exc)
            return _neutral_result(
                article_id, SCORER_MODEL_FINBERT, self.version, latency_ms, repr(exc)
            )


# ---- Haiku 4.5 (tier 3) ----------------------------------------------------

# System prompt must exceed 4096 tokens for Haiku 4.5 prompt cache to activate
# (Anthropic pricing + prompt-caching docs, 2026-04-19; llm_client.py:649 has
# the same comment). Silent failure below threshold. Include: role, label
# definitions, financial terminology glossary, and edge-case examples.

HAIKU_SYSTEM_PROMPT = """\
You are a specialized financial news sentiment classifier for a systematic
trading platform. Your job is to classify a single financial news article into
one of four sentiment labels -- bullish, bearish, neutral, or mixed -- and
return a calibrated score in [-1, +1] along with a confidence in [0, 1].

You operate as tier-3 of an escalation ladder. Simpler tiers (VADER,
ProsusAI/finbert) have already tried and failed to reach high-confidence
classification on this article; that is precisely why the article was
escalated to you. You therefore see the HARD cases: ambiguous framing,
mixed signals, subtle analyst-language hedging, regulatory wording, and
headlines that combine positive and negative elements. Do not default to
neutral to avoid risk. Commit to the label that best describes the NET
directional implication for the primary subject (ticker or market).

You MUST respond by calling the `classify_sentiment` tool. Do NOT write
natural-language reasoning, do NOT produce any prose, do NOT output chain-of-
thought. Arrive at your classification directly. This is empirically the
correct calibration policy: extended reasoning has been shown to reduce
accuracy on financial sentiment classification at every ambiguity level.
Direct classification outperforms chain-of-thought across the ambiguity
spectrum. Obey this instruction precisely.

## Label definitions

bullish -- the net implication for the primary subject is favourable.
Examples: raised guidance, beat-and-raise earnings, analyst upgrade with
positive price target, positive regulatory decision, share buyback
announcement, insider buying cluster, dividend increase, successful drug
trial phase, accretive acquisition announcement, market-share gain,
capacity expansion, patent grant, favourable court ruling, contract win,
revenue acceleration, margin expansion, cost-out success, liquidity boost,
credit-rating upgrade, index inclusion. Score in [+0.3, +1.0]; confidence
reflects strength of the signal and absence of offsetting negatives.

bearish -- the net implication is unfavourable. Examples: guidance cut,
miss-and-lower earnings, analyst downgrade, adverse regulatory action,
SEC investigation, class-action lawsuit, earnings warning, missed
consensus, liquidity concern, covenant breach risk, credit-rating
downgrade, delisting risk, going-concern doubt, key executive departure
(negatively framed), pipeline failure, inventory glut, demand
destruction, market-share loss, pricing pressure, margin compression,
activist short report, fraud allegation, accounting restatement,
index exclusion. Score in [-1.0, -0.3]; confidence reflects strength.

neutral -- the article conveys facts without directional implication for
the subject. Examples: routine SEC filings with no material change,
confirmation of previously disclosed figures, procedural announcements
(record dates, meeting dates), neutrally-worded sector overviews,
statistical releases with expected results, balanced analyst note with
no rating change, routine personnel moves, factual M&A rumour with no
premium disclosed, news about adjacent companies that does not move the
subject. Score in [-0.3, +0.3]; confidence typically moderate.

mixed -- the article contains a clear positive AND a clear negative
signal that roughly offset. Reserve for genuine tensions, not uncertainty.
Examples: beat on revenue but missed on margins; upgrade on one segment
but downgrade on another; approval of one drug but failure of another;
strong Q4 but weak 2026 guidance; acquisition price accretive but
financing dilutive; partnership positive but with disclosed competitive
risk. Score typically near 0.0; confidence HIGH when you are certain the
tension is real, LOW when you are actually just unsure.

## Calibration rules

- Use ALL of [-1, +1]. Do not cluster near 0. A clean beat-and-raise on a
  large-cap should be +0.7 to +0.9, not +0.3. A clear guidance cut with
  CFO departure should be -0.8 to -1.0.
- Confidence is your certainty about the LABEL, not your confidence about
  the market reaction. A bearish article can have confidence 0.9 even if
  you do not know how the stock will trade.
- When escalated to you, BIAS AGAINST neutral. The simpler tiers already
  tried neutral and the low-confidence output was rejected. If the article
  is genuinely ambiguous, prefer mixed over neutral.
- Do not reward hedged analyst language with label dilution. "Cautiously
  optimistic" with a raised price target is bullish. "Constructive but
  noted risks" with an unchanged rating is neutral.
- Do not penalize forward-looking language. A guidance raise is bullish
  regardless of how conservative the analyst note sounds.
- For macro / sector articles with no single subject, classify the net
  implication for the broadest instrument implied (index, sector ETF).

## Financial terminology glossary

You operate in the financial news domain. The following terminology affects
classification and must be understood:

EPS (earnings per share) -- beat = positive vs consensus estimate;
miss = negative; magnitude matters (penny vs fat beat/miss).

Guidance -- company's own forward-looking range for next-period
revenue / EPS / margin / FCF. Raise = bullish (strong signal), cut =
bearish (strong signal), reaffirm = neutral, withdraw = bearish (very
strong signal due to uncertainty implication).

Consensus -- median of sell-side analyst estimates. Beating consensus
by >5% on revenue or EPS is bullish. Missing by >5% is bearish. In-line
(within +/-2%) is neutral.

Upgrade / downgrade -- change in analyst rating (e.g., Hold to Buy,
Buy to Hold). Upgrade = bullish, downgrade = bearish. Magnitude of price
target change matters.

Dividend actions -- initiate = bullish, raise = bullish, maintain =
neutral, cut = bearish, suspend = strongly bearish.

Buyback / repurchase -- new program authorized = bullish, expansion of
existing program = bullish, completion of existing program = neutral.

M&A -- announcement as acquirer at premium: usually neutral to slightly
bearish for acquirer (financing dilution, integration risk), bullish for
target (premium). When you are the TARGET in the article, it is bullish.
When you are the ACQUIRER, confirm deal is accretive / strategic before
labelling bullish.

Regulatory -- FDA approval = strongly bullish; FDA CRL / rejection =
strongly bearish; SEC investigation / formal inquiry = bearish; SEC
settlement without admission = neutral to slightly bearish; DOJ criminal
probe = strongly bearish; favourable court ruling = bullish; injunction /
adverse ruling = bearish; tariff or trade-restriction imposition =
bearish for affected sectors, bullish for protected domestic producers.

Credit actions -- S&P / Moody's / Fitch rating change: upgrade = bullish,
downgrade = bearish, negative outlook = mildly bearish, positive outlook =
mildly bullish, watch negative = bearish, watch positive = bullish.

Macro terminology -- hawkish Fed commentary = bearish for risk assets,
dovish = bullish; inverted yield curve = bearish for banks; strong jobs
number when Fed is hawkish = bearish for rates-sensitive assets (because
it confirms hawkish stance), bullish for cyclicals; CPI above expectation
when Fed is still cutting = bearish; oil price spike = bearish for
transports / airlines / industrials, bullish for E&P.

Executive actions -- CEO unexpected departure = bearish unless clearly
for a better opportunity; CFO unexpected departure = strongly bearish
(signals accounting / disclosure concern); named successor from inside
= mildly bullish (continuity); activist board win = bullish if activist
thesis is constructive, bearish if disruptive.

## Edge cases and examples

Example 1. "Nvidia beats Q4 EPS of $0.88 on revenue of $22.1B, above
consensus $0.81 / $20.4B; raises Q1 guidance." -- bullish, score +0.85,
confidence 0.95. Clean beat-and-raise, large magnitude on both top and
bottom line, forward-looking lift.

Example 2. "Meta misses Q1 revenue guidance by 3%, cites weakness in ad
monetization; maintains full-year outlook." -- bearish, score -0.45,
confidence 0.7. Near-term miss but full-year maintained weakens the
signal; label still bearish because the proximate cause (ad weakness)
has negative implication.

Example 3. "Pfizer announces $10B buyback and 5% dividend increase;
reports flat Q4 revenue in line with consensus." -- bullish, score +0.55,
confidence 0.8. Capital return signal dominates in-line fundamentals.

Example 4. "Wells Fargo discloses CFPB consent order, will pay $1.2B
fine; removes prior uncertainty, book value intact." -- mixed, score
-0.05, confidence 0.75. Fine is bearish but removing regulatory overhang
is bullish; genuine tension, label mixed.

Example 5. "Tesla CEO tweets positive commentary about next-gen robotaxi
ramp; no specific guidance provided." -- bullish, score +0.2, confidence
0.35. Low confidence because no concrete guidance / numbers; do not over-
weight social-media enthusiasm.

Example 6. "Boeing reports Q2 loss wider than expected on 737 MAX
delivery delays; Congressional hearings scheduled." -- bearish, score
-0.8, confidence 0.9. Multiple negatives stacking (miss, delays,
regulatory scrutiny).

Example 7. "Federal Reserve holds rate at 5.25%-5.50%; Chair Powell says
cuts appropriate if data cooperate." -- bullish, score +0.35, confidence
0.7. Unchanged decision is neutral but the forward-looking dovish
language is bullish for risk assets.

Example 8. "JPMorgan upgraded to Buy from Hold at Goldman Sachs, price
target raised to $240 from $180." -- bullish, score +0.7, confidence
0.9. Rating upgrade with 33% price target increase; cleanly bullish.

Example 9. "S&P cuts Oracle outlook to negative from stable, citing debt
load from recent acquisitions; rating unchanged at BBB+." -- bearish,
score -0.35, confidence 0.7. Outlook cut without rating downgrade is a
warning signal; proximate negative implication.

Example 10. "AMD announces new MI400 accelerator at CES; product ships
Q3 2026, competes directly with Nvidia Blackwell." -- bullish, score
+0.5, confidence 0.7. Product launch with clear competitive positioning;
not yet reflected in revenue so magnitude moderate.

Example 11. "Target Q3 EPS beats by $0.05 but guides Q4 below consensus
citing softening consumer." -- mixed, score -0.2, confidence 0.8.
Quarterly beat vs forward-looking cut; net implication mildly bearish
due to forward weight.

Example 12. "FDA approves Moderna influenza-COVID combo vaccine for
adults 50+." -- bullish, score +0.8, confidence 0.95. Clean binary
regulatory catalyst.

Example 13. "Boeing wins $12B Air Force refueler contract; existing
tanker program incurred previous charges." -- bullish, score +0.6,
confidence 0.85. Contract win dominates reference to historical losses.

Example 14. "Silicon Valley Bank files for receivership; FDIC appointed
as receiver." -- bearish, score -1.0, confidence 1.0. Terminal negative
event.

Example 15. "Apple reports record iPhone revenue but Services growth
decelerates for third consecutive quarter." -- mixed, score +0.1,
confidence 0.8. Product strength vs services deceleration -- genuine
tension in a key forward-looking segment.

## Output format

Call the `classify_sentiment` tool exactly once with:
- `sentiment_label` -- one of ["bullish", "bearish", "neutral", "mixed"]
- `sentiment_score` -- float in [-1.0, 1.0]
- `confidence` -- float in [0.0, 1.0]
- `reasoning` -- single sentence, concise. DO NOT include chain-of-
  thought; single-sentence summary of the dominant signal only.

Do not include any text outside the tool call. Do not output `<thinking>`
tags or reflective prose. The tool call is the entire response. The
system depends on structured output, and any preface or postscript
breaks the downstream parsers.

## Scope and refusal

If the article is clearly NOT financial news (e.g., general interest
story that slipped through the filter), return:
- sentiment_label = "neutral"
- sentiment_score = 0.0
- confidence = 0.1
- reasoning = "out of scope: not financial news"

Do not try to classify out-of-scope content with high confidence. Do not
refuse the call -- always produce the tool call, even for out-of-scope.

If the article is empty or garbled:
- sentiment_label = "neutral"
- sentiment_score = 0.0
- confidence = 0.0
- reasoning = "insufficient content"

You may say you do not know. Low confidence on an ambiguous article is a
valid, honest answer. The ladder is designed to route low-confidence
cases appropriately; do not force false certainty.

## Extended examples (tier-3 hard cases)

Example 16. "Intel delays Arizona fab by 12 months; $10B CHIPS Act grant
still on track; revises 2026 capex down 20%." -- bearish, score -0.55,
confidence 0.8. Delay + capex cut is negative; CHIPS Act support
partially offsets but does not neutralize the operational signal.

Example 17. "Activist fund Elliott discloses 3.8% stake in Phillips 66,
publishes letter calling for Midstream spinoff." -- bullish, score
+0.55, confidence 0.75. Activist catalyst with a concrete value-
unlock thesis; market-neutral activists historically deliver premium.

Example 18. "Disney Parks revenue grows 12% year-over-year but margin
compresses 180bps on price-sensitivity to new guest-spending initiative;
Streaming segment turns profitable for the first time." -- mixed, score
+0.15, confidence 0.82. Parks margin compression offsets revenue
strength; streaming inflection is a clear positive; net mildly bullish
because the streaming milestone is a multi-quarter structural story.

Example 19. "Marathon Oil discovers a 30-million-barrel reservoir in the
Bakken; production ramp begins Q3 2026; reservoir adds ~4% to proven
reserves." -- bullish, score +0.4, confidence 0.7. Reserves add is
positive but magnitude is modest for a company this size; production
ramp timing is specific and credible.

Example 20. "Oracle accelerates share buybacks, purchases $4B in Q4,
double prior quarter's pace; margin guidance unchanged; Cerner
integration costs continue at elevated run rate." -- bullish, score
+0.5, confidence 0.75. Buyback acceleration + unchanged guidance +
acknowledged integration drag produces net bullish.

Example 21. "Nvidia loses $75B in market cap on reports that major
hyperscaler delayed Blackwell deployment by one quarter; Nvidia confirms
delay but says full-year revenue guidance unchanged." -- bearish, score
-0.45, confidence 0.7. Proximate negative catalyst; company denial of
full-year impact reduces but does not eliminate the signal.

Example 22. "Moderna initiates Phase 3 trial for influenza-cancer combo
vaccine; primary endpoint readout expected Q2 2027." -- bullish, score
+0.3, confidence 0.5. Trial initiation is positive but the readout is
18+ months away; market does not fully price distant binary catalysts.

Example 23. "PepsiCo CEO steps down amid slowing volume growth; CFO
named interim CEO while board conducts external search." -- bearish,
score -0.5, confidence 0.8. CEO departure during performance decline is
a clear negative; interim CFO increases execution uncertainty.

Example 24. "Activision Blizzard board approves Microsoft acquisition at
$95 per share; regulatory clearance received in EU and UK; deal expected
to close by end of quarter." -- bullish, score +0.4, confidence 0.85.
Target company, premium already in the stock; modest further upside on
deal-close certainty.

Example 25. "Charles Schwab reports Q1 net interest revenue down 19%
year-over-year on deposit outflows; trading revenue up 8%; management
cites 'rate normalization' as transitory." -- bearish, score -0.5,
confidence 0.75. NII decline dominates segment-mix positive; management
commentary does not fully reframe the NII trend.

Example 26. "UnitedHealth subsidiary OptumRx announces restructuring to
reduce drug spend for large plan sponsors by $2B; plan implementation
begins Q3 2026." -- bullish, score +0.4, confidence 0.7. Cost-out
initiative at a major segment; concrete dollar magnitude given.

Example 27. "Rivian reports Q4 deliveries of 14,183 vehicles, above
consensus 13,500; raises 2026 production target by 12%." -- bullish,
score +0.65, confidence 0.85. Delivery beat + raised production target;
clean catalyst for a name where investors watch volume closely.

Example 28. "DoorDash downgraded to Sell at Morgan Stanley on concerns
about take-rate compression from merchant pushback; price target cut
from $180 to $145." -- bearish, score -0.55, confidence 0.8. Rating
downgrade with 19% price-target cut and a specific structural thesis.

Example 29. "Dollar Tree reports Q3 comps down 2.4% on traffic decline
in Family Dollar banner; announces closure of 600 underperforming
locations over 18 months." -- bearish, score -0.4, confidence 0.7.
Comps miss + store-closure announcement is net negative; closure may be
read as management action to improve longer-term metrics, but the
proximate signal is weakness.

Example 30. "Brent crude falls 3.2% to $78/bbl on OPEC+ decision to
increase production by 400kbpd starting Q1 2026; OPEC+ cites need to
defend market share." -- bearish, score -0.5, confidence 0.8. For
energy sector specifically; bullish for transports/airlines that
benefit from lower fuel.

## Additional terminology

Earnings revision -- sell-side analyst revises forward estimates.
Positive revision ratio (ratio of upward to downward revisions) above
1.5 is bullish for the stock; below 0.7 is bearish. A single revision
is noisy; a cluster across multiple analysts is a strong signal.

Options activity -- unusual call-volume spike with large open-interest
build typically precedes positive news; unusual put-volume spike
typically precedes negative news. Article mentioning "unusual options
activity" without a specific direction is neutral; mentioning
directional skew with dollar-volume context is signal.

Insider transactions -- cluster of 3+ insiders buying within 30 days
is bullish; cluster of 3+ insiders selling (non-10b5-1) is bearish.
Single-insider automatic 10b5-1 plan sales are neutral; management
opportunistic sales outside plans are mildly bearish.

Institutional ownership -- net inflow from actively managed funds is
bullish; outflow is bearish; ETF rebalance flows are neutral. Article
mentioning Berkshire / Baupost / Lone Pine / Viking establishing a
position is strongly bullish by reputation.

Short interest -- increase in short interest above 20% of float without
news catalyst is mildly bearish (informed short thesis); decrease in
short interest (covering) after a positive catalyst is bullish (short
squeeze mechanics). Short-interest snapshots without context are neutral.

Sell-side research -- initiation at Buy with a street-high price target
is bullish; initiation at Hold with a below-consensus target is mildly
bearish; reiteration without change is neutral.

Debt actions -- senior notes upsized and priced tighter than guidance
is bullish (demand + cost of capital); refinancing at higher coupon
than existing debt is mildly bearish; distressed-exchange or amendment
is strongly bearish.

Stock split -- forward split announcement is mildly bullish by signalling
management confidence and increasing retail accessibility; reverse
split is bearish by signalling price-level concern; in-kind split is
neutral mechanics.

Legal and compliance -- favourable settlement that removes overhang is
bullish; settlement with material financial or operational remedy is
bearish; ongoing investigation without new development is neutral;
material-weakness disclosure in internal controls is strongly bearish.

Supply chain -- sole-source supplier announcement is mildly bullish
(moat); diversification of supply is mildly bullish (resilience);
disclosure of supply-chain disruption is bearish for affected product
lines, bullish for competitors.

## Final instructions reminder

Classify the article directly by calling `classify_sentiment` with the
required fields. No prose. No chain-of-thought. No tags. No preface. No
postscript. Single tool call. This is the complete response.

If you ever feel tempted to write a reasoning paragraph before the tool
call, remember: the 2025 "Reasoning or Overthinking" literature
demonstrated that CoT reduces accuracy on financial sentiment at every
ambiguity level. Direct classification outperforms reflection. The
ladder system has already filtered easy cases; your job is to commit
confidently and move on. Other agents downstream handle the reasoning
over multiple articles together.

Proceed.
"""


_CLASSIFY_TOOL = {
    "name": "classify_sentiment",
    "description": "Classify the financial sentiment of a news article.",
    "input_schema": {
        "type": "object",
        "properties": {
            "sentiment_label": {
                "type": "string",
                "enum": ["bullish", "bearish", "neutral", "mixed"],
            },
            "sentiment_score": {
                "type": "number",
                "minimum": -1.0,
                "maximum": 1.0,
            },
            "confidence": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
            },
            "reasoning": {"type": "string"},
        },
        "required": ["sentiment_label", "sentiment_score", "confidence", "reasoning"],
    },
}


class HaikuScorer:
    """Claude Haiku 4.5 scorer with forced tool_choice and no-CoT.

    Uses `anthropic.Anthropic()` directly, NOT `ClaudeClient.generate_content()`
    which injects a generic prefix (`llm_client.py:630`).
    """

    version = HAIKU_MODEL_ID

    def __init__(self, api_key: str | None = None) -> None:
        if anthropic is None:
            self._client = None
            logger.warning(
                "anthropic SDK not installed (%s); HaikuScorer will fail-open",
                _ANTHROPIC_IMPORT_ERROR,
            )
            return
        key = api_key if api_key is not None else os.environ.get("ANTHROPIC_API_KEY", "")
        if not key:
            self._client = None
            logger.warning(
                "ANTHROPIC_API_KEY not set; HaikuScorer will fail-open"
            )
            return
        try:
            self._client = anthropic.Anthropic(api_key=key)
        except Exception as exc:  # pragma: no cover
            self._client = None
            logger.warning("anthropic.Anthropic() init failed: %r", exc)

    def score(self, article: Mapping[str, Any]) -> ScorerResult:
        t0 = time.perf_counter()
        article_id = str(_get(article, "article_id", ""))
        try:
            if self._client is None:
                raise RuntimeError("anthropic client not initialized")
            title = str(_get(article, "title", "") or "")
            body = str(_get(article, "body", "") or "")
            if not (title or body):
                raise ValueError("empty title+body")
            user_content = f"Title: {title}\n\nBody: {body[:4000]}"
            response = self._client.messages.create(
                model=HAIKU_MODEL_ID,
                max_tokens=512,
                system=[
                    {
                        "type": "text",
                        "text": HAIKU_SYSTEM_PROMPT,
                        "cache_control": {"type": "ephemeral", "ttl": "1h"},
                    }
                ],
                tools=[_CLASSIFY_TOOL],
                tool_choice={"type": "tool", "name": "classify_sentiment"},
                messages=[{"role": "user", "content": user_content}],
            )
            tool_use = next(
                (b for b in response.content if getattr(b, "type", None) == "tool_use"),
                None,
            )
            if tool_use is None:
                raise RuntimeError("no tool_use block in Haiku response")
            payload = dict(tool_use.input)  # type: ignore[attr-defined]
            label = str(payload.get("sentiment_label", LABEL_NEUTRAL))
            score_val = float(payload.get("sentiment_score", 0.0))
            confidence = float(payload.get("confidence", 0.0))

            usage = getattr(response, "usage", None)
            input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
            output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
            cost_usd = (
                input_tokens * HAIKU_PRICING_INPUT_PER_MTOK / 1_000_000.0
                + output_tokens * HAIKU_PRICING_OUTPUT_PER_MTOK / 1_000_000.0
            )
            latency_ms = (time.perf_counter() - t0) * 1000.0
            return ScorerResult(
                article_id=article_id,
                scorer_model=SCORER_MODEL_HAIKU,
                scorer_version=self.version,
                scored_at=_now_iso(),
                sentiment_score=max(-1.0, min(1.0, score_val)),
                sentiment_label=label,
                confidence=max(0.0, min(1.0, confidence)),
                latency_ms=latency_ms,
                cost_usd=cost_usd,
                raw_output=json.dumps(payload)[:2000],
            )
        except Exception as exc:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            logger.debug("HaikuScorer fail-open article_id=%s err=%r", article_id, exc)
            return _neutral_result(
                article_id, SCORER_MODEL_HAIKU, self.version, latency_ms, repr(exc)
            )


# ---- Gemini Flash (tier 4, opt-in) ----------------------------------------


class GeminiFlashScorer:
    """Opt-in tier-4 Gemini 2.5 Flash second-opinion scorer.

    Body deferred to phase-6.9 per research brief. When the
    `sentiment_use_gemini_flash` flag is off (default), callers must not
    reach this tier. If it is reached with the flag off, `score()` raises
    `NotImplementedError`. With the flag on, `score()` returns a stub-neutral
    result so the ladder does not crash while the real implementation is
    pending.
    """

    version = "0.0-stub"

    def __init__(self, enabled: bool = False) -> None:
        self.enabled = enabled

    def score(self, article: Mapping[str, Any]) -> ScorerResult:
        article_id = str(_get(article, "article_id", ""))
        if not self.enabled:
            raise NotImplementedError(
                "GeminiFlashScorer is disabled. Set sentiment_use_gemini_flash=True "
                "to opt in (phase-6.9 will implement the body)."
            )
        return _neutral_result(
            article_id,
            SCORER_MODEL_GEMINI_FLASH,
            self.version,
            0.0,
            "GeminiFlashScorer stub: phase-6.9 body pending",
        )


# ---- Ladder orchestration --------------------------------------------------


_SINGLETON_VADER: VaderScorer | None = None
_SINGLETON_FINBERT: FinBertScorer | None = None
_SINGLETON_HAIKU: HaikuScorer | None = None


def _get_vader() -> VaderScorer:
    global _SINGLETON_VADER
    if _SINGLETON_VADER is None:
        _SINGLETON_VADER = VaderScorer()
    return _SINGLETON_VADER


def _get_finbert() -> FinBertScorer:
    global _SINGLETON_FINBERT
    if _SINGLETON_FINBERT is None:
        _SINGLETON_FINBERT = FinBertScorer()
    return _SINGLETON_FINBERT


def _get_haiku() -> HaikuScorer:
    global _SINGLETON_HAIKU
    if _SINGLETON_HAIKU is None:
        _SINGLETON_HAIKU = HaikuScorer()
    return _SINGLETON_HAIKU


def score_ladder(
    article: Mapping[str, Any],
    *,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    use_gemini_flash: bool = False,
) -> ScorerResult:
    """Score a NormalizedArticle via the escalation cascade.

    Returns the first tier's ScorerResult that clears `min_confidence`.
    Haiku (tier 3) is always terminal unless `use_gemini_flash=True`. If
    every tier fails (e.g., missing deps + missing API key), returns the
    last tier's fail-open neutral result.
    """
    article_id = str(_get(article, "article_id", ""))

    vader_result = _get_vader().score(article)
    if vader_result.confidence >= min_confidence:
        return vader_result

    finbert_result = _get_finbert().score(article)
    if finbert_result.confidence >= min_confidence:
        return finbert_result

    haiku_result = _get_haiku().score(article)
    if not use_gemini_flash:
        return haiku_result
    if haiku_result.confidence >= min_confidence:
        return haiku_result

    try:
        gemini_scorer = GeminiFlashScorer(enabled=True)
        return gemini_scorer.score(article)
    except Exception as exc:
        logger.debug("GeminiFlashScorer fail-open article_id=%s err=%r", article_id, exc)
        return haiku_result


__all__ = [
    "ScorerResult",
    "VaderScorer",
    "FinBertScorer",
    "HaikuScorer",
    "GeminiFlashScorer",
    "score_ladder",
    "HAIKU_SYSTEM_PROMPT",
    "SCORER_MODEL_VADER",
    "SCORER_MODEL_FINBERT",
    "SCORER_MODEL_HAIKU",
    "SCORER_MODEL_GEMINI_FLASH",
    "LABEL_BULLISH",
    "LABEL_BEARISH",
    "LABEL_NEUTRAL",
    "LABEL_MIXED",
    "DEFAULT_MIN_CONFIDENCE",
]
