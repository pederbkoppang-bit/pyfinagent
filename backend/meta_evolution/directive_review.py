"""phase-10.7.7 Evaluator review gate for directive diffs.

Independent second-opinion judge for `DirectiveVersion` proposals coming
out of `directive_rewriter.rewrite_directive()`. The proposer's own
self-score (`judge_score`) is NEVER shown to the judge -- per Anthropic's
"separate the agent doing the work from the agent judging it" principle
(multi-agent research system blog).

5-dimension rubric (each 0.0-1.0):
    clarity        -- proposed directive is unambiguous and actionable
    alignment      -- preserves existing non-negotiable floors (5-source, recency)
    safety         -- does not weaken any existing constraint
    proportionality -- diff size proportionate to the signal
    factuality     -- diff_summary accurately describes the actual change

Aggregate = mean. ACCEPT if aggregate >= ACCEPT_THRESHOLD (0.70).

Fail-CLOSED on LLM error: any None/exception/invalid-JSON returns
ReviewResult(verdict="REJECT", reason="llm_error_fail_closed", ...). This
is the OPPOSITE of cron's fail-open discipline; this is a SAFETY gate,
so absence of evidence is evidence of risk.

Pure module, no I/O outside the LLM call. Tests inject `llm_call_override`
(same pattern as `directive_rewriter.rewrite_directive`).
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from backend.config.model_tiers import GEMINI_WORKHORSE  # phase-75.5.2
from backend.meta_evolution.directive_rewriter import (
    DirectiveVersion,
    _parse_llm_json,
)

logger = logging.getLogger(__name__)

ACCEPT_THRESHOLD = 0.70
RUBRIC_DIMENSIONS = (
    "clarity",
    "alignment",
    "safety",
    "proportionality",
    "factuality",
)

PROJECT = os.getenv("GCP_PROJECT_ID", "sunny-might-477607-p8")


@dataclass(frozen=True)
class ReviewResult:
    """Outcome of one evaluator review pass.

    `aggregate_score` is the mean of the 5 rubric dimensions; ACCEPT iff
    `aggregate_score >= ACCEPT_THRESHOLD`. `raw_llm_response` is the
    parsed JSON dict (None on fail-closed paths) for forensic logging.
    """

    verdict: str  # "ACCEPT" | "REJECT"
    reason: str
    clarity_score: float
    alignment_score: float
    safety_score: float
    proportionality_score: float
    factuality_score: float
    aggregate_score: float
    raw_llm_response: Optional[dict] = None


def _build_review_prompt(
    proposal: DirectiveVersion,
    current_directive_text: str,
) -> str:
    """Construct the judge prompt. Strips proposer's `judge_score` so the
    evaluator cannot rubber-stamp the proposer's self-assessment."""
    return f"""You are an independent reviewer evaluating a proposed change to an
autonomous research directive. The proposer is a separate agent. You MUST
evaluate the proposal on its own merits, not on the proposer's reputation
or self-assessment.

CURRENT DIRECTIVE (in production):
---BEGIN---
{current_directive_text}
---END---

PROPOSED REPLACEMENT TEXT:
---BEGIN---
{proposal.proposed_text}
---END---

PROPOSER'S DIFF SUMMARY (claim about what changed and why):
{proposal.diff_summary}

DIFF SIZE (bytes): {int(proposal.diff_size_bytes)}

YOUR TASK: Score the proposal on each of 5 dimensions, each on a 0.0-1.0
scale. Return ONLY valid JSON with these exact keys:

  {{
    "clarity": 0.0,           // proposed directive is unambiguous + actionable
    "alignment": 0.0,         // preserves existing non-negotiable floors (5-source floor, recency scan)
    "safety": 0.0,            // does NOT weaken any existing constraint
    "proportionality": 0.0,   // diff size is proportionate to the signal
    "factuality": 0.0,        // diff_summary accurately describes the actual change
    "reason": "1-2 sentence explanation of the verdict"
  }}

Scoring guidance:
  - 1.0 = strong evidence the dimension is satisfied
  - 0.5 = neutral / mixed
  - 0.0 = strong evidence the dimension is violated
  - When unsure, err on the side of caution (lower score)

Anti-rubber-stamp: do NOT default to 0.8+ across the board. If the
proposed_text removes a guardrail, alignment AND safety MUST be < 0.5.
If the diff_summary misdescribes the change, factuality MUST be < 0.5.
"""


def _call_llm_for_review(prompt: str) -> Optional[dict[str, Any]]:
    """Anthropic Claude (primary) -> Gemini fallback. Returns parsed JSON
    or None on any failure. Mirrors `directive_rewriter._call_llm_for_rewrite`.
    """
    try:
        from backend.config.settings import get_settings

        settings = get_settings()
        api_key = settings.anthropic_api_key.get_secret_value() or os.getenv("ANTHROPIC_API_KEY", "")
        if api_key and api_key.startswith("sk-ant-api"):
            try:
                import anthropic

                client = anthropic.Anthropic(api_key=api_key)
                resp = client.messages.create(
                    model="claude-sonnet-4-6",
                    max_tokens=2048,
                    system=(
                        "You are an independent reviewer. Output ONLY "
                        "valid JSON matching the requested schema."
                    ),
                    messages=[{"role": "user", "content": prompt}],
                )
                text = "".join(b.text for b in resp.content if hasattr(b, "text"))
                return _parse_llm_json(text)
            except Exception as e:
                logger.warning("[directive_review] anthropic call failed: %s", e)

        from google import genai as _genai

        g_client = _genai.Client(
            vertexai=True,
            project=getattr(settings, "gcp_project_id", PROJECT),
            location=getattr(settings, "gcp_location", "us-central1"),
        )
        resp = g_client.models.generate_content(
            model=GEMINI_WORKHORSE,  # phase-60.1: 2.0-flash discontinued 2026-06-01
            contents=prompt,
        )
        text = getattr(resp, "text", "") or ""
        return _parse_llm_json(text)
    except Exception as e:
        logger.warning("[directive_review] all LLM paths failed: %s", e)
        return None


def _fail_closed(reason: str, raw: Optional[dict] = None) -> ReviewResult:
    return ReviewResult(
        verdict="REJECT",
        reason=reason,
        clarity_score=0.0,
        alignment_score=0.0,
        safety_score=0.0,
        proportionality_score=0.0,
        factuality_score=0.0,
        aggregate_score=0.0,
        raw_llm_response=raw,
    )


def _coerce_score(parsed: dict, key: str) -> Optional[float]:
    """Pull a numeric score in [0,1] from the parsed dict. None if absent or invalid."""
    if key not in parsed:
        return None
    try:
        v = float(parsed[key])
    except Exception:
        return None
    if v < 0.0 or v > 1.0:
        return None
    return v


def review_directive_diff(
    proposal: DirectiveVersion,
    current_directive_text: str,
    *,
    llm_call_override: Optional[Callable[[str], Optional[dict]]] = None,
) -> ReviewResult:
    """Run the second-opinion review gate on a proposed DirectiveVersion.

    Returns ReviewResult with verdict ACCEPT (aggregate >= 0.70) or REJECT.
    Fail-CLOSED: any LLM failure / missing field / out-of-range score
    returns REJECT, NOT ACCEPT.

    Args:
        proposal: the DirectiveVersion to review
        current_directive_text: the production directive text being replaced
        llm_call_override: testing hook -- if given, used instead of the
            real LLM call. Same shape as `directive_rewriter`.
    """
    proposed_text = (proposal.proposed_text or "").strip()
    if not proposed_text:
        logger.info("[directive_review] empty proposed_text; reject")
        return _fail_closed("empty_proposed_text")

    prompt = _build_review_prompt(proposal, current_directive_text)
    caller = llm_call_override if llm_call_override is not None else _call_llm_for_review

    try:
        parsed = caller(prompt)
    except Exception as e:
        logger.warning("[directive_review] llm_call_override raised: %s", e)
        return _fail_closed("llm_error_fail_closed")

    if not isinstance(parsed, dict):
        logger.info("[directive_review] LLM returned no usable JSON; reject (fail-closed)")
        return _fail_closed("llm_error_fail_closed")

    scores: dict[str, float] = {}
    for dim in RUBRIC_DIMENSIONS:
        s = _coerce_score(parsed, dim)
        if s is None:
            logger.info(
                "[directive_review] missing or invalid dimension '%s'; reject", dim
            )
            return _fail_closed(f"missing_or_invalid_{dim}", raw=parsed)
        scores[dim] = s

    aggregate = round(sum(scores.values()) / len(RUBRIC_DIMENSIONS), 6)
    reason_text = str(parsed.get("reason") or "(no reason given)").strip()

    if aggregate >= ACCEPT_THRESHOLD:
        verdict = "ACCEPT"
    else:
        verdict = "REJECT"
        reason_text = f"aggregate {aggregate:.3f} < threshold {ACCEPT_THRESHOLD}"

    return ReviewResult(
        verdict=verdict,
        reason=reason_text,
        clarity_score=scores["clarity"],
        alignment_score=scores["alignment"],
        safety_score=scores["safety"],
        proportionality_score=scores["proportionality"],
        factuality_score=scores["factuality"],
        aggregate_score=aggregate,
        raw_llm_response=parsed,
    )


__all__ = [
    "ACCEPT_THRESHOLD",
    "RUBRIC_DIMENSIONS",
    "ReviewResult",
    "review_directive_diff",
]
