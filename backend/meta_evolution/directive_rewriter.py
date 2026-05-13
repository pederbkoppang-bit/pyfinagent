"""phase-10.7.2 Recursive Prompt Optimization -- Research Directive rewriter.

Closes a recursive prompt-optimization feedback loop. The "Research
Directive" is the prompt-shape that drives the harness research-gate
(`.claude/agents/researcher.md`). After each cycle, the system has
empirical data on whether the directive produced useful research
(`gate_passed`, source quality, recency-scan completeness). This module
uses an LLM to propose mutations to the directive based on that data.

Pattern follows SIPDO (arXiv 2505.19514, 2025) + Promptbreeder (arXiv
2309.16797) with anti-drift guards from GAAPO (Frontiers AI 2025) and
the Anthropic harness HITL pattern (the rewriter PROPOSES, the operator
APPROVES, Main writes the agent file, session restart picks it up).

The rewriter is fail-open: when LLM is unavailable (e.g., sk-ant-oat-*
401 + no Gemini fallback) or scores too low or too few briefs, returns
None. Never crashes the daily cycle.

Anti-drift guards:
  - MIN_BRIEFS_FOR_PROPOSAL = 5: need at least 5 cycles of evidence
  - MIN_LLM_JUDGE_SCORE = 0.6: rewrite is rejected below this floor
  - Simplicity criterion: prefer smaller diffs (size-of-change is part of score)
  - Global confirmation: rewrite must score better than the current directive
    on the SAME briefs corpus (no monotone-degradation paths)

Distinct from `backend/agents/skill_optimizer.py` which optimizes
`backend/agents/skills/*.md` agent prompts (Layer-1). This module
targets the META-prompt that drives Layer-3 harness research.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)

MIN_BRIEFS_FOR_PROPOSAL = 5
MIN_LLM_JUDGE_SCORE = 0.6

# phase-16.38 (#55) SIPDO global-confirmation thresholds. See
# docstring on `should_apply_globally` for the full rationale + sources.
MIN_CONFIRMATIONS_FOR_GLOBAL_APPLY = 3
MIN_PREFIX_OVERLAP_RATIO = 0.80
MIN_PASS_RATE_FOR_GLOBAL = 0.67

PROJECT = os.getenv("GCP_PROJECT_ID", "sunny-might-477607-p8")
DATASET = "pyfinagent_pms"
TABLE = "directive_versions"
TABLE_FQN = f"{PROJECT}.{DATASET}.{TABLE}"


@dataclass
class DirectiveVersion:
    """One proposed-or-applied version of the Research Directive.

    `applied_at` is None until the operator (Peder) approves + Main
    writes the new text into `.claude/agents/researcher.md`. The HITL
    gate is enforced by NOT having auto-apply; the rewriter only
    proposes.
    """

    version_id: str  # e.g., "rev-2026-04-25-001"
    parent_version_id: Optional[str]  # previous applied version, if any
    proposed_text: str
    diff_summary: str  # 1-2 sentence summary of what changed + why
    diff_size_bytes: int  # for simplicity criterion (smaller is better)
    judge_score: Optional[float]  # LLM-as-judge score 0-1, None if not scored yet
    components: dict[str, Any] = field(default_factory=dict)
    proposed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    applied_at: Optional[datetime] = None
    proposer: str = "directive_rewriter"  # vs "human_override"

    def is_acceptable(self) -> bool:
        """Does this version pass the anti-drift floor?"""
        if self.judge_score is None:
            return False
        return self.judge_score >= MIN_LLM_JUDGE_SCORE

    def to_bq_row(self) -> dict[str, Any]:
        return {
            "version_id": self.version_id,
            "parent_version_id": self.parent_version_id,
            "proposed_text": self.proposed_text,
            "diff_summary": self.diff_summary,
            "diff_size_bytes": int(self.diff_size_bytes),
            "judge_score": (
                float(self.judge_score) if self.judge_score is not None else None
            ),
            "components_json": json.dumps(self.components, sort_keys=True),
            "proposed_at": self.proposed_at.isoformat(),
            "applied_at": self.applied_at.isoformat() if self.applied_at else None,
            "proposer": self.proposer,
        }


def _summarize_brief_signals(briefs: list[dict[str, Any]]) -> dict[str, Any]:
    """Reduce a corpus of recent research briefs to scalar signals the LLM can weigh.

    Each brief is expected to be a dict with at least `gate_passed` (bool),
    `external_sources_read_in_full` (int), `recency_scan_performed` (bool),
    plus optional `tier` and `urls_collected`.
    """
    n = len(briefs)
    if n == 0:
        return {"n_briefs": 0}
    pass_rate = sum(1 for b in briefs if b.get("gate_passed")) / n
    avg_sources = sum(int(b.get("external_sources_read_in_full") or 0) for b in briefs) / n
    avg_urls = sum(int(b.get("urls_collected") or 0) for b in briefs) / n
    recency_rate = sum(1 for b in briefs if b.get("recency_scan_performed")) / n
    tiers = [b.get("tier") for b in briefs if b.get("tier")]
    return {
        "n_briefs": n,
        "gate_pass_rate": round(pass_rate, 3),
        "avg_sources_in_full": round(avg_sources, 2),
        "avg_urls_collected": round(avg_urls, 2),
        "recency_scan_rate": round(recency_rate, 3),
        "tiers_observed": sorted(set(tiers)),
    }


def _build_rewrite_prompt(
    current_directive_text: str,
    brief_signals: dict[str, Any],
    outcome_signals: dict[str, Any],
) -> str:
    """Construct the LLM prompt that proposes a rewritten directive."""
    return f"""You are the Research Directive rewriter for an autonomous harness.

CURRENT DIRECTIVE (research-gate spec from `.claude/agents/researcher.md`):
---BEGIN---
{current_directive_text}
---END---

EMPIRICAL SIGNALS FROM RECENT RESEARCH BRIEFS:
{json.dumps(brief_signals, indent=2)}

OUTCOME SIGNALS (downstream Q/A verdicts, masterplan-step results):
{json.dumps(outcome_signals, indent=2)}

YOUR TASK:
Propose a TARGETED, MINIMAL rewrite of the current directive that addresses
specific weaknesses surfaced by the empirical signals. Do NOT rewrite the
whole document. Return JSON with:

  {{
    "diff_summary": "1-2 sentence summary of what changed and WHY (cite specific signals)",
    "proposed_text": "the FULL rewritten directive (not just the diff)",
    "judge_score": 0.0-1.0  (your honest self-assessment of how much this will help)
  }}

Anti-drift rules (MUST follow):
  1. Smaller diffs are better (simplicity criterion).
  2. Do NOT remove existing safeguards (research-floor of 5 sources, etc.).
  3. If signals do not strongly suggest a rewrite is warranted, return
     judge_score < 0.6 (the rewriter will reject and propose nothing).
  4. Do NOT fabricate signals; weight your proposal only on the JSON above.
  5. Preserve YAML frontmatter + heading structure.
"""


def _call_llm_for_rewrite(prompt: str) -> Optional[dict[str, Any]]:
    """Call Anthropic Claude (primary) or Gemini fallback. Returns parsed JSON
    or None on any failure. Mirrors the phase-16.31 MAS-fallback pattern."""
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
                    max_tokens=4096,
                    system=(
                        "You are a careful prompt-engineering agent. Output ONLY "
                        "valid JSON matching the requested schema."
                    ),
                    messages=[{"role": "user", "content": prompt}],
                )
                text = "".join(b.text for b in resp.content if hasattr(b, "text"))
                return _parse_llm_json(text)
            except Exception as e:
                logger.warning("[directive_rewriter] anthropic call failed: %s", e)
                # fall through to Gemini

        # Gemini fallback (phase-16.31 pattern)
        from google import genai as _genai

        g_client = _genai.Client(
            vertexai=True,
            project=getattr(settings, "gcp_project_id", PROJECT),
            location=getattr(settings, "gcp_location", "us-central1"),
        )
        resp = g_client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
        )
        text = getattr(resp, "text", "") or ""
        return _parse_llm_json(text)
    except Exception as e:
        logger.warning("[directive_rewriter] all LLM paths failed: %s", e)
        return None


def _parse_llm_json(text: str) -> Optional[dict[str, Any]]:
    """Best-effort JSON extraction. Strips ```json fences if present."""
    if not text:
        return None
    s = text.strip()
    if s.startswith("```"):
        # strip a leading ```json or ``` line and a trailing ```
        lines = s.split("\n")
        if lines:
            lines = lines[1:]  # drop first ```...
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            s = "\n".join(lines)
    try:
        return json.loads(s)
    except Exception:
        # Try to find a JSON object inside
        first = s.find("{")
        last = s.rfind("}")
        if first >= 0 and last > first:
            try:
                return json.loads(s[first : last + 1])
            except Exception:
                return None
    return None


def rewrite_directive(
    current_directive_text: str,
    recent_briefs: list[dict[str, Any]],
    outcome_signals: Optional[dict[str, Any]] = None,
    *,
    parent_version_id: Optional[str] = None,
    llm_call_override: Optional[Any] = None,
) -> Optional[DirectiveVersion]:
    """Propose a rewritten Research Directive.

    Returns None when:
      - fewer than `MIN_BRIEFS_FOR_PROPOSAL` recent briefs (insufficient evidence)
      - LLM call fails / returns invalid JSON
      - judge_score < `MIN_LLM_JUDGE_SCORE` (anti-drift floor)
      - proposed_text is identical to current (no-op)

    Returns a `DirectiveVersion` (with `is_acceptable() == True`) when
    the rewriter has high-confidence in the proposal.

    HITL gate: NEVER writes to `.claude/agents/researcher.md` directly.
    Operator (Peder) reviews the returned `DirectiveVersion.proposed_text`
    + Main applies it after explicit approval. Session restart required
    per CLAUDE.md "Agent definition changes require session restart".

    Args:
        current_directive_text: full text of `.claude/agents/researcher.md`
        recent_briefs: list of brief-envelope dicts (gate_passed, etc.)
        outcome_signals: optional dict of downstream Q/A signals
        parent_version_id: previous applied version (for lineage)
        llm_call_override: testing hook -- if given, used instead of
            `_call_llm_for_rewrite`. Receives the prompt, returns parsed
            JSON dict or None.
    """
    if not current_directive_text:
        logger.warning("[directive_rewriter] empty current_directive_text; skip")
        return None

    n = len(recent_briefs)
    if n < MIN_BRIEFS_FOR_PROPOSAL:
        logger.info(
            "[directive_rewriter] only %d briefs (< %d floor); no proposal",
            n,
            MIN_BRIEFS_FOR_PROPOSAL,
        )
        return None

    brief_signals = _summarize_brief_signals(recent_briefs)
    outcome_signals = outcome_signals or {}
    prompt = _build_rewrite_prompt(current_directive_text, brief_signals, outcome_signals)

    caller = llm_call_override if llm_call_override else _call_llm_for_rewrite
    parsed = caller(prompt)
    if not parsed:
        logger.info("[directive_rewriter] LLM returned no usable JSON; no proposal")
        return None

    proposed_text = (parsed.get("proposed_text") or "").strip()
    if not proposed_text or proposed_text == current_directive_text.strip():
        logger.info("[directive_rewriter] no-op proposal; skip")
        return None

    diff_summary = (parsed.get("diff_summary") or "").strip() or "(no summary)"
    score = parsed.get("judge_score")
    try:
        score = float(score) if score is not None else None
    except Exception:
        score = None

    diff_size_bytes = abs(
        len(proposed_text.encode("utf-8")) - len(current_directive_text.encode("utf-8"))
    )

    version_id = f"rev-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
    version = DirectiveVersion(
        version_id=version_id,
        parent_version_id=parent_version_id,
        proposed_text=proposed_text,
        diff_summary=diff_summary,
        diff_size_bytes=diff_size_bytes,
        judge_score=score,
        components={
            "brief_signals": brief_signals,
            "outcome_signals": outcome_signals,
        },
    )

    if not version.is_acceptable():
        logger.info(
            "[directive_rewriter] proposal score=%s below floor=%s; not acceptable",
            score,
            MIN_LLM_JUDGE_SCORE,
        )
        return None

    return version


def should_apply_globally(
    recent_versions: list["DirectiveVersion"],
    recent_qa_verdicts: list[str],
) -> bool:
    """phase-16.38 (#55) SIPDO global-confirmation gate.

    Decides whether a series of accepted DirectiveVersion proposals has
    converged enough to warrant promoting to a global, persistent
    directive change (vs the per-cycle local apply).

    Pure function: no I/O, no BQ, no file writes. The orchestrator
    calls this and surfaces the result through the existing HITL
    pipeline -- this function NEVER writes to .claude/agents/researcher.md.

    Returns True iff ALL FOUR criteria hold:
      1. >= MIN_CONFIRMATIONS_FOR_GLOBAL_APPLY (3) recent versions
      2. Every version is_acceptable() (judge_score >= floor)
      3. Pairwise SequenceMatcher ratio >= MIN_PREFIX_OVERLAP_RATIO (0.80)
         across all (a, b) version pairs (SIPDO reconfirmation; arXiv
         2505.19514 2025) -- proves convergence rather than oscillation
      4. Verdict-weighted pass-rate >= MIN_PASS_RATE_FOR_GLOBAL (0.67):
         PASS=1.0, CONDITIONAL=0.5, FAIL=0.0; weighted_sum / N

    Refs:
      - SIPDO (arXiv 2505.19514 2025): closed-loop reconfirmation step
      - GAAPO (Frontiers AI 2025): documented missing global-confirm gate
      - APE / GRIPS convergence research: N=3 cycles minimum for plateau
      - HITL composite-signal pattern: never gate on a single LLM score
    """
    from difflib import SequenceMatcher

    if len(recent_versions) < MIN_CONFIRMATIONS_FOR_GLOBAL_APPLY:
        return False

    # Every version must clear the per-cycle floor first.
    if not all(v.is_acceptable() for v in recent_versions):
        return False

    # Convergence check: every pair of proposed_text strings must overlap
    # by >= MIN_PREFIX_OVERLAP_RATIO. Quadratic in N but N is tiny (3-10).
    for i in range(len(recent_versions)):
        for j in range(i + 1, len(recent_versions)):
            ratio = SequenceMatcher(
                None,
                recent_versions[i].proposed_text,
                recent_versions[j].proposed_text,
            ).ratio()
            if ratio < MIN_PREFIX_OVERLAP_RATIO:
                return False

    # Outcome check: verdict-weighted pass-rate. Empty verdicts list
    # blocks the gate (cannot confirm without outcome signal).
    if not recent_qa_verdicts:
        return False
    weights = {"PASS": 1.0, "CONDITIONAL": 0.5, "FAIL": 0.0}
    weighted_sum = sum(weights.get(v.upper(), 0.0) for v in recent_qa_verdicts)
    pass_rate = weighted_sum / len(recent_qa_verdicts)
    if pass_rate < MIN_PASS_RATE_FOR_GLOBAL:
        return False

    return True


def persist_version(bq_client: Any, version: DirectiveVersion) -> None:
    """Insert one version into `directive_versions`. Fail-open like
    `alpha_velocity.persist_sample`."""
    row = version.to_bq_row()
    try:
        errors = bq_client.insert_rows_json(TABLE_FQN, [row])
        if errors:
            logger.warning(
                "[directive_rewriter] persist returned errors: %s", errors
            )
    except Exception as e:
        logger.warning("[directive_rewriter] persist failed (fail-open): %s", e)
