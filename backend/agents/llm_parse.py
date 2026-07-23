"""phase-75.5 (llmeng-04): the shared LLM JSON-parse helper.

Criterion 3 of masterplan step 75.5 refers to "the shared JSON-parse helper" -- but at
the time the step was written **there was no such thing**. Several independent copies
existed instead, each taking a bare `text: str` and therefore structurally incapable of
telling a TRUNCATED response from a merely malformed one. This module is that shared
helper.

## Known duplicated sites -- NOT A COMPLETE ENUMERATION

These are the truncation-blind LLM-JSON parsers identified so far. **This list is
explicitly non-exhaustive and no total count is asserted here.** Treat it as "at least
these", never as "exactly these":

  - `backend/agents/debate.py:122::_parse_json`
  - `backend/agents/risk_debate.py:118::_parse_json`             (byte-identical to the above)
  - `backend/agents/orchestrator.py:309::_parse_json_with_fallback`
  - `backend/meta_evolution/directive_rewriter.py:212::_parse_llm_json`
  - `backend/agents/agent_definitions.py:353::parse_llm_classification`
        -- on parse failure returns a SILENT DEFAULT routing to `AgentType.MAIN`
  - `backend/agents/evaluator_agent.py:412::_parse_evaluation_response`
        -- on parse failure returns a conservative FAIL verdict

A proper behavioural census is queued as masterplan step **75.5.8**; the rewiring of the
named sites is **75.5.5**.

## WHY THERE IS NO NUMBER IN THIS DOCSTRING

Earlier revisions asserted "three", then "four". Both were wrong, and both were wrong the
same way: a **count carried forward from a summary and never measured against a stated
definition**. Reviewers caught "three" on the fifth pass and "four" on the sixth. The
second correction was not a measurement either -- it just moved the number.

The defect is not arithmetic. It is asserting a **bounded** count for a population whose
boundary was never defined operationally.

Demonstration, and a third instance of the same error: an earlier revision of THIS
paragraph asserted that "takes a bare `str` and json-loads it" matches "~36 functions
under `backend/`" -- a number written inside the section explaining why numbers are not
asserted here. It was never measured. When two reviewers each executed that stated rule
independently, they got **17** and **20** -- neither of them 36, and not each other.
Two careful measurements of "the same rule" disagreeing is the proof that the rule was
never operational: it does not say whether `json_io.loads` counts, whether the parameter
must be annotated, or whether `self` shifts the position.

The lesson stands without any number: that phrasing sweeps in GCS loaders, JWE decrypt
and BigQuery readers, so it is the wrong discriminator. The property that matters is "is
an LLM-response parser that cannot see truncation", and establishing it needs a
behavioural census keyed on LLM-provenance -- not a name-shaped regex, which can only
re-confirm the members whose names already fit the pattern you drew around them.

So: this docstring names what it knows, states that it is incomplete, and points at the
step that will measure it. **If you are adding to this list, do not replace the
"non-exhaustive" wording with a number unless you have run the census.**

## The retry-ownership rule (read before adding any retry here)

**This module NEVER retries. Not once, not conditionally, not "just for truncation".**

It only *observes*: it reports whether the response was cut off and marks the result
`degraded`. Retrying here would create a second, invisible re-request layer stacked on
top of one that already exists -- which is precisely the failure this design forbids.

**The sole owner of the max_tokens re-request on the `generate_content` path is:**

> `ClaudeClient.generate_content`'s phase-4.14.4 MF-26/27 `stop_reason` dispatch in
> `backend/agents/llm_client.py` (~:1656-1681), which re-requests once with
> `max_tokens = min(max_tokens * 2, 8192)` and only when the truncated response ends in
> a `tool_use` block.

A **separate, pre-existing** owner exists on a **different** path: the Layer-2 MAS tool
loop in `backend/agents/multi_agent_orchestrator.py` (~:1363-1394), which re-requests at
`min(_max_tokens * 2, 32768)` guarded by `_mt_retried_turn`. phase-75.5 deliberately did
**not** unify the two -- they sit on different call paths and merging them is queued as
its own step. Knowing both exist is exactly why this helper adds no third.

Note also that owner #1 retries **only** on a `tool_use` tail; plain-text truncation is
logged and returned as-is. So a `degraded=True` result from this helper is the *expected*
outcome for truncated prose, not a sign the retry failed.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional

from backend.utils import json_io

logger = logging.getLogger(__name__)


def clean_json_output(text: str) -> str:
    """Strip markdown code fences that models wrap around JSON."""
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"```$", "", text)
    return text.strip()


def parse_llm_json(
    response: Any,
    label: str,
    *,
    text: Optional[str] = None,
) -> tuple[Optional[dict], bool]:
    """Parse an LLM response as JSON, reporting truncation instead of retrying.

    Args:
        response: an `LLMResponse` (preferred -- carries `stop_reason`) or a raw
            string. Passing a string is supported for the legacy call sites but
            loses truncation detection, which is the whole point of this helper.
        label: agent name for logging.
        text: optional pre-extracted/cleaned text overriding `response.text`.

    Returns:
        `(parsed_dict_or_None, degraded)`.

        `degraded=True` means the provider truncated the response, so a parse
        failure is a KNOWN-CAUSE failure rather than an unexplained one. Callers
        must treat a degraded result as "the output is incomplete" -- never as a
        silent pass. **This function issues no retry**; see the module docstring
        for the single owning layer.
    """
    raw = text if text is not None else getattr(response, "text", response)
    truncated = bool(getattr(response, "is_truncated", lambda: False)())

    if truncated:
        # Deliberately WARNING, not DEBUG: a truncated response that happens to
        # parse is the dangerous case -- valid-looking JSON missing its tail.
        logger.warning(
            "%s: provider stopped with stop_reason=%r -- output is TRUNCATED. "
            "Marking degraded. No retry is issued here by design; the single "
            "max_tokens re-request owner is ClaudeClient.generate_content's "
            "MF-26/27 dispatch (llm_client.py:~1656-1681).",
            label, getattr(response, "stop_reason", None),
        )

    try:
        data = json_io.loads(clean_json_output(raw if isinstance(raw, str) else str(raw)))
    except (json.JSONDecodeError, ValueError, TypeError):
        if truncated:
            logger.warning(
                "%s: JSON parse failed AND the response was truncated -- the cause "
                "is a too-small output budget, not a malformed model reply.", label,
            )
        else:
            logger.warning("%s returned invalid JSON, using raw text", label)
        return None, truncated

    if not isinstance(data, dict):
        return None, truncated

    # Marking a SUCCESSFUL parse degraded is intentional: truncated JSON can still
    # parse (e.g. a closed object missing later fields), and that silent-partial case
    # is worse than an outright failure because nothing downstream questions it.
    if truncated:
        try:
            setattr(response, "degraded", True)
        except Exception:  # pragma: no cover -- raw-string callers
            pass
    return data, truncated


__all__ = ["parse_llm_json", "clean_json_output"]
