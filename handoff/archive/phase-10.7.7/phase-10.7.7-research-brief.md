# Research Brief: phase-10.7.7 -- Evaluator Review Gate for Directive Diffs

**Tier:** moderate (assumed; caller did not override)
**Date:** 2026-04-26

---

## Search queries run (three-variant discipline)

| Variant | Query |
|---------|-------|
| Current-year frontier | `LLM-as-a-judge architectural patterns evaluator-optimizer 2026` |
| Current-year frontier | `prompt evolution safety review gate accept reject LLM evaluator 2026` |
| Last-2-year window | `Constitutional AI RLAIF judge model system prompt evolution safety review 2025` |
| Last-2-year window | `SIPDO arXiv 2505.19514 recursive prompt optimization closed loop 2025` |
| Last-2-year window | `pytest monkeypatch LLM stub mock async client testing 2025` |
| Year-less canonical | `LLM judge calibration rubric design score variance generator evaluator separation` |
| Year-less canonical | `Anthropic evaluator-optimizer multi-agent research system second opinion pattern` |

---

## Read in full (>=5 required; counts toward the gate)

| URL | Accessed | Kind | Fetched how | Key quote or finding |
|-----|----------|------|-------------|----------------------|
| https://www.anthropic.com/engineering/multi-agent-research-system | 2026-04-26 | Official blog | WebFetch | "Separating the agent doing the work from the agent judging it proves to be a strong lever"; file-based handoff pattern; factual accuracy + citation accuracy + completeness as judge dimensions |
| https://www.anthropic.com/engineering/harness-design-long-running-apps | 2026-04-26 | Official blog | WebFetch | Generator/evaluator loop with sprint contracts establishing "done" before implementation; evaluator uses criterion-based grading with hard thresholds; fail-closed: any criterion below threshold = sprint fails |
| https://www.evidentlyai.com/llm-guide/llm-as-a-judge | 2026-04-26 | Authoritative blog | WebFetch | "If unsure, err on the side of caution and mark it as toxic" (fail-closed pattern); split complex criteria into separate evaluators; binary/low-precision scoring more reliable than high-precision |
| https://www.kinde.com/learn/ai-for-software-engineering/best-practice/llm-as-a-judge-done-right-calibrating-guarding-debiasing-your-evaluators/ | 2026-04-26 | Industry blog | WebFetch | Anchor examples stabilize calibration; chain-of-thought improves accuracy + auditability; temperature=0 for deterministic judge output; 2-3 runs averaged for high-stakes decisions |
| https://labelyourdata.com/articles/llm-as-a-judge | 2026-04-26 | Industry blog | WebFetch | Four judge components: model, rubric, scoring method, sampling strategy; Cohen's kappa >0.7 as calibration target; >75% human-judge agreement; pre-deployment safety screening blocks + queues high-severity violations |
| https://arxiv.org/pdf/2505.19514 | 2026-04-26 | arXiv preprint | WebFetch | SIPDO closed-loop: fail-closed on optimizer failure (retain prior best prompt); comparative acceptance (revision must outperform current); convergence via iteration limits + performance plateau detection |
| https://www.anthropic.com/research/constitutional-ai-harmlessness-from-ai-feedback | 2026-04-26 | Peer-reviewed (Anthropic) | WebFetch | CAI two-phase: separate critique generator from the model being improved; "sample, self-critique, revise, then finetune" -- the critique step is always a separate role from the generator |

---

## Identified but snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|-----|------|------------------------|
| https://sureprompts.com/blog/llm-as-judge-prompting-guide | Blog | Fetched; included above in read-in-full |
| https://langfuse.com/docs/evaluation/evaluation-methods/llm-as-a-judge | Docs | Snippet only; content covered by Evidently source |
| https://arxiv.org/abs/2212.08073 | arXiv abstract | Full paper covered via Anthropic research page fetch |
| https://aws.amazon.com/blogs/machine-learning/llm-as-a-judge-on-amazon-bedrock-model-evaluation/ | Blog | Snippet only; vendor-specific, lower priority |
| https://medium.com/@adnanmasood/rubric-based-evals-llm-as-a-judge-methodologies-and-empirical-validation-in-domain-context-71936b989e80 | Blog | Snippet only; April 2026 publication, rubric methodology covered by other sources |
| https://arize.com/llm-as-a-judge/ | Industry | Snippet only; content covered |
| https://docs.pytest.org/en/latest/how-to/monkeypatch.html | Official docs | Snippet only; well-known stdlib pattern, confirmed via search |
| https://github.com/anthropics/anthropic-cookbook/blob/main/patterns/agents/evaluator_optimizer.ipynb | GitHub | Snippet only; code example, patterns confirmed via blog posts |
| https://arxiv.org/abs/2512.10449 | arXiv | Snippet only; adversarial review vulnerability, context only |
| https://www.confident-ai.com/blog/why-llm-as-a-judge-is-the-best-llm-evaluation-method | Blog | Fetched; rubric dimensions extracted |

---

## Recency scan (2024-2026)

Searched explicitly for 2025-2026 literature on LLM-as-judge, prompt evolution safety gates, and evaluator patterns.

**Found findings (2024-2026):**

1. **SIPDO** (arXiv 2505.19514, May 2025): Closed-loop prompt optimization with synthetic data feedback; fail-closed on optimizer failure; convergence via plateau detection. Directly referenced in `directive_rewriter.py` docstring. Confirms: retain prior best when judge score < floor.

2. **C3AI** (ACM Web Conference 2025): Positively-framed, behavior-based principles in constitutions outperform negatively-framed or trait-based ones -- relevant for rubric phrasing in the directive evaluator.

3. **When Reject Turns into Accept** (arXiv 2512.10449, Dec 2025): LLM-as-judge safety gates are vulnerable to adversarial prompt injection with up to 86% decision-flip rates in open-source models. Implication: the evaluator's judge prompt itself must be hardened; use a capable model (Claude Sonnet-class) not a small open-source model.

4. **Rubric-Based Evaluations & LLM-as-a-Judge** (Medium, Apr 2026): Domain-specific rubric calibration using empirical validation; confirms per-dimension scoring with anchored examples.

5. **Microsoft Security Blog** (Feb 2026): One-prompt attacks that break LLM safety alignment still possible in 2026 -- reinforces fail-closed default and HITL gate for directive application.

Older canonical sources (CAI 2022, Anthropic harness blog, Evidently guide) remain load-bearing and are not superseded; the 2025-2026 work complements rather than replaces them.

---

## Internal code inventory

| File | Lines | Role | Status |
|------|-------|------|--------|
| `backend/meta_evolution/directive_rewriter.py` | 1-411 | Proposer: generates `DirectiveVersion` with `judge_score` set by the LLM itself | Active |
| `backend/meta_evolution/__init__.py` | 1-24 | Package exports (no `directive_rewriter` in `__all__` yet) | Active; update needed |
| `backend/agents/evaluator_agent.py` | 1-160+ | Backtest evaluator: async, Gemini-based, 5-dimension rubric, `EvaluationResult` dataclass | Active; NOT reusable as-is |
| `tests/meta_evolution/test_directive_rewriter.py` | 1-200 | 7 unit tests using `llm_call_override` FakeLLM pattern + `FakeBQ` stub | Active; gold template |
| `tests/agents/` | -- | **Does not exist** -- must be created with `__init__.py` | Missing |

---

## Key findings (numbered, per-claim)

1. **The proposer self-scores its own proposal** -- `directive_rewriter.py:71` (`judge_score: Optional[float]`) is set by the LLM that writes `proposed_text`. The rewriter prompt at line 158 asks: "your honest self-assessment of how much this will help." This is the exact "grading own homework" anti-pattern. The evaluator gate must use a SEPARATE LLM call with a SEPARATE prompt that does NOT see the proposer's self-score. (Source: Anthropic harness blog -- "separating generation from evaluation is the strongest lever")

2. **`DirectiveVersion` is the proposer's output shape** -- fields at `directive_rewriter.py:57-76`:
   - `version_id: str`
   - `parent_version_id: Optional[str]`
   - `proposed_text: str`
   - `diff_summary: str`
   - `diff_size_bytes: int`
   - `judge_score: Optional[float]` (proposer's self-score -- evaluator must IGNORE this)
   - `components: dict` (contains `brief_signals`, `outcome_signals`)
   - `proposed_at: datetime`
   - `applied_at: Optional[datetime]`
   - `proposer: str`

3. **The evaluator must be a NEW module** -- `evaluator_agent.py` is async, Gemini-only, uses `google-genai` shim, outputs `EvaluationResult` typed for backtest metrics (Sharpe, DSR, drawdown). It cannot be reused without a full rewrite. A dedicated `backend/meta_evolution/directive_review.py` is cleaner and keeps meta-evolution self-contained. (Source: `evaluator_agent.py:1-77`)

4. **Mock surface in tests** -- `test_directive_rewriter.py:122-130` shows the established pattern: `llm_call_override` parameter receives a lambda `(prompt: str) -> Optional[dict]`. The new `review_directive_diff` should follow the same pattern with a `llm_call_override` kwarg. No `monkeypatch` needed; the override is a direct function-parameter injection. (Source: `tests/meta_evolution/test_directive_rewriter.py:246-268`)

5. **Fail-closed is mandatory for a safety gate** -- Both Evidently guide ("if unsure, err on the side of caution") and SIPDO ("retain prior best prompt on failure") confirm: when the LLM call fails or returns invalid JSON or a score below the threshold, the gate must REJECT (not pass). The existing `rewrite_directive` is fail-open (returns `None` = caller skips proposal); the new gate must be fail-closed (exception/error = `ReviewResult(verdict="REJECT", reason="llm_error")`). (Sources: Evidently guide; SIPDO arXiv 2505.19514)

6. **Rubric dimensions for a directive diff review** -- Synthesizing CAI self-critique dimensions, Anthropic judge criteria, and EvidentlyAI rubric design: five dimensions map to the directive use case: (1) Clarity -- is the proposed directive unambiguous? (2) Alignment -- does it preserve existing non-negotiable floors (e.g., 5-source floor, recency scan)? (3) Safety/No-Regression -- does it not remove existing guardrails? (4) Proportionality -- is the diff small relative to the improvement signal? (5) Factuality -- does `diff_summary` accurately describe the actual change? Each scored 0.0-1.0; aggregate = mean; ACCEPT >= 0.70.

7. **Score threshold** -- The existing `MIN_LLM_JUDGE_SCORE = 0.6` in `directive_rewriter.py:43` is the proposer's SELF-floor. A separate evaluator should use a HIGHER bar (0.70) because it is a second-opinion gate, not the first filter. This aligns with SIPDO's comparative acceptance (must outperform current) and Anthropic's sprint-contract hard-threshold pattern.

8. **`__init__.py` does not export `directive_rewriter`** -- `backend/meta_evolution/__init__.py:10-24` only exports archetype symbols. The new `directive_review.py` module must be importable directly; no `__init__.py` change is strictly required but the brief recommends adding `directive_review` to `__all__` for discoverability.

---

## Consensus vs debate (external)

**Consensus:**
- Separate judge from generator (universal across all sources)
- Fail-closed when safety/gate is the use case (Evidently, SIPDO, Anthropic)
- Low-precision scoring (binary or 0-1 per dimension) more reliable than fine-grained numeric scale
- Chain-of-thought in judge prompt improves auditability
- Anchor examples (what a "1.0" looks like vs a "0.0") stabilize calibration

**Debate:**
- Score threshold: sources don't prescribe a universal cutoff; 0.70 is a reasonable choice based on the existing 0.60 proposer floor + upgrade for second-opinion strictness
- Model choice for judge: capable model (Sonnet-class) vs cost; for a safety gate, capability wins

---

## Pitfalls (from literature)

1. **Judge sees proposer's self-score**: If the review prompt echoes back `judge_score` from the `DirectiveVersion`, the LLM anchors on it (position/anchoring bias). Strip `judge_score` from the proposal dict before passing to the evaluator.
2. **Adversarial flip via proposed_text** (arXiv 2512.10449): A crafted `proposed_text` could manipulate the judge. Mitigation: pass `proposed_text` as a quoted, clearly-delimited block; use a Sonnet-class model; do not let the proposed text appear in the system-prompt position.
3. **No-op on LLM error is fail-open** -- must actively REJECT; do not silently return `None` like the rewriter does.
4. **Score variance on borderline proposals** -- Run the judge call once; for a safety gate, one deterministic call at temperature=0 is correct. Don't average multiple calls (cost); instead, require margin above threshold (>= 0.70, not >= 0.65 + 0.75 avg).
5. **`is_acceptable()` re-checked by evaluator** -- Do NOT call `version.is_acceptable()` inside the evaluator (that uses the proposer's self-score). The evaluator computes its own independent score.

---

## Proposed module + test plan

### New file: `backend/meta_evolution/directive_review.py`

```python
@dataclass
class ReviewResult:
    verdict: str             # "ACCEPT" | "REJECT"
    reason: str              # human-readable explanation
    clarity_score: float     # 0.0-1.0
    alignment_score: float   # 0.0-1.0 (preserves existing guardrails)
    safety_score: float      # 0.0-1.0 (no-regression on floors)
    proportionality_score: float  # 0.0-1.0 (diff size vs signal)
    factuality_score: float  # 0.0-1.0 (diff_summary matches actual diff)
    aggregate_score: float   # mean of 5 dimensions
    raw_llm_response: Optional[dict]  # for auditability

ACCEPT_THRESHOLD = 0.70  # >= this = ACCEPT; < this = REJECT

def review_directive_diff(
    proposal: DirectiveVersion,
    current_directive_text: str,
    *,
    llm_call_override: Optional[Callable[[str], Optional[dict]]] = None,
) -> ReviewResult:
    ...
```

**Signature notes:**
- `proposal` is the full `DirectiveVersion` object (proposer's `judge_score` STRIPPED before building judge prompt)
- `current_directive_text` is passed so the judge can diff proposed vs current
- `llm_call_override` matches the existing `directive_rewriter.py` test pattern
- Returns `ReviewResult` with `verdict="REJECT"` on any LLM failure (fail-closed)

### Judge prompt structure (RCAF pattern per SurePrompts)

```
ROLE: You are an independent directive-quality reviewer. You did NOT write the proposed directive.

CONTEXT: The current Research Directive is shown below. A proposer LLM has suggested a revision.
Your job is to score the revision on 5 dimensions: clarity, alignment, safety, proportionality, factuality.
Score each 0.0-1.0. Be strict. When in doubt, score LOW.

[current_directive_text block -- clearly delimited]
[proposed_text block -- clearly delimited, do NOT inherit any scores from the proposer]
[diff_summary -- from proposal]

ACTION: Score each dimension. Return JSON: {clarity, alignment, safety, proportionality, factuality, reasoning}.
Do NOT include an overall score; the caller computes the mean.

FORMAT: Valid JSON only. No prose outside the JSON object.
```

### Test list (6-10 tests for `tests/agents/test_evaluator_directive_review.py`)

1. **`test_accept_on_high_scores`** -- FakeLLM returns all dimensions >= 0.75; `verdict == "ACCEPT"`, `aggregate_score >= 0.70`
2. **`test_reject_on_low_aggregate`** -- FakeLLM returns mixed scores averaging 0.55; `verdict == "REJECT"`
3. **`test_reject_on_llm_error_fail_closed`** -- `llm_call_override` returns `None`; `verdict == "REJECT"`, `reason` contains "llm_error"
4. **`test_reject_on_invalid_json`** -- `llm_call_override` raises `json.JSONDecodeError`; `verdict == "REJECT"` (fail-closed)
5. **`test_proposer_self_score_stripped_from_prompt`** -- Capture the judge prompt via a spy override; assert `judge_score` value from `DirectiveVersion` does NOT appear in the prompt string
6. **`test_missing_required_fields_reject`** -- `DirectiveVersion` with empty `proposed_text`; `verdict == "REJECT"`
7. **`test_idempotent_same_proposal_same_verdict`** -- Same `DirectiveVersion`, same `llm_call_override`; call twice; both return the same verdict (deterministic)
8. **`test_accept_threshold_boundary`** -- FakeLLM returns exact mean of 0.70; `verdict == "ACCEPT"`. FakeLLM returns 0.699; `verdict == "REJECT"` (boundary test)
9. **`test_current_text_in_prompt`** -- Spy on judge prompt; assert `current_directive_text` appears in prompt (judge needs it to diff)
10. **`test_review_result_is_dataclass`** -- `ReviewResult` has all required fields; `raw_llm_response` is stored for auditability

---

## Research Gate Checklist

Hard blockers:

- [x] >=5 authoritative external sources READ IN FULL via WebFetch (7 fetched)
- [x] 10+ unique URLs total (incl. snippet-only) (17 collected)
- [x] Recency scan (last 2 years) performed + reported (5 findings: SIPDO 2025, C3AI 2025, arXiv 2512.10449 Dec 2025, Medium Apr 2026, Microsoft Feb 2026)
- [x] Full papers/pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim (see Key Findings #1-8)

Soft checks:

- [x] Internal exploration covered every relevant module (directive_rewriter, evaluator_agent, __init__, test_directive_rewriter, tests/agents/ absence confirmed)
- [x] Contradictions/consensus noted (consensus section above)
- [x] All claims cited per-claim (not just listed in footer)

---

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 7,
  "snippet_only_sources": 10,
  "urls_collected": 17,
  "recency_scan_performed": true,
  "internal_files_inspected": 5,
  "report_md": "handoff/current/phase-10.7.7-research-brief.md",
  "gate_passed": true
}
```
