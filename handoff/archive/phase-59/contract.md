# Contract — Step 59.3

**Step id:** 59.3 — Stress-test doctrine for the Fable 5 release (harness-free comparison)
**Date:** 2026-06-11
**Phase:** phase-59 (operator answer 2026-06-11: include the stress test)
**Researcher gate:** PASSED — `handoff/current/research_brief.md` (tier=moderate, 7 external sources read in full, 45 URLs, recency scan; 15 internal files; envelope `gate_passed: true`)

## Research-gate summary

Step selection: **55.2 (ops + agent-quality audit)** — pure analysis (a bare CODE step would write conflicting code), richest artifact chain (7 files), and the only chain whose Q/A hostile-re-derived every claim against live BQ → 10 pre-registered ground-truth anchors (GT1-GT10) + 3 operator-voice premise probes (P1 "llm_call_log is the ground truth of what fired", P2 "lite runs rag/earnings_tone/insider/patent/news", P3 SNDK "7.0→5.0"). Contamination design (the validity threat): 13-row leakage inventory; the 56.x/57.1 fixes INVERT live-code ground truth (dead button removed; REJECT now flag-gated) → code reads pinned in a disposable worktree at `70a8242b` (the commit before the 55.2 PASS), whose own gotcha (a partial research_brief_55.2.md captured by the 18:35 chore commit) is neutralized by `rm -rf <wt>/handoff` + restoring ONLY the two original inputs (goal doc + 55.1 post-mortem — input parity with the original chain); prompt-level do-not-read list for what the pin can't reach (live handoff/, CHANGELOG, masterplan, memories; BQ date-filter ≤2026-06-10, forbid harness_learning_log/agent_memories); scoring weights PROCESS dimensions over answer-matching to absorb residual leakage, and suspiciously-specific unexplained matches count as leakage evidence not skill. Five scored dimensions (D1 accuracy /10 with −1 confident-wrong; D2 premise-catching /3; D3 evidence-rigor counts + fabrication spot-test; D4 coverage /9 operative requirements; D5 overclaim/hedge counts) + D6 overhead (reported: harness ≈31-70 min, 3 sessions, ~100KB/7 artifacts). Judging is RUBRIC-ANCHORED CHECKLISTS against this pre-registration — never holistic pairwise (position bias peaks when both candidates are good, arXiv:2406.07791; Claude-judges-Claude self-preference, arXiv:2410.21819). Per-component decision rules PRE-REGISTERED in brief §E (anti-motivated-reasoning), with honest scope limits: handoff durability + QA deterrence are unobservable in a single pass; component verdicts are ATTRIBUTED-NOT-ISOLATED (Anthropic's documented method is component-at-a-time; this all-at-once run measures the joint effect — a component-wise confirmation run is the operator-gated follow-up; arXiv:2604.07236 corroborates that stronger models substitute for structure rather than erase its value). Recency: NO independent Fable 5 agentic evals exist yet (2 days post-release) — 59.3 generates primary evidence; running before the June-22 Max window end is $0 marginal. Execution: bare-task prompt (criteria withheld — the contract phase is itself under test) pasted VERBATIM from brief §D (Main holds the answers; improvising the prompt risks leak-by-framing); ONE run, no retries/prompt-iteration (scaffolding by another name); technical-failure re-runs disclosed.

## Hypothesis

A single bare Fable 5 pass will land several GT anchors but miss premise corrections and coverage items that the harness's researcher gate and contract phase produced, and will carry more overclaims than the QA-gated chain — yielding evidence-based KEEP verdicts for most components with at most MODIFY candidates; if the bare pass instead scores at-threshold, the pre-registered rules force honest PRUNE-candidate recommendations (operator-gated).

## Immutable success criteria (verbatim from .claude/masterplan.json, step 59.3)

1. "one representative, already-completed masterplan step is re-run WITHOUT the harness (single Fable 5 pass: no researcher/qa subagents, no contract/handoff scaffolding) and the output is saved verbatim; the chosen step and why it is representative are justified in the comparison doc"

2. "the comparison evaluates harness-vs-harness-free output on at least 3 named dimensions (e.g. factual accuracy/citation quality, criteria coverage, error-catching depth) with concrete examples from both artifacts, and renders a keep/prune/modify verdict for each major scaffolding component (researcher gate, contract phase, separate qa, handoff files, turn caps)"

3. "any prune/modify proposal is presented as an OPERATOR-GATED recommendation with expected savings and risks -- nothing is removed from the harness in this step; an honest 'keep everything' is a valid outcome"

4. "the comparison + verdict live in handoff/current/59.3-stress-test-comparison.md with the evidence excerpts in live_check_59.3.md"

**Verification command (immutable):** `test -f handoff/current/59.3-stress-test-comparison.md && test -f handoff/current/live_check_59.3.md`

## Plan

1. Worktree setup per brief §B Layer 1 (pin `70a8242b`, scrub handoff+masterplan, restore the 2 input docs into `<wt>/handoff/context/`).
2. Spawn ONE general-purpose subagent, `model: fable`, with the §D prompt VERBATIM (bare-task, blinded, output to `handoff/current/59.3-harness-free-output.md`). One run; save telemetry (tokens/tool-uses/duration) for D6.
3. Score per §C rubrics against GT1-GT10 / P1-P3 / the 9 operative requirements (rubric-anchored; fabrication spot-test by re-running 2-3 of the agent's boldest queries; flag suspicious unexplained matches as leakage).
4. Render per-component verdicts via the §E pre-registered rules (attributed-not-isolated labeling; component-wise confirmation run proposed as the operator-gated follow-up where relevant).
5. Write `59.3-stress-test-comparison.md` (selection justification, leakage disclosure, dimension tables with concrete examples from BOTH artifacts, verdicts, operator-gated recommendations with savings + risks) + `live_check_59.3.md` (harness-free output excerpts, telemetry, the 3-dimension+ table, verdicts).
6. Teardown the worktree. experiment_results.md → ONE fresh Q/A → harness_log → flip (CLOSES phase-59).

## Constraints

- The bare run must not modify the repo (single output doc only; worktree disposable); review-only; $0 (Max-session subagent; bounded BQ reads); no retries/prompt-iteration; all residual leakage disclosed; prune/modify = operator-gated proposals only; "keep everything" valid.

## References

- handoff/current/research_brief.md (researcher 59.3, gate_passed: true; Anthropic harness-design [re-read in full], Fable-5 announcement, arXiv:2604.07236, arXiv:2406.07791, arXiv:2410.21819, arXiv:2502.17521, arXiv:2603.05344; snippet tier incl. arXiv:2605.30621)
- Archived 55.2 chain: handoff/archive/phase-55.2/ (brief/contract/audit/critique) — the harness side of the comparison
- CLAUDE.md "Stress-test doctrine (Anthropic)" section; operator answer 2026-06-11 (include)
