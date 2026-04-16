# MAS / Harness Loop — F1 (retry loop) + F2 (research-on-demand)

**Drafted:** 2026-04-16
**Cycle type:** infrastructure fix (no masterplan step; tracked in harness_log for audit)
**Research tier:** moderate (WebFetch of both Anthropic canonical posts in full)

## Hypothesis
Wiring an explicit retry loop (evaluator FAIL → revert + re-plan with critique, escalate to certified fallback after N consecutive failures) + a planner-triggered researcher spawn (when the planner emits `research_needed=True`) closes the two critical gaps between our harness and the Anthropic multi-agent reference pattern.

## Source-of-truth citations (researcher WebFetch'd these in full)

- **https://www.anthropic.com/engineering/multi-agent-research-system** — "The LeadResearcher synthesizes these results and decides whether more research is needed — if so, it can create additional subagents or refine its strategy." Brief format: "an objective, an output format, guidance on the tools and sources to use, and clear task boundaries." Scaling: "simple fact-finding ~1 agent; complex research 10+ subagents." No hardcoded retry cap — qualitative "sufficient information gathered" threshold.
- **https://www.anthropic.com/engineering/building-effective-agents** — Evaluator-optimizer loop: "One LLM call generates a response while another provides evaluation and feedback in a loop." **Feedback**, not just score — "both are required."
- **https://code.claude.com/docs/en/best-practices** — "Include tests, screenshots, or expected outputs so Claude can check itself. This is the single highest-leverage thing you can do." Durable-state-before-mutation: "Every action Claude makes creates a checkpoint."
- **https://code.claude.com/docs/en/agent-teams** — Plan-approval reject-and-revise loop: "If rejected, the teammate stays in plan mode, revises based on the feedback, and resubmits." Auto mode: "aborts if the classifier repeatedly blocks actions" (finite cap exists, number not published). Brief structure for spawned teammates: topic + file scope + focus areas + output format.

## Success criteria

### F1 (retry loop)

1. `run_harness.py` main loop introduces a `consecutive_fails` counter initialized to 0 before the cycle loop.
2. On `grades["verdict"] == "PASS"` or `"CONDITIONAL"`: reset `consecutive_fails = 0`.
3. On `grades["verdict"] == "FAIL"`: increment `consecutive_fails`; call `save_best_params(pre_cycle_best)` (revert, per agent-teams "revise-not-restart").
4. When `consecutive_fails >= MAX_CONSECUTIVE_FAIL` (3): emit certified-fallback to `CERTIFIED_FALLBACK_BEST_PATH` if configured, write a clear warning to `handoff/harness_log.md`, and exit the cycle loop early.
5. `previous_critique` (the full `grades` dict, not just the verdict) is passed forward into the next cycle's `run_planner()` call — already the case; verify the dict contains `{verdict, statistical_validity, robustness, simplicity, reality_gap, weak_periods, cost_2x_sharpe, composite}` so the planner can re-plan with structured feedback.

### F2 (research-on-demand)

6. `run_planner()` output gains an optional `plan["research_needed"]` bool + `plan["research_brief"]` dict.
7. `research_brief` follows Anthropic's format: `{objective, output_format, tool_scope, task_boundaries}`.
8. New helper `run_planner_with_research(cycle, previous_critique)` wraps `run_planner`; if `research_needed`, spawns the `researcher` agent up to `MAX_RESEARCH_ITER=3` times and re-invokes the planner with `research_context` injected.
9. Heuristic gap detection inside `run_planner`: when the planner detects a **plateau** (its existing `PLATEAU` suggestion) AND the planner's `excluded_params_count >= 10`, set `research_needed = True` with a `research_brief` describing "recover Sharpe above plateau — research alternative strategy switches or regime-specific parameter sets."
10. Researcher output written to `handoff/current/research.md`; planner reads it on re-invocation.

### F3 (shared)

11. Both features are feature-flagged with constants at top of `run_harness.py` (`MAX_CONSECUTIVE_FAIL`, `MAX_RESEARCH_ITER`, `CERTIFIED_FALLBACK_BEST_PATH`) so they're operator-tunable without code changes.
12. No new external dependencies.
13. `python -c "import ast; ast.parse(open('scripts/harness/run_harness.py').read())"` passes.
14. `python scripts/harness/run_harness.py --dry-run --cycles 1` still returns `HARNESS COMPLETE` (F2 only triggers on live, non-dry-run cycles when plateau criteria are met).

## Out of scope
- LLM-backed planner (replacing the rule-based heuristic entirely). F2 uses the EXISTING heuristic's `PLATEAU` signal to trigger research, without replacing the planner's core logic. Full LLM-planner is a future phase.
- Parallel Planner/Generator execution (Anthropic F5 optimization, acceptable deviation).
- Mandatory worktree isolation for verifiers (F4, acceptable deviation).
- Parallel TaskCompleted hook fan-out for both verifiers (F3 verifier parallelism).

## Anti-patterns guarded
- **Revert vs restart**: per agent-teams "prefer revise/revert over restart"; we revert `optimizer_best.json` and pass the critique forward, we don't tear down the cycle state.
- **Structured critique, not bare score**: `previous_critique` is the full `grades` dict, not `verdict` alone.
- **Durable state before mutation**: `pre_cycle_best = copy.deepcopy(load_best_params())` already runs before `run_generator`; we don't remove it.
- **Not a hardcoded retry topology**: `MAX_CONSECUTIVE_FAIL` and `MAX_RESEARCH_ITER` are implementer-tunable; Anthropic doesn't prescribe a number.

## Files to modify
- `scripts/harness/run_harness.py` — retry loop + research helper + constants.
- `handoff/current/mas-harness-fixes-contract.md` — this file (PLAN artifact).
- `handoff/harness_log.md` — LOG entry at end.

## Verification
- Syntax: `python -c "import ast; ast.parse(open('scripts/harness/run_harness.py').read())"`.
- Dry-run: `source .venv/bin/activate && python scripts/harness/run_harness.py --dry-run --cycles 1` → `HARNESS COMPLETE`.
- Retry smoke: force a synthetic FAIL verdict 3 times; confirm the loop exits with a certified-fallback log line.
- Research-brief smoke: force `plateau=True` with 10+ excluded params; confirm `run_planner_with_research` spawns researcher.md and writes `handoff/current/research.md`.
- Both verifiers (qa-evaluator + harness-verifier) spawn in parallel; expect `ok:true` from both.
