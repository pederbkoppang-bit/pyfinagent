# Contract -- Cycle 2: step 27.6 BLOCKED-state evidence + operator action surface

**Cycle:** 2 (production-readiness mode)
**Date:** 2026-05-26
**Step targeted:** masterplan `27.6` "End-to-end smoke verify: full path on Claude" (P0)
**Class:** verification cycle (NOT a trading-policy change; citation floor does NOT apply)
**Status flip:** NONE -- 27.6 stays `pending` because the criteria cannot be satisfied without operator action. Operator-approved cycle-3 path: Claude Code CLI routing to bypass the credit-exhausted Anthropic rail.

**File-collision note (FOURTH occurrence today):** the autonomous-loop's parameter-optimization sprint clobbered `handoff/current/contract.md` four times today (19:56, 20:36, 20:47, and likely-again). Both Layer-3 harness (this cycle) and the harness optimizer write to this same path. Re-writing the cycle-2 content over the sprint stub each time is the current workaround. Permanent deconfliction (separate paths or discriminator field) is on the follow-up backlog. This document supersedes the parameter-optimization stub for cycle 2.

## Research gate

- Researcher `aa204309cdc5f0761`, tier=moderate, 6 sources read in full, 8 snippet-only, 14 URLs, recency scan performed, internal_files_inspected=5, **gate_passed=true**.
- Brief: `handoff/current/research_brief_phase_27_6_smoke.md`.

## N* delta

- **B primary:** convert "operator doesn't know the cycle is broken" into "operator has a verbatim audit-grade artifact naming the blocker, the BQ evidence, the remediation chain, and the cost rail". Today's autonomous run at 20:00:41 CEST failed silently from the operator's perspective (UI showed "0 trades"); underlying cause is Anthropic credit exhaustion + a wrong-model setting (`claude-opus-4-7` not `claude-sonnet-4-6`).
- **R secondary:** the BLOCKED-state evidence enables cycle 3 to ship the Claude Code routing layer with full context.

## Empirical findings (researcher Section 2)

Today's cycle `cycle_id=c870fdab` at 2026-05-26 20:00:41 CEST:

| # | Criterion (verbatim from masterplan) | Status | Evidence |
|---|---|---|---|
| 1 | model = claude-sonnet-4-6 via settings API | **FAIL** | currently set to `claude-opus-4-7` |
| 2 | full cycle completed, status=completed | PASS | 20:06:36 CEST cycle complete logged |
| 3 | lite_mode=False observed in Step 3 log | PASS | `Step 3 -- Analyzing 4 new + 9 re-evals (lite_mode=False)` |
| 4 | zero "Full orchestrator failed" lines | **FAIL** | 13 of 13 failed (Anthropic credit exhaustion 400) |
| 5 | min 14/15 analyses persisted to BQ analysis_results | **FAIL** | 0 rows for 2026-05-26 |
| 6 | OutcomeTracker step 9 logged | unknown | Step 9 gated on `closed_tickers != []`; no closures today |

Five of six criteria FAIL. Step 27.6 cannot close PASS today.

## Structural finding (out of scope; follow-up backlog)

Researcher Section 7 documents a **shared-credit anti-pattern**: backend uses ONE Anthropic API key for BOTH the full orchestrator and the lite Claude fallback at `autonomous_loop.py:1322-1328`. When the key fails, both paths fail in unison. Portkey 2026 "shared credit pool failure mode". The operator-approved cycle-3 Claude Code routing bypasses this entirely (Max-subscription flat-fee rail).

## Scope -- 1 new evidence artifact, ZERO code

### NEW

1. `handoff/current/live_check_27.6.md` -- BLOCKED-state evidence with verbatim cycle_id, BQ query + result, 6-criterion table, operator remediation chain, BLOCKED header so future readers don't mis-parse as PASS. Supersedes the prior 2026-05-17 capture; preserves git history.

### ZERO code changes

Verification-only. No backend, no frontend, no tests.

## Operator action (approved direction; cycle-3 scope)

The operator approved 2026-05-26: route through Claude Code CLI for testing phase until production Anthropic key is set up. Cycle 3 will implement:

- Feature flag `paper_use_claude_code_route: bool = False` (default OFF, operator opt-in).
- `backend/agents/llm_client.py::claude_code_invoke()` shells out to `claude --print --output-format json <prompt>` on the Max-subscription rail.
- Stage-1 / Stage-2 / Stage-3 call sites in `orchestrator.py` switch on the flag.
- Cycle 4+ flips the flag ON and re-runs the 27.6 verification.

Alternative path (if Claude Code routing proves infeasible in cycle 3): top up Anthropic credits + flip model setting + trigger fresh cycle.

## Immutable success criteria (cycle 2 itself, NOT 27.6)

1. `handoff/current/live_check_27.6.md` exists.
2. Artifact contains verbatim `cycle_id=c870fdab`.
3. Artifact contains the 6-criterion table.
4. Artifact contains the operator remediation chain.
5. Artifact contains a BLOCKED header so future readers don't mis-parse it as PASS.
6. Researcher brief `handoff/current/research_brief_phase_27_6_smoke.md` exists with gate_passed=true.
7. `handoff/current/experiment_results.md` documents this cycle's outputs.
8. ZERO code changes (frontend or backend).
9. ZERO new npm deps.
10. NO `npm run build`, NO `rm -rf .next/*`.
11. `masterplan.json` `27.6.status` UNCHANGED at `pending` (no premature flip).
12. **THIS contract.md** (cycle-2 trading-verification content + File-collision preamble + researcher `aa204309cdc5f0761` cite) is on-disk at commit time. If the autonomous-loop overwrites it again between now and Q/A, Main re-writes immediately before re-spawning Q/A.

## /goal integration gates

1. AST parse N/A (no code). 2. Log LAST. 3. No self-evaluation. 4. Citation floor N/A (verification cycle). 5. Researcher gate_passed=true.
