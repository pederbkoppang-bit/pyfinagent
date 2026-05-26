# Experiment Results -- Cycle 2: step 27.6 BLOCKED-state evidence

**Date:** 2026-05-26
**Phase:** verification cycle. No SSOT or trading-policy change. No masterplan flip.
**Result:** GENERATE complete; awaiting Q/A.

## What changed (1 new artifact, ZERO code)

### NEW

1. `handoff/current/live_check_27.6.md` -- BLOCKED-state evidence artifact for masterplan step 27.6. Documents:
   - cycle_id=c870fdab + timestamps from today's 20:00 run.
   - Per-criterion table (6 criteria, 2 PASS + 3 FAIL + 1 unknown).
   - BQ verbatim query + result (`analyses_persisted=0` for 2026-05-26).
   - Root cause: Anthropic API credit exhaustion + wrong model (`claude-opus-4-7` vs required `claude-sonnet-4-6`).
   - Structural finding: shared API key between full orchestrator and lite-mode fallback (Portkey "shared credit pool failure mode").
   - Operator remediation chain (3 commands).
   - "BLOCKED" status header so the artifact's grep tokens are NOT mis-parsed as PASS.

### MODIFIED

- `handoff/current/contract.md` -- re-written (autonomous-loop sprint contract clobbered it at 20:36:54 UTC; cycle 2's content restored).
- `handoff/current/research_brief_phase_27_6_smoke.md` -- written by researcher `aa204309cdc5f0761`.

ZERO code changes. ZERO new npm deps. ZERO masterplan status flips.

## Operator-pending decision: Claude Code routing path (cycle 3 candidate)

Operator asked mid-cycle: "could we test using claude code here intead untill we have proven our consept". The pragmatic answer is YES via the Claude Agent SDK + `claude --print --output-format json` subprocess invocation:

- The Max subscription's flat-fee tier covers Claude Code (this CLI) + spawned subagents (researcher, qa). Same rail would cover backend's autonomous-loop analysis pipeline if it called `claude` instead of `api.anthropic.com` directly.
- Implementation path (sketched): add `claude_code_invoke()` to `backend/agents/llm_client.py` behind a feature flag `paper_use_claude_code_route: bool = False` (default OFF, operator opt-in). Replaces direct API calls in `orchestrator.py` Stage-1 / Stage-2 / Stage-3 paths.
- Tradeoffs:
  * PRO: zero per-token cost during testing phase; bypasses the credit-exhaustion blocker.
  * PRO: same Max rail already proven by harness Researcher + Q/A subagents.
  * CON: subprocess cold-start ~200ms per ticker (13 tickers = ~2.6s overhead; acceptable for daily cycle).
  * CON: Claude Code's prompt-shape is markdown-first; need explicit JSON schema in prompt + `--output-format json` to preserve structured-output contract.
  * CON: undocumented as a production-pattern by Anthropic for unattended async use; rate-limit ceiling per Max plan applies.

Operator action: confirm direction. If GO, cycle 3 spawns researcher → contract → generate the routing layer behind a feature flag → Q/A. If NO-GO, cycle 3 picks a code-only masterplan step (36.2 ATR-scaled stops looks like the cleanest P2).

## Verification (verbatim)

```
$ test -f handoff/current/live_check_27.6.md && echo "present"
present

$ grep -c "BLOCKED" handoff/current/live_check_27.6.md
4

$ grep -c "cycle_id=c870fdab" handoff/current/live_check_27.6.md
2

$ grep -c "analyses_persisted=0" handoff/current/live_check_27.6.md
2

$ grep -E "27\.6.*\"status\"" .claude/masterplan.json | head -2
        "id": "27.6",
        "status": "pending"
(masterplan status UNCHANGED -- no premature flip)

$ git diff --stat HEAD -- backend/ frontend/
(empty -- ZERO code changes)

$ git diff HEAD -- frontend/package.json
(empty -- ZERO new deps)
```

## Memory-rule compliance

- ZERO frontend changes.
- ZERO backend changes.
- ZERO new npm deps.
- NO `npm run build`, NO `rm -rf .next/*`.
- ZERO emojis introduced.

## Not in scope

- Operator action items 1-3 (top up credits + flip model + trigger fresh cycle).
- Claude Code routing layer implementation (cycle 3 candidate, operator decision pending).
- Shared-credit anti-pattern fix (provider fallback / split keys; researcher Section 7 recommended follow-up step).

## Cycle 2 surfaces to operator

1. **Production blocker:** Anthropic credits exhausted; today's full-orchestrator path returned 13/13 failures. 0 analyses persisted.
2. **Settings mismatch:** model is `claude-opus-4-7`, 27.6 requires `claude-sonnet-4-6`.
3. **Cycle 3 fork:** confirm Claude Code routing path (Y/N). If Y, cycle 3 implements; if N, cycle 3 closes 36.2 (ATR-scaled stops) or similar code-only step.
4. **Recurring file collision:** `handoff/current/contract.md` overwritten by autonomous-loop sprint contract three times today (19:56, 20:36, ...). Layer-3 harness + harness-optimizer share the path; needs deconfliction.
