---
step: phase-16.35
title: Research -- Claude Code / Claude Code Remote as Anthropic API substitute
cycle_date: 2026-04-25
harness_required: true
forward_cycle: true
parent_phase: phase-16
deliverable: docs/architecture/claude-code-as-api-substitute.md
---

# Sprint Contract -- phase-16.35

User's question: *"add a new step where we use claude code and claude code remote instead for #21 Anthropic key swap (user action) research whether this could work and also give me upsides and downsides"*

Pure research deliverable. No code. The output is a design doc.

## Research-gate summary

`handoff/current/phase-16.35-research-brief.md`. tier=moderate, 8 in-full, 18 URLs, recency scan, gate_passed=true.

## Key findings (all 2026 sources, decisive)

1. **April 4, 2026 Anthropic policy change** ended third-party subscription quota access. Max can no longer pay for programmatic SDK calls. (PYMNTS.com)

2. **Agent SDK requires `ANTHROPIC_API_KEY=sk-ant-api03-*`** — Max subscription cannot pay for SDK calls. Officially documented. (Agent SDK overview, Anthropic support)

3. **"Claude Code Remote" is a mobile-sync feature, NOT a cloud API.** Phone connects to a session running on your Mac. No programmatic endpoint exposed. (Remote Control docs)

4. **Setting `ANTHROPIC_API_KEY` in env silently switches Claude Code CLI from subscription to API billing.** Real risk if user adds key to `backend/.env` and any shell sourcing leaks it. (Anthropic support article 11145838)

5. **Grey area:** `claude -p "prompt"` without `ANTHROPIC_API_KEY` set technically uses subscription quota, but Anthropic policy prohibits this for "third-party developers offering claude.ai login." Personal automated use is unenforced but not guaranteed.

6. **Subprocess Claude Code adds 500-1500ms latency per call** + complex async integration, for zero cost benefit at pyfinagent volumes (~$3-10/mo native API).

## Hypothesis

The honest answer to the user's question is **NO** — Claude Code / Claude Code Remote cannot substitute for the API key. Anthropic explicitly closed this loophole in April 2026. The recommendation: pay the $3-10/mo for a real `sk-ant-api03-*` key (the existing #21 path), and ALSO be careful about ANTHROPIC_API_KEY env-var scoping so it doesn't leak into Claude Code shell sessions.

The doc presents this conclusion with full evidence + 2 alternative-path notes (subscription grey-area pros/cons, hypothetical future Claude Code Remote API).

## Success Criteria (verbatim, immutable)

```
test -f docs/architecture/claude-code-as-api-substitute.md && grep -qE 'Recommendation|Upsides|Downsides|Implementation sketch' docs/architecture/claude-code-as-api-substitute.md
```

- doc_exists
- claude_code_sdk_evaluated
- claude_code_remote_evaluated
- cost_comparison_present
- implementation_sketch_present
- honest_recommendation

## Plan steps

1. (DONE during research gate) Researcher wrote `docs/architecture/claude-code-as-api-substitute.md` (528 lines)
2. Verify file exists + grep matches required keywords
3. Spawn Q/A to audit doc honesty + completeness

## What Q/A must audit

1. Doc exists at the canonical path; verification grep PASSES
2. All 10 sections present (current state, Option A, Option B, upsides, downsides, implementation sketch, cost analysis, reliability, recommendation, what-this-closes)
3. Cited sources are real (spot-check at least 1 URL via curl)
4. Recommendation is honest given the April 4, 2026 policy reality
5. The env-var-scoping caveat is documented (otherwise user could brick their Claude Code subscription)
6. Doc explicitly states whether this CLOSES #21 or not (it doesn't — it confirms #21 is the only path)
