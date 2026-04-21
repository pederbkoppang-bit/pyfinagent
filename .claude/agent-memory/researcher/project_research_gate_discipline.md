---
name: Research gate discipline — >=5 sources floor and stale file locations
description: Phase-4.16.1 raised the research gate floor to >=5 sources read in full; stale files that still say 3-5 identified in phase-4.16.3
type: project
---

Phase-4.16.1 raised the mandatory source floor from >=3 to >=5 sources read in full per harness step. Phase-4.16.3 audited which files still contain the old floor.

**Why:** Anthropic's multi-agent research system documented that agents default to SEO-optimized content farms over authoritative sources. The >=5 floor plus read-in-full (not abstract-only) directly counters this failure mode. The 7-of-9 phase-4.8 research-gate misses drove the phase-4.16 rule-hardening effort.

**How to apply:**
- When citing the research gate floor, always say ">=5" not "3-5". The old number appears in `.claude/context/research-gate.md` line 10 and may appear in older contracts/briefs — do not repeat it.
- Stale files to update (as of 2026-04-18):
  - `.claude/context/research-gate.md`:10 — says "3-5", needs ">=5 (raised in phase-4.16.1)"
  - `.claude/context/mas-architecture.md`:13-14 — lists old 4-agent topology (QA Evaluator + Harness Verifier separately); correct is 3-agent MAS per `docs/runbooks/per-step-protocol.md`
- If a `feedback_research_gate_min_three_sources.md` memory file exists in the main agent's memory path, its title and body should be updated to say ">=5".
