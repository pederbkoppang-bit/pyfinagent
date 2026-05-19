# Live check — phase-29.6 (extract code-review heuristics to skill)

**Step ID:** phase-29.6
**Date:** 2026-05-19

## Pre/post line count

```
qa.md before: 439 lines (after phase-29.4 edits)
qa.md after:  221 lines
delta:        -218 lines (~50% reduction)

SKILL.md (new): 234 lines
```

## Content preservation evidence

All phase-29.4 OWASP additions present in new SKILL.md:
- `rag-memory-poisoning` ✓
- `unbounded-llm-loop` ✓
- `agent_config.system_prompt` (enhanced LLM07 cue) ✓
- `BM25 corpus is not subject to Vec2Text` (negation bullet) ✓
- OWASP LLM Top-10 v2.0 source link ✓

All phase-16.59 framework preserved:
- 5 dimensions (Security / Trading-domain correctness / Code quality / Anti-rubber-stamp / LLM-evaluator anti-patterns)
- Top-15 ranked heuristics with severity dispatch
- Simultaneous-presentation rule (cycle-2 spawn guard)
- 5 negation lists

## qa.md frontmatter

```yaml
$ head -19 .claude/agents/qa.md
---
name: qa
description: MUST BE USED ...
tools: Read, Bash, Glob, Grep, SendMessage
model: opus
maxTurns: 12
# phase-29.2 (2026-05-18): codified Opus 4.7 + max effort per operator
# directive (overnight pre-approval). ...
effort: max
memory: project
color: green
permissionMode: plan
skills:
  - code-review-trading-domain
---
```

`skills:` is in YAML block-list form per official sub-agents doc (inline JSON-array form is undocumented).

## qa.md body tail (cross-reference block)

```
$ tail -12 .claude/agents/qa.md
[constraints section ends with: `docs/runbooks/per-step-protocol.md` §4 EVALUATE for full text.]

---

> **Code-review heuristics moved (phase-29.6).** The 5-dimensional code-
> review framework ... are now in
> `.claude/skills/code-review-trading-domain/SKILL.md` — preloaded into
> this Q/A subagent's context at spawn via the `skills:` frontmatter
> entry above. Phase-16.59 research basis preserved at
> `handoff/archive/phase-16.59/research_brief_16_59.md`.
```

## Skill discoverability (this session)

The system surfaced `code-review-trading-domain` in the available-skills list immediately after the SKILL.md was written (visible in this cycle's tool-loading message). This confirms the skill is correctly placed under `.claude/skills/<slug>/SKILL.md` and the frontmatter is parseable.

## Post-restart operator recipe

```
1. /clear or restart Claude Code
2. Confirm Q/A frontmatter snapshot includes `skills:` field (any tool-loading message will show
   code-review-trading-domain in the available-skills list with the skill's full description)
3. Spawn fresh Q/A with prompt: "List the top 5 heuristics from your code-review-trading-domain skill"
   Expected: Q/A returns 5 heuristics from the Top-15 list (secret-in-diff, kill-switch-reachability,
   stop-loss-always-set, prompt-injection-path, broad-except-silences-risk-guard)
   If Q/A says "I don't see code-review-trading-domain", the preload didn't fire — investigate:
   - `grep skills .claude/agents/qa.md` should show the block-list
   - `ls .claude/skills/code-review-trading-domain/` should show SKILL.md
4. Run a small phase-X cycle and confirm Q/A applies the heuristics correctly (verify a planted
   violation, e.g. an obvious API_KEY=... string, triggers secret-in-diff and FAILs the cycle)
```

## Honest disclosure

THIS overnight session's Q/A (called next) is using the OLD snapshotted qa.md with 230 lines of inline heuristics. The preload activation lands on the NEXT (morning) session. This Q/A subagent will verify the file change but won't itself benefit from the cleaner preload path.

If the morning session's first Q/A spawn shows the skill NOT preloaded (rare; covered by GitHub issue #29441 only for team-spawned process-level agents, which this isn't), the workaround is to set `skills:` to inline JSON-array form `skills: ["code-review-trading-domain"]` — but the researcher confirmed the YAML block-list IS the documented canonical form so this should not be needed.
