---
name: research-gate-depth-86-73
description: Design research on scaling the research gate's depth (single-researcher floors vs parallel fork) and on who assesses difficulty; the env.tier gap and the unused opts.floors seam
metadata:
  type: project
---

Step 86.73 design research (2026-08-13). Brief:
`handoff/current/research_brief_86.73.md`.

**Anthropic's own guidance MOVED against fan-out, and the project cites the older
post.** `researcher.md:255-270` justifies the multi-subagent fork with the 2025
multi-agent-research post ("more than 10 subagents"). The **2026-01-23** official
post `claude.com/blog/building-multi-agent-systems-when-and-how-to-use-them` is
later and narrower: *"Start with single-agent systems"*, multi-agent *"reserved
for cases where they provide clear benefits that justify the additional cost"*,
**3-10x tokens** for equivalent tasks, and *"the primary benefit of
parallelization is thoroughness, not speed"*. Cite both or you are quoting stale
vendor guidance.

**Two verified code facts that any tier/fork design inherits:**
- `grep -n "env\.tier" .claude/workflows/research-gate.js` returns **ZERO hits**.
  `enforceGate` never compares the tier the agent RETURNED against the tier the
  caller REQUESTED. The schema (`:285`) pins the returned string to
  `VALID_TIERS`, but a researcher asked for `complex` that returns `simple`
  raises no violation. The anti-de-escalation ratchet exists only in prose
  (`researcher.md:204`, `:401`).
- `enforceGate` already takes per-call floors —
  `const floors = (opts && opts.floors) || {...}` at `research-gate.js:365` —
  and `grep -n "floors:"` finds **no caller anywhere**, including the checker.
  It is a live but UNEXERCISED seam: per-tier floors need no new plumbing, but a
  first use needs a mutation cell, because nothing has ever proven a raised floor
  actually fails a brief that meets only the old one.

**Why:** the operator asked for evidence, not preference, on (a) raise floors on
one researcher vs (b) fork 2-3 parallel researchers, and on caller-declared vs
self-assessed difficulty.

**How to apply:** for (a) vs (b) — duplication is the LARGEST measured
multi-agent failure mode (MAST FM-1.3 step repetition = **17.14%**;
`arxiv.org/html/2503.13657v2`), and the anti-duplication value in parallel deep
research comes from an explicit plan DAG, not from parallelism (removing it cost
coverage 4.31->4.10 and 4.7x runtime; `arxiv.org/html/2604.24978`). At matched
compute, multi-agent debate "significantly underperforms simple self-consistency"
(Huang et al., `ar5iv.labs.arxiv.org/html/2310.01798`). Fan-out onto this rail is
additionally blocked by structure, already written down at
`research-gate.js:190-200`: one brief path, one stage-2 verifier, no cross-branch
de-dup.

For difficulty assessment — do NOT cite Snell et al.
(`arxiv.org/html/2408.03314v1`, "model-predicted difficulty bins largely overlap
with oracle") as support for an agent VERBALLY rating its own task. Snell's
estimate is a learned verifier's score averaged over 2,048 samples. Verbalized
self-confidence is "comparatively insensitive to task difficulty" (ECE 43.4% on
expert-level, `arxiv.org/html/2506.00582`), and **web-search agents are the
worst-calibrated tool class measured** — mean confidence 0.86-0.97 when WRONG
(`arxiv.org/html/2601.07264`). The two usable levers are the AFCE result (elicit
the estimate BEFORE and SEPARATELY from the work; ECE -58.4%) and a
one-directional ratchet (escalate-only), which is the same construction already
used for `audit_class` at `rules/research-gate.md:169-171`.

**What is NOT in the literature:** no source measures an agent deliberately
DE-ESCALATING its own difficulty rating to finish sooner. Premature-exit work
(`arxiv.org/html/2505.17616v2`) explicitly does not separate "task done" from
"minimise compute". The ratchet is a cheap precaution, not a fix for a measured
phenomenon — say so rather than implying evidence exists.

See [[project_multi_agent_harness_guidance]], [[project_research_gate_discipline]].
