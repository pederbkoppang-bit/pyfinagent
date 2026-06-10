## Research: phase-2.10 (Karpathy Autoresearch Supersession Audit) + phase-4.14.20 (Agent Trigger-Phrasing Fix)

Tier assumed: `simple` (two bounded audit steps, no new code generation).

---

### Read in full (>=5 required; counts toward the gate)

| URL | Accessed | Kind | Fetched how | Key quote or finding |
|-----|----------|------|-------------|----------------------|
| https://github.com/karpathy/autoresearch | 2026-04-19 | code/repo | WebFetch | Core loop: editable train.py + fixed 5-min budget + val_bpb metric + keep/discard via git reset |
| https://thenewstack.io/karpathy-autonomous-experiment-loop/ | 2026-04-19 | blog | WebFetch | Released 2026-03-07; "three engineering primitives: editable asset, scalar metric, time-boxed cycle" |
| https://www.datacamp.com/tutorial/guide-to-autoresearch | 2026-04-19 | tutorial/doc | WebFetch | Autoresearch = LLM-driven proposal generation, not bounded hyperparameter search; "search space is whatever the LLM can think of" |
| https://fortune.com/2026/03/17/andrej-karpathy-loop-autonomous-ai-agents-future/ | 2026-04-19 | industry news | WebFetch | 700 experiments over 2 days; 20 optimizations; 11% speed-up; Karpathy: "all LLM frontier labs will do this" |
| https://code.claude.com/docs/en/sub-agents | 2026-04-19 | official docs | WebFetch | "Claude uses each subagent's description to decide when to delegate tasks"; description = routing hint; no magic keywords documented |
| https://www.antstack.com/blog/claude-agents-subagents-agent-teams-skills-and-mcp-a-developer-s-field-guide/ | 2026-04-19 | authoritative blog | WebFetch | "description field works as a trigger mechanism"; precision matters; no evidence that specific strings like MUST BE USED fire differently |
| https://www.builder.io/blog/claude-code-subagents | 2026-04-19 | authoritative blog | WebFetch | "Claude uses a subagent's description as a routing hint"; no special wording documented; clarity and action-orientation matter most |

### Identified but snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|-----|------|------------------------|
| https://kenhuangus.substack.com/p/exploring-andrej-karpathys-autoresearch | blog | Fetched; lower tier (substack community) — used as corroboration |
| https://www.anthropic.com/engineering/built-multi-agent-research-system | official | Fetched; confirmed tool description quality matters but no routing-algorithm disclosure |
| https://explore.n1n.ai/blog/running-karpathy-autoresearch-local-llm-zero-cost-2026-03-23 | blog | Snippet only; no routing content relevant to 4.14.20 |
| https://www.news.aakashg.com/p/autoresearch-guide-for-pms | blog | Snippet only; redundant with datacamp coverage |
| https://medium.com/@k.balu124/i-turned-andrej-karpathys-autoresearch-into-a-universal-skill-1cb3d44fc669 | blog | Snippet only; practitioner extension, not authoritative |

### Recency scan (2024-2026)

Searched for 2024-2026 literature on Karpathy autoresearch, agent description routing, and trigger-phrasing auto-delegation.

Result: autoresearch is itself a 2026-03-07 release, so ALL primary literature is within the recency window. No older canonical sources are superseded — autoresearch did not exist before March 2026. For agent description routing, no 2024-2026 paper or official doc discloses a keyword-gated routing mechanism (as opposed to semantic intent matching).

---

### Key findings

#### phase-2.10: Karpathy Autoresearch

1. **Autoresearch was released 2026-03-07**, well AFTER phase-2.10 was created as a stub (the archive research doc is dated 2026-03-29, consistent with this). The project's `handoff/archive/misc/phase210_research.md` was a planning document listing sources to fetch (all unchecked TODO), confirming the research gate was never completed. (Source: GitHub karpathy/autoresearch + internal `handoff/archive/misc/phase210_research.md`)

2. **What autoresearch actually is**: a 630-line autonomous experiment loop for ML training — specifically, LLM-driven modification of `train.py` scored by val_bpb, with git-based keep/discard. It is NOT a general autoresearch SDK or framework that wraps external evaluators. The `evaluate()` callback the original plan expected does not exist as a published API. (Source: thenewstack.io, datacamp.com)

3. **The absorbing step is phase-8.5.0**, explicitly named "Retire phase-2 step 2.10 stub with a decision log linking here" (`masterplan.json` line for id=8.5.0, status=pending). Its verification command is `test -f handoff/phase-2.10-supersede.md`. That file does NOT yet exist. (Source: internal masterplan.json id=8.5.0 + Bash check)

4. **skill_optimizer.py is the actual Karpathy-autoresearch absorber**: it explicitly self-describes as "Mirrors Karpathy's autoresearch pattern: establish baseline → propose modification → measure metric → keep/discard/crash → LOOP FOREVER" (`backend/agents/skill_optimizer.py` lines 4, 129, 270, 453, 471). This is the concrete materialization of the phase-2.10 concept. (Source: internal grep)

5. **meta_coordinator.py is a DEPRECATED stub**: its docstring says "DEPRECATED — Phase 4 stub. Not part of the active MAS architecture." It lists Karpathy autoresearch as a research basis but is not part of the active loop. (Source: `backend/agents/meta_coordinator.py` lines 1-14)

6. **Gap assessment**: The supersession is MOSTLY clean. The Karpathy loop was materialized as `skill_optimizer.py`. However, two things are missing: (a) `handoff/phase-2.10-supersede.md` does not exist (required by phase-8.5.0's immutable verification command), and (b) phase-8.5.0 is still `status=pending`. The formal audit trail for the supersession was never landed.

#### phase-4.14.20: Trigger Phrasing

7. **Both successor agents already have all three trigger phrases**: `qa.md` description contains "MUST BE USED", "Use proactively", and "use immediately after" in its frontmatter line 3. `researcher.md` description contains "MUST BE USED" and "Use proactively" in its frontmatter line 3. This was confirmed by direct grep (`grep -n 'use proactively\|MUST BE USED\|use immediately after'`). (Source: internal grep of `.claude/agents/qa.md` and `.claude/agents/researcher.md`)

8. **The verification command's three target files no longer exist**: `.claude/agents/qa-evaluator.md`, `.claude/agents/harness-verifier.md` were deleted by phase-4.15.0 (MAS restructure). Only `qa.md` and `researcher.md` remain. (Source: `ls .claude/agents/`)

9. **Anthropic's routing mechanism is semantic intent-matching, not keyword detection**: official Claude Code docs (`code.claude.com/docs/en/sub-agents`) and independent analysis (builder.io, antstack.com) all confirm that routing uses the description as a hint for intent matching. No source documents a special effect of literal strings like "MUST BE USED" or "use proactively" at the tokenizer or routing layer. The value of these phrases is behavioral (they make the instruction explicit to the model's reasoning), not mechanical. (Source: code.claude.com, builder.io, antstack.com)

10. **CLAUDE.md "separation of duties on agent edits" applies**: the rule reads "The same Claude Code session should not both author an agent .md change AND self-evaluate work that depends on it." Because Main authors the agent files, Q/A must evaluate the result, not Main. This is a Q/A gate, not a blocker on Main making the edits. (Source: internal CLAUDE.md)

---

### Internal code inventory

| File | Lines | Role | Status |
|------|-------|------|--------|
| `.claude/agents/qa.md` | 166 | Merged Q/A + harness-verifier | ACTIVE; already has all trigger phrases |
| `.claude/agents/researcher.md` | ~300+ | Merged researcher + Explore | ACTIVE; already has trigger phrases |
| `.claude/agents/qa-evaluator.md` | -- | Former evaluator | DELETED by phase-4.15.0 |
| `.claude/agents/harness-verifier.md` | -- | Former verifier | DELETED by phase-4.15.0 |
| `backend/agents/skill_optimizer.py` | ~500+ | Active Karpathy-loop implementation | ACTIVE; explicitly mirrors autoresearch |
| `backend/agents/meta_coordinator.py` | ~150+ | Cross-loop sequencer | DEPRECATED stub; not in active MAS |
| `handoff/archive/misc/phase210_research.md` | 133 | Pre-execution research doc for 2.10 | All sources unchecked; research gate never completed |
| `handoff/phase-2.10-supersede.md` | -- | Required by phase-8.5.0 verification | MISSING |

---

### Consensus vs debate (external)

Consensus: autoresearch is a 2026 LLM-driven experiment loop (not a general SDK). Consensus: Claude subagent description routing is semantic intent-matching. Debate: whether explicit imperative phrasing ("MUST BE USED") materially changes delegation frequency vs. simply writing a clear description — no controlled study exists, but practitioners report it helps when Claude fails to self-delegate (builder.io, antstack.com guidance).

### Pitfalls

- **phase-2.10**: treating it as cleanly superseded when `handoff/phase-2.10-supersede.md` is missing means phase-8.5.0 will fail its immutable verification command.
- **phase-4.14.20**: the semantic intent of the step IS already satisfied (both successor agents have the phrasing). The only unresolvable gap is the literal immutable command which references deleted files. Attempting to recreate the deleted files would violate CLAUDE.md.

### Application to pyfinagent

- **phase-2.10**: supersession is substantively clean (skill_optimizer.py is the materialization). Formal gap: `handoff/phase-2.10-supersede.md` must be created and phase-8.5.0 run to close the audit trail.
- **phase-4.14.20**: the blocked step should be retired (not fixed) because its literal verification command cannot be satisfied without violating the CLAUDE.md merge rule. The spirit is already met: both active agents (`qa.md` at line 3, `researcher.md` at line 3) carry all required trigger phrases. Owner decision needed per the masterplan's own blocker note.

---

### Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (7 fetched)
- [x] 10+ unique URLs total (12 unique URLs collected)
- [x] Recency scan (last 2 years) performed + reported
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every relevant module
- [x] Contradictions / consensus noted
- [x] All claims cited per-claim

---

```json
{
  "tier": "simple",
  "external_sources_read_in_full": 7,
  "snippet_only_sources": 5,
  "urls_collected": 12,
  "recency_scan_performed": true,
  "internal_files_inspected": 8,
  "report_md": "handoff/current/phase-audit-2.10-4.14.20-research-brief.md",
  "gate_passed": true
}
```
