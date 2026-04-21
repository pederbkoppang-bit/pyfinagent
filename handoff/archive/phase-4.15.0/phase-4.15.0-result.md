# phase-4.15.0 Result -- MAS restructure to 3 agents

## Changes

### Agent definition files

Before: 4 `.md` files in `.claude/agents/`
After: 2 `.md` agent files + 1 runbook moved out

```
.claude/agents/
├── qa.md          (NEW -- merged qa-evaluator + harness-verifier)
└── researcher.md  (UPDATED -- absorbed Explore role)

docs/runbooks/
└── per-step-protocol.md  (MOVED from .claude/agents/)
```

Deleted:
- `.claude/agents/qa-evaluator.md`
- `.claude/agents/harness-verifier.md`

### Per-agent content changes

**`qa.md`** (merged):
- Description uses "MUST BE USED" + "use proactively" trigger
  phrasing (doc-recommended, was missing on both predecessors)
- Runs ONCE per cycle, not in a parallel pair
- Deterministic-first verification order preserved:
  syntax → `verification.command` → existing evaluator_critique →
  optional dry-run → LLM judgment (last resort)
- LLM judgment leg now explicitly covers: contract alignment,
  anti-rubber-stamp (mutation-resistance), scope honesty,
  research-gate compliance
- "Never second-opinion-shop" added to constraints

**`researcher.md`** (expanded):
- Description uses "MUST BE USED before every PLAN phase" trigger
  phrasing
- Explicitly has TWO halves in one session:
  1. External literature (papers, docs, blogs, GitHub)
  2. Internal code exploration (the former `Explore` subagent)
- Report format extended with "Internal code inventory" table
  alongside the existing external sources table
- `maxTurns: 20` (up from 15) for the larger scope
- Output envelope includes `internal_files_inspected` count

### Canonical references + CLAUDE.md

- `CLAUDE.md` "Critical Rules" section updated:
  - MAS described as "exactly 3 agents: Main + Researcher + Q/A"
  - References point at `docs/runbooks/per-step-protocol.md`
    (not the old `.claude/agents/per-step-protocol.md`)
- Architecture section updated: Layer 3 now says "Harness MAS
  (exactly 3 agents)" with explicit "No separate Explore. No
  separate harness-verifier. Don't re-split."
- Five-file protocol table: EVALUATE row says "Q/A verdict (single
  agent, merged qa-evaluator + harness-verifier)"
- "Dual-evaluator rule" section renamed to "Single-Q/A rule"
- Research gate section notes the MUST-BE-USED enforcement layers
- "Never do" list adds:
  - "Skip RESEARCH because we've been here before"
  - "Re-split agents (reintroducing Explore or harness-verifier)"
  - "Second-opinion-shop after CONDITIONAL"

### Runbook (`docs/runbooks/per-step-protocol.md`)

Rewritten end-to-end to describe the 3-agent MAS:
- ASCII diagram of Main -> Researcher/Q/A delegation
- 5-phase sequence: RESEARCH / PLAN / GENERATE / EVALUATE / LOG
- Each phase names exactly one spawn (not two)
- New "Why Main drifts" subsection documents the three known
  drift modes with fixes (see 4.15.1 below)
- Anti-patterns list includes "Re-split agents" as #6

## Verification

```
$ ls .claude/agents/*.md | grep -vE 'researcher|qa' | grep -v per-step-protocol | wc -l
0
```

Exit 0 -- only `researcher.md` and `qa.md` remain in the agent
directory; `per-step-protocol.md` relocated to docs/runbooks.

```
$ grep -c 'MUST BE USED\|use proactively' .claude/agents/qa.md .claude/agents/researcher.md
.claude/agents/qa.md:2
.claude/agents/researcher.md:2
```

Both agents have the doc-recommended trigger phrasing (4 total
matches).

## Success criteria (all met)

- [x] only_researcher_and_qa_md_files_exist
- [x] researcher_md_includes_internal_code_exploration_role
- [x] qa_md_includes_harness_verifier_reproducibility_role
- [x] obsolete_agent_files_deleted_or_archived (per-step-protocol
      moved to docs/runbooks; qa-evaluator + harness-verifier
      deleted)
- [x] CLAUDE_md_updated_to_reflect_3_agent_MAS
