# Research Brief — Step 67.3: Codify WRITE-FIRST discipline in researcher.md

**Status: IN PROGRESS (write-first skeleton).** This file is written
incrementally as sources are read, per the very discipline this step
codifies. If it ends with `gate_passed: false`, the gate was not met
and the gaps are listed honestly.

Tier: **simple** (floors still apply: >=5 read-in-full, >=10 URLs,
recency scan, three query variants).

## Question

Step 67.3 asks Main to (1) codify WRITE-FIRST / incremental-brief
discipline in `.claude/agents/researcher.md`, (2) prune stale
scaffolding, and (3) weaken NO research-gate floor. This brief covers
the internal half (where the discipline belongs, what is stale, what
floors must be preserved) and the external half (Anthropic's Fable 5
prompting guidance + long-horizon research-agent output discipline).

---

## Read in full (>=5 required; counts toward the gate)
| # | URL | Accessed | Kind | Fetched how | Key finding |
| --- | --- | --- | --- | --- | --- |
| 1 | https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/prompting-claude-fable-5 | 2026-07-09 | official doc | WebFetch (full) | De-prescription: "steer most behaviors with a brief instruction rather than enumerating each behavior by name"; "Skills developed for prior models are often too prescriptive for Claude Fable 5 and can degrade output quality." "Construct a memory system": "Store one lesson per file with a one-line summary at the top." Early-stopping failure mode = "text-only statement of intent ('I'll now run X') without issuing the corresponding tool call" -> remedy: "Before ending your turn, check your last paragraph. If it is a plan... do that work now with tool calls." "Ground progress claims": "Before reporting progress, audit each claim against a tool result." PITFALL: "Don't instruct Claude to reproduce its reasoning in the response" -> triggers reasoning_extraction refusal + Opus fallback. |
| 2 | _pending fetch_ | | | | |

## Identified but snippet-only (context; does NOT count toward gate)
| URL | Kind | Why not fetched in full |
| --- | --- | --- |
| https://arxiv.org/html/2604.13018v1 (Autonomous Long-Horizon Engineering for ML Research) | paper | artifact-mediated project state; may fetch |
| https://zylos.ai/research/2026-05-15-long-horizon-agent-goal-persistence/ | blog | artifact-based context bridges |
| https://www.llamaindex.ai/blog/long-horizon-document-agents | blog | vendor |
| https://arxiv.org/pdf/2606.11926 (Hypothesis-Tree Refinement) | paper | externalized state |

## Recency scan (2024-2026)
IN PROGRESS. Current-year 2026 search returned arXiv:2606.11522 "Search
Discipline for Long-Horizon Research Agents" (emphasizes "writing long
instructions to files and leaving artifacts for audit, avoiding partial
reads") and arXiv:2604.13018 (artifact-mediated project state vs
"transient conversational handoffs"). Fable 5 prompting guide (2026)
directly supersedes prior-model prescriptive-prompt advice. Full finding
after fetches.

## Key findings
1. (Fable 5 doc) The 132K-burn failure mode is a NAMED Fable behavior:
   ending a turn on a "statement of intent" without the tool call.
2. (Fable 5 doc) Over-prescriptive skills DEGRADE Fable output -> the
   write-first section must be SHORT (goal + invariant, not a procedure).
3. (Anthropic multi-agent, snippet) Subagents "write outputs directly to
   a filesystem instead of funnelling everything through the lead agent"
   -- the artifact pattern; the brief file IS that artifact.

## Internal code inventory
| File | Lines | Role | Status |
| --- | --- | --- | --- |
| .claude/agents/researcher.md | 258-265 "Domain context" | agent prompt | STALE point-in-time facts (see below) |
| .claude/rules/research-gate.md | whole file | how-to mechanics | current; owns floors + PDF strategy + envelope |
| ARCHITECTURE.md | 492-561 | MADR reference record | current; owns the 5-source floor rationale |
| docs/runbooks/per-step-protocol.md | 33 | 3-agent diagram | STALE: labels Researcher "(sonnet)" |
| backend/backtest/experiments/optimizer_best.json | 27-28 | current-best source of truth | sharpe=1.1704633 (rounds to 1.1705, MATCHES); dsr=0.9525811 (NOT 0.9984 in researcher.md) |
| feedback_researcher_write_first.md (auto-memory) | -- | the incident record | verbatim 132K/53-call burn evidence |

## Application to pyfinagent
See "Recommendations" section (populated after external fetches).

## Research Gate Checklist
IN PROGRESS -- 1 of >=5 read in full so far.

```json
{ "gate_passed": false, "note": "in-progress skeleton; 1/5 read in full" }
```
