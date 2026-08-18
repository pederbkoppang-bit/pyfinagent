# Experiment results -- step 86.28 (CURRENT STATE)

> **Full six-cycle history** (what was built each cycle, every follow-up):
> `handoff/current/experiment_results_86.28_history.md` (587 lines,
> byte-identical copy before compaction, md5 9f0266cb42797998bd24c62820236a77).
> Nothing deleted; moved out of the mandatory read path after two Q/A drops.

## What ships

Three defects in `.claude/workflows/research-gate.js`, plus the checker and
docs that cover them.

**1. Silent tier downgrade -> ABSENT vs UNSUPPORTED.** An unsupported tier
was replaced with `moderate` and the substitution reached only the agent
PROMPT -- payload, not response, so no caller could detect it. Now:

- `tierAbsent` / `tierSupported` / `tierUnsupported` are distinct;
- an UNSUPPORTED tier **refuses to spawn**, before a max-effort researcher
  session is spent, mirroring the file's own args-boundary doctrine;
- `tier_requested` / `tier_applied` / `tier_supported` are in the RETURN
  VALUE (the RFC 7240 `Preference-Applied` shape from the research);
- an ABSENT tier still defaults to `moderate` with no violation, unchanged.

`'deep'` is deliberately NOT added to `VALID_TIERS` -- see live_check S7.

**2. Uncorroborated self-reports.** `recency_scan_performed` and
`urls_collected` answered only to themselves inside a gate whose thesis is
recompute-never-trust. Both are now checked against the brief by the
EXISTING stage-2 verifier (no new agent, no new spawn). Named as STRUCTURAL
because that is what they are: `recency_section_present` says a section
exists, not that the scan was substantive.

**3. Doc drift.** `researcher.md` and `CLAUDE.md` claimed
`agentType:'general-purpose'`; the code pins `'researcher'` and the checker
asserts it. Both corrected, CLAUDE.md's self-contradiction retracted in
place, and all citations switched to grep-for-symbol -- line numbers in
these files went stale three times inside this one step.

## File list

| File | Change |
|---|---|
| `.claude/workflows/research-gate.js` | tier classification, refuse-to-spawn, tier violation, 2 corroboration checks, stage-2 schema + prompt, 3 return fields. **Frozen since cycle 3**; comment-only edits before that. |
| `scripts/qa/verify_research_gate_workflow.mjs` | compliant fixtures, stage-2 simulation, `opts` passthrough, `[6b]`/`[6c]`/`[6d]` sections, `[7b]` driver-mutant matrix, fixture-fidelity + anchor-uniqueness assertions |
| `CLAUDE.md`, `.claude/agents/researcher.md` | `agentType` corrected; symbols not line numbers |

## Verification

Baseline `ALL GREEN: 40 passed, 0 failed` (independently re-derived at the
base commit by two Q/As). Current: **92 passed, 0 failed**. Ladder
40 -> 61 -> 64 -> 73 -> 78 -> 92, zero checks removed or renamed, verified
by symmetric difference of check NAMES rather than totals.

Full captured output, mutation matrices and live-spawn returns are in
`live_check_86.28.md`.

## Not done, deliberately

| Item | Why |
|---|---|
| Add `'deep'` to `VALID_TIERS` | researcher.md's "Multi-subagent fork option" makes deep's fourth listed element a CONDITIONAL multi-subagent producer fork. Enabling it ships fan-out onto an N=1 artifact rail and pre-empts an open operator decision. Disclosed, not resolved. |
| Corroborate `coverage.dry` | "Dry" is K executed search rounds with no new findings -- a property of executed discovery, not of a file. Any file-derived proxy is false assurance (EBTE). |
| Wire `opts.floors` | Zero callers; its only consumer would be tier-aware floors, which depend on the deep decision. |
| Change the envelope `tier` enum | The agent reports the tier it actually operated at, always a supported value, so the enum is not lying. The requested-vs-applied distinction is the SCRIPT's to report, and it now does. |

## What each cycle cost, and what it found

Six evaluate cycles, two of which dropped without a verdict. Every finding
from cycle 3 onward was about the CHECKER or the EVIDENCE, not shipped
behaviour -- including one FAIL for a transcript I typed instead of
captured. That FAIL was correct and is recorded in full in the history file
rather than summarised away.
