# Sprint Contract -- phase-2.10 + phase-4.14.20 Audit Cycle

**Written:** 2026-04-19 PRE-commit.
**Steps audited:** `phase-2.10` (status=superseded) + `phase-4.14.20` (status=blocked).
**Cycle purpose:** formalize both non-actionable resolutions with harness artifacts per the user's explicit request ("full harness with mas system on this resolved non-actionable").

## Research-gate summary

Researcher spawned today. Envelope: `{tier: simple, external_sources_read_in_full: 7, snippet_only_sources: 5, urls_collected: 12, recency_scan_performed: true, internal_files_inspected: 8, gate_passed: true}`. Brief at `handoff/current/phase-audit-2.10-4.14.20-research-brief.md`.

Key findings (quoted):
- **phase-2.10**: "Karpathy autoresearch concept was materialized in `backend/agents/skill_optimizer.py`, which explicitly self-describes as 'Mirrors Karpathy's autoresearch pattern' (lines 4, 129, 270, 453)." Supersession is substantively clean. Bureaucratic gap: `handoff/phase-2.10-supersede.md` does not exist; phase-8.5.0 (pending) has immutable `test -f handoff/phase-2.10-supersede.md` — so the missing file is a forward blocker.
- **phase-4.14.20**: "`qa.md` line 3 has 'MUST BE USED', 'Use proactively', and 'use immediately after'. `researcher.md` line 3 has 'MUST BE USED' and 'Use proactively'. Both active successor agents already carry all required phrases." Literal immutable cmd cannot be satisfied (targets two deleted files: `qa-evaluator.md` + `harness-verifier.md`). Semantic intent is already satisfied by the phase-4.15.0 agent merge.

## Hypothesis

Both steps are correctly non-actionable AS IMPLEMENTATION items, but are missing formal audit artifacts. One targeted file-creation (phase-2.10) + one status change + audit note (phase-4.14.20) closes the forward-blocker chain without violating CLAUDE.md immutability rules.

## Success criteria

**phase-2.10 deliverables:**
1. `handoff/phase-2.10-supersede.md` exists with:
   - Header: "phase-2.10 Karpathy Autoresearch Integration — supersede record"
   - Supersession decision + date
   - Absorber: `backend/agents/skill_optimizer.py` (lines 4, 129, 270, 453 confirm self-described Karpathy-autoresearch mirror)
   - Cross-reference to phase-8.5.0 which gates on this file's existence
   - Signature of who authored the supersession (this audit cycle)
2. Masterplan `phase-2.10` status remains `superseded` (no change; this is formalization, not re-opening).

**phase-4.14.20 deliverables:**
3. Grep evidence captured in experiment_results showing current state of `.claude/agents/qa.md` and `.claude/agents/researcher.md` — phrases present.
4. Masterplan `phase-4.14.20` status changed from `blocked` to `superseded` with `superseded_by: phase-4.15.0` in a new field (status change is allowed; editing immutable verification command is NOT allowed — we leave `verification` untouched).
5. A one-paragraph note in the audit-record captured as `handoff/phase-4.14.20-supersede.md` documenting: the original contract's intent (apply trigger phrasing to 3 agent files), the phase-4.15.0 merger that collapsed 2 of those 3 files into `qa.md`, the current state showing the phrasing lives correctly in the successor agents, the decision to retire the step rather than resurrect deleted files.

**Immutable verification commands retained (unchanged):**
- phase-2.10 has no immutable cmd in masterplan (research confirmed; verified below).
- phase-4.14.20 immutable cmd untouched. Because the cmd can no longer be satisfied, we document this fact — we do NOT edit the cmd. Status change to `superseded` acknowledges the immutable cmd is stranded by the phase-4.15.0 refactor.

**Read-only verification for the audit:**
- `grep -c 'use proactively\|MUST BE USED\|use immediately after' .claude/agents/qa.md` -> >= 2
- `grep -c 'use proactively\|MUST BE USED\|use immediately after' .claude/agents/researcher.md` -> >= 1 (any of the phrases satisfies; researcher has 2 per research)
- `test -f handoff/phase-2.10-supersede.md` -> exit 0
- `test -f handoff/phase-4.14.20-supersede.md` -> exit 0
- `grep -c superseded .claude/masterplan.json` -> increased by 1 (phase-4.14.20 added; phase-2.10 was already superseded)

## Plan steps

1. Create `handoff/phase-2.10-supersede.md`.
2. Create `handoff/phase-4.14.20-supersede.md`.
3. Update `.claude/masterplan.json` phase-4.14.20: status `blocked` -> `superseded`; add `superseded_by: phase-4.15.0` field. Preserve `verification` block verbatim.
4. Run read-only verification commands, capture into experiment_results.
5. This audit cycle does NOT touch `.claude/agents/*.md` (CLAUDE.md separation-of-duties rule on agent edits; the research already established no edits are needed — phrasing is already present).

## Non-goals

- Not re-splitting `qa.md` back into `qa-evaluator.md` + `harness-verifier.md` (CLAUDE.md forbids).
- Not amending any immutable verification cmd.
- Not editing `.claude/agents/*.md` files (nothing to add; separation-of-duties rule).
- Not re-opening phase-2.10 (substantively resolved by skill_optimizer.py).

## References

- `handoff/current/phase-audit-2.10-4.14.20-research-brief.md`
- `backend/agents/skill_optimizer.py:4,129,270,453` (Karpathy-autoresearch-mirror self-ref)
- `.claude/agents/qa.md:3` + `.claude/agents/researcher.md:3` (trigger-phrase evidence)
- `handoff/archive/phase-4.15.0/` (MAS restructure that merged agents)
- CLAUDE.md: "Never edit verification criteria in masterplan.json — they are immutable" + "Separation of duties on agent edits"

## Researcher agent id

`a382f98251f8a31f5`
