---
step: phase-23.8.3
title: Correct misleading "DEPRECATED" headers (closes audit R-6 by header correction, not deletion)
cycle_date: 2026-05-11
harness_required: true
verification: 'source .venv/bin/activate && python3 tests/verify_phase_23_8_3.py'
research_brief: (researcher subagent, 2026-05-11; gate_passed=true; 6 sources read in full, 16 URLs, recency scan)
audit_basis: docs/audits/dev-mas-2026-05-11/04-remediation.md (R-6 — but supersedes the "delete" framing)
---

# Contract — phase-23.8.3

**Step**: phase-23.8.3 — Correct misleading "DEPRECATED" file headers
that incorrectly mark two live-imported modules as stubs.

**Date**: 2026-05-11.

**Status target**: pending → done.

**Hypothesis**:

The 2026-05-11 dev-MAS audit (R-6 / C-A7) recommended deleting
`backend/agents/meta_coordinator.py` and `backend/autonomous_harness.py`
based on their self-declared "DEPRECATED — Phase 4 stub" headers.
Cycle 37's research gate revealed both files are LIVE with active
importers. The correct closing action for R-6 is therefore to
**update the headers** so the documentation matches reality —
deletion is impossible without significant refactor work that
provides no offsetting benefit.

The research gate for this cycle confirmed:

1. LLMs treat file headers as authoritative ground truth — a 2025
   arXiv study (`2504.04372v2`) found fault-localization accuracy
   dropped to 24.55% under misleading comments.
2. Stale headers actively degrade autonomous-agent decision-making
   (`propelcode.ai` 2026).
3. Google's SWE Book (Abseil ch15) requires deprecation notices to
   be **actionable** — a `DEPRECATED` header on a live-imported file
   with no migration path is by definition inactionable noise.
4. The HARNESS-DOC stress-test doctrine directly applies: "every
   component in a harness encodes an assumption ... worth stress
   testing." The DEPRECATED header encodes the assumption "this file
   is unused" — falsified by a one-line grep.

## Research-gate summary

Researcher subagent ran 2026-05-11 (tier: simple). JSON envelope:
`{"external_sources_read_in_full": 6, "snippet_only_sources": 10,
"urls_collected": 16, "recency_scan_performed": true,
"internal_files_inspected": 7, "gate_passed": true}`.

Key external citations (≥5 sources read in full via WebFetch):

1. HARNESS-DOC
   (`https://www.anthropic.com/engineering/harness-design-long-running-apps`)
   — stress-test doctrine quote.
2. arXiv 2504.04372v2 (2025) — "LLMs often try to extract a code's
   semantic information from the comments in addition to the code
   itself." Empirical evidence comment drift degrades model
   performance.
3. propelcode.ai (2026) — "stale docstrings don't just confuse
   humans — they actively degrade agent decision-making in
   follow-up tasks."
4. abseil.io/resources/swe-book/html/ch15.html — Google SWE Book on
   deprecation: notices must be actionable, must have a migration
   path. Inactionable notices are noise.
5. pensero.ai (2026) — Knight Capital dead-code-reactivation
   example ($440M loss in 45 min); 20-30% onboarding cost premium.
6. EFFECTIVE-DOC
   (`https://www.anthropic.com/engineering/building-effective-agents`)
   — "Add complexity only when it demonstrably improves outcomes."
   Header correction is strictly less complex than deletion +
   import-chain refactor.

Recency scan: no 2024-2026 source defends preserving misleading
deprecation notices for "historical fidelity".

### New finding from research gate

The researcher discovered **two additional files** that
cross-reference the misleading "DEPRECATED" label as a contrast
disambiguator:

- `backend/meta_evolution/__init__.py:7` —
  "Distinct from the DEPRECATED `backend/agents/meta_coordinator.py`
  Phase-4 stub."
- `backend/meta_evolution/alpha_velocity.py:18` — same phrasing.

These are not imports; they're documentation labels. They need to
be reworded to match the corrected header semantics so the cross-
reference remains accurate.

## Plan steps

### G-1 — Correct `backend/agents/meta_coordinator.py` header

Replace the opening "DEPRECATED — Phase 4 stub. Not part of the
active MAS architecture." with an accurate description that:
- Acknowledges the file is ACTIVE
- Names the live importers (`autonomous_loop.py`,
  `skill_optimizer.py`)
- Preserves the existing "should not be extended" guidance (the
  guidance is sound; only the "DEPRECATED — Phase 4 stub" label is
  wrong)
- Cites the audit closure: phase-23.8.3, recommendation R-6

### G-2 — Correct `backend/autonomous_harness.py` header

Replace the opening "DEPRECATED — Phase 4 stub. Not part of the
active MAS architecture." with an accurate description that:
- Acknowledges the file is ACTIVE (used by
  `scripts/risk/phase4_9_redteam.py` for FINRA 15-09 negative
  tests)
- Names the live exports (`promote_strategy`, `PromotionBlocked`,
  `_BLOCKLIST_PATH`)
- Preserves the "should not be extended" guidance
- Cites the audit closure: phase-23.8.3

### G-3 — Update contrast labels in `backend/meta_evolution/`

- `backend/meta_evolution/__init__.py:7` — change "DEPRECATED
  `backend/agents/meta_coordinator.py` Phase-4 stub" → "the
  legacy `backend/agents/meta_coordinator.py` module (still
  active for autonomous_loop + skill_optimizer; do not extend —
  see phase-23.8.3 closure of audit R-6)".
- `backend/meta_evolution/alpha_velocity.py:18` — same change.

### G-4 — Document audit R-6 closure

Append a short closure note to
`docs/audits/dev-mas-2026-05-11/04-remediation.md` (NOT a
rewrite of the original R-6 finding — that is a historical
record). The note explains:
- R-6 was based on file headers that turned out to be misleading.
- Cycle 37 research gate caught the issue (live importers).
- Cycle 40 (this cycle) closed R-6 by correcting the headers.
- Future cycles should NOT delete these files until / unless a
  proper refactor cycle removes their importers.

### G-5 — Verifier `tests/verify_phase_23_8_3.py`

Immutable claims:

1. `meta_coordinator.py` header no longer contains
   "DEPRECATED — Phase 4 stub".
2. `meta_coordinator.py` header contains "ACTIVE" + explicit
   importer references.
3. `autonomous_harness.py` header no longer contains
   "DEPRECATED — Phase 4 stub".
4. `autonomous_harness.py` header contains "ACTIVE" + explicit
   caller references.
5. `backend/meta_evolution/__init__.py` no longer contains
   "DEPRECATED `backend/agents/meta_coordinator.py`".
6. `backend/meta_evolution/alpha_velocity.py` no longer
   contains "DEPRECATED `backend/agents/meta_coordinator.py`".
7. `04-remediation.md` contains an R-6 closure note citing
   phase-23.8.3.
8. **No regressions**: both files still import successfully (the
   real failure mode if I accidentally break the file while
   editing the header).
9. **No regressions**: `autonomous_loop.py` and
   `skill_optimizer.py` and `phase4_9_redteam.py` (the live
   importers) all still import.
10. `handoff/harness_log.md` Cycle 40 contains a verbatim R-6
    closure note referencing this step.

## Files expected to change

| File | Type | Change |
|---|---|---|
| `backend/agents/meta_coordinator.py` | edit | lines 1-6 header rewritten |
| `backend/autonomous_harness.py` | edit | lines 1-11 header rewritten |
| `backend/meta_evolution/__init__.py` | edit | line 7 contrast label updated |
| `backend/meta_evolution/alpha_velocity.py` | edit | line 18 contrast label updated |
| `docs/audits/dev-mas-2026-05-11/04-remediation.md` | edit | append R-6 closure note |
| `tests/verify_phase_23_8_3.py` | NEW | 10-claim verifier |
| `handoff/current/contract.md` | NEW (this file) | contract |
| `handoff/current/experiment_results.md` | NEW (later) | by GENERATE |
| `handoff/current/evaluator_critique.md` | NEW (later) | by Q/A |
| `handoff/harness_log.md` | append | Cycle 40 with R-6 closure framing |
| `.claude/masterplan.json` | edit | new step 23.8.3 pending → done |

## Immutable success criteria

(Same as G-5 verifier claims above.)

## Rollback note

Single-commit revert. The only logic change is text in comment
blocks. No code execution path changes. No tests need updating.
The only risk is breaking the docstring at the top of a Python
file, which would surface as an `import` failure — caught by
claims 8 and 9.

## Out of scope (explicit)

- **R-5** (qa.md fail-mode change) — separate session.
- **R-6 actual delete** — would require refactoring
  `autonomous_loop.py:19,50,462-488,896-897`,
  `skill_optimizer.py:825`, and `phase4_9_redteam.py:58` to remove
  their dependencies. This cycle does NOT do that. The audit's
  R-6 framing of "just delete" is now corrected: future delete
  needs a refactor cycle first.
- **qa.md follow-on** — deferred from cycle 38 per
  separation-of-duties.
- **Stop hook H-3 cleanup** — separate cycle if at all.
- **Auto-commit hook auto-fire diagnostic** — observed
  empirically 3 cycles in a row; separate diagnostic cycle.

## References

- `docs/audits/dev-mas-2026-05-11/04-remediation.md` R-6 — the
  proposal that this cycle closes.
- `docs/audits/dev-mas-2026-05-11/02-per-agent.md` C-A7 — the
  related finding.
- `handoff/archive/phase-23.8.0/contract.md` — cycle 37's
  deferral of R-6 with the live-importer evidence.
- arXiv 2504.04372v2 (2025) — empirical grounding for
  "LLMs trust misleading comments" hazard.
- abseil.io SWE Book ch15 — "deprecation must be actionable".
- Researcher JSON envelope: `gate_passed: true`.
