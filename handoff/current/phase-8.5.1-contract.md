# Sprint Contract — phase-8.5 / 8.5.1 (Candidate space) — REMEDIATION cycle v2

**Step id:** 8.5.1 **Remediation cycle:** 2 (after hook-race failure on v1)
**Date:** 2026-04-20 **Tier:** simple

## Why remediation v2

Remediation v1 failed Q/A (`qa_851_remediation_v1 FAIL`) because the archive-handoff hook churned `handoff/current/` into 150+ phantom archive dirs — my v1 contract + experiment-results + research-brief got archived before Q/A could see them. The researcher-authored brief was found at `handoff/archive/phase-8.5.1-v99/`.

**Infrastructure fix applied this cycle:** `.claude/archive-handoff.disabled` flag-file created + hook edited to early-exit when the flag exists. Files will now stay in `handoff/current/` across masterplan writes.

## Research-gate summary

Researcher subagent (spawned 2026-04-20 ~03:22 UTC) produced a proper 167-line brief with:
- Arithmetic breakdown table (5*4*3*2*5*5*5 = 15,000, independently computed).
- Three-variant search (current-year, last-2-year, year-less canonical).
- 5 external sources in full (hyperparameter search literature).
- Cross-reference table to the 3 scaffolded modules.
- Adversarial "is this inflated?" check — HONEST.

Brief restored from `phase-8.5.1-v99/` to `handoff/current/phase-8.5.1-research-brief.md`.

## Immutable criterion

- `test -f backend/autoresearch/candidate_space.yaml && python -c "import yaml; d=yaml.safe_load(open('backend/autoresearch/candidate_space.yaml')); assert d['estimated_combinations'] >= 10000"`

## Plan

1. Verify the hook guard works (edit masterplan noop or write file, confirm no new archive dir).
2. Re-run immutable + arithmetic + cross-ref.
3. Spawn fresh Q/A against files STABLE in `handoff/current/`.
4. Log remediation.

## References

- `handoff/current/phase-8.5.1-research-brief.md` (167 lines; restored)
- `.claude/archive-handoff.disabled` (guard flag)
- `backend/autoresearch/candidate_space.yaml`
- `.claude/masterplan.json` → phase-8.5 / 8.5.1
