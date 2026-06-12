---
name: committed-criterion-gitignore-check
description: Any criterion saying "file X committed" requires git check-ignore + git ls-files verification; .gitignore:24 (*.log) silently defeated 3 prior cycles' evidence logs
metadata:
  type: project
---

When a masterplan success criterion says an artifact must be "committed", ALWAYS run
`git check-ignore -v <path>` and `git ls-files --error-unmatch <path>` before issuing
PASS. The auto-commit hook stages via `git add -A`, which SILENTLY skips gitignored
paths -- no error, no warning.

**Why:** phase-17.4 (evaluated 2026-06-12): criterion "handoff/current/
alpaca-researcher-dryrun.log committed" was defeated by the global `*.log` rule at
`.gitignore:24`. Three prior attempts' logs sat untracked on disk in
handoff/archive/misc/ while the step's notes field claimed "Log committed" -- a
VERIFICATION_DEFECT that only `git ls-files` exposed. The lone tracked precedent
(`alpaca-researcher-dryrun.log.test`) only got in because its suffix evades the
pattern. First Q/A spawn returned CONDITIONAL on exactly this; fix = `git add -f`
at the literal criterion path before the flip. Fix CONFIRMED LANDED in commit
6684c9c7 (delta re-eval PASS, 2026-06-12). Nuance: once the path is tracked,
`git check-ignore` exits 1 (tracked files are exempt from ignore rules) -- do not
misread exit-1 as "was never ignored"; and ordinary `git add -A` DOES stage future
modifications to the now-tracked path. Moved/archived copies at new `*.log` paths
still need their own `git add -f`.

**How to apply:** (1) for every "committed"/"staged" criterion, run both git commands
and quote the .gitignore rule if it matches; (2) ordering trap: if the
archive-handoff hook MOVES a tracked file to an ignored destination path at the
status flip, `git add -A` stages the deletion and drops the new path -- the move
becomes a delete; (3) flag any NEW criterion naming a `*.log` evidence path as an
authoring hazard (prefer .md/.txt). Also note the phase-17 precedent set here:
per-step artifact files (contract_<sid>.md, research_brief_<sid>.md,
evaluator_critique_<sid>.md) are the accepted pattern when a stale step interleaves
with in-flight work, because the step's own verification command can overwrite the
rolling slots (run_harness.py:355 writes contract.md; researcher branch overwrites
research_brief.md).
