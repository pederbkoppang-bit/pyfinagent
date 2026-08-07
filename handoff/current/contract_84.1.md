# Contract — Step 84.1: memory-link auditor reconciliation (tooling only)

- **Step id:** 84.1 (P1, phase-84)
- **Tier (named field):** T3 — executor Main (Opus 5, effort max); Q/A via qa-verdict Workflow (opus/max).
- **Date:** 2026-08-07, autonomous drain, cycle 176

## Research-gate summary

`handoff/current/research_brief_84.1.md` — gate_passed: **true** (envelope in the brief; today's corpus state RE-MEASURED because the step's 2026-08-06 figures were already stale: main 75 files/13 problems, qa 37/61, researcher 90/105 with 3 NO POINTER; taxonomy over N=406 links: exact 230 (56.7%) / same-dir-normalized 127 (31.3%) / cross-corpus 37 (9.1%) / unresolvable 12 (3.0%) — the step's "83% same-dir" has decayed to 72.2% of non-exact in ONE day; direction holds, ratio doesn't). Decisive findings:

1. **yaml.safe_load CRASHES on 2 real main-corpus files** (unquoted `description:` values containing ": ") — frontmatter `name:` must be extracted by line-anchored regex; 4 files have no extractable name (index tolerates None).
2. **Collision policy needed (criteria silent)**: report AMBIGUOUS naming all candidates, do NOT fail (the target exists; erroring re-creates the defect being fixed; all four external tools resolve ambiguity by preference, never error). Measured today: 0 colliding keys — a guard, not a behaviour change.
3. **Criterion-3 trap**: a literal "exit 0 iff no unresolvable AND no NO POINTER" reading would DISARM the DANGLING POINTER check — the exact 2026-07-26 incident the tool was built for. Safe resolution: keep DANGLING + MALFORMED failing; build every criterion-3 fixture with zero of both so BOTH readings agree; add a non-criterion regression test that a dangling pointer still exits 1.
4. Bounded link regex + code stripping removes 3 false unresolvables today (two multi-line greedy matches + a C++ `[[nodiscard]]`); the "BROKEN WIKILINK" string is retired (verified: no cron/hook/CI greps it); the `memory files: N   pointers: M` header is kept VERBATIM (an external masterplan step shells it).
5. **Post-fix projection: all three real corpora STILL exit 1** (genuine unresolvables + researcher NO POINTERs) — correct; criterion 6 asserts structure only.

## Immutable success criteria — the 9 in `.claude/masterplan.json` 84.1 `verification.criteria`, C1-C9 as read 2026-08-07 (verbatim in the step record; command: `cd /Users/ford/.openclaw/workspace/pyfinagent && .venv/bin/python -m pytest backend/tests/test_phase_84_1_memory_link_resolution.py -q`)

## Explicit decisions

- **D1 — resolution LADDER** (ordered, deterministic class): exact filename → same-dir normalized (stem OR frontmatter-name alias) → cross-corpus → unresolvable.
- **D2 — normalization**: casefold, '-'→'_' BEFORE stripping AT MOST ONE leading type prefix (feedback_/project_/reference_/user_); docstring states the implied consequence (feedback_x and project_x share a key).
- **D3 — frontmatter name via line-anchored regex**, never bare yaml (finding 1).
- **D4 — AMBIGUOUS class**: report-all-candidates, PASS (finding 2).
- **D5 — criterion-3 reading**: dangling/malformed KEEP failing; dual-reading-agreeing fixtures; extra dangling-still-fails regression test.
- **D6 — bounded regex `\[\[([^\[\]\n|#]{1,120})\]\]` + fenced/inline code stripped** before matching (adopted; fixture expectations account for it).
- **D7 — the `:30` dead `is_dir()` guard fixed** (`root/MEMORY.md`.is_dir() leg was dead; a MEMORY.md-as-directory would have crashed at read_text).
- **D8 — output contract**: header verbatim; stable prefixes NORMALIZED/CROSS-CORPUS/UNRESOLVABLE/AMBIGUOUS LINK + unchanged NO POINTER/DANGLING/MALFORMED lines + a closing census line.
- **D9 — sibling corpora**: production default = the known three roots minus the audited; tests inject `--sibling DIR` (repeatable) so cross-corpus cases build in tmp_path.
- **D10 — criteria 8/9**: the auditor stays read-only (byte+mtime invariance test); zero memory-file edits this step.

## Plan

1. Rewrite `audit_memory.py` resolution per D1-D9 (NO POINTER/DANGLING/MALFORMED logic unchanged).
2. `backend/tests/test_phase_84_1_memory_link_resolution.py`: criterion-5's six fixture cases + exit-code truth table both directions (C3) + real-corpora structural run (C6) + byte/mtime invariance (C8) + dangling-still-fails regression (D5) + ambiguous-class case (D4) + the yaml-crash fixture (a frontmatter that breaks yaml.safe_load must not crash the auditor).
3. Mutations (C7 + extras): (m1) drop the normalization fold → ≥2 named fixture cases red, exact-match case green; (m2) drop the frontmatter-alias lookup → ≥1 named case red; (m3) make unresolvable non-failing → C3 direction-B red; (m4) make normalized failing → C3 direction-A red; (m5) auditor writes a file → C8 red.
4. Lint gate BEFORE Q/A (the twice-today lesson). Live run of the auditor over the three real corpora recorded (structure only) in the results.
5. experiment_results → qa-verdict → transcribe → harness_log → flip (via Edit so the hook fires). Re-derive every fenced measurement after the final edit.

## References

`research_brief_84.1.md` (Obsidian/Alias-Linker/VS-Code resolution-order sources, slugify collision practice, wikilink-rules spec; today's re-measured taxonomy with the stated denominator rule).
