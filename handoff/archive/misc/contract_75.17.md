# Contract -- Step 75.17: absent-path verification family (triage + repair)

- **Step id**: 75.17 (phase-75, Audit75 S17) -- sonnet-tier -> **Sonnet executor GENERATE** (8th delegated; Main reviews + re-measures).
- **Date**: 2026-07-24
- **BOUNDARY**: NO immutable verification criteria amended anywhere -- `verification.command` + `success_criteria` of every touched step stay BYTE-IDENTICAL (asserted against the pre-75.17 commit); repair = `superseded_record` sibling annotations (4.17.9 house shape at masterplan:4870, NOT the 68.5 shape) + the committed sweep tool + guard tests. Repo-only; no network (version pins, if any, from requirements.lock per the 75.16 durable rule).

## Research-gate summary (gate PASSED, audit-class: coverage.dry=true, 4 rounds / 2 dry)

Workflow `wf_8e13812e-3ce` (opus/max, moderate, audit-class). Envelope: `6 read-in-full (ISACA proof-vs-policy, ADR supersession, polymorphic-field handling, doc-rot...), snippet=10, urls=16, recency=true, internal=12, gate_passed=true, coverage={rounds:4, dry_rounds:2, dry:true}`. Brief + FULL census table: `research_brief_75.17.md`; re-runnable reference implementation: `handoff/current/census_75_17_reference.py`.

**Corrections adopted (binding):**
1. Shape census re-measured at HEAD: **720 dict / 126 str / 13 list / 24 None** (883 steps; the node's 674 was 75.2.1-era). The sweep must handle all four (naive `.get` crashes on the 13 list-shaped).
2. **THE DEFINITIVE GENUINE SET = 10 steps**: 9 go_live_drills (4.17.2-8, 4.17.11, 4.17.12 -- class (i): the plan names NEVER matched disk; the real work exists as `smoke_test_4_17_N.py`, git --diff-filter=A empty for the plan names) + **4.14.26** (class (ii): two of its 10 grepped skill files deleted by phase-26.4 commit f7e24d0a -- a NEW member neither prior sweep named). EXCLUDED as already-annotated: 4.17.9, 4.14.24, AND 4.14.4 (the node names only two; the repo confirms three).
3. Both prior counts were over-counts, reconciled by TWO missing filters: (a) **negative-assertion detection** (`! test -f`, `test ! -f`, `test -f X ||`, python `assert not os.path.exists`) -- 4.14.19/16.50 are absence-asserting and RUNNABLE; (b) **current-disk check** -- 8.5.4/10.2 name runtime TSVs that exist NOW. Remaining candidates were extraction artifacts (frontend-relative, URL routes, truncated plist, globs, shell vars).
4. Hard cases the sweep must handle: 7.12 glob-prefix re-resolution (`alt_data_ic_*.tsv` matches an existing file); 75.16's python assert-not-exists + tolerated-missing bare basename (negation matched against the FULL path).
5. Exactly-one-superseded_record-per-step repo-wide invariant (current holders: 4.14.4/4.14.24/4.17.9/68.5).

## Plan (research recommendations adopted verbatim)

1. **Committed sweep** `scripts/qa/sweep_absent_verification_paths.py` (sweep_ascii_logger.py idiom; the classifier as an IMPORTABLE module -- step 75.19 is chartered to reuse it for the preflight gate; do NOT add a nightly workflow here). All resolution/skip/negation rules from the brief; status=done scope; git-classification never-existed vs retired.
2. **Annotations**: superseded_record sibling on each of the 10 (class-i template: already_broken_before_retirement true, retired_by_commit null, on_disk_equivalent smoke_test_4_17_N.py, scope_disclosure; class-ii template for 4.14.26: retired_by_commit f7e24d0a, retired_in_step 26.4). The three annotated steps untouched.
3. **Guard tests** `backend/tests/test_phase_75_17_verification_paths.py` (must match the node's immutable verification command): byte-identity of command+success_criteria for all 10 vs the pre-75.17 commit (git show); exactly-one-superseded_record repo-wide; the sweep returns EXACTLY the empty genuine set post-annotation (or the 10 pre-annotation against the git-show tree); all-4-shapes fixture handling.
4. **Mutation matrix M1-M10 per the brief** including the MANDATORY M7 fixture mutation (dict-only fixture must fail the all-4-shapes test) and M8 byte-identity break + M9 double-annotation + the resolver non-flag proofs (M3-M6, M10).
5. Coordinate-with-75.19 note in the results (classifier importability documented); go_live_drills annotations belong HERE, never double-annotated there.

**Verification**: the node's immutable command verbatim exit 0; full suite vs the 8/9-red environment-dependent baseline (CI-equivalent selection stays green); ruff derived scope; masterplan diff = ONLY the 10 superseded_record inserts (byte-identity proof for everything else).

## NOT in scope
Amending any criterion; the preflight-gate recalibration (75.19); nightly wiring; the three already-annotated steps.

## References
research_brief_75.17.md (+ census table + reference impl); masterplan:4870 (4.17.9 shape); audit_phase75; qa.md sec4b; the 75.16 read-only-lookup durable rule.
