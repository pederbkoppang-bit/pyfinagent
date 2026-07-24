# Contract — phase-75.19: recalibrate the masterplan preflight gate, then triage its true residue

- **Step id:** 75.19 (phase-75, P3, harness_required, executor: opus-tagged → Main-on-Fable GENERATE; Researcher + Q/A gates opus/max via Workflow structured-output)
- **Date:** 2026-07-24
- **Boundary (from step text):** NO immutable verification criteria amended — the deliverable is a trustworthy CHECKER plus annotations, never a criterion rewrite. Coordinate with 75.17: the go_live_drills cluster belongs to that step — do not double-annotate.

## Research-gate summary (gate PASSED — wf_209bbec6-ace)

Envelope: `tier=moderate, external_sources_read_in_full=7, snippet_only_sources=11, urls_collected=45, recency_scan_performed=true, internal_files_inspected=8, gate_passed=true`. Brief: `handoff/current/research_brief_75.19.md`.

Load-bearing findings (each re-measured by Main, not trusted from the 2026-07-20 step text):

1. **Live baseline (2026-07-24):** `preflight_verify_masterplan: scanned 863 steps, 151 broken, 8 unparseable` with **222 BROKEN lines** (217 path + 5 import) spanning exactly **151 distinct step ids**. The step text's 819/141/212/28 figures have drifted; all numbers below are re-derived.
2. **Status breakdown of the 151:** 82 done / 40 pending / 10 deferred / 5 dropped / 1 superseded / 13 in non-`steps` containers. 12 of the 82 done ids already carry `superseded_record` (75.17's go_live_drills cluster + 4.14.24/4.14.26) — the preflight ignores annotations.
3. **Container census (measured):** verification-bearing steps live in `phases[].steps[]` (859), `phases[].subphases[]` (13: 38.10–38.13 done/pending mix, 46.0–46.8 pending), `archived_legacy_steps[]` (3, duplicates), `archived_dropped_steps[]` (1: 5.5). The preflight's recursive `_walk_steps` drags in ALL of these; the 75.17 classifier's `flat_steps` sees ONLY `phases[].steps[]`.
4. **True residue after recalibration = 0.** Full-fidelity walk + status filter + `superseded_record` filter + `fp_reason` adjudication + status-gated import leg → **0 absent-path, 0 import** genuine defects. All 5 import-broken lines are pending phase-5 steps. All 14 genuinely-unrunnable done steps were already annotated (3 by 75.2.1, 11 by 75.17).
5. **The 8 shlex-unparseable are all done steps** with nested-quote `python -c`/`bash -c` commands (23.2.7, 23.3.2, 70.2, 70.3, 70.5, 71.2, 71.4, 71.5); the classifier's regex extractors scan all 8 cleanly (0 genuine). Checker limitation, not a defect — own bucket, never "broken"; commands are immutable and stay untouched.
6. **List-shaped verifications:** `_extract_command` silently returns None for the 13 list-shaped verifications; `verif_commands` handles all four shapes.
7. **External canon:** Google SWE book ch.20 + Sadowski CACM 2018 — a check survives only under ~10% effective false positives (FindBugs died of this); current preflight is ~100% effective-FP, which is exactly why nobody acts on it. 2026 harness literature (arXiv 2607.07405, 2606.06324, 2607.00871) treats noisy validators as first-class harness flaws.
8. **Anti-vacuous-guard (qa.md §4c, live in roster — probe passed this session):** "residue==0 on live" alone is tautology-risk; tests need a POSITIVE synthetic-defect fixture plus ≥1 FIXTURE mutation (criterion 6).

## Hypothesis

Swapping the preflight's recursive walk + ad-hoc path heuristic for (a) a full-fidelity container-explicit walk and (b) the imported 75.17 adjudication core (`verif_commands`/`_extract_candidates`/`fp_reason`), adding a status-gated import leg, regex-fallback scanning for shlex-unparseable commands, and an internally-consistent summary, drops the reported count from 151 steps/222 lines (~100% effective-FP) to the true genuine residue of **0**, with zero unresolvable ids — turning an ignored cry-wolf gate into a trustworthy one, and confirming with reproducible evidence that **no new `superseded_record` is owed** (criterion-(d) triage resolves to "nothing to annotate").

Design deviation from researcher R2, with evidence: R2 proposed restricting to `flat_steps` (canonical-only). That would silently skip the 13 live `subphases[]` steps (38.10 is `done`) — a recall loss. Instead the walk is full-fidelity with explicit container semantics: INCLUDE `steps[]` + `subphases[]`; EXCLUDE `archived_legacy_steps[]`/`archived_dropped_steps[]` (archive duplicates) and everything under `superseded_record` (annotation) — excluded containers are counted and reported, so the change is auditable. Zero '?' ids holds by construction (ids come from the nodes themselves).

## Plan

1. Rewrite `scripts/meta/preflight_verify_masterplan.py` around the imported 75.17 core (no copied adjudicators — a second copy drifts from the census):
   - full-fidelity walk with container-explicit include/exclude + bucket counts;
   - report a step broken ONLY if `status=="done"` and no `superseded_record`; non-done and annotated steps → excluded buckets;
   - path leg = `verif_commands` → `_extract_candidates` → `fp_reason` (imports via file-location load since `scripts/qa` is not a package);
   - import leg kept (classifier is path-only) but status-gated identically;
   - shlex `ValueError` → regex-extractor fallback still scans; done-step unparseables → `unverifiable-by-shlex(scanned-via-regex)` bucket;
   - one auditable summary line: total steps by container, scanned done/unannotated, GENUINE broken lines AND distinct step count, excluded-bucket counts; a self-check asserts emitted rows == summary counts; exit code keys off GENUINE only.
2. Write `backend/tests/test_phase_75_19_preflight_calibration.py`: synthetic fixture masterplans pinning (a) one example per status class on the same absent path; (b) each transient/non-source class by construction (handoff/ output, gitignored path, URL fragment `/openapi.json`, frontend-relative `lib/icons.ts`, truncated `/Library/LaunchAgents/com.py`); (c) POSITIVE genuine-defect fixture (done step, absent non-transient path → MUST report); (d) list-shaped verification; (e) `subphases` inclusion + archive exclusion; (f) `superseded_record` exclusion; (g) summary/rows consistency; (h) shlex-fallback path.
3. Run the recalibrated gate on the live masterplan → expect CLEAN/exit 0; triage = nothing to annotate (criterion 5 satisfied with sweep evidence; go_live_drills untouched).
4. Mutation matrix (recorded verbatim in live_check_75.19.md): production mutations (drop status filter; drop superseded_record skip; restore blind recursive walk; drop transient exclusion; break summary consistency) AND ≥1 fixture mutation (flip the positive fixture's absent path to an existing one → its test must fail). Every mutation must flip a test.
5. live_check_75.19.md: before/after preflight output verbatim, verification-command output (exit 0), `git diff --stat`, byte-identity statement for annotations (none owed — evidence attached), mutation evidence.
6. Q/A via qa-verdict Workflow (opus/max), verdict transcribed verbatim; harness_log append; status flip last.

## Immutable success criteria (copied VERBATIM from .claude/masterplan.json step 75.19)

> "command": ".venv/bin/python -m pytest backend/tests/test_phase_75_19_preflight_calibration.py -q"

1. "The recalibrated preflight does NOT report a step as broken solely because an artifact of a pending/deferred/dropped/superseded step is absent; a fixture pins one example of each status so a regression to status-blindness fails"
2. "Transient and non-source references (handoff/ outputs, gitignored paths, URL fragments like /openapi.json, frontend-relative paths like lib/icons.ts, truncated paths like /Library/LaunchAgents/com.py) are excluded by construction, each pinned by a fixture rather than by an allowlist of observed strings"
3. "Every step id in the report resolves to a real masterplan step -- zero '?' ids -- and the summary line's counts are internally consistent with the emitted rows (the current run emits 212 BROKEN lines while summarizing 141 broken steps and 8 unparseable, and a test asserts rows and summary agree)"
4. "The report distinguishes GENUINELY-UNRUNNABLE done steps from calibration noise, and the residue count is stated with the sweep output backing it -- no count asserted anywhere without reproducible evidence"
5. "Any genuinely-unrunnable done step surviving triage gains a superseded_record sibling per the 75.2.1 shape, with verification.command and verification.success_criteria BYTE-IDENTICAL to their pre-step values (asserted against git show); the go_live_drills cluster owned by 75.17 is excluded and no step carries two superseded_record entries"
6. "Each new behavioral guard is mutation-tested with the evidence recorded verbatim in live_check_75.19.md, INCLUDING at least one mutation of a test fixture (per the phase-75.2.1 lesson that mutating production alone missed a fixture which could not represent the failure)"

## References

- `handoff/current/research_brief_75.19.md` (7 sources read in full; Google SWE ch.20; Sadowski et al. CACM 2018; Vera-Perez pseudo-tested methods; arXiv 2607.07405 / 2606.06324 / 2607.00871; ReVeal arXiv 2506.11442 via qa.md §4c)
- `scripts/qa/sweep_absent_verification_paths.py` (75.17 importable core — reuse mandate)
- `scripts/meta/preflight_verify_masterplan.py` (recalibration target)
- masterplan `superseded_record` exemplar: step 4.14.4 (75.2.1 shape)
- CLAUDE.md harness protocol; `.claude/rules/research-gate.md`; qa.md §4b/§4c
