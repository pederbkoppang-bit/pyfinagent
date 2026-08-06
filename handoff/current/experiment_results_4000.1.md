# Experiment results -- 4000.1: research gate + measured baselines

Date: 2026-08-06. Author: Main. Contract: contract_4000.1.md (written before
these artifacts; research brief written before the contract).

## What was built / changed

| Artifact | Path |
|---|---|
| Research brief (researcher, Workflow rail, write-first) | handoff/current/research_brief_4000.1.md |
| Contract | handoff/current/contract_4000.1.md |
| Baseline artifact, sections (a)-(f) + defects register | handoff/current/cc_rail_baseline_4000_1.md |
| Check script (criterion 1) | scripts/qa/verify_phase_4000_1_baseline.sh |
| This file | handoff/current/experiment_results_4000.1.md |

No production code touched. This step's produced-file set, derived from git
status + mtimes (criterion 9 + Q/A cycle-1 disclosure fix): the four suffixed
handoff/current/ files, scripts/qa/verify_phase_4000_1_baseline.sh, AND one
researcher auto-memory file written by this step's researcher spawn --
.claude/agent-memory/researcher/project_cc_rail_e2e_4000_1.md. Criterion 9's
binding half holds: no backend/, frontend/, or settings change belongs to this
step. No flag flips. No rail calls by Main; the researcher made ONE envelope
probe call (permitted by the step's non-scope).

## Research gate (criterion 7)

Workflow run wf_5d34fa5e-635, agentType=researcher, opus/max. Envelope
(captured return, verbatim): tier=moderate, external_sources_read_in_full=6,
snippet_only_sources=23, urls_collected=29, recency_scan_performed=true,
internal_files_inspected=17, gate_passed=true, wrote_brief_file=true.

## Verification command output (criterion 1-6), verbatim

```
$ bash scripts/qa/verify_phase_4000_1_baseline.sh
artifact: /Users/ford/.openclaw/workspace/pyfinagent/handoff/current/cc_rail_baseline_4000_1.md
ok   [a: live flag capture]
ok   [b: process age vs 78.2]
ok   [c: rail row rule]
ok   [d: cadence baseline]
ok   [e: entrypoint]
ok   [f: E1-E7 definitions]
RESULT: PASS (all six sections present with required content)
exit=0
```

## Mutation demonstration (criterion 8), verbatim

Section (d) deleted from a COPY (sed range delete into scratchpad
mutated_baseline.md); the real artifact untouched:

```
$ bash scripts/qa/verify_phase_4000_1_baseline.sh $SCRATCH/mutated_baseline.md
...
FAIL [d: cadence baseline]: missing marker: ## Section (d)
RESULT: FAIL (1 section check(s) failed)
exit=1
```

Observed exit codes: real artifact 0, mutated copy 1.

## Key measured findings (full detail in the baseline artifact)

1. paper_use_claude_code_route is ALREADY True (GET capture 2026-08-06T10:33:35Z)
   -- phase-78.1 steady state. 4000.3 reframes to verify-and-characterize.
2. Metered path already dark: ZERO anthropic-direct rows in llm_call_log (7d)
   vs 615 CC-rail rows.
3. Rail health defect: 27% of rail calls fail with timeout-shaped latencies
   (116-150s vs claude_code_timeout_s=150); worst opus-4-8 43%, haiku-4-5 61%.
4. E8 cadence baseline: 7 trades / 3 closed round trips in 28d (0.75 RT/week);
   32 RTs all-time, reconciled exactly against paper_round_trips (n=32).
5. Backend process (2026-08-05 17:38) post-dates 78.2 -- no restart in 4000.3.
6. Research: `claude -p` context-load overhead dominates cost (45,580
   cache-creation tokens on a 9-token probe); modelUsage collapse bug class;
   flat-fee policy is paused-not-stable; --bare default flip is a watch item.

## Deviations / notes for Q/A

- Handoff files are step-suffixed (contract_4000.1.md etc.) because another
  session's rolling contract.md is mid-flight on phase-82 (precedent
  contract_82.46.md; phase-4000 CONCURRENCY RAIL).
- The discovered-defects register (D1-D3, W1-W2) is recorded in the baseline
  artifact; D1/D3 will be queued as masterplan steps in the same edit as the
  4000.1 status flip, per feedback_queue_discovered_defects_in_masterplan.
- Criterion 6's "artifact git timestamp precedes any 4000.3 flag flip" is
  trivially satisfiable now (no 4000.3 window has occurred); the commit lands
  at the status flip.

## Follow-up (cycle 2) -- Q/A CONDITIONAL findings fixed, 2026-08-06

Cycle-1 verdict: CONDITIONAL (evaluator_critique_4000.1.md, verbatim). All
three findings fixed; evidence CHANGED before this re-spawn:

1. Capture fidelity (Invalid_Precondition): baseline line 4 no longer claims
   blanket verbatim; section (a) now states the fenced block is a 2-key
   PROJECTION of the 45-key GET body, discloses the 307-redirect trap, and
   carries the exact producing command. Precision notes also applied: W2 cite
   is now backend/services/autonomous_loop.py:2347-2358 (basename collision
   disclosed); the complement label now distinguishes the rail clause (exact
   De Morgan complement) from the metered clause (narrowed); third 78.2 commit
   acf89271 added.
2. Scope disclosure (Overgeneralization): the produced-file set in this file
   now names all SIX files including the researcher auto-memory file
   .claude/agent-memory/researcher/project_cc_rail_e2e_4000_1.md.
3. Guard hardening (vacuity survivors): the check script now extracts each
   section's own text (awk index-anchored) and asserts markers IN-SECTION,
   fenced blocks in (a) and (d), both computed cadence figures in (d), the
   recorded **NO** conclusion in (b), and substantive E1-E7 bodies in (f).

Re-run of the full mutation matrix against the hardened script, verbatim:

```
control: exit=0 | RESULT: PASS (all six sections present with required in-section content)
secdel_d: exit=1 | RESULT: FAIL (1 section check(s) failed)          <- criterion-8 whole-section delete
v1_a_fence_gone: exit=1 | RESULT: FAIL (1 section check(s) failed)   <- QA survivor V1, now killed
v2_d_sql_counts_gone: exit=1 | RESULT: FAIL (1 section check(s) failed) <- QA survivor V2, now killed
v3_b_yes: exit=1 | RESULT: FAIL (1 section check(s) failed)          <- QA survivor V3, now killed
v4_f_stripped: exit=1 | RESULT: FAIL (1 section check(s) failed)     <- QA survivor V4, now killed
== real artifact ==
RESULT: PASS (all six sections present with required in-section content)
exit=0
```
