---
step: phase-17.1
cycle_date: 2026-04-24
agent: qa
verdict: CONDITIONAL
---

# Q/A Critique -- phase-17.1

## Harness-compliance audit (5 items)

1. research_gate: PASS -- Pre-existing brief at `handoff/current/alpaca-mcp-research-brief.md` is dated 2026-04-24 (today), JSON envelope on lines 200-210 reports `gate_passed=true`, `external_sources_read_in_full=5`, `urls_collected=15`, `recency_scan_performed=true`. Spot-check of Alpaca README URL returned HTTP/2 200. Sources are real (not hallucinated). Brief legitimately pre-dates 17.2-17.8 work. NOTE: researcher subagent was NOT re-spawned this cycle, but the retrospective framing makes that defensible -- re-spawning would produce a duplicate-with-worse-timestamp brief.
2. contract_before_generate: FAIL (soft) -- Contract frontmatter is `step: phase-17.1`, dated today, and correctly flags `retrospective: true`. However the REAL protocol breach is upstream: 17.2-17.8 shipped with NO step-17.1-specific contract. The contract itself openly admits this in its "Honest framing" section. The breach is disclosed but not un-done. Per-protocol ordering was violated for the parent phase-17 work-cycle; this cycle is cleanup bookkeeping.
3. experiment_results_committed: PASS -- `handoff/current/experiment_results.md` exists with `step: phase-17.1` frontmatter, includes verbatim verification command output (exit=0, returns 2), and has an explicit "Known caveats / honest disclosures" section calling out all four integrity issues (retrospective closure, parent-already-done, 17.4 blocked, no live code exercised).
4. log_last: PASS -- `grep -c "phase-17.1" handoff/harness_log.md` returns 0. Main has not appended early.
5. no_verdict_shopping: PASS -- Pre-existing `evaluator_critique.md` is for task #50 (MASTERPLAN PLANNING, 2026-04-24 earlier cycle), not phase-17.1. It's the rolling file getting overwritten by this spawn. No prior phase-17.1 Q/A verdict exists. I am the first Q/A for phase-17.1.

## Deterministic checks

- verification_command_exit: 0
- verification_command_output: `2` (grep count of `gate_passed` tokens in research brief)
- curl_check_README: HTTP/2 200 (alpaca-mcp-server README is real, reachable)
- research_brief_gate_envelope: present on lines 200-210, `gate_passed: true`
- research_brief_frontmatter: `gate_passed: true` on line 5
- contract_frontmatter_step: `phase-17.1` (correct)
- experiment_results_frontmatter_step: `phase-17.1` (correct)
- harness_log_phase_17_1_count: 0 (log-last respected)
- masterplan_phase_17_status: `done`
- masterplan_phase_17_1_status: `pending` (awaiting this Q/A)
- masterplan_phase_17_4_status: `in-progress` (NOT being flipped)
- git_diff_stat: only handoff/current/contract.md + experiment_results.md + audit jsonls + auto-regen frontend/next-env.d.ts + deleted scheduled_tasks.lock. No code/import/config regressions.

## Success criteria

1. alpaca-mcp-research-brief.md exists with gate_passed=true: PASS -- file exists, frontmatter has `gate_passed: true`, JSON envelope has `"gate_passed": true`.
2. contract.md present for this cycle: PASS -- file exists with step=phase-17.1 frontmatter and cycle_date=2026-04-24.
3. no_regressions: PASS -- git diff shows only markdown handoff artifacts + audit append-only streams + auto-generated next-env.d.ts + a lock-file deletion. Zero Python, TypeScript, or masterplan verification-criteria mutations. Verification command in masterplan.json matches the command run verbatim.

## LLM judgment

- scope_honesty: The experiment_results.md is unusually honest. It explicitly enumerates what was touched this cycle (2 files) vs pre-existing (5 files), discloses the FRED-key sibling work as unrelated, and states "No live code exercised." Contract's "Honest framing" leads with the protocol breach rather than burying it. This is the opposite of rubber-stamp-bait -- Main is asking Q/A to flag the breach. I will comply.

- retrospective_legitimacy: Closing 17.1 retrospectively is defensible ONLY BECAUSE the research brief genuinely pre-dates the 17.2-17.8 work (2026-04-24 on the brief, matching commit 89dd4400 plan and 70cbf355 implementation). The *substance* of the research gate was honored in-timeline; only the contract artifact and masterplan flip slipped. Marking 17.1 `done` now records the truth (research happened) more accurately than leaving it `pending` forever (which would falsely imply research didn't happen). Alternative of `blocked`/`needs_replan` would be misleading -- it's not blocked, it's retroactively documented. PASS with the breach-note in violated_criteria as a soft violation.

- data_integrity: The parent `phase-17 = done` flipped before children 17.1 and 17.4 reached terminal state is a real integrity anomaly that pre-dates this cycle. This cycle does NOT make it worse -- it actually makes it BETTER by moving 17.1 from `pending` to `done`, reducing the parent-children inconsistency from 2 open children to 1 (17.4 remains in-progress, which is correct). Post-closure state: 1 child in-progress, 7 done, parent done. Still anomalous but improving.

- spillover_17_4: Confirmed. 17.4 is explicitly NOT being flipped. Contract line 99 lists step 9 (Researcher MCP dry-run) as `BLOCKED`. experiment_results.md line 74-79 states "17.4 remains blocked... unchanged from the prior 17.4 attempt and is orthogonal to 17.1's closure." No sneak-through. 17.4 will require a separate cycle with `ALPACA_API_KEY_ID` exported to shell at session start.

## Verdict

{
  "ok": true,
  "verdict": "CONDITIONAL",
  "reason": "Mechanical verification passes: all 3 immutable success criteria met, no regressions, honest scope framing, 17.4 spillover avoided, log-last respected. Soft violation retained for audit-trail visibility: retrospective contract-after-GENERATE for the 17.2-17.8 work cycles is a documented protocol-ordering breach per feedback_contract_before_generate.md. CONDITIONAL (not PASS) keeps the breach visible in violated_criteria rather than papering over it.",
  "violated_criteria": ["contract_before_generate_soft_violation"],
  "violation_details": [
    {
      "violation_type": "Missing_Assumption",
      "action": "mark_step_done(phase-17.1) after phase-17.2..17.8 already done",
      "state": "contract.md for phase-17.1 authored 2026-04-24, but commits 89dd4400 (plan) + 70cbf355 (impl) + 17.2-17.8 step-closures all landed before this contract existed",
      "constraint": "feedback_contract_before_generate.md: contract MUST be written before GENERATE; order is research -> contract -> generate -> qa"
    }
  ],
  "certified_fallback": false,
  "checks_run": [
    "harness_compliance_5_item",
    "verification_command",
    "curl_readme_liveness",
    "gate_envelope_parse",
    "frontmatter_step_match",
    "harness_log_not_prepended",
    "masterplan_status_walk",
    "git_diff_regression_scan",
    "prior_critique_check_no_verdict_shopping",
    "spillover_17_4_check"
  ]
}

## Main's next actions (if accepting CONDITIONAL as closing verdict)

CONDITIONAL here is the "acknowledged breach" flavor, not the "fix-and-respawn" flavor. The breach being flagged is HISTORICAL (17.2-17.8 already shipped without contract) -- it is literally un-fixable without time travel. The legitimate closure path is:

1. Accept CONDITIONAL as terminal for this cycle (do NOT respawn Q/A seeking PASS on the same evidence -- that IS verdict-shopping per `feedback_qa_harness_compliance_first.md` and CLAUDE.md `never do` list).
2. Append `handoff/harness_log.md` cycle entry recording verdict=CONDITIONAL with the soft violation noted.
3. Flip `.claude/masterplan.json` phase-17.1 `status: pending -> done` with a `completed_at` timestamp AND a note in the step's `notes` field referencing this critique so the breach stays discoverable.
4. Commit with a message that names the soft violation, e.g. `phase-17.1: retrospective close (CONDITIONAL -- contract-after-generate breach documented)`.

If Main instead wants a PASS verdict, the ONLY legitimate path is to update `feedback_contract_before_generate.md` (or a superseding rule doc) to explicitly carve out "retrospective closure of steps whose substantive work shipped before the contract artifact, provided the research brief pre-dates the downstream work" as an approved pattern, get Peder's sign-off on that carve-out, THEN respawn Q/A on the updated evidence (new rule file = genuinely changed evidence, not verdict-shopping). Without that carve-out, PASS would be rubber-stamping.
