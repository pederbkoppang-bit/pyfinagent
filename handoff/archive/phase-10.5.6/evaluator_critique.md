---
step: phase-10.5-batch (covers 10.5.0, 10.5.1, 10.5.2, 10.5.3, 10.5.4, 10.5.5, 10.5.6, 10.5.8)
cycle_date: 2026-04-24
agent: qa
verdict: CONDITIONAL
---

# Q/A Critique -- phase-10.5 batch closure

## Harness-compliance audit (5 items)

1. **research_gate: PASS (pre-existing, spot-checked)** -- researcher subagent was NOT re-spawned this cycle. Pre-existing per-step briefs at `handoff/current/phase-10.5.N-research-brief.md` exist for N in {0,1,2,3,4,5,6,8}. Spot-checked **10.5.8** (accessibility -- most skeptical given axe was not re-run): JSON envelope present, `external_sources_read_in_full=6`, `urls_collected=14`, `recency_scan_performed=true`, `gate_passed=true`. Also spot-checked 10.5.0: envelope present, 6 read-in-full / 16 URLs / recency_scan=true / gate_passed=true. Both briefs satisfy the floor (>=5 sources). Briefs legitimately pre-date this cycle, paired 1:1 with the shipped deliverables. Defensible under the phase-17.1 retrospective-closure precedent.

2. **contract_before_generate: FAIL (soft, disclosed)** -- Contract frontmatter: `step: phase-10.5-batch`, `retrospective: true`, `batch: true`. The contract openly flags the breach in its "Honest framing" section lines 22-36. The 8 deliverables shipped in commit 1122a021 days before this contract existed. Same flavor as phase-17.1 (CONDITIONAL). Disclosed, not hidden.

3. **experiment_results_committed: PASS** -- `handoff/current/experiment_results.md` exists with correct frontmatter, per-step evidence tables, and an explicit "Broken verification commands (summary)" section at lines 151-160 honestly calling out the 10.5.0 pytest stdlib-shadow bug and the 10.5.2 missing-audit-script. Caveats section (lines 170-179) enumerates 8 known gaps including unmeasured p95, un-rerun axe, and the archive-hook interaction.

4. **log_last: PASS** -- `grep -c "phase-10.5-batch" handoff/harness_log.md` returns 0. `grep -c "phase=10.5" handoff/harness_log.md` returns 0 (no prior `phase=10.5` historical cycle entry either; batch closure will be the first log entry for this series). Log-last discipline respected.

5. **no_verdict_shopping: PASS** -- `grep -c "phase-10.5-batch" handoff/current/evaluator_critique.md` returns 0 BEFORE this spawn. Prior content of `evaluator_critique.md` was the phase-17.1 rolling critique from the immediately prior cycle. No prior phase-10.5-batch Q/A verdict exists. I am the first Q/A for this batch.

## Batching judgment

- **batching_acceptable: YES_WITH_CAVEATS** (expressed as CONDITIONAL below)
- **reasoning:**
  - (a) All 8 deliverables shipped in ONE commit (1122a021), so the "evidence boundary" per step is genuinely shared -- splitting into 8 separate Q/As would produce 8 near-identical audits of the same commit. That's 8x ceremony with ~1x information gain.
  - (b) Each of the 8 steps has INDEPENDENTLY VERIFIABLE success criteria (separate test filters, separate endpoints, separate BQ view). I was able to per-step verify each, so per-step granularity is preserved in this single spawn.
  - (c) Batching is ONLY defensible for RETROSPECTIVE closure of already-shipped code. It would NOT be defensible for forward work (where 8 independent contract-before-generate cycles are required).
  - (d) The cost of batching: single Q/A has less "adversarial surface area" than 8 independent Q/As. I mitigate by returning CONDITIONAL and enumerating per-step verdicts, so Main can see which specific steps have weaker evidence (10.5.0 p95/cron, 10.5.8 axe).
  - This precedent should NOT generalize to "always batch retrospective closures." The phase-17.1 template closed one step; this batches 8. If future cycles try to batch 20, Q/A should push back harder.

## Deterministic checks (per step)

- **10.5.0:** Live `GET /api/sovereign/red-line?window=30d` -> `series_len=31 >= 25` **PASS**. `sovereign_api.py` exists (548 lines) + wired in `main.py:327-328`. As-written pytest command fails (pre-existing `backend/calendar/` stdlib-shadow bug), but run correctly from repo root it is 7/7 PASS (per experiment_results). p95 latency and cron_slots criteria NOT independently verified this cycle -- disclosed.
- **10.5.1:** `python scripts/migrations/create_strategy_deployments_view.py --verify` -> `[verify] view_exists: PASS`, `[verify] at_least_one_champion_row: PASS`, `[verify] ALL CHECKS PASS` **PASS** (re-ran live in this Q/A).
- **10.5.2:** `curl -sI http://127.0.0.1:3000/sovereign` -> `HTTP/1.1 302 Found` with `location: /login` -> route reachable + NextAuth guard active **PASS**. `frontend/src/app/sovereign/page.tsx` exists. As-written audit script `scripts/audit/sovereign_route.js` does not exist (pre-existing defect, disclosed).
- **10.5.3-10.5.6 (spot-check):** Re-ran `npm run test -- --filter=RedLineMonitor` live: `Test Files 1 passed (1), Tests 4 passed (4), Duration 1.15s` **PASS**. The 4 tests correspond 1:1 to the 4 immutable criteria. Experiment_results reports analogous PASS for ComputeCostBreakdown (5/5), AlphaLeaderboard (4/4), StrategyDetail (4/4) in background tasks this cycle. Not all re-run by me but evidence is consistent.
- **10.5.8:** Re-ran `node frontend/scripts/audit/sovereign_consistency.js` live: `phosphor_icons_only PASS`, `no_emoji_in_ui PASS`, `dark_theme_token_0f172a PASS`. `npm run axe` + `npm run lint` NOT re-run this cycle (disclosed). `wcag_2_1_aa_pass` rests on shipping-time lighthouse artifact only.

## LLM judgment

- **batching_rigor:** The evidence table in `experiment_results.md` gives genuine per-step coverage (each step has its own file list, verification command, as-written result, run-correctly result, and success-criteria mapping). It does NOT lean on "if one PASS, I assume the rest PASS" shortcuts -- each step has independent evidence. Batching did not erode audit rigor in this case; the common-commit framing actually made the evidence denser, not sparser.

- **broken_command_disclosure:** Honest and complete. The 10.5.0 stdlib-shadow bug is correctly diagnosed (backend/calendar package shadowing `import calendar` from `zoneinfo`'s test path). The 10.5.2 missing `sovereign_route.js` is correctly flagged as a never-written script, not a deleted-by-accident one. Main's workarounds are legitimate evidence of deliverable (pytest from repo root IS the correct invocation; HTTP 302 + grep + shell-shape inspection IS the route_reachable criterion in practice). The two steps are ELIGIBLE for closure; the command bugs are orthogonal cleanup tickets. **However**, Q/A should insist these two defects get their own tickets (recommend: `phase-cleanup-verification-commands`) to prevent the "works-around-reality" pattern from becoming normalized.

- **research_gate_spotcheck:** 10.5.8 brief (spot-checked) is dated and cited -- not a placeholder. Code state HAS drifted slightly since the brief (sovereign_consistency.js shipped exactly to the brief's recommendations), but drift is consistent with the brief, not diverged from it. 10.5.0 brief similarly intact. Stale-brief risk is LOW because the briefs pre-date the code by days, not months.

- **axe_skip_legitimacy:** NOT fully legitimate. The immutable criterion is `wcag_2_1_aa_pass` with verification command `npm run axe`. "Code unchanged since shipping" is a reasonable heuristic but not the contract. The shipping-time lighthouse_home_sovereign.json exists but lighthouse != axe-core, and Main admits "npm run axe + npm run lint: not re-run this cycle". This is the weakest-evidence step in the batch. **Recommend: either (a) Main re-runs `npm run axe` before flipping 10.5.8 to done (Main offered this in experiment_results line 149 -- accept the offer), or (b) 10.5.8 stays `pending` for one more cycle while 10.5.0-10.5.6 flip to done now.** I will NOT block the batch on this (other 7 steps are clean) but 10.5.8's per-step verdict is CONDITIONAL.

- **archive_hook_health:** `.claude/.archive-baseline.json` exists and is healthy. Inspected: `seen_done` array contains 10.0, 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7, 10.8, 10.9, 10.10, 10.11, 10.12, 11.0-11.4, ... The 8 batch steps (10.5.0-10.5.6, 10.5.8) are NOT yet in the set, which is correct (they're still pending). On the upcoming masterplan flip, the hook will add all 8 to `seen_done`. Semantics appear correct for batch writes. Q/A recommends Main grep-verify the 8 IDs appear in `seen_done` AFTER the flip as a post-closure sanity check.

## Verdict

```json
{
  "ok": true,
  "verdict": "CONDITIONAL",
  "reason": "All 8 deliverables exist and meet their core immutable criteria under correct invocation. Live spot-checks this cycle confirmed 10.5.0 red-line endpoint (31 series >= 25), 10.5.1 BQ view verify ALL CHECKS PASS, 10.5.2 route 302->login reachable, 10.5.3 RedLineMonitor 4/4 tests, 10.5.8 sovereign_consistency.js PASS. Research gate honored via pre-existing per-step briefs (10.5.0 and 10.5.8 spot-checked, gate envelopes valid). Batching is accepted as a retrospective-only pattern with caveats. CONDITIONAL (not PASS) retains visibility on: (a) contract-before-generate soft breach disclosed in contract; (b) 10.5.0 p95_latency + cron_slots criteria not independently measured this cycle; (c) 10.5.8 wcag_2_1_aa_pass relies on shipping-time lighthouse artifact, axe NOT re-run; (d) two broken verification commands (10.5.0 pytest cd-bug, 10.5.2 missing sovereign_route.js) are worked-around not fixed; (e) no fresh researcher spawn (pre-existing briefs only). Per-step: 7 CONDITIONAL-pass, 10.5.8 CONDITIONAL-with-axe-rerun-recommended.",
  "violated_criteria": [
    "contract_before_generate_soft_violation",
    "batched_qa_retrospective_only",
    "10.5.0_p95_and_cron_unverified_this_cycle",
    "10.5.8_axe_not_rerun_this_cycle",
    "broken_verification_commands_worked_around_not_fixed"
  ],
  "violation_details": [
    {
      "violation_type": "Missing_Assumption",
      "action": "batched mark_step_done(10.5.0..10.5.8 minus 10.5.7) after GENERATE shipped in commit 1122a021",
      "state": "contract.md dated 2026-04-24 post-dates the 8 deliverables by days; Main frames this as retrospective closure",
      "constraint": "feedback_contract_before_generate.md: contract MUST be written before GENERATE"
    },
    {
      "violation_type": "Threshold_Not_Met",
      "action": "assert wcag_2_1_aa_pass for 10.5.8",
      "state": "npm run axe not re-run this cycle; reliance on shipping-time lighthouse_home_sovereign.json (lighthouse != axe-core)",
      "constraint": "10.5.8 verification command: `cd frontend && npm run axe && npm run lint && node scripts/audit/sovereign_consistency.js`"
    },
    {
      "violation_type": "Unjustified_Inference",
      "action": "claim p95_latency_under_800ms and cron_slots_zero_declared for 10.5.0",
      "state": "experiment_results explicitly states 'UNMEASURED this cycle' for p95 and 'not directly verified this cycle' for cron",
      "constraint": "10.5.0 immutable success criteria require direct verification, not 'endpoints feel fast locally'"
    }
  ],
  "per_step_verdicts": {
    "10.5.0": "CONDITIONAL -- deliverable + endpoint live-verified (31 series); p95 and cron_slots unmeasured; verification cmd broken (worked around). Safe to flip to done with caveat.",
    "10.5.1": "PASS -- BQ view verify re-run live this cycle, ALL CHECKS PASS.",
    "10.5.2": "CONDITIONAL -- route reachable (302 login redirect confirmed); verification cmd broken (sovereign_route.js missing). Safe to flip with caveat.",
    "10.5.3": "PASS -- tests re-run live this cycle, 4/4 PASS mapped 1:1 to criteria.",
    "10.5.4": "PASS -- evidence from background task btddmv0gz, 5/5 PASS (not re-run by Q/A but consistent).",
    "10.5.5": "PASS -- evidence from background task bzkjayjk2, 4/4 PASS (not re-run by Q/A but consistent).",
    "10.5.6": "PASS -- evidence from background task bd7ofey76, 4/4 PASS (not re-run by Q/A but consistent).",
    "10.5.8": "CONDITIONAL -- sovereign_consistency.js re-run live PASS (phosphor/emoji/theme); axe NOT re-run this cycle, wcag_2_1_aa_pass relies on shipping-time lighthouse only. Recommend Main re-run `npm run axe` before flipping 10.5.8 OR hold 10.5.8 pending for one more cycle while flipping the other 7."
  },
  "certified_fallback": false,
  "checks_run": [
    "harness_compliance_5_item",
    "batching_judgment",
    "research_brief_envelope_spotcheck_10.5.8",
    "research_brief_envelope_spotcheck_10.5.0",
    "live_curl_redline_endpoint",
    "live_curl_sovereign_route",
    "live_bq_view_verify",
    "live_sovereign_consistency_js",
    "live_npm_test_redlinemonitor",
    "file_existence_check_deliverables",
    "harness_log_not_prepended",
    "prior_critique_no_verdict_shopping",
    "archive_baseline_state_file_inspection"
  ]
}
```

## Main's next actions (if accepting CONDITIONAL as closing verdict)

CONDITIONAL here is the "acknowledged breach + one weak-evidence step" flavor. Recommended closure path:

1. **Accept CONDITIONAL as terminal for 7 of 8 steps** (10.5.0, 10.5.1, 10.5.2, 10.5.3, 10.5.4, 10.5.5, 10.5.6). The contract-before-generate breach is historically un-fixable; the 10.5.0 p95/cron gaps and 10.5.0/10.5.2 command defects are disclosed and orthogonal.
2. **For 10.5.8 specifically, pick ONE:**
   - (preferred) Re-run `cd frontend && npm run axe 2>&1 | tail -20`, append the output to experiment_results as an addendum, then flip 10.5.8 to done with a note citing this critique. That upgrades 10.5.8 to a clean close. No Q/A re-spawn needed -- the axe output is new evidence Main can attach without invalidating this critique.
   - (alternative) Leave 10.5.8 pending, flip only the other 7 in this batch, open a tiny 10.5.8-axe-rerun cycle next.
3. **Append `handoff/harness_log.md`** -- single cycle block covering the batch, verdict=CONDITIONAL, with violated_criteria enumerated.
4. **Flip masterplan statuses** in one write (7 or 8 depending on #2 above). Post-flip: `grep -c "10.5.0" .claude/.archive-baseline.json` etc. to sanity-check the archive hook added all IDs to `seen_done`.
5. **Open a cleanup ticket** (phase-cleanup or similar) for the two broken verification commands (10.5.0 pytest cd-shadow, 10.5.2 missing sovereign_route.js). Do NOT normalize the "works-around-reality" pattern by leaving them un-fixed indefinitely.

**Do NOT respawn Q/A on the same evidence hoping for PASS** -- that is verdict-shopping per `feedback_qa_harness_compliance_first.md`. If Main runs axe and attaches the output, the evidence has genuinely changed and a fresh Q/A would be legitimate, but is also unnecessary -- axe PASS + this critique's CONDITIONAL already map to a defensible close.
