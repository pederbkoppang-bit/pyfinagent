# Evaluator critique — phase-86.3

**Cycle 186, EVALUATE pass 1.** Launched via the Workflow structured-output rail
(`.claude/workflows/qa-verdict.js`, run `wf_68da249f-138`). Agent: `qa`, model `opus`,
effort `max`, 42 tool uses, 192,947 tokens, 658s.

Main did NOT author this verdict. The JSON below is the Q/A agent's captured return
value, transcribed **VERBATIM** — no edits, no paraphrase, no omissions.

## Verdict: **CONDITIONAL**  (`ok: false`)

```json
{
  "ok": false,
  "verdict": "CONDITIONAL",
  "reason": "Criteria 1, 2, 3, 6, 7 MET and independently reproduced; criterion 4's channel enumeration is INCOMPLETE (the subprocess channel is absent from both the table and the \"Channels NOT contained\" list, and backend/tests/test_phase_4000_2_cc_rail_smoke.py:204 shells out to scripts/qa/smoke_cc_rail_e2e.py whose --backend-url DEFAULTS to http://localhost:8000 at :469 and which PUTs live settings \u2014 a child process loads no conftest, so the guard is structurally absent there); criterion 5 is literally NOT met (13 other tests changed status, disclosed honestly, accounting verified complete but 1 un-root-caused); and the \"verbatim\" capture of the IMMUTABLE verification command in experiment_results_86.3.md:99-102 records \"4 passed\" while the shipped tree collects 5 and I measured \"5 passed\" (contradicted by the neighbouring 17-passed block in the same document, 17-12=5). No new failures, no production code touched, live journal byte-identical (62 lines / 90e0303130fc546df82e33fe1ebb7c782efd75d74e3b7877e16f76fcdbddf653) across every command I ran including a 4-mutant matrix. All fixable in artifacts + the follow-up step's scope; no code change required.",
  "violated_criteria": [
    "criterion_4_channel_enumeration_incomplete_subprocess",
    "criterion_5_thirteen_other_tests_changed_status",
    "verbatim_verification_command_capture_stale"
  ],
  "violation_details": [
    {
      "violation_type": "Missing_Assumption",
      "action": "Re-derived the live-host-reaching population over backend/tests/ with an independent rule (network-client grep UNION loopback-reference grep) instead of the author's :8000 text grep",
      "state": "backend/tests/test_phase_4000_2_cc_rail_smoke.py is absent from the criterion-4 table in both experiment_results_86.3.md sec.7 and live_check_86.3.md sec.3, and 'subprocess' is absent from the 'Channels NOT contained' list (which names only httpx, raw socket, filesystem). That file runs subprocess.run([sys.executable, scripts/qa/smoke_cc_rail_e2e.py, *argv]) at :204; the script calls urllib.request.urlopen at :90 and its --backend-url argparse default is 'http://localhost:8000' at :469, and it issues settings PUTs. A subprocess loads no conftest, so the phase-86.3 guard is structurally absent in that process. MEASURED MITIGATION: I checked all 11 run_smoke(...) call sites (:219,:232,:244,:251,:259,:275,:290,:357,:373,:387,:401,:415 via live_args) and every one passes an explicit ephemeral-stub --backend-url, so there is no live egress today. test_phase_82_11_autoresearch_failure_paging.py:610 is a second subprocess instance (dead port, benign). By the author's own inclusion standard this file is the same class as test_phase_76_9_2_max_bridge.py, which IS listed.",
      "constraint": "criterion 4: 'the fix is stated in terms of the CHANNEL, not the file: enumerate every test that reaches a live host and say which are now contained and which are not'"
    },
    {
      "violation_type": "Threshold_Not_Met",
      "action": "Compared the current 12 failing node ids against the 26 baseline node ids in live_check_85.4.md sec.5, and re-ran the six files behind the 11-test attribution",
      "state": "13 tests other than this step's own changed status (26 failed -> 12 failed). VERIFIED IN THE AUTHOR'S FAVOUR: the current 12 are a strict subset of the baseline 26 node-for-node (zero new failures), the accounting is arithmetically complete (11+1+1+1 = 14 = 26-12), and the six files behind the 11-test claim collect exactly 106 tests and are 106 passed today, matching the author's forced-paused '11 failed, 95 passed' population exactly. RESIDUAL: test_phase_23_2_15_verify_23_1_smoke.py::test_phase_23_2_15_known_pass_scripts_still_pass is explicitly not root-caused; it shells out to verify scripts with a documented PATH-sensitivity (6 of 8 fail in a PATH-minimal shell, per its own docstring), so it is plausibly environmental. The baseline is provably confounded (captured under a different live kill-switch pause state - the 36.28 coupling).",
      "constraint": "criterion 5: 'no other test changes status vs a measured baseline'"
    },
    {
      "violation_type": "Contradiction",
      "action": "Ran the immutable verification command: bash -c 'source .venv/bin/activate && python -m pytest backend/tests/test_phase_23_2_4_pause_resume_no_deadlock_live.py -q --timeout=120'",
      "state": "Measured '5 passed, 1 warning in 2.05s'; --collect-only lists 5 node ids; git shows the file has had 5 test functions since c4ff90fa. experiment_results_86.3.md:99-102 presents a block labelled as that command's verbatim output reading '....' / '4 passed, 1 warning in 3.75s'. It is internally contradicted by section 4.2 in the same document ('17 passed' for this file plus the 12-test guard file: 17-12=5). The stale capture corresponds to an intermediate state before test_phase_86_3_mutation_the_audit_redirect_is_load_bearing - the criterion-3 substitute proof - existed, so that block does not evidence the load-bearing test passing. The 17-passed block does, and the command itself is green (exit 0).",
      "constraint": "qa.md sec.4b: a 'verbatim' capture must be regenerated, never edited; a number in a verbatim artifact that does not reproduce is a Contradiction finding"
    }
  ],
  "certified_fallback": false,
  "checks_run": [
    "harness_compliance_audit_5_item",
    "research_gate_envelope_gate_passed_8_sources_recency_scan",
    "mtime_ordering_research_1108_lt_contract_1112_lt_code_1118_1119_1126_lt_results_1136",
    "log_last_zero_86.3_entries_in_harness_log_masterplan_status_pending",
    "no_verdict_shopping_cycle_1_zero_prior_CONDITIONALs",
    "immutable_verification_command_5_passed_exit_0",
    "guard_test_file_12_passed",
    "ruff_F821_F401_F811_E9_derived_scope_3_files_non_empty_all_checks_passed",
    "backend_runtime_smoke_api_health_200_via_original_backend_is_up_probe",
    "git_scope_audit_no_production_code_changed",
    "masterplan_verification_block_unedited_only_86.6_added",
    "criterion_1_journal_62_lines_sha256_reproduced_plus_git_diff_shows_prefix_8_row_damage",
    "criterion_3_literal_replay_of_ORIGINAL_prefix_post_helper_against_live_url_BLOCKED",
    "mutation_matrix_control_plus_4_mutants_all_killed",
    "root_conftest_load_verified_both_trees_via_trace_config",
    "tests_tree_collection_AB_confcutdir_identical_725_7errors",
    "tests_tree_zero_mutating_calls_to_8000_derived",
    "criterion_4_independent_census_rederivation",
    "criterion_5_node_id_subset_check_and_106_test_population_rerun",
    "criterion_7_byte_identical_diff_exit_0_plus_skipif_retained",
    "code_review_heuristics"
  ],
  "harness_compliance_ok": true,
  "notes": "HARNESS COMPLIANCE CLEAN (5/5): research_brief_86.3.md gate_passed:true, 8 sources read in full (floor 5), recency scan present, coverage.audit_class:false so dry:false is informational; mtimes order research(11:08) < contract(11:12) < code(11:18/11:19/11:26) < results(11:36); experiment_results + live_check present; log-last correct (zero 86.3 entries in harness_log, masterplan status still pending); cycle 1 so no verdict-shopping and the 3rd-CONDITIONAL counter is 0 (CONDITIONAL is permitted here, not auto-escalated).\n\nDISCLOSURES ADJUDICATED. (2) Criterion-3 split ADEQUATE - and I went further than the author: I extracted the ORIGINAL pre-fix _post_state_transition from c4ff90fa~1, pointed it at the real http://localhost:8000 under the shipped guard, and both POSTs were BLOCKED with the phase-86.3 refusal while the journal sha stayed 90e0303130fc546d and the original _backend_is_up() GET probe still returned True. That refutes vacuity shape #7 (re-implemented test) - the guard catches the actual historical writer's code, not an approximation. The live row-rise half of the criterion is independently in git: `git diff handoff/kill_switch_audit.jsonl` shows exactly 8 uncommitted rows in two 4-row clusters at 2026-08-08 22:29 and 22:37 (54 -> 62). (3) Entailment (a) AND (b) HOLDS, plus a third fact the author did not name: _append_audit at kill_switch.py:433 is unbound with signature (event, **fields) and reads the module global _AUDIT_PATH at call time (:440), and the rewritten cycle test proves the cycle actually CALLS it by asserting 3 rows in the tmp journal. (a)+(b)+(c) => removing the redirect writes live. Sound. (4) Filesystem deferral LEGITIMATE - criterion 4 is scoped to live HOSTS, criterion 1 was measured green regardless, 86.6 is genuinely filed in the masterplan with the false-negative recall test written into its criteria. (5) Agent-tool research fallback MITIGATION SUFFICIENT - I re-verified two of the four load-bearing findings myself (the 76_9_2 ephemeral-port POST; the skipif-at-import with a narrow except tuple). (6) Crash disclosure ADEQUATE and the fix is sound (test_phase_80_2_error_response_contract.py:38 is the in-repo idiom). (7) The corrected census is right as far as it goes, but see the criterion-4 finding.\n\nMUTATION MATRIX (all run in-process via pytest.main(plugins=[in-memory module]) with a SINK installed first so a neutered guard could not reach :8000 - no repo file was written or modified, and the live journal hashed 90e0303130fc546d before and after every run): control 12 passed; m1 _LIVE_BACKEND_PORT->9999 6 failed; m2 _MUTATING_VERBS->frozenset() 6 failed; m3 _resolve_method->always GET 6 failed; m4 guard removed from the urlopen chain 8 failed (additionally killing test_the_guard_is_actually_installed and test_the_slack_guard_still_chains_alongside_it). KILL MECHANISM NAMED: under m1/m2/m3 the pytest.raises(RuntimeError, match=\"phase-86.3 test guard\") assertions were not satisfied and the sink's URLError propagated; under m4 the chain-walk preconditions failed first. 4/4 killed, differing kill SETS - the guard tests are not vacuous.\n\nSTRUCTURAL CLAIMS I CONFIRMED. --trace-config shows `conftest` from the repo root registered for BOTH backend/tests and tests/, and registered BEFORE backend.tests.conftest, so the chain order claim and criterion 6's \"protected by existing, not by opting in\" both hold for the in-process HTTP channel. tests/ collection is IDENTICAL with and without the root conftest (725 collected, 7 errors, both ways via --confcutdir), and I derived that zero files under tests/ make a mutating call to :8000 - so the unmeasured second tree carries no regression I can find. Criterion 7 is byte-identical from the def to EOF (diff exit 0) and its @pytest.mark.skipif is retained at :298, now the only skipif in the file.\n\nNOT APPLICABLE: 1c live-UI capture gate (no UI claims, diff touches no frontend/**). No BLOCK-level code-review heuristic fired: no secrets, no kill-switch reachability or stop-loss or perf-metrics change, the two except clauses in conftest.py are narrowed and logged rather than bare, and no production execution path changed.\n\nTO CLEAR TO PASS (artifacts + scope only, no code change): (1) add the subprocess/child-process channel to criterion 4 - list backend/tests/test_phase_4000_2_cc_rail_smoke.py (and test_phase_82_11_autoresearch_failure_paging.py:610) in the enumeration, add \"subprocess\" to the Channels-NOT-contained list with the scripts/qa/smoke_cc_rail_e2e.py:469 localhost:8000 default and the measured fact that all 11 current call sites pass an explicit stub URL, and fold that channel into 86.6's scope (or file it) so it is queued rather than unnamed; (2) regenerate experiment_results_86.3.md sec.4.1 from the shipped tree (5 passed) or relabel it as an intermediate capture; (3) state in the harness_log append that criterion 5 is not literally met and why the baseline is confounded, rather than logging a clean pass.\n\nKEY PATHS: /Users/ford/.openclaw/workspace/pyfinagent/conftest.py, /Users/ford/.openclaw/workspace/pyfinagent/backend/tests/test_phase_86_3_live_egress_guard.py, /Users/ford/.openclaw/workspace/pyfinagent/backend/tests/test_phase_23_2_4_pause_resume_no_deadlock_live.py, /Users/ford/.openclaw/workspace/pyfinagent/backend/tests/test_phase_4000_2_cc_rail_smoke.py, /Users/ford/.openclaw/workspace/pyfinagent/scripts/qa/smoke_cc_rail_e2e.py, /Users/ford/.openclaw/workspace/pyfinagent/handoff/current/experiment_results_86.3.md, /Users/ford/.openclaw/workspace/pyfinagent/handoff/current/live_check_86.3.md"
}
```

---

## Follow-up — Main's response to EVALUATE pass 1 (cycle-2 flow)

Per CLAUDE.md's canonical cycle-2 flow: the blockers were fixed and the handoff
files updated, then a **fresh** Q/A was spawned on **changed evidence**. This is
not verdict-shopping — the evidence differs from what pass 1 graded.

**The Q/A was right on all three findings. None is argued away.**

### 1. `criterion_4_channel_enumeration_incomplete_subprocess` — FIXED

The subprocess channel was genuinely missing, and it belonged in the first
enumeration. **Independently re-derived by Main before acting on it**, and the
measurement went further than the verdict stated:

- `test_phase_4000_2_cc_rail_smoke.py:202` — `run_smoke()` is
  `subprocess.run([sys.executable, str(SCRIPT), *argv], ...)`. **A child process
  loads no conftest, so the guard is structurally absent there.**
- `scripts/qa/smoke_cc_rail_e2e.py:469` — `--backend-url` defaults to
  `http://localhost:8000`.
- **The script mutates**, which the verdict asserted and Main confirmed at
  source: `http_json("PUT", f"{base}/api/settings/", …)` at `:289-290` and
  `http_json("POST", f"{base}/api/analysis/", …)` at `:307`. An invocation
  without `--backend-url` would **PUT settings on the operator's live backend**.
- **Latent, not active:** `base` is an ephemeral stub
  (`ThreadingHTTPServer(("127.0.0.1", 0))` at `:176`), and **all 12 call sites**
  pass an explicit `--backend-url`. `grep -n "run_smoke(" | grep -v live_args |
  grep -v backend-url | grep -v "^202:"` returns nothing.

Added to the enumeration in `experiment_results_86.3.md` §7 and
`live_check_86.3.md` §3 + new §3a, and to both "Channels NOT contained" lists.
**Folded into step 86.6**, whose name, audit_basis and criteria were widened
(Part B, +3 criteria — 86.6 is unstarted, so this is scoping a new step, not
amending a graded one).

*Recorded against myself:* this is the same error in kind as the one already
written down in this step. A worktree relocates file paths but not a socket; a
conftest guard covers the parent process but not a child. I made the
process-boundary version of the mistake after writing the transport-boundary
version down.

### 2. `verbatim_verification_command_capture_stale` — FIXED

§4.1 claimed `4 passed, 1 warning in 3.75s`. The shipped tree collects **5**
and measures `5 passed, 1 warning in 2.05s`. The capture predated
`test_phase_86_3_mutation_the_audit_redirect_is_load_bearing` — the criterion-3
substitute proof — so the block did not evidence the load-bearing test passing,
and it was already contradicted by §4.2 in the same document (17 − 12 = 5).
**Regenerated by re-running the command, not edited**, and the five node ids are
now listed alongside it.

### 3. `criterion_5_thirteen_other_tests_changed_status` — ACCEPTED, not fixable

Literally not met, and it is not something a code change can repair: the
2026-08-08 baseline was captured under a different live kill-switch pause state.
The Q/A independently verified the accounting is complete and arithmetically
sound (11+1+1+1 = 14 = 26−12), that the current 12 are a strict node-for-node
subset of the baseline 26, and that the six files behind the 11-test attribution
collect exactly 106 tests — matching the forced-paused `11 failed, 95 passed`
population.

It also contributed a lead I did not have on the one un-root-caused test:
`test_phase_23_2_15` shells out to verify scripts with a **documented
PATH-sensitivity** (its own docstring records 6 of 8 failing in a PATH-minimal
shell), so its 08-08 failure is plausibly environmental. **Still not
root-caused, and still not claimed.**

Per the Q/A's instruction, the `harness_log.md` append states plainly that
criterion 5 is **not literally met** and why the baseline is confounded, rather
than logging a clean pass.

### Not changed

No production code. No threshold, gate or allowlist. No other step's
`verification` block — `git diff` over `.claude/masterplan.json` shows zero
changes to any `success_criteria` or `command` outside the new 86.6.
