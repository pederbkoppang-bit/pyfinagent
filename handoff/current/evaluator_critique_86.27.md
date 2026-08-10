# phase-86.27 -- EVALUATE (cycle 1)

Launched on the Workflow structured-output rail (`.claude/workflows/qa-verdict.js`,
run `wf_a9d0c946-3bc`). **Transcribed VERBATIM from the captured return value.**
Main records the verdict; Main never authors it.

## ok

`true`

## verdict

`"PASS"`

## certified_fallback

`false`

## harness_compliance_ok

`true`

## reason

All 7 immutable criteria MET, each verified by MY OWN execution rather than by reading the author's claims. Criterion 2 (the step) independently confirmed: I derived 10 spellings proven absent from the tracked tree via `git grep -F -c` (127.0.1, 0177.0.0.1, 0x7f.1, 127.65537, [0:0:0:0:0:0:0:1], [::0.0.0.1], [0000:...:ffff:7f00:0001], 2130706434, plus interface-derived 3232257621 / 0xc0a85655 / 030052053125); 6 reach the live backend at HTTP 200 and ALL are refused -- and still refused with the string layer monkeypatched to False, i.e. by the socket-address authority alone. Zero mutating bytes left my process (sentinel = socket.socket.connect replaced by sys.audit + raise; control asserted before any probe). Also refused on the requests/urllib3 family; example.com and 8.8.8.8 stay allowed. Immutable command 40 passed; new module 50 passed; ruff F821/F401/F811 clean over the git-derived 7-file scope (non-empty guard + xargs -0). Full suite reproduces the author's numbers EXACTLY (16 failed / 3351 passed / 12 skipped / 5 xfailed / 1 xpassed, 360.70s vs 360.18s); I root-caused the 3 most-plausibly-attributable failures and all name causes orthogonal to this step (a 2026-07-23 masterplan timestamp baseline; a kill-switch daily_loss assertion; effortLevel=max vs xhigh from the 2026-08-04 operator change) -- "delta attributable = 0" survives falsification. Anti-vacuity is settled by the e2e control I ran myself: DISARMED sends all 14 mutating PUTs (SENT:200), ARMED refuses all 14, GETs SENT:200 in BOTH modes -- so refusals are the guard, not an unreachable stub, and conftest constraint 1 (GETs must not raise) holds. Mutation matrix 7/7 killed on re-run with anchors asserted unique and the tracked source unchanged; I added 5 mutants the author did not include (E1/E3/E4 KILLED, E2/E5 survived as equivalent mutants, no hole). Harness compliance clean (gate_passed true, 19 sources, 43 URLs, recency + coverage.dry; research 08:55 < contract 08:57 < code 09:11-09:15 < results 09:25; not yet logged; status still pending; cycle 1, 0 prior CONDITIONALs). No production module touched -- the diff is conftest.py + scripts/qa + backend/tests only; all seven files byte-identical to 9bda4e6d; handoff/kill_switch_audit.jsonl still sha256 ea78508b..., 64 lines at evaluation time. Five NOTE-level findings recorded in notes, none blocking; the sharpest is an undisclosed degraded-mode UNDER-refusal on this machine's six global IPv6 addresses, latent behind two independently measured conditions.

## violated_criteria

_(none)_

## violation_details

_(none)_

## checks_run

- harness_compliance_audit_5_item
- verification_command_immutable
- python_lint_gate_ruff_F821_F401_F811_derived_scope
- backend_runtime_smoke_live_get_api_health
- syntax_ast_parse
- scoped_test_run_new_module_50_passed
- full_backend_suite_360s_failure_set_rederived
- criterion2_independent_novel_spelling_probe_non_networking_sentinel
- criterion2_socket_layer_isolated_string_layer_neutered
- criterion2_requests_urllib3_client_family_probe
- criterion1_pre_fix_predicate_reconstruction_verified_verbatim_vs_cad38647
- criterion3_collect_only_22_tests
- criterion3_independent_wider_bind_census
- criterion4_independent_29_input_adversarial_junk_battery
- criterion5_independent_latency_and_memoisation_measurement
- criterion6_drift_alarm_docstring_and_replacement_oracle_review
- criterion7_mutation_matrix_rerun_7of7
- criterion7_five_additional_mutants_authored_by_evaluator
- e2e_disarmed_armed_control_executed_independently
- degraded_mode_psutil_absent_probe
- consumer_contract_sweep_removed_symbol
- frozen_parametrize_table_byte_identity_diff
- unintended_production_change_check_per_file_sha256
- kill_switch_audit_integrity_sha256
- code_review_heuristics
- evaluator_critique
- mutation_test

## notes

FIVE NOTE-LEVEL FINDINGS (PASS-with-flag; none blocking, none a criterion miss).

N1 -- UNDISCLOSED, and the sharpest thing I found. experiment_results §5 says the psutil-absent path "degrades to 'only a globally routable address is provably remote', which OVER-refuses". That is one-sided. Measured: this machine has SIX globally-routable IPv6 addresses (2001:4654:6451:0:* on en1). With the psutil import forced to fail, `_is_this_machine('2001:4654:6451:0:31:6467:1ea6:1852')` returns False and `address_is_live_backend((that, 8000))` returns False -- the degraded path classifies one of THIS machine's own addresses as REMOTE, i.e. it also UNDER-refuses. Materiality is bounded by two conditions I measured rather than assumed: (a) psutil IS installed, so the branch is unreached; (b) `lsof -nP -iTCP:8000` shows uvicorn bound IPv4-only (`TCP *:8000 (LISTEN)`, IPv4) and I confirmed `GET http://[2001:4654:...]:8000/api/health` and `http://[::1]:8000/` both return Connection refused -- no IPv6 spelling reaches the book today. Not a criterion-2 miss (criterion 2 concerns spellings that REACH the backend). Suggested 1-line fix: in the `not interfaces_enumerable()` branch of live_backend_origin.py:181-186, return True unconditionally ("cannot prove remote") instead of `not ip.is_global`. If uvicorn is ever moved to a dual-stack bind, this stops being latent.

N2 -- criterion 4 residual. The author's 24 inputs x 3 predicates / 0 raised reproduces. My own 29-input adversarial battery (87 calls) found 27/29 total and TWO escapes, both requiring an object whose OWN dunder raises an uncaught type: an object with a raising `__str__` escapes `except (ValueError, AttributeError, TypeError, UnicodeError)` at scripts/qa/live_backend_origin.py:265 (`is_live_backend`, RuntimeError), and an object whose `__int__` raises OverflowError escapes `except (TypeError, ValueError)` at :333 (`address_is_live_backend`). Both fail CLOSED -- the request does not go out -- and no production call site can supply such an object (conftest passes `_resolve_url(req)` strings; the PEP-578 hook passes a real sockaddr). Every input class the criterion NAMES is total: unparseable URLs, None, bytes, bytearray, int, float, nan, bool, `object()`, dict/list/set/tuple/iterator, IPv4Address. The word "TOTAL" in the docstrings at :152 and :254 is one notch stronger than what is proven.

N3 -- test name promises more than the body asserts. `test_no_stub_in_this_repo_can_ever_bind_the_live_port` asserts only the sysctl fact that 8000 is outside 49152-65535; the repo-wide census its name implies exists only as prose in live_check §C, and that prose scoped itself to `bind((`. I ran the WIDER census myself (`bind((` plus `ThreadingHTTPServer((`/`HTTPServer((` across backend/tests, tests, scripts): 7 server binds, every port derived from `_free_port()` (which itself binds port 0) or literal 0; `scripts/ops/anthropic_max_bridge.py` defaults to PORT 18797 and is env-overridden with an ephemeral port in tests; zero literal-8000 binds anywhere. The claim holds under my broader derivation -- criterion 3 MET -- but the guard is narrower than its name.

N4 -- a "verbatim" invariant that no longer reproduces, through no fault of the code. experiment_results:9 asserts `git diff 9bda4e6d HEAD -- conftest.py scripts/qa backend/tests` is EMPTY. It is not: commit 294a9a09 (a CONCURRENT session's 86.28 cycle-3, 09:26:39, fifty seconds after Main's own artifact commit 1b19b264) changed `scripts/qa/verify_research_gate_workflow.mjs`, which falls inside the `scripts/qa` glob. The claim was true when written. The 86.27 CODE is unaffected -- I verified per-file sha256 for all seven touched files against `git show 9bda4e6d:<f>`: conftest.py, live_backend_origin.py, mutation_matrix_86_27.py, reproduce_86_27_spellings.py, smoke_cc_rail_e2e.py, test_phase_86_27_live_origin_class.py, test_phase_86_6_subprocess_channel.py all IDENTICAL, and `git status --porcelain` on that scope is empty (graded == committed). Lesson for the next artifact: pin the invariant to the FILE LIST, not to a directory glob shared with a concurrent session.

N5 -- mutation-matrix completeness (the author does not overclaim; recorded for the record). I built 5 cells the author did not include. E1 (delete the `is_loopback` half of the line M2 mutates, probe `_is_this_machine('127.0.0.55')`) KILLED -- so BOTH halves of that one line are separately load-bearing and M2 only proved one. E3 (degraded-mode conservative answer) and E4 (skip the degraded branch), both under simulated psutil-absence, KILLED. E2 (drop `if str(ip) in own_addresses(): return True`) SURVIVED and E5 (drop the AF_UNIX `isinstance(address,(str,bytes,bytearray))` guard) SURVIVED -- I checked both for a behavioural differential and both are EQUIVALENT mutants, not holes: E2 is subsumed by the immediately following `own_addresses(refresh=True)` leg (a memoised fast path, cost-only), and E5 reaches the same False by `int('t')` raising ValueError into the existing handler. So no new hole. All my mutants were hermetic (temp copies, child processes) and I asserted the tracked source unchanged after every run.

WHAT I VERIFIED BUT DID NOT REPRODUCE, stated as a bound on this verdict: I did NOT re-run all 13 pre-existing failures in a worktree at cad38647 -- creating a worktree is a filesystem mutation I decline as a read-only evaluator, and a concurrent session is live in this tree. Instead I reproduced the full-suite aggregate exactly and root-caused the 3 members most plausibly attributable to 86.27 (including `test_masterplan_diff_touches_only_the_ten_sibling_insertions`, the one that could have been broken by this step's masterplan insertion -- it fails on a removed `"updated_at": "2026-07-23"` line, a stale phase-75.17 baseline, not on step 86.29). I also did NOT reproduce the author's exact criterion-5 instrumentation counts (59 calls / 61.38 ms / 32.293 ms worst); my independent spot-measures corroborate the MAGNITUDE and show the author reported the more conservative figure (my cold example.com resolve was 1.541 ms vs the author's 32.293 ms), memoisation works (0.0007 ms), 50 numeric canonicalisations cost 0.279 ms total, and an unresolvable `.invalid` host returns True (fail-safe) in 10.1 ms without hanging.

CODE-REVIEW HEURISTICS: no BLOCK, no WARN. No secret in diff. No trading-domain surface touched -- the diff contains zero production modules (`git diff --name-only cad38647 HEAD -- '*.py'` outside backend/tests, scripts/qa, conftest.py is EMPTY), so kill-switch/stop-loss/perf-metrics/max-position heuristics are N/A rather than passed-by-assertion. `subprocess.run` is list-form with `shell=False` and `sys.executable` throughout (safe per the negation list), including the `git grep --fixed-strings --` call that takes a runtime-derived spelling. The two `except Exception  # noqa: BLE001` sites at live_backend_origin.py:112/121 return None into the CONSERVATIVE degraded mode, and conftest's import-failure handler logs `_log.error` and degrades to PORT-ONLY over-refusal -- both fail in the safe direction and are documented, so `broad-except-silences-risk-guard` does not fire. `consumer-contract-break`: conftest's `_LOOPBACK_HOSTS` was deleted; my quoted-glob sweep (the first attempt was eaten by zsh and I reran it rather than report a sweep that never executed) finds every surviving reference is either the deliberate verbatim historical reconstruction in reproduce_86_27_spellings.py, the `not in src` tripwire assertion, or a comment -- no live consumer. `LOOPBACK_HOSTS` is retained in the authority module for external importers and is explicitly consulted by nothing. `illusory-guard` is refuted by the disarmed/armed differential I executed myself.
