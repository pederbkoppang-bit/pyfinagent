STATUS: COMPLETE -- write-first record, still NOT a verdict
STEP: 86.120
WRITTEN: 2026-08-18T08:20:19Z
COMPLETED: 2026-08-18T08:32:00Z

# Q/A write-first record -- step 86.120 (CYCLE 2 re-grade)

Role: Layer-3 Q/A evaluator, Workflow rail. Read `.claude/agents/qa.md` in full at
runtime (STEP 0 binding). This file is a crash-survival record, NOT a verdict.

## Prior-attempt / verdict evidence (gathered as EVIDENCE only)
- `qa_wip.py 86.120 --spawned-at 2026-08-18T08:20:19Z`:
  `source_present: true`, `attempt_number: 2`, `attempt_number_status: "ok"`,
  `attempt_number_is_lower_bound: false`, `prior_attempts: 1`,
  `records_retained: 2` (GAUGE, includes my own record), `records_pruned_known: null`,
  prior record = `verdict_wip_86.120__20260818T075356Z.md`.
- `verdict_history_86_21.py --step 86.120 --evidence-only`: `status: no_rows_for_step`,
  `verdicts: (none)`.
- CROSS-CHECK: `prior_attempts` (1) > ledger verdict count (0) => **LEDGER IS STALE**
  for this step. Sequence from the ledger: UNKNOWN. Separate observation (different
  quantity, not reconstructible from the other): attempt_number = 2.
- Cycle 1's verdict is transcribed in `handoff/current/evaluator_critique_86.120.md`
  (CONDITIONAL, run `wf_a285260f-0dd`). Treated as EVIDENCE, not ground truth; every
  number re-derived below.

## A. Harness compliance (5 items)
1. RESEARCH GATE: `handoff/current/research_brief_86.120.md` exists, 36,779 bytes,
   mtime 2026-08-18T09:31:11 local -- earliest of the chain. (envelope checked below)
2. CONTRACT-BEFORE-GENERATE: mtime chain is scrambled by the CYCLE-2 prose edits
   (contract 10:15:25 > claude_code_client.py 10:14:11). Cycle 1's Q/A recorded the
   ORIGINAL cycle-1 chain as research 09:31:11 < contract 09:35:22 < settings 09:41 <
   tests 09:49:40 < client 09:50:40 < experiment_results 09:51:50 -- i.e. ordering held
   at GENERATE time; the later contract mtime is the documented cycle-2 correction pass.
   NOT a breach; recorded so a later reader does not misread the mtimes.
3. experiment_results_86.120.md present (14,681 bytes).
4. LOG-LAST: `grep -F "86.120" handoff/harness_log.md` -> ZERO rows. masterplan
   86.120 status = "pending". Intact.
5. NO-VERDICT-SHOPPING: evidence CHANGED between spawns -- suite 27 -> 30 tests
   (sha of the test file differs), plus prose corrections. Production file sha256
   UNCHANGED (see B). This is the documented cycle-2 flow, not a re-grade on
   unchanged evidence.

## B. Deterministic checks (all re-derived by me, not read)
- IMMUTABLE COMMAND `.venv/bin/python -m pytest backend/tests/test_phase_86_120_cc_rail_limit_aware_cooldown.py -q`
  -> **30 passed, 1 warning in 2.13s; RAW_EXIT=0** (exit captured bare, not via a pipe).
- sha256 `backend/agents/claude_code_client.py` =
  `76b47a217489eb5be665db2d6eb354181bde5d2746c515c8da63c6f8dde5dcb1`
  -- BYTE-IDENTICAL to the value cycle 1 captured. Confirms Main's claim that ZERO
  production code changed this cycle; all three fixes are test-only.
- `git diff --stat`: claude_code_client.py +322/-2 (unchanged from cycle 1),
  settings.py +6 (the single new Field, verified by reading the diff),
  charts.py +13/-3 (this session's separately-disclosed NaN fix, out of scope here).
- LINT gate, scope DERIVED (`git diff --name-only HEAD -- '*.py'` UNION
  `git ls-files --others --exclude-standard -- '*.py'`), 5 files, xargs-fed (no
  unquoted-variable word-split): `uvx ruff check --select F821,F401,F811` ->
  **All checks passed!, exit 0**. POSITIVE CONTROL run: a planted
  `x = definitely_undefined_name` on stdin -> F821, exit 1. Gate proven able to go red.
- RUNTIME SMOKE: imported `backend.agents.claude_code_client`,
  `backend.config.settings`, `backend.api.charts` in the venv -- OK.
  `Settings().claude_rail_cooldown_default_hours` = 6.0.
  `cooldown_status()` live = `{'cooldown_active': False}`.
  `rail_guard_status()` live keys include `cooldown_active`.
  Live backend `GET /api/health` -> **http=200**.
- WIDER REGRESSION, scope DERIVED (
  `grep -rln "claude_code_client\|claude_code_invoke\|rail_guard" backend/tests/*.py | grep -v conftest`
  = 17 files, + the charts suite = 18): **1 failed, 366 passed in 43.82s**.
  Reproduces Main's 366/1 exactly. The single failure
  `test_phase_60_3_data_integrity.py::test_60_3_flag_defaults_off` is pre-existing and
  unrelated (asserts a `paper_data_integrity_enabled` default that this environment's
  .env overrides; settings.py diff touches nothing near it).
  NOTE on Main's prose: "17 files, plus this step's own suite ... 18 files total" is
  loosely worded -- the step's own suite is already INSIDE the 17. The total (18) and
  the counts reproduce, so this is wording only.

## C. INDEPENDENT MUTATION MATRIX (in-memory; tree never written)
Method: read the production source, mutate in memory with a uniqueness-asserted
anchor, exec into a module injected as `backend.agents.claude_code_client` (+ parent
package attribute), run the step's suite against it. sha256 re-read after EVERY cell:
`76b47a21...` throughout, `TREE_UNCHANGED=True` on every cell.

| Cell | Mutation | Result |
|---|---|---|
| **C0** | null mutant (no-op) | **30 passed** -- harness proven non-distorting |
| **M10** | delete `cooldown_record_hit(_limit)` from `claude_code_invoke` | **KILLED** 1 failed/29 passed -- `test_a_real_classified_failure_through_generate_content_actually_persists_the_cooldown` |
| **M3** | delete `cooldown_clear_on_success()` from `claude_code_invoke` | **KILLED** 1 failed/29 passed -- `test_a_real_success_after_cooldown_removes_the_persisted_state_file` |
| **M6** | revert tz fallback to `now.tzinfo` | **KILLED** 1 failed/29 passed -- `test_bare_reset_time_with_no_explicit_timezone_uses_host_local_zone_not_utc` |
| M1 | disable the pre-subprocess cooldown gate | KILLED 5 failed/25 passed (same 5 test names as cycle 1) |
| M4 | delete `_rail_guard_open_for_quota(...)` (N=1 accelerated trip) | KILLED 1 failed |
| **M12** | make `cooldown_clear_on_success` a no-op at its DEFINITION (not call site) | KILLED 2 failed -- incl. the new M3 test |
| **M13** | `cooldown_status()` always reports inactive | KILLED 11 failed |
| **M15** | `_cooldown_write(None)` never unlinks the file | KILLED 2 failed -- incl. the new M3 test |

M12/M13/M15 are MINE, not Main's. They matter because they prove the three new guards
are bound to BEHAVIOUR at multiple depths (definition site, status derivation, and the
actual file unlink), not merely to the one call-site literal each was written against.
All three previously-surviving cycle-1 mutants are now killed. **Zero survivors.**

### C-bis. Attribution differential (the one real finding)
`test_a_real_classified_failure_...persists_the_cooldown` carries this inline comment
above its second half:

    # a SECOND call must now spawn ZERO subprocesses, proving the persisted
    # state (not just the in-memory breaker) is what gates the next call.

MEASURED FALSE. Under **M1 alone** (the persisted-cooldown gate fully disabled) this
test stays **GREEN** -- so `run2.assert_not_called()` is NOT satisfied by the persisted
cooldown. I ran the discriminating differential **M1+M4** (gate disabled AND the N=1
accelerated breaker trip removed): the test then goes **RED** (7 failed/23 passed).
=> the second call is blocked by the IN-MEMORY N=1 breaker, not by the persisted state.
This is qa.md 4c vacuity shape #11 (mis-attributed kill mechanism) -- the same class
cycle 1 flagged, recurring inside the fix for it.
SEVERITY: NOTE, not capping. The test's FIRST assertion (`cooldown_active() is True` +
`cooldown_kind == "weekly"`) is the genuine, load-bearing M10 killer (M10 kills exactly
this test), and criterion 3's cross-cycle/post-restart requirement is separately and
genuinely covered by `test_generate_content_skips_subprocess_across_two_cycles_and_a_restart`,
which M1 DOES kill. No criterion loses coverage; the defect is in a code comment's
attribution, and the fix is a one-line comment correction.

### C-ter. Fixture / harness inspection (qa.md 4c: mutate the FIXTURE too)
- `_isolated_cooldown_file` (autouse) monkeypatches `ccc._COOLDOWN_PATH` to a tmp_path
  and calls `rail_guard_reset()` on entry AND exit -- real isolation, no residue on the
  production `backend/agents/_cache/cc_rail_cooldown.json`. Verified: live
  `cooldown_status()` is `{'cooldown_active': False}` after all my runs.
- `_mock_completed` returns a REAL `subprocess.CompletedProcess`, not a dict/duck stub
  -- this is NOT the phase-75.2.1 wrong-type-fixture shape.
- Disclosed fixture limitation (accurate, and the test says so itself): the M6 pin
  `pytest.skip`s at UTC+0. It is genuinely load-bearing on this host (Europe/Oslo,
  non-zero offset -- M6 killed it here), but it would be INERT in a UTC CI runner.
  Honest disclosure in the docstring; recorded as a known bound, not a finding.

## D. Criterion-by-criterion (evidence cited)
1. MET -- `classify_limit_failure` reads the JSON envelope `result` first, falls back to
   raw stdout (matching the :441-443 comment about plain-text-on-stdout).
   Corrected contract wording ("overlapping, not a subset" vs run_away_session.sh:242)
   verified accurate: that regex carries `usage limit` / `out of credit` alternatives the
   new classifier does not, and misses `weekly`/`Opus` which it does.
2. MET -- persistence proven at the PRODUCTION entry point by the new test; M10 kills it.
3. MET -- guard sits at the top of `claude_code_invoke` (before binary resolution);
   `test_generate_content_skips_subprocess_across_two_cycles_and_a_restart` asserts
   `run.assert_not_called()` across 2 `rail_guard_reset()` cycles + a fresh client
   (simulated restart); M1 kills it. Both call sites covered (direct
   `claude_code_invoke` for ticket_queue_processor).
4. MET -- `claude_rail_cooldown_default_hours` Field(6.0, ge=0.5, le=192.0) with a full
   description; live default read as 6.0. Self-clearing proven by the FILE-existence
   assertion (the only property that discriminates cleared-vs-expired); M3/M12/M15 kill it.
5. MET -- `test_MUTATION_removing_limit_detection_reverts_to_generic_breaker` drives the
   mutation through `generate_content` and asserts all three of: 3 subprocess spawns,
   `cooldown_active() is False`, `consecutive_failures == 3` (reverted to the old breaker).
6. MET -- `test_MUTATION_removing_the_pre_subprocess_guard_lets_subprocess_run_again`,
   with a control assertion (`cooldown_active() is True`) before the mutation.
7. MET -- `test_generic_failure_still_feeds_the_old_breaker_and_never_engages_cooldown`
   + `test_generic_breaker_still_trips_at_the_configured_threshold`. Diff reading confirms
   the 66.1 counter/threshold logic is untouched; the new path is additive.
8. MET -- `test_an_always_on_signal_agent_respects_cooldown_via_shared_entry_point` drives
   real `make_client` -> ClaudeCodeClient -> claude_code_invoke; M1 kills it. Corrected
   citation VERIFIED: `backend/services/pead_signal.py:300` reads
   `make_client(getattr(settings, "pead_signal_model", "claude-haiku-4-5"), None, settings, enable_prompt_caching=False)`.
   The softened "functionally equivalent" wording is now accurate.
9. MET -- `test_make_client_routing_breach_guard_still_fires`; `Routing breach` guard
   confirmed live at `backend/agents/llm_client.py:2213`.
10. MET -- `rail_guard_status()` live keys on the running interpreter include
    `cooldown_active`; consumers (`autonomous_loop.py:1951/:2651`, `llm_client.py:2157`)
    are `.get()`-based, so the added keys are non-breaking.
11. MET -- zero survivors across 9 independent cells + C0 control + the M1+M4
    differential; byte-identical tree confirmed on every cell.

## C-quater. EXTENDED matrix -- defensive branches added by this step (MINE)
| Cell | Mutation | Result |
|---|---|---|
| **M17** | `cooldown_status()` except-branch: `active = True` -> `active = False` (fail-SAFE inverted to fail-OPEN on a corrupt/unparseable persisted record) | **SURVIVED -- 30 passed** |
| **M18** | `_cooldown_write`: replace atomic `tmp + os.replace` with a direct `write_text` | **SURVIVED -- 30 passed** |
| M19 | drop the `> now + 9 days` implausibility clamp on a parsed retry_at | KILLED 1 failed -- `test_cooldown_never_trusts_a_past_or_implausible_parsed_time` |
| **C0b** | null control on the llm_client injection path | **30 passed** -- harness non-distorting |
| **M16** | weaken llm_client.py's `Routing breach` guard so it never raises ($0-metered net) | **KILLED** 1 failed/29 passed -- `test_make_client_routing_breach_guard_still_fires`. Log line confirms the test drives the REAL fallthrough (llm_client.py:2198 -> :2213), so criterion 9 is genuinely load-bearing, not a tautology. |

**M17 is the one finding of substance this cycle.** `claude_code_client.py:470` reads
`active = True  # a corrupt record fails toward SAFE (still cooling down)`. Inverting it
to fail-OPEN leaves all 30 tests green. `grep -n corrupt` over the suite returns NOTHING
-- the branch has ZERO coverage. It matters more than a generic uncovered `except`
because the claimed property was LOAD-BEARING in a prior judgement: cycle 1's own
code-review reasoning declined to flag the seven new broad `except Exception` blocks on
the stated ground that they "fail toward MORE cooling (a corrupt cooldown record reads
as still-active)". That safety property was asserted, relied upon, and never tested.
Inverted, a corrupt state file makes the rail resume spawning `claude` subprocesses
against an exhausted quota -- precisely the condition the step exists to prevent.
FIX (cheap, ~5 lines, test-only): write a non-JSON byte string to `_COOLDOWN_PATH` and
assert `cooldown_active() is True`; confirm it goes RED under `active = False`.

**M18 is recorded as a KNOWN BOUND, not a demand.** Atomicity cannot be proven without
crash injection, and requiring a mutant for it would make every defensive branch a new
matrix obligation -- the recursive-evidence-work trap. Shipped behaviour is correct.

## Findings summary
- PRODUCT defects (shipped code behaves wrong): **NONE.** The implementation is correct
  and I verified it working end-to-end independently.
- Criteria 1-10: **all MET**, each independently re-derived (not read).
- Criterion 11: met for every criterion-bearing guard (9 cells, zero survivors), but
  ONE new defensive branch (M17) has no killing mutant -> not literally complete.
- BLOCKING: none.
- NOTE (1): the mis-attributed inline comment in
  `test_a_real_classified_failure_through_generate_content_actually_persists_the_cooldown`
  (second-call assertion is satisfied by the in-memory breaker, not the persisted state).
  Fix = correct the comment. No coverage gap.
- NOTE (2): experiment_results' "17 files, plus this step's own suite ... 18 files total"
  double-counts the step's own suite in the prose; the total and all counts reproduce.
- Known bound (disclosed by the author): the M6 tz pin skips at UTC+0.
