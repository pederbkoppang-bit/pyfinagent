# Evaluator critique -- 66.1 Restore the decision path (Cycle 68, 2026-07-06/07)

Q/A single-agent evaluation (merged qa-evaluator + harness-verifier), first spawn
for this step (prior-CONDITIONAL count = 0).

## VERDICT: CONDITIONAL

Criteria 1, 2, 4 VERIFIED with independent deterministic reproduction. Criterion 3
is wall-clock-gated (scheduled cycle 2026-07-07 18:00 UTC) and honestly PENDING --
masterplan step remains `status: pending`, live_check section 4 is explicitly
marked PENDING with the SQL to be pasted. Per the contract's own plan step 7 and
the tasking note, CONDITIONAL is the expected honest outcome; PASS is barred while
criterion 3 evidence does not exist, FAIL is not warranted for a by-design pending
wall-clock gate.

## 1. Harness-compliance audit (5 items)

1. **Researcher before contract: PASS.** research_brief_66.1.md mtime 00:16:54,
   JSON envelope verbatim: `tier: moderate, external_sources_read_in_full: 7,
   urls_collected: 80, recency_scan_performed: true, internal_files_inspected: 15,
   gate_passed: true`. contract_66.1.md (mtime 00:19:13) research-gate summary
   reflects the brief's load-bearing findings (import-path root cause, choke
   point, deduper P1 bypass, transition-latch canon).
2. **Contract before generate: PASS with one accuracy note.** Authoring order by
   mtime: brief 00:16:54 -> contract 00:19:13 -> code files 00:22:16-00:25:17 ->
   code commit 27d40df5 00:31:37 -> results 00:32. The contract PREDATES all code
   mtimes. NOTE: the claim that the contract was committed "in an earlier commit
   than the code commit" is NOT what git shows -- `git log --follow` says
   contract_66.1.md and research_brief_66.1.md were FIRST committed inside
   27d40df5 itself (the auto-commit hook commits on masterplan/status events, so
   no mid-step earlier commit existed). The protocol requirement (contract
   authored before GENERATE) is satisfied by mtime evidence; the commit-separation
   claim is corrected here for the record. Does not degrade the verdict.
3. **experiment_results_66.1.md: PASS.** Present, verbatim immutable-command
   output, drill transcript, deploy evidence, and a real "Honest disclosures"
   section (see LLM judgment).
4. **Log-last: PASS.** `grep "phase=66.1" handoff/harness_log.md` -> no match.
   (The three "Cycle 68" hits at lines 3845/11627/17287 are 2026-04/05 entries
   from earlier goals' cycle numbering, not this step.)
5. **No verdict-shopping: PASS.** No prior evaluator_critique_66.1.md existed
   before this file; retry_count=0 in masterplan.

## 2. Deterministic checks (run by Q/A, not trusted from files)

### a. Immutable verification command (verbatim)

```
$ source .venv/bin/activate && python -m pytest backend/tests/test_phase_66_1_rail_guard.py -q
........                                                                 [100%]
8 passed in 0.17s
EXIT=0
```

### b. Criterion 1 -- probe gate (VERIFIED)

- Gate placement: `backend/agents/claude_code_client.py:528`
  `_blocked = _rail_guard_blocked()` runs BEFORE `claude_code_invoke` (:538);
  blocked branch (:529-534) returns empty `LLMResponse` with NO `_log_cc_call`
  (no llm_call_log row) and no subprocess spawn. The skip-shape is identical to
  the pre-existing error-shape (:556-560), supporting the "healthy path
  byte-identical / degraded shape unchanged" claim.
- Loop wiring: `backend/services/autonomous_loop.py:221` `rail_guard_reset(_cycle_id)`
  BEFORE the probe (:222); probe-failure branch calls `rail_guard_disable(_rail_detail)`
  (:231) before the P1.
- Mutation-resistance: test_rail_guard_probe_gate_zero_invocations monkeypatches
  `claude_code_invoke` to record attempts, runs 10 calls, asserts
  `attempts == []` AND `skipped_calls == 10` AND `rail_skipped is True` AND the
  guard adds no page (`_isolated_guard == []`). Removing the :528 gate makes the
  mock fire -> `attempts == []` fails. Real behavioral coverage, not tautology.

### c. Criterion 2 -- breaker + exactly-once page (VERIFIED, live-reproduced)

- Threshold: `backend/config/settings.py:176` `claude_rail_breaker_threshold:
  int = Field(20, ge=1, le=500, ...)` -- bounded Field, default 20; read via
  `_rail_guard_blocked`'s companion `_rail_breaker_threshold()`
  (claude_code_client.py:105-109, `getattr(get_settings(), ...)`).
- Exactly-once latch: `_rail_guard_record_failure` (:169-211) trips on
  `consecutive_failures >= threshold and not open`, pages only `if not paged`
  (:183-185); `rail_guard_disable` (:124-133) sets `paged = True`, so the
  probe-disable path consumes the latch (unit-covered by
  test_rail_guard_no_page_when_probe_already_paged: disable + 7 failures -> zero
  pages).
- Deduper reasoning INDEPENDENTLY CONFIRMED: `backend/services/observability/
  alerting.py:46` `_CRITICAL_SEVERITIES = frozenset({"P0", "P1", ...})`; :75
  `if severity in _CRITICAL_SEVERITIES:` -> single occurrence fires (P1 bypasses
  the 3-in-5-min consecutive-failure dedup by phase-62.7 design). Therefore a
  caller-side latch is REQUIRED for exactly-once -- the implementation has it.
- LIVE server-side verification (read-only conversations.history via
  settings.slack_bot_token on C0ANTGNNK8D, oldest=1783290000, 115 messages
  scanned): **exactly ONE** breaker message.
  ```
  api_ok= True scanned= 115 breaker_hits= 1
  ts= 1783377037.178179
  text= '[P1] Claude Code rail breaker OPEN -- 20 consecutive failures; remaining
  rail calls skipped this cycle -- claude_code_rail: cycle_id=drill-66.1-1783377034 |
  consecutive_failures=20 | threshold=20 | last_error=claude CLI '
  ```
  Matches the claimed ts and permalink digits exactly; not two messages.
- Breaker unit test asserts field-level: threshold(5) -> exactly 5 real attempts,
  breaker eats calls 6-9 (skipped_calls==4), exactly ONE page with
  severity==P1, source==claude_code_rail, error_type==breaker_open.

### d. Zero-pages root-cause fix (VERIFIED)

- `python -c "import importlib.util; print(importlib.util.find_spec('backend.services.alerting'))"`
  -> `None` (module genuinely does not exist).
- autonomous_loop.py corrected imports: `raise_cron_alert` at :233, :764, :936,
  :970 (4 sites) plus `raise_cron_alert_sync` at :1448, :1471 -- all from
  `backend.services.observability.alerting`. ZERO occurrences of the broken
  `from backend.services.alerting` import.

### e. Criterion 3 -- honestly PENDING (VERIFIED)

- masterplan phase-66/66.1 `status: "pending"` (confirmed via jq).
- live_check_66.1.md section 4 explicitly "PENDING, wall-clock-gated" with the
  SQL template for the closing session.
- Deploy preconditions reproduced: backend PID 31182 (launchd child, PPID 1,
  caffeinate wrapper 31184 -- the disclosed single-instance shape, no zombie
  double-uvicorn), `ps -o lstart=` -> `Tue Jul 7 00:31:39 2026`, postdating the
  code commit 27d40df5 (00:31:37 +0200) and all changed-file mtimes
  (00:22-00:25). The running process holds the guard code.

### f. Criterion 4 -- policy doc + scope (VERIFIED)

- `docs/runbooks/claude-rail-degraded-mode.md` exists: rail-down => HOLD
  (fail-safe, current behavior); "Gemini fallback: NOT implemented; config-gated
  + operator-token if ever proposed"; operator recovery runbook; probe as the
  half-open check.
- `git show 27d40df5 --stat`: claude_code_client.py, settings.py (+6 lines, ONLY
  the new bounded Field -- verified via diff), autonomous_loop.py,
  cycle_health.py, test file, runbook, contract, brief. NO gate/threshold/
  risk-cap/trailing-stop/orchestrator-scoring files touched. Matches the 66.2
  scope boundary in the contract.

### g. Regression (adjacent suite)

```
$ python -m pytest backend/tests/test_phase_60_4_observability.py -q
16 passed, 1 warning in 1.99s
EXIT=0
```

### h. Criteria integrity

Programmatic whitespace-normalized diff of the contract's 4 numbered criteria vs
masterplan phase-66/66.1 success_criteria: `identical_normalized=True` for all 4
(differences are line-wrapping only).

## 3. Code-review heuristics (5 dimensions) -- no BLOCK, no WARN

- Security: no secrets in diff; no new LLM input path; subprocess uses list args
  (no shell=True); no dep changes.
- Trading-domain: no execution-path change -- the guard alters only the ALREADY-
  FAILING rail path (doomed subprocess -> instant empty response of the same
  shape); rail-down => hold REDUCES trade activity (fail-safe direction);
  kill-switch/stop-loss/perf_metrics untouched; no new LLM-output-to-execution
  path.
- Code quality: `except Exception` at claude_code_client.py:210 is the paging
  fail-open ("paging must never break the rail") with a logged warning -- the
  established observability convention, not a silenced risk guard (NOTE only).
  ASCII-only log strings confirmed in the new code.
- Anti-rubber-stamp: 8 behavioral tests with attempt-recording mocks and
  field-level page assertions + a REAL end-to-end drill (real client, real
  breaker, real Slack delivery) that Q/A independently re-verified server-side.
  No tautological assertions found.
- Evaluator anti-patterns: first spawn, evidence-cited verdict, no shopping.

## 4. LLM judgment

- **Contract alignment:** work matches the contract plan 1:1; criteria copied
  verbatim; hypothesis (import bug silenced pages; gate+breaker make >20 silent
  consecutive failures structurally impossible) is supported by the test that
  asserts a page fires at threshold and by the live drill.
- **Scope honesty: STRONG.** The results file volunteers (i) drill telemetry
  pollution -- 20 labeled `cc_rail:drill_66_1` ok=false rows plus ~14 unlabeled
  rows from a failed first drill attempt, cost 0, all pre-deploy timestamps; and
  (ii) a latent `_resolve_claude_binary` doc/behavior mismatch (shutil.which
  preferred over the CLAUDE_CODE_BINARY env override, docstring claims the
  reverse) deliberately NOT fixed. **Ruling: deferral is CORRECT scope
  discipline.** Changing binary-resolution order is a runtime behavior change to
  the rail outside all four immutable criteria; bundling it into a P0 alerting
  fix mid-step would be uncontracted scope expansion (the exact pattern the
  harness forbids), and the mismatch is doc-vs-behavior with zero cost impact,
  now on the defect register. Fix belongs in its own step with its own test.
- **Research-gate compliance:** contract cites the brief per-claim; envelope
  gate_passed=true; the load-bearing findings (deduper bypass -> caller latch;
  page-on-transition canon; per-cycle window reset) are visibly implemented.

## 5. What closes criterion 3 (for the fresh Q/A after the scheduled cycle)

Paste into live_check_66.1.md section 4: ok=true `agent LIKE 'cc_rail%'` rows
from `pyfinagent_data.llm_call_log` with DATE(ts)=2026-07-07 originating from the
18:00 UTC SCHEDULED cycle (not a manual run, per the 39.1 lesson), then re-spawn
Q/A on the changed evidence (sanctioned cycle-2 flow).

## JSON verdict

```json
{
  "ok": false,
  "verdict": "CONDITIONAL",
  "reason": "Criteria 1/2/4 verified with independent deterministic reproduction (immutable cmd 8 passed EXIT=0; gate-before-invoke + no-log-row at claude_code_client.py:528-534; reset-before-probe at autonomous_loop.py:221; bounded Field default 20; exactly-once latch confirmed incl. deduper P1-bypass at alerting.py:46/:75; LIVE Slack read-back: exactly one breaker P1 at ts=1783377037.178179; find_spec None + 4 corrected imports; runbook hold-policy + config-gated-OFF; scope clean; criteria byte-identical). Criterion 3 is wall-clock-gated to the 2026-07-07 18:00 UTC scheduled cycle and honestly PENDING (masterplan status=pending, live_check section 4 PENDING, deploy preconditions verified: lstart 00:31:39 postdates commit 00:31:37).",
  "violated_criteria": ["criterion_3_scheduled_cycle_cc_rail_bq_rows_pending"],
  "violation_details": [
    {
      "violation_type": "Missing_Assumption",
      "action": "close criterion 3 before the next scheduled trading cycle has run",
      "state": "step deployed 2026-07-07 00:31:39; next scheduled cycle 2026-07-07 18:00 UTC; zero post-deploy scheduled-cycle ok=true cc_rail llm_call_log rows can exist yet",
      "constraint": "criterion 3: 'A SCHEDULED (not manual) trading cycle after deploy writes ok=true agent LIKE cc_rail% rows to pyfinagent_data.llm_call_log (BQ row paste in live_check; scheduled-run evidence per the 39.1 lesson)'"
    }
  ],
  "certified_fallback": false,
  "checks_run": ["harness_compliance_audit", "verification_command", "code_inspection_criterion1", "mutation_resistance_review", "settings_field_bounds", "latch_logic_inspection", "deduper_p1_bypass_confirmation", "live_slack_readback", "find_spec_broken_module", "import_count_grep", "masterplan_status", "deploy_lstart_reproduction", "runbook_read", "scope_git_stat", "regression_suite_60_4", "criteria_byte_diff", "code_review_heuristics", "llm_judgment"]
}
```
