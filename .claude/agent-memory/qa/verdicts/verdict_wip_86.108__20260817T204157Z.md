STATUS: COMPLETE -- write-first record, still NOT a verdict
STEP: 86.108
WRITTEN: 2026-08-17T20:41:57Z

## Q/A write-first record (crash-survival evidence, NOT a verdict)

Spawn: Workflow rail, step 86.108 EVALUATE. qa.md read in full at 20:41:57Z.

### Attempt / sequence evidence
- qa_wip.py 86.108 --spawned-at 2026-08-17T20:41:57Z:
  attempt_number=1 (status=ok, is_lower_bound=false), prior_attempts=0,
  source_present=TRUE, records_retained=1 (gauge, incl. my own), prior_records=[].
- verdict_history_86_21.py --step 86.108 --evidence-only: status=no_rows_for_step,
  verdicts=(none).
- CROSS-CHECK prior_attempts(0) vs ledger rows(0): no staleness signal.

### A. HARNESS COMPLIANCE -- CLEAN
mtime chain: research_brief 20:43:25 < contract 20:53:38 < code 22:30..22:38 <
live_check 22:40:48 < experiment_results 22:41:23.  ORDER OK.
Gate cited wf_8581f683-d24 (15 full-read sources, 35 URLs, audit-class dry@12).
Criteria in contract are VERBATIM-identical to .claude/masterplan.json (compared).
LOG-LAST: masterplan status="pending"; `grep -F 86.108 handoff/harness_log.md` -> 0 hits.
NO VERDICT-SHOPPING: attempt 1, no prior verdict on disk.

### B. DETERMINISTIC -- what reproduced
1. IMMUTABLE CMD -> `parses`, EXIT=0.  REPRODUCES.
2. RUFF F821,F401,F811 over derived scope (git diff HEAD '*.py' UNION
   git ls-files --others '*.py' = 13 files, + census_invalid_json_86_108.py which is
   COMMITTED at 471f6e26 and thus invisible to diff-vs-HEAD): exit=1, 2 findings.
     - debate.py:16 F401 `typing.Callable`  -> PRE-EXISTING (reproduced on
       `git show HEAD:backend/agents/debate.py`).
     - scripts/qa/mutation_86_108.py:34 F401 `sys` -> NEW, step-owned.
3. 20 passed in 2.54s. REPRODUCES.
4. mutation_86_108.py re-run: CONTROL rc=0 collected=20; KILLED=12/12; RESTORE
   VERIFIED. REPRODUCES. Kill attributions audited -- M1/M2 credited to a
   misleadingly-named test that DOES assert by_site/by_agent_kind (correct);
   M4/M5 correct; M6 discriminates because the test hardcodes (bool,int,float)
   independently of _SCALAR_TYPES.
5. Regression sweep: `1 failed, 543 passed, 3068 deselected`. REPRODUCES EXACTLY.
   The 1 failure is test_phase_40_2 effortLevel xhigh-vs-max: `.claude/settings.json`
   unmodified (git status empty) and grep of the changed-module names in that test
   file = 0. Unrelated claim CONFIRMED.
6. census --rotated-only: TOTAL 2859, compact 2371 / json 488. REPRODUCES.
7. era_rail: rotated 2859 / incl. live 2874; internal sums all check
   (939+792+640+192+91+146+59=2859; calls 7248; cc 40; zero-cc eras 954).
8. INDEPENDENT BQ CROSS-CHECK of the hardcoded RAIL_MIX (I queried
   pyfinagent_data.llm_call_log myself): anthropic 6385 / gemini 823 /
   claude-code 40 = 7248. Era column sums equal each. NUMBERS ARE TRUE.
9. Population: Settings 264 / FullSettings 45 / gated_scalar_names 168. REPRODUCES.
   No secret-named field in the population.
10. _FIELD_TO_ENV dead rows: 5, and exactly the 5 ASK-1 names. REPRODUCES.
11. Runtime smoke: all 9 changed/new backend modules + backend.main import OK.
12. Live routes on pid 41635: /api/settings/flags 404, /api/observability/parse-failures
    404, /api/observability/latency 200, /api/health 200. REPRODUCES §10 verbatim.
13. I drove the 3 real emit sites + both routes IN-PROCESS myself: 200/200,
    records_seen=3, reconciles=True, by_site correct, flags payload correct,
    typo'd name -> requested_but_unknown. Independent of Main's capture.
14. Auth: neither new path is in _PUBLIC_PATHS -> both auth-gated. No new
    _update_env_var caller. SettingsUpdate untouched in the diff.
15. debate.py:55 == `"response_schema": ModeratorConsensus`. Citation EXACT.

### C. FINDINGS

**F1 (BLOCKING-class) -- the rail stamp is a FLAG READING, not the transport,
and its guard cannot fail.**
- `current_rail()` takes NO arguments (`() -> str`) and reads only
  `paper_use_claude_code_route`. `record_parse_failure` receives no model/client.
- `llm_client.py:2142-2145` enters the CC rail only when
  `model_name.startswith("claude-") AND paper_use_claude_code_route`.
- Live: the flag is **True** right now, `current_rail()` -> `claude_code`, while
  my own BQ query shows 823 gemini-provider calls in the same period. So a
  Gemini-served parse failure is stamped `rail=claude_code` TODAY.
  `paper_rail_failforward_enabled` can also serve Vertex-Gemini with the flag on.
- EXECUTED MUTATION (in-memory via pytest.main(plugins=[...]), no file written):
  a mutant identical to current_rail() except the attribution is INVERTED, with
  the except-path behaviour preserved.
    control current_rail() = claude_code | mutant() = gemini_or_direct
    20 passed -> rc=0 -> **SURVIVED**
  (My first, cruder mutant replaced the whole function and killed via the
  "settings explode" test -- a MIS-ATTRIBUTED kill by my own probe. Redone
  precisely; the precise mutant survives.)
- Sole coverage is `assert rec["rail"] in {"claude_code","gemini_or_direct","unknown"}`
  -- membership, not correctness. Author cell M12 sets rail to `""`, i.e. the one
  value OUTSIDE the vocabulary, so it is the only mutation that assertion can catch.
- `grep -rn current_rail backend/ scripts/` outside the module returns only the
  M12 anchor. No other guard exists.
- NO DISCLOSURE: grep of contract / live_check / experiment_results / the module
  for any flag-vs-transport caveat returns nothing, while three artifacts assert
  the opposite ("the attribution that cannot be recovered from history is
  available on every event from here on"; "the rail is on the record";
  "stamped with the rail that was in force when it happened").

**F2 (WARN) -- criterion 6 says "every new guard"; the rail guard's surviving
mutation is absent from the matrix.** The matrix procedure is excellent (control
first, exit-5 rejected, collection-count pinned, NAMED test required, SHA-256
restore) but it licenses only "these 12 were killed".

**F3 (WARN) -- the era script's re-derivation path is broken.**
`RAIL_QUERY` contains the placeholder `<one row per rotation window: era, lo, hi>`
so it is not executable, and the comment says "re-run it with --refresh-help for
the SQL" -- `--refresh-help` DOES NOT EXIST (`error: unrecognized arguments`).
live_check §3 repeats "embedded in the script so it is re-derivable". The NUMBERS
are true (I verified them against live BQ), the shipped re-derivation path is not.

**F4 (WARN) -- new F401 `sys` in scripts/qa/mutation_86_108.py:34;** ruff gate exits 1.

**F5 (NOTE) -- "368 Moderator" lacks its population.** Contract says 359 (x3);
live_check §4 (x2) and experiment_results (x1) say 368. I nearly filed this as a
Contradiction. It is NOT: 368 is the DEFAULT census (incl. the live log), 359 is
`--rotated-only`. Deltas across all agents sum to exactly the 15 live lines. But
§4 states 368 with no qualifier, immediately after §2/§3 which are explicitly
`--rotated-only` -- in a step whose own rule is "every rate printed with its
denominator or not at all".

**F6 (NOTE) -- live_check §6 renders an in-process TestClient call as `$ GET ...`
with `"pid": 22814`,** which reads as an HTTP call to the server; §10 discloses
the routes are 404 on pid 41635, so a careful reader can reconcile it.

### C2. WHAT IS GENUINELY STRONG (recorded so the next spawn does not re-litigate)
- The C1 impossibility finding is correct, measured, and was raised in the
  CONTRACT rather than discovered during GENERATE.
- The census refuses to print a rate and says why. The era table computes its one
  supported claim instead of asserting it, and records a prior false draft.
- C4 is the best part: population DERIVED from a stated rule, cannot go stale,
  cannot admit a str-typed field by construction; divergence driven end to end.
- C5 corroborated structurally by me, not only by the author's stat.
- No gate loosened, no default verdict fabricated, `_judge_parse_fail_fallback`
  untouched (verified in the diff), ASK-2 filed rather than silently fixed.

### D. BLOCKED CHECK (disclosed)
`stat`/`ls -l` on `backend/.env` was DENIED by my permission surface (twice).
Criterion 5's "no .env write" therefore rests on (a) Main's own stat -- an
author-supplied evidence leg -- plus (b) my independent structural corroboration:
the new route is GET-only with no write path, `SettingsUpdate` is untouched in the
diff, and `_update_env_var`'s only callers are the 4 pre-existing PUT sites.
I treat the block as authoritative and did not work around it.

### E. GATES DELIBERATELY NOT RUN, WITH THE REASON
- qa.md 1b (frontend eslint/tsc): the working tree carries a PEER session's
  uncommitted frontend/** + sovereign_api.py + autonomous_loop.py edits. None is in
  86.108's contract, criteria or file list; experiment_results explicitly disclaims
  autonomous_loop. Running the frontend gate would attribute a peer's in-flight
  edits to this step.
- qa.md 1c (live UI capture): this step makes NO UI claim (contract, criteria and
  live_check are backend-only). N/A.

COMPLETED: 2026-08-17T20:52:33Z
(read from `date -u`, not narrated -- an earlier draft of this line carried an
estimated 20:57:41Z that I had not read from a clock; corrected before returning.)
