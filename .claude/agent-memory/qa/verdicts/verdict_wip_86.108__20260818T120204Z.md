STATUS: COMPLETE -- write-first record, still NOT a verdict
STEP: 86.108
WRITTEN: 2026-08-18T12:02:04Z

# Q/A write-first record -- step 86.108 (cycle 4)

Spawned via Workflow rail. Prior sequence per Main's ADVISORY disclosure:
CONDITIONAL, CONDITIONAL, CONDITIONAL (to be verified against the ledger myself).

## Plan
- A. harness-compliance audit (5 items)
- B. deterministic: immutable command, git status/diff scope, ruff lint, scoped tests, runtime smoke
- C. claim auditing: re-derive every number in experiment_results / live_check
- D. mutation matrix: the whitelist AST guard (V2), control-first, byte-identical restore
- E. criteria 1-6 MET/NOT MET

## Findings (appended as established)

### Attempt / sequence evidence
- qa_wip.py 86.108 --spawned-at 2026-08-18T12:02:04Z: source_present=True,
  attempt_number=4 (status ok, lower_bound=True), prior_attempts=3,
  records_retained=4 (gauge, not counter).
- verdict_history_86_21.py --evidence-only: status=ok, 3 verdicts,
  CONDITIONAL -> CONDITIONAL -> CONDITIONAL.
- CROSS-CHECK: prior_attempts (3) == ledger rows (3) -> ledger NOT stale.

### Deterministic
- IMMUTABLE COMMAND: `ast.parse(orchestrator.py)` -> `parses`, EXIT=0. REPRODUCES.
- Regression sweep re-run by me: `567 passed, 3143 deselected, 1 warning in 7.48s`
  -> the CAPTURE reproduces (counts identical; only wall-clock differs).
- masterplan 86.108 status=pending (not flipped). harness_log has
  `Cycle 1247 -- phase=86.108 result=PARKED (3rd-CONDITIONAL rule)` -- a PARK row,
  not a result claim for this cycle.

### MUTATION MATRIX (mine, independent) -- control GREEN first
CONTROL (unmutated, AST-channel harness): 37 passed, rc=0.
CONTROL_FULL (unmutated, full-source harness): 37 passed, rc=0.

AST-channel cells:
- Q3 `_client_model_name(None)`               -> KILLED (rc=1) [reproduces Main]
- Q4 `... or "claude-opus-4-8"`               -> KILLED (rc=1) [reproduces Main]
- Q1 `_client_model_name("claude-opus-4-8")`  -> SURVIVED 37/37
- Q6 `_client_model_name("")`                 -> SURVIVED 37/37
- Q2 `_client_model_name(self.general_client)`-> SURVIVED 37/37
- Q5 `_effective_model_name("claude-opus-4-8", model)` (debate) -> SURVIVED the
  AST guard alone

FULL-SOURCE cells (module injected into sys.modules AND read_text patched, so
BOTH the behavioural and the AST channel see the mutation):
- Q1 -> SURVIVED 37/37   (mod_repl=1 disk_repl=2)
- Q2 -> SURVIVED 37/37   (mod_repl=1 disk_repl=2)
- Q5 -> KILLED  (rc=1)   -> the debate sites ARE behaviourally covered; the
  orchestrator sites are NOT. Attribution confirmed, not assumed.

BEHAVIOURAL DIFFERENTIAL of the survivors (executed):
- control `_client_model_name(synthesis)` -> 'gemini-2.5-flash',
  resolve_rail -> ('gemini_or_direct', 'measured: not a claude- model...')
- Q1      `_client_model_name("claude-opus-4-8")` -> None,
  resolve_rail -> ('unknown', 'no_model_in_scope_at_emit_site')
  => a FALSE 'unknown' WITH A FALSE BASIS STRING while a model IS in scope --
     verbatim the outcome `_accepted_model_name_arg`'s docstring says it rejects.
- Q2      `_client_model_name(general=claude)` -> 'claude-opus-4-8',
  resolve_rail -> ('claude_code', ...)  => full RAIL INVERSION.

FINDING: the whitelist rejects the TOKEN `None` and non-Call shapes, but accepts
any OTHER constant and any wrong-but-in-scope client. Same semantic class,
different token -- the third consecutive narrowing of this same guard
(cycle-3 blacklist `ast.Constant`; cycle-4 whitelist + `Constant None`).
No repo file was modified: mutations applied in-memory only.
Q5 killer NAMED (not assumed): test_run_debate_records_the_real_client_model
  -> AssertionError: assert 'claude-opus-4-8' == 'gemini-2.5-flash'.

### Harness compliance (5 items)
1. research gate: brief_status COMPLETE, gate_passed true, 15 read-in-full,
   recency_scan true, urls 35. Contract cites the brief. CLEAN.
2. order: brief 20:43:25 < contract 20:53:38 < commit 8200283c 23:49:23. CLEAN.
3. experiment_results_86.108.md + live_check_86.108.md present. CLEAN.
4. log-last: masterplan 86.108 status=pending (NOT flipped); criteria + command
   byte-identical to 8200283c~1 (verified by json compare). harness_log carries
   only `Cycle 1247 result=PARKED`, a park record, not a verdict claim. CLEAN.
5. no verdict-shopping: evidence CHANGED (8200283c 23:49 after the 21:42
   cycle-3 verdict; f7685272 today 12:01Z). CLEAN.

### Other deterministic
- ruff F821,F401,F811 over 8200283c's OWN py scope (11 files): exit 1, single
  finding backend/agents/debate.py:16 F401 typing.Callable. PROVEN PRE-EXISTING:
  reproduces on the 8200283c~1 copy; Callable count 1 before and after. Queued
  as real step 86.113. Not owned by this step.
- runtime smoke: all 8 changed backend modules import OK in the venv.
- LIVE endpoints (contradicting the artifact's own 404 disclosure -- backend was
  restarted, pid 25117 started 2026-08-18 13:54:49 local):
    /api/settings/flags = 200, /api/observability/parse-failures = 200,
    /api/observability/latency = 200 (control), /api/health = 200.
  Exercised /api/settings/flags?only=... -> reports pid 25117 (the RUNNING
  process), divergent_count 0, population_total 169.
- SECRET SAFETY of the new route, EXECUTED not read: every in_force value is
  bool/int/float (0 exceptions over the full population); no 32+char opaque
  string in the payload; only secret-SHAPED name is auth_enforce_allowlist=False
  (a bool). Route is NOT in _PUBLIC_PATHS -> authenticated.
- 168-vs-169 drift is NOT a finding: the peer session's uncommitted settings.py
  adds `claude_rail_cooldown_default_hours: float` (86.120). Main's 168 was
  correct when claimed.
- V3 reproduces: 75.5.5 PRESENT status=pending. 86.112/86.113/86.114 all PRESENT.
- V5 reproduces: parse_llm_json has ZERO production callers (grep over
  backend/ + scripts/: only tests, its own def/__all__, and two comments).
- criterion 5: backend/.env mtime 2026-08-17T13:06:04Z, BEFORE all step work;
  new route is @router.get only; SettingsUpdate untouched in the diff.
- census: era_rail_86_108.py reproduces ROTATED ONLY 2859.

### EVIDENCE-CLASS FINDINGS (product is sound; these are artifact/guard scope)
F1. ORPHANED PROSE, introduced by TODAY's remediation f7685272. Both files had
    the sweep LINE replaced but not the sentence under it:
      experiment_results_86.108.md:61 `567 passed, 3143 deselected` (zero fails)
      :62-65  "-> the 1 failure is PRE-EXISTING and unrelated: test_phase_40_2..."
      live_check_86.108.md:394 same capture; :397-403 "The single failure is
      pre-existing and unrelated: test_phase_40_2..."
    MEASURED: that test is now GREEN -- `pytest test_phase_40_2_...py` -> 8 passed
    -- repaired by 86.118 (commit 1bf26bf8) to assert effortLevel == "max". It is
    still SELECTED by the -k pattern (3 tests), so the zero-failure capture is
    honest and the surrounding prose is simply false.
    CONSEQUENCE: masterplan step 86.112, filed by this step, is now MOOT.
    Also experiment_results:221 "Both now carry the measured 560" -- they carry 567.
F2. -k RECALL HOLE. live_check:390 calls it "Regression sweep over every adjacent
    suite". KNOWN-MEMBER RECALL TEST (population = backend/tests files importing
    any 86.108-changed module, a set I chose, not the author): 18 known, 13
    selected, 5 MISSED -- test_phase_23_2_14_no_reentrant_locks.py,
    test_phase_32_3_sector_exposure.py, test_phase_70_5_reschedule.py,
    test_phase_82_10_freshness_paging.py, test_phase_86_41_quant_isolation.py.
    STATED HONESTLY: I ran all five -> 39 passed. No regression was hidden; the
    defect is the CLAIM's support, not a red.
F3. WHITELIST IS NOT SOLE-SUFFICIENT (Main's explicit grade-hard question).
    Executed above: Q1/Q6/Q2 survive 37/37 under full-source mutation while Q5 at
    a debate site is killed by a real behavioural driver -- so the AST guard is
    proven the SOLE coverage for the 3 orchestrator sites and it admits in-class
    mutants. Its own docstring overclaims: "a mutation has to look like the real
    thing to pass" -- `_client_model_name("claude-opus-4-8")` passes a string
    literal where a client object is required, does NOT look like the real thing,
    and passes.
F4. STALE DISCLOSURE (safe direction): experiment_results:124-130 and live_check
    §10 say both routes return 404 on pid 41635. Both return 200 on pid 25117.

### Criteria mapping (all six)
C1 MET  -- census reproduces 2,859 by committed script; population rule + command
           in the output; rail delivered as an era bucket with the per-event
           non-derivability, an UPPER bound, and an under-tagging caveat stated
           in the output itself rather than a footnote. Not inherited from 86.69.
C2 MET  -- live_check §4: four transports, guarantees quoted with doc URLs, the
           in-repo refutation (Moderator declares response_schema and still fails
           359/2,859 and 368/2,874, both with denominators). Lands BEFORE any
           change; no schema/prompt change was made.
C3 MET  -- record-level ledger at 4 emit sites; kinds separated; return values
           unchanged; route live and returning 200. No gate loosened; the
           pre-existing fabricated APPROVE_REDUCED is untouched and filed as 86.114.
C4 MET  -- GET /api/settings/flags exercised LIVE by me against pid 25117;
           in_force vs env_file with computed divergent; secret-safety verified
           by execution over the whole population.
C5 MET  -- .env mtime predates all step work; GET-only route; SettingsUpdate
           untouched; ASK-1/2/3 numbered.
C6 MET as worded -- control GREEN first (I reproduced: 37 passed, rc=0),
           19/19, SHA-256 restore. SEE F3: the guard's own semantic class is not
           covered, which is a scope finding against the guard, not a criterion miss.

VERDICT RETURNED: CONDITIONAL (worst-of-3-lenses: correctness PASS,
reproduces PASS, scope-honesty CONDITIONAL). All six criteria MET; every
finding is EVIDENCE/guard-scope class with a named minimal fix.

COMPLETED: 2026-08-18T12:13:17Z


