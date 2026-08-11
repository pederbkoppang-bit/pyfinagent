STATUS: COMPLETE -- write-first record, still NOT a verdict
STEP: 86.21
WRITTEN: 2026-08-11T07:23:30Z

CYCLE 4 evaluation. Prior: c1 CONDITIONAL, c2 CONDITIONAL, c3 FAIL (3rd-CONDITIONAL rule fired;
c3 stated all six criteria MET). Counter reset by the FAIL, so any verdict is available on merits.

## Log (appended as established)
- [start] Read .claude/agents/qa.md in full. Scope per prompt: no production module, no frontend,
  no UI claim -> qa.md 1c/1d do not bind. Verified against git: only scripts/qa/*.py changed.

## A. HARNESS COMPLIANCE -- CLEAN (5/5)
1. Research gate: research_brief_86.21.md envelope gate_passed=true, external_sources_read_in_full=7
   (>=5), urls_collected=26 (>=10), recency_scan_performed=true. mtime 2026-08-09T23:58:36Z.
2. Contract-before-generate: contract first-committed dc621419 2026-08-10 00:01:20 +0200;
   scripts first-committed 7897cb8c 2026-08-10 00:06:37 +0200. Research < contract < code. OK.
3. experiment_results_86.21.md present (mtime 2026-08-11T09:21:35Z, cycle-4 commit 8074e371).
4. LOG-is-last / not-yet-closed: masterplan 86.21 status=pending, retry_count=0. harness_log has
   exactly ONE row for the step: "## Cycle 199 -- 2026-08-10 -- phase=86.21 result=FAIL" (:32340).
   Zero result=CONDITIONAL rows -> 3rd-CONDITIONAL escalation NOT armed. Confirmed three ways:
   log grep = 0 CONDITIONAL; the step's OWN tool on the ledger prints consecutive=0 / armed=False;
   prompt disclosure c1 COND, c2 COND, c3 FAIL. FAIL reset the run.
5. No verdict-shopping: evidence CHANGED since c3 (commit 8074e371 touched both scripts and
   experiment_results). Documented cycle-2 flow, not a re-spawn on unchanged evidence.

## B. DETERMINISTIC
- IMMUTABLE CMD: bash -c 'grep -c "^## Cycle" handoff/harness_log.md && ls handoff/current/
  evaluator_critique_*.md | head -3'  ->  1212 + 3 filenames, exit=0.
  NOTE (on the record since cycle 1): this command cannot go red for any defect this step could
  have. Its exit 0 is evidence of NOTHING. Real evidence is the two re-runnable commands below.
- No unintended production change. git status --short shows only hook-appended audit JSONLs
  (pre_tool_use / config_change / instructions_loaded / away_ops health / prompt_leak_redteam),
  one researcher memory file, and Q/A WIP verdict files. git diff HEAD --stat: 6 files, all
  append-only logs. Zero production/frontend/backend files touched.
- Derived .py scope (git is the authority; zsh word-split avoided by while-read, empty-set guard
  armed and it FIRED once on my first attempt, correctly): 2 files --
  scripts/qa/mutation_matrix_86_21.py, scripts/qa/verdict_history_86_21.py.
  ruff --select F821,F401,F811 -> "All checks passed!", exit=0.
  RUFF LIVENESS PROBE: piped `x = undefined_name_zzz` -> F821 raised, exit=1. Tool proven alive,
  so the clean result is a real zero, not a resolver that measured nothing.
- ast.parse: both files parse.
- REPRODUCED: `python scripts/qa/verdict_history_86_21.py --self-test` -> 18 case lines, PASSED,
  exit 0. Matches the cycle-4 stated rule (grep -cE '^   \(' == 18) exactly.
- REPRODUCED: `python scripts/qa/mutation_matrix_86_21.py` -> 14 cells, ALL 14 KILLED, exit 0,
  target md5 c7b4220a2fa894d64c38baacd2627842, integrity unchanged. Matches the stated rule
  (one tuple literal in MUTANTS == 1 cell; AST count = 14).
- LIVE: counter on 36.17 -> 6 verdicts, consecutive=0, grep contrast 3, predicate-mismatch cause.
  counter on 86.21 -> 3 verdicts (COND,COND,FAIL), consecutive=0, grep contrast 0 row(s) while
  the step is pending with three recorded verdicts. That is criterion 1 reproducing LIVE, by me.
- Ledger IS git-tracked (git ls-files hit) and NOT gitignored (git check-ignore exit=1), so the
  §5 auditability claim survives the *.log-gitignore trap that has defeated this project before.

## C. INDEPENDENT MUTATION WORK (my own harness: in-memory exec, __file__ pinned, zero repo write)
Control rc=0 on every run; target md5 unchanged after every run.

ATTACK (a) -- are (vi-b)/(vi-c)/(vi-d) genuine assertions on the SHIPPED _report?
  YES, genuine, NOT re-implemented: they capture the shipped function's stdout under
  redirect_stdout. Proof by construction -- my A4 (off-by-one in the printed consecutive count,
  `{c}` -> `{c + 1}`) was KILLED. A print-layer mutant dies, so the print layer is really observed.
  BUT the coverage is PARTIAL. 10 print-layer mutants, 1 killed, 9 SURVIVED:
    A1  SURVIVED  printed `auto-FAIL armed : False` hardcoded on the KNOWABLE branch  <-- worst
    A2  SURVIVED  unknowable branch prints `auto-FAIL armed : False` instead of
                  "UNKNOWN -- treat as ARMED"                                          <-- worst
    A10 SURVIVED  ordinal hardcoded "1st" -> "a further CONDITIONAL would be the 1st" at c=2
    A8  SURVIVED  status line always prints "ok"
    A9  SURVIVED  verdicts line always prints "(none)"
    A5  SURVIVED  predicate-mismatch CAUSE explanation gutted
    A6  SURVIVED  predicate branch swaps g<->c in its own explanation
    A3  SURVIVED  LEDGER_MISSING bootstrap NOTE deleted
    A7  SURVIVED  LEDGER_EMPTY truncation NOTE deleted
  A1/A2 are the material ones: both are fail-OPEN lies on the arming line, the same class and
  direction as S1 (`consecutive : 0`) which cycle 4 treated as the flagship survivor. Cycle 4
  asserted the consecutive line and the NOT-KNOWABLE refusal but never the ARMED line, in either
  branch. A2 is literally the cycle-3 finding ("prose fail-closed, properties fail-open") moved
  one layer down into the print output.
  Also: (vi-d)'s own comment says "BOTH cause branches must be reachable, and each must print its
  OWN explanation" -- but the code asserts only the blindness branch. A5/A6 prove the else branch
  is unasserted. The comment overclaims what the assertion does.

ATTACK (b) -- is verify_broken_scoring() vacuous?  YES, w.r.t. the claim made for it.
  B3 (does the actual fix work?): injected three guard-IRRELEVANT mutants as real MUTANTS cells --
     syntax error, unimportable module, broken indent. All three scored BROKEN, matrix rc=3,
     and it did NOT print "ALL N MUTANTS KILLED". THE SUBSTANTIVE CYCLE-4 FIX IS REAL. Verified.
  B1 (does the self-check observe that fix?): restored the exact cycle-3 defect -- routed a
     load-time failure back to `killed += 1`. verify_broken_scoring STILL printed
     "the BROKEN branch is reachable, guard-irrelevant mutants cannot be scored as kills",
     the matrix proceeded, and printed "ALL 14 MUTANTS KILLED -- every guard IN THIS MATRIX can
     fail." at rc=0. So the self-check does NOT observe main()'s routing decision at all.
  B2 (what does it actually observe?): made _load swallow compile errors -> self-check returned
     False, "REFUSING TO SCORE", rc=5. So it pins ONE thing: that importlib raises on bad syntax.
     That is an upstream library fact -- qa.md 4c shape #6, "library-fact assertion posing as a
     fixture pin". The commit message's "so the fixed path is observed working rather than
     assumed" is FALSE as measured; the docstring's own wording ("require _load to raise") is
     honest, and the claim built on it is not.
  Severity: WARN, not BLOCK -- the genuine mechanism exists and I verified it (B3), so this is a
  vacuous guard ALONGSIDE a genuine one (qa.md 4c verdict wiring), not sole coverage.

## D. CLAIM AUDIT (qa.md 4b -- point the instrument at the prose)
- 18 self-test cases: REPRODUCES under the stated rule. 14 matrix cells: REPRODUCES (AST count).
- STALE VERBATIM BLOCKS, both inside the sections that present criterion evidence:
  * §6 (criterion 5) self-test capture shows 15 case lines -- missing (vi-b)/(vi-c)/(vi-d).
    Current output has 18. The block does not reproduce.
  * §7 (criterion 6) matrix capture shows md5 9ece5e79b6568feaaced32628fbfb144 and
    "ALL 11 MUTANTS KILLED". Current md5 is c7b4220a2fa894d64c38baacd2627842 and there are 14
    cells. Neither figure reproduces, and §7 CONTRADICTS §10's "14 cells, all killed" in the
    same document. Cycle 4's own fix #3 was withdrawing an unreproducible number; it shipped
    while leaving two unreproducible verbatim blocks in the same artifact.
  Severity WARN: the CORRECT figures are stated in §10 and do reproduce, and §7's block carries
  no cycle label that would mark it historical.
  * live_check_86.21.md §3 heading says "(FIVE statuses; 15 self-test cases)" over the same
    15-line block. THIRD unreproducible capture of the same two commands. live_check §1 and §2
    -- the parts the immutable live_check text actually requires -- DO reproduce: I re-ran the
    counter on 36.17 (byte-identical) and re-ran both git commands at 688ac349 / 7145f566
    (pending / 1 verdict / 0 rows, then pending / 2 verdicts / 0 rows).

## E. BEHAVIOURAL DIFFERENTIALS for the two strongest survivors (not equivalent mutants)
  A1, step at consecutive=2 (rule genuinely ARMED), through the shipped _report:
     baseline  rc=0  "auto-FAIL armed : True  (a further CONDITIONAL would be the 3rd)"
     A1-mutant rc=0  "auto-FAIL armed : False (a further CONDITIONAL would be the 3rd)"
  A2, corrupt ledger (count NOT KNOWABLE):
     baseline  rc=1  "auto-FAIL armed : UNKNOWN -- treat as ARMED until the source is fixed"
     A2-mutant rc=1  "auto-FAIL armed : False"
  Both are real fail-OPEN differentials on the escalation line. Mitigation on A2: the exit code
  stays 1, so a machine consumer of the exit code is still fail-closed; only the human-facing
  text lies. No mitigation on A1 -- exit 0 both ways.

## F. CRITERION MAP (all six MET; findings are WARN-level, not criterion misses)
  C1 MET -- re-derived by me at both commits, AND reproduced LIVE on 86.21 itself.
  C2 MET -- separate ledger; harness_log untouched by every 86.21 commit; reason stated at
            verdict_history_86_21.py:16-31 and experiment_results §3.
  C3 MET -- self-test (i) loads the exact five-verdict sequence -> consecutive=2, armed=True,
            reset-on-FAIL exercised; live six-row history correctly gives 0, disclosed not trimmed.
  C4 MET -- "ADVISORY, not authoritative" at :40-48 and §5, with the residual named (a writer
            Main does not control) and recorded as NOT DONE in §8. Ledger is git-tracked and not
            gitignored, so the weaker auditability claim actually holds.
  C5 MET -- four statuses, property + exit-code assertions through the real entry point.
            GAP: the printed arming line is unasserted in BOTH branches (A1/A2).
  C6 MET -- corrupt/empty/missing all NOTICE; 14/14 killed, control green, md5 stable; and I
            verified the cycle-4 BROKEN-scoring fix independently (B3).

## G. VERDICT SHAPE
  worst-of-lenses: correctness=PASS-level, does-it-reproduce=CONDITIONAL (3 stale captures),
  scope-honesty=CONDITIONAL (two overclaims). min() = CONDITIONAL.
  3rd-CONDITIONAL check: 0 result=CONDITIONAL rows in harness_log for 86.21; ledger says
  consecutive=0; prompt disclosure agrees (COND,COND,FAIL). Escalation NOT armed -> CONDITIONAL
  is available on the merits and is what I return.

COMPLETED: 2026-08-11T07:41:05Z
