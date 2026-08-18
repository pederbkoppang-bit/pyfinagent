STATUS: COMPLETE -- write-first record, still NOT a verdict
STEP: 86.108
WRITTEN: 2026-08-17T21:04:32Z

# Q/A write-first record -- step 86.108 (cycle 3 per Main's advisory: attempts_used=2)

## Plan
A. Harness-compliance audit (5 items)
B. Deterministic: immutable command, git scope, ruff gate, pytest, runtime smoke
C. Criteria 1-6 MET/NOT MET with cited evidence
D. Independent mutation work (fixture + harness shapes, per qa.md 4c)

## Log
- 21:04:32Z read .claude/agents/qa.md in full. Write-first record created.

## Attempt / sequence evidence (gathered, NOT applied as a trigger)
- `qa_wip.py 86.108 --spawned-at 2026-08-17T21:04:32Z`:
  attempt_number=2 (status=ok, is_lower_bound=false), prior_attempts=1,
  source_present=TRUE, records_retained=2 (GAUGE, incl. mine),
  records_pruned_known=null, prior_records=[verdict_wip_86.108__20260817T204157Z.md].
- `verdict_history_86_21.py --step 86.108 --evidence-only`: status=ok,
  detail="1 verdict(s) from the ledger", verdicts=CONDITIONAL.
- CROSS-CHECK prior_attempts(1) vs ledger rows(1): 1 > 1 is FALSE -> no staleness signal.
- NOTE: Main's advisory says attempts_used=2 (would make this attempt 3); the two
  auto sources agree on attempt_number=2. Main is the constrained party; advisory only.

## A. HARNESS COMPLIANCE
(pending)

## B. DETERMINISTIC -- reproduced so far
1. IMMUTABLE CMD `ast.parse(orchestrator.py)` -> `parses`, EXIT=0. REPRODUCES.
2. RUFF F821,F401,F811 over DERIVED scope (git diff HEAD '*.py' UNION
   git ls-files --others, 13 files, fed via `xargs ... < file` after BSD xargs
   rejected `-a`): exit=1, ONE finding -- `F401 typing.Callable`
   backend/agents/debate.py:16. REPRODUCED AT HEAD on `git show HEAD:...` ->
   PRE-EXISTING, matches Main's disclosure. The cycle-1 `sys` F401 in
   mutation_86_108.py is GONE (cycle-2 fix #3 CONFIRMED).
   Tool positive control: it found something, so the invocation is not vacuous.
   census_invalid_json_86_108.py (committed, so invisible to diff-vs-HEAD):
   All checks passed, exit=0.
3. `.venv/bin/python -m pytest backend/tests/test_phase_86_108_parse_failure_ledger.py -q`
   -> **29 passed in 2.29s**. REPRODUCES experiment_results:42.
4. `.venv/bin/python scripts/qa/mutation_86_108.py` -> CONTROL rc=0 collected=29;
   **KILLED=14/14 SURVIVORS=none UNSCORABLE=none**. REPRODUCES.
   INDEPENDENT RESTORE PROOF (md5 before/after, not the script's own line):
   parse_failure_ledger bc5dc0d9d92dd07fbe6639d005d12ed3, debate b6856c411341d1fc7c9e95d475c80da4,
   llm_parse cd003f18564f08df4b20770e518d7211, gated_flags 66bcb6777db394f20862899ef01c0b4c
   -- byte-identical after the run. Criterion 6's restore leg HOLDS.
   Scoring rule audited: control-first, exit-5 rejected, collection-count pinned
   to control, NAMED test required, SHA-256 restore, `finally` deliberately
   without `return`. Genuinely strict.
5. `resolve_rail` MIRRORS the client. Verified against llm_client.py:2142-2145
   (`model_name.startswith("claude-") and paper_use_claude_code_route`), and
   make_client's ONLY earlier branch is `startswith("gemini-")` (:2119), so for a
   claude- model the CC gate is first. Boundary probe agrees case-for-case:
   'claude'->gemini_or_direct, 'Claude-opus-4-8'->gemini_or_direct (both
   startswith-false in BOTH places). :2204-2220 RAISES rather than silently
   billing Anthropic-direct, so `claude_code` is the right stamp there.
6. LIVE flag state read from the venv: paper_use_claude_code_route=True,
   paper_rail_failforward_enabled=False, paper_failforward_model='gemini-2.5-flash'.
   So the failforward->unknown branch is NOT active today; claude- models do
   record `claude_code`. All three declared on the Settings model.
7. `parse_llm_json` production callers: ZERO (only tests + this step's suite).
   The docstring's "still has no production callers (step 75.5.5)" REPRODUCES.
   (`_parse_llm_json` in meta_evolution/directive_rewriter.py is a different fn.)
8. `_effective_model_name(declared, model)` = `declared or model.model_name or None`
   MIRRORS `_generate_with_retry`'s own `effective_model_name = model_name or
   model.model_name` (debate.py:73, risk_debate.py:66). Claim HOLDS.

## C. FINDINGS ESTABLISHED

**QA-F1 (WARN, guard-vacuity at the CALL-SITE seam) -- the model-name resolver
has ZERO coverage; a hardcoded-wrong-model mutant SURVIVES 29/29, reinstating
the cycle-1 defect one seam upstream of where cycle 2 fixed it.**
- The matrix mutates INSIDE `_parse_json` / `resolve_rail` / `gated_flags`.
  No cell mutates a PRODUCTION CALL SITE's `model_name=` argument.
- `test_emit_sites_pass_the_model_through_to_the_record` calls `_parse_json`
  DIRECTLY with an explicit `model_name=`, so it cannot observe a call-site
  regression, despite its docstring "The threading is real, not just an
  accepted kwarg."
- EXECUTED (in-memory, pytest.main(plugins=[...]), NO repo file written;
  control run first at rc=0/29 passed; each cell asserts the patch is LIVE
  before scoring):
    QA1 debate._effective_model_name -> None                : SURVIVED (proof=None)
    QA2 risk_debate._effective_model_name -> None           : SURVIVED (proof=None)
    QA3 debate._effective_model_name -> "claude-opus-4-8"   : SURVIVED
        (proof='claude-opus-4-8'; every Gemini-served debate parse failure would
         be stamped rail=claude_code -- the EXACT cycle-1 misattribution)
- QA1/QA2 degrade to an honest `unknown` (criterion-3 compliant), so on their own
  they are near-equivalent mutants. **QA3 is the load-bearing survivor**: it is a
  wrong-but-in-vocabulary attribution, which is the shape criterion 1 exists to
  prevent, and no assertion anywhere can see it.
- Coexists with GENUINE behavioral guards (resolve_rail truth table + M13/M14 +
  the record-level threading test), so WARN not BLOCK per qa.md 4c wiring.

**QA-F2 (WARN, claim does not reproduce) -- experiment_results:44-45.**
  Block headed "## Verbatim verification output" contains:
    `$ uvx ruff check --select F821,F401,F811 <the 11 files this step owns>`
    `All checks passed!    EXIT=0`
  The same document defines exactly 11 owned files (5 new + 6 modified, its own
  tables at :10-27). I ran ruff on EXACTLY those 11: **exit=1**, `F401
  typing.Callable backend/agents/debate.py:16`. The claimed green does NOT
  reproduce over the population the document itself defines. Two defects:
  (a) an ELIDED argument list inside a block labelled verbatim is unreproducible
      by construction (qa.md 4b);
  (b) it is a hand-assembled scope whose green does not reproduce (qa.md 1a).
  It also CONTRADICTS the step's own disclosure that the debate.py:16 F401 is
  pre-existing -- i.e. Main knows the finding is in a step-owned file.
  UNDERLYING LINT STATE IS HONEST (I reproduced pre-existence at HEAD); the
  defect is in the PROSE, not the code.
  ROOT CAUSE ISOLATED BY EXECUTION: swapping census_invalid_json_86_108.py IN
  and debate.py OUT gives an 11-file set that prints "All checks passed!" exit=0.
  So the pasted green is over an 11-file set from which the ONE file carrying a
  finding is absent, while the 11 files the document itself enumerates exit 1.
  Same block appears at live_check:345-346, and live_check:349 states "The ruff
  gate is clean on this step's files" -- false for the step's own file list.
  live_check:416 DOES disclose the debate.py finding, so nothing is concealed;
  the two statements simply contradict each other.

**QA-F3 (WARN, carried-forward capture) -- the regression sweep in
experiment_results:48 is the CYCLE-1 run, pasted unchanged into the cycle-2
document under "## Verbatim verification output".**
  Claimed: `1 failed, 543 passed, 3068 deselected`.
  I measured:  `1 failed, 552 passed, 3068 deselected` (same -k filter).
  ARITHMETIC RECONCILES EXACTLY: the step suite went 20 -> 29 tests in cycle 2
  (+9); 543 + 9 = 552, and `deselected` is IDENTICAL (3068) because the step
  file matches the `parse` token in the -k expression, so all 29 are selected.
  Per project_verbatim_paste_drift_arithmetic this is STALE TRANSCRIPTION, not
  untested change -- the 9 new tests all pass. The 1 failure reproduces exactly
  (test_phase_40_2 effortLevel xhigh-vs-max) and IS unrelated: `git status
  --short .claude/settings.json` is empty.
  Second instance of the same class as QA-F2 in the same section.

## D. WHAT I VERIFIED AS SOUND (so a re-spawn does not re-litigate)
- All FIVE cycle-1 findings are genuinely fixed, each verified by execution:
  (1) resolve_rail mirrors the client predicate, boundary-for-boundary;
  (2) the guard asserts the VALUE (M13/M14 KILLED; truth table green);
  (3) the `sys` F401 is gone; (4) `--sql` prints executable SQL and the
  `--refresh-help` string survives only inside the comment recording the old
  defect; (5) both Moderator figures ship with populations, and BOTH reproduce
  (359/2859 rotated-only, 368 default incl. live).
- The record seam IS genuinely guarded. I ran two discriminating probes:
    QA4 resolve_rail -> always claude_code : KILLED (10 failed / 19 passed)
    QA5 emit sites stop FORWARDING model_name: KILLED (2 failed / 27 passed)
  So `test_record_carries_agent_kind_site_and_rail` and
  `test_emit_sites_pass_the_model_through_to_the_record` are non-vacuous.
  This is what makes QA-F1 a PRECISE finding rather than a broad one: the
  guarded seam is `_parse_json(model_name=..) -> record -> resolve_rail`; the
  UNGUARDED seam is `run_debate/run_risk_debate/orchestrator -> _parse_json(...)`.
- Nothing outside the step's own test file references the ledger
  (`grep -rl` over backend/tests + scripts = 3 files, all step-owned), and the
  two files mentioning run_risk_debate use `inspect.getsource` / an extracted
  fallback -- neither reaches a ledger call site. So the gap is real, and the
  machinery to close it (a driver for run_debate with a fake client) is cheap.
- Criterion 4 security, verified BY EXECUTION not by reading:
  Settings 264 / FullSettings 45 / gated 168; zero non-scalar overlap; zero
  secret-ish names admitted; `gated_flag_report(only=["anthropic_api_key",
  "ALLOWED_EMAILS","paper_swap_enabled"])` -> flags=['paper_swap_enabled'],
  requested_but_unknown=['ALLOWED_EMAILS','anthropic_api_key'] -- the arbitrary
  .env read is refused BEFORE `_read_env_raw` is reached, and `_read_env_raw`
  re-guards with `if key in keys` (M9 kills its removal). Neither new route is
  in `_PUBLIC_PATHS` (backend/main.py:562) -> both auth-gated.
- Criterion 5 structurally corroborated: the settings_api diff is a GET-only
  addition; `SettingsUpdate` untouched; `_update_env_var`'s only callers remain
  the 4 pre-existing PUT sites (:501,:544,:546,:548).
- Criterion 3: `_judge_parse_fail_fallback` is untouched in the diff; every
  emit site preserves its return value; M1/M2/M4/M5 + the four real-function
  control-flow tests are all KILLED/green. No gate loosened, no verdict fabricated.
- NOT-YET-IN-FORCE reproduced verbatim: pid 41635 (started 15:57:16 local =
  13:57:16Z), /api/observability/parse-failures 404, /api/settings/flags 404,
  /api/observability/latency 200, /api/health 200.
- Runtime smoke: all 9 changed/new backend modules + backend.main import clean.

## E. HARNESS COMPLIANCE -- CLEAN (5/5)
- Research gate BEFORE contract: research_brief_86.108.md present (52,295 B);
  contract cites wf_8581f683-d24, 15 sources read in full, 35 URLs, audit-class
  dry after 12 rounds.
- Contract BEFORE generate: research_brief 20:43:25 < contract 20:53:38 <
  test file 22:58:36 / mutation script 22:59:07 < evidence 23:03:10-23:03:47.
  **CAVEAT I MUST STATE:** the mtimes of parse_failure_ledger.py / debate.py /
  llm_parse.py / gated_flags.py now read 23:08:xx because MY OWN matrix re-run
  rewrote and restored them. Their CONTENT is byte-identical (md5 before/after),
  so the 23:08 stamps are my artefact and are NOT evidence of a late edit. I did
  not report an ordering violation from data I perturbed myself.
- experiment_results present (8,903 B) + live_check present (26,516 B).
- LOG-LAST: `grep -F 86.108 handoff/harness_log.md` -> 0 hits; masterplan
  status="pending" and the masterplan diff adds `"status": "pending"`, no flip.
- NO VERDICT-SHOPPING: evidence CHANGED materially between cycles -- product
  rewrite (current_rail -> resolve_rail + threading), 20 -> 29 tests, 12 -> 14
  mutation cells, era --sql repaired, F401 removed. This is the documented
  cycle-2 flow, not a re-spawn on unchanged evidence.
- Criteria in contract_86.108.md are VERBATIM-identical to .claude/masterplan.json
  (I re-read the masterplan and compared all six); immutable command matches.

## F. BLOCKED / NOT-RUN CHECKS, DISCLOSED
- `stat -f ... backend/.env` was DENIED by my permission surface (same block the
  cycle-1 Q/A hit). Treated as authoritative; not worked around. Criterion 5's
  mtime leg is therefore AUTHOR-SUPPLIED; my independent leg is structural (§D).
- qa.md 1b (frontend eslint + tsc) NOT RUN: the tree carries a PEER session's
  uncommitted frontend/** (9 files), backend/api/sovereign_api.py (a `1y`
  red-line window, matching the peer's RedLineMonitor.tsx edit) and
  backend/services/autonomous_loop.py. None is in 86.108's contract, criteria or
  file list. Running the gate would attribute a peer's in-flight edits to this
  step. NOTE: experiment_results disclaims autonomous_loop but does NOT mention
  sovereign_api.py; I attribute it to the peer on the diff's own content, not to
  this step.
- qa.md 1c (live UI capture): this step makes NO UI claim anywhere in its
  contract, criteria, diff or live_check. N/A; no capture needed or taken.
- Scratchpad note: I used two shell `>` redirects into the SESSION SCRATCHPAD
  (outside the repo) to hold a derived file list and a `git show HEAD:` copy.
  No repo state was written; the Write tool was correctly BLOCKED by
  qa-write-guard when I attempted a scratchpad probe file, and I switched to a
  heredoc instead of seeking a way around it.

## G. CRITERION MAP
1. MET (substance). Per-agent counts re-derived by a committed script with rule
   + command stated; I reproduced 602/359/342/314/310/309/307/264/52 of 2859.
   "rates" deliberately NOT printed -- no denominator exists and the script says
   so, which serves "measured, not inherited" better than a fabricated rate.
   Per-EVENT historical rail split proven non-derivable and pre-declared in the
   CONTRACT; era bucket delivered with an explicit SUPPORTED/NOT-SUPPORTED
   block and an under-tagging caveat. Prospective rail now genuinely measured
   from the model. Deviation judged ACCEPTABLE.
2. MET. Four transports with verbatim guarantees + doc URLs, the in-repo
   refutation (359/2859 and 368/2874 Moderator failures under a declared
   response_schema at debate.py:55 -- citation exact), plus the constraint-tax
   and death-loop literature. Landed before any schema change, and NO schema
   change was made.
3. MET. Record-level countable degradation at all 4 emit sites; legacy warnings
   kept verbatim; no gate loosened; no verdict fabricated; recorder failure
   counted not hidden.
4. MET. Read-only auth-gated route; population DERIVED (168) and secret-safe by
   construction, proven by execution; in_force vs env_file with computed
   divergence; NOT-YET-IN-FORCE honestly disclosed and reproduced.
5. MET (with the disclosed author-supplied mtime leg). ASK-1/2/3 numbered; P5
   deviation explained rather than silently taken.
6. **NOT FULLY MET.** Control-GREEN-first VERIFIED (rc=0, collected=29) and
   byte-identical restore VERIFIED INDEPENDENTLY by md5. But "every new guard":
   `_effective_model_name` (NEW in cycle 2, 2 modules, 6 call sites) and the 3
   orchestrator inline `getattr(client,'model_name',None)` expressions have NO
   test and NO mutation cell. QA3 -- a hardcoded-claude mutant at that seam --
   SURVIVED 29/29 while reinstating the exact cycle-1 misattribution.

## H. VERDICT REASONING
Worst-of-lenses (P1 money-adjacent):
  correctness lens      -> PASS (every shipped line I checked is right)
  does-it-reproduce lens-> CONDITIONAL (2 "verbatim" blocks do not reproduce)
  scope-honesty lens    -> PASS (deviations pre-declared; ASKs numbered; the
                           NOT-YET-IN-FORCE + non-derivability disclosures are
                           unusually good)
  guard-vacuity lens    -> CONDITIONAL (QA-F1, WARN per qa.md 4c wiring: a
                           genuine behavioral guard coexists, so WARN not BLOCK)
min = CONDITIONAL.
NOT FAIL: the product is correct, criteria 2-5 are met with strong executed
evidence, criterion 1's substance is delivered, and all five cycle-1 findings
are genuinely fixed.
NOT PASS: criterion 6's "every new guard" has a named, reproduced, executable
gap on precisely the code cycle 2 added, and two blocks labelled verbatim do
not reproduce.
Anti-sycophancy check: I am not repeating the prior verdict reflexively -- all
five prior findings are CLOSED and I say so; my three findings are new and
independently derived. Nor am I flipping to PASS under a detailed rebuttal.

Lesson saved: .claude/agent-memory/qa/feedback_a_fix_can_relocate_the_defect_one_seam_upstream.md
(+ MEMORY.md pointer under "Guards that stop one seam short").

COMPLETED: 2026-08-17T21:17:56Z
