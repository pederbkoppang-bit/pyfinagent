STATUS: COMPLETE -- write-first record, still NOT a verdict
STEP: 86.84
WRITTEN: 2026-08-17T10:55:25Z
COMPLETED: 2026-08-17T11:07:06Z

# Q/A write-first record -- step 86.84, cycle 8 (Workflow rail)

Spawn stamp: 20260817T105525Z. Role file `.claude/agents/qa.md` read in full from disk.

## Attempt / sequence evidence (gathered, not applied)
- `qa_wip.py 86.84 --spawned-at 2026-08-17T10:55:25Z`: source_present=True,
  attempt_number=8, prior_attempts=7, attempt_number_status='ok',
  attempt_number_is_lower_bound=True, records_pruned_known=None, records_retained=8 (GAUGE).
- `verdict_history_86_21.py --step 86.84 --evidence-only`: status=ok, 7 verdicts,
  sequence = CONDITIONAL -> CONDITIONAL -> CONDITIONAL -> CONDITIONAL -> FAIL -> NO_VERDICT -> FAIL.
- Cross-check: prior_attempts (7) == ledger rows (7). attempt_number (8) exceeds it by
  exactly this in-flight spawn, so the ledger is NOT demonstrably stale.

## A. HARNESS COMPLIANCE -- CLEAN
1. research-gate-before-contract: research_brief_86.84.md envelope brief_status=COMPLETE,
   external_sources_read_in_full=11, urls_collected=19, recency_scan_performed=true,
   gate_passed=true; "## Recency scan (2024-2026)" at :365. Contract :7-11 cites it.
2. contract-before-generate: contract mtime 10:20Z < rail_turn_cap.py 10:45Z <
   mutate_rail_turn_cap.py 10:47Z < experiment_results/live_check 10:54Z. (stat prints LOCAL; converted.)
3. experiment_results_86.84.md present (352 lines, Cycles 4-8).
4. log-last: masterplan status = "pending"; harness_log has only IN-PROGRESS / EVIDENCE-ADDED
   rows for 86.84, no closing result row.
5. no-verdict-shopping: evidence CHANGED since cycle 7 -- rail_turn_cap.py (+19),
   mutate_rail_turn_cap.py (+29/-10), and all three artifacts modified.

## B. DETERMINISTIC
1. `python3 scripts/qa/rail_turn_cap.py --verify` -> **EXIT 0**,
   "VERIFY: PASS -- controls green, turn-exhaustion claim holds."
   590 records / 1288 spawns / 0 missing transcripts.
   qa n=43 dropped=0 non-emitters=0 p50=41 p90=55 max=62 >old-cap(30)=39;
   researcher n=14 dropped=0 non-emitters=0 p50=19 p90=35 max=38 >old-cap(40)=0.
2. `python3 scripts/qa/mutate_rail_turn_cap.py --verify` reproduced EXACTLY as claimed:
   cells=35, real survivors=0, errors=0,
   kills by mode {'VERIFY':28,'ORACLE':1,'INJECTED_TRUTH':2,'MUST_STAY_GREEN':2}, VERIFY: PASS.
   Control asserted GREEN before any mutant (harness aborts otherwise).
3. Byte-identical restore INDEPENDENTLY verified -- md5 recorded by ME before the run and
   re-read after all my own work: rail_turn_cap.py 24fcdf2856269472651b3735ade32798,
   qa.md 4c9faa6d7eb14aba70eea2fc7f804727, researcher.md a9592ee0950e55d24fc3e1bb65d5c26f.
4. ruff F821,F401,F811 over a DERIVED scope (git diff HEAD + ls-files --others, 8 .py files,
   non-empty guard passed, xargs -0): "All checks passed!" exit 0.
5. Frontend gate (qa.md is in the phase diff): `npx tsc --noEmit` exit 0; `npx eslint src`
   0 errors / 55 warnings, exit 0.
6. criterion-4/7 executed tests: `node scripts/qa/verify_rail_retry.mjs` 38/38 exit 0
   (F1 exhaustion yields no value; F2 rethrows the original error; F3 gate_passed still
   recomputed by enforceGate after the retry loop; F4 the loop assigns no verdict field);
   `node scripts/qa/verify_research_gate_workflow.mjs` 124/124 exit 0.

## C. INDEPENDENT RE-DERIVATION OF THE CYCLE-8 DELTA  <-- THE FINDINGS

Delta: collect() carries errored=bool(entry.get("error")); non-emitter predicate adds
`and not s.get("errored")`; errored_n added to post-removal rows; cells S12/S13.

### FINDING A -- the headline claim does not reproduce (Contradiction)
live_check_86.84.md:607-608 verbatim: "The two 529 entries that reddened cycle 7 are counted
under `errored_n` and excluded from the loss signal, correctly."
MEASURED: **errored_n = 0 on BOTH post-removal rows.**
Corpus census of error-bearing workflow_agent entries: **7 total**, dated 2026-07-18..2026-08-06,
agentTypes general-purpose/Explore only -- ZERO qa, ZERO 529, ZERO post_removal.

### FINDING B -- the fix is INERT on the live corpus (Unjustified_Inference)
 (i)  HEAD's committed PRE-cycle-8 rail_turn_cap.py verifies GREEN today, post-removal rows
      byte-identical (dropped=0, killed_n=0, non_emitters=0, p50/p90/max identical).
 (ii) Deleting `and not s.get("errored")` from the working-tree source, NO injection:
      verify_ok=True, problems=[].
 (iii) errored_n = 0 everywhere it is computed.
TRUE MECHANISM: the two run RECORDS were REWRITTEN by a same-runId re-dispatch.
  wf_2fafe515-6a2  record birth 10:29:57Z -> mtime 10:48:05Z  | 1 entry, 2 transcripts on disk
  wf_80376bff-7ae  record birth 10:29:44Z -> mtime 10:44:12Z  | 1 entry, 2 transcripts on disk
(birth times match the cycle-7 critique's "born 10:29:44Z/10:29:57Z" exactly.)
The failed attempts survive ON DISK and still carry the 529:
  agent-a8a91688009ccc502.jsonl turns=38 so=False  ("API Error: 529 Overloaded", 12 hits)
  agent-a5fb7a57499f4fb8a.jsonl turns=10 so=False  ("API Error: 529 Overloaded",  4 hits)
but their workflow_agent entries are GONE, and collect() iterates workflowProgress.
CONTRAST (proves this is a distinct mechanism, not the 86.81 in-script retry):
  wf_078f4125-57a and wf_a6ea31e7-9b9 -- birth == mtime, **2 entries** each, the failed
  40-turn attempt fully visible. In-script retry APPENDS; re-dispatch REPLACES.

### FINDING C -- false premise shipped in PRODUCTION SOURCE (Contradiction)
rail_turn_cap.py:395-402 comment: "...turning the immutable command permanently red
(the corpus is append-only)". The corpus is NOT append-only at the record level, and the
red was not permanent -- it cleared with the fix inert. Same premise at
experiment_results_86.84.md:323-324.

### FINDING D -- undisclosed false-negative channel on the step's own recurrence trigger
The named trigger (criterion 6) is the POST-REMOVAL NON-EMITTER floor. Measured today it has
BOTH failure directions: it fired at cycle 7 on an environmental 529 (false positive, which
cycle 8 addresses), and it can be silenced by record rewrite (false negative) -- two real
post-removal qa non-emitters exist on disk right now while the floor reports non_emitters=0.
Undisclosed. Main recorded the adjacent observation ("a resumed run shares its run_id with
its dead predecessor", experiment_results:344) but routed it only to the verdict-ledger key
doctrine, never to the corpus its own immutable command reads.

### FINDING E -- report claim false (Contradiction)
live_check:604-605 "for each role, `killed_n` and `errored_n` are reported beside
`non_emitters`". The printed report (rail_turn_cap.py:802-808) emits only
n / dropped / non-emitters / p50 / p90 / max / >old-cap. Neither field is printed anywhere.

### FINDING F -- NOTE (cosmetic)
mutate_rail_turn_cap.py:393 hardcodes "no false positive on an operator abort"; S12 prints it
for an ERRORED spawn. Also experiment_results:340-341 "S2/S7/S13 anchors are REBUILT FROM THE
SOURCE BYTES programmatically" is an authoring claim not observable in the artifact; the
loud-failure half IS true (run_source_cell raises on a missing anchor) and applies to all cells.

## CRITERION MAP
1 NOT MET (cycle-7 causal story adopted, not re-derived; published figure measures 0)
2 MET   live_check:141-155, #20625 + #41143 cited, NOs stated
3 MET   uncensored sample reproduced by me: qa p50=41 p90=55 max=62, 39/43 above the old 30
4 MET   verify_rail_retry.mjs 38/38 exit 0
5 MET   rail_drop_rate.py:14-20/234-240, qa-verdict.js:629/650, research-gate.js:879-880
6 NOT MET (trigger has a live, undisclosed false-negative channel -- Finding D)
7 MET   verify_research_gate_workflow.mjs 124/124 exit 0; F3 pins enforceGate after retry
8 MET   matrix reproduced exactly; control green first; byte-identical restore; M6/M6b reported

VERDICT RETURNED: FAIL. Nothing in the tree was modified by this evaluation (md5s re-read
after all work and unchanged).
