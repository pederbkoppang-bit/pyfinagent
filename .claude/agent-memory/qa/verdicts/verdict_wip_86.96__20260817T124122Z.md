STATUS: COMPLETE -- write-first record, still NOT a verdict
STEP: 86.96
WRITTEN: 2026-08-17T12:41:22Z

# Q/A write-first record -- step 86.96 (string-args dispatch-kill class on the Q/A rail)

Launch: Workflow structured-output rail, args delivered as a REAL OBJECT.

## Attempt / sequence evidence
- `qa_wip.py 86.96 --spawned-at 2026-08-17T12:41:22Z`: source_present=true,
  attempt_number=1 (status ok, not a lower bound), prior_attempts=0, prior_records=[].
- `verdict_history_86_21.py --step 86.96 --evidence-only`: status=`no_rows_for_step`,
  verdicts=(none). prior_attempts (0) == ledger rows (0) -> not stale for this step.
- FIRST evaluation. No verdict-shopping risk.

## A. Harness compliance -- 5/5 CLEAN
1. research gate: brief_status COMPLETE, gate_passed true, 8 sources read in full
   (>=5), urls_collected 30 -- I independently counted exactly 30 distinct URLs in
   the brief. "Recency scan (2024-2026) -- PERFORMED" section present (3 findings).
2. order by mtime: brief 11:45:27 < contract 12:01:42 < experiment_results 12:05:46
   < live_check 12:06:14. CORRECT.
3. experiment_results_86.96.md present (6,452 B). Contract copies all 7 criteria
   VERBATIM (compared against masterplan, char counts 267/310/156/249/158/170/77).
4. log-last: `grep -cF "phase=86.96" handoff/harness_log.md` = **0**; masterplan
   86.96 status = **pending**. Not yet logged, not yet flipped. CORRECT.
5. no-verdict-shopping: attempt 1.

## B. Deterministic
- IMMUTABLE COMMAND: `bash -c 'source .venv/bin/activate && node --check
  .claude/workflows/qa-verdict.js && echo parses'` -> `parses`, **EXIT=0**.
  Re-run AFTER two mid-eval commits landed: still `parses`, EXIT=0.
- Checker family, bare exit codes measured by me:
  verify_prompt_render_86_90.mjs EXIT=0 (113 passed, 0 failed)
  verify_workflow_args_boundary.mjs EXIT=0 (96)
  verify_research_gate_workflow.mjs EXIT=0 (124)
  Contract cites the pre-step checker as 95/0; 95 + 18 new [7] checks = 113. Reconciles.
- DERIVED diff scope for this step = commits 65e38c3b + 6345166d ONLY:
  65e38c3b -> handoff artifacts + scripts/qa/verify_prompt_render_86_90.mjs (+96)
  6345166d -> masterplan.json, qa-verdict.js, research-gate.js, 2 handoff files
  ZERO *.py and ZERO frontend/** in scope, so lint gates 1a/1b and smoke gate 1d are
  NOT triggered. Running ruff on an empty set would be a FALSE PASS (qa.md's own
  empty-set guard), so it was correctly not run. No UI claims -> gate 1c not triggered.
- Uncommitted tree changes (sovereign_api.py, 5 frontend components, perf_results.tsv)
  are a SEPARATE in-flight workstream (RedLineMonitor gains a "1y" window etc.),
  outside this step's commits. Not 86.96's.
- masterplan diff in 6345166d = the **86.90** status flip pending->done. 86.96's
  immutable criteria are BYTE-UNCHANGED.
- Comment-only claim VERIFIED BY DERIVATION: filtering the added lines of both
  workflow scripts for non-`//`, non-blank content returns the EMPTY SET.
- Checker does NOT mutate the tree: md5 of both workflow scripts and the checker
  IDENTICAL before and after a full run (mutants are in-memory strings + os.tmpdir).

## C. Independent re-derivation -- everything reproduced

### Census (crit 5) -- REPRODUCED, drift disclosed
My own walk of every `*/workflows/wf_*.json` under `~/.claude/projects`
(602 records now, 0 unparseable):
  {'STRING_PARSES': 390, 'ABSENT': 96, 'OBJECT': 112, 'STRING_FAILS': 4}
vs the artifact's 585-record run {OBJECT 95, STRING_PARSES 390, STRING_FAILS 4,
ABSENT 96}. Delta = +17, ALL in OBJECT -- the object-first launches made since.
STRING_PARSES / ABSENT / STRING_FAILS reproduce EXACTLY.
Arithmetic reconciles across all three measurements: brief 394/484=81.4% (580 recs),
results 394/489=80.6% (585), mine 394/506=77.9% (602).

### The 4-event failure class -- REPRODUCED MEMBER-FOR-MEMBER
  wf_b098cab6-87b research-gate 2026-08-06T14:26:44Z 'Unterminated string' pos 123 len 3201
  wf_8375665b-f5a research-gate 2026-08-09T15:01:31Z "Expecting ',' delimiter" pos 4911 len 4911 (pos==len: TRUNCATION)
  wf_1f6b0398-020 qa-verdict    2026-08-16T09:14:58Z "Expecting ',' delimiter" pos 4939 len 5481
  wf_88302c2a-d20 qa-verdict    2026-08-16T09:15:30Z "Expecting ',' delimiter" pos 5536 len 6090

### Reproduction (crit 1) -- DETERMINISTIC
Sliced the SHIPPED classifyArgs and drove it on both stored payloads, 3 runs each:
6/6 THROW `qa-verdict: args are PRESENT but not parseable as JSON (typeof=...`,
matching each record's `error` field verbatim. On top of the author's 4/4.

### Bisection (crit 2) -- REPRODUCED IN BOTH PARSERS
  python: pos 4939 / 5536 "Expecting ',' delimiter"
  node:   "Expected ',' or ']' after array element in JSON at position 4939 / 5536"
  char at pos = '}' ; context `...commands and diff."},"known` / `...cycle 2's did not."},"known`
  SUBSTITUTION } -> ] : BOTH PARSE  => minimal failing input is ONE character
  INSERTION  ] before } : BOTH FAIL => the bracket is WRONG, not MISSING
  ESCAPED QUOTES ruled out: wf_f7d084d8-76c(2) wf_091e2312-0d8(8) wf_ea569c91-52a(8)
    wf_4575b02b-eb0(2) -- all 4 PARSE. Exactly as claimed.
  SIZE ruled out -- and I found STRONGER evidence than the artifact's synthetic
    6,088-char control: **13 PARSING production string payloads exceed 6,090 chars**
    (max 7,358 = wf_a09bbe42-f2e); 35 exceed 5,481.

### Localisation (crit 3) -- per-layer evidence verified, not elimination
  MARSHALLING INNOCENT is genuinely two-sided: each record's `error` field carries
  the script's OWN runtime-computed `len=` and `preview=`, and both match the stored
  args bytes exactly (sha1 f678987c / 151e86ef -- the artifact's cited values).
  PARSE INNOCENT: node and python refuse the same bytes at the same offsets, and 390
  production strings traverse the path successfully. I confirmed the two scripts'
  classifyArgs are behaviourally IDENTICAL (diff is comments + brace style only), so
  "the identical path" is accurate.
  CALLER GUILTY: 4/394 strings fail, 0/112 objects.

### Crit 4 -- LIVE production corroboration beyond any fixture
The 7 criteria in THIS spawn's prompt are identical to `.claude/masterplan.json`'s,
i.e. the object path transmitted them verbatim end-to-end on a real launch. The four
most recent qa-verdict launches all carry `args` as a real object (dict).

## D. MY OWN MUTATION MATRIX vs section [7]
Replicated the checker's runDriver/runDriverRaw and its [7] predicates VERBATIM.
CONTROL on the unmutated shipped source: **GREEN, 0 failures**.

  M-B  render truncates criteria at 3        -> KILLED (obj c4, order, str c4 RED)
  M-C  each criterion loses its last byte    -> KILLED (all 8 byte checks + order RED)
  M-D  STRING PATH ONLY newline collapse     -> KILLED (str c2 + IDENTICAL-bytes RED)
  M-E  guard fails OPEN (parse error eaten)  -> KILLED (dies-LOUD RED)
  M-A  render truncates criteria at 4        -> *** SURVIVED ***
  M-A2 render truncates criteria at 5        -> *** SURVIVED ***

M-A/M-A2 survive the WHOLE checker family: by enumeration, the largest criteria
fixture anywhere in scripts/qa/*.mjs is the 4-element ADVERSARIAL_CRITERIA (every
other is 1 element), and `grep -rn "criteria.length" scripts/ backend/` outside the
workflow scripts returns NOTHING. Reachable: this spawn carries 7 criteria, so
#5-#7 lie outside the guard's reach. Harm class = the 86.90 class exactly (silent
partial loss, invisible from the verdict).
Compounding: contract plan item 6 named THREE cells including "adversarial corpus
neutered" -- the fixture-directed one -- and only TWO landed; the substitution is
not disclosed as a deviation. experiment_results section 6 states the reach globally
where the evidence licenses per-criterion-within-a-4-element-fixture (the parenthetical
"per-criterion byte assertions" does name the bounded mechanism, which softens it).
NOT vacuity: the guard demonstrably CAN fail (5 distinct kills, 4 of them mine), so
the illusory-guard heuristic does not fire. Bounded fixture + prose overclaim.
FIX: lengthen ADVERSARIAL_CRITERIA past any realistic count and/or assert
rendered-criterion-count == input count.

Scoring hygiene of the shipped cells, checked: a non-unique anchor is reported
unscorable (no silent no-match pass); a mutant that fails to build scores
UNSCORABLE, never KILLED; both controls are genuinely false on the clean source.

## E. Criteria disposition
1 MET  2 MET  3 MET  4 MET  5 MET (residual R2)  6 MET (residual R1)  7 MET

Crit 7 airtight: the only rail edit is a `//` comment; commit 65e38c3b touched NO
workflow script; the checker mutates nothing on disk. Nothing here can convert a
non-PASS to a PASS.

## F. Residuals -- EVIDENCE-quality, for QUEUEING not iteration
R1 [crit 6] section [7] fixture-cardinality blind spot (M-A/M-A2 above) + the
   global reach sentence in experiment_results section 6 + the dropped third
   mutation cell. Fix is one line of fixture.
R2 [crit 5] live_check sections 1-3 carry `$` lines that are DESCRIPTIONS
   ("$ node <slice classifyArgs ... and drive both stored payloads>",
   "$ python3 json.loads on both payloads") and the census block carries no `$`
   line at all, in a file headed "Verbatim command output for every figure".
   Substance holds -- the population rule in results section 5 let me reproduce
   the census EXACTLY from prose alone -- but the literal clause is unmet there.
   Sections 4-5 do carry real commands.
R3 [not a criterion; spawn-prompt only] The prompt claims the classifyArgs comment
   was "placed inside the byte-identical phase-86.90 block and therefore MIRRORED
   verbatim into research-gate.js". FALSE: it sits at qa-verdict.js:76-90, OUTSIDE
   the block (118..317), and `grep -c phase-86.96 research-gate.js` = **0**. The
   ARTIFACTS make no such claim. Substantive tail: research-gate.js has the
   behaviourally identical classifyArgs and carried 2 of the 4 historical failures,
   but no object-first contract note.
R4 [NOTE] contract plan item 4 promised sha256 byte equality; shipped guard uses
   exact-substring containment + full prompt-byte identity. Both establish byte
   equality; results section 4 describes what was actually done. Plan deviation only.

## G. Mid-evaluation tree movement
HEAD moved during this evaluation (d33aabe2 phase-86.21 cycle 7 + its changelog
row). It touches only handoff/current/experiment_results_86.21.md -- the graded
surface is untouched, and I re-ran the immutable command and the checker afterwards:
both still EXIT=0 / 113 green. Graded 86.96 artifacts carry no uncommitted change.

VERDICT RETURNED: PASS (see the structured return -- this file is not the verdict).

COMPLETED: 2026-08-17T13:02:47Z
