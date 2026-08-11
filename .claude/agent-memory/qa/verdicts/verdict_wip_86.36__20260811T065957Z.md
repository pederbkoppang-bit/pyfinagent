STATUS: COMPLETE -- write-first record, still NOT a verdict
STEP: 86.36
WRITTEN: 2026-08-11T06:59:57Z
COMPLETED: 2026-08-11T07:09:59Z

# Q/A write-first record -- step 86.36 (WIP, NOT a verdict)

Spawn: Workflow rail, evaluator independent of Main.
Filename carries the phase-86.36 STAMP per qa.md lines 112-129 (stamped-name
directive). NOTE: my spawn prompt's STEP 0b told me to use the FIXED name
`verdict_wip_86.36.md`; qa.md on disk (the binding source of truth per STEP 0)
mandates the stamped form. I followed qa.md. This is itself evidence for
disclosed weak-point (3).

## Live observation before any of my own writes (ls of verdicts/ at 06:59:57Z)
    6286  10 aug 12:51  verdict_wip_86.24.md
    8543  11 aug 08:39  verdict_wip_86.25.md
     528  11 aug 08:59  verdict_wip_86.29__20260811T065922Z.md   <-- STAMPED, 35s old
   11479  11 aug 08:48  verdict_wip_86.29.md
    8415  10 aug 17:11  verdict_wip_86.30.md
   13751  11 aug 08:32  verdict_wip_86.31.md
   10346  11 aug 08:48  verdict_wip_86.34.md
    8653  10 aug 17:59  verdict_wip_86.37.md

FINDING (pre-emptive, weak-point 3): a REAL, independent Q/A spawn (peer session,
step 86.29) HAS produced a stamped record at 06:59:22Z -- 35 seconds before my
own first write. The stamped-name directive demonstrably reaches live spawns.
Also note 86.29 now has BOTH a legacy-named record (11479 B, 08:48) and a
stamped one (528 B, 08:59) COEXISTING -- exactly the property criterion 2
demands, observed in production rather than in a scratch sink.

## A. HARNESS COMPLIANCE -- CLEAN (all 5)
mtimes (UTC+2 local as stat prints them, monotonic order is what matters):
  research_brief_86.36.md   08:45:31
  contract_86.36.md         08:48:27   <- research < contract  OK
  qa_wip.py                 08:53:25   <- contract < artifact  OK
  qa.md                     08:52:24   <- contract < artifact  OK
  verify_wip_retention      08:54:25
  mutation_matrix           08:55:31
  experiment_results        08:57:16
  live_check                08:58:11
1. research-gate-before-contract  MET (brief exists, precedes contract; gate wf_5dc835fa-de3)
2. contract-before-generate       MET (contract 08:48 precedes every generated artifact)
3. experiment_results present     MET
4. log-last                       MET -- `grep -cE "phase=86\.36 " handoff/harness_log.md` = 0,
                                  masterplan status = "pending". Not yet logged/flipped.
5. no-verdict-shopping            N/A -- first Q/A cycle for 86.36 (0 result= lines).
3rd-CONDITIONAL rule: 0 prior CONDITIONALs for this step id -> does NOT bind.

## B. DETERMINISTIC
B1. IMMUTABLE COMMAND
    bash -c 'source .venv/bin/activate && python scripts/qa/verify_qa_write_first_86_31.py'
    -> EXIT 0.  "ALL GREEN -- 197 passed, 0 failed"
    Section [9] of that run picked up MY OWN file:
      "live artifact verdict_wip_86.36__20260811T065957Z.md carries a valid marker
       status=INCOMPLETE written=2026-08-11T06:59:57Z bytes=1965"
    -> criterion 3's phase-86.31 assertions all still pass; signature change is
       backward compatible (run_stamp defaults to None -> legacy path).

B2. CRITERION 5 -- GUARD UNCHANGED
    git diff HEAD -- .claude/hooks/qa-write-guard.sh          => 0 bytes
    git diff 5595055c^ HEAD -- .claude/hooks/qa-write-guard.sh => 0 bytes
    last commit touching it = d23a981e (phase-86.31), PREDATES this step.
    -> UNCHANGED, and the 86.31 role predicate re-proved green by B1
       (DENY x 18 classes x 6 qa-role spawn names; ALLOW for Main + researcher).

B3. UNINTENDED PRODUCTION CHANGES: none. git status shows only
    researcher MEMORY.md + audit jsonl (hook-appended) + 3 untracked memory
    files (mine, the peer's 86.29 stamped record, a researcher memory).
    Commit 5595055c = exactly 7 files, explicit pathspecs.

## C. CRITERION EVIDENCE (re-run by me, not pasted)
C1 REPRODUCE -- MET. reproduce_wip_destruction_86_36.py exit 0.
   same path both spawns; 4386 -> 124 bytes, LOST 4262;
   "spawn 1's analysis still recoverable : False". Simulation, and it SAYS so
   (scratch tempdir; the real sink is being written by a live peer).
C2 COEXISTENCE -- MET. verify_wip_retention exit 0.
   verdict_wip_86.36__20260811T060000Z.md = 965B  and
   verdict_wip_86.36__20260811T064000Z.md = 110B  both on disk, DISTINCT paths,
   each with its own WRITTEN; cycle1 has COMPLETED, cycle2 does not.
   audit_memory.py before/after a stamped probe: exit 1 vs 1, output IDENTICAL.
   NOTE: auditor baseline rc=1 (pre-existing red) -- the asserted property is
   UNCHANGED, which is right, but see positive-control check below.
   *** REAL-WORLD CONFIRMATION, not a fixture: the live sink right now holds
   verdict_wip_86.29.md (11477B, COMPLETE) AND
   verdict_wip_86.29__20260811T065922Z.md (528B, INCOMPLETE) side by side. ***
C3 RESOLUTION + STALENESS -- MET. spawn1->cycle1 record, spawn2->cycle2 record,
   cycle1 listed as prior_records (not merged); future spawn -> STALE;
   unparseable spawned_at -> IDENTITY_UNKNOWN. Plus B1 section [5] green.
C4 NEVER A VERDICT -- MET (asserted, not stated): 3 reports incl. the newest,
   "verdict" not in rep AND is_verdict is False, with a >=3 cardinality floor.
C5 GUARD -- MET (see B2).
C6 MUTATION -- in progress below.

## Attacks
(3) REFUTED. The stamped path HAS now been written by a real Q/A: the peer's
    86.29 record at 06:59:22Z and mine at 06:59:57Z. qa.md's directive reached
    me -- I read lines 112-129 and followed the STAMPED form even though my own
    spawn prompt's STEP 0b still told me the FIXED name. Disclosure was honest
    but is already stale in the step's favour.
(4) CONFIRMED as disclosed: grep for prune_wip_records over the repo returns
    only qa_wip.py (def), verify_wip_retention (2 test calls), mutation_matrix
    (anchor strings), and the two handoff docs that DISCLOSE it. No production
    caller. Retention is a proven function property, unbounded in practice.
(1) NOT VACUOUS -- proven by execution, three ways. I loaded the matrix module
    from its REAL path (avoiding the /tmp REPO=parents[2] trap Main hit) and
    swapped MUTATIONS:
      P1-MISATTRIB-PROBE  MIS-ATTRIB   (genuinely red, expected name never appears)
      P2-STALE-ANCHOR     ANCHOR       (0 matches refused, not silently scored)
      P3-DEFAULT-KEEP     SURVIVED     (so SURVIVED is reachable too)
    KILL-ATTRIBUTION AUDIT (vacuity shape 11): for all 5 shipped cells I printed
    the REAL failing lines; the credited name is among them in 5/5. No cell is
    credited to an assertion that did not fire.
(2) SEAM IS SOUND. PYFIN_QA_WIP_OVERRIDE lives in the CHECKER, not in the
    production subject; qa_wip.py has no seam. Proven isolated: running the
    IMMUTABLE command with PYFIN_QA_WIP_OVERRIDE=/nonexistent/evil.py still
    yields "ALL GREEN -- 197 passed, 0 failed". The gate cannot be bent by it.
    Residual (theoretical): pointing the var at a pristine copy would green the
    86.36 checker; no CI/production path sets it, and the matrix digests the
    tracked subject before/after (unchanged, da6db96dddb9b9fc).
(5) BACKWARD COMPAT HOLDS. All 7 legacy-named records resolve. 86.29 shows the
    new behaviour working in production: path=stamped(528B), priors=[legacy],
    retained=2 -- the legacy record is LISTED, not destroyed.
POSITIVE CONTROL on the audit_memory leg (my own, on a temp COPY of the corpus):
    baseline rc=1 13007 chars; verdicts/ probe -> IDENTICAL (invisible);
    SAME file at TOP level -> 13164 chars, adds
      "- NO POINTER: verdict_wip_99.99__...md exists but MEMORY.md never links it"
      "- MALFORMED FRONTMATTER: verdict_wip_99.99__...md"
    => the auditor DOES scan .claude/agent-memory/qa (_KNOWN_CORPORA line 62)
       and the non-recursive root.glob("*.md") is what hides verdicts/.
       The [2c] assertion is therefore NOT vacuous. (My first control was
       broken -- passed a positional instead of --dir, rc=2. Probe fixed.)

## LINT GATES
RUFF (scope DERIVED: git diff --name-only 5595055c^ 5595055c -- '*.py', 4 files):
  F401 [*] `re` imported but unused
    --> scripts/qa/mutation_matrix_86_36.py:25:8
  Found 1 error.   RUFF_STEP_EXIT=1     <-- IN-SCOPE GATE FAILURE (qa.md 1a)
  Confirmed dead: no `re.` usage in the file. It is a fossil of the two earlier
  regex-based matchers that were both wrong -- residue of the disclosed defect.
ESLINT (qa.md 1b fires because the diff touches .claude/agents/qa.md):
  26 errors, grouped by dir = {'.next-audit-36-12': 13, '.next-functional': 13}
  ZERO in src/. Pre-existing build-dir noise; this step touches no frontend file.
  NOT attributed to 86.36.

## *** BLOCKING FINDING -- THE PRIMARY LAUNCH PATH STILL TEACHES THE OLD NAME ***
.claude/workflows/qa-verdict.js STEP 0b (lines 96-100) was NOT updated by this
step (`git diff --name-only 5595055c^ 5595055c | grep -c qa-verdict.js` = 0) and
still injects, verbatim, into every Workflow-rail Q/A prompt:
    'create .claude/agent-memory/qa/verdicts/verdict_wip_' + stepId + '.md ...'
    '... The stamp is load-bearing: the path is FIXED'
    'per step, so a later cycle that drops before its first write would ...'
grep for YYYYMMDDTHHMMSSZ|run_stamp|stamped in that file = 0 hits.
So the FIRST-CLASS launch path (CLAUDE.md + qa.md both designate it primary)
tells the Q/A to write the DESTRUCTIVE fixed name and asserts a premise this very
step falsifies. A Q/A obeying the more proximate, explicitly-"binding" STEP 0b
reintroduces exactly the destruction 86.36 removes.
NOT caught by any guard: the 86.31 checker's section [6] anchors for qa-verdict.js
("carries 'verdict_wip_'", /create \.claude/agent-memory/qa/verdicts//) all PASS
on the stale text, so section [6] is blind to this divergence.
Mitigating: I resolved the conflict in qa.md's favour (STEP 0 designates qa.md the
single source of truth) and so did the peer's 86.29 Q/A at 06:59:22Z -- 2 of 2
live spawns used the stamped name. So it is a latent trap, not an active outage.

## NOTE-level (do not degrade the verdict)
N1 "195 passed" in the block labelled "## 5. Verbatim" does NOT reproduce; I get
   197. FULLY EXPLAINED, not fabricated: section [9] emits one PASS per live
   artifact; there were 7 at experiment_results mtime (06:57:16Z UTC) and 9 now
   (peer's 06:59:22Z + mine 06:59:57Z). 195+2=197. The count is non-deterministic
   by construction and that is not disclosed beside the number.
N2 REFUTED IN THE STEP'S FAVOUR: experiment_results section 6 and live_check
   section G both say "the stamped path has never been written by a REAL Q/A /
   every record on disk today is legacy-named". True when written; false 2 min
   later. My own record IS the first-party proof, and the immutable command's
   section [9] independently picked it up:
     "live artifact verdict_wip_86.36__20260811T065957Z.md carries a valid marker
      status=INCOMPLETE written=2026-08-11T06:59:57Z bytes=1965"
N3 SURVIVING MUTANT (mine): DEFAULT_KEEP 3 -> 1 SURVIVES -- no assertion pins the
   default, because every call passes keep= explicitly. Behavioural differential:
   NONE within the reachable call graph, since prune_wip_records has zero
   production callers (attack 4 confirmed by my own repo-wide grep). Doubly dead.
N4 3 of the 5 "Verbatim" commands (`$ python scripts/qa/...`) do not run as
   written: bare `python` is not on PATH (exit 127). Only the immutable line
   carries `source .venv/bin/activate`.
N5 live_check section F's 86.29 row (COMPLETE/11479/retained=1) now reads
   INCOMPLETE/528/retained=2 -- the peer wrote a newer record after the capture.
   Explained, not a defect; it is the new feature working.
N6 Real destruction instances corroborated FROM GIT by me, not taken on trust:
   `git show 5285699b:...verdict_wip_86.30.md | wc -c` = 7380 (matches the
   masterplan's cited figure); 86.34 at 630fa95b = 796 bytes (matches the cited
   "-> 796"). The 4,921 start is absent from git precisely because it was
   destroyed before any commit -- which is the defect itself.

## RESEARCH GATE
contract_86.36.md:16-21 cites wf_5dc835fa-de3, tier moderate, brief 24,395 chars,
10 sources read in full (floor 5), 18 URLs (floor 10), gate_passed RECOMPUTED by
the script rather than self-reported. COMPLIANT.

## VERDICT REACHED (the return value is the deliverable, not this file)
CONDITIONAL. All 6 immutable criteria MET with executed evidence; harness
compliance clean; no unintended production change; guard byte-unchanged.
Two fixable blockers: (B1) qa-verdict.js STEP 0b still teaches the destructive
fixed name + a false premise on the primary launch path, unguarded; (B2) ruff
exits 1 on an in-scope file introduced by this step.
3rd-CONDITIONAL rule: 0 prior result= entries for 86.36, so this is the FIRST --
the rule does not bind and CONDITIONAL is permitted.
