STATUS: COMPLETE -- write-first record, still NOT a verdict
STEP: 86.84
WRITTEN: 2026-08-14T17:38:24Z

# Q/A cycle 2 on step 86.84 (write-first record)

HEAD 04976701; work commit 85127353. qa_wip attempt_number=2 (source_present true,
identity_checked true). verdict_history_86_21 --evidence-only: no_rows_for_step.
attempt_number(2) > ledger count(0) => LEDGER IS STALE; sequence unreliable.
Known from the critique file itself: cycle 1 = CONDITIONAL.

## 1. Reproduction (deterministic)
- `python3 scripts/qa/rail_turn_cap.py --verify` -> EXIT 0 (bare run). 1.8s.
  VERIFY: PASS. 572 records / 1325 spawns / 0 missing.
- `node --check .claude/workflows/qa-verdict.js` -> 0
- `node --check .claude/workflows/research-gate.js` -> 0
- `node scripts/qa/verify_research_gate_workflow.mjs` -> EXIT 0, 124 passed 0 failed
  (reproduces Main's claim exactly)
- `node scripts/qa/verify_rail_retry.mjs` -> EXIT 0, 38 passed 0 failed. Section [F]
  is the executed drop-never-PASS test: F1 exhausted retry yields NO value, F2
  rethrows original error, F3 research-gate RECOMPUTES gate_passed, F4 retry loop
  assigns no verdict field. NOT cited in any 86.84 artifact (I found it myself).
- LINT GATE (diff touches *.py) -- scope derived from the COMMIT:
  `git diff --name-only 85127353^ 85127353 -- '*.py'` = rail_drop_rate.py, rail_turn_cap.py
  `uvx ruff check --select F821,F401,F811` -> EXIT 1
    F401 `collections.Counter` imported but unused -> scripts/qa/rail_turn_cap.py:112:25
  Pre-existing from c1797888 (also this step, cycle 1). Parent also had 0 uses.
- No frontend/** in the diff -> 1b N/A. No UI claim -> 1c N/A.
- Backend smoke 1d N/A (no backend/** in the diff).

## 2. The fix is applied (task item 1) -- CONFIRMED
YAML-parsed both frontmatters (yaml.safe_load on the block between the first two ---):
  qa.md        keys [color, description, effort, memory, model, name, permissionMode,
               skills, tools]   maxTurns present: False   model=opus effort=max
  researcher.md keys [color, description, effort, memory, model, name, permissionMode,
               tools]           maxTurns present: False   model=opus effort=max
Parent (85127353^) had maxTurns=30 / 40 respectively. qa.md BODY is byte-identical
(45398 == 45398). researcher.md lost only the pin line. No non-comment maxTurns in
either frontmatter.

## 3. Timeline honesty (task item 2)
- Corpus span: 2026-07-13T19:56:14Z .. 2026-08-14T15:22:51Z (566 runs w/ agent spawns).
- Spawns with record timestamp >= CAP_REMOVED_AT (2026-08-14T17:35:00Z): **0**.
  Straddlers (startTime < boundary <= timestamp): **0**. Rows with no timestamp: **0**.
  => the boundary misclassifies NOTHING in the current corpus.
- IS THE HARDCODE HONEST? Independently corroborated FROM GIT:
    git rev-list -1 --before=<d> main -- .claude/agents/qa.md, then parse frontmatter
    2026-06-01 qa=12 res=30 | 2026-06-12 qa=30 res=40 | 2026-07-01 qa=30 res=40
    2026-08-01 qa=30 res=40 | 2026-08-14T12:00 qa=30 res=40
  Over the ENTIRE corpus window the pins were 30/40. HISTORICAL_CAPS is CORRECT,
  and YES there is a git-derived alternative (demonstrated above) -- so the
  hardcode is a maintainability WARN, not an honesty finding.
- LATENT DEFECT (forward-looking): the boundary is the FILE-EDIT instant, but
  Main's own (correct) claim is that the removal is not in force until the next
  session's roster snapshot. In the window [17:35Z, next session start) a capped
  run is scored cap=None. SIMULATED: injected a synthetic qa spawn,
  timestamp 2026-08-14T18:00Z, turns=30, status=failed ->
  effective_cap("qa","2026-08-14T18:00:00Z") = None, and verify goes
  RED with "CLAIM BROKEN: a dropped spawn is not at its cap -- turn exhaustion is
  no longer necessary on every drop". Fails LOUD but MISDIAGNOSES: it indicts the
  diagnosis when the real cause would be the boundary definition.

## 4. MY OWN MUTATION MATRIX (task item 3) -- independent of Main's M4/M5
Technique: importlib.util load of rail_turn_cap.py into a fresh module per cell;
mod.REPO repointed at a tempfile.mkdtemp mirror of .claude/agents. ZERO writes to
the repo -- md5s after the whole matrix:
  947818e1d05040ff85c8b9b193d8c03b  scripts/qa/rail_turn_cap.py   (unchanged)
  4c9faa6d7eb14aba70eea2fc7f804727  .claude/agents/qa.md          (unchanged)
  a9592ee0950e55d24fc3e1bb65d5c26f  .claude/agents/researcher.md  (unchanged)
CONTROL observed GREEN FIRST: verify_ok=True, all_pins_removed=True.

  cell                                      result            live_caps
  M4r qa pin restored `maxTurns: 30`        KILLED (red)      {'qa': 30}
  M5r qa pin `maxTurns: 60`                 KILLED (red)      {'qa': 60}
  M9  researcher pin `maxTurns: 40` only    KILLED (red)      {'researcher': 40}
  M8  `maxTurns:30` (no space)              KILLED (red)      {'qa': 30}
  M7c `maxTurns : 30` (space before colon)  KILLED (red)      {'qa': 30}
  M7  `maxTurns: "30"` (quoted)             SURVIVED (green)  {'qa': None}
  M7b `maxTurns: 30  # restored`            SURVIVED (green)  {'qa': None}
  M6  qa.md ABSENT                          SURVIVED (green)  {'qa': None}
  M6b both agent files ABSENT               SURVIVED (green)  {both None}

SURVIVOR DIFFERENTIALS (a survivor is only a finding if it is behaviourally real):
- M7b is a REAL, NON-EQUIVALENT survivor. yaml.safe_load("maxTurns: 30  # restored")
  -> 30, type int. A LIVE integer pin of 30, and the guard reports
  "all pins removed: True". The regex is `^\s*maxTurns\s*:\s*(\d+)\s*$` --
  a trailing YAML comment defeats it. This is not exotic: EVERY other line of that
  frontmatter block is a `#` comment, so "restore the pin with a note" is the most
  likely shape a future editor would use. BLOCKING-adjacent vacuity.
- M7 `"30"` -> yaml gives str '30'. Whether Claude Code coerces a quoted scalar is
  NOT determined here; report as unresolved secondary, not a confirmed live pin.
- M6/M6b: absent-subject vacuity (the guard passes over a missing subject). Lower
  severity -- a vanished qa.md breaks loudly elsewhere. Same class as the project's
  oracle-with-silent-fallback lesson.

TIMELINE MUTATIONS (fresh collect() per cell -- my first pass reused a cached
DATA and every cell falsely SURVIVED; the cap is baked in at collect time, not
analyse time. Probe defect on my side, corrected):
  M11  CAP_REMOVED_AT -> 2026-01-01   KILLED  capped_n 395->0  "only 0 capped spawns"
  M11b CAP_REMOVED_AT -> 2026-08-01   KILLED  capped_n 68      "nothing to test"
  M12  HISTORICAL_CAPS qa=31          KILLED  "CLAIM BROKEN: a dropped spawn is not at its cap"
  M12b HISTORICAL_CAPS qa=29          KILLED  "C2 FAILED: 45 capped spawns exceed their cap"
  M13  HISTORICAL_CAPS researcher=41  KILLED  "CLAIM BROKEN"
  M14  CAP_REMOVED_AT -> 2027         SURVIVED (equivalent: whole corpus already
       precedes the boundary)
=> the hardcoded constants ARE load-bearing (+-1 kills the claim), and they are
   externally corroborated by git. Good.

## 5. Cycle-1 findings (task item 4) -- each verified
- F4 killed as third status: LANDED. collect() carries dropped/completed/killed
  explicitly; detector denominator 1277 -> **1267** (confirmed in --json).
  C3 now prints 10 spawns at turns [1,1,2,2,2,3,4,5,6,16], 0 at a cap.
- F5 at-cap non-emitters in completed runs: LANDED. at_cap_non_emitters=49, of
  which 2 in completed runs ['wf_078f4125-57a','wf_a6ea31e7-9b9'].
- THE 49-vs-50 DISPUTE: **Main is right, my predecessor was wrong.** 57 spawns sit
  at a cap (48 dropped + 9 completed-at-cap: qa ok@cap 6 + researcher ok@cap 3).
  Detector: 1 of the 48 dropped DID emit StructuredOutput -> 47 dropped
  non-emitters. 9 completed-at-cap, 7 emitted, 2 did not. 47 + 2 = **49**.
  Arithmetic checks out.
- NOTE-A 0/50 at-risk: LANDED, computed (uncapped_at_risk=50, drops 0,
  vs capped 12.2%), with an in-output instruction to quote the ratio not the raw total.
- F1/F2/F3: F1 and F2 landed in live_check + masterplan + day report. See finding
  V-3 below for what did NOT propagate.

## 6. Criterion 5 (task item 5) -- PARTIAL
- research-gate.js: CLEAN replacement. "appeared to split by MODEL" (past tense),
  then "THAT MODEL SPLIT IS CONFOUNDED AND THE MECHANISM IS NOW PROVEN", with the
  old UNPROVEN sentence quoted as history. Correct shape.
- qa-verdict.js: mostly clean, BUT :391 still asserts
  "P(0 drops in 73 | true rate 11.2%) = 2e-4 -- the model split SURVIVES."
  un-annotated, and the correction at :422 says "THE MODEL SPLIT REPORTED BELOW IS
  CONFOUNDED" when the split is reported ABOVE it (:385-387). A surviving
  contradictory copy inside one comment block -- exactly what criterion 5 forbids.
- rail_drop_rate.py: the correction is a PURE ADDITION (diff deletes nothing). The
  "mechanism is UNPROVEN" claim was never in this file (grep: absent), so that half
  is satisfied; but the model split at :14-16 survives verbatim, AND the script's
  RUNTIME OUTPUT still prints "=== by model === opus-5[1m] 11.7% / fable-5 3.0% /
  opus-4-8[1m] 0.0%" with NO caveat anywhere in the output. The confounded split is
  still freshly produced on every run.
- Both JS files pass node --check; verify_research_gate_workflow.mjs 124/0 green.

## 7. Is removing the cap right? (central judgment) -- YES, and the risk is
   adequately bounded. Measured independently:
  uncapped n=930: p50 9, p90 23, p95 32, p99 53, max 93.
  >30 turns: 50 | >40: 25 | >60: 5.
  tokens on the 49 >30-turn uncapped spawns with token data: median 172,297,
  MAX 259,390 (vs median 78,705 for <=30-turn uncapped).
  => the observed uncapped ceiling is ~93 turns / ~259K tokens. Not unbounded.
  qa COMPLETED turn histogram is near-FLAT from 8 to 30 (counts 8-22 per bucket,
  6 sitting exactly at 30) -- the textbook signature of right-censoring pressed
  against a cap, and independent corroboration of the diagnosis.
  The trade: pay up to ~2-3x tokens on the ~13% of evaluations that need >30 turns,
  instead of losing 100% of those tokens. Clearly favourable.
  CAVEAT Main does not state: "self-terminating" is EMPIRICAL, not structural --
  the 930 uncapped spawns are different ROLES with different workloads, so the
  qa/researcher uncapped distribution is still unobserved. Main's own
  re-measurement commitment is the right answer to this. Quoting "p50 7-12" in a
  cost argument is the weakest framing of the tail, but the max 93 IS quoted
  alongside it, so this is a NOTE not a finding.

## 8. Scope (task item 6) -- CLEAN
git show --stat 85127353: qa.md, researcher.md, qa-verdict.js, research-gate.js,
day_report, live_check_86.84.md, harness_log.md, rail_drop_rate.py, rail_turn_cap.py.
- Both JS diffs are COMMENT-ONLY (verified: every +/- line outside `//` = none).
- qa.md body byte-identical; researcher.md lost only the pin.
- No threshold, no gate, no verdict semantics touched.
- Peer-session files (backend/api/sovereign_api.py, frontend/src/*) are dirty in the
  working tree but are NOT in the commit. Not swept in.

## 9. Honesty about force (task item 7) -- CONFIRMED
harness_log Cycle 218 carries "NOT IN FORCE YET ... no behavioural claim about the
uncapped rail can be made from this session", plus an explicit OPERATOR REVIEW
REQUESTED header per CLAUDE.md separation-of-duties. Swept for forward claims
("no longer drop", "the rail is fixed", "since the removal", ...) across live_check,
contract, day report and both agent files: ZERO hits.

## 10. FINDINGS

V-1 (BLOCKING, Missing_Assumption) -- the step's own `verification.live_check`
    requires live_check_86.84.md to carry "the re-derived turn-cap measurement with
    its controls, THE EVIDENCE BEHIND THE CHOSEN REMEDY, AND THE MUTATION MATRIX".
    It carries the measurement + controls only. The remedy evidence lives in
    contract_86.84.md / day report; the mutation matrix exists NOWHERE in the repo
    (grep across handoff/, .claude/agents/, scripts/qa/ for the M4/M5 cells: 0 hits)
    -- only as three lines of commit-message prose. No control observation recorded,
    no per-cell record, no restore evidence.

V-2 (BLOCKING, Contradiction) -- live_check_86.84.md was NOT brought up to date
    with the remediation, and it is the artifact named by the criterion. As of the
    same commit that edited it:
      :239 §6 "No agent `.md` was edited. No cap was changed." -- FALSE.
      :83-84 "`.claude/agents/qa.md:6` -- `maxTurns: 30`" / researcher.md:6 -- FALSE.
      :206 "no remedy has been chosen or applied" + "still in flight at session
           freeze" -- FALSE.
      :223 "brief_status: INCOMPLETE, 7 sources read in full, gate NOT passed" --
           FALSE; the brief's own envelope is brief_status COMPLETE,
           external_sources_read_in_full 11, urls_collected 19, gate_passed TRUE.
      :44 and :172 detector "1257/1277" -- stale, now 1257/1267 (F4).
      :71 table row "researcher ... ok p50 23" -- does not reproduce; current is 24
           (F4 removed killed spawns from the ok population). A fenced block
           presented as the measurement that no longer reproduces.
      :112 killed turns "6/3/5/4/16/2/2/1/1" = 9 values; the script reports 10
           spawns [1,1,2,2,2,3,4,5,6,16].
      :207-219 the graceful-degradation RETRACTION is carried as authoritative,
           while contract_86.84.md:100-114 and the day report both say that
           retraction is "wrong in scope". Main knows and deferred it
           ("owed to live_check_86.84.md in the next cycle").

V-3 (Contradiction) -- the 49 correction did NOT propagate to the masterplan
    audit_basis, which still reads "the at-cap non-emitter population is 50 not 48",
    and still reads "1257/1277" twice. The masterplan was committed at 89efc7d8
    (17:20:46Z), before the 49 fix landed at 17:37Z. This is the artifact the
    cycle-1 Q/A specifically flagged as "the audit_basis a future executor is told
    to re-derive against".

V-4 (Contradiction, THE THIRD OVERCLAIM asked for) -- day_report:582-588 claims
    "each now carries the confound correction at source RATHER THAN A NOTE BESIDE
    IT". In rail_drop_rate.py the correction IS a note beside it (pure addition,
    nothing deleted, and the runtime output still prints the split uncaveated), and
    qa-verdict.js:391 still asserts "the model split SURVIVES". Also
    day_report:568 "it never flipped to COMPLETE (7 sources read in full, gate NOT
    passed)" -- contradicted by the brief's envelope AND by the same file's own
    later section "Research gate: PASSED ... 11 sources, gate_passed: true".
    Also day_report:582 points at "Session 4 below"; grep "Session 4" returns only
    that reference -- no such section exists.

V-5 (Circular_Reasoning / vacuity) -- the remediation guard is defeated by
    `maxTurns: 30  # restored` (M7b), a live int-30 YAML pin. Main's 2-cell matrix
    tested only the bare-pin shape and therefore reported 0 survivors; the matrix is
    too narrow rather than the guard being sound. Criterion 8 says "report survivors
    rather than dropping them" -- there are 4, one of them real.

V-6 (Invalid_Precondition) -- lint gate red on the step's own new file:
    F401 `collections.Counter` imported but unused, scripts/qa/rail_turn_cap.py:112:25,
    ruff exit 1. Pre-existing from c1797888 (cycle 1 of this same step).

V-7 (WARN) -- CAP_REMOVED_AT is the file-edit instant, not the first-session-after-
    restart instant, contradicting Main's own roster-snapshot claim. Zero runs
    affected today; simulated a post-boundary capped drop and it fails LOUD but with
    a message that indicts the diagnosis rather than the boundary.

V-8 (NOTE) -- HISTORICAL_CAPS is hardcoded where git can supply it (demonstrated);
    the values are correct across the whole corpus window, so this is
    maintainability, not honesty.

V-9 (NOTE) -- protocol: there is no `experiment_results_86.84.md`. The rolling
    handoff/current/experiment_results.md is phase-82.6 (2026-08-06). The GENERATE
    artifact of the five-file protocol is absent, and evaluator_critique_86.84.md
    carries no cycle-2 Follow-up section (§3 still reads "None of the above is fixed
    yet ... no fresh Q/A has been spawned").

V-10 (NOTE) -- criterion 4's executed evidence (verify_rail_retry.mjs section [F],
    38/38 green) is real and I ran it, but it is cited in NO 86.84 artifact. The
    property holds; the mapping to the criterion is missing.

## 11. CRITERION MAP
1 diagnosis re-derived, population rule beside ratios, disagreements reported: MET
  (I re-derived the table, the model cross-tab, the 49, the git cap timeline, the
  uncapped tail; Main reported and RESOLVED a disagreement with the cycle-1 Q/A's 50)
2 remedy from EVIDENCE incl. the NO answers, cited: MET (contract:29-56 -- no
  per-call turn budget in agent() opts; #20625 closed as not planned; absent maxTurns
  = "No limit"; and the uncapped-default-subagent option answered NO with
  qa-verdict.js:264-273 as the reason)
3 right-censoring addressed head-on: MET (removal instead of a number, plus a
  committed re-measurement of the uncensored distribution)
4 executed test that exhaustion yields NO VERDICT / never PASS / never gate_passed:
  MET ON THE PROPERTY, unmapped in the artifacts (V-10). verify_rail_retry.mjs [F]
  38/38 and verify_research_gate_workflow.mjs 124/0 both green, run by me.
5 model attribution corrected AT SOURCES, REPLACES not beside: PARTIALLY MET
  (research-gate.js clean; qa-verdict.js:391 surviving copy; rail_drop_rate.py
  runtime output uncaveated) -- V-4
6 recurrence answered + re-measurement trigger named: MET (removal has no number to
  outgrow; "RE-MEASURE once uncapped" named in qa.md + harness_log)
7 no gate loosened, no verdict semantics change, DEMONSTRATED: MET (comment-only JS
  diffs verified mechanically; 124/0 + 38/38 green; qa.md body byte-identical)
8 mutation-test every new guard, control green first, byte-identical restore, report
  survivors: NOT MET -- V-1 (no matrix on disk) + V-5 (4 survivors, 1 real)

CORRECTION: the line below first read COMPLETED: 2026-08-14T17:52:10Z, which I typed
without reading a clock -- the exact defect the cycle-1 critique s4 flagged in its
own predecessor. Replaced with a read value (date -u), measured 52s later.
COMPLETED: 2026-08-14T17:53:02Z
