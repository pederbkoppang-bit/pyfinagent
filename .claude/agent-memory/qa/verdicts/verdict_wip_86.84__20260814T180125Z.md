STATUS: COMPLETE -- write-first record, still NOT a verdict
STEP: 86.84
WRITTEN: 2026-08-14T18:01:25Z

Cycle 3. HEAD f57eb6e0 (24c57a47 + auto-changelog).
qa_wip attempt_number=3 (status ok, identity_checked true, prior_attempts 2).
verdict_history_86_21 --evidence-only: no_rows_for_step. CROSS-CHECK:
attempt_number 3 > ledger count 0 => LEDGER IS STALE; sequence from the ledger
is UNKNOWN. The two prior verdicts are transcribed verbatim in
handoff/current/evaluator_critique_86.84.md and both read CONDITIONAL.

## Deterministic reproduction (all exit codes captured bare, no pipes)
- rail_turn_cap.py --verify: exit 0 under .venv/bin/python AND /usr/bin/python3
- mutate_rail_turn_cap.py --verify: exit 0 under both; 15 cells, control green
  first, 3 labelled survivors; md5 of qa.md/researcher.md/rail_turn_cap.py
  unchanged (I md5'd independently outside the harness)
- node scripts/qa/verify_rail_retry.mjs exit 0 (38/0); [F] section green
- node scripts/qa/verify_research_gate_workflow.mjs exit 0 (124/0)
- ruff F821,F401,F811 over a git-derived scope (3 .py files, non-empty asserted): 0
- rail_drop_rate.py RUNTIME prints the confound caveat under the by-model table

## Independent re-derivation -- REPRODUCES
table, controls, 1257/1267 vs 1/48, killed [1,1,2,2,2,3,4,5,6,16] in 6 killed
RUNS, 0/50 at-risk, 57 at-cap / 49 non-emitters / 2 in completed runs,
qa.md body 45,398 chars identical, dropped tail 48/48 tool_result with last
tool Bash 37 / Edit 4 / Write 2 / Read 2 / WebFetch 1 / WebSearch 1 +
StructuredOutput 1 (wf_d4e2e794-567), uncapped turns max 93 p50 9 p90 23 p95 32
p99 53, criteria byte-identical across c1797888..HEAD (never amended),
JS diffs 0 non-comment lines, no peer-session files in either commit.

## FINDINGS
- F-A (Contradiction) live_check §0 + masterplan audit_basis: "347 of 347
  completed qa/researcher spawns end on a tool_result". MEASURED:
  status==completed => 343, all 343 end on tool_result. The 347 population is
  "not dropped", which includes 4 spawns in killed runs (wf_4c70d707-88e,
  wf_0471dd22-909, wf_d63b872f-e33, wf_8375665b-f5a) and those 4 do NOT end on
  a tool_result. False under either population. Same F4 class the step fixed in
  code; second correction of this figure (393/394 -> 347/347) still wrong.
- F-B (Overgeneralization) live_check §6: "observed uncapped ceiling ~93 turns /
  ~259K tokens". MEASURED: 259,390 is the p99 of uncapped spawn tokens; ceiling
  is 416,614; the 93-turn spawn used 217,384. Understates the disclosed cost
  tail 1.6x inside the residual-risk paragraph.
- F-C (Missing_Assumption, WARN) criterion 8 incomplete + a live vacuity.
  MEASURED by in-memory mutation of rail_turn_cap.py:
    C1 count_turns->0            => C1 FAILED fires (live)
    C3 killed bucket relabelled  => C3 FAILED fires (live)
    detector -> always False     => detector control fires (live)
    detector -> always True      => VERIFY STAYS GREEN  <-- vacuous direction
  The detector control compares raw COUNTS (1267 vs 48) not rates, so a detector
  stuck True passes and silently zeroes at_cap_non_emitters (the 49). No cell in
  the shipped matrix targets C1, C2, C3 or the detector; 4 of 8 new guard
  families unmutated, and the untested set contained the hole.
- F-D (WARN) parse_cap strength is interpreter-dependent and undisclosed.
  /usr/bin/python3 (what bare `python3` resolves to here) has NO PyYAML, so the
  shipped command takes the regex fallback. On that path !!int 30, &anchor 30,
  *alias, 0x1e read as UNCAPPED. Both realistic restore shapes survive-proof on
  both paths; both commands exit 0 on both interpreters.
- F-E (WARN) V-7 residual, SIMULATED not reasoned: effective_cap('qa',
  2026-08-15T01:30:00Z) = None, and injecting one such drop makes --verify exit
  1 with "CLAIM BROKEN ... the diagnosis must be revisited". A spawn of THIS
  session after 00:00Z is still capped by the roster snapshot, so the window is
  open tonight. Disclosed in-code with a named follow-up.

## HARNESS ATTACK (Main's question: can a cell report KILLED without the guard
## doing the work?) -- NO, proven by neutering:
  A1 remediation branch -> `if False`  => the 7 pin cells become REAL SURVIVORS
  A2 verify() always green             => 12 real survivors, rc=1
  A3 verify() always red               => control RED, matrix aborts, 0 scored
  MX a cell writing the real agents dir => "CHANGED ... modified the real tree",
     rc=1 -- the md5 restore proof is live, not decorative.

## PARSE_CAP PROBE (14 shapes, PyYAML path): every LIVE pin detected
  0 / !!int 30 / &anchor / *alias / dup-key / tab / hex 0x1e / CRLF+pin => KILLED
  30.0 and "thirty" => sentinel -1 => KILLED (redder, safe direction)
  nested key / trailing second --- block / last-wins-null dup => uncapped, and
  none of those is a live pin under the loader either, so correct not evasive.

CRITERION MAP: 1 PARTIAL (F-A, F-B) | 2 MET | 3 MET | 4 MET | 5 MET | 6 MET |
7 MET | 8 SUBSTANTIALLY MET, not complete (F-C).

Verdict returned: CONDITIONAL.

CORRECTION: this line first read COMPLETED: 2026-08-14T18:44:10Z, typed without
reading a clock. Replaced with a value read from `date -u`.
COMPLETED: 2026-08-14T18:19:32Z
