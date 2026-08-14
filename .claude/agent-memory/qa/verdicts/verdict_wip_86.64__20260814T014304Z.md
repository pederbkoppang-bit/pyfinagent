STATUS: COMPLETE -- write-first record, still NOT a verdict
STEP: 86.64
WRITTEN: 2026-08-14T01:43:04Z

# Q/A write-first record -- step 86.64 (attempt 1)

Subject: `.claude/hooks/qa-write-guard.sh` header rewrite + `.claude/settings.json`
statusMessage rewrite. Commit 30b1b08a. Claimed outcome: the criterion-4 licensed exit.

## Counters
- `qa_wip.py 86.64` -> records_retained=1 (that is MY OWN record), prior_records=[],
  source_present=TRUE. => **ATTEMPT 1**.
- `verdict_history_86_21.py --step 86.64` -> status=no_rows_for_step, verdicts=(none),
  consecutive=0, auto-FAIL NOT armed. Two sources AGREE at 0 prior; no staleness flag.
- F1b: attempt 1 of 5. No escalation.
- masterplan: 86.64 status=pending (NOT flipped), retry_count/max_retries absent.

## Harness compliance (5 items) -- ALL PASS
1. research gate: brief_status COMPLETE, gate_passed true, 9 read-in-full (floor 5), 39
   urls_collected (I independently counted 39 distinct http URLs -- agrees), recency scan
   at :125. Contract cites run wf_bb618099-661.
2. contract-before-generate (UTC mtimes): brief 23:11:33 < contract 23:33:01 < guard
   00:53:59 < settings 00:54:31 < experiment_results 00:55:11 < live_check 00:55:38.
3. experiment_results_86.64.md + live_check_86.64.md both present.
4. log-last: not flipped, no harness_log result row.
5. no-verdict-shopping: attempt 1. N/A.

## Deterministic -- ALL REPRODUCE
- IMMUTABLE `bash -c 'bash -n .claude/hooks/qa-write-guard.sh && echo guard-parses'`
  -> `guard-parses`, **exit 0**.
- Worktree guard == committed after-version (cmp, byte-identical). md5 3eb5acfea3ec356ac6b289011f49912d.
- SCOPE derived from git: commit touches exactly 4 files -- the guard, settings.json, and
  the 2 new handoff artifacts. ZERO .py, ZERO frontend/**, ZERO backend/** -> qa.md gates
  1a/1b/1d N/A (derivation printed, not assumed). No production or trade-path file.
  `git status` otherwise clean apart from the audit jsonl and my own WIP file.
- settings.json: `jq -e` valid; qa-write-guard matcher still `Write|Edit`; effortLevel
  still `max`; 8 hook events present. Round-trip claim HOLDS.
- "0 executable (non-comment, non-blank) lines changed": 79 before, 79 after, diff EMPTY.

## C5 six cells -- RE-DRIVEN BY ME: 2,0,0,0,2,0 -- EXACT match to the claim.
## C3 nine shapes -- RE-DRIVEN BY ME: all nine exit codes match the claim.

## C3: I supplied the proof the criterion stipulates, which the artifact omits
Criterion 3 requires the property "proven by making the guard's own helper RAISE". The
artifact's three cells are malformed JSON (a HANDLED except branch), empty payload (not an
error at all -- normal allow path) and python3-absent (helper never runs). None is a raise.
I forced three genuine uncaught raises:
  agent_type=5        -> AttributeError 'int' has no attribute 'strip'   -> exit 0
  file_path=123       -> AttributeError 'int' has no attribute 'replace' -> exit 0
  file_path=["a","b"] -> AttributeError 'list' ... 'replace'             -> exit 0
Tracebacks OBSERVED in the guard log. PROPERTY HOLDS (falls to `case *) exit 0`).
The helper's try/except wraps ONLY the json parse; everything after is unprotected and
relies on the bash default -- a path the artifact's evidence never touched.
NOTE: the real log shows Main DID drive `agent_type: 7` at 00:52:59.888 and it raised, so
the work happened; the write-up just does not report it.

## C1: CORROBORATED, and by a stronger route than the artifact used
- Log-aliveness POSITIVE CONTROL: the last row of the real guard log was MY OWN WIP Write
  (01:43:14Z, agent_type='qa', agent_id a432021376adb291e); 40 rows in the last hour. The
  zero delta is NOT a dead log.
- The 10293 figure CORROBORATES against the log's own line/timestamp map: file line 10293
  carries ts 00:50:43Z and the next row is 00:52:40Z, so the BEFORE/AFTER bracket around
  the real Bash write is internally consistent; Main's synthetic A/B rows land after it.
- KNOWN-MEMBER RECALL from a source Main did not curate: over the guard-log window
  (2026-07-24T11:57Z -> 2026-08-14T01:43Z) the platform PreToolUse audit stream recorded
  **26,934 Bash tool calls**. Guard rows with a REAL platform key set (>=10 keys incl.
  session_id/transcript_path) = 260, and they are **Write 120 / Edit 140 / Bash 0**.
  Recall = **0 / 26,934**.
- TRAP AVOIDED: the guard log DOES hold 309 tool_name=Bash rows. All 309 are SYNTHETIC
  drives (5-key/0-key signatures; an 18-each identity sweep). Zero carry a real key set.
  The contract's F7 flagged this at 307; Main correctly did not cite them.
- STRONGEST FORM, performed live BY ME: I am agent_type='qa' per the guard's own log. My
  Write tool calls produced one guard row each (01:43:14, 01:51:13). My Bash tool calls
  that created files outside the memory dir at 01:44:21 and 01:51:28 produced **ZERO**
  guard rows.

## FINDINGS

**F-1 (BLOCKING for C4) MIS-ATTRIBUTED MECHANISM -- confirmed by executed mutation.**
The guard header credits the A/B to the matcher: "the settings.json matcher is Write|Edit,
so a write issued through Bash is never routed here at all. Measured: ... exit 2 via
tool_name=Write and exit 0 via tool_name=Bash". The A/B is a PIPED-PAYLOAD drive -- the
matcher is not in its causal path at all. The exit 0 comes from `:148`
`if is_qa_role(agent_type) and tool_name in ("Write", "Edit"):`.
MUTATION (scratchpad copy only; repo md5 unchanged): control on the unmutated copy with
qa+Bash -> exit 0 GREEN; widen ONLY `:148` to `("Write","Edit","Bash")`, matcher untouched
-> **exit 2**; byte-identical restore. So `:148` is the operative gate for the Bash path.
Consequence: the corrected description -- which IS this step's deliverable -- would send a
maintainer to widen the matcher and believe the accidental-breach channel closed, when
`:148` would still allow Bash. The contract named this in advance (plan item 1, "the
doubly-blind trap ... a test that only fixes the matcher will still show 'no fire' for the
wrong reason"). Neither artifact mentions `:148`.

**F-2 (BLOCKING for C4) the contract-mandated DECIDABILITY basis is absent.**
The contract states in bold that criterion 4's conclusion "may **not** rest on 'the control
cannot be made sound because Bash is uninterceptable' ... Any honest conclusion must rest
on **DECIDABILITY**", and supplies both sides (CARE 85.64% F1; CVE-2025-66032 $IFS bypass).
grep over experiment_results / live_check / the guard header / settings.json: no
"decidab", no CARE, no IFS, no CWE-693, no "Protection Mechanism Ignored". Shipped C4 rests
on (1) Channel -- restated as the matcher -- and (2) Identity. Leg (1) is the one the
contract forbade, restated in configuration terms: a matcher is CHANGEABLE, so it is not a
reason the control cannot be MADE sound. Leg (2) IS sufficient on its own, so the
CONCLUSION survives; the stated REASONING does not.

**F-3 (WARN) the corrected header over-generalizes on the very point it corrects.**
Header: "PATH INDETERMINATE on a qa Write/Edit -> FAIL-CLOSED, exit 2." The five ENUMERATED
shapes are all correct (I reproduced each at exit 2). But a TRUTHY non-string file_path is
also path-indeterminate and fails **OPEN**: file_path=123 and file_path=["a","b"] both
exit 0, because `.replace` raises before the containment check. Real boundary: falsy/absent
-> "" -> deny; truthy non-string -> raise -> allow.

**F-4 (WARN) `CLAUDE_AGENT_TYPE=qa-86-64-c1` in the live_check is INERT.**
The hook reads agent_type from the stdin JSON payload only; it never reads the environment.
So the artifact's real-Bash leg was issued by Main (agent_type='') with a decorative env
var, and no single measurement in the artifact satisfies C1's conjunction (Bash channel AND
qa-shaped agent_type in one drive). I closed that gap live (see C1 above), so C1 is met on
the evidence available to me -- but not by the artifact standing alone.

**F-5 (NOTE) NotebookEdit not queued.** Disclosed in three places, correctly not fixed
(no criterion owns it; the constraint forbids weakening/changing guard behaviour here). I
found no masterplan step queued for it. Should be queued together with the `:148` gate.

## Criterion mapping
- C1 MET (artifact + my independent corroboration; see F-4 for the artifact-alone caveat)
- C2 MET (enumeration sound; my census over 98 platform tool names found no additional
  local-FS writer; table header correctly scoped to the matcher; NotebookEdit found and
  disclosed -- a tool that appears in ZERO of 177,938 audit records, which is itself
  evidence the list was read rather than guessed)
- C3 MET IN SUBSTANCE, evidence-reporting gap (see F-3 and the raise proof above)
- C4 **NOT MET as delivered** (F-1 + F-2; outcome legitimate, description defective)
- C5 MET / correctly N/A (no guard added; 0 exec lines VERIFIED; six cells reproduce; I
  additionally ran a real mutation with green control and byte-identical restore)

## Answer to "did I take the easy exit?"
No. The measurement work is real and better than most, and the honesty (self-corrected
predictions, unprompted NotebookEdit disclosure, no C5 overclaim) is genuine. But criterion
4's deliverable IS the text, and the text credits the wrong mechanism and omits the basis
the contract said the conclusion must rest on. Fixable in two text blocks.

COMPLETED: 2026-08-14T01:55:41Z
