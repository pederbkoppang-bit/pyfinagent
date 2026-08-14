# live_check — step 86.81

Evidence captured from the live rail and from re-runnable checkers, 2026-08-14.

---

## 1. THE LIVE FORCED DROP — the retry firing on the real Workflow rail

The stochastic drop cannot be summoned on demand, so the fault was **injected by
instruction**: the probe agent reads a marker file, and on the `FIRST` branch it flips
the marker and ends its turn **without emitting the schema call**. Attempt 2 reads the
flipped marker and emits normally. The probe carries `agentRetryingDrops` **verbatim**
out of the shipped `.claude/workflows/qa-verdict.js` (brace-matched, sha256 of the span
stamped into the probe), and was launched by `scriptPath` from a temp dir — never from
`.claude/workflows/`, which is a dispatch directory.

Generator: `scripts/qa/gen_live_retry_probe.mjs` (re-runnable).

### Run `wf_9f387ad8-b5c` — run record, read back from disk

```
marker on disk: SECOND
status      : completed
agentCount  : 2
LOGS ARRAY  :
    'qa-verdict: StructuredOutput DROP on attempt 1/2 -- retrying'
error       : None
retry present in dispatched script: True

--- per-agent transcripts in this run ---
  agent-a093d72a017ef0fe2.js  emitted_StructuredOutput=True
  agent-a4e6793aee0f111c4.js  emitted_StructuredOutput=False
```

Workflow return value:

```json
{"probe":"live-retry-86.81",
 "retry_span_sha256":"1366d49acf843666f8cac718d289c6b4303f55700c54219ce312b5de75bde974",
 "result":{"attempt_seen":"SECOND","marker_value_read":"SECOND"},
 "error":null}
```
`agents_done: 2`, **`agents_empty_result: 1`** — one agent returned nothing (the drop),
one returned the value (the recovery).

### Trigger verification — the run is a genuine recovery, not a compliant agent

AgentChaos §4.4 requires that runs where the fault never fired be **filtered out, not
scored**. All four independent facts hold:

| requirement | observed |
|---|---|
| the drop branch actually executed | marker on disk reads `SECOND` |
| the retry actually caught a drop | run record logs `StructuredOutput DROP on attempt 1/2 -- retrying` |
| two spawns for one logical call | `agentCount: 2`, one transcript with no `StructuredOutput` tool_use |
| the run recovered rather than exhausting | `status: completed`, `error: None`, value returned on attempt 2 |

### The FIRST live attempt was INVALID and is reported, not hidden

Run `wf_ba771f51-1f7` returned `{"attempt_seen":"FIRST"}` with `agentCount: 1`. The
agent flipped the marker and then **called StructuredOutput anyway**, ignoring the drop
instruction:

```
  TOOL Read: {"file_path": ".../attempt_marker.txt"}
  TOOL Bash: printf 'SECOND\n' > .../attempt_marker.txt
  TEXT[assistant]: Marker updated.
  TOOL StructuredOutput: {"attempt_seen": "FIRST", "marker_value_read": "FIRST"}
```

No drop, no retry. The prompt was then sharpened to state that non-emission is the
**correct** outcome on that branch and that the runtime's nudge is part of the test.

**A defect in my own probe, found by this:** the probe returned `fault_fired: false` on
the SECOND run too — on which the fault demonstrably *had* fired. The extracted retry
span closes over the **runtime's** global `log`, so the recording wrapper the probe
installed was never called. A probe returning a clean answer it had no way to dirty is
the `suspect-the-clean-check` class. The misleading fields were removed from the
generator; the authoritative channel is the run record's own `logs` array, quoted above.

---

## 2. THE METRIC — contamination shown RED, then GREEN, on the SAME fixture

Three synthetic runs. Ground truth: **R1** carries the real shipped `qa-verdict.js` in
its `script` field (which contains the retry log literal exactly once) and **empty
logs** — it never retried. **R2** retried once. **R5** failed with a `TypeError`, which
is not a drop.

```
--- PRE-FIX reader (commit f88f8190) ---
    EXHAUSTED (lost runs) :    1   33.3%
    RETRIED   (recovered) :    4   <- runs the pre-fix code would have LOST

--- CORRECTED reader (this step) ---
    EXHAUSTED (lost runs) : 0
    RETRIED   (recovered) : 1

Truth: retried=1 (R2 only), exhausted=0 (R5's error is not a drop).
```

The pre-fix reader reports **4 retries where 1 occurred** — it counted the drop literal
embedded in each record's `script` field, i.e. its own subject matter — and classifies
an unrelated `TypeError` as a rail drop. This is the same self-match that produced 38
phantom drops out of 81 and was corrected in `f88f8190` **in the `exhausted` predicate
only**; the `RETRIED` counter kept scanning the blob.

### The before/after split, on the real corpus

```
=== BEFORE vs AFTER the retry (launch instant 2026-08-14T10:15:17Z, commit 6b4df8f9) ===
  before    runs= 564  exhausted= 44 (7.8%)  retried=0
  on/after  runs=   3  exhausted=  0 (0.0%)  retried=0

  NOTE: only 3 run(s) have LAUNCHED since the fix -- too few to call a rate.
```

The shipped reader put **every run of that whole day** in the "after" bucket. Split on
the launch instant, the true post-fix population is **3**, and all three are
`research-gate` runs.

---

## 3. MUTATION MATRIX — control GREEN first, 6/6 killed, subject unmodified

`node scripts/qa/mutation_matrix_86_81.mjs`

```
subject sha256 BEFORE : 2c2c8692f468eec6971dc45d250828bf6719fba2f4e6fc5dbbf30a4628cbb844

=== CONTROL (unmutated) -- must be GREEN before any cell means anything ===
  control exit=0 GREEN

=== MATRIX ===
  KILLED   M1-DELETE-DROP-STRING-GUARD
  KILLED   M2-MAXATTEMPTS-ONE
  KILLED   M3-BARE-AGENT-CALL-SITE
  KILLED   M4-TSD-SWALLOW-REAL-BUG
  KILLED   M5-TSD-SILENT-EXHAUSTION
  KILLED   M6-WRONG-DROP-LITERAL

6/6 killed
ALL CELLS KILLED

subject sha256 AFTER  : 2c2c8692f468eec6971dc45d250828bf6719fba2f4e6fc5dbbf30a4628cbb844
tracked file unchanged: YES
```

Each cell names the assertion it is aimed at and that assertion must be among the
failures — red alone is not scored as a kill. Operator choice follows EMSE 2021's
survival ranking: **M4 is TSD on the non-drop `throw`** (~75% survival, the most
dangerous mutant here — it makes the retry silently re-run a real bug at ~185K tokens
an attempt), and it was killed by `A3b ...with NO retry`.

---

## 4. VERIFICATION COMMAND

```
$ node scripts/qa/verify_rail_retry.mjs
ALL GREEN: 38 passed, 0 failed
EXIT=0
```

Prior art asserted rather than rebuilt:

```
$ node scripts/qa/verify_research_gate_workflow.mjs
ALL GREEN: 124 passed, 0 failed
EXIT=0
```

---

## 5. REACHABILITY — measured, not read off documentation

| launch form | evidence | verdict |
|---|---|---|
| `Workflow({name})` | three dispatches launching 07:37:05Z / 08:11:45Z / 09:04:38Z all carried a **byte-identical 18,321-char** script matching the commit of 00:28:27Z | **stale, up to 8h36m, across two intervening commits** |
| `Workflow({scriptPath})` | 88-second pickup of `fedcffff`; 102-second pickup of `6b4df8f9`; and a 62-second A/B — NAME took 18,321 chars, scriptPath one minute later took 22,961 | **delivers the on-disk file at dispatch** |

Every dispatched script byte-matches a specific commit, so this is provable rather than
inferred. The live drive in §1 is itself a fourth confirmation: it was launched by
`scriptPath` and its record shows `retry present in dispatched script: True`.

---

## 6. THE RETRACTED-FIGURE SWEEP

```
=== C7 SWEEP, tracked tree, retracted figures only ===
--- 21\.8%
.claude/workflows/qa-verdict.js:394:// "21.8%", a research-gate "53.4%", a "4x amplification" between the two
--- 53\.4%
.claude/workflows/qa-verdict.js:394:// "21.8%", a research-gate "53.4%", a "4x amplification" between the two
--- 4x amplification
.claude/workflows/qa-verdict.js:394:// "21.8%", a research-gate "53.4%", a "4x amplification" between the two
    CLEAN  39 times and completed 34
    CLEAN  76 of all 80
    CLEAN  1 run in 5
```

The single surviving hit is the **retraction notice itself** — the one place that names
the figures in order to forbid them. `research-gate.js` no longer restates them and now
points at that single notice instead, so a correction cannot decay back into a source.

### A defect found by this sweep

`.claude/workflows/qa-verdict.js.export.mjs` still carried all three retracted figures.
Commit `f237bb8d`, subject *"chore: stop a stray workflow test-copy from being
committable"*, **committed the very file it set out to exclude** — `git add -A` staged
it in the same commit, and a `.gitignore` rule cannot affect an already-tracked path:

```
$ git log --oneline --diff-filter=A -- .claude/workflows/qa-verdict.js.export.mjs
f237bb8d chore: stop a stray workflow test-copy from being committable
$ head -4 .claude/workflows/qa-verdict.js.export.mjs | grep -n "name:"
2:  name: 'qa-verdict',
```

It carried `name: 'qa-verdict'` — a **duplicate registered workflow name inside the
dispatch directory**, which is precisely the hazard that commit meant to remove.
Untracked with `git rm --cached`, which lets the existing ignore rule take effect; the
file is **left on disk** because `f237bb8d` recorded it as a concurrent session's
working file.

```
$ git check-ignore -v .claude/workflows/qa-verdict.js.export.mjs
.gitignore:106:.claude/workflows/*.export.mjs	.claude/workflows/qa-verdict.js.export.mjs
```

**Still owed to the operator:** a duplicate `name: 'qa-verdict'` remains on disk in the
dispatch directory. Untracking removes it from the shipped tree but not from whatever
the local registry scans. Deleting another session's working file is not mine to do.
