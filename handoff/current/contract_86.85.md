# Contract -- step 86.85

**Step id:** 86.85
**Name:** the verdict ledger is never written for the step being evaluated, so the
3rd-CONDITIONAL auto-FAIL rule has no input and cannot ever fire.
**Date:** 2026-08-15. **Status entering:** `pending`.

---

## 1. Research gate -- PASSED

`handoff/current/research_brief_86.85.md`, envelope on disk:

```json
{"brief_status": "COMPLETE", "tier": "moderate", "external_sources_read_in_full": 8,
 "snippet_only_sources": 15, "urls_collected": 23, "recency_scan_performed": true,
 "internal_files_inspected": 10, "gate_passed": true}
```

Floors per `.claude/rules/research-gate.md`: >=5 read in full (8), >=10 URLs (23),
recency scan performed (yes). Not audit-class, so `coverage.dry` is informational.

**Findings that FORCE the design** (F-numbers are the brief's):

- **F2 / A** -- the Q/A **cannot** write. `research-gate.js:52-55` states an
  `import fs` makes the script **unlaunchable (SyntaxError)** -- a runtime property
  of Workflow scripts, not a policy. Per ESAA (arXiv:2602.23193 §3, §6.5) that is
  the *recommended* topology reached by accident: the agent emits intentions, a
  deterministic orchestrator writes. **The writer is therefore Main-at-the-seam.**
- **F1** -- record **before** the irreversible effect. Here the irreversible effect
  is Main *acting on* the verdict (transcribing, fixing, flipping the step).
- **F4 / C** -- dedup key `(step_id, run_id)`; `run_id` is the rail's own
  `wf_<uuid>`. Fallback `(step_id, cycle)`.
  **CORRECTED (cycle-1 Q/A, C2):** this line said `run_id` "is present on 33/35
  existing rows", a figure carried forward from the research brief and never
  re-derived. **No predicate yields 33.** Population = every non-blank line of
  `handoff/verdict_ledger.jsonl` at `d1c4a79d~1`; command
  `git show d1c4a79d~1:handoff/verdict_ledger.jsonl | python3 -c "..."`:
  total rows **35**, `run_id` key present **35**, non-empty **35**, `wf_`-prefixed
  **35**, non-`wf_` values `[]`. So it is **35 of 35 on every predicate**.
  Corrected against a re-run, not by editing the digit.
- **F8 / E** -- on silence: **alert always** (NIST AU-5 base), **fail-closed on the
  DECISION that consumes the ledger**, and **never fail-closed on the harness**
  (AU-5(4) / `CrashOnAuditFail` ships disabled by default).
- **F9** -- "unknown" is a third value. The reader already returns `None`, never
  `0` (`verdict_history_86_21.py:98-99`). Nothing here may weaken that.
- **F10 / F** -- **one append-only JSONL with a dedup key.** No projections, no
  snapshots, no rehydration, no CQRS. Scope discipline *is* the finding.

## 2. Localisation -- criterion 1, discharged BEFORE building

Commands and their output, run 2026-08-15:

```
$ wc -c handoff/verdict_ledger.jsonl   -> 10814 bytes, 35 rows
$ python3 (count by key)               -> recorded_by: {'main': 35}
                                          verdict: {CONDITIONAL 18, PASS 7, FAIL 5, NO_VERDICT 5}
                                          distinct step_ids: 10 ; 86.74 rows: 0 ; max date: 2026-08-11
$ python scripts/qa/verdict_history_86_21.py --step 86.74 --evidence-only
                                       -> status=no_rows_for_step, verdicts=(none)
$ python scripts/qa/verdict_history_86_21.py --step 86.21 --evidence-only   [POSITIVE CONTROL]
                                       -> status=ok, 5 verdicts: C -> C -> FAIL -> C -> C
```

**VERDICT: CAUSE = NEVER-WRITTEN.** Not wrong-key (the positive control returns
rows for 86.21 through the same reader and the same key), not pruned (the file is
append-only and its max date is 2026-08-11, before 86.74's verdicts existed), not
only-after-close (86.21 is `pending` and has rows).

**Criterion 1 also asks: if 86.74's verdicts ARE on disk somewhere, say so and
re-scope. They partly are, and the step still stands.** They exist as **10 WIP
artifacts** (`.claude/agent-memory/qa/verdicts/verdict_wip_86.74__*.md`) and as
**3 narrative cycle sections** in `evaluator_critique_86.74.md`. Neither is a
substitute:

- `qa_wip.py`'s own guidance calls a WIP record *"EVIDENCE FOR THE NEXT SPAWN ONLY
  -- never transcribe it into evaluator_critique.md, never feed it to the verdict
  gate, never count it as a verdict"*, and `records_retained` is *"a gauge, not a
  counter"*.
- the critique is prose; no machine can read a sequence out of it.

**Freshly measured this session, and it is the strongest evidence the step has:**
`handoff/harness_log.md` **cannot** carry a per-step sequence either. 86.74's
cycle 5 was never given a row at all, and cycle numbers in that file are **not
unique** -- two independent 193/194/195 runs exist. So all three candidate
substitutes fail, and the ledger is the only structure that can supply
`args.verdict_sequence`.

## 3. Hypothesis

If Main appends one row to `handoff/verdict_ledger.jsonl` at the moment a verdict
returns and **before** acting on it, keyed `(step_id, run_id)` and refusing
duplicates, then `verdict_history_86_21.py --step <id>` will return a real sequence
for the in-flight step, `qa-verdict.js::enforceEscalation` will receive it via
`args.verdict_sequence`, and the 3rd-CONDITIONAL auto-FAIL becomes computable
instead of `null`.

**This session produced the motivating instance.** 86.74's cycle 7 returned
CONDITIONAL; the escalation machinery reported `sequence_status: "not_supplied"`,
`consecutive_conditionals: null`, `would_auto_fail: null`, because the sequence
arrived as prose in `extra` rather than as data. Main had to compute the auto-FAIL
by hand. The ledger is precisely the missing input.

## 4. MEASURED before designing -- the hook question, settled

The brief flagged as **UNPROVEN** whether a `PostToolUse` hook can see the returned
verdict. Measured with a temporary probe hook in the gitignored
`.claude/settings.local.json`, then restored **byte-identical** (sha256
`8f03f194...66` before and after):

| question | answer |
|---|---|
| Does a PostToolUse hook receive `tool_response` at all? | **YES.** Top-level keys include `tool_response`, plus `tool_input`, `tool_name`, `tool_use_id`, `session_id`, `transcript_path`, `permission_mode`, `effort`, `cwd`, `duration_ms`. |
| Does the matcher fire on `Workflow`? | **YES** -- `tool_name: "Workflow"` observed. |
| Does `tool_response` carry the **verdict**? | **NO.** Shape is `{runId, scriptPath, status, summary, taskId, taskType, transcriptDir, workflowName}`. |

**Why, and why it is structural rather than a quirk:** Workflows *always* run in the
background ("this tool returns immediately with a task ID"), so PostToolUse fires at
**launch**, and `tool_response` is the launch receipt. There is no foreground mode,
so **a PostToolUse hook can never author a verdict row.**

**Consequence for this step, taken as the brief's own stated fallback (§B):** the
writer is an **explicit call by Main at the seam**, not a hook. A hook remains
viable *later* as a pure alarm -- it sees `runId` at launch, which is exactly the
key needed to notice a launched run that never got a verdict row -- but that alarm
is **out of scope here** (86.85 = the WRITER only) and is recorded as a follow-up.

## 5. IMMUTABLE success criteria -- copied VERBATIM from `.claude/masterplan.json`

1. the failure is LOCALISED before anything is built: state, with the command and its output, whether the ledger is never written, written under a different key, written and then pruned, or written only after the step closes -- and if the three verdicts on 86.74 ARE on disk somewhere, say so and re-scope this step rather than building a writer for records that already exist
2. the population rule is stated beside every count of ledger rows, and the enumeration command is quoted
3. cross-session persistence is DEMONSTRATED, not asserted: a verdict written in one process invocation is read back by a separate process invocation, since the Layer-3 per-step loop runs across sessions
4. the 3rd-CONDITIONAL rule is proven to FIRE end-to-end by driving it -- three CONDITIONAL verdicts on one step id must produce the auto-FAIL on the next pass, shown by execution rather than by reading the source
5. the interaction with 86.79 (records_retained counts the current spawn; pruning can undercount) and 86.45 (a rail drop recorded in the ledger clears a real escalation) is resolved explicitly in writing, or the conflict is recorded as a numbered blocker and this step is scoped around it
6. a rail drop must not be recorded in a way that CLEARS an escalation, and a missing row must not be readable as 'no prior verdict' -- an absent field supports 'not recorded', never 'did not happen', so prove the recorder ran before treating a zero as evidence
7. verdict semantics are UNCHANGED: nothing here may turn a non-PASS into a PASS, demonstrated under every flag combination
8. mutation-test every new guard with the control observed GREEN first and a byte-identical restore

**Verification command (immutable):**
`bash -c 'source .venv/bin/activate && python -c "import ast; ast.parse(open(\"scripts/qa/verdict_history_86_21.py\").read()); print(\"parses\")"'`

**live_check:** `live_check_86.85.md` carrying the localisation evidence, the
cross-process read-back, and the driven 3rd-CONDITIONAL auto-FAIL.

## 6. Plan

1. **`scripts/qa/verdict_ledger_write.py`** -- the writer, and the only new
   production surface. Appends exactly one row to `handoff/verdict_ledger.jsonl`.
   - dedup key `(step_id, run_id)`, fallback `(step_id, cycle)` when `run_id` is
     absent; a duplicate key is **refused, not overwritten** (Azure: reject an event
     matching an existing entity+event id) and exits non-zero so the caller sees it.
   - **event-time vs write-time kept separate** (`recorded_at` = write time,
     `date`/`cycle` = event identity), so a backfill can never masquerade as history
     (brief pitfall 4 -- 12 existing rows share one microsecond stamp).
   - **fail LOUD**: any write failure prints to stderr and exits non-zero. It must
     never fail silently, because a silent writer manufactures exactly the
     `LEDGER_EMPTY` state the reader is built to refuse.
   - **append-only**: never rewrites a row (F7); corrections are new labelled rows.
2. **`--emit-sequence`** on the writer (or reuse the reader) so Main can obtain the
   array to pass as `args.verdict_sequence` -- closing the loop that failed today.
3. **Tests** `backend/tests/test_phase_86_85_verdict_ledger_write.py`: dedup refusal,
   NO_VERDICT does not clear an escalation, absent row reads as unknown not zero,
   cross-process read-back, and the driven 3rd-CONDITIONAL auto-FAIL.
4. **Mutation matrix** with control GREEN first and byte-identical restore.
5. Backfill 86.74's six verdicts as **labelled** rows (F7: appended and labelled,
   never rewriting), so the ledger's first real consumer has data.

## 7. Explicitly OUT of scope -- criterion 5's boundary, resolved in writing

- **86.45** owns whether a `NO_VERDICT` row grades. This step **records** a rail
  drop faithfully and **changes no counting semantics**. Note the current consumers
  already skip it (`qa-verdict.js` `enforceEscalation`: `if (v === 'NO_VERDICT')
  continue` -- it neither extends nor resets), so recording a drop **cannot** clear
  an escalation today. Criterion 6 is satisfied by preserving that, not by
  re-deciding it.
- **86.79** owns the `records_retained` off-by-one in the *parallel* `qa_wip.py`
  trail. This step writes a different file and does not read `records_retained`.
- **86.71** owns the cumulative attempt budget that would be the ledger's second
  consumer. Out of scope; the ledger merely makes it possible.
- **86.21** owns the counter's in-flight blindness.
- **The PostToolUse alarm** (measured viable above) is a follow-up, not this step.
- **No flag promotion, no `.env` write, no manual cycle, no restart.**

## 8. References

- `handoff/current/research_brief_86.85.md` (gate passed, 8 sources)
- ESAA, arXiv:2602.23193 §3, §3.2, §6.5 -- record at the seam; agent must not write
- arXiv:2606.04990 §2.3, §3.4, §4.2 -- self-authored trails are weak evidence
- Azure Architecture Center, *Event Sourcing pattern* -- idempotency, versioning
- Confluent, *exactly-once semantics* -- producer-assigned logical-event key
- NIST SP 800-53 AU-5 and AU-5(4); Microsoft `CrashOnAuditFail`
- Dudycz, *event sourcing is not an audit log* -- adversarial, scope discipline
