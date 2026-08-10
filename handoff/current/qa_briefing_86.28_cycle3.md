# Q/A briefing -- step 86.28 (current)

Written by Main, the author under evaluation. **Do not trust it. Verify
anything you rely on.** It exists only to cut reading, because two Q/A
spawns dropped mid-read at 197K and 185K tokens.

## Read this much

| File | Lines | What it is |
|---|---|---|
| `experiment_results_86.28.md` | 72 | what ships, file list, deliberate omissions |
| `live_check_86.28.md` | 194 | captured checker output, mutation matrices, live spawns, disclosed gaps, the operator decision |
| `evaluator_critique_86.28.md` | 187 | verdict LEDGER + the latest verdict verbatim |
| `contract_86.28.md` | 210 | PLAN (its line citations are annotated as PLAN-time, frozen) |

**663 lines total.** The full six-cycle history is preserved verbatim in
`*_86.28_history.md` (1,915 lines) -- byte-identical copies made before
compaction, with md5s recorded in each lean file's header. Nothing was
deleted. Read them only if you need a specific cycle's detail.

## Verdict history (verify from the critique's ledger)

CONDITIONAL, CONDITIONAL, **FAIL**, CONDITIONAL, CONDITIONAL, plus two
DROPPED spawns that returned no verdict. F1: `retry_count: 1` of 3. The
FAIL reset the consecutive-CONDITIONAL counter, so a CONDITIONAL is still
available to you. Dropped runs are not verdicts.

## The one-line state

`.claude/workflows/research-gate.js` has been **frozen since cycle 3** and
was comment-only before that; three consecutive Q/As found the shipped code
correct under every probe. Checker: **92 passed, 0 failed** (baseline 40).
Every finding from cycle 3 onward has been about the CHECKER or the
EVIDENCE, not shipped behaviour -- including the FAIL, which was for a
transcript I typed instead of captured.

## Commands that settle most of it

```
node scripts/qa/verify_research_gate_workflow.mjs        # expect ALL GREEN: 92 passed, 0 failed
grep -n "VALID_TIERS = " .claude/workflows/research-gate.js   # expect no 'deep'
grep -nE "researcher\.md:[0-9]" .claude/workflows/research-gate.js  # expect none
git log --oneline -1 -- .claude/workflows/research-gate.js     # expect the cycle-3 commit
```

Arithmetic audit for any transcript block: `passed + failed` must equal a
real suite size (40 / 61 / 64 / 73 / 78 / 92). A total matching none of
those is a fabricated capture -- that is exactly what the FAIL was for, and
it is the cheapest test you can run on this step's evidence.

## Known residuals, disclosed not defended

- Criterion 9: the full stage-1+stage-2 path has not been re-run live
  post-change (both post-change live runs take the refusal branch, spawning
  nothing). A prior Q/A measured the failure direction as safe -- omitted
  stage-2 fields make the gate fail CLOSED -- and proved the shipped bytes
  executably identical to the version the live runs exercised.
- `coverage.dry` and `opts.floors` deliberately untouched, reasons in
  `live_check` S6.
- The `n()` `-1` sentinel message is confusing; queued, not patched.
- The deep-tier divergence is an OPERATOR decision, `live_check` S7.
