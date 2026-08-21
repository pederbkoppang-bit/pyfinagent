# Experiment Results -- step 90.15

> **STATUS: BUILT, VERIFIED DETERMINISTICALLY, NOT EVALUATED. NOT CLOSEABLE.**
> No research gate and no Q/A were spawned. The operator's instruction of 2026-08-21 was
> to stop spending evaluation cycles while the harness's own known defects are unfixed;
> this step IS one of those defects, and it is the only one that was **live on every Q/A
> spawn**. It is landed as a repair so the next evaluation runs on a sound rail. Nothing
> here has been independently graded, and the step stays `pending`.
>
> **Why no research gate:** the diagnosis was produced by an independent evaluator
> (`wf_01b37b7d-fd2`) and reproduced by me twice by execution. Research exists to stop
> building from assumption; there is no assumption here to test.

**Step:** 90.15 -- all three sibling-leak guards compute one seam UPSTREAM of the object
actually returned. **Date:** 2026-08-21.
**Research gate: PASSED (enforced)** -- `wf_7af57a73-cd4`, 9 sources read in full, 37 URLs,
`self_report_disagreed: false`, brief `handoff/current/research_brief_90.15.md`. **Run AFTER
this build, on the operator's instruction, and it found an open residual -- see section 0.**

## 0. THE GATE FOUND A RESIDUAL THIS BUILD DOES NOT CLOSE (filed as 90.16)

**The fix closes the RECONSTRUCTION seam and not the MUTATION seam.** Reproduced by me in a
shadow tree before filing:

```
MUTANT: a post-guard MUTATION of the returned object; `return returned` left intact
  returned.caller_note = 'x'
  return returned
  checks run: 100 (floor 100)
  failed:     0
exit: 0  (0 == SURVIVED -- the residual is real)
```

**And the part that matters more:** cells **M18 and M19 look like they cover this and do
not.** They die to the SOURCE SCAN `/\nreturn returned\n/`, **not to any guard firing** --
because no checker observes the object the module actually returns. The behavioural drives
exercise an *extracted* guard against hand-built objects, which is a different thing. So the
matrix reports two kills for a property that is not behaviourally covered at all.

**What the literature says about the shape:** this is textbook **CWE-367** time-of-check /
time-of-use -- *"access() and fopen() operate on filenames rather than on file handles"* --
and CWE-367's preferred remedy is to **remove the window, not police it**, which is what
construct-once did for the reconstruction half. **CWE-441** supplies the provenance rule for
judge pipelines: *"the immutability of the identity of the initiator must be maintained."*
**OWASP** prescribes exactly the allowlist/denylist pairing this file now has -- the three
per-object filters are denylists, the key-set guard is the allowlist, with the allowlist
primary. **RFC 9457 dissents**: consumers MUST ignore unknown members, so the
producer/consumer split is worth stating rather than assuming away.

**Not fixed here, deliberately.** The obvious remedy is to seal or freeze the returned
object -- and that **changes what the Q/A rail returns on every spawn**. Landing it ungraded
would be the shipped-fix-that-never-ran pattern on the live rail. Filed as **90.16** with its
own immutable command.

---

## 1. The defect, reproduced before it was touched

All three guards were evaluated against `merged`; the function then returned a *different*
object one line later. Anything spread in at that final statement reached no guard.

**RED-FIRST, against the pre-fix code at commit `d564ad58`:**

```
$ flatten severity_routing at the final return, then run the SHIPPED checker
  checks run: 87 (floor 74)
  failed:     0
PRE-FIX + flatten exit: 0   (0 == SURVIVED, the defect)
```

The same shape survived for `escalation` and `research_routing`, which is why this is P0
rather than a 90.2 residual: those two guards ship on **every Q/A spawn** and carried the
identical seam since phase-86.78 and phase-86.72. The consequence is caller-authored fields
surfacing as top-level siblings of `ok` / `verdict` in the object Main transcribes VERBATIM
-- the doer/judge blur those guards exist to prevent.

## 2. The fix removes the seam rather than adding a fourth guard

`returned` is constructed ONCE, every guard runs against it, and it is returned unchanged:

```js
const merged   = { ...verdict, escalation, research_routing, severity_routing }
const untouched = Object.keys(verdict).every(k => merged[k] === verdict[k])
const returned = { ...merged, verdict_unmodified: untouched }
// ... three per-object filters, now `k in returned` ...
// ... then the positive-completeness guard ...
return returned            // no spread, no second construction step
```

**Plus a positive-completeness guard, because the three filters are structurally
insufficient.** Each answers only *"did MY keys leak?"* -- none can see a spread of some
future caller object. The new guard asserts the whole top-level key set: anything that is
neither a judge key nor one of four named caller siblings throws.

**It runs LAST, on purpose.** Running it first swallowed the specific 86.78 / 86.72 / 90.2
messages that name *which* object leaked, and a sibling checker asserts on exactly those
strings -- caught by running that checker rather than by reasoning about it. Specific
diagnosis first, catch-all last.

## 3. Verification -- deterministic, and the whole rail, not just this file

```
90.2 immutable command   exit 0    (92 checks over a floor of 79, 21 mutation cells)
verify_escalation_86_78          exit 0
verify_research_gate_workflow    exit 0
verify_prompt_render_86_90       exit 0
verify_workflow_args_boundary    exit 0
verify_rail_retry                exit 0

  ok   N0   SURVIVED  expected SURVIVED
  ok   M18  KILLED    expected KILLED     flatten at the FINAL return -- the mutant that survived before
  ok   M19  KILLED    expected KILLED     a caller key belonging to NO named sibling -- only the completeness guard kills it
  ok   QX   ERROR     expected ERROR
```

`verify_prompt_render_86_90.mjs` drives the workflow body end to end, so the return path is
executed, not only the sliced pure functions.

## 4. Two defects this work surfaced in my own prior output

**A checker had been red since 90.2 landed, and I never ran it.** `verify_prompt_render_86_90`
anchors a mutation cell on the `const merged = ...` line. Phase-90.2 added `severity_routing`
to that line **hours earlier**, the anchor stopped matching, and the cell reported
`anchor unique -- found 0`. Its anchor-uniqueness guard did exactly its job -- refused to run
a no-op cell and said so -- but **I changed a shared file and did not re-run the checkers
that anchor on it.** Anchor updated, with the reason recorded at the site.

**A red control made two live mutants look dead.** While co-changing the checker, one
source-scan assertion pinned a literal the fix had moved. Every cell then carried that
failure, so M18 and M19 reported KILLED while genuinely surviving. **A red control does not
merely invalidate the run -- it manufactures kills.** Fixed, then the cells failed honestly
and were fixed for real.

**And the cause of that second one is worth keeping:** section I's source scans read the
tracked file from disk, so they were blind to *every* mutant -- they tested the repository,
not the subject. `load()` now carries the subject source with the module and the scans read
that. This is why L1/L2 had needed behavioural drives to be meaningful at all.

## 5. What is NOT done

- **No Q/A verdict.** The step is not closeable and is not flipped.
- The masterplan criteria for 90.15 are untouched and remain the bar this must be graded
  against; criterion 3 (the judge-authored carve-out cell) and criterion 6 (the ledger
  sha256 pair) are **not** yet covered by a dedicated cell.
