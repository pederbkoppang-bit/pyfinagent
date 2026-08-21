# Experiment Results -- step 90.12

> **STATUS: BUILT AND VERIFIED, GATE RUN AFTER THE FACT, NOT EVALUATED. NOT CLOSEABLE.**
> No Q/A. Landed as a harness repair under the operator's 2026-08-21 instruction to stop
> spending evaluation cycles while the harness's own filed defects are unfixed. **The step
> stays `pending` and ungraded.**
>
> **CORRECTED.** This header previously justified skipping the research gate: *"the
> diagnosis came from an independent evaluator and I reproduced it twice by execution --
> there is no assumption left for a research gate to test."* The operator overruled that,
> the gate was run at `wf_69d5b66e-684` (**PASSED**, 8 sources read in full, 26 URLs), and
> it found a **live blind spot** -- see section 8. The justification is replaced, not
> annotated.

**Step:** 90.12 -- the mutation matrix's ERROR discriminator is vacuous over the production
fail-open handler. **Date:** 2026-08-21.

**This is the defect that FAILED step 90.1 at cycle 5.** Closing it is what makes 90.1's
criterion 5 clause 3 satisfiable at all.

---

## 1. The defect

`_drive_traceback` decided *"this mutant could not run"* by requiring the literal string
`Traceback (most recent call last)` on a drive's stderr. `attempt_gate.handle_hook` ends in
a blanket `except Exception` that prints **one line** -- `[attempt-gate] INTERNAL ERROR --
NameError: ... -- failing OPEN` -- and returns 0.

**That handler is correct and stays.** A broken gate must not break the harness. But it
means no failure raised inside the hook's try block *ever* produces a traceback, so the scan
returned `None` for the entire class and those mutants scored **KILLED** where criterion 5
clause 3 requires **ERROR**.

The harm is not nominal: a call-site rename defeats **no guard at all** and still fails 7 of
25 checks, three of them belonging to criteria 2, 3 and 4 — so **a build that never runs
green-washes three criteria at once.**

## 2. The fix -- read the TYPE, not the shape

A traceback is one way an exception type reaches stderr; a fail-open handler formatting it
into a message is another. **Neither is the property.** The property is: *did the code fail
to resolve a name?* That stays typed, so a domain exception remains a KILL whichever way it
arrives. Renamed `_drive_traceback` → `_drive_unresolvable`, because the old name described
the mechanism it happened to use rather than the question it answers.

## 3. Red-first, and the baseline is not a retyping

`verify_error_discriminator_90_12.py` **extracts the pre-fix discriminator from git** at
`d564ad58` — the tree the cycle-5 Q/A actually evaluated — and scores each mutant with
**both** implementations from a **single** observation pass, so the pair is a true
differential on identical evidence:

```
  cell   BEFORE (traceback-only)    AFTER (typed)              expected
  QA1    not-ERROR                  ERROR                      ERROR
  QA1b   not-ERROR                  ERROR                      ERROR
  QA1c   not-ERROR                  ERROR                      ERROR
  QX2    ERROR                      ERROR                      ERROR
  DOM    not-ERROR                  not-ERROR                  KILL
  N0     not-ERROR                  not-ERROR                  KILL_NONE
```

- **QA1 / QA1b / QA1c** are the three call-site renames the Q/A authored (`read_ledger`,
  `extract_step_id_claim`, `extract_step_id`), each at a single call site inside
  `handle_hook`. All three flip not-ERROR → ERROR.
- **QX2** is the *definition* rename. It scored ERROR **before and after** — the pre-fix
  scan already caught that sub-class, which is precisely why the cycle-4 fix looked
  complete. Keeping it in the table is what shows the fix closed a *different* sub-class
  rather than the one already covered.
- **DOM** plants an `AssertionError` inside the hook's try block, so it reaches stderr
  through the **same** fail-open handler in the **same** one-line shape. It must stay a
  KILL, and it does. A separate check proves that drive really exercised the handler, so
  the assertion is not vacuous.

**Why DOM matters:** scoring "any exception" as ERROR silently *deletes* legitimate cells.
That already happened once, in 90.1 cycle 4, to a cell whose whole purpose was to
reintroduce a bug raising `AssertionError`. **An over-eager probe is as bad as a blind one.**

## 4. No silent cell loss (criterion 5)

```
shipped matrix: exit=0  KILLED 15 | SURVIVED 0 (excl. N0) | ERROR 0 | null mutant survived: True
  [PASS] its tally is UNCHANGED by this fix -- measured by me BEFORE the edit and again after
  [PASS] ...and the cell roster is non-empty, so the tally is not vacuous -- 17 cells scored
```

## 5. Verification

```
$ bash -c 'python3 scripts/qa/mutation_matrix_90_1.py --verify && python3 scripts/qa/verify_error_discriminator_90_12.py --self-test'
  checks run: 20 (floor 20)
  failed:     0
EXIT 0

$ python3 scripts/harness/attempt_gate.py --self-test && python3 scripts/qa/mutation_matrix_90_1.py --verify
EXIT 0      # step 90.1's own immutable command, unaffected
```

## 6. Two defects in my own checker, found by running it

- **A roster keyed by the wrong element.** `dict(re.findall(...))` over `(score, id)` pairs
  collapses 17 cells onto 3 keys; the non-vacuity check reported *"2 cells scored"* and
  failed. **A roster keyed by the wrong element is not a roster.**
- **A containment probe that matched its own source.** The first version grepped this file
  for write verbs — and the literal list of those verbs *is in the file*, so it failed
  against itself. This is the same self-referential trap that broke the equivalent check in
  `criteria_shape_90_9.py`, twice in one day. Replaced with an empirical `git status`
  measurement. Its first version then ate one character off exactly one path, because
  `stdout.strip()` removes the leading space of only the *first* porcelain line and a
  fixed-width slice assumed a column that had moved.

## 7. What is NOT done

- **No Q/A verdict.** Not closeable, not flipped.
- `AttributeError` remains in `UNRESOLVABLE_ERRORS` and *can* legitimately be a domain
  error. That is pre-existing, unchanged here, and stated rather than quietly inherited.


## 8. What the research gate returned, after the fact

**Gate:** `wf_69d5b66e-684`, PASSED (enforced), 8 sources read in full, 26 URLs, recency
scan performed, `self_report_disagreed: false`. Brief:
`handoff/current/research_brief_90.12.md`.

**A LIVE BLIND SPOT IT FOUND, now closed.** `UnboundLocalError` was missing from
`UNRESOLVABLE_ERRORS`. It **subclasses** `NameError` — but the printed name is
`UnboundLocalError:`, which does **not contain** the substring `NameError`, and this scan
matches type names **as strings**, so the subclass relationship does not carry. A mutant
that moves a binding after its use raises it and was scored KILLED. New cell **UBL**:
`not-ERROR` before, **ERROR** after.

**A deliberate exclusion it justified.** `TypeError` stays out of the list, and now with a
reason on the record: **cosmic-ray issue #310** is this defect in the wild *in reverse* — a
`TypeError`, a legitimate domain error, was classed non-viable and the mutant mis-scored.
Adding it would trade a false negative for a false positive, and a false positive here
**silently deletes a cell**.

**The doctrine is settled prior art, not a local invention.** Excluding non-viable mutants
from the score **denominator** is what Stryker does (score = detected/valid), what PIT calls
`NON_VIABLE`, what cosmic-ray calls "incompetent", and what Google frames as
unproductive/arid nodes. So the ERROR bucket is standard practice.

**The published risk is the ORACLE's precision, not the doctrine.** The best equivalent-mutant
detector reaches 94.33% precision; Google validated arid-node suppression on 100 labelled
nodes (99 correct). The field's discipline is to **measure the false-exclusion rate on a
labelled sample** — which is why exception type is **rung 4 of a parse → import → run → type
ladder** rather than the whole instrument, and why that laddering is now stated in the source.

**Corroboration of the exact failure mode.** cosmic-ray issue #310 is the mirror case in a
real tool: a non-viable mutant scored SURVIVED because *the crash landed outside the
observation window* — precisely what a fail-open handler manufactures.

**A gap the gate reports honestly:** no source treats fail-open-handler testing as a named
methodology problem. This is not covered territory, and the brief says so rather than
padding a citation.

**Not done:** the labelled-sample false-exclusion rate is **not** measured over a sample large
enough to quote a percentage. Seven labelled cells is a smoke test, not an oracle validation.
