# experiment_results -- step 86.118

**Step:** the backend test suite has 18 PRE-EXISTING failing tests, so it cannot
detect a regression and every step's "suite is green" check is measuring a red
baseline. **P2, verification-integrity.**

**Immutable verification command:**

```
$ bash -c 'source .venv/bin/activate && python -c "import ast; ast.parse(open(\"backend/tests/conftest.py\").read()); print(\"parses\")"'
parses
exit=0
```

Full evidence with verbatim command output: **`live_check_86.118.md`**.

## What this step SHIPS

**19 red -> 8 red, and not one test was weakened.** No bulk xfail, no skip, no
deleted assertion, no widened tolerance, no pinned seed. Eleven tests repaired,
four defects filed as their own numbered steps, and the residue named
individually with a disposition each.

| file | what changed |
|---|---|
| `backend/tests/test_phase_57_1_reject_binding.py` | fixture PINS the flag under test instead of inheriting `backend/.env`; the "ships default-OFF" claim re-aimed at the declared field default |
| `backend/tests/test_phase_60_3_data_integrity.py` | asserts the declared field default rather than the deployed value |
| `backend/tests/test_phase_40_2_claude_code_v2_1_140_features.py` | effort pin tracks the operator-documented `max`, so the guard still catches a clobber |
| `backend/tests/test_phase_75_prompt_contracts.py` | archive-aware artifact resolver |
| `backend/tests/test_phase_75_sre_ops.py` | archive-aware resolver + a rebuilt `launchctl bootstrap` oracle carrying its own known-bad fixtures |
| `backend/tests/test_phase_75_17_verification_paths.py` | shape census git-pinned to `BASELINE_COMMIT` |
| `scripts/qa/sweep_absent_verification_paths.py` | classifier learns that `\|\|`-joined arms are ALTERNATIVES |
| `scripts/qa/mutation_86_118.py` | criterion 7 -- 13 cells over 7 targets, built on `guardlib` |

## The result that matters more than eleven repairs

**The suite is not hermetic: it inherits the operator's `backend/.env`.**

Four of the nineteen failures shared this ONE root cause. `Settings` loads
`backend/.env`, so every field a test fixture does not explicitly name falls
through to the deployment. Tests whose comments read *"flag-OFF"* and *"ships
default-OFF"* were running **flag-ON**, because the operator had legitimately
promoted two flags.

```
FIXTURE-BUILT Settings (exactly as the tests build it):
  paper_risk_judge_reject_binding = True
  paper_data_integrity_enabled    = True
DECLARED code defaults:
  paper_risk_judge_reject_binding = False
  paper_data_integrity_enabled    = False
```

Neutralising exactly those two overrides and changing nothing else: `4 passed`.

The consequence is bigger than four red tests. **The same commit is green or red
depending on what is deployed**, any future flag promotion silently reddens
tests unrelated to that flag's purpose, and every past "the suite is green"
claim was conditional on `.env` in a way nobody stated. The CLASS is filed as
**86.125**; this step repaired the four instances.

## Results, by criterion

**Criterion 1 -- RE-MEASURED, twice, with the command stated.** `19 failed,
3672 passed` and `19 failed, 3673 passed`. The 19 FAILED names are byte-identical
across both runs. The `passed` count moved by one because a **concurrent peer
session is committing to this same tree** -- said plainly, because "two runs
agreed" is a weaker claim than it looks when the tree is moving underneath them.
`pytest-randomly` is absent (**86.119**), so both runs share one collection
order and prove nothing about order-independence.

**Criterion 2 -- every failing test classified with the evidence read or run to
reach it.** Full table in `live_check_86.118.md` §3. Buckets: 4 env-leakage,
2 classifier false-positive, 2 consumed-evidence, 2 census-vs-live-artifact,
1 superseded policy, 1 proxy assertion, 1 exit-code drift, 1 lifecycle
meta-test, 1 rotated log, 3 candidate PRODUCT-DEFECT, and 1 order-dependent
(outside scope).

**Criterion 3 -- every PRODUCT-DEFECT candidate filed, none closed by editing
the test that found it.** **86.126** (swap engine emits 1 swap SELL where 2 are
expected) and **86.123** (`nightly_outcome_rebuild` produces an outcome where
none should exist, plus a live-BigQuery Mystery Guest). **86.124** owns the four
non-product tests left red. **86.125** owns the env-leakage class.

**Criterion 4 -- nothing weakened.** Two assertions were re-aimed and each is
argued with the measurement that justifies it. The `is` identity comparison in
`test_off_identity_prompts_are_verbatim_constants` was deliberately left as
`is`. The one deselect used anywhere is in the mutation matrix's control, names
a single test explicitly, and is disclosed in the script and the live_check.

**Criterion 5 -- the ORDERING-ARTIFACT class is EMPTY in scope, measured.**
18 FAILS_ALONE / 1 PASSES_ALONE. The single order-dependent test is the 19th
failure and lies outside the named 12 files, so this is a measured `n=1 outside
scope` rather than a failure to find shared state among the 18.

**Criterion 6 -- post-work counts reported, residue named.** `8 failed, 3684
passed`. Each of the 8 is named with its disposition in `live_check_86.118.md`
§5. **A smaller honest red count beats a green one that proves nothing.**

**Criterion 7 -- 13 cells over 7 targets, 13 KILLED, 0 SURVIVED, 0 UNSCORABLE**,
every control observed GREEN first, SHA-256-verified restore per target,
1 EQUIVALENT-BY-DESIGN declared up front with its measurement.

## Defects this step found in ITS OWN work

Recorded because the matrix caught them rather than review:

1. **A red control silently made three cells UNSCORABLE.** `guardlib`'s
   `pytest_runner` took one suite string and passed `"a.py b.py"` as a SINGLE
   argv element -- one nonexistent path -- so the target's control read as
   legitimately RED while the same command typed by hand was green. Fixed by
   expressing multiplicity through TYPE and never splitting a string, since
   splitting would have "fixed" it while breaking any path containing a space.
2. **A cell that mutated unreachable code.** M3c mutated a `raise` that is never
   reached while the artifact resolves, so it changed nothing and SURVIVED.
   Re-aimed at the return path.
3. **A cell whose survival WAS the finding.** M4b showed that asserting the flag
   on the fixture instance restates the fixture and cannot fail -- which is
   exactly why the assertion was re-aimed at the declared default. Declared
   EQUIVALENT-BY-DESIGN with its measurement rather than deleted.
4. **A step-ID collision with a concurrent session.** The first filing attempt
   guessed 86.121/86.122; a peer session had just filed those. The guard
   refused to overwrite them, but it also silently DROPPED two of my filings.
   IDs are now computed from the live maximum, and both peer steps were verified
   intact afterwards.

## A hypothesis tested and discarded rather than reported

The swap failure looked like a fifth env-leakage victim -- same fixture shape,
and the reject-binding flag plausibly blocking a swap SELL. Driven with both
promotions neutralised it fails **identically** (`1 failed in 0.23s` vs `1 failed
in 0.24s`). So it stays a candidate product defect and is filed as **86.126**.
Recorded because a plausible-and-wrong consolidation is exactly what an
unmeasured classification would have shipped.

## Scope honesty -- what this step did NOT do

- It did **not** install `pytest-randomly` (**86.119**); doing so here would
  make this step's before/after delta unreadable.
- It did **not** enable `xfail_strict`. The silent `1 xpassed` is a real
  finding, but flipping that flag changes the outcome of every xfail in the repo
  and is its own operator-gated change (**86.124**).
- It did **not** fix the 19th test (`test_phase_86_6_subprocess_channel`); it is
  outside the named 12 files, is classified, and is handed to **86.119**.
- It did **not** touch production behaviour. The only non-test file changed is
  `scripts/qa/sweep_absent_verification_paths.py`, an evidence classifier.
- It did **not** promote or un-promote any flag, and `backend/.env` was not
  written.
