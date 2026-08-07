---
name: oracle-with-silent-fallback-survives-absent-subject
description: A test oracle that reads a live subject but silently falls back to a checked-in snapshot passes even when the subject is ABSENT; mutate the live read to 404 (not just to empty) and complementarily blank the snapshot to prove which branch is load-bearing
metadata:
  type: feedback
---

When a test resolves ground truth as "live system if reachable, else checked-in
snapshot", the author's own vacuity mutant is almost always "make the resolver
return `{}`" -- which the non-empty assert kills, producing a false sense of
rigor. **The mutation that matters is making the LIVE read raise**, i.e.
simulating the subject not existing at all. The fallback then silently
substitutes the author's hand-written snapshot and the suite stays green.

Worked instance (83.0 cycle 3): `_resolve_schema(table, snapshot)` wrapped
`bigquery.Client.get_table` in `try/except: pass` and returned
`dict(snapshot)`. Patching `get_table` to raise 404 -- exactly the
pre-migration state -- left **all 11 tests passing**, so the C1/C2 schema
guards could not distinguish "migration ran" from "tables never created" in a
credential-less runner.

**Run the pair, not one mutant** -- one alone cannot tell you which branch fed
the assertion:
- **Break the live read** (make it raise). Survives => the fallback is a silent
  self-attestation path.
- **Blank the snapshot** (fixture-side, via a `pytest_collection_finish` plugin
  so you never write the tree). Survives => the live branch is what actually
  fed the assertions on THIS rig.
- **Corrupt the live result** (return a schema object missing the column /
  with the wrong mode). Must KILL, or the guard is vacuous everywhere, not
  just offline.

Verdict wiring: if the corrupt-live mutant KILLS and you have independently
read the live subject yourself, the criterion is MET and the fallback is a
NOTE-level robustness gap (named fix: skip/xfail when the subject is
unreachable instead of substituting). If the corrupt-live mutant SURVIVES too,
the oracle is vacuous and that is blocking.

**Why:** vacuity shape #5 (fixture that cannot represent the failure) hides
inside anything shaped like graceful degradation, and `except Exception: pass`
is what converts a missing subject into a passing test.

**How to apply:** any `try: <read live> except: return <snapshot|default>`
inside a test helper. Also applies to cached fixtures, recorded HTTP cassettes,
and "golden file if API key absent" patterns.

Related: [[feedback_mutate_without_touching_the_tree]] (the injection technique
that makes all three mutants free), [[feedback_survivor_needs_behavioural_differential]]
(a survivor still needs its behavioural diff before you call it a finding),
[[feedback_neutralize_the_write_chokepoint_probe]] (the complementary
composition check when each link is individually mutation-proven).
