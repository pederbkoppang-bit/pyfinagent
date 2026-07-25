---
name: mutate-the-library-for-upstream-pins
description: A library-fact-shaped test is not automatically vacuous -- mutate the installed dependency in site-packages to decide whether it is a real upstream pin
metadata:
  type: feedback
---

A test that asserts a fact about a dependency (`assert json.dumps(x, allow_nan=False)
raises`) is NOT automatically vacuity shape #6. Decide it by **mutating the library**,
not by eyeballing the shape: patch the installed file under
`.venv/lib/pythonX.Y/site-packages/...`, run the test, restore in a `finally`, and
md5-verify the restore alongside the project files.

**Why:** phase-80.1 cycle 2 (2026-07-25). Cycle 1 correctly killed
`test_the_fixture_really_carries_a_non_finite_float` as a library-fact assertion posing
as a fixture pin. A second test in the same suite,
`test_plain_jsonresponse_still_raises_proving_the_defect_is_real`, had the identical
shape. Flipping `allow_nan=False` -> `True` at `starlette/responses.py:198` (starlette
1.0.0) **killed it** -- so it was a genuine upstream pin that fires if the dependency
ever stops 500ing on NaN, i.e. it detects when the whole fix becomes unnecessary.
Condemning it on shape alone would have been a false finding. The distinguishing test is
executable, so run it.

The secondary tell is the docstring's CLAIM: the vacuous one claimed to detect a change
in the local fixture (false); the real one claimed only to be a control proving the
defect premise (true). Shape #6 is specifically "*posing* as a fixture pin" -- an
honestly-labelled upstream control is a different thing.

**How to apply:** whenever a guard asserts something about stdlib or a third-party
package, name the concrete upstream change that would break it and execute that change
before calling it vacuous. Same snapshot/restore/md5 discipline as any other mutation --
the venv must come back byte-identical. Relates to
[[derived-scope-lint-use-xargs]] (both are "execute the check, don't reason about it").
