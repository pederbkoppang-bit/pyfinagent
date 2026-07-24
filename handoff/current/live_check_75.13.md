# live_check -- Step 75.13 (deps-01, deps-02, deps-08, deps-09)

Date: 2026-07-24. Verbatim captures; exit codes via rc=$? immediately.

## 1. Immutable verification command (exit 0) -- Main's independent verbatim run

```
$ <masterplan 75.13 verification.command: python3 assert chain on requirements.lock / requirements.txt / pip-audit.yml>
immutable_exit=0
```

## 2. New test file + full suite (Main re-runs)

```
$ .venv/bin/python -m pytest backend/tests/test_phase_75_deps.py -q
12 passed in 0.04s
FAIL SET IDENTICAL TO BASELINE
10 failed, 1428 passed, 12 skipped, 5 xfailed, 1 xpassed, 1 warning in 91.09s (0:01:31)
```

## 3. Environment-mutation proof (the step's core boundary)

```
$ .venv/bin/pip freeze | shasum -a 256      # Main, post-step
8df19b228e083e35c4c0de62...  (matches the executor's pre-step hash; 303 lines both)
$ grep -c "==" backend/requirements.lock
303
$ git diff --stat HEAD -- scripts/autoresearch/run_nightly.sh
(empty -- the 75.11 paging seam untouched)
```

## 4. M6 command-vs-test delta (Main independently reproduced)

Moving `PyYAML==6.0.3` into a comment: parsed-line test FAILED (kills)
while the immutable command still exited 0 -- the documented substring
blind spot; the new tests out-bite the command. Restored byte-exact,
12/12 green post-restore.

## 5. Flag-gated / UI note

No live-loop behavior, no UI surface, no service restart needed. The
pip-audit.yml lock-audit first fires on the next GitHub push (inert
locally). The fresh-install dry-check is documented (NOT executed) in
experiment_results_75.13_draft.md.
