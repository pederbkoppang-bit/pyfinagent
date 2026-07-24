# live_check -- Step 75.16 (sre-ops-08, gap6-03/04/05/08, pins, Dockerfiles, migrations)

Date: 2026-07-24. Verbatim captures; rc=$? discipline.

## 1. Immutable verification command (exit 0)

```
immutable_exit=0     (Main re-run after every edit; executor's M8 proof: FAILS
                      on every leg against the pre-fix git-show tree, 0 post-fix)
```

## 2. Tests + CI-equivalent selection (Main re-runs)

```
$ .venv/bin/python -m pytest backend/tests/test_phase_75_deploy_surface.py -q
44 passed in 1.06s
$ ...test_phase_75_deploy_surface.py + test_phase_75_ci_gates.py
60 passed in 8.59s
$ <3 env overrides false> pytest backend/tests/ -q -m "not requires_live"
1510 passed, 2 skipped, 16 deselected, 5 xfailed, 1 xpassed  (0 failed)
```
Raw suite (executor shell): 8 failed -- strict subset of the 9-red baseline
(the absent 23_2_15 is the documented PATH-shell-dependent test). Zero new.

## 3. THE M1 DELTA PROOF (Main-reproduced, both directions)

Mutant: traceback restored into the quant yield under a RENAMED variable
(err_msg) -- the exact escape hatch the research found in the immutable
command. Result: new AST guard 1 failed (KILLED); immutable command on the
SAME mutant: exit 0 (the documented blind spot). Post-restore 44/44.

## 4. Change surface + boundary

```
22 files changed, +224/-607 (net -383). Deletions: deploy_agents.sh,
cloudbuild.yaml (+RETIRED.md), 4x scripts/debug. New: response.py (pure),
test_phase_75_deploy_surface.py (44 tests).
```
Zero gcloud/docker calls. ONE disclosed executor deviation: two pip-index
PyPI version lookups (network, information-only, $0) -- surfaced in both
drafts for the Q/A. Live quant stream contract preserved (single-line
ERROR:, tokens untouched -- diff-verified). The DEPLOYED quant copy still
runs the old code until the operator redeploys (documented; the repo is
now safe to deploy).
