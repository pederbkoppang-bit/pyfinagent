---
name: child-process-escapes-conftest-guards
description: A conftest/monkeypatch guard covers only the pytest process; any test that subprocess.run()s a script is an uncovered channel — grep the script's argparse DEFAULTS, not the call sites
metadata:
  type: feedback
---

When grading a "we closed the channel" step whose mechanism is a `conftest.py`
patch (urlopen, socket, env, singleton), **enumerate the child-process channel
yourself**: `grep -rn "subprocess\.\(run\|Popen\|check_output\)" <test tree>`.
A subprocess loads NO conftest, so the guard is structurally absent there, and
it will not appear in any census keyed on "call sites in test files".

**Why:** phase-86.3 (2026-08-09) shipped a repo-root `conftest.py` refusing
MUTATING urllib verbs aimed at `localhost:8000`. Its criterion-4 enumeration
listed 5 files and a "Channels NOT contained" list naming httpx / raw socket /
filesystem — but **not subprocess**. I re-derived the population with a broader
rule and found `backend/tests/test_phase_4000_2_cc_rail_smoke.py` shelling out to
`scripts/qa/smoke_cc_rail_e2e.py`, whose `--backend-url` **defaults to
`http://localhost:8000`** and which issues settings PUTs. No live egress today
(all 11 call sites pass an explicit stub URL — measured), but the channel is
uncontained and was undisclosed. Same census rule had already missed a member
twice before in that step.

**How to apply:** for any guard installed in-process, the finding is not "is
there a live POST today" but "is there a path where the guard is absent". Check
in this order: (1) subprocess/Popen in the test tree; (2) the shelled script's
argparse `default=` for a live URL — the default is the danger, not the call
site; (3) threads/child interpreters. Report it under the criterion that demands
CHANNEL enumeration, and check whether the follow-up step's scope actually
covers it. Related: [[feedback-isolation-must-cover-every-channel]],
[[census-the-declared-label-space]].
