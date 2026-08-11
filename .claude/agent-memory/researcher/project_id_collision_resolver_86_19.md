---
name: id-collision-resolver-86-19
description: Step 86.19 -- masterplan.json duplicate-id collisions; the ARCHIVE wins because JSON key order decides, phase-6.5 is a cross-type (phase vs step) collision no archive-exclusion can close, and the defect is latent not live
metadata:
  type: project
---

Measured 2026-08-10 on `.claude/masterplan.json` (1348 id-bearing nodes, 4 duplicate ids).

**Fact 1 -- the winner is decided by JSON key-insertion order, not by "live beats archive".**
`live_check_gate.py::find_step` is depth-first first-match over `node.values()`. In
`phases[36]` the key order puts `archived_legacy_steps` BEFORE `steps`, so `find_step('5.1')`
returns the **archived** twin (status=pending) and never sees the live done step. Whoever last
hand-edited the file decided the resolution. RFC 8259 enumerates three duplicate behaviours
(last-wins / error / keep-all); "first match wins" is a fourth, unnamed one.

**Fact 2 -- `phase-6.5` is a CROSS-TYPE collision and an archive exclusion cannot close it.**
Both twins are live: a PHASE at `/phases[13]` and a STEP at `/phases[12]/steps[4]`. Only
type-tagged lookup (`LIVE_STEP_CONTAINERS`) closes it. Do not assume "exclude the archives"
is the whole fix -- it fixes 3 of the 4.

**Fact 3 -- the defect is LATENT, not live.** None of the 4 colliding ids currently carries a
`verification.live_check`, so `gate_decision` returns `proceed` for all four today. It arms the
moment live step 5.1 gets a live_check: the gate would read the twin that has none and silently
permit. Do not report this as a currently-firing bug.

**Fact 4 -- green-ability of a uniqueness assertion (matters for immutable criteria).**
Per-type: live steps 1230 / phases 114, **0 duplicates each** -> green today. Cross-type
(phases ∪ live steps): **exactly 1** (`phase-6.5`). A cross-type criterion is therefore
uncloseable unless the step also resolves phase-6.5.

**Fact 5 -- the shared-exclusion-set drift seam is ALREADY open.** `ARCHIVE_CONTAINERS` is
defined once (`preflight_verify_masterplan.py:90`) but
`backend/tests/test_phase_75_19_preflight_calibration.py:151-152` re-declares the two names as
string literals instead of importing it; `live_check_gate.py` and
`auto-commit-and-push.sh::load_done_ids` don't know the concept exists. `handoff/archive/phase-5.1/`
and `handoff/archive/phase-6.5/` exist on disk, so the ids are load-bearing outside the JSON --
renumbering breaks real references.

**Why:** researched for the 86.19 research gate; the objective asked data-fix vs resolver-fix.
**How to apply:** recommend scoping the RESOLVER (k8s namespace idiom), never renumbering the
archive. See [[project_phase86_killswitch_channels]] and
[[feedback_immutable_criteria_must_be_green_able]] for the green-ability rule this interacts with.
