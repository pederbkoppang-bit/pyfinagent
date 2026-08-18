---
name: queued-is-a-claim-that-must-reproduce
description: "queued rather than swept in" is a reproducible claim -- walk .claude/masterplan.json for it; 86.90 asserted it 4x with the masterplan untouched since a commit PREDATING the work
metadata:
  type: feedback
---

Treat every **"queued"**, **"filed"**, **"tracked as its own step"** sentence in a
contract or experiment_results as a claim to REPRODUCE, exactly like a numeric one.
The reproducing check is a walk of `.claude/masterplan.json` for a step whose text
covers the named defect -- not a grep for a keyword, which matches unrelated step prose
(measured 86.90: `harness-self-audit` and `unknown key` each returned exactly 1 hit, both
inside *other* steps' descriptions; `86.92` matched a numeric edge-ratio value).

The cheapest decisive check is the **git log on the masterplan itself**:
`git log --oneline -5 -- .claude/masterplan.json`. On 86.90 its newest commit PREDATED
the step's own work commit, which settles all four claims at once -- nothing could have
been queued.

**Why:** phase-86.90 asserted "queued" for (a) three affected PASS re-grades, (b) a
sibling concat defect in `harness-self-audit.js`, (c) stronger unknown-key handling,
(d) a stale generated artifact. None existed as steps. The standing project rule is
"own step per standing rule" -- so "queued" prose without a step means the follow-up is
**lost**, and it reads as done to the next reader. The same session HAD filed 86.91
properly in its own commit, so this is an omission, not disregard -- which is why it is
WARN, not a criterion miss.

**How to apply:** run the masterplan walk whenever an artifact defers work. Also check
the inverse: a live-RED re-runnable checker that the author correctly identifies as
pre-existing (86.90: `verify_workflow_args_boundary.mjs`, 3 cells, fixture drift) is
STILL unqueued red -- name it. Related: [[recheck-prior-remediation-list]],
[[structural-fix-needs-a-mechanism]].
