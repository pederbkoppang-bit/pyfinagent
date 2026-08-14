---
name: retry-never-ran-86-81
description: Step 86.81 — a workflow run record's `script` field embeds the DISPATCHED source, so "did my fix actually run?" is measurable, not inferable; the 86.81 retry had never executed once
metadata:
  type: project
---

The `script` field of every Claude Code workflow run record
(`~/.claude/projects/<proj>/<session>/workflows/*.json`) embeds the **dispatched
source**. That makes "which version of my code actually ran?" a MEASUREMENT
rather than an inference from commit time.

Measured 2026-08-14 on step 86.81: `agentRetryingDrops` (the StructuredOutput
retry, commit `6b4df8f9`) was **absent from the dispatched script on all 18 runs
that day**, including both drops timestamped AFTER the commit instant. Exactly
one run carried the new code. The fix had never executed once while being
reasoned about as if it had.

**Why:** this beats the `grep -c` over `workflows/scripts/<wf>-wf_*.js` that the
goal doc recommends, because that only covers snapshot copies — the two stale
drops ran with `scriptPath=.../.claude/workflows/qa-verdict.js`, i.e. the **real
on-disk path**, and were stale anyway. So `scriptPath` dispatch is NOT by itself
evidence that the new code ran.

**How to apply:** before analysing whether a harness fix worked, read the
`script` field of the runs you are analysing and assert a string only the new
version contains. Related: three more measurement traps found in
`scripts/qa/rail_drop_rate.py` in the same pass — (a) `logs` is EMPTY on all 44
failed runs, so a log-derived "RETRIED" counter is structurally blind on exactly
the exhausted runs it is meant to explain; (b) the before/after split compares
`timestamp[:10]` to a fix-date, so a mid-day fix misattributes 15 of 18
pre-commit runs to "after"; (c) the `exhausted` predicate's second disjunct still
scans the whole blob, which embeds the source — the same self-match that produced
38 phantom drops, fixed in the first disjunct and left latent in the second.

See [[suspect-the-clean-check]], [[named-workflow-script-snapshots]],
[[committed-is-not-in-force]]. Brief:
`handoff/current/research_brief_86.81.md`.
