---
name: stepid-grep-escape-dot
description: harness_log step-id greps must escape the dot or use grep -F; "67.6" regex matched "67/67 tests" and inflated the CONDITIONAL counter input (67.6 eval)
metadata:
  type: project
---

When counting prior verdicts for a step-id in `handoff/harness_log.md` (the
3rd-CONDITIONAL auto-FAIL check), never grep the bare id: `grep "67.6"` matched
`67/67 tests pass` in an unrelated 67.3 entry because the unescaped dot matches
`/`. Use `grep -F "phase=67.6 "` or `grep "phase=67\.6 "` (anchored to the
cycle-header `phase=<id> result=` format) instead.

**Why:** a false-positive count can either wrongly trigger the 3rd-CONDITIONAL
auto-FAIL or mask a real prior-verdict history -- both corrupt the verdict.
Caught during the 67.6 evaluation (2026-07-10) where the naive grep returned a
67.3 line.

**How to apply:** in every compliance audit, run the CONDITIONAL/prior-entry
count with `-F` on the literal `phase=<id> ` token, and eyeball the matched
lines before trusting the count. Related: [[verbatim-paste-drift-arithmetic]]
(same class: numeric claims need reconciliation, not transcription).
