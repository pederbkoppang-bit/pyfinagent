---
name: contract-order-mtime-fallback
description: Single-commit steps defeat git-timestamp contract-before-GENERATE checks; pre_tool_use_audit.jsonl has NO file paths (only ts/tool/verdict/reason) -- use filesystem mtime chain instead
metadata:
  type: project
---

When a masterplan step lands brief + contract + code + tests + results in ONE
commit (the normal pattern under the auto-commit hook), `git log --diff-filter=A`
timestamps cannot prove contract-before-GENERATE ordering, and
`handoff/audit/pre_tool_use_audit.jsonl` cannot either -- its records carry only
`ts`, `tool`, `verdict`, `reason` (no `file_path`), verified 2026-07-08 during
the 61.2 evaluation.

**Why:** compliance item 2 of every Q/A tasking asks for contract-before-build
evidence; two obvious evidence sources are dead ends and burn budget.

**How to apply:** go straight to `stat -f '%m %Sm %N' <handoff files> <code files> | sort -n`.
An untouched contract keeps its authorship mtime, so a clean monotonic chain
(brief -> contract -> code -> tests -> results -> live_check -> commit time) is
positive evidence. Disclose that it is mtime-based (weaker than commit history --
a later edit to the contract would reset it). Related: [[verbatim-paste-drift-arithmetic]].
