---
name: workflow-run-record-locations
description: Where a Workflow run's SCRIPT return lives vs the PER-AGENT returns, how to corroborate a "verbatim" block against both, and why a char count can differ by exactly 2
metadata:
  type: reference
---

Two different objects, and grading a "regenerated verbatim" block against the
wrong one produces a spurious mismatch (36.27 cycle 3, 2026-08-09).

Base: `~/.claude/projects/-Users-ford--openclaw-workspace-pyfinagent/<sessionId>/`

| What | Path | Shape |
|---|---|---|
| **SCRIPT return value** | `workflows/<runId>.json` -> key `result` | the object the workflow script returned; also `agentCount`, `totalToolCalls`, `totalTokens`, `durationMs`, `status`, `scriptPath` |
| **PER-AGENT returns** | `subagents/workflows/<runId>/journal.jsonl` | JSONL, records `{type:'started'|'result', agentId, result}` — one pair per stage |

Measured on `wf_9880694c-d30`: script `result` = 40 leaves; journal stage-1
result = 23 leaves, stage-2 = 5 leaves. So comparing a regenerated script-result
block against the journal reports a mismatch because they are **different
objects, not disagreeing ones**.

**ANTI-CIRCULARITY TRICK — use the split, don't just avoid it.** The journal is
an INDEPENDENT second file, so it corroborates the first:
`result.envelope == stage-1 journal return` and
`result.brief_verification == stage-2 journal return` (both exactly True on
36.27). Also `stat` the workflow json: its mtime must PREDATE the artifact that
claims to reproduce it (12:28:11 vs 12:54:35), else the source could have been
fitted to the block. A "verbatim" block corroborated by two files + an mtime
ordering is materially stronger than one `EXACT MATCH: True`.

**Char-count disagreements of exactly 2 are a counting method, not content.**
`len(json.dumps(s)) == len(s) + 2` — the two delimiter quotes. A verdict saying
"1193 chars" against an author's "1191" is quoted-JSON-literal vs character
content; confirm with `len(s)`, `len(s.encode())`, and the non-ASCII count
before treating it as a discrepancy. Related: [[verbatim-paste-drift-arithmetic]].

**How to grade the regeneration:** re-emit `json.dumps(result, indent=2)`
yourself and byte-compare to the fenced block; then prove your own check is not
vacuous by mutating the block (drop a leaf, flip a value, truncate one char,
insert a space) and confirming each goes False against a passing control.
See [[mutate-without-touching-the-tree]].
