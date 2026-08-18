---
name: reference-wf-run-record-corpus-parsing
description: Where the wf_* Workflow run records actually live, and the args-is-a-JSON-string trap that makes a naive parse recover 15.8% of the corpus AND invert the conclusion
metadata:
  type: reference
---

The Workflow run records are **JSON files**, not directories:

```
~/.claude/projects/-Users-ford--openclaw-workspace-pyfinagent/<session-uuid>/workflows/wf_*.json
```

Measured 2026-08-17: **582 files across 44 session dirs**. There are **no top-level
`wf_*` directories** and **no `journal.jsonl`** — two masterplan audit_bases
(86.71, 86.72) describe the population that way and the artifacts do not exist at
that path. Glob `<project>/*/workflows/wf_*.json`; there is no deeper nesting
(measured: `*/*/workflows/wf_*.json` = 0).

**THE TRAP — `args` is a JSON STRING on most records.** Measured over 582:

| `type(args)` | count |
|---|---|
| `str` (parseable with `json.loads`) | 390 |
| `str` (needs regex fallback) | 4 |
| `dict` | 92 |
| `None` | 96 |

A naive `d["args"]["step_id"]` recovers **92 of 582 (15.8%)** — and the survivors are
**recency-biased**, because the dict form is newer. That subsample's top steps were
86.94 (6 qa / **2** researcher) and 86.97 (5 qa / **2** researcher), which reads as
*"the researcher IS re-engaged."* Parsing the string form gives **483 of 582** and the
**opposite** answer: 36.8 = 9 qa / **0** researcher, 75.5 = 7/**0**, 36.12 = 6/**0**,
78.2 = 6/**0**. Same corpus, same question, inverted conclusion — the sampling rule
was the whole finding.

**Why:** an unstated population rule is not reproducible, and a partial parse that
fails *silently* looks exactly like a complete one. Same class as
[[reference_json_value_returns_null_for_objects]] and the `agent_type` spawn-name
trap ([[reference_agent_type_is_the_spawn_name]]).

**How to apply:** when counting runs by step or role, (1) parse `args` when it is a
`str`, (2) state the recovery count AND the unrecovered count side by side, (3) never
report a per-step split without saying how `step_id` was recovered. Role comes from
the script name (`research*` → researcher, `qa*` → qa); distribution is
`qa-verdict` 307 / `research-gate` 79. See also
[[reference_workflow_run_record_fields]].
