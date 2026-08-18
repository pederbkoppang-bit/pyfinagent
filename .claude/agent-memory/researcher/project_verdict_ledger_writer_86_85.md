---
name: verdict-ledger-writer-86-85
description: 86.85 — the ledger's reader AND consumer are already fail-closed; only the WRITER is missing. Workflow scripts cannot write by RUNTIME PROPERTY. Workflow/Agent ARE matchable tool_names but no hook here reads tool_response.
metadata:
  type: project
---

Step 86.85 (verdict ledger never written for the step being evaluated). Four
measurements that a future session should not re-derive, and one it must.

**1. The Workflow rail cannot write the ledger — that is a RUNTIME PROPERTY, not a
policy.** `.claude/workflows/research-gate.js:52-55` says an `import fs from
'node:fs'` makes the script **UNLAUNCHABLE** (`SyntaxError: Unexpected identifier
'fs'`). `qa-verdict.js:301-303` states the same for itself. So "the evaluator should
just append its own row" is not a design option to be argued about — it is
unimplementable on the primary rail.

**2. The missing piece is ONLY the writer.** Do not re-audit the reader or the
consumer; both are already correct and already fail-closed:
- `scripts/qa/verdict_history_86_21.py:98-99` returns `None`, never `0`, for
  `LEDGER_MISSING`/`LEDGER_EMPTY`/`UNPARSEABLE`, with the docstring rule *"a caller
  that treats None as 0 has reintroduced the defect"* (`:87-89`).
- `.claude/workflows/qa-verdict.js:311-312` — *"FAILS CLOSED. An absent or unusable
  sequence yields `null`, never `0`."*

**3. `handoff/verdict_ledger.jsonl` is 100% backfill and it is VISIBLE.** 35/35 rows
`recorded_by=main`; `recorded_at` absent on 14/35; of the 21 that have it, **12 share
one microsecond timestamp** and 7 share another. Two bulk backfills, zero seam
writes. It is only detectable as backfill *because* write-time is stamped separately
from event-time — so any writer must keep those two fields distinct.

**4. `Workflow` and `Agent` ARE matchable PostToolUse tool_names — measured, not
assumed.** `handoff/audit/pre_tool_use_audit.jsonl` carries `Workflow` 600x and
`Agent` 1,225x as `tool_name`. The registered PostToolUse matchers in
`.claude/settings.json` are only `Bash`/`Write`/`Edit`, so nothing observes the
agent seam today, but a matcher on those names would fire.

**THE ONE THING STILL UNMEASURED — measure it before designing around it:** whether a
PostToolUse hook receives the tool's *return payload*. `grep -l tool_response
.claude/hooks/*.sh` returns **no matches** — every hook in this repo reads
`tool_input` only. If `tool_response` is unavailable, a hook can still raise a loud
alarm on an un-recorded verdict (NIST AU-5(2)) even if it cannot author the row;
`auto-commit-and-push.sh` + `live_check_gate.py` are the in-repo precedent for
"hold the action until the artifact exists".

**Why:** 86.85's obvious fix (have the Q/A write its own row) is blocked by #1, and
its second-obvious fix (a hook writes it) is blocked-or-not by an unmeasured fact.
Skipping straight to either wastes a cycle.

**How to apply:** scope a 86.85 contract to *writer + alarm* only. `(step_id,
run_id)` is the dedup key (`run_id` present on 33/35 rows). Leave the adjacent
questions to their own steps: [[project_conditional_counter_86_21]] (in-flight
blindness), 86.45 (does `NO_VERDICT` grade), 86.71 (cumulative budget consumer),
[[project_counter_correctness_86_79]] (`records_retained` off-by-one).

Literature backing lives in `handoff/current/research_brief_86.85.md`: ESAA
(arXiv:2602.23193) independently derives this exact topology — agent emits validated
JSON with no write access, deterministic orchestrator appends, record-before-effect,
`idempotency_key` in the agent's output contract.
