---
name: shared-identity-authz-86-33
description: qa-write-guard authorizes on a self-chosen string; agent_id is a documented-but-unread field; general-purpose really wrote evaluator critiques; widening the prefix would break ~50 legitimate workflow-subagent writes
metadata:
  type: project
---

Step 86.33 research (authorization on an unreliable/shared subject identifier).
Findings that are NOT derivable by re-reading the guard, because they came from
measuring its 7,224-record log and from the upstream hooks doc.

**`agent_id` exists and nothing reads it.** The Claude Code hooks doc
(https://code.claude.com/docs/en/hooks) lists `agent_id` AND `agent_type` as common
PreToolUse fields. `qa-write-guard.sh` reads only `agent_type` and logs only
`agent_type`/`tool_name`/`file_path` -- so the project has ZERO data on whether a
non-self-chosen discriminator is even populated. Adding it to the log line is
decision-free and is the only way to answer "distinguish two callers presenting the
same identity" by measurement. Do not assume it is populated or stable -- measure.

**`agent_type` is the spawn name, and upstream says so.** Verbatim: *"Agent name
(for example `"Explore"` or `"security-reviewer"`). Present when the session uses
`--agent` or the hook fires inside a subagent."* The guard's own file HEADER still
cites the old "frontmatter name" reading and contradicts its `is_qa_role` docstring.

**The residual is exercised, but not in the direction the docstring implies.**
Measured from `handoff/logs/qa_write_guard.log`: `general-purpose` issued 15 real
Write/Edit events on `evaluator_critique_82.7.md` / `_82.5.md` on 2026-08-04 (spread
over two multi-hour windows -- not a probe). So an evaluator-shaped artifact WAS
authored under an unmatched name. Meanwhile `workflow-subagent` did ~50 legitimate
production writes (28 backend / 13 frontend / 5 scripts / 4 handoff).

**Why:** the guard's docstring argues the unmatched identities are "indistinguishable
from legitimate writers." Half-right: true for `workflow-subagent`, false as a claim
that nothing evaluator-shaped ever wrote under an unmatched name.

**How to apply:** do NOT widen the `qa-`/`qa_` prefix -- it would block those ~50
legitimate writes and break GENERATE. The literature's transplantable answer is
`no_new_privs`-style monotonic restriction (invert to a narrow permit so an
unrecognised name gets LESS authority), not a longer name list. Also note
`research-gate.js:734` pins a THIRD Layer-3 agentType, `'Explore'`, with ZERO events
in the log. Both checkers hard-code identity LISTS, so neither can see an identity
not on its list -- see [[guard-from-instance-not-class]].

**Log-vs-reality trap:** the log conflates real agent writes with checker-driven
synthetic probes, because both drive the same hook. Discriminate by timestamp spread,
not by path shape -- five different identities "wrote" `frontend/src/lib/api.ts`
within the SAME second (2026-08-10T10:07:32). Absolute-vs-relative path does NOT
discriminate; `verify_qa_write_first_86_31.py` drives absolute paths too.
