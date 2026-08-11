---
name: shared-identity-authz-86-33
description: agent_type is a UNION of definition-name and caller-chosen label (72 values, 2 definition files); agent_id is now MEASURED -- populated on 100% of real subagent writes, 0% of Main's, but opaque and role-free; agent_id presence cleanly separates real events from synthetic probes
metadata:
  type: project
---

Step 86.33 research (authorization on an unreliable/shared subject identifier).
Findings that are NOT derivable by re-reading the guard -- they came from
measuring its log and from the upstream docs. **Updated 2026-08-11 after the P0
log-only commit `8a9a4293` made `agent_id` measurable.**

**`agent_id` is now MEASURED, and the earlier "do not assume it is populated"
caution is resolved.** Post-P0 (Claude Code v2.1.227): `agent_id` is populated on
**63 of 63** real subagent Write/Edit events and on **0 of 77** of Main's own
calls. Shape: exactly 17 chars, `a` + 16 lowercase hex. 18 distinct values; **zero
appear under more than one `agent_type`**. The hooks doc states its purpose
verbatim: *"Use this to distinguish subagent hook calls from main-thread calls."*
So it is a reliable **subagent-vs-Main** discriminator and an **instance**
identifier -- but it is NOT a role attribute: it is opaque, joins to nothing on
disk, and you can only recover a role from it via the `agent_type` in the same
record, which is the untrusted field. Per NIST SP 800-162 that is an *identifier*,
not an *attribute*; authorizing on it would need an id->role registry that does
not exist.

**`agent_type` is a UNION of two namespaces with no discriminator.** The
sub-agents doc says `name` is a *"Unique identifier ... Hooks receive this value
as `agent_type`"* and that *"identity comes only from the `name` frontmatter
field"* -- which would make it a definition attribute. **The installed build
contradicts that**: `.claude/agents/` holds only `qa.md` and `researcher.md`, yet
the log carries **72 distinct `agent_type` values**. The extra ~70 arrive as
invocation-time labels / dynamic `--agents` JSON occupying the same field. This is
exactly the RFC 9700 s2.6 anti-pattern (*"SHOULD NOT allow clients to influence
their `client_id`"*).

**Why:** a prefix allowlist keyed on that field cannot be repaired by widening.
86.31 already widened `== "qa"` to the `qa`/`qa-`/`qa_` prefix, and
`quality-auditor` (11 events, semantically Q/A) still walks past, as does
`general-purpose` -- which authored `evaluator_critique_82.5.md` / `_82.7.md`,
the artifact Main is contractually the verbatim scribe for.

**How to apply:** invert to a narrow PERMIT list (fail-safe defaults), or better,
move enforcement to the platform's own non-forgeable point -- the `tools` /
`disallowedTools` frontmatter, which the runtime enforces against the loaded
definition and which a spawn label cannot steer (the built-in `Explore` agent is
documented as "read-only tools; Write and Edit are denied", and
`harness-self-audit.js` already relies on this). **Blocker:** `qa.md`'s
`memory: project` re-injects Write/Edit on purpose, so a blanket `disallowedTools`
kills the `verdict_wip_*.md` write-first mechanism 86.31 built. See
[[guard-from-instance-not-class]].

**Fail-closed breaks the researcher rail if you forget the `research*` namespace.**
821 events across **31 distinct** `research*`/`res-*` spellings write
`handoff/current/research_brief_*.md` and the researcher MEMORY.md; write-first is
non-negotiable. Note the irony: a permit list keyed on names inherits the SAME
spelling-drift weakness on the allow side. Classify `workflow-subagent` (82 events,
wrote `backend/services/kill_switch.py`) and `general-purpose` (24) before any flip.

**Log-vs-reality trap -- now cleanly solvable.** The log conflates real agent
writes with checker-driven synthetic probes because both drive the same hook.
Post-P0 the clean discriminator is **`agent_id` presence**: every one of the 30
role-typed rows lacking `agent_id` was a probe (`/tmp/x.md`, `/tmp/evil.md`,
`.claude/agent-memory/qa/../../../etc/x`). Prefer that over the old
timestamp-spread heuristic. Absolute-vs-relative path does NOT discriminate.

**Fail-open is not theoretical here.** The log holds **32 Python `SyntaxError`
tracebacks** -- a window where the guard allowed everything while still appearing
installed. Anderson 1972's criteria A (tamper-proof) and B (always invoked) are
both unmet today, independent of the identity question. See
[[fail-open-guards-hide-their-own-breakage]].

**Stale comment to fix:** `.claude/workflows/research-gate.js:47-48` still says the
guard "matches `agent_type == 'qa'` only" -- untrue since the 86.31 widening. The
conclusion (researcher is not blocked) still holds; the stated reason does not.
