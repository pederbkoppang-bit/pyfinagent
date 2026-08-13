---
name: write-guard-channels-86-64
description: qa-write-guard blindness -- "hooks cannot intercept Bash" is FALSE (PreToolUse fires for every tool except EndConversation); the gap is the MATCHER; bypassPermissions skips PROMPTS not DENY rules; the log's 307 Bash records are all synthetic
metadata:
  type: project
---

Findings from the phase-86.64 research gate (2026-08-14). Full brief:
`handoff/current/research_brief_86.64.md`.

**The premise almost everyone gets wrong.** `qa-write-guard.sh:18` says "Write/Edit hooks
do not intercept Bash". True as written, but the natural reading -- *hooks can't see Bash*
-- is FALSE. Anthropic's permissions doc: "PreToolUse hooks run before the permission
prompt, **for every tool except `EndConversation`**", and "A hook that exits with code 2
stops the tool call before permission rules are evaluated." The Bash channel is
unintercepted **by this hook's `matcher: "Write|Edit"` registration**
(`.claude/settings.json:35`), not by platform limitation. CWE-693 calls this the
**"ignored"** mode ("a mechanism is available and in active use ... but the developer has
not applied it in some code path"), NOT "missing".

**Why:** it changes the design space. "Cannot be made sound" is a *decidability* argument
(CARE arXiv:2607.21642v2 gets 85.64% F1 on shell-command verification, and CVE-2025-66032
defeated Claude Code's OWN command validator via `$IFS` rewriting), never a *capability*
argument. Reaching the right conclusion from the wrong premise still misleads the reader.

**How to apply:** before writing "the platform can't do X" about hooks, check the
tools-reference (45 canonical tools) and the permissions doc. Grep the matcher in
`settings.json`, not just the hook body.

**`bypassPermissions` skips PROMPTS, not DENY rules.** `settings.json:171` sets it
project-wide, but `permissions.deny` (27 entries at `:183-210`) is still a live hard
boundary -- the doc is explicit that "Claude Code evaluates deny and ask rules regardless
of what a PreToolUse hook returns". BUT deny rules are **session-global**: they cannot be
conditioned on `agent_type`, which is the structural reason this control is a hook and not
a permission rule. Don't propose "just use a deny rule" without that caveat.

**Measurement traps in `handoff/logs/qa_write_guard.log`:**
- The 307 `"tool_name": "Bash"` records are **ALL synthetic checker fixtures**. Zero
  platform-delivered Bash events have ever reached the hook (correctly -- wrong matcher).
- Discriminator: real records carry a **12-key** `payload_keys` set + a 17-hex `agent_id`
  (`a0daa10a3d87d331b`); probes carry **5 keys** + `agent_id: "aX"`.
- The log is **gitignored** (`.gitignore:76`, NOT tracked) -- which contradicts
  `qa-write-guard.sh:68` ("this log is committed to the repo"). The keys-only logging
  decision is right; its stated rationale is wrong.

**Two platform behaviours worth remembering:**
- `MultiEdit` **does not exist** on the current platform (0 rows in the tools reference vs
  1 for `Write`). Any doc naming it is stale.
- `Edit` with a non-matching `old_string` **never reaches the PreToolUse hook** -- the
  tool's own validation short-circuits first (measured by the 86.31 Q/A,
  `handoff/archive/phase-86.31/evaluator_critique.md:80`). A deny-probe built that way
  proves nothing.

**The channel has already been used, benignly, first-party.** phase-82.39's Q/A wrote:
"the qa-write-guard hook correctly blocked my one attempted scratchpad Write, so I moved
everything to stdin". A blocked agent reaches for the alternate path by DEFAULT. That is
CWE-424 *Improper Protection of Alternate Path* observed in this repo.

**Fail-open vs fail-closed -- the reconciliation.** Saltzer's fail-safe defaults argues
fail-closed; the project forbids it (a guard that bricks its caller when the guard breaks
is worse). CARE resolves it: fail closed **only** "if parsing fails **and** the raw or
normalized command still contains strong high-risk indicators" -- narrow fail-closed on
positive danger evidence, fail-open on internal error. Use that shape, not a blanket deny.

**Mediate the RESOURCE, not the CHANNEL.** 6 direct write channels + 3 indirect, and the
tool list grows every release. A post-hoc tree diff (per-step-protocol.md:155-174
POST-VERDICT CLEANLINESS) observes the object rather than the path, so it is complete over
an unbounded channel set in a way no matcher list can be. See
[[guard-from-instance-not-class]] and [[research-gate-discipline]].
