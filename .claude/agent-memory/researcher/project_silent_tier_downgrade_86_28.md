---
name: silent-tier-downgrade-86-28
description: ABSENT vs UNSUPPORTED is the deciding distinction for any defaulted parameter; enum is NOT stripped by structured outputs so it can make an honest report unrepresentable; coverage.dry is unverifiable in principle
metadata:
  type: project
---

Step 86.28 research. Three reusable facts, none derivable by reading the code alone.

**1. The disposition question has a settled answer, and it is not "severity".** When a caller names
a capability the implementation lacks, protocol design offers exactly two legitimate dispositions --
**fail closed** (LDAP control criticality TRUE -> `unavailableCriticalExtension`, RFC 4511 §4.1.11;
HTTP `Expect` -> 417; TLS `inappropriate_fallback`, RFC 7507 §3) or **proceed with a
machine-readable signal in the RESPONSE** (RFC 7240 `Prefer` MUST be ignored, but only because
`Preference-Applied` exists -- §3 says the client "might not be capable of reliably determining if
the preference was (or was not) applied simply by examining the payload"). Silent substitution is
endorsed by no source. **The deciding variable is whether the caller can detect the substitution
from the response** -- not how severe it is. A disclosure buried in the artifact/payload the caller
is trying to assess is the payload, not the response, and does not count.

**Why:** RFC 9413 §6 -- *"Hiding the consequences of protocol variations encourages the hiding of
issues"* -- plus §4.1 on entrenchment. RFC 7507 §1 adds the second half: a catch-all fallback
misfires on unrelated inputs (network glitches read as legacy servers), so an `else -> default`
branch cannot tell a typo from a deliberate stricter request.

**How to apply:** any time a workflow/config layer computes a single `xDefaulted` boolean, check
whether it spans both ABSENT (defaulting legitimate, no referent violated) and UNSUPPORTED (caller
named something real). One boolean cannot carry two dispositions, and the human-readable string it
drives will be FALSE in one branch. Mirror the ABSENT/UNUSABLE/INCOMPLETE trichotomy the args
boundary already uses.

**2. `enum` is NOT stripped on the wire.** Anthropic structured outputs strips
`minimum`/`maximum`/`minLength` and caps `minItems` at 1 (see [[schema-floors-not-enforceable]]),
but `enum` + `additionalProperties:false` DO bind. Consequence: an enum on a self-report field is
the same trap as `const: true` -- it can make an honest answer **unrepresentable**. In 86.28 the
envelope's `tier` enum meant the agent could not report the tier it was actually asked for.

**How to apply:** before putting an enum on any field an agent must report about ITSELF or about its
INPUT, ask whether every truthful value is in the enum. Floors -> assert in JS. Honesty fields ->
plain type.

**3. Some self-reported fields are unverifiable in principle, and the answer is labelling, not a
proxy.** A claim about a document's CONTENT is checkable from the artifact; a claim about the
PROCESS that produced it may not be. `coverage.dry` ("K consecutive search rounds surfaced nothing
new") is a property of executed discovery, so no read-only file check establishes it even in
principle. The 2025-2026 literature's answer is to demote such claims to *non-authorizing* /
*advisory* so they cannot carry a pass (EBTE arXiv:2607.25364v2 §IV-B; Proof-or-Stop
arXiv:2607.14890 §3), NOT to invent a proxy. Corollary trap: a substring/presence check is
necessary-but-not-sufficient and must be DESCRIBED that way -- EBTE §XVI is explicit that structural
conformance is not semantic verification. And a check run over URLs the agent itself supplied is a
*consistency* check, not an independent one.

**Measured anchors (2026-08-10, re-verify before citing):** `grep -c -i deep
.claude/workflows/research-gate.js` = 0; `scripts/qa/verify_research_gate_workflow.mjs` baseline
`ALL GREEN: 40 passed, 0 failed`; `opts.floors` at research-gate.js:269 has zero callers.
Evidence base: EviBound arXiv:2511.05524 measured prompt-level self-reflection at **100%**
false-completion claims (8/8 claimed, 0/8 verified) falling to **0%** with a post-hoc artifact gate,
at only **~8.3%** added execution time -- the cheapest strong result in the brief.

Full brief: `handoff/current/research_brief_86.28.md`.
Related: [[schema-floors-not-enforceable]], [[guard-from-instance-not-class]],
[[operations-that-cannot-fail-loudly]].
