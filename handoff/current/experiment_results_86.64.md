# Experiment results — step 86.64

**Step:** 86.64 — the qa-write-guard cannot see the write channel that would actually be used to evade it
**Date:** 2026-08-14 ~07:00 CEST
**Contract:** `handoff/current/contract_86.64.md` | **Gate:** PASSED (`3ddc97bc`)
**Immutable command:** `bash -n .claude/hooks/qa-write-guard.sh && echo guard-parses` → **`guard-parses`, exit 0**

> **Outcome: the control CANNOT be made sound, and criterion 4 explicitly licenses that as
> an acceptable result** — *"accuracy of the claim is itself an acceptable outcome."* So the
> deliverable is a corrected description, not a new guard. **No executable line changed.**

---

## C1 — the Bash-channel evasion, DEMONSTRATED

A single A/B where **only the channel differs** — same qa identity, same target path:

```
identity=qa-86-64-c1, target=handoff/current/evaluator_critique.md

  tool_name=Write  ->  exit 2   BLOCKED  ("qa-write-guard: BLOCKED -- read-only on file contents")
  tool_name=Bash   ->  exit 0   ALLOWED
```

And driven for real rather than simulated — a Bash write under a qa-shaped identity:

```
guard log lines BEFORE : 10293
  (write executed via Bash, 45 bytes, succeeded)
guard log lines AFTER  : 10293
entries for this write :     0     <- the hook never fired
```

**The hook is not bypassed; it is never reached.** `settings.json`'s matcher is `Write|Edit`,
so a Bash tool call is not routed to it at all.

## C2 — write-capable tools, ENUMERATED from the platform tool list

Every tool in this session's list that can create or modify a **local file**, each stated:

| tool | writes local FS | intercepted by matcher `Write\|Edit` |
|---|---|---|
| `Write` | yes | **YES** |
| `Edit` | yes | **YES** |
| `NotebookEdit` | yes (`.ipynb` cells) | **NO** |
| `Bash` | yes (any shell redirect, `tee`, `python -c`, `sed -i`, …) | **NO** |
| `Agent` / `Workflow` | indirectly — spawn subagents that write | only the *spawned* call is hooked, under its own `agent_type` |
| `Artifact` | no — **reads** a local file, publishes remotely | n/a |
| MCP writers (Airtable `create_records_for_table`, Slack `send_message`, Figma `create_new_file`, …) | no — remote services | n/a |

**2 of 4 direct local-FS writers are intercepted.** `NotebookEdit` is a second unguarded
channel the step did not name; it is narrower than Bash (notebooks only) but is the same
class, and widening the matcher to `Write|Edit|NotebookEdit` would close it. **Not done
here** — it would be a behavioural change to a security hook, and no criterion owns it.
Queued as a finding rather than slipped in.

## C3 — fail direction, measured, and the docstring was WRONG

The docstring claimed: *"FAIL-OPEN by design: missing fields, malformed JSON, or an internal
error must never brick the session — **only an explicit qa-outside-memory match blocks**."*
**The second half is false.** Measured across payload shapes, qa identity + `tool_name=Write`:

| shape | exit | direction |
|---|---:|---|
| malformed JSON | 0 | fail-OPEN ✓ *(as claimed)* |
| empty payload | 0 | fail-OPEN ✓ |
| `python3` absent from PATH | 0 | fail-OPEN ✓ |
| **no `tool_input` key** | **2** | **fail-CLOSED** |
| **`tool_input: null`** | **2** | **fail-CLOSED** |
| **`tool_input: {}`** | **2** | **fail-CLOSED** |
| **`file_path: ""`** | **2** | **fail-CLOSED** |
| **`tool_input: "oops"`** (non-dict) | **2** | **fail-CLOSED** |
| non-qa identity, any shape | 0 | allow ✓ |

**I predicted "allow" for two of those and was wrong** — measuring corrected it. Mechanism:
`normpath("")` is `"."`, and `"."` is not inside the memory dir, so an unreadable path is
treated as *outside* and denied.

**Criterion 3 is satisfied**: it requires fail-open *on internal error*, and all three
genuine internal-error paths do fail open. The path-indeterminate case is a different
category, and **denying when the control cannot read its target is the correct direction**
— so the behaviour is kept and the description corrected.

## C4 — the claim corrected: CONVENTION CHECK, not a boundary

Two independent reasons the control cannot be made sound, both measured:

1. **Channel** — the matcher is `Write|Edit`; Bash is never routed (C1).
2. **Identity** — `agent_type` is chosen by the **spawner** (RFC 9700 §4.15), so a caller
   that wants past it simply does not name itself `qa-*`.

It therefore raises the cost of an **accidental** breach and cannot stop a **deliberate**
one. Corrected in both places criterion 4 names:

- **`.claude/hooks/qa-write-guard.sh`** — header rewritten: convention-check framing, the
  measured A/B, and the mixed fail direction replacing the false fail-open claim.
- **`.claude/settings.json`** — `statusMessage` rewritten from *"read-only outside
  .claude/agent-memory/qa/ … fail-open"* to the convention-check description with the
  mixed fail direction and the named covering control.

The covering control for a deliberate write is unchanged and now stated in both: the
Main-side post-verdict `git status` cleanliness rule (`per-step-protocol.md` §4).

## C5 — mutation test: NOT APPLICABLE, and why that is not a dodge

C5 reads *"mutation-test **any guard added**"*. **No guard was added** — the outcome is a
corrected description, so there is no new guard to mutate. What I proved instead is the
property that makes that safe:

```
executable (non-comment, non-blank) lines changed: 0
```

and all six behaviour cells reproduce byte-identically after the edit:

| case | before | after |
|---|---:|---:|
| qa Write outside memory | 2 | **2** |
| qa Write inside memory | 0 | **0** |
| qa Bash, same target | 0 | **0** |
| malformed JSON | 0 | **0** |
| qa Write, `tool_input: null` | 2 | **2** |
| non-qa Write outside | 0 | **0** |

`settings.json` round-tripped; matcher still `Write|Edit`; `effortLevel` still `max`; all 8
hook events intact.

---

## Scope honesty

- **No executable line of the guard changed**, proven mechanically, not asserted.
- **`NotebookEdit` is a second unguarded channel** the step did not name. Disclosed, not
  fixed — closing it changes hook behaviour and no criterion owns it.
- **The 10293-line guard log is a live artifact**; the BEFORE/AFTER equality is the evidence
  for C1, and it will move as the session continues.
- **No Q/A has graded this**, and the step is not flipped.
