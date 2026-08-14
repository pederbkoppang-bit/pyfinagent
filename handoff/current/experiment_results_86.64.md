# Experiment results — step 86.64

**Step:** 86.64 — the qa-write-guard cannot see the write channel that would actually be used to evade it
**Date:** 2026-08-14 ~07:00 CEST
**Contract:** `handoff/current/contract_86.64.md` | **Gate:** PASSED (`3ddc97bc`)
**Immutable command:** `bash -n .claude/hooks/qa-write-guard.sh && echo guard-parses` → **`guard-parses`, exit 0**

> **Outcome: the control CANNOT be made sound, and criterion 4 explicitly licenses that as
> an acceptable result** — *"accuracy of the claim is itself an acceptable outcome."* So the
> deliverable is a corrected description, not a new guard. **No executable line changed.**

---

## C1 — the Bash-channel evasion — CORRECTED after the cycle-1 Q/A

> **Two defects in my cycle-1 evidence, both found by the Q/A, both reproduced by me.**

**(i) My "qa-shaped identity" was INERT.** I drove
`CLAUDE_AGENT_TYPE=qa-86-64-c1 bash -c "…"` and called it a write under a qa identity. The
hook reads `agent_type` from the **stdin JSON payload only** (`:70 d.get("agent_type")`) and
never consults the environment. That write was issued by Main with `agent_type=""`. **No
single measurement in the cycle-1 artifact satisfied C1's conjunction** of Bash-channel AND
qa-shaped identity.

The Q/A closed it live from its own session: its `agent_type='qa'` **Write** calls each
produced a guard row (01:43:14Z, 01:51:13Z) while its **Bash** calls creating files outside
the memory dir (01:44:21, 01:51:28) produced **zero** rows.

**(ii) My A/B credited the WRONG MECHANISM.** I wrote that the matcher is why Bash exits 0.
A **piped-payload drive never touches the matcher** — the matcher decides whether the
platform *routes* a call to the hook, and piping bypasses routing entirely.

**Mutation-proven, and I reproduced it independently:**

```
CONTROL  (scratchpad copy, unmutated)  qa + tool_name=Bash   -> exit 0
MUTANT   widen ONLY the in-script gate `tool_name in (...)` to include "Bash",
         settings.json matcher UNTOUCHED                      -> exit 2
repo md5 unchanged: 3eb5acfea3ec356ac6b289011f49912d
```

**So there are TWO gates and my text conflated them:**

| gate | decides | evidenced by |
|---|---|---|
| `settings.json` matcher `Write\|Edit` | whether the platform **routes** a call here | the log: **0 of 26,934** platform Bash calls produced a row over 21 days |
| `qa-write-guard.sh` in-script gate | what the script does with a payload it **has** received | the A/B above |

**This matters operationally:** a maintainer following my cycle-1 text would widen the
matcher and believe the accidental-breach channel closed — while the in-script gate still allows Bash.
The contract predicted exactly this: *"even with the matcher fixed, :124 gates on
`tool_name in ("Write","Edit")` and would allow Bash anyway. Both must change."*

**Log aliveness (positive control):** the Q/A's own Write appears as the log's last row, so
the zero-for-Bash is a measured zero, not a dead log.

## C2 — write-capable tools — RE-ENUMERATED from the audit stream (cycle-3)

> **My cycle-1/2 table failed a known-member recall test against a member MY OWN CONTRACT
> had already named.** `contract_86.64.md:81` reads *"the Q/A can call Bash … and
> `browser_take_screenshot`."* Grep for `screenshot|playwright|browser_` over my artifacts
> returned **no mention**. I typed a plausible list instead of deriving one — the exact
> failure `qa.md` §4b calls *"a scan that cannot locate its own already-known members is a
> FAILED gate, not a partial pass."*

**Scope DERIVED, with the command stated:**

```
handoff/audit/pre_tool_use_audit.jsonl -> 97 distinct tool names, 178,006 events
```

**Local-filesystem writers in that derived set, by observed volume:**

| tool | events | writes locally | intercepted by matcher `Write\|Edit` |
|---|---:|---|---|
| `Bash` | 82,291 | yes — any redirect, `tee`, `sed -i`, … | **NO** |
| `Edit` | 6,026 | yes | **YES** |
| `Write` | 5,169 | yes | **YES** |
| `mcp__playwright__browser_run_code_unsafe` | **391** | yes — arbitrary code with Playwright API access | **NO** |
| `mcp__playwright__browser_take_screenshot` | **91** | yes — caller-supplied filename | **NO** |
| `mcp__paper-search-mcp__download_arxiv` | **3** | yes — downloads a PDF | **NO** |
| `NotebookEdit` | 0 observed | yes (`.ipynb`) — capable, never used | **NO** |

**So it is 2 of 6 observed local writers intercepted, not "2 of 4".** The old denominator
was wrong because the table asserted *"MCP writers … no — remote services"*, and that row is
**falsified by measurement**: `find .playwright-mcp -type f` returns **307** local files
(191 `.yml`, 114 `.log`, 1 `.png`, 1 `.json`).

**The three MCP writers matter more than NotebookEdit**, which I had flagged: two of them are
on the **Q/A's own declared tools line** (`qa.md:4`), so they are reachable by the very role
this guard constrains — and `browser_run_code_unsafe` executes arbitrary code.

### The declared-vs-effective discrepancy the contract demanded be settled

Contract plan item 2: *"qa.md's `tools:` line lists no Write and no Edit, yet this session's
agent listing showed both … which is exactly what this criterion exists to settle."* I never
addressed it. Settled now:

- `qa.md:4` declares `Read, Bash, Glob, Grep, SendMessage, mcp__playwright__browser_*` — **no
  `Write`, no `Edit`**.
- The Q/A **holds `Write` and `Edit` regardless**, because `memory: project` frontmatter makes
  the loader auto-enable them (the reason this hook exists at all — see the WHY block at the
  top of `qa-write-guard.sh`).
- **Consequence:** the declared allowlist is *not* the effective tool set, so an enumeration
  taken from `qa.md` would be wrong in both directions. The audit stream is the only honest
  source, which is why the derivation above uses it.

**Still not fixed, and now a larger gap than disclosed in cycle 1:** widening the matcher to
`Write|Edit|NotebookEdit` would not touch the three MCP writers either. No criterion of this
step owns a behavioural change, so this is enumerated and queued rather than slipped in.

## C3 — fail direction — CORRECTED TWICE, and it splits on TRUTHINESS

The original docstring claimed *"only an explicit qa-outside-memory match blocks."* False.
**My cycle-1 replacement was also wrong**: I labelled the deny leg "PATH INDETERMINATE",
and its own subject falsifies that.

**Measured boundary (qa identity + Write/Edit):**

| shape | exit | leg |
|---|---:|---|
| missing `tool_input` / `null` / `{}` / `file_path: ""` / non-dict `tool_input` | **2** | **falsy → `""` → `normpath("")` is `"."` → deny** |
| **`file_path: 123`** | **0** | **truthy non-string → `.replace()` raises → ALLOW** |
| **`file_path: ["a","b"]`** | **0** | same |
| malformed JSON / empty payload / `python3` absent | 0 | hook-itself-broke → fail-open |
| **`agent_type: 5`** (genuine uncaught raise) | **0** | fail-open |
| non-qa identity, any shape | 0 | allow |

**So the deny leg is NARROWER than "cannot read the target" — a garbled path sails through.**

**My cycle-1 fail-open proof did not prove what it claimed.** The Q/A's point is exact: none
of my three cases is an *uncaught raise of the helper*. Malformed JSON hits the **handled**
`except Exception: print("allow malformed-payload")` branch; an empty payload is the
ordinary allow path, not an error; `python3` absent means the helper never runs at all. The
criterion asks for the property *"proven by making the guard's own helper raise."*

Genuine uncaught raises — `agent_type: 5` (`'int' has no attribute 'strip'`), `file_path:
123` and `file_path: ["a","b"]` (`'int'/'list' has no attribute 'replace'`) — **all exit 0**,
with tracebacks observed in the guard log. **The property holds; my evidence for it did
not.** Note the helper's `try/except` wraps only the JSON parse: everything after relies on
the bash-level `case *) exit 0` default, which cycle-1 never exercised.

## C4 — the claim corrected: CONVENTION CHECK, not a boundary

> **My cycle-1 reasoning was the reasoning the contract FORBADE.** I grounded the
> conclusion on *Channel + Identity*, where "channel" meant *"Bash is never routed here."*
> The contract says in bold that C4's conclusion **may not** rest on that, because
> **Bash IS interceptable** — this is CWE-693 *"Protection Mechanism Ignored"*, not
> *missing*. A matcher is a changeable configuration, so "the matcher excludes Bash" is a
> reason the control **is not** sound, never a reason it **cannot be made** sound. Regrounded
> below.

**Why it cannot be made sound — two real limits:**

**1. DECIDABILITY.** A hook that intercepts Bash must then decide whether an arbitrary shell
command writes a guarded path. The evidence cuts both ways and both halves belong in the
answer:

- **CARE (2026): 85.64% F1** on shell-command verification — non-trivial, **not hopeless**.
- **CVE-2025-66032** defeated **Claude Code's own validator** via `$IFS` rewriting — shell
  is adversarially hard to parse, so it is **not solved** either.

A control whose correctness depends on winning that arms race is not a boundary.

**2. IDENTITY — decisive on its own.** `agent_type` is chosen by the **spawner**
(RFC 9700 §4.15; established phase-86.33). A caller that wants past this simply does not
name itself `qa-*`. **No amount of channel coverage repairs an unauthenticated identity**,
which is why the conclusion survives even though my original leg (1) was wrong.

**Corrected in both places criterion 4 names — AND IN CYCLE 2 THAT SENTENCE WAS FALSE.**

> **A false past-tense claim of mine, caught by the cycle-2 Q/A.** Cycle 2 shipped this
> heading while `.claude/settings.json` **was not touched at all**:
> `git diff --name-only b59a7038 722158a8` contains no `settings.json`, and the
> `statusMessage` string was **byte-identical** before and after (md5 unchanged). So the
> statusMessage still carried BOTH cycle-1 defects — it credited the piped A/B to the
> matcher, and it kept the falsified *"fail-CLOSED when a qa Write has no readable
> file_path"* category. I corrected the guard header and both artifacts, then wrote that I
> had corrected two places. **Actually rewritten in cycle 3** (new md5
> `490dc442bf699ee3872113e18f1c00ff`), verified by asserting the two defective phrases are
> absent and four required ones present, with the matcher confirmed still `Write|Edit`.



- **`.claude/hooks/qa-write-guard.sh`** — header now separates the **two gates** (the
  routing matcher vs the in-script gate), states the mutation proof that the gate is operative for a
  piped payload, grounds unsoundness on decidability + identity, and replaces the
  falsified "PATH INDETERMINATE" category with the measured truthiness split.
- **`.claude/settings.json`** — `statusMessage` rewritten to the convention-check
  description with the mixed fail direction and the named covering control.

The covering control for a deliberate write is unchanged: the Main-side post-verdict
`git status` cleanliness rule (`per-step-protocol.md` §4).

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
- **A Q/A graded cycle 1: CONDITIONAL** (`wf_19fbea36-8c1`), transcribed verbatim in
  `evaluator_critique_86.64.md`. C1/C2/C3/C5 MET; **C4 NOT MET as delivered** — the
  corrected description credited the wrong mechanism and omitted the decidability grounding
  the contract required. Both are fixed above. **A fresh Q/A must grade the changed
  evidence. The step is NOT flipped.**
- **The Q/A corroborated C1 far beyond my artifact**: guard-log recall of Bash =
  **0 of 26,934** platform Bash calls over the log's 21-day window, with the log proven
  alive by its own Write appearing as the last row.
