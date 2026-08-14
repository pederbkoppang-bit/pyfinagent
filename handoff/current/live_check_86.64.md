# Live check — step 86.64

**Date:** 2026-08-14 ~07:05 CEST
**Required shape:** *"the driven Bash-channel write and the hook log showing no entry for it"*

---

## The required evidence — CORRECTED after the cycle-1 Q/A

**Cycle-1's evidence for this criterion was defective in two ways. Both are withdrawn.**

**(i) The env var was INERT.** I drove `CLAUDE_AGENT_TYPE=qa-86-64-c1 bash -c "…"`. The hook
reads `agent_type` from the **stdin JSON payload only** (`:70`), never the environment — so
that write ran as Main with `agent_type=""`, and **no measurement in cycle 1 satisfied C1's
conjunction** of Bash-channel AND qa-shaped identity.

**Closed live by the cycle-1 Q/A from its own session** (`agent_type='qa'`,
`agent_id a432021376adb291e`):

```
its Write calls -> one guard row each   (01:43:14Z, 01:51:13Z)
its Bash calls creating files outside the memory dir -> ZERO rows (01:44:21, 01:51:28)
```

**(ii) The A/B credited the WRONG GATE.** A piped payload never reaches the routing matcher.
Mutation, reproduced by me on a scratchpad copy:

```
CONTROL  unmutated, qa + tool_name=Bash                      -> exit 0
MUTANT   widen ONLY the in-script gate in qa-write-guard.sh to include "Bash",
         settings.json matcher UNTOUCHED                     -> exit 2
repo md5 unchanged: 3eb5acfea3ec356ac6b289011f49912d
```

**Two gates, separately evidenced:**

| gate | evidence |
|---|---|
| routing matcher `Write\|Edit` | **0 of 26,934** platform Bash calls produced a log row over 21 days |
| `qa-write-guard.sh` in-script gate | the piped A/B above |

**A maintainer who widens only the matcher closes nothing** — the in-script gate still allows Bash.

## The immutable command

```
$ bash -c 'bash -n .claude/hooks/qa-write-guard.sh && echo guard-parses'
guard-parses
exit=0
```

Proves the guard **parses**. It observes no behaviour — the A/B above carries C1.

## Fail direction, measured (C3)

```
malformed JSON / empty payload / python3 absent  -> exit 0   FAIL-OPEN  (as documented)
no tool_input / null / {} / "" / non-dict        -> exit 2   FAIL-CLOSED (NOT documented)
non-qa identity, any shape                       -> exit 0   allow
```

The docstring claimed *"only an explicit qa-outside-memory match blocks"*. Five shapes block
without being such a match. **Behaviour kept, description corrected.**

## Change safety (C5 substitute — no guard was added)

```
executable (non-comment, non-blank) lines changed in qa-write-guard.sh : 0
six behaviour cells after the edit : 2, 0, 0, 0, 2, 0  (identical to before)
settings.json: round-tripped; matcher still Write|Edit; effortLevel still max;
               all 8 hook events intact
```

## What this artifact does NOT license

- **It does not close the step.** Cycle 1 graded **CONDITIONAL** (`wf_19fbea36-8c1`); the C4 blockers are fixed and a **fresh Q/A must grade the changed evidence**.
- **`NotebookEdit` remains an unguarded local-FS write channel** — enumerated in
  `experiment_results_86.64.md` C2, deliberately not fixed here.
- The control is now described as a **convention check**. Nothing should be relied on it
  that assumes a boundary; the covering control for a deliberate write is Main's
  post-verdict `git status` cleanliness rule (`per-step-protocol.md` §4).
