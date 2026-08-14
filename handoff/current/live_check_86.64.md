# Live check — step 86.64

**Date:** 2026-08-14 ~07:05 CEST
**Required shape:** *"the driven Bash-channel write and the hook log showing no entry for it"*

---

## The required evidence, verbatim

```
guard log lines BEFORE : 10293
  $ CLAUDE_AGENT_TYPE=qa-86-64-c1 bash -c "echo '...' > <target outside the memory dir>"
  wrote via Bash        : YES (45 bytes)
guard log lines AFTER  : 10293
guard entries for this write: 0   <- the hook never fired
```

**Positive control — the guard is alive and DOES deny on the same identity and path:**

```
$ printf '{"agent_type":"qa-86-64-c1","tool_name":"Write",
           "tool_input":{"file_path":"handoff/current/evaluator_critique.md"}}' | bash .claude/hooks/qa-write-guard.sh
qa-write-guard: BLOCKED -- the qa evaluator is read-only on file contents
(Write/Edit allowed only under .claude/agent-memory/qa/; see
per-step-protocol.md section 4 and phase-75.20.1)
    exit=2

$ ... same identity, same path, "tool_name":"Bash" ...
    exit=0
```

**Only the channel differs.** A zero log delta from a dead hook would prove nothing; the
control shows the hook fires, denies, and logs — and still sees nothing on the Bash path.

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

- **It does not close the step.** No Q/A has graded it.
- **`NotebookEdit` remains an unguarded local-FS write channel** — enumerated in
  `experiment_results_86.64.md` C2, deliberately not fixed here.
- The control is now described as a **convention check**. Nothing should be relied on it
  that assumes a boundary; the covering control for a deliberate write is Main's
  post-verdict `git status` cleanliness rule (`per-step-protocol.md` §4).
