---
name: a-pasted-blocks-section-header-is-a-claim
description: A "regenerated" evidence block paired section [5]'s header with section [8]'s ok-lines and dropped a member; re-run the checker at the tree the block claims and see which header precedes the lines
metadata:
  type: feedback
---

When an artifact pastes tool output under a heading, **the heading is part of the
claim**. Re-run the tool at the tree the block says it came from and check which
section header actually precedes those lines.

**Why:** phase-86.37 cycle 6. `live_check_86.37.md:12` said *"Every block below is
regenerated from the shipped tree."* Section 3's fence carried the header
`[5] stage-2 verification is itself load-bearing -- absent verification FAILS CLOSED`
over six `ok` lines that belong to `[8] structural`, and silently dropped a seventh
member of that group (`ABSENT is reported DISTINCTLY from INCOMPLETE`). I rebuilt
the checker at **every** tree in the step's history and located the header above the
born-inert lines each time:

```
d3bb1dfb 110 ok -> [8]   133060b0 117 ok -> [8]
23270f29 121 ok -> [8]   HEAD     124 ok -> [8]
```

No run produces that header/body pairing, so the block was assembled, not
regenerated. Five prior Q/A cycles read it and did not notice — the `ok` lines were
all real and all green, so nothing *inside* the fence looked wrong.

**How to apply:** for any fenced block that names a section, grep the block's first
line against live output. Then **decide direction**: this one *understated* the
evidence (a mislabelled header + an omitted PASSING assertion) — materially unlike
the same artifact's two earlier misses, which *overstated* (a stale `121 passed`, a
false `+3 = 86.28` attribution). Direction of error is what separates a queue-class
residual from a capping finding. Reconstructing old trees is cheap: `git show
"${C}:path"` for BOTH the tool and its subject into a scratchpad mini-repo, then run
it there. **Brace the variable** — bare `$C:path` in zsh fires the `:s` history
modifier and git silently receives a mangled ref; my control tree failed identically
to the mutants, which is how I caught it. See
[[regenerating-a-capture-leaves-the-authored-summary-stale]] and
[[check-the-attribution-not-just-the-count]].
