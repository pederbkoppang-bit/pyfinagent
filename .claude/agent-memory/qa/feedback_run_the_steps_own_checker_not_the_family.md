---
name: run-the-steps-own-checker-not-the-family
description: A shared commit broke the step's OWN checker on a whole-line literal; the artifact reported green over a DIFFERENT family of scripts, and the aborted mutation matrix produced ZERO cells while still printing pages of PASS lines
metadata:
  type: feedback
---

Always run the checker and the mutation matrix **the step itself shipped**, by name,
and read their **exit codes** — never accept a "family green" roll-up.

**Why:** 86.78 cycle 3. Commit `77f15b4d` was labelled `phase-86.72/86.78` and changed
`.claude/workflows/qa-verdict.js:835` from `const merged = { ...verdict, escalation }`
to `{ ...verdict, escalation, research_routing }` for the *sibling* step. The 86.78
checker asserted the **whole line as an exact literal**
(`verify_escalation_86_78.mjs:159`), so it went red — exit 1, 1 of 51 — and
`mutation_matrix_86_78.mjs` then **ABORTED on its own control** with **zero cells**
executed. The experiment_results was updated **23 minutes after** that commit and
reported "Family green at write time (136/124/96)" — prompt-render, research-gate,
args-boundary — a family that **excludes both scripts the step built**. All three
family scripts really were green; the two that carried criteria 3/4/5/6 were not.
The live_check's own re-run header still read `# 37 checks, exit 0` / `# 10 cells,
exit 0` against measured `exit 1` / `exit 1` / **0 cells**.

**How to apply:**
- `git show --stat` every commit in the step's set; if a commit is labelled for TWO
  steps, assume it broke the other one's probes until you have run them.
- An aborted matrix still prints the checker's PASS lines after the ABORT banner.
  **Count the `KILLED`/`SURVIVED` lines** (`grep -cE "^ +(KILLED|SURVIVED)"`) and read
  the exit code; a page of green text with 0 cells is the failure mode.
- Prefer a PROPERTY assertion (prefix match / key membership) over a whole-line
  literal in a checker — an additive third key is a legal change that a literal
  cannot survive. Related: [[feedback_byte_presence_pin_is_satisfied_by_a_comment]],
  [[feedback_matrix_row_sums_pin_the_tree]].
- When their instrument is down, build your own: extract the real function's source
  span and `import()` it from a `data:` URL, **control green first**, then mutate in
  memory. Zero tree writes, and it tells product defect from instrument defect.
  See [[feedback_mutate_without_touching_the_tree]].
