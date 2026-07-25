---
name: rerun-whole-compound-verification-command
description: Always re-run the FULL `a && b` immutable verification command yourself; a pasted second-half output never proves the first half passed, and new frontend test files are inside tsc's include so they turn `tsc --noEmit` red
metadata:
  type: feedback
---

When a step's immutable `verification.command` is a compound `cmd1 && cmd2`, run the
**whole compound** and read its exit code. Never accept an artifact that shows only
`cmd2`'s output as proof that `cmd1` passed.

**Why:** phase-80.3 cycle 1. The command was
`cd frontend && npx tsc --noEmit -p tsconfig.json && grep -n 'Handle' src/components/AgentMap.tsx`.
Both `experiment_results_80.3.md` §2 and `live_check_80.3.md` §B presented, in a block
labelled verbatim, `tsc exit=0` followed by three grep hit lines. Re-running the command
three times gave exit **1** with three `TS2698: Spread types may only be created from
object types` errors — in `AgentMap.handles.test.tsx`, the guard file the step itself
added — so `grep` never executed at all (`&&` short-circuits). The product code was
clean; only the new test file was red. A reader who trusts the pasted grep lines
concludes the whole command passed.

**The specific frontend trap:** `frontend/tsconfig.json` `include` is
`["next-env.d.ts", "**/*.ts", "**/*.tsx", ...]` with `exclude: ["node_modules"]` — so
`*.test.tsx` files ARE type-checked by the immutable command even though vitest itself
never type-checks them. A test can be 3/3 green in vitest while making `tsc --noEmit`
exit non-zero. Loose casts written to satisfy a wide props type are the usual culprit
(here `<AgentNode data={X as never} {...({} as never)} />` — spreading `never`).
`frontend/next.config.js` has no `typescript: { ignoreBuildErrors: true }`, so this class
of error also reaches `next build`.

**How to apply:** on any frontend step, run the immutable command verbatim yourself and
report its exit code before grading criteria. Do NOT run `npm run build` to check the
build impact — it writes `.next`, which the operator's live :3000 dev server shares
(see [[feedback_second_next_dev_breaks_operator_3000]]); `npx tsc --noEmit` is the safe
authority and is what the criterion names anyway. Related:
[[verbatim-paste-drift-arithmetic]] (a "verbatim" block that does not reproduce is a
Contradiction finding, not a transcription nit).
