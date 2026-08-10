---
name: mutate-without-touching-the-tree
description: Run full mutation testing with ZERO repo writes -- sys.modules injection for Python, a vite alias to a gitignored node_modules/.cache copy for the frontend (plus a CONTROL run)
metadata:
  type: feedback
---

Mutation-test without editing a single tracked file. Beats the
cp-to-scratchpad-and-restore dance ([[restore-mutations-from-worktree-backup]]):
nothing to restore, so a crashed run cannot leave the tree dirty.

**Python** — inject a mutated module into `sys.modules` BEFORE pytest imports the
test module, then call `pytest.main([...])` in the SAME process:
read the real source, `assert old in src` then `.replace(old, new, 1)`, build a
module via `importlib.util.module_from_spec`, `exec(compile(mutated, path, 'exec'),
mod.__dict__)`, register under the dotted name AND `setattr` it on the parent
package. `from backend.services.perf_metrics import f` in the test file then binds
the mutant. Works for the API module too, so "does the payload still carry the
field" is mutatable.

**Frontend** — write the mutated copy into `frontend/node_modules/.cache/<step>/`
(gitignored, and bare imports like `react/jsx-dev-runtime` resolve because it sits
inside the project), then run `npx vitest run --config <scratch>.ts` with an alias
`{ find: /^\.\/cockpit-helpers$/, replacement: <mutant path> }`. Two gotchas: a
scratch config outside the project needs
`NODE_PATH=<frontend>/node_modules` or it cannot `require('vitest/config')`; and a
copy left in /tmp fails module resolution entirely.

**Node/ESM checker scripts** — when the subject is a script whose own checker
reads it from a REPO-RELATIVE path, rebuild a minimal repo in `mkdtemp` and exec
the real checker there: `fs.mkdirSync(d+'/.claude/workflows',{recursive:true})`,
write the MUTATED subject and the (possibly mutated) checker into their real
relative slots, then `execFileSync('node',[d+'/scripts/qa/<checker>.mjs'])` and
grep stdout for `ALL GREEN|^FAILED|  FAIL `. Zero repo writes, no restore, and it
runs the AUTHOR'S checker rather than a re-implementation. Drive it all from
`node --input-type=module -e '...'` so no scratch file is created either (a bare
`rm` anywhere in the command gets the whole Bash call denied). This also lets you
mutate the FIXTURE, not just the subject — 86.28: reverting `makeBrief` to its
pre-change shape turned 8 checks red, which is what PROVED the fixture change was
truthful rather than masking a weakened guard.

**Why:** qa.md forbids Edit/Write on production files but §4c requires EXECUTING
mutations — this resolves the conflict instead of picking a side, and the verdict
can state `git diff --stat` empty without a restore step.

**How to apply:** always run a CONTROL first (empty mutation list, same alias/
injection path). A green control is what separates "the guard died" from "my
harness is broken" — 80.40 cycle 3: control 12/12 green, then F1/F2/F3 each killed
a NAMED assertion. Pair with [[survivor-needs-behavioural-differential]].
