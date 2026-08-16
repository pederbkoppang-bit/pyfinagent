---
name: uncalled-function-86-97
description: 86.97 -- an AST extractor keyed on DEFINITION node types can never see a call site; the mutant's input is byte-identical, so no future assertion in that file can kill it
metadata:
  type: project
---

The checker's INPUT is unchanged, so the defect is unfixable by adding assertions
to that checker.

**Why:** `scripts/qa/verify_changelog_flip_86_91.py:88-92` (`detector_source`)
filters `tree.body` for `ast.FunctionDef` / `ast.Assign` / `ast.AnnAssign` --
three DEFINITION node classes. A bare call is `ast.Expr(value=ast.Call(...))`
and binds no name, so it can never match, and the `NEEDED` tuple at `:78` cannot
help because `NEEDED` matches names BOUND BY THE NODE. MEASURED read-only
2026-08-16: deleting `_log_decision(bump_type)` (hook `:262`) leaves the
extracted `SHIPPED` string **byte-identical** (at commit 52358053: 7,597 bytes,
sha1 `f7458a6ab1f5fe96`; at the phase-86.97 HEAD: 8,617 bytes, sha1
`072056e58af2befa` -- the count moves with any edit inside the four extracted
names, so cite the commit or re-derive; the byte-IDENTICAL property is what is
invariant), with a positive control (hook source did change, +26
bytes; anchor count 1). Every one of the file's 42 assertions runs against
`SHIPPED`, so this is not "the mutant survived" -- it is "the mutant is
invisible", and adding a 43rd assertion cannot change that. Two call sites are
affected, not one: `bump_type = _flip_magnitude()` (hook `:214`, inside an
`ast.If`) is also absent from the extract, and the checker manufactures BOTH
calls itself (`:132`, `:534`).

**How to apply:** when a guard extracts source and execs it, ask "what node
classes can my filter never match?" before adding cases. The literature name is
*pseudo-tested* -- Vera-Perez et al. (ar5iv 1807.05030 §2.1/§4.1: 2,540 methods,
1%-46% prevalence) and Niedermayr et al. (ar5iv 2103.08480 §IV-A: 291/2041 =
14%, coverage 89%->82% when recounted, and **14 of 25 were side-effect methods**
-- exactly `_log_decision`'s shape). The fix is to DRIVE THE PROCESS (bats-core
`run` / `run -N`, asserting on the FILE not env vars, since `run` uses a
subshell), which is also the only instrument that reaches the three bash
`exit 0` paths at hook `:28`/`:33`/`:37` -- they are outside the heredoc, so no
Python-extraction approach touches them at all. ShellCheck cannot help: SC2317
is reachability not invocation, and it does not parse heredoc bodies.

Silent-path classification (measured 28 of 56 commits on 2026-08-16 hit the
recursion guard): `:27-29` recursion guard = legitimately silent BUT must be
counted (otherwise "guard fired" and "hook never ran" are indistinguishable --
the same unknowable-denominator defect one level up); `:32-34` CHANGELOG absent
and `:36-38` heading renamed = MUST-LOG, both are silent total kills. Any
bash-side write must be `|| true` because `set -euo pipefail` is at `:7`.

Dissent worth keeping: Vera-Perez et al. §4.4 -- developers judged only **30 of
101 (30%)** pseudo-tested methods worth acting on, and "it is not reasonable to
prescribe the absolute absence (zero)". So guard `:262`/`:214`, do NOT guard
`:355` (`lines.insert`, same node class) -- its effect is visible in CHANGELOG.md.

Brief: `handoff/current/research_brief_86.97.md`.
Related: [[project_version_bump_swallow_86_91]], [[project_fixture_rot_dead_gate_86_92]].
