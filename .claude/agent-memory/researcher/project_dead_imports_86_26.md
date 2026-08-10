---
name: dead-imports-86-26
description: F401 dead-import removal in the learn-loop modules -- why absence of an "X as X" alias does NOT prove a name is unconsumed, why compute_benchmark_return is safe, and that pyfinagent has NO ruff config file at all
metadata:
  type: project
---

Step 86.26 (2026-08-10) removes 7 ruff F401 findings from `outcome_tracker.py`,
`memory.py`, `bias_detector.py`, `conflict_detector.py`. Findings that were NOT
obvious from the step text and cost real tool calls to establish:

**1. The `X as X` re-export convention is TYPE-CHECKER-facing only -- it proves
nothing at runtime.** PEP 484 / the Python Typing Spec say only `import X as X`,
`from Y import X as X`, and `from Y import *` re-export, and "all other imported
symbols are considered private by default". But **mypy's `implicit_reexport`
defaults to `True`** ("imported values to a module are treated as exported"), and
CPython imposes no rule at all -- `from mod import name` always works. So the
absence of an alias is NOT evidence that nothing consumes the name. **A repo-wide
consumer grep is the only sound proof.** Do not let a reviewer accept the alias
check as sufficient.

**Why:** this is the exact shape of `guard_from_instance_not_class` -- a
necessary-but-not-sufficient test presented as a proof.
**How to apply:** on any dead-name removal, run the FROM-this-module grep; the
`__all__` / alias check is a supplement, never the argument.

**2. `compute_benchmark_return` at `outcome_tracker.py:17` is NOT a re-export**
(the masterplan explicitly suspected it was). Its only non-defining consumer,
`backend/tests/test_dod4_tier1_coverage_investment.py:346`, imports it from
`backend.services.perf_metrics` -- the DEFINING module. The benchmark math
outcome_tracker relies on survives anyway because `beat_benchmark` calls it
internally at `perf_metrics.py:560`, and outcome_tracker uses that alias.

**3. pyfinagent has NO ruff configuration file anywhere** -- no `pyproject.toml`,
no `ruff.toml`, no `setup.cfg` at the repo root (the only hit is a stale
`.venv.py313.bak/testing/tox.ini`). Consequences: ruff runs stable/non-preview
defaults, there are ZERO `per-file-ignores`, and `lint.pyflakes.allowed-unused-imports`
is empty. Every ruff finding in this repo is therefore a raw default-behaviour
finding, and introducing a config is a repo-wide behaviour change, never a
drive-by.

**4. Strongest deadness proof is textual occurrence count, not the linter.** Each
of the 7 names occurs EXACTLY ONCE in its file -- on its own import line. A name
with no second textual hit cannot be reached by a string annotation, a `# type:`
comment, a doctest, or an `eval`. That single grep beats enumerating F401 blind
spots one at a time.

**5. Removing one name from a multi-name `from` statement carries zero
side-effect risk** -- the module still imports, `sys.modules` still gets its
entry, the body still executes. Side-effect risk only exists for removals that
delete a WHOLE import statement. 2 of the 7 here are name-level, 5 are
statement-level (all stdlib `json`/`typing`).

**6. Do NOT widen to `__init__.py`.** F401's `__init__.py` autofix is in preview
and actively unstable (ruff issues #12513 infinite loop, #15805, #15858, #16609),
and `lint.ignore-init-module-imports` was deprecated in 0.4.4.

Brief: `handoff/current/research_brief_86.26.md`. Related:
[[immutable-criteria-must-be-green-able]] -- the F401 command here IS green-able
because it is scoped to the four files the change touches.
