# Research Brief -- step 86.26

**Tier:** simple (caller-specified). **Audit-class:** NO (coverage reported for information only).
**Date:** 2026-08-10. **Status:** COMPLETE (written incrementally, write-first).
**Gate:** PASSED -- 6 sources read in full, 26 URLs, recency scan performed.

## Research: Safely removing unused imports in Python -- re-exports, side-effect imports, ruff F401 blind spots, `__all__` / PEP 484 no-implicit-reexport, and proving a name is not consumed elsewhere

### Objective (verbatim from caller)

> Safely removing unused imports in Python: when is an apparently-unused import actually a
> re-export or an import-for-side-effect, what does ruff F401 miss or over-report, `__all__`
> and re-export conventions (PEP 484 no-implicit-reexport), and how to prove a name is not
> consumed elsewhere before deleting it

### Internal scope (verbatim from caller)

> The exact F401 findings over `backend/services/outcome_tracker.py`, `backend/agents/memory.py`,
> `backend/agents/bias_detector.py`, `backend/agents/conflict_detector.py` -- re-derive them with
> ruff rather than trusting the step text. For EACH flagged name, establish whether any other
> module imports that name FROM this module (a re-export), whether it appears in `__all__`, and
> whether the import has a side effect. `compute_benchmark_return` in outcome_tracker is the
> specific suspect.

### Search queries run (three-variant discipline, `.claude/rules/research-gate.md`)

| # | Variant | Query actually run |
|---|---|---|
| 1 | year-less canonical | `ruff F401 unused import re-export __all__ side effect false positive` |
| 2 | current-year frontier | `removing unused imports safely Python import side effects registration 2026` |
| 3 | last-2-year window | `ruff F401 preview mode __init__.py unused import behaviour change 2025` |

All three variants were run. The four normative sources (typing spec, PEP 484, mypy, CPython
language reference) were reached by **direct URL**, not by search -- disclosed here rather than
back-filled as fake queries.

---

### Read in full (>=5 required; counts toward the gate)

| # | URL | Accessed | Kind | Fetched how | Key quote / finding |
|---|---|---|---|---|---|
| 1 | https://docs.astral.sh/ruff/rules/unused-import/ | 2026-08-10 | official doc (ruff/Astral) | WebFetch, full page | F401 rationale: unused imports "add a performance overhead at runtime, and risk creating import cycles. They also increase the cognitive load of reading the code." THREE sanctioned escape hatches: redundant alias `from module import member as member`; an `__all__` entry; and `importlib.util.find_spec` instead of import-to-probe. **Fix safety is context-dependent**: removal is a SAFE fix in a regular module; in `__init__.py` third-party/stdlib fixes are "unsafe because the module's interface changes", and NO fix is offered when multiple `__all__` declarations exist. Options: `lint.ignore-init-module-imports`, `lint.pyflakes.allowed-unused-imports`. Preview mode is stricter -- "both different and more import statements being marked as unused". |
| 2 | https://typing.python.org/en/latest/spec/distributing.html | 2026-08-10 | official spec (Python Typing Spec, successor to PEP 484 prose) | WebFetch, full section | Normative re-export rule. Exactly three forms re-export: `import X as X` (module), `from Y import X as X` (symbol), and `from Y import *` (per `Y.__all__`, else all public globals). **"All other imported symbols are considered private by default."** `__all__` "overrides all other rules above, allowing imported symbols or symbols whose names begin with an underscore to be included in the interface." |
| 3 | https://mypy.readthedocs.io/en/stable/config_file.html | 2026-08-10 | official doc (mypy) | WebFetch, full page | `implicit_reexport` **defaults to `True`**: "By default, imported values to a module are treated as exported and mypy allows other modules to import them." When false, "mypy will not re-export unless the item is imported using from-as or is included in `__all__`." Worked example distinguishes `from foo import bar` (not re-exported) from `from foo import bar as bar` / `__all__ = ['bar']` (re-exported). "Note that mypy treats stub files as if this is always disabled." |
| 4 | https://docs.python.org/3/reference/import.html | 2026-08-10 | official doc (CPython language reference) | WebFetch, full page | Import is executable and side-effecting: "While certain side-effects may occur, such as the importing of parent packages, and the updating of various caches (including `sys.modules`), only the `import` statement performs a name binding operation." Module execution "is the key moment of loading in which the module's namespace gets populated." `sys.modules` is "a cache of all modules that have been previously imported" -- a second import returns the cached object WITHOUT re-executing the body. Submodule binding is itself a side effect: "When a submodule is loaded using any mechanism ... a binding is placed in the parent module's namespace to the submodule object." |
| 5 | https://peps.python.org/pep-0484/ | 2026-08-10 | PEP (peer-reviewed-tier normative standard) | WebFetch, full page | The origin of no-implicit-reexport: "Modules and variables imported into the stub are not considered exported from the stub unless the import uses the `import ... as ...` form or the equivalent `from ... import ... as ...` form." Clarified: "only names imported using the form `X as X` will be exported, i.e. the name before and after `as` must be the same." Exception: "all objects imported into a stub using `from ... import *` are considered exported." And "submodules automatically become exported attributes of their parent module when imported." PEP 484 does NOT discuss `__all__` in stubs -- that came later (see source 2). |
| 6 | https://docs.astral.sh/ruff/settings/ | 2026-08-10 | official doc (ruff/Astral) | WebFetch, full page | `lint.ignore-init-module-imports` -- **default `true`**, **deprecated in 0.4.4**: "Avoid automatically removing unused imports in `__init__.py` files. Such imports will still be flagged, but with a dedicated message suggesting that the import is either added to the module's `__all__` symbol, or re-exported with a redundant alias (e.g., `import os as os`)." It "will be removed in a future version because F401 now recommends appropriate fixes for unused imports in `__init__.py` (currently in preview mode)." `lint.pyflakes.allowed-unused-imports` -- **default `[]`**, a `list[str]` of imports exempted from F401. |

**Failed fetch, recorded honestly (does NOT count toward the gate):**
`https://pypi.org/project/autoflake/` -- WebFetch returned only PyPI's client-side error shell
("A required part of this site couldn't load"), no README content. Autoflake's stdlib-only-by-default
policy therefore appears in this brief ONLY as a search snippet (see snippet table), never as a
read-in-full claim.

### Identified but snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|---|---|---|
| https://github.com/astral-sh/ruff/issues/717 | issue tracker (quasi-official) | "Don't mark re-exported symbols as unused (F401)" -- the canonical ruff thread on the redundant-alias convention; superseded by the official rule doc (#1 above), which is the authoritative statement. |
| https://github.com/astral-sh/ruff/issues/18893 | issue tracker | "F401 (unused imports) False Negatives" -- evidence that F401 UNDER-reports in some shapes; directional only, no normative content. |
| https://github.com/astral-sh/ruff/issues/12780 | issue tracker | "F401 false positive when importing in a type-checking block" -- narrow `TYPE_CHECKING` interaction; not applicable to the four target files (none use `TYPE_CHECKING`). |
| https://github.com/charliermarsh/ruff/issues/1401 | issue tracker | Historic F401 false-positive report; superseded by current ruff 0.16.0 behaviour, which I measured directly instead. |
| https://github.com/charliermarsh/ruff/issues/2044 | issue tracker | Historic F401 false-positive report; same reason. |
| https://github.com/astral-sh/ruff/issues/15805 | issue tracker | Preview-mode aliasing behaviour on F401 fixes; relevant only if preview were enabled (it is not -- no ruff config exists in this repo). |
| https://www.jetbrains.com/help/inspectopedia/Python-Unused-imports-PyUnresolvedReferences.html | vendor doc (JetBrains) | Duplicates the side-effect-suppression guidance already covered by ruff + autoflake official docs; lower authority for this question. |
| https://bigdatagurus.wordpress.com/2025/08/16/the-ghost-in-the-machine-unused-python-imports/ | community blog | Tier-5 community source; its substantive claim (imports are executable statements, so combine automation with tests) is corroborated by the Python language reference read in full. |
| https://www.pythontutorials.net/blog/is-there-a-way-to-remove-unused-imports-for-python-in-vs-code/ | community blog | Tier-5, tooling-workflow only, no normative content. |
| https://intellij-support.jetbrains.com/hc/en-us/community/posts/115000035164-Unconfigurable-Auto-Remove-of-Unused-Python-Imports | forum | Tier-5 forum thread; IDE-behaviour complaint, not applicable. |
| https://pypi.org/project/autoflake/ | tool doc | **Fetch ATTEMPTED and FAILED** (PyPI error shell). Snippet-level only: autoflake "by default ... only removes unused imports for modules that are part of the standard library, as other modules may have side effects that make them unsafe to remove automatically." Corroborates the risk model but is NOT read-in-full. |
| https://github.com/astral-sh/ruff/issues/12513 | issue tracker | F401 preview-mode infinite fix loop in `__init__.py` with `__all__`; preview is off here (no config), so not applicable. |
| https://github.com/astral-sh/ruff/issues/16609 | issue tracker | "No way to autofix F401 errors in `__init__.py`"; none of the four target files is `__init__.py`. |
| https://github.com/astral-sh/ruff/issues/15858 | issue tracker | "F401 is not doing the 'right' thing with `__init__.py`"; same reason. |
| https://github.com/astral-sh/ruff/issues/15571 | issue tracker | "Ruff is not fixing F401"; superseded by direct measurement on 0.16.0 (all 7 reported fixable). |
| https://www.cyber.airbus.com/en/newsroom/stories/2025-10-python-code-quality-with-ruff-one-step-at-a-time-part-3 | industry blog (Airbus Cyber, Oct 2025) | Tier-4 adoption narrative; no normative content beyond the ruff docs read in full. |
| https://readmedium.com/autoflake-remove-unused-imports-unused-variables-from-python-code-4774c1117099 | content mirror | Mirror of a third-party autoflake write-up; low authority, superseded by the ruff official docs. |
| https://github.com/serious-scaffold/ss-python/pull/685 | repo PR | Incidental config-bump PR surfaced by the 2025 query; no bearing on the question. |
| https://pypi.org/project/ruff/0.0.178/ | package index | Ancient ruff release page; irrelevant to 0.16.0 behaviour. |
| https://www.jetbrains.com/help/inspectopedia/Python-Unused-Imports-Pyunresolvedreferences-Pyunresolvedreferences.html | vendor doc | Near-duplicate of the other Inspectopedia page already listed. |

---

## Internal code inventory (re-derived, not trusted from the step text)

**Tooling:** `ruff 0.16.0`, run from the repo venv. **Python 3.14.4.**
**No ruff configuration file exists anywhere in the repo** -- `find . -maxdepth 3 \( -name pyproject.toml -o -name ruff.toml -o -name .ruff.toml -o -name setup.cfg \)` (excluding `.venv`/`node_modules`) returns only `./.venv.py313.bak/testing/tox.ini`, which is a backup venv, not project config. Consequences that matter for this step: ruff runs on **stable (non-preview) defaults**, there are **no `per-file-ignores`**, and `lint.pyflakes.allowed-unused-imports` is empty. Any behaviour attributed to preview mode in the ruff docs does NOT apply here.

### Re-derived F401 finding set -- 7 findings, exactly matching the step text

Command: `ruff check --select F401 --no-cache --output-format concise <4 files>` -> `Found 7 errors. [*] 7 fixable with the --fix option.`

| # | File:line:col | Flagged name | Statement shape | Other names on the same statement |
|---|---|---|---|---|
| 1 | `backend/agents/bias_detector.py:12:8` | `json` | `import json` (whole statement) | none |
| 2 | `backend/agents/conflict_detector.py:8:8` | `json` | `import json` (whole statement) | none |
| 3 | `backend/agents/conflict_detector.py:11:20` | `typing.Optional` | `from typing import Optional` (whole statement) | none |
| 4 | `backend/agents/memory.py:12:8` | `json` | `import json` (whole statement) | none |
| 5 | `backend/services/outcome_tracker.py:11:32` | `datetime.timedelta` | `from datetime import datetime, timedelta, timezone` | **`datetime`, `timezone` -- both live** |
| 6 | `backend/services/outcome_tracker.py:12:20` | `typing.Optional` | `from typing import Optional` (whole statement) | none |
| 7 | `backend/services/outcome_tracker.py:17:63` | `perf_metrics.compute_benchmark_return` | `from backend.services.perf_metrics import compute_return_pct, compute_benchmark_return, beat_benchmark as _beat_benchmark` | **`compute_return_pct`, `beat_benchmark as _beat_benchmark` -- both live** |

Line numbers are current as of 2026-08-10 and match the masterplan's "RE-DERIVE, line numbers drift" caveat exactly; nothing has drifted.

### Per-name safety verdict (the three questions the caller asked, answered for each name)

**Q1: Is any flagged name re-exported (imported FROM this module by another module)?**
Measured with `grep -rEn "from (backend\.services\.outcome_tracker|backend\.agents\.memory|backend\.agents\.bias_detector|backend\.agents\.conflict_detector) import .*(json|Optional|timedelta|compute_benchmark_return)"` repo-wide (excluding `.venv`, `.git`). **Result: ZERO matches.** The single grep hit was `.claude/masterplan.json:17587`, which is English prose about `detect_biases` -- not a Python import statement and not one of the seven flagged names. So **no flagged name is a re-export.**

**Q2: Does any flagged name appear in `__all__`?**
`grep -n "__all__"` over all four files: **no `__all__` in any of them.** This is not because the codebase lacks the idiom -- `grep -rn "^__all__" backend/` returns **94** declarations (e.g. `backend/autoresearch/gate.py:86`, `backend/metrics/sortino.py:136`, `backend/middleware/catch_all_errors.py:63`). The four target modules simply never adopted it, so the `__all__` escape hatch is not in play and cannot be silently broken.

**Q3: Does any flagged import have a side effect?**
All four modules involved are pure: `json` and `typing` and `datetime` are stdlib with no registration/plugin side effects, and `backend.services.perf_metrics` is a pure-function metrics module. Critically, for the two multi-name statements (#5, #7) the **module-level import survives the edit** because other names on the same statement are live -- so even if `datetime` or `backend.services.perf_metrics` had an import side effect, removing one name from the list could not lose it. That reduces the side-effect question to the four whole-statement removals (#1, #2, #3, #4, #6), all of which are stdlib `json`/`typing`.

### Additional evidence that the names are genuinely dead (F401 blind-spot sweep)

The known F401 blind spots are names referenced only from places the linter does not resolve. I checked each against the four files:

| Blind spot | Check run | Result |
|---|---|---|
| Name used only in a `# type:` comment | `grep -nE "# *type:"` over the 4 files | **none** |
| Name used only inside `eval()` / `exec()` | `grep -nE "eval\(\|exec\("` | **none** |
| Name resolved at runtime via `typing.get_type_hints()` | `grep -n "get_type_hints"` | **none** |
| Deferred/string annotations (`from __future__ import annotations`) | `grep -n "from __future__"` | **none in any of the 4 files** |
| Dynamic re-export via `importlib` / `getattr` / `__import__` | `grep -rEn "importlib.*(outcome_tracker\|agents\.memory\|bias_detector\|conflict_detector)\|getattr\((outcome_tracker\|memory\|bias_detector\|conflict_detector)"` over `backend/` + `scripts/` | **none** |

**Strongest single piece of evidence:** each flagged name occurs **exactly once** in its own file -- on the import line itself.
- `grep -nE "\bjson\b"` returns exactly one line per file: `bias_detector.py:12`, `memory.py:12`, `conflict_detector.py:8`.
- `grep -n "Optional"` returns exactly one line per file: `outcome_tracker.py:12`, `conflict_detector.py:11`.
- `grep -nE "\btimedelta\b"` over `outcome_tracker.py` returns exactly one line: `:11`.

A name with zero textual occurrences beyond its own binding site cannot be reached by a string annotation, a `type:` comment, a doctest, or an `eval` -- all of those would have produced a second textual hit. This is a materially stronger proof than "ruff says unused".

### `compute_benchmark_return` -- the named suspect, RESOLVED: genuinely dead, NOT a re-export

The masterplan flagged this one as "may be a re-export other modules import FROM outcome_tracker". **It is not.** Every occurrence in the repo (excluding `.venv`/`node_modules`/`.git`):

| Location | Nature |
|---|---|
| `backend/services/perf_metrics.py:536` | **the definition** (`def compute_benchmark_return(holding_days: int, annual_rate: float = 0.10) -> float:`) |
| `backend/services/perf_metrics.py:560` | internal call, from inside `beat_benchmark` in the SAME module |
| `backend/services/outcome_tracker.py:17` | **the dead import under review** |
| `backend/tests/test_dod4_tier1_coverage_investment.py:346,348,350,352` | test -- and it imports `from backend.services.perf_metrics import compute_benchmark_return`, i.e. **from the defining module, NOT from outcome_tracker** |
| `CHANGELOG.md:3530`, `docs/coverage_tier_overrides.md:112`, `handoff/current/evaluator_critique_86.22.md`, `.claude/masterplan.json:24361` | prose/documentation only |

The benchmark comparison that `outcome_tracker` actually performs is not lost by the deletion: `outcome_tracker.py` calls `_beat_benchmark(return_pct, holding_days)`, and `beat_benchmark` internally calls `compute_benchmark_return` at `perf_metrics.py:560`. The geometric benchmark math therefore still executes -- through the alias that IS used. Removing the name from the import list changes nothing at runtime.

### Consumers of the four modules (blast-radius map)

| Module | Live production consumers | Test consumers |
|---|---|---|
| `backend/agents/bias_detector.py` | `backend/agents/orchestrator.py:38` -- `from backend.agents.bias_detector import detect_biases` | `backend/tests/test_phase_86_22_outcome_tracker_bias_detector_vocabulary.py:149` |
| `backend/agents/conflict_detector.py` | `backend/agents/orchestrator.py:43` -- `from backend.agents.conflict_detector import detect_conflicts` | `test_phase_86_22_...py:180,191` (`_check_recommendation_alignment`) |
| `backend/services/outcome_tracker.py` | **no live `from backend.services.outcome_tracker import ...` in `backend/` or `scripts/`** -- every grep hit is in `handoff/archive/*`, `handoff/harness_log.md`, or `.claude/.masterplan.json.bak.*` (historical phase-16.21/16.26 verification commands) | -- |
| `backend/agents/memory.py` | same shape: hits are archival handoff docs + masterplan backups, not live code | -- |

In all cases the imported names are `detect_biases` / `detect_conflicts` / `_check_recommendation_alignment` / `evaluate_recent` / `retrieve_memories` -- **none of the seven flagged names.** No consumer can break.


---

## Recency scan (last 2 years, 2024-2026) -- MANDATORY SECTION

Searched with `removing unused imports safely Python import side effects registration 2026`
(current-year frontier) and `ruff F401 preview mode __init__.py unused import behaviour change 2025`
(last-2-year window). **Result: 3 new findings in the 2024-2026 window; NONE supersedes the canonical
re-export rule, but two change the operating envelope for this step.**

1. **F401's `__init__.py` behaviour is actively in flux (2024-2026).** `lint.ignore-init-module-imports`
   was **deprecated in ruff 0.4.4** and is slated for removal because "F401 now recommends appropriate
   fixes for unused imports in `__init__.py` (currently in preview mode)"
   (https://docs.astral.sh/ruff/settings/, accessed 2026-08-10). Multiple 2025 issue reports
   (astral-sh/ruff #12513, #15805, #15858, #16609) show preview-mode `__init__.py` autofix converting
   imports to redundant aliases and, in one case, looping. **Impact on 86.26: NONE -- none of the four
   target files is an `__init__.py`, and no ruff config exists so preview is off. But it is a live
   reason NOT to widen this step's scope to package `__init__.py` files.**
2. **`lint.pyflakes.allowed-unused-imports` exists** as a config-level allow-list (default `[]`).
   This is a newer, cleaner alternative to scattering `# noqa: F401`. Noted for completeness; the
   masterplan explicitly forbids silencing the detector in this step, so it must NOT be used here.
3. **No change to the re-export rule itself.** The canonical convention (PEP 484's `X as X`, 2015)
   is unchanged and has merely been re-homed into the living Python Typing Spec
   (https://typing.python.org/en/latest/spec/distributing.html), which added the `__all__` override
   and the wildcard clause. mypy's `implicit_reexport` still defaults to `True`. Nothing in the
   2024-2026 window invalidates the older canonical sources.

---

## Key findings (each cited per-claim)

1. **"Unused" is a *static* verdict; import is a *dynamic* statement.** Only the `import` statement
   performs name binding, but importing still mutates global state: "certain side-effects may occur,
   such as the importing of parent packages, and the updating of various caches (including
   `sys.modules`)" (CPython language reference, https://docs.python.org/3/reference/import.html,
   accessed 2026-08-10). This is exactly why an unused import can still be load-bearing -- the value
   of the import may be the *execution of the module body*, not the bound name.
2. **Only three forms constitute a re-export; everything else is private.** `import X as X`,
   `from Y import X as X`, and `from Y import *`. "All other imported symbols are considered private
   by default" (Python Typing Spec, https://typing.python.org/en/latest/spec/distributing.html,
   accessed 2026-08-10). Origin: "only names imported using the form `X as X` will be exported, i.e.
   the name before and after `as` must be the same" (PEP 484, https://peps.python.org/pep-0484/,
   accessed 2026-08-10).
3. **BUT the *runtime* has no such rule -- the convention is type-checker-facing only.** mypy's
   `implicit_reexport` **defaults to `True`**: "By default, imported values to a module are treated as
   exported and mypy allows other modules to import them" (https://mypy.readthedocs.io/en/stable/config_file.html,
   accessed 2026-08-10). **Consequence: the absence of an `X as X` alias does NOT prove nothing
   re-imports the name.** At runtime `from mod import name` always works regardless of convention. So
   the alias/`__all__` check is *necessary but not sufficient* -- a repo-wide consumer grep is the
   only sound proof. This is the single most important nuance for 86.26.
4. **Ruff sanctions exactly three ways to keep an intentionally-unused import**, and using
   `importlib.util.find_spec` instead of import-to-probe (https://docs.astral.sh/ruff/rules/unused-import/,
   accessed 2026-08-10). Fix safety is contextual: safe in a regular module, "unsafe because the
   module's interface changes" for third-party/stdlib imports in `__init__.py`.
5. **Removing ONE name from a multi-name `from` statement cannot lose an import side effect**, because
   the module is still imported and `sys.modules` still gets its entry -- the module body executes on
   first import either way (https://docs.python.org/3/reference/import.html, accessed 2026-08-10).
   Side-effect risk is therefore confined to removals that delete an entire import statement.

## Consensus vs debate (external)

**Consensus:** the `X as X` redundant-alias convention is settled and identical across PEP 484, the
Python Typing Spec, mypy, and ruff. All four sources read in full agree on the same three re-export
forms and on `__all__` as the override.

**Genuine debate / divergence:** *how aggressive automated removal should be.* Ruff removes unused
imports as a **safe** fix in ordinary modules; autoflake's default is the opposite -- stdlib-only,
precisely because non-stdlib modules "may have side effects that make them unsafe to remove
automatically" (snippet-only, https://pypi.org/project/autoflake/). These two defaults encode
opposite risk appetites for the *same* operation. For 86.26 the divergence is moot in one direction
and instructive in the other: six of the seven names are stdlib (`json`, `Optional`, `timedelta`),
which even autoflake's conservative default would remove; the seventh
(`compute_benchmark_return`) is first-party, which autoflake would NOT touch by default -- so it
earns the extra proof it has been given below.

## Pitfalls (from the literature), mapped to whether each applies here

| Pitfall | Applies to 86.26? |
|---|---|
| Import kept for a registration side effect (Django signals, plugin registries, `import module_that_registers`) | **NO** -- all four modules are stdlib or a pure metrics module; two of the seven removals do not even delete the statement. |
| Name is a re-export consumed via `from thismodule import name` | **NO** -- repo-wide grep returns zero consumers for all seven names. |
| Name listed in `__all__` | **NO** -- none of the four files declares `__all__` at all. |
| Name referenced only in a string/deferred annotation | **NO** -- each name occurs exactly once in its file (its own import line). |
| Name referenced only in a `# type:` comment / `eval` / `exec` / `get_type_hints` | **NO** -- all four greps empty. |
| `__init__.py` interface change (the ruff "unsafe fix" case) | **NO** -- no target file is an `__init__.py`. This is the main reason not to widen scope. |
| Absence of an `X as X` alias mistaken for proof of non-consumption | **GUARDED** -- see key finding 3; the consumer grep, not the alias convention, is what proves it here. |
| Silencing the detector instead of removing the code (`# noqa: F401`, `allowed-unused-imports`) | **FORBIDDEN by the masterplan step text** (`.claude/masterplan.json:24361`: "DO NOT add a per-file noqa to make the gate green"). |

## Application to pyfinagent (external findings -> internal file:line anchors)

- **All 7 F401 findings are safe to delete.** Verdict basis: zero re-export consumers (grep), zero
  `__all__` membership, zero side-effect exposure, and exactly-one textual occurrence per name.
- **`compute_benchmark_return` at `backend/services/outcome_tracker.py:17` -- the masterplan's named
  suspect -- is NOT a re-export and is safe to remove.** The only non-defining consumer,
  `backend/tests/test_dod4_tier1_coverage_investment.py:346`, imports it from
  `backend.services.perf_metrics` (its defining module, `perf_metrics.py:536`), not from
  `outcome_tracker`. The benchmark math outcome_tracker actually relies on still runs, because
  `beat_benchmark` calls `compute_benchmark_return` internally at `perf_metrics.py:560` and
  outcome_tracker imports and uses that as `_beat_benchmark`.
- **Two removals are name-level, five are statement-level.** `outcome_tracker.py:11` keeps
  `datetime`/`timezone`; `outcome_tracker.py:17` keeps `compute_return_pct` and
  `beat_benchmark as _beat_benchmark`. Per key finding 5 these two carry **zero** side-effect risk.
  The five statement-level deletions (`json` x3 at `bias_detector.py:12`, `conflict_detector.py:8`,
  `memory.py:12`; `Optional` x2 at `conflict_detector.py:11`, `outcome_tracker.py:12`) remove the
  stdlib module import entirely -- still safe, `json`/`typing` have no registration side effects.
- **Do NOT adopt `__all__` in these files as part of this step.** It is a 94-occurrence repo idiom
  (`backend/autoresearch/gate.py:86`, `backend/metrics/sortino.py:136`, etc.) but adding it here
  would be a public-interface change, not a dead-code removal, and would exceed the step's scope.
- **Do NOT widen the ruff rule set or add config.** There is currently **no ruff config file in the
  repo at all**, so introducing one is a repo-wide behaviour change well outside this step; and the
  masterplan text forbids widening (`.claude/masterplan.json:24361`).
- **Proof obligation for the Q/A gate:** the durable evidence is the *delta*, not the absolute count.
  Before: 7 F401 over these four files. After: expected 0. The 86.22 Q/A already established
  pre==post per file against rev `4b7dab7b`, so this step's own delta is `7 -> 0` with no other
  file touched. Re-running `ruff check --select F401 <4 files>` is the natural verification command,
  and unlike the phase-81.0 postmortem case it is **green-able**, because it is scoped to exactly the
  files the change touches rather than the whole repo.
- **Regression guard suggestion for Main (not a research finding, flagged as advice):** the four
  modules have live consumers at `backend/agents/orchestrator.py:38` (`detect_biases`) and `:43`
  (`detect_conflicts`), plus `backend/tests/test_phase_86_22_outcome_tracker_bias_detector_vocabulary.py:149,180,191`.
  Importing all four modules and re-running that 86.22 test file is a cheap proof that no import
  statement was over-deleted.

---

## Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch -- **6** (ruff rule doc, Python
      Typing Spec, mypy config doc, CPython language reference, PEP 484, ruff settings doc). A 7th
      (autoflake on PyPI) was attempted and FAILED; it is excluded from the count and recorded as
      snippet-only.
- [x] 10+ unique URLs total -- **26** (6 read in full + 20 snippet-only).
- [x] Recency scan (last 2 years) performed + reported -- see the dedicated section (3 findings).
- [x] Full pages read (not abstracts) for the read-in-full set -- all 6 were full-page WebFetches; no
      arXiv PDFs were involved, so the html/ar5iv/pdfplumber chain was not needed.
- [x] file:line anchors for every internal claim -- every internal assertion carries a `file:line` or
      a reproduced command.

Soft checks:
- [x] Internal exploration covered every relevant module -- all 4 targets, plus `perf_metrics.py`,
      `orchestrator.py`, and the two consuming test files.
- [x] Contradictions / consensus noted -- see "Consensus vs debate": ruff-safe-by-default vs
      autoflake-stdlib-only-by-default is a real divergence in risk appetite, and mypy's
      `implicit_reexport=True` default contradicts the naive reading of the `X as X` convention.
- [x] All claims cited per-claim (URL + access date inline, or the exact command run).

**Scope-honesty notes (disclosed, not buried):**
- The ruff settings page defers its default-rule-selection list to a separate page I did not fetch, so
  I make **no claim** about which rules ruff enables by default here. It does not matter: I passed
  `--select F401` explicitly, so the measurement is independent of the default set.
- This brief exceeds the `simple` tier's ~300-word guidance. The overage is entirely tables and
  per-name evidence, which the caller's internal scope explicitly required (three properties
  established for each of seven names). The external analysis itself is held to `simple` depth.

## JSON envelope

```json
{
  "tier": "simple",
  "external_sources_read_in_full": 6,
  "snippet_only_sources": 20,
  "urls_collected": 26,
  "recency_scan_performed": true,
  "internal_files_inspected": 8,
  "coverage": {
    "audit_class": false,
    "rounds": 2,
    "dry_rounds": 0,
    "K_required": 2,
    "new_findings_last_round": 3,
    "dry": false
  },
  "summary": "All 7 ruff-0.16.0 F401 findings re-derived and confirmed safe to delete. No __all__ in any of the four files; zero repo-wide consumers import any flagged name FROM these modules; no side-effect exposure. compute_benchmark_return is NOT a re-export -- its only consumer imports it from perf_metrics, and the benchmark math survives via beat_benchmark. Key nuance: mypy's implicit_reexport defaults to True, so the absence of an 'X as X' alias does not prove non-consumption -- the consumer grep does.",
  "brief_path": "handoff/current/research_brief_86.26.md",
  "gate_passed": true
}
```

**Status: COMPLETE.**
