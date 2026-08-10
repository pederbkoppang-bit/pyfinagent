# phase-86.26 -- CONTRACT

**Step:** P3 -- dead imports in the learn-loop modules, surfaced by a Q/A lint
gate that cannot currently pass.

**Research gate:** PASSED -- `wf_1c1bd9f0-b5f`, tier `simple`, **6 sources read
in full**, 26 URLs, 20 snippet-only, recency scan performed, 8 internal files.
Brief: `handoff/current/research_brief_86.26.md` (32,558 chars); the gate script
independently re-read it and confirmed all 6 claimed URLs appear.

---

## 1. Why this is worth a step rather than a drive-by deletion

`qa.md` rule 1a reads a non-zero lint exit as FAIL. With 7 F401 findings
standing in these modules, **every future step that touches them must either
re-derive a per-file pre/post delta by hand or accept a red gate.** The gate has
been made unusable on this file set -- the same shape as an immutable
verification command that is already red for unrelated reasons (phase-81.0; auto
-memory `immutable_criteria_must_be_green_able`). Removing the dead imports
restores the gate rather than merely tidying.

## 2. What the research changes about the approach

The load-bearing nuance, from the brief: **mypy's `implicit_reexport` defaults
to `True`, and CPython itself has no re-export rule at all.** So the *absence*
of an `X as X` alias or an `__all__` entry does **NOT** prove a name is
unconsumed -- any module can import any name from any other module. The only
sound proof is to ask the repository who imports that name FROM that module.
That is what `scripts/qa/verify_unused_imports_86_26.py` does, by AST rather
than grep (a bare grep matches the definition site and every unrelated use).

Also from the brief: no target file is an `__init__.py`, where ruff's F401
autofix is preview-gated and unstable; and the repo has no ruff config, so
preview mode is off.

## 3. Immutable success criteria -- copied VERBATIM from `.claude/masterplan.json`

1. The finding count is RE-DERIVED, not copied from this step text; if it
   differs from 7, report the difference rather than the expected number.
2. For EACH name removed, show the grep proving no other module imports it from
   this one. A name that IS re-exported is kept and the reason recorded.
3. No `noqa` comment and no change to the rule set is used to reach green.
4. The immutable command above exits 0.
5. The full backend test suite shows no NEW failure attributable to the
   removals, established by a diff of the failure sets and not by a count.

**Immutable verification command** (unmodified):

```
bash -c 'source .venv/bin/activate && uvx ruff check --select F821,F401,F811 \
  backend/services/outcome_tracker.py backend/agents/memory.py \
  backend/agents/bias_detector.py backend/agents/conflict_detector.py \
  backend/agents/skill_optimizer.py backend/api/portfolio.py \
  backend/slack_bot/formatters.py backend/services/recommendation_vocab.py'
```

## 4. Measured before the contract was written

Re-derived with ruff, matching the step text exactly (7, no line drift):

| file | line | name |
|---|---|---|
| `agents/bias_detector.py` | 12 | `json` |
| `agents/conflict_detector.py` | 8 | `json` |
| `agents/conflict_detector.py` | 11 | `typing.Optional` |
| `agents/memory.py` | 12 | `json` |
| `services/outcome_tracker.py` | 11 | `datetime.timedelta` |
| `services/outcome_tracker.py` | 12 | `typing.Optional` |
| `services/outcome_tracker.py` | 17 | `perf_metrics.compute_benchmark_return` |

**The named suspect resolves clean.** `compute_benchmark_return` has exactly one
non-defining consumer, `test_dod4_tier1_coverage_investment.py:346`, and it
imports the name **from `perf_metrics`** -- the origin -- not through
`outcome_tracker`. So it is not a re-export. No file in scope declares
`__all__`.

**The verification method is itself validated** before its clean report is
trusted: `is_buy_intent` and `canonical_recommendation` (wired into seven
modules by phase-86.22) must return several hits, and two nonexistent probes
must return none. Measured 8, 9, 0, 0. A scanner that returned zero for
everything -- a typo in the module path, say -- would otherwise declare every
import safe to delete.

## 5. Traps

- **Do not `ruff --fix` blindly.** The criterion requires a per-name consumer
  proof, and an autofix produces no evidence.
- **Do not add `noqa`** or narrow the rule set to reach green (criterion 3).
- **`json` appears three times** in three different files -- each is its own
  removal with its own proof, not one finding.
- **Count the delta, not the total**, when checking the suite (criterion 5): the
  tree carries 14 pre-existing failures whose membership churns with the wall
  clock (phase-86.24), so a bare count would be misleading in both directions.
- Two removals are **name-level** (the `from X import a, b, c` line survives
  minus one name); five are whole-statement. Do not delete a whole line that
  still carries a used name.

## 6. Plan

1. Re-derive the findings (done above; re-run at fix time).
2. Prove each name's safety with the validated consumer scan.
3. Remove the 7 names -- name-level where the statement carries survivors.
4. AST-parse and import-smoke every touched module.
5. Immutable command -> expect exit 0.
6. Full suite; diff the failure SET against the pre-change baseline.
7. Q/A; transcribe verbatim; log; flip.

## 7. References

- `handoff/current/research_brief_86.26.md`
- `scripts/qa/verify_unused_imports_86_26.py` (validated consumer scan)
- ruff F401 docs; PEP 484 re-export; mypy `implicit_reexport`
