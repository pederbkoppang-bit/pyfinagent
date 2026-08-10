# phase-86.26 -- GENERATE

**Step:** P3 -- dead imports in the learn-loop modules, surfaced by a Q/A lint
gate that cannot currently pass.
**Contract:** `handoff/current/contract_86.26.md`
**Research:** `handoff/current/research_brief_86.26.md` (gate PASSED,
`wf_1c1bd9f0-b5f`, 6 sources read in full, 26 URLs)

---

## 1. Criterion 1 -- the count, RE-DERIVED

Run with ruff rather than copied from the step: **7 findings, and they match the
step text exactly with no line drift.** Reported here as a confirmation rather
than an assumption, because the criterion requires the difference to be
reported if there is one. There is none.

| file | line | name | removal |
|---|---|---|---|
| `agents/bias_detector.py` | 12 | `json` | whole statement |
| `agents/conflict_detector.py` | 8 | `json` | whole statement |
| `agents/conflict_detector.py` | 11 | `typing.Optional` | whole statement |
| `agents/memory.py` | 12 | `json` | whole statement |
| `services/outcome_tracker.py` | 11 | `datetime.timedelta` | **name-level** |
| `services/outcome_tracker.py` | 12 | `typing.Optional` | whole statement |
| `services/outcome_tracker.py` | 17 | `perf_metrics.compute_benchmark_return` | **name-level** |

Two are name-level: their `from X import a, b, c` statement carries surviving
names and only the dead one is dropped. Deleting those lines wholesale would
have removed `datetime`, `timezone`, `compute_return_pct` and `_beat_benchmark`
along with them.

## 2. Criterion 2 -- the per-name proof, and why a grep would not do

`scripts/qa/verify_unused_imports_86_26.py` asks the repository, for each name,
whether ANY module does `from <this module> import <name>` -- by AST, because a
bare grep for the name matches its definition site and every unrelated use and
would report a false positive on every single one.

```
file                                         line  name                              re-exported?              verdict
backend/agents/bias_detector.py                12  json                              none                      safe to remove
backend/agents/conflict_detector.py             8  json                              none                      safe to remove
backend/agents/conflict_detector.py            11  Optional                          none                      safe to remove
backend/agents/memory.py                       12  json                              none                      safe to remove
backend/services/outcome_tracker.py            11  timedelta                         none                      safe to remove
backend/services/outcome_tracker.py            12  Optional                          none                      safe to remove
backend/services/outcome_tracker.py            17  compute_benchmark_return          none                      safe to remove

7 safe / 0 needing a decision
```

**The named suspect resolves clean.** `compute_benchmark_return` has exactly one
non-defining consumer -- `test_dod4_tier1_coverage_investment.py:346` -- and it
imports the name **from `perf_metrics`**, the origin, not through
`outcome_tracker`. It was never a re-export. `outcome_tracker`'s benchmark maths
survives via `beat_benchmark`, which is still imported and still used.

No file in scope declares `__all__`.

### The method is validated before its clean report is believed

A scanner that returns zero for *everything* -- a typo in the module path, say
-- would declare every import safe to delete and look exactly like this one.
So it is probed in both directions first:

```
METHOD VALIDATION -- can the consumer-finder actually find consumers?
  OK    backend.services.recommendation_vocab.is_buy_intent             hits=8   expected=some
  OK    backend.services.recommendation_vocab.canonical_recommendation  hits=9   expected=some
  OK    backend.services.recommendation_vocab.no_such_name_xyzzy        hits=0   expected=none
  OK    backend.does.not.exist.is_buy_intent                            hits=0   expected=none
  Method validated in both directions.
```

`is_buy_intent` is the known-positive precisely because phase-86.22 wired it
into seven modules last night.

### What the research changed about the approach

From the brief, and it is load-bearing: **mypy's `implicit_reexport` defaults to
`True` and CPython has no re-export rule at all.** So the absence of an
`X as X` alias or an `__all__` entry proves nothing -- any module may import any
name from any other. Had I reasoned from "there is no explicit re-export
marker", I would have been asserting a convention the language does not enforce.
Only the consumer scan settles it.

**Stated limit:** the scan finds STATIC `from X import Y`. A dynamic
`getattr(module, "name")` or an `importlib` lookup would not appear. The script
prints this itself rather than leaving it implicit.

## 3. Criterion 3 -- no `noqa`, no rule-set change

```
$ git diff -U0 -- backend/ | grep -c noqa
0
```

The rule set in the immutable command is untouched, and the repo has no ruff
config at all (measured in the brief), so nothing was narrowed elsewhere either.

## 4. Criterion 4 -- the immutable command

```
$ bash -c 'source .venv/bin/activate && uvx ruff check --select F821,F401,F811 \
    backend/services/outcome_tracker.py backend/agents/memory.py \
    backend/agents/bias_detector.py backend/agents/conflict_detector.py \
    backend/agents/skill_optimizer.py backend/api/portfolio.py \
    backend/slack_bot/formatters.py backend/services/recommendation_vocab.py'
All checks passed!
exit=0
```

**This is the point of the step.** The gate now passes on this file set, so a
future step touching these modules gets a usable signal instead of a red gate it
must reason around.

## 5. Criterion 5 -- the failure SET, diffed

The criterion says "established by a diff of the failure sets and not by a
count", and that wording matters here: the tree carries 14 pre-existing failures
whose *membership* churns with the wall clock (phase-86.24, filed tonight), so a
count can match while the membership has moved.

```
before: 14 failed, 3291 passed
after : 14 failed, 3303 passed

NEW  (attributable to the removals): (none)
GONE (attributable to the removals): (none)
```

**The failure set is identical, member for member.** The +12 passed are the
phase-86.12 tests committed between the two runs.

## 6. Diff

```
backend/agents/bias_detector.py     | 1 -
backend/agents/conflict_detector.py | 2 --
backend/agents/memory.py            | 1 -
backend/services/outcome_tracker.py | 5 ++---
4 files changed, 2 insertions(+), 7 deletions(-)
```

The two insertions are the rewritten name-level import lines. All four modules
AST-parse and import cleanly.

## 7. Not claimed

- **No behaviour change.** Nothing but import statements was touched.
- **The scan cannot see dynamic access** (`getattr`, `importlib`) -- stated
  above and printed by the script.
- **This does not make the lint gate green repo-wide**, only on the eight files
  in the immutable command's scope.
