# Contract -- 67.2 Bug-catching upgrade (skill heuristic + NameError fix)

Step: masterplan phase-67 / 67.2 (P0, depends_on 67.1). Research gate: PASSED
(moderate tier; research_brief_67_2.md -- 6 sources read in full, 14 URLs, recency
scan done, 11 internal files).

## Research-gate summary

- Bug re-confirmed live in venv: `parse_llm_classification('not json {')` ->
  `NameError: name 'json' is not defined`. agent_definitions.py imports only
  dataclasses/enum/typing/resolve_model (:23-27); :396 references json.JSONDecodeError.
  json_io.py:15-16 EXPLICITLY documents call sites are expected to keep `import json`
  for their except clauses -- the module simply never did.
- BLAST RADIUS: sole caller multi_agent_orchestrator.py:982 (_classify_via_llm), NOT
  wrapped in try/except -- on any non-JSON Communication-agent output the NameError
  propagates into the Slack/iMessage classify path; the graceful-default at :397-402
  is dead code today.
- SECOND latent break (researcher): `json_io.loads("5")` -> int; `data.get(...)` at
  :368 raises AttributeError -- also uncaught by the current tuple.
- Fix variant chosen: (a) `import json` at module top + broaden tuple to
  `(json.JSONDecodeError, AttributeError, KeyError, TypeError)` -- matches 5 sibling
  modules (planner_agent:13, risk_debate:15, orchestrator:17, debate:9,
  evaluator_agent:30 all pair top-level `import json` with json_io use). Variants (b)
  ValueError-widening and (c) import-from-json_io rejected (off house style; json_io
  exports no JSONDecodeError).
- Heuristic home: Dimension 3 (Code quality) table; severity WARN -> BLOCK when a live
  unverified consumer is found by grep; plus Top-15 append as #16. Negation list
  disambiguates from rename-as-refactor (Dim 4) and exempts additive/flag-off/private/
  grep-verified changes.
- 67.1 interlock: pytest-timeout 2.4.0 installed by 67.1 (this step's immutable
  verification command uses --timeout=60 and could not pass without it). ruff gate
  (67.1) covers the F821 class deterministically; the heuristic covers what F821
  provably cannot see (renamed keys, argv-vs-stdin, SDK-kwarg-vs-CLI-flag, casing).
- Bonus from 67.1's teeth demo: F401 unused `typing.Optional` at :25 -- clean it in
  the same diff.

## Hypothesis (falsifiable)

Importing json + broadening the except tuple restores the designed graceful-default
path (malformed LLM routing output -> Main-routed ClassificationResult instead of a
propagating NameError), provable by a behavioral test that fails on the old code and
passes on the new; and a consumer-contract-break heuristic in the skill gives Q/A a
named check for the operator's most frequent shipped-bug class.

## Success criteria (verbatim from .claude/masterplan.json 67.2 -- IMMUTABLE)

1. ".claude/skills/code-review-trading-domain/SKILL.md gains a consumer-contract-break
   heuristic (interface/CLI-flag/output-shape/schema change without every consumer
   verified; the Q/A greps consumers itself) with a severity rating and a negation
   list; every pre-existing heuristic name remains present (criteria-erosion guard)"
2. "backend/agents/agent_definitions.py::parse_llm_classification degrades gracefully
   on malformed input: the exception tuple references only imported names, and the
   documented repro now returns the default Main-routed ClassificationResult instead
   of raising NameError"
3. "A behavioral test exercises the malformed-classification path (not tautological,
   not over-mocked) and passes in the venv"
4. "Fresh Q/A PASS with the 67.1 gates applied to this diff (lint-gate output over the
   changed files included in the critique)"

## Design (files)

1. `backend/agents/agent_definitions.py`: add `import json` to the module imports;
   broaden the :396 tuple to `(json.JSONDecodeError, AttributeError, KeyError,
   TypeError)`; remove the unused `Optional` import (F401). NO other logic change.
2. `backend/tests/test_agent_definitions_classification.py` (NEW, house template: from
   __future__, sys.path insert, direct import, plain def test_*, no mocks): (1)
   'not json {' -> MAIN + confidence 0.5 (the documented repro; implicitly asserts no
   NameError); (2) '5' and '[1,2]' -> MAIN (AttributeError path); (3) fenced ```json
   QA/complex payload -> QA + COMPLEX (happy path preserved); (4) bare valid research
   JSON -> RESEARCH + triggers_harness true.
3. `.claude/skills/code-review-trading-domain/SKILL.md`: append `consumer-contract-break`
   row to the Dimension 3 table + #16 in the Top-15 list + negation list, per the
   brief's drafted text. APPEND-ONLY: all 15 existing Top-ranked names + all dimension
   heuristic names remain byte-present.
4. live_check_67.2.md: pre-fix NameError repro (captured 2026-07-09 pre-change) +
   post-fix graceful output, both verbatim from the venv; ruff over the changed .py
   files post-fix (expect clean).

## Anti-patterns guarded

- criteria-erosion: append-only skill edit; Q/A greps all pre-existing heuristic names.
- tautological-assertion / over-mocked-test: pure-function tests, no mocks, asserts on
  routing outcomes not on not-None.
- financial-logic-without-behavioral-test: N/A (no financial math touched) -- but the
  behavioral-test discipline applies to the fixed path anyway (criterion 3).
- consumer-contract-break itself: the fix widens an except tuple -- callers unaffected
  (sole caller verified); no interface shape changes.

## Out of scope

multi_agent_orchestrator.py (adding a caller-side try/except is a design change the
graceful-default makes unnecessary); any other F-class findings elsewhere in backend/;
trading logic; the stale "Opus 4.6" comments in agent_definitions.py docstring (doc
drift, not this step's criteria -- register for a doc pass).

## Risk

- The broadened tuple could mask genuinely unexpected exceptions -> mitigated by
  keeping the tuple narrow (4 named classes, no bare Exception) and the fallback's
  reasoning string carrying the exception repr for observability.
- Test brittleness on prompt-format drift -> tests pin the PARSER contract (JSON in ->
  routing out), not LLM behavior.
