---
step: phase-10.7.3
title: Algorithm Discovery archetype seed library
cycle_date: 2026-04-25
harness_required: true
forward_cycle: true
parent_phase: phase-10.7
deliverables:
  - backend/meta_evolution/archetype_library.py
  - tests/meta_evolution/test_archetype_library.py
  - backend/meta_evolution/__init__.py (export ARCHETYPES)
---

# Sprint Contract -- phase-10.7.3

## Research-gate summary

`handoff/current/phase-10.7.3-research-brief.md`. tier=moderate, 6 in-full,
16 URLs, recency scan present, gate_passed=true. Internal: 9 files inspected.

## Hypothesis

A canonical 6-archetype seed library — mirroring `STRATEGY_REGISTRY` from
`backend/backtest/backtest_engine.py:32-38` plus one forward-declaration
(`sentiment_event_driven`) — gives the meta-evolution loop (10.7.2 directive
rewriter, future 10.7.4-10.7.8 governance) machine-readable structured
metadata it can mutate. Pattern mirrors 10.7.1 / 10.7.2: `@dataclass` +
module-level constant tuple + pure Python (no I/O).

## Key references (from research)

1. **QuantEvolve** (arXiv 2510.18569v1, 2025) — C+1 island seeding pattern; one
   simple representative seed per category. Direct design reference.
2. **AlphaEvolve** (DeepMind blog, 2025) + **DeepEvolve** (arXiv 2510.06056,
   2025) — single best-known seed; diversity emerges from evolution, not
   from exhaustive taxonomy.
3. **Internal:** `backend/backtest/backtest_engine.py:32-38` STRATEGY_REGISTRY
   (5 strategies); `backend/backtest/quant_optimizer.py` AVAILABLE_STRATEGIES
   (6 incl. blend); `backend/agents/skills/quant_strategy.md` (240 lines,
   complementary human-readable guide).
4. **Internal pattern:** `backend/meta_evolution/alpha_velocity.py` (10.7.1,
   161 lines) — dataclass + tuple + factory + FakeBQ test pattern.

## Plan steps

1. Implement `backend/meta_evolution/archetype_library.py` (~140 LOC):
   - `Archetype` dataclass with 7 fields (`strategy_id`, `name`,
     `description`, `default_params`, `expected_regime`, `directive_template`,
     `is_implemented`).
   - Module-level `ARCHETYPES: tuple[Archetype, ...]` of 6 entries (5 mirror
     STRATEGY_REGISTRY + 1 forward-declaration `sentiment_event_driven`).
   - Module-level constants: `ALLOWED_REGIMES` (8 values),
     `IMPLEMENTED_STRATEGY_IDS` (mirror of STRATEGY_REGISTRY + blend).
   - `__post_init__` validation in dataclass: required fields non-empty;
     `expected_regime in ALLOWED_REGIMES`.
2. Add `tests/meta_evolution/test_archetype_library.py` (~90 LOC, 7 tests).
3. Export `ARCHETYPES` from `backend/meta_evolution/__init__.py`.
4. Run immutable verification command + pytest.
5. Spawn Q/A.

## Success Criteria (verbatim, immutable from masterplan)

```
python -c "from backend.meta_evolution.archetype_library import ARCHETYPES; assert len(ARCHETYPES) == 6"
```

Plus:
- `archetype_library_module_imports`: file exists + clean import.
- `six_archetypes_present`: tuple length is exactly 6.
- `unique_strategy_ids`: no duplicates across the 6.
- `implemented_ids_in_registry`: every `is_implemented=True` archetype's
  `strategy_id` is in `STRATEGY_REGISTRY` (or `blend`).
- `forward_declaration_flagged`: at least one archetype has
  `is_implemented=False` (the `sentiment_event_driven` one).
- `tests_pass`: `pytest tests/meta_evolution/test_archetype_library.py -v`
  exits 0 with all 7 tests passing.

## What Q/A must audit

1. Module imports cleanly (deterministic check).
2. `len(ARCHETYPES) == 6` (immutable verification, exit 0).
3. Pytest 7/7 PASS.
4. No duplicate `strategy_id` values.
5. `is_implemented=True` archetypes' IDs are all in STRATEGY_REGISTRY (no
   silent fallback to `triple_barrier`).
6. `expected_regime` values are all in the 8-allowed-set.
7. `default_params` keys map to keys in `quant_optimizer._PARAM_BOUNDS`
   (spot-check at least one archetype).
8. `directive_template` strings are non-empty AND contain a `{name}` or
   `{strategy_id}` placeholder (so 10.7.2 rewriter can substitute).
9. No mutation of `STRATEGY_REGISTRY` or other engine code.
10. No I/O in the module (pure data).
