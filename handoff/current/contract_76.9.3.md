# Contract — Step 76.9.3 (duckduckgo retriever import-broken: pin + install `ddgs`)

Date: 2026-07-25 | Cycle: 157 | Executor: Main (Opus 5) | Research gate: **PASSED**

## Research-gate summary

`handoff/current/research_brief_76.9.3.md` — tier `simple`, envelope
`{"external_sources_read_in_full":6,"snippet_only_sources":15,"urls_collected":21,`
`"recency_scan_performed":true,"internal_files_inspected":9,"gate_passed":true}`.
Three-variant search discipline visible; recency scan reports the
`duckduckgo_search` → `ddgs` rename (2025-07-06) and releases through 2026-05-15.

Load-bearing findings:

1. **Pin = `ddgs==9.14.4`** — latest stable (2026-05-15), satisfies upstream's own
   `ddgs>=9.0.0` spec for byte-identical retriever code; the required import surface
   (`from ddgs import DDGS`, zero-arg `DDGS()`, `.text(query, region=, max_results=)`)
   is unchanged across 9.x. Matches the manifest's `==` convention.
2. **Why pip was satisfied while runtime broke**: gpt-researcher 0.14.8's *code*
   imports `ddgs`, but its dist METADATA still declares the frozen
   `duckduckgo-search>=4.1.1`. So the resolver was green and only the runtime path
   failed — which is why this went unnoticed until the live memo logs.
3. **THE GUARD TRAP (measured, then re-measured by Main 2026-07-25)**: the step's own
   immutable command
   `from gpt_researcher.retrievers.duckduckgo.duckduckgo import Duckduckgo`
   **succeeds today with ddgs absent** — `check_pkg('ddgs')` runs inside `__init__`,
   not at module import. A module-import-only guard is a guard that cannot fail
   (`feedback_mutation_test_guards_and_fixtures`). **The guard MUST construct.**
   Main's own reproduction, verbatim:
   `MODULE IMPORT: OK (immutable cmd would PASS)` / `CONSTRUCT: FAILED -> ImportError
   Unable to import ddgs. Please install with 'pip install -U ddgs'`
4. **Install method**: targeted `pip install ddgs==9.14.4`, NOT a full `-r` re-run —
   the manifest carries no `langchain-core` pin and a bare `-r` re-install has
   silently bumped it before (documented incident).

## Hypothesis

Pinning + installing `ddgs==9.14.4` restores the DuckDuckGo retriever as a real
third leg of the retriever stack (currently semantic_scholar + arxiv only, both
externally rate-limited), and a **construct-level** behavioral guard makes any
future upstream rename fail LOUD at test time instead of silently degrading
nightly memo quality.

## Immutable success criteria (verbatim from `.claude/masterplan.json`)

1. ddgs pinned in requirements-autoresearch.txt with the researched compatible
   version and installed; the retriever import succeeds
2. One live DDG search via the gpt_researcher retriever class returns results
   (recorded verbatim)
3. A guard test fails loudly if the DDG retriever import breaks again (mutation:
   uninstall/rename-sim -> guard red)

Immutable verification command:
```
.venv/bin/python -c "from gpt_researcher.retrievers.duckduckgo.duckduckgo import Duckduckgo" && .venv/bin/python -m pytest backend/tests/test_phase_75_deps.py -q
```
**Disclosed weakness (not amended — immutable):** leg 1 of this command passes even
in the fully broken state (finding 3). It is therefore NOT evidence of the fix; the
pytest leg carries the real weight. Recorded so Q/A does not mistake a green
immutable command for a working retriever.

## Plan

1. Append the `ddgs==9.14.4` pin to `scripts/autoresearch/requirements-autoresearch.txt`
   in the file's existing exact-pin + consumer-anchor comment shape.
2. Targeted install into `.venv` (`pip install ddgs==9.14.4`); record the env delta
   verbatim; re-verify no `langchain-core` drift.
3. Criterion 2: one live sub-query-shaped retrieval
   (`Duckduckgo("<query>").search(max_results=3)`) recorded verbatim. Network call,
   run once; the pytest guard stays offline-deterministic.
4. Criterion 3 — three guards in `backend/tests/test_phase_75_deps.py`, mirroring the
   existing manifest (:206-212) and behavioral (:218-286) shapes:
   - **manifest pin test** — `_parse_requirements` yields `("==","9.14.4")` for ddgs.
   - **behavioral construct guard** — `Duckduckgo("probe")` constructs and
     `hasattr(r.ddg,"text")`; exercises the real `check_pkg` + `from ddgs import DDGS`
     + `DDGS()` chain, offline.
   - **loud-failure counterpart** — with `importlib.util.find_spec('ddgs')`
     monkeypatched to None, `pytest.raises(ImportError, match="pip install -U ddgs")`
     on construction: proves the failure mode is the LOUD verbatim message from the
     live logs, not a silent empty result.
5. **Mutation matrix** (Main, after GENERATE; every guard must be killed):
   - N1 delete/loosen the manifest pin → manifest test red.
   - N2 rename-sim (`find_spec('ddgs')` → None) → construct guard red.
   - N3 **STUB mutation**: make the loud-failure guard's monkeypatch a no-op → that
     test red (proves the negative test is load-bearing, not self-satisfying).
   - N4 **anti-tautology**: assert the construct guard fails against the PRE-install
     state (already captured above as the reproduced "before").
6. Q/A via a fresh subagent (Opus 5) on changed evidence → `harness_log.md` append →
   masterplan flip (log LAST, before the flip).

## Boundaries

- `requirements-autoresearch.txt` + `.venv` install + `backend/tests/test_phase_75_deps.py`.
- **`run_memo.py` NOT edited** (explicitly out of the step's stated boundary).
- The retriever-order config is NOT changed — this step restores the third leg that
  the existing order already references; reordering is out of scope.
- No production `backend/requirements.txt` change (autoresearch manifest only).

## References

- `handoff/current/research_brief_76.9.3.md` (gate, 6 sources read in full)
- Installed ground truth: `.venv/lib/python3.14/site-packages/gpt_researcher/retrievers/duckduckgo/duckduckgo.py`
  + `.../utils.py:44-60` (`check_pkg`, the verbatim error string)
- `backend/tests/test_phase_75_deps.py:206-212` (manifest shape), `:218-286` (behavioral shape)
- `feedback_mutation_test_guards_and_fixtures` — a guard that cannot fail does not count.
