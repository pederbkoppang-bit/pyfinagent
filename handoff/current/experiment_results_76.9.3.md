# Experiment results — Step 76.9.3 (ddgs pin: restore the DuckDuckGo retriever leg)

Date: 2026-07-25 | Cycle: 157 | Execution: Main (Opus 5) GENERATE

## What was changed (3 files, all inside the stated boundary)

### 1. `scripts/autoresearch/requirements-autoresearch.txt` (EDIT, +10 lines)

Added the pin `ddgs==9.14.4` with a consumer-anchored comment in the file's existing
convention, recording *why pip was green while the runtime broke*: gpt-researcher
0.14.8's **code** imports `ddgs` (upstream spec `ddgs>=9.0.0`) while its dist
**METADATA** still declares the frozen pre-rename `duckduckgo-search>=4.1.1`. The
resolver was therefore satisfied and only the runtime path failed — which is why this
survived the 75.13 pin sweep unnoticed until the live memo logs surfaced it.

Version rationale (from the gate): 9.14.4 is latest stable (2026-05-15), satisfies
upstream's own floor for byte-identical retriever code, and preserves the exact
import surface the installed retriever uses (`from ddgs import DDGS`, zero-arg
`DDGS()`, `.text(query, region=, max_results=)`) across all of 9.x.

### 2. `.venv` install

`pip install ddgs==9.14.4` — **targeted, not `-r`**, because the manifest carries no
`langchain-core` pin and a bare `-r` re-run has silently bumped it before. Verified:
langchain-core 1.4.8 before **and** after. Env delta: +ddgs, +fake-useragent, +h2,
+hpack, +hyperframe, +socksio; primp 1.2.2 → 1.3.1. No removals.

### 3. `backend/tests/test_phase_75_deps.py` (EDIT, +4 tests, +`import pytest`)

Mirrors the file's existing manifest (:206-212) and behavioral (:218-286) shapes:

- `test_autoresearch_requirements_manifest_pins_ddgs` — parses a REAL requirement
  line via the existing `_parse_requirements`, so a name mentioned only in a comment
  does not satisfy it; asserts `("==","9.14.4")`.
- `test_ddg_retriever_constructs_with_real_ddgs_installed` — **constructs**
  `Duckduckgo(...)`, exercising the real `check_pkg('ddgs')` → `from ddgs import
  DDGS` → `DDGS()` chain, then asserts the client exposes `.text()` (what the
  retriever actually calls). Offline: `DDGS` is a lazy proxy whose instantiation only
  assigns attributes — the http client is created lazily later and the sole network
  path is gated behind an `api_url` the retriever never passes — so network fires only
  on `.text()`, which the test never calls.
- `test_ddg_retriever_fails_loud_when_ddgs_missing` — with
  `importlib.util.find_spec('ddgs')` monkeypatched to `None`, asserts
  `pytest.raises(ImportError, match="pip install -U ddgs")`, proving the failure mode
  is the LOUD verbatim message from the live logs rather than a silent empty result.

- `test_installed_ddgs_version_matches_the_manifest_pin` (**added in cycle-2**, closing
  a vacuity the 76.9.3 Q/A found in the three guards above): asserts
  `importlib.metadata.version("ddgs") == the parsed pin`. Without it, all three earlier
  guards pass on a venv holding *any* ddgs — e.g. 9.0.0 against a 9.14.4 pin — because
  construction and the error string are version-independent. This is the only assertion
  in the suite that compares the manifest against the **live venv**.

**Why the guards construct instead of import** (the load-bearing design fact,
measured twice — by the gate and again by Main): the step's own immutable command
`from ...duckduckgo import Duckduckgo` **succeeds with ddgs absent**, because
`check_pkg` runs inside `__init__`. An import-only guard here could never fail. This
is disclosed in the contract rather than silently "fixed", since the criteria are
immutable.

## Verification (verbatim)

```
$ .venv/bin/python -c "from gpt_researcher.retrievers.duckduckgo.duckduckgo import Duckduckgo" && .venv/bin/python -m pytest backend/tests/test_phase_75_deps.py -q
16 passed, 1 warning in 2.39s
IMMUTABLE exit=0
```

12 pre-existing tests unchanged and green; 4 new (derived: `git show HEAD:...| grep -c '^def test_'` = 12, worktree = 16).

## Criteria status

| # | Criterion | Status | Evidence |
|---|-----------|--------|----------|
| 1 | ddgs pinned + installed; retriever import succeeds | MET | live_check §2-3: pin line, `Successfully installed ddgs-9.14.4`, `CONSTRUCT OK -> ddgs.ddgs.DDGS` |
| 2 | One live DDG search returns results (verbatim) | MET | live_check §3: 3 real results with titles/hrefs/bodies |
| 3 | Guard fails loudly if the import breaks again (mutation → red) | MET | live_check §5: N1-N5 each killed exactly its guard; N2 is a REAL `pip uninstall`; N5 covers the cycle-2 version-agreement guard |

## Disclosures

- **The install desynchronized the shared venv from `backend/requirements.lock`
  (found by Q/A cycle-1, independently reproduced by Main; queued as 76.9.4).**
  `backend/requirements.lock` is a `pip freeze` snapshot of THIS venv and carries the
  autoresearch closure (`duckduckgo_search==8.1.1`:76, `gpt-researcher==0.14.8`:104,
  `primp==1.2.2`:209, `sentence-transformers==5.5.1`:263); its header declares it "the
  DEPLOYED graph that `.github/workflows/pip-audit.yml` scans and that a fresh install
  should reproduce byte-for-byte". After this step the venv holds `ddgs 9.14.4` and
  `primp 1.3.1`, while
  `grep -inE '^(ddgs|fake.useragent|h2|hpack|hyperframe|socksio)==' backend/requirements.lock`
  exits 1 (ZERO hits). So `pip install -r backend/requirements.lock` into a fresh env
  would NOT reproduce the working retriever — **the same manifest-vs-runtime
  divergence class this step exists to fix.**
  Compounding it: `grep -c autoresearch .github/workflows/pip-audit.yml` returns 0 —
  the workflow audits exactly 5 files and the autoresearch manifest is not among them,
  so `ddgs` + its 5 new transitive deps currently sit in **zero** CVE-scan surfaces.
  Regenerating the lock is deliberately NOT done here: it is a full freeze of a shared
  venv that may carry unrelated drift, and it is outside this step's stated boundary
  (`requirements-autoresearch.txt` + install + tests). Queued as **76.9.4** per the
  operator standing rule that a discovered defect gets its own research-gated step
  rather than a prose mention.
  Why the suite stayed green over it: the existing lock tests assert header/count/prefix
  only, never freeze-equality — which is itself part of what 76.9.4 must fix.


- Pre-existing, untouched: `gpt-researcher 0.14.8 requires numpy<2.3.0` vs installed
  numpy 2.4.4. Not caused by this step (no numpy in the install set) and out of
  boundary; queue separately if it ever bites.
- `duckduckgo_search==8.1.1` deliberately left installed (stale metadata dep of
  gpt-researcher; nothing imports it). Removing it is out of boundary.
- `run_memo.py` NOT edited (explicit boundary). The retriever ORDER at
  `run_memo.py:283` was already `semantic_scholar,arxiv,duckduckgo`; this step
  restores the third leg that order already referenced, and changes no ordering.
