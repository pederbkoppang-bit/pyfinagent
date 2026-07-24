# Experiment results -- Step 75.13 (Python dependency integrity)

Date: 2026-07-24. **Execution model: GENERATE delegated to a Sonnet-4.6
executor (5th delegated step); Main wrote the contract, reviewed every
diff by eye, and independently re-measured every headline figure.
Executor draft (5 disclosed judgment calls) preserved at
`handoff/current/experiment_results_75.13_draft.md`.**

## What was built (all FILE WRITES -- zero pip install/uninstall)

- **deps-01**: NEW `backend/requirements.lock` -- 22-line `#` header
  (regeneration command, both sync commands, date, pin count, the pip
  repeatable-installs citation) + the verbatim 303-line `pip freeze`
  snapshot. `.github/workflows/pip-audit.yml` now audits BOTH the lock
  (deployed graph) AND the floors file (hypothetical resolution) --
  disclosed additive widening of "point it at the lock" -- and the lock is
  in the push/PR paths filters + failure-artifact upload.
- **deps-02**: NEW `scripts/autoresearch/requirements-autoresearch.txt`
  (gpt-researcher==0.14.8 + langchain-huggingface==1.2.1 +
  sentence-transformers==5.5.1, all read from the live freeze).
  `run_memo.py` gained `_gpt_researcher_guard()` called BEFORE
  `_embedding_preflight()` -- a missing gpt_researcher now fails loudly
  through the existing rc-1 -> 75.11 paging path instead of being maskable
  by the intentional 51.4 embedding soft-skip. run_nightly.sh UNTOUCHED
  (git diff empty -- boundary proof).
- **deps-09**: real declaration lines with consumer-anchoring comments:
  `exchange-calendars==4.13.2` (hyphen form per the PyPA-normalization
  gotcha), `numpy==2.4.4` (39 direct-import sites), `PyYAML==6.0.3`
  (CAPS), `pytest==9.0.3`, `python-dateutil==2.9.0.post0`,
  `google-cloud-storage==3.10.1` (the unguarded compliance WORM writer).
- **deps-08**: fpdf2 line + its comment DELETED (zero whole-file residue);
  xlrd comment enhanced (pandas .xls engine, macro_regime.py:59,154).
- NEW `backend/tests/test_phase_75_deps.py` -- 12 tests: PARSED-line
  requirement asserts (comment mentions cannot satisfy), lock header +
  pin-count, yml parsed via yaml.safe_load, the guard behavioral test
  (find_spec->None => rc 1 with `_embedding_preflight` proven UNREACHED)
  + the pass-through sanity test.

## Verification (ALL figures independently re-measured by Main)

- Immutable command (verbatim from the masterplan): **exit 0**.
- `test_phase_75_deps.py`: **12 passed** (Main re-run).
- Full suite (Main re-run): **10 failed / 1428 passed** -- fail set
  BYTE-IDENTICAL to baseline (comm diff empty); 1428 = 1416 + exactly the
  12 new tests. Zero regressions.
- Ruff over the git-derived scope + new test file: clean.
- **Environment-mutation proof**: `pip freeze | shasum -a 256` identical
  before (executor) and after (Main re-measured): `8df19b228e08...` --
  303 lines both. Zero installs/uninstalls.
- Mutation matrix: executor **7/7 KILLED** (lock truncation, unpin,
  fpdf2 restore, yml pointer removal, guard removal, stub-break of the
  test's own monkeypatch target) **+ M6, the documented command-vs-test
  delta, independently REPRODUCED by Main**: moving `PyYAML==6.0.3` into
  a comment fails the parsed-line test while the IMMUTABLE COMMAND STILL
  PASSES (exit 0) -- measured proof the new tests out-bite the command's
  substring asserts, exactly the weakness the research gate called out.

## Change surface (measured)

3 modified (backend/requirements.txt, .github/workflows/pip-audit.yml,
scripts/autoresearch/run_memo.py) + 3 NEW (backend/requirements.lock,
scripts/autoresearch/requirements-autoresearch.txt,
backend/tests/test_phase_75_deps.py). Masterplan diff = ONLY Main's
75.13.1 queue insert (+21 lines, verified). The executor correctly
flagged-and-excluded the concurrent hook/daemon diffs from its surface.

## Not verified live / documented-not-executed

- The fresh-install dry-check (throwaway venv + `uv pip sync` +
  autoresearch closure) is DOCUMENTED verbatim in the draft but NOT
  executed (boundary: no installs). The lock's correctness rests on it
  being a byte-verbatim freeze of the running venv.
- pip-audit.yml changes are GitHub-side; first exercised on the next push.
- The find_spec guard's loud path composes with the 75.11 paging seam,
  which was live-validated last night (arXiv-429 -> consecutive_fails: 1).

## Queued this step

- **75.13.1** (pending): classify + declare-or-guard the undeclared
  OPTIONAL import families (torch, transformers, statsmodels, fredapi,
  voyageai, timesfm, chronos, vaderSentiment) -- research side-finding,
  out of 75.13 scope.
