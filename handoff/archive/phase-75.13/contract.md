# Contract -- Step 75.13: Python dependency integrity (lockfile, undeclared runtime deps, dead + implicit declarations)

- **Step id**: 75.13 (phase-75, Audit75 S13) -- P1, executor sonnet-tier
- **Date**: 2026-07-24
- **Author**: Main (contract + review). **GENERATE delegated to a Sonnet-4.6 executor** (5th delegated step).
- **BOUNDARY**: lock generated from the LIVE venv snapshot (`pip freeze` -- no resolver run); no .env edits; **NO pip install/uninstall of ANY kind** (declaration-only changes; `pip freeze` before == after must be byte-identical); no edits to the 75.11 paging seam in run_nightly.sh; the phase-51.4 `_embedding_preflight()` soft-skip is INTENTIONAL and stays.

## Research-gate summary (gate PASSED)

Workflow `wf_fc61bc04-4cd` (researcher, opus/max, tier=moderate).
Envelope: `external_sources_read_in_full=7, snippet_only=9, urls=17, recency_scan=true, internal_files=11, gate_passed=true`.
Brief: `handoff/current/research_brief_75.13.md`.

**Step-text corrections adopted (binding):**
1. Path fix: the champion/challenger import site is `backend/autoresearch/monthly_champion_challenger.py:326-327`.
2. **deps-02 re-scoped**: the gpt_researcher PACKAGE ImportError is ALREADY loud (run_memo.py:114 broad except -> rc 1 -> the 75.11 seam logs FAIL + pages after 3). The real silent dry-up is the phase-51.4 `_embedding_preflight()` exit-0 soft-skip (run_memo.py:201-204) which is an INTENTIONAL away-ops design. The fix = a NEW `importlib.util.find_spec('gpt_researcher') is None -> FAIL + return 1` guard placed BEFORE the preflight. Do NOT revert 51.4; do NOT touch the 75.11 seam.
3. **Text-assert gotchas** (the immutable command is case/spelling-sensitive): requirements.txt must carry the HYPHEN form `exchange-calendars==4.13.2` (pip freeze emits the underscore form; both are pip-identical per the PyPA name-norm spec -- the lock keeps the underscore, its assert is prefix-only case-insensitive); `PyYAML==6.0.3` must be CAPS (lowercase fails the assert); `fpdf2` must leave ZERO residue anywhere in requirements.txt (whole-file substring assert) -- delete the comment line too.
4. **xlrd already declared** (requirements.txt:20) -- the work is ENHANCING its comment (pandas .xls engine, macro_regime.py:154), not adding a line.
5. pytest is criterion-required but NOT command-asserted (and `pytest` is already a substring via pytest-cov) -- add the explicit `pytest==9.0.3` line anyway; the Q/A must check it by eye since the command cannot.
6. **The verification command is necessary-but-not-sufficient** (measured weaknesses: `==` count includes comments; substring checks pass on comments; the yml check passes on a comment; pytest/seam/header/freeze-equality never asserted) -- experiment_results must carry the real-line evidence and the Q/A must verify by reading, not just running.

**Key measured findings:** live venv = 303 freeze lines (>=150 floor cleared 2x), NO -e/git+/quirk lines; exact pins captured in the brief (pandas 3.0.1, yfinance 1.2.0, numpy 2.4.4, exchange_calendars 4.13.2, gpt-researcher 0.14.8, fpdf2 2.8.7, xlrd 2.0.2, PyYAML 6.0.3, pytest 9.0.3, python-dateutil 2.9.0.post0, google-cloud-storage 3.10.1). All silent-degrade quotes verified (markets.py:203-204 fail-OPEN; the drill's hardcoded-2026 holiday set; UNGUARDED `from google.cloud import storage` at compliance_logger.py:122 WORM writer + earnings_tone.py:38,56). fpdf2 is a TRUE orphan (zero live imports, empty Required-by) -- removal breaks nothing. gh-action-pip-audit takes requirements via `inputs:`; auditing the LOCK = auditing the deployed graph (Semgrep 2026 rationale).

## Hypothesis

The dependency surface becomes reproducible and honest via pure file writes -- a 303-pin freeze lock, real declaration lines for the six undeclared runtime deps, the orphan removed, CI pointed at the lock, and a find_spec guard that makes the nightly memo fail loudly -- with `pip freeze` before/after byte-identical proving zero environment mutation.

## Immutable success criteria + command

Copied verbatim in the masterplan node (the executor and Q/A read them there). Command: the python3 assert chain on requirements.lock (>=150 `==` pins, exchange* prefix present), requirements.txt substrings (`exchange-calendars`, `numpy`, `PyYAML`, `python-dateutil`, `google-cloud-storage`, NO `fpdf2`), and pip-audit.yml containing `requirements.lock`. Criteria additionally require (not command-asserted): the pytest declaration, the run_nightly/run_memo loud-fail seam, the lock regeneration header, freeze before==after, and the documented (unexecuted) fresh-install dry-check.

## Plan steps

1. `backend/requirements.lock`: `.venv/bin/pip freeze` output + prepended `#` header (regeneration command, sync commands, date, ~pin count). Header stays comments-only so the pin-count assert is unperturbed.
2. `backend/requirements.txt`: real requirement lines `exchange-calendars==4.13.2`, `numpy==2.4.4`, `PyYAML==6.0.3`, `pytest==9.0.3`, `python-dateutil==2.9.0.post0`, `google-cloud-storage==3.10.1`; DELETE the fpdf2 line AND its comment (zero residue); enhance the xlrd comment.
3. `.github/workflows/pip-audit.yml`: real `--requirement backend/requirements.lock` in the run step + add the lock to the push/PR paths filters. Inert locally.
4. `scripts/autoresearch/requirements-autoresearch.txt`: pinned `gpt-researcher==0.14.8` + its embedding closure (langchain-huggingface, sentence-transformers) with a header naming the launchd consumer.
5. `run_memo.py`: early `find_spec('gpt_researcher')` guard (FAIL print + return 1) BEFORE `_embedding_preflight()`; composes with the 75.11 seam (unmodified).
6. Tests (extend `backend/tests/test_phase_75_sre_ops.py` or a small new file -- executor's call, disclosed): real-line asserts (a parsed-requirements check, not substrings) for the six declarations + fpdf2 absence + lock header presence + the run_memo guard (behavioral: monkeypatch find_spec -> None => rc 1 before any preflight call).
7. Mutation matrix: truncate the lock below 150; unpin one requirements line (`==` -> `>=`); restore fpdf2; drop the yml pointer; remove the find_spec guard; comment-only-satisfy probe (move a declaration into a comment -- the parsed-line test must fail while the immutable command would still pass, PROVING the test out-bites the command).
8. Verification: immutable command exit 0; `pip freeze` before/after byte-compare; full backend suite vs the 10-test baseline; ruff over the git-derived scope; the documented fresh-install dry-check block in experiment_results (NOT executed).
9. live_check_75.13.md: verbatim command output + git diff --stat + the freeze-equality proof. No UI; no flag-gated behavior.

## Queued this step

- **75.13.1** (discovered defect, own step): classify + declare-or-guard the undeclared optional third-party imports surfaced by the research import-vs-declared diff (torch, transformers, statsmodels, fredapi, voyageai, timesfm, chronos, vaderSentiment) -- explicitly OUT of 75.13 scope.

## Explicitly NOT in scope

- Any pip install/uninstall; any .env edit; the 51.4 embedding soft-skip; the 75.11 paging seam; the optional-import families (queued 75.13.1); floor bumps in requirements.txt beyond the named additions.

## References

- `handoff/current/research_brief_75.13.md` (7 read-in-full: pip Repeatable-Installs, PEP 508, PyPA name-norm spec, pip-tools, gh-action-pip-audit, Semgrep 2026 lock-audit rationale)
- `handoff/current/audit_phase75/confirmed_findings.json` (deps-01/02/08/09)
