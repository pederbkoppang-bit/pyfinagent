---
name: credential-free-ci-lane-85-2
description: Step 85.2 CI-lane research — the step premise named only 1 of 2 blockers; clearing collection still leaves 46 red; pydantic defaults are the LOWEST-precedence source; a tracked ops artifact couples CI to operator state
metadata:
  type: project
---

Research findings for masterplan step 85.2 (repair the credential-free `e2e-smoke.yml`
lane), measured 2026-08-07 against HEAD `55df9006`.

**Why:** the step asserted a single root cause (pydantic `ValidationError` on four
required `backend/config/settings.py` fields). Measuring the real CI log refuted the
completeness of that premise and surfaced a much larger residual.

**How to apply:** before designing any "make CI green" step, re-derive the failure
census from the ACTUAL run log, not from the step text — and measure the post-fix
population before promising green.

## The five load-bearing facts

1. **Two independent blockers, not one.** Run 31154911052's 7 collection errors split
   4 (pydantic) + 3 (`ModuleNotFoundError: aiohttp`). Root cause of the second, pinned
   to the install log: `WARNING: slack-bolt 1.30.0 does not provide the extra 'async'`.
   `backend/requirements.txt:55` is `slack-bolt[async]>=1.18.0`; slack-bolt 1.30.0
   (2026-07-15) removed that extra, so `aiohttp` stopped being installed and **pip still
   exited 0**. The operator's `.venv` pins 1.27.0 + aiohttp, so the Mac is structurally
   blind to it. **Class of defect: an unpinned `pkg[extra]` can silently evaporate on a
   dependency release — pip warns, never fails.**

2. **Fixing collection does NOT make the lane green.** Measured full secretless run
   (detached worktree + `env -i` + `HOME` pointed away from gcloud ADC):
   `46 failed, 2817 passed, 13 skipped, 16 deselected, 4 xfailed, 4 errors` across 23
   files. Dominant causes: `DefaultCredentialsError` (ADC absent) reached from tests that
   have nothing to do with credentials, and live-BQ tests that were never marked
   `requires_live` (only 16 tests carry that marker).

3. **A pydantic default cannot change production behaviour.** Documented source
   precedence is init args > env vars > dotenv > secrets dir > **defaults** (lowest). With
   a real `backend/.env` supplying a key, a default is never consulted. Corollary trap:
   `env_file=(example, real)` TUPLE layering also works and is documented, but it
   backfills a key the real `.env` FORGOT — the masking failure mode. Corollary 2:
   `Field(...)` on a `str` already accepts an EMPTY value, so today's "strictness" does
   not catch `GCP_PROJECT_ID=`.

4. **The repo's fail-fast premise is already partly false.**
   `os.getenv("GCP_PROJECT_ID", "sunny-might-477607-p8")` — the REAL project id hardcoded
   as a fallback — appears in 12+ modules (spend.py, sortino.py, slot_accounting.py,
   sovereign_api.py, _production_fns.py, directive_*.py, ...). Never claim end-to-end
   startup validation on that variable without qualifying it.

5. **A TRACKED ops artifact decides CI's verdict.** `handoff/kill_switch_audit.jsonl` is
   version-controlled, so a fresh clone inherits whatever kill-switch state was last
   committed; 8 `test_paper_trading_v2` failures were `REFUSING BUY ... kill switch is
   PAUSED (pause_reason='manual')`. CI outcome is coupled to committed operator state.

## Context traps

- **"It used to be green" is not a baseline.** Last green run `29987399582`
  (2026-07-23); 15 consecutive failures since. phase-75.15 removed
  `continue-on-error: true` on 2026-07-24 — its own contract says the lane
  "structurally CANNOT block" before that. Its green was captured LOCALLY with the
  operator's `.env` on disk, which could never exercise the four missing fields.
- `get_settings()` is `@lru_cache`'d (`settings.py:626`) — a test calling it gets an
  object built from the operator's `.env`. Instantiate `Settings(_env_file=None)` or
  `cache_clear()`, else the test is a false green on the Mac.
- `gh` token carries the `workflow` scope, so `gh workflow run` dispatch is available;
  the `pull_request` trigger is dead here (direct-to-main policy).
- Related: [[immutable-criteria-must-be-green-able]] — a criterion demanding a green
  workflow is bound to ~46 pre-existing unrelated failures.

Brief: `handoff/current/research_brief_85.2.md` (archived to
`handoff/archive/phase-85.2/` on close).
