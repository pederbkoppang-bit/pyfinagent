---
step: phase-23.6.0
title: Ship dotenv-syntax validator + document operator-fix for backend/.env leading-space bug — experiment results
date: 2026-05-10
verdict_class: PASS_PENDING_QA
verification_command: 'python3 tests/verify_phase_23_6_0.py'
---

# Experiment Results — phase-23.6.0

## What was done

4 new files + 1 edit. **No changes to backend code or backend/.env**
(sandbox-blocked).

### File 1 (NEW) — `scripts/validators/check_dotenv_syntax.py`

Pure-Python stdlib-only validator. ~140 lines.
- 5 regex-based rules (2 CRITICAL, 2 WARNING, 1 INFO).
- Pure function `scan_text(text) -> list[Finding]` for testability.
- CLI with `--strict` flag (gate WARNING findings on exit code).
- Exit semantics: 0 clean / 1 critical / 2 file-error / 1 warning-with-strict.

CRITICAL rules (the load-bearing ones):
- `^[A-Za-z_][A-Za-z0-9_]*=\s+\S` — leading space after `=` (the
  exact pattern that crashed autoresearch nightly since 2026-04-24).
- `^\s+[A-Za-z_]` — leading whitespace before key.

### File 2 (NEW) — `tests/services/test_dotenv_syntax.py`

17 pytest cases covering each rule + idempotency + main() exit
codes. Uses synthetic in-memory fixtures (`tempfile`-backed for
the main() tests). Does NOT touch the real `backend/.env`
(sandbox-blocked + would make tests non-portable).

### File 3 (NEW) — `handoff/runbooks/dotenv-leading-space-fix.md`

Operator runbook with:
- Pre-fix scan command (validator + manual `grep`).
- Idempotent `sed` fix:
  ```
  sed -i '' 's/^\([A-Z_][A-Z0-9_]*\)=  *\([^ ]\)/\1=\2/' backend/.env
  ```
- Verify-clean command.
- Restart sequence (`launchctl bootout` + `bootstrap` + `kickstart`).
- Dashboard verification curl.
- Cross-references to phase-23.3.5 + 23.5.19 archives.
- Pre-commit hook documentation.

### File 4 (NEW) — `tests/verify_phase_23_6_0.py`

4-check verifier per the contract:
1. validator runs clean=0 / dirty=1 with CRITICAL surfaced.
2. pytest test_dotenv_syntax.py passes.
3. runbook contains required tokens (sed pattern, regex,
   launchctl bootstrap, validator path).
4. `.git/hooks/pre-commit` invokes the validator.

### Edit — `.git/hooks/pre-commit`

Added a 13-line block at the bottom that:
- Skips fast when no `.env` is staged.
- Loops over staged `.env` files and runs the validator.
- On non-zero exit, prints the validator output + the runbook
  reference and aborts the commit.

## Verbatim verifier result

```
$ python3 tests/verify_phase_23_6_0.py
=== phase-23.6.0 verifier ===
  [PASS] validator runs (clean=0, dirty=1): validator clean=0 dirty=1 with CRITICAL surfaced
  [PASS] pytest test_dotenv_syntax passes: 17 passed in 0.01s
  [PASS] runbook contains required tokens: runbook contains all 4 required tokens
  [PASS] pre-commit hook invokes validator: .git/hooks/pre-commit invokes the validator

PASS (4/4)
EXIT=0
```

## Smoke runs of the validator

```
$ python3 scripts/validators/check_dotenv_syntax.py /tmp/clean.env
OK /tmp/clean.env (clean)
EXIT=0

$ python3 scripts/validators/check_dotenv_syntax.py /tmp/dirty.env
CRITICAL /tmp/dirty.env:1: [leading_space_after_eq] Leading space after '=' (KEY= value) — bash sources this as KEY="" + run `value` as command
         | ALPHAVANTAGE_API_KEY= TV5O5XN8IS2NLR6X

summary: 1 CRITICAL, 0 WARNING, 0 INFO
EXIT=1
```

The validator catches the EXACT line shape that's been crashing
autoresearch (`ALPHAVANTAGE_API_KEY= ...`).

## Test suite summary

```
$ .venv/bin/python -m pytest tests/services/test_dotenv_syntax.py -q
17 passed in 0.02s
```

Coverage:
- Each of the 5 rules has at least 1 explicit test.
- Idempotency (same input → same findings).
- main() exit codes for clean / critical / warning / strict / missing-file.
- Quoted-value bypass for inline-comment + trailing-whitespace rules.

## Why this matters / what the operator does next

This step ships **defensive tooling**. The operator must run the
runbook to actually fix `backend/.env` — the harness is sandbox-
blocked from the file.

**Next operator action:** open a Mac terminal and run:

```bash
cd /Users/ford/.openclaw/workspace/pyfinagent
python3 scripts/validators/check_dotenv_syntax.py backend/.env
```

If the validator finds CRITICAL lines, follow the runbook at
`handoff/runbooks/dotenv-leading-space-fix.md`.

If the validator exits 0 but autoresearch still shows `failed`,
the bug has moved — likely a python entrypoint error in the
autoresearch script. Tail `handoff/autoresearch.log` for the
real error (separate masterplan step).

## Sibling verifiers — no regressions

All 25 prior phase-23.5 verifiers green; phase-23.6.0 PASS (4/4).

## What this step does NOT do

- Edit `backend/.env` (sandbox-blocked + operator-action).
- Migrate to a secret manager.
- Refactor the autoresearch python entrypoint.
- Install dotenv-linter (Rust CLI) globally.
- Add `python-dotenv` as a dependency (the bug it can't catch is
  exactly the bug we're guarding).

## Artifact files

- `handoff/current/contract.md`
- `handoff/current/experiment_results.md` (this file)
- `handoff/current/phase-23.6.0-research-brief.md`
- `tests/verify_phase_23_6_0.py` (NEW)
- `scripts/validators/check_dotenv_syntax.py` (NEW)
- `tests/services/test_dotenv_syntax.py` (NEW; 17 tests)
- `handoff/runbooks/dotenv-leading-space-fix.md` (NEW)
- `.git/hooks/pre-commit` (extended +13 lines)

## How to re-run

```bash
cd /Users/ford/.openclaw/workspace/pyfinagent
python tests/verify_phase_23_6_0.py
```
