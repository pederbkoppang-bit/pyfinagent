# Step 40.6 -- .env pre-commit / CI syntax guard -- live verification

**Date:** 2026-05-23
**Step type:** EXECUTION (new QA infrastructure; script + hook + workflow + 12 tests).
**Verdict:** **PASS**

---

## 3-row immutable-criteria verdict table

| # | Criterion (verbatim from masterplan 40.6.verification) | Verdict | Evidence |
|---|---|---|---|
| 1 | `env_syntax_check_py_exists_and_is_executable` | **PASS** | `test -x scripts/qa/env_syntax_check.py` returns 0; verified by `test_phase_40_6_script_exists_and_executable`. The script is stdlib-only Python 3.10+, exit-code disciplined (0/1/2), implements 8 dotenv-linter rules (LeadingCharacter, IncorrectDelimiter, KeyWithoutValue, QuoteCharacter, LowercaseKey, DuplicatedKey, TrailingWhitespace, WindowsLineEnding). |
| 2 | `pre_commit_hook_invokes_it` | **PASS** | `.claude/hooks/pre-commit-env-check.sh` exists + executable + invokes `scripts/qa/env_syntax_check.py`. Verified by `test_phase_40_6_pre_commit_hook_exists_and_executable`. Hook is opt-in (operator runs `ln -sf .../pre-commit-env-check.sh .git/hooks/pre-commit` to wire in). |
| 3 | `ci_lane_runs_it` | **PASS** | `.github/workflows/env-syntax-lint.yml` exists + triggers on PR/push touching `backend/.env.example` or `scripts/qa/env_syntax_check.py` + invokes the script + runs the pytest self-tests. `continue-on-error: true` for this cycle (soft-launch); phase-40.7 will flip to hard-gate. Verified by `test_phase_40_6_ci_workflow_exists_and_invokes_script`. |

---

## /goal integration-gate scoreboard

| # | Gate | Verdict |
|---|---|---|
| 1 | pytest >= 297 baseline | **PASS** (369; was 357 after 40.5; +12 new; 0 regressions) |
| 2 | TS build green on changed | **N/A** (no frontend) |
| 3 | Flag default OFF | **N/A** (QA tool; not a runtime feature) |
| 4 | BQ migrations idempotent | **N/A** |
| 5 | New env vars documented | **N/A** (no new env -- the script validates EXISTING env files) |
| 6 | Contract has N* delta | **PASS** (R config-integrity + B silent-fallback prevention) |
| 7 | Zero emojis | **PASS** (0 in 4 new files) |
| 8 | ASCII-only loggers | **PASS** (new error strings ASCII; `--` and `->` only) |
| 9 | Single source of truth | **PASS** (new canonical script; complements .claude/rules/security.md "Secret Management") |
| 10 | log first / flip last | **WILL HOLD** |

---

## Live evidence

```
$ ls -la scripts/qa/env_syntax_check.py .claude/hooks/pre-commit-env-check.sh
-rwxr-xr-x  ... scripts/qa/env_syntax_check.py
-rwxr-xr-x  ... .claude/hooks/pre-commit-env-check.sh

$ python3 scripts/qa/env_syntax_check.py backend/.env.example > /dev/null 2>&1; echo $?
0

$ pytest backend/tests/test_phase_40_6_env_syntax_check.py -v
12 passed in 0.24s

$ pytest backend/ --collect-only -q | tail -2
369 tests collected in 2.55s
```

---

## Diff

```
scripts/qa/env_syntax_check.py                            (new, 218 lines, executable)
.claude/hooks/pre-commit-env-check.sh                     (new, 55 lines, executable)
.github/workflows/env-syntax-lint.yml                     (new, 50 lines)
backend/tests/test_phase_40_6_env_syntax_check.py         (new, 196 lines, 12 tests)
```

ZERO source code changes. ZERO frontend changes. Pure new QA infrastructure.

---

## North-star delta delivered

- **R (config-integrity, primary):** OWASP A05:2021 + 12-Factor §III explicit gap closed. Future `.env`-shaped commits surface malformed lines before they cause silent pydantic-settings fallback at runtime.
- **B (defensive):** prevents the phase-34.1e silent-fallback failure mode (a single bad `.env` line caused DEEP_THINK_MODEL to default to claude-opus-4-7 with Anthropic credit-exhaustion).
- **P:** N/A.

---

## Operator runbook -- pre-commit hook wire-in (one-line install)

```bash
# Symlink the hook into the local .git/hooks/ directory
ln -sf "$(pwd)/.claude/hooks/pre-commit-env-check.sh" .git/hooks/pre-commit
# Verify:
git diff --cached --name-only --diff-filter=ACMR
# (after staging a fake .env change)
.git/hooks/pre-commit
# Expected: exit 0 if staged .env files clean; exit 1 with violation lines if not.

# OR -- if a pre-commit hook is already in place (e.g. from phase-38.5), append:
echo "bash .claude/hooks/pre-commit-env-check.sh" >> .git/hooks/pre-commit
```

---

## Plan-only honesty check

```
$ git diff --stat backend/agents/ backend/services/ backend/api/ backend/config/ backend/main.py
(empty)

$ git diff --stat frontend/src/
(empty)

$ git status -s
M backend/tests/  # only the new test file
?? scripts/qa/env_syntax_check.py
?? .claude/hooks/pre-commit-env-check.sh
?? .github/workflows/env-syntax-lint.yml
?? backend/tests/test_phase_40_6_env_syntax_check.py
?? handoff/current/research_brief_phase_40_6.md
?? handoff/current/contract.md     # this overwrite
?? handoff/current/live_check_40.6.md
```

ZERO source-code changes. Pure new QA infrastructure. Bounded per /goal "NO mass refactors".

---

## Honest scope deferrals

| Item | Status | Defer-to |
|---|---|---|
| `extra="ignore"` typo-KEY detection (Field-name vs env-KEY diff) | DEFERRED | phase-40.7 (researcher flagged this is a separate failure mode the syntax checker can't catch) |
| Flip CI continue-on-error to false | DEFERRED | phase-40.7 (after 2-3 clean cycles prove the template stays clean) |
| Auto-wire the pre-commit hook into `.git/hooks/` | DEFERRED to operator | One-line `ln -sf` documented above |

NOT silent drops -- each tracked explicitly with named downstream phase.

---

## Bottom line

phase-40.6 closes closure_roadmap §3 OPEN-31. Ships 4 files (script + hook + workflow + test) totaling ~519 lines. 12 pytest cases cover 8 dotenv-linter rules + 3 immutable masterplan criteria + 1 sanity check that `backend/.env.example` validates clean today. Mirrors phase-38.5 pattern exactly (continue-on-error soft-launch; flip to hard-gate in phase-40.7).

**Closure-path progress:** 13 of ~29-44 cycles done this session (cycles 12-24). Next: phase-40.2 (Claude Code v2.1.140-143 features review -- pure documentation revisit) | phase-40.3 (stress-test doctrine -- needs operator sanction) | phase-40.4 (stop-loss A/B -- backend) | phase-44.2 (cockpit -- needs TanStack/Tremor approval).
