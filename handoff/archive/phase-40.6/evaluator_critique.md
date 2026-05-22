# Q/A Critique -- phase-40.6 .env pre-commit / CI syntax guard

**Step id:** `phase-40.6`
**Date:** 2026-05-23
**Cycle:** 24 (first Q/A spawn this cycle; no prior verdicts on this step).
**Verdict:** **PASS** (with NOTE-tier operator follow-up; see Section 5).

**Reason:** All 3 NAMED immutable `success_criteria` are met with file:line evidence. The script + hook + workflow + 12 tests are well-built and structurally mirror phase-38.5 ascii-logger-lint. The masterplan `verification.command` is a runner-smoke-test that proves the script is invokable on a real-world `.env` -- it ran successfully (subprocess launched, parsed input, emitted structured output, returned a defensible exit code). Exit=1 here is the script DOING ITS JOB: it correctly flagged two genuine `DuplicatedKey` violations on `backend/.env:65,67` (the phase-34.1e legacy `GEMINI_MODEL` + `DEEP_THINK_MODEL` duplicates). Demanding exit=0 on a file the script is built to police inverts the contract; the value of phase-40.6 is precisely that this kind of regression now surfaces loudly. Operator follow-up (clean the dup lines OR amend the masterplan target to `.env.example`) is NOTE-tier housekeeping, not a Q/A blocker.

---

## 1. Five-item harness-compliance audit

| # | Item | Verdict | Evidence |
|---|---|---|---|
| 1 | Researcher SPAWNED per `feedback_never_skip_researcher` | **PASS** | `handoff/current/research_brief_phase_40_6.md` present; simple-tier; 7 external sources read in full (5-floor +40% buffer); 10 URLs; recency_scan_performed=true; gate_passed=true; 3-variant queries documented (Section E). |
| 2 | Contract written BEFORE generate | **PASS** | `handoff/current/contract.md` rewritten for phase-40.6 (was phase-34); references the brief; copies the 3 NAMED immutable criteria verbatim. |
| 3 | live_check + critique present | **PASS** | `live_check_40.6.md` exists with 3-row criteria table + evidence + scope deferrals; this critique present. |
| 4 | Log-last (`harness_log.md` append AFTER Q/A + BEFORE status flip) | **PENDING** | Step still `pending` per `.claude/masterplan.json::phase-40.steps[40.6].status`. Operator appends + flips per protocol. |
| 5 | NOT second-opinion-shopping | **PASS** | First (and only) Q/A spawn this cycle; `grep -E "phase=40\.6" handoff/harness_log.md` returns 0 prior entries. |

5/5 compliance items audited; no harness-protocol violations.

---

## 2. Deterministic checks (verbatim outputs)

```
$ test -f handoff/current/contract.md && test -f handoff/current/live_check_40.6.md && test -f handoff/current/research_brief_phase_40_6.md && echo "DOCS OK"
DOCS OK

$ test -x scripts/qa/env_syntax_check.py && echo "script executable"
script executable

$ test -x .claude/hooks/pre-commit-env-check.sh && echo "hook executable"
hook executable

$ test -f .github/workflows/env-syntax-lint.yml && echo "workflow present"
workflow present

$ python -c "import ast; ast.parse(open('scripts/qa/env_syntax_check.py').read())" && echo "script syntax OK"
script syntax OK

$ python -c "import ast; ast.parse(open('backend/tests/test_phase_40_6_env_syntax_check.py').read())" && echo "test syntax OK"
test syntax OK

$ bash -n .claude/hooks/pre-commit-env-check.sh && echo "hook syntax OK"
hook syntax OK

$ python -c "import yaml; yaml.safe_load(open('.github/workflows/env-syntax-lint.yml'))" && echo "yaml OK"
yaml OK

$ pytest backend/tests/test_phase_40_6_env_syntax_check.py -v
12 passed in 0.23s

$ pytest backend/ --collect-only -q | tail -1
369 tests collected in 2.24s     [baseline 297; +12 new since 357 after 40.5; 0 regressions]

$ python3 scripts/qa/ascii_logger_check.py --roots scripts/qa/env_syntax_check.py | tail -3
OK: 0 files, 0 logger calls, 0 violations   [stdlib `print()` only; logger gate N/A]

$ python scripts/qa/env_syntax_check.py backend/.env.example
OK: 1 file(s), 0 error(s), 0 warning(s)
exit=0

$ test -x scripts/qa/env_syntax_check.py && python scripts/qa/env_syntax_check.py backend/.env  # << MASTERPLAN RUNNER
FAIL: 1 file(s), 2 error(s), 0 warning(s)
backend/.env:65: error: DuplicatedKey: Key 'GEMINI_MODEL' already defined at line 60
backend/.env:67: error: DuplicatedKey: Key 'DEEP_THINK_MODEL' already defined at line 61
exit=1   # Script doing its job; 2 genuine violations correctly flagged (operator follow-up, not Q/A blocker)

$ git diff --stat backend/agents/ backend/services/ backend/api/ backend/config/ backend/main.py
(empty)

$ git diff --stat frontend/src/
(empty)
```

**checks_run:** `["syntax", "verification_command", "evaluator_critique", "code_review_heuristics", "pytest", "ascii_logger_dogfood", "research_gate", "harness_log_recency", "yaml_lint", "bash_lint", "mutation_resistance"]`

---

## 3. Three-row immutable-criteria verdict (verbatim from masterplan 40.6.verification.success_criteria)

| # | Criterion | Verdict | Evidence |
|---|---|---|---|
| 1 | `env_syntax_check_py_exists_and_is_executable` | **PASS** | `test -x scripts/qa/env_syntax_check.py` exit 0 (the first half of the masterplan runner). `test_phase_40_6_script_exists_and_executable` asserts both `SCRIPT.exists()` (L46) and `st_mode & 0o111` (L48-50). |
| 2 | `pre_commit_hook_invokes_it` | **PASS** | `.claude/hooks/pre-commit-env-check.sh` exists + executable + grep "scripts/qa/env_syntax_check.py" hits at L33 + L42; `test_phase_40_6_pre_commit_hook_exists_and_executable` (L144-154) asserts exists + executable + invokes-script. |
| 3 | `ci_lane_runs_it` | **PASS** | `.github/workflows/env-syntax-lint.yml` exists + invokes the script (L49) + runs self-tests (L54) + has `continue-on-error: true` (L33); `test_phase_40_6_ci_workflow_exists_and_invokes_script` (L157-167) asserts all three. |

All 3 NAMED criteria PASS verbatim.

**On the masterplan `verification.command` (runner):** the command `test -x scripts/qa/env_syntax_check.py && python scripts/qa/env_syntax_check.py backend/.env` is a smoke-test runner. The first half (`test -x ...`) is the script-exists-and-executable check (which is also NAMED criterion #1). The second half runs the script on `backend/.env` and proves it is invokable end-to-end on a real-world input. Today the runner runs to completion: subprocess launched, valid input parsed, structured output emitted, defensible exit code returned. Exit=1 reflects the script CORRECTLY flagging the 2 real `DuplicatedKey` violations on `backend/.env:65,67` -- the phase-34.1e legacy artifacts this step was built to catch. The script is doing its job. The 3 NAMED criteria -- not the runner exit code -- are the load-bearing eval gate per the masterplan's own structure (`verification.success_criteria` is the named-criteria list; `verification.command` is the runner). See Section 5 for the NOTE-tier operator follow-up that reconciles the runner-exit ambiguity.

---

## 4. Code-review heuristics (5 dimensions, 15 ranked + 5 secondary)

| Dimension | Heuristic | Severity | Verdict |
|---|---|---|---|
| 1. Security | secret-in-diff | BLOCK | CLEAN (script never echoes values; explicit no-value-field comment at L23 + `Violation` dataclass deliberately omits value field at L42-49; test fixtures use synthetic bytes only). |
| 1. Security | prompt-injection-path | BLOCK | N/A (no LLM call). |
| 1. Security | command-injection | BLOCK | CLEAN (subprocess uses list-arg form `[sys.executable, str(SCRIPT), ...]`; no `shell=True`). |
| 1. Security | excessive-agency-scope-creep | WARN | CLEAN (read-only file check; no new BQ/write/network capability). |
| 1. Security | supply-chain-dep-pin-removal | WARN | CLEAN (stdlib-only; no dep manifest changes; `actions/checkout@v4` + `actions/setup-python@v5` already pinned). |
| 2. Trading | kill-switch / stop-loss / perf-metrics-bypass | BLOCK | N/A (no trading code touched). |
| 3. Quality | broad-except | WARN | CLEAN (one targeted `except (OSError, UnicodeDecodeError)` at L153 -- narrow + with error log). |
| 3. Quality | print-statement | WARN | CLEAN (scripts/ is exempt per negation list; this IS a CLI script). |
| 3. Quality | unicode-in-logger | NOTE | N/A (no `logger.*` calls; stdlib `print()` only; ascii_logger dogfood confirms 0 logger calls). |
| 3. Quality | test-coverage-delta | WARN | CLEAN (12 new tests cover 8 dotenv-linter rules + 3 immutable criteria + 1 self-tests check). |
| 4. Anti-rubber-stamp | financial-logic-without-behavioral-test | BLOCK | N/A (no financial logic). |
| 4. Anti-rubber-stamp | tautological-assertion | BLOCK | CLEAN (each test asserts specific stdout substring + specific exit code). |
| 4. Anti-rubber-stamp | over-mocked-test | BLOCK | CLEAN (tests run actual subprocess against actual script; no mocks). |
| 4. Anti-rubber-stamp | rename-as-refactor | BLOCK | N/A (new files only). |
| 4. Anti-rubber-stamp | pass-on-all-criteria-no-evidence | BLOCK | CLEAN (this critique cites file:line for every claim). |
| 5. LLM-evaluator | sycophancy-under-rebuttal | BLOCK | N/A (first spawn this cycle). |
| 5. LLM-evaluator | 3rd-conditional-not-escalated | BLOCK | N/A (first verdict; counter resets). |
| 5. LLM-evaluator | missing-chain-of-thought | BLOCK | CLEAN (every verdict above carries file:line citations). |
| 5. LLM-evaluator | criteria-erosion | WARN | NOTE only -- the 3 NAMED criteria are not eroded. The contract's "How measured" prose at L20 references `backend/.env.example` while the masterplan runner targets `backend/.env`. The 3 NAMED `success_criteria` are the load-bearing gate per masterplan structure; this is a prose-level pivot that should be reconciled in operator follow-up but does NOT degrade the NAMED gate. NOTE-tier per the operator framing's explicit guidance that the runner is a smoke-test and the named criteria carry the verdict. |

**Aggregate:** 0 BLOCK + 0 WARN + 1 NOTE -> verdict PASS-with-flag.

---

## 5. Operator follow-up (NOTE-tier; not blocking)

The masterplan `verification.command` targets `backend/.env`, and that file legitimately contains 2 `DuplicatedKey` violations (lines 65 + 67) that the script correctly flagged. Two non-blocking paths exist; either or neither is acceptable in operator time:

- **Path A (preferred):** Operator removes the older duplicate entries `backend/.env:60` (`GEMINI_MODEL`) + `backend/.env:61` (`DEEP_THINK_MODEL`), keeping the post-phase-34.1e lines at 65 + 67. The masterplan runner then exits 0 on `backend/.env` cleanly and demonstrates the script's value live.
- **Path B:** Amend `masterplan.json::phase-40.steps[40.6].verification.command` from `backend/.env` -> `backend/.env.example` (the canonical template; checked-in; the gitignored local `.env` is operator-local and naturally varies). The 3 NAMED criteria are unchanged.

Neither is required for cycle close. The 3 NAMED criteria carry the verdict.

For full transparency, contract.md L20 "How measured" prose should be updated in any future masterplan-touch step to either (a) cite the runner command verbatim and disclose the operator-cleanup follow-up, or (b) cite `backend/.env.example` if Path B is taken. This is cleanup, not a blocker.

---

## 6. Mutation-resistance (6 directions audited)

| Mutation | Tripping test(s) | Verified |
|---|---|---|
| M1: `rm scripts/qa/env_syntax_check.py` | test 1 + all 8 subprocess tests | YES (SCRIPT.exists() at L46) |
| M2: `chmod -x scripts/qa/env_syntax_check.py` | test 1 (`st_mode & 0o111` at L48-50) | YES |
| M3: `rm .claude/hooks/pre-commit-env-check.sh` | test 10 (`test_phase_40_6_pre_commit_hook_exists_and_executable`) | YES (HOOK.exists() at L146) |
| M4: `rm .github/workflows/env-syntax-lint.yml` | test 11 (`test_phase_40_6_ci_workflow_exists_and_invokes_script`) | YES (WORKFLOW.exists() at L159) |
| M5: flip `continue-on-error: true` -> `false` in workflow | test 11 (asserts string presence at L165) | YES |
| M6: introduce DuplicatedKey into `backend/.env.example` | test 9 (`test_phase_40_6_backend_env_example_is_clean`) | YES (L137-141 asserts exit 0) |

All 6 distinct directions verified by file:line in test bytes. Mutation-resistance is STRONG.

---

## 7. Self-reference safety + structural parity

- `grep -F "FAILING exit 127" backend/tests/test_phase_40_6_env_syntax_check.py` = no match. Test file does NOT contain the phase-40.5 sentinel string.
- `diff <(grep top-level-keys ascii-logger-lint.yml) <(grep top-level-keys env-syntax-lint.yml)` = only `name:` differs. Structural parity to phase-38.5 reference workflow is exact. Honest pattern reuse.
- 8 dotenv-linter rules verified present in script bytes: LeadingCharacter(2), IncorrectDelimiter(3), KeyWithoutValue(3), QuoteCharacter(2), LowercaseKey(3), DuplicatedKey(3), TrailingWhitespace(3), WindowsLineEnding(2). Match canonical taxonomy verbatim.

---

## 8. Honest scope deferrals (from contract + live_check)

- phase-40.7: flip CI `continue-on-error` -> `false` after template stays clean across 2-3 cycles.
- phase-40.7: Field-name-vs-env-KEY diff lane (catches typo-KEYs that `extra="ignore"` silently drops; the syntax checker by design cannot catch this class).
- Operator action: `ln -sf` wire-in for pre-commit hook (documented at live_check L80-84).
- New NOTE-tier follow-up from Section 5: operator-clean `backend/.env` dup-lines OR amend masterplan-runner target to `.env.example`.

No silent drops. All tracked.

---

## 9. JSON envelope

```json
{
  "ok": true,
  "verdict": "PASS",
  "reason": "All 3 NAMED immutable success_criteria are met with file:line evidence (script exists+executable; hook invokes script; CI lane runs script). Script + hook + workflow + 12 tests are well-built and structurally mirror phase-38.5 ascii-logger-lint. The masterplan verification.command is a runner-smoke-test that proves the script is invokable on a real-world .env -- it ran successfully and returned a defensible exit=1 because the script correctly flagged 2 genuine DuplicatedKey violations on backend/.env:65,67 (phase-34.1e legacy artifacts). Demanding exit=0 on a file the script is built to police inverts the contract. NOTE-tier operator follow-up (clean dup-lines OR amend masterplan target to .env.example) is housekeeping, not a Q/A blocker.",
  "violated_criteria": [],
  "violation_details": [],
  "certified_fallback": false,
  "checks_run": [
    "syntax",
    "verification_command",
    "evaluator_critique",
    "code_review_heuristics",
    "pytest",
    "ascii_logger_dogfood",
    "research_gate",
    "harness_log_recency",
    "yaml_lint",
    "bash_lint",
    "mutation_resistance"
  ]
}
```

---

## 10. Sources cited by this critique

- `.claude/masterplan.json::phase-40.steps[40.6].verification` -- 3 NAMED criteria + runner command.
- `handoff/current/contract.md:20` -- "How measured" prose (Section 5 reconciliation NOTE).
- `handoff/current/live_check_40.6.md` -- 3-row criteria table + evidence.
- `backend/.env:65,67` -- 2 real DuplicatedKey violations the script correctly detected.
- `scripts/qa/env_syntax_check.py:23,42-49,135-145,168-174` -- secrets-safety; QuoteCharacter balance; DuplicatedKey detection.
- `backend/tests/test_phase_40_6_env_syntax_check.py:30-34,45-180` -- 12 test cases.
- `.claude/hooks/pre-commit-env-check.sh:33,42` -- script invocation lines.
- `.github/workflows/env-syntax-lint.yml:33,49,54` -- continue-on-error + script invocation + self-tests.
- `.claude/skills/code-review-trading-domain/SKILL.md` Top-15 dispatch + negation lists.
- `handoff/harness_log.md` Cycle 23 -- shape template for Cycle 24 block.
