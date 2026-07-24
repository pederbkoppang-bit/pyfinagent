# Experiment results -- Step 75.11 (SRE hardening)

Date: 2026-07-24. **Execution model: GENERATE delegated to a Sonnet-4.6
executor; Main wrote the contract, reviewed the deliverables, and
independently re-measured every headline figure. Executor draft (5 named
deviations) preserved at
`handoff/current/experiment_results_75.11_draft.md`.**

## What was built (contract plan steps 1-7)

- **sre-ops-01**: `scripts/ops/rotate_logs.sh` (cp+truncate, modeled on
  healthcheck.sh:246-255; the four logs at their REAL paths -- backend.log
  + frontend.log at repo root, slack_bot.log + auto-push.log under
  handoff/logs/; 50MB caps, gzip, keep-10) + `com.pyfinagent.logrotate.plist.template`
  + the health.jsonl mtime>2h liveness alarm seam (bot-token page pattern
  reused). NOTHING bootstrapped -- machine actions are operator token
  **OPS-ROTATE-BOOTSTRAP** in `ops_rotate_runbook_75.11.md`.
- **sre-ops-02**: start_services.sh -> `launchctl kickstart -k` primary
  path; legacy branch flag-gated with scoped SIGTERM `pkill -f 'uvicorn
  backend.main'`; all backend.log truncation removed.
- **sre-ops-09**: `com.pyfinagent.frontend.plist.template` (production
  `next start -p 3000` via `frontend_start.sh` build wrapper) -- fixes the
  research-corrected reality that the LIVE plist runs `next dev` today.
  Live plist untouched; one authority; second-next-dev memory honored.
- **sre-ops-04**: `run_ablation.sh` reusing run_nightly.sh's phase-62.6
  sanitized-grep sourcing verbatim + FAIL-rc log + consecutive-failure
  paging seam (counter file); ablation plist TEMPLATE points at it;
  run_nightly.sh gained the paging seam only (it already logged FAIL rc).
  The backend/.env:81 quote repair is DRAFTED as operator token 2 in the
  runbook -- .env NOT edited (sha256 identical before/after, both runs).
- **sre-ops-05**: pkill/killall rail in pre-tool-use-danger.sh (narrow
  target regex python|uvicorn|next|slack_bot; exit 2; stderr points at
  launchctl kickstart; existing CLAUDE_ALLOW_DANGER escape covers it).
- **sre-ops-07**: `"$GTIMEOUT" -k 10 120` on the away-session git pull
  (existing `if !` already routes rc=124 to the offline branch -- additive
  only, per research); `-m 15` on the mention-checker curl; `-k 60 3600`
  ceiling on run_cycle.sh's claude call (disclosed judgment call,
  unexercised in production).
- **pysvc-05**: main.py formatter branches swapped (debug -> Compact,
  default -> Json), proven redaction-safe + red-set-safe by research
  before shipping.

## Change surface (measured, code-scoped)

`git diff --stat HEAD -- backend/ scripts/ .claude/hooks/ .claude/masterplan.json`:
**8 files, +138/-24** (main.py, pre-tool-use-danger.sh, 5 scripts,
masterplan 75.11.1 insert) + **7 NEW files** (6 under scripts/ops/ +
backend/tests/test_phase_75_sre_ops.py, 25 tests) + handoff artifacts.
Boundary held: backend/.env sha256 IDENTICAL before/after (executor
measured, hash in draft); zero live launchd mutations; zero service
restarts; templates carry NO secret literals (Main's independent
long-literal grep: clean).

## Verification (ALL figures independently re-measured by Main)

- Immutable command: **25 passed, exit 0** (Main re-run).
- Ruff F821/F401/F811: clean over the git-derived scope + new test file
  (Main re-run; NOTE the executor installed ruff 0.16.0 into .venv as a
  lint-time convenience -- deviation 1, mirrors the pdfplumber precedent,
  not added to requirements.txt).
- bash -n: clean on all 9 touched/new shell files (Main re-run).
- **Danger-hook behavioral (Main-driven, real script, zero real kills)**:
  `pkill -9 uvicorn` -> exit 2 with the kickstart-pointer message;
  `pkill -9 SomeRandomApp` -> exit 0; escape hatch -> exit 0.
- Full suite (Main re-run): **10 failed / 1416 passed** -- fail set
  BYTE-IDENTICAL to baseline (comm diff empty); 1416 = 1391 + exactly the
  25 new tests. Zero regressions -- including the formatter flip, whose
  red-set-safety the research proved in advance.
- Formatter branch verified in source (debug -> Compact at main.py:101-104).
- Mutation matrix: executor **9/9 KILLED** including M8 (regex-matches-
  everything stub -- proves the allow-side assert is real) and M9 (planted
  40-char secret literal -- proves the no-secrets scan bites). The
  executor DISCLOSED that M3 initially SURVIVED a naive substring check
  and fixed the test before re-running (the vacuous-guard doctrine
  applied by the executor itself).

## Executor deviations (5, all disclosed in the draft; Main-endorsed)

1. ruff installed into .venv (lint-time tool, not a production dep).
2. **A real latent defect found in shipped code**: calling
   `setup_logging()` more than once per process closes the real stderr
   (fresh TextIOWrapper per call + `root.handlers.clear()` GC chain) and
   destroys foreign root handlers (crashed the full suite until the test
   snapshot/restores root handlers). Pre-existing, unrelated to the
   branch swap; production calls it once -> dormant. **Queued as
   masterplan step 75.11.1** per the discovered-defects rule.
3. Its own comments tripped its own literal scans twice (the banned
   `pkill -9 uvicorn` token in a comment; a 46-char memory slug reading
   as a secret-shaped literal) -- prose reworded, scans kept strict.
4. The no-secrets check narrowed to actual plist `<key>` definitions so
   documenting the live plists' secret-embedding finding stays possible.
5. run_cycle GTIMEOUT ceiling 3600/-k 60 is a disclosed judgment call.

## Not verified live

- No launchd agent was bootstrapped; rotation + frontend-authority +
  ablation fixes are INERT until the operator runs OPS-ROTATE-BOOTSTRAP
  (runbook). The formatter flip + danger rail are live-on-next-restart /
  immediately for new hook invocations respectively. backend.log remains
  at ~112MB until the operator bootstraps rotation. No UI surface.
