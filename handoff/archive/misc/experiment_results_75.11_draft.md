# Experiment results (executor draft) -- Step 75.11 (SRE hardening)

Executor: Sonnet-4.6 (per contract's `[executor: sonnet-4.6/high]` tag).
GENERATE only -- no self-evaluation, no masterplan/harness_log edits, no
git commit/push. All figures below were run-then-written by me and are
verbatim tool output, not transcribed from memory.

## Boundary compliance (measured, not asserted)

- `backend/.env` was never opened for content and never edited. sha256
  recorded BEFORE any edit and again AFTER all edits below -- **identical**:
  `2df2608773b19fd766c7d28197f91f7a829cd192c984fd27bfffa954efb28015`
  (6128 bytes), both times.
- No file under `~/Library/LaunchAgents/` was created, modified, or
  removed. No `launchctl bootstrap`/`bootout`/`load`/`unload`/`kickstart`
  command was executed by me at any point in this session.
- No real `pkill`/`killall`/service restart was run. The danger-hook test
  (criterion 4) drives the REAL hook script via env vars only -- the hook
  matches the command STRING and never executes it, so there is zero
  process side effect (confirmed by design and by the hook's own source).
- No live log file was rotated, truncated, or deleted.
- Automated proof of the first two points is now part of the immutable
  test file: `test_c6_no_launchctl_bootstrap_executed_in_ops_scripts`
  (scans every `scripts/ops/*.sh` line, excluding comments, for the
  literal string) and the `.env` hash equality documented above.

## What was built (contract plan steps 1-7)

1. **sre-ops-01** -- `scripts/ops/rotate_logs.sh` (cp+truncate modeled
   verbatim on `scripts/away_ops/healthcheck.sh:246-255`; covers the four
   REAL log paths: `$REPO/backend.log`, `$REPO/frontend.log`,
   `handoff/logs/slack_bot.log`, `handoff/logs/auto-push.log`; 50MB size
   cap per log, gzip archives, keep-10 retention pruning) +
   `scripts/ops/com.pyfinagent.logrotate.plist.template` (StartInterval
   1800s user agent, `__REPO_ROOT__` placeholder, no secrets) + a
   watchdog-liveness alarm seam inside the same script (pages via the
   bot-token pattern reused verbatim from healthcheck.sh's P1 fallback
   when `handoff/away_ops/health.jsonl` is >2h stale, with an
   incident-latch so it re-pages only once per incident) +
   `handoff/current/ops_rotate_runbook_75.11.md` (both operator tokens).
2. **sre-ops-02** -- `scripts/start_services.sh` rewritten: primary path
   is `launchctl kickstart -k gui/$UID/com.pyfinagent.{backend,frontend}`;
   the direct-launch path survives only behind `LEGACY_DIRECT=1`, using a
   scoped SIGTERM-then-5s-wait `pkill -f 'uvicorn backend.main'` (never
   `-9`); the `> backend.log` single-arrow truncation is gone everywhere
   (legacy path now appends, `>>`).
3. **sre-ops-09** -- `scripts/ops/frontend_start.sh` (pre-start build
   wrapper: builds if `.next/BUILD_ID` is missing or older than
   `src`/`package.json`/`next.config.*`, then `exec npx next start -p
   3000`) + `scripts/ops/com.pyfinagent.frontend.plist.template` (points
   at the wrapper; the LIVE plist -- which runs `next dev` today -- is
   untouched; start_services.sh's frontend leg is kickstart-only).
4. **sre-ops-04** -- `scripts/ops/run_ablation.sh` (the
   `scripts/autoresearch/run_nightly.sh:19-27` sanitized-grep sourcing
   block copied verbatim inside a marked region, never a raw
   `. backend/.env`; logs `FAIL rc=$rc`; a consecutive-failure counter
   file gates a bot-token page after `PAGE_AFTER_N=3` -- default, env
   overridable) + `scripts/ops/com.pyfinagent.ablation.plist.template`
   pointing at the wrapper. `scripts/autoresearch/run_nightly.sh` gained
   the identical paging seam only (it already logged FAIL rc).
5. **sre-ops-05** -- `.claude/hooks/pre-tool-use-danger.sh` gained a
   pkill/killall rail inside the existing Bash-tool block (right after
   the `cmd` extraction, before the rm-rf gate): blocks when the command
   matches `(pkill|killall)` AND the target matches
   `python|uvicorn|next|slack_bot`; exit 2 + stderr pointing at
   `launchctl kickstart -k gui/$UID/com.pyfinagent.<svc>`; inherits the
   pre-existing `CLAUDE_ALLOW_DANGER=1` escape (checked first, unchanged).
6. **sre-ops-07** -- `scripts/away_ops/run_away_session.sh:107`'s git
   pull now runs under `"$GTIMEOUT" -k 10 120` (additive only -- the
   pre-existing `if ! ...` still routes any nonzero rc, including
   gtimeout's 124, to the offline branch; no new branch added);
   `scripts/slack_mention_checker.sh:38`'s curl gained `-m 15`;
   `scripts/mas_harness/run_cycle.sh`'s claude call is now wrapped in
   `"$GTIMEOUT" -k 60 3600` (a new `GTIMEOUT` var was added -- the script
   had none before).
7. **pysvc-05** -- `backend/main.py:98-101` formatter branch order
   corrected: `settings.debug -> CompactFormatter`, default ->
   `JsonFormatter` (matches the pre-existing comment's stated intent,
   which the code had inverted).

## Change surface (measured)

`git diff --stat HEAD` (code-relevant subset; the four `handoff/audit/*`,
`handoff/.cycle_heartbeat.json`, `handoff/kill_switch_audit.jsonl`,
`handoff/current/contract.md`, `handoff/current/research_brief.md` lines
are hook-appended session audit trail from this session's own tool calls,
not authored content):

```
 .claude/hooks/pre-tool-use-danger.sh          |  15 +
 backend/main.py                               |   7 +-
 scripts/autoresearch/run_nightly.sh           |  28 ++
 scripts/away_ops/run_away_session.sh          |   7 +-
 scripts/mas_harness/run_cycle.sh              |   9 +-
 scripts/slack_mention_checker.sh              |   4 +-
 scripts/start_services.sh                     |  73 +++--
```

New (untracked) files:
```
backend/tests/test_phase_75_sre_ops.py
handoff/current/ops_rotate_runbook_75.11.md
scripts/ops/com.pyfinagent.ablation.plist.template
scripts/ops/com.pyfinagent.frontend.plist.template
scripts/ops/com.pyfinagent.logrotate.plist.template
scripts/ops/frontend_start.sh
scripts/ops/rotate_logs.sh
scripts/ops/run_ablation.sh
```
(`handoff/current/contract_75.11.md` / `research_brief_75.11.md` are
Main's/researcher's prior artifacts, not mine.)

## Verification (verbatim)

**Immutable command:**
```
cd /Users/ford/.openclaw/workspace/pyfinagent && .venv/bin/python -m pytest backend/tests/test_phase_75_sre_ops.py -q
```
Output:
```
.........................                                                [100%]
25 passed, 1 warning in 2.26s
```
Exit code: **0**.

**Ruff `--select F821,F401,F811`** over the git-derived scope, re-derived
AFTER the last edit: `git diff --name-only HEAD -- '*.py'` -> `backend/main.py`
(1 file) + the new untracked test file `backend/tests/test_phase_75_sre_ops.py`
explicitly included -> **scope_files=2**. `ruff` was not installed in
`.venv` at session start (`No module named ruff`); installed via
`.venv/bin/pip install ruff` (a lint-time convenience tool, not added to
`backend/requirements.txt`, mirroring the research-gate's pdfplumber
precedent) -> ruff 0.16.0.
```
.venv/bin/ruff check --select F821,F401,F811 backend/main.py backend/tests/test_phase_75_sre_ops.py
All checks passed!
```
Exit code: **0**.

**`bash -n`** on every touched/new `.sh` file:
```
scripts/ops/rotate_logs.sh                              OK
scripts/ops/frontend_start.sh                           OK
scripts/ops/run_ablation.sh                             OK
scripts/start_services.sh                               OK
.claude/hooks/pre-tool-use-danger.sh                    OK
scripts/away_ops/run_away_session.sh                    OK
scripts/slack_mention_checker.sh                        OK
scripts/mas_harness/run_cycle.sh                        OK
scripts/autoresearch/run_nightly.sh                     OK
```

**`plutil -lint`** on the three new `.plist.template` files (placeholders
did not break XML validity -- they only replace `<string>` element
content):
```
scripts/ops/com.pyfinagent.ablation.plist.template: OK
scripts/ops/com.pyfinagent.frontend.plist.template: OK
scripts/ops/com.pyfinagent.logrotate.plist.template: OK
```

**`ast.parse`**: clean on `backend/main.py` and the new test file
(explicit check, not just ruff's implicit parse).

**Behavioral proof of the danger-hook rail (criterion 4), driven live
before the pytest wrapper existed:**
```
pkill -9 uvicorn -> exit=2
killall next -> exit=2
pkill -f "uvicorn backend.main" -> exit=2
pkill SomeUnrelatedTool -> exit=0
pkill -9 uvicorn + CLAUDE_ALLOW_DANGER=1 -> exit=0
```
All five match the contract's expected behavior; the same five are now
permanent tests (`test_c4_*`).

**Full-suite regression** (`'.venv/bin/python -m pytest backend/tests/ -q'`):
- WITHOUT `test_phase_75_sre_ops.py` (baseline check, run first to isolate
  whether my new file was the cause of an earlier crash -- see Deviations):
  `10 failed, 1391 passed, 12 skipped, 5 xfailed, 1 xpassed, ... in 92.73s`.
- WITH `test_phase_75_sre_ops.py` (after the fix described in Deviation 2):
  `10 failed, 1416 passed, 12 skipped, 5 xfailed, 1 xpassed, ... in 88.75s`.
- The 10 failing test IDs are **byte-identical** between both runs and
  match the documented standing baseline exactly:
  `test_phase_23_2_10_watchdog_log_present_and_fresh`,
  `test_phase_23_2_15_known_pass_scripts_still_pass`,
  `test_phase_23_2_6_backend_log_has_skipping_buy_evidence`,
  `test_phase_23_2_9_backend_log_has_prewarm_evidence`,
  `test_reject_binding_main_path_off_emits_on_blocks`,
  `test_reject_binding_swap_path_off_emits_on_blocks`,
  `test_off_identity_prompts_are_verbatim_constants`,
  `test_60_1_claude_code_rail_declares_latency_profile`,
  `test_60_3_flag_defaults_off`, `test_swap_framework_fills_zero_buy_gap`.
- Delta: 1416 - 1391 = **25**, exactly the new test count. **Zero
  regressions attributable to this step**, including in the
  log-evidence family the formatter flip was flagged as a risk for
  (`test_phase_23_2_6`/`test_phase_23_2_9` are in the pre-existing
  baseline fail set already, unrelated to the formatter branch -- their
  failure mode is unchanged by pysvc-05, confirmed by them being present
  in the WITHOUT-my-file run too, before `main.py` even mattered to them).

**`.env` hash before/after** (recorded in the scratchpad before any edit,
recomputed after all edits):
```
before: sha256=2df2608773b19fd766c7d28197f91f7a829cd192c984fd27bfffa954efb28015 size=6128
after:  sha256=2df2608773b19fd766c7d28197f91f7a829cd192c984fd27bfffa954efb28015 size=6128
```
Identical.

## Mutation matrix (scripted, scratchpad
`/private/tmp/.../scratchpad/mutation_matrix_75_11.py`, exactly-once +
byte-restore pattern, mirroring the `mutation_matrix_75_9.py` convention)

**9/9 KILLED, 0 survivors** (final run, after the M3 test-quality fix
below):

| # | Mutation | Target file | Paired test(s) | Result |
|---|---|---|---|---|
| M1 | Strip the backend kickstart line | `scripts/start_services.sh` | `test_c2_kickstart_backend_and_frontend` | KILLED |
| M2 | Restore a raw `. backend/.env` line | `scripts/ops/run_ablation.sh` | `test_c3_ablation_wrapper_no_raw_dot_env_uses_sanitized_block` | KILLED |
| M3 | Revert the frontend wrapper's `exec` line to `next dev` | `scripts/ops/frontend_start.sh` | `test_c3_frontend_plist_template_wrapper_runs_next_start` | KILLED (see below) |
| M4 | Remove `-m 15` from the curl | `scripts/slack_mention_checker.sh` | `test_c5_slack_mention_checker_curl_m15` | KILLED |
| M5 | Unwrap the git-pull gtimeout | `scripts/away_ops/run_away_session.sh` | `test_c5_run_away_session_git_pull_gtimeout_wrapped` | KILLED |
| M6 | Invert the formatter branch back | `backend/main.py` | `test_c6_debug_true_uses_compact_formatter` + `test_c6_debug_false_uses_json_formatter` | KILLED |
| M7 | Neuter the danger-rail target regex (match nothing) | `.claude/hooks/pre-tool-use-danger.sh` | `test_c4_pkill_uvicorn_blocked` + 2 more | KILLED |
| M8 (STUB) | Make the danger-rail regex match everything | `.claude/hooks/pre-tool-use-danger.sh` | `test_c4_pkill_unrelated_allowed` | KILLED -- proves the allow-side assertion is real, not vacuous |
| M9 | Insert a fake 40-char secret literal in a plist template | `scripts/ops/com.pyfinagent.logrotate.plist.template` | `test_c6_no_plaintext_secrets_in_templates` | KILLED |

Every file was verified byte-identical to its pre-mutation state
immediately after each mutation's paired test ran (the harness raises if
restore fails; it never did). Final full-file re-run after all 9
mutations were applied-and-restored: `25 passed` (unchanged).

**One real test-quality defect found and fixed during the mutation
pass** (M3): my first version of `test_c3_frontend_plist_template_wrapper_runs_next_start`
did a loose `"next start -p 3000" in wrapper_text` substring check.
`frontend_start.sh` legitimately mentions that exact substring THREE
times: once in a narrative comment about `start_services.sh`'s OLD
behavior, once in an `echo` diagnostic line, and once in the real `exec`
line. Mutating only the real `exec` line to `next dev` left the other two
occurrences intact, so the naive substring check kept passing (SURVIVED
on the first mutation-matrix run). Fixed by anchoring the assertion to
the actual executable statement via
`re.search(r"^\s*exec\s+npx\s+next start\b", wrapper, re.M)` plus an
explicit negative check for a `next dev` exec line. Re-ran the full
mutation matrix after the fix: M3 now KILLED. This is exactly the
"guard that can't fail" class of defect the mutation-matrix step is
required to catch (feedback_mutation_test_guards_and_fixtures) -- caught
by the harness itself, not glossed over.

## Deviations (all disclosed)

1. **`ruff` was not installed** in `.venv` at session start
   (`.venv/bin/python -m ruff` -> `No module named ruff`). Installed via
   `.venv/bin/pip install ruff` (0.16.0) to run the required lint
   command; not added to `backend/requirements.txt` (a lint-time
   convenience, mirrors the research-gate's documented pdfplumber
   precedent for the same reason -- a tool needed to satisfy this step's
   own verification, not a production dependency).
2. **A real interpreter crash was hit and fixed mid-step, not silently
   worked around.** My first draft of the criterion-6 behavioral test
   called the REAL `backend.main.setup_logging()` three times in one
   pytest process. `setup_logging()` wraps `sys.stderr.buffer` in a FRESH
   `io.TextIOWrapper` every call and does `root.handlers.clear()`; when a
   stale handler (and its TextIOWrapper) is garbage-collected after being
   cleared, CPython's TextIOWrapper `__del__` closes the underlying
   buffer it wraps. Calling `setup_logging()` more than once against the
   REAL `sys.stderr` in one process is therefore latently unsafe
   (pre-existing behavior of the shipped function, unrelated to the
   pysvc-05 branch swap -- both branches construct the same
   TextIOWrapper). First fix attempt (swap in a throwaway buffer-backed
   `sys.stderr` for the duration of each call) made the ISOLATED test
   file pass, but the FULL SUITE run still crashed
   (`ValueError('I/O operation on closed file.')`, `lost sys.stderr`)
   later in the run, because `root.handlers.clear()` was also wiping out
   pytest's OWN log-capture handler on the root logger, corrupting later
   tests' log capture. Final fix: snapshot the root logger's handlers +
   level BEFORE each call and force-restore them in a `finally` block
   (`test_phase_75_sre_ops.py::_run_setup_logging_and_capture`), so the
   root logger is provably unchanged after each of the three calls.
   Verified: full suite (`backend/tests/ -q`) now completes cleanly with
   my file included, no crash, exactly the baseline fail set + 25 passed.
3. **Two of my own comments tripped my own text-marker tests on the first
   pytest run** (both are real, not test bugs): (a)
   `scripts/start_services.sh`'s explanatory comment about the OLD design
   literally quoted `pkill -9 uvicorn` and `> backend.log` -- both banned
   substrings per criterion 2's literal wording (no such substring
   ANYWHERE in the file, comments included). Reworded the comment to
   describe the old behavior without using those exact tokens. (b)
   `scripts/ops/com.pyfinagent.frontend.plist.template`'s memory
   cross-reference used the literal auto-memory slug
   `feedback_second_next_dev_breaks_operator_3000` (46 contiguous
   word-characters), which is indistinguishable from a base64-ish secret
   by the no-plaintext-secrets scan. Reworded to prose that describes the
   same memory without a single 30+-char contiguous token. Both fixes are
   in the shipped files, not in the test (the test's secret-shape
   heuristic is correct; my prose was the false positive source).
4. **The no-secrets test's known-var-name check was initially too
   strict**: it failed on `com.pyfinagent.frontend.plist.template`
   because that template's own comment documents (correctly, as a
   research finding) that the LIVE plist embeds `AUTH_SECRET`. Narrowed
   the check to fail only if a name is DEFINED as an actual plist
   `<key>` element (i.e. this template sources a value for it), not
   merely discussed in prose -- discussing the live plist's secret-shaped
   keys is exactly the security finding this step reacts to and must
   stay documented.
5. **`run_cycle.sh`'s new `GTIMEOUT` ceiling (3600s, `-k 60`) is a
   judgment call**, not measured from an existing precedent in that file
   (it had no prior gtimeout usage). Chosen to match the
   `-k 60 "$CAP"` idiom already used at
   `scripts/away_ops/run_away_session.sh:160` and to be generous enough
   for a full research->plan->generate->evaluate->log->push cycle
   without being unbounded. No live cycle was run to time-validate this
   number; it is a repo-shipped ceiling, not yet exercised in production.

## Not verified live

- No live BQ/backend/frontend interaction. No launchd agent bootstrapped,
  no live plist touched, no real service restarted, no real log rotated.
- The 3600s `run_cycle.sh` gtimeout ceiling has not been exercised against
  a real multi-hour cycle.
- The rotation/frontend/ablation plist TEMPLATES have not been installed
  anywhere -- `plutil -lint` proves they are well-formed XML, not that
  they behave correctly once `__REPO_ROOT__` is substituted and loaded;
  that is explicitly the OPS-ROTATE-BOOTSTRAP operator token's job.
- `backend/.env:81`'s quote defect is described but not fixed (its own
  operator token, `ENV-QUOTE-REPAIR-81`, drafted in the runbook).
- No UI surface touched; no Playwright capture applicable to this step.

## Operator tokens drafted (full text in `handoff/current/ops_rotate_runbook_75.11.md`)

- **`OPS-ROTATE-BOOTSTRAP`** -- bootstraps the three new/replacement
  launchd agents (logrotate, frontend, ablation) from their templates.
- **`ENV-QUOTE-REPAIR-81`** -- fixes the unbalanced quote at
  `backend/.env:81` (a direct `.env` edit, out of this step's boundary).
