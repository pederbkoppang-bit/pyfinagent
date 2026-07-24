# Contract -- Step 75.11: SRE hardening (log rotation, single service authority, unattended timeouts, pkill guard, formatter fix)

- **Step id**: 75.11 (phase-75, Audit75 S11) -- P1, executor sonnet-tier
- **Date**: 2026-07-24
- **Author**: Main (contract + review). **GENERATE delegated to a Sonnet-4.6 executor** (same model as 75.9/75.10: Main reviews + independently re-measures before Q/A).
- **BOUNDARY (step text + research)**: NO backend/.env edits (the .env:81 quote repair is an OPERATOR token); NO machine launchd bootstraps or modifications to live plists (bootstrap = operator token **OPS-ROTATE-BOOTSTRAP**; repo ships scripts + plist TEMPLATES + runbook only); **plist templates MUST NOT hardcode secrets** (the live plists embed plaintext AUTH_SECRET/AUTH_GOOGLE_SECRET/CLAUDE_CODE_OAUTH_TOKEN -- a research security finding; templates source from env/sourced-file, and no artifact echoes live secret values).

## Research-gate summary (gate PASSED)

Workflow `wf_9d109381-31a` (researcher, opus/max, tier=moderate).
Envelope: `external_sources_read_in_full=7, snippet_only=18, urls=25, recency_scan=true, internal_files=14, gate_passed=true`.
Brief: `handoff/current/research_brief_75.11.md`.

**Step-text corrections adopted (binding):**
1. **sre-ops-09 understated**: the LIVE `com.pyfinagent.frontend.plist` runs `next dev --port 3000` TODAY -- the fix must both collapse to one authority AND flip the surviving one to `next start` (template-only; live plist untouched).
2. **Log paths**: backend.log + frontend.log live at REPO ROOT (launchd StandardOutPath); only slack_bot.log + auto-push.log are under handoff/logs/. Rotation targets the four at their ACTUAL paths.
3. **backend.log measured at 112MB** (not 84MB); last rotation Jul 6 -- the rotation authority has been DEAD ~17 days with health.jsonl frozen at 2026-07-06 and nobody paged (the liveness alarm addresses a currently-firing gap).
4. **sre-ops-04 split**: run_nightly.sh ALREADY logs FAIL rc (:41-45) -> needs only the PAGING seam; the raw `. backend/.env` lives in the ablation PLIST ProgramArguments -> new wrapper script + plist TEMPLATE pointing at it.
5. **sre-ops-07 additive-only**: run_away_session.sh:107's `if ! git pull` already routes ANY nonzero rc to the offline branch -- the gtimeout wrap only converts an infinite hang into rc=124; no new branch needed.
6. **newsyslog RULED OUT** for the launchd logs (rename+SIGHUP reopen model breaks on launchd-held FDs); cp+truncate confirmed correct AND empirically proven on-machine (healthcheck produced .gz archives Jun 12 + Jul 6 with no restart; O_APPEND seeks to EOF post-truncate).

**Key cleared risks (measured):**
- **pysvc-05 is redaction-safe**: Python docs -- handler filters run BEFORE format(); SecretRedactionFilter rewrites record.msg before either formatter, so Compact<->JSON is invisible to it.
- **pysvc-05 is red-set-safe**: the three backend.log log-evidence tests match plain substrings that json.dumps preserves verbatim; cycle_health.py does not read backend.log; the log viewer is cosmetic. settings.debug=False measured live -> the flip changes the operator's default log format to JSON (DEBUG=true restores compact for interactive dev -- the fix makes `debug` the intuitive human-readable toggle, matching the existing comment).
- **sre-ops-05 self-lockout bounded**: the hook matches only top-level Bash TOOL command strings (never executes them); pkill inside scripts is unaffected; CLAUDE_ALLOW_DANGER=1 escape already exists at :72-75 and is checked first. Behaviorally testable offline via `CLAUDE_TOOL_NAME=Bash CLAUDE_TOOL_INPUT='{"command":"pkill -9 uvicorn"}'` -> exit 2 (+ALLOW -> exit 0). Official hooks doc confirms exit-2-blocks + stderr-is-reason.
- **launchctl kickstart -k** is the proven idiom already used at healthcheck.sh:172.

## Hypothesis

The seven SRE gaps close as repo-shipped scripts/templates/runbook + two surgical code fixes (formatter branch swap; danger-hook rail) with zero live-system mutation in this step -- machine actions deferred to OPS-ROTATE-BOOTSTRAP -- provable offline by a test file whose text-asserts are each paired with a breaking mutation and whose formatter + danger-hook legs are behavioral.

## Immutable success criteria (copied VERBATIM from .claude/masterplan.json step 75.11)

verification.command:
```
cd /Users/ford/.openclaw/workspace/pyfinagent && .venv/bin/python -m pytest backend/tests/test_phase_75_sre_ops.py -q
```

1. "New backend/tests/test_phase_75_sre_ops.py passes and asserts (reading the script files as text): a rotation plist template + rotation script exist under scripts/ops/ covering the four named logs with cp+truncate, and a watchdog-liveness (health.jsonl mtime) alarm seam exists; the runbook + OPS-ROTATE-BOOTSTRAP operator token are drafted in handoff/current/"
2. "start_services.sh contains launchctl kickstart for backend and frontend, contains NO 'pkill -9 uvicorn' / 'pkill -9 \"next dev\"' outside the flag-gated legacy branch (which uses scoped pkill -f 'uvicorn backend.main' with SIGTERM), and no '> backend.log' truncation"
3. "A frontend plist template running 'next start' with a pre-start build wrapper exists; an ablation wrapper script exists using the sanitized-sourcing block (no raw '. backend/.env') and logging FAIL rc with a paging seam; the plist template points at the wrapper"
4. "pre-tool-use-danger.sh blocks pkill/killall whose target matches python|uvicorn|next|slack_bot with the CLAUDE_ALLOW_DANGER escape hatch (test feeds sample commands through the hook's pattern and asserts block/allow)"
5. "run_away_session.sh git pull is gtimeout-wrapped falling into the offline branch on rc=124; slack_mention_checker curl carries -m 15; run_cycle.sh claude call is gtimeout-capped (text asserts)"
6. "main.py setup_logging: settings.debug selects CompactFormatter and the default path selects JsonFormatter (assert the corrected branch order); executor edits no .env and bootstraps no machine agents -- machine actions are operator-token items only"

verification.live_check: "handoff/current/live_check_75.11.md: verbatim output of this step's verification command (exit 0) + git diff --stat proving the change surface; for any flag-gated live-loop behavior an ON-vs-OFF $0 diff, and for UI-touching parts a Playwright/curl capture. Findings covered: sre-ops-01, sre-ops-02, sre-ops-09, sre-ops-04, sre-ops-05, sre-ops-07, pysvc-05"

## Plan steps

1. **sre-ops-01**: `scripts/ops/rotate_logs.sh` (cp+truncate modeled on healthcheck.sh:246-255; the four logs at their REAL paths; size caps + gzip) + `scripts/ops/com.pyfinagent.logrotate.plist.template` (user-space StartInterval; env-sourced, NO hardcoded secrets) + liveness alarm seam reading `handoff/away_ops/health.jsonl` mtime>2h + runbook `handoff/current/ops_rotate_runbook_75.11.md` + operator token **OPS-ROTATE-BOOTSTRAP** drafted in handoff/current/.
2. **sre-ops-02**: start_services.sh -> `launchctl kickstart -k gui/$UID/com.pyfinagent.{backend,frontend}` (healthcheck.sh:172 idiom); legacy path behind an explicit flag with scoped SIGTERM `pkill -f 'uvicorn backend.main'` + wait; no backend.log truncation anywhere.
3. **sre-ops-09**: `scripts/ops/com.pyfinagent.frontend.plist.template` running `next start -p 3000` + pre-start build wrapper; start_services frontend leg = kickstart only (one authority; honors the second-next-dev memory). Live plist NOT touched.
4. **sre-ops-04**: ablation wrapper script reusing run_nightly.sh:19-27's sanitized-grep sourcing verbatim + FAIL-rc logging + paging seam (bot-token page after N consecutive failures); ablation plist TEMPLATE points at the wrapper; run_nightly.sh gains the paging seam only.
5. **sre-ops-05**: pkill/killall rail inside pre-tool-use-danger.sh's Bash block (after :95), target regex NARROW (python|uvicorn|next|slack_bot), exit 2 + stderr pointing at `launchctl kickstart -k`; existing CLAUDE_ALLOW_DANGER escape covers it.
6. **sre-ops-07**: `"$GTIMEOUT" -k 10 120` on run_away_session.sh:107 git pull; `-m 15` on slack_mention_checker.sh:38 curl; `"$GTIMEOUT"` cap on run_cycle.sh:60 claude.
7. **pysvc-05**: swap main.py:98-101 branch order (debug -> Compact, default -> Json). BEHAVIORAL test: both debug values -> assert formatter class; + an `api_key=SECRET...` record through the JSON path -> emitted message REDACTED (branch order + redaction survival in one test).
8. **Tests**: text asserts EACH paired with a named breaking mutation; behavioral legs for criteria 4 (drive the real hook script via env vars, zero real kills) and 6 (formatter). Criterion-1/3 file-existence asserts must also check content markers (cp+truncate lines, sanitized-sourcing block, no plaintext secrets in templates).
9. **Mutation matrix**: strip kickstart; restore raw sourcing; revert template to next dev; remove -m 15; unwrap gtimeout; invert formatter back; neuter the danger-rail regex; STUB mutation on the hook test (feed an allowed command and assert the test would catch a hook that blocks everything).
10. **live_check_75.11.md**: verbatim pytest exit 0 + git diff --stat + the danger-hook behavioral transcript. No UI; no flag-gated live-loop behavior (all machine actions deferred to the operator token).

## Explicitly NOT in scope

- backend/.env (any edit; the :81 quote repair is an operator token drafted in the runbook)
- Bootstrapping/modifying ANY live launchd agent or plist (OPS-ROTATE-BOOTSTRAP)
- Restarting services; deleting/truncating any live log in this step
- newsyslog-based designs (ruled out)

## References

- `handoff/current/research_brief_75.11.md` (7 read-in-full: Apple/launchd + newsyslog man + logrotate copytruncate + launchctl kickstart + Claude Code hooks reference + GNU timeout + Python logging filter docs)
- `handoff/current/audit_phase75/confirmed_findings.json` (sre-ops-01/02/04/05/07/09, pysvc-05)
- feedback_second_next_dev_breaks_operator_3000; feedback_mutation_test_guards_and_fixtures; CLAUDE.md Harness Protocol
