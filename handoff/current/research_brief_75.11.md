# Research Brief — Step 75.11 (SRE hardening)

**Tier:** moderate (caller-specified). NOT audit-class.
**Step:** 75.11 — SRE hardening: always-on log rotation, single service
authority, unattended-wrapper timeouts, pkill guard, log-formatter fix.
**Executor:** sonnet-4.6/high. **Started:** 2026-07-24 (WRITE-FIRST skeleton).

7 findings in scope: sre-ops-01 (log rotation agent), sre-ops-02
(start_services.sh kickstart), sre-ops-09 (single frontend authority),
sre-ops-04 (ablation wrapper), sre-ops-05 (danger-hook pkill rail),
sre-ops-07 (gtimeout wraps), pysvc-05 (main.py formatter inversion).

BOUNDARY (immutable): NO .env edits (sanitized sourcing instead;
backend/.env:81 quote repair is an OPERATOR token). NO machine launchd
bootstraps (repo ships files+runbook; bootstrap = operator token
OPS-ROTATE-BOOTSTRAP).

---

## Status: COMPLETE — gate_passed: true

_(envelope at tail is authoritative for the gate)_

---

## Internal code inventory (re-anchoring the 7 findings)

All 7 anchors re-hunted live on 2026-07-24. Result: anchors are ACCURATE this
time (rare) with a few important corrections/additions below.

| Finding | File:line (verified) | State on disk |
|---|---|---|
| sre-ops-01 | `scripts/away_ops/healthcheck.sh:246-255` (rotation block; audit said :248 -> the `if` is :248, block spans 246-255) | Rotates ONLY `backend.log`, cap `52428800` (50MB) via `cp` -> `: > "$BLOG"` -> `gzip`. cp+truncate rationale in comment :242-243 ("launchd FDs carry O_APPEND"). NO authority for frontend.log / slack_bot.log / auto-push.log. |
| sre-ops-02 | `scripts/start_services.sh:11` `pkill -9 uvicorn`, `:12` `pkill -9 "next dev"`, `:18` `> backend.log` (TRUNCATES), `:23` `npx next start -p 3000` | No `launchctl kickstart` anywhere. `:18` truncates the launchd-owned backend.log with `>`. `:12` pkills `next dev` but `:23` starts `next start` (mismatch). |
| sre-ops-09 | frontend authority split | `~/Library/LaunchAgents/com.pyfinagent.frontend.plist` runs **`next dev --port 3000`** (NOT next start!); `start_services.sh:23` runs `next start -p 3000`. TWO authorities, DIFFERENT modes. plist has `KeepAlive=true`, logs to `frontend.log`. |
| sre-ops-04 | `~/Library/LaunchAgents/com.pyfinagent.ablation.plist` ProgramArguments uses raw `. backend/.env` | Confirmed: `bash -c "cd ... && set -a && . backend/.env && set +a && ..."`. Crashes on backend/.env:81 unbalanced quote. StandardOut = `handoff/ablation.launchd.log` (0 bytes -> job dies before logging). `handoff/logs/ablation.launchd-v3.log` = 0 bytes. Reuse block = `run_nightly.sh:19-27`. |
| sre-ops-05 | `.claude/hooks/pre-tool-use-danger.sh` | NO pkill/killall rail. Escape hatch `CLAUDE_ALLOW_DANGER=1` ALREADY exists + checked FIRST at :72-75. Bash-cmd block is `:78`+; `cmd` extracted :83-95. Invocation contract below. |
| sre-ops-07 | `run_away_session.sh:107` git pull; offline branch :108-110; `$GTIMEOUT` defined `:21` = `/opt/homebrew/bin/gtimeout` (installed, verified). `slack_mention_checker.sh:38` curl (no `-m`); `mas_harness/run_cycle.sh:59-66` claude call (`"$CLAUDE_BIN"` starts :60, no cap) | Existing gtimeout idiom in same file: `:141` `"$GTIMEOUT" -k 5 20 ...`, `:160` `"$GTIMEOUT" -k 60 "$CAP" ...`. |
| pysvc-05 | `backend/main.py:98-101` | **INVERTED**: `if settings.debug: JsonFormatter() else: CompactFormatter()`. Comment :97 says "compact for local dev, JSON for production" -- code does the opposite. Fix: swap so `debug -> CompactFormatter`, default -> `JsonFormatter`. Formatters defined :43-71; both use `record.getMessage()` (:53, :67). |

### Corrections to the 75.11 step text (verified drift)
- **sre-ops-09 understated the drift**: the live frontend plist runs `next dev`,
  not merely "a second authority" -- so the fix is BOTH (a) collapse to one
  authority AND (b) flip the surviving authority from dev to `next start`. The
  "second next-dev breaks operator :3000" memory is doubly relevant: the plist
  IS the `next dev` today.
- **sre-ops-01 log paths**: audit lists `handoff/logs/slack_bot.log` +
  `handoff/logs/auto-push.log`. Verified those two ARE under `handoff/logs/`,
  BUT `backend.log` + `frontend.log` are at REPO ROOT (launchd StandardOutPath),
  not `handoff/logs/`. The rotation agent must target all four at their ACTUAL
  paths (2 root, 2 under handoff/logs/).
- **health.jsonl is 17 days stale** (mtime 2026-07-07, last line ts
  2026-07-06T23:03) -> the away-watchdog rotation authority is not merely
  "dead," it stopped ~Jul 7 and NOBODY was paged -- direct justification for
  the mtime>2h liveness alarm. `health.jsonl` lives at
  `handoff/away_ops/health.jsonl`; written by `healthcheck.sh:266-270`.

### Live-state measurements (the load-bearing numbers)
- `backend.log` = **112MB** (audit said 84MB; still growing, 2.2x its 50MB cap).
  `frontend.log` = 15MB, `handoff/logs/slack_bot.log` = 5.0MB,
  `handoff/logs/auto-push.log` = 692KB.
- Prior rotations exist: `handoff/logs/backend.log.20260706T225648Z.gz` (Jul 6,
  last), `...20260612T104931Z.gz` (Jun 12). None since Jul 6 = 18 days = the
  watchdog death window.
- **`settings.debug = False`** on the operator's live machine, `log_level=INFO`.
  This is the pivot for pysvc-05's risk (below).
- launchd service states now: backend=running, slack-bot=running,
  frontend=**not running**, away-watchdog=**not running** (loaded, StartInterval
  1800s, but not firing). No `test_phase_75_sre_ops.py` yet.
- SECURITY: `com.pyfinagent.frontend.plist` and `.away-watchdog.plist` embed
  PLAINTEXT secrets (AUTH_SECRET, AUTH_GOOGLE_SECRET, CLAUDE_CODE_OAUTH_TOKEN).
  The executor's rotation/frontend plist TEMPLATES must NOT hardcode secrets;
  read them from the environment / a sourced env, and the brief/tests must never
  echo the live values.

### pre-tool-use-danger.sh invocation contract (for the sre-ops-05 test)
- Hook reads env `CLAUDE_TOOL_NAME` / `CLAUDE_TOOL_INPUT`; if empty AND stdin is
  non-interactive, parses stdin JSON for `tool_name` + `tool_input` (:19-43).
- For Bash, the command is `tool_input.command` (:83-88).
- **Exit 2 = block**, exit 0 = allow (PreToolUse convention, :57-68). Block also
  prints a human message to stderr.
- `CLAUDE_ALLOW_DANGER=1` short-circuits to allow at :72-75 (BEFORE any pattern)
  -> a new pkill rail automatically inherits this escape.
- TEST can drive the REAL script offline with NO real kills by setting
  `CLAUDE_TOOL_NAME=Bash CLAUDE_TOOL_INPUT='{"command":"pkill -9 uvicorn"}'` and
  asserting exit 2; and `CLAUDE_ALLOW_DANGER=1` + same -> exit 0. The pattern is
  matched against the command STRING; the hook never executes it, so no process
  is signalled. (This is the behavioral leg.)
- SELF-LOCKOUT check: the hook fires on the Bash TOOL's top-level command only.
  A pkill INSIDE a script (`start_services.sh`'s own `pkill`, or the legacy
  scoped `pkill -f 'uvicorn backend.main'`) is a subprocess, NOT a Bash tool
  call -> unaffected. So `bash scripts/start_services.sh` still runs; only a
  top-level `pkill ...` typed as a Bash tool command is gated. The rail should
  match `pkill`/`killall` targeting `python|uvicorn|next|slack_bot` and point
  the operator at `launchctl kickstart -k`. Because THIS session runs under the
  hook, keep the target regex narrow (those 4 tokens) so unrelated `pkill`
  (e.g. `pkill -f some_test_fixture`) is not caught, and rely on
  CLAUDE_ALLOW_DANGER=1 as the documented escape.

### pysvc-05 -- SecretRedactionFilter safety + operator-visible consequence
- `SecretRedactionFilter` (`backend/services/observability/log_redaction.py:30-47`)
  is a **handler-level** `logging.Filter` added at `main.py:107-108`. It operates
  on `record.getMessage()` and rewrites `record.msg` in place + clears
  `record.args` (:40-44), returning True always. **It runs in `Handler.handle()`
  BEFORE `emit()`/`format()`** -- so it sees the RAW record, never the formatted
  line. Switching the formatter Compact<->JSON does NOT change what the filter
  sees or redacts; the already-redacted `getMessage()` is what BOTH formatters
  serialize (Compact :53, JSON :67). The audit's worry ("match inside JSON
  strings?") is MOOT: redaction happens on `key=value` text BEFORE json.dumps
  wraps it. CONFIRMED SAFE, no new leak. (JsonFormatter only emits
  timestamp/level/module/message/stack_trace; stack_trace redaction is unchanged
  by the swap -- pre-existing behavior for both formatters.)
- **HIGHEST-RISK operator consequence, quantified**: debug=False today ->
  CompactFormatter (colored). After the fix, debug=False -> JsonFormatter, so
  the operator's live `backend.log` (launchd-managed, debug=False) FLIPS from
  colored-compact to JSON lines, AND a manual `uvicorn` run in the operator's
  terminal shows JSON instead of colored compact. Mitigation: `DEBUG=true` for
  interactive local dev now yields the human-readable compact format (the fix
  makes `debug` the intuitive human-toggle, matching the :97 comment's intent).
- **Consumer audit (does anything PARSE the compact format and break on JSON?)**:
  Enumerated every backend.log/frontend.log reader in the repo:
  | Consumer | Match mechanism | Breaks on JSON? |
  |---|---|---|
  | `test_phase_23_2_9_ticker_meta_latency.py:76` | `text.count("Prewarming ticker-meta cache")` | NO -- substring survives inside JSON `message` |
  | `test_phase_23_2_13_governance_watcher.py:77,111,135` | `text.count("governance: immutable limits loaded")` etc | NO -- plain substrings, no quotes/backslash/newline |
  | `test_phase_23_2_6_sector_cap_emit.py:240+` | `count("Skipping BUY")`; ALREADY rotation-archive-aware | NO |
  | `backend/services/cycle_health.py` | `readlines()` at :186/:351/:378 are NOT backend.log (grep: no backend.log ref) | N/A -- not a consumer |
  | `backend/api/cron_dashboard_api.py:147` | log-VIEWER tail served to /cron UI | Cosmetic only -- renders JSON lines; no functional break; UI-touching -> live_check curl/Playwright |
  | `scripts/launchd/backend_watchdog.sh` | SIGUSR1 dump target, does not PARSE format | NO |
  The red-set tests count MESSAGE SUBSTRINGS; `json.dumps` escapes only
  `"`/`\`/control chars, none of which appear in those substrings, so they match
  verbatim in JSON too -- even in a MIXED-format log during the restart
  transition. **The immutable criterion is satisfiable with zero red-set
  breakage.** The executor MUST still spot-check any NEW red-set pattern for
  embedded quotes before relying on it.

---

## External research

### Read in full (7; counts toward the gate)
| URL | Accessed | Kind | Fetched | Key finding |
|---|---|---|---|---|
| https://code.claude.com/docs/en/hooks | 2026-07-24 | official doc | WebFetch | PreToolUse: exit 2 blocks the tool call, "Claude Code ignores stdout... stderr text is fed back to Claude as an error message". Bash cmd = `tool_input.command`. "command hooks can block by exit code alone" -- exit 2 sufficient, no JSON needed. Also a NEWER structured path: exit 0 + `hookSpecificOutput.permissionDecision` = allow/deny/ask/defer ("either exit codes alone OR exit 0 + JSON, not both"). |
| https://docs.python.org/3/library/logging.html | 2026-07-24 | official doc | WebFetch | VERBATIM: "filters attached to handlers are consulted **before an event is emitted by the handler**". Sequence = filter() -> format() -> emit(). `Handler.handle()`: "Conditionally emits... depending on filters... Wraps the actual emission". `record.message` (=`msg % args`) "is only set when `Formatter.format()` is invoked". Confirms handler-filter sees RAW record, not formatted text. Also the descendant-logger rule (why redaction is at handler, not logger, level). |
| https://eclecticlight.co/2019/08/27/kickstarting-and-tearing-down-with-launchctl/ | 2026-07-24 | authoritative blog | WebFetch | `kickstart` "run a particular service immediately, either to replace an existing service...". `-k` = "kill the running service before restarting it". Target `user/uid/name`. Atomic stop+start under launchd supervision. |
| https://www.datadoghq.com/blog/log-file-control-with-logrotate/ | 2026-07-24 | industry | WebFetch | copytruncate: copy then truncate the SAME file in place; "any data that was logged after the copy process but before the truncate operation will be lost" (the data-loss window). Used when a daemon "cannot handle termination signals like SIGHUP to close and reopen its log file descriptor"; keeps the same open FD writing uninterrupted. |
| https://patelhiren.com/blog/macos-newsyslog-openclaw-logs/ | 2026-07-24 | practitioner blog | WebFetch | macOS newsyslog runs via a system launchd daemon "every 30 minutes"; config fields logfile/owner:group/mode/count/size/when/flags. CRITICAL ownership caveat: system newsyslog creates rotated files `root:root` -> a USER launchd service then gets "Permission denied" and "the daemon fails to start" -- must set owner:group. |
| https://www.man7.org/linux/man-pages/man1/timeout.1.html | 2026-07-24 | official man | WebFetch | `-k DURATION` "also send a KILL signal if COMMAND is still running this long after the initial signal". Default signal TERM. Exit 124 on timeout (unless --preserve-status), 137 if KILL(9)-ed, else the command's status. DURATION suffix s/m/h/d. |
| https://man.freebsd.org/cgi/man.cgi?newsyslog.conf(5) | 2026-07-24 | official man | WebFetch | newsyslog "rotates by renaming rather than truncating. The process is notified via signal to reopen its log" (pid_file + signal, default SIGHUP). Flags: Z=gzip, J=bzip2, B=binary (suppress the ASCII rotation marker). This RENAME+signal model is the load-bearing mismatch for launchd stdout logs (see Key findings #2). |

### Identified but snippet-only (context; does NOT count toward gate)
| URL | Kind | Why not fetched |
|---|---|---|
| https://github.com/logrotate/logrotate/issues/34 | code/canonical | copytruncate data-loss debate; datadog covered the mechanism |
| https://community.splunk.com/.../Why-copytruncate-logrotate-does-not-play-well | community | copytruncate FD race, lower tier |
| https://www.gnu.org/software/coreutils/timeout | official doc | HTTP 429; man7 mirror read instead |
| https://linuxcommandlibrary.com/man/timeout | doc mirror | redundant with man7 |
| https://www.morphllm.com/claude-code-hooks | blog | redundant with official hooks doc |
| https://thepromptshelf.dev/blog/claude-code-hooks-complete-reference-2026/ | blog | redundant with official hooks doc |
| https://ss64.com/mac/launchctl.html | doc mirror | kickstart syntax, covered by eclecticlight |
| https://developer.apple.com/forums/thread/768741 | forum | KeepAlive-disable Q, tangential |
| https://developer.apple.com/library/archive/.../CreatingLaunchdJobs.html | official doc | plist background, not rotation-specific |
| https://www.launchd.info/ | doc | launchd tutorial, general |
| https://richard-purves.com/2017/11/08/log-rotation-mac-admin-cheats-guide/ | blog | newsyslog cheat guide, covered |
| https://codedmemes.com/lib/newsyslog-automatic-log-rotation/ | blog | newsyslog perf, covered |
| https://en.wikipedia.org/wiki/Log_rotation | encyclopedia | copytruncate vs create overview |
| https://www.toptal.com/developers/python/in-depth-python-logging | blog | logging handlers, covered by official |
| https://osxhub.com/macos-process-management-commands-guide/ | blog | pkill/killall/launchctl compare |
| https://runebook.dev/en/docs/python/library/logging/logging.Filter.filter | doc mirror | filter traps, covered |
| https://www.man7.org/linux/man-pages/man1/kill.1.html | official man | signal numbers |
| https://discussions.apple.com/thread/6752216 | forum | "DNS stops logging after rotation" = the rename-without-reopen failure, corroborates Key finding #2 |

URLs collected: ~25. Read in full: 7. Snippet-only: 18.

---

## Recency scan (2024-2026)
Searched 2026 + 2025 frontier + year-less canonical for all seven topics
(hooks, launchctl, copytruncate, newsyslog, python logging, gtimeout).
- **NEW (2026)**: the Claude Code hooks contract has EVOLVED past bare exit
  codes -- exit 0 + `hookSpecificOutput.permissionDecision` (allow/deny/ask/
  defer) is the structured path now (official doc, current). The pyfinagent
  hook uses the exit-2 path, which remains fully supported ("block by exit
  code alone"); no migration required, but the executor MAY use either and
  should stay consistent with the file's existing exit-2 idiom.
- **NO material change** to: launchctl kickstart -k (stable since 10.10),
  copytruncate semantics (logrotate long-settled), newsyslog rename+signal
  model (BSD-stable), GNU timeout -k/exit-124 (coreutils-stable), Python
  logging filter/handler ordering (the only add is the 3.12 filter-return-a-
  record feature, which pyfinagent does NOT need -- it has ONE handler, so
  in-place mutation has no cross-handler side effect).
- Older canonical sources (eclecticlight 2019, BSD man pages) remain
  authoritative; no 2025-2026 work supersedes them.

---

## Key findings
1. **The formatter switch (pysvc-05) is redaction-SAFE, provably.** Python docs
   (verbatim): handler filters are "consulted before an event is emitted by the
   handler" -- sequence filter()->format()->emit(). `SecretRedactionFilter`
   redacts `record.getMessage()` and rewrites `record.msg` BEFORE either
   formatter runs, so Compact<->JSON is invisible to the filter; the JSON
   `message` field carries the already-redacted text (`main.py:67` serializes
   `record.getMessage()`). The audit's "match inside JSON strings?" worry is
   moot -- redaction precedes json.dumps. (Source: docs.python.org logging;
   backend/services/observability/log_redaction.py:38-47.)
2. **newsyslog is the WRONG native tool for launchd StandardOutPath logs.**
   FreeBSD man (verbatim): newsyslog "rotates by renaming rather than
   truncating. The process is notified via signal to reopen its log." But
   backend.log/frontend.log are captured by launchd (StandardOutPath), so the
   FD is held by LAUNCHD, not the child -- there is no process to SIGHUP into
   reopening, and after a rename the launchd FD keeps writing to the moved
   inode (Apple-forum "DNS stops logging after rotation" is exactly this
   failure). Plus system-newsyslog creates `root:root` files a user agent
   cannot write (patelhiren). CONCLUSION: the cp+truncate design in the step is
   CORRECT and newsyslog is NOT the drop-in native alternative for these two
   logs. (A user-space `newsyslog -f custom.conf` on a StartInterval agent
   COULD rotate slack_bot.log/auto-push.log IF their writer reopens on signal
   -- it does not for these either, so cp+truncate is uniform-correct.)
3. **cp+truncate is required (not mv) for O_APPEND launchd FDs** -- empirically
   proven on THIS machine: healthcheck.sh's cp+truncate produced
   `handoff/logs/backend.log.*.gz` on Jun 12 + Jul 6 with NO backend restart.
   POSIX O_APPEND writes seek-to-EOF atomically; after `: > file` EOF=0 so the
   same held FD grows the file fresh. datadog confirms the copytruncate
   trade-off is a small data-loss window ("data logged after the copy but
   before the truncate... will be lost") -- ACCEPTABLE for backend.log (INFO
   noise; the money records live in BigQuery, not the log).
4. **The danger-hook rail is behaviorally testable offline with ZERO kills.**
   Official hooks doc: exit 2 blocks, stderr is the reason, Bash cmd =
   `tool_input.command`. The hook matches the command STRING and never executes
   it, so a test sets `CLAUDE_TOOL_NAME=Bash CLAUDE_TOOL_INPUT='{"command":
   "pkill -9 uvicorn"}'` -> assert exit 2; add `CLAUDE_ALLOW_DANGER=1` -> assert
   exit 0. Escape hatch ALREADY exists (:72-75) and is checked first, so the new
   rail inherits it. Self-lockout is bounded: the hook only sees the Bash TOOL's
   top-level command, so `pkill` inside a script (start_services' own pkill,
   the legacy scoped `pkill -f 'uvicorn backend.main'`) is unaffected.
5. **settings.debug=False on the live machine -> the fix flips backend.log to
   JSON.** Quantified operator impact: the launchd backend (debug=False) and any
   manual `uvicorn` run switch from colored-compact to JSON lines. The three
   red-set log-evidence tests survive (they `text.count()` plain message
   substrings that json.dumps preserves verbatim), and cycle_health.py does not
   read backend.log; only the /cron log-viewer renders differently (cosmetic).
   Mitigation: `DEBUG=true` for interactive dev restores compact -- the fix
   makes `debug` the intuitive human-readable toggle (matching the main.py:97
   comment's stated intent).
6. **The rotation authority has been dead ~17 days and nobody was paged** --
   health.jsonl frozen at 2026-07-06, backend.log at 112MB (2.2x cap). This is
   the concrete justification for the watchdog-liveness (health.jsonl mtime>2h)
   alarm the step requires; it is a real, currently-firing gap.
7. **launchctl kickstart -k is the supervised-restart primitive** (eclecticlight:
   "-k = kill the running service before restarting it") -- replacing
   start_services' `pkill -9` + `nohup` (which orphans an unsupervised process
   while launchd KeepAlive crash-loops the twin). healthcheck.sh:172 already
   uses `kickstart -k gui/$UID/...` as the proven idiom to copy.
8. **GNU timeout -k gives the two-stage TERM->KILL + rc=124** the offline branch
   needs; the existing `if ! git pull` at run_away_session.sh:107 ALREADY routes
   any nonzero (incl. 124) to the offline branch, so the gtimeout wrap is purely
   additive -- it makes a HANG return instead of holding the lock forever.

---

## Application to pyfinagent (external -> file:line)
- **sre-ops-01**: cp+truncate (Key #2,#3) for all four logs at their REAL paths
  (`backend.log`,`frontend.log` at repo root; `handoff/logs/slack_bot.log`,
  `handoff/logs/auto-push.log`). Model the rotation SCRIPT on
  healthcheck.sh:246-255; ship a `scripts/ops/` rotation script + a user
  StartInterval plist TEMPLATE (bootstrap = OPS-ROTATE-BOOTSTRAP operator
  token). Liveness alarm reads `handoff/away_ops/health.jsonl` mtime.
- **sre-ops-02**: `start_services.sh:11-12,18,23` -> `launchctl kickstart -k
  gui/$UID/com.pyfinagent.{backend,frontend}` (copy healthcheck.sh:172); drop
  the `> backend.log` truncation; legacy path behind a flag using scoped
  `pkill -f 'uvicorn backend.main'` + SIGTERM-then-wait.
- **sre-ops-09**: FLIP `com.pyfinagent.frontend.plist` template from `next dev`
  to `next start -p 3000` + a pre-start build wrapper; reduce start_services'
  frontend leg to kickstart. (The plist runs `next dev` TODAY -- verified.)
- **sre-ops-04**: ablation wrapper REUSING run_nightly.sh:19-27 sanitized block
  (replaces ablation.plist's raw `. backend/.env`); add the PAGING seam
  (run_nightly.sh already logs FAIL rc at :41-45, so it needs paging ONLY).
- **sre-ops-05**: add pkill/killall rail inside the `if [ "$TOOL" = "Bash" ]`
  block (pre-tool-use-danger.sh:78+), targets matching
  `python|uvicorn|next|slack_bot`, exit 2, stderr pointing at
  `launchctl kickstart -k`; escape via existing CLAUDE_ALLOW_DANGER=1.
- **sre-ops-07**: wrap run_away_session.sh:107 git pull in `"$GTIMEOUT" -k 10
  120` (var defined :21, idiom at :141/:160); `slack_mention_checker.sh:38`
  curl gets `-m 15`; `run_cycle.sh:60` claude call gets a `"$GTIMEOUT"` cap.
- **pysvc-05**: swap main.py:98-101 so `if settings.debug: CompactFormatter()
  else: JsonFormatter()`. Behavioral test: instantiate setup_logging under both
  debug values, assert `handler.formatter` class; plus feed
  `"api_key=SECRETVALUE123"` through the JSON path and assert the emitted
  message is redacted (proves branch order AND redaction survival together).

### Vacuous-guard mutations the Q/A must see (per feedback_mutation_test_guards)
- pysvc-05: BEHAVIORAL (not a source-scan) -- invert the branch back and the
  formatter-class assertion must FAIL; break the filter and the redaction
  assertion must FAIL. Do NOT settle for `grep JsonFormatter main.py`.
- sre-ops-05: BEHAVIORAL -- feed a matching pkill and a non-matching command
  through the REAL hook; both block/allow assertions must flip if the rail is
  stripped. Mutate the escape hatch too (CLAUDE_ALLOW_DANGER=1 must allow).
- sre-ops-02/07/09/04: text-asserts -- for each, name the mutation that breaks
  it (strip the kickstart line; restore raw `. backend/.env`; revert plist to
  `next dev`; remove `-m 15`; unwrap gtimeout). A guard that cannot fail is a
  finding.

---

## Internal code inventory (files inspected)
| File | Lines read | Role | Status |
|---|---|---|---|
| scripts/away_ops/healthcheck.sh | 1-273 (full) | rotation authority + frontend restart | live idiom source |
| scripts/start_services.sh | 1-40 (full) | pkill+nohup service start | needs rewrite (sre-ops-02) |
| .claude/hooks/pre-tool-use-danger.sh | 1-264 (full) | PreToolUse guard | needs pkill rail (sre-ops-05) |
| scripts/autoresearch/run_nightly.sh | 1-47 (full) | sanitized-source template | REUSE :19-27 |
| scripts/slack_mention_checker.sh | 1-82 (full) | curl poller | :38 needs -m 15 |
| scripts/mas_harness/run_cycle.sh | 1-80 (full) | claude cycle wrapper | :60 needs gtimeout |
| scripts/away_ops/run_away_session.sh | 80-149 | git-pull sync + gtimeout idiom | :107 needs gtimeout |
| backend/main.py | 1-144 | setup_logging + formatters | :98-101 inverted (pysvc-05) |
| backend/services/observability/log_redaction.py | 1-48 (full) | handler-level redaction | safe across switch |
| ~/Library/LaunchAgents/com.pyfinagent.{frontend,away-watchdog,ablation,backend}.plist | full | live plists (READ-ONLY) | frontend=next dev; ablation=raw .env |
| backend/tests/test_phase_23_2_{6,9,13}*.py | matchers | red-set log-evidence tests | substring-safe vs JSON |
| backend/api/cron_dashboard_api.py | 140-160 | log-viewer endpoint | cosmetic only |

---

## Research Gate Checklist
Hard blockers (all satisfied):
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (7)
- [x] 10+ unique URLs total (~25)
- [x] Recency scan (last 2 years) performed + reported
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every relevant module (7 scripts + main.py +
      redaction + 4 plists + 3 red-set tests + log-viewer)
- [x] Contradictions / consensus noted (newsyslog-vs-cp+truncate; hooks
      exit-code-vs-JSON path)
- [x] Claims cited per-claim with URL/file:line

---

## JSON envelope

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 7,
  "snippet_only_sources": 18,
  "urls_collected": 25,
  "recency_scan_performed": true,
  "internal_files_inspected": 14,
  "coverage": {
    "audit_class": false,
    "rounds": 1,
    "dry_rounds": 0,
    "K_required": 2,
    "new_findings_last_round": 0,
    "dry": false
  },
  "summary": "All 7 anchors re-verified live (accurate this cycle). Key drift: frontend.plist runs `next dev` not `next start` (sre-ops-09 fix must flip mode AND collapse authority); backend.log now 112MB (audit 84MB); backend.log+frontend.log are at repo root, only slack_bot/auto-push under handoff/logs. pysvc-05 is provably redaction-SAFE: Python docs confirm handler filters run BEFORE formatting, so SecretRedactionFilter (rewrites record.msg pre-format) is formatter-independent; the three red-set backend.log tests count plain substrings that survive json.dumps, so the criterion is satisfiable with zero breakage. settings.debug=False live, so the fix flips the operator's backend.log to JSON (mitigate via DEBUG=true for interactive). newsyslog is the WRONG native tool for launchd StandardOutPath logs (rename+SIGHUP model; launchd owns the FD) -- cp+truncate is uniform-correct and empirically proven on-machine. Danger-hook rail is behaviorally testable offline (exit 2 blocks; CLAUDE_ALLOW_DANGER=1 escape already exists at :72; self-lockout bounded to top-level Bash commands). gtimeout -k gives TERM->KILL+rc=124; the existing `if ! git pull` already routes 124 to offline, wrap is additive. health.jsonl 17 days stale = live justification for the mtime alarm.",
  "brief_path": "handoff/current/research_brief_75.11.md",
  "gate_passed": true
}
```
