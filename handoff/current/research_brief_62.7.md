# Research Brief — phase-62.7 (goal-away-ops): Pre-departure dress rehearsal (Sunday 2026-06-14)

Tier: moderate (caller-set). Date: 2026-06-12. Researcher: Layer-3 (merged Explore).
Status: COMPLETE. Scope note: the 5-part internal audit pushed tool calls (~27) and length past the
moderate guideline; depth of analysis kept at moderate. Job = validate/harden the EXISTING checklist
at `handoff/away_ops/dress_rehearsal.md`, not invent one.

## Read in full (>=5 required; counts toward the gate)

| URL | Accessed | Kind | Fetched how | Key quote / finding |
|---|---|---|---|---|
| https://support.apple.com/guide/deployment/intro-to-filevault-dep82064ec40/web | 2026-06-12 | official doc | WebFetch full | "After a user turns on FileVault on a Mac, their credentials are required during the boot process" — auto-login impossible with FV on. NEW: "On a Mac with Apple silicon with macOS 26 or later, FileVault can be unlocked over ssh after a restart if Remote Login is turned on" |
| https://support.apple.com/guide/mac-help/protect-data-on-your-mac-with-filevault-mh11785/mac | 2026-06-12 | official doc | WebFetch full | "Users unlock the encrypted disk with their login password" — FV gates boot on a password |
| https://ss64.com/mac/pmset.html | 2026-06-12 | man-page mirror | WebFetch full | "autorestart - Automatic restart on power loss (value = 0/1)"; set via `sudo pmset -a autorestart 1`; read via `pmset -g`. Related: womp, powernap, sleep 0 to disable system sleep |
| https://support.apple.com/guide/mac-help/software-update-settings-on-mac-mchla7037245/mac | 2026-06-12 | official doc | WebFetch full | Exact toggle names: "Download new updates when available", "Install macOS updates", "Install Security Responses and system files"; "Update Tonight" requires Mac on + plugged in. No deferral mechanism documented for non-MDM Macs |
| https://eclecticlight.co/2025/02/27/how-your-mac-can-update-macos-when-you-dont-want-it-to/ | 2026-06-12 | authoritative blog (Howard Oakley) | WebFetch full | Hidden `com.apple.SUOSUScheduler.tonight.install` DAS-CTS activity installs + restarts "DoItLater = 1" even against user settings; "Activities scheduled by DAS-CTS are hidden from the user"; clicking "tonight" on a notification arms it |
| https://principlesofchaos.org/ | 2026-06-12 | canonical reference | WebFetch full | "it is the responsibility and obligation of the Chaos Engineer to ensure the fallout from experiments are minimized and contained" (minimize blast radius); experiment on production, but contained |
| https://adhoc.team/2023/01/31/engineering-chaos-a-guide-to-building-a-game-day/ | 2026-06-12 | practitioner guide | WebFetch full | "Share the planned scenario and expected results with relevant individuals and leadership to review and approve"; estimate resolution time beforehand; timebox 2-3h; blameless postmortem |

## Identified but snippet-only (does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|---|---|---|
| https://queue.acm.org/detail.cfm?id=2371516 (Google DiRT, "Weathering the Unexpected") | peer-reviewed/ACM | 403 Forbidden on fetch; Ad Hoc guide substituted as drill-design authority |
| https://docs.aws.amazon.com/wellarchitected/latest/reliability-pillar/rel_testing_resiliency_game_days.html | official doc | JS shell returned no body; principles covered by the two fetched drill sources |
| https://support.apple.com/guide/mac-help/set-your-mac-to-automatically-log-in-at-startup-mchlp1675/mac | official doc | Fetch returned only the guide's TOC; FV/auto-login claim sourced from the deployment doc instead |
| https://discussions.apple.com/thread/256183030 + /256199871 | community | Tahoe (macOS 26) FV/auto-login stuck-conflicting-states bug; FV enabled-by-default on Tahoe upgrades |
| https://learn.jamf.com/.../Deferring_maOS_Software_Upgrades_and_Updates.html | vendor doc | MDM-only deferral (config profile); this Mac is unmanaged |
| https://lifetips.alibaba.com/.../change-a-macs-software-update-frequency-with-a-terminal | community | Big Sur+ deprecated `softwareupdate --ignore` |
| ~11 further hits (oneuptime 2026 game-day guide, youngju.dev 2026 chaos guides x2, medium, harness.io, FileWave, Intune, Swif, addigy x2, howtoisolve, oreilly) | mixed | Redundant with fetched set; retained as URL-collection evidence |

Search-query variants run: year-less canonical ("FileVault automatic login disabled macOS Apple support"),
last-2-year ("game day disaster recovery drill design chaos engineering safety abort criteria 2025"),
current-year ("macOS defer software updates softwareupdate schedule pmset autorestart 2026").

## Recency scan (2024-2026)

Performed (2025/2026-scoped queries above). Findings: (1) macOS Tahoe/26 enables FileVault by default
on some upgrades and has a known bug leaving FV + auto-login in conflicting states — directly relevant:
do NOT toggle FV pre-departure. (2) macOS 26 adds FileVault unlock over ssh post-restart (Apple
deployment doc) — a new remote-recovery option if FV were ever on. (3) eclecticlight (Feb 2025): hidden
SUOSUScheduler tonight-install can restart the Mac despite settings. (4) Game-day/chaos canon unchanged
2024-2026 (2026-dated guides restate blast-radius/abort/timebox principles).

## Key findings (external)

1. Auto-login and FileVault are mutually exclusive — "credentials are required during the boot process" with FV on (Apple deployment doc). Moot here: FileVault is OFF on this Mac (machine-verified below); the checklist's "decision" collapses to "enable auto-login, leave FV off, don't touch FV".
2. `pmset autorestart` covers POWER LOSS only — it does not recover from a kernel panic or an update-restart-to-login-screen without auto-login (ss64 man page). Auto-login is the complement, not a substitute.
3. Update deferral on an unmanaged Mac = turning OFF "Download new updates" + "Install macOS updates" toggles; there is no supported defer-N-days knob outside MDM (Apple doc + Jamf). Residual risk: a clicked "tonight" notification arms a hidden DAS-CTS install+restart (eclecticlight) — the operator must not click "tonight"/"now" on any update notification before departure, and disabling download removes the staged payload.
4. Drill design canon: minimize blast radius and contain fallout (principlesofchaos.org); pre-share scenario + expected results, estimate resolution time, timebox, blameless notes (Ad Hoc). The run-sheet below is that pre-shared document.

## Internal code inventory

| File | Lines | Role | Status |
|---|---|---|---|
| handoff/away_ops/dress_rehearsal.md | 1-69 | The checklist under validation (A keystrokes / B tokens / C drills / D sign-off) | Sound skeleton; hardening below |
| backend/services/kill_switch.py | :36 audit path; :61-104 boot replay; :152-182 pause()+P1; :166 `_MANUAL_TRIGGERS`; :184-194 resume(); :196-217 SOD/peak; :230-264 evaluate_breach (pure, no state flip) | Kill-switch core | Peak ratchet is one-way (:212-217) — never inject a fake-high NAV |
| backend/api/paper_trading.py | :480-518 GET /kill-switch; :521-527 POST /pause (trigger hardcoded "manual"); :530-570 POST /resume (re-evaluates breach, 409 if breached); :631-642 /flatten-all | Operator routes | /pause can NEVER exercise the alert path (manual is silenced at kill_switch.py:166-167) |
| backend/services/paper_trader.py | :1014-1062 check_and_enforce_kill_switch | The real breach path (peak ratchet, SOD roll, flatten+pause trigger="limit_breach") | Only fires inside a cycle |
| backend/services/autonomous_loop.py | :1018-1024 | The check the daily cycle consults: `trader.check_and_enforce_kill_switch()` + `_ks_state().is_paused()` -> "kill_switch_halted" | This is the "trading blocked" proof anchor |
| backend/agents/mcp_servers/risk_server.py | :65-92 | MCP kill_switch tool: synthetic `current_nav` -> evaluate_breach, "Does NOT flip state" | The safe breach-math leg |
| backend/services/observability/alerting.py | :100-115 deduper consecutive_threshold=3 (P1 NOT critical-exempt); :147-156 returns False before sending when `slack_webhook_url` empty | P1 dispatch | BLOCKER: see finding I-1 |
| scripts/away_ops/healthcheck.sh | :42-59 ks probe+audit fallback; :85-106 frontend-only restart (kickstart -> 113 -> bootstrap); :108-157 P1 w/ bot-token fallback; :119 HEALTHCHECK_TEST_P1 | Watchdog payload | Bot-token fallback exists HERE only, not in kill_switch.py |
| backend/slack_bot/operator_tokens.py | :40-41 jsonl+cursor paths; :43-46 grammar; :52-55 KNOWN_TOKEN_ENV_MAP **EMPTY** (both entries commented); :92-126 append+dedupe+line ACK; :158-185 advance_cursor (temp+rename mtime opens 62.0 gate) | Token plumbing | `handoff/operator_tokens.jsonl` and `handoff/away_ops/tokens_cursor` DO NOT EXIST yet — never fired live |
| scripts/away_ops/prompt_am.md | :29-34 reading order (calendar = selection authority); :36-45 Step-0 token order, (d) KILL SWITCH: RESUME -> POST resume; :47-57 one-step + 4h cap + WIP checkpoint | AM session contract | Gap: Step-0(d) doesn't explicitly order cursor advance for non-env tokens (finding I-4) |
| handoff/away_ops/approved_plan_2026-06-12.md | :157-166 calendar (Sunday AM = leftovers of "62.0-62.6, 61.1, 62.8"; W1 Mon AM = 61.2); :176-184 the 5 rehearsal verifications | Calendar authority | Sunday-AM pick is ambiguous once pre-dep rows are exhausted (finding I-2) |
| .claude/masterplan.json | 62.7 + 63.1 blocks | Immutable criteria; 63.1 has depends_on_step: null | 63.1 is the ideal bounded Sunday step |
| Machine probes (this Mac, 2026-06-12) | sw_vers; fdesetup; defaults read loginwindow; pmset -g; softwareupdate --schedule; defaults read SoftwareUpdate | Item-4 ground truth | macOS 26.5 (25F71); FileVault OFF; autoLoginUser NOT set; autorestart=1 ALREADY; sleep=1 held only by caffeinate; AutomaticallyInstallMacOSUpdates=1 with 26.5.1 (25F80) QUEUED |
| handoff/archive/phase-62.8/research_brief.md | :114 | Confirms map-empty + grandfather manifest design | Done step; consistent |

## Internal findings (the five questions)

### I-1. Kill-switch drill design (riskiest drill) — recommended recipe

The three candidate mechanisms decompose; use a 4-leg composite. NEVER use synthetic NAV injection
(option b): `update_peak` is ratchet-up-only (kill_switch.py:212-217) — a fake high peak is
unrecoverable through any API; a fake SOD corrupts the daily-loss anchor for the rest of the UTC day.

- **Leg A — breach math, read-only**: risk MCP `kill_switch(current_nav = 0.88 * peak_nav)` ->
  `trailing_dd_breached: true` returned with NO state flip (risk_server.py:65-92, docstring explicit).
  Proves evaluate_breach math against live SOD/peak.
- **Leg B — pause + restart survivability**: one-shot venv
  `get_state().pause(trigger="drill_62_7_simulated_breach", details={"drill": True})` appends the audit
  pause line and exercises the SAME alert dispatch code as a real breach (trigger not in
  `_MANUAL_TRIGGERS`, kill_switch.py:166-181). Then `launchctl kickstart -k gui/$(id -u)/com.pyfinagent.backend`
  — boot replay (kill_switch.py:71-76) loads paused=true into the LIVE process. This deliberately also
  proves rail-5's restart-survivability (a reboot mid-pause must stay paused). Backend restart is safe:
  61.1 made the scheduler forward-only (no double-fire), and Sunday has no cron anyway. Why not POST
  /pause: paper_trading.py:526 hardcodes trigger="manual", which kill_switch.py:166 silences — the API
  route can never test the alert path.
- **Leg C — P1 alert: EXPECT FAILURE with current config (pre-drill decision required).** Two
  independent blockers: (1) `slack_webhook_url` is EMPTY on this machine, so `raise_cron_alert*`
  returns False before sending (alerting.py:147-156; the 62.5 Q/A BLOCK finding, healthcheck.sh:109-113)
  — healthcheck got a bash bot-token fallback, kill_switch.py did NOT; (2) even with a webhook, the
  deduper's consecutive_threshold=3 swallows a one-shot P1 (alerting.py:111-113; P1 is not
  threshold-exempt). A REAL breach while away pages NOBODY — flatten+pause still protects capital
  (rail 5), but operator awareness waits for the next digest (<=9h). The masterplan criterion says
  "P1 alert -> Slack", so as-is this leg FAILS honestly. Options: (a) RECOMMENDED — Saturday
  operator-present live-fix in alerting.py (observability, not a rail-6 trading file): P0/P1 bypass
  the consecutive threshold + bot-token chat.postMessage fallback when webhook empty (mirror the
  proven healthcheck.sh:139-148 path), with failing->passing regression test + Q/A; (b) accept +
  re-line the criterion evidence as "alert attempt logged + paused state lands in health.jsonl/digest"
  — requires acknowledging the criterion is not met verbatim. Do not configure the legacy
  SLACK_WEBHOOK_URL just for this (checklist already recommends SKIP, dress_rehearsal.md:20) — it
  would still leave blocker (2).
- **Leg D — token -> resume -> restore**: operator phones `KILL SWITCH: RESUME` (creates
  handoff/operator_tokens.jsonl — first live line ever; bot thread-ACKs line number). Main-as-session
  runs Step-0: `unapplied_tokens()` -> key "KILL SWITCH" absent from KNOWN_TOKEN_ENV_MAP -> NO env
  change authorized -> POST /api/paper-trading/resume {confirmation:"RESUME"} -> route re-evaluates
  breach against live NAV (paper_trading.py:556-567) -> healthy (we never touched NAV) -> 200 resumed
  -> `advance_cursor()` (creates tokens_cursor — first ever).
- **"Restored to pre-drill", precisely**: GET /api/paper-trading/kill-switch BEFORE vs AFTER must be
  field-identical on {paused:false, pause_reason:null, sod_nav, sod_date, peak_nav} (paused_at and
  auto_resume_alerted_at are null on both sides — cleared by resume, kill_switch.py:188-190).
  Positions/cash are untouched because the drill deliberately has NO flatten leg — flattening the real
  paper book on a closed-market Sunday would distort the live experiment (the engine WORKS; biggest
  risk is regression into it). Flatten coverage = unit tests + the /flatten-all manual route's
  existence. `kill_switch_audit.jsonl` gains EXACTLY two lines (pause + resume) and they REMAIN —
  append-only by design (kill_switch.py:36,109-119); the trigger string `drill_62_7_simulated_breach`
  self-documents so future forensics can't misread it as a real breach. Never hand-edit the jsonl.
- **Abort criteria**: /resume returns 409 -> the system believes a limit is REALLY breached -> stop the
  drill, investigate before any further action (the 409 is the resume-precondition working). Risk MCP
  tool errors -> skip leg A, note it. Cross-evidence: early-fire the watchdog while paused; its
  health.jsonl line shows `kill_switch_paused":"true"` (healthcheck.sh:42-59).

### I-2. AM-session real-step drill

Selection logic: prompt_am.md:29-32 — calendar in approved_plan_2026-06-12.md:157-166 is the ordering
authority, then "next pending step per the calendar" from masterplan. Sunday's AM row ("62.0-62.6,
61.1 close, 62.8") will be exhausted by Sunday; the next row is W1-Mon 61.2 — so an unpinned Sunday
kickstart either fires 61.2 a day early or dithers. 62.7 itself is not pickable (needs the operator;
it IS the rehearsal). Fix: Saturday evening, add one line to handoff/current/active_goal.md (reading-
order item 2, the "calendar pointer"): "2026-06-14 AM session: execute 63.1 (route walk)". 63.1 is
ideal — depends_on_step null, P0, read-only against the running app (Playwright walk of 22 routes),
zero trading-code surface, genuinely useful Sunday work, fits the 4h gtimeout cap (prompt_am.md:55-57).
Let the real 07:30 CEST scheduled fire count as the drill (dress_rehearsal.md:34-35 explicitly allows
this); operator reviews session.log START/END/COST lines + the auto-commit/push at breakfast instead of
watching live. Abort: phone `HALT-DEV` (Step-0a, prompt_am.md:38) — which doubles as a live HALT-DEV
test; the wrapper's sentinel downgrade path was already proven 2026-06-12 (session.log: synthetic
$99 metered figure -> "sentinel FAILED -- downgrading to digest-only").

### I-3. Token-consumed drill — Sunday's tokens do NOT exercise the .env path (confirmed)

Path traced: commands.py token handler (before catch-all :184) -> append_operator_token
(operator_tokens.py:92-126, dedupe + 1-based line ACK) -> next session Step-0 (prompt_am.md:36-45) ->
KNOWN_TOKEN_ENV_MAP validate -> `advance_cursor()` FIRST (temp+rename refreshes mtime, opening the
62.0 PreToolUse .env gate for 6h, operator_tokens.py:158-164) -> edit backend/.env -> kickstart
backend -> live_check cites the jsonl line. KNOWN_TOKEN_ENV_MAP is EMPTY (operator_tokens.py:52-55;
both entries commented out pending 61.5/65.2) — so SDK-CREDIT / MAS-PLIST / WEBHOOK / TEST-TOKEN are
recorded acknowledgments, and AWAY_MODE_ENABLED is an operator keystroke (dress_rehearsal.md:12), NOT
a token. Nothing on Sunday's list touches the semantic-cursor + hook-gate + .env-write chain — and the
first live use would otherwise be 65.2's "EU SCREENER: ON" fired UNATTENDED in W1. That asymmetry
justifies a drill token. Design: register `"AWAY DRILL": "AWAY_DRILL_NOOP_FLAG"` in
KNOWN_TOKEN_ENV_MAP (+1 line; + a row in backend/tests/test_phase_62_2_operator_tokens.py); grep
proves the flag has zero consumers. Operator phones `62.7 AWAY DRILL: ON`; Main-as-session executes
Step-0(c) verbatim; verify: tokens_cursor JSON carries {applied_line, token_sha256, key:"AWAY DRILL"},
backend/.env contains the line, the PreToolUse gate permitted the write (no block in
handoff/audit/pre_tool_use_audit.jsonl), backend restarted healthy. Then the OPERATOR hand-deletes the
.env line (operator keystroke, present) so .env ends byte-identical. Keep the registration permanently
as the documented per-departure drill key. If Main rules this too invasive, the honest gap statement
is: "the .env-write chain is unit-tested (62.2) but its first live execution will be unattended."

### I-4. Stale-token hazard (new finding, from the trace)

prompt_am.md Step-0(d) tells sessions to POST resume on `KILL SWITCH: RESUME` but only spells out
cursor advancement in the (c) env-write branch. If a session resumes WITHOUT advancing the cursor, the
token stays "unapplied" forever — and a LATER session could re-apply the stale RESUME after a FUTURE
real breach, violating rail 5 (only a fresh token may resume). The drill must verify the cursor ends
past the RESUME line; recommend a one-line prompt_am.md clarification ("every consumed token advances
the cursor, env-mapped or not") shipped Saturday with the other prep.

### I-5. pmset / auto-login / FileVault / updates — machine-verified state (run 2026-06-12, read-only)

| Check | Command (read-only) | Result TODAY | Action for checklist |
|---|---|---|---|
| OS | `sw_vers` | macOS 26.5 / 25F71 (Tahoe) | — |
| FileVault | `fdesetup status` | **"FileVault is Off."** | Leave OFF (auto-login available; do NOT toggle FV — Tahoe has a known FV/auto-login stuck-state bug) |
| Auto-login | `defaults read /Library/Preferences/com.apple.loginwindow autoLoginUser` | key does not exist -> NOT enabled | Enable in System Settings > Users & Groups > Automatic login; verify the command then returns the username. (Stores an obfuscated password in /etc/kcpassword — acknowledged tradeoff) |
| Power-loss restart | `pmset -g` | **autorestart already 1** — item machine-PASSES today | Paste output as the PASS line |
| Sleep | `pmset -g` | **`sleep 1` — held awake ONLY by caffeinate processes** | NEW ITEM: `sudo pmset -a sleep 0`; verify `pmset -g` shows sleep 0 (see Risk R-3) |
| Update schedule | `softwareupdate --schedule` | "Automatic checking ... turned on" | `sudo softwareupdate --schedule off` (optional) |
| Auto-install | `defaults read /Library/Preferences/com.apple.SoftwareUpdate` | AutomaticDownload=1, **AutomaticallyInstallMacOSUpdates=1, AutoInstallProductKeys contains MSU_UPDATE_25F80_patch_26.5.1** (queued!) | Turn OFF "Install macOS updates" + "Download new updates when available" (System Settings > General > Software Update > Automatic Updates); verify both keys read 0 and the queued key clears; do this SATURDAY, not Sunday (R-2) |
| ADC / gh | `gcloud auth application-default print-access-token` / `gh auth status` | (healthcheck already probes both, healthcheck.sh:82-83) | Run fresh at sign-off, paste exit status |

"Signed item-by-item" therefore means: each section-A box gets the verbatim command output pasted
under it with a timestamp — machine-verified, not clicked.

### I-6. Frontend bootout drill (62.5 residual) — exact sequence

1. Operator (sessions are hook-blocked + rail 9): `launchctl bootout gui/$(id -u)/com.pyfinagent.frontend`
2. Confirm down: `curl -s -o /dev/null -w '%{http_code}' http://localhost:3000/login` -> 000/!=200.
3. EITHER wait <=30 min (watchdog StartInterval) OR early-fire (operator keystroke):
   `launchctl kickstart gui/$(id -u)/com.pyfinagent.away-watchdog` (no `-k`; just runs the job now).
4. Expected (healthcheck.sh:85-106): detect -> `kickstart -k` fails exit 113 on the booted-out label ->
   `launchctl bootstrap` fallback from ~/Library/LaunchAgents/com.pyfinagent.frontend.plist ->
   sleep 15 -> re-probe -> health.jsonl line with `restart_note":"kickstart-113-then-bootstrap
   before=absent/http... after=running:<pid>/http200"`, restarts_performed=1, restart_failed=false.
5. Verify: `tail -1 handoff/away_ops/health.jsonl` + curl /login -> 200.
Known flake: Next dev cold start can exceed the 15s settle -> after-probe non-200 -> restart_failed=true
false alarm; re-fire the watchdog once (second pass finds it running+200 and clears). Abort: if note
says "kickstart-and-bootstrap-FAILED" -> manual `launchctl bootstrap gui/$(id -u)
~/Library/LaunchAgents/com.pyfinagent.frontend.plist` + read handoff/away_ops/healthcheck_err.log.
The P1-after-2-fails leg needs no re-test (proven via HEALTHCHECK_TEST_P1=1 in 62.5).

## Risks & gotchas

- **R-1 (BLOCKER-class): the kill-switch P1 criterion cannot PASS as-is** — webhook empty + deduper
  threshold-3 double-kill the dispatch (I-1 leg C). Decide fix-vs-accept SATURDAY; a Sunday surprise
  here would burn the rehearsal window.
- **R-2: a 26.5.1 auto-install is QUEUED on this Mac right now** (AutoInstallProductKeys,
  FirstInstallTonight stamps through 2026-06-10). Disable auto-install/download Saturday; never click
  "tonight"/"now" on update notifications pre-departure (eclecticlight: hidden DAS-CTS activity
  installs + restarts regardless of settings once armed).
- **R-3: sleep=1 single point of failure.** The Mac stays awake only because com.pyfinagent.backend's
  plist wraps uvicorn in `caffeinate -i -s` (plus ad-hoc terminal caffeinates that die). If the backend
  job is ever booted-out/crash-looping, the Mac sleeps in ~1 min and EVERYTHING stops — launchd
  StartInterval timers (the watchdog) do not fire during sleep, and womp can't help (nothing sends
  magic packets). `sudo pmset -a sleep 0` is the defense-in-depth fix; autorestart alone only covers
  power loss, and an update- or panic-restart without auto-login = total outage (hence item I-5 row 3).
- **R-4: stale RESUME token** could un-pause a future real breach if the cursor isn't advanced (I-4).
- **R-5: drill ordering.** Run the token-plumbing drill BEFORE the kill-switch drill — leg D depends on
  the jsonl/ACK/cursor path working (game-day principle: escalate difficulty; estimate resolution time).
- **R-6: append-only artifacts.** kill_switch_audit.jsonl and operator_tokens.jsonl keep drill lines
  forever — that is the design; self-documenting trigger/key strings are the mitigation, never edits.
- **R-7: AWAY_MODE_ENABLED must precede the PM kickstart** or the digest lacks away sections; after the
  .env append, restart the bot (`launchctl kickstart -k gui/$(id -u)/com.pyfinagent.slack-bot` — it is
  launchd-supervised now; rail 9 allows it).
- **R-8: timebox.** Ad Hoc: 2-3h max. The run-sheet keeps operator-attention time ~75 min across the day.

## Timed run-sheet — Sunday 2026-06-14 (CEST)

| # | When | Item | Op-min | Depends | Abort criterion |
|---|---|---|---|---|---|
| 0 | Sat eve | Prep ships w/ tests + Q/A: active_goal pin "AM=63.1"; alerting.py P1 fix (or documented accept); AWAY DRILL key registered; prompt_am cursor clarification; update deferral done (R-2) | 10 (review) | — | any prep FAIL -> drill that leg is observe-only |
| 1 | 07:30 (auto) | AM session fires, executes 63.1 headlessly | 0 | 0 | HALT-DEV from phone |
| 2 | 09:00 | Review AM evidence: session.log START/END/COST, push landed, route_walk artifacts | 10 | 1 | session crashed -> recovery prompt path, not abort |
| 3 | 09:15 | Section A keystrokes + machine-verify paste (mv mas-harness plist, pmset sleep 0, auto-login enable, ADC/gh refresh) | 15 | — | per-item explicit waiver note |
| 4 | 09:35 | Section B tokens from phone: TEST TOKEN: PING (expect thread-ACK w/ line no.), SDK CREDIT: STOP-ON-EXHAUSTION, MAS PLIST: MOVED, WEBHOOK: SKIP, 62.7 AWAY DRILL: ON | 5 | bot running | no ACK in 60s -> check slack-bot label, kickstart -k it |
| 5 | 09:45 | Token-consumed drill: Step-0 run, cursor CREATED, .env write + backend kickstart + live_check; operator deletes the noop line after | 10 | 4 | hook blocks write -> capture audit line, stop (gate bug found) |
| 6 | 10:00 | Kill-switch drill legs A->B->C->D (snapshot, MCP math, one-shot pause, backend kickstart, P1 observation, watchdog early-fire for paused-state line, phone RESUME, POST resume, snapshot diff) | 15 | 4,5 | /resume 409 -> STOP + investigate; never hand-edit jsonl |
| 7 | 10:20 | Frontend bootout drill + watchdog early-fire; verify health.jsonl + /login 200 | 10 | — | bootstrap also fails -> manual bootstrap, read healthcheck_err.log |
| 8 | 10:30 | AWAY_MODE_ENABLED=true append + bot restart (R-7); spot-check one watchdog health line from today | 5 | 3 | — |
| 9 | 22:00 (auto) | PM session kickstart fires; 23:00 evening digest with away sections | 5 (review) | 8 | digest missing sections -> check AWAY_MODE + bot restart |
| 10 | 23:15 | Section D sign-off: every box checked/waived, PASS lines timestamped, transcript appended; close 62.7 per its criteria | 10 | all | any unwaived FAIL -> 62.7 stays open, fix Monday pre-departure |

Total operator attention ~85 min, within the game-day timebox. Expected-results doc = this run-sheet
(Ad Hoc: "share the planned scenario and expected results ... to review and approve").

## Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (7)
- [x] 10+ unique URLs total (~26 incl. snippet-only)
- [x] Recency scan (2024-2026) performed + reported
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered kill-switch, routes, loop, MCP, tokens, prompts, watchdog, calendar, masterplan, live machine state
- [x] Contradictions noted (ACM/AWS fetch failures disclosed; criterion-vs-reality conflict R-1 surfaced, not papered over)
- [x] Claims cited per-claim

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 7,
  "snippet_only_sources": 17,
  "urls_collected": 26,
  "recency_scan_performed": true,
  "internal_files_inspected": 14,
  "report_md": "handoff/current/research_brief_62.7.md",
  "gate_passed": true
}
```
