# Goal Prompt -- goal-away-ops (3-week unattended operation)

Set by operator 2026-06-12 via plan-mode approval (4 question rounds + ExitPlanMode accept).
Full approved plan (verbatim copy): handoff/away_ops/approved_plan_2026-06-12.md
Design basis: architect agent 2026-06-12 + 3 Explore agents (infra, tests/markets, backlog).

## Objective (one sentence)

Run pyfinagent unattended for 3 weeks -- two scheduled Claude sessions/day fixing verified
bugs live and building behavior changes dark, proving every feature works as designed via a
full test matrix, and demonstrating healthy multi-market (US/KR/EU) paper trading -- with
daily Slack digests, operator reply tokens as the only authority for behavior changes, and
safe defaults everywhere.

## Binding operator decisions (verbatim)

$0 new metered LLM spend (Max plan + existing Gemini pipeline; the approved $25 58.1 window
continues) | All markets live, kill-switch auto-pause 10% trailing / 4% daily | "Bug fixes
live, behavior changes dark"; NO new masterplan phases beyond 62-65 | Daily Slack digest +
reply tokens; no reply = safe default | 2 sessions/day; Mac stays on | Kill-switch resume
ONLY on `KILL SWITCH: RESUME` token; auto-resume stays OFF | FRED rotation deferred |
Must-complete: phase-61 chain, full test matrix, all-markets proof, 44.7/44.8 + 43.0 |
Backlog disposition: "Confirm disposition" (36.2-36.6, 37.3.1, 40.3.1, 40.8.2, 40.7, 40.1
to deferred with audit notes).

## CRITICAL constraints (violating any is an automatic FAIL)

The 10 rails in docs/runbooks/away-ops-rules.md (created by 62.0) are binding on every away
session. Headline rails: token gate on every behavior flag; masterplan freeze (62-65 only);
no force-push / never touch files a session didn't edit; $0 metered with sentinel
enforcement; kill-switch stays paused absent the token; order/sizing/risk/screener/fee
files are trading-behavior BY DEFINITION (dark+token); one step per AM session; claude.ai
connectors never load-bearing headless; unsure => do nothing + token ask. Plus the standing
rules: 5-file harness loop per step, immutable criteria, no emojis, Playwright evidence for
UI claims, BQ rows for data claims, phase-58.1 undisturbed.

## Masterplan installation payload (canonical; install byte-for-byte)

```json
[
  {
    "id": "phase-62",
    "name": "Away-ops infrastructure (P0, PRE-DEPARTURE, operator present) -- goal-away-ops. Scheduled sessions cannot bootstrap their own scheduler, so all of phase-62 lands before departure and ends with a live dress rehearsal (62.7). Design: handoff/away_ops/approved_plan_2026-06-12.md. Install gate: operator plan-mode approval 2026-06-12.",
    "status": "pending",
    "depends_on": ["phase-60"],
    "gate": null,
    "steps": [
      {
        "id": "62.0",
        "name": "Hard-rules file + away goal install + backlog disposition -- docs/runbooks/away-ops-rules.md (10 rails), active_goal.md refresh, deferred-status flips for 36.2-36.6/37.3.1/40.3.1/40.8.2/40.7/40.1 with audit notes, pre-tool-use-danger.sh away patterns (force-push, launchctl bootout, .env flag edit without fresh token cursor).",
        "status": "pending",
        "harness_required": true,
        "priority": "P0",
        "depends_on_step": null,
        "audit_basis": "Operator-approved plan 2026-06-12 (handoff/away_ops/approved_plan_2026-06-12.md); backlog disposition reply verbatim 'Confirm disposition (Recommended)'; prior away-week unauthorized-install lesson (reverted ad349f57).",
        "verification": {
          "command": "cd /Users/ford/.openclaw/workspace/pyfinagent && test -f docs/runbooks/away-ops-rules.md && python3 -c \"import json; mp=json.load(open('.claude/masterplan.json')); ids={'36.2','36.3','36.4','36.5','36.6','37.3.1','40.3.1','40.8.2','40.7','40.1'}; steps={s['id']: s for p in mp['phases'] for s in p.get('steps',[])}; bad=[i for i in ids if steps.get(i,{}).get('status')!='deferred']; assert not bad, bad; print('deferred OK')\"",
          "success_criteria": [
            "docs/runbooks/away-ops-rules.md contains the 10 numbered rails from the approved plan verbatim and is referenced by every kickoff prompt",
            "the 10 disposition steps are status=deferred with an audit note citing the operator's verbatim 'Confirm disposition' reply; a git diff in experiment_results.md proves ONLY status/audit fields changed (verification criteria byte-identical)",
            ".claude/hooks/pre-tool-use-danger.sh (or equivalent PreToolUse hook) blocks: git push --force variants, launchctl bootout/unload of pyfinagent labels, and edits adding/changing PAPER_* flag lines in backend/.env when handoff/away_ops/tokens_cursor has no fresh matching token -- each pattern unit-tested by invoking the hook with a synthetic payload",
            "handoff/current/active_goal.md points at goal_away_ops.md with the away calendar"
          ],
          "live_check": "live_check_62.0.md with the hook-block transcript for all three synthetic payloads and the masterplan diff summary"
        },
        "retry_count": 0,
        "max_retries": 3
      },
      {
        "id": "62.1",
        "name": "Slack bot under launchd + restart on current code -- com.pyfinagent.slack-bot.plist (KeepAlive=true, mirrors backend plist PATH/venv shape); kill stale manual PID (running 2026-06-05 code).",
        "status": "pending",
        "harness_required": true,
        "priority": "P0",
        "depends_on_step": "62.0",
        "audit_basis": "Explore audit 2026-06-12: bot is a manual process PID 26147 started 2026-06-05, NOT in launchd, no auto-restart -- the #1 unattended-operation gap; digests + 7 phase-9 data jobs + watchdog all die silently with it.",
        "verification": {
          "command": "launchctl print gui/$(id -u)/com.pyfinagent.slack-bot | grep -E 'state|pid' && ps -o lstart= -p $(launchctl print gui/$(id -u)/com.pyfinagent.slack-bot | awk '/pid =/{print $3}') && git log -1 --format=%ci -- backend/slack_bot/",
          "success_criteria": [
            "com.pyfinagent.slack-bot launchd agent exists with KeepAlive=true, mirroring com.pyfinagent.backend.plist's environment shape; the old manual PID is dead (kill -0 fails)",
            "ps lstart of the launchd-managed bot process is LATER than the newest git commit touching backend/slack_bot/ (verbatim paste of both)",
            "a morning or evening digest observed in Slack from the NEW process (permalink or screenshot path in live_check_62.1.md)"
          ],
          "live_check": "live_check_62.1.md with launchctl print excerpt, lstart-vs-commit paste, and the digest permalink"
        },
        "retry_count": 0,
        "max_retries": 3
      },
      {
        "id": "62.2",
        "name": "Inbound operator-token handler in the Socket-Mode bot -- parses token grammar + reserved words (KILL SWITCH: RESUME, HALT-DEV, RESUME-DEV), operator-user+channel allowlist, threaded ACK, appends {ts,user,channel,raw,step,key,value} to handoff/operator_tokens.jsonl.",
        "status": "pending",
        "harness_required": true,
        "priority": "P0",
        "depends_on_step": "62.1",
        "audit_basis": "Explore audit 2026-06-12: zero inbound token handling exists (commands.py handlers are read-only); claude.ai Slack MCP is absent in headless sessions, so bot-side ingestion is the ONLY viable path for the operator's 'daily digest + reply tokens' decision.",
        "verification": {
          "command": "cd /Users/ford/.openclaw/workspace/pyfinagent && source .venv/bin/activate && python -m pytest backend/tests -k 'operator_token or 62_2' -q && tail -3 handoff/operator_tokens.jsonl",
          "success_criteria": [
            "a message handler registered BEFORE the catch-all @app.message at backend/slack_bot/commands.py:184 parses ^(?:(?P<step>[0-9][0-9.]*)\\s+)?(?P<key>[A-Z][A-Z0-9 _-]+):\\s*(?P<value>.+)$ plus reserved words and appends the structured line to handoff/operator_tokens.jsonl",
            "only the operator's Slack user ID in the configured channel is accepted; unit tests assert other users/bots/channels are ignored and malformed lines are NOT written",
            "live round-trip: operator sent a real test token (e.g. 'TEST TOKEN: PING') and the jsonl line + the bot's threaded ACK are pasted verbatim in live_check_62.2.md"
          ],
          "live_check": "live_check_62.2.md with the verbatim jsonl line and ACK permalink from the live round-trip"
        },
        "retry_count": 0,
        "max_retries": 3
      },
      {
        "id": "62.3",
        "name": "Scheduled-session plists + wrapper + kickoff prompts -- com.pyfinagent.away-session-{am,pm}.plist (07:30 / 22:00 local, tz confirmed vs the 18:00 UTC cycle), scripts/away_ops/run_away_session.sh (lockfile w/ stale reap, sentinel pre-flight -> digest-only downgrade, dirty-tree -> recovery prompt, git pull --rebase w/ offline fallback, gtimeout 4h/2h, claude -p pinned claude-opus-4-8, exit-0 discipline), prompts am/pm/recovery/digest-only.",
        "status": "pending",
        "harness_required": true,
        "priority": "P0",
        "depends_on_step": "62.0",
        "audit_basis": "Architect design 2026-06-12 cloning scripts/mas_harness/run_cycle.sh (proven headless claude -p pattern); Opus 4.8 pin because Fable 5 draws usage credits after 2026-06-22 (mid-window); dirty-tree recovery instead of run_cycle's abort because an abort policy deadlocks 3 unattended weeks.",
        "verification": {
          "command": "plutil -lint ~/Library/LaunchAgents/com.pyfinagent.away-session-am.plist ~/Library/LaunchAgents/com.pyfinagent.away-session-pm.plist && bash -n scripts/away_ops/run_away_session.sh && grep -c 'END session' handoff/away_ops/session.log",
          "success_criteria": [
            "both plists lint clean, use StartCalendarInterval (AM 07:30, PM 22:00 local; tz cross-checked against date and the 18:00 UTC cycle in the contract), invoke the wrapper with am|pm, and carry the same EnvironmentVariables block as com.pyfinagent.mas-harness.plist",
            "wrapper implements ALL of: shared lockfile handoff/.away-session.lock with stale-PID reap; sentinel pre-flight failure -> prompt_digest_only.md (never abort silently); dirty-tree -> prompt_recovery.md branch; git pull --rebase with offline-mode fallback; gtimeout 14400 (am) / 7200 (pm); claude -p --dangerously-skip-permissions --model claude-opus-4-8 with --max-turns 250/120; every failure path logs to handoff/away_ops/session.log and exits 0",
            "a manually-kickstarted dry-run session (no-op prompt) produced START/END lines in session.log, and a second concurrent kickstart logged SKIP (lockfile proof)"
          ],
          "live_check": "live_check_62.3.md with plutil output, the dry-run session.log lines, and the concurrent-SKIP proof"
        },
        "retry_count": 0,
        "max_retries": 3
      },
      {
        "id": "62.4",
        "name": "Guardrail/budget sentinel -- scripts/away_ops/sentinel.sh: $0-metered assert vs pinned existing-Gemini-pipeline baseline, kill-switch state read, backend/.env behavior-flag vs operator_tokens.jsonl reconciliation; failure downgrades the session to digest-only mode.",
        "status": "pending",
        "harness_required": true,
        "priority": "P0",
        "depends_on_step": "62.0",
        "audit_basis": "Operator $0-metered decision; away-week lesson that drift must be caught by deterministic checks, not model judgment; compute-burn surface pinned in the script header at build time.",
        "verification": {
          "command": "cd /Users/ford/.openclaw/workspace/pyfinagent && source .venv/bin/activate && bash scripts/away_ops/sentinel.sh; echo exit=$?",
          "success_criteria": [
            "sentinel prints {metered_llm_usd_today, baseline_usd, kill_switch_paused, flags_match_tokens, ok} JSON and exits 0 healthy; the metered figure source (BQ table/endpoint) is pinned in the script header",
            "tamper tests: a synthetic cost row above baseline AND a behavior flag line with no matching token each make sentinel exit non-zero with a named gate failure (test transcript in experiment_results.md)",
            "run_away_session.sh pre-flight wires sentinel failure to the digest-only prompt (wrapper test asserts the prompt path switch)"
          ],
          "live_check": "live_check_62.4.md with the healthy JSON and both tamper-test transcripts"
        },
        "retry_count": 0,
        "max_retries": 3
      },
      {
        "id": "62.5",
        "name": "Daily self-check + restart authority -- scripts/away_ops/healthcheck.sh (backend/frontend/slack-bot launchctl state, /api/health, :3000 probe, kill-switch state, cycle freshness <26h, ADC token probe, disk >20GB, gh auth) + com.pyfinagent.away-watchdog.plist (30-min StartInterval); auto-restart via launchctl kickstart -k; 2 consecutive failed restarts -> P1 Slack alert.",
        "status": "pending",
        "harness_required": true,
        "priority": "P0",
        "depends_on_step": "62.1",
        "audit_basis": "3-week unattended risk table (Mac reboot, service death, ADC expiry, disk fill from 397MB log + Playwright captures); restart authority limited to the three pyfinagent service labels per rail 9.",
        "verification": {
          "command": "cd /Users/ford/.openclaw/workspace/pyfinagent && bash scripts/away_ops/healthcheck.sh && tail -1 handoff/away_ops/health.jsonl",
          "success_criteria": [
            "healthcheck verifies all listed probes and appends a structured JSON line to handoff/away_ops/health.jsonl",
            "drill: a deliberately-stopped frontend (launchctl stop) is auto-recovered via kickstart -k with before/after states logged",
            "com.pyfinagent.away-watchdog.plist runs healthcheck every 30 min independently of the two daily sessions; 2 consecutive failed restarts raise a P1 via the existing raise_cron_alert_sync path"
          ],
          "live_check": "live_check_62.5.md with the drill transcript and a health.jsonl line from the watchdog (not a manual run)"
        },
        "retry_count": 0,
        "max_retries": 3
      },
      {
        "id": "62.6",
        "name": "Ops hygiene batch (operator-present window) -- 397MB backend.log rotation (live file <50MB, history archived), DoD-1 cron fixes (langchain_huggingface install for autoresearch cron; ablation exit=1 root-caused and fixed or documented-disabled), masterplan 39.1 executed under its own criteria.",
        "status": "pending",
        "harness_required": true,
        "priority": "P1",
        "depends_on_step": "62.0",
        "audit_basis": "Open ops asks from cycle_block_summary.md (DoD-1 carried items); disk-fill risk; 39.1 is owner-gated and the owner is present this weekend only.",
        "verification": {
          "command": "cd /Users/ford/.openclaw/workspace/pyfinagent && test $(stat -f%z backend.log) -lt 52428800 && source .venv/bin/activate && python -c \"import langchain_huggingface; print('lh OK')\"",
          "success_criteria": [
            "backend.log live file is under 50MB with rotation in place (newsyslog entry or copytruncate script wired to a schedule); the historical log is archived compressed, not deleted, pending the deferred FRED rotation",
            "the autoresearch nightly cron exits 0 on a dry invocation with langchain_huggingface importable from ITS venv; ablation exit=1 is root-caused with the fix applied or the job documented-disabled with an audit note",
            "masterplan step 39.1 is closed via its own immutable verification (cross-referenced, not duplicated)"
          ],
          "live_check": "live_check_62.6.md with the dry-run cron transcript and log-size evidence"
        },
        "retry_count": 0,
        "max_retries": 3
      },
      {
        "id": "62.7",
        "name": "Pre-departure dress rehearsal (operator watching) -- real kickstart of both sessions (one genuine masterplan step executed headlessly), token round-trip from the operator's phone, healthcheck auto-restart drill, kill-switch drill (simulated breach -> flatten+pause -> P1 alert -> KILL SWITCH: RESUME -> resumed), signed pre-departure checklist (auto-login/FileVault decision, pmset autorestart on, macOS updates deferred, gcloud ADC + gh auth refreshed, Slack tokens valid).",
        "status": "pending",
        "harness_required": true,
        "priority": "P0",
        "depends_on_step": "62.5",
        "audit_basis": "Verification section of the approved plan; every prior unattended failure was discovered AFTER the operator left -- the rehearsal moves discovery to T-1 day.",
        "verification": {
          "command": "test -f handoff/away_ops/dress_rehearsal.md && grep -c PASS handoff/away_ops/dress_rehearsal.md",
          "success_criteria": [
            "one full simulated day executed with the operator watching: AM kickstart attempting a real masterplan step, PM kickstart sending a real digest, phone token consumed by the next session, healthcheck auto-restart drill, kill-switch drill -- all PASS lines timestamped in dress_rehearsal.md",
            "the kill-switch drill ran in paper, evidence in kill_switch_audit.jsonl, and trading state was restored to pre-drill",
            "the pre-departure checklist is signed item-by-item by the operator (auto-login or FileVault implications acknowledged, pmset autorestart on, updates deferred, ADC + gh refreshed)"
          ],
          "live_check": "live_check_62.7.md = dress_rehearsal.md with all drill transcripts and the signed checklist"
        },
        "retry_count": 0,
        "max_retries": 3
      },
      {
        "id": "62.8",
        "name": "Away-mode digest sections -- formatters.py format_away_digest_sections(): trades by market (EU:0 flagged), NAV + DD vs caps + kill-switch state, shipped-today (commits + steps flipped), open token asks with exact reply strings + age, system health from health.jsonl, defect-register delta; evening digest full set, morning compact (asks + health); behind away_mode_enabled ops flag.",
        "status": "pending",
        "harness_required": true,
        "priority": "P0",
        "depends_on_step": "62.1",
        "audit_basis": "Operator 'daily Slack digest + reply tokens' decision; reuses formatters.py block helpers (format_morning_digest :323, format_evening_digest :422); away_mode_enabled is an ops flag, not trading behavior.",
        "verification": {
          "command": "cd /Users/ford/.openclaw/workspace/pyfinagent && source .venv/bin/activate && python -m pytest backend/tests -k 'away_digest or 62_8' -q",
          "success_criteria": [
            "format_away_digest_sections() renders all six sections from fixture data, stays under the 50-block Slack cap, and is appended in the evening digest only when away_mode_enabled is true (OFF path byte-identical)",
            "morning digest gains only the compact asks+health sections; unit tests cover empty-state and populated variants",
            "one LIVE evening digest observed in Slack containing the new sections (permalink in live_check_62.8.md)"
          ],
          "live_check": "live_check_62.8.md with the live digest permalink and a rendered-section screenshot path"
        },
        "retry_count": 0,
        "max_retries": 3
      }
    ]
  },
  {
    "id": "phase-63",
    "name": "Full-app live audit -> verified defect register -> fixes (week 1 + rolling) -- goal-away-ops. Covers the operator's screenshot bugs across ALL areas (reports, positions/currency, dashboard numbers, new pages) by auditing the RUNNING app, not screenshots.",
    "status": "pending",
    "depends_on": ["phase-62"],
    "gate": null,
    "steps": [
      {
        "id": "63.1",
        "name": "Playwright walk of all 22 routes (find frontend/src/app -name page.tsx; incl. one concrete /sovereign/strategy/[id]) -- per route: full-page screenshot, console errors, failed (4xx/5xx) network requests -> handoff/away_ops/route_walk_<date>/walk_summary.json.",
        "status": "pending",
        "harness_required": true,
        "priority": "P0",
        "depends_on_step": null,
        "audit_basis": "Operator screenshot report spans all four problem areas; only 8/22 routes have any E2E baseline; the walk replaces screenshot interpretation with live evidence.",
        "verification": {
          "command": "cd /Users/ford/.openclaw/workspace/pyfinagent && python3 -c \"import json,glob; d=json.load(open(sorted(glob.glob('handoff/away_ops/route_walk_*/walk_summary.json'))[-1])); assert d['routes_visited']>=22, d; print(d['routes_visited'], d.get('console_error_routes'))\"",
          "success_criteria": [
            "every page.tsx route visited via Playwright against the running app (NextAuth wall or the documented LIGHTHOUSE_SKIP_AUTH=1 port-3100 bypass), including one concrete strategy [id]",
            "per-route artifacts: screenshot, console messages, failed-request list; walk_summary.json enumerates routes_visited, console_error_routes, failed_request_routes",
            "the on-disk route list is reconciled against the walk (any delta is itself a defect row)"
          ],
          "live_check": "live_check_63.1.md with walk_summary.json verbatim and the artifact directory listing"
        },
        "retry_count": 0,
        "max_retries": 3
      },
      {
        "id": "63.2",
        "name": "BQ cross-check of displayed numbers -- for every number-bearing page (cockpit /, /paper-trading/*, /performance, /learnings, /sovereign): displayed value vs API response (curl) vs independent BQ aggregate, side-by-side; mismatches beyond rounding become defect rows.",
        "status": "pending",
        "harness_required": true,
        "priority": "P0",
        "depends_on_step": "63.1",
        "audit_basis": "Operator-reported 'dashboard/performance numbers wrong'; the 8-day audit precedent (displayed fees/P&L vs BQ truth) proved this method.",
        "verification": {
          "command": "cd /Users/ford/.openclaw/workspace/pyfinagent && test -f handoff/away_ops/defect_register.md && grep -c '^| DEF-' handoff/away_ops/defect_register.md",
          "success_criteria": [
            "each number-bearing page has a displayed-vs-API-vs-BQ triple recorded with the SQL pasted verbatim",
            "every mismatch beyond rounding is a DEF- row with route, severity, reproduction, displayed-vs-truth values, suspected file, and {pure-bug | trading-behavior} classification",
            "zero metered LLM calls used (BQ + curl only)"
          ],
          "live_check": "live_check_63.2.md with at least three side-by-side triples pasted verbatim"
        },
        "retry_count": 0,
        "max_retries": 3
      },
      {
        "id": "63.3",
        "name": "Verified defect register published -- handoff/away_ops/defect_register.md consolidating 63.1+63.2 findings, P0/P1/P2 triage, digest summary; operator screenshot areas all covered or explicitly cleared.",
        "status": "pending",
        "harness_required": true,
        "priority": "P0",
        "depends_on_step": "63.2",
        "audit_basis": "Single source of truth for the 63.4 fix queue and the 63.5 exit re-walk; register rows are the away window's bug-fix authority scope.",
        "verification": {
          "command": "cd /Users/ford/.openclaw/workspace/pyfinagent && grep -cE '^\\| DEF-[0-9]+ \\|' handoff/away_ops/defect_register.md && grep -c 'SCREENSHOT-AREA' handoff/away_ops/defect_register.md",
          "success_criteria": [
            "every console-error route, failed-request route, and number mismatch from 63.1/63.2 appears as exactly one DEF- row (no silent drops; duplicates merged with cross-references)",
            "all four operator-reported screenshot areas map to register rows or an explicit ALL-CLEAR entry with evidence",
            "the register summary appeared in a Slack digest (sections wired in 62.8)"
          ],
          "live_check": "live_check_63.3.md with the register header, row count, and digest permalink"
        },
        "retry_count": 0,
        "max_retries": 3
      },
      {
        "id": "63.4",
        "name": "Fix queue execution (rolling AM slots) -- one register fix per AM slot under the full harness loop; failing-then-passing regression test + fresh Q/A PASS + live re-capture per fix; any fix touching order/sizing/risk/screener/fee paths is reclassified trading-behavior -> dark+token.",
        "status": "pending",
        "harness_required": true,
        "priority": "P0",
        "depends_on_step": "63.3",
        "audit_basis": "Operator authority decision 'bug fixes live, behavior changes dark'; per-fix evidence pattern from phase-61; this step stays in_progress across the window and closes with 63.5.",
        "verification": {
          "command": "cd /Users/ford/.openclaw/workspace/pyfinagent && ls handoff/away_ops/fixes/ | grep -c 'DEF-'",
          "success_criteria": [
            "every shipped fix has handoff/away_ops/fixes/DEF-<n>.md containing: the failing-then-passing test transcript, the fresh Q/A PASS envelope, and the live re-capture (Playwright or BQ) showing the corrected behavior in the running app",
            "zero fixes shipped live that touch order-placement, sizing, risk, screener-threshold, or fee files -- any such defect carries a dark build + token ask instead (sentinel reconciliation clean all window)",
            "all P0 register rows are addressed (fixed or dark+token-pending) before any P1 row is started"
          ],
          "live_check": "live_check_63.4.md indexing every per-fix evidence file"
        },
        "retry_count": 0,
        "max_retries": 3
      },
      {
        "id": "63.5",
        "name": "Regression re-walk (week-2 end) -- rerun the 63.1 walk + 63.2 cross-checks; every register row resolves to FIXED (evidence link) or KNOWN-OPEN (cause + plan); zero NEW console-error routes vs the 63.1 baseline.",
        "status": "pending",
        "harness_required": true,
        "priority": "P1",
        "depends_on_step": "63.4",
        "audit_basis": "Closes the loop on 'all features of our app work as designed' for the UI surface; prevents fix-regressions from accumulating silently in week 3.",
        "verification": {
          "command": "cd /Users/ford/.openclaw/workspace/pyfinagent && python3 -c \"import json,glob; runs=sorted(glob.glob('handoff/away_ops/route_walk_*/walk_summary.json')); a,b=json.load(open(runs[0])),json.load(open(runs[-1])); new=set(b['console_error_routes'])-set(a['console_error_routes']); assert not new, new; print('no new error routes')\" && grep -vc 'FIXED\\|KNOWN-OPEN' /dev/null; grep -cE 'FIXED|KNOWN-OPEN' handoff/away_ops/defect_register.md",
          "success_criteria": [
            "a second full walk exists with zero NEW console-error routes vs the 63.1 baseline",
            "every DEF- row carries a final status: FIXED with evidence link, or KNOWN-OPEN with root cause and a concrete post-return plan",
            "the re-walk summary appeared in the weekly digest"
          ],
          "live_check": "live_check_63.5.md with both walk summaries diffed and the final register status counts"
        },
        "retry_count": 0,
        "max_retries": 3
      }
    ]
  },
  {
    "id": "phase-64",
    "name": "Test matrix build-out (weekends, markets closed) -- goal-away-ops. Functional UI E2E for all routes + backend gap tests + multi-market e2e; 'all features work as designed', proven by tests that keep running nightly.",
    "status": "pending",
    "depends_on": ["phase-62"],
    "gate": null,
    "steps": [
      {
        "id": "64.1",
        "name": "Functional-E2E Playwright project -- new testDir tests/e2e-functional as a second project in frontend/playwright.config.ts; assertion style: primary data region renders, zero console.error, zero 5xx -- NO screenshot comparisons, so the Linux-baseline caveat does not apply and the suite runs on the Mac.",
        "status": "pending",
        "harness_required": true,
        "priority": "P0",
        "depends_on_step": null,
        "audit_basis": "Explore audit: zero functional E2E exists (visual-regression only, 8/22 routes, Linux-only baselines); playwright.config.ts:5-14 documents the macOS baseline failure mode this project shape avoids.",
        "verification": {
          "command": "cd /Users/ford/.openclaw/workspace/pyfinagent/frontend && LIGHTHOUSE_SKIP_AUTH=1 npx playwright test --project=functional --reporter=line --grep smoke",
          "success_criteria": [
            "the functional project exists in playwright.config.ts with testDir tests/e2e-functional and no screenshot assertions",
            "a smoke spec passes on the Mac against the port-3100 auth-bypass server",
            "NEXT_PUBLIC_E2E_TESTING polling suppression is honored per the existing config note"
          ],
          "live_check": "live_check_64.1.md with the green smoke-run transcript"
        },
        "retry_count": 0,
        "max_retries": 3
      },
      {
        "id": "64.2",
        "name": "Functional specs for all 22 routes -- load + one key interaction per route family; suite completes <15 min so the PM session runs it nightly.",
        "status": "pending",
        "harness_required": true,
        "priority": "P0",
        "depends_on_step": "64.1",
        "audit_basis": "The 'all features work as designed' must-complete; route inventory from find frontend/src/app -name page.tsx (22 routes incl. nested paper-trading/* and sovereign/strategy/[id]).",
        "verification": {
          "command": "cd /Users/ford/.openclaw/workspace/pyfinagent/frontend && LIGHTHOUSE_SKIP_AUTH=1 npx playwright test --project=functional --reporter=line",
          "success_criteria": [
            "one spec file per route family, >=22 routes covered, all green on the Mac",
            "each spec asserts: primary data region renders (testid), zero console.error, zero 5xx network responses",
            "full run completes in under 15 minutes (timed transcript)"
          ],
          "live_check": "live_check_64.2.md with the timed full-run transcript"
        },
        "retry_count": 0,
        "max_retries": 3
      },
      {
        "id": "64.3",
        "name": "Backend gap tests -- kill-switch state machine (pause/resume/breach/auto-resume-off), currency/money paths (KR/EU add-on averaging, fx fallback, 61.3 follow-through), per-market screener units, learnings reader error-vs-empty.",
        "status": "pending",
        "harness_required": true,
        "priority": "P1",
        "depends_on_step": null,
        "audit_basis": "Explore audit gap list (kill_switch, currency paths, screener, learnings had no dedicated tests); these are exactly the subsystems the away window depends on.",
        "verification": {
          "command": "cd /Users/ford/.openclaw/workspace/pyfinagent && source .venv/bin/activate && python -m pytest backend/tests -k '64_3 or kill_switch_machine or currency_path or screener_market or learnings_reader' -q",
          "success_criteria": [
            "new test files cover all four gap areas and pass; the requires_live quarantine list does not grow",
            "the kill-switch tests assert the stays-paused policy (auto-resume OFF) that rail 5 depends on",
            "currency tests assert KR avg_entry stays KRW-scale on add-on buys and EU rows stay EUR-scale (mirrors 61.3 criteria)"
          ],
          "live_check": "live_check_64.3.md with the green run transcript and per-area test counts"
        },
        "retry_count": 0,
        "max_retries": 3
      },
      {
        "id": "64.4",
        "name": "Multi-market e2e -- fixture-replayed US/KR/EU cycle (recorded yfinance fixtures, no network) asserting per-market funnel counts >0 and currency invariants; plus one requires_live-marked smoke excluded from default runs.",
        "status": "pending",
        "harness_required": true,
        "priority": "P1",
        "depends_on_step": "65.1",
        "audit_basis": "No end-to-end multi-market test exists (Explore audit); EU funnel counters from 65.1 define the assertions; fixtures keep it $0 and deterministic.",
        "verification": {
          "command": "cd /Users/ford/.openclaw/workspace/pyfinagent && source .venv/bin/activate && python -m pytest backend/tests -k 'multi_market_e2e' -q -m 'not requires_live'",
          "success_criteria": [
            "a fixture-replayed cycle produces screening->ranking->order-intent output for ALL THREE markets with per-market funnel counts >0 (EU under the 65.2 thresholds via test flag)",
            "currency invariants asserted in the same test (KR KRW-scale, EU EUR-scale)",
            "the requires_live variant exists and is excluded from default/CI runs"
          ],
          "live_check": "live_check_64.4.md with the green run and the per-market funnel count output"
        },
        "retry_count": 0,
        "max_retries": 3
      },
      {
        "id": "64.5",
        "name": "CI wiring + nightly runner -- credential-free functional subset added to .github/workflows/e2e-smoke.yml; PM session runs the full local suite nightly with results in the digest.",
        "status": "pending",
        "harness_required": true,
        "priority": "P2",
        "depends_on_step": "64.2",
        "audit_basis": "Keeps the matrix alive after the away window; e2e-smoke.yml already runs the credential-free backend subset nightly.",
        "verification": {
          "command": "cd /Users/ford/.openclaw/workspace/pyfinagent && grep -c 'e2e-functional' .github/workflows/e2e-smoke.yml && grep -c 'playwright test --project=functional' scripts/away_ops/prompt_pm.md",
          "success_criteria": [
            "e2e-smoke.yml gains the functional subset job (credential-free; no secrets)",
            "the PM kickoff prompt includes the nightly local run and the digest carries pass/fail counts",
            "at least one nightly run evidenced in handoff/away_ops/ with its digest permalink"
          ],
          "live_check": "live_check_64.5.md with the CI diff and one nightly-run digest permalink"
        },
        "retry_count": 0,
        "max_retries": 3
      }
    ]
  },
  {
    "id": "phase-65",
    "name": "All-markets trading proof (diagnose week 1, prove weeks 2-3) -- goal-away-ops. EU zero-trades diagnosis -> dark per-market screener fix + token -> >=5-trading-day three-market proof with currency-correct fills and no unexplained kill-switch events.",
    "status": "pending",
    "depends_on": ["phase-62"],
    "gate": null,
    "steps": [
      {
        "id": "65.1",
        "name": "EU zero-trades funnel diagnosis -- per-gate counters (universe 40 -> screener -> price_quality -> calendar -> rank -> order) over a replayed cycle; confirm/refute ranked causes: US-tuned min_avg_volume=100000/min_price=5.0 (backend/tools/screener.py:93-99) vs DAX-40 (universe_lists.py:19-26), price_quality R3, calendar gate; counters become permanent structured-log lines (no behavior change).",
        "status": "pending",
        "harness_required": true,
        "priority": "P0",
        "depends_on_step": null,
        "audit_basis": "EU live since 2026-06-01 with ZERO trades (BQ-verified in the 8-day audit); Explore ranked causes with screener thresholds at 60% confidence; the proof step 65.4 is wall-clock-gated so diagnosis must land week 1.",
        "verification": {
          "command": "cd /Users/ford/.openclaw/workspace/pyfinagent && source .venv/bin/activate && python -m pytest backend/tests -k 'eu_funnel or 65_1' -q && python3 -c \"import json; d=json.load(open('handoff/away_ops/eu_funnel.json')); print(json.dumps(d['EU'], indent=1))\"",
          "success_criteria": [
            "eu_funnel.json records, per market, the count surviving each named gate for one replayed cycle; the EU collapse point is identified with the exact threshold values that excluded each ticker",
            "ranked-cause verdict recorded with per-ticker evidence for >=5 DAX names",
            "funnel counters ship as permanent structured-log lines with NO behavior change (flag-free, log-only diff)"
          ],
          "live_check": "live_check_65.1.md with eu_funnel.json verbatim and the per-ticker exclusion table"
        },
        "retry_count": 0,
        "max_retries": 3
      },
      {
        "id": "65.2",
        "name": "Per-market screener thresholds built DARK -- config flag default OFF (e.g. notional-based volume floor or per-market overrides); OFF path byte-identical; ON-vs-OFF replay shows EU candidates surviving; token ask published: reply '65.2 EU SCREENER: ON'.",
        "status": "pending",
        "harness_required": true,
        "priority": "P0",
        "depends_on_step": "65.1",
        "audit_basis": "Screener thresholds are trading-behavior by definition (rail 6) -> dark+token even though they only ENABLE designed behavior; replay evidence pattern from phase-60.2.",
        "verification": {
          "command": "cd /Users/ford/.openclaw/workspace/pyfinagent && source .venv/bin/activate && python -m pytest backend/tests -k 'screener_per_market or 65_2' -q && grep -c '65.2 EU SCREENER: ON' handoff/away_ops/pending_tokens.json",
          "success_criteria": [
            "the per-market threshold mechanism exists behind a settings flag default False with the OFF path byte-identical (test asserts identical screener output flag-OFF vs pre-change)",
            "ON-vs-OFF replay over recorded cycles shows >=3 EU candidates surviving the screener flag-ON with the exact threshold rationale documented",
            "the token ask with the verbatim reply string is in pending_tokens.json and appeared in a digest; the flag is NOT flipped without the token"
          ],
          "live_check": "live_check_65.2.md with the replay table and the digest permalink carrying the ask"
        },
        "retry_count": 0,
        "max_retries": 3
      },
      {
        "id": "65.3",
        "name": "US+KR health baseline -- BQ per-market trade counts, win rate, exit-reason mix, holding-day distribution since 2026-06-01, written to handoff/away_ops/market_health_baseline.md with explicit 'healthy' thresholds for the 65.4 proof.",
        "status": "pending",
        "harness_required": true,
        "priority": "P1",
        "depends_on_step": null,
        "audit_basis": "'Healthy trades in all markets' needs a definition BEFORE the proof window or 65.4 becomes unfalsifiable; the 8-day audit provides the method (BQ aggregates, verbatim rows).",
        "verification": {
          "command": "cd /Users/ford/.openclaw/workspace/pyfinagent && test -f handoff/away_ops/market_health_baseline.md && grep -c 'HEALTHY-THRESHOLD' handoff/away_ops/market_health_baseline.md",
          "success_criteria": [
            "per-market aggregates (trades, win rate, exit reasons, holding days) since 2026-06-01 with the SQL pasted verbatim",
            "explicit HEALTHY-THRESHOLD lines that 65.4 will be judged against (e.g. no market >X% of NAV in fees, stop-out rate <Y%)",
            "post-churn-fix (61.1 flags ON) trend noted separately from the pre-fix baseline"
          ],
          "live_check": "live_check_65.3.md = the baseline doc itself"
        },
        "retry_count": 0,
        "max_retries": 3
      },
      {
        "id": "65.4",
        "name": "Three-market proof (wall-clock-gated, week 3) -- >=1 filled paper trade per market (US, KR, EU) on >=3 distinct trading days each within >=5 trading days after the 65.2 token timestamp; EUR-scale entry prices; no unexplained kill-switch event; judged against the 65.3 thresholds. If the token never arrives: documented blocked-on-token with diagnosis + replay ready, closing one day post-return.",
        "status": "pending",
        "harness_required": true,
        "priority": "P0",
        "depends_on_step": "65.2",
        "audit_basis": "The operator's 'see trades from all markets with the best stocks possible' directive, made falsifiable; wall-clock gating documented so an operator-silent window degrades safely instead of failing.",
        "verification": {
          "command": "cd /Users/ford/.openclaw/workspace/pyfinagent && test -f handoff/current/live_check_65.4.md && grep -c 'market=' handoff/current/live_check_65.4.md",
          "success_criteria": [
            "verbatim BQ rows showing >=1 filled paper trade per market on >=3 distinct trading days each within the >=5-trading-day window after the 65.2 token timestamp in operator_tokens.jsonl (or the documented blocked-on-token fallback)",
            "EU trades show EUR-scale entry prices and pass the 61.3 currency assertions",
            "no kill_switch_audit.jsonl pause event in the window, or any event is documented with trading left paused per rail 5; results judged against the 65.3 HEALTHY-THRESHOLD lines"
          ],
          "live_check": "live_check_65.4.md with the per-market BQ rows and the threshold judgment table"
        },
        "retry_count": 0,
        "max_retries": 3
      }
    ]
  }
]
```

## active_goal.md refresh payload

Installed by 62.0; see that step. The calendar table lives in
handoff/away_ops/approved_plan_2026-06-12.md and is the AM-session ordering authority.

## References (read before PLAN of any step)

- handoff/away_ops/approved_plan_2026-06-12.md (the operator-approved plan, incl. calendar
  + risk table)
- docs/runbooks/away-ops-rules.md (after 62.0 creates it -- binding rails)
- scripts/mas_harness/run_cycle.sh (headless wrapper precedent)
- handoff/current/goal_phase61_churn_integrity.md (in-flight sibling goal)
- Architect + Explore agent outputs: session transcript dir, 2026-06-12
