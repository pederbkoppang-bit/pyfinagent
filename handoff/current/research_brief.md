# Research Brief -- phase-62.3 (goal-away-ops): away-session plists + run_away_session.sh

Tier: moderate-to-complex (caller-set). Date: 2026-06-12. Researcher: Layer-3 (merged Explore).

Step scope: com.pyfinagent.away-session-{am,pm}.plist (07:30 / 22:00 local, StartCalendarInterval,
mas-harness EnvironmentVariables block) + scripts/away_ops/run_away_session.sh (shared lock w/
stale reap, sentinel pre-flight -> digest-only, dirty-tree -> recovery, pull --rebase w/ offline
fallback, gtimeout 14400/7200, claude -p pinned opus-4-8 --max-turns 250/120, exit-0 discipline)
+ prompts am/pm/recovery/digest-only. FO-1 (62.0) binding: prompts read away-ops-rules.md FIRST
and quote the rails inline.

## Read in full (6; counts toward gate)

| URL | Accessed | Kind | Fetched how | Key finding |
|---|---|---|---|---|
| https://www.launchd.info/ | 2026-06-12 | authoritative doc | WebFetch full | "If the system is asleep, the job will be started the next time the computer wakes up. If multiple intervals transpire... coalesced into one event"; bootstrap gui/`id -u` is the modern load verb; EnvironmentVariables: no shell expansion |
| https://code.claude.com/docs/en/headless | 2026-06-12 | official doc | WebFetch full | "Starting June 15, 2026, Agent SDK and `claude -p` usage on subscription plans will draw from a new monthly Agent SDK credit, separate from your interactive usage limits"; default -p "loads the same context an interactive session would" (hooks/CLAUDE.md/MCP) -- never pass --bare; stdin pipe capped 10MB (v2.1.128); background tasks reaped ~5s after final result (v2.1.163); --output-format json exposes total_cost_usd |
| https://support.claude.com/en/articles/15036540-... | 2026-06-12 | official doc | WebFetch full | Agent SDK credit: Pro $20 / Max 5x $100 / Max 20x $200 per month; covers `claude -p`; exhausted => "Agent SDK requests stop until your credit refreshes" unless usage credits enabled (then standard API rates); interactive Claude Code unaffected |
| https://www.gnu.org/software/coreutils/manual/html_node/timeout-invocation.html | 2026-06-12 | official doc | WebFetch full | default signal TERM; -k/--kill-after grace runs FROM the first signal; exits 124 (timeout), 137 (KILL), 125/126/127; --foreground means "any children of command will not be timed out" -- do NOT use it; default mode signals timeout's own process group |
| https://flokoe.github.io/bash-hackers-wiki/howto/mutex/ | 2026-06-12 | authoritative community canon | WebFetch full | check-then-create is "two separate steps... not an atomic operation"; mkdir and `set -o noclobber; echo $$ > lock` are the atomic primitives; PID liveness (kill -0) stale-reap is "unreliable" (PID reuse) but the practical standard; trap-based cleanup |
| https://keith.github.io/xcode-man-pages/launchd.plist.5.html | 2026-06-12 | official man page | WebFetch full | StartCalendarInterval = crontab-like local-time semantics, sleep->coalesced wake fire; StartInterval: "If the job is running during an interval firing, that interval firing will likewise be missed" (per-label no-overlap); ExitTimeOut = SIGTERM->SIGKILL gap; AbandonProcessGroup=false kills the job's pgid on death |

## Snippet-only (27 URLs; context, not gate)

alvinalexander.com launchd examples; github.com/seamusdemora UsingLaunchd...; emorydunn.github.io
StartCalendarInterval; homebrew-autoupdate#59; forums.macrumors.com launchd; institute.sfeir.com
headless x3 (cmd-ref/faq/cheatsheet); angelo-lima.fr; agentpatterns.ai headless-claude-ci;
backgroundclaude.com/headless; wmedia.es; hidekazu-konishi.com claude CI/CD; linuxize.com timeout;
man7.org timeout(1); github coreutils timeout.c; putorius.net; howtogeek.com timeout;
daemon-systems.org timeout(1); adrian.idv.hk bashlock; commandinline.com flock; baeldung.com
single-instance; bash-hackers gabe565 mirror; tobru.ch mkdir-lock; blog.darnell.io launchctl;
developer.apple.com archive ScheduledJobs ("if the machine is off when the job should have run,
the job does not execute until the next designated time"); apple forums #52369; manpagez
launchd.plist; deniapps.com wake-support. Why not full: redundant with the 6 canonical reads.

## Search queries (three-variant discipline)

Year-less canonical: launchd StartCalendarInterval missed/sleep/coalesced; GNU timeout SIGTERM
kill-after 124; bash locking mkdir/noclobber/flock/PID stale. Current-year: claude code headless
-p --max-turns flags 2026. Recency-scoped: macOS launchd behavior change 2025/2026 Sequoia/Tahoe.

## Recency scan (2024-2026)

THREE findings that supersede assumptions baked into the approved plan (written 2026-06-12):
(1) **June 15, 2026 Agent SDK credit** -- `claude -p` leaves the interactive subscription pool
3 days into the away window; Max 5x = $100/mo, Max 20x = $200/mo; exhaustion = headless requests
STOP until refresh. The plan's "Opus 4.8 not Fable" pin addressed the June-22 Fable cliff but
missed this. Two Opus sessions/day x 3 weeks plausibly exceeds $100-200 at API-equivalent burn.
(2) `--bare` (new 2026): would skip hooks/CLAUDE.md/MCP -- must NOT be used; default `-p` loads
them (the away design depends on auto-commit/live-check/danger hooks firing headless).
(3) v2.1.163 background-task reaping + v2.1.128 10MB stdin cap -- both benign for us (prompt
files ~4-40KB). launchd semantics: NO 2024-2026 behavior changes found (Sequoia/Tahoe) -- stable.

## Internal code inventory (file:line)

| Anchor | Finding |
|---|---|
| scripts/mas_harness/run_cycle.sh:14-18 | CLAUDE_BIN=/Users/ford/.local/bin/claude (matches `which claude`); LOCKFILE/LOGFILE under handoff/ |
| run_cycle.sh:23-33 | lock = PID file + `kill -0` stale reap + `trap rm EXIT`; SKIP line + exit 0 on held lock. NOTE: check-then-write race (Bash Hackers); harden with `set -o noclobber` atomic create, keep PID+reap |
| run_cycle.sh:40-45 | dirty-tree ABORT exit 0 (away wrapper replaces with recovery-prompt branch per criteria) |
| run_cycle.sh:46-50 | `git checkout main` + `pull --rebase`; pull fail = exit 1 (away wrapper: offline fallback + exit 0; also `git rebase --abort` on conflict before proceeding) |
| run_cycle.sh:59-71 | `"$CLAUDE_BIN" -p --dangerously-skip-permissions --model claude-opus-4-8 < "$PROMPT"` -- prompt via STDIN (avoids argv limits); failure logs FAIL + tail -20 (but exits nonzero -- away must exit 0) |
| run_cycle.sh:73-77 | logging idiom: `[date -Iseconds] START/END cycle` + tail -5 sentinel + `---` |
| ~/Library/LaunchAgents/com.pyfinagent.mas-harness.plist:7-15 | EnvironmentVariables block to clone VERBATIM: CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1, HOME=/Users/ford, PATH=/Users/ford/.local/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin |
| plist:16-17,25-34 | ExitTimeOut 1500; RunAtLoad false; logs handoff/mas-harness.launchd.log; StartInterval 1800 (away swaps for StartCalendarInterval Hour/Minute); WorkingDirectory=repo |
| launchctl state | `launchctl print gui/501/...mas-harness` = "Could not find service"; print-disabled = "enabled" => plist is BOOTOUT'd, not disabled. RISK: file still in ~/Library/LaunchAgents => next login/reboot AUTO-RELOADS it => 30-min go-live cycles revive and race away sessions (different lockfiles). Sessions CANNOT fix it themselves -- hook :162-163 blocks `launchctl disable` on com.pyfinagent.*. Operator must neutralize pre-departure (move/rename plist) |
| scripts/mas_harness/cycle_prompt.md:8-39,76-99 | prompt template: hard rules inline at top, failure-mode sentinel lines (NOOP/BLOCKED/FAILED), reading list, permission context ("writes to .git/.claude... hang in -p mode -- avoid"). FO-1: away prompts put docs/runbooks/away-ops-rules.md FIRST in the reading list AND quote the 10 rails inline |
| docs/runbooks/away-ops-rules.md:30-32,43-50 | prompt-level enforcement + token mechanics. WORDING TRAP: :45-47 says "apply them FIRST... then advance the cursor" -- literal order DEADLOCKS (the .env write would hit a stale-cursor hook block) |
| backend/slack_bot/operator_tokens.py:21-24,158-185 | OPERATIVE order (module docstring + advance_cursor): validate vs KNOWN_TOKEN_ENV_MAP -> advance_cursor(line_no, rec) (temp+rename refreshes mtime = OPENS the 6h gate) -> THEN write backend/.env -> restart -> live_check cites line. unapplied_tokens() :140-155 returns (line_no, record) past cursor.applied_line. KNOWN_TOKEN_ENV_MAP :52-55 is EMPTY today (61.5/65.2 register later); HALT-DEV/RESUME-DEV reserved bare :46 |
| .claude/hooks/pre-tool-use-danger.sh:178-199,218-251 | .env gate = tokens_cursor mtime < 21600s, Bash + Edit/Write coverage; :152-171 force-push/launchctl guards |
| handoff/away_ops/approved_plan_2026-06-12.md:112-132 | wrapper spec verbatim (AM-overrun=>PM SKIP; sentinel fail=>digest-only never silent; recovery = complete-or-revert ONLY the crashed session's files; --max-turns 250/120; WIP checkpoint `chore(away-wip):` at ~80% budget); :172 "07:30 start clears the observed 01:20 reset" |
| handoff/cycle_history.jsonl (last) | cycle 78d253f5: 2026-06-11T18:00:00Z -> 19:10:40Z (70.7 min) -- confirms ~19:10 UTC typical end |
| handoff/logs/mas-harness{,-v2}.log | 156 + 177 "END cycle" lines = 333 successful headless `claude -p` cycles; Phase 4.4.x commits authored+pushed headlessly => hooks/git work in -p (now also doc-confirmed) |
| Live env probes | claude 2.1.173; `--max-turns` PARSES (hidden from --help; negative control `--definitely-not-a-real-flag` => "unknown option") -- criteria buildable as written. gtimeout ABSENT (no coreutils; brew present at /opt/homebrew/bin) => GENERATE must `brew install coreutils`. caffeinate -i -s wraps uvicorn (backend plist, PID 84682); pmset: "sleep prevented by caffeinate", autorestart=1. zdump Europe/Oslo: DST only Mar 29 / Oct 25 |
| .claude/settings.json hooks | PreToolUse danger + PostToolUse changelog/masterplan-sync/archive-handoff/commit-reminder/auto-commit-and-push -- all load in default -p mode |

## Timezone math (window 2026-06-12 .. ~2026-07-06, CEST=UTC+2 fixed; EDT=UTC-4 fixed)

- No DST boundary in window (Oslo: Oct 25; US: Nov 1). 07:30 CEST = 05:30 UTC; 22:00 CEST = 20:00 UTC.
- AM 07:30-11:30 CEST (4h cap) -- clear of everything; >6h past the observed 01:20 limit reset.
- 18:00 UTC autonomous cycle = 20:00 CEST, typical end 19:10 UTC (21:10 CEST) => PM at 22:00 CEST
  starts ~50 min after; only a 2.8x-duration outlier cycle would overlap (and they are separate
  processes; the paper loop has its own lock).
- Evening digest 17:00 ET = 21:00 UTC = 23:00 CEST = 60 min INTO the PM session; sent by the
  slack-bot process from durable state (commits/health.jsonl/pending_tokens.json -- 62.8 sections),
  not from a synchronous PM handoff. Ordering ACCEPTABLE iff the PM prompt front-loads
  pending_tokens.json refresh + health evidence in its first hour; later PM output rolls to the
  next morning compact digest. Morning digest 08:00 ET = 14:00 CEST (after AM ends 11:30).
- StartCalendarInterval uses local time (crontab(5) semantics); no Weekday key => fires 7d/week,
  matching the plan's weekend calendar rows.

## Failure matrix + GO/NO-GO

| Failure | Detection | Wrapper behavior | Next-session recovery |
|---|---|---|---|
| claude crash / nonzero exit | $? + captured tail | log FAIL + tail -20, exit 0; lock freed by trap | dirty tree likely => pre-flight branches to prompt_recovery.md (complete-or-revert ONLY crashed session's files) |
| Limit-hit (01:20-style session cap; June-15 SDK credit exhaustion) | output matches rate_limit/credit signature; `--output-format json` total_cost_usd loggable | log LIMIT_HIT distinctly, exit 0; calendar keeps firing | AM/PM are 10.5h apart (never share a 5h window); credit exhaustion = EVERY later session no-ops => digest shows shipped-today empty; P1 ask via pending_tokens |
| git pull fails -- offline | pull exit != 0 + no network (curl -m 5 github 4xx/timeout) | log OFFLINE, skip pull, continue (commits queue locally) | next session pushes backlog; auto-push hook already fail-opens |
| git pull fails -- rebase conflict | .git/rebase-merge present | `git rebase --abort`, log CONFLICT, downgrade to digest-only, exit 0 | recovery prompt inspects divergence; never leaves mid-rebase tree |
| sentinel fail OR missing (62.4 ships later) | exit != 0 / file absent | fail-closed: prompt_digest_only.md + log reason (criteria: never abort silently) | sentinel re-checked next fire; digest carries P1 |
| lock stale (SIGKILL/power loss) | PID file + kill -0; harden: noclobber atomic create + `ps -p PID -o command=` name-check (PID-reuse guard) | reap + proceed; concurrent kickstart => SKIP line exit 0 (criterion 3) | none needed |
| Mac asleep at fire | caffeinate -s active while backend lives; if asleep anyway, launchd fires coalesced on wake | session runs late; lock + gtimeout bound it | n/a |
| Mac powered off at fire | Apple doc: job does NOT run retroactively after power-on | fire lost; autorestart=1 + auto-login (62.7) restore services | next scheduled fire resumes; digest gap signals it |
| gtimeout kill at cap | exit 124 (TERM) / 137 (KILL after -k grace) | log BUDGET_KILL, exit 0; prompts checkpoint WIP at ~80% budget so the kill is clean | recovery prompt finishes/reverts WIP commit |
| mas-harness zombie revival post-reboot | plist on disk + "enabled" => auto-bootstraps at login | out of wrapper's hands (hook blocks sessions running `launchctl disable`) | OPERATOR pre-departure: move/rename the plist (62.7 checklist) |

**GO**, conditional on 4 contract legs: (1) `brew install coreutils` in GENERATE (gtimeout absent
today -- verify `which gtimeout` post-install; plist PATH already covers /opt/homebrew/bin);
(2) June-15 Agent SDK credit: quantify plan tier + decide usage-credits enablement with operator
pre-departure; wrapper logs per-session cost (json output) + LIMIT_HIT; (3) operator neutralizes
the mas-harness plist file pre-departure; (4) prompts encode the FO-2 order from operator_tokens.py
(advance_cursor BEFORE .env write), overriding away-ops-rules.md:45-47's looser wording. Plus
FO-1: all four prompts list away-ops-rules.md FIRST + quote rails inline (cycle_prompt.md shape).

## Consensus vs debate

Consensus: launchd sleep=coalesced-wake-fire / power-off=skip (launchd.info, man page, Apple
archive); atomic lock = mkdir/noclobber, PID-reap imperfect but standard (all locking sources);
timeout -k TERM-then-KILL with 124/137 (GNU + man7 + linuxize). Debate: flock superiority is
Linux-centric -- macOS ships no flock(1), so the noclobber+PID hybrid is correct here. SFEIR/
community prefer `--permission-mode dontAsk` over `--dangerously-skip-permissions` for CI;
this repo's criteria + 333-cycle precedent pin the latter -- keep it (immutable criteria).

## Research Gate Checklist

- [x] >=5 authoritative external sources READ IN FULL via WebFetch (6)
- [x] 10+ unique URLs total (33)
- [x] Recency scan (2024-2026) performed + reported (3 findings; June-15 credit is P0)
- [x] Full pages read for the read-in-full set
- [x] file:line anchors for every internal claim
- [x] Internal exploration covered every relevant module
- [x] Contradictions / consensus noted (rules-file vs module ordering; flock vs noclobber)
- [x] All claims cited per-claim

## JSON envelope

```json
{
  "tier": "complex",
  "external_sources_read_in_full": 6,
  "snippet_only_sources": 27,
  "urls_collected": 33,
  "recency_scan_performed": true,
  "internal_files_inspected": 16,
  "report_md": "handoff/current/research_brief.md",
  "gate_passed": true
}
```
