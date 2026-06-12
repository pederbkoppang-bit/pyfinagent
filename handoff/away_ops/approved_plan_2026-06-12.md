# 3-Week Away-Operations Plan — pyfinagent autonomous fix/test/trade

## Context

The operator is away for 3 weeks. The system is mid-development: phase-61 in flight (61.1
awaits tonight's cycle evidence), operator-reported bugs across ALL UI areas, only 8/22
routes have any E2E coverage (zero functional E2E), EU has produced zero trades since
2026-06-01. Directive: schedule agents to fix the system, prove every feature works as
designed (UI + app), test all markets, and demonstrate profitable multi-market trading —
autonomously, with daily Slack contact. The last unattended period produced silent defects
and one unauthorized masterplan install (reverted); this plan is built around those lessons.

## Operator decision record (verbatim, 4 question rounds, 2026-06-12)

| Decision | Answer |
|---|---|
| New metered LLM spend | "$0 - Max plan only" (Claude Code flat fee + existing Gemini pipeline) |
| In-flight $25 window (58.1) | "Continue the $25 window" ($0 applies to NEW spend) |
| Trading posture | "All markets + auto-pause guardrails" (10% trailing DD / 4% daily; existing kill-switch) |
| Change authority | "Bug fixes live, behavior changes dark" + NO new masterplan phases beyond this plan |
| Communication | "Daily Slack digest + reply tokens"; no reply = safe default; existing channel |
| Cadence | "2 scheduled sessions/day"; Mac stays on + plugged in |
| Kill-switch resume | "Only on your Slack token" (RESUME TRADING); auto-resume stays OFF |
| FRED key rotation | "Defer" (log truncation + redaction still happen) |
| Must-complete | ALL FOUR: phase-61 chain · full test matrix · all-markets proof · 44.7/44.8 + 43.0 |
| Screenshot bugs | All four areas → week-1 full live-app audit supersedes screenshot triage |
| Backlog disposition | "Confirm disposition" (defer list below) |

Pre-plan, same session: flags 60.2/60.3/57.1 ON by operator keystroke + backend restarted
(phase-60 code live); 61.1 Q/A CONDITIONAL pending tonight's 18:00 UTC cycle.

## New phases (the ONLY ones permitted; appended to .claude/masterplan.json on approval)

### phase-62 — Away-ops infrastructure (P0, PRE-DEPARTURE, operator present)
Sessions can't bootstrap their own scheduler — all of 62 lands before departure, ending in a
live dress rehearsal. Steps:
- **62.0** Hard-rules file (`docs/runbooks/away-ops-rules.md`) + away goal install + backlog
  disposition status-flips (defer list below, audit notes, criteria untouched)
- **62.1** Slack bot under launchd (`com.pyfinagent.slack-bot.plist`, KeepAlive) + restart on
  current code (kills stale manual PID 26147 running 06-05 code)
- **62.2** Inbound operator-token handler in the Socket-Mode bot → appends
  `{ts,user,channel,raw,step,key,value}` to `handoff/operator_tokens.jsonl`; only the
  operator's Slack user ID in the approvals channel; threaded ACK; unit-tested grammar
  (`61.5 FEE TABLE: ON`, `KILL SWITCH: RESUME`, `HALT-DEV`, ...). Token ingestion MUST be
  bot-side: claude.ai connectors are absent in headless sessions.
- **62.3** Two launchd plists (clone `com.pyfinagent.mas-harness.plist` shape):
  `away-session-am` 07:30 local (~4h cap), `away-session-pm` 22:00 local (~2h cap, after the
  18:00 UTC cycle) + `scripts/away_ops/run_away_session.sh` wrapper + kickoff prompts
  (am/pm/recovery/digest-only variants)
- **62.4** Guardrail sentinel (`scripts/away_ops/sentinel.sh`): $0-metered assert vs pinned
  Gemini baseline, kill-switch state, .env-flag-vs-token reconciliation; failure ⇒ session
  downgrades to digest-only mode
- **62.5** `healthcheck.sh` + 30-min away-watchdog plist: backend/frontend/slack-bot alive
  (launchctl kickstart -k restart authority), cycle freshness <26h, ADC token probe, disk
  >20GB, gh auth; 2 consecutive failed restarts ⇒ P1 Slack alert
- **62.6** Ops hygiene while operator present: 397MB backend.log rotation, DoD-1 cron fixes
  (langchain_huggingface, ablation exit=1), 39.1 autoresearch exit-1
- **62.7** Dress rehearsal with operator watching: kickstart both sessions for real, token
  round-trip from phone, healthcheck auto-restart drill, kill-switch drill; pre-departure
  checklist signed (auto-login/FileVault decision, `pmset autorestart on`, defer macOS
  updates, refresh gcloud ADC + gh auth)
- **62.8** Away-mode digest sections in `formatters.py` (trades by market w/ EU:0 flagged,
  NAV + DD vs caps + kill-switch state, shipped-today commits + steps flipped, open token
  asks with exact reply strings + age, system health, defect-register delta), behind a new
  `away_mode_enabled` ops flag

### phase-63 — Full-app live audit → defect register → fixes (week 1 + rolling)
- **63.1** Playwright walk of ALL 22 routes (incl. `/sovereign/strategy/[id]`): screenshot,
  console errors, failed network calls per route → `walk_summary.json`
- **63.2** BQ cross-check of every displayed number (cockpit NAV, trades, performance,
  learnings) — displayed vs API vs independent BQ aggregate, side-by-side
- **63.3** Verified defect register (`handoff/away_ops/defect_register.md`), P0/P1/P2,
  each row classified {pure-bug | trading-behavior}
- **63.4** Fix queue: one fix per AM slot, failing-then-passing regression test + fresh Q/A
  PASS + live re-capture; order/sizing/risk-path files auto-reclassify to dark+token
- **63.5** Week-2 regression re-walk: every register row FIXED (evidence) or KNOWN-OPEN (cause)

### phase-64 — Test matrix build-out (weekends — markets closed)
- **64.1** Functional-E2E Playwright project (assertions: data renders, zero console.error,
  zero 5xx — NO screenshots, so the Linux-baseline caveat doesn't apply; runs on the Mac)
- **64.2** Specs for all 22 routes via the documented `LIGHTHOUSE_SKIP_AUTH=1 ... --port
  3100` bypass; suite <15 min so the PM session runs it nightly
- **64.3** Backend gap tests: kill-switch state machine, currency/money paths, per-market
  screener, learnings error-vs-empty
- **64.4** Multi-market e2e: fixture-replayed US/KR/EU cycle (no network) asserting
  per-market funnel counts >0 + currency invariants; plus a `requires_live`-marked smoke
- **64.5** CI wiring + PM-session nightly runner

### phase-65 — All-markets trading proof (diagnose wk 1, prove wk 2-3)
- **65.1** EU zero-trades funnel diagnosis with per-gate counters (universe 40 → screener →
  price_quality → calendar → rank → order) over a replayed cycle; prime suspect: US-tuned
  `min_avg_volume=100_000`/`min_price=5.0` (backend/tools/screener.py:93-99) vs DAX-40
  high-priced/lower-share-volume names; counters become permanent structured-log lines
- **65.2** Per-market screener thresholds built DARK flag-OFF + ON-vs-OFF replay; digest
  token ask: reply `65.2 EU SCREENER: ON`
- **65.3** US+KR health baseline (so "healthy" is defined before the proof)
- **65.4** Proof (wall-clock-gated): ≥1 filled trade per market on ≥3 distinct days each
  within ≥5 trading days after the token timestamp; EUR-scale prices; no unexplained
  kill-switch event. If the token never comes: documented blocked-on-token, closes in one
  day post-return.

## Backlog disposition (operator-confirmed)

EXECUTE: 61.1-close → 61.2 → 61.3 → 61.4 → 61.5(dark) · 62→63→64→65 · 44.7 → 44.8 · 43.0 ·
39.1 + DoD-1 fixes (in 62.6) · 35.3 · 58.1 (self-closing) · best-effort tail:
44.3/44.5/44.9/44.10.
DEFER in 62.0 (status flip + audit note ONLY, criteria untouched): 36.2-36.6 (superseded;
long-hold engine ruled untouchable), 37.3.1 + 40.3.1 (event-gated on future model releases),
40.8.2 (low value), 40.7 (behavior change — barred while away), 40.1 (needs operator's
OpenAlex account — return-day ask). Unchanged: 13.0, 53.4, phase-5 (already blocked/deferred).

## Scheduled-session mechanics (key design points)

- Wrapper `scripts/away_ops/run_away_session.sh` evolves the proven
  `scripts/mas_harness/run_cycle.sh` pattern: shared PID-lockfile with stale reap (AM
  overrun ⇒ PM logs SKIP, exits 0); sentinel pre-flight (fail ⇒ digest-only prompt, never
  silent); dirty-tree ⇒ recovery prompt (inspect, complete-or-revert ONLY the crashed
  session's files) instead of run_cycle's abort (an abort policy would deadlock 3 weeks);
  `git pull --rebase` with offline-mode fallback; `gtimeout` 4h/2h; every failure exits 0 so
  the calendar never stalls.
- Invocation: `claude -p --dangerously-skip-permissions --model claude-opus-4-8
  --max-turns 250` (PM 120). **Pinned Opus 4.8, NOT fable**: Fable 5 draws usage credits
  after 2026-06-22, mid-window. Researcher/Q/A subagent pins stay as committed.
- Each session reads in order: away-ops-rules.md (binding) → active_goal.md →
  masterplan.json → new lines in operator_tokens.jsonl (cursor file). **Tokens applied
  first** (flag flip + restart + live_check citing the jsonl line), then exactly ONE
  masterplan step (AM) under the full researcher → contract → generate → Q/A → log → flip
  loop; the existing auto-commit/push + live_check-gate hooks fire identically headless
  (mas-harness precedent). PM session: healthcheck, wall-clock evidence collection (61.1c4,
  65.4, 35.3), nightly E2E subset, pending_tokens.json refresh, digest send.
- Sessions checkpoint WIP (`chore(away-wip):` commit + handoff note) at ~80% budget so
  gtimeout kills are clean.

## Safety rails (docs/runbooks/away-ops-rules.md, enforced in 3 layers)

1. No .env behavior-flag change without a verbatim matching operator_tokens.jsonl line;
   live_check cites the line. No reply = stays dark/paused.
2. Masterplan freeze: no phases/steps beyond 62-65 + existing backlog; criteria immutable.
3. Git: no force-push, no history rewrite, never touch files a session didn't edit, main-only.
4. $0 metered: dev sessions never call Gemini/Anthropic APIs; sentinel asserts daily
   compute-burn ≤ pinned pipeline baseline; breach ⇒ digest-only + P1 ask. ($25 58.1 window
   + existing Gemini pipeline exempt.)
5. Kill-switch stays paused after any breach; only `KILL SWITCH: RESUME` token resumes;
   auto-resume hysteresis stays OFF.
6. Live-fix authority: failing→passing regression test + fresh Q/A PASS + live evidence;
   order/sizing/risk/screener/fee files = trading-behavior by definition ⇒ dark+token.
7. claude.ai connectors (Figma, Slack MCP) never load-bearing headless; Slack I/O =
   bot artifacts only.
8. One step per AM session; evidence-only PM; exit at budget; never chain.
9. launchctl authority limited to backend/frontend/slack-bot kickstart; never touch
   watchdog/session plists.
10. Unsure ⇒ do nothing, write a token ask, move on. Operator can reply `HALT-DEV` anytime.
Enforcement: prompt-level (rules read first) + hook-level (extend pre-tool-use-danger.sh:
block force-push, launchctl bootout, .env flag edits without fresh token cursor) +
sentinel-level (daily flag-vs-token reconciliation).

## Calendar (departure assumed Monday; today Fri 2026-06-12 → pre-dep = Fri-Sun)

| | AM (dev step) | PM (evidence + digest) |
|---|---|---|
| Pre-dep Fri-Sun | 62.0-62.6, 61.1 close (tonight's cycle), 62.8 | 62.7 dress rehearsal Sunday |
| W1 Mon-Fri | 61.2 → 65.1 → 65.2(+token ask) → 61.3 → 63.4 fixes | 63.1 walk → 63.2 cross-checks → 63.3 register → fix slots → 65.3 baseline |
| W1 weekend | 64.1 + 64.2 functional E2E | 64.2 + nightly wiring |
| W2 Mon-Fri | 61.4 → 61.5(dark+token ask) → 44.7 | 63.4 fix slots, EU funnel monitoring, 63.5 re-walk Fri |
| W2 weekend | 64.3 backend gap tests | 64.4 multi-market e2e + 64.5 CI |
| W3 Mon-Fri | 44.8 → 43.0 DoD audit → best-effort 44.x | 65.4 evidence → 35.3 → final 3-week report digest |

## Top risks (full table in architect output)

Mac reboot (62.7 checklist: auto-login/pmset/update-deferral) · Slack bot death (fixed by
62.1 launchd KeepAlive; bot down ⇒ safe default holds) · session crash ⇒ dirty tree
(recovery prompt, not abort) · Claude usage limits (07:30 start clears the observed 01:20
reset; wrapper retries) · ADC expiry (62.7 refresh + daily probe) · operator silence
(everything has a safe default; worst case = EU proof closes one day post-return).

## Verification (before departure — 62.7 dress rehearsal, operator watching)

1. Token round-trip: phone → bot → operator_tokens.jsonl → next session ACKs in digest.
2. Both scheduled sessions kickstarted for real; one executes a genuine masterplan step
   headlessly with auto-commit/push + hooks verified.
3. Kill-switch drill: simulated breach → flatten+pause → P1 alert → `KILL SWITCH: RESUME`
   → resumed. All paper, evidence in handoff.
4. Healthcheck auto-restart drill (stop frontend, watch it come back).
5. Signed pre-departure checklist (power/login/updates/ADC/gh/Slack tokens).

## Critical files

- `scripts/mas_harness/run_cycle.sh` — proven headless wrapper to clone
- `backend/slack_bot/commands.py` (token handler before catch-all at :184) +
  `formatters.py` (:323/:422 digest extension)
- `.claude/masterplan.json` — phases 62-65 appended on approval
- `backend/tools/screener.py:93-99` — EU prime suspect
- `~/Library/LaunchAgents/com.pyfinagent.mas-harness.plist` — plist template
