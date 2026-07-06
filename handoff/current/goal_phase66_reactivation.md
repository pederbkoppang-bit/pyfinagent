# goal-phase66-reactivation -- Post-vacation analysis + goal prompt (INSTALLED)

Status: INSTALLED 2026-07-06 -- operator approval recorded in-session ("do it", return-day
session, operator present). Masterplan phase-66 (6 steps, all pending) added same session.
Author: return-day session 2026-07-06 (operator present, /login completed).
Evidence: all numbers below were measured live this session -- BQ (`financial_reports.paper_trades`,
`paper_portfolio`, `pyfinagent_data.llm_call_log` via ADC Python client), kill-switch MCP,
`handoff/cycle_history.jsonl`, `handoff/away_ops/session_*.json` + `session.log`, `git log`,
`launchctl list`. No estimates unless labeled.

---

## PART 1 -- What happened while you were away (2026-06-13 .. 2026-07-06)

### 1.1 Money (the north star)

| Metric | Value | Source |
|---|---|---|
| NAV today | $23,997.71 | kill-switch MCP + paper_portfolio (updated 2026-07-06 18:05 UTC) |
| Peak NAV | $24,124.77 | kill-switch MCP |
| P&L since inception | +19.99% vs benchmark +5.23% (alpha ~ +14.8pp) | paper_portfolio |
| P&L over the away month | ~flat (NAV was "~24,021 region" on 06-18 per the recovery record; $23,998 today) | pending_tokens + live |
| **Open positions today** | **0 -- portfolio is 100% cash** (current_cash == total_nav) | paper_portfolio |
| Last BUY | 2026-06-10 (SNDK, swap_buy) | paper_trades |
| Trades after 06-10 | exactly 2, both `stop_loss_trigger` SELLs: SNDK 06-23, 000660.KS 07-03 | paper_trades |
| Trading cycles | ran every trading day; 23 completed + 3 timeouts (06-16/18/19); **every cycle since 06-11 executed n_trades=0** | cycle_history.jsonl |
| Kill switch | never paused the whole window | kill-switch MCP + audit |

Interpretation: the engine **failed safe, not profitable**. Risk rails all worked --
stops executed mechanically, kill-switch never fired, NAV was preserved. But the BUY
side died on 2026-06-10, existing positions bled out through their stops, and the book
has been sitting in cash earning nothing since 07-03. The +20% / +14.8pp alpha is the
pre-vacation result, frozen. A money engine that is 100% cash has zero expected alpha:
**reactivation is the single highest-value action available.**

### 1.2 Root cause chain (all confirmed live, not inferred)

1. **cc_rail (the Claude Code CLI decision rail, phase-60 lite/deep path,
   `backend/agents/claude_code_client.py`) never had a healthy day in production.**
   It went live with the 06-12 restarts: 06-11 2ok/2fail, 06-12 36ok/45fail, and from
   **06-15 onward 100% failure** -- 2,400+ consecutive failed calls (ECONNRESET 06-15..19,
   then `401 Invalid authentication credentials` from 06-20). Sample rows: ~5s latency,
   0 input/0 output tokens, ok=false. No circuit breaker: it retried ~162 calls per cycle,
   every cycle, for three weeks.
2. **The same expired OAuth credential killed all away dev sessions.** Timeline from
   session_*.json: 06-13/14/15 AM healthy (Cycles 64-66: 62.1 restart, 62.2 verify,
   61.1 PASS+flip). ECONNRESET 06-15..19 (2 recovery sessions 06-18/19 AM got through).
   **From 06-20: 34 consecutive sessions died in ~10s with 401** through 07-06. Last real
   commit: 2026-06-19 (`7be476b3`). Phases 63/64/65 (live audit, test matrix, all-markets
   proof): **0% executed.**
3. **Nothing paged.** healthcheck/away-watchdog monitor app services (backend/frontend/bot
   -- all of which stayed healthy and are still running now), not Claude-credential
   validity. The component whose job was to raise P1s (the away session) was the dead
   component. 17 days of identical failures, zero alerts.
4. **The METERED-BREACH P1 (06-17 $16.51 / 06-18 $42.00) is root-caused and is phantom
   accounting, not real spend.** The sentinel's `metered_llm_usd_today` sums
   `llm_call_log.session_cost_usd` where provider=anthropic. The breach days are 100%
   agent=`cc_rail` rows -- 137/137 and 207/207 FAILED calls -- and a subset of failed rows
   carry a flat $0.50 `session_cost_usd` with 0 tokens. The phase-60.4 writer in
   claude_code_client.py deliberately logs cost 0 ("flat-fee rail"); the $0.50 rows enter
   via another path (candidate: com.pyfinagent.claude-code-proxy, PID 1269, or a
   log_llm_call default -- to be pinned in 66.3). Since cc_rail is Max flat-fee by design
   and 401/ECONNRESET calls bill nothing, the "$460 metered since 06-12" figure is
   contaminated: real metered spend for the window is ~$8.24 Gemini + ~$0.07 claude-code.
   Two real bugs: (a) failed calls logged with nonzero cost; (b) the sentinel counts the
   flat-fee rail as metered.
5. **Away wrapper stuck in a recovery loop (secondary).** Session artifacts dirtied the
   tree after 06-19 -> every session selected prompt_recovery.md; `git pull --rebase`
   failed on unstaged changes -> OFFLINE MODE; then the 401 killed the session before it
   could recover anything. Benign by design (fail-closed), but it means the first healthy
   session must sweep ~3 weeks of session/audit artifacts.
6. **Gemini deep pipeline ran at trickle volume the entire window (1-11 calls/day,
   ~$0.5/day)** -- roughly the same as the pre-departure days, so possibly by design
   (deep-pipeline gating), but with cc_rail dead this means NO analysis path was
   producing BUY candidates. Whether 4-5 Gemini calls/day is the intended deep-pipeline
   duty cycle needs a one-time verification (AW-4 "dead deep pipeline" was a 59.3 finding
   once already).

### 1.3 What held (keep, don't touch)

- Trailing stops, kill-switch monitoring, sector caps (untested -- no buys), the
  fail-closed sentinel + wrapper degrade ladder, launchd KeepAlive on backend/frontend/bot,
  P1 bot-token paging path (live-proven 06-12; it was never *triggered*, which is the gap
  -- see 66.4). The 62.x pre-departure engineering did its job; the failure was upstream
  of all of it (one credential, no probe on it).

### 1.4 Open operator asks (pending_tokens.json, unchanged since 06-19)

| Ask | Disposition proposed by this analysis |
|---|---|
| METERED-BREACH-RECURRING (P1) | Close via 66.3 -- root-caused above (phantom cost on failed flat-fee calls). No real overspend found. |
| TEST-TOKEN-62.2 | Send `TEST TOKEN: PING` in C0ANTGNNK8D now that you're back -> flips 62.2. |
| FABLE-HEADLESS | KEEP OPUS OVERRIDE for any future away window; Fable pins work in attended sessions (this one). |
| SDK-CREDIT | Moot for now (operator home); re-decide before the next away window. |
| WEBHOOK | Configure on a free evening or SKIP permanently (bot-token path is proven). |
| MAS-PLIST-ZOMBIE | Still on disk -- do the `mv` (1 keystroke) or accept the reboot race. |
| AUTORESEARCH-SPEND | Decide: `AUTORESEARCH SPEND: RESUME` or keep $0 preflight-only. |
| FRED key rotation | Return-day ask from pre-departure -- now due. |
| (from Cycle 58, unfiled) | Alpaca paper account shows short_market_value -$13,842.89 on a long-only system -- fold into 66.2 verification or the defect register. |

---

## PART 2 -- NEW GOAL PROMPT (install as `handoff/current/active_goal.md` payload on approval)

Written per Fable 5 prompting practice: full specification up front in one turn; goals and
constraints rather than step-by-step scripts; explicit boundaries and autonomy grants;
progress claims must be grounded in tool evidence. The harness protocol (researcher gate,
Q/A evaluate, five files, live_check) applies to every step exactly as CLAUDE.md specifies
-- this goal does not restate it, it relies on it.

```markdown
# goal-phase66-reactivation -- Make the engine earn again, and make this failure
# mode impossible to repeat silently

## Why (read this before any step)

pyfinAgent exists to make the most money: net, risk- and cost-adjusted, OOS-robust P&L
(charter 2026-06-01). Today the paper portfolio is 100% cash at $23,997.71 because the
Claude decision rail (cc_rail) has been dead since 2026-06-15 -- first on network errors,
then on an expired OAuth credential -- and no alert existed for either. A portfolio in
cash has zero expected alpha; every day unreactivated is a day of forfeited edge. The
sibling failure: the same credential silently killed 34 consecutive scheduled dev
sessions and nobody was paged. Phase-66 has exactly two outcomes: (1) the engine is
analyzing and trading again through its normal gates, and (2) any future death of a
decision path or credential pages the operator within one cycle instead of three weeks.

## Priorities (strict order; do not start N+1 while N has an unmet P0 criterion)

### 66.0 Recovery re-baseline (P0, first session, mostly mechanical)
Outcome: repo and ops state match reality again. The three-week artifact backlog
(handoff/away_ops/session_*.json, audit JSONLs, this draft) is committed and pushed;
away-session plists' recovery loop is confirmed exited (next scheduled session picks a
normal prompt on a clean tree); pending_tokens.json entries are re-dispositioned per the
Part-1 table with operator replies recorded through the 62.2 handler where a token is the
mechanism. Done when: clean `git status`, push landed, pending_tokens updated, and the
METERED-BREACH P1 carries its root-cause note.

### 66.1 Restore the decision path (P0, the money step)
Outcome: cc_rail executes real analyses again in the live loop. Auth is already fixed
(operator /login 2026-07-06) -- verify, then make the rail self-protecting:
- a cheap pre-cycle health probe (claude_code_health_probe exists) that, on failure,
  SKIPS the rail for that cycle instead of firing ~162 doomed calls;
- a circuit breaker: after N consecutive rail failures, stop retrying and emit ONE P1
  page through the proven bot-token path; never again 2,400 silent failures;
- decide-and-document the degraded-mode policy: when the Claude rail is down, does the
  cycle fall back to the Gemini deep path or hold? (Fail-safe default = hold, which is
  current behavior; a fallback is a behavior change -> config-gated, default OFF,
  operator token to enable.)
Done when: a live trading cycle writes ok=true cc_rail rows to
pyfinagent_data.llm_call_log (BQ evidence in the live_check), and a forced-failure drill
(temporarily broken binary path or equivalent) shows probe-skip + single P1 page.
Immutable criterion: no cycle may ever again log >20 consecutive failed cc_rail calls
without a page.

### 66.2 Redeploy capital through the NORMAL path (P0)
Outcome: the 100%-cash book returns to being a portfolio. Constraint that overrides
urgency: DO NOT force, script, or lower gates to manufacture trades. With 66.1 live, run
the normal cycles and verify decisions flow end-to-end (signals -> risk judge -> execution).
Done when EITHER (a) the first BUY lands in paper_trades with risk_judge_decision
recorded, via the ordinary pipeline, OR (b) after 5 consecutive healthy-rail trading
days with zero BUYs, a Q/A-verified diagnosis distinguishes "gates correctly reject
current candidates" from "pipeline defect" with per-stage evidence (candidate counts at
each gate, from BQ/logs -- not narrative). Fold in the two open integrity checks: the
Alpaca short_market_value -$13,842.89 anomaly, and whether paper_portfolio's single
US row is the intended representation for the EU/KR markets that traded in June.

### 66.3 Cost-truth restoration (P1)
Outcome: the sentinel's metered figure means "dollars actually billed". Pin the writer of
the flat $0.50 session_cost_usd on FAILED cc_rail calls (candidates: claude-code-proxy,
log_llm_call default) and fix it to log 0 on auth/connection failures with 0 tokens;
either exclude the flat-fee (Max) rail from the metered gate or tag rows
metered-vs-flat-fee at write time; keep failure COUNTS as their own first-class signal
(they were the real story). Done when: a hand-audit of one day's billed spend matches
the sentinel figure, and a replayed 06-18 dataset no longer breaches.

### 66.4 Credential-expiry resilience (P1)
Outcome: an expiring Claude credential pages within 24h, ever after. Add a daily auth
probe to healthcheck.sh (a $0/near-$0 headless check that distinguishes 401 from network
error) wired to the existing P1 bot-token path with drill mode; the away wrapper treats
401 as page-once-and-stop-burning-slots, not retry-forever. If a proactive credential
freshness check (token age / expiry surface) is feasible, prefer warning BEFORE expiry.
Done when: a drill with an invalidated credential produces exactly one P1 page and a
clean skip, evidenced in the live_check.

### 66.5 Away-backlog triage (P2, planning-only)
Outcome: phases 63/64/65 (untouched: live-audit defect register, test matrix,
all-markets proof) are re-scoped for attended operation -- keep/merge/drop each step
with one-line rationale, operator sign-off on the result. No build work inside 66.5.

## Boundaries (binding)

- Trailing-stop engine untouchable (phase-61 ruling stands). Hysteresis family remains
  banned absent HYSTERESIS: AUTHORIZE.
- Behavior changes to trading ship config-gated default-OFF with ON-vs-OFF evidence;
  pure bug fixes (66.3 cost logging, probe/breaker plumbing) ship live but test-covered.
- No new metered LLM spend without operator approval; cc_rail is flat-fee, Gemini is
  metered -- know which one you are touching.
- Never fabricate operator evidence (tokens, jsonl lines) or trading evidence; a
  criterion that cannot be met honestly is reported as blocked, not satisfied.
- Progress claims must cite a tool result from the session (BQ row, command output,
  Slack permalink). "Fixed" claims about scheduled jobs require SCHEDULED-run evidence,
  not a manual rerun (the 39.1 lesson).
- Minor implementation choices: decide and note them, don't ask. Scope changes,
  destructive actions, anything touching live trading behavior: ask first.

## Definition of done for the whole goal

The engine holds positions again (or has a Q/A-verified no-trade justification per
66.2b); one full week of cycles with zero silent cc_rail failures; sentinel spend figure
audit-clean; credential drill passed; backlog re-scoped. Each step through the full
harness loop -- researcher gate, contract, generate, fresh Q/A, log-last, live_check.
```

---

## PART 3 -- Return-day checklist for you (5 minutes, in order)

1. Approve/edit this goal -> say the word and the session installs phase-66 into the
   masterplan (the away freeze needs your go-ahead; nothing self-installs).
2. Send `TEST TOKEN: PING` in the approvals channel (closes 62.2's last criterion).
3. `mv ~/Library/LaunchAgents/com.pyfinagent.mas-harness.plist ~/Library/LaunchAgents/disabled.com.pyfinagent.mas-harness.plist.bak` (the reboot-zombie ask, still open).
4. Decide AUTORESEARCH-SPEND and WEBHOOK (both one-word replies).
5. FRED key rotation (pre-departure deferral, now due).
