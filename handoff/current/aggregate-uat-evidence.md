# Aggregate UAT Evidence Bundle -- pre-paper-trading-Monday 2026-04-27

**Bundle assembled:** 2026-04-25 (Saturday)
**Paper trading goes live:** Monday 2026-04-27 14:00 ET (post-16.18 TZ fix)
**Bundle author:** Main (this Claude Code session)
**Audience:** Q/A subagent + Peder (final approver)

This is the input for **phase-16.23 (AGGREGATE Go/No-Go verdict)**, which closes the long-running **phase-16.15** ("Go/No-Go verdict, Q/A spawn required, NO self-evaluation"). Q/A renders the Go / No-Go decision based on the evidence below. Main does NOT self-judge.

---

## 1. Phase-16 status snapshot

| Step | Status | Title | This session? |
|------|--------|-------|---------------|
| 16.1 | done | Infrastructure readiness | no |
| **16.2** | **in-progress** | Layer 1 analysis pipeline | no (was Vertex 429 blocked; 16.21 surfaced missing wrapper) |
| **16.3** | **in-progress** | MAS Orchestrator round-trip | no (was Anthropic key blocked; 16.20 surfaced missing function) |
| 16.4 | done | Paper-trading cycle live code path | no |
| 16.5 | done | Self-improving loops health | no |
| 16.6 | done | Kill switch + risk guards drill | no |
| 16.7 | done | HITL champion/challenger gate | no |
| 16.8 | done | Slack bot + scheduled jobs | no (autoresearch exit=127 carry-forward) |
| 16.9 | done | Backtest + quant optimizer short loop | no |
| 16.10 | done | Frontend full-page sweep | no |
| 16.11 | done | Auth + OWASP headers | no |
| 16.12 | done | Observability freshness + perf_tracker | no |
| 16.13 | done | Drills aggregate gate | no |
| 16.14 | done | Harness MAS full cycle dry-run | no |
| **16.15** | **in-progress** | Go/No-Go verdict | (this aggregate cycle) |
| 16.16 | done | Backend correctness re-verification | YES (this session, PASS) |
| 16.17 | done | Frontend correctness re-verification | YES (PASS) |
| 16.18 | done | Live API smoke + OWASP | YES (PASS, after TZ fix) |
| 16.19 | done | Trading mechanics drills | YES (PASS, after 2 drill-script fixes) |
| 16.20 | done | MAS orchestrator round-trip | YES (CONDITIONAL — function missing) |
| 16.21 | done | Layer-1 analysis + outcome/memory | YES (CONDITIONAL — 3 wrappers missing) |
| 16.22 | done | Operational layer | YES (PASS, after 3 small aliases) |
| 16.23 | pending | AGGREGATE go/no-go (this cycle) | YES (in flight) |

**Tally for this session's 7 cycles (16.16-16.22):** 5 PASS + 2 CONDITIONAL.

## 2. Monday-blocker bugs found and FIXED in this session

| Cycle | Blocker | Fix | Lines | Status |
|-------|---------|-----|-------|--------|
| 16.18 | APScheduler `add_job(..., "cron", ...)` had no `timezone=` -- daily cycle would fire 14:00 CEST = 08:00 EDT, **90 min before market open** | Add `timezone=ZoneInfo("America/New_York")` to the cron registration in `paper_trading.py:651-660` + `from zoneinfo import ZoneInfo` import | +2 / -0 | LIVE post-restart; verified `next_run: 2026-04-27T14:00:00-04:00` |
| 16.19 | `alpaca_shadow_drill.py` reused `client_order_id="uat-17.6-{sym}-{i}"` -- Alpaca rejects forever (incl terminal orders), drill 100% fails on re-run | Switched to timestamped `uat-shadow-{run_ts}-{sym}-{i}` | +5 / -2 | drill PASSes 5/5 |
| 16.19 | `kill_switch_test.py` used `importlib.util` to load `signals_server.py` without REPO_ROOT on sys.path -- `from backend.utils import json_io` resolved to gpt-researcher's site-packages `backend` (which has `utils.py` not `utils/`) | `sys.path.insert(0, str(REPO_ROOT))` at top | +8 / 0 | drill PASSes 4/4 |
| 16.22 | Verification command imported `build_app` (alias missing) + 2 routes at wrong prefixes | 3 small aliases (`build_app = create_app`; `/api/observability/freshness`; `/api/cost-budget/status`) | +24 / -1 across 3 files | All PASS |

**Five Alpaca paper orders cleaned up:** drill 1 left 5 weekend orders ACCEPTED that would have filled at Monday market open. Cancelled all manually. Q/A re-ran the drill and cancelled its own 5 too. Paper portfolio is **clean** for Monday open (verified: 0 open uat-shadow* orders).

## 3. Live system health (verified within last 60 minutes)

- **Backend** PID 54732 (post-16.22 bounce). `/api/health` HTTP 200. version=6.5.85. 3 in-app MCP servers all `status: ok` (data, backtest, signals). limits_digest=edf822591bb1...
- **Frontend** PID 7586. All 8 authed routes return 302 (NextAuth login redirect, expected for unauthenticated). Login route 200.
- **APScheduler** active. `next_run: 2026-04-27T14:00:00-04:00` (mid-session, 2h before close — correct trigger).
- **Kill switch**: paused=false. sod_nav = peak_nav = current_nav = $9,499.50. No breaches. limits 4% daily / 10% trailing-DD intact.
- **OWASP headers** all 5 present on /api/health (x-content-type-options, x-frame-options, x-xss-protection: 0, referrer-policy, cache-control: no-store).
- **Alpaca paper account** (PA3VQZZLAKE2): ACTIVE, 0 open orders, no positions, $200k buying_power.
- **Tests**: backend pytest 177/178 (1 env skip vaderSentiment). Frontend vitest 34/34. AST audit 258 backend.py files clean.
- **BQ**: strategy_deployments view PASS, 1 champion row (seed_0000) + 1 approved (UAT-REAL-2026-04). Sovereign endpoints serve 31 red-line series + 2 leaderboard entries + 31 daily compute-cost rows.
- **Cost-budget**: $0.0004 daily / $1.91 monthly under $5/$50 caps.
- **launchd**: 7 jobs registered (mas-harness, claude-code-proxy, ablation, openclaw gateway, autoresearch, backend, frontend). 3 with non-zero last-exits (backend=-15 SIGTERM expected; autoresearch=127 ENOENT carry-forward; openclaw.gateway=1 transient).
- **FRED API integration**: live, 7 series, signal=EASING (verified earlier in this session).

## 4. CONDITIONAL closures and what they mean for Monday

### 16.20 (MAS orchestrator) -- CONDITIONAL
- Verification command imports `run_orchestrated_round` which doesn't exist
- MAS layer-2 is **NOT on Monday critical path**: zero references in `autonomous_loop.py` + `paper_trading.py`. Daily cycle uses Layer 1 (Gemini), not MAS Claude
- Anthropic key in `backend/.env` is `sk-ant-oat-*` (OAuth bearer, hard-401s the Messages API). User reminder logged
- 4 follow-up tickets filed (#20-#23). 16.3 EXPLICITLY remains in-progress per Q/A condition (NOT silently closed by 16.20)

### 16.21 (Layer-1 + outcome/memory) -- CONDITIONAL
- Verification command imports 3 functions that don't exist (`run_analysis_pipeline`, `evaluate_recent`, `retrieve_memories`)
- Underlying classes (`OutcomeTracker`, `FinancialSituationMemory`) DO exist; just missing module-level wrappers
- Daily cycle has **0 references** to the missing wrappers. NOT a Monday blocker
- 3 follow-up tickets filed (#24-#26). 16.2 EXPLICITLY remains in-progress per Q/A condition

### Q/A 16.21 escalation clause (HONORED at 16.22)
> "A third structurally-identical CONDITIONAL must FAIL, otherwise the harness is being used as a logger instead of a corrector."

16.22 was on track for the same pattern. Main took the corrector path with 3 minimal aliases (~24 lines pure delegation). Cycle PASSed. Clause honored.

## 5. Aspirational gaps (NOT shipped, NOT blocking Monday)

- **Champion/challenger gate scheduler** (Explore agent's "highest-leverage learning loop"): code is wire-ready, scheduler doesn't fire. User declined the wiring this session. Phase-10.6 in roadmap.
- **Authenticated-home Lighthouse harness** (closes 10.5.7 CONDITIONAL): follow-up ticket #8.
- **3 broken verification commands** (10.5.0 cd-shadow, 10.5.2 missing audit script, 10.5.7 lighthouse `--url`): follow-up #9. Now joined by 5 more (16.20 missing function, 16.21 x3, 16.18-fixed-but-similar pattern).
- **6 cron jobs missing TZ** (slack_bot/scheduler.py x4 + autoresearch/cron.py + mcp_health_cron.py): follow-up #19. Same bug as 16.18 fix — daily-trade trigger is fixed; digests/redteam/health-check still in CEST.
- **Doc-reconciliation sweep** (#26, #29): pre-flight script to diff verification commands vs live code BEFORE steps open. Long-term answer to the recurring 16.20/16.21/16.22 alias pattern.
- **Adaptive LLM provider fallback** (Q/A 16.20): orchestrator hard-fails on Anthropic 401 with no Gemini fallback. Follow-up #22.

## 6. Monday-readiness summary -- the critical 7

For paper trading to succeed on Monday 2026-04-27:

| # | Need | Status |
|---|------|--------|
| 1 | Backend healthy + scheduler armed for 14:00 ET trigger | ✅ verified live (TZ fix in place) |
| 2 | Kill switch armed, paused=false | ✅ verified live |
| 3 | Alpaca paper account clean (no stale orders) | ✅ all UAT shadows cancelled |
| 4 | OWASP headers + auth gate | ✅ 5/5 headers, 8/8 routes 302 |
| 5 | Sovereign endpoints serving (red-line, leaderboard, compute-cost) | ✅ 31 + 2 + 31 rows |
| 6 | Quant strategy seed in BQ (seed_0000 champion) | ✅ verified live |
| 7 | Cost-budget governance + observability | ✅ both endpoints 200, $1.91/mo under $50 cap |

**0 of 7 critical needs are blocked.**

## 7. Q/A's task

Render PASS / CONDITIONAL / FAIL as the AGGREGATE Go/No-Go verdict for Monday paper-trading.

**Suggested decision criteria for Q/A:**
- **PASS** if: Monday-critical path is intact, all 7 critical needs verified, no Monday-blocker bug undocumented, follow-up tickets are filed for known gaps.
- **CONDITIONAL** if: critical path intact but 1+ disclosure feels incomplete; explicit conditions and follow-ups must be tracked.
- **FAIL** if: a real Monday-day-of risk is undocumented or the bundle hides material concerns.

Q/A must NOT rubber-stamp. The user explicitly said "stop rigging dual-evaluator PASS" earlier this session.

If PASS: Main flips both 16.23 AND 16.15 to done (the latter is what the user actually cares about).
If CONDITIONAL or FAIL: Main does NOT flip 16.15. 16.23 closes with the Q/A's verdict, 16.15 stays in-progress until conditions met.

**16.2 and 16.3 are NOT closed by this verdict** regardless of outcome (Q/A's prior conditions stand).
