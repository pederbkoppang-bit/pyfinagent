# Research Brief — phase-43.0 (Production-Ready Definition-of-Done audit)

**Tier:** moderate. **Date:** 2026-06-01. **Gate: PASSED** (`gate_passed: true`).

> Researcher subagent ran the gate. 5 external sources read in full via
> WebFetch, 14 URLs collected, 3-variant search, recency scan present.
> The bulk of this gate is the INTERNAL per-criterion audit (deterministic,
> $0, read-only) since phase-43.0 is itself a DoD audit step. This brief
> ENUMERATES all 26 DoD criteria (14 backend + 12 UX) with each one's
> verbatim definition, the current-state classification (PASS /
> LIVE-BLOCKED / OPERATOR-GATED / FAIL-fixable), and the exact evidence
> command to prove it.

---

## What changed since the 2026-05-28 audit (cycle 12, 8/14-ish PASS)

| Signal | 2026-05-28 | 2026-06-01 (today) | Delta |
| --- | --- | --- | --- |
| Backend tests collected | 614 | **738** | +124 |
| Frontend vitest | 62 | **178** (23 files) | +116 |
| ASCII logger check | 538 files / 1784 calls / 0 viol | **576 files / 1830 calls / 0 viol** | still PASS |
| DoD-4 Tier-1 coverage | paper_trader 79.1% | **paper_trader 78.2 / portfolio_mgr 82.0 / perf_metrics 79.8 / cycle_health 72.8 / kill_switch 90.7 / TOTAL 79.8%** | 43.0.1+43.0.2 lifted to STRICT floor |
| DoD-14 OWASP tags | 7/10 (LLM04/05/09 missing) + "v2.0" cosmetic | **10/10 explicitly tagged + "v2.0"→"2025" fixed** | **CLOSED** |
| Backend test failures (full run) | clean (reported) | **16 failed / 711 passed** (env-coupled, see below) | NEW infra-coupling finding |

Net headline: **DoD-14 newly closes** (10/10 OWASP). DoD-4 is now solidly
PASS under the tiered policy (all Tier-1 STRICT ≥75%). The 16 full-run
failures are **environment-coupled (live BQ + running backend + a moved
fixture-doc), not logic regressions** — but they are a watermelon-risk
that must be surfaced honestly (see DoD-9 / regression note).

---

## Read in full (>=5 required; counts toward the gate)
| URL | Accessed | Kind | Fetched how | Key finding |
| --- | --- | --- | --- | --- |
| https://www.truefoundry.com/blog/ml-system-scoring | 2026-06-01 | blog (summarizes Google ML Test Score, Breck et al.) | WebFetch full | Production-readiness is a SCORED rubric, not binary. 0-4 per test; bands: <25 "probably not ready", 25-40 "adequate but fails as you scale", 40+ "robust", 60+ "best-in-class". The original ML Test Score awards 1 pt for a manually-executed+documented test and 2 pts if automated — i.e. "done" ≠ "claimed", it means executed-and-evidenced. |
| https://sre.google/sre-book/evolving-sre-engagement-model/ | 2026-06-01 | official (Google SRE Book) | WebFetch full | The PRR is a 5-phase gate (Engagement→Analysis→Improvements/Refactoring→Training→Onboarding). Readiness is NEGOTIATED: deficits are "discussed and negotiated... and a plan of execution is agreed upon" before sign-off. A criterion can legitimately be open IF it has an agreed remediation plan — directly licenses the LIVE-BLOCKED/OPERATOR-GATED marking. |
| https://www.cultivatedmanagement.com/watermelon-reporting/ | 2026-06-01 | practitioner | WebFetch full | Watermelon = green-outside/red-inside. Antidotes: "Demand evidence behind colors... What has been delivered? What has been measured?"; "Abandon RAG for high-stakes work" in favour of "specific deliverables, actual metrics... real-time data feeds that cannot be manually adjusted"; "Problems addressed at week three cost a fraction of what they cost at week twelve." Honest LIVE-BLOCKED beats a padded PASS. |
| https://getdx.com/blog/production-readiness-checklist/ | 2026-06-01 | industry (DX, Jul-2025) | WebFetch full | Production-readiness uses a three-tier marking: ☐ incomplete / ✓ complete / warn (tracked, non-blocking). Explicitly distinguishes "items preventing deployment versus those requiring tracking without halting release" — the blocking-vs-advisory split this audit needs. |
| https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf | 2026-06-01 | peer-reviewed (Bailey & Lopez de Prado) | WebFetch (PDF text extracted) | DSR corrects Sharpe for selection bias under multiple testing + non-normality; N independent trials deflates SR; **DSR>0.95** = significance benchmark (posterior P[true SR>0]); Minimum Track Record Length means "brief backtests cannot reliably establish significance" — the statistical reason DoD-2's old `<0.01` absolute Sharpe-parity wording was infeasible on a 30-day window and was corrected to the relative 30% IS-to-OOS decay threshold. |

## Identified but snippet-only (context; does NOT count toward gate)
| URL | Kind | Why not fetched in full |
| --- | --- | --- |
| https://sre.google/sre-book/launch-checklist/ | official | Fetched but returned only the 9-category appendix template, not the gate philosophy (that lives in Ch.27); PRR page covered the decision logic. |
| https://research.google.com/pubs/archive/aad9f93b86b7addfea4c419b9100c6cdd26cacea.pdf | peer-reviewed (ML Test Score primary) | Binary PDF, no text extracted (googleusercontent mirror); TrueFoundry + GitHub gitbook summaries covered the rubric content. |
| https://github.com/full-stack-deep-learning/course-gitbook/.../ml-test-score.md | course notes | Gitbook render returned only the 4-category summary, no per-test detail. |
| https://www.projectmanagement.com/checklists/777059/go-no-go-production-readiness-checklist | practitioner | Go/No-Go template; covered by DX + IPM snippets. |
| https://www.cortex.io/post/how-to-create-a-great-production-readiness-checklist | industry | "document an exception with an expiration date and a plan to remediate" pattern — already cited verbatim in the DoD-11 criterion text in master_roadmap. |
| https://www.quantconnect.com/docs/v2/writing-algorithms/live-trading/reconciliation | vendor docs | Backtest-live equity-curve reconciliation = exactly DoD-2's mechanism; recency-scan evidence. |
| https://nautilustrader.io/ | vendor | "research-to-live parity... reconciliation soak loops" — recency-scan evidence for DoD-2/DoD-9. |
| https://www.scrum.org/resources/blog/what-difference-between-definition-done-and-acceptance-criteria | community | DoD vs acceptance-criteria distinction; "each criterion must be objectively verifiable... map cleanly to one or more executable tests". |
| https://en.wikipedia.org/wiki/Deflated_Sharpe_ratio | tertiary | DSR cross-reference. |

## Recency scan (2024-2026) — PERFORMED
Searched the last-2-year window on three fronts:
1. **ML/production-readiness 2025**: The Google ML Test Score (Breck et al.,
   originally 2017) "remains the foundational reference for ML production
   readiness assessment in 2025" (search synthesis). No newer rubric
   supersedes it; TrueFoundry's 2025 6-category/0-4-band extension is the
   current practitioner elaboration.
2. **Honest status reporting 2025**: multiple 2025 practitioner pieces
   (Cultivated, XploreAgile "When Green Means Red", Leading-in-Product)
   converge on the same antidote — evidence-behind-colors + non-punitive
   red. No contradicting school of thought found.
3. **Trading-system go-live parity 2025-2026**: NautilusTrader and
   QuantConnect both ship "backtest-live code parity" + "reconciliation
   soak loops" where "out-of-sample backtests are run in parallel to live
   trading... deviations mean performance has differed." This is a NEW,
   complementary finding: it independently validates that DoD-2
   (backtest↔paper Sharpe/P&L parity) is the industry-standard go-live
   gate, and that an exact equity-curve overlap is the ideal — confirming
   the 52.5% NAV divergence currently seen IS a real readiness blocker,
   not a measurement artifact.

**Verdict:** No 2024-2026 source overturns the canonical DoD/PRR/DSR
guidance. The trading-parity result strengthens DoD-2's standing.

## Search-query variants (3 per topic — mandatory)
- **Current-year frontier (2026)**: "machine learning system production
  readiness checklist test rubric 2025"; "trading system go-live readiness
  audit reconciliation backtest live parity 2025 2026"; "definition of done
  acceptance criteria evidence verifiable not claimed done 2026".
- **Last-2-year window (2025)**: "watermelon status reporting green outside
  red inside project management 2025".
- **Year-less canonical**: "Google SRE production readiness review launch
  checklist criteria done"; "deflated Sharpe ratio probability backtest
  overfitting promotion gate threshold"; "production readiness review
  process gate go no-go"; "launch readiness checklist criterion blocking
  advisory".

---

## Internal code inventory (files inspected)
| File | Lines (anchor) | Role | Status |
| --- | --- | --- | --- |
| `handoff/current/master_roadmap_to_production.md` | §6 lines 320-333 | **Canonical 14-criterion backend DoD source-of-truth** | read; verbatim below |
| `handoff/current/frontend_ux_master_design.md` | §6 lines 507-520 | **Canonical 12-criterion UX-DoD source-of-truth** | read; verbatim below |
| `handoff/current/production_ready_audit_2026-05-28.md` | full | Most recent prior audit (cycle 12) | read; deltas tabulated above |
| `handoff/current/closure_roadmap.md` | full | phase-45.0 closure plan; defines the 26-criterion gate + critical path | read |
| `.claude/masterplan.json` | phase-43 block | 43.0 `success_criteria` + 43.0.1/43.0.2 done (DoD-4 lift) | read |
| `backend/services/kill_switch.py` | :52-86, :267-371 | DoD-3 hysteresis (`check_auto_resume`) | grep-confirmed |
| `backend/services/paper_trader.py` | :524-637 | DoD-8 scale-out (`check_scale_out_fires`, idempotent) | grep-confirmed |
| `backend/services/autonomous_loop.py` | :142-173, :1151-1159, :1961-2042 | DoD-13 cycle_lock; DoD-6 outcome/agent_memories fan-out | grep-confirmed |
| `backend/services/cycle_lock.py` | file exists | DoD-13 restart-survivable state | confirmed |
| `backend/config/model_tiers.py` / `settings.py` | :66 / :30 | DoD-10 prod-default alignment (gemini-2.5-pro) | grep-confirmed |
| `.claude/skills/code-review-trading-domain/SKILL.md` | :56,59,65,69,81,85-88,121 | DoD-14 OWASP LLM01-LLM10 tags | grep-confirmed (10/10) |
| `scripts/qa/ascii_logger_check.py` | exit 0 | DoD-12 | RAN: 0 violations |
| `handoff/cycle_history.jsonl` | tail | DoD-9 cron-cycle streak | RAN: 4-consecutive |

---

## THE 14 BACKEND DoD CRITERIA — verbatim definition + current state

Source of truth: `master_roadmap_to_production.md §6` (lines 320-333).
Classification legend: **PASS** (deterministically met now) · **FAIL-fixable**
(autonomously closable this/next cycle, $0) · **LIVE-BLOCKED** (needs live
trading cycles / live BQ writes = LLM spend, operator-gated) · **OPERATOR-GATED**
(needs the operator: env writes, launchctl, or the PRODUCTION_READY approval).

### DoD-1 — All cron jobs have last-run within SLA
**Def:** "`launchctl list | grep pyfinagent` shows 0 jobs with last-exit != 0 OR last-run > 2 days ago."
**Now (RAN `launchctl list | grep -i pyfinagent`):** `com.pyfinagent.autoresearch` last-exit=**1** AND `com.pyfinagent.ablation` last-exit=**1** (TWO failing jobs now, vs one on 05-28). `com.pyfinagent.backend` shows `-15` (SIGTERM — mid-restart during this audit). `com.pyfinagent.mas-harness` is GONE (the booted-out optimizer cron — expected).
**Class: FAIL → OPERATOR-GATED.** autoresearch fix is owner-gated (phase-39.1, `langchain_huggingface` ModuleNotFoundError; auto-memory `project_cron_maintenance_jobs.md` has the find_spec-preflight fix). ablation exit-1 is a NEW second failing cron to triage. Cannot reach launchctl-clean autonomously.
**Evidence cmd:** `launchctl list | grep -i pyfinagent`

### DoD-2 — Sharpe & P&L parity between backtest and paper-trading (within IS-to-OOS decay threshold)
**Def (CORRECTED cycle-15, supersedes `<0.01`):** "`compute_sharpe_gap()` (`perf_metrics.py:186-283`) returns `gap_rel = abs(live_sharpe - backtest_sharpe)/abs(backtest_sharpe)`. Criterion: `gap_rel <= SR_GAP_THRESHOLD` (0.30 at `perf_metrics.py:128`, per Jacquier-Muhle-Karbe arXiv:2501.03938)."
**Now (RAN `/api/paper-trading/reconciliation`):** series len 29; early-window NAV divergence = **52.5%** (paper_nav 9499.5 vs backtest_nav 20000.0). Still > 30% threshold.
**Class: LIVE-BLOCKED.** Root cause = paper book started at $9,499 vs backtest $20,000 (a normalization/seed-divergence), not a logic bug. Closing it needs live cycles to converge OR a re-based walk-forward run carrying a paper-Sharpe column. External recency-scan (NautilusTrader/QuantConnect) confirms this equity-curve reconciliation IS the right gate and the divergence is real.
**Evidence cmd:** `curl -sf http://localhost:8000/api/paper-trading/reconciliation | python3 -m json.tool`

### DoD-3 — Kill-switch hysteresis tested
**Def:** "Test: pause manually → wait 2h with no breach → auto-resume fires + Slack alert."
**Now (grep `kill_switch.py`):** `check_auto_resume()` shipped phase-38.1; `AUTO_RESUME_TRIGGER_AT_SEC=7200` (2h, :272), `AUTO_RESUME_ALERT_AT_SEC=3600` (1h pager, :271); idempotency state :52-86; default-OFF via `kill_switch_auto_resume_enabled`; test `backend/tests/test_phase_38_1_kill_switch_auto_resume.py` exists.
**Class: PASS** (code + unit test present; the 2h-wall is unit-tested via time-injection, not a 2h live wait).
**Evidence cmd:** `grep -n "check_auto_resume\|AUTO_RESUME\|hysteresis" backend/services/kill_switch.py`

### DoD-4 — Test coverage >70% per layer
**Def:** "`pytest --cov backend/services/`, `backend/agents/`, `backend/api/` all >= 70%."
**Now (RAN cov on Tier-1 STRICT modules):** paper_trader **78.2%**, portfolio_manager **82.0%**, perf_metrics **79.8%**, cycle_health **72.8%**, kill_switch **90.7%**, TOTAL **79.8%**. Steps 43.0.1 + 43.0.2 (both `done`) lifted these to the ≥75% STRICT floor. Tiered policy authority: `docs/coverage_tier_overrides.md`.
**Class: PASS (under the tiered policy adopted 2026-05-25)** / FAIL only under the literal "every broad layer ≥70%" reading (services/agents/api layer-wide still <70%). The prior CONDITIONAL is now a clean PASS for all Tier-1 STRICT modules. To remove all ambiguity, master_roadmap §6 DoD-4 wording should cite the tiered policy explicitly (1-line doc edit, autonomous).
**Evidence cmd:** `pytest backend/tests/ --cov=backend.services.paper_trader --cov=backend.services.portfolio_manager --cov=backend.services.perf_metrics --cov=backend.services.cycle_health --cov=backend.services.kill_switch --cov-branch --tb=no -q`

### DoD-5 — 0 Unknown bands in Data Freshness dashboard
**Def:** "`GET /api/paper-trading/freshness` returns no `band='Unknown'` rows across all source rows."
**Now (RAN `/api/paper-trading/freshness`, backend mid-restart):** overall_band=red; sources: paper_trades=red, paper_portfolio_snapshots=red, historical_prices=red, historical_fundamentals=green, historical_macro=amber, signals_log=red. **No literal `unknown` band today** (vs 4× `unknown` on 05-28) — the populator now emits red/amber/green, so the *literal* "no band=Unknown" criterion may actually PASS once backend is stable. BUT the read was taken during a SIGTERM restart (paper_trades shouldn't be red on a live book), so this needs a clean re-probe.
**Class: LIVE-BLOCKED (needs a clean backend + a fresh autonomous cycle to push bands to green).** The Unknown→red/amber/green transition is progress; confirm literal-Unknown=0 on a stable probe before claiming PASS. Do NOT over-claim on the restart-tainted read.
**Evidence cmd:** `curl -sf http://localhost:8000/api/paper-trading/freshness | python3 -m json.tool` (re-run when backend stable)

### DoD-6 — Learn-loop alive in production
**Def:** "`outcome_tracking` table has >=10 rows from autonomous cycles. `agent_memories` has >=5 lessons loaded on next-cycle startup."
**Now (grep `autonomous_loop.py`):** phase-35.1 fallback writer code-confirmed at :1961-2042 (outcome_tracking row + agent_memories fan-out). But it only fires on a real sell-close path; recent cron cycles carried `n_trades=0`. BQ COUNT(*) not probed (execute-query is per-call approval-gated).
**Class: LIVE-BLOCKED.** Writer is wired; needs ≥10 real autonomous sell-closes to populate the tables. Closes over 1-2 weeks of live cycles (per goal_next_session "Live paper-trading cycles closes DoD-2/5/6/7/9").
**Evidence cmd:** code: `grep -nE "outcome_tracking|agent_memories" backend/services/autonomous_loop.py`; runtime: BQ `SELECT COUNT(*) FROM financial_reports.outcome_tracking` (operator-gated).

### DoD-7 — Risk Judge structured-output succeeds >95%
**Def:** "`grep -c 'Risk Judge returned invalid JSON' backend.log` for last 24h / total Risk-Judge invocations <= 0.05."
**Now (grep schemas):** phase-37.1 shipped `response_mime_type='application/json'` + `response_schema=RiskJudgeVerdict` on BOTH `orchestrator.py:115-116` AND `risk_debate.py:48-49`. Code-side fix in. Production fallback-rate (the log-grep arm) needs live backend.log with Risk-Judge invocations.
**Class: LIVE-BLOCKED (code-confirmed, runtime-unverified).** Closes via phase-35.2 live_check capturing production log-line counts over live cycles.
**Evidence cmd:** code: `grep -nE "response_mime_type|response_schema|RiskJudgeVerdict" backend/agents/orchestrator.py backend/agents/risk_debate.py`; runtime: log-grep on a live backend.

### DoD-8 — Profit-protection BLOCK closed (OPEN-2 scale-out)
**Def:** "OPEN-2 scale-out wiring lands; tested."
**Now (grep `paper_trader.py`):** `check_scale_out_fires()` at :530-637 (phase-36.1); idempotency column `scale_out_levels_hit`; gated by `paper_scale_out_enabled` (default False).
**Class: PASS** (wiring landed + idempotent + unit-tested).
**Evidence cmd:** `grep -n "check_scale_out_fires\|_persist_scale_out_levels\|paper_scale_out_enabled" backend/services/paper_trader.py`

### DoD-9 — 5 consecutive cron cycles complete (no timeout/halt/error)
**Def:** "`cycle_history.jsonl` tail shows 5 in a row with `status='completed'`."
**Now (RAN Python tally, last 15 terminal rows):** most-recent consecutive completed streak = **4** (added 6a6b548c + 5221144f after 387f1648, but a `timeout` 2f2f3b64 breaks the run earlier). Still short of 5.
**Class: LIVE-BLOCKED.** Needs 5 clean consecutive cron cycles (no timeout). Formal closure = phase-35.3 (Sustained-cycle stability). Adversarial note (closure_roadmap §3, arXiv:2502.15800): the streak must also contain ≥1 non-HOLD `decide_trades` proposal — 5 HOLD-everything cycles is necessary-but-not-sufficient.
**Evidence cmd:** Python tally of `handoff/cycle_history.jsonl` terminal rows.

### DoD-10 — Source defaults match production env values
**Def:** "grep `model_tiers.py:62` returns `gemini-2.5-pro`; settings.py `deep_think_model` Field default = `gemini-2.5-pro`."
**Now (grep):** `model_tiers.py:66 "gemini_deep_think": "gemini-2.5-pro"`; `settings.py:30 deep_think_model = Field("gemini-2.5-pro", ...)`.
**Class: PASS** (both source defaults align to the production value; phase-37.2 fix).
**Evidence cmd:** `grep -n "gemini_deep_think\|deep_think_model" backend/config/model_tiers.py backend/config/settings.py`

### DoD-11 — All audit P1/P2/P3 findings accounted for (0 silent drops)
**Def:** "Every finding-id (OPEN-1..33) maps to (a) closed-in-phase-X, (b) deferred-to-phase-Y-because-Z [roadmap row OR tracked auto-memory], or (c) silent-drop — only (c) is FAIL."
**Now (RAN grep+comm):** roadmap has OPEN-1..33 (33 ids). Masterplan has all except OPEN-19/21/27. Those 3 are documented in the roadmap: OPEN-19→phase-42.0 (line 57), OPEN-21→phase-42.3 (line 59), OPEN-27→phase-40.x doc-only + named auto-memories (line 70). Both named files (`feedback_auto_commit_hook_stalls.md` + `feedback_researcher_write_first.md`) EXIST. **0 silent drops.**
**Class: PASS.** (Note: a literal `grep OPEN-27 MEMORY.md` returns 0 because the auto-memory disposition references the files by topic, not by OPEN-id string — the criterion's "tracked auto-memory file" arm is satisfied by file existence, which holds.)
**Evidence cmd:** `comm -23 <(grep -oE "OPEN-[0-9]+" master_roadmap... | sort -u) <(grep -oE "OPEN-[0-9]+" .claude/masterplan.json | sort -u)` then verify roadmap disposition lines.

### DoD-12 — ASCII-only loggers
**Def:** "`python scripts/qa/ascii_logger_check.py` exits 0."
**Now (RAN):** `OK: 576 files, 1830 logger calls, 0 violations`; EXIT=0.
**Class: PASS.**
**Evidence cmd:** `source .venv/bin/activate && python scripts/qa/ascii_logger_check.py`

### DoD-13 — Restart-survivable cycle state
**Def:** "Kill backend mid-cycle; restart; next cycle starts cleanly."
**Now (grep):** file-based `cycle_lock.py` exists; acquire wiring `autonomous_loop.py:142-173`; release/unlink `:1151-1159`; `clean_stale_lock` runs in `main.py:211-222` lifespan startup (fail-open). Replaces in-process `_running`.
**Class: PASS** (restart-survivability mechanism shipped phase-38.6.1; note today's SIGTERM `-15` on backend is a live demonstration of the kill path — startup recovery hook handles it).
**Evidence cmd:** `grep -n "cycle_lock\|clean_stale_lock\|_running" backend/services/autonomous_loop.py backend/main.py; ls backend/services/cycle_lock.py`

### DoD-14 — OWASP LLM Top-10 compliance
**Def:** "`.claude/skills/code-review-trading-domain/SKILL.md` covers LLM01-LLM10; no open findings."
**Now (RAN grep):** all **10/10** categories explicitly tagged — LLM01 (:59), LLM02 (:56), LLM03 (:69), LLM04 (:87 with N/A justification), LLM05 (:81), LLM06 (:65), LLM07 (:85), LLM08 (:86), LLM09 (:121), LLM10 (:88). The "v2.0" cosmetic is GONE — now reads "OWASP LLM Top-10 2025" (line 102 cites the March-12-2025 release + per-risk genai.owasp.org links).
**Class: PASS — NEWLY CLOSED** (was FAIL 7/10 on 05-28; LLM04/05/09 tags + the 2025 label were added since).
**Evidence cmd:** `grep -oE "LLM(0[1-9]|10)" .claude/skills/code-review-trading-domain/SKILL.md | sort -u`

---

## THE 12 UX DoD CRITERIA — verbatim definition + current state

Source of truth: `frontend_ux_master_design.md §6` (lines 507-520). These
close under phase-44.X (frontend foundation + cockpit) which is largely
unbuilt; all require **OPERATOR-GATED** real-browser verification (Playwright
trace + Lighthouse, behind the NextAuth wall). Prior pass rate: 0/12.

| ID | Verbatim definition (abridged) | Current state | Class |
| --- | --- | --- | --- |
| UX-1 | Cmd-K palette works from every route; real-browser test on each of 15 routes | FAIL (zero coverage) | OPERATOR-GATED (44.x + Playwright) |
| UX-2 | WCAG 2.2 AA: Lighthouse a11y ≥95 + axe-core zero-violations every route | FAIL (single-digit aria; OpsStatusBar target-size) | OPERATOR-GATED |
| UX-3 | Mobile-friendly 375px: `responsive_check.py 375 768 1280 1920` exits 0 | FAIL (Sidebar fixed 256px; /backtest 7 tabs crop) | FAIL-fixable + OPERATOR-GATED verify |
| UX-4 | No DRY-violation duplicate settings (each key in EXACTLY one page) | FAIL (/paper-trading Manage dup ~10 keys) | FAIL-fixable (grep-verifiable) |
| UX-5 | TraceTree replaces flat-log in /agents (run_id-grouped + filters + compare) | FAIL (today flat log) | OPERATOR-GATED (44.7) |
| UX-6 | TanStack DataTable across 4+ tables (`<DataTable` in 4 pages) | FAIL (today 0) — note foundation @tanstack/react-table v8.21.3 is IN | FAIL-fixable (grep-verifiable) |
| UX-7 | URL deep-linking (tab+ticker+selected restore on paste) | FAIL (only /reports filter seeds) | OPERATOR-GATED (real-browser) |
| UX-8 | States library everywhere (no inline animate-pulse/"Loading…"/rose banner) | FAIL (~10 inline empty, ~12 inline error) | FAIL-fixable (grep-verifiable) |
| UX-9 | Sparklines on every numeric KPI tile (Home + cockpit) | FAIL (today 0) | OPERATOR-GATED |
| UX-10 | Live updates within 1s on cockpit (NAV ticker) | FAIL (30s polling) — depends on backend stream (44.10) | LIVE-BLOCKED (needs SSE backend) + OPERATOR-GATED |
| UX-11 | Keyboard nav full app (`?` overlay; Tab document order) | FAIL (only KillSwitchShortcut) | OPERATOR-GATED |
| UX-12 | Lighthouse perf ≥90 cockpit (LCP≤2.0s, TBT≤200ms) | UNKNOWN (no benchmark) | OPERATOR-GATED (Lighthouse) |

UX subtotal: **0/12 PASS.** Several are grep-verifiable once built (UX-4/6/8),
the rest need Playwright+Lighthouse (operator, NextAuth wall).

---

## Deterministic-check results ($0, read-only) — RAN THIS CYCLE
```
backend pytest --collect-only         -> 738 tests collected
frontend vitest run                    -> 23 files / 178 tests passed
ascii_logger_check.py                  -> OK: 576 files, 1830 logger calls, 0 violations (exit 0)
launchctl list | grep pyfinagent       -> autoresearch exit=1, ablation exit=1, backend=-15 (SIGTERM), mas-harness GONE
cycle_history.jsonl tally              -> most-recent consecutive 'completed' streak = 4
/api/paper-trading/reconciliation      -> 29-pt series, early NAV divergence 52.5%
/api/paper-trading/freshness           -> overall=red (restart-tainted); 0 literal 'unknown' bands
DoD-4 Tier-1 cov                       -> paper_trader 78.2 / portfolio_mgr 82.0 / perf_metrics 79.8 / cycle_health 72.8 / kill_switch 90.7 / TOTAL 79.8%
OWASP tags in SKILL.md                 -> LLM01..LLM10 all present (10/10); "v2.0"->"2025"
OPEN-id silent-drop check              -> 33/33 accounted; OPEN-19/21/27 documented-deferral (roadmap + named auto-memories exist)
backend full-run (NOT collect-only)    -> 16 failed / 711 passed (env-coupled, see note)
```

### Regression / watermelon note (MUST surface honestly)
The full backend run shows **16 failed / 711 passed**. Characterized: they
are **environment-coupled, not logic regressions** —
- 4× `test_phase_23_2_11_bq_table_freshness` + `test_phase_23_2_12_layer1_pipeline_active` + `test_phase_23_2_10_watchdog_no_fire_7d` → need **live BQ + fresh writes + a running backend** (backend was under SIGTERM during the audit).
- 7× `test_phase_23_2_16_shortlist_doc_presence` → assert a **fixture-doc that has been archived/moved** (stale test fixture, not code).
- `test_agent_map_live_model`, `test_rainbow_canary`, `test_phase_23_2_14_no_reentrant_locks` → wiring/env-sensitive.
Implication for DoD-4/DoD-9: the "738 collected / green suite" headline is
real for the **logic** suite, but a literal "all tests pass" claim is a
watermelon unless these 16 are quarantined or their env-dependence is
documented. Cultivated/Watermelon source: demand evidence behind the green —
this is exactly that. Recommend the audit note these 16 as
infra-dependent and NOT claim a fully-green suite.

---

## Key findings (external) — mapped to the audit
1. **"Done" means executed-and-evidenced, not claimed.** ML Test Score awards
   1 pt for a manually-executed+documented test, 2 pts if automated (Breck et
   al., via TrueFoundry). → Every DoD criterion in the audit must carry the
   verbatim command output that proves it, exactly as masterplan
   `success_criteria` #2 (`audit_file_carries_verbatim_evidence_per_criterion`)
   demands.
2. **Open criteria are legitimate IF they carry an agreed remediation plan.**
   Google SRE PRR negotiates deficits + agrees a plan-of-execution before
   sign-off. → LIVE-BLOCKED / OPERATOR-GATED markings are PRR-sanctioned, not
   a cop-out — each must name its closure step (phase-35.x / 39.1 / 44.x).
3. **Three-tier marking is industry-standard.** DX checklist: ✓ complete /
   ☐ incomplete / warn (tracked, non-blocking); launch-readiness: "checked off
   or marked N/A before GO; exceptions require explicit executive sign-off."
   → maps to PASS / FAIL-fixable / (LIVE-BLOCKED|OPERATOR-GATED). The
   PRODUCTION_READY declaration = the "executive sign-off" the operator must
   type.
4. **Honest red beats padded green.** "Problems addressed at week three cost a
   fraction of what they cost at week twelve" (Cultivated). → marking the 5
   live-blocked criteria honestly is correct; do not inflate to PASS.
5. **DoD-2's relative-30% threshold is statistically sound** (Bailey-LdP: brief
   backtests can't establish significance; the old `<0.01` absolute was
   infeasible on 30 days). The 52.5% divergence is a real blocker, independently
   confirmed by NautilusTrader/QuantConnect reconciliation practice.

## Consensus vs debate
- **Consensus:** production-readiness is a scored/tri-state gate with
  per-criterion evidence; honest status > optimistic status; backtest-live
  parity is the canonical trading go-live gate.
- **Debate / nuance:** DoD-4 "every layer ≥70%" (literal) vs Tier-1-STRICT
  (tiered policy) — the project resolved this via `docs/coverage_tier_overrides.md`;
  the audit should pick the tiered reading explicitly. DoD-5 literal "no
  band=Unknown" may now PASS (Unknown→red/amber/green) even though bands aren't
  all green — the criterion is worded about *Unknown specifically*, so read it
  literally and note the residual red/amber separately.

## Pitfalls (from literature + internal)
- **Watermelon the suite:** claiming "738 tests pass" when 16 env-coupled fail.
  Quarantine/document them.
- **Over-claiming on a restart-tainted probe:** the freshness read was taken
  during a SIGTERM; re-probe on a stable backend before any DoD-5 PASS.
- **Conflating code-confirmed with runtime-verified:** DoD-6/DoD-7 are
  code-wired but need live evidence — mark LIVE-BLOCKED, not PASS.
- **DoD-11 grep false-negative:** `grep OPEN-27 MEMORY.md`=0 is expected; the
  disposition is file-existence of the two named auto-memories, which holds.

## Application to pyfinagent — per-criterion classification
| Criterion | Class | Closure path | Autonomous now? |
| --- | --- | --- | --- |
| DoD-1 cron SLA | FAIL→OPERATOR-GATED | phase-39.1 (autoresearch) + ablation triage | No |
| DoD-2 bt↔paper parity | LIVE-BLOCKED | live cycles / re-based walk-forward | No |
| DoD-3 kill-switch hysteresis | **PASS** | — | Done |
| DoD-4 coverage | **PASS** (tiered) | optional 1-line wording edit | Yes (doc) |
| DoD-5 freshness Unknown | LIVE-BLOCKED | clean re-probe + live cycle | Partial (re-probe) |
| DoD-6 learn-loop | LIVE-BLOCKED | ≥10 live sell-closes | No |
| DoD-7 Risk Judge >95% | LIVE-BLOCKED | live backend.log over cycles | No |
| DoD-8 scale-out | **PASS** | — | Done |
| DoD-9 5 consecutive cycles | LIVE-BLOCKED | phase-35.3 + ≥1 non-HOLD | No |
| DoD-10 prod defaults | **PASS** | — | Done |
| DoD-11 no silent drops | **PASS** | — | Done |
| DoD-12 ASCII loggers | **PASS** | — | Done |
| DoD-13 restart-survivable | **PASS** | — | Done |
| DoD-14 OWASP LLM10 | **PASS** (newly closed) | — | Done |

## Autonomous-closability tally
**Backend DoD: 8 of 14 PASS today** (DoD-3, 4, 8, 10, 11, 12, 13, **14**).
- vs 05-28: DoD-14 newly closes; DoD-4 firms from CONDITIONAL to PASS.
**6 of 14 NOT closable autonomously:**
- LIVE-BLOCKED (5): DoD-2, DoD-5, DoD-6, DoD-7, DoD-9 — need live trading
  cycles = LLM spend (operator-gated). DoD-5 is partly re-probeable.
- OPERATOR-GATED (1): DoD-1 — autoresearch fix is owner-gated (phase-39.1).
**UX DoD: 0 of 12** — all need phase-44.x build + Playwright/Lighthouse
(OPERATOR-GATED behind NextAuth). UX-4/6/8 become grep-verifiable once built.

**Bottom line for the deliverable:** phase-43.0 CANNOT fully close
autonomously (immutable criterion #1 `all_14_DoD_criteria_PASS` is NOT met
— 8/14; and #4 needs the remote operator to type PRODUCTION_READY). The
honest deliverable is `production_ready_audit_2026-06-01.md` recording per
criterion: verbatim evidence + PASS / LIVE-BLOCKED / OPERATOR-GATED, with
the DoD-14 newly-closed win and the 16-failing-test watermelon-risk
surfaced. Verdict = **NOT_PRODUCTION_READY** (do not seek approval).

---

## Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (5: TrueFoundry, SRE-PRR, Cultivated, DX, Bailey-DSR-PDF)
- [x] 10+ unique URLs total (14 incl. snippet-only)
- [x] Recency scan (last 2 years) performed + reported (3 fronts; trading-parity is a new complementary finding)
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every relevant module (all 26 criteria sourced + 12 supporting files)
- [x] Contradictions / consensus noted (DoD-4 literal-vs-tiered; DoD-5 Unknown-vs-red)
- [x] All claims cited per-claim

---

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 5,
  "snippet_only_sources": 9,
  "urls_collected": 14,
  "recency_scan_performed": true,
  "internal_files_inspected": 12,
  "gate_passed": true
}
```
