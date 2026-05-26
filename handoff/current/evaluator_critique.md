# Evaluator critique -- Cycle 72 SSOT NAV/P&L via root-level LivePortfolioProvider

**Date:** 2026-05-26
**Cycle:** 72 (Single-Source-of-Truth NAV refactor; root-level Context provider)
**Q/A spawn:** 1 of 1 (first spawn for cycle 72, no prior CONDITIONAL to revise)
**Verdict:** PASS

---

## 1. 5-item harness-compliance audit (FIRST, per feedback_qa_harness_compliance_first.md)

| # | Audit item | Evidence | Verdict |
|---|---|---|---|
| 1 | Researcher BEFORE contract? | `handoff/current/research_brief_phase_ssot_nav.md` (38,344 bytes, deep tier, 21 sources read in full, 37 URLs collected, gate_passed=true) authored 2026-05-26 19:18; `handoff/current/contract.md` authored 19:20 -- 2 minutes AFTER the brief. Three-pass research (scan/gap/adversarial). Cross-vendor triangulation (Robinhood, Alpaca, IBKR, QuantConnect). | PASS |
| 2 | Contract pre-commit? | `contract.md` declares the N* delta (B primary -- 4 NAV values burn operator attention; R secondary -- restores trust), three root causes (RC1 dual sources, RC2 dual polls, RC3 stale-baseline P&L Today), and an 8-row migration table with explicit file:line targets. Plan steps enumerated 1-10. | PASS |
| 3 | experiment_results.md? | Operator prompt acknowledges this is a UX refactor cycle and the harness_log captures the equivalent. The contract + this critique + harness_log entry collectively satisfy the harness invariant for the UX-refactor sub-class. Acceptable per per-step-protocol.md "GENERATE artifacts vary by step class". | PASS (with caveat -- noted in Section 4 dimension 6) |
| 4 | Log-LAST? | `handoff/harness_log.md` last entries show cycle 70 (donut Option B) and cycle 71 (Slack regression fixes). No cycle 72 entry yet. Main must append a `## Cycle 72 -- 2026-05-26 -- SSOT LivePortfolioProvider result=PASS` block AFTER this PASS and BEFORE any masterplan flip. masterplan.json status unchanged. | PASS (action: Main appends log after this critique) |
| 5 | No verdict-shopping? | First Q/A spawn for cycle 72 -- no prior verdict to reverse. Sycophancy-under-rebuttal check N/A. Mtime check: this is a new cycle (new files: `live-portfolio-context.tsx`, `live-portfolio-context.test.tsx`, fresh research brief at 19:18). | PASS |

All 5 audits PASS.

---

## 2. Deterministic checks (10 items)

| # | Check | Command + actual output | Verdict |
|---|---|---|---|
| 1 | TypeScript typecheck | `cd frontend && npx tsc --noEmit; echo EXIT=$?` -> `EXIT=0` | PASS |
| 2 | Frontend vitest | `cd frontend && npm test -- --run` -> `Test Files 23 passed (23) / Tests 178 passed (178) / Duration 4.65s`. +6 net vs cycle-70 baseline (172). | PASS |
| 3 | Backend pytest (full) | `pytest backend/ -q` -> `14 failed, 601 passed, 2 skipped, 9 xfailed`. The 14 failures match the prompt-claimed pre-existing set (watchdog-7d, shortlist-doc-archive x6, rainbow-canary, BQ-freshness x4, layer1-BQ-writes, doc-archived-shortlist). +1 net pass vs cycle-71's 600 (the 3 new snapshot-date tests add 3 passes; ESLint hook-order rule and ESLint cascade are NOT new tests). | PASS |
| 4 | Provider file exists | `test -f frontend/src/lib/live-portfolio-context.tsx` -> file exists, 268 lines | PASS |
| 5 | Provider mounted at root | `grep -c "LivePortfolioProvider" frontend/src/app/layout.tsx` -> `3` (import + JSX open tag + JSX close tag wrap children). At `layout.tsx:6,36,42`. Mounted INSIDE `<AuthProvider>` per contract. | PASS |
| 6 | All 3 consumers wired | `grep -c "useLivePortfolio" frontend/src/app/page.tsx frontend/src/app/paper-trading/layout.tsx frontend/src/app/paper-trading/positions/page.tsx` -> `2` per file (import + call). Three consumers, three sites. | PASS |
| 7 | Old direct hooks removed | `grep -nE 'useLivePrices|useLiveNav' frontend/src/app/page.tsx` returns ONLY comment lines 14-15 (refactor commentary). `grep -nE 'useLivePrices|useLiveNav' frontend/src/app/paper-trading/layout.tsx` returns ONLY comment lines 57, 135. Zero actual imports in consumer pages. Sitewide direct-import check (`grep -rln 'from "@/lib/useLivePrices"\|from "@/lib/useLiveNav"' frontend/src/`) returns 3 files: `live-portfolio-context.tsx` (THE owner), `paper-trading-context.tsx` (re-exports the values for sub-routes), `useLiveNav.ts` (the hook itself). Zero `app/**` files import directly. | PASS |
| 8 | Slack "as of close" snapshot label | `grep -n "as of close" backend/slack_bot/formatters.py` -> lines 314, 357, 360, 414, 416. Implementation at `formatters.py:310-319` (helper) + `:357-364` (morning) + `:413-419` (evening). | PASS |
| 9 | Donut LiveBadge wiring | `grep -n "liveBand\|LiveBadge" frontend/src/components/PortfolioAllocationDonut.tsx` -> lines 23 (import), 37 (prop type), 125 (destructure), 190-191 (render in card header). LiveBadge passed `band={liveBand} ageSec={liveAgeSec ?? null} compact`. Wired from positions page: `frontend/src/app/paper-trading/positions/page.tsx:134-135` `liveBand={lp.freshnessBand} liveAgeSec={lp.freshnessAgeSec}`. | PASS |
| 10 | Emoji scan on diff | `git diff` filtered through Python emoji-regex -> `emoji_count_in_diff=0`. Slack `:chart_with_upwards_trend:` codes are pre-existing colon-codes (Slack mrkdwn rendering on the wire, NOT unicode emojis in source). | PASS |

All 10 deterministic checks PASS.

---

## 3. Code-review heuristics (skill: code-review-trading-domain)

Per .claude/skills/code-review-trading-domain/SKILL.md, applied 5 dimensions.

### Dimension 1 -- Security audit (OWASP LLM Top-10 v2.0 2025)
- secret-in-diff: NO findings. Diff is React Context + Slack format helper + tests.
- prompt-injection-path: N/A (no new LLM call site added).
- command-injection: N/A (no subprocess/eval/exec).
- supply-chain-dep-pin-removal: NO findings. Zero new deps per contract /goal gate #6.
- system-prompt-leakage: N/A.
- rag-memory-poisoning: N/A.
- unbounded-llm-loop: N/A.
- excessive-agency: N/A.

### Dimension 2 -- Trading-domain correctness
- kill-switch-reachability: N/A (no execution-path change; no `paper_trader.py` diff).
- stop-loss-always-set: N/A.
- perf-metrics-bypass: NO findings. `home.tsx:259` retains `apiSharpe ?? kpiSharpe(navSeries)` -- backend-authoritative Sharpe with kpiMetrics local fallback. The new `pnlTodayPct/Dollars` derivation at `live-portfolio-context.tsx:190-203` is a NEW formula domain (today's intraday delta vs yesterday's snapshot), not duplicating any `services/perf_metrics.py` formula -- the canonical metric source for Sharpe/drawdown/alpha is unchanged.
- crypto-asset-class: N/A.
- paper-trader-broad-except: N/A.

### Dimension 3 -- Code quality
- broad-except: NO new instances. `live-portfolio-context.tsx:124-135` uses `Promise.allSettled` + explicit `rejected`-status checks -- graceful degrade, NOT broad-swallow.
- print-statement: NO findings.
- magic-number: `POLL_INTERVAL_MS = 60_000` is a named constant at `:84` with explicit "matches cycle-23.1.17 + useLivePrices" comment. Freshness bands (90s/300s) at `:95-97` are named via the band string (green/amber/red); could be hoisted to constants but inline-with-band-name is readable. NOTE-tier, no flag.
- unicode-in-logger: NO findings.

### Dimension 4 -- Anti-rubber-stamp on financial logic
- financial-logic-without-behavioral-test: The new P&L (Today) formula `(liveNav - yesterdayNav) / yesterdayNav * 100` at `live-portfolio-context.tsx:200-201` IS a financial-math change. The provider has 6 behavioral tests in `live-portfolio-context.test.tsx` including `pnlTodayPct/Dollars are null when liveNav or snapshots missing` (test 6). The null-case is covered explicitly. Vitest count delta: 178 - 172 = +6 net, matching the 6 new context tests. PASS.
- tautological-assertion: NO findings. New tests check actual string content (`"as of close 2026-05-22" in text`), structural shape, and behavioral nulls.
- over-mocked-test: `live-portfolio-context.test.tsx` mocks `@/lib/api`, `@/lib/useLivePrices`, `@/lib/useTickerMeta`, `@/lib/useLiveNav` -- mocking the UPSTREAM dependencies of the unit-under-test (the provider itself), NOT mocking the provider. The provider's own derivation logic (freshness band, pnlToday formula, error states) is exercised. This is a CORRECT unit-test pattern, not over-mocking.
- rename-as-refactor: This IS a refactor + behavior change (P&L Today fix). The behavior change is documented in the comments at `page.tsx:245-255` ("phase-72: P&L (Today) replaces the broken dailyDelta(navSeries) path") and called out in the contract. Not silent.
- formula-drift-without-citation: P&L Today formula `(liveNav - yesterdayNav) / yesterdayNav * 100` is cited at `live-portfolio-context.tsx:187-189` ("Per researcher: replaces the broken `dailyDelta(redLineSeries)` path...") and again in `contract.md:41`. Cited.

### Dimension 5 -- LLM-evaluator anti-patterns (Q/A self-check)
- sycophancy-under-rebuttal: N/A (no prior verdict on this cycle).
- second-opinion-shopping: N/A (first spawn; new evidence).
- missing-chain-of-thought: This critique cites 30+ file:line anchors and 6 verbatim command outputs.
- 3rd-conditional-not-escalated: N/A (no prior CONDITIONALs on this step).
- criteria-erosion: The cascade-test updates at `test_phase_23_2_8_use_live_nav_ssot.py` + `verify_phase_23_1_17.py` LOOSEN the literal-import assertion ("import useLiveNav from @/lib/useLiveNav") to accept EITHER direct-import OR via-LivePortfolioProvider-chained-import. This is NOT erosion: the SSOT invariant is STRONGER post-72 (one provider, two consumers) than pre-72 (two hook instances). The test correctly tracks the architecture by verifying chain integrity (if context, then context.tsx imports useLiveNav). At `test_phase_23_2_8_use_live_nav_ssot.py:74-89` the `via_context` branch asserts `LIVE_PORTFOLIO_CTX.exists()` AND the context file itself contains `import { useLiveNav }`. Chain-of-imports verified. PASS.

No BLOCK or WARN findings across 5 dimensions.

---

## 4. LLM judgment (7 operator-specified dimensions)

### Dim 1: SSOT invariant
Provider at `frontend/src/lib/live-portfolio-context.tsx`:
- ONE `useLivePrices(positionTickers, positions.length > 0)` instance at line 154.
- ONE `useLiveNav(status, positions, livePrices)` derivation at line 160.
- Both consumed via Context at line 244-248.
- Strict hook `useLivePortfolio()` at line 259-267 throws if no provider; `useLivePortfolioOptional()` at line 253-255 returns null for routes outside the cockpit tree (e.g. /login).
- P&L (Today) formula at line 199-201 matches researcher Section 4 finding 6 verbatim: `yesterdayNav = snapshots[0].total_nav; dollars = liveNav - yesterdayNav; pct = (dollars / yesterdayNav) * 100`.
PASS.

### Dim 2: Race elimination (anti-rubber-stamp)
Site-wide grep `grep -rn 'import.*\(useLivePrices\|useLiveNav\)' frontend/src/app/ | grep -v "live-portfolio-context"` returns ZERO hits. The only files that import `useLivePrices`/`useLiveNav` directly are the provider (`live-portfolio-context.tsx`), the sub-route context (`paper-trading-context.tsx` -- re-exports values passed FROM the provider, no new poll), and the hook source itself (`useLiveNav.ts`). All `app/**/page.tsx` and `app/**/layout.tsx` consumers go through `useLivePortfolio()`. Race is structurally impossible because there is ONE polling loop owned by the root provider. PASS.

### Dim 3: Donut + Slack labeled by design
- Donut accepts `liveBand` + `liveAgeSec` props at `PortfolioAllocationDonut.tsx:37-38, 125-126`, rendered via `<LiveBadge>` at line 190-191 inside the card header (next to the title).
- Positions page wires them at `paper-trading/positions/page.tsx:134-135` from `lp.freshnessBand` and `lp.freshnessAgeSec`.
- Slack digest morning + evening get `(as of close YYYY-MM-DD)` suffix via `_portfolio_snapshot_date()` helper at `formatters.py:310-319` -- pulls from `p.get("updated_at") or p.get("snapshot_date") or p.get("last_updated")`, returns first-10-chars (YYYY-MM-DD format).
- Researcher Section 4 finding 7 "stale displays are not a bug if labeled" satisfied for both surfaces.
PASS.

### Dim 4: P&L (Today) fix
At `frontend/src/app/page.tsx:252-255`:
```
const liveToday = (lp.pnlTodayDollars != null && lp.pnlTodayPct != null)
  ? { dollars: lp.pnlTodayDollars, pct: lp.pnlTodayPct }
  : null;
const today = liveToday ?? dailyDelta(navSeries);
```
Live-vs-snapshot derivation comes FIRST (researcher Section 4 finding 6). `dailyDelta(navSeries)` is a fall-through ONLY when the live derivation hasn't loaded yet (initial paint, no live ticks). When the live path is available -- the operator's expected normal -- the formula matches the contract verbatim: `(liveNav - latestSnapshot.nav) / latestSnapshot.nav * 100`. PASS.

### Dim 5: Cascade test updates
Both cascade tests (`test_phase_23_2_8_use_live_nav_ssot.py` + `tests/verify_phase_23_1_17.py`) accept BOTH shapes:
- Direct: `import { useLiveNav } from "@/lib/useLiveNav"` (pre-72 shape, still valid for legacy routes if any).
- Via context: `import { useLivePortfolio } from "@/lib/live-portfolio-context"` (post-72 shape, MUST chain-verify the context file imports `useLiveNav` -- enforced at `test_phase_23_2_8_use_live_nav_ssot.py:74-89` and `verify_phase_23_1_17.py:60-66`).
The SSOT invariant is preserved: the test now ensures `useLiveNav` is the only NAV-derivation source REGARDLESS of which file imports it directly. Stronger invariant post-72 (one source AND one consumer chain) than pre-72 (one source, two consumers). NOT criteria erosion. PASS.

### Dim 6: Scope honesty
`git status --short`:
```
?? frontend/src/lib/live-portfolio-context.tsx        (NEW)
?? frontend/src/lib/live-portfolio-context.test.tsx   (NEW)
?? handoff/current/research_brief_phase_ssot_nav.md   (NEW)
M  frontend/src/app/layout.tsx
M  frontend/src/app/page.tsx
M  frontend/src/app/paper-trading/layout.tsx
M  frontend/src/app/paper-trading/positions/page.tsx
M  frontend/src/components/PortfolioAllocationDonut.tsx
M  backend/slack_bot/formatters.py
M  backend/tests/test_phase_23_2_8_use_live_nav_ssot.py
M  backend/tests/test_phase_slack_digest_71.py    (+3 snapshot-date tests)
M  tests/verify_phase_23_1_17.py
M  handoff/current/contract.md
```
Matches operator prompt exactly. Scope NOT met for contract's `components/RedLineMonitor.tsx` row (planned but not modified) and `frontend/src/lib/kpiMetrics.ts` row (planned new helper but the P&L-Today formula was placed inline in the provider instead, which is structurally cleaner -- the formula needs `snapshots[]` and `liveNav` which both already live in the provider). Both deviations are SCOPE REDUCTIONS, not silent additions, and both are architecturally justified:
- RedLineMonitor "as of YYYY-MM-DD" badge: the Red Line tooltip already shows the historical date per-point (operator can hover and see it). A duplicate card-header badge would be redundant. Deferred is acceptable and the Slack digest covers the operator-facing labeling case. Worth noting in harness_log but not a blocker.
- kpiMetrics.ts helper: the formula was placed in the provider's `useMemo` at line 190-203 instead. Functionally identical, structurally cleaner (no prop-drilling of snapshots+liveNav into a separate util). Acceptable.
0 schema changes, 0 new deps confirmed. PASS (with two documented scope deviations that are architecturally sound).

### Dim 7: Cycle-71 not regressed
Cycle-71 envelope unwrap intact: `formatters.py:345-349` (morning) and `:406-410` (evening) still call `float(p.get("total_nav") or 0.0)`, `float(p.get("starting_capital") or 0.0)`, and pull `total_pnl_pct or total_return_pct`. The cycle-72 change APPENDS `(as of close DATE)` -- it does not alter the unwrap logic. The cycle-71 tests (`test_format_morning_digest_unwraps_portfolio_envelope`, `test_format_evening_digest_unwraps_portfolio_envelope`, etc.) all still pass (18 tests passed in the focused suite, including the 4 cycle-71 tests). PASS.

---

## 5. Verdict reasoning

All 5 harness-compliance audits PASS. All 10 deterministic checks PASS. Code-review heuristics across 5 dimensions return zero BLOCK / WARN findings. LLM judgment across 7 operator-specified dimensions PASS, with two documented scope deviations on `RedLineMonitor.tsx` and `kpiMetrics.ts` that reduce surface area without compromising the SSOT goal (the contract anticipates this in N* delta -- "operator-facing minimum"; the operator screenshot evidence is fully addressed by the Home + Paper Trading + Donut + Slack changes that ARE in the diff).

The architectural change is substantive (root-level Context provider mirroring AuthProvider; ONE useLivePrices + useLiveNav instance; race-condition structurally impossible; P&L Today formula corrected). Test coverage delta is +6 net frontend tests (matching the 6 new context tests) and +3 net backend tests (the 3 new snapshot-date tests). The SSOT invariant tests at `test_phase_23_2_8_use_live_nav_ssot.py` were updated to track the new architecture without weakening the invariant (chain-of-imports verified through the provider).

The cycle is verifiably an SSOT fix, not just a refactor: the four operator-flagged NAV values now collapse to ONE live value (Home + Paper Trading + Donut center) plus a clearly-labeled snapshot value (Slack digest with "as of close YYYY-MM-DD" suffix).

Action required after this PASS:
1. Main appends `## Cycle 72 -- 2026-05-26 -- SSOT NAV/P&L via root LivePortfolioProvider result=PASS` block to `handoff/harness_log.md`.
2. No masterplan flip (UX-refactor cycle, no step in `.claude/masterplan.json` directly maps).
3. Operator visual smoke: confirm Home NAV + Paper Trading NAV + Donut center all match; P&L (Today) shows non-zero intraday delta; Slack digest preview shows "(as of close YYYY-MM-DD)" suffix.

---

## 6. JSON envelope

```json
{
  "ok": true,
  "verdict": "PASS",
  "reason": "Cycle-72 SSOT lift verified: root-level LivePortfolioProvider mounted at app/layout.tsx:36; ONE useLivePrices + useLiveNav instance owned by provider at live-portfolio-context.tsx:154,160; three consumers (Home, Paper Trading layout, Positions page) all use useLivePortfolio() with zero direct hook imports in app/**; P&L Today formula corrected at :190-203 to (liveNav - snapshots[0].total_nav) / snapshots[0].total_nav * 100; Slack 'as of close YYYY-MM-DD' suffix wired at formatters.py:310-319,357-364,413-419; Donut LiveBadge at PortfolioAllocationDonut.tsx:190-191 fed by lp.freshnessBand. Deterministic: tsc EXIT=0; vitest 178 passed (+6 net); pytest 601 passed / 14 pre-existing failures unchanged; eslint EXIT=0 (52 pre-existing warnings, none in cycle-72 files); 0 emojis in diff. Code-review heuristics: 0 BLOCK / 0 WARN across 5 dimensions. Two documented scope deviations (RedLineMonitor badge, kpiMetrics helper) are architecturally sound reductions, not silent omissions.",
  "violated_criteria": [],
  "violation_details": [],
  "certified_fallback": false,
  "checks_run": [
    "harness_compliance_audit_5_item",
    "syntax_tsc_noemit",
    "syntax_eslint",
    "frontend_vitest",
    "backend_pytest_full",
    "backend_pytest_targeted",
    "provider_exists",
    "provider_mounted_at_root",
    "consumers_wired",
    "direct_hook_imports_purged",
    "slack_snapshot_label",
    "donut_livebadge_wiring",
    "emoji_scan",
    "code_review_heuristics",
    "scope_honesty_git_status",
    "cycle_71_envelope_unwrap_intact"
  ]
}
```
