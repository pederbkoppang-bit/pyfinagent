# Goal — phase-70: Make the fund changeable, diversified, and honestly traded

_Operator prompt 2026-07-13 (verbatim intent): "there are still several bugs. our app
hasn't bought any new stock from other sectors. I'm not able to change setting max
position per sector. we need as many trades as possible to make our program even better.
use ultracode, find as many bugs as possible, test them with Playwright, then make new
steps in the masterplan to fix all of them. audit first, then fix."_

This goal converts the 2026-07-13 ultracode audit (31 agents, 6 dimensions,
adversarial-verify; **17 confirmed / 8 refuted**; register
`handoff/current/confirmed_findings.json` + live evidence
`handoff/current/audit_2026-07-13_live_evidence.md`) into fixes for the three
operator-reported symptoms, without regressing the working +20%-NAV / +14%-alpha engine.

## What the audit actually found (root causes, not guesses)

**S1 — "I can't change MAX POSITIONS PER SECTOR."** The save path works end-to-end on the
running app (Playwright: edit → `PUT /api/settings/ 200` → persists → survives reload →
scheduled loop reads fresh). The *literal* blocker is a **frontend controlled-input bug**
(`cockpit-helpers.tsx:467`), **confirmed live**: select-all + Backspace does NOT clear the
field — it snaps back to the stored value; typing then *appends* (2 → type "5" → "25"),
which exceeds the max (20) and fails Save with a generic 422. Compounding it: a
`risk_overrides` shadow (`portfolio_manager.py:304`) can silently override the UI cap with
**no in-app way to see or clear it**, and the 30%/sector NAV cap + other active knobs have
**no settings-UI editor at all**.

**S2 — "no new stock from other sectors."** NOT a universe-size problem — the universe is
S&P 500 + international. It is a **monosector funnel**: ranking + top-N truncation
(`analyze top-5`, `autonomous_loop.py:838`) collapses candidates to the momentum-leading
sector (semiconductors), so only semis ever become BUY candidates; the per-sector cap then
blocks the overflow into a swap path that **only rotates within the same sector**
(`portfolio_manager.py:620/675`). Live proof: 1 position (AMD/Tech), ~97% cash, 61 lifetime
trades across only **20 unique tickers** — all semis/tech. A `sector_neutral` lever exists
but a 2026-06-01 replay found **HARD sector-neutral hurts long-only returns**, so the fix
must be *soft*, profit-aware diversification, not a hard switch.

**S3 — "as many trades as possible."** Throughput is *starved*, not busy: capital barely
deployed (~97% cash) because a stack of **silent BUY-gates** fire with no operator
visibility — a hidden $1 per-cycle session budget that is *half* the visible $2 cap
(`autonomous_loop.py:90`), a 5% price-tolerance gate (`paper_trader.py:182`), and a lite
analyzer that defaults to HOLD on parse failure (`autonomous_loop.py:2250`). Plus two
money-path bugs that make the book *shrink*: a non-atomic swap that can execute the SELL and
silently drop the BUY (`portfolio_manager.py:675`, explains the 2→1 position drop) and
avg-entry-price corruption on non-US add-ons (`paper_trader.py:308`).

"More trades" is reframed to the north star: **more diversified, higher-quality capital
deployment and more clean round-trips for the learning loop** — NOT more churn of the same
20 names (which also fights the phase-61 churn-integrity work).

## Definition of done

Every one of the 17 confirmed findings is fixed or explicitly dispositioned across the
phase-70 steps, with the three operator symptoms verifiably closed:

1. **S1 closed:** an operator can clear a numeric settings field and type any in-range value
   that saves and takes effect; out-of-range input is blocked/clamped client-side with a
   specific message; an active `risk_overrides` shadow is visible and clearable from the
   Manage tab; every active BUY-gating knob (incl. the 30%/sector NAV cap) is editable in the
   UI. Proven by a live Playwright capture of the clear-then-type flow.
2. **S2 closed:** a normal cycle can produce BUY candidates from ≥2 distinct GICS sectors and
   the book can hold positions in ≥2 sectors, via SOFT diversification (respecting the
   2026-06-01 hard-neutral-hurts finding) — flag-gated, backtest-validated, DARK-until-token.
   Enrichment failure no longer collapses everything into one "Unknown" bucket.
3. **S3 closed:** the silent BUY-gates are logged + surfaced + tunable and reconciled with the
   operator-visible caps; the non-atomic swap and non-US avg-price bugs are fixed fail-safe so
   the book stops silently shrinking. Trade throughput measurably rises on paper WITHOUT a
   rise in same-name churn (round-trip diversity up, not turnover-for-its-own-sake).

## Boundaries (binding)

- **$0 metered, free APIs only, paper-only.** No live-money surface. LLM cost changes need
  Peder's approval.
- **Do-no-harm:** the working engine's guardrails stay byte-untouched unless a step's whole
  point is to fix one — kill-switch limits, stop-losses, sector caps as *risk* limits,
  DSR≥0.95 / PBO≤0.5 promotion gates. Behavior changes to the live loop ship **flag-gated,
  default-OFF (DARK-until-token)** with an ON-vs-OFF $0 diff in the live_check.
- **Hysteresis stays banned; `historical_macro` stays frozen** (no optimizer runs that need
  it) unless the operator records the un-freeze token.
- **Full 5-file harness protocol per step** (research gate → contract → experiment_results →
  Q/A critique → harness_log); **harness stays exactly 3 agents** (Main + Researcher + Q/A);
  Q/A runs the live Playwright gate on every UI-touching step; no self-evaluation.
- **Diversification must not be bought with returns:** any S2/S3 change that could lower
  risk-adjusted OOS P&L must clear a paper/backtest check before its activation token.

## Refuted / dismissed (checked, not fixed — for auditor trust)

8 candidate findings were adversarially refuted (e.g. "settings PUT never reloads" — false,
`get_settings.cache_clear()` runs; "scheduler holds stale caps" — false, `_scheduled_run`
reads fresh; the os.environ-over-.env path — real only if the backend was started via
`set -a; . .env`, not the case on the running instance). See `confirmed_findings.json`
`refuted[]` for the full list + reasons.

## Step map (all 17 findings covered)

- **70.0** — Research gate + design pack (soft-diversification algebra that respects the
  2026-06-01 hard-neutral replay; atomic-swap + cross-sector-rotation design; gate-visibility
  design). Opens the phase; offline, $0.
- **70.1 (P1, S1)** — Make the setting changeable: fix clear-snapback (#4, live-confirmed) +
  out-of-range client validation (#13); surface & clear `risk_overrides` shadow in the UI +
  `api.ts` wiring (#12); add editors for the NAV%/sector cap and other active knobs (#11,#16).
- **70.2 (P1, S2)** — Soft, profit-aware cross-sector diversification of the analyzed top-N
  (#1,#2) + robust sector attribution so enrichment failure can't freeze the book in one
  "Unknown" bucket (#5,#14). Flag-gated, backtest-validated, DARK-until-token.
- **70.3 (P1/P2, S3 + money-path)** — Swap/rotation correctness: cross-sector-capable, atomic,
  cash-bounded, $50-floor-aware swap (#3,#9); fix non-US avg-entry-price unit mix (#10).
- **70.4 (P2, S3)** — Un-gate throughput by making silent BUY-blockers visible + tunable and
  reconciling the hidden $1 session budget with the visible $2 cap (#6,#7,#8).
- **70.5 (P3, general)** — Settings-surface completeness + observability: deposits reflected in
  Starting-capital display (#17); reschedule APScheduler on `paper_trading_hour` save (#15).
