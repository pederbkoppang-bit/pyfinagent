# Active Goal — Best-in-class elevation + remote-working go-live (Opus 4.8)

Set by operator 2026-06-01. Scope: **FULL best-in-class push**. Supersedes the 2026-05-28 goal
(spent: the zero-trades diagnosis is resolved; priorities 1/3/4/5/7 shipped; 47.2 parked on the
operator LLM-spend gate).

## North star
(verbatim `masterplan.json::goal`) "Ship an Intelligence Engine trading system that maximizes
Net System Alpha = Profit − (Risk Exposure + Compute Burn) by dynamically shifting capital to the
highest-earning strategy, recursively self-improving under hard risk caps, within a 15-slot daily
Claude-routine budget." **AND** make pyfinAgent best-in-class on every axis it touches — quant
correctness, UX, data-stack, algorithms — with every change gated by the deep-research discipline.

## Scope — the next natural steps, in order
1. **phase-50.6** — Multi-market UI. Close via the harness. Cycle-32 (`goal-multimarket-ux`)
   shipped only the paper-trading page; 50.6 still needs the **backtest-page** treatment + a
   **multi-currency NAV-breakdown widget** + a **`paper_markets` settings toggle** + `live_check_50.6.md`.
2. **phase-43.0** — Definition-of-Done audit (14-criterion production-ready gate). Run it; close
   what is autonomously closable; honestly mark live-blocked criteria.
3. **phase-53** — Best-in-class elevation + remote-working go-live (NEW, 2026-06-01):
   - 53.1 Algorithm/quant elevation (portfolio construction / execution-cost / overfitting-control; measure-and-gate; do-no-harm)
   - 53.2 UX elevation (ONE consistent layout/design/animation across all pages + WCAG AA)
   - 53.3 Data-stack elevation (BQ cost/perf + partition/cluster + freshness/lineage)
   - 53.4 Remote-working SessionStart hook (web bootstrap; path-portable; fail-open) + `claude.yml` pin 4-7→4-8
   - 53.5 **E2E smoke capstone** (CI `e2e-smoke.yml` + `aggregate.sh` green) ← CLOSES the goal

## Opus-4.8 workflow policy (speed without quality loss)
- **Parallelism WITHIN the harness only**: parallel WebFetch sub-queries inside the single
  researcher spawn; dynamic-workflow / parallel fan-out of INDEPENDENT file edits; parallel
  smoke-test execution; adaptive thinking; mid-conversation system messages to re-assert the gates.
- **Effort pinned**: Main `xhigh`, Researcher + Q/A `max`. Fast-mode OK for mechanical loops, NOT
  for research synthesis or Q/A judgment.
- **FORBIDDEN**: fanning out the Plan→Generate→Evaluate gate; >1 Q/A or an evaluator panel;
  lowering Researcher/Q/A effort; re-splitting Researcher/Q/A. One masterplan flip per step.

## Founding principle — deep-research gate (non-negotiable)
Every step: `researcher` FIRST — ≥5 sources read in full (WebFetch), 10+ URLs, 3-pass search
(current-year / last-2-year / year-less), mandatory recency scan, source-quality hierarchy, JSON
envelope `gate_passed: true` — then `contract.md` → GENERATE → fresh `qa` (no self-eval) →
`harness_log.md` append → masterplan flip. CONDITIONAL/FAIL → fix + update files + FRESH `qa`
(no verdict-shopping). The elevation themes are research TARGETS, not foregone conclusions; the
gate decides what ships.

## Done-definition (HARD STOP)
- 50.6 four criteria met (incl. operator visual `live_check`) + 43.0 audit run + phase-53 closed.
- **Remote working ACTIVE**: the SessionStart hook fires on a fresh session and injects the current
  step; `e2e-smoke.yml` present; `bash scripts/smoketest/aggregate.sh` GREEN (exit 0) +
  `python scripts/harness/run_harness.py --dry-run --cycles 1` appends `harness_log.md`.
- Write `handoff/current/cycle_block_summary.md` and stop.

## Constraints / gates
- **Branch**: develop on `claude/upbeat-euler-Y3evq` (this remote session). Local runs are
  main-based — merge the branch before resuming locally.
- **OPERATOR-GATED**: LLM API spend (live cycles/trades), pip installs, BQ DROP / unqualified
  DELETE. Otherwise full approval to proceed + commit per step.
- **DO-NO-HARM**: the working US pure-quant momentum core (+20% NAV) must not regress — every
  alpha/construction change is config-gated + default-off + measured ON-vs-OFF before any live flip.
- True LIVE-trade end-to-end needs Monday EU/KR data + LLM spend (operator-gated); offline/dry-run
  smoke covers the rest now.

## Stop conditions
- SOFT STOP: 12 cycles elapsed OR a blocker needing the operator → write a summary + a crisp ask.
- Per-step operator go for visual `live_check`s (NextAuth wall) + any operator-gated cost/destructive action.

## Cycle ledger (this run)
- (appended per cycle)
