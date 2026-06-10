# Live-check evidence — phase-24.1 — Trading-Execution + Governance Audit

**Step:** 24.1 — Trading-execution + governance audit (P0)
**Date:** 2026-05-12
**Live-check field:** `ls docs/audits/phase-24-2026-05-12/24.1-execution-trading-findings.md && head -30 docs/audits/phase-24-2026-05-12/24.1-execution-trading-findings.md`

---

## Verbatim command output

```
$ ls docs/audits/phase-24-2026-05-12/24.1-execution-trading-findings.md
docs/audits/phase-24-2026-05-12/24.1-execution-trading-findings.md

$ head -30 docs/audits/phase-24-2026-05-12/24.1-execution-trading-findings.md
---
bucket: 24.1
slug: execution-trading
cycle: 2
cycle_date: 2026-05-12
researcher_gate: {"tier": "complex", "external_sources_read_in_full": 5, "snippet_only_sources": 11, "urls_collected": 16, "recency_scan_performed": true, "internal_files_inspected": 10, "gate_passed": true}
---

# Findings — phase-24.1 — Trading-Execution + Governance Audit

## Executive summary

The most user-impactful live bug in pyfinagent is confirmed: `check_stop_losses()` at `backend/services/paper_trader.py:414-423` is orphan code — defined once, called from zero production paths. The autonomous daily loop wires `check_and_enforce_kill_switch()` at `autonomous_loop.py:314` (NAV-level protection works) but has no equivalent per-position stop-enforcement step. As a result 6 of 11 current positions (ON, INTC, TER, DELL, GLW, CIEN) carry `stop_loss_price=None`, and TER has accumulated -12.30% unrealized P&L with zero sell action because `portfolio_manager.py:82-88` silently bypasses the stop check when `stop_loss_price` is None (Python truthiness on `if stop and current ...`). A second-order governance gap compounds the problem: `backend/governance/limits.yaml:28` defines `max_sector_weight_pct: 0.30` (immutable) but the value is never consulted by `decide_trades()`, which instead uses an unrelated count-based cap `paper_max_per_sector` defaulting to 0 (disabled). Five P0/P1 phase-25 candidates close the gap: wire stop-enforcement step, backfill missing stops with same-cycle re-check, no-sells-watchdog, sector-weight enforcement from immutable limits, and notional-cap enforcement in `execute_buy()`. A sixth P0 candidate (no-stop-on-entry hard block) prevents future regressions.

## Code-grounded findings

### F-1: `check_stop_losses()` is orphan code (paper_trader.py:414-423)

Verbatim grep:
```

The findings doc exists at the expected path; the head-30 output shows the YAML frontmatter (with `gate_passed: true`) plus the full executive summary documenting the orphan stop-loss + TER -12.30% + governance gap. This satisfies the live_check requirement that an operator can independently audit the artifact on disk.

**Audit anchor for next bucket:** 24.4 (P0 — agent topology + per-agent rationale flow — Trader/RiskJudge byte-identical aliasing).
