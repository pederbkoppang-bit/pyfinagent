---
step: phase-24.1
cycle: 2
cycle_date: 2026-05-12
result: PASS_PENDING_QA
verification_command: 'source .venv/bin/activate && python3 tests/verify_phase_24_1.py'
title: Trading-execution + governance audit (stop-loss orphan, missing stops, zero sells, sector caps, position limits)
---

# Experiment Results — phase-24.1 — Trading-Execution + Governance

**Cycle:** phase-24 cycle 2
**Date:** 2026-05-12
**Step ID:** 24.1
**Priority:** P0
**Action:** Phase-24 is READ-ONLY by charter. Produced one findings doc + research brief + contract. No backend/frontend/scripts/.claude code changes.

---

## Artifacts produced

| File | Bytes (~) | Purpose |
|---|---|---|
| `handoff/current/research_brief.md` | 17 KB | Researcher subagent brief; gate_passed=true with 5 sources read in full |
| `handoff/current/contract.md` | 5 KB | Sprint contract for phase-24.1 (this cycle) |
| `docs/audits/phase-24-2026-05-12/24.1-execution-trading-findings.md` | 22 KB | Findings doc — 6 phase-25 candidates, 11-position stop-status table, governance gap |

**No backend/, frontend/, .claude/, or scripts/ files were modified.** Phase-24 is read-only by charter.

---

## Verbatim verifier output

Command: `source .venv/bin/activate && python3 tests/verify_phase_24_1.py`

```
=== phase-24.1 (execution-trading) verifier ===
  [PASS] findings_md_exists_at_docs_audits_phase_24_2026_05_12_24_1_execution_trading_findings_md
  [PASS] research_gate_envelope_present_with_gate_passed_true
  [PASS] external_sources_count_at_least_5
  [PASS] canonical_url_cited_verbatim_paper_trader_py
  [PASS] recency_scan_2024_2026_section_present
  [PASS] at_least_three_phase_25_candidate_steps_proposed
  [PASS] each_candidate_step_has_files_list_with_absolute_paths
  [PASS] each_candidate_step_has_draft_verification_command
  [FAIL] harness_log_has_phase_24_24_1_cycle_entry
         -> harness_log.md must contain `## Cycle N -- ... phase=24.1 result=...` header
  [PASS] executive_summary_section_present
  [PASS] findings_cites_paper_trader_py_414_423_orphan_with_grep_evidence
  [PASS] findings_tags_all_11_current_portfolio_positions_by_stop_presence
  [PASS] findings_documents_ter_minus_12_30_no_sell_case
  [PASS] findings_audits_governance_limits_loader_watcher
FAIL (13/14) EXIT=1
```

**Interpretation:** 13/14 PASS. The single FAIL is `harness_log_has_phase_24_24_1_cycle_entry`, which is expected per the log-last protocol (CLAUDE.md::Critical Rules + scripts/audit/phase_24_audit_prompt.md:217-218). After the Q/A PASS and the log append, the verifier returns 14/14.

---

## Live-check evidence

Command: `ls docs/audits/phase-24-2026-05-12/24.1-execution-trading-findings.md && head -30 docs/audits/phase-24-2026-05-12/24.1-execution-trading-findings.md`

To be captured in `handoff/current/live_check_24.1.md` before auto-push fires.

---

## Hypothesis verdict

**CONFIRMED.** All four sub-hypotheses validated:

1. **`check_stop_losses()` is orphan code** — verbatim `grep -rn "check_stop_losses" backend/ scripts/ tests/` shows only the definition at `paper_trader.py:414` + 4 audit-prompt mentions. Zero production callers.
2. **6 positions have `stop_loss_price=None`** — ON, INTC, TER, DELL, GLW, CIEN (~15 days old, pre-phase-23.1.8 fallback).
3. **TER -12.30% no-sell case** — `portfolio_manager.py:82` short-circuits on None via `if stop and current ...` truthiness; `check_stop_losses` is orphan; kill-switch is NAV-level only.
4. **Governance gap** — `limits.yaml:28` defines `max_sector_weight_pct: 0.30` and `max_position_notional_pct: 0.05` (both immutable, watcher works), but `decide_trades()` and `execute_buy()` never consult them. The trade path uses an unrelated `paper_max_per_sector` count cap defaulted to 0.

## Phase-25 candidates emitted (6)

1. **phase-25.1 (P0)** — Wire `check_stop_losses()` into the daily loop with auto-sell
2. **phase-25.2 (P0)** — Backfill missing stops with same-cycle re-check
3. **phase-25.3 (P1)** — "No-sells-in-N-days" anomaly watchdog
4. **phase-25.4 (P1)** — Connect `limits.yaml:max_sector_weight_pct` to `decide_trades()`
5. **phase-25.5 (P1)** — Enforce `max_position_notional_pct` in `execute_buy()`
6. **phase-25.6 (P0)** — "No-stop-on-entry" hard block in `execute_buy()`

---

## Next phase

EVALUATE phase. Q/A subagent will:
1. Run 5-item harness-compliance audit
2. Re-run `python3 tests/verify_phase_24_1.py` independently
3. LLM-judgment legs (contract alignment, mutation-resistance, anti-rubber-stamp, scope honesty, research-gate compliance)
4. Return verdict envelope

On Q/A PASS:
- Append `## Cycle 43 -- 2026-05-12 -- phase=24.1 result=PASS` to `handoff/harness_log.md`
- Write `handoff/current/live_check_24.1.md` with head-30 evidence
- Flip masterplan step 24.1 to done
- Auto-commit-and-push hook fires
