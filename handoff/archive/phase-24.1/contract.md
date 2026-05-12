# Sprint Contract — phase-24.1 — Trading-Execution + Governance Audit

**Cycle:** phase-24 cycle 2
**Date:** 2026-05-12
**Step ID:** 24.1
**Step name:** Trading-execution + governance audit (stop-loss orphan, missing-stops-on-entry, zero-sells-ever, sector caps, position limits)
**Priority:** P0
**Depends on:** 24.0 (charter — DONE 2026-05-12 cycle 42)
**Harness required:** true
**Audit basis:** `backend/services/paper_trader.py:414-423` orphan `check_stop_losses`; TER -12.30% no-stop evidence; `backend/governance/`; operator bug report 2026-05-12

---

## Research-gate summary

Researcher subagent ran 2026-05-12. **`gate_passed: true`** with envelope:

```json
{
  "tier": "complex",
  "external_sources_read_in_full": 5,
  "snippet_only_sources": 11,
  "urls_collected": 16,
  "recency_scan_performed": true,
  "internal_files_inspected": 10,
  "gate_passed": true
}
```

Five sources fetched in full: Anthropic harness-design, arxiv 2604.27150 (stop-loss parameterization, April 2026), Frontiers 2024 (disposition-effect debiasing), Alpaca order state docs, Semnet agentic-governance 2026. Three-variant search-query discipline applied across 4 topics (stop-loss design, disposition effect, broker order lifecycle, AI trading safety). Recency scan surfaced 3 new findings; none supersede canonical Anthropic patterns. Full brief at `handoff/current/research_brief.md`.

---

## Hypothesis

`check_stop_losses()` at `paper_trader.py:414-423` is orphan code. The daily-loop scheduler does not call it. The entry path does not write `stop_loss` on every new position. Governance limits-loader watcher does not block entries on sector-cap breach.

**Researcher verdict: CONFIRMED.** All four sub-hypotheses validated:
1. `check_stop_losses()` defined once (`paper_trader.py:414`), zero production callers (verbatim grep evidence in brief)
2. 6 positions (ON, INTC, TER, DELL, GLW, CIEN) have `stop_loss_price=None`
3. TER at -12.30% never sold; no caller for `check_stop_losses()` + portfolio_manager.py:82 silently bypasses None stops via `if stop and current...` short-circuit
4. Governance gap: `limits.yaml:max_sector_weight_pct: 0.30` defined but never consulted by `decide_trades()` (which uses unrelated `paper_max_per_sector` count-cap defaulted to 0)

---

## Success criteria (immutable — copied verbatim from `.claude/masterplan.json` step 24.1 `verification.success_criteria`)

1. `findings_md_exists_at_docs_audits_phase_24_2026_05_12_24_1_execution_trading_findings_md`
2. `research_gate_envelope_present_with_gate_passed_true`
3. `external_sources_count_at_least_5`
4. `canonical_url_cited_verbatim_paper_trader_py`
5. `recency_scan_2024_2026_section_present`
6. `at_least_three_phase_25_candidate_steps_proposed`
7. `each_candidate_step_has_files_list_with_absolute_paths`
8. `each_candidate_step_has_draft_verification_command`
9. `harness_log_has_phase_24_24_1_cycle_entry`
10. `executive_summary_section_present`
11. `findings_cites_paper_trader_py_414_423_orphan_with_grep_evidence`
12. `findings_tags_all_11_current_portfolio_positions_by_stop_presence`
13. `findings_documents_ter_minus_12_30_no_sell_case`
14. `findings_audits_governance_limits_loader_watcher`

**Verifier command:** `source .venv/bin/activate && python3 tests/verify_phase_24_1.py`
**Live check:** `ls docs/audits/phase-24-2026-05-12/24.1-execution-trading-findings.md && head -30 docs/audits/phase-24-2026-05-12/24.1-execution-trading-findings.md`

---

## Plan steps

1. **Write findings doc** at `docs/audits/phase-24-2026-05-12/24.1-execution-trading-findings.md` with:
   - YAML frontmatter (bucket, slug, cycle, cycle_date, researcher_gate)
   - Executive summary (1 paragraph TL;DR — orphan stop, 6 positions no-stop, TER -12.30% no-sell, governance gap)
   - Code-grounded findings — `paper_trader.py:414-423` orphan + verbatim grep; portfolio_manager.py:82-88 silent-bypass; governance gap with file:line; 11-position stop-status table
   - External-research summary — 5 verbatim URLs (Anthropic harness, arxiv 2604.27150, Frontiers 2024, Alpaca, Semnet)
   - Recency scan (2024-2026) section — 3 findings
   - **Coverage matrix** (optional but useful)
   - Proposed phase-25 candidates (≥5) with `Files:` blocks + absolute paths + draft verification commands + priority
   - Open questions
   - References (all 16 URLs collected)
2. **Write `experiment_results.md`** at `handoff/current/experiment_results.md` with verbatim verifier output.
3. **Spawn Q/A subagent** with 5-item harness-compliance audit first.
4. **On Q/A PASS**, append harness_log.md with `## Cycle 43 -- 2026-05-12 -- phase=24.1 result=PASS` block.
5. **Write live_check evidence** at `handoff/current/live_check_24.1.md` (ls + head -30 of findings doc).
6. **Flip masterplan status to done** — auto-commit-and-push fires.

**Phase-24 is READ-ONLY**: no code changes. The 5+ phase-25 candidates are the deliverable.

---

## References

External (read in full):
- https://www.anthropic.com/engineering/harness-design-long-running-apps
- https://arxiv.org/html/2604.27150
- https://www.frontiersin.org/journals/behavioral-economics/articles/10.3389/frbhe.2024.1345875/full
- https://docs.alpaca.markets/docs/orders-at-alpaca
- https://www.semnet.co/post/agentic-ai-governance-in-2026-preventing-data-leaks-and-cves

Internal anchors:
- `backend/services/paper_trader.py:414-423` (orphan check_stop_losses)
- `backend/services/autonomous_loop.py:314` (kill_switch wired; no stop-loss step)
- `backend/services/portfolio_manager.py:82-88` (silent None-stop bypass)
- `backend/services/portfolio_manager.py:194` (max_per_sector count-cap, default 0)
- `backend/services/portfolio_manager.py:288-329` (`_extract_stop_loss` only for NEW buys)
- `backend/governance/limits.yaml:28` (max_sector_weight_pct: 0.30 — never read)
- `backend/governance/limits_loader.py` (watcher kills process on file change but limits unused in trade path)
- `backend/services/kill_switch.py` (NAV-level; works)
- `backend/api/paper_trading.py` (/portfolio surfaces stop_loss_price)
