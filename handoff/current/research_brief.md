# Research Brief — phase-57.1 (Binding RiskJudge gate + concentration-aware prompt context)

**Tier:** complex. **Date:** 2026-06-11. **Step:** 57.1 (phase-57 FEATURE; operator reply `PHASE-57: FEATURE` 2026-06-11).
**Spec-of-record:** `handoff/archive/phase-55.3/55.3-synthesis-checkpoint.md` §2.6 FEATURE paragraph.
**Cites:** F-3 (RiskJudge REJECT advisory-only — portfolio_manager.py:185 records, :194-198 log-only; 3 away-week REJECTs executed incl. DELL 06-03), F-8 (lite RiskJudge system prompt phantom "10% cap" vs config paper_max_per_sector_nav_pct=30.0; rationales say "no portfolio sector breakdown was provided").

> STATUS: IN PROGRESS — writing incrementally.

---

## A. Internal fix-design (file:line)

### A0. Gate-site topology (decide_trades flow, portfolio_manager.py)

`decide_trades(current_positions, candidate_analyses, holding_analyses, portfolio_state, settings, candidates_by_ticker)` returns `list[TradeOrder]`, sells appended first, buys second, final stable-sort `SELL`-before-`BUY` at :398.

Two loops matter:
- **Candidate-build loop :148-198** — iterates `candidate_analyses`, skips held/non-BUY-rec, extracts `risk_assessment = analysis.get("risk_assessment", {})`, computes position_pct/stop_loss, appends a dict to `buy_candidates` **including `"risk_judge_decision": risk_assessment.get("decision", "")` at :185**. The advisory log-only branch is :193-198 (`if decision and decision != "APPROVE_FULL": logger.info(...)`). THIS is F-3's exact site.
- **BUY-emit loop :256-365** — sorts `buy_candidates` by final_score desc (:201), then for each candidate runs (in order): position-cap break (:257), available-cash break (:269), sector-COUNT cap -> `sector_blocked.append` + continue (:276-286), position-size + $50-floor (:288-299), sector-NAV-pct cap continue (:301-318), FF3 factor-corr cap continue (:320-334), then `orders.append(TradeOrder(... risk_judge_decision=cand["risk_judge_decision"] ...))` at :336-356, decrement available_cash + increment sector trackers (:357-365).

Every BUY that ever executes flows through the `orders.append(TradeOrder(...))` at :336. The `risk_judge_decision` string is already carried on every candidate dict (:185) and on every emitted TradeOrder (:342). So the binding gate has TWO clean placement options (recommended option in A1).

_(more below)_

---

## B. External research

_(populated below)_

---

## Draft immutable success criteria

_(populated below)_

---

## JSON envelope

_(emitted at end)_
