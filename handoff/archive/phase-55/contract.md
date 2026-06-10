# Contract — Step 55.3

**Step id:** 55.3 — Synthesis + operator checkpoint
**Date:** 2026-06-10
**Phase:** phase-55 (review-only; $0; NO fixes; NO LLM trading-cycle spend)
**Researcher gate:** PASSED — `handoff/current/research_brief_55.3.md` (tier=complex, 7 external sources read in full incl. the three pre-anchored papers + an adversarial 2026 source, 17 URLs, recency scan; envelope `gate_passed: true`)

## Research-gate summary

The brief IS the strategic chapter's evidence base. Internal: BQ-measured burn (lite $0.05-0.17/cycle ≈ $0.01/ticker CONFIRMED; full $1.08-4.06/cycle ≈ $0.19-0.27/ticker effective — the UI's "$0.50-2.00/ticker" at `manage/page.tsx:230` is overstated 2-10x; `settings.py:316`'s "$1-3/day" is authoritative); the 53.1 binding precedent mechanics (`analytics.py:239 sharpe_diff_test`, promote rule `p<0.05 AND delta>=+0.05 AND ci_low>0` + gross do-no-harm leg; rejected mechanism = rank-hysteresis band at `rebalance_band.py:22`); MinTRL horizon menu (Bailey-LdP exact form: ~539 dailies ≈ 2.1y at backtest Sharpe 1.17; ~2,820 ≈ 11y at SR 0.5; a 1-2wk window is 2 orders of magnitude short → the live window is a sanity/stress gate, NOT a skill proof); the finetune-vs-features finding map (churn is the only lever-shaped finding and its obvious lever family already failed 53.1; the dominant cluster is reliability/correctness-shaped). External (7 in full): FINSABER arXiv:2505.07078v5 (short-window LLM wins vanish; min-holding/cooldown named as THE guardrail), StockBench arXiv:2510.02209 (LLM agents mostly lose to B&H even zero-cost), arXiv:2602.14233 (structural-validity gate: failed requirements → stress-test interpretation only, no deployable-alpha claim), KTD-Fin arXiv:2605.28359 [ADVERSARIAL] (adding tools/capabilities did NOT improve returns; 9/10 negative selection alpha → steer FEATURES toward reliability, not analytical depth), Bailey-LdP MinTRL formulas, arXiv:2509.04541 (band turnover regularization > linear penalties; objective-level > post-hoc), ThinkNewfound (longer holding helps momentum net of costs). 53.1-differentiation: score-hysteresis is the SAME family as the rejected band (auto-FAIL risk — recommend AGAINST); minimum holding period is the evidence-best LEVER (time axis, mechanistically orthogonal); binding-RiskJudge-REJECT + concentration-aware sizing is the evidence-best FEATURE (F-F/F-G/B11; KF3/KF5).

## Hypothesis

The consolidated evidence supports a reliability-features-first recommendation with a single time-axis churn lever as the LEVER candidate; the operator block can price the live window honestly (burn $0.50-40 over 1-2wk depending on mode; value = sanity-gate + DoD closure, NOT skill proof per MinTRL) — all at $0 from existing findings + the research brief.

## Immutable success criteria (verbatim from .claude/masterplan.json, step 55.3)

1. "a ranked findings table (severity x N*-impact) consolidates 55.1 + 55.2, with each finding carrying a stable ID (F-1, F-2, ...) that phase-56 fixes must reference and an owner column (fix-in-56.x / operator-gated / WONTFIX), written in blameless systemic framing (why it passed silently, not who); the table separates CODE-CONFIRMED findings from DATA-INFERRED ones"

2. "the strategic chapter passes the research gate (>=5 sources read in full + recency scan, per .claude/rules/research-gate.md) covering LLM-trading-agent evaluation (incl. the 2025-2026 evidence that short-window LLM-trading wins usually vanish under longer, cost-inclusive evaluation), paper-trading statistical power (compute and STATE the MinTRL number from the observed Sharpe + skew/kurtosis -- 8 days of dailies cannot establish significance), and agent-skill ROI; it compares the system against a passive buy-and-hold baseline AND the existing US-momentum-core baseline using the 55.1 regime-vs-skill attribution, and concludes finetune-vs-features with explicit reasoning grounded in the findings + literature"

3. "the chapter concludes with a recommendation plus a tight one-paragraph spec for EACH candidate phase-57 variant: a LEVER variant (exactly ONE of score-hysteresis/persistence, minimum holding period, sector-concentration cap, turnover budget -- chosen from the evidence; measured ON-vs-OFF via the $0 replay/backtest on the production universe reporting Sharpe/return/turnover/maxDD; subject to the SAME Ledoit-Wolf SR-difference robustness gate as 52.3/53.1 (p<0.05 AND delta>=+0.05 AND CI_low>0); config-gated default-off; US momentum core byte-identical unless the flag is enabled; re-proposing the 53.1-rejected no-trade band in naive or renamed form is an automatic FAIL) and a FEATURE variant (the top capability gap from the strategic chapter, e.g. full-mode agents in the autonomous path, per-market benchmark fetches (^KS11), concentration limits as a feature -- with measurable acceptance criteria of the same rigor); the FULL masterplan payload is authored at install time for the CHOSEN variant only, per this goal's 'Phase-57 required shape' section (two complete throwaway payloads are deliberately NOT pre-built)"

4. "an OPERATOR DECISION block is posted to the operator Slack channel: LLM burn estimate ($/cycle from llm_call_log cost columns, fallback token counts x current pricing, x planned cycles over a 1-2 week window), expected value (which of DoD-2/5/6/7/9 close; projected go-live-gate delta from baseline 1/5), the finetune-vs-features recommendation, and the verbatim reply grammar 'LLM SPEND: APPROVED <budget> | DECLINED' + 'PHASE-57: LEVER | FEATURE'; phase-56 may start once phase-55 is done, but phase-57 installation and any phase-58 live cycle are HARD-gated on the verbatim replies"

**Verification command (immutable):** `cd /Users/ford/.openclaw/workspace/pyfinagent && test -f handoff/current/55.3-synthesis-checkpoint.md && test -f handoff/current/live_check_55.3.md`

## Plan

1. Consolidate 55.1 B1-B15 + 55.2 F-A1..F-I into final stable IDs F-1..F-19, ranked severity × N*-impact, owner column, blameless framing, CODE-CONFIRMED vs DATA-INFERRED split.
2. Write the strategic chapter grounded in research_brief_55.3.md (KF1-KF6): cost-inclusive LLM-trading evidence; MinTRL STATED (377-dailies observed-|SR| figure from 55.1 + the horizon menu: ~539 dailies / 2.1y at backtest SR 1.17, ~2,820 / 11y at SR 0.5); agent-skill ROI (KTD-Fin adversarial); baselines (passive SPY +2.49% vs fund −2.26% away week; US-momentum-core Sharpe 1.17/DSR 0.95 vs the lite-LLM book) using the 55.1 attribution; conclude finetune-vs-features with explicit reasoning.
3. Recommendation + both one-paragraph specs (LEVER = minimum holding period with the Ledoit-Wolf gate verbatim; FEATURE = binding RiskJudge REJECT + concentration-aware sizing with regression-fixture acceptance criteria). State explicitly that score-hysteresis was considered and rejected as same-family-as-53.1.
4. Compose + post the OPERATOR DECISION block to Slack #ford-approvals (C0ANTGNNK8D): burn table ($/cycle measured from BQ; honest llm_call_log undercount caveat per F-E), expected value (DoD-2/5/6/7/9 closure; gate delta from baseline 1/5 — current reading 2/5), recommendation, verbatim reply grammar. Record the Slack timestamp.
5. Write `55.3-synthesis-checkpoint.md` + `live_check_55.3.md` (ranked table + research-gate JSON envelope + Slack ts).
6. Fresh Q/A → harness_log append → masterplan flip. Phase-56 may start after the flip; phase-57 install + phase-58 live cycles stay HARD-gated on the two verbatim operator replies.

## Constraints

- $0; review-only; no fixes; the Slack post is the one outward action (explicitly mandated by immutable criterion 4).
- The chapter's recommendation must be evidence-grounded (findings + literature), with the adversarial source (KTD-Fin) addressed, not buried.
- 53.1's REJECT is binding: no naive/renamed no-trade band.
- MinTRL framing must prevent the operator from reading the live window as a skill proof.

## References

- handoff/current/research_brief_55.3.md (gate_passed: true; KF1-KF6; both candidate spec drafts)
- handoff/archive/phase-55.1/55.1-away-week-postmortem.md (B1-B15); handoff/current/55.2-ops-skill-audit.md (F-A1..F-I)
- handoff/current/cycle_block_summary.md (DoD-2/5/6/7/9 definitions; operator follow-ups)
- handoff/current/goal_post_away_review.md (phase-57 required shape; checkpoint mechanics)
- External: arXiv:2505.07078v5, arXiv:2510.02209, arXiv:2602.14233, arXiv:2605.28359, Bailey-LdP DSR/MinTRL, arXiv:2509.04541, ThinkNewfound; arXiv:2603.27539 (re-cited from 55.2)
- Code: analytics.py:239,292; rebalance_band.py:22; no_trade_band_replay.py:133-147; portfolio_manager.py:185-198; settings.py:308,316-318; optimizer_best.json; paper_metrics_v2.py:33
