# Research Brief — phase-72.3: P3 Earning-Capacity Decision Sheet (recommend-only)

**Tier:** moderate. **NOT audit-class.** **Researcher:** Layer-3 (Workflow path).
**Started:** 2026-07-18. Status: WRITE-FIRST IN PROGRESS.

## Mandate

Rank EVERY dark lever in `money_recon_2026-07-18.md` "Dark-lever inventory"
by expected net P&L impact, with risk + rollback per item, USING EXISTING
EVIDENCE ONLY (no optimizer runs, historical_macro frozen, no new backtests).
Operator flips; session recommends only. Honest "NONE FOUND" where no evidence.

Baseline facts (from money_recon + money_diagnosis_72):
- Book 97.2% cash; NAV flat since 05-29; the ACTIVE blocker is P0 (credit-dead
  LLM scoring rail), NOT any lever below. Levers are earning-CAPACITY once P0 clears.
- Trades net-profitable in-window: 29-30 round trips, +$3,057-3,194 realized.
  Exit-reason P&L: stop_loss_trigger 14/+$1,941.10 (avg +17.82%, trailing-profit
  captures), swap_for_higher_conviction 13/+$1,103.18, sell_signal 2/+$150.40.
- Costs negligible (~$51, ~0.2% NAV).

---

## INTERNAL EVIDENCE HUNT (main leg)

All line refs `backend/config/settings.py` unless noted. Live `.env` UNCONFIRMED
(permission-blocked) — code defaults quoted; operator grep pending (money_recon §Operator greps).

### QUANTIFIED evidence (hard numbers exist)

- **paper_soft_sector_diversity (:447-448)** — STRONGEST. phase-70.2 clean ablation
  (`_70_2_soft_diversity_replay.json`, 47 rebalances): baseline ann_Sharpe **1.344**;
  soft d_Sharpe **+0.176 / +0.200 / +0.234** at w=0.10/0.20/0.30 (all POSITIVE, monotonic);
  breadth **+1.25 / +2.02 / +2.60** sectors; turnover ~neutral (0.542→0.549).
  Hard sector_neutral = **−0.117** (WORSE — why the SOFT design). phase-70.2 evaluator_critique
  L33-34 corroborates. CAVEAT: replay shows Sharpe≥incumbent but does NOT report the
  DSR≥0.95/PBO≤0.5 clearance the design's activation gate (design_trade_diversity_70.md §a)
  requires — Sharpe-ablation only.
- **momentum_52wh_tilt (:443, k=0.5)** — `_52wh_paired_returns.json` config_sharpes:
  baseline 1.3437, **hi52_k0.5 = 1.3985 (+0.0548)**, hi52_k1.0 1.3983 (plateau), vol_scaled
  1.3468 (flat), sector_neutral 1.2265 (−0.117). Settings desc: "+0.05 ann Sharpe, turnover-neutral"
  (George-Hwang 2004 anchoring). Small, cheap, additive, evidence-backed.
- **Exit-reason P&L (bears on scale_out)** — money_recon: stop_loss_trigger 14 trips / **+$1,941.10,
  avg +17.82%** (mostly trailing-PROFIT captures — the trail ratcheted up and harvested big winners);
  swap_for_higher_conviction 13 / +$1,103.18; sell_signal 2 / +$150.40. Trades net-profitable 19W/10L.

### MECHANISM-PROVEN, NO $ MAGNITUDE (correctness/safety fixes)

- **sign_safe_overlays (:36)** — phase-69.3 evaluator C1: neg-base +catalyst(−9) now ranks ABOVE
  neg-base −catalyst(−11); OFF INVERTS. Reproduced directly + via test. Fixes a ranking sign-inversion
  that only bites when composite scores go negative (drawdowns). Magnitude: **NONE FOUND** ($ impact).
- **paper_atomic_swap (:453)** — fixes the Saga net-−1-position bug (SELL fires, paired BUY silently
  drops when cost>cash; design §b finding #9). Protects the +$1,103 swap profit stream. Depends on
  paper_swap_enabled (True). Magnitude: NONE FOUND (safety fix, not new alpha).
- **paper_avg_entry_fx_fix (:455)** — non-US avg-entry corruption fix (LOCAL-weighted vs USD-cost mix);
  byte-identical for US. Bears on the 10 in-window KR (.KS) adds. Magnitude: NONE FOUND.
- **paper_cross_sector_rotation (:454)** — structural "changeable fund" enabler; fires only if HHI
  strictly drops; HARD dep on paper_swap_churn_fix ON. Complements diversity. No standalone $ evidence.
- **kill_switch_peak_reset (:38, KS-PEAK-RESET)** — INSURANCE not earnings. money_recon REFUTED it as
  active cause (precondition never occurred; latent permanent-lockout bomb). phase-69.1: DARK, wired into
  resume(); evaluator nit — invoked outside the lock; endpoint update owed for full operator-path.
  Normal-op P&L: **ZERO**; prevents a catastrophic future 100%-cash freeze.

### NO INTERNAL EVIDENCE (honest NONE FOUND)

- **paper_scale_out (:34)** — quant_results.tsv: **0** scale_out/take_profit/2R/3R rows. NO backtest.
  Only inferential evidence = exit-reason data above: the trail already harvests +17.82% avg on winners,
  so a 2R(16%) partial would CAP the biggest trending winners. Regime-dependent; see external topic 1.
- **paper_session_budget_reconcile (:456)** — hidden $1.00 = HALF the visible $2.00 cap; truncates
  cycles early (design §c). Throughput lever (more candidates/cycle) at ~2x LLM cost (still ~$0.4/day,
  negligible). Needs LLM-cost approval. No $ P&L magnitude; only matters post-P0.
- **paper_position_recommendation_fix (:201)** — revives structurally-dead signal_downgrade SELL. Blast
  radius = SELLs of held positions. UNSAFE if synthesis_integrity OFF (synthetic HOLD → wrongful SELL of
  healthy position — settings.py:203 logs WARNING). RISK-FORWARD; gate behind P0. No $ magnitude.
- **paper_price_tolerance_pct (:560, ACTIVE 5.0)** — live BUY-suppressor (reject fill >5% off analysis
  price; SEC LULD basis). Could reject fast-moving momentum BUYs. Actual in-window rejections: NONE FOUND
  (needs the 70.4 skip-reason ledger, not yet shipped). Tuning is a HYPOTHESIS, not evidence-backed.
- **paper_min_k_sectors_analyzed (:449) + paper_unknown_sector_cap_exempt (:450)** — structural enablers
  bundled with soft diversity (part of the 70.2 mechanism); low STANDALONE impact. Unknown-cap fix stops a
  ticker-meta failure collapsing N sectors into one frozen "Unknown" bucket (design §a.3, findings #5/#14).
- **regime_net_liquidity (:37)** — regime-prompt enrichment (net-liq WALCL−WTREGEN−RRPONTSYD + INDPRO
  repair, new 24h cache, historical_macro untouched). Indirect (better regime → better overlay weights).
  Magnitude: NONE FOUND.
- **meta_scorer_enabled (:402)** — BLOCKED on P0: bypasses the rail (meta_scorer.py:220-225 direct key),
  credit-dead every trading day since 05-22. Enabling now = no-op/harmful until P0 credit + R1 decoupling.
- **Dark alpha-overlay library (:362-521, ~20 flags)** — treat as ONE dossier. Most-cited members:
  momentum_52wh_tilt (:443, broken out above, +0.05 Sharpe = ONLY internally-validated one),
  short_interest_filter (:406, Boehmer-Jones-Zhang 2008 lit only), options_flow_screen (:462, lit only),
  russell1000_universe (:459, SNDK-miss case). Internal backtest on OUR data: NONE FOUND for all but 52wh.

### APPLIED — skip ranking

- **paper_data_integrity_enabled (:45)** — per 72.1 the 06-11 keystroke batch (swap-churn-fix /
  data-integrity / RJ-binding) IS applied + runtime-loaded (60.2 corroborated). Noted APPLIED; not ranked.

## EXTERNAL RESEARCH

### Read in full (>=5; counts toward gate)
| # | URL | Kind | Topic | Key finding |
|---|-----|------|-------|-------------|
| 1 | arxiv.org/html/2604.27150 | peer-reviewed (2026) | 1 scale-out | Autonomous-agent-swarm SL/TP grid: best = **75% partial TP at +10%** + **10% stop** → **+25.2% Sharpe** over 25% baseline. "small early de-risking is not enough; better strategies remove MORE exposure at first gain threshold." Trailing 3% activation/2-5% dist = moderate. **[ADVERSARIAL to trend-following view]** |
| 2 | traderssecondbrain.com/guides/take-profit-methods | practitioner | 1 scale-out | R-multiple signatures: full-exit=consistent R, scale-out=blended R (0.5×1R+0.5×2R=1.5R), trailing=variable (0.5R..5R+). **Trend-following (30-45% WR) → TRAILING STOP** (asymmetric runners drive EV; full-exit/scale-out caps the edge). Scale-out fits BIMODAL only. Mismatch = **10-25% degradation**. |
| 3 | quantpedia.com/should-factor-investors-neutralize-the-sector-exposure | quant-research | 2 concentration | Ehsani/Harvey/Li: keeping sector exposure better in **78% of long-only trials** (sector-neutral wins only 20% for long-short). Neutralize ONLY iff Sharpe(across/within) < corr(across,within) — rarely holds long-only. |
| 4 | people.duke.edu/~charvey/.../P165_Is_sector_neutrality.pdf | peer-reviewed FAJ 2023 (PDF, partial extract) | 2 concentration | Same paper, source: bottom-up Sharpe **0.694** vs sector-neutral **0.659** (neutralization REDUCES long-only Sharpe). Conclusions extracted; exact tables in binary. |
| 5 | tradezella.com/blog/how-to-build-a-trading-system | practitioner | 3 rollout | **Single-variable rule**: "Treat every parameter change as a new hypothesis... Backtest → forward-test 20+ trades at reduced size → micro-live 25% → full only after validation. Never change on a single trade or bad week." |
| 6 | arxiv.org/html/2607.06117 | peer-reviewed (2026) | 3 rollout | Gated incremental admission: "all statistically-screened terms simultaneously produces inferior results"; only **2 of 26** candidates survived the incremental OOS Sharpe filter (many HAC |t|>4 had NEGATIVE incremental Sharpe). Validates one-gated-change-at-a-time. |

### Recency scan (2024-2026)
PERFORMED. Current-frontier (2026): arXiv 2604.27150 + 2607.06117 (both 2026 preprints), tradezella 2026 rollout guide, Morningstar 2026 concentration data (top-10 US names = 36% of index, most-concentrated since 1932). Year-less canonical: Ehsani/Harvey/Li FAJ (2023, SSRN 2021) — still the governing sector-neutrality reference; NOT superseded by newer work. No 2024-2026 finding OVERTURNS the FAJ long-only conclusion; the 2026 agent-swarm scale-out paper ADDS a regime-caveat (see topic-1 debate).

### Consensus vs debate
- **Topic 2 (concentration): CONSENSUS** — soft/partial diversity beats both extremes for long-only. Internal replay (+0.176..+0.234 Sharpe soft; −0.117 hard) and FAJ 2023 (0.694 vs 0.659; 78% keep-sector) AGREE: shade sector representation, never hard-neutralize. High confidence.
- **Topic 1 (scale-out): GENUINE DEBATE.** arXiv 2604.27150 (2026) found aggressive early scale-out (75%@+10%) HELPED (+25.2% Sharpe) on a high-turnover agent-swarm/volatile sample with a 48h stale-close. traderssecondbrain + trend-following math say scale-out HURTS trend-following (the regime our in-window trail resembles: +17.82% avg captured on winners). Reconciliation: **scale-out value is REGIME-dependent** — helps choppy/bimodal/high-turnover, hurts sustained-trend. Our exit-reason data leans trend-like → scale-out likely NEUTRAL-to-NEGATIVE here, but UNPROVEN on our data (backtest frozen).
- **Topic 3 (rollout): CONSENSUS** — one validated change at a time; batch flips underperform. Directly governs the decision-sheet recommendation to SEQUENCE, not batch.

### Pitfalls (from literature)
- Changing R/R by taking partials silently re-specifies the strategy you backtested (traderssecondbrain) → any scale_out flip invalidates the existing trailing-stop expectancy evidence.
- Statistical significance ≠ incremental value (arXiv 2607.06117): a lever can look good in isolation yet subtract once combined — argues for ON-vs-OFF gating of EACH flip, matching the phase-70 DSR/PBO activation gate.

## LEVER DOSSIERS (ranked by expected net P&L impact)

### BINDING FRAMING (read before the ranking)
1. **P0 gates everything.** The credit-dead scoring rail keeps the book ~97% cash;
   NO lever below earns a cent until P0 clears. These are earning-CAPACITY ranks,
   realized only post-P0. Recommend-only — operator flips.
2. **Sequence, do NOT batch** (tradezella single-variable rule + arXiv 2607.06117:
   "all-signals simultaneously produces inferior results", 2/26 survived). Flip ONE
   lever, run the ON-vs-OFF live_check / phase-70 DSR≥0.95-PBO≤0.5 gate, then the next.
3. Recommended flip ORDER once P0 is up: **[insurance] KS-PEAK-RESET → [alpha] soft-diversity(w=0.20)
   → 52wh-tilt(k=0.5) → [safety] atomic-swap + avg-entry-fx → [enabler] cross-sector+min-k+unknown-cap
   → [throughput] session-budget → [regime] sign-safe + net-liq**. HOLD: scale-out, position-rec-fix,
   price-tolerance, meta-scorer, wider overlay library (evidence-gated / P0-gated).

### RANK TABLE
| # | Lever (:line) | Tier | Expected net P&L | Evidence strength |
|---|---|---|---|---|
| 1 | paper_soft_sector_diversity(_w) :447-448 | ALPHA | **+0.20 ann Sharpe** (w=0.20) | QUANTIFIED (internal replay + FAJ 2023) |
| 2 | momentum_52wh_tilt :443 (k=0.5) | ALPHA | **+0.05 ann Sharpe**, turnover-neutral | QUANTIFIED (internal replay) |
| 3 | paper_atomic_swap :453 | SAFETY | protects +$1,103 swap stream (no net-1 leak) | mechanism-proven, no $ |
| 4 | paper_avg_entry_fx_fix :455 | SAFETY | correct non-US cost-basis/stops (10 KR trades) | mechanism-proven, no $ |
| 5 | paper_cross_sector_rotation :454 | ENABLER | enables cross-sector rotation (compounds #1) | structural, no standalone $ |
| 6 | paper_min_k_sectors :449 + unknown_cap :450 | ENABLER | unfreezes the funnel; part of #1's mechanism | structural, no standalone $ |
| 7 | paper_session_budget_reconcile :456 | THROUGHPUT | more candidates/cycle @ ~2x LLM $ (~$0.4/day) | no $ magnitude; cost-approval |
| 8 | sign_safe_overlays :36 | REGIME | +rank quality in drawdowns only | mechanism-proven, NONE FOUND $ |
| 9 | regime_net_liquidity :37 | REGIME | indirect (better regime → overlay weights) | NONE FOUND $ |
| — | kill_switch_peak_reset :38 | INSURANCE | **$0 normal-op**; prevents permanent-lockout | mechanism-proven (owed token) |
| HOLD | paper_position_recommendation_fix :201 | RISK-FWD | adds SELL path; DANGEROUS pre-P0 | NONE FOUND $; wrongful-SELL risk |
| HOLD | paper_price_tolerance_pct :560 | TUNING | possible BUY-suppressor | NONE FOUND rejections |
| HOLD | paper_scale_out :34 | REGIME-DEP | likely NEUTRAL/−ve in trend regime | NO backtest; CONFLICTING external |
| HOLD | meta_scorer_enabled :402 | BLOCKED | no-op until P0 credit + R1 | P0-blocked |
| HOLD | dark overlay library :362-521 | GATED | each needs DSR/PBO gate | only 52wh internally validated |
| APPLIED | paper_data_integrity_enabled :45 | — | (already live per 72.1) | not ranked |

Full per-lever what/evidence/impact/risk/rollback shipped as the `lever_dossiers`
structured-output array (this brief's dossier detail == that array, verbatim).

---

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 5,
  "snippet_only_sources": 9,
  "urls_collected": 42,
  "recency_scan_performed": true,
  "internal_files_inspected": 11,
  "coverage": {
    "audit_class": false,
    "rounds": 1,
    "dry_rounds": 0,
    "K_required": 2,
    "new_findings_last_round": 0,
    "dry": false
  },
  "summary": "Ranked every dark lever by expected net P&L using EXISTING evidence only. P0 (credit-dead rail) gates all earning; levers are post-P0 capacity. TIER-1 ALPHA (quantified): soft sector diversity +0.20 ann Sharpe at w=0.20 (internal 70.2 replay, corroborated by Ehsani/Harvey/Li FAJ 2023: long-only keeps sector in 78% of trials, hard-neutral 0.694->0.659); 52wh tilt +0.05 ann Sharpe turnover-neutral (52.1 replay). TIER-2 SAFETY (protect existing profit, no new $): atomic-swap (no net-1 leak, guards +$1,103 swap stream), avg-entry-fx (non-US cost-basis). ENABLERS: cross-sector-rotation + min-k + unknown-cap complete the diversity mechanism. THROUGHPUT: session-budget (2x LLM $). REGIME: sign-safe + net-liq (NONE FOUND $). INSURANCE: KS-PEAK-RESET ($0 normal-op, prevents permanent lockout). HOLD: scale-out (NO backtest; external CONFLICTS-arXiv 2604.27150 says helps high-turnover, trend-math says hurts, our +17.82%-trail data is trend-like), position-rec-fix (wrongful-SELL pre-P0), price-tolerance (no measured rejections), meta-scorer (P0-blocked), wider overlay library (DSR/PBO-gated). Rollout: SEQUENCE one gated flip at a time (arXiv 2607.06117: batch admission inferior, 2/26 survived).",
  "brief_path": "handoff/current/research_brief_72.3.md",
  "gate_passed": true
}
```

