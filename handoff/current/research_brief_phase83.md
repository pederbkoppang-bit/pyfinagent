# Phase-83 Design Pack — Market-News Thematic Engine (pre-registered)

**Step 83.1, written 2026-08-07.** DESIGN ONLY — changes no live behaviour, writes no signal code. This pack LANDS the completed 2026-08-04 8-lens audit-class research (`handoff/current/phase83_research_raw/{research,synthesis,verdicts}.json`, three adversarial verification passes) as an auditable artifact; it does not re-run it. Where a criterion asks for something the corpus never recorded, this pack says so instead of fabricating (see "Corpus limits").

## 1. Verdict carried forward

`synthesis.go_no_go = "descope"`. The corpus's own bottom line: the thematic engine is worth building ONLY as a slow score-overlay on the existing monthly reconstitution, with the honest pre-registered expectation that the gate step (83.5) FAILS on free data at gate-grade trial counts. A null result is planned-for and informative — the pre-registration exists precisely so that outcome cannot be talked past (kill rule owned by 83.1.1).

## 2. Gate thresholds — READ FROM SOURCE (criterion 2)

The promotion gate that actually governs research promotion is `backend/autoresearch/gate.py::PromotionGate` (frozen dataclass, gate.py:19-30), read at runtime, never hardcoded from memory:

- min_dsr = 0.95 (gate.py:21)
- max_pbo = 0.20 (gate.py:22)
- min_pbo_trials = 10 (gate.py:30)

**The 0.50 figure every 2026-08-04 research lens quoted is a DIFFERENT decision**: `backend/services/promotion_gate.py:37` `PBO_CEILING = 0.5` feeds `evaluate_stage` (live allocation staging, PSR-parity basis at :56, strict-`<` comparator at :57) — not the research-promotion gate (whose comparator rejects on `>` at gate.py:61). Provenance of the error: the spawn prompt itself briefed "PBO<=0.5" (`research_prompt_market_news.md:32`); `synthesis.headline_findings[0]` and `killed_options[2]` already carry the correction. Pre-registering against the loose ceiling would have baked a 2.5x calibration error into this artifact.

PBO is computed ONLY via `compute_pbo_checked` (refusal ≠ pass — step 83.0.3), and every producer feeding the gate ALWAYS emits `pbo_n_trials`, because the gate.py:43-45 legacy carve-out makes the trials floor inert for producers that omit it (83.0.3 disclosure).

## 3. Design decisions (from `synthesis.design_decision`, condensed with the numbers that ground them)

- **Theme representation**: VERSIONED, FROZEN-AT-BIRTH, APPEND-ONLY registry — keyword + confirmation-gate rail (the only representation with a published human-validation number: GPR correlates 0.93 annually with a hand-coded 7,000+ article index; point-in-time by construction; the shape `defense_signal.py` already ships). Any spec change mints a NEW theme_id with a supersedes pointer. Intensity is an attention SHARE (self-normalizing).
- **Entry**: on ACCELERATION — positive and rising 6-month attention-intensity difference. Never on birth (news novelty forecasts **-2.821%/yr per sd**; negative cross-sectional risk premium). Never on confirmation (thematic-ETF launch marks the valuation peak: **-0.50%/mo FFC4 for the following 60 months**). Only the 6-month difference generates spread; 1-week and 3-month differences do not.
- **Exit**: ON A CLOCK — hard exit 6 months, absolute cap 12 months (drift "persists for about a year"; reverses by year 2). Minimum hold ≥ 8 weeks, TIME-BASED (a signal-based exit converts a slow design into a fast one; the empirical failure shape: short-term reversal at 305%/mo turnover, 6.75%/yr cost).
- **Implementation**: a score overlay on the EXISTING monthly reconstitution — never a separate trading sleeve (the live monthly design already spends 133 bps/yr at the repo's 10bp assumption, ~2.8x the 48 bps/yr central alpha budget; adding a sleeve doubles spend, overlaying does not).
- **Label horizon (pre-registered)**: **126 trading days** (the calibrated 6-month hold) — explicitly DISTINCT from the engine's holding-days-derived 1.5× purge horizon (135 calendar days, `backtest_engine.py:274` × `:962`).

## 4. Candidate designs — closed-vocabulary cost classification (criterion 7)

**Closed vocabulary**: `survives_costs` | `marginal` | `fails_costs` | `untestable_on_free_data`. Source: Lens-4 derived turnover/cost table (T = 2 × rebalances/yr × fraction replaced; reconciles exactly with the repo's phase-53.1 replay 133 bps/yr AND Ke-Kelly-Xiu's published 46%/yr). Costs quoted at the repo-default 10bp one-way; the classification rule is Tmax = alpha budget / one-way cost (Chen-Velikov central alpha 48 bps/yr; strongest-anomaly 240 bps/yr).

| # | Candidate design | T (×NAV/yr) | Cost @10bp (bps/yr) | Classification |
|---|---|---|---|---|
| 1 | Continuous rank tilt (incremental, ~8% names/mo) | 1.9 | 19 | survives_costs |
| 2 | Quarterly full thematic basket | 8.0 | 80 | marginal |
| 3 | Monthly top-N (LIVE today, 0.555 replaced/mo) | 13.3 | 133 | marginal |
| 4 | Event entry + 8-week time exit | 13.0 | 130 | marginal |
| 5 | Event entry + 4-week time exit | 26.0 | 260 | fails_costs |
| 6 | Weekly rotation | 104 | 1,040 | fails_costs |
| 7 | Daily headline (KKX VW 91.4%/day) | 460.7 | 4,607 | fails_costs |

Unclassified candidates: **0**. Classification notes: #1 clears every alpha/cost combination (2-10x headroom). #2-#4 sit inside the budget at c=1.6-3bp on all alpha assumptions but at c=10bp clear only the strongest-anomaly assumption (fail the central 48 bps/yr budget by up to 2.8x) — hence `marginal`, not `survives_costs`. #5-#7 fail every combination (#7 needs 29× the central budget). No candidate is `untestable_on_free_data` — the vocabulary term exists because the free-data constraint binds elsewhere (per-theme sample adequacy, section 7), and future candidates must use it rather than omit a classification.

## 5. Negative evidence (criterion 3 — failure modes each with source AND number)

1. **Thematic ETFs structurally underperform**: **-3.1%/yr risk-adjusted after fees**; about **-6%/yr in the first five years post-inception** (~-30% cumulative risk-adjusted); **FFC4 alpha -3.24%/yr vs -0.24%/yr** for broad-based, with a fee gap of only 0.13%/yr — an order of magnitude too small to explain it (Ben-David/Franzoni/Kim/Moussawi, peer-reviewed; read in full).
2. **The theme→beneficiary mapping's intellectual foundation has decayed**: Cohen-Frazzini ECONOMIC-LINK (customer-supplier) predictability, value-weighted — the implementation pyfinagent would use — drops from **1.30%/mo (1978-2004) to 0.62%/mo (2005-2018, t=1.54, statistically insignificant)**, a 52% decline.
3. **LLM-era backtest inflation**: DeepSeek-driven news strategy collapses from **+20.73% to -1.04%** when evaluated strictly post-training-cutoff; LLM stock-pick accuracy falls from **80.58% to 45.70%** past the knowledge cutoff — the sharpest known measurement of look-ahead living in model weights.
4. **Sentiment-model sign instability**: the same corpus scored by three model generations flips a strategy from **+23.67%** to **-1.73%** to **-11.47%** — scorer_version pinning (83.0 schema) is load-bearing, not hygiene.
5. **Survivorship in thematic products**: Morningstar — **3 of 4** thematic funds shuttered over 15 years; **1 in 10** both survived and outperformed.
6. **Theme-timing gap**: investors in thematic vehicles realize **4.9pp/yr** less than the vehicles themselves (vs 0.5pp for broad funds) — the entry-timing failure mode the acceleration rule exists to avoid.
7. **Anomaly base rates**: Hou-Xue-Zhang replication — **65% / 82% / 96%** failure rates depending on the significance bar.

(Two additional Lens-7 counter-evidence findings exist but are `read_in_full=false` in the corpus and are therefore not counted toward the floor here.)

## 6. Reference-case table (criterion 4)

Free-source coverage windows are the corpus's MEASURED values (`synthesis.design_decision.data_sources`): GDELT `gkg_partitioned` **4,169 daily partitions, 2015-02-17..2026-08-04, ~99.6% calendar completeness**, licence verbatim *"unlimited and unrestricted use for any academic, commercial, or governmental use of any kind without fee"*; SEC EDGAR full-text (US-government public domain, floor 2001); Caldara-Iacoviello GPR (CC-BY, daily since 1985, already fetched at `backend/services/macro_regime.py`).

**Cost-to-hold cells are DERIVED** from the Lens-4 design-level table under the pre-registered slow-overlay implementation (monthly reconstitution overlay, 6-month hold ⇒ candidate #3/#4 shape, 130-133 bps/yr at the repo-default 10bp; 21-40 bps/yr at measured large-cap costs 1.6-3bp). No per-case cost figure exists in the corpus — these are derivations with the holding assumption stated, labelled DERIVED.

| Reference case | Free data source (coverage window contains the case) | Cost-to-hold (DERIVED, slow-overlay, 10bp / 1.6-3bp) |
|---|---|---|
| COVID / pharma (2020) | GDELT gkg_partitioned (2015-02-17..2026-08-04) + GPR (1985..) | DERIVED: ~130 bps/yr / 21-40 bps/yr |
| AI-datacenter / memory (2023-24) | GDELT gkg_partitioned (2015-02-17..2026-08-04) + SEC EDGAR full-text (2001..) | DERIVED: ~130 bps/yr / 21-40 bps/yr |
| Ukraine / defense (2022) | GDELT gkg_partitioned (2015-02-17..2026-08-04) + GPR (1985..) | DERIVED: ~130 bps/yr / 21-40 bps/yr |
| Iran-US / oil (2019-20, 2024-25) | GDELT gkg_partitioned (2015-02-17..2026-08-04) + GPR (1985..) | DERIVED: ~130 bps/yr / 21-40 bps/yr |

This corrects the corpus's own early error that COVID and Ukraine are "unbacktestable on free data" (reached considering only Alpha Vantage; GDELT's measured window covers all four cases). **Honesty note**: the four cases were never traced END-TO-END through a real pipeline — `synthesis.residual_risks[6]`, `verdicts[1].missing_coverage[5]`, and `verdicts[2].missing_coverage[2]` say so independently. This table asserts coverage and derived cost, not a completed trace.

## 7. Sample adequacy (verbatim-grounded, with anchor corrections)

The corpus's honest answer: **NO** — not for a gate-grade DSR/PBO verdict on a single theme series, and NOT because of calendar length. Four multiplicative constraints: (1) overlapping labels — at a 135-day purge horizon a 4.4-year corpus yields ~12 independent spans, GDELT's 11.46y ~30; (2) the DSR hurdle SR* = √V × E[max_N] is INDEPENDENT of T — trial count dominates (N 45→10 moves the floor more than 4.4y→41y of history); (3) the real ceiling is PBO ≤ 0.20, 2.5× tighter than every lens assumed; (4) CSCV needs ≥10 genuinely diverse configurations (near-identical columns → high PBO regardless of N; 82.3 measured 0.967-0.979 pairwise on the short window).

**Anchor corrections (this pack supersedes the step texts and auto-memory on these):** the 135-day figure is `backtest_engine.py:274` (`holding_days: int = 90`) × `:962` (`horizon_days = int(self.holding_days * 1.5)`) — the `:665` anchor circulating in the phase-83 step texts is macro-coverage logging, not the purge horizon. And `compute_deflated_sharpe`'s `variance_of_srs` default **0.5 is a VARIANCE** (`math.sqrt(var_srs)` at `analytics.py:429`), so the "V" the lenses argued over is the module default, and reading 0.5 as a standard deviation is off by √2 — 83.1.1 must MEASURE V, not assume it.

## 8. Killed options (from `synthesis.killed_options`, 15 total — headline entries)

Theme birth as entry signal (bearish aggregate); thematic-ETF-confirmation entry (peak marker); signal-based exits (turnover amplifier); daily/weekly rotation designs (#6/#7 above); LLM re-scoring of historical corpora without cutoff discipline (negative-evidence #3); pooled cross-strategy PBO matrices (answers the wrong question); Alpha Vantage as backfill corpus (non-commercial ToS, 25 req/day, ≥19× density discontinuity — operator decision pending, writer is source-agnostic per 83.0).

## 9. Corpus limits — recorded so completeness cannot be inferred (contract D6)

`verdicts[2]` (completeness-critic) returned **materially_flawed** with 10 never-researched angles, out of scope for 83.1 and listed so no reader infers coverage: price-mediated commodity/factor propagation, return-based theme construction, options-flow propagation, free analyst-revision proxies, concentration × risk-gate interaction, V as the highest-leverage unmeasured number, and four further angles enumerated in `verdicts.json`. The citation auditor (`verdicts[0]`) verified 13 primary sources in full against the lenses' 87 self-reports — the envelope below reports the self-reported figure with this label.

## 10. Pre-registration (criterion 5)

Machine-readable ranking file: `backend/backtest/experiments/preregistration_phase83_ranking.json`

**PREREGISTRATION_SHA256: 7b34649297c7f21c4f4a67621743cb01b5724f8316d0956d7a94ed7f61b4e5f0** (recomputed in cycle 2 after the append-only amendment adding the content rule; prior hash a22cb12f... superseded per the amendment policy)

The file pre-registers: the ranking criteria (measured-V DSR, checked-PBO gate with always-emitted `pbo_n_trials`, tie-breakers), the trial-budget cap (45), the label horizon (126 trading days), the entry rule, the artifact-population rule (globs `*_phase_83_*.json` + `phase83*.json`; the naive `*83*` rule is REJECTED — measured 71 false positives), the kill-rule pointer (83.1.1), and the pre-registered expected outcome (descope; 83.5 failing is the planned-for null). Amendment policy is append-only with hash recomputation. Phase-83 backtest artifacts existing at registration: **0** (measured). Ordering guard: no phase-83 backtest artifact may carry an mtime earlier than the ranking file (strict `st_mtime_ns` comparison; enforced by `backend/tests/test_phase_83_1_design_pack.py`, mutation-tested).

## Envelope (criterion 1)

`coverage.dry` is **null** because the 2026-08-04 run — briefed audit-class with K=2 (`research_prompt_market_news.md:28,114-120`) — persisted NO coverage object in any of the three raw JSON files (grep-verified 2026-08-07) and Cycle 1137's log entry carries no envelope. Reporting `true` would be fabrication; null with this stated reason is the honest value. Derivation rules: `external_sources_read_in_full` = sum of per-lens self-reports (the citation auditor verified 13 primary in full — see section 9); `snippet_only_sources` = count of distinct URLs flagged `read_in_full=false` = **16**; `urls_collected` = distinct http(s) `source_url` values = **79** (CORRECTED in cycle 2: the cycle-1 figure 85 did not reproduce under this rule — the Q/A's 12-variant sweep found 79 under the internally consistent http-only-distinct rule, which also yields the pack's own 63 full-read and 16 snippet-only counts, so the two candidate snippet rules AGREE at 16 and the previously-disclosed "16 vs 22" disagreement was an artifact of the wrong 85 input); `internal_files_inspected` = Lens-8's self-reported internal-artifact count.

```json
{
  "tier": "complex",
  "external_sources_read_in_full": 87,
  "snippet_only_sources": 16,
  "urls_collected": 79,
  "recency_scan_performed": true,
  "internal_files_inspected": 10,
  "coverage": {
    "audit_class": true,
    "dry": null,
    "dry_reason": "the 2026-08-04 run persisted no coverage object; asserting dry=true would fabricate a record that does not exist"
  },
  "gate_passed": true
}
```
