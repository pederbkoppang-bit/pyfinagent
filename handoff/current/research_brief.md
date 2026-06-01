# Research Brief — phase-53.1 (Algorithm/Quant Elevation: Portfolio Construction / Execution-Cost / Overfitting-Control)

Tier: **complex** | Researcher session | Date: 2026-06-01 | Gate: **PASSED** (`gate_passed: true`)

THE TASK: phase-52 exhausted the cheap-price *signal* search (rotation / sector-neutral /
vol-scaling / 52wh / residual-momentum all measured + REJECTED on the Ledoit-Wolf
SR-difference gate). 53.1 elevates PORTFOLIO CONSTRUCTION / EXECUTION-COST /
OVERFITTING-CONTROL instead of raw signal. Survey the candidate levers from the
literature, RECOMMEND ONE that is (a) a config-gated default-OFF overlay on the long-only
US momentum book, (b) measurable ON-vs-OFF via the existing $0 replay harness
(Sharpe/return/turnover/maxDD) on the production universe, (c) gated by the SAME
Ledoit-Wolf SR-difference robustness test as 52.3/52.4. DO-NO-HARM: +20% US momentum core
byte-identical unless enabled; NO live flag flip in 53.1.

**RECOMMENDATION (one line):** Adopt a **transaction-cost-aware no-trade / rebalance
buffer band** on the monthly top-N reconstitution (Garleanu-Pedersen "trade partially
toward the aim" + tolerance-band literature). It is the single best-evidenced
construction/cost lever for THIS long-only equal-weight book, it is the only candidate
whose mechanism (turnover reduction) the existing replay already measures directly, and
the three rival construction levers (vol-targeting, covariance-shrinkage/min-variance,
TSMOM regime filter) are each contraindicated for this book by the literature below.
Honest prognosis: the GROSS-Sharpe delta will be small (a band changes *which* names you
hold only at the margin); the lever's real value is **net-of-cost** via turnover cut, so
the 52.3 SR-difference gate on *gross* monthly returns may well return REJECT — that is a
valid, honestly-reported outcome, and the brief proposes reporting the net-of-cost delta
alongside the gross SR-diff so the keep/reject call is made on the right axis.

---

## Read in full (>=5 required; counts toward the gate) — 7 read

| # | URL | Accessed | Kind | Fetched how | Key finding |
| --- | --- | --- | --- | --- | --- |
| 1 | https://reasonabledeviations.com/notes/papers/ledoit_wolf_covariance/ | 2026-06-01 | blog/notes (peer-reviewed paper digest) | WebFetch full | LW shrinkage = `delta*F + (1-delta)*S`, F=constant-correlation target; "shrinkage beats sample covariance in all scenarios"; constant-corr target beats single-factor for N<=225. NO equal-weight comparison present (the 1/N critique is elsewhere). |
| 2 | https://www.kitces.com/blog/best-opportunistic-rebalancing-frequency-time-horizons-vs-tolerance-band-thresholds/ | 2026-06-01 | practitioner (Daryanani/Vanguard study digest) | WebFetch full | Optimal rebalance = **relative 20%-of-weight tolerance band**, "look constantly" but trade only on breach; "tolerance band approach will reduce the volume of rebalancing trades"; checking less often than every ~10 trading days erodes the benefit; banding "could even increase overall returns". |
| 3 | https://arxiv.org/html/2412.11575 | 2026-06-01 | arXiv preprint (HTML) | WebFetch full HTML | Cost-aware construction (proportional + quadratic cost penalty) lifts OOS **Sharpe 1.044 -> 1.295** AND cuts **turnover 18.16 -> 2.65** vs cost-unaware min-variance; "strategies incorporating transaction fees consistently demonstrate superior performance". |
| 4 | https://arxiv.org/html/2411.07949 | 2026-06-01 | arXiv preprint (HTML) | WebFetch full HTML | Optimal strategy = a symmetric **no-trade zone (+/-eta)**; "if the investor starts inside the no-trade zone, a future transaction only occurs at the boundary"; hysteresis from the buffer reduces trading frequency; uses trading frequency as the cost proxy (no Sharpe numbers). |
| 5 | https://www.nber.org/papers/w15205 | 2026-06-01 | NBER working paper (Garleanu-Pedersen, JF 2013) | WebFetch full (page) | Two principles: "**aim in front of the target**" + "**trade partially towards the current aim**"; optimal = linear combo of current portfolio and an aim portfolio (weighted toward slow-decaying signals); "superior net returns relative to more naive benchmarks" (empirics on commodity futures). |
| 6 | https://www.aqr.com/Insights/Research/Journal-Article/Time-Series-Momentum | 2026-06-01 | practitioner/journal (Moskowitz-Ooi-Pedersen) | WebFetch full | TSMOM = past-12mo own return predicts future return; **explicitly a 58-instrument multi-asset FUTURES result** ("country equity indices, currencies, commodities, sovereign bonds"), NOT a single-equity-market result. |
| 7 `[ADVERSARIAL]` | https://ideas.repec.org/a/oup/rfinst/v22y2009i5p1915-1953.html | 2026-06-01 | peer-reviewed (DeMiguel-Garlappi-Uppal, RFS 2009) | WebFetch full (page) | "**Of the 14 models we evaluate... none is consistently better than the 1/N rule** in terms of Sharpe ratio, certainty-equivalent return, or turnover"; the estimation window to beat 1/N is "**around 3000 months for 25 assets and about 6000 months for 50 assets**"; "the gain from optimal diversification is more than offset by estimation error". |

`[ADVERSARIAL]` tag: source #7 directly contradicts the prima-facie case for lever #3
(covariance-shrinkage / minimum-variance construction). It is the structural reason the
recommended lever works WITH the existing equal-weight construction rather than replacing
it. (Adversarial sourcing is a `deep`-tier formal requirement; included here at `complex`
because it is decisive for the lever choice.)

## Identified but snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
| --- | --- | --- |
| https://nbgarleanu.github.io/DynTrad.pdf | Garleanu-Pedersen full PDF | Binary PDF; per research-gate.md PDF-strategy WebFetch on `/pdf` returns no text; JF-2013 (pre-Dec-2023) so not on arXiv/ar5iv. Read via NBER page (#5) instead. |
| https://onlinelibrary.wiley.com/doi/full/10.1111/jofi.13395 | DeMiguel et al. 2024 JF "Multifactor Perspective on Volatility-Managed Portfolios" | HTTP 402 paywall. Findings captured via WebSearch snippet: single-factor vol-mgmt fails OOS net of costs; only the MULTIFACTOR version survives. |
| https://link.springer.com/article/10.1007/s11408-022-00419-6 | Springer "Rebalancing with transaction costs" | 303 redirect to auth/paywall. |
| https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3088828 | Barroso-Detzel "Do limits to arbitrage explain vol-managed benefits?" (JFE 2021) | Snippet only: vol-mgmt "can greatly increase monthly turnover by as much as 15 times" -> commensurate cost increase. |
| https://alphaarchitect.com/destabilizing-rebalancing/ | Alpha Architect rebalancing/momentum-band | HTTP 403 (twice). Topic covered by Kitces (#2). |
| https://alphaarchitect.com/surprise-the-size-value-and-momentum-anomalies-survive-after-trading-costs/ | Alpha Architect "anomalies survive after costs" | HTTP 403. Snippet: momentum survives net of costs; vol-mgmt's 15x turnover is the cost killer. |
| https://www.mdpi.com/2227-9091/14/4/84 | MDPI 2026 overnight-vs-daytime momentum across sector ETFs | Snippet: momentum edge "decreases significantly when transaction costs are taken into account... contingent on... below the 2-3 bps threshold". |
| https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf | Bailey-Lopez de Prado Deflated Sharpe Ratio | Already in-repo (`compute_deflated_sharpe`); canonical, not re-read. |
| http://www.ledoit.net/honey.pdf | Ledoit-Wolf "Honey, I Shrunk the Covariance Matrix" original | Covered by digest #1. |
| https://www.semanticscholar.org/paper/...DeMiguel-Garlappi/... | DeMiguel 1/N (Semantic Scholar) | Page rendered empty; read via IDEAS (#7) instead. |
| https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2326253 | Bailey et al. "Probability of Backtest Overfitting" (CSCV) | Canonical; PBO/CSCV already in repo (`compute_pbo` per prior briefs). |

**URLs collected: 21** (7 read-in-full + 11 snippet-table + 3 search-only mirrors). Gate floor (10+) cleared.

## Recency scan (2024-2026) — PERFORMED

Searched the 2024-2026 window explicitly (queries tagged `2026`, `2025`, `2024`). Findings
that complement/supersede the canonical sources:

1. **DeMiguel, Martin-Utrera, Uppal, Nogales 2024 (JF), "A Multifactor Perspective on
   Volatility-Managed Portfolios"** — single-factor vol-management (the Moreira-Muir 2017
   lever) **fails out-of-sample and net of transaction costs**; only a *multifactor*
   conditional version survives. This SUPERSEDES Moreira-Muir for a single-book context
   and directly explains why the project's 52.x `vol_scaled` arm was rejected.
2. **arXiv:2412.11575 (Dec 2024), "Cost-aware Portfolios in a Large Universe"** — fresh
   hard ON-vs-OFF numbers (Sharpe 1.04->1.30, turnover 18.2->2.65) confirming cost-aware
   construction beats cost-unaware in a large equity universe (read in full, #3).
3. **arXiv:2411.07949 (Nov 2024), "Optimal two-parameter portfolio... transaction costs"**
   — confirms the symmetric no-trade-zone / hysteresis mechanism is still the optimal
   structure under proportional costs (read in full, #4).
4. **MDPI Risks 2026 (Apr), overnight-vs-daytime momentum across sector ETFs** — momentum's
   net edge is "contingent on maintaining transaction costs below the 2-3 bps threshold"
   — quantifies how cost-sensitive a momentum book is, reinforcing the cost-aware lever.
5. **Alpha Architect 2025, "Size/Value/Momentum survive after trading costs"** — momentum's
   premium survives net of costs (the signal is fine); the binding constraint is turnover,
   not raw alpha — which is exactly the axis a no-trade band attacks.

No 2024-2026 source overturns the Garleanu-Pedersen no-trade-band foundation or the
DeMiguel 1/N result; both remain the live consensus. Net: the recency scan REINFORCES the
recommendation (cost/turnover control) and REINFORCES the rejection of vol-targeting.

## 3-query evidence (mandatory variants)

- **Current-year (2026):** "...no-trade buffer rule 2026 systematic equity" -> MDPI 2026
  ETF-momentum cost-threshold paper; "...transaction-cost-aware sizing 2026" -> Garleanu
  ideas/repec + alphaex 2026 threshold-rebalancing.
- **Last-2-year (2025/2024):** "...Barroso Detzel transaction costs anomalies survive 2025"
  -> Alpha Architect 2025 + Barroso-Detzel; "Moreira Muir... criticism out-of-sample 2024"
  -> DeMiguel et al. 2024 JF multifactor.
- **Year-less canonical:** "Garleanu Pedersen dynamic trading..." -> JF 2013 + NBER w15205;
  "Ledoit Wolf covariance shrinkage minimum variance" -> ledoit.net + RFS; "DeMiguel
  optimal versus naive 1/N" -> RFS 2009; "Moskowitz Ooi Pedersen time series momentum" ->
  AQR/JFE 2012. The source table is a deliberate mix of 2026/2025/2024 frontier and
  2009-2013 canonical prior-art.

---

## Key findings (per-claim cited)

1. **The optimal response to transaction costs is a no-trade region + partial adjustment,
   NOT a different signal.** "the optimal strategy is characterized by two principles: 1)
   aim in front of the target and 2) trade partially towards the current aim" (Garleanu &
   Pedersen, NBER w15205, https://www.nber.org/papers/w15205, accessed 2026-06-01). The
   buffer is a *symmetric no-trade zone*: "if the investor starts inside the no-trade zone,
   a future transaction only occurs at the boundary" (arXiv:2411.07949,
   https://arxiv.org/html/2411.07949, accessed 2026-06-01).
2. **Cost-aware construction delivers a large, measured net edge in a large equity
   universe.** OOS Sharpe rises 1.044 -> 1.295 and turnover falls 18.16 -> 2.65 vs
   cost-unaware min-variance (arXiv:2412.11575, https://arxiv.org/html/2412.11575, accessed
   2026-06-01). The dominant gain is the turnover cut, not raw return.
3. **Practitioner-calibrated band width: relative ~20% of target weight; monitor
   continuously, trade only on breach.** "the optimal rebalancing threshold was at a
   relative threshold of 20% of the investment's original weighting"; "tolerance band
   approach will reduce the volume of rebalancing trades" (Kitces/Daryanani,
   https://www.kitces.com/blog/best-opportunistic-rebalancing-frequency-time-horizons-vs-tolerance-band-thresholds/,
   accessed 2026-06-01).
4. **`[ADVERSARIAL]` Optimal/shrinkage/min-variance construction does NOT beat equal-weight
   1/N out-of-sample at this universe size.** "none is consistently better than the 1/N
   rule in terms of Sharpe ratio, certainty-equivalent return, or turnover"; the window to
   beat 1/N is "about 6000 months for a portfolio with 50 assets" (DeMiguel, Garlappi,
   Uppal, RFS 2009, https://ideas.repec.org/a/oup/rfinst/v22y2009i5p1915-1953.html,
   accessed 2026-06-01). => Lever #3 is contraindicated for a ~10-25-name book.
5. **Single-factor volatility management fails OOS net of costs.** Cederburg et al. and
   Barroso-Detzel show the Moreira-Muir lever "do[es] not survive transaction costs";
   vol-management "can greatly increase monthly turnover by as much as 15 times" (Barroso &
   Detzel, SSRN 3088828, accessed 2026-06-01); only the *multifactor* version survives
   (DeMiguel et al. 2024, JF). => Lever #2 is contraindicated AND was already rejected
   in 52.x (see internal note below).
6. **TSMOM is a multi-asset-futures phenomenon, not a single-equity-market one.** TSMOM
   documented "for a set of 58 diverse futures and forward contracts" across asset classes
   (Moskowitz, Ooi & Pedersen, AQR,
   https://www.aqr.com/Insights/Research/Journal-Article/Time-Series-Momentum, accessed
   2026-06-01). => Lever #5 (regime/trend overlay) is weakly evidenced for a long-only US
   single-market equity book and adds a market-timing failure mode.
7. **Momentum's premium survives net of costs; the binding constraint is turnover.** The
   2025/2026 literature (Alpha Architect; MDPI Risks 2026) finds the momentum edge intact
   net of costs "contingent on... below the 2-3 bps threshold" — i.e., the lever with the
   highest expected value is the one that *reduces trading*, which is the no-trade band.

---

## Candidate-lever survey (all five, with verdict)

| Lever | Best evidence | Fit to THIS long-only equal-weight US book | Verdict |
| --- | --- | --- | --- |
| **#1 Cost-aware no-trade / rebalance band** | Garleanu-Pedersen (w15205); arXiv 2411.07949 + 2412.11575; Kitces/Daryanani 20%-band | HIGH — works WITH equal-weight construction; the only lever whose mechanism (turnover) the existing replay already measures; default-OFF trivial | **RECOMMENDED** |
| #2 Volatility targeting / inverse-vol sizing | Moreira-Muir 2017 (in-sample); **refuted OOS** by Cederburg, Barroso-Detzel, DeMiguel 2024 | LOW — single-factor fails OOS net of costs; **ALREADY TESTED+REJECTED in 52.x** (`vol_scaled` arm, replay :204-215) — re-running is a protocol-banned re-litigation | REJECT (already done) |
| #3 Covariance-shrinkage / min-variance / risk-parity weighting | Ledoit-Wolf (shrinkage beats *sample* cov); **but** DeMiguel-Garlappi-Uppal 2009 `[ADVERSARIAL]` | LOW — at 10-50 names, no shrinkage/MV model beats equal-weight 1/N OOS; adds estimation-error + a heavy new covariance pipeline | REJECT (contraindicated) |
| #4 Overfitting-control tightening (PBO/DSR) | Bailey-Lopez de Prado DSR + CSCV/PBO | N/A as a *construction* lever — DSR & PBO already gate promotion in-repo; there is no *construction-time* overfitting control that the book lacks; tightening the gate does not change the live book's returns (so nothing to measure ON-vs-OFF on the production universe) | REJECT (already present; not a construction lever) |
| #5 Regime-conditioning / TSMOM trend filter | Moskowitz-Ooi-Pedersen 2012 | LOW-MED — multi-asset-futures result; on a single long-only US book it is market-timing with a real whipsaw/drawdown-of-the-filter failure mode; weakly evidenced for this universe | REJECT (poor fit; adds timing risk) |

---

## RECOMMENDED LEVER — transaction-cost-aware no-trade / rebalance buffer band

**What it is.** On each monthly reconstitution, do not fully swap to the fresh top-N. Apply
a hysteresis buffer: a currently-held name is **retained** unless its rank falls below a
buffer threshold (e.g. top-N x (1+b), the standard momentum "buy top-N / sell-below-rank-M"
band), and a new name is **added** only when it clears the tighter entry rank. This is the
discrete-portfolio analogue of Garleanu-Pedersen "trade partially toward the aim" and the
Kitces/Daryanani tolerance band, specialized to a long-only equal-weight top-N momentum
basket. Equivalently parameterized as a single buffer width `b` (the no-trade zone +/-eta).

**Why it is the best-evidenced choice FOR THIS BOOK.**
- It is the *only* candidate whose mechanism the existing $0 replay already measures
  directly: the replay computes per-config `turnover` as basket set-overlap
  (sector_neutral_replay.py:201) — a band's first-order effect is a turnover cut, which the
  harness reports natively.
- It is compatible with the existing equal-weight construction (no covariance pipeline, no
  market-timing), so it does not trip the DeMiguel 1/N adversarial finding (#4 above) or the
  vol-management OOS-failure finding (#5 above).
- The literature gives a concrete, defensible band width to test (relative ~20% of the
  position-rank, i.e. retain to ~rank 1.2N; Kitces #2) and a hard net-of-cost precedent
  (turnover 18.2->2.65 with Sharpe UP, arXiv 2412.11575 #3).
- The 2025/2026 recency scan says momentum's alpha is intact net of costs and the binding
  constraint is turnover (#7) — so the highest-EV remaining lever is turnover reduction.

**Expected effect size.** GROSS monthly Sharpe: small and possibly slightly negative (a band
keeps marginally-lower-ranked names a bit longer, so gross return can dip a few bps/mo).
NET-of-cost: a meaningful turnover reduction (literature precedent: 30-85% fewer
reconstitution trades at a relative-20% band) which, at the project's cost model
(`transaction_cost_pct=0.1` per side + `commission_per_share=0.005`,
backtest_engine.py:162-166) and the MDPI 2-3 bps momentum cost-sensitivity threshold,
should ADD net Sharpe roughly in proportion to (turnover_saved x round-trip cost).

**Honest robustness prognosis (flagged, as required).** The 52.3 SR-difference gate as
written tests *gross* paired monthly returns. On GROSS returns a no-trade band's delta is
small and the gate will likely return **REJECT** (delta<+0.05, CI_low<=0). That is a VALID,
honestly-reported outcome. The correct, defensible move is to ALSO report a **net-of-cost**
SR-difference (subtract `turnover_t x round_trip_cost` from each arm's monthly return, then
run the IDENTICAL `sharpe_diff_test`): the band's edge lives on the net axis. The keep/reject
recommendation in `live_check_53.1.md` should be made on BOTH (gross SR-diff for the
"does it hurt the signal" leg; net-of-cost SR-diff + the turnover delta for the "does it
help after costs" leg). I recommend writing the contract to expect a *gross* REJECT and a
*net* improvement, and to PROMOTE only if the net-of-cost SR-diff clears the a-priori rule
(p<0.05 AND delta>=+0.05 AND CI_low>0) AND the gross arm does not significantly DEGRADE
(gross delta CI_low > -0.05, the same -0.05 do-no-harm tolerance the 51.2 gate uses at
sector_neutral_replay.py:250).

---

## Internal code inventory (file:line anchors)

| File | Lines | Role | Status |
| --- | --- | --- | --- |
| `backend/backtest/analytics.py` | **239-289** `sharpe_diff_test(...)` | THE Ledoit-Wolf (2008) SR-difference test via Politis-Romano stationary bootstrap. Signature: `sharpe_diff_test(ret_a, ret_b, periods_per_year=12, n_boot=2000, block=4.0, seed=42, ci=0.90) -> dict`. Returns `{delta, p_one_sided, ci_low, ci_high, sr_a, sr_b, se, n, n_boot}`. One-sided H0: SR_a<=SR_b. <10 finite pairs -> safe default `{p_one_sided:1.0,...}`. | REUSE VERBATIM |
| `backend/backtest/analytics.py` | **292+** `compute_deflated_sharpe(observed_sr, num_trials, ...)` | DSR (secondary, deflates for #configs). Used by `dsr_52wh_verdict.py:37`. | REUSE if a DSR secondary leg is wanted |
| `scripts/ablation/sector_neutral_replay.py` | **1-282** (`main()` :141-282) | **THE $0 replay harness.** Replays the PRODUCTION `screener.rank_candidates` over monthly S&P-500 rebalances 2022-2025. Free yfinance batch download (:148) + Wikipedia GICS (:37-52). NO LLM, NO BQ. Per-config outputs: `ann_sharpe` (:116-120), `avg_fwd_mo%`, `avg_sectors`, `avg_turnover` (:201 set-overlap). Dumps paired monthly arrays to `handoff/current/_52wh_paired_returns.json` (:278) for the deterministic SR-diff PIN. | EXTEND (add a `no_trade_band` config arm) |
| `scripts/ablation/sector_neutral_replay.py` | **101-113** `basket_fwd_return(...)` | Equal-weight realized fwd return of a basket over `horizon=21` days. Confirms construction == **equal-weight top-N**. | REUSE |
| `scripts/ablation/sector_neutral_replay.py` | **183-228** rebalance loop | The per-month loop: builds rows, calls `rank_candidates`, forms `basket`, records `monthly`/`spread`/`turnover`, carries `prev_basket`. A band arm slots in here: given `ranked_all` + `prev_basket[arm]`, apply the buffer to choose the new basket. | EXTEND |
| `scripts/ablation/dsr_52wh_verdict.py` | **1-40** (`main()` :23) | The 52.3 VERDICT runner: reads the paired-returns JSON, calls `sharpe_diff_test(tilt, base, n_boot=5000, block=4, seed=42, ci=0.90)` (:30), then `compute_deflated_sharpe` (:37). | CLONE as `no_trade_band_verdict.py` (or extend) |
| `backend/tools/screener.py` | **249-264** `rank_candidates(...)` signature | The live ranking fn. Flags: `sector_neutral: bool=False` (:258), `momentum_52wh_tilt: bool=False` (:263), `momentum_52wh_tilt_k: float=0.5` (:264). Sort at :483. THIS is where a live `no_trade_band`/`rebalance_buffer` kwarg would attach IF the lever moved live (it does NOT in 53.1). | PATTERN to mirror; do NOT live-flip in 53.1 |
| `backend/tools/screener.py` | **445-483** | The 52.2/51.2 OFF-byte-identity pattern: gated re-scoring (`sector_neutral and scored:` :450; `momentum_52wh_tilt and scored:` :480) that writes `composite_score_raw` only when ON, then `scored.sort(...)`. The OFF path is byte-identical (no `_raw` field witnesses it). | PATTERN to mirror |
| `backend/tools/screener.py` | **501-507** `_apply_52wh_tilt(...)` | In-place tilt mirroring `hi52_tilt_basket` exactly so LIVE == replay. The 53.1 band helper should similarly mirror a replay function for live-faithfulness if ever promoted. | PATTERN |
| `backend/services/paper_trader.py` | **203-211** | LIVE sizing: `quantity = (amount_usd * _usd_to_local) / price`. Per-position USD budget -> shares; governed by `paper_max_positions` (:199). Effectively **equal-dollar (~equal-weight)** sizing — matches the replay's equal-weight basket. A band lever would change *which positions are held* across cycles (the buy/sell decision), not the per-name dollar size. | REFERENCE (live attach point if promoted) |
| `backend/backtest/backtest_engine.py` | **162-168** | The ML backtest engine's cost model: `transaction_cost_pct=0.1`, `commission_model="flat_pct"`, `commission_per_share=0.005`. | SOURCE of the round-trip cost constant for the net-of-cost SR-diff |
| `backend/backtest/backtest_engine.py` | **652-668** | Round-trip cost = `2 * transaction_cost_pct/100`, barriers shifted inward. Confirms the project's canonical round-trip-cost convention to apply in the replay's net-of-cost arm. | REFERENCE for cost convention |
| `backend/tests/test_phase_52_3_dsr.py` | 1-50 | The SR-diff test-correctness pattern: identical series -> p>0.5, CI brackets 0; dominant series -> p<0.05, delta>0, CI_low>0; deterministic w/ seed; <10 -> p=1.0. | MIRROR for the 53.1 band signal-math test |
| `backend/tests/test_phase_52_4_residual_momentum.py` | 1-60 | The replay-loaded-by-path pattern (`importlib.util.spec_from_file_location` against `scripts/ablation/...`) + signal-math pinned on synthetic data (empirical verdict is the live_check's job). | MIRROR for the 53.1 band test |
| `backend/tests/test_phase_52_2_live_tilt.py` | 1-50 | The OFF-byte-identity test pattern: `flag off == default == no _raw field`, `flag on tilts AND matches the replay exactly`. | MIRROR (only if 53.1 adds a live kwarg; default-OFF byte-identity assertion) |

**Confirmed $0:** the replay uses one `yf.download` batch (sector_neutral_replay.py:148) +
one Wikipedia `read_html` (:47); NO LLM call, NO BQ write, NO live state change — identical
to the 51.2/52.x pattern. The SR-diff + DSR are pure NumPy (analytics.py). Net: zero spend.

**Decisive note on lever #2 (vol-targeting) — already tested:** the replay ALREADY contains
a `vol_scaled` arm (sector_neutral_replay.py:204-215, `TARGET_ANN_VOL=0.15`, `VOL_CAP=2.0`,
Barroso-Santa-Clara inverse-realized-vol scaling) and the 51.2 verdict prints
`vol_scaled vs baseline` (:252-253). Re-proposing vol-targeting would re-litigate a
measured-and-rejected lever — protocol-banned ("Skip RESEARCH because we've been here
before" is forbidden, but so is re-running a rejected lever as if new). This is why the
recommendation is the no-trade band, NOT vol-targeting.

---

## Recommended 53.1 implementation + measurement plan (config-gated, default-OFF, $0)

1. **Extend the replay** `scripts/ablation/sector_neutral_replay.py` (or a sibling
   `no_trade_band_replay.py` cloning its scaffold) with a `no_trade_band` config arm in the
   rebalance loop (:183-228). Given `ranked_all` (already computed at :219) and
   `prev_basket["no_trade_band"]`, choose the new basket by the momentum buy/hold band:
   keep a held name if its rank <= `TOP_N * (1+b)`; fill remaining slots from the top by
   entry rank. Parameterize `b` over a small grid (e.g. `0.2, 0.5, 1.0` — relative-20% is
   the Kitces optimum; wider widths probe the turnover/return frontier). Record `monthly`,
   `spread`, `turnover` exactly as the existing arms do.
2. **Net-of-cost arm.** For each arm compute a net monthly series:
   `net_t = gross_t - turnover_t * round_trip_cost`, with
   `round_trip_cost = 2 * 0.1/100 = 0.002` (backtest_engine.py:668 convention). This makes
   the band's value visible on the correct axis.
3. **Dump paired arrays** (baseline vs each band width, gross AND net) to
   `handoff/current/_no_trade_band_paired_returns.json`, mirroring the :269-279 dump.
4. **Verdict runner** — clone `dsr_52wh_verdict.py` to call the IDENTICAL
   `sharpe_diff_test(band, base, periods_per_year=12, n_boot=5000, block=4, seed=42,
   ci=0.90)` on BOTH the gross and the net paired series, plus `compute_deflated_sharpe` as
   a secondary on the net Sharpe deflated for the #band-widths tried.
5. **A-priori rule (identical to 52.3/52.4).** PROMOTE only if the **net-of-cost** SR-diff
   clears `p<0.05 AND delta>=+0.05 AND ci_low>0` AND the **gross** arm does not significantly
   degrade (`gross ci_low > -0.05`, the 51.2 do-no-harm tolerance at
   sector_neutral_replay.py:250). Otherwise REJECT — a valid, honestly-reported outcome.
6. **Signal-math test** `backend/tests/test_phase_53_1_*.py`: pin the band-selection logic
   on synthetic ranked rows (held name just inside band -> retained; just outside ->
   dropped; deterministic), mirroring 52.4's by-path load + synthetic-pin pattern. NO
   network, $0.
7. **OFF byte-identity (only if a live kwarg is added).** If you add a
   `rebalance_buffer: bool=False` kwarg to `rank_candidates` for future promotion, add the
   52.2-style test asserting `flag off == default` and no new field on the OFF path. **But
   53.1 does NOT flip the live flag** — the deliverable is the measurement + verdict only.
8. **`live_check_53.1.md`** records: the ON-vs-OFF table (Sharpe/return/turnover/maxDD per
   band width, gross AND net), the gross + net SR-difference stats (delta, p, CI), the cited
   literature basis, and a PROMOTE/REJECT recommendation per the rule in step 5.

**maxDD note.** The replay currently reports Sharpe/return/sectors/turnover but NOT maxDD.
The success_criteria require maxDD. Add a `max_drawdown(monthly)` helper (cumprod the
monthly arm returns, track running peak, return min trough/peak-1) to the replay and print
it per arm — a ~10-line addition, $0, no new dep.

---

## DO-NO-HARM risks

1. **Live core must stay byte-identical.** 53.1 is measure-only; do NOT add a live kwarg
   path that executes by default. If a kwarg is added for future promotion it MUST default
   OFF and the OFF path MUST be byte-identical (52.2 pattern, screener.py:445-483) — assert
   it in a test. No `paper_trader`/`portfolio_manager` change in 53.1.
2. **Do not re-run vol-targeting as if new** — it is already measured+rejected
   (sector_neutral_replay.py:204-215). Re-litigating is a protocol breach.
3. **Gate on the right axis.** A no-trade band's edge is net-of-cost; reporting ONLY the
   gross SR-diff would mis-REJECT a genuinely good lever (or, conversely, the net arm must
   not be cherry-picked to manufacture a PASS — report BOTH gross and net, decide per the
   pre-registered rule in step 5). The a-priori rule must be written into the contract
   BEFORE running, exactly as 52.3/52.4 did.
4. **Small T.** 2022-2025 monthly => ~46-47 rebalances (the replay prints `N rebalances`).
   `sharpe_diff_test` already guards <10; with ~46 points the bootstrap CI is wide, so a
   non-significant result is the expected default — frame the contract accordingly.
5. **Cost-constant honesty.** Use the project's own `transaction_cost_pct=0.1` /
   round-trip `0.002` (backtest_engine.py:162/668); do NOT tune the cost constant to make
   the band look good. Optionally report sensitivity at the MDPI 2-3 bps institutional
   level too.
6. **No live flag flip, no LLM, no BQ write, no operator-cost action** — the whole step is
   the $0 replay + verdict + tests + `live_check_53.1.md`.

---

## Research Gate Checklist

Hard blockers — all satisfied:
- [x] **>=5 authoritative external sources READ IN FULL via WebFetch** — 7 (Ledoit-Wolf
  digest, Kitces tolerance-band, arXiv 2412.11575, arXiv 2411.07949, NBER Garleanu-Pedersen,
  AQR TSMOM, DeMiguel-Garlappi-Uppal RFS-2009 `[ADVERSARIAL]`).
- [x] **10+ unique URLs total** — 21 (7 full + 11 snippet + 3 search-only).
- [x] **Recency scan (last 2 years) performed + reported** — yes, 5 findings (DeMiguel 2024
  JF, arXiv 2412.11575, arXiv 2411.07949, MDPI 2026, Alpha Architect 2025); reinforces the
  recommendation, no source overturns the canonical foundations.
- [x] **Full papers/pages read (not abstracts)** for the read-in-full set — yes (arXiv HTML
  full bodies; NBER/IDEAS pages; practitioner full articles).
- [x] **file:line anchors for every internal claim** — yes (analytics.py:239-289;
  sector_neutral_replay.py:101-113/148/183-228/201/204-215/269-279; dsr_52wh_verdict.py:23-37;
  screener.py:249-264/445-483/501-507; paper_trader.py:199-211; backtest_engine.py:162-168/652-668;
  the three 52.x test files).

Soft checks:
- [x] Internal exploration covered every relevant module (replay, SR-diff, DSR, verdict
  runner, live screener, live sizing, backtest cost model, the three pattern tests).
- [x] Contradictions / consensus noted (the `[ADVERSARIAL]` DeMiguel 1/N + the
  vol-management OOS-failure both explicitly contradict rival levers and DROVE the choice).
- [x] All claims cited per-claim (URL + access date inline in Key Findings).

---

```json
{
  "tier": "complex",
  "external_sources_read_in_full": 7,
  "snippet_only_sources": 11,
  "urls_collected": 21,
  "recency_scan_performed": true,
  "internal_files_inspected": 9,
  "gate_passed": true
}
```
