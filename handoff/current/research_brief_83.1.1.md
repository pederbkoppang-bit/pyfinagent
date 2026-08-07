# Research Brief -- step 83.1.1 (tier=moderate)

**Topic:** MEASURE the phase-83 gate arithmetic before building -- required-SR
for DSR>=0.95 via the repo's own `compute_deflated_sharpe`, V measured from an
actually-run trial set, independent-label-span counts at horizon 126,
`compute_pbo_checked` feasibility, and a kill rule written FIRST.

**Status:** COMPLETE. Gate PASSED (7 sources read in full, 27 URLs, recency scan done).
**Started:** 2026-08-07

Assumption stated up front: tier = `moderate` (caller-specified). Non-audit-class
(caller did not set `coverage.audit_class`), so `coverage` is informational.

---

## 0. Scratch / progress log

- [x] Read `.claude/agents/researcher.md` in full
- [x] Read `.claude/rules/research-gate.md` in full
- [x] Read masterplan step 83.1.1 + verification criteria verbatim
- [x] Internal: `backend/backtest/analytics.py::compute_deflated_sharpe` FULL
- [x] Internal: `compute_pbo_checked`
- [x] Internal: 82.3 candidate backtests -> per-trial Sharpe inventory (V)
- [x] Internal: `preregistration_phase83_ranking.json`
- [x] Internal: `test_phase_83_1_design_pack.py` mtime idiom
- [x] Prototype the DSR->required-SR inversion (printed output)
- [x] External: DSR / expected-max / V semantics (7 read in full)
- [x] External: overlapping-label independent-observation counting
- [x] Recency scan 2024-2026

---

## 1. Internal code inventory

| File | Lines | Role | Status |
|---|---|---|---|
| `backend/backtest/analytics.py` | 384-447 | `compute_deflated_sharpe` -- the ONLY DSR implementation | live, 1 prod caller |
| `backend/backtest/analytics.py` | 766-777 | the repo's canonical DSR call site (`generate_report`) | live |
| `backend/backtest/analytics.py` | 752-754 | how the repo derives `variance_of_srs` today | live -- **semantic mismatch, see 1.3** |
| `backend/backtest/analytics.py` | 208-273 | `compute_pbo_checked` (refuses instead of false-good 0.0) | live |
| `backend/backtest/analytics.py` | 276-328 | `compute_pbo` (raw CSCV; returns 0.0 on undersized input) | live |
| `backend/backtest/analytics.py` | 197-205 | `PBO_CEILING_LIVE=0.20`, `PBO_MIN_TRIALS_GATE_GRADE=10` | live constants |
| `backend/backtest/backtest_engine.py` | 274, 962 | `holding_days=90` x `horizon_days=int(holding_days*1.5)` -> 135d purge | live |
| `backend/backtest/experiments/preregistration_phase83_ranking.json` | whole | 83.1 pre-registration: horizon 126, cap 45, ranking | frozen, hash-recorded |
| `backend/tests/test_phase_83_1_design_pack.py` | 59-65, 186-228 | strict `st_mtime_ns` ordering guard + its mutation test | REUSE VERBATIM |
| `backend/tests/test_phase_83_0_3_pbo_false_pass.py` | 85-101 | monkeypatch **routing spy** idiom | REUSE for criterion 2 |
| `backend/backtest/experiments/results/20260804T025319Z_phase_82_3_full_sample_3strat.json` | 3x8 runs | actually-run trial set (2018-2025) | **the V source** |
| `backend/backtest/experiments/results/20260804T041628Z_phase_82_3_short_window_4strat.json` | 4x8 runs | actually-run trial set (short window) | second V source |
| `scripts/harness/run_82_3_candidate_backtests.py` | 12,632 B | producer of the above | reference only |
| `backend/autoresearch/strategy_backtest_adapter.py` | 159, 262-268 | the gate-side DSR/PBO wrapper | live |

### 1.1 `compute_deflated_sharpe` -- exact shape (analytics.py:384-447)

```python
def compute_deflated_sharpe(
    observed_sr: float, num_trials: int, variance_of_srs: float = 0.5,
    skewness: float = 0.0, kurtosis: float = 3.0,
    T: int = 252, periods_per_year: int = 1) -> float
```

Mechanics, line by line:

- `:417` guard -- returns **0.0** if `num_trials < 1 or T < 10 or observed_sr == 0`.
- `:422-424` de-annualization: `sr = observed_sr/sqrt(ppy)`, `var_srs = variance_of_srs/ppy`.
  So `observed_sr` is passed **ANNUALIZED** and `variance_of_srs` is likewise an
  **annualized variance of annualized trial Sharpes**.
- `:429-432` expected max:
  `e_max = sqrt(var_srs) * ((1-0.5772)*Phi^-1(1-1/max(N,2)) + 0.5772*Phi^-1(1-1/(max(N,2)*e)))`
  -- the Bailey/LdP Euler-Mascheroni two-quantile form, gamma hardcoded as the
  literal `0.5772`.
- `:435-437` `se_sr = sqrt((1 - skew*sr + (kurt-1)/4*sr^2)/T)` -- **T is the
  PER-PERIOD observation count** (daily return count when ppy=252). T enters ONLY here.
- `:443` `z = (sr - e_max)/se_sr`; `:446` `dsr = Phi(z)`.

**TRAP 1 (confirmed, decisive for criterion 1): `max(num_trials, 2)` at `:430-431`
means N=1 and N=2 produce the IDENTICAL DSR.** Criterion 1 mandates N in
`{1, 10, 45, 100}`; the N=1 row is therefore not "no deflation" -- it is the N=2
row. The step artifact must RECORD this collapse rather than present N=1 as an
undeflated baseline. (Independently recorded for 82.25.)

**TRAP 2: T enters only through `se_sr`, and `e_max` has no T at all.** So the
annualized required-SR decomposes exactly as
`SR*_ann = sqrt(V) * E[max_N] + z* * sqrt(ppy) * se_pp`, confirming the step
name's claim that the deflation FLOOR is T-independent and trial-count-driven.
The T term is the only lever history buys.

### 1.2 `compute_pbo_checked` -- exact shape (analytics.py:208-273)

`compute_pbo_checked(pnl_matrix, S=16) -> dict` with keys
`pbo, n_trials, n_obs, gate_grade, column_corr_mean, column_corr_max,
columns_diverse, refused`. Refusal paths: not 2-D; `N < 2`; `T < S*2`.
`gate_grade = N >= 10` (`PBO_MIN_TRIALS_GATE_GRADE`). `columns_diverse =
corr_mean < 0.99`. NOTE the refusal branches return a dict WITHOUT the
`column_corr_*`/`columns_diverse` keys at all -- criterion 5 records
"verbatim", so the recorder must use `.get()` and record absence as absence.

### 1.3 FINDING: the repo's live V is NOT a cross-trial dispersion

`analytics.py:752-754`:

```python
window_sharpes = [w.sharpe_ratio for w in result.windows if w.sharpe_ratio != 0]
sr_variance = float(np.var(window_sharpes)) if len(window_sharpes) > 1 else 0.5
```

This is the variance of per-WINDOW Sharpes **inside a single run** -- a
time-dispersion, not the Bailey/LdP dispersion of Sharpes **across the N trials
of the search**. It also uses `np.var` (ddof=0, population) and silently falls
back to the 0.5 default at <2 windows. 83.1.1 measures the correct quantity;
the divergence from the live path is a recordable finding (and a candidate
follow-up step), not something to fix in this step.

---
## 2. External research

### 2.1 Read in full (>=5 required; counts toward the gate) -- 7

| # | URL | Accessed | Kind | Fetched how | Key finding |
|---|---|---|---|---|---|
| 1 | https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf | 2026-08-07 | peer-reviewed (JPM 2014) | WebFetch returned binary -> **pypdf 6.10.2 local extraction, 22 pp / 47,819 chars** | Eq.(1) VERBATIM: `E[max{SR_n}] ~ E[{SR_n}] + sqrt(V[{SR_n}])((1-g)Z^-1(1-1/N) + g Z^-1(1-1/(Ne)))`. **V is a VARIANCE** (`V[{SR_n}]`, sqrt applied). Author's own code: `maxZ=(1-emc)*ss.norm.ppf(1-1./numTrials)+emc*ss.norm.ppf(1-1./(numTrials*np.e)); return mu+sigma*maxZ` |
| 2 | https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf | 2026-08-07 | peer-reviewed (2015) | WebFetch binary -> **pypdf, 34 pp / 64,198 chars** | CSCV: `S=16` -> "12,780 combinations"; logit `l_c = ln(w_c/(1-w_c))`; PBO = "the rate at which optimal IS strategies underperform the median of the OOS trials"; worked cases at PBO 74% (overfit) and 0.04% (real) |
| 3 | https://arxiv.org/pdf/2507.07107 | 2026-08-07 | preprint, Jul 2025 | WebFetch binary -> **pypdf, 18 pp / 49,239 chars** | RECENCY: cites Bailey/LdP DSR + "Harvey et al. raised multiple-testing concerns over 300+ published factors" but **applies no measured V and reports no DSR** -- a live example of the exact gap 83.1.1 closes |
| 4 | https://rdrr.io/github/braverock/quantstrat/man/SharpeRatio.deflated.html | 2026-08-07 | reference implementation (R) | WebFetch, full page | `.deflatedSharpe(sharpe, nTrials, varTrials, skew, kurt, numPeriods, periodsInYear=252)`; `varTrials` = *"variance of Sharpe ratios of the trials"*; **"the documentation does not explicitly explain how to calculate or obtain varTrials"** -- the canonical gap |
| 5 | https://quanterlab.com/articles/foundations-dsr | 2026-08-07 | practitioner | WebFetch, full page | Calibration ladder (>0.95 strong / 0.80-0.95 borderline / <0.50 artifact); explicitly does NOT say how to estimate V; a 100-cell sweep with zero true edge yields expected max SR ~2.0 |
| 6 | https://reasonabledeviations.com/notes/adv_fin_ml/ | 2026-08-07 | academic notes (AFML Ch.4/Ch.7) | WebFetch, full page | *"Labels in finance are not IID ... different labels may look at the same set of returns"*; average uniqueness -> effective sample size; purge = drop training labels whose information horizon overlaps the test label |
| 7 | https://www.mql5.com/en/articles/19850 | 2026-08-07 | practitioner, **2025-10-29** | WebFetch, full page | RECENCY: concurrency = count active events per bar; uniqueness = mean reciprocal concurrency over the lifespan; worked example `(1/3+1/4+1/3+1/2)/4 ~ 0.354`; **no closed-form span/horizon rule is offered** |

### 2.2 Identified but snippet-only (does NOT count toward the gate)

| URL | Kind | Why not fetched in full |
|---|---|---|
| https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551 | SSRN landing | abstract page only; the full text is source #1 |
| https://en.wikipedia.org/wiki/Deflated_Sharpe_ratio + `..._Ratio` | encyclopaedia | **both casings returned HTTP 404** on 2026-08-07 -- recorded as an attempted-and-failed fetch |
| https://www.pm-research.com/content/iijpormgmt/40/5/94 | journal of record | paywalled |
| https://www.pm-research.com/content/iijpracapp/2/3/110 | "Practical Applications" | paywalled |
| https://quantdare.com/deflated-sharpe-ratio-how-to-avoid-been-fooled-by-randomness/ | blog | duplicate of source #1's formula |
| https://marti.ai/qfin/2018/05/30/deflated-sharpe-ratio.html | blog | duplicate |
| https://sdm.lbl.gov/oapapers/ssrn-id2507040-bailey.pdf | paper (LBNL) | same author group, superseded by #1/#2 for this question |
| http://boston.qwafafew.org/wp-content/uploads/sites/4/2017/01/Lopez_de_Prado_Sharpe.pdf | slides | MinTRL slides; not the V question |
| https://pdfs.semanticscholar.org/c215/d0a2064ce1a3565d276475abc84305418f0f.pdf | paper | duplicate of #1 |
| https://www.foliolab.ai/docs/metrics/deflated-sharpe-ratio | vendor doc | low tier |
| https://paperswithbacktest.com/course/deflated-sharpe-ratio | course | low tier |
| https://medium.com/balaena-quant-insights/deflated-sharpe-ratio-dsr-33412c7dd464 | blog | low tier |
| https://moonsat.medium.com/how-top-quants-find-winning-strategies-and-reject-the-other-99-complete-framework-800c8dd295ef | blog, Jun 2026 | recency hit, low tier |
| https://quantdecoded.com/en/the-sharpe-ratio-measuring-risk-adjusted-returns | blog | low tier |
| https://arxiv.org/pdf/2512.22476 (AutoQuant) | preprint | recency hit; crypto auto-tuning, tangential |
| https://toc.library.ethz.ch/objects/pdf03/e01_978-1-119-48208-6_01.pdf | book TOC | TOC only |
| https://www.garp.org/hubfs/Whitepapers/a1Z1W0000054x6lUAA.pdf | whitepaper | "10 reasons ML funds fail" -- background |
| https://www.quantconnect.com/forum/discussion/565/... | forum | community tier |
| https://www.nomura.com/events/.../10_00_Marcos_Lopez_de_Prado_20150510.pdf | slides | duplicate |
| https://www.quantmemo.com/concepts/triple-barrier-labeling | blog | background |

**URLs collected: 27 unique** (7 read in full + 20 snippet-only).

### 2.3 Search-query composition (3-variant discipline)

1. **Year-less canonical** -- `deflated Sharpe ratio Bailey Lopez de Prado expected maximum Sharpe Euler-Mascheroni variance of trial Sharpe ratios`
2. **Year-less canonical (2nd topic)** -- `Lopez de Prado overlapping labels uniqueness number of independent observations concurrency financial machine learning`
3. **Last-2-year window** -- `deflated Sharpe ratio multiple testing trial count 2025 2026 backtest overfitting practice`
4. **Current-year frontier** -- `required Sharpe ratio threshold number of trials deflated Sharpe 2026 quant research effective trials`

### 2.4 Recency scan (2024-2026) -- PERFORMED

Result: **3 new findings in the window; NONE supersedes the 2014/2015 canonical formulation.**

- `arXiv:2507.07107` (Jul 2025, source #3) cites Bailey/LdP and Harvey's 300+-factor multiple-testing critique but reports **no DSR and no V estimate**. Confirms the practice gap rather than closing it.
- MQL5 label-concurrency article (2025-10-29, source #7) restates AFML Ch.4 concurrency/uniqueness unchanged; still **no closed-form "span / horizon" rule** in the literature -- criterion 4's `span_days / horizon` is a deliberately conservative simplification of average-uniqueness, and the brief must say so.
- `arXiv:2512.22476` (AutoQuant, crypto auto-tuning) and a Jun-2026 practitioner piece restate the "more trials -> higher required SR" result with no new estimator.

**Conclusion: the 2014 Eq.(1) and the 2015 CSCV algorithm remain the state of the art; the repo's implementations are faithful to both.** No plan change required on recency grounds.

### 2.5 CRITICAL METHOD WARNING (recorded because it nearly corrupted this brief)

`WebFetch` on `deflated-sharpe.pdf` returned a **fabricated** summary: it asserted
`E[max SR_N] ~ (1-g)/sqrt(T) * sqrt(2 ln N) + SR`, that *"V represents the standard
deviation of trial Sharpe ratios, not variance"*, and that the deflation threshold
**depends on T**. All three are FALSE against the extracted source text. The correct
readings are Eq.(1) above; V is a variance; and the `E[max]` term contains no T at
all. This is the `.claude/rules/research-gate.md` "Binary PDF" failure mode producing
confident wrong text rather than an empty result -- **more dangerous than a refusal.**
Every PDF claim in this brief comes from the pypdf-extracted `.txt`, not the fetch summary.

---

## 3. MEASUREMENTS (all figures recorded, none asserted against a target)

### 3.1 V measured from the ACTUALLY-RUN 82.3 trial sets

Source A -- `results/20260804T025319Z_phase_82_3_full_sample_3strat.json`, sample
**2018-01-01..2025-12-31**, 3 strategies x K=8 configs = **24 runs**:

| strategy | K | Sharpes | var ddof=0 | var ddof=1 | pbo | matrix |
|---|---|---|---|---|---|---|
| triple_barrier | 8 | 0.6090, 0.6090, 0.5376, 0.6127, 0.6127, 0.5662, 0.5615, 0.5663 | 0.000774 | 0.000884 | 0.7486 | (1661, 8) |
| stretch_regime | 8 | 0.6858, 0.3909, 0.8246, 0.6685, 0.4522, 0.4604, 0.4751, 0.6077 | 0.019538 | 0.022329 | 0.1960 | (1535, 8) |
| reversion_sigma | 8 | 0.6781, 0.5828, 0.5597, 0.5597, 0.5359, 0.5997, 0.5206, 0.4837 | 0.002977 | 0.003402 | 0.3968 | (1661, 8) |
| **POOLED** | **24** | mean 0.5733, min 0.3909, max **0.8246** | **0.007829** | **0.008169** | -- | -- |

Source B -- `results/20260804T041628Z_phase_82_3_short_window_4strat.json`, sample
**2024-07-01..2025-12-31**, 4 x 8 = **32 runs**: pooled mean 1.3060, **V(ddof=1) = 0.167921**
(ddof=0 0.162673); per-strategy V 0.024164..0.164901; matrices (119, 8) and (55, 8).

**Finding: V is ~20x larger on the short window than on the full sample** (0.1679 vs
0.0082). Trial-Sharpe dispersion on a short sample is dominated by *estimation* noise,
not by genuine cross-config differences. Any V borrowed from a short window inflates
the deflation floor. RECORD both; do not average them.

**Honest caveat (must be carried into the artifact):** both figures come from K=8
configs of 3-4 *existing* strategy families, not from a 45-trial thematic search.
They are the best available empirical prior, NOT the phase-83 V. 83.5 must re-measure
V on its own trial set.

### 3.2 Required annualized Sharpe for DSR >= 0.95 -- via `compute_deflated_sharpe`

Method: **bisection over the repo function** (200 iterations, bracket [1e-9, 50],
`periods_per_year=252`, `skewness=0.0`, `kurtosis=3.0`). Monotonicity of DSR in
`observed_sr` verified empirically over 4,000 points before inverting (True).
**16,080 invocations of `compute_deflated_sharpe`** in the prototype run.

T grid from `exchange_calendars` XNYS `sessions_in_range` (calendar constructed with
`start="1980-01-01"` -- **default XNYS bounds begin 2006-08-07 and raise
`DateOutOfBounds` on EDGAR/GPR starts**):

| source | window | cal days | **T sessions** | years | **spans @126** | spans @135 cal |
|---|---|---|---|---|---|---|
| GDELT gkg_partitioned | 2015-02-17..2026-08-04 | 4186 | **2883** | 11.46 | **22.88** | 31.01 |
| SEC EDGAR full-text | 2001-01-01..2026-08-04 | 9346 | **6434** | 25.59 | **51.06** | 69.23 |
| Caldara-Iacoviello GPR | 1985-01-01..2026-08-04 | 15190 | **10477** | 41.59 | **83.15** | 112.52 |
| 82.3 full-sample (CONTROL, not a free-source window) | 2018-01-01..2025-12-31 | 2921 | 2011 | 8.00 | 15.96 | 21.64 |

**Required annualized SR (recorded, not asserted):**

| V | T | N=1 | N=2 | N=10 | N=45 | N=100 |
|---|---|---|---|---|---|---|
| **0.5** (module default, NOT a measurement) | 2883 | 0.8542 | 0.8542 | 1.6009 | **2.0692** | 2.2782 |
| 0.5 | 6434 | 0.6932 | 0.6932 | 1.4396 | 1.9075 | 2.1164 |
| 0.5 | 10477 | 0.6227 | 0.6227 | 1.3690 | 1.8368 | 2.0456 |
| 0.5 | 2011 | 0.9503 | 0.9503 | 1.6973 | 2.1658 | 2.3749 |
| *floor sqrt(V)E[max_N] (T-free)* | -- | 0.3675 | 0.3675 | 1.1134 | 1.5808 | 1.7894 |
| **0.008169** (MEASURED, full-sample pooled n=24) | 2883 | 0.5334 | 0.5334 | 0.6288 | **0.6886** | 0.7153 |
| 0.008169 | 6434 | 0.3725 | 0.3725 | 0.4679 | 0.5277 | 0.5543 |
| 0.008169 | 10477 | 0.3021 | 0.3021 | 0.3975 | 0.4572 | 0.4839 |
| 0.008169 | 2011 | 0.6295 | 0.6295 | 0.7249 | 0.7847 | 0.8114 |
| *floor (T-free)* | -- | 0.0470 | 0.0470 | 0.1423 | 0.2021 | 0.2287 |
| **0.167921** (MEASURED, short-window pooled n=32) | 2883 | 0.6995 | 0.6995 | 1.1322 | **1.4034** | 1.5244 |
| 0.167921 | 6434 | 0.5386 | 0.5386 | 0.9711 | 1.2421 | 1.3631 |
| 0.167921 | 10477 | 0.4681 | 0.4681 | 0.9005 | 1.1716 | 1.2925 |
| 0.167921 | 2011 | 0.7956 | 0.7956 | 1.2284 | 1.4997 | 1.6208 |
| *floor (T-free)* | -- | 0.2130 | 0.2130 | 0.6452 | 0.9161 | 1.0370 |
| **0.022329** (MEASURED, within-strategy max) | 2883 | 0.5641 | 0.5641 | 0.7218 | 0.8207 | 0.8648 |
| 0.022329 | 10477 | 0.3328 | 0.3328 | 0.4904 | 0.5893 | 0.6333 |
| *floor (T-free)* | -- | 0.0777 | 0.0777 | 0.2353 | 0.3341 | 0.3781 |

**Three decisive facts:**

1. **N=1 and N=2 are byte-identical in every row** -- `max(num_trials, 2)` at
   `analytics.py:430-431`. The N=1 column is NOT an undeflated baseline. (The paper's
   own formula is undefined at N=1: `Z^-1(1-1/1) = Z^-1(0) = -inf`; the clamp is a
   sound guard, but its consequence must be recorded, not presented as a result.)
2. **The floor rows carry no T.** Extending history 2883 -> 10477 sessions (11.5y ->
   41.6y, +263%) cuts the required SR at N=45/V=0.5 from 2.0692 to 1.8368 (-11%).
   Cutting N from 45 to 10 at the same V cuts it 2.0692 -> 1.6009 (-23%). **Trial
   budget beats archive depth**, exactly as 83.1 pre-registered.
   [MAIN CORRECTION 2026-08-07, 83.1.1 cycle 4: both percentages above are the V=0.5
   slice, as the arithmetic on these lines itself states -- but the CONCLUSION is
   V-conditional and REVERSES at the measured V=0.008169: -33.6% history vs -8.7%
   trials, so archive depth dominates at the measured V. Queued against the 83.1
   preregistration text as 83.1.5.]
3. **V dominates both.** At N=45/T=2883 the required SR spans **0.6886 -> 2.0692
   (3.0x)** across the measured-vs-default V range. This single unmeasured number is
   what produced the two contradictory 2026-08-04 verdicts.

### 3.3 PBO feasibility via `compute_pbo_checked` (payloads verbatim)

Shape feasibility (`S=16`; synthetic matrices -- shape-only properties):

| T | N | `refused` | `gate_grade` | keys returned |
|---|---|---|---|---|
| 2883 | 45 | `None` | `True` | 8 keys incl. `column_corr_*`, `columns_diverse` |
| 2883 | 10 | `None` | `True` | 8 keys |
| 2883 | 9 | `None` | **`False`** | 8 keys |
| 2883 | 2 | `None` | `False` | 8 keys |
| 2883 | 1 | `'N=1 < 2; compute_pbo would return a false-good 0.0'` | `False` | **5 keys only** |
| 119 | 8 | `None` | `False` | 8 keys |
| 32 | 10 | `None` | `True` | 8 keys |
| 31 | 10 | `'T=31 < S*2=32; compute_pbo would return a false-good 0.0 that PASSES the ceiling'` | `False` | **5 keys only** |

=> On any free source the T constraint (`T >= 32`) is trivially met (2883 / 6434 /
10477). **The entire PBO constraint reduces to N >= 10 AND column diversity.**

Intended matrix shape **(T=2883, N=45)**, verbatim payloads:

```json
{
 "independent_columns":    {"pbo": 0.3059052059052059, "n_trials": 45, "n_obs": 2883, "gate_grade": true,
                            "column_corr_mean": 0.0010598741494548662, "column_corr_max": 0.05544104465329114,
                            "columns_diverse": true, "refused": null},
 "near_duplicate_columns": {"pbo": 0.7042735042735043, "n_trials": 45, "n_obs": 2883, "gate_grade": true,
                            "column_corr_mean": 0.997539110834522, "column_corr_max": 0.9977523457893794,
                            "columns_diverse": false, "refused": null},
 "mild_config_spread":     {"pbo": 0.7757575757575758, "n_trials": 45, "n_obs": 2883, "gate_grade": true,
                            "column_corr_mean": 0.9183022470396465, "column_corr_max": 0.9246776737051763,
                            "columns_diverse": true, "refused": null}
}
```python
# backend/backtest/gate_feasibility.py
def required_annualized_sr(num_trials, T, variance_of_srs, target_dsr=0.95,
                           periods_per_year=252, skewness=0.0, kurtosis=3.0,
                           lo=1e-9, hi=50.0, iters=200):
    from backend.backtest import analytics          # FUNCTION-SCOPED, resolved at CALL time
    f = analytics.compute_deflated_sharpe           # attribute lookup, NOT a bound name
    ...
```

A module-level `from ... import compute_deflated_sharpe` **binds the name at import
time and makes `monkeypatch.setattr(analytics, ...)` invisible** -- criterion 2 would
pass vacuously. This is the exact trap the 83.0.3 test comment documents
(`test_phase_83_0_3_pbo_false_pass.py:88-90`).

Inversion method: **bisection, not closed form.** Closed form exists at skew=0/kurt=3
(the z-equation is quadratic in SR) but would *reimplement the deflation*, which
criterion 2 forbids. Bisection is monotone-safe: verified True over 4,000 points, and
provable -- `d/dSR[(SR-e)/sqrt((1+SR^2/2)/T)] > 0` for `e >= 0, SR >= 0`.

### 7.2 The kill rule -- written FIRST, write-once

**File:** `backend/backtest/experiments/killrule_phase83.json`
(deliberately NOT under `results/`: it must not join the 83.1 artifact population,
whose globs + `phase_tag` content rule only scan `results/`).

Shape -- each clause is `(recorded_quantity, comparison, threshold, threshold_source, stops)`:

| id | recorded quantity | comparison | threshold | threshold source |
|---|---|---|---|---|
| K1 | `required_annualized_sr[N=trial_budget_cap, T=T_chosen_source, V=V_measured_on_phase83_trials]` | `>` | `0.8246` | **measured**: best full-sample Sharpe the repo has produced across the 82.3 24-run trial set (`stretch_regime[...]`, 2018-2025). If the gate demands more than the repo's best-ever full-sample result, it is out of reach. |
| K2 | `independent_label_spans_at_126[chosen_source]` | `<` | `16` | `compute_pbo(S=16)` partitions T into 16 ordered subsets; fewer independent spans than subsets means at least one CSCV subset contains no independent observation. |
| K3 | `pbo_checked.refused is not None` OR `gate_grade is False` OR `pbo > PromotionGate.max_pbo` | any true | `0.20` read at runtime from `backend/autoresearch/gate.py:22` | live gate, never hardcoded |
| K4 | `actual_trial_count` | `>` | `45` | `preregistration_phase83_ranking.json::ranking_criteria.trial_budget_cap` |

**K1 discriminates on the numbers already measured** -- and that is the point:
V=0.008169 -> 0.6886 < 0.8246 CONTINUE; V=0.167921 -> 1.4034 > 0.8246 STOP;
V=0.5 -> 2.0692 > 0.8246 STOP. A rule that cannot fire either way is not a kill rule.
**K2 is likewise live:** GDELT 22.88 passes, EDGAR 51.06 and GPR 83.15 pass, and the
82.3 control at 15.96 would FAIL -- proof the threshold is not cosmetic.

### 7.3 mtime ordering + mutation mechanics (criteria 6-7)

Reuse `test_phase_83_1_design_pack.py:59-65` verbatim (strict `<` on `st_mtime_ns`;
equal mtimes PASS, because a fresh `git checkout` stamps every file identically -- so
the guard is real locally and inert in a clean CI clone; **disclose this**).

```python
def _assert_kill_rule_predates_all_results() -> None:
    k_ns = KILL.stat().st_mtime_ns
    arts = _step_artifacts()
    assert arts, "artifact population EMPTY -- criterion 6 would be vacuously green"
    for a in arts:
        assert not a.stat().st_mtime_ns < k_ns, (
            f"{a.name} predates the kill rule ({a.stat().st_mtime_ns} < {k_ns})")

def test_c7_touching_kill_rule_forward_makes_c6_fail():
    orig = KILL.stat().st_mtime_ns                      # SAVE
    try:
        t = max(a.stat().st_mtime_ns for a in _step_artifacts()) + 60_000_000_000
        os.utime(KILL, ns=(t, t))                       # mutate FORWARD
        with pytest.raises(AssertionError):
            _assert_kill_rule_predates_all_results()
    finally:
        os.utime(KILL, ns=(orig, orig))                 # NON-DESTRUCTIVE restore
```

The `finally` restore is mandatory: 83.1 mutates a throwaway artifact and `unlink`s it,
but 83.1.1 must mutate the **real, tracked** kill-rule file, so an unrestored `utime`
leaves criterion 6 permanently red for the rest of the suite.

**Operational consequence: the kill-rule file is WRITE-ONCE.** Any later edit (even a
typo fix) pushes its mtime past the results and turns criterion 6 red. Write it, then
never touch it.

### 7.4 Test-file skeleton -- `backend/tests/test_phase_83_1_1_gate_feasibility.py`

```python
_REPO   = pathlib.Path(__file__).resolve().parents[2]
KILL    = _REPO / "backend/backtest/experiments/killrule_phase83.json"
ART     = _REPO / "backend/backtest/experiments/gate_feasibility_83_1_1.json"  # the recorded figures
PREREG  = _REPO / "backend/backtest/experiments/preregistration_phase83_ranking.json"

def _step_artifacts() -> list[pathlib.Path]: ...   # every result file THIS step produces

# C1 grid recorded, nothing asserted against a threshold
def test_c1_required_sr_grid_recorded_for_all_N_and_T():
    g = json.loads(ART.read_text())["required_annualized_sr"]
    assert {1, 10, 45, 100} <= set(g["trial_counts"])            # N floor from the criterion
    assert len([t for t in g["sample_lengths"] if t["free_source"]]) >= 2
    assert g["periods_per_year"] == 252 and g["return_frequency"] == "daily"   # ppy<->frequency
    for cell in g["cells"]:
        assert isinstance(cell["required_sr"], (int, float))     # PRESENCE + type only
    assert len(g["cells"]) == len(g["trial_counts"]) * len(g["sample_lengths"])
    # NOTE: deliberately NO assertion of any required_sr against any threshold.

# C2 routing spy -- the ONLY guard that proves the module function ran
def test_c2_inversion_invokes_module_compute_deflated_sharpe(monkeypatch):
    import backend.backtest.analytics as analytics
    from backend.backtest import gate_feasibility
    calls, real = [], analytics.compute_deflated_sharpe
    def spy(*a, **k): calls.append(1); return real(*a, **k)
    monkeypatch.setattr(analytics, "compute_deflated_sharpe", spy)
    gate_feasibility.required_annualized_sr(num_trials=45, T=2883, variance_of_srs=0.5)
    assert calls, "the inversion did not invoke analytics.compute_deflated_sharpe"

def test_c2_mutant_reimplementation_fails():        # mutation: inline the deflation -> spy stays empty
    ...

# C3 V carries its integer sample size
def test_c3_measured_V_has_trial_count_ge_2():
    v = json.loads(ART.read_text())["variance_of_trial_sharpes"]
    assert isinstance(v["n_trials_measured_over"], int) and not isinstance(v["n_trials_measured_over"], bool)
    assert v["n_trials_measured_over"] >= 2
    assert len(v["trial_sharpes"]) == v["n_trials_measured_over"]      # no orphan V
    assert v["source_artifact"] and v["ddof"] in (0, 1)

# C4 horizon == pre-registered 126, NOT the engine's 135
def test_c4_spans_use_preregistered_horizon_not_engine_1_5x():
    import inspect
    from backend.backtest.backtest_engine import BacktestEngine
    hd = inspect.signature(BacktestEngine.__init__).parameters["holding_days"].default   # 90
    engine_horizon = int(hd * 1.5)                                                       # 135
    prereg = json.loads(PREREG.read_text())["label_horizon"]["trading_days"]             # 126
    rec = json.loads(ART.read_text())["independent_label_spans"]
    assert rec["horizon_trading_days"] == prereg
    assert rec["horizon_trading_days"] != engine_horizon, "126 and 135 silently equated"
    assert rec["span_units"] == "trading_sessions"        # the +45% unit trap
    for s in rec["per_source"]:
        assert s["free_source"] in (True, False) and s["spans"] == pytest.approx(s["T_sessions"] / prereg)

# C5 verbatim PBO payload
def test_c5_pbo_checked_payload_recorded_verbatim():
    p = json.loads(ART.read_text())["pbo_feasibility"]
    for k in ("refused", "gate_grade", "columns_diverse"):
        assert k in p["payload"]          # present even when null
    assert p["matrix_shape"] == [p["payload"]["n_obs"], p["payload"]["n_trials"]]
    assert p["S"] == 16 and "seed" in p

# C6 / C7 ordering + its mutation  (see 7.3)
def test_c6_kill_rule_predates_every_result_artifact(): _assert_kill_rule_predates_all_results()
def test_c7_touching_kill_rule_forward_makes_c6_fail(): ...
```

## 8. Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL -- **7** (4 WebFetch full pages +
      3 PDFs extracted in full with pypdf per `research-gate.md` Step 3)
- [x] 10+ unique URLs total -- **27**
- [x] Recency scan (2024-2026) performed + reported -- section 2.4
- [x] Full papers read, not abstracts -- 47,819 / 64,198 / 49,239 chars extracted
- [x] file:line anchors for every internal claim -- section 1

Soft checks:
- [x] Internal exploration covered every relevant module
- [x] Contradictions noted -- section 2.5 (fabricated fetch), 5 (V-estimation silence)
- [x] Claims cited per-claim -- section 4

Known gaps (disclosed, not blocking): DSR-paper **Appendix 3** (determining N when
trials are not independent) is referenced at p.8 but its body did not survive pypdf
layout extraction -- P5 records the conservative over-count choice instead.
Wikipedia's DSR article 404'd under both casings.

[MAIN CORRECTION 2026-08-07 covering the JSON envelope summary below without editing
it (kept parseable): its '+263% buys -11% / cutting N buys -23%' lever claim is the
V=0.5 slice (REVERSES at the measured V: -33.6%/-8.7%), and its PBO ranges
0.41-0.77 / 0.18-0.51 are this session's PROTOTYPE run, superseded by the step's
recorded measurement (noise 0.2027-0.6524, edge 0.3025-0.8095 -- no edge seed clears
the 0.20 ceiling).]

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 7,
  "snippet_only_sources": 20,
  "urls_collected": 27,
  "recency_scan_performed": true,
  "internal_files_inspected": 14,
  "coverage": {
    "audit_class": false,
    "rounds": 1,
    "dry_rounds": 0,
    "K_required": 2,
    "new_findings_last_round": 0,
    "dry": false
  },
  "summary": "Measured, not assumed. V from the 82.3 24-run full-sample trial set = 0.008169 (ddof=1); the short-window 32-run set gives 0.167921 -- a 20x spread. Required annualized Sharpe for DSR>=0.95 at N=45, T=2883 (GDELT sessions) swings 0.6886 (measured V) to 2.0692 (module default 0.5), which is exactly why the two 2026-08-04 verdicts disagreed. The deflation floor carries no T: +263% history buys -11%, cutting N 45->10 buys -23%. N=1 and N=2 are identical because of max(num_trials,2) at analytics.py:430. Independent spans at the pre-registered 126 TRADING days: GDELT 22.88, EDGAR 51.06, GPR 83.15. The binding constraint is PBO, not DSR: at the intended (2883,45) shape pure noise gives PBO 0.41-0.77 across 5 seeds and a genuine edge gives 0.18-0.51 against a 0.20 ceiling. Bailey/LdP Eq.(1) confirms variance_of_srs is a VARIANCE. A WebFetch PDF summary fabricated the opposite; all PDF claims here come from pypdf extraction.",
  "brief_path": "handoff/current/research_brief_83.1.1.md",
  "gate_passed": true
}
```
