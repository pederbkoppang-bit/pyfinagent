# Research Brief -- phase-82.25

**Step:** 82.25 (P1) -- `_load_previous_best` RESETS THE DSR TRIAL COUNT ON EVERY WARM START
**Tier:** moderate (caller-specified). Audit-class: false.
**Researcher:** Layer-3 merged researcher + internal explorer
**Date:** 2026-08-05
**Status:** COMPLETE -- gate passed (6 sources read in full, recency scan performed)

## Question

Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014) deflates a Sharpe by `N`, the number of
trials searched. `backend/backtest/quant_optimizer.py` resets `self.num_trials = 1` on both
warm-start paths. **Is `N` the trial count of THIS run, or of the ENTIRE search history that
produced the candidate?** And what is the honest default when the prior count is unknown?

**Answer, in one line:** the literature is unambiguous -- `N` is a *meta-research variable*
scoped to the research process that produced the discovery, not to the session. Resetting is
the failure mode the DSR paper was written to attack.

---

## Search queries run (three-variant discipline)

| # | Query | Variant |
|---|-------|---------|
| 1 | `Deflated Sharpe Ratio Bailey Lopez de Prado number of trials selection bias backtest overfitting` | **YEAR-LESS canonical** (surfaced the 2014 primary paper -- a year-locked query would have buried it) |
| 2 | `deflated Sharpe ratio effective number of trials multiple testing 2025 2024 quantitative finance` | **Last-2-year window** |
| 3 | `Lopez de Prado false strategy theorem effective number of trials clustering DSR 2026` | **Current-year frontier** |

---

## Sources read IN FULL (6; >=5 required; counts toward the gate)

### S1 -- Bailey & Lopez de Prado (2014), "The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting and Non-Normality", *Journal of Portfolio Management* 40(5) 94-107. **THE PRIMARY SOURCE.**
- URL: `https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf` (accessed 2026-08-05)
- Tier: **peer-reviewed** (JPM; = SSRN 2460551 working paper, "This version: July 31, 2014")
- Fetched how: `WebFetch` -> binary PDF captured to disk -> text extracted with `pypdf` (47,819 chars, verbatim grep). **Full text, not abstract.**

**Verbatim quotes bearing directly on 82.25:**

> "we will argue that the most important piece of information missing from virtually all backtests published in academic journals and investment offerings is **the number of trials attempted**. Without this information, it is impossible to assess the relevance of a backtest. Put bluntly, **a backtest where the researcher has not controlled for the extent of the search involved in his or her finding is worthless**, regardless of how excellent the reported performance might be."

> "selection bias is ubiquitous in the financial literature, where backtests are often published **without reporting the full extent of the trials involved in selecting that particular strategy**." (Conclusions)

> "The Deflated Sharpe Ratio (DSR) **incorporates information about the unselected trials**" (Conclusions)

> "Equation 1 tells us that, **as the number of independent trials (N) grows, so will grow the expected maximum** of {SR}."

> "**It is critical to understand that the N used to compute E[max{SR}] corresponds to the number of INDEPENDENT trials. Suppose that we run M trials, where only N trials are independent, N<M. Clearly, using M instead of N will overstate E[max{SR}].**" (Appendix 3)

> "the counter of trials **cannot be turned back**." (footnote 1, "When should we stop testing?")

> Worked example: SR=2.5 annualized, N=1000, V[SR]=0.5, T=1250, skew=-3, kurt=10 -> DSR ~0.90 -> the investor DECLINES. "**Should the strategist have made his discovery after running only N=46 independent trials, the investor may have allocated some funds, as DSR would have been 0.9505**, above the 95% confidence level."

**What S1 establishes for this step:**
1. **N is scoped to the DISCOVERY, not the session.** "the extent of the search involved in his or her finding"; "the full extent of the trials involved in selecting that particular strategy". A warm-started optimizer inherits a *selected* parameter vector; the search that selected it is part of N by definition.
2. **Direction:** N up -> E[max SR] up -> DSR down. Monotone.
3. **Over-counting is the SAFE direction.** Appendix 3 is explicit that M >= N and "using M instead of N will overstate E[max{SR}]" -- which *lowers* DSR. Over-counting cannot manufacture significance; it can only withhold it. **This is the decisive argument for shipping a plain cumulative counter without a correlation-based effective-N estimator.**
4. N=46 vs N=1000 flips a 0.95 gate. This repo's gate (`dsr_threshold=0.95`) is exactly the threshold in the worked example, and the example turns on N alone.
5. The paper explicitly rejects the reset model: "the counter of trials cannot be turned back."

### S2 -- Lopez de Prado & Lewis (2018/2019), "Detection of False Investment Strategies Using Unsupervised Learning Methods" (SSRN 3167017; *Quantitative Finance* 19(9)).
- URL: `https://codemacher.com/wp-content/uploads/2021/02/Detection-of-false-investment-strategies-using-unsupervised-learning-methods_M.LopezDePrado_and_M.Lewis_2018.pdf` (accessed 2026-08-05)
- Tier: **peer-reviewed**. Fetched how: `WebFetch` -> PDF saved -> `pypdf` extraction (49,739 chars, 25 pages), verbatim grep. **Full text.**

> "this theorem required the estimation of **two meta-research variables, in the sense that they are variables related to the research process itself, rather than the outcome of the research**. These two meta-research variables in question are: (1) The estimation of the number of effectively uncorrelated tests (E[K]); and (2) the variance of the SR across the K effectively uncorrelated tests."

> "**There are two major reasons why (5)-(6) are usually unknown. First, it is common for researchers to hide, NOT TRACK, not report or underreport (5)-(6). The motivations may vary, and they could range all the way between negligence and outright fraud. Regardless of the motivations, the implication is that ignorance of (5)-(6) makes it impossible to assess whether a discovery is false.**"

> "Second, even those careful and knowledgeable researchers who track every single trial that takes place face the problem that trials are not usually independent. **The number of independent trials K is less or equal to the number of trials N.**"

> False Strategy Theorem: "Given a sample of IID-Gaussian Sharpe ratios {SR_k}, k=1..K ... E[max_k{SR_k}] / sqrt(V[{SR_k}]) ~= (1-g)Z^-1[1-1/K] + g Z^-1[1-1/(Ke)]"

**Decisive for this step.** The trial count is a property of the **research process**, not of the artifact. `_load_previous_best` resetting to 1 is literally the "not track" failure mode named in the paper. NOTE the notation flip vs S1: here `N` = raw trials, `K` = independent trials; in S1 `N` = independent trials and `M` = raw. Same inequality either way.

### S3 -- Lopez de Prado, "Advances in Quantitative Meta-Strategies" (Nomura 9th Annual Global Quantitative Investment Strategies Conference, 2015-05-10).
- URL: `https://www.nomura.com/events/9th-annual-global-quantitative-investment-strategies-conference/resources/upload/10_00_Marcos_Lopez_de_Prado_20150510.pdf` (accessed 2026-08-05)
- Tier: **official / author's own presentation**. Fetched how: `WebFetch` -> PDF saved -> `pypdf` (19,012 chars, 39 slides). **Full deck.**
- Carries the same E[max SR_n] formula and the "Expected Maximum Sharpe Ratio as the number of independent trials N grows" exhibit. Lists "**Number of independent trials**" as one of the five variables a rigorous selection process must recognise. Confirms the process-level (not run-level) framing: the metric captures overfitting risk across all backtests performed during development.

### S4 -- "AutoQuant: An Auditable Expert-System Framework for Execution-Constrained Auto-Tuning in Cryptocurrency Perpetual Futures", arXiv:2512.22476 (Dec 2025). **[RECENCY / most directly analogous prior art]**
- URL: `https://arxiv.org/html/2512.22476` (accessed 2026-08-05) -- fetched via the arXiv **HTML** chain per `.claude/rules/research-gate.md` (never the `/pdf/` URL). Tier: **preprint**.
- This is the closest published analogue to pyfinagent's situation: an auto-tuner that runs repeated optimisation and must report a DSR.

> "we approximate the total number of effective trials as N_total in [N_opt x N_windows, N_opt x N_windows x N_scen]"

> "The upper bound additionally counts the N_scen cost-scenario grid evaluations used in stable-candidate screening as **distinct selection degrees of freedom**."

> "In the replication materials we provide **helper code for recomputing the same statistic under alternative trial-count assumptions**."

> "the DSR-style statistic serves as a **consistency check** for the walk-forward and cost-stress evidence rather than as standalone proof of economic significance"

**Application:** the 2025 state of practice is (a) **accumulate** across stages -- every selection degree of freedom counts, (b) report N as a **range/bound** when exact independence is unknown, (c) make the assumption **re-computable and auditable** rather than pretending to a single true value. That is a direct template for criterion 3.

### S5 -- QuanterLab, "Deflated Sharpe Ratio (DSR) and Multiple-Testing Correction".
- URL: `https://quanterlab.com/articles/foundations-dsr` (accessed 2026-08-05). Tier: **practitioner**. Fetched in full via `WebFetch`.
- Confirms the inverse N->DSR relation and the "effective trials" concept. **Explicitly provides NO guidance** on cross-run accumulation or unknown N -- recorded here as evidence that the practitioner literature is silent on exactly the question 82.25 must decide, so the decision must be made and documented locally rather than cited.

### S6 -- PapersWithBacktest, "Deflated Sharpe Ratio Explained (Algo Trading)".
- URL: `https://paperswithbacktest.com/course/deflated-sharpe-ratio` (accessed 2026-08-05). Tier: **practitioner**. Fetched in full via `WebFetch`.
- Frames the operational question as "**How many configurations did I try to arrive at this result?**" -- note "to arrive at this result", i.e. the whole path, not the last leg. Also notes over-counting "would artificially raise the benchmark, making it harder to pass the DSR test" -- corroborating S1's conservative-direction argument.

---

## Snippet-only sources (context; do NOT count toward the gate)

| URL | Kind | Why not read in full |
|---|---|---|
| `https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551` | abstract page | SSRN landing page; the full text was obtained from S1 instead |
| `https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3221798` (False Strategy Theorem) | abstract page | superseded by S2, which restates the theorem in full |
| `https://pdfs.semanticscholar.org/c215/d0a2064ce1a3565d276475abc84305418f0f.pdf` | paper PDF | **ATTEMPTED AND FAILED** -- image-only PDF, zero extractable text |
| `https://www.ams.org/notices/201405/rnoti-p458.pdf` ("Pseudo-Mathematics and Financial Charlatanism") | peer-reviewed | **ATTEMPTED AND FAILED** -- HTTP 403 from ams.org; `davidhbailey.com/dhbpapers/pseudo-math.pdf` returned 404. Its core claim ("with enough trials any Sharpe is achievable") is captured verbatim via S2's False Strategy Theorem |
| `https://en.wikipedia.org/wiki/Deflated_Sharpe_ratio` | encyclopedia | **DOES NOT EXIST.** Search engines list the title, but a `curl` of the URL returns Wikipedia's "Wikipedia does not have an article with this exact name" page. Do not cite it |
| `https://sdm.lbl.gov/oapapers/ssrn-id2507040-bailey.pdf` ("Statistical Overfitting and Backtest Performance") | peer-reviewed | identified, not needed once S1+S2 were read in full |
| `https://www.semanticscholar.org/paper/...5b68e5b9f179...` | index page | index entry for S2 |
| `https://tradersunion.com/interesting-articles/deflated-sharpe-ratio/` | community | low tier |
| `https://medium.com/balaena-quant-insights/deflated-sharpe-ratio-dsr-33412c7dd464` | community | low tier |
| `https://www.researchgate.net/publication/286121118_...` | index page | paywalled index |
| `https://www.allaboutalpha.com/blog/2020/01/23/false-positives-and-machine-learning/` | blog | secondary commentary on S2 |
| `https://www.scirp.org/reference/referencespapers?referenceid=4289650` | citation stub | reference stub only |

**URLs collected: 18** (6 read in full + 12 snippet-only).

---

## Recency scan (last 2 years, 2024-2026) -- MANDATORY SECTION

**Performed.** Query variants 2 and 3 above were scoped to 2024/2025/2026.

**Result: ONE new finding that COMPLEMENTS (does not supersede) the canonical 2014/2018 sources.**

- **arXiv:2512.22476 (AutoQuant, Dec 2025)** -- S4 above. First source found that treats trial
  counting in a *repeated auto-tuning* setting, which is exactly pyfinagent's warm-start case.
  Its practice -- accumulate across all stages, report N as a bound, ship helper code so a
  reader can recompute under alternative assumptions -- is the current state of the art and is
  what this brief recommends for criterion 3.
- **No 2024-2026 source contradicts the 2014 DSR paper on the meaning of N.** The direction
  (N up -> DSR down), the process-level scope, and the M>=N inequality are all unchanged.
- The one methodological refinement post-2014 is the **ONC clustering estimator** for the
  *effective* number of trials (S2, 2018; still cited as current in the 2025/2026 practitioner
  material). It is a REFINEMENT that would *reduce* N below the raw count -- i.e. it would make
  the DSR *higher*, not lower. **Skipping it is the conservative choice and does not weaken
  this step.** Explicitly out of scope for 82.25.

---

## Internal code inventory (every claim re-derived 2026-08-05; command shown)

| File | Anchor | Role | Status |
|---|---|---|---|
| `backend/backtest/quant_optimizer.py` | :151, :223, :226, :256, :285, :416, :716, :792, :821, :863 | the defect | LIVE |
| `backend/backtest/analytics.py` | :384-447, :743, :768, :795 | DSR formula + report | LIVE |
| `backend/backtest/experiments/optimizer_best.json` | whole file | warm-start source 1 | LIVE, **schema v1** |
| `backend/backtest/experiments/results/*.json` | 10 files for run 60617e0b | headline evidence | LIVE |
| `backend/backtest/experiments/quant_results.tsv` | header + 537 rows | trial ledger candidate | LIVE |
| `backend/autoresearch/meta_dsr.py` | :1-11, :46-70, :73-75 | cumulative-N doctrine | **DEAD for this purpose** |
| `backend/autoresearch/rotation_runner.py` | :161-166 | reads `dsr` from the artifact | LIVE |
| `backend/services/paper_go_live_gate.py` | :42, :97-104 | DSR>=0.95 go-live gate | LIVE |
| `backend/tests/test_phase_82_22_optimizer_best_provenance.py` | :31-49 (`_optimizer()`), :51-57 (`_saved()`), :194-200 | fixture precedent | LIVE |
| `backend/autoresearch/strategy_backtest_adapter.py` / `strategy_selector.py` | :155-162, :233 / :66, :109 | separate `num_trials` | LIVE, **unaffected** |

### 1. Complete lifecycle of `self.num_trials`

Command: `grep -n "num_trials" backend/backtest/quant_optimizer.py`. All 9 hits, each read in context:

| Line | Statement | Meaning | Legitimate? |
|---|---|---|---|
| `:151` | `self.num_trials = 0` | `__init__` seed. Runs BEFORE `_load_previous_best()` (called at `:157`). | YES -- a neutral pre-load seed |
| `:223` | `generate_report(baseline_result, num_trials=1)` | the cold-start baseline is genuinely trial #1 | **YES** |
| `:226` | `self.num_trials = 1` | cold-start baseline only (inside the `else` of `if self._warm_started`) | **YES -- this one is CORRECT and must NOT be changed** |
| `:256` | `self.num_trials += 1` | per-iteration increment. **Placed BEFORE the `try:` at `:274`, so crashed experiments DO increment.** | YES (and consistent with counting search effort) |
| `:285` | `generate_report(result, num_trials=self.num_trials)` | **the only place the optimizer computes a DSR from the counter.** This is where the fix becomes observable | -- |
| `:416` | `export_best()` returns `"num_trials": self.num_trials` | API/report surface | -- |
| `:716` | `self.status_callback(self.num_trials, ...)` | UI progress | -- |
| `:792` | `"num_trials": getattr(self, "num_trials", None)` in `_save_best_params()` | **the 82.22 write-side hook, with an in-code comment naming step 82.25** | YES |
| `:821` | `self.num_trials = 1` | warm start from `optimizer_best.json` | **NO -- THE DEFECT** |
| `:863` | `self.num_trials = 1` | warm start from `result_store.load_latest()` | **NO -- THE DEFECT** |

**CONFIRMED:** the step description's line numbers (:821, :863, :226, :151, :256, :285, :416, :716, :792) are all correct as of `HEAD` (last touch to this file: `5c1f3f8f fix(82.16)`). The step's count of two warm-start resets is correct; there is no third.

**Edge case the step did not name:** both resets sit INSIDE `if prev_sharpe is not None:`. A source file with `params` but no `sharpe` applies the params, does NOT set `_warm_started`, and leaves `num_trials == 0` from `:151`; `run_loop()` then takes the cold-start branch and `:226` sets it to 1. So there is a third, *silent* path where inherited params are scored as trial 1 without any warm-start flag at all. Worth a guard.

### 2. What `optimizer_best.json` ACTUALLY contains today

Command: `python3 -c "import json; d=json.load(open('backend/backtest/experiments/optimizer_best.json')); print(list(d.keys()))"`

```
['params', 'sharpe', 'dsr', 'run_id', 'kept', 'discarded', 'saved_at']
```
```
sharpe    = 1.1704633657934074
dsr       = 0.9525811126193078
run_id    = '60617e0b'
kept      = 0
discarded = 10
saved_at  = '2026-07-24T11:04:51.243740+00:00'
```

Presence check for every 82.22 field: `schema_version` ABSENT, `metrics_run_id` ABSENT,
**`num_trials` ABSENT**, `warm_started_from` ABSENT, `metrics_source_artifact` ABSENT.

> ### STALE STEP CLAIM (high value)
> The step says: *"persist it in optimizer_best.json -- step 82.22 added a `num_trials` field
> for exactly this"*. **That is true of the WRITER and false of the FILE.** 82.22 changed
> `_save_best_params()` (`quant_optimizer.py:792`) but the on-disk artifact was last written
> `2026-07-24T11:04:51Z`, before 82.22 landed (`be04da12`, 2026-08-04), and the optimizer has
> not run since (`historical_macro` is frozen -- no optimizer runs, per the phase-66 memo).
> **The live file is schema v1 with no `num_trials`.** Consequences:
> - Criterion 1's "fixture whose source file records a prior count" must be a **synthetic**
>   `tmp_path` fixture; it cannot be the live artifact.
> - Criterion 3's "fixture with no recorded prior count" is **exactly the live artifact's
>   shape** -- so the no-prior-count branch is the branch that will actually execute on the
>   next real warm start. It is the primary path, not the edge case. Design it first.

**The precise defect shape is therefore worse than "the field exists but is not read":**
`_load_previous_best` reads `params` (`:810`), `best_sharpe`/`sharpe` (`:816`),
`best_dsr`/`dsr` (`:817`), `metrics_run_id`/`run_id` (`:830`), `metrics_source_artifact`
(`:832`) -- **and never reads `num_trials` at all**, from either source. The write side
(`:792`) and the read side (`:809-834`) are disconnected.

### 3. How DSR is computed, and the direction

`backend/backtest/analytics.py:384-447` `compute_deflated_sharpe(observed_sr, num_trials, variance_of_srs=0.5, skewness=0.0, kurtosis=3.0, T=252, periods_per_year=1)`.

```python
:417    if num_trials < 1 or T < 10 or observed_sr == 0:
:418        return 0.0
:429    e_max_sr = math.sqrt(var_srs) * (
:430        (1 - 0.5772) * stats.norm.ppf(1 - 1 / max(num_trials, 2))
:431        + 0.5772 * stats.norm.ppf(1 - 1 / (max(num_trials, 2) * math.e))
:432    )
:443    z = (sr - e_max_sr) / se_sr
:446    dsr = float(stats.norm.cdf(z))
```

This is Bailey's Eq. (1)/(2) verbatim. `num_trials` enters ONLY through `e_max_sr`, which is
strictly increasing in `N`; `z` is therefore strictly decreasing in `N`; `norm.cdf` is
monotone. **DSR is strictly decreasing in `num_trials`. Confirmed analytically, not asserted.**

Guards: **two different clamps at two layers.** `compute_deflated_sharpe` returns `0.0` for
`num_trials < 1` (`:417`); `generate_report` instead passes `num_trials=max(num_trials, 1)`
(`:768`) so a 0 never reaches the guard through that door. There is also an internal
`max(num_trials, 2)` at `:430-431` -- so N=1 and N=2 produce the **same** `e_max_sr`. No upper
ceiling. **A cumulative counter must never emit 0**, or the two layers silently disagree.

### 4. VERIFYING THE HEADLINE MEASUREMENT -- **REPRODUCED, not folklore**

Command:
```
grep -l "60617e0b" backend/backtest/experiments/results/*.json     # -> 10 files
python3 ... json.load(f)['analytics'] -> (num_trials, deflated_sharpe, sharpe)
```

| file | num_trials | deflated_sharpe | sharpe |
|---|---|---|---|
| `20260724T080231Z_60617e0b-exp01.json` | 2 | **0.6387492887307706** | 0.6455483635957818 |
| `...exp02` | 3 | 0.344347531304153 | 0.6505601279884126 |
| `...exp03` | 4 | 0.16791184161814215 | 0.6455483635957818 |
| `...exp04` | 5 | 0.05625863455607464 | 0.5747299343087512 |
| `...exp05` | 6 | 0.057481413032332326 | 0.6455483635957818 |
| `...exp06` | 7 | 0.036671993878682634 | 0.6455483635957818 |
| `...exp07` | 8 | 0.006017459920086288 | 0.5383536163549719 |
| `...exp08` | 9 | 0.009004286723854782 | 0.5415592531805858 |
| `...exp09` | 10 | 0.012049984963969612 | 0.6455483635957818 |
| `...exp10` | 11 | **0.008813951184271042** | 0.6455483635957818 |

**The step's numbers are exact and reproducible from disk.** 0.6387 at N=2 -> 0.0088 at N=11.
This is NOT inherited folklore.

**The measurement is actually STRONGER than the step claims.** exp01 (N=2, DSR 0.6387) and
exp10 (N=11, DSR 0.0088) have the **identical Sharpe, 0.6455483635957818**. Same strategy, same
returns, same everything -- **a 72x DSR difference produced by the trial counter alone.** That
is the cleanest possible demonstration that `num_trials` dominates the statistic, and it is a
ready-made, zero-cost fixture: the two artifacts are already on disk. (Note the fall is not
strictly monotone across the table -- exp04->exp05 and exp07->exp08 rise -- because the Sharpe
varies too; the monotonicity claim holds only at fixed Sharpe, which the exp01/exp10 pair
provides.)

### 5. Criterion 3 -- the decision when no prior count is recorded

Options, with the Bailey/LdP verdict on each:

| Option | Mechanism | Verdict |
|---|---|---|
| **A. Treat absent as 1** (status quo) | current `:821`/`:863` | **INDEFENSIBLE.** Asserts first-attempt discovery. S2 names this exact behaviour ("not track") as what makes a discovery unassessable |
| **B. Refuse to report a DSR** | emit `None`/sentinel when N unknown | **Most literally faithful to S2** ("ignorance ... makes it impossible to assess whether a discovery is false"). But high blast radius: ~15 `dict.get`-based consumers; `paper_go_live_gate` and `rotation_runner._incumbent_dsr_from_optimizer_best` would each fail in a different direction (one closed, one open). **Not recommended as the default path**, but a good *reporting* companion |
| **C. Documented conservative constant** | a named module constant, e.g. `_UNKNOWN_PRIOR_TRIALS`, explicitly != 1, recorded in the run's output | **RECOMMENDED.** Satisfies "documented decision rather than silently defaulting". Over-counting is the safe direction (S1 App.3, S6) |
| **D. Infer from `quant_results.tsv`** | count rows / runs | Defensible in DIRECTION but **has a trap**: `DELETE /api/backtest/optimize/history` (`backend/api/backtest.py:838-857`) deletes the TSV *and* `optimizer_best.json` *and* all result JSONs. An inferred count silently resets to 0 -- re-creating the same defect through a different door. Use only as a *floor*, never as the sole source |

Measured, for option D's plausibility (`python3 csv.DictReader` over the TSV):
`537` data rows, `36` distinct base `run_id`s, `27` `BASELINE` rows, `11` rows for `60617e0b`.
Status breakdown: `crash 300`, `discard 193`, `BASELINE 27`, `evaluated 7`, `seed_test 6`,
`keep 2`, `dsr_reject 2`. **300 of 537 rows are crashes** -- a crashed backtest produces no
Sharpe, so it is not a member of `{SR_k}` under the False Strategy Theorem, yet the in-run
counter at `:256` already counts it (the increment precedes the `try:`). Whichever way the
contract rules, it must rule **explicitly** -- and staying consistent with the existing `:256`
semantics (count crashes) is both simpler and more conservative.

**Recommended resolution (hybrid C, with D as a floor and B as reporting):**
read `num_trials` from the artifact when present; when absent, resolve to a **named documented
constant** and **record which branch was taken** in the persisted payload
(`trials_provenance: "carried" | "assumed_unknown" | "fresh"`). This mirrors S4's 2025 practice
exactly: accumulate, bound rather than pretend, and make the assumption auditable.

### 6. Blast radius -- who reads `num_trials`, and does a larger count break anything?

Command: `grep -rn "num_trials" --include="*.py" --include="*.ts" --include="*.tsx" . | grep -v .venv | grep -v node_modules | grep -v backend/tests`

**Same-name-different-variable (NOT affected -- do not "fix" these):**
`backend/autoresearch/strategy_backtest_adapter.py:155,162,172,233`,
`strategy_candidate_producer.py:164-181`, `strategy_selector.py:66,109`,
`rotation_runner.py:219,248,278`. These carry their **own** `num_trials` = the number of seed
configs in a bake-off (`n = num_trials if num_trials is not None else len(grid)`). They never
touch the optimizer's counter.

**Genuinely downstream of the optimizer's counter:**
- `quant_optimizer.py:285` -> `generate_report` -> `analytics.py:768` -> the DSR. **This is the
  behaviour change.**
- Keep/discard: `:318 if delta > 0 and trial_dsr >= self.dsr_threshold` (threshold `0.95`,
  `:141`/`:146`) and `:349 elif delta > 0 and trial_dsr < self.dsr_threshold` (the
  `DSR_REJECT` branch, `:355`). **A larger N makes KEEP strictly harder.** Given the live file
  already records `kept=0, discarded=10`, expect the fix to push future runs further toward
  zero keeps. That is honest, but Main must SAY it in the contract -- otherwise it will look
  like the step broke the optimizer.
- `:416 export_best()` and `:716 status_callback` -- display only.
- `:792 _save_best_params()` -- persists the counter (82.22).
- `scripts/go_live_drills/dsr_oos_test.py:112-114` asserts `S6 num_trials > 1`; a larger count
  makes this **easier**, no break. `scripts/go_live_drills/evaluator_criteria_test.py:107-112`
  buckets on `>5` / `>1` -- also directionally easier.

**The promotion / go-live surfaces read the persisted `dsr`, not `num_trials`:**
`backend/services/paper_go_live_gate.py:42 DSR_THRESHOLD = 0.95` (+ `:97-104` reads
`optimizer_best.json`), `backend/autoresearch/rotation_runner.py:161-166
_incumbent_dsr_from_optimizer_best()`, `backend/autoresearch/promoter.py:26
DSR_MIN_FOR_PROMOTION = 0.95`, `backend/autoresearch/gate.py:21 min_dsr = 0.95`.

> **The single biggest go-live risk in this step.** The live file's `dsr = 0.9525811126193078`
> clears the 0.95 gate by **0.0026**. If the fix ever *recomputes* that persisted figure at a
> higher N, `dsr_ge_95` flips to FAIL and the go-live gate closes. **The fix must change only
> FUTURE computations at `:285`; it must NOT retroactively re-deflate the persisted `dsr`.**
> Re-deflating an inherited number without re-running the backtest would also be fabrication --
> the inherited `dsr` was computed at whatever N its own run used, and that N is unrecorded
> (the file is schema v1). State this boundary explicitly in the contract.
>
> Related, already-known: per 82.22, that `sharpe=1.1704.../dsr=0.9525...` pair does not even
> belong to `60617e0b` -- it belongs to `52eb3ffe-exp10`, four months earlier
> (`quant_optimizer.py:742-762`).

**Dead-but-relevant prior art:** `backend/autoresearch/meta_dsr.py:1-11` already states this
step's exact doctrine --

> "When running many trials in parallel (the autoresearch loop), the per-trial DSR is NOT the
> right statistic -- **you must recompute DSR at the cumulative sample size across ALL trials,
> including abandoned ones.** This is the 'multiple testing correction' Bailey & Lopez de Prado
> describe for DSR."

and exposes `meta_dsr(trials, *, cumulative_n=None)` (`:46`) plus `required_dsr(cumulative_n)`
(`:73`). **But it is dead for this purpose**: `grep -rn "meta_dsr\|TrialLedger"` shows the only
production import is `from backend.autoresearch.meta_dsr import LOOSE_DSR_MIN` at
`backend/agents/evaluator_agent.py:48` -- the constant only. `meta_dsr()` and `TrialLedger`
have **zero production callers** (one test script, `scripts/harness/autoresearch_meta_dsr_test.py`).
Its penalty formula is admittedly a stand-in: `penalty = 0.1 * sqrt(log(max(2, n)))`, with the
docstring conceding "this scaffold uses the qualitatively-correct monotone penalty only".
**Reuse the doctrine and the `cumulative_n` vocabulary; do NOT reuse the formula** -- the real
one is already correct at `analytics.py:429-432`.

### 7. Existing test precedent for the fixture

`backend/tests/test_phase_82_22_optimizer_best_provenance.py`:
- `:20-24` `_checker()` -- loads `scripts/qa/check_optimizer_best_provenance.py` by spec.
- `:27-29` `class _Engine` -- a two-field stub (`_strategy_params`, `stop_check`).
- **`:32-49` `_optimizer(**attrs)`** -- **the fixture builder to copy.** It bypasses `__init__`
  via `QuantStrategyOptimizer.__new__(QuantStrategyOptimizer)` and hand-sets
  `best_params/best_sharpe/best_dsr/num_trials/kept/discarded/_run_id/_warm_started/
  _warm_started_from_run_id/_warm_started_from_artifact`, with `**attrs` overrides.
  Docstring: *"we are testing the save/provenance logic, not the constructor's disk I/O"*.
- `:51-57` `_saved(monkeypatch, tmp_path, opt)` -- monkeypatches `qo._BEST_PARAMS_PATH` to a
  `tmp_path` file, calls `_save_best_params()`, returns the parsed JSON.
- `:194-200` already asserts `"num_trials" in d` with the message **"num_trials is required as
  input to step 82.25"**, and `test_num_trials_is_persisted_for_the_deflation_fix` asserts
  `d["num_trials"] == 7`. **82.22 deliberately built 82.25's input.**
- `:213-235` and `:253-283` already contain monotonicity guards over `compute_deflated_sharpe`
  and over run 60617e0b's recorded pairs.

**CAUTION for 82.25's own test:** `_optimizer()` bypasses `__init__`, so it cannot exercise
`_load_previous_best` (which `__init__` calls at `:157`). The new test needs a **second**
builder that runs the real constructor with `_BEST_PARAMS_PATH` monkeypatched to a `tmp_path`
fixture, plus a stub engine -- note `_Engine._strategy_params` has only `strategy` and
`tp_pct`, and `_apply_params_to_engine` will be called on it. Check what that method requires
before reusing `_Engine` verbatim.

---

## Recommendation for the contract

**Design.**
1. **Read the counter on warm start.** In `_load_previous_best`, replace `self.num_trials = 1`
   at `:821` with a resolution of the prior count from the artifact
   (`data.get("num_trials")`), and at `:863` from the result-store payload
   (`latest.get("analytics", {}).get("num_trials")` -- `analytics.py:795` writes it there).
2. **Leave `:226` alone.** The cold-start baseline genuinely IS trial 1. Only the two warm-start
   sites are defective. A fix that also touches `:226` is over-reach.
3. **Never emit 0.** Resolve to `max(resolved, 1)` at the assignment site so the two clamp
   layers (`analytics.py:417` vs `:768`) cannot disagree.
4. **Name the unknown branch.** A module constant (e.g. `_UNKNOWN_PRIOR_TRIALS`) with a comment
   citing S1 App.3 (over-count is conservative) and S2 ("not tracking makes a discovery
   unassessable"). It must NOT be 1.
5. **Record the decision in the artifact.** Add `trials_provenance` ("carried" /
   "assumed_unknown" / "fresh") to `_save_best_params()`'s payload alongside the existing
   `num_trials` at `:792`. This is what turns criterion 3's "documented decision" from a comment
   into an auditable fact, and it matches S4's "recomputing under alternative trial-count
   assumptions".
6. **Do not retroactively re-deflate the persisted `dsr`.** Changed behaviour starts at the next
   `:285` call.
7. **Scope discipline:** effective-N / ONC clustering (S2) is explicitly OUT of scope. Note in
   the contract that the plain cumulative count over-deflates relative to the ideal, that this
   is the SAFE direction per S1 Appendix 3, and that the repo already documents the same
   trade-off at `strategy_backtest_adapter.py:45` ("plain num_trials=K over-deflates -- the SAFE
   direction").

**Test design (criterion 4 -- must fail against reset-to-1).**
- Build a `tmp_path` `optimizer_best.json` carrying `num_trials: 11` (mirroring run 60617e0b),
  construct a real `QuantStrategyOptimizer` with `_BEST_PARAMS_PATH` monkeypatched, and assert
  `o.num_trials == 11`. Against `HEAD` this yields `1` -> **the test fails today.** Prove that
  in `experiment_results.md` by running it before the fix.
- **Do NOT assert `num_trials > 1`** -- changing `= 1` to `= 2` would pass it. Assert **equality
  with the recorded value**, and separately assert that the first post-warm-start report is
  computed at `prior + 1`.
- For criterion 2, `compute_deflated_sharpe` is deterministic and pure: assert
  `dsr(sr, N=cumulative) < dsr(sr, N=1)` on identical inputs. The **exp01/exp10 pair already on
  disk** (identical Sharpe 0.6455483635957818, DSR 0.6387 vs 0.0088) is a free real-data anchor.
- For criterion 3, a fixture with **no** `num_trials` key -- i.e. the live file's actual shape --
  must resolve to the documented constant and stamp `trials_provenance: "assumed_unknown"`.
  Mutation-test it: flipping the constant to 1 must fail the test.

**Traps.**
1. The live `optimizer_best.json` has **no** `num_trials`. Criterion 1's fixture must be
   synthetic; criterion 3's branch is the one that fires in production.
2. Both `= 1` sites are inside `if prev_sharpe is not None:` -- a params-only artifact leaves the
   counter at `0` from `:151` and falls through to the cold-start `:226`. Third silent path.
3. `:256` increments **before** the `try:`, so crashes count (300 of 537 TSV rows). Keep that
   semantic; don't accidentally change it.
4. `max(num_trials, 2)` at `:430-431` means N=1 and N=2 give the same `e_max_sr` -- a test that
   compares N=1 vs N=2 will find **no difference**. Compare against a materially larger N.
5. Inferring N from the TSV is undermined by `DELETE /api/backtest/optimize/history`
   (`api/backtest.py:838-857`), which deletes the TSV, `optimizer_best.json` and the result JSONs
   together. Floor only, never sole source.
6. A larger N makes the `:318` KEEP branch strictly harder and the `:349` `DSR_REJECT` branch
   more likely. Disclose this as an intended consequence, with the `kept=0/discarded=10` baseline.
7. `paper_go_live_gate.py:42` gates at DSR>=0.95 and the live artifact sits at 0.95258 -- 0.0026
   of headroom. Do not touch the persisted number.
8. `strategy_*`/`rotation_runner` `num_trials` is a **different variable** (seed-config count).
   Out of scope.
9. `meta_dsr.py`'s `cumulative_n` is the right vocabulary but its penalty formula is a
   placeholder. Don't wire the formula in.

**Where the masterplan step description is WRONG or STALE.**
- **STALE:** *"step 82.22 added a `num_trials` field for exactly this [to optimizer_best.json]"*
  -- 82.22 added it to the **writer** (`:792`); the **live artifact does not contain it**
  (schema v1, `saved_at 2026-07-24`, pre-82.22). Reading this as "the field is on disk" would
  send the implementer to write a test against a key that isn't there.
- **UNDERSTATED (in the step's favour):** the step says DSR "falls monotonically 0.6387 ->
  0.0088". Reproduced -- but the fall is monotone only *at fixed Sharpe*; across the ten
  artifacts the Sharpe also moves (exp04->exp05 and exp07->exp08 rise). The far stronger and
  fully clean statement is that **exp01 and exp10 share the identical Sharpe
  0.6455483635957818** and differ 72x in DSR on the trial count alone.
- **CONFIRMED, not stale:** `:821` and `:863` are exactly where the step says, and there are
  exactly two warm-start resets. Every other line number the step guessed (`:151`, `:226`,
  `:256`, `:285`, `:416`, `:716`, `:792`) is also correct at `HEAD`.

---

## Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch -- **6** (S1-S6; 2
      peer-reviewed, 1 author presentation, 1 preprint, 2 practitioner)
- [x] 10+ unique URLs total -- **18** (6 full + 12 snippet-only)
- [x] Recency scan (last 2 years) performed + reported -- section above, 1 complementing finding
- [x] Full papers / pages read (not abstracts) -- S1 and S2 extracted with `pypdf` (47,819 and
      49,739 chars) and grepped verbatim; S3 39 slides; S4 arXiv HTML chain; S5/S6 full pages
- [x] file:line anchors for every internal claim, each with the command that produced it

Soft checks:
- [x] Internal exploration covered every module named in the scope, plus 4 the scope did not
      name (`meta_dsr.py`, `paper_go_live_gate.py`, `rotation_runner.py`, `api/backtest.py`)
- [x] Contradictions noted (S1 `N`=independent/`M`=raw vs S2 `K`=independent/`N`=raw; S5/S6
      silent on cross-run accumulation)
- [x] Claims cited per-claim with URL + access date

**Failed fetch attempts, disclosed:** `ams.org/notices/201405/rnoti-p458.pdf` (403);
`davidhbailey.com/dhbpapers/pseudo-math.pdf` (404); `pdfs.semanticscholar.org/c215/...`
(image-only PDF, no text); `en.wikipedia.org/wiki/Deflated_Sharpe_ratio` (**article does not
exist** -- verified by curl, despite appearing in search results twice).

---

## JSON gate envelope

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 6,
  "snippet_only_sources": 12,
  "urls_collected": 18,
  "recency_scan_performed": true,
  "internal_files_inspected": 12,
  "coverage": {
    "audit_class": false,
    "rounds": 3,
    "dry_rounds": 0,
    "K_required": 2,
    "new_findings_last_round": 1,
    "dry": false
  },
  "summary": "N in the Deflated Sharpe Ratio is a meta-research variable scoped to the research process that produced the discovery, not to one optimisation session (Bailey & Lopez de Prado 2014; Lopez de Prado & Lewis 2018: 'ignorance ... makes it impossible to assess whether a discovery is false'). Over-counting trials is the conservative direction (App.3: using M>=N overstates E[max SR], lowering DSR), so a plain cumulative counter is defensible without effective-N clustering. quant_optimizer.py:821 and :863 both reset num_trials=1 on warm start; :226 (cold baseline) is legitimate. The live optimizer_best.json is schema v1 and does NOT carry num_trials -- 82.22 changed the writer only, so criterion 3's unknown-prior-count branch is the production path. Run 60617e0b's headline is reproduced exactly from disk, and is stronger than claimed: exp01 and exp10 share Sharpe 0.6455483635957818 but differ 72x in DSR on trial count alone. Blast radius: a larger N makes the :318 KEEP branch harder; the persisted dsr=0.95258 clears the 0.95 go-live gate by 0.0026 and must NOT be retroactively re-deflated.",
  "brief_path": "handoff/current/research_brief_82.25.md",
  "gate_passed": true
}
```
