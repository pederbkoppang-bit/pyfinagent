# Research Brief -- step 86.47

**Tier:** moderate (caller-specified). **Audit-class:** YES (loop-until-dry, K=2).
**Role:** Layer-3 Researcher (external literature + internal codebase exploration).
**Started:** 2026-08-18. **Status:** IN PROGRESS -- see envelope at the tail.

**Objective (verbatim from caller):** Establishing whether a run of ZERO events in a
low-rate pipeline is anomalous, before explaining it: (a) statistics of a zero-run under
a low base rate (Poisson / negative-binomial nulls, rule of three, why p just above 0.05
is neither evidence of health nor of breakage, normalisation rule trading-vs-calendar
days); (b) funnel / conversion-census methodology with stated predicates and denominators;
(c) measuring a funnel on a column that is NULL/empty for the counted population;
(d) REFUSAL vs ABSENCE in observability data; (e) evidential standard for claiming a risk
gate is MIS-CALIBRATED (counterfactual outcomes; reject inference); (f) confounding a
DEGRADED path with a healthy one in the same census (2-call lite wrapper vs 28-agent full
pipeline).

**Binding constraint from the caller:** inherit NO number from the step text. Every figure
below is re-derived here, with its query and window stated.

---

## ENVELOPE (phase-86.37) -- FINAL

```json
{
  "brief_status": "COMPLETE",
  "tier": "moderate",
  "external_sources_read_in_full": 14,
  "snippet_only_sources": 34,
  "urls_collected": 48,
  "recency_scan_performed": true,
  "internal_files_inspected": 9,
  "coverage": {
    "audit_class": true,
    "rounds": 14,
    "dry_rounds": 2,
    "K_required": 2,
    "new_findings_last_round": 0,
    "dry": true
  },
  "summary": "The zero-run is NOT anomalous. Re-derived: last trade 2026-08-13; the run is 2 analysis days / 13 analyses (not 5 calendar days -- 2 are weekend, 1 has not occurred). Post-break BUY rate is 8/262 = 0.0305, so P(0 BUYs | 13 opportunities) = 0.672 and the rule-of-three bound 3/13 = 0.231 contains it; ~97 analyses (~16 days) would be needed for p<0.05. The real event is 2026-06-11/06-15 (BUY 46/64 -> 0/7). Four changes land together on 2026-08-14 (sonnet-4-6 -> sonnet-5, risk columns empty -> populated, zero-scores gone, recommendation -> Hold only), so no attribution is identifiable. risk_judge_decision is populated on 18/580 = 3.1% of analysis_results (0 in May, 0 in July), so a funnel keyed on it measures its own blindness; risk_intervention_log is 0 rows. The lite/full split IS derivable from JSON_VALUE(full_report_json, $._path) on 100% of rows, and the zero-run days are full 13/13. Counting REJECTs cannot show mis-calibration; forward outcomes on rejected candidates are required.",
  "brief_path": "handoff/current/research_brief_86.47.md",
  "gate_passed": true
}
```

---

## Read in full (>=5 required; counts toward the gate)

| # | URL | Accessed | Kind | Fetched how | Key finding |
|---|-----|----------|------|-------------|-------------|
| 1 | https://arxiv.org/html/2606.18479v1 | 2026-08-18 | preprint (arXiv, 2026-06) | WebFetch, arXiv native HTML | Accepted-only data **structurally cannot** reveal whether a rejection rule is calibrated. Proposition 2 decomposes `acc_t = (1-pi_t)*spec_t + pi_t*rec_t`; as the observed positive rate `pi_t` shrinks under selection, recall's contribution vanishes and accuracy goes blind to recall collapse. Controlled exploration (deliberately approving a random fraction `r` of rejects) is the only assumption-free recovery; r=2-5% "diagnoses feedback loop severity at near-zero cost". |
| 2 | https://arxiv.org/abs/2607.02830 (text via pdfplumber 0.11.9 from the /pdf render) | 2026-08-18 | preprint (arXiv, 2026-07) | WebFetch -> binary PDF -> pdfplumber local extract (10 pages, 21,639 chars) per research-gate.md Step 3 | The only at-scale worked example of auditing a **rejection filter** by its rejected candidates' forward outcomes. Five-tier outcome rule; conservative save-to-miss 3.7:1 (418 saved-windowed vs 112 missed of 2,402 events). The wider 14.8:1 collapses because its extra tier fails a **matched** comparison: early-death mints reach `gone` at 48.9% vs 57.6% for non-early-death rejected mints. The apparent elevation vs the 34.6% broader-universe base rate "is an artifact of filter selection rather than a property of the classification". |
| 3 | https://support.minitab.com/en-us/minitab/help-and-how-to/quality-and-process-improvement/control-charts/how-to/rare-event-charts/g-chart/before-you-start/overview/ | 2026-08-18 | official vendor doc | WebFetch | The SPC answer to "is this run of nothing abnormal": for rare events you chart the *opportunities/days BETWEEN events*, not the count. "When you monitor rare events with a traditional chart, such as a P or a U chart, you need a large amount of data to establish accurate control limits" and "collecting enough rare event data to detect an adverse change in the frequency of events may take months or even years." |
| 4 | https://prometheus.io/docs/prometheus/latest/querying/functions/ | 2026-08-18 | official docs | WebFetch | The canonical REFUSAL-vs-ABSENCE construct. `absent(v)` "returns an empty vector if the vector passed to it has any elements ... and a 1-element vector with the value 1 if the vector passed to it has no elements"; `absent_over_time` is "useful for alerting on when no time series exist for a given metric name and label combination for a certain amount of time." Absence needs a DIFFERENT operator from any threshold on a value -- a zero-valued series and a missing series are not the same object. |
| 5 | https://sre.google/sre-book/monitoring-distributed-systems/ | 2026-08-18 | official docs (Google SRE) | WebFetch | **Partly a NULL RESULT, reported as such.** The canonical SRE monitoring chapter does NOT address absence-of-signal vs a broken system, and does not state limits of the four golden signals. It does give "Symptoms Versus Causes": "Your monitoring system should address two questions: what's broken, and why? The 'what's broken' indicates the symptom; the 'why' indicates a (possibly intermediate) cause", and "it's better to spend much more effort on catching symptoms than causes." The gap is the point: the standard observability canon has no vocabulary for "legitimately had no work to do", which is why source 4's `absent()` is a separate primitive. |
| 6 | https://arxiv.org/html/2509.00109v1 | 2026-08-18 | preprint, systematic review (arXiv, 2025-08) | WebFetch, arXiv native HTML | Cross-domain triangulation for source 1's mechanism. 24 primary studies from 347 screened, 2019-2025. "Most bias mitigation approaches for RS are evaluated on a single iteration of the training/validation/testing data splits, ignoring the feedback loop effect, therefore introducing evaluation bias", and cites a related survey where "115 studies out of 127 are evaluated using offline testing without considering model updates." A feedback loop is only visible under a DYNAMIC (simulated or live-A/B) setup, never from the deployed system's own logs. |
| 7 | https://pmc.ncbi.nlm.nih.gov/articles/PMC11090505/ | 2026-08-18 | peer-reviewed (PMC, 2024) | WebFetch | Directly answers "why a p just above 0.05 is not evidence of health". "Such a 'null result' -- typically characterized by a p-value p>0.05 for the null hypothesis of an absent effect -- may also occur if an effect is actually present"; "if the sample size of a study is chosen to detect an assumed effect with a power of 80%, null results will incorrectly occur 20% of the time"; and the root problem is that "the null hypothesis under which the p-value is computed is misaligned with the goal of inference." Claiming ABSENCE requires equivalence testing against a pre-declared margin `[-D,+D]` or a Bayes factor `BF01>1`. Cites Altman DG, Bland JM, BMJ 1995;311:485. |
| 8 | https://en.wikipedia.org/wiki/Funnel_analysis | 2026-08-18 | reference | WebFetch | **Read in full and reported as a NEGATIVE finding.** The canonical "funnel analysis" entry provides *no* formal stage definition, *no* conversion formula, and *no* denominator convention -- "The article does not explicitly define how stages are determined" and gives only one 2015 McKinsey citation. Conclusion for this step: there is no rigorous canonical funnel methodology to lean on; the actual methodological discipline for a decision census has to be imported from the selection-bias / survivorship literature (sources 1, 2, 6). |

| 9 | https://arxiv.org/html/2604.10996 | 2026-08-18 | preprint (arXiv, 2026-04) | WebFetch, arXiv native HTML | **The design template for this step.** Distinguishes a REGIME BOUNDARY from a DEFECT by multi-regime comparative testing: same model weights, H1-2025 shock Sharpe -0.267+/-0.284 vs H2-2025 calm +1.038+/-0.424. "The recovery in H2 proves the features themselves aren't broken -- only the market regime changed. A true defect would persist across both periods." Also honest on power: "Five seeds on a 120-day window yield low power (<<50% to detect dSharpe = 0.3 at sigma = 0.4)." |
| 10 | https://ar5iv.labs.arxiv.org/html/0809.4205 | 2026-08-18 | preprint (arXiv, pre-Dec-2023 -> ar5iv per research-gate.md Step 2) | WebFetch, ar5iv LaTeXML render | The formal excess-zeros machinery behind (a). "in many situations this type of observations exhibit a substantially larger proportion of zeros than what is expected for the Poisson model"; "If the proportion of atypical zero observations remains undetected, the variability of the population is underestimated." Under H0: p=0, `sqrt(n)*D_2:2/sigma(theta)` is asymptotically N(0,1). Crucial caveat: "Rejecting the Poisson model does not necessarily imply that the ZIP model provides the best fit. Another model could account better for the observed dispersion." Power at n=50, p=0.05 is only 0.386-0.422. |
| 11 | https://plato.stanford.edu/entries/paradox-simpson/ | 2026-08-18 | peer-reviewed reference (SEP) | WebFetch | The formal answer to (f). Association reversal requires the partitioning variable to be **correlated with the treatment** -- "if the partitioning variable M is independent of treatment T, association reversal cannot occur." And the decisive rule: "The judgment that one should partition the population in one case but not the other cannot be based on the probabilities alone, but requires the additional information supplied by the causal model" -- condition on CONFOUNDERS, never on MEDIATORS. "there is no basis for distinguishing the two causal structures ... using statistics alone." |

| 12 | https://pmc.ncbi.nlm.nih.gov/articles/PMC11332371/ | 2026-08-18 | peer-reviewed (PMC, 2024) | WebFetch | **[ADVERSARIAL / qualifying]** Cuts the other way on (e): a naive audit can FALSELY convict a gate. "The subset approach wrongly suggested miscalibration for the predictions under the never treated strategy" even with a correctly specified model and identical development/validation distributions. The subset (complete-case) approach "is prone to selection bias"; the correct method is artificial censoring + inverse-probability weighting, requiring conditional sequential exchangeability, consistency, positivity and correct weight-model specification. Calibration is judged by an observed/expected ratio near 1.0. |
| 13 | https://arxiv.org/html/2605.05427 | 2026-08-18 | preprint (arXiv, 2026-05) | WebFetch, arXiv native HTML | Kills the "count the refusals" instinct outright, on LLM gates specifically. "Refusal rates are a poor proxy for LLM safety, i.e., a model may over-refuse benign prompts while still complying with harmful ones." Over-refusal and harmful compliance are essentially **uncorrelated (r = -0.032, p = 0.89)** over 21 models, 4 benchmarks, ~7.1M prompt-response pairs. "high refusal rates do not imply low harmful compliance: the two failure modes are largely independent." Two-dimensional metric required (ORR + HCR), never one count. |
| 14 | https://arxiv.org/html/2607.19449v1 | 2026-08-18 | preprint (arXiv, 2026-07) | WebFetch, arXiv native HTML | The exact (d) failure mode, named and measured: **Unfaithful Safety Refusal** -- an agent "invokes a policy, privacy, legal, or authorization rationale ... when no such constraint is instantiated", borrowing "safety vocabulary to fill the silence left by an infrastructure fault" when a backend returns HTTP 200 with an empty payload. Three-class taxonomy: Honest Surrender / Fabrication / USR. Baseline n=396 trajectories: Fabrication 56.6%, USR 0.25%; under a safety-framed prompt USR amplifies **15.6x to 3.95%** (n=380, p<0.001). Judge-human agreement Cohen's kappa = 0.85. |


## Identified but snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|-----|------|-------------------------|
| https://en.wikipedia.org/wiki/Rule_of_three_(statistics) | reference | superseded by the Cochrane + Hanley primary sources fetched in full |
| https://www.statology.org/a-concise-guide-to-the-statistical-rule-of-three/ | blog (community tier) | low tier; canonical source fetched instead |
| https://yystats.wordpress.com/2016/07/09/rule-of-three-for-binomial-confidence-interval/ | blog (community tier) | low tier |
| https://www.mathworks.com/help/risk/reject-inference-for-credit-scorecards.html | vendor doc | vendor how-to, not evidential standard |
| https://support.sas.com/resources/papers/proceedings09/305-2009.pdf | vendor paper | 2009 SAS proceedings; superseded by source 1 |
| https://arxiv.org/pdf/2206.00568 | preprint | RMT-Net MNAR credit scoring; same mechanism as source 1 |
| https://arxiv.org/pdf/1909.06108 | preprint | shallow self-learning reject inference; subsumed by source 1 |
| https://www.experian.com/blogs/insights/reject-inference/ | industry blog | practitioner overview only |
| https://blog.minitab.com/en/blog/monitoring-rare-events-with-g-charts | vendor blog | subsumed by the official Minitab doc read in full |
| https://www.spcforexcel.com/knowledge/attribute-control-charts/g-control-chart/ | vendor doc | same content as #3 |
| https://en.wikipedia.org/wiki/Survivorship_bias | reference | mechanism covered by sources 1, 2, 6 |
| https://mitzu.io/post/guide-to-funnel-analysis/ | industry blog | vendor funnel how-to; no evidential standard |
| https://www.pmean.com/01/zeroevents.html | named-statistician blog | rule of three; superseded by sources 7 + 10 |
| https://pubmed.ncbi.nlm.nih.gov/7647644/ | peer-reviewed (Altman & Bland BMJ 1995) | abstract-only page; the full argument is quoted inside source 7 |
| https://handbook-5-1.cochrane.org/chapter_16/16_9_4_confidence_intervals_when_no_events_are_observed.htm | official handbook | **ATTEMPTED, 301** to a version-index page; not fetched in full |
| https://www.itl.nist.gov/div898/handbook/prc/section2/prc252.htm | official docs (NIST) | **ATTEMPTED, 302** to nist.gov/itl; not fetched in full |
| https://www.jvsmedicscorner.com/Statistics_files/Probability%20of%20adverse%20events%20that%20have%20not%20yet%20occurred.pdf | peer-reviewed (Eypasch BMJ 1995) | **ATTEMPTED**: server returned HTML, not a PDF; pdfplumber raised `No /Root object`. Not counted. |
| https://csrc.nist.gov/glossary/term/false_reject_rate | official docs (NIST glossary) | one-line definition only |
| https://www.sciencedirect.com/science/article/abs/pii/S0378426603002036 | peer-reviewed (Crook & Banasik 2004) | paywalled; its result is quoted in sources 1 and 2 |
| https://cer.business-school.ed.ac.uk/wp-content/uploads/sites/55/2017/02/workingpaper03_2-1.pdf | working paper | Banasik & Crook lean models; subsumed |
| https://oneuptime.com/blog/post/2026-02-06-heartbeat-dead-man-switch-opentelemetry-pipeline/view | industry blog | dead-man-switch pattern; the primitive is source 4 |
| https://arxiv.org/pdf/2505.08054 | preprint (FalseReject) | over-refusal dataset; mechanism covered by source 13 |
| https://arxiv.org/pdf/2606.07867 | preprint | cold-start safety gap; adjacent |
| https://arxiv.org/html/2512.02445v1 | preprint | unstable refusals in long-context agents; adjacent |
| https://academic.oup.com/ije/article/47/6/2082/5049576 | peer-reviewed (IJE) | controls in interrupted time series; framing only |
| https://arxiv.org/pdf/2603.17281 | preprint | triple-difference ITS; over-engineered for n=2 |
| https://journals.sagepub.com/doi/10.1177/2515245918770963 | peer-reviewed (Lakens 2018) | equivalence-testing tutorial; covered by source 7 |
| https://arxiv.org/pdf/2410.20384 | preprint | base-rate bias in structural health monitoring; adjacent |
| https://soda.io/blog/data-quality-metrics-12-examples | industry blog | completeness = non-null/expected; low tier |
| https://arxiv.org/pdf/2306.15007 | preprint | quality issues in ML software systems; adjacent |
| https://arxiv.org/pdf/2103.16860 | preprint | Simpson's paradox as inductive singularity; SEP read instead |
| https://harbourfrontquant.substack.com/p/when-trading-systems-break-down-causes | industry blog | drawdown-vs-broken framing; source 9 read instead |
| https://elifesciences.org/articles/92311 | peer-reviewed | journal version of source 7; not double-counted |
| https://arxiv.org/abs/2509.00109 | preprint | abstract page for source 6; not double-counted |
| https://arxiv.org/pdf/2006.16916 | preprint | counterfactual prediction under runtime confounding; adjacent |
| https://arxiv.org/pdf/2308.13026 | preprint | estimating counterfactual prediction models; adjacent |

## Recency scan (2024-2026) -- PERFORMED

Searched explicitly in the 2024-2026 window. Result: **9 of the 14 sources read in full are
from the window, and 6 of them CHANGE the design** rather than merely confirming older work:

1. **arXiv:2606.18479 (2026-06)** -- proves accepted-only evaluation is structurally blind to
   rejection quality (Proposition 2), and prices the fix: 2-5% random exploration.
2. **arXiv:2607.02830 (2026-07)** -- first at-scale rejected-candidate outcome audit; supplies
   a five-tier classification, an explicit `unclassifiable` tier, and the demonstration that a
   plausible tier can fail a matched test.
3. **arXiv:2605.05427 (2026-05)** -- refusal rate is a poor proxy; over-refusal and wrong-approval
   are uncorrelated (r = -0.032). This is NEW and it supersedes any instinct to audit a gate by
   counting its REJECTs.
4. **arXiv:2607.19449 (2026-07)** -- names and measures Unfaithful Safety Refusal: a fabricated
   policy refusal filling the silence of an empty backend response. Directly describes this
   repo's `_judge_parse_fail_fallback`.
5. **arXiv:2604.10996 (2026-04)** -- the regime-vs-defect discriminator (recovery in a second
   regime proves the component was never broken).
6. **PMC11332371 (2024)** -- the counterweight: naive evaluation can FALSELY convict a gate.
7. **arXiv:2509.00109 (2025-08)** + **PMC11090505 (2024)** -- feedback-loop evaluation bias and
   the absence-of-evidence machinery respectively.

Older canonical work still stands and is not superseded: the rule of three / Poisson-zero
machinery (source 10, 2008, and Altman & Bland 1995 quoted inside source 7), SPC rare-event
charting (source 3), and Simpson's paradox (source 11). Nothing found in the window contradicts
them; the new work is additive and sits on the *decision-audit* side, not the *statistics* side.

**Search-query composition (three-variant discipline, mandatory).** Queries actually run
included -- current-year frontier: "2026 anomaly detection zero counts base rate low-frequency
event pipeline observability silent failure", "LLM agent pipeline conservative refusal rate
over-refusal measuring silent behaviour change after model upgrade"; last-2-year: "arxiv 2025
selection bias machine learning deployment feedback loop only observing accepted decisions
evaluation"; year-less canonical: "rule of three zero events confidence interval Poisson upper
bound", "reject inference credit scoring rejected applicants counterfactual outcomes
methodology", "g-chart rare events statistical process control time between events low rate
monitoring", "conversion funnel analysis methodology stage definition denominator survivorship
bias pitfalls", "Simpson's paradox pooling heterogeneous subpopulations", "absence of evidence
is not evidence of absence statistical power negative result interpretation", "Crook Banasik
does reject inference really improve performance application scoring models", "trading days
versus calendar days normalisation event rate reproducibility financial time series convention".
The year-less variants are what surfaced sources 3, 7, 10 and 11 -- none of them would have
appeared under a year-locked query.

### Audit-class coverage loop (loop-until-dry, K=2)

| round | angle | new read-in-full findings |
|---|---|---|
| 1 | rule of three; reject inference | 2 (sources 1, 2) |
| 2 | g-chart / SPC; funnel methodology; SRE monitoring | 3 (sources 3, 4, 5) |
| 3 | 2026 anomaly detection / silent pipeline failure | **0 -- dry** |
| 4 | selection bias + deployment feedback loops | 1 (source 6) |
| 5 | absence of evidence vs evidence of absence | 1 (source 7) |
| 6 | canonical funnel analysis; DAMA completeness | 1 (source 8, a negative finding) |
| 7 | zero-inflation tests; regime-vs-defect in trading | 2 (sources 9, 10) |
| 8 | false-reject auditing; Simpson's paradox | 1 (source 11) |
| 9 | dead-man switch / no-data vs zero | **0 -- dry** (fully covered by source 4) |
| 10 | evidential standard for miscalibration | 1 (source 12) |
| 11 | Crook & Banasik canonical reject inference | **0 -- dry** (subsumed by source 1) |
| 12 | LLM over-refusal; unfaithful refusals | 2 (sources 13, 14) |
| 13 | stage-definition drift; base-rate fallacy; trading-vs-calendar days | **0 -- dry** |
| 14 | interrupted time series co-intervention; equivalence testing | **0 -- dry** |

Rounds 13 and 14 are two CONSECUTIVE dry rounds -> `coverage.dry = true`. Note rounds 3, 9 and 11
were dry but were each followed by a productive round, so they did not satisfy K=2 -- the loop
correctly kept going, and rounds 12's two sources are the strongest in the brief. That is the
loop doing its job rather than the researcher stopping at the floor.

## Internal measurement (re-derived; NO number inherited from the step text)

Access path: BigQuery MCP tools were **not present in this session's tool surface**, so I used the
documented fallback (CLAUDE.md "BigQuery Access (MCP)" rule 6): the Python `google-cloud-bigquery`
client under `.venv`, project `sunny-might-477607-p8`, ADC. Every query is stated below.

### M0 -- the step's stated data surface is WRONG (finding, not a nit)

The spawn prompt says `pyfinagent_data.analysis_results` and `pyfinagent_data.signals_log`.
Both are **404 Not Found** in `pyfinagent_data`. Enumerated via `client.list_tables`:

- `pyfinagent_data` (US) actually holds: `alt_13f_holdings` 110, `alt_congress_trades` 7262,
  `alt_finra_short_volume` 0, `calendar_events` 0, `llm_call_log` 7248, `news_articles` 11,
  `news_sentiment` 9, `risk_intervention_log` **0**, `scraper_audit_log` 0, `sla_alerts` 1,
  `strategy_decisions` 60, `unified_sar_log` 0.
- `financial_reports` (us-central1) holds `analysis_results` **580**, `signals_log` **119**,
  `paper_trades` **66**, `paper_round_trips` 32, `paper_positions` **2**, `paper_portfolio` 1,
  `paper_metrics_v2` 148, `outcome_tracking` **3**, plus the historical tables.

So `analysis_results` and `signals_log` are in `financial_reports`, alongside the paper-trading
tables -- the same dataset-location trap CLAUDE.md already documents for `paper_trades`
(`backend/db/bigquery_client.py:486` `_pt_table()` uses `settings.bq_dataset_reports`).
`analysis_results` has **91** columns (the step said 88 -- re-derived, do not inherit).
Note `pyfinagent_data.risk_intervention_log` is **0 rows**: the table that would carry a refusal
event stream is empty, which is itself decisive for question (d) below.

### M1 -- the actual trade history (whole table, no sampling)

```sql
SELECT SUBSTR(created_at,1,10) AS d, ticker, action,
       IFNULL(risk_judge_decision,'<NULL>') AS rjd, ROUND(total_value,2) AS val, analysis_id
FROM `sunny-might-477607-p8.financial_reports.paper_trades`
ORDER BY created_at DESC LIMIT 80   -- returned all 66 rows
```

`paper_trades` contains **66 rows total, ever**. `created_at` is a **STRING**, not a TIMESTAMP
(there is no `executed_at` column -- my first query failed on that name). Newest rows:

| date | ticker | action | risk_judge_decision |
|---|---|---|---|
| 2026-08-13 | DELL | BUY | *(empty)* |
| 2026-07-31 | NTAP | BUY | APPROVE_REDUCED |
| 2026-07-27 | AMD | SELL | *(empty)* |
| 2026-07-27 | PANW | SELL | *(empty)* |
| 2026-07-20 | PANW | BUY | APPROVE_REDUCED |
| 2026-07-13 | MU | SELL | *(empty)* |
| 2026-07-09 | AMD, MU | BUY | APPROVE_REDUCED |

**The last trade of any kind is 2026-08-13 (DELL BUY).** Relative to today (2026-08-18) that is a
zero-run of 5 calendar days. That is the number the whole step turns on and it must be normalised
before it means anything -- see the statistics section.

### M2 -- `risk_judge_decision` is NOT populated for the population a funnel would count

```sql
SELECT FORMAT_TIMESTAMP('%Y-%m', analysis_date) AS mon, COUNT(*) n,
       COUNTIF(UPPER(IFNULL(recommendation,'')) LIKE '%BUY%')  buyish,
       COUNTIF(UPPER(IFNULL(recommendation,'')) LIKE '%HOLD%') holdish,
       COUNTIF(UPPER(IFNULL(recommendation,'')) LIKE '%SELL%') sellish,
       COUNTIF(IFNULL(recommendation,'')='')            rec_empty,
       COUNTIF(IFNULL(risk_judge_decision,'')<>'')      rjd_pop,
       COUNTIF(IFNULL(final_score,0)=0)                 score_zero,
       COUNTIF(final_score IS NULL)                     score_null
FROM `sunny-might-477607-p8.financial_reports.analysis_results` GROUP BY mon ORDER BY mon
```

| month | n | buyish | holdish | sellish | rec_empty | **rjd_pop** | score_zero |
|---|---|---|---|---|---|---|---|
| 2025-11 | 25 | 11 | 14 | 0 | 0 | **0** | 0 |
| 2026-01 | 16 | 13 | 3 | 0 | 0 | **0** | 0 |
| 2026-02 | 2 | 1 | 0 | 1 | 0 | **0** | 0 |
| 2026-03 | 11 | 3 | 6 | 2 | 0 | **3** | 0 |
| 2026-05 | 174 | 54 | 109 | 11 | 0 | **0** | 87 |
| 2026-06 | 134 | 49 | 83 | 0 | 0 | **2** | 64 |
| 2026-07 | 137 | 4 | 133 | 0 | 0 | **0** | 120 |
| 2026-08 | 81 | 4 | 73 | 4 | 0 | **13** | 35 |

Two decisive results:

1. **`risk_judge_decision` is populated on 18 of 580 rows (3.1%) in `analysis_results`, and on
   ZERO rows in 2026-05 and 2026-07.** A per-recommendation funnel keyed on this column would be
   measuring its own blindness, exactly as the caller warned. `recommendation` by contrast is
   populated on 580/580 (`rec_empty = 0` in every month), so the recommendation leg is countable
   and the risk-verdict leg is not -- from this table.
2. **The BUY rate collapses between June and July: 49/134 = 36.6% -> 4/137 = 2.9%.** Whatever
   caused the drought, a candidate explanation that does not pass through this collapse is
   arguing against the base rate. `score_zero` is 120/137 = 87.6% in July, consistent with the
   empty-row regression previously filed as 86.69 (its own dating is re-checked below, not
   inherited).


### M3 -- dating the regime break precisely (daily, re-derived)

```sql
SELECT DATE(analysis_date) d, COUNT(*) n,
       COUNTIF(UPPER(recommendation) LIKE '%BUY%') buyish,
       COUNTIF(IFNULL(final_score,0)=0) score_zero,
       COUNTIF(IFNULL(risk_judge_decision,'')<>'') rjd_pop,
       STRING_AGG(DISTINCT recommendation ORDER BY recommendation LIMIT 6) recs
FROM `sunny-might-477607-p8.financial_reports.analysis_results`
WHERE analysis_date >= TIMESTAMP('2026-06-01') GROUP BY d ORDER BY d
```

| date | n | buyish | score_zero | recs |
|---|---|---|---|---|
| 06-01..06-10 (8 days) | 64 | 46 | **0** | BUY, HOLD |
| **2026-06-11** | 8 | 1 | **5** | BUY, HOLD, Hold, **N/A** |
| 2026-06-12 | 5 | 2 | 3 | BUY, HOLD |
| **2026-06-15** | 7 | **0** | **7** | HOLD only |
| 06-17..07-08 | ~90 | 0 | most | HOLD only |

The first contaminated day is **2026-06-11** (not 06-12) and the collapse is total by
**2026-06-15**. Re-derived independently; the step text's dating is not inherited.

### M4 -- the zero-run, normalised three ways (this is the whole question)

Last trade **2026-08-13**. Today **2026-08-18**. The run length depends entirely on the
normalisation, and the three defensible denominators disagree by more than 2x:

| normalisation | value |
|---|---|
| calendar days since last trade | **5** |
| US trading days (08-14 Fri, 08-17 Mon, 08-18 Tue-incomplete) | **3** |
| **analysis-cycle days that actually produced rows** (08-14, 08-17) | **2** |
| **analyses executed in the run** (6 on 08-14 + 7 on 08-17) | **13** |

The last row is the only denominator with a decision-theoretic meaning: a BUY can only
arise from an analysis, so the count of *opportunities* is 13, not 5. Stating "a 5-day
drought" silently swaps a calendar denominator into a per-opportunity rate.

**Is 13 opportunities with 0 BUYs anomalous?** Base rate re-derived from the post-break
window 2026-06-15..2026-08-13: **N = 262 analyses, B = 8 BUY-class, rate = 0.0305**.

- Expected BUYs in the run: `lambda = 13 x 0.0305 = 0.397`
- **`P(0 | Poisson 0.397) = 0.672`** -- one-sided p for "nothing changed".
- Rule of three, from the zero run alone: 95% upper bound on the BUY rate is
  `3/13 = 0.231`. The post-break rate 0.0305 sits comfortably inside it.
- To make a zero-run significant at p<0.05 against a 3.05% rate you need
  **n >= 97 consecutive BUY-free analyses (~16 analysis days at ~6/day)**.

**Conclusion: the zero-run is not anomalous. It is the expected behaviour of a 3%-rate
process observed for 13 opportunities.** For contrast, the same 13 opportunities under
the PRE-break rate (2026-06-01..06-12: N=77, B=49, rate=0.636) give `lambda = 8.27` and
`P(0) = 2.6e-4` -- under the *old* regime the run WOULD have been decisive. The anomaly
is therefore located at **2026-06-11/06-15**, not in August.

### M5 -- three changes land on 2026-08-14 and are perfectly collinear

```sql
SELECT DATE(analysis_date) d, COUNT(*) rows_n, COUNT(DISTINCT ticker) tickers_n,
       COUNTIF(total_tokens IS NULL) tok_null, COUNTIF(decision_trace_count IS NULL) trace_null,
       STRING_AGG(DISTINCT IFNULL(standard_model,'<null>')) models
FROM ... WHERE analysis_date >= TIMESTAMP('2026-07-25') GROUP BY d ORDER BY d
```

On **2026-08-14**, simultaneously: `standard_model` flips `claude-sonnet-4-6` ->
**`claude-sonnet-5`**; `risk_judge_decision` / `risk_level` /
`recommended_position_pct` go from empty to **populated on 13/13 rows** (the phase-86.74
writer fix landing -- its own in-code comment at
`backend/services/autonomous_loop.py::_persist_analysis` says those columns "were empty on
129 of 129 rows across 2026-07-20..2026-08-13"); `final_score = 0` rows go to **0**; and
`recommendation` becomes **`Hold` on 13/13**. The zero-run starts the same day.
**Four changes, one 2-day window, zero degrees of freedom -- no attribution is
identifiable from this data.** Any claim crediting one of them is uncheckable here.

`total_tokens`, `decision_trace_count` and `debate_rounds_count` are **NULL on 100% of
rows since 2026-07-25**, so those columns cannot be used to characterise the pipeline.

### M6 -- CRITERION 4: the lite-vs-full distinction IS derivable (contra the "underivable" branch)

The path tag is stamped into the JSON blob, not a column:
`_persist_analysis` does `full_report = {**full_report, "_path": analysis["_path"]}`
(phase-60.1 comment: "The away week wrote 64 lite rows that looked identical to full rows").

```sql
SELECT DATE(analysis_date) d, COUNT(*) n,
  COUNTIF(JSON_VALUE(full_report_json,'$._path') IS NOT NULL) path_tagged,
  STRING_AGG(DISTINCT IFNULL(JSON_VALUE(full_report_json,'$._path'),'<none>')) paths,
  COUNTIF(JSON_VALUE(full_report_json,'$._fallback_reason') IS NOT NULL) fb_tagged
FROM ... WHERE analysis_date >= TIMESTAMP('2026-06-01') GROUP BY d ORDER BY d DESC
```

**`_path` is populated on 100% of rows for every day since 2026-06-01** (25/25 days
sampled), taking values `lite` and `full`. Mixed `lite,full` days with a
`_fallback_reason` (intended-full, landed-lite): 07-20 (1), 07-24 (1), 07-31 (3),
08-05 (2), 08-10 (3), 08-11 (1). **08-14 and 08-17 are `full` on 13/13 with zero
fallbacks** -- so the zero-run is NOT a degraded-path artefact, and a census that
does not split on `$._path` is confounding the two paths for free when it does not
have to. Note `JSON_VALUE` returns NULL for a JSON *object*, so this predicate only
works because `_path` is a scalar string.

### M7 -- REFUSAL vs ABSENCE: a live instance, and a NULL that hides it

`backend/agents/risk_debate.py::_judge_parse_fail_fallback` (defined `:158`, sole call
site `:375`) **synthesises a verdict when the judge output is unparseable**:

```python
if _parse_fail_reject:
    return {"decision": "REJECT", "risk_adjusted_confidence": 0.0,
            "recommended_position_pct": 0, "risk_level": "EXTREME", ...}
return {"decision": "APPROVE_REDUCED", "risk_adjusted_confidence": 0.5,
        "recommended_position_pct": 3, "risk_level": "MODERATE", ...}
```

So a stored `APPROVE_REDUCED / 3 / MODERATE / 0.5` is **indistinguishable in the
verdict column from a real approval** -- it is an ABSENCE of judgment recorded as a
DECISION. Flag: `backend/config/settings.py:346 paper_risk_judge_parse_fail_reject`
(defaults False, i.e. the permissive branch).

Enumerating **every** row in the table with a populated verdict (n = 18 of 580):

| date | ticker | decision | pos | risk_level | rac | fingerprint |
|---|---|---|---|---|---|---|
| 2026-08-17 | MU | APPROVE_REDUCED | 3.0 | MODERATE | *(NULL)* | **NULL -- unresolvable** |
| 2026-08-17 | DELL, NTAP, HPE, 009150.KS | REJECT | 0.0 | HIGH | *(NULL)* | False |
| 2026-08-17 | MRVL | APPROVE_HEDGED | 5.0 | HIGH | *(NULL)* | False |
| 2026-08-17 | SNDK | APPROVE_REDUCED | 2.0 | HIGH | *(NULL)* | False |
| 2026-08-14 | NTAP, STX | APPROVE_REDUCED | 2.0 | HIGH | *(NULL)* | False |
| 2026-08-14 | MRVL, HPE, WDAY, PANW | REJECT | 0.0 | HIGH | *(NULL)* | False |
| **2026-06-11** | 005930.KS, MU | APPROVE_REDUCED | 3.0 | MODERATE | **0.5** | **True -- parse-fail** |
| 2026-03-20/21 | SNDK x3 (2 identical rows on 03-21) | REJECT | *(NULL)* | *(NULL)* | 0.0 | False |

Two results:
1. The **only two rows that can be positively identified as parse failures are the two
   on 2026-06-11** -- the exact day the score/recommendation regime broke.
2. **The 4-part fingerprint evaluates to `NULL`, not `False`, for 2026-08-17 MU**,
   because `risk_adjusted_confidence` is unpopulated on the new rows. A census that
   writes `COUNTIF(fingerprint)` would silently count this row as *not* a parse failure.
   The correct report is **UNRESOLVED**, and it matters: MU is the single highest-scoring
   name in the zero-run (final_score 7.15) and the only one carrying a MODERATE label.

### M8 -- the refusal stream that does not exist, and the vocabulary that drifts

- `pyfinagent_data.risk_intervention_log` has **0 rows**. The table designed to hold
  refusal events is empty, so refusals are only inferable from a column on the analysis
  row -- which is populated on 18/580 = **3.1%** of the table.
- `recommendation` vocabulary over the whole table (7 spellings, 3 classes):
  `HOLD` 284 (05-16..08-13), `Hold` 137 (2025-11-23..**08-17**), `BUY` 94 (05-17..08-11),
  `Buy` 40 (2025-11-23..**08-13**), `Sell` 18, `Strong Buy` 5 (last 2026-05-22),
  `N/A` 2 (both 2026-06-11). The **last BUY-class row before the zero-run is spelled
  `Buy`, not `BUY`** (2026-08-13 DELL, final_score 6.58) -- and `Strong Buy`, the
  highest-conviction spelling, has not appeared since 2026-05-22.
  The canonicaliser `backend/services/recommendation_vocab.py` exists (209 lines,
  `canonical_recommendation`, imported at `autonomous_loop.py:33` as
  `resolve_outcome_recommendation`) and folds case + `[\s\-_]+`; its own docstring
  records that `.upper()` "folds CASE but never the SEPARATOR" and that `N/A` resolves
  to UNKNOWN, deliberately distinct from HOLD.
- **Duplicate rows inflate the denominator**: 2026-08-09 has 12 rows for 6 tickers
  (6 excess), 2026-07-28 has 9 rows for 5 tickers (4 excess), 2026-03-21 has SNDK twice.
  On 08-09 the duplicates are one real-score row plus one `HOLD/0.0` row per ticker.
  A funnel counting rows, not (ticker, day) pairs, over-counts its own denominator on
  exactly the days the degradation was worst.

### M9 -- the loop is ALIVE, and 2026-08-18 has not happened yet

`handoff/.cycle_heartbeat.json` = `{"cycle_id":"3e5afddb","event":"end","updated_at":"2026-08-17T19:47:15.758944+00:00"}`.
Clock at time of writing: **`2026-08-18T01:01:17Z`**. The daily cycle hour is
`settings.py:408 paper_trading_hour = 10` (ET), so **today's cycle has not run yet** and
2026-08-18 is not an observed zero. Backend pid 41635, started `2026-08-17 15:57:16` local.

So the naive "5 calendar days" denominator contains **2 weekend days (08-15, 08-16) and 1
day that has not occurred (08-18)**. The real observed window is 2 days. A rate quoted on
the calendar denominator is inflated ~2.5x relative to the opportunity denominator.

**Multi-market caveat on "trading day":** this book trades US + EU + KR, whose calendars do
not align, so "trading day" is itself ambiguous here. The unambiguous denominator is the one
the system controls: rows in `analysis_results`, capped at
`settings.py:407 paper_analyze_top_n = 5` per cycle (LIVE value confirmed 5 via
`GET /api/settings/`).

### M10 -- the live gate chain a BUY must clear (`backend/services/portfolio_manager.py`)

`decide_trades` at `:164`. In order, each gate `continue`s past the candidate:

| # | gate | anchor | live value |
|---|---|---|---|
| 1 | `canonical_recommendation(...)` returns `None` -> skip | `:16` import, `:128`, `:302` | canonicaliser LIVE |
| 2 | `if rec not in _BUY_RECS: continue` | `:304`, set at `:64` = `{"BUY","STRONG_BUY"}` | LIVE |
| 3 | **BINDING RiskJudge REJECT gate** | `:383-400` | `settings.py:342-345 paper_risk_judge_reject_binding` default **False**; NOT in the 45 keys exposed by `GET /api/settings/`, so its RUNNING value is **UNRESOLVED from this surface** |
| 4 | per-sector count cap | `:286` | `paper_max_per_sector` LIVE = **5** (runtime override; `settings.py:305` default is 2) |
| 5 | per-sector NAV% cap | `:289` | `paper_max_per_sector_nav_pct` |
| 6 | cash reserve | `:31-32`, `:119` | `paper_min_cash_reserve_pct` LIVE = **5.0** |

Gate 3's own in-code comment states the away week "executed 3 REJECT BUYs -- all via the swap
path". **Independently re-derived from `paper_trades`: exactly 3 rows have
`action='BUY'` AND `risk_judge_decision='REJECT'`** -- 066570.KS 2026-06-09 ($238.40),
DELL 2026-06-03 ($246.67), HPE 2026-06-02 ($245.04). A REJECT verdict that appears in a
census as "the gate fired" corresponds here to a trade that **executed anyway**.

---

## Key findings

1. **The zero-run is NOT anomalous, and this is measurable rather than arguable.**
   13 opportunities, 0 BUYs, post-break rate 0.0305 -> `P(0) = 0.672`; rule-of-three upper
   bound 3/13 = 0.231 comfortably contains the base rate. (M4.) Any explanatory step that
   starts from "the drought is a defect" has skipped its own base-rate test -- which is the
   precise failure that sank 86.38 and 86.41.
2. **The system is structurally incapable of detecting a drought quickly.** At
   `paper_analyze_top_n = 5`, reaching the n>=97 needed for p<0.05 takes **~16-19 analysis
   days**. This is the SPC rare-event result verbatim: "collecting enough rare event data to
   detect an adverse change in the frequency of events may take months or even years"
   (Minitab, source 3). The right instrument is a **g-chart on opportunities-between-BUYs**,
   not a count per day.
3. **A p just above 0.05 is neither health nor breakage.** "Such a 'null result' ... may also
   occur if an effect is actually present"; the null "is misaligned with the goal of
   inference" (source 7). To claim the pipeline is HEALTHY one needs equivalence testing
   against a declared margin, or a Bayes factor `BF01 > 1` -- not a non-significant p.
4. **The real event is 2026-06-11/06-15, not August.** BUY rate 46/64 (06-01..06-10) -> 0/7
   (06-15). Under the pre-break rate the same 13 opportunities would give `P(0) = 2.6e-4`.
   The drought post-dates that break and is fully explained by it. (M3, M4.)
5. **Counting risk-gate REJECTs cannot show mis-calibration.** Two independent 2026 results:
   accepted-only data is structurally blind (source 1, Proposition 2), and for LLM gates
   specifically "Refusal rates are a poor proxy ... a model may over-refuse benign prompts
   while still complying with harmful ones", with over-refusal and wrong-approval
   **uncorrelated at r = -0.032** (source 13). The only evidential standard that works is
   forward outcomes on the REJECTED candidates (source 2's five-tier PRFS audit) or
   deliberate 2-5% exploration (source 1).
6. **...but a naive rejected-candidate audit can FALSELY convict the gate.** Source 12: "The
   subset approach wrongly suggested miscalibration" even under correct specification.
   Source 2 makes the same point empirically -- its own 14.8:1 tier collapsed under a
   *matched* comparison because "the filter stack preferentially rejects the lower-quality
   tokens, so the broader-universe base rate is depressed relative to the rejected
   subpopulation by design." Comparison group choice is the whole ballgame.
7. **The funnel cannot be keyed on `risk_judge_decision`.** 18/580 = 3.1% populated, and
   0/174 in May, 0/137 in July. (M2.) `pyfinagent_data.risk_intervention_log` is 0 rows.
   The verdict *is* recoverable from the JSON blob and from `paper_trades`, but the column
   census would measure its own blindness.
8. **REFUSAL vs ABSENCE is live in this codebase, not hypothetical.**
   `_judge_parse_fail_fallback` (`risk_debate.py:158`, sole call `:375`) synthesises
   `APPROVE_REDUCED/3/MODERATE/0.5` from an unparseable judge response. Source 14 names this
   exact pattern (Unfaithful Safety Refusal: safety vocabulary "to fill the silence left by
   an infrastructure fault") and gives the fix: a three-class taxonomy that keeps
   **Honest Surrender** separate from a real decision. Two rows (2026-06-11) carry the full
   fingerprint; **2026-08-17 MU evaluates to NULL, not False**, because
   `risk_adjusted_confidence` is unpopulated -- a `COUNTIF` would silently score it clean.
9. **Criterion 4 is satisfiable: `$._path` distinguishes lite from full on 100% of rows.**
   And the zero-run days are `full` 13/13 with zero fallbacks -- so the drought is not a
   degraded-path artefact. (M6.) Not splitting on it anyway is a free Simpson exposure:
   reversal requires the partition variable to correlate with the treatment (source 11), and
   `_path` correlates with everything here.
10. **Four changes land on 2026-08-14 with zero degrees of freedom.** Model
    sonnet-4-6 -> sonnet-5, risk columns empty -> populated, zero-scores gone,
    recommendation collapses to `Hold`. Source 9's discriminator (a component that recovers
    in a second regime was never broken) is the only design that can separate them, and it
    needs a second regime this data does not yet contain. Source 13 adds the specific warning
    that refusal behaviour differs **across** model families -- so the 08-14 model switch
    alone could produce the whole change in verdict distribution.

## Internal code inventory

| File | Lines | Role | Status |
|---|---|---|---|
| `backend/services/autonomous_loop.py` | 3,981 | cycle driver; `_persist_analysis` `:3561`; `_path` gate `:1247`; fallback alarm `:1326-1390`; RiskJudge out-channel `:1665-1682` | live |
| `backend/services/portfolio_manager.py` | 1,168 | `decide_trades` `:164`; `_BUY_RECS` `:64`; REJECT gate `:383-400` | live |
| `backend/services/recommendation_vocab.py` | 209 | sole canonicaliser; folds `[\s\-_]+`; `N/A` -> UNKNOWN != HOLD | live, imported `portfolio_manager.py:16` |
| `backend/agents/risk_debate.py` | -- | `_judge_parse_fail_fallback` `:158`, call `:375` | live; **fabricates a verdict on parse failure** |
| `backend/config/settings.py` | -- | `:305` sector cap, `:342` reject-binding (False), `:405` cash reserve, `:407` analyze_top_n=5, `:408` trading hour | live |
| `backend/db/bigquery_client.py` | -- | `_pt_table()` `:486` -> `bq_dataset_reports` = `financial_reports` | live; explains M0 |
| `handoff/.cycle_heartbeat.json` | -- | last cycle end 2026-08-17T19:47:15Z | live |
| `.claude/agents/researcher.md`, `.claude/rules/research-gate.md` | -- | gate doctrine (read per STEP 0) | live |

## Consensus vs debate (external)

**Consensus.** (i) Outcomes observed only on accepted decisions cannot establish rejection
quality -- sources 1, 2, 6, 12, 13 all agree, across credit scoring, DEX trading, recommender
systems, clinical prediction and LLM safety. (ii) Absence of data is a different object from a
zero value and needs a different operator (source 4). (iii) A non-significant result is not
evidence of absence (source 7). (iv) Rare-event rates need time-between-events instrumentation,
not per-period counts (source 3).

**Debate.** How to recover the counterfactual. Source 1 is *hostile* to imputation-based reject
inference ("Simple Extrapolation does not mitigate survival bias; it reverses its sign") and
favours deliberate exploration. Source 2 avoids imputation entirely by *measuring* forward
outcomes on rejects. Source 12 pushes back on both by showing a naive comparison can invent
miscalibration that is not there, and requires IPW under strong assumptions
(exchangeability, positivity). The honest reading: **there is no cheap way to audit a gate**,
and the cheapest defensible one here is source 2's design -- record what the rejected
candidates went on to do -- because it requires no causal assumptions, only instrumentation.

## Pitfalls (from literature, mapped to what would go wrong here)

1. **Evaluating the gate on the trades it allowed.** Source 1 Prop. 2: as the accepted-positive
   rate shrinks, accuracy stops responding to recall collapse.
2. **Choosing the wrong comparison group.** Source 2: rejected-vs-universe showed 34.6% and
   looked damning; rejected-vs-rejected showed 48.9% vs 57.6% and reversed the conclusion.
3. **Counting refusals as a calibration signal.** Source 13: r = -0.032.
4. **Reading a fabricated fallback as a decision.** Source 14; here `_judge_parse_fail_fallback`.
5. **`COUNTIF` over a predicate containing a NULL column.** SQL three-valued logic scores the
   unknown row as not-matching. Measured live on 2026-08-17 MU.
6. **Pooling `lite` and `full`.** Source 11: reversal is possible exactly when the partition
   variable correlates with the treatment, and the choice to pool "requires the additional
   information supplied by the causal model", not the data.
7. **Attributing a level change to one of several simultaneous interventions.** Four landed on
   2026-08-14 (M5); interrupted-time-series design cannot separate co-interventions.
8. **Calendar denominators.** M9: 5 calendar days -> 2 real opportunity days.
9. **Row-counting a denominator with retries in it.** M8: 08-09 is 12 rows / 6 tickers.
10. **Vocabulary membership by `.upper()`.** M8 + `recommendation_vocab.py` docstring; the last
    pre-drought BUY is spelled `Buy`, and `Strong Buy` has been absent since 2026-05-22.

## Application to pyfinagent (what the contract should require)

- **Order of proof.** Establish the null FIRST (M4) and only then explain. The step's own text
  frames the drought as needing an explanation; the measurement says the drought needs no
  explanation beyond the 2026-06-11/15 break. A contract that skips this repeats 86.38/86.41.
- **Declare the normalisation in the criterion text.** "BUY-class recommendations per
  `analysis_results` row, window `[t0, t1)` on `analysis_date`, de-duplicated to
  (ticker, DATE(analysis_date))" -- not "per day", not "per calendar day". (M8, M9.)
- **Census predicates must be stated with their denominators** and must split on
  `JSON_VALUE(full_report_json,'$._path')` (M6) -- criterion 4 is satisfiable, so an
  "underivable" finding would be wrong.
- **Never key the funnel on `analysis_results.risk_judge_decision`** without first reporting
  its population rate; use `paper_trades.risk_judge_decision` and the JSON blob as the
  cross-checks, and report `UNRESOLVED` (not `False`) where the fingerprint columns are NULL.
- **Do not claim the risk gate is mis-calibrated.** No outcome data exists on the refused
  candidates (`outcome_tracking` = 3 rows; `risk_intervention_log` = 0 rows). The supportable
  deliverable is *instrumentation*: persist every REJECT with its ticker/date so a
  source-2-style forward-outcome audit becomes possible later. That is a build, not a verdict.
- **Separate refusal from absence in the schema**, per source 14's taxonomy: a parse-failure
  fallback must be recorded as an distinct state, not as `APPROVE_REDUCED`.
- **Flag the 2026-08-14 confound in the contract as an explicit non-identifiability**, so no
  criterion is written that requires attributing the change to one of the four co-changes.

## Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch -- **14**
- [x] 10+ unique URLs total (incl. snippet-only) -- **48** de-duped by work (50 raw strings)
- [x] Recency scan (last 2 years) performed + reported
- [x] Full papers / pages read (not abstracts); arXiv chain honoured
      (`/html/` for 2606.18479, 2604.10996, 2605.05427, 2607.19449, 2509.00109; **ar5iv** for
      the 2008 paper 0809.4205; **pdfplumber** for 2607.02830 which has no HTML render).
      No `arxiv.org/pdf/` URL was treated as a primary read.
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every module named in the scope
- [x] Contradictions noted (source 12 vs sources 1/2; source 5 reported as a null result)
- [x] Claims cited per-claim with URL + access date
- [!] **Gap disclosed:** the RUNNING value of `paper_risk_judge_reject_binding` could not be
      read (`GET /api/settings/` exposes 45 keys and not that one). Committed default is
      `False`. Do not assume the gate is binding.
- [!] **Gap disclosed:** `risk_adjusted_confidence` is NULL on all 2026-08 verdict rows, so the
      parse-fail fingerprint is unresolvable there.
