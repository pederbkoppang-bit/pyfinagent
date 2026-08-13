# Research Brief — step 86.59 (v2 — re-run of failed gate `wf_f1f11c5c-6da`)

**Tier:** complex (caller-specified; not self-selected)
**Audit-class:** NO (coverage reported for information only; `coverage.dry` not required)
**Objective:** why pure trailing-return momentum ranking produces near-zero cross-sectional
turnover day-over-day, and what makes a stock screener select a varied candidate set without
weakening it — cross-sectional standardisation, residual/idiosyncratic momentum, short-horizon
reversal, sector-neutralisation, explore-exploit / bandit candidate generation, turnover-aware
and diversity-penalised portfolio construction, and out-of-sample validation of each.

**Brief path:** `handoff/current/research_brief_86.59_v2.md`
**Date:** v1 body 2026-08-12; v2 additions 2026-08-13

---

## ENVELOPE (born inert — flipped to COMPLETE only as the final act)

> **v3 pass, 2026-08-13 (attempt 3).** The v2 body below was written by a run that DROPPED before
> its final act, so this envelope was never updated past its born-inert seed: it still said
> `external_sources_read_in_full: 6` while the read-in-full table below carries **10** rows, and
> `snippet_only_sources: 0` / `urls_collected: 0` were never filled in at all. The v3 pass does
> **verification and ownership, not new research**: every row of the read-in-full table is checked
> for substantive content actually present in this file, the counts are re-derived from the file
> on disk, and only then is `brief_status` flipped. **Tier discrepancy, disclosed:** the v1/v2 body
> was produced under a `complex`-tier spawn (header above); this v3 spawn is `moderate`. The
> envelope reports `moderate` (my caller's tier). No new depth was authored at complex scope.
> Counts below are LIVE as of the v3 pass and are updated in place as each check lands.

```json
{
  "brief_status": "COMPLETE",
  "tier": "moderate",
  "external_sources_read_in_full": 10,
  "snippet_only_sources": 45,
  "urls_collected": 55,
  "recency_scan_performed": true,
  "internal_files_inspected": 5,
  "coverage": {
    "audit_class": false,
    "rounds": 5,
    "dry_rounds": 1,
    "K_required": 2,
    "new_findings_last_round": 0,
    "dry": false
  },
  "summary": "Verification pass on the v2 body; no new research. All 10 claimed read-in-full sources carry >=2 distinctive content markers in the prose - none dropped. The two decisive PDFs were re-downloaded and re-extracted with pypdf and every load-bearing quote regex-verified (Hanauer & Windmueller 65pp/121,826ch; Novy-Marx & Velikov 61pp/103,894ch - both byte-identical to v2's report). Both findings now stand as my own measurements: (A) _zscore is defined at screener.py:532 and called ONLY at :607-610 inside _apply_multidim_momentum (:564), so the live composite at :299-305 sums RAW returns - declared weights 0.40/0.35/0.25 are not the effective weights and reweighting before standardising tunes a dead knob; (B) Garleanu-Pedersen - low turnover is CORRECT for a slow predictor, so the defect is the ABSENCE of a fast signal, bounded by Novy-Marx & Velikov's per-side ~50% line. Residual momentum is not high-turnover once the convention mismatch is normalised (H&W sums both legs; N-M&V is per-side). Counts re-derived on disk: 10 / 55 / 5, correcting the seed's 6 and the stale '30 URLs' that was v1's over-claim. Three stale line anchors corrected in DISAGREEMENT 3.",
  "brief_path": "handoff/current/research_brief_86.59_v2.md",
  "gate_passed": true
}
```

**Gate logic, evaluated explicitly:** `external_sources_read_in_full` 10 ≥ 5 ✓ AND
`recency_scan_performed` true ✓ AND every hard-blocker checklist item satisfied ✓ AND
`coverage.audit_class` false (so `coverage.dry` is informational and does not gate) ✓ →
**`gate_passed: true`**. `dry_rounds: 1` records that the v3 round added **zero** new read-in-full
sources, which is expected for a verification pass and is reported rather than dressed up as
coverage.

---

## WHY THIS IS v2 — the v1 gate failure was MECHANICAL, not substantive

Run `wf_f1f11c5c-6da` (2026-08-12) self-reported `urls_collected: 30` and `gate_passed: true`.
The calling script recomputed **FALSE**: only **13 distinct URLs** were literally present in the
28,878-byte brief on disk. The cause is visible in v1's own snippet table — 18 of its 24
"snippet-only" rows are written as **bare arXiv IDs** (`arXiv:2601.04062`), not as URLs. A regex
over the file cannot count an identifier that is not a URL. **The research was real; the
accounting was not.** v2 fixes that by writing every evaluated source as a full absolute URL, and
by re-deriving `urls_collected` from what is actually on this page rather than from memory.

v2 has three jobs: **(1)** close the declared residual-momentum gap (v1 Evidence-gaps §1);
**(2)** record every URL as a URL; **(3)** re-verify — not re-derive — v1's two load-bearing
findings and preserve its slate-composition-over-score-mutation reasoning.

---

## JOB 1 — RESIDUAL / IDIOSYNCRATIC MOMENTUM: v1's declared gap is now CLOSED

v1 Evidence-gaps §1 read: *"Residual / idiosyncratic momentum: NO source read in full."* **That
gap is closed.** Three sources read in full in v2 (#7, #8, #9 in the read-in-full table). The
canonical Blitz-Huij-Martens 2011 paper itself remains paywalled (SSRN abstract only — recorded
in the snippet table with its failure mode), but its construction and its results are reported
verbatim by two independent sources that I did read in full, and one of those (Hanauer &
Windmüller) **re-implements it on 88 years of U.S. data and publishes the turnover number that
decides this step.**

### (i) The exact formula and the factor model used to residualise

From Hanauer & Windmüller (2019), §2.3, extracted verbatim from the PDF text
(`http://wp.lancs.ac.uk/mhf2019/files/2019/09/MHF-2019-076-Matthias-Hanauer.pdf`):

> "Instead of using the individual stocks' raw returns from t−12 to t−2, we orthogonalize them
> with respect to a Fama-French three-factor model. Thereby, stock returns are adjusted for
> their risk factor exposure. We follow Gutierrez and Prinsky (2007), Blitz et al. (2011), and
> Blitz et al. (2018) and regress **the past 36 months' returns** of all valid stocks within the
> investment universe on country-specific factors of the Fama-French three-factor model."

Equation (8) — the residualising regression, estimated per stock on a rolling 36-month window:

```
R_i,t − R_f,t = α_i + β_RMRF,i·RMRF_t + β_SMB,i·SMB_t + β_HML,i·HML_t + ε_i,t
```

Equation (9) — the signal is the **12-2 month cumulative residual divided by the standard
deviation of those same residuals** (i.e. an information-ratio / t-stat form, NOT a raw sum):

```
ε̂^(12−1)_i,t  =  ( Σ_{j=2..12} ε̂_i,t−j )  /  sqrt( Σ_{j=2..12} (ε̂_i,t−j − ε̄_i)² )
```

> "As in Gutierrez and Prinsky (2007), Blitz et al. (2011), and Blitz et al. (2018), we calculate
> the cumulative idiosyncratic return for each stock by scaling the 12-2 month idiosyncratic
> returns with their volatility."

Independently corroborated by QuantPedia (`https://quantpedia.com/strategies/residual-momentum-factor/`),
which describes the same construction: rank on *"past 12-month residual returns, excluding the
most recent month, standardized by the standard deviation of the residual returns over the same
period"*, residuals from *"the Fama and French three factors as independent variables"*
calculated *"over the past 36 months"*, **rebalanced monthly**.

**Naming note (removes a real ambiguity):** Hanauer & Windmüller footnote 6 states
*"Gutierrez and Prinsky (2007) and Blitz, Huij, and Martens (2011) use the terms abnormal return
momentum and residual momentum, respectively, but **the definitions are identical**."* So
"residual momentum" = "idiosyncratic momentum" = "abnormal-return momentum" (iMOM). Do not treat
them as three candidate signals.

**A cheaper variant is explicitly endorsed by the literature — this matters for us.** Footnote 7:

> "Chaves (2016) in this regard shows that also a simplified version of idiosyncratic momentum
> that is based on **one-factor (market) unscaled residuals** works. Blitz et al. (2018) confirm
> that **most of the performance improvement comes from orthogonalizing returns with the market
> factor** and that the inclusion of additional Fama-French factors leads to small further
> improvements as more of the stock specific momentum is isolated."

### (ii) Reported turnover — THE NUMBER THIS STEP TURNS ON

Hanauer & Windmüller Table 6 (verbatim from the extracted text), *"Average long-short portfolio
turnover (monthly, in %)"*, where §3.3 defines the measure as *"the average (over time)
**one-way** portfolio turnover of the **long leg plus the short leg**"*:

| Strategy | US turnover %/mo (1930-2017) | Global %/mo (1991-2017) |
|---|---|---|
| MOM (standard 12-2 momentum) | **53.79** | **50.32** |
| cvol6M (constant-vol-scaled) | 80.63 | 70.69 |
| dyn (dynamic OOS-scaled) | 82.22 | 81.06 |
| **iMOM (idiosyncratic/residual)** | **65.32** | **62.59** |

> "Idiosyncratic momentum also incurs **higher** portfolio turnover than standard momentum."

#### THE CONVENTION TRAP — verified at source, and it reverses the naive conclusion

**Do not compare 65.32% directly against Novy-Marx & Velikov's ~50% line. The two papers measure
different things, and the naive comparison gives the wrong answer.** I re-fetched w20721 and read
its own definition rather than trusting a paraphrase (61 pp / 103,894 chars, `curl` + `pypdf`).

- **N-M&V's "one-sided monthly turnover" is PER SIDE — an average, not a sum.** Table 4 caption,
  verbatim: *"average turnover (**average over the long and short side**)"*; body: *"monthly
  average turnover of **each side** of the strategy"*; and the explanatory gloss: *"if the long
  side of a strategy turns over 20% per month, the realized long/short spread will be at least
  20 bps per month lower than the gross spread."*
- **H&W's figure is a SUM.** §3.3, verbatim: *"the average (over time) one-way portfolio turnover
  of the long leg **plus** the short leg."*

Normalising H&W onto N-M&V's convention (÷2, assuming symmetric legs):

| Strategy | H&W summed %/mo (US) | → N-M&V per-side equivalent | vs N-M&V 50% line |
|---|---|---|---|
| MOM | 53.79 | **≈ 26.9%** | well under |
| **iMOM** | 65.32 | **≈ 32.7%** | **under** |
| cvol6M | 80.63 | ≈ 40.3% | under |
| dyn | 82.22 | ≈ 41.1% | under |

**This cross-validates against N-M&V's own independent measurement.** N-M&V place momentum in
their *mid-turnover* bin and state, verbatim: *"These are all rebalanced monthly, and have average
turnover on each of the long and the short side of **between 14% and 35% per month**."* The
converted H&W momentum figure (26.9%) lands inside that band, and the converted iMOM figure
(32.7%) lands inside it too, near its top. Two independent papers, two datasets, one consistent
picture once the convention is normalised.

**Consequence for this step: residual momentum is NOT a high-turnover strategy by the standard
this project has adopted.** It sits in the same mid-turnover regime as the momentum leg already
in production, roughly +6pp per side. The earlier worry that it might breach the ~50% line was an
artefact of comparing a summed number to a per-side threshold.

*Caveat, stated plainly:* the ÷2 step assumes the two legs turn over symmetrically — H&W publish
only the sum, so this is a normalisation, not a measurement. It is corroborated by N-M&V's
independent 14-35% band rather than resting on the assumption alone. **pyfinagent is long-only, so
the long leg is the only one we would trade; Main should still measure our own realised per-side
turnover in replay before treating ~33% as our number.**

*Worth flagging as a literature-hygiene note:* H&W themselves write *"for all strategies, the legs
on average generate a turnover of more than 50% per month"* while citing *"Novy-Marx and Velikov
(2016) state that most published factors with above a 50% turnover per month are not profitable
after trading costs"* and, in footnote 9, *"Novy-Marx and Velikov (2016) find momentum to deliver
significant after-transaction cost returns."* Those three sentences are only mutually consistent
once you notice the convention mismatch — a published paper made exactly the comparison error
this section warns against. Anyone reading H&W alone would have inherited it.

### (iii) Does the edge survive transaction costs? — YES, by the widest margin in the study

H&W use Grundy & Martin's **break-even round-trip cost**: the cost level that would render the
strategy's return statistically insignificant.

| | MOM | cvol6M | dyn | **iMOM** |
|---|---|---|---|---|
| US, break-even round-trip @5% sig. | 0.62% | 1.02% | 1.03% | **0.77%** |
| US, @1% sig. | 0.46% | 0.92% | 0.93% | **0.70%** |
| Global, @5% sig. | 0.35% | 0.76% | 0.52% | **0.87%** |
| Global, @1% sig. | 0.14% | 0.61% | 0.39% | **0.79%** |

> "Panel B shows that for the global sample, **idiosyncratic momentum clearly gives the highest
> bounds for all types of round-trip costs**."

So iMOM buys ~24% more cost headroom than standard momentum in the U.S. (0.77 vs 0.62) and
~2.5× more globally (0.87 vs 0.35), *while* carrying only ~6pp more turnover per side. That is a
strictly better cost-per-unit-of-turnover trade than either volatility-scaling variant. H&W's own
caveat, verbatim: *"Our approach does not explicitly test the after-trading cost performance ...
Rather, this break-even cost study reveals how profitable each strategy remains when assuming a
certain level of transaction costs."* It is an **upper-bound argument, not a net-return
measurement** — and I flag it as such rather than as evidence of realised net profitability.

**The strongest cost evidence is not H&W's break-even bound but N-M&V's direct net measurement,
which explicitly covers this family.** For their mid-turnover bin (where momentum sits), w20721
states verbatim: *"only the net issuance, earnings momentum strategy based on cumulative abnormal
three day return around the prior earnings announcement, and **momentum and its derivative
anomalies**, achieve net excess returns that are statistically significant."* Residual momentum is
precisely a derivative of momentum, constructed on the same 12-2 window over the same universe at
the same monthly cadence, with per-side turnover ~6pp higher. **N-M&V measured net spreads, not
break-even bounds — that is the difference between "survives costs" and "would survive costs if
they were below X".** Two independent methodologies therefore agree.

Corroborated a third time by Seppä-Lassila (2020), who re-derives the same measures on U.S. daily
data 1926-2020 and concludes verbatim: *"I confirm the findings of Barroso and Santa-Clara (2015)
and Hanauer and Windmüller (2020) that the profitability of volatility-scaled momentum and
residual momentum **does not come from overly high turnover** that would in fact make the
strategies unprofitable by generating too much trading. Even though the turnovers of the enhanced
strategies are generally somewhat higher than that of the total return momentum, the higher
average returns increase the level of tolerable transaction costs to an even higher level compared
to the traditional momentum."*

**A liquidity result that cuts directly against our biggest cost risk.** v1's Pitfall 5 warned
that cost concentrates in small-cap / high-idiosyncratic-vol names and that a diversity rule
reaching deeper into sectors reaches into exactly those names. Residual momentum goes the *other*
way — verbatim from Seppä-Lassila: *"Blitz et al. (2020) show that **both legs of the residual
momentum are more concentrated on large-cap stocks with lower idiosyncratic volatility**, which
would suggest better liquidity and lower transaction costs than in the traditional momentum."*
That is the same structural fix N-M&V and QuantPedia identify for short-term reversal
(large-cap restriction), except here it falls out of the construction instead of being bolted on.

Two further quality results, verbatim: Blitz et al. (2011) *"document that idiosyncratic momentum
exhibits **only half of the volatility** of standard momentum without any significant decrease in
returns"*; and iMOM *"experiences **no long-term reversals**"* (Gutierrez & Prinsky 2007).

H&W's spanning tests (Table 7) give iMOM **α = 0.32, t(α) = 5.67** (U.S.) and **α = 0.54,
t(α) = 8.17** (Global) against a FF+MOM factor set — i.e. it is *not* a repackaging of the
momentum leg we already have. Both clear this project's Harvey-et-al. **t ≥ 3.0** bar with room to
spare, and they do so *conditioning on standard momentum already being in the model*. Table 8:
U.S. iMOM avg. return **0.64%/mo, t = 8.90, annualized Sharpe 0.95, max DD −25.52%** (1930-2017).
For contrast this project's own hard sector-neutralisation replay measured **−0.166** Sharpe.
*(All of these are zero-cost **long-short factor** statistics; a long-only screener captures at
most the long leg, so do not carry the 0.95 Sharpe into a pyfinagent expectation.)*

QuantPedia's standalone
backtest of the factor reports **CAGR 9.18%, Sharpe 0.34, max DD −59.74%, vol 15.27%, 1926-2009**,
~1000-stock universe, 2 deciles, equally weighted, **monthly** rebalance.

### (iii-b) The DISAGREEING view — residual momentum is not unambiguously better

Sources #7-#8-#10 all favour iMOM. Deliberately seeking the qualifying case, CXO Advisory
(`https://www.cxoadvisory.com/momentum-investing/idiosyncratic-pure-or-residual-momentum-as-a-stock-return-predictor/`,
read in full) reviews the same Blitz-Hanauer-Vidojevic evidence and lands materially cooler:

- **Gross return is LOWER, not higher.** Idiosyncratic 1.39%/mo vs conventional 1.54%/mo
  (Dec 1925 - Dec 2015, U.S.). The Sharpe advantage (0.48 vs 0.25 monthly) comes **entirely from
  halved volatility**, not from more return: *"Idiosyncratic momentum portfolios reliably generate
  gross average returns comparable to those of conventional momentum, with half the volatility."*
  **A long-only screener that never harvests the short leg and does not lever to a vol target
  captures the numerator, not the denominator — so it may capture very little of the advantage.
  This is the most important caveat in this brief for our specific use case.**
- **Sub-period instability:** underperformance versus conventional momentum during **1940-2000**,
  weakness **after the early 2000s**, and it *"absorbed significant portion of 2009 crash"*.
  A 90-year headline hides two long stretches where the edge was absent.
- **Frictions the headline ignores:** returns are gross; net is reduced by *"monthly reformation
  and shorting costs"*, shorting is capacity-constrained, and the strategy is characterised as
  *"beyond the reach of most investors"*.
- On turnover it is more equivocal than H&W: *"this turnover is modestly higher than that of the
  conventional momentum factor portfolio (which tends to be high compared to turnovers of other
  factor portfolios)"* — agreeing on direction, warning that momentum's own base is already high.

There is also a **theoretical objection** in the literature (surfaced in the recency scan, not
read in full): that idiosyncratic stock or industry momentum should not persist, as it would be
inconsistent with the existence of rational arbitrageurs. Empirics across markets nonetheless keep
measuring it. **Recorded as an open theoretical dispute, not as a settled result.**

**Net assessment:** residual momentum is well-evidenced as a *risk-reduction* improvement to
momentum (half the volatility, no long-term reversal, higher break-even costs, large-cap tilt) and
is **not** well-evidenced as a *raw-return* improvement. For pyfinagent that is still valuable —
but the honest framing for the contract is "a better-behaved momentum signal", not "a bigger one".

### (iv) Data requirements versus what pyfinagent already stores

| Requirement | Full FF3 form | Market-only form (Chaves 2016 / Blitz 2018) | pyfinagent today |
|---|---|---|---|
| Per-stock return history | **36 months** rolling | 36 months rolling | Only ~126 trading days (~6 mo) is used by the live composite; longer history is available from the price store but is **not** in the screener's input path |
| Factor series | RMRF **+ SMB + HML**, country-specific | **RMRF only** | No SMB/HML series is stored; a market/benchmark return series **does** exist (benchmark plumbing per `project_multimarket_benchmark_fx_changesite`) |
| Per-stock regression | 3-var OLS per stock per month | 1-var OLS per stock per month | Not implemented anywhere in `screener.py` |
| Rebalance cadence | **Monthly** | Monthly | Loop runs **daily** — see the cadence mismatch flagged in Application A8 below |

**Bottom line for the contract:** the market-only variant is the affordable one — it needs a
36-month return matrix plus one market series, no SMB/HML, and is endorsed by the literature as
capturing *"most of the performance improvement"*. The FF3 variant would require sourcing and
storing two additional factor series per market (US + EU + KR), which is a data-ingestion project,
not a screener change.

---

## METHOD NOTE — v1's WebSearch outage did NOT recur in v2

v1 recorded WebSearch returning *"this session has used its web search budget (200 of 200
WebSearch calls)"* on its first attempt (session-shared budget, exhausted before that researcher
was spawned — auto-memory `reference_websearch_budget_is_session_shared`). **In v2 WebSearch is
available and was used**, which is how the residual-momentum gap was closed: the query
`residual momentum Blitz Huij Martens idiosyncratic momentum turnover` surfaced the Hanauer &
Windmüller working paper that v1's arXiv-only substitute searches could never have reached
(it is not on arXiv). *Lesson worth carrying: v1's gap was caused by a tooling outage, not by an
absence of literature.*

The v1 method note below is retained unchanged for provenance.

## METHOD NOTE (v1, retained) — WebSearch was unavailable (disclosed, not hidden)

`WebSearch` returned **"this session has used its web search budget (200 of 200 WebSearch
calls)"** on my FIRST search attempt — the budget is session-shared and was exhausted before
this researcher was spawned. `WebFetch` is unaffected.

Search substitutes actually used (each a real search pass, not a snippet harvest):
- **arXiv search UI via WebFetch** — `https://arxiv.org/search/?searchtype=all&query=...`
  returns real result lists with IDs, titles, dates. Four queries run; two produced results,
  two returned arXiv's "produced no results" page (reported below, not hidden).
- **Semantic Scholar Graph API via curl** — rate-limited during this run (1 of 5 queries).
- **arXiv Atom API via curl** — returned empty in this environment; abandoned.

**Three-variant discipline** (required by `.claude/rules/research-gate.md`) is satisfied and
visible in the tables: **year-less canonical** prior art (Novy-Marx & Velikov; Gârleanu &
Pedersen; Daniel & Moskowitz; short-term reversal), **last-2-year** work (arXiv 2024-2025), and
**current-year 2026** frontier (arXiv 2601.*, 2602.*, 2603.*, 2606.*, 2607.*).

PDF handling follows `.claude/rules/research-gate.md`: `WebFetch` was **never** called on an
`arxiv.org/pdf/` URL. NBER PDFs were extracted locally with `pypdf` (project venv) because
WebFetch PDF summarisation is a known quote-fabrication risk here (auto-memory
`reference_webfetch_pdf_summaries_fabricate_quotes`). **Every NBER quote below was
regex-verified against the extracted text**; none is a model summary.

---

## Read in full (10 — counts toward the gate)

Rows 1-6 were read in the v1 session (2026-08-12); rows 7-10 in the v2 session (2026-08-13).
**Row 1 was independently RE-FETCHED and re-read in v2** to verify its turnover convention at
source — 61 pp / 103,894 chars, byte-count identical to v1's report, which corroborates that v1's
read was genuine. Provenance is labelled per row rather than blurred.

| # | URL | Accessed | Kind | Fetched how | Verified core finding |
|---|-----|----------|------|-------------|------------------------|
| 1 | https://www.nber.org/system/files/working_papers/w20721/w20721.pdf | v1 2026-08-12; **re-read v2 2026-08-13** | peer-reviewed WP | curl + pypdf, 61 pp / 103,894 ch | Novy-Marx & Velikov, *A Taxonomy of Anomalies and Their Trading Costs* — the 50%-turnover survival line; **v2 verified the line is PER-SIDE, and that "momentum and its derivative anomalies" achieve significant NET spreads** |
| 2 | https://www.nber.org/system/files/working_papers/w15205/w15205.pdf | 2026-08-12 | peer-reviewed WP | curl + pypdf, 47 pp / 69,840 ch | Gârleanu & Pedersen, *Dynamic Trading with Predictable Returns and Transaction Costs* — aim portfolio, partial adjustment |
| 3 | https://www.nber.org/system/files/working_papers/w20439/w20439.pdf | 2026-08-12 | peer-reviewed WP | curl + pypdf, 52 pp / 118,626 ch | Daniel & Moskowitz, *Momentum Crashes* — vol-scaled dynamic momentum ~doubles Sharpe |
| 4 | https://arxiv.org/html/2408.09168v1 | 2026-08-12 | paper (CS/IR) | WebFetch, full text | Lichtenberg et al., *Multinomial Blending* — slate composition without touching scores; MMR baseline drifts |
| 5 | https://arxiv.org/html/2601.08717v1 | 2026-08-12 | paper (q-fin) | WebFetch, full text | Garcia & Messud — HHI diversity penalty vs. bounded-degradation form; synthetic energy assets |
| 6 | https://quantpedia.com/strategies/short-term-reversal-in-stocks/ | 2026-08-12 | industry | WebFetch, full page | Short-term reversal: 1-week rank, weekly rebalance, large-cap restriction is the cost fix |
| 7 | https://quantpedia.com/strategies/residual-momentum-factor/ | 2026-08-13 | industry | WebFetch, full page | **[JOB 1]** Residual momentum construction: FF3 residuals over 36 mo, 12-2 window, standardised by residual SD, **monthly** rebalance; CAGR 9.18%, Sharpe 0.34, maxDD −59.74%, 1926-2009 |
| 8 | http://wp.lancs.ac.uk/mhf2019/files/2019/09/MHF-2019-076-Matthias-Hanauer.pdf | 2026-08-13 | working paper (peer-reviewed venue submission) | curl + pypdf, 65 pp / 121,826 ch | **[JOB 1 — the decisive source]** Hanauer & Windmüller, *Enhanced Momentum Strategies*. Eq. 8/9 formula; **iMOM turnover 65.32% US / 62.59% Global (long+short SUMMED)**; break-even round-trip 0.77%/0.87% — highest globally; spanning α t = 5.67 / 8.17 |
| 9 | https://www.cxoadvisory.com/momentum-investing/idiosyncratic-pure-or-residual-momentum-as-a-stock-return-predictor/ | 2026-08-13 | industry review | WebFetch, full page | **[JOB 1 — QUALIFYING/ADVERSARIAL]** Gross return LOWER (1.39 vs 1.54%/mo); Sharpe edge is entirely halved vol; underperformed 1940-2000 and weak post-2000s |
| 10 | https://aaltodoc.aalto.fi/bitstreams/5735c930-793b-4ec7-92db-f29e7f122f27/download | 2026-08-13 | MSc thesis (**tier-5, student work — weight accordingly**) | curl + pypdf, 43 pp / 81,620 ch | **[JOB 1]** Seppä-Lassila, *Risk-managed residual momentum*. Independent replication 1926-2020: residual momentum's profit *"does not come from overly high turnover"*; both legs concentrate in **large caps with lower idiosyncratic vol** |

## Identified but snippet-only (26 — context; does NOT count toward the gate)

**This table is the direct fix for the v1 gate failure.** v1 wrote 18 of these rows as bare arXiv
identifiers (`arXiv:2601.04062`), which a URL regex cannot count — hence 13 counted against 30
claimed. Every row below now carries a **full absolute URL**, plus the query that surfaced it.

| URL | Kind | Query that surfaced it | Why not read in full |
|---|---|---|---|
### v1 rows (URLs preserved, arXiv IDs converted to absolute URLs)

| URL | Kind | Query that surfaced it | Why not read in full |
|---|---|---|---|
| https://www.nber.org/system/files/working_papers/w18098/w18098.pdf | WP | v1 NBER direct | **Fetched and REJECTED** — expected Asness-Moskowitz-Pedersen; extracted text is *"Market Design in Cap and Trade Programs"* (Holland & Moore). Wrong paper; excluded rather than mis-cited |
| https://arxiv.org/abs/2408.09168 | abs page | v1 arXiv search | Superseded by the `/html/` full read (#4) |
| https://arxiv.org/abs/2601.08717 | abs page | v1 arXiv search | Superseded by the `/html/` full read (#5) |
| https://www.aqr.com/Insights/Research/Journal-Article/Craftsmanship-Alpha-An-Alternative-to-Factor-Investing | industry | v1 year-less canonical | **Dead** — resolves to AQR 404 / fraud-warning page |
| https://alphaarchitect.com/the-limits-of-anomalies-trading-costs/ | blog | v1 year-less canonical | HTTP **403**, bot-blocked |
| https://en.wikipedia.org/wiki/Momentum_(finance) | encyclopaedia | v1 year-less canonical | Tier-5, deliberately unused |
| https://arxiv.org/abs/2601.04062 | paper | turnover/rebalancing 2026 | "transaction costs, turnover control, and regularization" in the training objective |
| https://arxiv.org/abs/2607.21170 | paper | turnover/rebalancing 2026 | Retention mechanism "reducing portfolio turnover" |
| https://arxiv.org/abs/2603.16904 | paper | turnover/rebalancing 2026 | Claims 44.5% transaction-cost reduction via fewer rebalances |
| https://arxiv.org/abs/2309.10152 | paper | turnover/rebalancing year-less | Turnover-sparsity (l0) constraint |
| https://arxiv.org/abs/2303.12751 | paper | turnover/rebalancing year-less | l1/l2 regularisation to "reduce an excessive turnover" |
| https://arxiv.org/abs/2206.14760 | paper | turnover/rebalancing year-less | Exact penalty function on the turnover constraint |
| https://arxiv.org/abs/2601.06507 | paper | turnover/rebalancing 2026 | Off-topic (ESG) |
| https://arxiv.org/abs/2410.04217 | paper | bandit/portfolio 2024-25 | ADTS/CADTS, "Sharpe ratio 20% higher"; strongest bandit candidate |
| https://arxiv.org/abs/2406.06552 | paper | bandit/portfolio 2024-25 | UCB for risk-adjusted arm choice |
| https://arxiv.org/abs/2205.05843 | survey | bandit/portfolio year-less | Best entry point if bandits are pursued |
| https://arxiv.org/abs/2606.23933 | paper | bandit/portfolio 2026 | 2026 frontier; portfolio-selection benchmark |
| https://arxiv.org/abs/2602.15972 | paper | bandit/portfolio 2026 | 2026; portfolio management with risky assets |
| https://arxiv.org/abs/2512.09850 | paper | bandit/portfolio 2025 | Statistical validity under weak arm separability |
| https://arxiv.org/abs/2211.14768 | paper | bandit/portfolio year-less | Closest bandit formulation to a fixed daily analyse-budget |
| https://arxiv.org/abs/2206.12463 | paper | bandit/portfolio year-less | Mean-variance Thompson Sampling |
| https://arxiv.org/abs/1709.04415 | paper | bandit/portfolio year-less | Coherent risk measures in MAB |
| https://arxiv.org/abs/1911.05309 | paper | bandit/portfolio year-less | Year-less canonical bandit-portfolio hit |
| https://arxiv.org/abs/2312.03294 | paper | bandit/portfolio year-less | MAB framework for strategy blending/switching |

### v2 rows — residual-momentum sweep (new, 2026-08-13)

Queries run, three-variant discipline (see §Recency scan for the year-scoped pass):
**(a) year-less canonical** `residual momentum Blitz Huij Martens idiosyncratic momentum turnover`;
**(b) publisher-scoped** `Robeco "residual momentum" OR "idiosyncratic momentum" Blitz research paper pdf`;
**(c) current-year** `idiosyncratic momentum residual momentum 2026 evidence`.

| URL | Kind | Query | Why not read in full |
|---|---|---|---|
| https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2319861 | paywalled paper | (a),(b) | **The canonical source** — Blitz, Huij & Martens, *Residual Momentum*, J. Emp. Fin. 18 (2011). SSRN abstract page only; no public full text. **This is v1's original blocker and it remains blocked** — closed instead via #7/#8/#10, which reproduce its construction and results |
| https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2947044 | paywalled paper | (b),(c) | Blitz, Hanauer & Vidojevic, *The Idiosyncratic Momentum Anomaly* (2017/2020). Abstract only; its findings reach the brief via #8 and #9 |
| https://www.sciencedirect.com/science/article/abs/pii/S1059056020300927 | paywalled journal | (b),(c) | Same paper, Int. Rev. Econ. & Fin. 69 (2020). Abstract-only behind Elsevier paywall |
| https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2929306 | paywalled paper | (b) | Huij & Lansdorp, *Residual Momentum and Reversal Strategies Revisited* — directly on the momentum/reversal interaction (v1's F8). Abstract only |
| https://assets.super.so/e46b77e7-ee08-445e-b43f-4ffd88ae0a0e/files/017c102d-5882-4e93-9f4c-4ef8500ef7d3.pdf | PDF mirror | (a) | **Attempted, FAILED** — `curl` returned a 243-byte S3 `<Error><Code>AccessDenied</Code>` XML body, not a PDF |
| https://alphaarchitect.com/residual-momentum-a-better-momentum/ | practitioner blog | (a) | **Attempted, FAILED** — HTTP **403** bot-block (same block v1 hit on a different Alpha Architect URL) |
| https://alphaarchitect.com/swedroe-spotlight-enhancing-momentum-strategies-via-idiosyncratic-momentum/ | practitioner blog | (a),(c) | **Attempted, FAILED** — HTTP **403** bot-block |
| https://www.researchgate.net/publication/332907764_Residual_Momentum | repost | (a) | ResearchGate "Request PDF" stub — no full text without an account |
| https://www.researchgate.net/publication/342147740_The_idiosyncratic_momentum_anomaly | repost | (a),(b),(c) | ResearchGate "Request PDF" stub |
| https://www.semanticscholar.org/paper/Residual-Momentum-Blitz-Huij/e75488daa31c7d76a3660a15b8f36df6cc06d434 | index | (a),(b) | Metadata/index page; v1 recorded Semantic Scholar's API as rate-limited |
| https://www.semanticscholar.org/paper/60a28ce523a197da7bd3ce587042d223faa1023d | index | (c) | *Idiosyncratic Momentum: U.S. and International Evidence* — index stub |
| https://onlinelibrary.wiley.com/doi/abs/10.1111/eufm.12247 | paywalled journal | (c) | Lin (2020), *Idiosyncratic momentum and the cross-section of stock returns: further evidence*, Eur. Fin. Mgmt — abstract only |
| http://www.efmaefm.org/0EFMAMEETINGS/EFMA%20ANNUAL%20MEETINGS/2022-Rome/papers/EFMA%202022_stage-3032_question-Full%20Paper_id-448.pdf | conference WP | (a) | Graef, *Firm-specific versus systematic momentum* — directly relevant; **not fetched: the 3-source residual-momentum floor was already met and read-in-full budget was spent on #8/#10** |
| https://www.diva-portal.org/smash/get/diva2:1672661/FULLTEXT01.pdf | MSc thesis | (b) | *Residual Momentum and Volatility-Managed Portfolios* — duplicate in kind and tier with #10, which was already read in full |
| http://wp.lancs.ac.uk/fofi2026/files/2026/03/FoFI-2026-035-Malek-Alkshaik.pdf | 2026 WP | (c) | *An Auto-Residual Factor Model* (v1 Jun 2025, FoFI 2026) — **the only genuinely 2025/2026 residual-factor hit**; see Recency scan |
| https://link.springer.com/article/10.1007/s11408-022-00417-8 | journal | (c) | *Momentum: what do we know 30 years after Jegadeesh and Titman?* — survey; paywalled |
| https://arxiv.org/abs/1910.13115 | paper | (c) | *Horse race of weekly idiosyncratic momentum strategies ... Chinese stock market* — **weekly** iMOM, adjacent to v1's F8 reversal branch; non-US market |
| https://arxiv.org/abs/2408.07497 | paper | (c) | *Forecasting stock return distributions with quantile neural networks* — off-topic for this step |
| https://factorinvestingtutorial.wordpress.com/9-residual-momentum-david-blitz/ | tutorial site | (b) | Tier-5 community content; superseded by #7/#8 |
| https://robeco.com/data | vendor data | (b) | Robeco's public factor-data page — cited by the search result as hosting a downloadable iMOM 2x3 factor. **Data, not literature**; a possible validation input for Main, not a research source |
| https://www.sciencedirect.com/author/23017878700/david-c-blitz | author index | (b) | Author listing page, no content |

---

## Recency scan (2024-2026) — PERFORMED

Explicit passes run for the 2024-2026 window (arXiv search UI, three query families:
turnover/rebalancing, bandit/portfolio, diversity/selection).

**Result: 3 findings that COMPLEMENT — and 0 that supersede — the canonical sources.**

1. **Diversity penalties have moved from ad-hoc weights to *bounded-degradation* programs.**
   arXiv:2601.08717 (Jan 2026) formulates diversification as *maximise diversity subject to a
   bounded ROI/CVaR loss* (tolerances Δp, Δr) rather than a free-floating penalty weight. This
   is directly relevant and is a **better template than our current `w`** (see Application §3).
2. **Turnover control is being pushed into the training objective**, not applied post-hoc —
   arXiv:2601.04062 (Jan 2026) embeds "transaction costs, turnover control, and regularization"
   in a decision-focused learning objective; arXiv:2607.21170 (Jul 2026) uses an explicit
   retention mechanism to "preserve high-quality assets across consecutive rebalancing windows".
   Note the direction of travel: recent work is trying to *reduce* turnover, not create it.
3. **Bandit-for-portfolio is an active but non-stationarity-limited literature.** 23 hits, with
   2026 entries (2606.23933, 2602.15972) concentrating on *non-stationary* reward drift — the
   exact regime a stock screener lives in. arXiv:2410.04217 (Oct 2024) reports Sharpe ~20%
   above classical approaches. **Read at snippet level only — lower confidence, flagged.**

Nothing found in the window overturns Novy-Marx & Velikov's cost result, Gârleanu-Pedersen's
partial-adjustment result, or Daniel-Moskowitz's crash/vol-scaling result. They remain load-bearing.

---

## Key findings (per-claim citations)

**F1 — Turnover is not free, and ~50% one-sided monthly turnover is the empirical survival
line.** Verbatim from the extracted text: *"Most of the anomalies that we consider with
one-sided monthly turnover lower than 50% continue to generate statistically significant net
spreads, at least when designed to mitigate transaction costs. Few of the strategies with
higher turnover do."* And: *"while transaction costs dramatically reduce the profitability of
many anomalies, especially those with high turnover, designing strategies to minimize
transaction costs significantly reduces these costs."* (Novy-Marx & Velikov 2014, w20721.)
This is the primary constraint on *any* proposal to make the screener churn more.

**F2 — The single most effective cost mitigation is a banding/abstention rule, not a cleverer
score.** *"...not actively trade into, is the single most effective simple cost mitigation
strategy"* (w20721). The paper also finds *"trading costs are persistent and significantly
positively associated with idiosyncratic volatility"*, and that costs *"for equal-weighted
strategies are generally two to three times as high"* than value-weighted. (Novy-Marx & Velikov 2014.)

**F3 — Optimal response to a moving signal is PARTIAL adjustment toward an "aim", and slow
signals correctly produce low turnover.** *"The optimal strategy is characterized by two
principles: 1) aim in front of the target and 2) trade partially towards the current aim."* The
aim is *"a weighted average of the current Markowitz portfolio (the moving target) and the
expected Markowitz portfolios on all future dates"*, and crucially *"predictors with slower mean
reversion (alpha decay) get more weight in the aim portfolio."* (Gârleanu & Pedersen, w15205.)
**This reframes the whole step:** a 1/3/6-month composite is a *slow* predictor, so low turnover
is the theoretically CORRECT behaviour for it. The defect is not "turnover too low" — it is
"the signal menu contains nothing fast".

**F4 — Volatility-scaling momentum is the best-documented Sharpe improvement, and it is a
SIZING change, not a selection change.** *"An implementable dynamic momentum strategy based on
forecasts of momentum's mean and variance approximately doubles the alpha and Sharpe Ratio of a
static momentum strategy, and is not explained by other factors."* Momentum's payoff is
option-like with asymmetric bear-market betas (*"most of the up- versus down-beta asymmetry in
bear markets is driven by the past losers"*). (Daniel & Moskowitz, w20439.) Our vol leg is a
step function (`vol > 0.6 → ×0.85`), a crude proxy for this.

**F5 — You can add diversity WITHOUT touching the score: change slate composition, not the
ranking function.** Multinomial blending *"first samples a content type according to c∼M(p) and
then selects the highest-scoring remaining candidate from that content type"*, preserving the
personalised order *within* each type. Its decisive property: *"The average exposure guarantees
are independent of the underlying scoring function h and therefore remain stable even after
model re-training or non-stationary user behavior."* (Lichtenberg et al. 2024, arXiv:2408.09168.)
This is the cleanest available answer to "varied without weakening".

**F6 — Score-penalty diversity (MMR-style) needs constant retuning and drifts.** The same paper
rejected MMR operationally because *"various market places required different MMR penalty
parameters and optimal diversification rates would change over time"* — MMR won on one metric
(+13.57% vs +18.82% podcast time; +2.76% vs +2.23% engagement) yet MB *"was globally launched due
to its operational advantages"*. (arXiv:2408.09168.) This is a direct warning about
`paper_soft_sector_diversity_w` as a hand-set constant.

**F7 — Prefer "maximise diversity subject to bounded performance loss" over a free penalty
weight.** Garcia & Messud give both forms; the second *"maximizes diversification while
controlling expected profit and risk degradation"* via explicit tolerances, letting the operator
state the acceptable loss (e.g. 10%) instead of guessing a weight. **Caveat, verbatim: evaluated
*"using synthetic data (energy assets)"* with 100 scenarios** — no equity evidence.
(arXiv:2601.08717, Jan 2026.)

**F8 — Short-horizon reversal is the only surveyed mechanism that creates genuine day-over-day
variation, and its cost problem has a known fix.** Rank on the past *week*, rebalance weekly,
universe = 100 largest caps: 16.25% p.a. net, Sharpe 1.09, maxDD -52.94% (1990-2009). *"the
impact of transaction costs on reversal profits can largely be attributed to excessively trading
in small cap stocks"* — restricting to large caps *"significantly reduces trading costs"*.
Blitz, van der Grient & Honarvar's enhanced version exists specifically to counter reversal's
*"tendency to go against short-term momentum in industry and factor returns"*. (Quantpedia.)

**F9 — Bandit/explore-exploit selection is real but the worst fit here (low confidence,
snippet-only).** The literature is large (23 hits) and 2026 work centres on non-stationary
drift. But a bandit needs a per-pull reward; this screener deep-analyses
`paper_analyze_top_n = 5` names/day (`settings.py:407`) with a delayed, noisy P&L reward. The
closest formulation is fixed-budget pure exploration (arXiv:2211.14768). **I did not read any
bandit paper in full — treat F9 as a pointer, not evidence.**

---

## Internal code inventory

| File | Lines | Role | Status |
|---|---|---|---|
| `backend/tools/screener.py` | :248-296 | `rank_candidates` signature (24 overlay params) | LIVE |
| `backend/tools/screener.py` | :299-305 | **momentum composite** | **LIVE — the whole ranking** |
| `backend/tools/screener.py` | :306-315 | RSI + vol step penalties | LIVE |
| `backend/tools/screener.py` | :317-405 | 13 `apply_*_to_score` overlays, each `if <signal>:` | all gated |
| `backend/tools/screener.py` | :420-430 | news-only injection at flat `5.0*1.10` | gated |
| `backend/tools/screener.py` | :443-452 | `_apply_multidim_momentum` | dark |
| `backend/tools/screener.py` | :454-484 | sector-neutral percentile re-score | dark |
| `backend/tools/screener.py` | :486-492 | `_apply_52wh_tilt` | dark |
| `backend/tools/screener.py` | :494-499 | `_apply_soft_sector_diversity` | dark |
| `backend/tools/screener.py` | :501-502 | **`sort()` + `[:top_n]`** | **LIVE — structural choke point** |
| `backend/tools/screener.py` | :505-539 | soft-diversity impl `(1-w)^j`, sign-safe | dark |
| `backend/tools/screener.py` | :541-553 | `_zscore` | **defined; unreachable on the live path** |
| `backend/services/autonomous_loop.py` | :169-209 | `_min_k_sector_slice` | dark (k=0) |
| `backend/services/autonomous_loop.py` | :982-1019 | `rank_candidates` call site | LIVE |
| `backend/services/autonomous_loop.py` | :1120-1122 | held-ticker exclusion | LIVE |
| `backend/services/autonomous_loop.py` | :1124-1129 | min-K vs plain top-N branch | LIVE, K=0 |
| `backend/autoresearch/gate.py` | :22-25 | `PromotionGate(min_dsr=0.95, max_pbo=0.20, min_pbo_trials=10)` | LIVE |
| `backend/services/promotion_gate.py` | :53-63 | staging `PBO_CEILING` check | LIVE |
| `backend/config/settings.py` | :407, :468-469, :487-489 | the three dark mitigations + `paper_analyze_top_n=5` | default OFF/0 |

### The composite, verbatim (`backend/tools/screener.py:299-305`)

```python
score = (
    mom_1m * 0.40 +
    mom_3m * 0.35 +
    mom_6m * 0.25
)
```

### ROOT CAUSE — why day-over-day turnover is ~zero (structural, not a data bug)

Every term is a **trailing cumulative return** over ~21 / 63 / 126 trading days. Rolling one day
forward swaps out one observation and swaps in one:

- `mom_1m` changes by O(1/21) of its window; `mom_3m` O(1/63); `mom_6m` O(1/126).
- The composite's day-over-day autocorrelation is therefore very close to 1.
- The **rank** vector is stickier still: an ordering change requires two names' scores to
  *cross*, i.e. a shock exceeding the gap between them.
- `sort()` + `[:top_n]` (`:501-502`) then takes the top slice of a near-static ordering, so
  **set** turnover is even lower than score turnover.

The RSI/vol legs cannot fix this: they are **multiplicative constants on discrete thresholds**
(`rsi>80 → ×0.7`, `rsi<20 → ×0.8`, `vol>0.6 → ×0.85`). They fire rarely and apply the *same*
constant to everyone in a bucket, so they move levels, not ordering, except exactly at a crossing.

**No cross-sectional standardisation exists on the live path.** `_zscore` (`:541-553`) is called
**only** by `_apply_multidim_momentum`, gated on `multidim_momentum_enabled`
(`settings.py:478`, default `False`). So raw returns of three different horizons are summed as if
commensurate. A 6-month return has roughly √6 ≈ 2.4× the dispersion of a 1-month return, so the
6m term's **effective** weight far exceeds its nominal 0.25 — making the composite *more*
persistent than the weights suggest, and meaning **the declared weights 0.40/0.35/0.25 are not
the weights actually in force.**

### Every overlay is default-OFF (measured from `settings.py`)

`pead_signal` :431 · `news_screen` :435 · `meta_scorer` :442 · `analyst_revisions` :450 ·
`sector_neutral_momentum` :468 · `sector_momentum` :471 · `multidim_momentum` :478 ·
`momentum_52wh_tilt` :483 · `paper_soft_sector_diversity` :487 (`w`=0.0 :488) ·
`paper_min_k_sectors_analyzed` :489 (=0) · `options_flow_screen` :506 ·
`insider_signal_screen` :516 · `call_transcript_gpr` :538 · `social_velocity` :544 ·
`defense_signal` :551 · `peer_leadlag` :558 · `ma_preannounce` :565 — **all `False`/0.**

If those defaults hold at runtime, `rank_candidates` computes the composite, skips every
`if <signal>:` block, sorts, truncates — so the **entire live ranking is three trailing returns
plus two step penalties**, and every mechanism that could add cross-sectional variation is dark.

> **LIMITATION (disclosed).** Reading `backend/.env` was **DENIED** by the permission system.
> These are *declared defaults*, not proof of the running process's values. Per this project's
> standing "committed is NOT in force" rule, **Main must re-derive the live values from the
> running backend** before relying on the "all overlays dark" framing.

### The two dark mitigations rest on non-equity citations — VERIFIED BY READING BOTH

- `settings.py:488` cites *"arXiv 2601.08717 shades-never-zeroes"*. The paper is Garcia &
  Messud (13 Jan 2026), HHI diversification evaluated *"using synthetic data (energy assets)"*.
  The *shape* of the claim (penalise concentration in the objective rather than eliminate
  candidates) is genuinely supported; the asset class, data and objective are not ours.
- `autonomous_loop.py:178` cites *"arXiv 2408.09168 multinomial-blend leader-pick"*. The paper
  is an **Amazon Music learning-to-rank** paper about mixing music/podcasts/videos into one
  slate. Legitimate for the *mechanism*; it carries **no return, Sharpe or cost result**.

Cross-domain transfer is a legitimate move — but the contract must not describe these as
validating the *financial* case. **Additionally, a fidelity gap:** MB's headline guarantee
(exposure independent of the scoring function, stable under retraining) derives from
**stochastic sampling** `c∼M(p)`. `_min_k_sector_slice` (`:188-200`) is a **deterministic**
top-k-sectors-by-peak leader pick. It does **not** inherit that guarantee, and it introduces no
exploration whatsoever.

### The one measured equity number already in the tree

`settings.py:487` records that the 2026-06-01 replay measured **-0.166 long-only Sharpe for
hard sector-neutralisation** — this project's own evidence, and it is *negative*. That is why
70.2 chose the soft `(1-w)^j` decay leaving each sector leader untouched. F5/F6 independently
support that ordering: prefer slate-composition changes over score mutation.

---

## JOB 3 — VERIFICATION PASS on v1's load-bearing findings (2026-08-13)

Task was to **verify, not re-derive**, and to **flag disagreements rather than silently rewrite**.
Both findings **survive**. Two disagreements found, both in *anchors*, neither in *substance*.

### Finding (A) — no cross-sectional standardisation on the live path: **CONFIRMED**

Re-read at source today:

- `backend/tools/screener.py:301-303` — the composite is exactly
  `mom_1m*0.40 + mom_3m*0.35 + mom_6m*0.25`, inside `if strategy == "momentum":` at `:299`.
  RSI penalties `:305-308` (`>80 → ×0.7`, `<20 → ×0.8`), vol penalty `:310-311` (`>0.6 → ×0.85`).
- `_zscore` is defined at `screener.py:532` and called at **exactly four sites — `:607`, `:608`,
  `:609`, `:610`** — and I confirmed all four are inside `_apply_multidim_momentum` (defined
  `:564`; the preceding def is `_apply_52wh_tilt` at `:545`). There is **no other call site in the
  file**. `multidim_momentum_enabled` defaults `False` (`settings.py:478`).
- **Therefore: the live composite sums raw multi-horizon returns with no standardisation.**
  Verified, not inherited.

> ### DISAGREEMENT 1 (anchors, flagged not silently fixed)
> v1's internal inventory places `_zscore` at `screener.py:541-553` and
> `_apply_multidim_momentum` at `:443-452`. **Both are wrong against the file as it stands
> today**: the true anchors are `:532` and `:564` (call sites `:607-610`). The caller's prompt
> carried the correct ones. v1's *conclusion* is unaffected — `_zscore` is still reachable only
> through the dark multidim path — but **Main must use `:532` / `:564` / `:607-610` in the
> contract, not v1's numbers.** v1's other anchors in that table are likewise unverified and
> should be re-derived before being copied. This is the standing "re-derive the line number
> before citing it again" trap, realised.

**On the "~2.4×" dispersion claim — supported, but label it as theory.** √6 ≈ 2.449 is the
dispersion ratio of a 6-month to a 1-month return **only under an IID/random-walk assumption**.
Real equity returns are not IID at these horizons, and momentum's own existence is evidence of
autocorrelation. **I did not measure the realised cross-sectional SD of `mom_1m` vs `mom_6m` on
our universe, and no source I read reports it for our data.** The *direction* is robust (longer
windows disperse more, so 0.25 on `mom_6m` buys more ranking influence than 0.40 on `mom_1m`);
the *magnitude* is an assumption. **Recommended: Main measures the three realised SDs from one
cycle's `screen_data` before the contract quotes 2.4×.** The load-bearing conclusion — *declared
weights are not effective weights, so reweighting before standardising tunes a dead knob* —
does not depend on the exact multiple.

### Finding (B) — Gârleanu-Pedersen / the missing-fast-signal framing: **CONFIRMED AND STRENGTHENED**

The v1 reasoning stands: low turnover is *correct* for a slow predictor, so the defect is the
**absence of any fast signal**, bounded on the other side by Novy-Marx & Velikov. v2 adds the
piece v1 was missing — **a candidate fast-ish signal with a verified turnover number and a
verified net-of-cost result** (JOB 1). Note the honest nuance: residual momentum is **not** a
*fast* signal. It is a 12-2 month, monthly-rebalanced signal — *slower* than our current 1m leg.
It improves signal **quality**, not signal **speed**. The genuinely fast candidate remains v1's
F8 short-horizon reversal. **These are complements, and v2's evidence says they interact:** Huij
& Lansdorp's *Residual Momentum and Reversal Strategies Revisited* exists precisely on that
interaction (paywalled — recorded in the snippet table), and H&W's iMOM *"experiences no
long-term reversals"*, which is exactly the property that lets a reversal leg coexist with it
instead of fighting it.

### Finding (C) — slate composition over score mutation: **PRESERVED, and now better supported**

v1 preferred min-K sector round-robin (slate composition) over soft-diversity (score mutation)
because mutating `composite_score` contaminates the DSR ≥ 0.95 / PBO ≤ 0.20 gates. **Preserved
verbatim.** v2 strengthens it: residual momentum is *also* a score change, so it inherits the same
contamination risk — but unlike a diversity penalty it is a **signal**, and its effect on the
gates is exactly what the gates are for. The distinction to carry into the contract is
*signal-vs-penalty*, not *score-vs-slate*: a better score is legitimately gate-measurable; an
arbitrary diversity penalty is not.

### Verified independently: the 86.60 unsorted-slice finding (context item)

The caller supplied this as background. I checked it because it changes what an overlay can do:

- `screen_universe` builds `results` by iterating the input ticker list (`for ticker in tickers:`
  at `screener.py:147`, `results.append(row)` at `:240`) and returns at `:246` — **with no sort
  anywhere before the return** (verified by scanning every `sort`/`return` occurrence below
  `:250`; the only `return` is `:246`).
- All eight overlay slices are literally `screen_data[: 2 * settings.paper_screen_top_n]` at
  `backend/services/autonomous_loop.py:749, :769, :833, :860, :884, :910, :938, :967` —
  confirmed line-by-line, all eight identical.
- **So the overlays see the first 20 tickers of the UNIVERSE LIST, not the top 20 of any
  ranking.** Confirmed. They are score adjustments inside a set momentum already chose, never
  entry paths. **This has a direct consequence for JOB 1: adding residual momentum as another
  overlay would inherit the same defect and could not change which names get analysed.** It must
  enter the *composite* at `screener.py:299-305` (or as a pre-rank transform), not as a ninth
  overlay.

### Live-value verification: the prescribed route DOES NOT EXPOSE THESE FLAGS

The caller directed me to read live values from `GET /api/settings/` rather than from
`settings.py` defaults, because `backend/.env` is DENIED. I did (HTTP **200**, 45 keys):

- **Confirmed live:** `paper_screen_top_n = 10`, `paper_analyze_top_n = 5` — matching the
  `settings.py:406-407` defaults. So the "20-element overlay slice" and the "5 names deep-analysed
  per day" figures are **runtime-verified**, not assumed.
- **NOT exposed — all five absent from the response:** `sector_neutral_momentum_enabled`,
  `multidim_momentum_enabled`, `paper_soft_sector_diversity_enabled`,
  `paper_soft_sector_diversity_w`, `paper_min_k_sectors_analyzed`. The endpoint returns 45 keys
  and none of them is one of these (the only sector-ish keys present are
  `sector_calendars_enabled=True`, `sector_calendars_lookahead_days=7`, `paper_max_per_sector=5`).

> ### DISAGREEMENT 2 / OPEN GAP — "the three mitigations are dark" is NOT runtime-verifiable
> today. `.env` is DENIED, and the settings API does not surface these flags. Their only
> consumers are `backend/services/autonomous_loop.py:693, :991, :994, :1124`, all via
> `getattr(settings, ..., False)`. **So both v1's framing and mine rest on the `settings.py`
> declared defaults, which per this project's standing "committed is NOT in force" rule is
> evidence, not proof.** I am reporting this rather than restating v1's LIMITATION as though the
> API had resolved it — it did not. If Main needs certainty, the available routes are a log line
> from a live cycle or a BQ record of the cycle's ranking parameters; **not** the settings
> endpoint. *(Note the step must not promote these flags regardless — so this gap blocks a
> claim, not the step.)*

---

## Consensus vs debate (external)

**Consensus.** (a) Turnover must be paid for and most high-turnover anomalies do not survive
costs (F1). (b) Optimal trading is *partial* adjustment toward an aim, never full rebalancing to
the target (F3). (c) Slow signals *should* produce low turnover (F3). (d) Diversity is safer
imposed on slate composition than on the score (F5/F6).

**Live debate — cited inside w20721.** Novy-Marx & Velikov record that *"More recently Frazzini
et al. (2014) have argued that 'actual trading costs are less than a tenth as large as, and
therefore the potential scale of these strategies is more than an order of magnitude la[rger]'"*.
So the size of the cost constraint is genuinely contested: N-M&V's estimates are the conservative
end, Frazzini-Israel-Moskowitz's live institutional data the permissive end. **A turnover-raising
proposal that only survives under FIM-style optimistic costs should be treated as unproven.**

**Second tension.** F8 (add a fast reversal signal) pushes turnover *up*; F1/F2 and the entire
2024-2026 recency window push it *down*. These are reconciled only by F3: add the fast signal to
the *forecast*, then let partial adjustment decide how much of it to trade.

---

## Pitfalls (from the literature)

1. **Churn for its own sake destroys value** — few >50%-turnover strategies keep significant net
   spreads (F1). Any picker change must report measured turnover, not assume it is affordable.
2. **Diversity penalties that mutate `composite_score` contaminate the metric the gates read.**
   `_apply_soft_sector_diversity` overwrites `composite_score` (`:538`); DSR/PBO are then computed
   on a score that is part signal, part penalty. `_min_k_sector_slice` does not have this problem.
3. **Hand-set penalty weights drift** and need per-market retuning (F6) — exactly the failure mode
   `paper_soft_sector_diversity_w` invites.
4. **Hard sector-neutralisation is already measured negative here** (-0.166 Sharpe).
5. **Cost is concentrated in small caps and high-idiosyncratic-vol names** (F2, F8) — a naive
   diversity rule that reaches deeper into sectors reaches into *exactly* those names.
6. **Bandit exploration needs dense reward**; 5 analysed names/day with delayed P&L is the
   sparsest plausible feedback (F9).
7. **Every picker variant tried raises the DSR bar.** DSR's N is the *iteration/trial count*
   (project memory `project_dsr_trial_count_reset_82_25`), so sweeping `w` and `K` over many
   values inflates N and makes `min_dsr=0.95` harder — a real cost of a wide search.
8. **Unstandardised weights are not the stated weights** — declaring 0.40/0.35/0.25 while
   summing raw multi-horizon returns misrepresents the model to anyone reading the config.

---

## Application to pyfinagent (external findings → file:line anchors)

**A1. Cross-sectional standardisation is the cheapest true fix, and the helper already exists.**
Z-score each horizon before weighting so nominal weights become effective weights. `_zscore` is at
`screener.py:541-553` but is reachable only through the dark multidim path. Making the *live*
composite standardised is a small, auditable change at `:299-305`. Note it is **not** a turnover
fix on its own (a monotone re-weighting of the same slow state) — it is a **correctness** fix.

**A2. Only a fast-decaying term creates real day-over-day variation.** Per F3, every currently
proposed overlay is another slow signal, so it cannot move day-to-day set membership much.
Short-horizon reversal (F8) is the one surveyed mechanism that does — but it must be
residualised against industry/factor momentum (Blitz-van der Grient-Honarvar) or it fights the
momentum leg, and it must be large-cap-restricted or costs eat it.

**A3. Prefer slate-composition over score mutation.** `_min_k_sector_slice`
(`autonomous_loop.py:169-209`, gated by `paper_min_k_sectors_analyzed`, `settings.py:489`) changes
only *which* names reach the deep-analyse slice and leaves `composite_score` untouched — so DSR/PBO
still measure the signal. `_apply_soft_sector_diversity` (`screener.py:494-499, 505-539`) mutates
the score. **On F5/F6 evidence, prefer the min-K lever; treat `w` as the riskier of the two.**

**A4. If a diversity penalty is used, bound its cost explicitly.** Replace a free-floating `w`
with the F7 form: maximise sector spread **subject to** a stated maximum Sharpe/return
degradation. That converts "how much diversity" into "how much am I willing to pay", which is
directly checkable against the gates instead of being a magic constant.

**A5. Consider making the min-K pick stochastic** to actually inherit the MB guarantee (F5) and to
introduce the only cheap exploration available — sample the sector, then take its leader, rather
than the deterministic top-k-by-peak at `autonomous_loop.py:188-200`. This is a genuine
explore-exploit mechanism that costs no new data feed. It does, however, make cycles
non-reproducible without a seed — a real tradeoff for this project's replay discipline.

**A6. Every change must still clear the immutable gates.** `PromotionGate(min_dsr=0.95,
max_pbo=0.20, min_pbo_trials=10)` at `autoresearch/gate.py:22-25`, plus the staging `PBO_CEILING`
at `promotion_gate.py:53-63`. Per project memory, **PBO — not DSR — has been the binding wall**.
Add a pre-gate sanity check from F1: report **one-sided monthly turnover** next to DSR/PBO, and
treat >50% as requiring explicit justification.

**A7. Do not describe the two arXiv citations as financial validation** in the contract. State
them as cross-domain mechanism transfers, and note the deterministic-vs-stochastic fidelity gap.

---

## v3 VERIFICATION & OWNERSHIP PASS (2026-08-13, attempt 3)

The v2 run dropped before its final act. This pass does **no new research**. Its job is to decide
whether the body above can be certified as a return, and it does that by **re-checking rather than
inheriting**. Everything below is a measurement made in the v3 session.

### V-1. Every claimed read-in-full source carries substantive content — **10/10, none dropped**

The check was deliberately not "does the URL appear" (that is the bar v1 failed in the opposite
direction). For each of the 10 rows I required **≥2 distinctive content markers** — a verbatim
quoted phrase or a published figure — present in the prose **outside** its own table row. A row
passing on its URL alone would have been dropped from the count.

| # | Source | Markers found | Verdict |
|---|---|---|---|
| 1 | Novy-Marx & Velikov w20721 | 4/4 (`one-sided monthly turnover lower than 50%`, `average over the long and short side`, `between 14% and 35%`, `momentum and its derivative anomalies`) | SUBSTANTIVE |
| 2 | Gârleanu & Pedersen w15205 | 2/2 (`aim in front of the target`, `slower mean reversion`) | SUBSTANTIVE |
| 3 | Daniel & Moskowitz w20439 | 2/2 (`approximately doubles the alpha`, `up- versus down-beta asymmetry`) | SUBSTANTIVE |
| 4 | Lichtenberg et al. 2408.09168 | 3/3 (`samples a content type according to`, `13.57`, `independent of the underlying scoring function`) | SUBSTANTIVE |
| 5 | Garcia & Messud 2601.08717 | 2/2 (`synthetic data (energy assets)`, `controlling expected profit and risk degradation`) | SUBSTANTIVE |
| 6 | Quantpedia short-term reversal | 3/3 (`52.94`, `16.25`, `excessively trading in small cap stocks`) | SUBSTANTIVE |
| 7 | Quantpedia residual momentum | 3/3 (`9.18`, `59.74`, `standardized by the standard deviation`) | SUBSTANTIVE |
| 8 | Hanauer & Windmüller | 4/4 (`65.32`, `long leg plus the short leg`, `most of the performance improvement comes from orthogonalizing`, `5.67`) | SUBSTANTIVE |
| 9 | CXO Advisory [ADVERSARIAL] | 4/4 (`1.39`, `1.54`, `half the volatility`, `1940-2000`) | SUBSTANTIVE |
| 10 | Seppä-Lassila (Aalto MSc, **tier-5**) | 2/2 (`does not come from overly high turnover`, `more concentrated on large-cap stocks`) | SUBSTANTIVE |

**Read-in-full count corrected upward from the born-inert seed: 6 → 10.** The seed value of 6 was
never updated after rows 7-10 landed. The floor is 5; it is met on rows 1-9 alone, i.e. **without
leaning on the single tier-5 student thesis** (#10), which is the weakest source in the set and is
corroborative rather than load-bearing.

### V-2. The two decisive external claims were RE-EXTRACTED, not inherited

Project memory records that WebFetch PDF summarisation has **fabricated quotes twice** here, so a
quote I did not extract myself is not a quote I can certify. I re-downloaded both decisive PDFs
with `curl` and re-extracted with `pypdf`, then regex-verified each load-bearing string:

- **Hanauer & Windmüller** — 65 pages / 121,826 chars, **byte-identical to v2's reported
  extraction**. All 11 probes FOUND. The turnover table is verbatim in the extracted text:
  `Turnover (in %) 53.79 80.63 82.22 65.32` (US) and `50.32 70.69 81.06 62.59` (Global), with the
  break-even row immediately below it: `Round-trip costs at 5% sign. level (in %) 0.62 1.02 1.03
  0.77` (US) and `0.35 0.76 0.52 0.87` (Global). The convention sentence is verbatim: *"Table 6
  shows the average (over time) one-way portfolio turnover of the long leg plus the short leg."*
  Footnote 7 verbatim: *"most of the performance improvement comes from orthogonalizing returns
  with the market factor."* Spanning row verbatim: `iMOM α 0.32 0.28 0.29 t(α) 5.67 4.99 5.21`.
- **Novy-Marx & Velikov w20721** — 61 pages / 103,894 chars, **byte-identical to v2's report**.
  All 6 probes FOUND, including **both** per-side convention sentences (*"average turnover (average
  over the long and short side)"* and *"monthly average turnover of each side of the strategy"*),
  the *"between 14% and 35% per month"* band, and the Frazzini *"less than a tenth as large"*
  counter-estimate.

**This is what certifies JOB 1's central conclusion.** The claim that residual momentum is *not*
high-turnover by this project's adopted standard rests entirely on a convention mismatch — H&W
report a **sum** of both legs, N-M&V a **per-side average**. I verified both definitions at source
in their own words. The ÷2 normalisation remains a normalisation, not a measurement (H&W publish
only the sum), and is still flagged as such above.

*Probe hygiene, disclosed:* my `0\.77` probe first matched a cell in a **correlation** table, not
the break-even row. The break-even figure is nonetheless confirmed — it is visible verbatim in the
context window of the `65.32` match quoted above. Reporting this because a probe that matches the
wrong place is exactly how a clean check lies.

*Incidental corroboration:* the v2 session's scratchpad still holds `revisited.pdf` at **243
bytes** — precisely the failed S3 `AccessDenied` fetch v2 documented in its snippet table. v2
recorded its failures accurately.

### V-3. Internal findings (A) and (B): re-verified at source, anchors corrected

| Claim | Verification run in v3 | Result |
|---|---|---|
| `_zscore` defined at `screener.py:532` | `grep -n "_zscore" backend/tools/screener.py` → 5 hits total | **CONFIRMED**: def at `:532`; calls at `:607,:608,:609,:610` and **nowhere else in the file** |
| Those calls sit off the live path | enclosing `def` scan: `:564 _apply_multidim_momentum`, next def `:626 _pct_change` | **CONFIRMED** — all four calls are inside `_apply_multidim_momentum`, gated by `multidim_momentum_enabled` |
| Composite at `:299-305`, weights 0.40/0.35/0.25 on RAW returns | `sed -n '295,315p'` | **CONFIRMED** verbatim — `if strategy == "momentum":` at `:299`, weights at `:302-304`, closing paren `:305` |
| `screen_universe` returns UNSORTED | `sed` at `:145-150`/`:238-248` + scan of every `sort`/`sorted`/`return` in `:100-250` | **CONFIRMED** — `for ticker in tickers:` `:147`, `results.append(row)` `:240`, `return results` `:246`, and the *only* match in that range is the bare return: **no sort anywhere before it** |
| Eight identical overlay slices | `grep -n "screen_data\[: *2 *\* *settings.paper_screen_top_n\]"` | **CONFIRMED — exactly 8**, at `:749, :769, :833, :860, :884, :910, :938, :967` |

So **finding (A)** — declared weights 0.40/0.35/0.25 are applied to raw returns of differing
dispersion, standardisation exists but is unreachable, therefore reweighting before standardising
tunes a dead knob — and **finding (B)** — low turnover is *correct* for a slow predictor, so the
defect is the absence of a fast signal — both stand as my own measurements, not as inherited text.
The preference for **slate composition over score mutation** is preserved unchanged.

> ### DISAGREEMENT 3 (anchors — flagged, not silently patched)
> v2 caught the stale `_zscore` anchor in its inventory table but **left the same stale number in
> Application A1**, which still reads `screener.py:541-553`. The correct anchor is **`:532`**
> (helper) with call sites **`:607-610`**, and `_apply_multidim_momentum` is at **`:564`**, not
> `:443-452`. Two further sub-anchors in the JOB-3 section are off by two: RSI penalties are at
> **`:307-310`** (not `:305-308`) and the vol penalty at **`:312-313`** (not `:310-311`).
> **Main must take anchors from this V-3 table, not from the inventory table or A1.** The
> conclusions are unaffected; only the line numbers were wrong. `screener.py` is 759 lines and
> `autonomous_loop.py` 3,752 as of this pass — re-derive before citing either again.

### V-4. Counts re-derived from the file on disk, with the rule stated

- **URLs collected: 55.** Rule: distinct absolute `http(s)://` strings literally present in this
  file, trailing punctuation stripped, **minus the one arXiv search-UI endpoint**
  (`arxiv.org/search/?searchtype=all&query=`), which is a tool invocation and not a candidate
  source. Raw distinct count is **56** under two independent terminator rules that agree — the
  control matters, because a single regex agreeing with itself proves nothing.
  *This does not match the 59 quoted to me in the spawn prompt; I report what I can measure on
  disk under a stated rule rather than inheriting a number I cannot reproduce.*
- **Read in full: 10** (V-1). **Snippet-only: 45.** 55 − 10 = 45, which independently matches the
  count of table rows whose first cell is a URL, minus the read-in-full rows. The v2 snippet
  section is *headed* "26" and lists 22 + 4 = 26 rows in its two sub-tables; the remainder are
  URLs cited in prose (method notes, dead/403 attempts, corroborations) that are genuinely
  evaluated-but-not-read. **I report 45 as the honest superset and leave the section heading as
  v2 wrote it** rather than restating its prose.
- **Internal files inspected: 5** — `screener.py`, `autonomous_loop.py`, `autoresearch/gate.py`,
  `services/promotion_gate.py`, `config/settings.py`. **Corrected downward from the seed's 6**,
  which counted something the inventory does not evidence. Of these, **2 were personally
  re-verified in v3** (`screener.py`, `autonomous_loop.py`); the other three are v2's reads with
  file:line anchors, and `backend/.env` remains **DENIED** and therefore uninspected.

### V-5. What this gate does and does not authorise

This step is **BLOCKED behind 86.69**: 81.2% of analyses persist as an empty placeholder scored
0.0 and labelled HOLD, so **no ranking change can pay off until that is fixed**. Clearing this
gate closes an open protocol obligation; it does **not** authorise ranking work. Nothing here was
implemented and no production code was edited — this pass was read-only.

---

## Evidence gaps (honest)

1. ~~**Residual / idiosyncratic momentum: NO source read in full.**~~ **SUPERSEDED by JOB 1 (v2)
   and confirmed in V-1 (v3).** This line is v1 text that the dropped v2 run never came back to
   revise; it now contradicts the body above and must not be read as current. The gap is
   **CLOSED**: three sources on this branch were read in full (#7 Quantpedia, #8 Hanauer &
   Windmüller, #10 Seppä-Lassila) plus #9 CXO Advisory as the qualifying/adversarial view, and
   the decisive numbers from #8 were re-extracted and regex-verified in v3. **The residual
   limitation, stated precisely:** the canonical Blitz, Huij & Martens (2011) paper itself remains
   **paywalled and unread** (SSRN abstract only); its construction and results reach this brief
   *via* sources that reproduce them, which is weaker than reading the original.
2. **Bandit/explore-exploit: snippet-level only** (F9). 23 candidates identified, none read.
3. **Sector-neutralisation external evidence is thin** — the strongest datapoint is this
   project's own -0.166 replay, not the external literature.
4. **`backend/.env` not readable** — runtime flag values unverified (see LIMITATION above).
5. **Frazzini et al. (2014)** — the adversarial cost estimate is quoted *via* w20721, not read
   at source.

---

## Research Gate Checklist — RE-DERIVED IN v3 (the v1 figures below were stale and are replaced)

> The version of this checklist left on disk by the dropped v2 run still claimed **"6" sources**
> and **"30 unique URLs"**. **30 is the exact over-claim that failed the v1 gate** (13 URLs were
> actually present). Both numbers are now re-derived from the file on disk under the rules stated
> in V-4 — not carried forward, and not rounded up.

Hard blockers:
- [x] **≥5 authoritative external sources READ IN FULL — 10.** All 10 pass the ≥2-content-marker
      test in V-1; none dropped. The floor is cleared on rows 1-9 alone, i.e. **without** the one
      tier-5 student thesis. Hierarchy is respected: 3 peer-reviewed NBER working papers +
      2 arXiv preprints + 3 industry/practitioner + 1 industry review + 1 tier-5 thesis.
- [x] **10+ unique URLs total — 55** (10 read in full + 45 evaluated-not-read), from a raw distinct
      count of 56 minus one search-UI endpoint; two independent terminator rules agree (V-4).
- [x] **Recency scan (2024-2026) performed + reported** — 3 complementary findings, 0 superseding.
      Three-variant query discipline is visible in the tables (year-less canonical, 2024-25 window,
      2026 frontier).
- [x] **Full papers/pages read, not abstracts** — arXiv `/html/` full texts, never `/abs/`, and
      **never** a WebFetch on an `arxiv.org/pdf/` URL. NBER + Lancaster + Aalto PDFs extracted
      locally with `pypdf`; the two decisive ones **re-extracted and regex-verified in v3** (V-2).
- [x] **file:line anchors for every internal claim** — and the load-bearing ones re-verified at
      source in V-3, with three stale anchors corrected in DISAGREEMENT 3.

Soft checks:
- [x] Internal exploration covered every module in the caller's scope (`screener.py` score/`_zscore`/
      `screen_universe`; `autonomous_loop.py` overlay slices).
- [x] Contradictions/consensus noted — Novy-Marx & Velikov vs Frazzini et al. on cost magnitude;
      F8 vs F1 on turnover direction; CXO Advisory as the qualifying view against #7/#8/#10.
- [x] Claims cited per-claim.
- [x] **Residual momentum branch NOW COVERED** — was the v1 gap; closed in JOB 1, confirmed V-1.
- [ ] **Open, disclosed, not padded over:** (a) Blitz-Huij-Martens (2011) itself is paywalled and
      unread — reached only via reproducing sources; (b) the three dark mitigations are **not**
      runtime-verifiable (`.env` DENIED, settings API does not expose them — DISAGREEMENT 2);
      (c) the ÷2 turnover normalisation is a normalisation, not a measurement; (d) the "≈2.4×"
      dispersion ratio is IID theory, unmeasured on our universe; (e) bandit/explore-exploit is
      snippet-level only (F9).

**gate_passed: true** — ≥5 read in full (10, verified individually), ≥10 URLs (55, measured on
disk), recency scan performed, hard blockers all satisfied, and the step is not audit-class. The
open items above are **disclosed limitations on specific claims**, not unmet floors; each one is
attached to the claim it bounds rather than left as a general caveat.
