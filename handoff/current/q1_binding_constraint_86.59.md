# Q1 — WHERE IS THE BINDING CONSTRAINT?

**Step:** 86.59 / 86.60 (owning) — answer to Q1 of
`handoff/current/prompt_candidate_picker_research.md`
**Date:** 2026-08-13
**Verdict:** **None of (a)/(b)/(c)/(d) as framed.** The binding constraint is
**upstream analysis availability**: ~4 of every 5 analyses that run return an
**empty placeholder scored 0.0 and persisted as `HOLD`**. This is a **dated
regression**, not a design limitation — it began between **2026-06-12 and
2026-06-15**.

**Operational consequence: STOP optimising the ranking.** Not for the reason the
prompt anticipated (it named (c) and (d) as the stop conditions), but the stop is
firmer: no ranking change can raise BUY count while 81% of the names the ranker
picks receive an empty analysis.

---

## Populations (every number below is derived from one of these two)

| ID | Population | Rule | Size |
|----|-----------|------|------|
| **A** | `sunny-might-477607-p8.financial_reports.analysis_results` | rows with `DATE(analysis_date)` in 2026-05-01..2026-08-13 | 511 rows |
| **B** | backend logs | every line matching `^{"timestamp"` in `backend.log` + all 6 rotated `handoff/logs/backend.log.*.gz`, concatenated once | 906,962 lines, 2026-07-24..2026-08-13 (21 days) |

Population B note: the 6 archives total 5,188,898 lines, but only 906,962 are
JSON-format; the two oldest archives (2026-06-12, 2026-07-06) are plain uvicorn
format and carry no `"timestamp"` field. **Population B therefore starts
2026-07-24, not at the oldest archive.** Every log count below is over B and
inherits that 21-day bound.

BigQuery MCP was **not attached this session** (only the claude.ai OAuth variant
surfaced); per CLAUDE.md BigQuery rule 6 all queries used the Python client with
ADC. Every query is date-bounded and `LIMIT`ed.

---

## 1. The prompt's 21.1% BUY rate is real, and it describes a system that no longer exists

Over the same 90 days the prompt cites I reproduce it: **111/511 = 21.7%**
BUY-class (population A, `UPPER(recommendation) LIKE '%BUY%'`). So the figure is
not wrong — it is an **average across a regime break**.

Split at the break:

| Regime | Analyses | BUY-class | Rate | Days |
|--------|---------:|----------:|-----:|-----:|
| **PRE** (..2026-06-12) | 251 | 103 | **41.0%** | 17 |
| **POST** (2026-06-15..) | 260 | 8 | **3.1%** | 44 |

**A 13.2x collapse on essentially unchanged analysis volume.**

Monthly, for orientation: 2026-05 31.0% · 2026-06 36.6% · **2026-07 2.9%** ·
2026-08 6.1%.

The monthly view *understates* the break, because June contains both regimes. At
daily resolution the last day with any BUY-class recommendation before the
collapse is **2026-06-12**; daily rates in the ten sessions before it run
40–100%. Sessions after 2026-06-15 are zero on all but three days in two months.

> This is the "a range can hide its counterexample" trap: the 90-day window and
> even the monthly bucket both smear the break away. The endpoints had to be
> checked at daily resolution to see it.

---

## 2. Mechanism: a failed analysis is persisted as a `HOLD`

The decisive test — if analyses became genuinely more bearish, `final_score` must
fall; if the score is intact but BUY vanished, the mapping changed. Neither
happened. **The score went degenerate.**

| Regime | n | `final_score == 0.0` | `> 0` | `mean(final_score \| >0)` |
|--------|--:|---------------------:|------:|------------------------:|
| PRE  | 251 | 95 (**37.8%**) | 156 | **6.14** |
| POST | 260 | 211 (**81.2%**) | 49 | **5.81** |

(`final_score` is a 0–10 scale. Zero NULLs in either regime.)

**When the pipeline works it still scores names the same** (6.14 → 5.81). The
engine did not turn bearish. It turned *absent*.

All **211** POST zero-score rows share one signature — verified, not inferred
from a sample:

- `summary` is the **empty string**: 211/211
- `debate_confidence` is **NULL**: 211/211
- text matching `degraded|placeholder|unavailable|failed` in `summary` or
  `recommendation_justification`: **0/211** — the rows are not *labelled*
  degraded, they are simply **blank**

Six most recent such rows, verbatim: `MRVL`, `NTAP`, `BAX`, `DELL`, `HPE`, `PANW`
— every one `score=0.0 rec='HOLD' why=''`.

**An operator reading this table sees `HOLD` and reads a judgment. It is the
absence of one.** This is the `recording-an-absence-as-a-value` class: the failed
analysis is not recorded as a failure, it is recorded as the most common valid
verdict, and a `HOLD` can never become a buy candidate.

---

## 3. The collapse decomposes into two roughly equal factors

| Factor | PRE | POST | Ratio |
|--------|----:|-----:|------:|
| Share of analyses producing a real score | 62.2% | 18.8% | **3.3x** |
| BUY conversion *among* real-score analyses | 57.7% | 16.3% | **3.5x** |
| Product | — | — | **11.6x** |
| Observed end-to-end (41.0% → 3.1%) | — | — | **13.2x** |

Both halves broke. Fixing only the emptiness recovers ~3.3x, not the full 13x.
I state the residual (13.2 vs 11.6) rather than reconciling it: the two factors
are computed on overlapping subsets and I did not derive a decomposition that
closes exactly.

---

## 4. Live-path corroboration (population B, 21 days)

- **19 cycles started** (`Paper trading: Step 1`) on 17 distinct days; only **13
  reached the trade-decision step** (Steps 6/7/8) on 11 days →
  **6 of 19 cycles (31.6%) never reached a trade decision at all.**
- **104 analyses; 5 BUY-class (4.8%)** — matching population A's POST regime, from
  an independent source.
- **Risk judge: 97 `Risk Judge rendering verdict`, of which 60 logged
  `judge response unparseable; fallback verdict=APPROVE_REDUCED`
  = 61.9%.** The majority of risk verdicts in this window are a **parse-failure
  fallback, not a judgment**. The headline "69.8% approve" (67/96) is therefore
  mostly not an approval decision.
- **Only 3 buy candidates reached the binding RiskJudge gate in 21 days**: NTAP
  `APPROVE_REDUCED`, HPE `BLOCKED`, CRWD `BLOCKED`.
- **1 trade**, from 13 `Trade decisions:` lines totalling `0 sells, 1 buys` — the
  NTAP entry on 2026-07-31, which matches the live position's `entry_date`.
- Supporting failure lines: 10 `Full orchestrator failed for <T>: QuantAgent
  failed ... 'NoneType' object has no attribute 'get' -> falling back to lite
  Claude analysis` (5 days); 11 `Degraded-scoring guard fired` (10 days); 8
  `Meta-scorer ran ENTIRELY on the no-<X> fallback` (7 days).
- 3 `UNRECOGNISED recommendation 'new_buy_signal'` (3 days) — **independent
  corroboration of step 86.58**, found without looking for it.

Cycle-level, from `handoff/cycle_history.jsonl`: the **2026-08-12** cycle recorded
`degradation: {'degraded': True, 'degraded_analyses': '6/6'}` with
`breaker_tripped: True`. The system logged that 6 of 6 analyses were degraded.

---

## 5. Answering (a)–(d) directly

**(a) Selection quality — the 5 names chosen are the wrong 5. NOT BINDING.**
Unfalsifiable at present and irrelevant while 81% of picks get an empty analysis.
15 distinct tickers over 104 analyses in 21 days confirms the repetition the
operator reported, but repetition is not what is suppressing trades.

**(b) Throughput — NEAREST, but not for the stated reason.** The prompt frames
throughput as "5 of 577 is too few". The binding throughput is one layer down:
of the ~5 analyses that run per cycle, **~4 return an empty placeholder**.
Effective *working*-analysis throughput is **≈0.9 per cycle, not 5**. At the
measured POST conversion of 16.3%, that is **≈0.15 BUYs per cycle** — about one
BUY every 7 cycles. Measured: 5 BUY-class in 17 cycle-days.

**(c) The risk gate — NOT BINDING, but independently defective.** It cannot be
suppressing volume: only **3 candidates reached it in 21 days**. It is not
rejecting good candidates in quantity because it is not being *offered* them.
Its own defect is the 61.9% unparseable-response fallback rate — a
**pipeline-quality** problem, exactly as the prompt anticipated, **not** a
threshold problem. **No loosening is proposed or warranted; the prohibition
stands untouched.**

**(d) Capital deployment — a real second-order multiplier, not binding.** Live
values read from the running process (`GET /api/settings/`, pid 99231):
`paper_max_positions=30`, `paper_min_cash_reserve_pct=5.0`,
`paper_max_per_sector=5`, `paper_analyze_top_n=5`, `paper_screen_top_n=10`.
Position sizing is the risk judge's `recommended_position_pct` (1–10, default
3.0). The book can hold ~30 positions and holds **1** (NTAP, $1,079.54 of
$23,900.18 NAV = 4.52%; cash $22,820.64 = 95.5%). The cap is **arrival rate**,
not sizing.

---

## 6. What this means for 86.59's planned fix

The fix 86.59 was scoped to make — reweight `mom_1m*0.40 + mom_3m*0.35 +
mom_6m*0.25`, standardise with `_zscore` first, add a bounded-turnover fast
signal — **is correctly diagnosed and cannot pay off yet.** Its own research
already established that reweighting before standardising tunes a disconnected
knob; this measurement adds that even a *correctly* standardised, faster ranker
feeds names into a pipeline that returns an empty `HOLD` for 4 of every 5 of
them.

**Recommended sequencing:** the emptiness regression is a strict prerequisite for
86.59 and 86.60. Both should stay open and unblocked-by-ranking-work until the
POST-break zero-score share is measured back down.

---

## 7. What I could NOT verify — stated, not padded over

1. **The CAUSE of the 2026-06-12/15 break is NOT established.** `git log
   --since=2026-06-11 --until=2026-06-16 -- backend/` shows only away-ops, Slack
   and alerting commits — nothing in the analysis pipeline. Untested hypotheses:
   (i) a backend restart in that window putting *earlier* phase-60 changes into
   force (the `committed-is-not-in-force` class — phase-60 is recorded as
   needing a restart for 60.2–60.4); (ii) a model/provider change; (iii) an
   upstream data-source failure surfacing as the `QuantAgent ... NoneType`
   error. **Each needs its own measurement. I am not asserting a cause.**
2. **I did not read the persist call site** that writes `final_score=0.0` with an
   empty summary. The link between the lite/degraded fallback and that row shape
   is inferred from co-occurrence of the log lines and the row signature, not
   from source. It is a strong inference, not a verified one.
3. **The 3 dark diversity flags are not exposed by `GET /api/settings/`.** The
   endpoint returns 45 keys; a filter matching `sector|diversity|min_k` returned
   `sector_calendars_enabled` and `sector_calendars_lookahead_days` but **none**
   of `sector_neutral_momentum_enabled`, `paper_soft_sector_diversity_enabled`,
   `paper_min_k_sectors_analyzed`. The filter is therefore live (positive
   control passed) and the absence is real — but it means **their live values
   remain unverified**, since reading `backend/.env` is denied. Criterion 4 of
   86.59 requires measuring these; that measurement needs another route.
4. **Recommendation vocabulary is case-inconsistent** in the 21-day window:
   `HOLD` 72 / `Hold` 23, `BUY` 4 / `Buy` 1, `Sell` 4 (no `SELL`). Adjacent to
   86.63's boundary guard. **Observation only — not queued work here**, and no
   criterion in 86.59 owns it.
5. The 08-13 cycle was **still running** while this was written (started
   18:00:00Z, 3,465s elapsed of a 10,800s budget at first check), so 2026-08-13
   contributes partial data to population B and none to the trade counts.
