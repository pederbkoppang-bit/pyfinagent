---
name: three-degradations-86-62
description: Step 86.62 triage — all 3 degradations chronic not transient; the promoted-404 is harmless because best_params NEVER reaches decide_trades; the AV rate limit ZEROES the social overlay via a keyword proxy; the cycle's own record says "clean"
metadata:
  type: project
---

Step 86.62 (2026-08-13) triaged the three 2026-08-11 cycle degradations. Every
"obvious" theory was refuted by the code; the real findings are elsewhere.

**Why:** the step was filed because three degradations logged and none was triaged.
The measured answer is that the logs are the ONLY surface they have — the durable
per-cycle record contradicts them.

**How to apply:** before blaming any of these three for a trading defect, re-check
the specific refutations below. Two of the three are provably inert.

## 1. The MetaCoordinator p95 has NO CONSUMER and measures the DASHBOARD

- `meta_coordinator.py:157-162` builds the reason string; `:120`
  `DEFAULT_LATENCY_THRESHOLD_MS = 500.0`; value from `gather_health` `:263-269`.
- `perf_tracker.summarize()` defaults to a **300-second window** (`perf_tracker.py:59`)
  and is called with no argument — but the decision fires **~80 min after cycle start**,
  so the window contains **none of the cycle's work**. It is the frontend poll set
  (`/live-prices`, `/kill-switch`, `/snapshots?limit=365`, ...).
- `summarize()` returns `total_requests` (`:109`) + `per_endpoint` (`:116`);
  `gather_health` reads **only** `p95_ms` (`meta_coordinator.py:267`). The denominator
  is computed and thrown away — the logged percentile is unfalsifiable.
- `_percentile` (`perf_tracker.py:144-154`) has **no minimum-N guard**.
- **`grep -rn "perf_opt" backend --include='*.py'` returns only the enum comment
  (`:110`) and the construction site (`:159`).** `summary["coordinator"]`
  (`autonomous_loop.py:1820`) has **no reader**. The decision is recorded, never acted on.
- Side effect that IS real: `perf_opt` is Priority 1 and returns early, so `quant_opt`
  (Priority 2) is unreachable on those cycles. Measured 21d: perf_opt 10, skill_opt 4,
  **quant_opt 0**, idle 0.

## 2. The promoted-strategy 404 is a MISSING TABLE and is CONSEQUENCE-FREE

- `404 ... reason: notFound` for `pyfinagent_data.promoted_strategies` **in location US**,
  and `pyfinagent_data` IS a US dataset — so not a location mismatch. A permission
  failure would be **403 accessDenied**, and the request carried a **Job ID**, proving
  `bigquery.jobs.create`. `scripts/migrations/create_promoted_strategies_table.py`
  exists with an `--apply` flag → the migration was apparently never applied.
- **THE KEY REFUTATION:** `best_params` is used at exactly three places in
  `backend/services/autonomous_loop.py` — `:499` assign, `:500-505` two summary fields,
  `:1850-1851` heartbeat `decided_strategy`. `decide_trades` (`:1662-1669`,
  signature `portfolio_manager.py:164-172`) **does not take it**. Live risk/sizing is
  `settings.paper_*`. Two in-repo comments say so outright:
  `strategy_backtest_adapter.py:43` and `strategy_registry.py:40`.
- So "the cycle ran on FALLBACK parameters" is a **non-event for that cycle's orders**.
  The worse finding it exposes: the promotion pipeline is not connected to live orders
  at all. That belongs to the bridge step, not to cycle error-handling.

## 3. The AV rate limit ZEROES the social overlay (Branch A is the production path)

- `social_sentiment.py:67-69` — the limiter response is **HTTP 200**, so
  `raise_for_status()` never fires and the `except` at `:145` is unreachable.
  `feed = []`.
- **Branch A** (`:75-76`, taken whenever `fallback_articles` exist — and
  `orchestrator.py:2041` passes `articles or fallback_articles or None`, while
  `alphavantage.py:82-84` itself substitutes 10 yfinance articles on ITS own limit):
  `_score_fallback_articles` keyword-scores yfinance headlines with a 20+20-word
  lexicon. **A NUMBER IS SCORED.** Two zero-substitutions inside it:
  `_keyword_score` `:44-45` `if total == 0: return 0.0`, and `:162`
  `avg_sentiment = ... if all_scores else 0`. `0.0` sits inside the NEUTRAL band
  (`:177-185`, |x| <= 0.15) — the imputed value is camouflaged as the most common
  legitimate answer.
- **Branch B** (`:77-81`, no fallback articles): returns `signal: NO_DATA` with **no
  `avg_sentiment` key** → `analysis.py:251` `.get()` → `None` → column NULL → OMITTED.
- Provenance IS produced (`:196` `"data_source": "yfinance_fallback"`) and then
  **dropped**: `bigquery_client.save_report` has `social_sentiment_score` (`:97`) and
  `social_sentiment_velocity` (`:145`) and **no social provenance parameter**.
- **Positive control that the absence is real:** the repo already persists exactly this
  provenance for a different tool — `orchestrator.py:2007 source="yfinance_fallback"`,
  aggregated by `bigquery_client.py:951-952` as `pct_yfinance_fallback_dominance`.

## 4. THE GENERAL FINDING — the cycle record says "clean"

`handoff/cycle_history.jsonl`, cycle `86667da7` (`completed_at 2026-08-11T19:21:29Z`):
`"degradation": null, "meta_scorer_degraded": false, "rail_skipped": false,
"breaker_tripped": false, "error_count": 0, "n_trades": 0`. Across **81 completed rows
the `degradation` key exists on only 2**, and both carry a different family
(`fallback_rate`/`degraded_analyses`). `meta_scorer_degraded` on 40/81;
`funnel`/`rail_skipped`/`breaker_tripped` on 26/81 — the schema grew over time and is
sparsely populated. **That is why nothing was triaged: the degradations live only in
free-text log lines, and the one durable artefact reports nothing happened.**

## Measurement notes that transfer

- Cycle counting: `zgrep -c "Paper trading: Step 1"` across
  `handoff/logs/backend.log.2026{0729,0804,0810}*.gz` + live `backend.log` gives
  4+5+6+4 = **19**, which **reproduced the caller's independently-stated 19**. Use that
  as the positive control for any 21-day log population claim.
- The gz rotation timestamp is when the file was CLOSED, so its contents PRECEDE it.
  `backend.log.20260724T064045Z.gz` is mostly OUTSIDE a 07-24..08-13 window.
- Pin `/usr/bin/grep` (the shell's `grep` is ugrep) and quote `--include='*.py'`
  (zsh globs it otherwise and the command errors with "no matches found").

Related: [[project_absent_upstream_data_86_41]] (the QuantAgent NoneType is a REMOTE
Cloud Function, not this repo), [[project_rec_vocabulary_86_20]], and the masterplan
steps 86.47 (drought), 86.60 (blind overlays), 86.69 (81% empty rows scored 0.0/HOLD) —
86.69 is the SAME failure class as finding 3 but a DIFFERENT instance; do not conflate.
