---
name: pbo-level-and-dead-gate-82-27
description: MEASURED 82.27 - CSCV columns are sweep-level configs (Bailey Algo 2.3 verbatim); the 82.23 min_pbo_trials floor is DEAD because build_per_strategy_candidates whitelists 5 keys; generate_report has 15 invocations not 16; "return 0.0 on N<2" has NO upstream precedent
metadata:
  type: project
---

Measured 2026-08-04 during the 82.27 research gate. Do not re-derive.

**The dead gate (the load-bearing one).** `backend/autoresearch/strategy_backtest_adapter.py:264-278`
emits `pbo_n_trials`/`pbo_gate_grade`/`pbo_column_corr_mean`/`pbo_columns_diverse`
alongside `pbo`, but `strategy_candidate_producer.py:115-123` rebuilds the
candidate from a hardcoded **5-key whitelist** (`strategy, dsr, pbo, params,
sharpe`) and silently drops the rest. Run live against the real modules:
`pbo_n_trials=3` -> gate verdict `{'promoted': True}`; with the key restored ->
`{'promoted': False, 'reason': 'pbo_trials_below_min:3<10'}`. So the phase-82.23
`min_pbo_trials=10` floor at `gate.py:43-53` is UNREACHABLE from the only live
producer. `_DEFAULT_K = 8` (adapter `:70`) is below that floor, so every adapter
run today would promote on a non-gate-grade PBO. Fix must ADD keys -- the
5-key block also carries the load-bearing skip at `:107-113`.

**Level.** Bailey/Borwein/Lopez de Prado/Zhu Algorithm 2.3, verbatim: each column
is "a vector of profits and losses over t = 1,...,T observations associated with
a particular model configuration"; "observations are synchronous for every row";
"N >> 10 is required"; "T should be chosen to be double of the number of
observations used by the investor to choose a model configuration". Guided
searches: "the columns of matrix M should be the final outcome of each guided
search ... and not the intermediate steps" -- which disqualifies
`quant_optimizer.py` twice (it is a guided search AND retains no per-iteration
series, only scalars to `quant_results.tsv:610`). Read-in-full path that works:
`curl https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf` + `pdfplumber`
(34 pp, 63,988 chars) -- both pdfplumber and pypdf are already in `.venv`.

**No upstream precedent for the false-good.** CRAN `pbo` (example uses `N <- 200`),
mlfinlab's CV module (generates paths, does NOT compute PBO), and the reference
Python walk-throughs all document ZERO small-N fallback. So `analytics.py:297-298`
`if N < 2 or T < S*2: return 0.0` is a pyfinagent-local invention, not a known
trap -- don't cite a precedent that doesn't exist.

**Counts, so nobody re-asserts them.** `generate_report` = **15 invocations**
(all single-run) + 1 monkeypatch substitution at
`backend/tests/test_phase_82_23_pbo_in_gate.py:216` + the `def` at
`analytics.py:741`. The circulating "16 call sites" counts the monkeypatch.
`quant_results.tsv` has **NO pbo column** (539 rows; the 7 "pbo" hits are inside
`params_json` on rows 533-539). Of 437 `experiments/results/*.json`, exactly **3**
carry a `"pbo"` key, all phase-82.3 outputs.

**Also true:** `risk_server.py:133-158` is a CONSUMER (matrix arrives as a tool
arg) but calls raw `compute_pbo` at `:142-143`, so it re-exposes the false-good on
the MCP surface. Nothing schedules `run_rotation_bakeoff` -- the weekly cron is
still DEFERRED (`rotation_runner.py:36`), so the gate path is operator-triggered.
A high PBO over near-identical columns is the PAPER's own documented benign case
(Sec 5.2 limitation 4), which is why the 0.7486 incumbent reading at column corr
0.967-0.979 is "not gate-grade", not "overfit".

Related: [[pbo-single-strategy-cpcv]], [[strategy-rotation-seed-set]],
[[psr-dsr-formulas]].
