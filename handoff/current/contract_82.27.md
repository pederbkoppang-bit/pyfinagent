# Contract -- phase-82.27

**Step**: RE-SPEC of 82.23 criterion 1 -- name the SWEEP-LEVEL producer, not `generate_report`.
**Status at write time**: pending. **Depends on**: 82.22 (done).

## 1. Research gate -- PASSED

`handoff/current/research_brief_82.27.md` (574 lines). Launched on the **Workflow
rail** per the standing operator instruction that both Layer-3 agents use it; the raw
envelope is archived at `handoff/current/qa_returns/w5j8o7fsj.output.json`.

```
gate_passed=true  tier=moderate  external_sources_read_in_full=6
snippet_only=22   urls_collected=28  recency_scan_performed=true
internal_files_inspected=13
```

### What the canonical source actually requires

Bailey, Borwein, Lopez de Prado & Zhu, *The Probability of Backtest Overfitting*
(read in full, 34pp, via curl + pdfplumber per the research-gate PDF chain).
Algorithm 2.3, verbatim: *"each column n = 1,...,N represents a vector of profits and
losses over t = 1,...,T observations associated with a particular model configuration
tried by the researcher. M is therefore a real-valued matrix of order (T x N)"*, rows
synchronous across the N trials.

Three findings that bind this step:

- **Guided search is explicitly excluded** (Sec 5.2): *"the columns of matrix M should
  be the final outcome of each guided search (i.e., after it has converged to a
  solution), and not the intermediate steps."*
- **N and T**: *"if the investor is sensitive to values of phi < 1/10 ... N >> 10 is
  required"*; *"T should be chosen to be double of the number of observations used by
  the investor to choose a model configuration."*
- **PBO must not be an optimisation objective** (Sec 5.2, citing Strathern): it
  evaluates a selection process and must not become the thing selected on. It stays
  out of `quant_optimizer`'s search objective. The promotion gate is its home.

External corroboration that 82.23's criterion was unsatisfiable: **no** implementation
surveyed computes PBO inside a single-run report object. CRAN `pbo` takes a
user-assembled frame (its own example uses N=200); the reference Python walk-through
stacks `R_mat` *after* the sweeps complete; mlfinlab generates N paths but computes PBO
in a separate module. The (T x N) matrix is always owned by whatever ran the sweep.

Also: **"return 0.0 on N<2" has no precedent anywhere.** CRAN `pbo`, mlfinlab and the
reference implementation all document zero small-N behaviour, and the paper gives only a
magnitude rule. `analytics.py:297-298` is a pyfinagent-local invention whose sentinel is
the best possible value on a ceiling gate. Do not cite a precedent for it.

### Recency scan (last 2 years) -- 2 findings, neither supersedes CSCV

- Arian, Norouzi M. & Seco (2024), *Knowledge-Based Systems* / SSRN 4778909 -- CPCV
  beats K-Fold / Purged K-Fold / Walk-Forward, **measured by lower PBO and higher DSR**.
  It changes how columns are *generated*, not what they are or who owns them.
- arXiv:2512.12924v1 (Dec 2025, read in full) -- cites CSCV as the reference method,
  notes walk-forward remains the industry standard, declines to deflate its own results.
  No 2026 work on PBO/CSCV surfaced.

### The researcher corrected me four times. All four measured; all four stand.

1. **"16 call sites" is 15.** The 16th is
   `test_phase_82_23_pbo_in_gate.py:216 monkeypatch.setattr(ad, "generate_report", ...)`
   -- a symbol substitution, not an invocation. The *substance* (every invocation
   receives one `BacktestResult`) is confirmed. Its advice is better than the count:
   express it as a **property**, not a number, so it cannot drift when a script is added.
2. **This step needs a WIRING CHANGE, not only tests** -- section 2.
3. **`quant_optimizer` is not a candidate producer** and must be *explicitly excluded*,
   not left open: it retains only scalars (`:610`), no per-iteration return series to
   stack, AND it is a guided search Bailey Sec 5.2 bans. Two independent disqualifications.
4. **`compute_pbo` is at `analytics.py:276`, not `:184`** (`:184` is now a comment block;
   `compute_pbo_checked` at `:208`; the false-good branch at `:297-298`). These have
   already moved once this phase -- re-derive before citing.

## 2. THE DEFECT THIS STEP MUST FIX -- the floor I shipped in 82.23 is DEAD

`backend/autoresearch/strategy_candidate_producer.py:115-123` rebuilds each candidate
from a **hardcoded 5-key whitelist** (`strategy`, `dsr`, `pbo`, `params`, `sharpe`). So
`pbo` reaches `PromotionGate.evaluate`, but `pbo_n_trials`, `pbo_n_obs`,
`pbo_gate_grade`, `pbo_column_corr_mean` and `pbo_columns_diverse` are discarded in
transit.

The researcher ran the real modules rather than reasoning about it:

```
without pbo_n_trials -> promoted: True
with    pbo_n_trials -> promoted: False, reason 'pbo_trials_below_min:3<10'
```

**The `min_pbo_trials = 10` floor I added in 82.23 cannot fire on the only live producer
path.** And `_DEFAULT_K = 8` (adapter `:70`) sits *below* that floor, so the default
sweep would be refused the moment the wiring works -- correct behaviour that must be
surfaced, not accommodated.

This is the class in auto-memory `feedback_operations_that_cannot_fail_loudly`: a gate
term that cannot fail is indistinguishable from one that passes.

**Live hop trace (I4), verified end to end:** `run_rotation_smoke.py:58` ->
`rotation_runner.run_rotation_bakeoff:241` -> `make_engine_backtest_fn:271` ->
`run_strategy_bakeoff:278` -> `strategy_candidate_producer.py:148` ->
`strategy_selector.py:150` -> `g.evaluate(c)` at `strategy_selector.py:94`.

**But nothing schedules it.** `rotation_runner.py:36` and
`strategy_candidate_producer.py:33` both still describe the weekly rotation CRON as
DEFERRED, and no rotation scheduling exists in `backend/services`, `backend/main.py` or
`backend/slack_bot`. The path is real and closed but **currently unexercised in
production** -- this fix is pre-emptive, not an outage repair. Say so plainly in the
results; do not dress it up as a live-money save.

## 3. Hypothesis

Naming `make_engine_backtest_fn` as the sweep-level PBO producer, and restoring the
provenance keys through `strategy_candidate_producer`, makes the promotion gate's PBO
term enforceable for the first time -- with its trial count and diversity number
attached, so a PBO computed over too few or near-identical configurations is refused
rather than believed.

## 4. Anchor decision

**`backend/autoresearch/strategy_backtest_adapter.py:167 make_engine_backtest_fn`.** The
only sweep-level producer that is library code (so a criterion written against it stays
true), sits on the path reaching `PromotionGate.evaluate`, already emits the 82.23
provenance fields (`:264-278`), and enforces Bailey's same-model-different-configuration
rule at `:94-108` with a hard `ValueError` on an unknown strategy so the engine cannot
silently fall back to `triple_barrier`.

`scripts/harness/run_82_3_candidate_backtests.py:143` is a genuine second sweep producer
but a one-shot evidence script that never calls the gate -- named the **offline evidence
producer**, not the anchor. `quant_optimizer` excluded per correction 3.
`risk_server.py:133-158` is a **consumer** (its `pnl_matrix` arrives as a tool argument),
though it re-exposes the false-good at `:142-143` via raw `compute_pbo` -- a one-line
swap, taken here as the same defect class.

## 5. Immutable success criteria -- copied VERBATIM from `.claude/masterplan.json`

Command: `source .venv/bin/activate && python -m pytest backend/tests/test_phase_82_23_pbo_in_gate.py backend/tests/test_phase_82_27_pbo_sweep_producer.py -q`

1. the SWEEP-LEVEL producer (a function that receives or builds N>=2 configuration series, NOT generate_report) emits a pbo together with its n_trials and its mean pairwise column correlation, asserted by a test that EXECUTES that producer against a stub engine and inspects the returned mapping -- an inspect.getsource token scan does not satisfy this
2. the single-run report path is asserted to emit NO pbo, with the test stating the reason (N=1 makes compute_pbo return a false-good 0.0 that passes the <=0.20 live ceiling), so the criterion-1 defect of 82.23 cannot be reintroduced
3. a fixture drives the producer with an under-length matrix (T < S*2) and asserts the emitted pbo is None/absent rather than 0.0, and that the refusal reason is reported
4. every criterion above is mutation-verified: the experiment_results mutation matrix names, for each guard, a concrete production mutation that makes it fail, and records the measured CONTROL count re-derived from the tree in the same run

## 6. Plan

1. Restore the dropped provenance keys through `strategy_candidate_producer.py`'s
   whitelist. Additive; no existing key changes meaning.
2. Swap `risk_server.py:142-143` to `compute_pbo_checked` so the false-good 0.0 is not
   re-exposed through the MCP surface.
3. Emit a **dropped-column count** beside every PBO. The adapter at `:212-216` silently
   drops a failed variant's column; the paper warns *"Hiding trials will lead to an
   underestimation of the overfit"*, so systematically dropping losers biases PBO DOWN.
4. New suite `backend/tests/test_phase_82_27_pbo_sweep_producer.py` -- criteria 1-3 by
   EXECUTING the producer against a stub engine, never by `inspect.getsource`.
5. Mutation matrix with a named production mutation per guard, CONTROL re-derived in the
   same run.

## 7. Scope honesty -- stated before starting

- The live `optimizer_best.json` stays mis-attributed; regenerating it needs an optimizer
  run gated on the frozen `historical_macro`. Unchanged from 82.22.
- No rotation CRON is added. Scheduling is out of scope and stays deferred.
- `_DEFAULT_K = 8` is BELOW the `min_pbo_trials = 10` floor. This step does **not**
  silently raise K to make the gate pass -- that would be tuning the test to the answer.
  It surfaces the conflict; choosing K is a separate decision.
- 82.23 stays permanently `pending`, superseded in place. Criteria immutable, untouched.

## 8. References

- Bailey, Borwein, Lopez de Prado & Zhu, *The Probability of Backtest Overfitting* --
  https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf (read in full)
- Arian, Norouzi M. & Seco (2024), SSRN 4778909 / Knowledge-Based Systems
- arXiv:2512.12924v1 (Dec 2025, read in full)
- CRAN `pbo` -- https://cran.r-project.org/web/packages/pbo/readme/README.html
- `handoff/current/research_brief_82.27.md` (full brief, 28 URLs)
- `handoff/current/qa_returns/w5j8o7fsj.output.json` (raw gate envelope)
