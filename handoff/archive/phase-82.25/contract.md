# Contract -- masterplan step 82.25

**Step id:** 82.25 (phase-82, P1, harness_required: true, depends_on 82.22)
**Date:** 2026-08-05 | **Cycle:** 1

---

## 1. Research gate summary

**Brief:** `handoff/current/research_brief_82.25.md` | **Envelope:** `gate_passed: true`
-- 6 sources read in full, 12 snippet-only, 18 URLs, recency scan performed, 12 internal
files inspected.

### Findings that decide the design

**(a) N is scoped to the RESEARCH PROCESS, not to one optimisation session.** Bailey &
Lopez de Prado 2014; Lopez de Prado & Lewis 2018: *"ignorance ... makes it impossible to
assess whether a discovery is false"*. So a cumulative count across warm starts is the
correct reading, and resetting to 1 is not a defensible simplification.

**(b) OVER-counting is the CONSERVATIVE direction.** DSR Appendix 3: using `M >= N`
overstates `E[max SR]` and therefore LOWERS the DSR. This matters enormously for the
design: it means a plain cumulative counter is defensible **without** needing
effective-N / trial-clustering machinery, because erring high errs safe.

**(c) The headline reproduces EXACTLY from disk, and is stronger than the step claimed.**
Run `60617e0b`: `exp01` and `exp10` share Sharpe `0.6455483635957818` and differ **72x in
DSR** on trial count alone. This is measured, not folklore -- unlike CLAUDE.md's
"~40-minute hang" which I had to retract in 82.13.

**(d) `:226` (cold baseline `num_trials = 1`) is LEGITIMATE.** Only the two warm-start
resets are the defect. The fix must not touch the baseline.

**(e) THE LIVE FILE IS SCHEMA v1 AND CARRIES NO `num_trials`.** Measured:
`keys = [discarded, dsr, kept, params, run_id, saved_at, sharpe]`. 82.22 changed the
**writer** only (`:792`, with an in-code comment naming 82.25 as its consumer). So
**criterion 3's "no recorded prior count" branch is the PRODUCTION path, not an edge
case.**

**(f) THE SINGLE BIGGEST RISK, and it is a go-live risk.** The live file's
`dsr = 0.9525811126193078` clears the `0.95` gate by **0.0026**
(`paper_go_live_gate.py:42`, `promoter.py:26`, `gate.py:21`). **If the fix ever RECOMPUTES
that persisted figure at a higher N, `dsr_ge_95` flips to FAIL and the go-live gate
closes.** Re-deflating an inherited number would also be fabrication: it was computed at
whatever N its own run used, and that N is unrecorded. **The fix changes only FUTURE
computations; it must never retroactively re-deflate a persisted `dsr`.**

**(g) A larger N makes the KEEP branch strictly harder** (`:318`, threshold 0.95). The
live file already records `kept=0, discarded=10`. Future runs will keep even less. **That
is honest and expected -- and it is stated here so it does not later look like this step
broke the optimizer.**

**(h) Dead prior art states this step's doctrine already.**
`backend/autoresearch/meta_dsr.py` says you must "recompute DSR at the cumulative sample
size across ALL trials, including abandoned ones" and exposes `cumulative_n` vocabulary --
but `meta_dsr()` has **zero production callers** and its penalty formula is a self-declared
stand-in. **Reuse the doctrine and the vocabulary; do NOT reuse the formula** -- the real
one at `analytics.py` is already correct.

---

## 2. Hypothesis

DSR is dominated by N (72x on identical Sharpe), and both warm-start paths silently reset
N to 1 -- so every carried-forward DSR is deflated as though the strategy had been found on
the first attempt. Reading the persisted count and accumulating it makes the reported
statistic honest. Where the prior count is unrecorded (the production path today), the
honest answer is **not** to invent `1`: it is to say the prior depth is unknown and mark
the resulting DSR as an **upper bound**.

---

## 3. Immutable success criteria (VERBATIM from `.claude/masterplan.json`)

1. `a warm-started optimizer carries a cumulative trial count rather than resetting to 1, asserted on a fixture whose source file records a prior count`
2. `the DSR reported after a warm start is deflated against the cumulative count, asserted by comparing against the same run deflated at num_trials=1`
3. `a fixture with no recorded prior count resolves per a documented decision rather than silently defaulting to 1`
4. `a test fails against the current reset-to-1 behaviour, so the guard cannot pass vacuously`

**Verification command (immutable):**
`source .venv/bin/activate && python -m pytest backend/tests/test_phase_82_25_trial_count_reset.py -q`

---

## 4. Plan

### 4.1 `backend/backtest/quant_optimizer.py` -- both warm-start paths

Replace `self.num_trials = 1` at both sites with a shared resolver:

```
prior = source.get("num_trials")
if isinstance(prior, int) and prior > 0:
    self.num_trials = prior            # CUMULATIVE: keep searching from here
    self.prior_trials_known = True
else:
    self.num_trials = 0                # the in-session counter starts clean
    self.prior_trials_known = False    # and the DSR is an UPPER BOUND
```

### 4.2 THE DOCUMENTED DECISION (criterion 3)

When the source records no prior count -- **today's production path** -- the fix does
**not** fabricate a number. Rationale, recorded in code:

- Assuming `1` is **anti-conservative**: it is the single most optimistic assumption
  available, and it is the exact defect being fixed.
- Inventing a large number would be fabrication -- the true depth is unrecorded.
- Per (b), erring HIGH is safe and erring LOW is dangerous. So the honest resolution is
  to mark the prior as **UNKNOWN** and label the resulting DSR as an **upper bound**
  (under-deflated), persisting `prior_trials_known: false` so a downstream reader can
  see it rather than infer it.

This is a *documented decision*, which is what criterion 3 asks for -- not a silent
default.

### 4.3 Persist the accumulation

`_save_best_params` already writes `num_trials` (82.22). Add `prior_trials_known` so the
next warm start inherits the honesty flag rather than laundering an unknown into a known.

### 4.4 EXPLICIT NON-GOALS -- the go-live boundary

- **The persisted `dsr` is NEVER recomputed or re-deflated.** Only future computations at
  the `generate_report` call site change. This is a hard boundary (§1f); a test pins it.
- No change to the `0.95` thresholds, to `paper_go_live_gate.py`, `promoter.py`, or
  `gate.py`.
- No change to `:226` (the legitimate cold baseline).
- No change to the same-named-but-unrelated `num_trials` in
  `strategy_backtest_adapter` / `strategy_candidate_producer` / `strategy_selector` /
  `rotation_runner` -- those count seed configs in a bake-off and are a different variable.
- `meta_dsr.py` is not wired up here (dead code with a stand-in formula; §1h).

### 4.5 Tests -- `backend/tests/test_phase_82_25_trial_count_reset.py`

Fixture shape copied from `test_phase_82_22_optimizer_best_provenance.py:32-49`
(`_optimizer(**attrs)` via `__new__`, `_saved()` via a monkeypatched `_BEST_PARAMS_PATH`).

- **C1:** a warm-start source recording `num_trials=7` yields a cumulative count, not 1.
- **C2:** DSR after a warm start is deflated against the cumulative count -- asserted by
  computing the SAME run at `num_trials=1` and showing the cumulative figure is strictly
  lower (direction verified against `analytics.compute_deflated_sharpe`, not assumed).
- **C3:** a source with no `num_trials` sets `prior_trials_known=False` and does NOT
  silently become 1; the flag is persisted.
- **C4:** a test that fails against the reset-to-1 behaviour (proved by mutation, not
  asserted).
- **Go-live boundary:** a persisted `dsr` is not mutated by a warm start.
- **Blast-radius guard:** the unrelated same-named `num_trials` in the adapter is
  untouched.

---

## 5. Files expected to change

`backend/backtest/quant_optimizer.py`,
`backend/tests/test_phase_82_25_trial_count_reset.py` (NEW), `.claude/masterplan.json`.

---

## 6. References

`handoff/current/research_brief_82.25.md`; Bailey & Lopez de Prado, *The Deflated Sharpe
Ratio* (JPM 2014) incl. Appendix 3 on `M >= N`; Lopez de Prado & Lewis 2018;
*Pseudo-Mathematics and Financial Charlatanism*; internal precedent
`backend/tests/test_phase_82_22_optimizer_best_provenance.py:32-49` and the dead-but-
correct doctrine in `backend/autoresearch/meta_dsr.py`.
