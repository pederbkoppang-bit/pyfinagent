# Contract -- step 86.59

**Step:** 86.59 -- the stock picker analyses the SAME 4-6 names every day
because its score is built only from slow trailing returns. **P1, money path.**

## Research-gate summary (what the gate CHANGED about the plan)

Gate **PASSED** on the re-run (`research_brief_86.59_rerun.md`, 38,007 chars;
8 sources read in full this session, 54 URLs collected, envelope COMPLETE).
The first attempt FAILED the gate on an over-claim (30 URLs claimed, 13 present)
and PLAN was correctly not entered; the re-run supplied the snippet-only table.

**The single most important finding is a REFRAME, and the contract must commit
to which thing it is buying.** "Re-selects the same 4-6 names daily" is **not
one defect**. It is three:

- **(a) a weighting-correctness bug** -- the declared weights are not the
  effective weights;
- **(b) a missing-orthogonal-signal gap** -- there is no fast signal at all;
- **(c) a slate-width choice** -- `paper_analyze_top_n = 5`.

The literature endorses fixing (a) and (b) and **explicitly warns against
treating (c) as a defect**: a slow predictor *correctly* produces low turnover,
and below roughly 50% one-sided monthly turnover most strategies survive costs
while few above it do. **This contract buys (a) and instruments for (b). It
does NOT buy (c), and it does not chase churn.**

**(a) is measured and specific.** `screener.py:301-305`:
`score = mom_1m*0.40 + mom_3m*0.35 + mom_6m*0.25`, applied to **RAW trailing
returns whose dispersions differ**. There is NO cross-sectional standardisation
on the live path: `_zscore` exists at `:532` but is reachable only through the
dark multidim path (`:438-439`, `:607-610`), verified end-to-end. Six-month
returns carry roughly 2.4x the dispersion of one-month, so **the term with the
smallest declared weight contributes the most ranking variance**. Any reweight
that does not standardise first is tuning a knob that is not connected to the
outcome. This is framed as **correctness**, not as a turnover fix.

**Three mitigations already exist and all default OFF** --
`sector_neutral_momentum_enabled`, `paper_soft_sector_diversity_enabled`,
`paper_min_k_sectors_analyzed`. Criterion 4 requires measuring what each does
BEFORE new code is written, so the step does not rebuild an existing mitigation.
The brief prefers `paper_min_k_sectors_analyzed` (changes only which names reach
the deep-analyse slice) over `paper_soft_sector_diversity_w` (overwrites
`composite_score`), and records this project's own **-0.166 replay** against the
hard sector-neutral version.

**There is no banding or hysteresis today** -- `screener.py:491-492` re-sorts
and re-slices unconditionally every run, with no incumbent bonus. If turnover
rises, banding is the evidenced fix, not more churn.

**Scope reference correction:** the file is `backend/tools/screener.py`.
`backend/services/screener.py` does not exist, and neither does
`candidate_picker.py`.

**Evidence gaps carried into the contract rather than hidden:** the two
quantified cost/turnover sources are a master's thesis (tier 3) and a
not-yet-peer-reviewed working paper; Blitz-Huij-Martens and Gutierrez-Kelley
were abstract-only (paywalled); no `Chaves` international-idiosyncratic-momentum
paper was located, reported as an absence rather than padded. The
residual-momentum recommendation therefore rests on **indicative**, not
peer-reviewed, magnitudes.

## Hypothesis

The ranking is frozen because every term is a trailing return over 21/63/126
trading days, so one extra day moves the composite by ~1/21, 1/63 and 1/126 of
its own window -- and because the composite is computed on unstandardised
inputs, the effective weights are not the declared ones. Correcting the
standardisation is a small, auditable, *correctness* change whose effect on
candidate selection can be measured out-of-sample against the existing gates.

## Immutable success criteria (copied verbatim from `.claude/masterplan.json`)

1. the day-over-day rank stability of the CURRENT score is MEASURED, not argued: recompute rank_candidates over consecutive historical sessions from stored price data and report the rank correlation and the top-10 turnover per day, with the command that produced them
2. any new or reweighted term is justified by an out-of-sample test, not by plausibility: show the change alters candidate selection on held-out dates AND report what it does to the existing gates (DSR, PBO) rather than only to turnover
3. candidate DIVERSITY is reported as a measured distribution before and after -- distinct tickers over N cycles and sector concentration -- and the N used is stated
4. the three existing dark flags are evaluated as a baseline BEFORE new code is written: measure what sector_neutral_momentum_enabled, paper_soft_sector_diversity_enabled and paper_min_k_sectors_analyzed each do to candidate turnover, so the step does not rebuild a mitigation that already exists
5. NO flag is promoted and NO .env is written by this step; operator-gated changes are recorded as numbered asks
6. flag-OFF parity is proven: with every new behaviour disabled the candidate list is byte-identical to today's, demonstrated against an oracle rather than by two passing examples
7. mutation-test every new guard: revert it and show the check goes red, with the control observed GREEN first and a byte-identical restore

**Immutable verification command:**
`bash -c 'source .venv/bin/activate && python -c "import ast; ast.parse(open(\"backend/tools/screener.py\").read()); print(\"parses\")"'`

**Immutable live_check:** `live_check_86.59.md` with the per-cycle
analysed-ticker list before and after, the measured rank-stability figure, and
the sector distribution.

## Plan

**P1 -- criterion 1, measured on stored prices.** Recompute `rank_candidates`
over consecutive historical sessions from stored price data; report Spearman
rank correlation and top-10 one-sided turnover per day, with the command. This
is the number that decides whether the composite is as frozen as claimed --
**and it is allowed to refute the step's premise.** Report it either way.

**P2 -- criterion 4 BEFORE any new code.** Measure each of the three dark flags'
effect on candidate turnover, off the live path, flags forced in-process (never
via `.env`). Establishes whether an existing mitigation already does the job.

**P3 -- the correctness fix (a).** Standardise cross-sectionally at
`screener.py:301-305` using the existing `_zscore`, behind a new default-OFF
flag. Framed and tested as correctness: assert that declared weights become the
effective weights, i.e. each term's contribution to ranking variance matches its
declared weight within a stated tolerance.

**P4 -- criterion 2, out-of-sample.** Show the change alters candidate selection
on held-out dates AND report DSR/PBO against the unchanged gates
(`min_dsr=0.95, max_pbo=0.20, min_pbo_trials=10`). **Report one-sided turnover
beside them**, treating >50% monthly as requiring explicit justification. **No
gate is loosened.**

**P5 -- criterion 3, diversity as a measured distribution** -- distinct tickers
over N cycles and sector concentration, before and after, with **N stated**.

**P6 -- criterion 6, parity against an ORACLE.** With the new flag OFF the
candidate list must be byte-identical to today's, demonstrated against a
recorded oracle rather than two passing examples.

**P7 -- criterion 7, mutations** with the control observed GREEN first and a
byte-identical restore; each cell scored, UNSCORABLE if its control was not
green.

## Scope honesty -- what this step does NOT do

- **It does not widen the slate** (`paper_analyze_top_n`). That is choice (c),
  and the literature warns against calling it a defect. Recorded as a numbered
  operator ask if wanted.
- **It does not promote any of the three dark flags** and writes no `.env`;
  it measures them off the live path.
- **It does not implement residual momentum.** That is finding (b) and needs
  factor returns plus a beta window `screen_universe()` does not produce; it is
  queued, not built here, and its supporting magnitudes are tier-3 sources.
- **It does not add hysteresis or an incumbent bonus.** Hysteresis remains
  banned by standing project decision; banding is recorded as the evidenced
  option if turnover rises.
- **It does not claim the trade drought as its consequence.** 86.47 owns the
  drought cause; two prior steps were filed on drought theories their own
  research gates refuted.
- **86.60 is the sibling** and owns the entry-path architecture; this step owns
  the score. Neither may claim the other's fix.

## References

`research_brief_86.59_rerun.md` (the reframe, the standardisation finding, the
turnover bounds, the honest evidence gaps); `research_brief_86.59_v2.md` and
`research_brief_86.59.md` (prior passes); `q1_binding_constraint_86.59.md`;
`contract_86.60.md` (the sibling boundary); `research_brief_86.60.md`
(finding I-5, the entry-path/score-adjustment split).
