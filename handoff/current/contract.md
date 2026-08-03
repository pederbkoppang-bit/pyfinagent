# Contract -- step 82.4

**Step id:** 82.4 | **Priority:** P2 | depends_on: ['82.1', '82.3']
PLAN phase. **Written BEFORE the 82.3 numbers exist -- deliberately.**

## Research gate

`handoff/current/research_brief_82.4.md` -- **gate_passed: true**
(6 sources read in full, 29 URLs, recency scan, 8 internal files).

## THE RANKING PROCEDURE, PRE-REGISTERED

The 82.3 backtests are RUNNING as this is written (pass A launched
2026-08-03T18:19:28Z). No result exists yet. The ranking rule and the tie-break
ORDER are fixed here, before any number is visible, because a rule chosen after
seeing the results is not a rule -- it is a rationalisation.

**Stage 1 -- GATES (binary, un-tradeable).** `DSR >= 0.95`, `PBO <= 0.5`,
net-of-cost return `> 0`. A strategy failing any gate is REPORTED AS FAILED and
is not ranked. Gates are not weighted against each other.

**Stage 2 -- PARETO frontier** over (net-of-cost return, PBO, turnover) among
gate-passers. Dominated strategies are listed as dominated, not scored.

**Stage 3 -- LEXICOGRAPHIC tie-break, in this declared order:**
`PBO (lower better)` -> `net-of-cost return (higher)` -> `turnover (lower)`.

**NO weighted composite.** arXiv:2508.00129 documents rank reversal and
transitivity violation as fundamental to weighted MCDA, and a composite would
hide the precise DSR-vs-PBO conflict the pack exists to expose. The repo's own
`rotation_log.jsonl` already encodes gate-then-rank
(`reason: "no_candidate_passed_gate"`, `ranked: []`); this matches that
vocabulary rather than inventing one. Note `candidate_selector.py:95` IS a
weighted composite but it ranks TICKERS in the screen, not strategies -- not a
precedent.

## Mermaid shape (research-determined)

One `flowchart LR` parent, four sibling `subgraph` blocks each `direction TB`,
**ONE** nesting level, and **ZERO cross-subgraph edges**. Per the Mermaid docs:
"If any of a subgraph's nodes are linked to the outside, subgraph direction will
be ignored." So a shared node (e.g. `_sigma_barriers`) must be REPEATED per
column, never linked -- one linking edge flattens all four columns into the
parent's LR direction. Equal node count per column is what aligns the rows;
`classDef` colours only the differing nodes. GENERATE must RENDER-VERIFY
(`npx -y @mermaid-js/mermaid-cli`); documented fallback is four separate blocks,
NOT adding a linking node. Zero mermaid blocks exist in the repo today.

## MANDATORY caveats (Bailey et al. + GIPS, both read in full)

1. **Declare the trial count.** N is NOT 4 and NOT 8 -- it includes the
   phase-82.2 label-design iterations. Under-declaring N inflates DSR.
2. **GIPS prohibits** presenting back-tested results "linked to actual
   performance results". The 82.3 figures must NEVER be spliced into, or placed
   adjacent-as-continuation with, the live paper-trading record (+18.86% /
   Sharpe 3.32). Separate sections, explicit labels.
3. `qarp` is **NOT EVALUABLE** on the full sample: `historical_fundamentals` has
   ZERO rows before 2024-06-30 (81.2% of the window blind) and yfinance serves
   only ~5-7 recent quarters, so no backfill is possible from the current
   source. Queued as 82.21.
4. `reversion_sigma` is purged at `holding_days*1.5 = 135d` against a 15-day
   label horizon (`backtest_engine.py:665` is strategy-blind; queued as 82.19).
   **A win is clean; a LOSS is CONFOUNDED** and must be reported as such.
5. Two passes with DIFFERENT evidential weight must be presented as two
   SEPARATE tables and never merged: full-sample (2018-2025, 3 strategies) and
   short-window (2024-07..2025-12, 4 strategies, ~6 walk-forward windows).

## THE FINDING THAT SHAPES THE RECOMMENDATION

The four columns are **not peers**. Column 1 is the LIVE funnel; columns 2-4 are
backtest LABEL METHODS. A bake-off winner changes NOTHING live while
`paper_analyze_top_n = 5` (`autonomous_loop.py:1035`) stands and no
registry-to-live bridge exists (`autonomous_loop.py:1649` consumes `strategy` as
a heartbeat label only). **So the top queued action the evidence can support is
the BRIDGE (82.6), not a strategy swap** -- whatever wins. The diagram must
label the lanes, mirroring `incumbent_live_strategy_spec.md` section 0.

## Immutable success criteria (verbatim from .claude/masterplan.json)

1. docs/ contains a design pack with one section per strategy (incumbent + 3) and a ranked recommendation with stated ranking criteria
2. every code claim in the design pack carries a file:line reference that resolves in the current tree
3. the design pack records the endogeneity caveat that holding-period-vs-outcome comparisons are tautological because a stopped-out trade has a short holding period by construction
4. one masterplan step is queued per recommended implementation action, each carrying its own verification criteria

**Verification command:** `source .venv/bin/activate && python -m pytest backend/tests/test_phase_82_4_design_pack.py -q`

## Plan

1. Write `docs/strategy/phase82_design_pack.md`: four mermaid columns, the
   pre-registered ranking above, the caveat block, and the two separated result
   tables (left EMPTY until 82.3 lands, then filled from the artifacts).
2. Render-verify the mermaid.
3. Fill the tables from `results/*_phase_82_3_*.json` -- transcribed, not
   retyped.
4. Apply the ranking MECHANICALLY to whatever the numbers say.
5. Queue one masterplan step per recommended action.
6. Fresh Q/A.

## Out of scope

No live-funnel change. No FigJam (operator decision: View seat on Starter = 6
MCP calls/MONTH). No re-running of 82.3.
