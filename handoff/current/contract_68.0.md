# Contract -- 68.0 Deep-research + design pack (execution cutover)

Step: masterplan phase-68 / 68.0 (P0, calendar-gated before 2026-07-12). Research
gate: PASSED (complex tier; research_brief_68.0.md -- 13 read in full, 48 URLs,
recency scan, 19 internal files, all four topics + DON'T-RE-FIX verified).

## Research-gate summary (load-bearing facts)

- PREMISE OVERTURNED (68.5 impact, operator decision pending -- see Risk): the AMD/MU
  "price defect" does not exist. 2026-07-09 closes: AMD $546.72, MU $991.64
  (yfinance re-fetch, date-pinned via MU's corroborated 07-08 close $948.80; MU ATH
  $1213.37 on 2026-06-25; AMD 52wk high $584.73). live_check_66.2.md:402's "real
  ~$150/~$110" was a stale world-knowledge anchor (AMD 52wk LOW is $137.59). The
  book is CORRECT; avg_entry does NOT poison stop math. DESC phantom ALREADY fixed
  (66.2 hotfix 9262ed36, drawdown_alarm.py:65-108 + regression tests). The REAL
  latent holes 68.5's sanity gate should target: unguarded SELL fill price with
  hash-synthetic fallback; tolerance gate fails open without price_at_analysis;
  independent quote must not be yfinance-vs-yfinance.
- Config: execution_router resolves mode PER-CONSTRUCTOR via os.getenv
  (execution_router.py:65-71, :268-269; "import time" docstring is stale); plist
  carries only 4 env keys today (no EXECUTION_BACKEND/ALPACA); pydantic env_file is
  model-only and settings has NO execution_backend field.
- Shadow: isolation structure already correct (bq_sim authoritative; alpaca leg
  try/except-swallowed) BUT the paired fill is DISCARDED and paper_trades has NO
  source column (dynamic INSERT rejects unknown keys) -> new table via migrations.
- Idempotency: client_order_id <=128 chars; duplicate-vs-ACTIVE -> 422; recovery via
  get_order_by_client_id. CUTOVER TRAP: execution_router.py:239-244 +
  paper_trader.py:260 book rejected/canceled orders at reference price -- must be
  fixed before 68.3 cutover.
- Guards: PKLIVE-prefix folklore is near-vacuous; real triple-enforcement =
  paper=True pin + key-shape rejection + no-live-construction path. Flatten MUST use
  close_all_positions(cancel_orders=True), NEVER account reset (invalidates keys).
  PDT applies on paper <$25k. alpaca-py 0.43.2 adequate.
- DON'T-RE-FIX verified standing (alerting imports autonomous_loop.py:233/:764/:971
  /:1005 + guard test; cc-rail guard :209-230).
- Gap disclosed: backend/.env is agent-locked; the operator verifies which ALPACA
  keys it holds (design doc carries this as an operator-verify item).

## Hypothesis (falsifiable)

A design pack grounded in the traced code paths (constructor-time env read, isolated
shadow leg, client_order_id idempotency, env-flip rollback, strengthened paper-only
guards) is sufficient for 68.1-68.3 to be implemented without touching risk surfaces
-- testable by the step's verification command plus Q/A review of the doc against
the brief's file:line evidence.

## Success criteria (verbatim from .claude/masterplan.json 68.0 -- IMMUTABLE)

1. "research_brief_68.0.md exists with an honest gate envelope and covers all four
   topics: env propagation into the launchd-started backend process, shadow-mode
   TRUE order semantics traced through the actual code path (file:line), alpaca-py
   paper order lifecycle + reconciliation primitives, and an AMD/MU price-defect
   hypothesis tree ranked by evidence"
2. "design_execution_cutover_68.md exists and specifies: config precedence (env >
   .env > default) with the exact propagation mechanism to the launchd process;
   shadow-mode isolation (shadow NEVER mutates bq_sim position state); order-id
   idempotency; rollback = single env flip back to bq_sim; existing PKLIVE/paper-only
   guards kept and named"
3. "No production code changed by this step (research + design only; git diff shows
   handoff/doc artifacts only)"
4. "Fresh Q/A PASS on the pack"

Criterion-1 note: the hypothesis tree's evidence-ranked CONCLUSION is that the top
hypothesis ("defect exists") is FALSE -- a valid, indeed the most valuable, outcome
of a hypothesis tree. Criterion-2 note: "PKLIVE...guards kept and named" is satisfied
by naming the existing guards AND documenting that the PKLIVE-prefix assumption is
folklore, replaced with stronger enforcement -- keeping guards does not mean keeping
a false assumption about how they work.

## Design (files)

1. NEW handoff/current/design_execution_cutover_68.md -- authored by Main (Fable)
   from the brief's "Design inputs" section; five mandated sections + sequencing
   consequences (68.5 premise status, cutover trap as a 68.3 prereq).
2. This contract + experiment_results_68.0.md + live_check_68.0.md.
3. NO production code (criterion 3).

## Anti-patterns guarded

- Criteria immutability: 68.5's now-unsatisfiable criteria are NOT edited by this
  step; the disposition is surfaced to the operator (their authority).
- Fabricated correction risk: the design forbids "correcting" the AMD/MU rows (the
  book is right; a correction would corrupt it).
- Stale-anchor class: the design's sanity gate uses an independent NON-yfinance
  quote source (the very failure mode that produced the false premise).

## Out of scope

All implementation (68.1+); any masterplan criteria edit; any BQ row mutation; the
operator's 68.5 disposition decision.

## Risk

- 68.5 as written cannot PASS (criteria 1-2 unsatisfiable, 4 pre-satisfied); if the
  operator doesn't re-scope it, the step will burn its retries -- surfaced
  immediately after this step closes.
- Real prices mean the ~$1560 book is ~3 shares total (fractional) -- fine for paper;
  PDT note stands.
