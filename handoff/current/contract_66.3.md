# Contract -- 66.3 Cost-truth restoration (goal-phase66-reactivation)

Step: 66.3 | Cycle 72 | 2026-07-07 | Operator present
Sequencing: operator-authorized early start (AskUserQuestion ~08:50 UTC; masterplan
edge relaxed 66.1 -> 66.0 with sequencing_note). The 66.1->66.2 P0 chain untouched.

## Research-gate summary

research_brief_66.3.md (tier moderate, gate_passed: true; 6 read-in-full / 75 URLs /
recency scan / 9 internal files). Load-bearing findings:
- THE WRITER, PINNED (criterion 1): there is NO flat-$0.50 constant.
  `session_cost_usd` is a CUMULATIVE PER-CYCLE NOMINAL GAUGE: log_llm_call lazy-fills
  the omitted kwarg from the autonomous_loop module accumulator
  (api_call_log.py:245-254 <- autonomous_loop.py:119-121), which grows by
  `analysis.get("total_cost_usd", 0.1)` per analysis (:893-895) and is never reset
  until cycle end (:1401-1403). Every row -- including FAILED 0-token cc_rail rows --
  is stamped with the running total, producing the observed 0.0 -> 0.5 staircase.
- THE SENTINEL BUG: sentinel.sh:67-71 SUMs the gauge across rows (a cumulative gauge
  summed row-wise over-counts roughly quadratically); the comment at :63-66 documents
  the false premise ("flat-fee rows log 0 by design ... a plain SUM IS the metered
  figure"). 06-18: rail rows account for $42.20 of the $43.00 "breach"; true nominal
  gauge peak was ~$1. The $8 baseline is the :40 constant, not flag_baseline.json.
- claude-code-proxy EXONERATED (no BQ code). No schema migration required for the
  chosen fix shape.
- DEEPER IMPLICATION: no true per-call cost column exists -- even "Gemini spend"
  figures computed by summing session_cost_usd are gauge artifacts. "Dollars actually
  billed" must be TOKEN-DERIVED (input_tok/output_tok x pinned price table) over
  metered providers, with flat-fee rail rows excluded by agent/provider filter (the
  criterion's sanctioned filtering option; documented).
- Recency flag: Claude Code's 2026-06-15 change routes headless usage through the
  Agent SDK credit -- the rail is credit-funded, still NOT per-token metered API;
  document in the definition of "metered".
- FinOps canon: never report estimated cost as billed; billed-vs-amortized separation
  (FOCUS spec); failed-call cost must be 0 when nothing was billed.

## Hypothesis

Two small, verifiable changes restore cost truth: (a) the shared writer stops
stamping the gauge on failed 0-token rows (they log 0 -- criterion 1 literal), and
(b) the sentinel's metered figure becomes token-derived dollars over metered
providers (rail excluded, unpriced models fail-visible via warnings), which makes
the 06-18 replay drop from $43.00 to <$1 and makes a live-day hand audit match by
construction plus an independent check.

## Immutable success criteria (verbatim from .claude/masterplan.json phase-66/66.3)

1. "The writer of nonzero session_cost_usd on failed cc_rail calls is pinned to
   file:line in the results, and fixed: auth/connection failures with 0 tokens log
   cost 0"
2. "The sentinel metered figure excludes flat-fee (Max subscription) rail rows,
   either by billing-class tagging at write time or provider/agent filtering, with
   the choice documented; consecutive-failure counts surface as a first-class
   sentinel warning"
3. "Replay of the 2026-06-18 window through the new logic yields a metered figure
   that does NOT breach the $8 baseline, AND one live day's sentinel figure matches
   a hand audit of actually-billed spend (both pastes in live_check)"

Verification command (immutable):
source .venv/bin/activate && python -m pytest backend/tests/test_phase_66_3_cost_truth.py -q

live_check: live_check_66.3.md with writer file:line, replay output, and the
live-day hand-audit comparison.

## Plan

1. Writer fix (api_call_log.log_llm_call): when ok=False AND input_tok==0 AND
   output_tok==0 AND session_cost_usd is None -> stamp 0.0 (never the gauge); guard
   protects every agent's failed calls, not just cc_rail. Docstring documents the
   gauge semantics explicitly.
2. New `scripts/away_ops/metered_spend.py` (importable module + CLI --date):
   token-derived metered dollars = sum over rows of metered providers
   (provider='gemini' and any non-rail 'anthropic'/'openai' API rows) of
   tokens x pinned price table (prices pinned with source comments; models observed
   in the last 30d priced; unknown metered models -> counted 0 + fail-visible
   warning naming them). Rail exclusion: agent LIKE 'cc_rail%' OR the claude-code
   provider (flat-fee/SDK-credit -- documented choice per criterion 2). Also emits
   rail_failures_today (count of ok=false rail rows) for the sentinel warning.
3. sentinel.sh: metered block delegates to metered_spend.py; adds
   `rail_failures_today` field + warnings[] entry when >= 20 (mirrors the 66.1
   breaker threshold); false-premise comment corrected; SENTINEL_TEST_* overrides
   preserved; replay via the module CLI (no sentinel structural change needed).
4. Tests (immutable command): writer-guard (failed 0-token -> 0.0; success path
   still gauge-fills; explicit kwarg respected), price-derivation unit tests
   (fixture rows -> expected dollars; unknown-model warning; rail exclusion),
   BQ-free (all monkeypatched/fixture-driven).
5. Evidence: replay `--date 2026-06-18` (expect <$1, no breach vs $8); live-day
   figure vs hand audit (manual token-math paste + sanity vs GCP billing export if
   fresh); full sentinel.sh run showing the new fields.
6. Fresh Q/A -> log Cycle 72 -> flip on PASS.

## Scope boundaries

No schema migration; no changes to the gauge accumulator itself (its semantics are
documented, not redefined -- the 58.x/62.x burn-audit consumers keep working); no
trading code; sentinel churn/kill-switch/flag gates untouched; no backfill of
historical rows (history stays as-written; the replay proves the new READ logic).

## References

research_brief_66.3.md; api_call_log.py:245-254; autonomous_loop.py:119-121/:893-895/
:1401-1403; sentinel.sh:40/:63-71; FOCUS spec billed-cost; Langfuse/Helicone cost
docs; Claude Code 2026-06-15 Agent-SDK credit change.
