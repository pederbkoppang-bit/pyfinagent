# Contract -- step 82.1

**Step id:** 82.1 (phase-82) | **Priority:** P1 | PLAN phase, before GENERATE.

## Research gate

`handoff/current/research_brief_82.1.md` -- **gate_passed: true**
(tier=moderate, audit_class=false, external_sources_read_in_full=6,
snippet_only=24, urls_collected=30, recency_scan_performed=true,
internal_files_inspected=11).

**Main independently re-verified every load-bearing claim. THREE were WRONG and
are NOT carried into the spec:**

| brief claim | measured reality |
|---|---|
| "once **10** positions are held, every buy is skipped" | `paper_max_positions` = **30** (`risk_overrides.get_effective` returns 30; settings default 30). Book holds 1. Not binding. |
| "`paper_max_per_sector = 2` should have bounded semis at 2" | effective value is **5**. Book holds 1. Not binding. |
| "a disarmed kill switch refuses every BUY at `paper_trader.py:275-286`" | REFUTED at `paper_trader.py:175-216`: `_kill_switch_refusal_for_buy` reads `is_paused()` and `baselines_present_in()` and its docstring states it reads baselines "**NEVER `armed`**" -- gating on `armed` is the exact money-path regression phase-36.9's evaluator caught. Live: `paused=false`, `baselines_present=true`. |

Claims that DID verify and are carried: the live book has no take-profit and no
time barrier (`paper_scale_out_enabled=False`) and stops at 8%
(`paper_default_stop_loss_pct`) rather than the optimizer's `sl_pct=12.92`;
concentration is generated at RANKING (`rank_candidates` defaults to
`strategy="momentum"`, screener scores momentum/RSI/SMA only) with all five
diversity levers OFF; overlays RANK rather than veto; and the silent-attrition
defect at `portfolio_manager.py:188-189` (confirmed by Main; queued as 82.14).

Its most valuable contribution is from the recency scan and is NOT a code fact:
the 2025-2026 literature names AI/memory/semis/Korea as **the** crowded trade of
the period, and the book's holdings are that list. So the concentration is the
ranker faithfully expressing a crowded factor rather than a ranker bug -- a
different and worse problem. MSCI (tested 1999-2024) and Resonanz both prescribe
crowding-score caps / dynamic sizing over hard sector-neutrality, agreeing with
this repo's own measured -0.166 Sharpe for hard sector-neutral on a long-only
book (`autonomous_loop.py:588-593`).

## Hypothesis

The live funnel is not a valuation-aware strategy that chose cash. It is a
momentum screener with a hard THROUGHPUT cap. Of ~583 names, `paper_screen_top_n
= 10` then `paper_analyze_top_n = 5` (`autonomous_loop.py:1034`) decide that
five names per cycle receive the 28-agent analysis -- so five are the only names
that can produce a BUY. ~1 BUY/cycle reproduces the 33 lifetime BUYs and the
~1-2 trades/week rate. The same slice, ranked momentum-only with every
diversity lever off, produces the single-sector book. Cash is an ARTEFACT OF
THROUGHPUT, not a view on valuation.

Falsifiable predictions: (a) no kill-switch refusal appears on the BUY path;
(b) the go-live gate cannot block a paper BUY at all; (c) position and sector
caps are slack (1 held vs 30/5).

## Immutable success criteria (verbatim from .claude/masterplan.json)

1. docs/ contains an incumbent strategy specification naming universe, signal, ranking, sizing, entry gates and exit rule, each claim carrying a file:line reference that resolves in the current tree
2. the specification states the measured trade counts (BUY, SELL, open positions) and reconciles them against the operator screenshot's trade count
3. a diagnosis names the binding constraint on turnover and quotes the live query output it rests on verbatim
4. a test asserts every file:line citation in the spec resolves to an existing file with at least that many lines

**Verification command:** `source .venv/bin/activate && python -m pytest backend/tests/test_phase_82_1_incumbent_spec.py -q`

## Plan

1. Write the incumbent spec to `docs/` -- universe, signal, ranking, sizing,
   entry gates, exit rule -- each claim carrying a file:line that resolves.
2. State the measured trade counts and reconcile them to the operator
   screenshot (32 BUY + 32 SELL = the "Trades (64)" tile; the 33rd BUY landed
   after the capture).
3. Name the ONE binding constraint (`paper_analyze_top_n`) and quote the live
   query output it rests on verbatim, including the refutations.
4. Ship the citation-resolver test.
5. Fresh Q/A. NO code changes to the live funnel -- spec and diagnosis only.

## Out of scope

Fixing the throttle or the concentration. Note for the record that raising
`paper_analyze_top_n` costs LLM spend (28 agents per name, against the standing
$0-metered constraint and the $25/day cap) whereas
`paper_min_k_sectors_analyzed > 0` re-allocates the SAME five slots and is free.
They are separable and must not be bundled.
