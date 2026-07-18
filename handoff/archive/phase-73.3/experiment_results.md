# Experiment Results — phase-73.3: D2c calibrated-sizing design

Date: 2026-07-18. Session: Fable 5 + ultracode, effort MAX; RESEARCH + DESIGN ONLY ($0 metered).

## What was built

1. **Research gate** (`wf_95688663-b4a`, opus/max, tier=moderate): gate_passed=true; 5 sources read in full (deliberation-calibration mechanics, overconfidence metrics, autorater calibration, small-sample binomial intervals canon, Kelly-under-estimation-error) plus the adversarial 2508.18868 recency qualifier honestly weighed; 10 internal files. Returned five `design_inputs` + verbatim `sample_size_math`. Brief: `research_brief_73.3.md`.
2. **`design_pack_73/c_calibrated_sizing.md` finalized (14,698 chars)** — sample-size math verbatim (Wilson lower bound 22-30pp under point estimate at today's N=10-15/bucket → scalar correctly ~1.0; calibrated-beats-uniform at ~40-50/bucket ≈ 100-150 total, coinciding with go-live TRADES_THRESHOLD=100; self-deferring empirical-Bayes design); five component specs verbatim (vote-share elicitation over already-persisted columns — zero new agents, zero migration, retro-backfillable; 2-bucket shrinkage map; the exact `portfolio_manager.py:388-392` seam with caps-downstream/REJECT-upstream non-bypassability; A/B promotion evidence; pairs pipeline starting NOW); decisions of record + one **surfaced operator decision** (s_max: defensive 1.0 vs 1.25-1.5).
3. **Executor build steps appended pending**: 73.3.1 pairs pipeline + calibration module [sonnet-4.6/high] (runs from day one, zero sizing effect), 73.3.2 sizing-seam scalar flag-gated dark + A/B harness [sonnet-4.6/high] (activation double-gated: operator token AND the ~100-150-trip data bar) — each with an immutable live_check.
4. Deferral discipline encoded end-to-end: three independent safeties (flag default OFF; empirical-Bayes self-defer at small N even flag-ON; lower-bound sizing so noise only shrinks bets).

## Verbatim verification output

```
$ bash -c 'test -f handoff/current/design_pack_73/c_calibrated_sizing.md && grep -Eqi "isotonic|bucket|calibrat" handoff/current/design_pack_73/c_calibrated_sizing.md'
73.3 VERIFICATION COMMAND EXIT: 0 (PASS)
```

## File list

- `handoff/current/contract.md` (73.3; gate → contract → GENERATE; write-first skeleton disclosed, precedented)
- `handoff/current/research_brief_73.3.md`
- `handoff/current/design_pack_73/c_calibrated_sizing.md`
- `.claude/masterplan.json` (73.3 in-progress; 73.3.1-73.3.2 appended pending)

## Scope honesty

No product code, no .env, no flags, no optimizer runs, no metered spend. The design claims no ECE gains from the multi-backbone paper (single-backbone caveat explicit), flags pre-72.0.1 backfill rows as degraded, and surfaces rather than silently decides the s_max question.
