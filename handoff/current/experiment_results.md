# Experiment Results — phase-73.0: D1 deep frontier study

Date: 2026-07-18. Session: Fable 5 + ultracode, effort MAX; RESEARCH + DESIGN ONLY ($0 metered, all on the Max rail).

## What was built

1. **Research gate** (`wf_f3f3c7ec-0fc`, opus/max, tier=complex): gate_passed=true; 5 anchor papers read IN FULL (The New Quant, agent survey, Profit Mirage, FinMem via ar5iv, LLM-judge overconfidence); all 18 non-anchor syllabus refs validated with access routes; 3 corrections (Look-Ahead-Bench canonical id 2601.13770; PiT-RAG true title; AlphaAgent IR/venue flagged unverified); 5-reader reading plan returned. Brief: `research_brief_73.0.md`.
2. **GENERATE fan-out** (`wf_70d10c32-d28`, 6/6 agents, ~546k tokens): five dimension-readers each read their sources IN FULL (with per-paper section/table citations) and returned ADOPT/ADAPT/REJECT/DEFER verdicts; a consistency synthesizer at effort max verified chain ordering against code, resolved 3 contradiction checks, flagged all abstract-only reads for confidence downgrade, and confirmed zero scale-constraint violations among adopts.
3. **`handoff/current/frontier_map_73.md` published (40,354 chars)** — 10 final dimension verdicts transcribed VERBATIM from the verified synthesis JSON (no paraphrase), plus: full chain-ordering verdict, per-reader read-in-full/not-fully-read coverage lists, consistency findings, and baseline corrections carried forward to the 73.1-73.5 contracts.

## Headline verdicts

- **Chain CONFIRMED** (#3→#2→#1→#4) with one decisive refinement: the baseline's #3 F-grade premise is **STALE** — the AFML purge + embargo were already built in phase-69.2 (verified at `backtest_engine.py:570-582,:662,:428-430` + `walk_forward.py:36,61`); residual #3 work is the **LLM-side** guards (post-cutoff eval, counterfactual audit on the live signal path — the historical backtest is quant-only GBM) + CPCV wiring as a complement. #4 (net-of-cost DSR) is parallelizable early (~10-line change feeding existing `compute_dsr`, gate.py untouched).
- **#2 keystone verified**: the exact outcome_tracker crash mechanism (:47/:50 naive-vs-aware, sibling guards :101-110, unguarded call :137) — a ~2-line fix unlocks the already-coded reflection path; FinMem decay as O(N) BM25 re-rank, embeddings REJECTED.
- **#1**: mechanism ADAPT (debate-derived vote-share via our existing bull/bear stack as the deliberation-calibrator; REJECT logprob — unavailable; REJECT isotonic at small samples — 2-3 bucket shrinkage on Wilson lower bound; bounded scalar at the sizing seam, caps stay downstream); live sizing DEFERRED until ~100-150 clean closed trades.
- **#5**: champion-bridge ADOPT build-dark (verified: best_params reaches only the heartbeat, never decide_trades); training-based self-evolution DEFERRED.
- **#6/#7 pilots**: DEFER behind #2/#3 respectively (PiT-RAG mechanism is a near-perfect future fit; factor-mining against a leaky scorer = industrialized overfitting; heavy frameworks REJECTED on cost/infra); smallest honest #7 step = an OOS rank-IC/ICIR gate, not a miner.
- **#8/#9/#10**: KEEP — no rebuilds; two wiring upgrades feed the existing gate; PBO 0.20 kept (stricter than charter 0.5) pending the 73.4 documentation.

## Verbatim verification output

```
$ bash -c 'test -f handoff/current/frontier_map_73.md && grep -Eqi "adopt" handoff/current/frontier_map_73.md && grep -Eqi "leakage" handoff/current/frontier_map_73.md && grep -Eqi "calibrat" handoff/current/frontier_map_73.md'
73.0 VERIFICATION COMMAND EXIT: 0 (PASS)
```

## Reading-coverage honesty (criterion 3)

Unreachable/partial sources are listed per-reader in the map with what was attempted: Look-Ahead-Bench body 404 (abstract+README only → Q4 confidence downgraded), CPCV-comparison 403 on both hosts (snippet + risklab.ai + AFML Ch.12), Amazon PDF unparseable locally, Man Group Bloomberg paywalled (snippet-level → #10 caveated). The three baseline-flagged 2026 arXiv IDs were verified genuine; AlphaAgent's "IR 1.5" confirmed in body as Table 2 IR=1.488.

## Scope honesty

No product code, no .env, no flags, no optimizer runs. The map corrects our own baseline (stale :587 premise) rather than defending it. All verdict paragraphs are verbatim from the synthesis return; Main authored only headers/assembly.
