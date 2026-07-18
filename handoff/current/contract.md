# Contract — phase-73.4: D2d cost-integrated promotion design

**Step id:** 73.4 (phase-73, depends_on 73.3 = done/PASS @9569fdc2)
**Session role:** Fable 5 + ultracode, effort MAX; RESEARCH + DESIGN ONLY. No product code, no .env, no flags, no optimizer runs, $0 metered.

## Research-gate summary (gate_passed: true)

Researcher via structured-output Workflow `wf_f5b30af7-e25` (opus/max, tier=moderate): 5 sources read in full (The New Quant §7.5/§7.10 at field-standard depth; QuantAgent token budgeting; Bailey/LdP DSR mathematics; transaction-cost modeling; GIPS net-of-fees), 27 URLs, recency scan (2026 arXiv:2607.10286 formalizes exactly this net-of-LLM-cost objective — with user-configured costs, vs our MEASURED tokens), 11 internal files. Brief: `research_brief_73.4.md`. Three structured `design_inputs` transcribed verbatim.

Load-bearing findings:
1. **Return-series transform, never a penalty**: PSR/DSR depend on realized skew/kurtosis, so costs must flow through the distribution; the N-trials deflation term is untouched → `gate.py` byte-identical.
2. **Two-seam nuance (prevents a fabricated cost)**: the quant-only GBM promotion backtest is ALREADY transaction-cost-net with structurally-zero token cost (Seam A — only an optional slippage haircut applies since bq_sim fills at close with no slippage model); token-cost netting is real only on the LIVE realized series (Seam B).
3. **Gauge-trap codified**: three safe per-cycle token-cost derivations (MAX-per-cycle gauge; token×MODEL_PRICING recompute; SUM of api_call_log's real per-call cost column) — never SUM(session_cost_usd).
4. **Cost-per-bp is the field-required diagnostic** reported ALONGSIDE the objective (no double-count; None-on-zero guard mirroring /efficiency).
5. **The PBO 'discrepancy' dissolves**: full census found exactly two thresholds — 0.5 per-candidate VETO cap (risk_server.py:28, = the charter floor) and 0.20 promotion bar (gate.py:21) — CORRECT nested gates on different objects; resolution = documentation (ARCHITECTURE.md primary, backend-backtest.md cross-ref; the operator-owned charter memory gets a RECOMMEND-ONLY annotation suggestion).

## Hypothesis

Promoting on a net-of-cost DSR computed on the correct seam, with the field-standard cost-per-bp diagnostic beside it and the two PBO gates documented as nested policy, makes "clears ALL costs incl. tokens" mechanically true without touching any immutable gate — the cheapest north-star alignment in the phase.

## Immutable success criteria (verbatim from .claude/masterplan.json step 73.4)

- "d_cost_promotion.md specifies the net-of-cost objective (which costs, measured where, charged how) and its integration at gate.py/perf_metrics without weakening any immutable gate"
- "The PBO 0.20-vs-0.5 discrepancy is resolved in writing (which value is intended, why, where documented) without editing any immutable criterion"
- "Executor-tagged build steps appended pending with live_checks (a gate run whose output shows the cost-charged objective); no code edited this session"

verification.command: `bash -c 'test -f handoff/current/design_pack_73/d_cost_promotion.md && grep -Eqi "token|cost" handoff/current/design_pack_73/d_cost_promotion.md && grep -Eqi "PBO" handoff/current/design_pack_73/d_cost_promotion.md'`

## Plan

1. GENERATE: design doc finalized verbatim from the gate (done, 9,536 chars); append executor build steps 73.4.1 (net-DSR + cost-per-bp reporting) and 73.4.2 (PBO policy documentation, docs-only) — pending, tagged, immutable live_checks.
2. `experiment_results.md` verbatim output → qa-verdict Workflow → transcribe → LOG (Cycle 122) → flip 73.4 done.

## References

- `handoff/current/research_brief_73.4.md`; `frontier_map_73.md` (#4 + #9 verdicts)
- arXiv 2510.05533 §7.5/§7.10; 2402.03755; 2607.10286; Bailey/LdP DSR; GIPS net-of-fees; BSIC slippage haircut
