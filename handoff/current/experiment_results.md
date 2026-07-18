# Experiment Results — phase-73.4: D2d cost-integrated promotion design

Date: 2026-07-18. Session: Fable 5 + ultracode, effort MAX; RESEARCH + DESIGN ONLY ($0 metered).

## What was built

1. **Research gate** (`wf_f5b30af7-e25`, opus/max, tier=moderate): gate_passed=true; 5 sources read in full (field reporting standard, token budgeting, DSR mathematics, transaction-cost modeling, GIPS net-of-fees) + 2026 recency (arXiv:2607.10286 — the field formalizes our exact objective with lower-fidelity user-configured costs); 11 internal files incl. the full PBO-threshold census. Brief: `research_brief_73.4.md`.
2. **`design_pack_73/d_cost_promotion.md` finalized (9,536 chars)** — three component specs verbatim: net-of-cost DSR as a return-series transform with the **two-seam nuance** (the GBM promotion backtest is already tx-net with structurally-zero token cost — netting token cost there would fabricate; Seam B live series is where it is real), the gauge-safe token-cost derivations (never SUM `session_cost_usd`), gross+net transition logging; the per-decision cost-per-bp diagnostic (field-required, no double-count, None-on-zero); the PBO census resolving the 'discrepancy' as **two correct nested gates** (0.5 veto cap / 0.20 promotion bar) with a docs-only resolution and a recommend-only charter-memory annotation.
3. **Executor build steps appended pending**: 73.4.1 net-DSR + cost-per-bp reporting [sonnet-4.6/high], 73.4.2 PBO nested-gate documentation [sonnet-4.6/high, docs-only] — each with an immutable live_check (the gate-run artifact showing dsr_gross vs dsr_net side-by-side satisfies the criteria's 'gate run whose output shows the cost-charged objective').
4. No immutable gate weakened anywhere: gate.py and risk_server.py thresholds untouched by design; the objective change happens strictly in the series fed to the existing function.

## Verbatim verification output

```
$ bash -c 'test -f handoff/current/design_pack_73/d_cost_promotion.md && grep -Eqi "token|cost" handoff/current/design_pack_73/d_cost_promotion.md && grep -Eqi "PBO" handoff/current/design_pack_73/d_cost_promotion.md'
73.4 VERIFICATION COMMAND EXIT: 0 (PASS)
```

## File list

- `handoff/current/contract.md` (73.4; gate → contract → GENERATE; write-first skeleton disclosed, precedented)
- `handoff/current/research_brief_73.4.md`
- `handoff/current/design_pack_73/d_cost_promotion.md`
- `.claude/masterplan.json` (73.4 in-progress; 73.4.1-73.4.2 appended pending)

## Scope honesty

No product code, no .env, no flags, no optimizer runs, no metered spend. The design prevents two subtle fabrications (token cost on the zero-token seam; double-counting cost in objective + diagnostic) rather than committing them, and routes the charter-memory annotation through the operator instead of editing an operator-owned file.
