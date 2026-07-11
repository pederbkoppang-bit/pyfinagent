# Experiment Results — Step 69.0 (P0 design pack, phase-69 audit burn-down)

- **Phase / step**: phase-69 → 69.0
- **Date**: 2026-07-11
- **Type**: DESIGN PACK ONLY (research + design; no production code)

## What was produced

1. **`handoff/current/research_brief_69.0.md`** — research-gate brief, **gate_passed: true**.
   8 external sources read in full (23 URLs collected), recency scan performed, all 4 design topics
   covered, 19 internal code sites re-verified against the register. Includes the DSR worked reference
   (0.9004) independently re-derived, the sign-safe overlay formula + proof, the FX fail-closed
   waterfall, the kill-switch peak-reset triggers, and the purge/embargo rule.
   - **Provenance**: two researcher subagent spawns (Fable, then Opus) each read all 8 sources but
     STALLED on the end-of-session flush and were stopped per CLAUDE.md STALL WATCH (Fable ~14 min
     transcript-idle; Opus ~4.5 min). Sources + DSR example + internal inventory were persisted
     incrementally (write-first partially held). Main (Opus) finalized the synthesis sections + the
     envelope from the already-read sources plus an independent re-derivation of the DSR reference —
     the "Main updates the stalled handoff file" pattern. Every synthesis claim traces to a "Read in
     full" source row. Full note in the brief's "Gate completion note (provenance)" section.

2. **`handoff/current/contract.md`** — step contract: research-gate summary, hypothesis, immutable
   success criteria copied VERBATIM from `.claude/masterplan.json` phase-69 → 69.0, plan, references.

3. **`handoff/current/design_audit_burndown_69.md`** — the design pack. Six sections, each element
   naming its exact file:line target and the do-no-harm invariant it preserves:
   - §1 FX degradation chain (yfinance→FRED→historical_fx_rates→last-known→BLOCK; direct-BQ last-known
     helper to avoid the `_usd_value_asof` mutual-recursion; execute_sell credit-last-known-else-block,
     never 1.0) — `paper_trader.py:388-392`, `fx_rates.py:78-104`.
   - §2 Kill-switch audited restart-replayable `peak_reset` state machine (new event + `_load_from_audit`
     replay branch + 2 authorized emit sites, DARK until `KS-PEAK-RESET: APPROVED`) + `current_nav<=0`
     null-breach guard — `kill_switch.py:212, :230-264, :61-106`.
   - §3 Sign-safe overlay algebra `score + abs(score)·(mult−1)` with the both-regimes proof + worked
     table — `news_screen:329`, `macro_regime:542/547`, pead/options/insider/peer_leadlag.
   - §4 Gate corrections: DSR de-annualization pinned to the 0.9004 BBLZ reference (+ the ≈√252 bug
     quantification), purge+embargo (1.5·holding_days), boundary business-day-snap, fracdiff-at-predict
     parity, go-live booleans — `analytics.py:292-335/654-661`, `backtest_engine.py:566-598/486-490/793-801`,
     `walk_forward.py:61`, `paper_go_live_gate.py:111`.
   - §5 Do-no-harm ledger (every immutable threshold byte-untouched; guard-behavior changes DARK-until-token).
   - §6 Downstream step map (69.1 money-path byte-coordinated with phase-68; 69.2 offline; 69.3 live flag-gated).

Also installed this cycle: **phase-69** (5 steps 69.0-69.4, immutable criteria) appended to
`.claude/masterplan.json` (pure additive merge — 92 prior phases byte-identical, count 92→93).

## Verification command output (verbatim)

Command (from masterplan 69.0):
```
bash -c 'test -f handoff/current/research_brief_69.0.md && test -f handoff/current/design_audit_burndown_69.md && grep -q "gate_passed" handoff/current/research_brief_69.0.md && grep -qi "last-known" handoff/current/design_audit_burndown_69.md && grep -Eqi "peak.reset" handoff/current/design_audit_burndown_69.md && grep -qi "sign-safe" handoff/current/design_audit_burndown_69.md && grep -Eqi "deflated sharpe|dsr" handoff/current/design_audit_burndown_69.md && grep -Eqi "purge|embargo" handoff/current/design_audit_burndown_69.md'
```
Result: **VERIFY EXIT=0 PASS**

## Proof: no production code changed (criterion 4)

- `git status --short` under `backend/` and `frontend/`: **nothing** (no production code files touched).
- `git diff --stat backend/ frontend/`: empty.
- Only artifacts changed: `handoff/current/*` (this step's five-file set + design pack), the audit
  register/goal draft (untracked), and `.claude/masterplan.json` (phase-69 install).

## Artifact shape

Four markdown artifacts under `handoff/current/`: `research_brief_69.0.md`, `contract.md`,
`design_audit_burndown_69.md`, `experiment_results.md`. The design pack is the implementation contract
for 69.1/69.2/69.3; each fix is specified with file:line + do-no-harm invariant + a red→green (or
fixture) reproduction-test sketch so the code steps are surgical.

## Known limitation / honesty

- The DSR reference (0.9004) and the ≈√252 inflation were re-derived by Main this session and match the
  Bailey paper's printed numerical example; the ACTUAL pinned unit test is authored in 69.2 (this step is
  design only). Q/A can independently recompute from the inputs in §4a.
- 69.1 (money-path code) must byte-coordinate with in-flight phase-68 work (68.5 fill-price gate shares
  `paper_trader.py`); this pack flags the shared surface but the coordination is enforced at 69.1 GENERATE.
