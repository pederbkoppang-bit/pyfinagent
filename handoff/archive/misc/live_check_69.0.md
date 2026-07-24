# Live Check — Step 69.0 (P0 design pack, phase-69 audit burn-down)

Design-pack step (offline, no live surface). This live_check quotes the design doc's four
sections + the research-gate envelope + the fresh Q/A verdict JSON, per the masterplan
`live_check` field for 69.0.

## Deterministic evidence (Main + Q/A independently)

- Verification command (verbatim from masterplan 69.0) → **EXIT 0 PASS**.
- `git status` under `backend/`/`frontend/` → **nothing** (no production code changed; required for a design-only step).
- phase-69 installed additively (92→93 phases): `[.steps[].id]` → `["69.0","69.1","69.2","69.3","69.4"]`.
- **DSR reference recomputed with scipy (Main AND Q/A independently)**: correct de-annualized path → **DSR=0.9004** (z=1.284); N=46→0.9505; Normal returns cross 0.95 near N=88; bug path (annualized SR + daily T) → **DSR=0.9999999** (z=5.29). Reproduces exactly.

## Design doc four sections (quoted from `design_audit_burndown_69.md`)

### §1 FX degradation chain (target 69.1)
> Introduce a last-known-rate degradation chain in `_usd_value_live(ccy)`:
> `cache → _fetch_yf → _fetch_fred → NEW (D) _last_known_usd_value(ccy) [direct historical_fx_rates read] → None only when no rate was EVER stored.`
> `_last_known_usd_value` queries `historical_fx_rates` **directly** — do NOT call `_usd_value_asof` (which degrades back to `_usd_value_live` → mutual recursion).
> execute_sell rule (`paper_trader.py:388-392`): replace unconditional `_l2u = 1.0` with credit-at-last-known, else **BLOCK + PAGE P1** (fail-closed) — never 1.0. Do-no-harm: no threshold changes; USD path byte-identical.

### §2 Kill-switch audited peak-reset state machine (target 69.1)
> New audit event `peak_reset` (guard-behavior change ⇒ DARK until `KS-PEAK-RESET: APPROVED`). Emit sites: (a) on flatten-to-cash → reset peak to post-flatten NAV; (b) on operator resume → re-anchor peak to current NAV. Replay branch in `_load_from_audit` (`kill_switch.py:61-106`) sets `_peak_nav` → restart-replayable + idempotent. Plus a `current_nav<=0` null-breach guard in `evaluate_breach` (`:230-264`) so a BQ-timeout `or 0.0` no longer renders a phantom 100% breach. Thresholds (4/10/8/30) byte-untouched.

### §3 Sign-safe overlay algebra (target 69.3)
> `def sign_safe(score, mult): return score + abs(score) * (mult - 1.0)`. Proof: score≥0 → `score*mult`; score<0 → `score*(2-mult)`; `∂score_out/∂mult = |score| ≥ 0` so a boost never lowers and a penalty never raises rank in both sign regimes. Sites: `news_screen:329`, `macro_regime:542/547`, pead/options/insider/peer_leadlag. Flag-gated + ON-vs-OFF live_check.

### §4 Gate corrections (target 69.2)
> DSR: de-annualize BOTH observed_sr and variance_of_srs (`SR_p=SR_ann/√ppy`, `V_p=V_ann/ppy`); reference pin DSR(SR_ann=2.5,T=1250,N=100,skew=−3,kurt=10,ppy=250)=**0.9004**, bug path ≈**0.9999999** (`analytics.py:292-335/654-661`). Purge+embargo: purge training samples whose `[sample_date, sample_date+1.5·holding_days]` overlaps `[test_start,test_end]` (`backtest_engine.py:566-598`, `walk_forward.py:61`). Boundary business-day-snap (`:486-490/512-516`). Fracdiff-at-predict parity (`:793-801`). Go-live booleans to documented spec (`paper_go_live_gate.py:111`). DSR≥0.95 / PBO≤0.5 byte-untouched.

## Research-gate envelope (from `research_brief_69.0.md`)

```json
{
  "tier": "complex",
  "external_sources_read_in_full": 8,
  "snippet_only_sources": 15,
  "urls_collected": 23,
  "recency_scan_performed": true,
  "internal_files_inspected": 19,
  "gate_passed": true
}
```

## Fresh Q/A verdict (agent qa, Opus; full critique in `evaluator_critique.md`)

```json
{
  "ok": true,
  "verdict": "PASS",
  "harness_compliance": {"research_gate":"PASS","contract_before_generate":"PASS","results_present":"PASS","log_last":"PASS","no_verdict_shopping":"PASS"},
  "checks_run": ["verification_command","dsr_reference_recompute_scipy","sign_safe_algebra_symbolic","criteria_verbatim_diff","git_no_code_change","fileline_anchor_spotcheck","harness_log_conditional_count","code_review_heuristics","research_brief","contract","experiment_results"],
  "violated_criteria": [],
  "certified_fallback": false,
  "notes": "All 5 criteria MET on a design-only step. DSR reference reproduces exactly with scipy; sign-safe algebra proven symbolically; criteria verbatim; git-confirmed zero production code changed; 4/4 file:line anchors real; research gate judged legitimate despite researcher-stall/Main-finalize provenance (DSR numbers reproduce = not fabricated, transparent disclosure). No blockers."
}
```

## Provenance note (transparency)
Two researcher subagent spawns (Fable, then Opus) and the first Q/A spawn (Opus) each did the work
but STALLED on their final flush and were stopped per CLAUDE.md STALL WATCH (a systemic harness-subagent
stall pattern observed this session). The research brief's sources + DSR example were persisted
incrementally; Main finalized the synthesis + envelope from the already-read sources. The second Q/A
spawn (Opus, write-verdict-early discipline) completed cleanly and independently re-verified the DSR
reference and sign-safe algebra — so the PASS rests on an independent evaluator, not Main's self-report.
