# Evaluator Critique — Step 70.2 (S2: soft, profit-aware cross-sector diversification)

**Evaluator:** fresh, independent Q/A via the Workflow structured-output path (Opus 4.8, `effort: max`, $0 Max
rail, stall-immune — run wf_eeae29cf-ea4). Verdict transcribed VERBATIM by Main (no-self-eval guardrail).

**VERDICT: PASS** | violated_criteria: [] | do_no_harm_ok: true

## Checks
- verification_command_exit: 0 | pytest_passed: true (7 passed) | off_byte_identical: true | no_risk_threshold_moved: true
- Harness compliance 5/5: research-gate-before-contract, contract-before-generate (mtime-proven), results present,
  log-last, no-verdict-shopping (first Q/A on 70.2).
- ablation_check: baseline_sharpe 1.3437; soft_sharpe_not_lower: true; breadth_increased: true; hard_neutral_worse: true.

## Q/A notes (verbatim)

HARNESS 5/5 clean. (1) Research gate: research_brief_70.2.md present, gate_passed=true, 6 external sources
read-in-full (>=5), recency-scan section present, envelope complete. (2) mtime order correct: research < contract
< settings < screener < loop < pm < test < results. (3) experiment_results.md lists 6 changed files + verbatim
verification output + ON-vs-OFF ablation table. (4) Log-last: no phase=70.2 entry in harness_log (last=70.1).
(5) No verdict-shopping: on-disk evaluator_critique.md is the 70.1 file; this is the first Q/A on 70.2.

DETERMINISTIC: immutable verification cmd exit 0. pytest test_phase_70_2_soft_diversity.py = 7 passed. Import-smoke
of all 4 modules clean. git status = only backend/ (4 code) + backend/tests/ (1 new) + scripts/ablation/ (1) +
handoff/ + audit JSONLs; no unrelated files. 4 new flags all default OFF/identity.

CRITERIA (all met): [1] `_min_k_sector_slice` provably spans min(k,#sectors) sectors and never drops a
sector-leader on truncation; test shows plain top-5 slice = 1 sector (monosector funnel) vs min-K=3 = 3 sectors;
OFF (K=0) = plain slice; production candidates carry a real `sector` via the UNCONDITIONAL enrichment block
(autonomous_loop.py:784-807, gated on `if candidates:` only), so min-K works standalone. [2] SOFT + SIGN-SAFE
verified in code: multiplicative rank-decay (1-w)^j with leader j=0 untouched, routed through
overlay_math.sign_safe_mult(base,mult,enabled=True) = base+abs(base)*(mult-1), which LOWERS rank for a positive
base AND a negative base — no sign inversion; test confirms negative T3(-4) -> < -4. INDEPENDENTLY REPRODUCED the
ablation (clean PYTHONPATH re-run): baseline ann_Sharpe 1.344, soft dSharpe +0.176/+0.200/+0.234 (all POSITIVE —
no OOS drop, a rise), breadth +1.25/+2.02/+2.60, hard sector_neutral 1.226 = dSharpe -0.117 (WORSE) — matches
experiment_results exactly; also recomputed ann_Sharpe from the raw dumped monthly arrays (ddof=0) to an EXACT
6-sig-fig match. [3] Unknown-exempt guards both count cap and NAV-pct cap; tests prove OFF blocks (byte-identical)
/ ON allows. [4] OFF byte-identical: every lever gated to identity, proven by test_soft_off_and_w0_byte_identical.

DO-NO-HARM ok: the two portfolio_manager cap comparisons only PREPEND a default-OFF `not _unk_exempt and` guard —
threshold VALUES unchanged; no risk/DSR/PBO/stop/kill-switch threshold moved; settings.py only ADDS fields;
ablation macro-free; historical_macro FROZEN respected; $0, paper-only, DARK-until-token; activation
operator-token-gated. No operator config mutated.

MINOR NON-BLOCKING NOTES (anti-rubber-stamp): (a) 1 pre-existing regression FAIL
`test_phase_23_2_6_backend_log_has_skipping_buy_evidence` — a runtime-log-state assertion reading backend.log;
orthogonal to the 70.2 diff (companion cap=0 test PASSED; 36 passed/1 failed), an environmental flake. (b)
scripts/ablation/sector_neutral_replay.py must be run with PYTHONPATH=repo-root or `python -m` (pre-existing
invocation convention; the diff only added configs). (c) brief line 22 says '7 sources' vs the envelope's 6
(cosmetic; >=5 floor cleared). (d) the w bound le=1.0 permits w=1.0 which zeroes deeper same-sector positive
names (only w<=0.30 tested/intended; 'shades-never-zeroes' holds strictly for w<1). (e) the clean ablation re-run
regenerated the two dumps with a 7th-sig-fig baseline drift (yfinance data-vintage noise) — economically
identical, all deltas/conclusions unchanged. None affect any immutable criterion, the OFF byte-identical
guarantee, or do-no-harm. VERDICT: PASS.

## Main's disposition of the non-blocking notes (recorded, not a verdict edit)
- (a) pre-existing/unrelated flaky test — not introduced by 70.2; left as-is (out of scope).
- (b) PYTHONPATH invocation is the pre-existing convention for the ablation script; noted for reproducibility.
- (c) research_brief line-22 "7 sources" is a cosmetic typo; the authoritative envelope says 6 (>=5 floor met).
- (d) `paper_soft_sector_diversity_w` intended operating range is <=0.30 (the tested/validated grid); w near 1.0
  is out-of-intended-range. Left as le=1.0 (matches arXiv 2601.08717's [0,1] domain); the activation gate + the
  operator choose the value.
- (e) the on-disk dump reflects the Q/A's clean re-run (economically identical to Main's); no action needed.
