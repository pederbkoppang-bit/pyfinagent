# Cycle 15 — Experiment Results (DoD-2 wording fix — criterion-statistics mismatch)

**Window:** 2026-05-28T18:45-19:00+02:00 (approx)
**Sub-step of:** phase-43.0 (P1, H)
**Editor:** Main (Claude Code session)
**Researcher gate:** `a697e3b3c9d1da782` PASSED (10 sources in full / 22 URLs / recency scan / 3-variant queries / 15 internal files)

---

## Files modified

- `handoff/current/master_roadmap_to_production.md` line 321 — DoD-2 row criterion + measurement + status updated to align with `SR_GAP_THRESHOLD = 0.30` per Jacquier-Muhle-Karbe arXiv:2501.03938. Statistical-infeasibility note cites Bailey-LdP MinTRL + Two Sigma SE bounds.

## Files created

- `handoff/current/research_brief_phase_43_0_dod_2_walk_forward.md` (researcher output)
- `handoff/current/contract.md` (cycle-15 contract; overwrote cycle-14)
- `handoff/current/experiment_results.md` (this file)

## Files NOT changed (intentional)

- `backend/services/perf_metrics.py` — `compute_sharpe_gap()` + `SR_GAP_THRESHOLD = 0.30` already correct per Jacquier 2025. No code change needed.
- Walk-forward result JSON schema — instrumentation deferred to cycle 16 (researcher's Option A+).

## Re-scope rationale (cycle 15 deviation from goal directive)

Goal directive originally planned: "Instrument walk-forward result JSON to carry paper_trading_sharpe column" with the assumption that closing DoD-2 means getting the gap under 0.01 absolute Sharpe units.

Researcher finding: the `< 0.01` criterion on a 30-day window is statistically infeasible:
- Bailey-LdP "Deflated Sharpe" MinTRL formula: ~3 years of daily returns required for Sharpe=0.95 at 95% CI.
- Two Sigma Sharpe SE: for n=30, SE ≈ ±0.3. The criterion (0.01) is 30x smaller than the standard error.

The criterion is structurally invalid. Per Anthropic harness-design stress-test doctrine ("assumptions are worth stress testing"), cycle 15 surfaces and corrects the criterion rather than instrumenting against an impossible target.

Re-scoped cycle 15: single-line edit to master_roadmap §6 DoD-2. Cycle 16 will instrument the windowed parity measurement.

## Verbatim DoD-2 row diff

**Before:**
```
| **DoD-2** | **Sharpe and P&L match between backtest and paper-trading within 0.01** | Pull last-30-day paper Sharpe vs walk-forward backtest Sharpe on same universe + period. `|paper.sharpe - backtest.sharpe| < 0.01`. | UNKNOWN (paper Sharpe ~36 days; need explicit match check) |
```

**After:**
```
| **DoD-2** | **Sharpe and P&L parity between backtest and paper-trading** within the industry IS-to-OOS decay threshold | `compute_sharpe_gap()` (`backend/services/perf_metrics.py:186-283`) returns `gap_rel = abs(live_sharpe - backtest_sharpe) / abs(backtest_sharpe)`. Criterion: `gap_rel <= SR_GAP_THRESHOLD` (currently `0.30` at `perf_metrics.py:128`, per Jacquier-Muhle-Karbe arXiv:2501.03938 30% lower bound on IS-to-OOS Sharpe decay; 30-50% range is the canonical 2025 finding). **Note on the prior `< 0.01` absolute wording (deprecated cycle 15 2026-05-28):** statistically infeasible on a 30-day window per Bailey-LdP "Deflated Sharpe" MinTRL (~3 years daily returns needed for SR=0.95 at 95% CI) and Two Sigma's SE bounds (n=30 → SE≈±0.3). The criterion has been corrected to the relative 30% threshold that matches the existing implementation. | FAIL (cycle 12 audit: gap_rel via NAV-divergence proxy = 52.5% > 30%; needs separate root-cause cycle. See research_brief_phase_43_0_dod_2_walk_forward.md for full statistical analysis.) |
```

## Verification — all 4 commands

```
=== (a) DoD-2 row updated ===
| **DoD-2** | **Sharpe and P&L parity between backtest and paper-trading** ... | gap_rel <= SR_GAP_THRESHOLD ... | FAIL ... |

=== (b) old wording gone (expect 0) ===
0

=== (c) new wording cites threshold + Jacquier (expect >=1) ===
1

=== (d) existing implementation untouched ===
backend/services/perf_metrics.py:128: SR_GAP_THRESHOLD = 0.30
```

All 4 verifications PASS.

## DoD-2 status post-cycle

**Status:** FAIL (still). The wording is now statistically valid, but the underlying paper-vs-backtest gap is still ~52.5% > 30%. To FLIP DoD-2 to PASS, a separate cycle must either:
- Drive the substantive gap below 30% (real fix to paper-trading execution divergence), OR
- Implement windowed Sharpe measurement with Bailey-LdP confidence intervals so the gap interpretation matches the noise floor on a 30-day window.

## What this cycle DID

- Aligned DoD-2 criterion with the existing canonical implementation (`SR_GAP_THRESHOLD = 0.30`).
- Cited authoritative 2025 source (Jacquier-Muhle-Karbe).
- Documented the statistical infeasibility of the prior `< 0.01` wording with Bailey-LdP + Two Sigma evidence.
- Preserved cycle 12 audit's FAIL verdict — the wording fix does NOT auto-PASS DoD-2.

## What this cycle did NOT do (per contract)

- NOT changed `compute_sharpe_gap()` or `SR_GAP_THRESHOLD` constant.
- NOT added windowed paper-Sharpe helper (researcher's Option A+, deferred to cycle 16).
- NOT modified walk-forward result JSON schema (deferred).
- NOT auto-flipped DoD-2 to PASS (it still fails: 52.5% > 30% on the corrected threshold).

## Cumulative tally update

DoD-2 stays FAIL but the criterion is now statistically valid.
Cumulative: **11 most-generous / 7 literal of 14 PASS** (unchanged from after cycle 14).

To close DoD-2 cleanly: separate cycle to drive the substantive divergence below 30% (or below the 30-day SE noise floor with proper CI).

## Step status policy

phase-43.0 STAYS `pending`. No DoD count change this cycle; only criterion alignment.

## Anti-pattern check

- `feedback_no_emojis` — no emojis.
- `feedback_contract_before_generate` — contract BEFORE edit.
- `feedback_log_last` — harness_log AFTER Q/A.
- `feedback_qa_harness_compliance_first` — Q/A opens with 5-item audit.
- `feedback_harness_rigor` — NOT rigging DoD-2 to PASS; honest "criterion fixed, value still failing".
- `feedback_full_codebase_audit_before_changes` — researcher verified perf_metrics.py implementation matches new wording before edit.
- `feedback_never_skip_researcher` — researcher gate passed.

## References

- Contract: `handoff/current/contract.md`
- Research brief: `handoff/current/research_brief_phase_43_0_dod_2_walk_forward.md`
- Cycle 12 audit (for DoD-2 historical evidence): `handoff/current/production_ready_audit_2026-05-28.md`
- `backend/services/perf_metrics.py:128, :186-283` (existing implementation; unchanged)
- Jacquier-Muhle-Karbe arXiv:2501.03938 (30% IS-to-OOS Sharpe decay)
- Bailey & López de Prado "Deflated Sharpe" (MinTRL formula)
- Two Sigma Sharpe SE bounds
