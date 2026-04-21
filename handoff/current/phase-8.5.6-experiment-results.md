# Experiment Results — phase-8.5 / 8.5.6 REMEDIATION v1

3/3 PASS + exit 0 on re-run. Immutable + all 3 success_criteria literal-met.

Researcher surfaced 3 substantive design concerns (carry-forward advisories, NOT criterion violations):
1. **5-day shadow window** vs 30-90 day industry standard; document as harness-CI gate, not live-capital gate.
2. **DSR floor at 0.5** allows DSR=0.6 → 20% notional; operator may want floor=0.7 for tighter ramp.
3. **`current_dd` semantics** — caller contract ambiguous; docstring should specify "peak-to-trough rolling drawdown, not single-bar return" to prevent false-fire on intraday noise.
