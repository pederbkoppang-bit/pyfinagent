# Research Brief — phase-8.5.6 Promotion path (closure)

Closure-style. phase-4.8 risk guard + phase-4.9 kill switch already on disk. Alpaca paper via phase-3.5.3.

Design:
- `Promoter.promote(trial)` — requires trial.shadow_trading_days >= 5 and trial.dsr >= 0.95.
- `Promoter.position_size(trial, capital)` — returns `capital * max(0, min(1, (trial.dsr - 0.5) * 2))` (tied to realized DSR).
- `Promoter.on_dd_breach(current_dd, kill_fn)` — triggers `kill_fn(reason)` when `abs(dd) > 0.10`.

`gate_passed: true` on closure.
