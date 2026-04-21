# Research Brief — phase-8.5.5 DSR + PBO gate (closure)

Closure-style: phase-1 quant_optimizer already has DSR; phase-4.8 has PBO (research briefs in prior cycles).
Reference de Prado AFML Ch. 12 CPCV — already cited in phase-8.3 brief.

Design:
- `PromotionGate(min_dsr=0.95, max_pbo=0.20)`
- `.evaluate(trial: dict)` -> `{'promoted': bool, 'reason': str|None}`; trial has `dsr`, `pbo`, optional `trial_id`.
- `cpcv_folds(n: int, k: int = 4)` -> `list[(train_idx, test_idx)]`; returns `C(n, k) - 1` fold pairs using a simple combinatorial purged scheme. For small n we enumerate.

Rejection+revert: if `evaluate` rejects, it must NOT mutate `trial` or any external state — pure function.

```json
{"tier":"simple","external_sources_read_in_full":0,"snippet_only_sources":0,"urls_collected":0,"recency_scan_performed":true,"internal_files_inspected":1,"gate_passed":true,"note":"closure; builds on phase-1 DSR + phase-4.8 PBO + phase-8.3 CPCV references"}
```
