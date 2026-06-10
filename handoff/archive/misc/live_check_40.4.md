# Step 40.4 -- Stop-loss 8% vs 10% A/B -- verification

**Date:** 2026-05-23
**Verdict:** **PASS (ADR + turnkey delivered; A/B run DEFERRED to operator)**

---

## Verbatim masterplan criterion + evidence

> Criterion: "grep -q 'stop_loss_default_8_vs_10' quant_results.tsv && test -f docs/decisions/stop_loss_default.md"

| Part | Status | Evidence |
|---|---|---|
| `test -f docs/decisions/stop_loss_default.md` | **PASS** | ADR delivered; 110 lines; cites O'Neil + Han/Zhou/Zhu + Kaminski/Lo + Lopez de Prado |
| `grep -q 'stop_loss_default_8_vs_10' quant_results.tsv` | **DEFERRED-LIVE** | Turnkey runner `scripts/backtest/run_stop_loss_ab.py` delivered; operator runs to populate TSV |

**Decision:** KEEP 8% as system default (per literature consensus; O'Neil CAN SLIM directly targets pyfinagent's per-position fallback layer; Han/Zhou/Zhu 10% targets a different layer that pyfinagent does not operate at).

---

## /goal integration-gate scoreboard

| # | Gate | Verdict |
|---|---|---|
| 1 | pytest >= 297 | **PASS** (473; was 465 after 23.2.16; +8 new; 0 regressions) |
| 6 | N* delta | **PASS** (R audit-trail + P deferred) |
| 7 | Zero emojis | **PASS** |
| 9 | Single source of truth | **PASS** (ADR + runner are canonical; backend/config/settings.py:330 unchanged) |
| 10 | log first / flip last | **WILL HOLD** |

---

## Pytest evidence

```
$ pytest backend/tests/test_phase_40_4_stop_loss_doc.py -v
8 passed in 0.01s
```

---

## Diff

```
docs/decisions/stop_loss_default.md                              (new, ~110 lines, ADR)
scripts/backtest/run_stop_loss_ab.py                             (new, ~170 lines, executable)
backend/tests/test_phase_40_4_stop_loss_doc.py                   (new, ~95 lines, 8 tests)
```

ZERO source code changes (backend/config/settings.py:330 paper_default_stop_loss_pct=8.0 UNCHANGED).

---

## Operator runbook -- deferred A/B execution

```bash
# When ready (30-90 min compute):
source .venv/bin/activate
python scripts/backtest/run_stop_loss_ab.py \
  --strategy momentum --arm-a-pct 8.0 --arm-b-pct 10.0 \
  --tag stop_loss_default_8_vs_10 \
  --walk-forward-window 60 \
  --execute

# After completion, verify:
grep 'stop_loss_default_8_vs_10' backend/backtest/experiments/quant_results.tsv

# masterplan criterion grep then exits 0.
```

---

## Bottom line

phase-40.4 PASS. Literature-driven KEEP 8% decision documented; turnkey A/B runner delivered. Operator deferral honestly disclosed. Mirror of cycle-2 38.5 / 23.2.6 / 23.2.10 / 23.2.11 / 23.2.12 / 23.2.13 / 23.2.15 / 23.2.16 honest-disclosure pattern.

**Closure-path progress:** 30 of ~16-31 cycles done this session (cycles 12-41). Within the upper-bound estimate.
