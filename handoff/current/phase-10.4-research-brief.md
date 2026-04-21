# Research Brief: phase-10.4 — Friday Promotion Gate Routine

**Tier:** moderate  **Accessed:** 2026-04-20

## Queries run (three-variant discipline)

1. Current-year frontier: "top-N strategy selection DSR Sharpe ranking quantitative portfolio promotion 2026"
2. Last-2-year: "strategy promotion live trading paper shadow staged rollout allocation ramp quant 2024 2025"
3. Year-less canonical: "Bailey Lopez de Prado deflated Sharpe ratio strategy selection multiple testing"

## Read in full (>=5 required)

| URL | Kind | Key finding |
|-----|------|-------------|
| https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf | Peer-reviewed (Bailey & Lopez de Prado) | DSR deflates SR by multiple-testing count; rank by deflated score; threshold DSR >= 0.95 = strong evidence |
| https://en.wikipedia.org/wiki/Deflated_Sharpe_ratio | Reference | Full DSR formula; gate at probability >= 0.95 |
| https://surmount.ai/blogs/ignal-to-execution-quant-strategy | Practitioner blog | Staged ramp: shadow -> pilot minimal capital -> scale; circuit-breaker on DD |
| https://quantpedia.com/multi-strategy-management-for-your-portfolio/ | Practitioner | Volatility-normalize before comparing; no hard count cap prescribed |
| https://arxiv.org/html/2510.18569v1 | arXiv Oct 2025 (QuantEvolve) | Combined score = Sharpe + IR + MaxDD equally weighted; confirms multi-metric ranking |
| https://finlego.com/blog/designing-a-real-time-ledger-system-with-double-entry-logic | Engineering blog | Idempotency via unique reference keys; upsert-on-key canonical |

## Recency scan (2024-2026)

- **QuantEvolve (arXiv 2510.18569, Oct 2025)** confirms multi-metric ranking over pure DSR
- No 2024-2026 work supersedes Bailey & Lopez de Prado (2014) on DSR gating; 0.95 threshold remains canonical
- 5% starting allocation: practitioner-conventional (multiple blogs), not a formal formula

## Key findings

1. **DSR ranking is canonical** for top-N. Rank by deflated SR desc; tie-break by PBO asc (lower overfitting = better).
2. **"5% starting allocation"** = capital slice allocated to the new strategy (passed as `capital` arg to `Promoter.position_size()` downstream). NOT the output of `position_size()`. Friday's job: record `starting_alloc=0.05` in ledger `notes`.
3. **Max N=3** is a count cap, not a statistical gate. Enforce as `min(top_n, max_n)` after ranking.
4. **Q/A-flagged edge from phase-10.3:** `thursday_batch.py:82-91` returns `already_fired=False` even when `append_row` returned `ok=False`. Friday MUST validate `thu_batch_id` is non-empty in the ledger row; fail-closed with `error="no_thursday_batch_on_ledger"` otherwise.
5. **Idempotency:** ledger upsert IS the slot counter. Check `fri_promoted_ids` populated in the week's row; if yes, return `already_fired=True`.

## Internal code inventory

| File | Role | Anchor |
|------|------|--------|
| `backend/autoresearch/gate.py` | `PromotionGate.evaluate(trial) -> {promoted, reason, trial_id}` | gate.py:24-39 |
| `backend/autoresearch/promoter.py` | `position_size(trial, capital)`; `capital` is a parameter | promoter.py:34 |
| `backend/autoresearch/weekly_ledger.py` | `append_row`/`read_rows`; upsert on `week_iso` | weekly_ledger.py:71-114 |
| `backend/autoresearch/thursday_batch.py` | Known bug: returns `already_fired=False` when ledger write fails | thursday_batch.py:82-91 |
| `scripts/harness/autoresearch_gate_test.py` | Reusable gate fixtures (good: dsr=0.99/pbo=0.10; bad: dsr=0.90/pbo=0.10) | lines 14, 28 |
| `scripts/harness/phase10_thursday_batch_test.py` | Test scaffold pattern (tempfile + 3 cases) | — |

## Final recommendation

**Module:** `backend/autoresearch/friday_promotion.py`

**Signature:**
```python
def run_friday_promotion(
    week_iso: str,
    *,
    candidates: list[dict],
    top_n: int = 1,
    max_n: int = 3,
    starting_allocation_pct: float = 0.05,
    ledger_path: Path | None = None,
) -> dict[str, Any]:
    """Returns {promoted_ids, rejected_ids, allocations, already_fired, error}."""
```

**Fail-closed on Thursday-missing:**
```python
row = next((r for r in read_rows(path=lpath) if r["week_iso"] == week_iso), None)
if row is None or not row.get("thu_batch_id"):
    return {"promoted_ids": [], "rejected_ids": [], "already_fired": False,
            "error": "no_thursday_batch_on_ledger"}
```

**Idempotency:** check `fri_promoted_ids` populated (and != "[]") in the week's row.

**Ranking:**
```python
gate = PromotionGate()
passed = [c for c in candidates if gate.evaluate(c)["promoted"]]
ranked = sorted(passed, key=lambda c: (-float(c["dsr"]), float(c["pbo"])))
promoted = ranked[:min(top_n, max_n)]
rejected = [c for c in candidates if c not in promoted]
```

**Ledger write:** `append_row(week_iso=..., fri_promoted_ids=promoted_ids, fri_rejected_ids=rejected_ids, notes=f"starting_alloc={starting_allocation_pct}")`. No schema change needed.

**Verification (`scripts/harness/phase10_friday_promotion_test.py`):** 4 cases mapping to masterplan `success_criteria`:
1. `case_consumes_exactly_one_slot` — 2nd call returns `already_fired=True`, 1 row
2. `case_reuses_phase_8_5_5_dsr_pbo_gate` — candidate with dsr=0.90 rejected
3. `case_promotion_at_5pct_starting_allocation` — ledger `notes` contains `starting_alloc=0.05`
4. `case_top_n_default_1_max_3` — default=1; top_n=3 yields 3; top_n=5 capped at max_n=3

## Pitfalls

1. Ranking by raw Sharpe instead of DSR — selects overfit.
2. Silent no-op on Thursday write failure — must fail-closed.
3. Re-firing check: `fri_promoted_ids` populated AND != "[]" (empty list string).
4. Passing `capital=total_portfolio` to `position_size()` — must be 5% slice.
5. Candidates without `dsr`/`pbo` fields — `PromotionGate` returns `{promoted: False, reason: "missing..."}`.

## JSON envelope

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 6,
  "snippet_only_sources": 8,
  "urls_collected": 14,
  "recency_scan_performed": true,
  "internal_files_inspected": 8,
  "report_md": "handoff/current/phase-10.4-research-brief.md",
  "gate_passed": true
}
```
