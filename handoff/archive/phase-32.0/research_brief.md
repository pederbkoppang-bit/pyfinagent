# phase-32.1 Research Brief — Breakeven-Stop Ratchet at +1R

**Tier:** moderate
**Date:** 2026-05-20
**Effort:** max
**Predecessor:** `handoff/archive/phase-31.0/research_brief.md` (deep canonical audit, 22 sources, gate_passed=true)

## Executive Summary (≤150 words)

[TBD after research complete]

## Topic 1: Recency-Scan Delta (post-2026-05-20 findings)

**Query variants run:**
- [ ] `breakeven stop systematic trading 2026`
- [ ] `profit-locking ratchet quant 2026`
- [ ] year-less canonical: `breakeven stop trailing`

**Read in full (>=5 required):**

| URL | Accessed | Kind | Fetched how | Key finding |
|-----|----------|------|-------------|-------------|

**Snippet-only:**

| URL | Kind | Why not in full |
|-----|------|------------------|

**Verdict:** [TBD - delta or no delta vs phase-31.0 P1.1]

## Topic 2: Adversarial Recheck (new Carver-style breakeven counter-arguments)

**Audit claim under test:** "Kaminski-Lo Proposition 2 is about TRAILING, not breakeven; breakeven is a one-shot mutation and is safe across all strategies."

**Sources surveyed:**

[TBD]

**Verdict:** [TBD]

## Topic 3: Internal Duplicate Audit

**Search patterns run on `backend/`:**
- `breakeven`
- `advance_stop`
- `move_to_breakeven`
- `stop_advance`
- `ratchet`

**Results:**

[TBD]

**Verdict:** [TBD - confirm NO separate breakeven helper exists]

## Topic 4: Schema Migration Pattern

**Source migration script:** [TBD - most recent under `scripts/migrations/`]

**Canonical pattern (verbatim):**

```python
[TBD]
```

**`_safe_save_position` compatibility check:** [TBD - lines 764-773]

## Topic 5: MFE Unit Reconciliation

**Lines 437-438 (pnl_pct computation):** [TBD verbatim]
**Lines 443-446 (mfe_pct):** [TBD verbatim]
**Verdict:** [TBD - confirm units shared with `settings.paper_default_stop_loss_pct`]

## Last-2-Year Recency Scan

[TBD]

## Adversarial Sourcing Section

[TBD]

## Implementation Crib Sheet

### Migration-script structure to copy

```python
[TBD]
```

### paper_trader.py:437-446 (wire-in site)

```python
[TBD]
```

### paper_trader.py:751 (_POSITION_RT_FIELDS update)

```python
[TBD]
```

## JSON Envelope

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 0,
  "snippet_only_sources": 0,
  "urls_collected": 0,
  "recency_scan_performed": false,
  "internal_files_inspected": 0,
  "gate_passed": false
}
```
