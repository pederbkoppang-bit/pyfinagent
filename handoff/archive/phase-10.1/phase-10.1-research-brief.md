# Research Brief: phase-10.1 Sprint calendar config (Mon-Fri + monthly anchor)

Tier: simple (closure audit -- YAML already on disk)

## YAML Audit Results

File: `backend/autoresearch/sprint_calendar.yaml`

| Criterion | Masterplan ID | Value found | Pass? |
|-----------|--------------|-------------|-------|
| `new_weekly_slots == 2` | new_weekly_slots_equals_2 | `new_weekly_slots: 2` (line 16) | YES |
| `thursday` key under `days` | thursday_and_friday_defined | present (line 19) | YES |
| `friday` key under `days` | thursday_and_friday_defined | present (line 24) | YES |
| `monthly_anchor` section exists | monthly_anchor_defined | present (line 30-38) | YES |
| File exists on disk | calendar_config_committed | confirmed | YES |

All four `success_criteria` from `.claude/masterplan.json` phase-10.1 are satisfied.

## YAML Content Summary (< 100 words)

`new_weekly_slots: 2`. Two `days` keys: `thursday` (slot_id `thu_batch`, 22:00 UTC, batch-trigger role) and `friday` (slot_id `fri_promotion`, 21:00 UTC, promotion-gate role). `monthly_anchor` fires on `last_trading_friday` as a `champion_challenger_sortino_gate` with HITL approval required (`hitl: true`), 20-day minimum challenger window, Sortino MAR defaulting to 0.045 (3M T-Bill). References cross-link phase-10.0 supersede doc, phase-8.5.2 budget.py, and phase-8.5.5 gate.py.

## Security / ASCII Compliance

`.claude/rules/security.md` requires ASCII-only logger messages. This YAML contains no logger calls -- it is a static config file loaded by Python. All characters in the file are ASCII-safe; the multi-line `notes` block (line 35-38) uses only ASCII punctuation. No compliance issue.

## Verification Command Dry-Run (manual trace)

```
test -f backend/autoresearch/sprint_calendar.yaml
  -> file exists: PASS

python -c "import yaml; d=yaml.safe_load(open('backend/autoresearch/sprint_calendar.yaml')); \
           assert d['new_weekly_slots'] == 2 and 'thursday' in d['days'] and 'friday' in d['days']"
  -> new_weekly_slots: 2 == 2: True
  -> 'thursday' in d['days']: True
  -> 'friday' in d['days']: True
  -> assert passes: PASS
```

## Research Gate Checklist

Hard blockers:
- [x] Recency scan: N/A for closure audit -- YAML is a new internal artifact, no external literature required
- [x] Internal file inspected: `backend/autoresearch/sprint_calendar.yaml` read in full (lines 1-44)
- [x] Masterplan criteria cross-checked: all 4 success_criteria verified against YAML content
- [x] Security rule checked: ASCII-only compliance confirmed

Soft checks:
- [x] Cross-references in YAML align with existing handoff structure (phase-10.0 supersede, phase-8.5.x gates)
- [x] `monthly_anchor.hitl: true` present -- HITL requirement not dropped
- [x] `min_challenger_days: 20` present -- duration guard intact

```json
{
  "tier": "simple",
  "external_sources_read_in_full": 0,
  "snippet_only_sources": 0,
  "urls_collected": 0,
  "recency_scan_performed": false,
  "internal_files_inspected": 2,
  "gate_passed": true,
  "note": "Closure audit only -- no external literature required. Gate passed on internal-inspection criteria alone. All four masterplan success_criteria confirmed satisfied by YAML on disk."
}
```
