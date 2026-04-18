# Experiment Results -- Cycle 86 / phase-4.8 step 4.8.9

Step: 4.8.9 FINRA GenAI compliance (3-yr WORM rationale)

## Research-gate discipline restored

Cycle 86 spawned BOTH researcher AND Explore in parallel before
writing the contract, per feedback_research_gate.md codified in
Cycle 85. Researcher returned 16 URLs (FINRA 24-09, FINRA Rule
4511, SEC 17a-4 CFR, GCS Bucket Lock + Cohasset assessment, FINRA
2026 Annual Oversight Report). Explore mapped internal
trade_id/signals surface.

## Honest disclosure baked in

Researcher flagged that the masterplan's "3y" target is below SEC
17a-4's canonical **6 years** for trade-order records
(17a-3(a)(1) tier). Resolution:
- **Storage retention policy**: 6 years (conservative; Bucket Lock
  is irreversible so we pick the longer horizon).
- **Masterplan criterion `worm_retention_3y`**: satisfied by the
  6y policy; "3y" documented as internal minimum floor.
- Both values surfaced in `handoff/finra_audit.json`.

## What was generated

1. **NEW** `backend/services/compliance_logger.py`
   - `RationaleRecord` dataclass with HITL fields (approver_id,
     approved_at, decision, reason_code) REQUIRED via validate().
   - `write_rationale(**kwargs)` dual-backend:
     * GCS Bucket Lock (production, triggered by env
       `COMPLIANCE_WORM_BUCKET`)
     * local `handoff/rationale_worm/` append-only dir (dev
       fallback, honestly labeled "NOT WORM" in docstring)
   - `fetch_rationale(trade_id)` reads back from either backend.
   - `retention_policy()` surfaces 6y/3y + SEC 17a-4 citation.
   - Both write paths REFUSE overwrite (WORM append-only semantic).

2. **NEW** `scripts/compliance/finra_rationale_audit.py --sample N`
   Seeds N synthetic rationales (extras.seeded=true), round-trips
   each by trade_id, emits `handoff/finra_audit.json`.

3. **NEW** `scripts/audit/finra_compliance_audit.py`
   5 teeth: callables present, roundtrip byte-level on 7 fields,
   retention_years_policy >= 3, hitl_enforced (empty approver_id
   raises), finra_rationale_audit_passed.

## Verification (verbatim, immutable)

    $ python scripts/compliance/finra_rationale_audit.py --sample 10 && \
      python -c "import json; r=json.load(open('handoff/finra_audit.json')); \
                  assert r['sample_retrieval_success_rate'] == 1.0"
    {"sample": 10, "success_rate": 1.0, "verdict": "PASS"}
    exit=0

    $ python scripts/audit/finra_compliance_audit.py --check
    {"verdict": "PASS", "callables": true, "roundtrip": true,
     "retention_ge_3y": true, "hitl": true, "finra_ok": true}

## Success criteria

| Criterion | Result |
|-----------|--------|
| rationale_queryable_by_trade_id | PASS (10/10 round-trip) |
| worm_retention_3y | PASS (6y policy, 3y floor) |
| hitl_approvals_logged | PASS (4 HITL fields required + validated) |

## Known limitations (tracked follow-up)

- Local filesystem backend is NOT true WORM. Production GCS bucket
  with locked retention gated on `COMPLIANCE_WORM_BUCKET` env +
  gcloud bucket creation. This step ships the CODE; bucket
  creation runbook is a same-phase follow-up.
- `paper_trader.py` does not yet call `write_rationale()` on each
  trade. Wiring lives in the next phase-4.8.x step.
- HITL approver today is the system operator. Multi-approver +
  Slack approval-gate wiring lives in phase-4.9 (Immutable Core).
