# Contract -- Cycle 86 / phase-4.8 step 4.8.9

Step: 4.8.9 FINRA GenAI compliance (3-yr WORM rationale)

## Research gate -- spawned researcher + Explore in parallel

Researcher (16 URLs): SEC 17a-4, FINRA Reg Notice 24-09, FINRA Rule
4511, FINRA 2026 Annual Oversight Report, GCS Bucket Lock +
Cohasset assessment, Federal Register 2022 17a-4 amendments.

Explore: trade_id + signals already exist in paper_trades; no WORM
config anywhere; HITL is agent-level Risk Judge, not human;
`scripts/compliance/` does not exist.

## Honest disclosure: 3y vs 6y mismatch

The masterplan says "3-yr WORM" but per SEC 17a-4, trade-order
records fall in the 6-year tier (17a-3(a)(1)). The researcher
flagged this explicitly. Decision for this cycle:
- **Storage retention policy**: 6 years (the conservative choice;
  Bucket Lock is irreversible so we pick the longer horizon).
- **Masterplan criterion `worm_retention_3y`**: the audit verifies
  retention >= 3 years (satisfied by the 6y target; "3y" is the
  internal floor documented alongside the real policy).
- The artifact records both values so a future reviewer sees the
  gap without confusion.

## Hypothesis

Ship a minimally honest compliance layer:
1. `backend/services/compliance_logger.py` writes one JSONL record
   per trade keyed by trade_id with HITL approver fields.
   - Production path: GCS bucket `pyfinagent-rationale-worm` with
     6-year locked retention (config documented; actual bucket
     creation requires gcloud + an owner decision, so the code
     writes to GCS when `COMPLIANCE_WORM_BUCKET` env is set,
     otherwise a local `handoff/rationale_worm/` append-only
     directory -- clearly labeled "local-dev fallback, NOT WORM").
2. `scripts/compliance/finra_rationale_audit.py --sample N` seeds
   N synthetic trade rationales, fetches each back by trade_id,
   asserts 100% retrieval, emits `handoff/finra_audit.json`.
3. `scripts/audit/finra_compliance_audit.py` verifies the
   implementation is not stub:
   (a) compliance_logger module has `write_rationale` + `fetch_
       rationale` callable functions.
   (b) round-trip test: write a synthetic record, read it back,
       compare every field byte-for-byte.
   (c) retention >= 3 years documented in handoff/finra_audit.json
       AND the constant in compliance_logger.py is >= 3y (in
       seconds).
   (d) HITL fields present in every written record: approver_id,
       approved_at, decision, reason_code.

## Scope

Files created:
1. **NEW** `backend/services/compliance_logger.py`
2. **NEW** `scripts/compliance/finra_rationale_audit.py`
3. **NEW** `scripts/audit/finra_compliance_audit.py`

## Immutable success criteria

1. rationale_queryable_by_trade_id -- round-trip write->fetch works
   for every sample trade_id.
2. worm_retention_3y -- retention policy >= 3 years (implemented
   as 6y, reported alongside).
3. hitl_approvals_logged -- every record has approver_id +
   approved_at + decision + reason_code.

## Verification (immutable)

    python scripts/compliance/finra_rationale_audit.py --sample 10 && \
    python -c "import json; r=json.load(open('handoff/finra_audit.json')); \
               assert r['sample_retrieval_success_rate'] == 1.0"

Plus: `python scripts/audit/finra_compliance_audit.py --check`.

## Anti-rubber-stamp

qa must:
- Confirm the 3y vs 6y gap is DISCLOSED in the artifact, not hidden.
- Verify round-trip test is real (not hardcoded equality).
- Check that HITL fields are REQUIRED in `write_rationale`
  signature -- a record missing approver_id should raise.
- Confirm GCS path is the production intent with local-dev
  fallback honestly labeled.

## References

- Researcher findings (Cycle 86 spawn, 16 URLs)
- Explore findings (Cycle 86 spawn)
- backend/services/paper_trader.py (trade_id + signals surface)
- backend/services/signal_attribution.py (agent rationale capture)
