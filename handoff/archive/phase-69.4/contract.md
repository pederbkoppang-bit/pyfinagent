# Contract — Step 69.4 (P2 hand-offs: route every confirmed audit defect to an owner)

- **Phase / step**: phase-69 → 69.4
- **Date**: 2026-07-11
- **Type**: HAND-OFFS / documentation. NO code execution. Zero live surface.
- **Boundaries**: $0 metered; no execution of any target phase (68.x / 61.x / 63.x); coverage must be exhaustive (no confirmed finding silently dropped).

## Research-gate summary

Researcher spawned before this contract. Brief: `handoff/current/research_brief_69.4.md` —
**gate_passed: true**, 5 external sources read in full (Origami Risk, Hyperproof, Webomates defect-triage,
TrustCloud, Stell RTM — all converge on: assign an OWNER to every finding, risk-triage, track-to-closure,
"nothing falls through the cracks", bidirectional traceability). The internal deliverable — a disposition
map for ALL 50 confirmed findings + an exhaustive 1..50 subsystem checksum (zero silent drops) + routing-
target verification — was authored by the researcher before it stalled on the external half (7th subagent
stall); Main read the 5 external sources + finalized the envelope.

**Routing targets verified to EXIST (all pending steps)**: 68.4 (learn-loop), 68.5 (fill-price gate),
68.6 (go-live tracker), 61.3 (money-display/currency), 63.3 (verified defect register). No hand-off files
into a void.

## Hypothesis

Every confirmed audit finding can be mapped to exactly one disposition — fixed-in-69.x, owned-by-an-in-flight-
69.x-step, filed-to-a-named-existing-owner-phase, or explicitly deferred-with-reason to the 63.3 catch-all —
such that a single coverage table + an exhaustive 1..50 checksum proves no confirmed defect is silently
dropped, with zero code execution.

## Immutable success criteria (verbatim from `.claude/masterplan.json` phase-69 → 69.4)

1. Learn-loop tz TypeError (outcome_tracker.py:50 / :118) filed as a hand-off to phase-68.4 (no 68.x execution in this step).
2. External-flow/deposit Sharpe corruption (perf_metrics.py:116) + get_paper_trades_in_window STRING/TIMESTAMP query bug (bigquery_client.py:957) filed to their phase-68.5 / 68.6 owners (no execution).
3. FX-1 residual root-cause handed to parked 61.3 (no 61.x execution).
4. The 30 contested findings + all Slack/UI display defects (formatters.py:247, _production_fns.py, cockpit-helpers.tsx, live-portfolio-context.tsx et al.) filed as 63.3-style seed entries, each with location + claim + verifier split (no 63.3 execution).
5. A coverage table maps every one of the 50 confirmed findings to a disposition {fixed-in-69.x | filed-to-<owner> | deferred-<reason>} so no confirmed defect is silently dropped. Fresh Q/A PASS.

## Plan (GENERATE)

Author `handoff/current/audit_phase69/handoffs_69.4.md` containing:
1. The full coverage table (all 50 confirmed findings → disposition, from the brief's disposition map) + the exhaustive 1..50 subsystem checksum.
2. Seed entries per owner: 68.4 (findings 12/15), 68.5+68.6 (13/38, + the 68.6-recommended 22/2/39), 61.3 (6/14), 63.3 (the 9 Slack/UI display defects + the 30 contested + the residual-19 with P-levels), each with location + claim (+ verifier split for contested).
3. Acknowledge FO-69.2-A (per-ticker FFD) already filed in `followons_69.2.md`.
4. A short note flagging the 3 residuals with a more-specific owner (22→68.6; 2/39→deposit cluster) and the money-ledger atomicity cluster (5/7/37) as recommendations for Main/operator (69.4 files them to 63.3 by default; no execution).

Then `experiment_results.md` (verification command output + no-execution proof) and a fresh Q/A (Workflow structured-output path).

## References
- `handoff/current/research_brief_69.4.md` (disposition map + 5 external sources).
- `handoff/current/audit_phase69/register.md` (50 confirmed / 30 contested / 4 refuted).
- `handoff/current/audit_phase69/followons_69.2.md` (FO-69.2-A).
- External: Origami Risk, Hyperproof, Webomates, TrustCloud, Stell (audit-remediation / defect-triage / traceability).
