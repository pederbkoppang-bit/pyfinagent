# Sprint Contract — phase-8.5 / 8.5.5 REMEDIATION v1 (full-breach)

**Tier:** moderate. Fresh Researcher + Fresh Q/A.

## Research-gate (grounded)
5 sources in full incl Wikipedia DSR, Wikipedia Purged CV, Balaena Quant, insightbig CPCV, Towards AI CPCV. DSR>=0.95 threshold grounded (95% confidence DSR is a CDF value; "strong evidence against noise"). PBO<=0.20 is conservative convention, not peer-reviewed standardized. CPCV C(6,2)=15 math confirmed. Conjunction posture correct (DSR + PBO measure orthogonal failure modes: statistical reality vs parameter overfit).

## Immutable
`python scripts/harness/autoresearch_gate_test.py` exit 0 + 4/4 PASS.

## Plan
Re-run test. Spawn fresh Q/A. Log.

## Advisory from researcher (to surface in Q/A)
- PBO<=0.20 is stricter than canonical (no single paper mandates 0.20). Defensible as safer; worth a note in the gate docstring.
