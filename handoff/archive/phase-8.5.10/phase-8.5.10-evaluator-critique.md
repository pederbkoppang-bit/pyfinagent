# qa_8510_v1 — PASS

TrialLedger records abandoned trials (n_abandoned field). meta_dsr penalty grows with N (fix disclosed: initial decay-formula replaced with `0.1*sqrt(log(N))`). required_dsr step-ups at N=50 boundary (0.95 -> 0.99). cpcv_applied_on correctly gates promoted-only, vacuously true for non-promoted or abandoned. 4/4 criteria PASS. PASS.
