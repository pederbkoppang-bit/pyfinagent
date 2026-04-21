# Contract — phase-8.5.10 REMEDIATION v1
Researcher confirms `0.1*sqrt(log N)` is defensible (actually strict). Raw `sqrt(2 log N)` as subtracted penalty is mathematically wrong (>1 at N=8). Canonical DSR uses it as threshold, not penalty. 5 sources in full. gate_passed: true.

Immutable: `python scripts/harness/autoresearch_meta_dsr_test.py` exit 0 + 4/4 PASS.
