# Contract — phase-8.5.9 REMEDIATION v1
5 sources in full (Google SRE postmortem × 2, OpenAI Cookbook self-evolving agents, Karpathy 2019 recipe, MLMastery seed-failure). Bucket-first ordering is canonical. gate_passed: true.

Pitfall flagged: `seed_target` extraction at script:34 truncates at first newline; OK for current postmortem.

Immutable: `test -f handoff/virtual_fund_postmortem.md && python scripts/harness/autoresearch_seed_from_postmortem.py --dry-run` exit 0.
