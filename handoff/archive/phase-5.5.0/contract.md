# Sprint Contract -- Cycle 45 / phase-5.5 step 5.5.1

Research gate: done (researcher agent; 14 URLs; NIST SP 800-30 1-5
ordinal, freshness from quant-data-vendor consensus, coverage from
QuantConnect + S&P Global, graduated SPOF per CMS, license per Open
Definition / Open Data Commons).

Rubric (per-axis 1-5; higher is better; SPOF is resilience not risk):
 - cost        : 5=free, 4=<$100/mo, 3=<$500, 2=<$2k, 1=>$2k
 - freshness   : 5=<5s, 4=<5min, 3=<1h, 2=daily/EOD, 1=>=7d
 - coverage    : 5=>=5000 US, 4=500-5000, 3=100-500, 2=<100, 1=single-name
 - spof        : 5=multi-provider_in_stack, 4=provider_multi-region,
                 3=documented_failover, 2=single_region_sla, 1=no_sla
 - license     : 5=public-domain/CC0, 4=paid_standard_saas, 3=paid_restrictive,
                 2=free_non-commercial, 1=ambiguous/prohibits_algo

Success criteria (immutable):
 - exit code 0
 - current_state.json parses with json.loads
 - every provider has cost, freshness, coverage, spof, license fields

Verification command (immutable):
 python3 scripts/audit/score_current_state.py \
   --input backend/data_audit/inventory.json \
   --output backend/data_audit/current_state.json

Plan:
 1. Copy handoff/data_sources_inventory.json to
    backend/data_audit/inventory.json (the immutable verification
    references that path; keep the two in sync).
 2. Write scripts/audit/score_current_state.py with a per-provider
    hard-coded rubric dict (auditable; can diff in PRs). Emits
    current_state.json with every provider carrying the 5 scalar
    fields + an `aggregate` score.
 3. Run verification.
 4. EVALUATE via qa-evaluator + harness-verifier.
 5. LOG + mark done.
