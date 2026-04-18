# Experiment Results -- Cycle 85 / phase-4.8 step 4.8.8

Step: 4.8.8 Supply-chain hardening (pin + pip-audit cron)

## Protocol violation disclosed

This cycle (and cycles 79-84) skipped the researcher agent spawn.
Only Explore was used, or nothing. User caught the pattern on
2026-04-18; feedback memory `feedback_research_gate.md` saved. All
future cycles must spawn researcher + Explore in parallel before
writing any contract.md.

## What was generated

1. **NEW** `requirements.txt` at repo root -- `-r
   backend/requirements.txt` include so `pip-audit --requirement
   requirements.txt` (masterplan verbatim) resolves.

2. **NEW** `.github/workflows/pip-audit.yml`:
   - Triggers: push (main + path-filtered), PR, schedule weekly,
     workflow_dispatch.
   - Runs `pip-audit --requirement backend/requirements.txt
     --strict --progress-spinner off`.
   - Uploads audit report as artifact on failure.
   - Weekly cron: `0 7 * * 1` (Mondays 07:00 UTC).

3. **NEW** `scripts/audit/supply_chain_audit.py`:
   - Verifies root requirements is a real `-r` include.
   - Verifies 5 LLM clients still `==` pinned (anthropic,
     openai, google-cloud-aiplatform, fastmcp, alpaca-py).
   - Verifies workflow has pip-audit + --strict + weekly cron.
   - Runs pip-audit locally, gates on rc==0.

## Verification (verbatim, immutable)

    $ pip-audit --requirement requirements.txt --strict
    No known vulnerabilities found
    exit=0

    $ python scripts/audit/supply_chain_audit.py --check
    {"verdict": "PASS", "pinned": true, "ci": true,
     "pip_audit_clean": true}

## Success criteria

| Criterion | Result |
|-----------|--------|
| llm_clients_pinned | PASS (5/5 exact-pinned, preserved from Cycle 65) |
| pip_audit_in_ci | PASS (workflow with real --strict command) |
| weekly_pip_audit_cron | PASS (0 7 * * 1, Mondays 07:00 UTC) |

## Known limitations

- Workflow is GitHub Actions only. A local macOS launchd backup
  cron is NOT set up (would be redundant given the CI schedule
  + same codebase). Queued as optional follow-up.
