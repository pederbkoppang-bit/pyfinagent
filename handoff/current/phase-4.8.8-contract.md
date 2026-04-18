# Contract -- Cycle 85 / phase-4.8 step 4.8.8

Step: 4.8.8 Supply-chain hardening (pin + pip-audit cron)

## Hypothesis

phase-3.7 step 3.7.6 (Cycle 65) already pinned the 5 critical LLM
clients exactly in `backend/requirements.txt`. This cycle closes
the remaining supply-chain gaps:
1. Root `requirements.txt` that pip-audit can consume (masterplan's
   immutable command expects a repo-root file).
2. GitHub Actions weekly pip-audit workflow (cron + push triggers).
3. Audit verifying everything is wired.

## Scope

Files created:

1. **NEW** `requirements.txt` at repo root -- `-r backend/requirements.txt`
   include directive so pip-audit resolves the pinned LLM clients.

2. **NEW** `.github/workflows/pip-audit.yml`:
   - on: push (to main), pull_request, schedule (cron weekly)
   - sets up Python 3.14 + installs pip-audit
   - runs `pip-audit --requirement backend/requirements.txt --strict`
   - fails the workflow on any known vulnerability

3. **NEW** `scripts/audit/supply_chain_audit.py`
   Verifies:
   (a) root requirements.txt exists and references backend/requirements.txt
   (b) backend/requirements.txt still has the 5 exact-pinned LLM
       clients from cycle 65
   (c) .github/workflows/pip-audit.yml exists with pip-audit
       command + `--strict` + cron schedule
   (d) local pip-audit --requirement backend/requirements.txt --strict
       exits 0 (no known vulnerabilities)

## Immutable success criteria

1. llm_clients_pinned -- 5 clients still exact-pinned (preserved
   from cycle 65: anthropic, openai, google-cloud-aiplatform,
   fastmcp, alpaca-py).
2. pip_audit_in_ci -- `.github/workflows/pip-audit.yml` exists
   with the `pip-audit` command.
3. weekly_pip_audit_cron -- workflow has a `schedule:` key with
   a weekly cron expression.

## Verification (immutable, from masterplan)

    pip-audit --requirement requirements.txt --strict

Plus: `python scripts/audit/supply_chain_audit.py --check`.

## Anti-rubber-stamp

qa must verify:
- root requirements.txt is a REAL include (-r backend/...), not a
  stub that silently bypasses LLM pins.
- workflow ACTUALLY runs pip-audit on the backend requirements.
  A workflow that only echoed "ok" would pass the file-existence
  check but fail the command-content check.
- cron schedule is genuinely WEEKLY (not "once in 1970" or
  commented out).
- --strict flag is present so transitive vulnerabilities also fail.

## References

- OWASP SCVS supply-chain verification standard
- pip-audit docs (https://pypi.org/project/pip-audit/)
- LiteLLM March 2026 incident (cycle 65 motivation)
- .github/workflows/claude-code-review.yml (existing workflow
  pattern)
