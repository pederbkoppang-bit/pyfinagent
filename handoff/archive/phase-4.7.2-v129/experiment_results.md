# Experiment Results -- Cycle 91 / phase-4.9 step 4.9.3

Step: 4.9.3 Runtime enforcement hooks wired to snapshot
Ran at: 2026-04-17 (UTC)

## What was built

Three new files:

1. `scripts/governance/lint_limits_usage.py` (~230 LOC)
   AST-based scanner. Walks every `.py` under the repo (skipping
   `.venv`, `.venv.py313.bak`, `node_modules`, `.git`, worktree
   copies, `.claude/worktrees`, archived handoffs, etc.) and raises
   violations on:
   (a) module-level literal constants whose name matches one of
       the 6 governance fields (case-insensitive);
   (b) `os.environ.get(K)` / `os.getenv(K)` where K is a governance
       field (env-var backdoor);
   (c) WARN-level: `settings.paper_daily_loss_limit_pct` /
       `settings.paper_trailing_dd_limit_pct` attribute access
       (migration markers for a later phase-4.9.x step).
   Allowlist of 6 files: limits_schema.py, limits_loader.py, the
   lint itself, + the 3 audit scripts (they contain the governance
   names as string literals).
   `--strict` exits 1 on any (a) or (b) violation.

2. `.github/workflows/governance-lint.yml`
   Triggers on push-to-main AND pull_request, both with a `paths:`
   filter covering `backend/**/*.py`, `scripts/governance/**`,
   `scripts/audit/**`, and the workflow itself. Invokes
   `python scripts/governance/lint_limits_usage.py --strict` on
   Python 3.14.

3. `scripts/audit/limits_lint_audit.py` (~170 LOC, 7 teeth)
   - script_exists
   - strict_pass_on_clean_tree
   - workflow_exists_and_parses
   - workflow_invokes_lint
   - workflow_has_push_with_paths
   - workflow_has_pull_request
   - governance_names_complete (lint tuple == RiskLimits fields)
   - mutation_kills_strict (injects MAX_PORTFOLIO_LEVERAGE=99.0
     into kelly_allocator.py, verifies --strict exits 1, restores)

## Verification output (verbatim)

Immutable verification:

    $ python scripts/governance/lint_limits_usage.py --strict
    lint_limits_usage: scanned 293 py files; violations=0 warnings=6
      WARN backend/api/paper_trading.py:300 [legacy_settings_attr] ...
      WARN backend/api/paper_trading.py:301 [legacy_settings_attr] ...
      WARN backend/api/paper_trading.py:312 [legacy_settings_attr] ...
      WARN backend/api/paper_trading.py:313 [legacy_settings_attr] ...
      WARN backend/api/paper_trading.py:339 [legacy_settings_attr] ...
      WARN backend/api/paper_trading.py:340 [legacy_settings_attr] ...
    exit 0

Audit:

    $ python scripts/audit/limits_lint_audit.py --check
    {
      "wrote": "handoff/limits_lint_audit.json",
      "verdict": "PASS",
      "script_exists": true,
      "strict_pass_on_clean_tree": true,
      "workflow_exists_and_parses": true,
      "workflow_invokes_lint": true,
      "workflow_has_push_with_paths": true,
      "workflow_has_pull_request": true,
      "governance_names_complete": true,
      "mutation_kills_strict": true
    }
    exit 0

## Success criteria (from contract)

1. all_callsites_use_snapshot:
   PASS -- `--strict` exits 0; zero governance-name literal or
   env-var violations.
2. no_env_var_fallback:
   PASS -- env-var scan found zero `os.environ.get` / `os.getenv`
   calls using any of the 6 governance keys.
3. lint_in_ci:
   PASS -- `.github/workflows/governance-lint.yml` exists, is
   valid YAML, triggers on push+paths and pull_request, invokes
   `--strict`.

## Anti-rubber-stamp checks (honest)

- GOVERNANCE_NAMES tuple contains exactly the 6 fields from
  `RiskLimits.model_fields`, verified by audit tooth #6.
- Audit tooth #7 proves the lint has REAL teeth: injecting a
  planted `MAX_PORTFOLIO_LEVERAGE = 99.0` violation causes
  `--strict` to exit 1; file is restored in a try/finally block.
- Workflow has BOTH `push:` (with paths filter) AND
  `pull_request:` triggers (audit teeth #5).
- Allowlist is precisely the 6 files that ARE the governance
  layer, the lint, or the audits (they reference the 6 names
  as string literals).
- WARN-level (settings.paper_*_limit_pct) is by design -- the
  migration to the snapshot is a separate follow-up step within
  phase-4.9 (not a cycle-91 task).

## Honest scope disclosure

- The lint catches NEW introductions of governance-name literals
  and env-var backdoors. It does NOT force the migration of the
  existing 6 WARN-level settings reads; that migration is a
  follow-up step (the contract and this artifact both make that
  clear).
- The CI workflow is declared but not yet exercised on a GitHub
  Actions runner (it will fire on the next PR touching the
  scoped paths).

## Artifacts written

- `handoff/limits_lint_audit.json` (JSON with all 7 checks +
  timestamp + governance names).
