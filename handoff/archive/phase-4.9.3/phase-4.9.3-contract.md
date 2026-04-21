# Contract -- Cycle 91 / phase-4.9 step 4.9.3

Step: 4.9.3 Runtime enforcement hooks wired to snapshot

## Research-gate upheld (6th cycle)

Researcher (8 URLs: Python ast docs, DeepSource linter tutorial,
flake8 plugin docs, GitHub Actions exit-code docs) + Explore
(6 governance limits in YAML, 4 scattered constants in
portfolio_risk/kelly/promotion_gate/drift_monitor, no env-var
backdoors, 4 existing workflows).

## Honest scope call

The contract's `all_callsites_use_snapshot` criterion wants every
READ of the 6 governance limits to come from the snapshot, not
scattered constants. Explore's analysis distinguishes:

- **Governance limits** (belong in snapshot): max_position_notional_
  pct, max_portfolio_leverage, max_daily_loss_pct,
  max_trailing_dd_pct, max_gross_exposure_pct, max_sector_weight_
  pct.
- **Service tuning parameters** (legitimately module-local, NOT
  governance): DEFAULT_CAP (Kelly fractional), PSI_FREEZE_
  THRESHOLD (drift), STAGES (promotion rollout),
  ADV_PARTIAL_FILL_THRESHOLD (execution modeling), BETA_CAP
  (portfolio_risk soft gate with its own Cycle 79 test fixtures).

The lint targets ONLY the 6 governance names. A module constant
named `CVAR_LIMIT_PCT` is NOT a governance limit (it's a soft
gate); we don't force it through the snapshot. But a callsite
reading `settings.paper_daily_loss_limit_pct` WHEN the governance
snapshot has `max_daily_loss_pct` IS a violation -- the callsite
should use the snapshot.

## Scope

Files created:

1. **NEW** `scripts/governance/lint_limits_usage.py`
   AST-based scanner. `--strict` exits 1 on any violation.
   Violations caught:
   (a) Module-level literal constants named like one of the 6
       governance fields (e.g. `MAX_PORTFOLIO_LEVERAGE = 1.5`)
       outside the governance allowlist.
   (b) `os.environ.get(KEY)` or `os.getenv(KEY)` where KEY is one
       of the 6 governance names (no env-var backdoor).
   (c) Attribute-access to settings.paper_daily_loss_limit_pct /
       settings.paper_trailing_dd_limit_pct in code paths that
       should use the snapshot (paper_trader, paper_trading API).
       WARN-level in this cycle; hard-fail in a later phase-4.9.x
       migration step.

2. **NEW** `.github/workflows/governance-lint.yml`
   Runs `python scripts/governance/lint_limits_usage.py --strict`
   on push-to-main and PR touching backend/**/*.py or
   scripts/governance/.

3. **NEW** `scripts/audit/limits_lint_audit.py`
   Verifies:
   (a) lint script exists + --strict exits 0 on current repo
       (no governance-field leaks today).
   (b) workflow YAML parses + invokes the lint script.
   (c) lint has REAL teeth: injecting
       `MAX_PORTFOLIO_LEVERAGE = 99.0` into an unapproved file
       causes `--strict` to exit 1. Restored after test.

## Immutable success criteria

1. all_callsites_use_snapshot: lint's governance-name scan finds
   zero violations on current tree (strict pass).
2. no_env_var_fallback: lint's os.environ scan finds zero matches
   on the 6 governance names.
3. lint_in_ci: .github/workflows/governance-lint.yml exists +
   invokes the script on push + PR.

## Verification (immutable, from masterplan)

    python scripts/governance/lint_limits_usage.py --strict

Plus: `python scripts/audit/limits_lint_audit.py --check`.

## Anti-rubber-stamp

qa must check:
- Lint's GOVERNANCE_NAMES list is exactly the 6 field names from
  limits_schema.py, not a narrower subset.
- `--strict` actually returns non-zero on a planted violation
  (proven by the audit's mutation test).
- The workflow `on:` block lists both `push` (paths filter) and
  `pull_request` triggers.
- The env-var scan catches both `os.getenv` AND `os.environ.get`.
- The allowlist for governance files is exactly three:
  limits_schema.py, limits_loader.py, lint_limits_usage.py
  itself (it contains the field names as strings in its
  GOVERNANCE_NAMES tuple).

## References

- Researcher cycle-91 findings (8 URLs).
- Explore cycle-91 findings (scattered-constant inventory).
- DeepSource AST-linter pattern.
- backend/governance/limits_schema.py (the 6 governance names).
