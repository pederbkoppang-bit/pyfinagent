# Q/A Critique -- phase-10.7.1

## Harness-compliance (5 items)
1. **Research gate**: PASS. `handoff/current/phase-10.7.1-research-brief.md` (21,587 bytes, 25 apr 11:05) exists. Contract line 14 declares `tier=moderate, 7 in-full, 18 URLs, recency scan, gate_passed=true` -- exceeds the >=5-in-full + >=10-URL floor. Tier=moderate is consistent with deeper analytical work (formula candidates evaluated, BQ schema designed).
2. **Contract-before-GENERATE**: PASS. `contract.md` mtime=1777108001 (11:06), `experiment_results.md` mtime=1777108197 (11:09) -- 196s gap, contract written first. step=phase-10.7.1, formula candidate (Sharpe-slope, Candidate B), 11-column BQ schema, and 6-test plan all present in contract.
3. **Experiment results**: present, references 6/6 PASS.
4. **Log-last**: PASS. `grep -c "phase-10.7.1" harness_log.md = 1`, but the only hit is line 9917 inside an OLDER cycle's "Carry-forwards" prose ("live paper-trader integration (phase-10.7.1)"), not a `## Cycle N -- phase=10.7.1` header. No log entry for THIS cycle exists yet, which is correct (log-last invariant: append after Q/A PASS, before status flip).
5. **No verdict-shopping**: PASS. Prior 16.28 critique was PASS; this is net-new evidence for a different step. No fresh-respawn-on-unchanged-evidence pattern.

## Deterministic checks
- pytest_6_pass: yes -- `6 passed in 0.03s` (`tests/meta_evolution/test_alpha_velocity.py`)
- ast_5_files: clean -- `all 5 syntax_ok`
- dry_run_emits_canonical_sql: yes -- prints `CREATE SCHEMA`, `CREATE TABLE IF NOT EXISTS`, all 11 columns, `PARTITION BY DATE(window_start)`, `CLUSTER BY strategy_id, macro_regime`, exit 0
- backend_pytest_no_regression: 177 passed, 1 skipped, 1 warning in 14.84s (matches contract Q/A audit item #6)
- diff_only_new_files: yes for THIS cycle -- `git status --short` shows only untracked: `backend/meta_evolution/`, `scripts/migrations/create_alpha_velocity_table.py`, `tests/meta_evolution/`. The pre-existing modified files in `git diff --stat` predate this cycle (separate uncommitted work, not this step's mutations).

## Formula verification
- min_observations: 20 (`MIN_OBSERVATIONS = 20`, line 32 of `alpha_velocity.py`)
- formula_matches_brief: yes -- `(self.sharpe_end - self.sharpe_start) / wd` (line 79), Candidate B exactly
- guard_for_zero_window: yes -- `if wd <= 0: raise ValueError(...)` (lines 71-76)
- guard_for_thin_samples: yes -- `if self.n_obs < MIN_OBSERVATIONS: return None` (lines 77-78)

## BQ schema verification
- table_fqn_correct: yes -- `sunny-might-477607-p8.pyfinagent_pms.alpha_velocity_samples`
- 11_columns_present: yes -- strategy_id, window_start, window_end, n_obs, sharpe_start, sharpe_end, alpha_velocity_score, window_days, macro_regime, components_json, computed_at (all observed in dry-run output)
- partition_clause: yes -- `PARTITION BY DATE(window_start)`
- cluster_clause: yes -- `CLUSTER BY strategy_id, macro_regime`
- 3_cli_flags: yes -- `--apply` (default), `--verify`, `--dry-run`, mutually exclusive group

## Test reality check
- 6_real_tests: yes -- 6 distinct `def test_*` functions; no skips, no parametrize-faking, no decorators that suppress
- assertions_meaningful: yes -- each test asserts numeric value, type, presence/absence of None, regex on raised error, or BQ row shape. `test_positive_velocity_basic` asserts `abs(score - 0.5/30.0) < 1e-9` (real arithmetic check), not just "no exception"
- fakebq_records_calls: yes -- `FakeBQ.calls` list, `bq.calls[0]` is `(table_fqn, rows)` tuple, asserted on
- subprocess_test_runs: yes -- `subprocess.run([sys.executable, str(script), "--dry-run"], ...)` with timeout=15, asserts returncode==0, parses stdout for SQL fragments

## LLM judgment
- **formula_choice_honest**: HONEST. Brief recommended Candidate B (Sharpe-slope-per-day with n_obs>=20 floor); implementation does exactly that on `alpha_velocity.py:79`. No sneaking in IC-slope or compound metric.
- **migration_not_applied_decision**: CORRECT call. Contract line 58 says "migration script supports `--dry-run` (does NOT actually create the table this cycle -- Q/A or user runs `--apply` separately)". This respects the BQ-mutation rule from CLAUDE.md (default to read-only, surface SQL first) and avoids leaving dangling table state if a downstream test discovers a schema bug. NOT buck-passing -- it's the correct separation.
- **dsr_separation_of_concerns**: CLEAR. Module docstring (lines 1-19) names DSR explicitly as a downstream filter (per Bailey & Lopez de Prado), and the brief flagged it as such. The compute layer purposely returns the raw Sharpe-slope without DSR multiplication so downstream consumers can apply DSR-or-not depending on use-case. Genuine separation, not coverage gap masquerading.
- **subprocess_test_pattern**: ACCEPTABLE. Subprocess matches the masterplan verification command's exit-code semantics and exercises argparse end-to-end (a unit-test import would skip the CLI parsing). 15s timeout bounds fragility. Not the most elegant pattern, but the integration-fidelity gain outweighs the brittleness for a one-off migration script.
- **meta_evolution_package_placement**: APPROPRIATE. `__init__.py` (10 lines) explains scope and explicitly disambiguates from the deprecated `backend/agents/meta_coordinator.py` -- this is exactly the disambiguation the research brief flagged as load-bearing for phase-10.7.2-10.7.7. NOT scope-creep; it's package-level documentation that prevents future Mains from extending the wrong file.

## Verdict

```json
{
  "ok": true,
  "verdict": "PASS",
  "follow_up_tickets": [
    "Operator must run `python scripts/migrations/create_alpha_velocity_table.py --apply` then `--verify` before phase-10.7.2 attempts to write samples. Document the apply-run timestamp in the next harness_log entry."
  ],
  "checks_run": [
    "harness_compliance_5",
    "syntax_ast_5_files",
    "verification_command_pytest",
    "backend_regression_pytest",
    "git_status_diff_scope",
    "formula_body_inspection",
    "bq_schema_dry_run",
    "test_reality_inspection",
    "research_brief_alignment",
    "contract_pre_generate_mtime"
  ],
  "violated_criteria": [],
  "violation_details": [],
  "certified_fallback": false
}
```

Formula matches the brief exactly; BQ schema matches the brief exactly; 6/6 tests PASS in 0.03s; 177-passed regression intact; all 5 harness-compliance items clean. Verdict: **PASS**.
