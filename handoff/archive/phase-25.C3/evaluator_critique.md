---
step: phase-25.C3
cycle: 72
cycle_date: 2026-05-12
verdict: PASS
qa_spawn: 1
violated_criteria: []
checks_run: ["harness_compliance_audit", "syntax", "verification_command", "behavioral_round_trips", "mutation_review", "scope_honesty"]
---

# Q/A Critique -- phase-25.C3 (first spawn)

## 5-item harness-compliance audit

1. **Researcher spawn for 25.C3** -- CONFIRM. `handoff/current/research_brief.md`
   header reads `phase-25.C3`; JSON envelope at EOF reports
   `external_sources_read_in_full=7`, `urls_collected=17`,
   `recency_scan_performed=true`, `gate_passed=true`. Three-variant search
   discipline visible.
2. **Contract pre-commit** -- CONFIRM. `handoff/current/contract.md` is
   phase-25.C3 with all three immutable criteria copied verbatim from
   masterplan (`strategy_registry_table_has_status_field_active_shadow_retired`,
   `actual_replacement_no_longer_hardcoded_false`,
   `monthly_approval_flips_status_from_shadow_to_active`).
3. **Results captured** -- CONFIRM. `experiment_results.md` lists code
   changes (4 files), verbatim verifier output (12/12 PASS), and four
   behavioral round-trips. AST gates documented.
4. **Log-last discipline** -- CONFIRM. `grep -c "phase=25.C3"
   handoff/harness_log.md` returns 0; the log append is correctly held
   until after this PASS verdict.
5. **No verdict-shopping** -- CONFIRM. First Q/A spawn for 25.C3 (no
   prior critique entry, no prior log entries for the step-id).
   3rd-CONDITIONAL counter is 0.

All five CONFIRM. Proceed.

## Deterministic checks (verbatim)

### Verification command

```
$ source .venv/bin/activate && python3 tests/verify_phase_25_C3.py
PASS: settings_real_capital_enabled_defaults_false
PASS: bq_update_promoted_strategy_status_signature
PASS: bq_update_uses_parameterized_sql_and_timeout_30
PASS: run_monthly_sortino_gate_accepts_real_capital_enabled_kwarg
PASS: actual_replacement_no_longer_hardcoded_false
PASS: deployment_log_notes_uses_derived_actual_replacement
PASS: record_approval_accepts_status_update_fn_kwarg
PASS: monthly_approval_flips_status_from_shadow_to_active
PASS: record_approval_rejection_does_not_flip_status
PASS: behavioral_actual_replacement_derived_from_real_capital_flag
PASS: behavioral_bq_update_status_round_trip
PASS: strategy_registry_table_has_status_field_active_shadow_retired

12/12 claims PASS, 0 FAIL
EXIT=0
```

### AST gates

- `backend/config/settings.py` -- OK
- `backend/db/bigquery_client.py` -- OK
- `backend/autoresearch/monthly_champion_challenger.py` -- OK
- `tests/verify_phase_25_C3.py` -- OK

### Spot-checks of wiring (file:line evidence)

- `backend/config/settings.py:145` -- `real_capital_enabled: bool =
  Field(False, ...)`. Default False: **SR 11-7 paper-only invariant
  preserved by default**.
- `backend/db/bigquery_client.py:762` -- `def
  update_promoted_strategy_status(` exists.
- `backend/autoresearch/monthly_champion_challenger.py:60` --
  `real_capital_enabled: bool = False,` kwarg on
  `run_monthly_sortino_gate`.
- `:81` -- `actual_replacement = bool(real_capital_enabled)`
  (derivation, no hardcoded False literal in result dict).
- `:199` -- `"actual_replacement": actual_replacement` snapshotted
  into state row (closes fire-to-approval drift gap).
- `:226` -- `status_update_fn: Callable[[str, str], None] | None =
  None` kwarg on `record_approval`.
- `:269-274` -- approval branch invokes
  `status_update_fn(challenger_id, "active")` inside try/except
  (fail-open).
- `:293, 307` -- deployment-log notes interpolates the snapshotted
  `actual_replacement` value; literal `actual_replacement=False` is
  gone.

## Per-criterion LLM judgment

### Criterion 1: `strategy_registry_table_has_status_field_active_shadow_retired`

PASS. The 25.A3 schema (`status STRING` with documented enum
`pending|active|paused|superseded|rolled_back`) is a strict superset
of the masterplan's informal `active_shadow_retired` triad. The
mapping (shadow -> pending; active -> active; retired -> superseded)
is documented in both `research_brief.md` (Pitfall #4 + the Status
name reconciliation table) and `contract.md` (Research-gate key
conclusions). Claim 12 verifies the migration names contain `active`
plus at least one retired-equivalent. No schema change needed -- this
is correctly scoped as a documentation + helper-existence claim, not
a DDL change.

### Criterion 2: `actual_replacement_no_longer_hardcoded_false`

PASS. Three independent checks confirm:
- **Claim 5**: result-dict literal `"actual_replacement": False`
  absent; verified by grep.
- **Claim 6**: `_emit_deployment_log_row` notes literal
  `actual_replacement=False` replaced with f-string interpolation
  reading the snapshotted state-row value.
- **Claim 10 (behavioral)**: gate-fire with
  `real_capital_enabled=True` -> `actual_replacement == True`; with
  `=False` (the default) -> `False`. This proves derivation, not
  just literal removal.
- **SR 11-7 invariant preserved**: both
  `Settings.real_capital_enabled` and the kwarg default are False,
  so behavior is byte-identical to pre-25.C3 unless an operator
  explicitly toggles the env var.

### Criterion 3: `monthly_approval_flips_status_from_shadow_to_active`

PASS. Three claims cover the behavioral surface:
- **Claim 8**: approval round-trip calls `status_update_fn` exactly
  once with `(challenger_id, "active")` -- wires fire on approval.
- **Claim 9**: rejection round-trip does NOT call
  `status_update_fn` -- asymmetry proven.
- **Claim 11**: direct test of `update_promoted_strategy_status`
  with fake BQ client asserts parameterized SQL pattern + `result(
  timeout=30)` -- BQ side proven.
- Fail-open try/except at lines 269-274 is consistent with the
  existing `bq_fn` contract.

## Anti-rubber-stamp mutation review

| Mutation | Claim that catches it | Verdict |
|---|---|---|
| Settings flag default flipped to True | claim 1 | covered |
| `actual_replacement = bool(real_capital_enabled)` -> `True` literal | claim 5 + 10 | covered |
| Skip `status_update_fn(...)` invocation on approval | claim 8 | covered |
| Invoke `status_update_fn` on rejection too | claim 9 | covered |
| Hardcode `actual_replacement=False` in deployment-log notes | claim 6 | covered |
| Drop `result(timeout=30)` from BQ UPDATE | claim 3 | covered |
| Drop parameterization (string-interpolate user input) | claim 3 | covered |
| Pass `"pending"` instead of `"active"` to `status_update_fn` | claim 8 (asserts call_args == (challenger_id, "active")) | covered |
| Snapshot drift (gate-fire stores True, approval reads False) | claim 6 reads snapshotted state-row value | covered |
| Schema-rename `status` -> `state` | claim 3 SQL grep `SET status = @new_status` | covered |

No spirit-breaking non-covered mutation identified.

## Scope honesty

CONFIRM. The contract's "Non-goals" section explicitly states "No
prior-active-row supersession (deferred per research finding 8)" and
research_brief.md key finding #8 lays out the rationale (criterion 3
says nothing about flipping the prior champion to `superseded`; the
existing reader naturally returns the new active row after the flip).
The downstream 25.R step is the right place to layer prior-row
supersession.

No overclaim in `experiment_results.md`: the live-check section
correctly states evidence is "pending in
`handoff/current/live_check_25.C3.md` after operator" runs the 4-step
BQ round-trip.

## Verdict

**PASS** (first spawn).

```json
{
  "ok": true,
  "verdict": "PASS",
  "reason": "All 3 immutable criteria met. Deterministic verifier 12/12 PASS, exit=0. 4 behavioral round-trips (approval-flip, rejection-no-flip, derived actual_replacement, BQ UPDATE round-trip) exercise real functions with fakes. SR 11-7 paper-only invariant preserved (real_capital_enabled defaults False at both Settings and kwarg). Scope honest: prior-active-row supersession deferred to 25.R explicitly.",
  "violated_criteria": [],
  "violation_details": [],
  "certified_fallback": false,
  "checks_run": ["harness_compliance_audit", "syntax", "verification_command", "behavioral_round_trips", "mutation_review", "scope_honesty"]
}
```

Ready to append harness_log.md and flip phase-25.C3 status -> done.
