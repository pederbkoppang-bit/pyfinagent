# Live-check placeholder -- phase-25.B3

**Step:** 25.B3 -- Daily loop reads latest promoted strategy via load_promoted_params()
**Date:** 2026-05-12

## Live-check field (per masterplan)
> "BQ promoted_strategies query returns active row; autonomous_loop log confirms params merged"

## Pre-deployment evidence
- 11/11 verifier PASS (`source .venv/bin/activate && python3 tests/verify_phase_25_B3.py`)
- 5 behavioral round-trips:
  - Happy path: fake bq returns row with params -> `load_promoted_params` returns those params + emits expected log line.
  - Empty path: fake bq returns None -> falls back to `load_best_params()` + log line `"No active promoted strategy in BQ"`.
  - Exception path: fake bq raises RuntimeError -> falls back + log line `"Promoted strategy BQ unavailable"` with exception detail.
  - JSON round-trip: reader inverts `TO_JSON_STRING(params)` via `json.loads` and pops `params_json` from return dict.
  - Malformed-JSON safe fallback: reader returns `params={}` (no exception).
- Backend AST clean for both touched files.

## Post-deployment operator workflow
1. (Prerequisite) Apply 25.A3 migration -- already covered in `live_check_25.A3.md`. Requires operator `--apply`.
2. Trigger a Friday promotion (real or simulated) with a non-None `bq_client` so a row lands in `promoted_strategies`.
3. Trigger the next daily cycle:
   ```
   curl -s -X POST http://localhost:8000/api/paper-trading/run-now \
     -H "Authorization: Bearer $TOKEN"
   ```
4. Verify the autonomous loop log shows the promoted params merge. Expected line (case-sensitive search):
   ```
   Loaded promoted params (DSR <dsr> week=<week_iso>): [<param keys...>]
   ```
5. Cross-check via BQ that the same row exists:
   ```sql
   SELECT strategy_id, week_iso, status, dsr, allocation_pct, promoted_at
   FROM `sunny-might-477607-p8.pyfinagent_data.promoted_strategies`
   WHERE status IN ('pending', 'active')
   ORDER BY promoted_at DESC, dsr DESC
   LIMIT 1;
   ```

## Fallback behavior (verified by verifier behavioral tests)
- BQ unavailable (network/auth/table-not-found): log warning `"Promoted strategy BQ unavailable, falling back to optimizer_best: <exc>"`, cycle continues using `optimizer_best.json`.
- BQ returns no row: log info `"No active promoted strategy in BQ, falling back to optimizer_best"`, cycle continues using `optimizer_best.json`.
- `optimizer_best.json` missing: returns `{}`; cycle still completes (existing Tier-3 behavior unchanged).

## Downstream
Unblocks **25.C3** (status state-machine; `pending` -> `active` flip + supersession), which unblocks **25.R** (auto-switching policy / red-line goal-c).

**Audit anchor for next bucket:** 25.C3 status state-machine.
