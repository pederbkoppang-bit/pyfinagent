# Contract -- Cycle 88 / phase-4.9 step 4.9.0

Step: 4.9.0 Schema and file for immutable limits

## Gate status

Owner approved via "continue 4.9" directive (2026-04-18).
masterplan.json phase-4.9.gate flipped to approved=true with
timestamp + approver note.

## Research-gate upheld

Spawned researcher (15 URLs: QuantConnect LEAN, SEC 15c3-1, AFML,
pydantic v2 ConfigDict, git signed tags, Millennium/Citadel pod
limits) + Explore in parallel before contract.

## Hypothesis

Ship a pydantic v2 frozen model + YAML file + singleton loader
satisfying the masterplan's verification:

    python -c "from backend.governance.limits_schema import load; \
               l=load(); assert l.max_position_notional_pct == 0.05 \
               and l.max_portfolio_leverage == 1.5"

Six canonical limits (converged across researcher + Explore):
1. `max_position_notional_pct = 0.05` (5% per name)
2. `max_portfolio_leverage = 1.5` (gross leverage cap)
3. `max_daily_loss_pct = 0.02` (daily loss kill-switch)
4. `max_trailing_dd_pct = 0.10` (trailing drawdown kill-switch)
5. `max_gross_exposure_pct = 1.00` (long-only fully invested)
6. `max_sector_weight_pct = 0.30` (single-sector concentration)

Immutability enforced via:
- pydantic v2 `model_config = ConfigDict(frozen=True, extra="forbid")`
  -> mutation attempt raises `ValidationError`; extra fields raise.
- `@lru_cache(maxsize=1)` on `load()` -> one parse per process.
- Explicit "DO NOT hot-reload" comment at module top.
- YAML file with banner warning + tag-signed-commit note.

## Scope

Files created:

1. **NEW** `backend/governance/__init__.py` -- package marker.
2. **NEW** `backend/governance/limits_schema.py` -- RiskLimits
   pydantic model + `load()` + `get_limits_digest()` (SHA-256 of
   the raw yaml file bytes; used by 4.9.2 startup loader).
3. **NEW** `backend/governance/limits.yaml` -- the six limits with
   banner warnings.
4. **NEW** `scripts/audit/immutable_limits_audit.py` -- verifies:
   - file exists, parses, validates
   - frozen behavior: `l.max_position_notional_pct = 0.99` raises
   - extra fields forbidden: inserting `rogue_field: 1.0` at parse
     time raises
   - `load()` is cached (second call returns SAME object by id)
   - all 6 required fields present with expected values
   - each field is a float in (0, 1] (sanity-ranged)

## Immutable success criteria

1. limits_file_exists -- `backend/governance/limits.yaml` present.
2. schema_validates -- `load()` returns a RiskLimits without errors.
3. six_limits_present -- all six named fields parse.

## Verification (immutable, from masterplan)

    python -c "from backend.governance.limits_schema import load; l=load(); assert l.max_position_notional_pct == 0.05 and l.max_portfolio_leverage == 1.5"

Plus: `python scripts/audit/immutable_limits_audit.py --check`.

## Anti-rubber-stamp

qa must verify:
- frozen is REAL (attempting mutation raises). A regression to
  a plain dataclass without frozen=True would NOT raise.
- extra="forbid" is REAL (a YAML with an extra key fails to load).
- load() caches (id(l1) == id(l2) for two sequential calls).
- YAML file has a human-visible banner warning, not just a
  comment somewhere random.
- Six fields exactly, no more and no less (guards against
  accidental limit-expansion via typo).

## References

- Researcher cycle-88 findings (15 URLs).
- QuantConnect LEAN risk models (defaults).
- pydantic v2 ConfigDict docs.
- SEC 15c3-1 haircut thresholds.
- Millennium/Citadel pod-level drawdown practice.
