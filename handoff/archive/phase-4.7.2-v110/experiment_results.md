# Experiment Results -- Cycle 88 / phase-4.9 step 4.9.0

Step: 4.9.0 Schema and file for immutable limits

## Gate flipped to approved

User directive "continue 4.9" recorded on masterplan
phase-4.9.gate: approved=true, approved_by="owner", timestamp set.

## Research-gate upheld

Spawned researcher (15 URLs: QuantConnect LEAN risk models + GitHub
source, SEC 15c3-1 Cornell LII, pydantic v2 ConfigDict, git signed
tags, Millennium/Citadel pod limits, AFML) + Explore (existing
portfolio_risk.py CVAR_LIMIT, kelly_allocator DEFAULT_CAP,
settings.py paper_daily_loss_limit_pct, docs/governance existing)
in parallel BEFORE writing the contract.

## What was generated

1. **NEW** `backend/governance/__init__.py` (package marker)
2. **NEW** `backend/governance/limits_schema.py`
   - `RiskLimits` pydantic v2 frozen model
     (`ConfigDict(frozen=True, extra="forbid")`), 6 fields with
     Field(gt=..., le=...) range validators.
   - `load()` wrapped in `@lru_cache(maxsize=1)` -- one parse per
     process, documented "DO NOT hot-reload" at module top.
   - `get_limits_digest()` returns 64-char SHA-256 of the YAML
     bytes for 4.9.2 boot-time fingerprint logging.
3. **NEW** `backend/governance/limits.yaml`
   Six limits with operator-visible banner + GPG-signed-tag
   requirement + institutional citation block:
   - max_position_notional_pct: 0.05
   - max_portfolio_leverage: 1.5
   - max_daily_loss_pct: 0.02
   - max_trailing_dd_pct: 0.10
   - max_gross_exposure_pct: 1.00
   - max_sector_weight_pct: 0.30
4. **NEW** `scripts/audit/immutable_limits_audit.py` with 7 teeth:
   file exists, schema validates, six exact fields, frozen
   enforcement, extra=forbid enforcement, load() cached, digest
   is 64-char hex.

## Verification (verbatim, immutable)

    $ python -c "from backend.governance.limits_schema import load; \
                  l=load(); \
                  assert l.max_position_notional_pct == 0.05 and \
                         l.max_portfolio_leverage == 1.5"
    exit=0

    $ python scripts/audit/immutable_limits_audit.py --check
    {"verdict": "PASS", all 7 teeth true}

## Success criteria

| Criterion | Result |
|-----------|--------|
| limits_file_exists | PASS |
| schema_validates | PASS |
| six_limits_present | PASS (exact set equality) |

## Mutation-resistance proven (harness-verifier)

Four independent mutations caught:
- out-of-range YAML value (2.0) -> range validator raises
- frozen=False in ConfigDict -> audit rc=1
- missing limit (comment out max_sector_weight_pct) -> rc=1
- (implicit) rogue extra field -> extra=forbid ValidationError

## Known limitations (tracked follow-up)

- Step 4.9.0 ships SCHEMA + FILE only. The limits are not yet
  READ by paper_trader / portfolio_manager / kill_switch; phase-
  4.9.2 startup loader + 4.9.3+ enforcement hooks wire them in.
- GPG-signed-tag CI enforcement is the deliverable of phase-4.9.1
  (next step).
- Existing runtime constants (`CVAR_LIMIT_PCT` in portfolio_risk,
  `paper_daily_loss_limit_pct` in settings) DUPLICATE some of
  these values. Cycle 88 does NOT remove them; the 4.9.2/4.9.3
  wiring step will consolidate to the single authoritative source.
