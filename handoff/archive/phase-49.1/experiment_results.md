# Experiment results -- phase-49.1: Runtime risk-limit control endpoint

**Date:** 2026-05-29 | **Result: built + live-verified** | $0 LLM | backend restarted (v6.28.16) to load the new router.

## What was built
A runtime operator-control surface for the four paper-trading deployment/concentration caps, tunable LIVE without a backend restart, fully bounded + confirmation-gated + audited. The P7 "risk limits" deliverable and the safe bridge for the (operator-only) "deploy idle cash" decision (see cycle_block_summary.md money analysis).

## Files changed/added
1. **`backend/services/risk_overrides.py`** (NEW, ~200 lines) -- file-backed override store mirroring `kill_switch.py`:
   - Module singleton `RiskOverrideState` + `threading.Lock`; append-only JSONL audit at `handoff/risk_overrides_audit.jsonl`; `_load_from_audit()` replay-on-init (restart-survivable, re-validates on replay).
   - `ALLOWED_KEYS` strict allowlist (the 4 deployment caps) + `BOUNDS` (type + min/max per key). Kill-switch loss limits are deliberately absent.
   - `_coerce_and_validate()` (validate-before-accept; `RiskOverrideError` on bad key / out-of-range / non-coercible).
   - API: `get_effective(key, default)`, `set_override(key, value, reason)`, `clear_override(key)`, `clear_all()`, `snapshot()`, `describe()`.
2. **`backend/services/portfolio_manager.py`** -- reader seam: the 4 caps now read via `risk_overrides.get_effective("<key>", <settings default>)` at the at-decide-time read points:
   - line 77 `paper_min_cash_reserve_pct`; lines 218/221 `paper_max_per_sector` + `paper_max_per_sector_nav_pct`; line 251 `paper_max_positions` (computed once before the buy loop; the diagnostic log + the swap-path nav_pct read at ~520 also use the effective value). Behaviour byte-identical when no override set.
3. **`backend/api/paper_trading.py`** -- `RiskLimitRequest` model + 3 routes: `GET /risk-limits` (effective values + bounds + overridden flags), `PUT /risk-limits` (confirmation `SET_RISK_LIMIT` + bounded + audited + `get_api_cache().invalidate("paper:*")`), `DELETE /risk-limits/{key}` (revert). Mirrors the existing `/pause` `/resume` `KillSwitchActionRequest` pattern.

## Verification command output (the masterplan immutable command)
```
$ python -c "import ast; ast.parse(open('backend/services/risk_overrides.py').read())"   -> syntax OK
$ python -c "from backend.services import risk_overrides as r; r.clear_all(); assert r.get_effective('paper_max_per_sector',2)==2; r.set_override('paper_max_per_sector',4,reason='test'); assert r.get_effective('paper_max_per_sector',2)==4; r.clear_override('paper_max_per_sector'); assert r.get_effective('paper_max_per_sector',2)==2; print('risk_overrides roundtrip OK')"
risk_overrides roundtrip OK
$ test -f handoff/current/live_check_49.1.md   -> exists
```
All three modules `ast.parse` clean; `import backend.api.paper_trading` registers the 3 risk-limits routes.

## Live verification (full evidence in live_check_49.1.md)
GET(defaults) -> PUT(set=4, effective->4) -> GET(overridden) -> PUT(999)=HTTP400 bounds -> PUT(daily_loss_limit_pct)=HTTP400 disallowed -> PUT(wrong confirmation)=HTTP400 -> DELETE(revert->2) -> GET(clean). Audit JSONL captured 4 rows. **Restart-survival proven**: set paper_max_positions=15 -> restart -> GET shows 15/overridden -> DELETE.

## Success criteria mapping (all 5 met)
1. risk_overrides.py exists/parses/mirrors kill_switch (singleton + JSONL audit + replay + lock) -- YES.
2. get_effective returns override-or-default; set_override bounded + audited; clear reverts -- YES (round-trip + HTTP 400 on 999).
3. portfolio_manager reads the 4 caps via get_effective at decide-time; kill-switch loss limits NOT mutable -- YES (5 read points wired; daily_loss_limit_pct rejected HTTP 400).
4. GET/PUT/DELETE exist with confirmation + bounds + cache-invalidate; live curl round-trip in live_check_49.1.md -- YES.
5. every mutation appends an audit row (key, old, new, reason, ts) -- YES (verbatim JSONL).

## Scope honesty
- This makes EXISTING caps operator-tunable; it does NOT change trading logic or alpha. No-override behaviour is identical to pre-49.1.
- `recommended_position_pct` (the lite-risk-judge 3% sizing in autonomous_loop.py ~1390) is NOT a settings field and is intentionally OUT OF SCOPE for this cycle (a future step can add it as a 5th knob).
- No UI wiring in this cycle (backend control surface only; UI consistency is separate P7 work behind the NextAuth real-browser-verification constraint).
