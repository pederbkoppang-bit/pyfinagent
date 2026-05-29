# Contract -- phase-49.1: Runtime risk-limit control endpoint

**Step id:** 49.1 | **Priority:** P2 (P7 operator control surface -- "risk limits" deliverable) | **depends_on:** 48.4
**Date:** 2026-05-29 | **harness_required:** true | **$0 LLM** (pure backend; backend restart once to load the new router)

## Research-gate summary (PASSED)
`handoff/current/research_brief.md` (researcher gate: **6 sources read in full, recency scan done, 30+ URLs, 6 internal files, gate_passed=true**). Decisive findings:
- **Process topology (Q1):** the daily loop runs in the **backend's own** APScheduler (`main.py:256` lifespan -> `init_scheduler` -> `paper_trading.py:1220` cron `paper_trading_daily` -> `_scheduled_run()` -> `run_daily_cycle`). The slack_bot scheduler only does digests/watchdog/red-team. So the loop + the API share the `com.pyfinagent.backend` process -> an override store is reachable by both. File-backed still preferred (survives restart; avoids the phase-38.13.1 lru_cache cross-worker desync).
- **Settings caching (Q2):** `get_settings()` is `@lru_cache` frozen at first call -> do NOT mutate the settings object; use a separate override store.
- **Override template (Q3):** `backend/services/kill_switch.py` is the proven pattern -- module singleton (`_state`) + append-only JSONL (`handoff/kill_switch_audit.jsonl`) + `_load_from_audit()` replay-on-init + `threading.Lock` + confirmation-gated handlers.
- **Integration seam (Q4):** the caps are read AT-DECIDE-TIME in `portfolio_manager.py` via `getattr(settings, "X")` at lines **74** (`paper_min_cash_reserve_pct`), **213** (`paper_max_per_sector`), **215/503** (`paper_max_per_sector_nav_pct`), **242** (`paper_max_positions`). `decide_trades` runs fresh each cycle -> an override injected here is picked up next cycle, no restart.
- **External (Q5 + best practices):** SEC Rule 15c3-5 + Fed FEDS-2025-034 (runtime limit changes are legitimate but MUST be audited + authorized + bounded); Knight Capital $440M (never let this surface disable the kill-switch breach checks; unique explicit keys; clear "cleared = default" semantics); OneUptime/Unleash (validate-before-accept, atomic, reversible, four-eyes for critical).

## Hypothesis
A file-backed `risk_overrides` store (mirroring `kill_switch.py`) + a `get_effective(key, default)` reader seam at the `portfolio_manager.py` at-decide-time read points + confirmation-gated/bounded/audited `GET/PUT/DELETE /api/paper-trading/risk-limits` will let the operator tune the four deployment caps **live, without a backend restart**, picked up by the next cycle, fully audited and reversible -- delivering the P7 "risk limits" control and the safe bridge for the (operator-only) "deploy idle cash" decision, with **zero change to trading logic** and the kill-switch breach checks left immutable.

## Success criteria (IMMUTABLE -- copied verbatim from .claude/masterplan.json step 49.1)
1. backend/services/risk_overrides.py exists, parses, and mirrors the kill_switch.py file-backed pattern: a module singleton + append-only JSONL audit at handoff/risk_overrides_audit.jsonl + replay-on-init (restart-survivable) + threading.Lock
2. get_effective(key, default) returns the active override when one is set and the passed settings default when not; set_override is BOUNDED (rejects out-of-range values with a clear error) and audited; clear_override/clear_all revert to settings defaults
3. portfolio_manager.py reads the four deployment caps (paper_max_per_sector, paper_max_per_sector_nav_pct, paper_min_cash_reserve_pct, paper_max_positions) via risk_overrides.get_effective(...) at the existing at-decide-time read points (lines ~74/213/215/242) so an override is picked up the NEXT cycle with no restart; the existing kill-switch loss-limit breach checks are NOT mutable through this surface (Knight Capital safety)
4. GET/PUT/DELETE /api/paper-trading/risk-limits exist: GET returns effective values + which keys are overridden; PUT is bounded + confirmation-gated + writes an audit row + invalidates the paper api_cache; DELETE reverts a key to its settings default; a LIVE curl round-trip (PUT->GET->DELETE) is captured verbatim in live_check_49.1.md showing the override set, read back, and reverted
5. every override mutation appends an audit row (key, old_value, new_value, reason, timestamp) to handoff/risk_overrides_audit.jsonl

**live_check:** REQUIRED -- live curl PUT/GET/DELETE round-trip against the running backend (port 8000) in live_check_49.1.md + the audit JSONL rows.

## Plan steps
1. **`backend/services/risk_overrides.py`** (mirror kill_switch.py): `BOUNDS` dict (per-key min/max), module singleton + `threading.Lock`, append-only JSONL at `handoff/risk_overrides_audit.jsonl`, `_load_from_audit()` replay-on-init, `get_effective(key, default)`, `set_override(key, value, reason=...)` (validates against BOUNDS, rejects out-of-range, appends audit row), `clear_override(key)`, `clear_all()`, `snapshot()`. Only the four deployment caps are allowed keys; kill-switch loss limits are NOT (guarded by the allowed-key set).
2. **`portfolio_manager.py` reader seam**: replace the four `getattr(settings, "X")` / `settings.X` reads (lines ~74/213/215/242) with `risk_overrides.get_effective("X", <existing default expression>)`. Behaviour identical when no override set (returns the settings value).
3. **`backend/api/paper_trading.py` endpoints**: `GET /risk-limits` (effective values + overridden-keys list + bounds), `PUT /risk-limits` (Pydantic body {key, value, confirmation, reason}; bounded; confirmation-gated like KillSwitchActionRequest; audit; `get_api_cache().invalidate("paper:*")`), `DELETE /risk-limits/{key}` (revert). Consistent with existing /pause /resume handlers.
4. **Verify**: syntax (ast.parse all 3); unit round-trip (the masterplan command's python -c); start/restart backend; **live curl PUT->GET->DELETE**; capture verbatim into `live_check_49.1.md`; confirm `risk_overrides_audit.jsonl` rows.
5. **EVALUATE**: spawn a FRESH qa (no self-eval). Then append harness_log.md (LAST), then flip masterplan 49.1 -> done.

## Anti-overfit / safety notes
- This does NOT change trading logic or the alpha; it makes existing caps operator-tunable. Default behaviour (no override) is byte-identical to today.
- Bounds prevent fat-finger (e.g., paper_max_per_sector in [0,20] matching settings Field ge/le; nav_pct in [0,100]; min_cash_reserve in [0,50]; max_positions in [1,50]).
- Kill-switch loss-limit breach checks remain un-mutable through this surface (explicit allowed-key allowlist).

## References
- handoff/current/research_brief.md (the gate-passing brief)
- backend/services/kill_switch.py (persistence template)
- backend/services/portfolio_manager.py:74,213,215,242,503 (integration seam)
- backend/api/paper_trading.py:510-558 (control-endpoint pattern)
- backend/config/settings.py:176,180,188,255 (the knobs + existing Field bounds)
- backend/main.py:256-265 (in-process scheduler)
- SEC Rule 15c3-5 / WilmerHale FAQ; Fed FEDS-2025-034; Knight Capital (Dolfing); OneUptime hot-reload; Unleash flag best-practices
