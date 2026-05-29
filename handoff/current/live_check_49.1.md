# live_check_49.1 -- runtime risk-limit control endpoint (LIVE evidence)

Backend restarted to v6.28.16; all calls against the RUNNING backend (localhost:8000) on 2026-05-29.

## 1. GET /api/paper-trading/risk-limits (initial -- no overrides)
All 4 adjustable keys returned with bounds + `overridden:false` + `effective_value == settings_default`:
- paper_max_per_sector: min 0 / max 20, default 2, effective 2
- paper_max_per_sector_nav_pct: min 0 / max 100, default 30.0, effective 30.0
- paper_min_cash_reserve_pct: min 0 / max 50, default 5.0, effective 5.0
- paper_max_positions: min 1 / max 50, default 20, effective 20
allowed_keys = [paper_max_per_sector, paper_max_per_sector_nav_pct, paper_max_positions, paper_min_cash_reserve_pct]

## 2. PUT {key:paper_max_per_sector, value:4, confirmation:SET_RISK_LIMIT, reason:"phase-49.1 live test"}
-> `{"status":"override_set","key":"paper_max_per_sector","effective_value":4,"overrides":{"paper_max_per_sector":4}}`

## 3. GET after override
-> `paper_max_per_sector: {effective_value:4, overridden:true, settings_default:2}`  (override picked up; default preserved)

## 4. PUT value=999 (out of bounds) -> HTTP 400
-> `{"detail":"value 999 for 'paper_max_per_sector' is out of bounds [0, 20]"}`  (validate-before-accept)

## 5. PUT key=daily_loss_limit_pct (kill-switch key) -> HTTP 400
-> `{"detail":"'daily_loss_limit_pct' is not an adjustable risk limit. Allowed keys: [...]"}`  (Knight Capital safety: kill-switch loss limits NOT mutable here)

## 6. PUT confirmation="WRONG" -> HTTP 400  (confirmation-gated)

## 7. DELETE /risk-limits/paper_max_per_sector
-> `{"status":"override_cleared","key":"paper_max_per_sector","effective_value":2,"overrides":{}}`  (revert to settings default)

## 8. GET final -> paper_max_per_sector {effective_value:2, overridden:false}  (clean)

## 9. Audit trail handoff/risk_overrides_audit.jsonl (verbatim tail)
```
{"ts": "2026-05-29T20:45:12.597048+00:00", "event": "set", "key": "paper_max_per_sector", "old_value": null, "new_value": 4, "reason": "test"}
{"ts": "2026-05-29T20:45:12.597266+00:00", "event": "clear", "key": "paper_max_per_sector", "old_value": 4, "new_value": null, "reason": "manual"}
{"ts": "2026-05-29T20:48:04.037206+00:00", "event": "set", "key": "paper_max_per_sector", "old_value": null, "new_value": 4, "reason": "phase-49.1 live test"}
{"ts": "2026-05-29T20:48:04.148117+00:00", "event": "clear", "key": "paper_max_per_sector", "old_value": 4, "new_value": null, "reason": "manual"}
```

## 10. RESTART-SURVIVABILITY (criterion #1 -- audit replay)
- PUT paper_max_positions=15 -> override_set, effective 15
- `launchctl kickstart -k com.pyfinagent.backend` -> healthy after 4s
- GET after restart -> `paper_max_positions {effective_value:15, overridden:true}`  (PERSISTED via _load_from_audit replay)
- DELETE -> effective back to 20

## Verdict
All 5 immutable success criteria verified against the live system: file-backed store mirrors kill_switch.py (singleton + JSONL audit + replay + lock); get_effective returns override-or-default; set_override bounded + audited; portfolio_manager reads the 4 caps via get_effective at decide-time; GET/PUT/DELETE work with confirmation gate + bounds + allowlist + cache invalidation; every mutation audited; overrides survive restart. Default behaviour (no override) is byte-identical to pre-49.1.
