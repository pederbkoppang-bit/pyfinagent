# Research Brief — phase-49.1: Runtime Risk-Limit Control Endpoint

Tier: moderate. Status: IN PROGRESS (incremental write).

## Scope
Operator API endpoint under `/api/paper-trading` to READ + ADJUST live
risk/deployment knobs without backend restart: audited, reversible,
bounded/validated. Knobs: `paper_max_per_sector` (count, 2),
`paper_max_per_sector_nav_pct` (NAV %, 30), `paper_min_cash_reserve_pct`
(5), `paper_max_positions` (10/env 20), lite-judge `recommended_position_pct`
(3% in autonomous_loop.py ~1390).

---

## INTERNAL CODE TRACE (5 gating questions) — DEFINITIVE

### Q1 — Process topology: WHERE does the live loop run? **SAME PROCESS as the API.**

DECISIVE FINDING: The live daily trading cycle is fired by the **backend's
OWN** APScheduler, NOT the slack_bot scheduler.

- `backend/main.py:256-265` (lifespan startup): creates `AsyncIOScheduler()`,
  calls `init_scheduler(scheduler)` (paper_trading.py), `scheduler.start()`.
- `backend/api/paper_trading.py:1220-1253` `init_scheduler` → `_add_scheduler_job`
  registers cron job `paper_trading_daily` → calls `_scheduled_run()`
  (line 1256) → `run_daily_cycle(settings)` (line 1260).
- `backend/api/paper_trading.py:947-965` `/run-now` → `asyncio.create_task(
  _run_cycle_background(settings))` → `run_daily_cycle` (line 1210).
- The slack_bot scheduler (`backend/slack_bot/scheduler.py:199-245`) ONLY
  schedules: morning digest, evening digest, watchdog health check, nightly
  prompt-leak red-team. It does NOT run the trading cycle. All its
  cross-process calls are `httpx` GETs to `/api/paper-trading/portfolio` etc.
  (lines 322/348/398) — pure read, for digests.

CONCLUSION: `run_daily_cycle` and ALL `/api/paper-trading/*` endpoints
execute in the **same `com.pyfinagent.backend` process / same event loop**.
=> An IN-PROCESS override store IS viable (setter API + reader loop share
memory). No cross-process IPC needed for the loop. (The slack_bot process
only needs the override for *display* in digests — it already reads
everything via HTTP, so a GET endpoint covers it.)

CAVEAT (uvicorn workers): the backend runs uvicorn with `--reload` in dev
(single worker). If a future deploy runs multiple uvicorn workers, an
in-memory-ONLY store desyncs across workers — the EXACT bug phase-38.13.1
hit with `paper_use_claude_code_route` (autonomous_loop.py:1285-1294 calls
`get_settings.cache_clear()` to cure it). For a local single-worker Mac
deployment ([[project_local_only_deployment]]) in-memory is fine TODAY, but
a file/BQ-backed store is restart-survivable AND worker-safe — strongly
preferred. The kill-switch already chose file-backed for exactly this reason.

### Q2 — `get_settings()` caching: `@lru_cache()`, frozen at first call.

`backend/config/settings.py:460-462`:
```python
@lru_cache()
def get_settings() -> Settings:
    return Settings()
```
Frozen at first-call. A `settings.paper_max_per_sector = X` mutation on the
cached instance WOULD propagate (same object every caller gets), BUT:
- `run_daily_cycle` (autonomous_loop.py:174): `settings = settings or
  get_settings()` — the cached singleton.
- `_scheduled_run` (paper_trading.py:1258) calls `get_settings()` fresh each
  cron fire — same cached singleton.
- pydantic-settings `BaseSettings` is **mutable by default** (no
  `model_config["frozen"]=True`). So `s.paper_max_per_sector = 4` works at
  runtime.
- BUT there is already a documented lru_cache cross-worker desync; the cure
  is `cache_clear()` + re-read (autonomous_loop.py:1292-1294).

CONCLUSION: Mutating the cached Settings singleton is technically possible but
FRAGILE: lost on restart, desyncs across workers, un-auditable, pollutes a
global. The clean pattern is a SEPARATE override store the reader consults at
decide-time, NOT settings-object mutation.

### Q3 — Existing override-persistence pattern: **FILE-BACKED JSONL** (kill_switch).

`backend/services/kill_switch.py` is the canonical template:
- Module-level singleton `_state = KillSwitchState()` (line 220), accessor
  `get_state()` (line 223). `threading.Lock`-guarded (line 46).
- **Persistence = append-only JSONL audit log** at
  `handoff/kill_switch_audit.jsonl` (line 36). On `__init__`,
  `_load_from_audit()` (line 61) replays the log to reconstruct state →
  **restart-survivable**. Every transition appends a row via `_append_audit`
  (line 108): `{ts, event, trigger, details}`.
- API handlers (paper_trading.py:510-558) call `_get_ks_state().pause/resume`,
  guarded by a `confirmation` string ("PAUSE"/"RESUME") — fat-finger guard.
- The loop reads it at decide-time: autonomous_loop.py:803-806
  `from backend.services.kill_switch import get_state; if ... _ks_state().is_paused()`.

This is the SAME shape the override store should take: singleton + file-backed
JSONL replay + threading.Lock + read-at-decide-time. In-process (loop + API
share it) AND restart-survivable AND auditable.

### Q4 — Where caps are consumed: **READ-AT-DECIDE-TIME** in portfolio_manager.

`backend/services/portfolio_manager.py`:
- `decide_trades(..., settings: Settings, ...)` (line 50-55) receives settings
  by reference from the loop (autonomous_loop.py:948 `settings=settings`).
- Caps read at decide-time, NOT cached at import:
  - line 213: `max_per_sector = int(getattr(settings, "paper_max_per_sector", 0) or 0)`
  - line 215: `getattr(settings, "paper_max_per_sector_nav_pct", 0.0) or 0.0`
  - line 242: `if remaining_positions >= settings.paper_max_positions`
  - line 74: `min_cash = nav * (settings.paper_min_cash_reserve_pct / 100.0)`
  - line 503: `max_sector_nav_pct = float(getattr(settings, "paper_max_per_sector_nav_pct", ...))`
All via `getattr(settings, "X")` at call time. => an override injected at this
read point (or visible on this object) is picked up the VERY NEXT cycle. No
restart. THIS is the integration seam.

NOTE on position-sizing default: `recommended_position_pct` 3% lives in
autonomous_loop.py (~1390, lite risk judge) — NOT a settings field. To make
it operator-tunable it must become a new settings field OR be read from the
override store at that point. (Verify exact line during GENERATE.)

### Q5 — Existing control endpoints (be consistent):
`backend/api/paper_trading.py`: `/start` (67) `/stop` (93) `/status` (105)
`/pause` (510) `/resume` (519) `/flatten-all` (562) `/kill-switch` GET (469)
`/gate` GET (667) `/run-now` (947) `/deposit` (1150). POST mutators use a
Pydantic `KillSwitchActionRequest` with a `confirmation` string gate. Cache
invalidation via `get_api_cache().invalidate("paper:*")` on every mutator.

### RECOMMENDED DESIGN (internal)
- New module `backend/services/risk_overrides.py` mirroring `kill_switch.py`:
  singleton `RiskOverrideStore`, file-backed JSONL at
  `handoff/risk_overrides_audit.jsonl`, `threading.Lock`, `_load_from_audit()`
  replay, `snapshot()`, `set_override(key, value, trigger, actor)`,
  `clear_override(key)`, `get_effective(key, settings_default)`.
- Bounds table per knob (reuse the pydantic `ge`/`le` already on the fields:
  `paper_max_per_sector` 0..20, `paper_max_per_sector_nav_pct` 0..100;
  `paper_min_cash_reserve_pct` currently UNBOUNDED → ADD 0..50;
  `paper_max_positions` UNBOUNDED → ADD 1..50; position_pct 0.5..20).
  Reject out-of-bounds with HTTP 422.
- Reader seam: in `decide_trades`, replace bare
  `getattr(settings, "paper_max_per_sector", 0)` with
  `risk_overrides.get_effective("paper_max_per_sector", settings.paper_max_per_sector)`
  (override wins if set, else settings default). ONE integration point; the
  loop picks it up next cycle automatically.
- API: `GET /api/paper-trading/risk-limits` (effective + defaults + bounds +
  who/when), `PUT /api/paper-trading/risk-limits` (set, bounded,
  confirmation-gated, audited), `DELETE .../risk-limits/{key}` (revert to
  settings default). Invalidate `paper:*` cache on write.
- Reversibility: DELETE clears the override → next cycle reads settings
  default. Full history in the JSONL.

---

## EXTERNAL RESEARCH

### Search-query variants run (3-variant discipline)
1. Current-frontier: "runtime feature flags risk limits trading system hot
   reload without redeploy 2026"; "dynamic risk limit management trading desk
   intraday limit override governance 2025".
2. Last-2-year: "pre-trade risk controls audit trail kill switch FIA SEC
   15c3-5 2025"; "...governance 2025".
3. Year-less canonical: "feature flag dynamic configuration management best
   practices"; "Knight Capital 2012 trading glitch feature flag"; "position
   sizing limits concentration cap operational risk fat finger controls".

### Read in full (>=5 required — counts toward gate)
| URL | Accessed | Kind | Fetched how | Key finding |
|-----|----------|------|-------------|-------------|
| federalreserve.gov/econres/feds/files/2025034pap.pdf (Li/Petrasek/Tian, FEDS 2025-034) | 2026-05-29 | peer-reviewed-adjacent (Fed working paper) | WebFetch→pdfplumber | "Even when limits occasionally get relaxed... such limit changes typically require multiple layers of approval." Loosening a risk limit is a multi-approval, costly-to-breach institutional action. |
| wilmerhale.com/.../sec-15c3-5-faqs | 2026-05-29 | official-analysis (law firm on SEC FAQ) | WebFetch (full) | Intraday limit increase IS permitted "in accordance with supervisory procedures"; "reasons for such modifications should be documented and retained as part of books and records"; controls must be under "direct and exclusive control" of the operator. |
| docs.getunleash.io/guides/feature-flag-best-practices | 2026-05-29 | authoritative vendor doc | WebFetch (full) | Four-eyes principle for critical changes; "tracking and auditing feature flag changes help... meet regulatory requirements"; flags (runtime, short-lived) vs configuration (static); unique names prevent stale-flag reuse. |
| henricodolfing.ch/.../knight-capital | 2026-05-29 | case study (industry) | WebFetch (full) | $440M loss from reused/stale flag + no second-person deploy review + NO kill switch / auto loss-limit + silent failure. Canonical fat-finger / un-audited-toggle disaster. |
| oneuptime.com/.../feature-flag-service-hot-reload-rust | 2026-01-25 | technical blog | WebFetch (full) | Hot-reload pattern: atomic swap (ArcSwap) so readers never block; file-based persistence loaded at startup; "validate config before accepting, reject invalid"; "if new config fails to load, keep the old one." |
| resonanzcapital.com/.../position-sizing-sell-discipline | 2026-05-29 | industry (allocator) | WebFetch (full) | Two-tier "review threshold THEN hard trim"; pre-committed limits "force discipline in good times, before hubris can tempt us to over-allocate"; compound/correlated exposure capped below sum of singles. |

### Snippet-only (context; does NOT count toward gate)
| URL | Kind | Why not full |
|-----|------|--------------|
| sec.gov/.../divisionsmarketregfaq-0 (Rule 15c3-5 FAQ) | official (primary) | HTTP 403 on WebFetch; content captured via search snippet + WilmerHale full-read of same FAQ |
| fia.org/.../FIA_WP_AUTOMATED TRADING RISK CONTROLS_FINAL.pdf | industry standard | Already cited in settings.py:433 (price-tolerance gate); known content |
| federalreserve.gov FINRA Sept-2025 risk-based margin framework | regulatory news | Recency-scan evidence; covered by search snippet |
| launchdarkly.com/blog/what-are-feature-flags | vendor | Redundant with Unleash full-read |
| configu.com/blog/feature-flags... | vendor | Fetched but weak on config-vs-flags distinction; superseded by Unleash |
| octopus.com/devops/feature-flags | vendor | Redundant |
| learn.microsoft.com/.../manage-feature-flags (Azure App Config) | official | Confirms separate config store + validation; redundant |
| krm22.com/limits-manager | industry product | Confirms "limits manager" as a product category |
| opensee.io/blog/challenges-of-intraday-risk-management | industry | "governance" is the hard part of intraday limits |
| momentslog.com/.../position-sizing-policy | industry | Review-threshold + hard-cap two-tier |

URLs collected total: 30+ across the 6 search passes.

### Recency scan (last 2 years, 2024-2026) — PERFORMED
Findings that COMPLEMENT the canonical sources:
- **Fed FEDS 2025-034 (Li/Petrasek/Tian)** — 2025 empirical confirmation that
  internal risk-limit relaxations require multiple approval layers and that
  binding limits measurably change behavior. Directly modernizes the
  "limits must be meaningful + governed" thesis. (NEW, complements.)
- **FINRA Sept-2025 risk-based intraday margin framework** — regulators are
  moving from static dollar thresholds to dynamic, volatility/concentration/
  correlation-based limits. Validates the *direction* of a runtime-tunable
  limit surface, but those are still governed changes, not free toggles.
- **FIA Automated Trading Risk Controls WP (July 2024)** — already the
  canonical pre-trade-gate reference in this codebase (settings.py:433).
  Reconfirms pre-trade, non-bypassable controls.
- **OneUptime hot-reload pattern (Jan 2026)** — current implementation
  pattern (atomic swap + validate-before-accept + keep-old-on-failure).
No 2024-2026 source CONTRADICTS the design; the consensus is unchanged and
strengthened: runtime limit changes are allowed but must be bounded,
documented/audited, reversible, and access-controlled. No finding supersedes
the kill-switch file-backed-JSONL pattern already in this repo.

### Consensus vs debate
CONSENSUS (strong, cross-domain — regulators + vendors + academia + incident
history): (1) runtime adjustment of risk limits without redeploy is legitimate
and expected; (2) every change MUST be documented/audited and retained;
(3) changes to risk controls should require elevated authorization (four-eyes
/ "multiple layers of approval" / confirmation); (4) values must be bounded
and validated before applying; (5) changes must be reversible. DEBATE: degree
of approval — feature-flag vendors say "four-eyes for critical"; Fed evidence
says institutional limit relaxation is multi-layer. For a single-operator local
deployment, a single-step *typed-confirmation* gate + full audit log is the
proportionate analogue (matches existing kill-switch `confirmation="PAUSE"`).

### Pitfalls (from literature)
- **Stale/ambiguous override names (Knight Capital)** — reusing or
  mis-scoping a control flag activated unintended behavior → $440M. MITIGATION:
  unique explicit keys (exact settings field names), a clear "cleared = falls
  back to settings default" semantics, and DELETE to revert.
- **No kill switch / no auto loss-limit (Knight Capital)** — runtime tuning
  must NEVER be able to disable the existing kill-switch breach checks. The
  override surface is for the deployment/concentration knobs ONLY; the
  daily-loss / trailing-DD limits stay enforced (autonomous_loop.py:803-806).
- **Unbounded values (OneUptime, Azure)** — validate-before-accept; reject
  out-of-bounds. Loosening `paper_max_per_sector_nav_pct` to 100 or
  `paper_min_cash_reserve_pct` to 0 is a fat-finger that concentrates the book
  → enforce sane max bounds (e.g. NAV-% sector cap <=50, min-cash >=0).
- **Un-audited change (Unleash, SEC 15c3-5)** — every set/clear must append a
  timestamped audit row (who/what/old/new/reason) — mirror
  handoff/kill_switch_audit.jsonl.
- **Worker desync (phase-38.13.1, internal)** — in-memory-only store desyncs
  across uvicorn workers; file-backed JSONL replay avoids it.

### Application to pyfinagent (external → internal anchors)
1. File-backed JSONL audit store (SEC "documented and retained"; Unleash
   "auditing... meet regulatory requirements") → new
   `backend/services/risk_overrides.py` mirroring
   `backend/services/kill_switch.py:36,108,220` → audit file
   `handoff/risk_overrides_audit.jsonl`.
2. Read-at-decide-time so next cycle picks it up without restart (OneUptime
   atomic-swap; SEC intraday-adjustment-allowed) → inject at
   `portfolio_manager.py:213,215,242,74,503` via
   `risk_overrides.get_effective(key, settings.<field>)`. Loop re-calls
   `decide_trades` every cycle (autonomous_loop.py:943-948), so no restart.
3. Bounds + validate-before-accept (OneUptime; Resonanz pre-committed limits)
   → reuse pydantic `ge`/`le` (settings.py:180,188) + ADD bounds for
   `paper_min_cash_reserve_pct` (0..50) and `paper_max_positions` (1..50);
   position_pct 0.5..20. HTTP 422 on violation.
4. Confirmation gate + audit actor (Knight four-eyes; Fed multi-approval)
   → PUT body carries `confirmation` string (mirror
   `KillSwitchActionRequest`, paper_trading.py:510) + `actor`/`reason`.
5. Reversibility (Unleash kill-switch semantics) → DELETE
   `.../risk-limits/{key}` clears override → settings default resumes next
   cycle; full history in JSONL.
6. Never weaken the kill-switch (Knight no-killswitch lesson) → risk-overrides
   surface excludes `paper_daily_loss_limit_pct` / `paper_trailing_dd_limit_pct`
   (those stay in settings + kill_switch breach path).

---

## JSON ENVELOPE

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 6,
  "snippet_only_sources": 10,
  "urls_collected": 30,
  "recency_scan_performed": true,
  "internal_files_inspected": 6,
  "gate_passed": true
}
```
