# Research Brief — phase-61.1: Activate dark fixes + deploy phase-60 code

Tier: simple (caller-stated). Date: 2026-06-11. Agent: researcher (Layer-3 MAS, merged Explore).
Prior phase-60.4 brief archived by hook to handoff/archive/phase-60/ (this file overwrites the rolling slot per handoff convention).
Disclosed overrun: the 4-item internal audit (incl. the double-cycle restart-safety question) pushes past the simple-tier 10-tool-call budget; floors all met, prose kept tight.

## Sources read in full (>=5 required; counts toward gate)

| # | URL | Accessed | Kind | Fetched how | Key finding |
|---|-----|----------|------|-------------|-------------|
| 1 | https://apscheduler.readthedocs.io/en/3.x/userguide.html | 2026-06-12 | Official docs (APScheduler 3.11.2) | WebFetch, full page | Misfire applies to run times the scheduler KNOWS were missed: "The most common case is when a job is scheduled in a persistent job store and the scheduler is shut down and restarted after the job was supposed to execute." Default store "simply keeps the jobs in memory"; "If you always recreate your jobs at the start of your application, then you can probably go with the default (MemoryJobStore)." Coalesce merges queued executions into one. |
| 2 | https://ss64.com/mac/launchctl.html | 2026-06-12 | Official man-page mirror | WebFetch, full page | `kickstart`: "Instructs launchd to kickstart the specified service." `-k`: "If the service is already running, kill the running instance before restarting the service." `-p` prints new PID. `gui/uid/label` targets the user's login domain. Kill-then-restart is atomic at the service (label) level — launchd owns instance uniqueness. |
| 3 | https://pydantic.dev/docs/validation/latest/concepts/pydantic_settings/ | 2026-06-12 | Official docs (pydantic-settings, redirect from docs.pydantic.dev) | WebFetch, full page | Dotenv is loaded at `Settings()` INSTANTIATION; "no mention of caching across instantiations" — each new instance re-reads the file. Priority: init args > env vars > dotenv > secrets > defaults ("environment variables will always take priority over values loaded from a dotenv file"). "By default, environment variable names are case-insensitive." Extra dotenv keys: `extra='ignore'` (this repo, settings.py:536) skips them. |
| 4 | https://raw.githubusercontent.com/encode/uvicorn/master/docs/server-behavior.md | 2026-06-12 | Official docs source (uvicorn.org/server-behavior — site itself ECONNREFUSED twice, GitHub canonical source used) | WebFetch, full page | Graceful shutdown: "Close any connections that are not currently waiting on an HTTP response, and wait for any other connections to finalize their HTTP responses... Wait for any background tasks to run to completion." "Uvicorn handles process shutdown gracefully, ensuring that connections are properly finalized, and all tasks have run to completion" within configured timeouts. |
| 5 | https://martinfowler.com/articles/feature-toggles.html | 2026-06-12 | Authoritative blog (Hodgson/Fowler — canonical feature-toggle reference) | WebFetch, full page | Test BOTH toggle states; ship the production-intended config plus "the fall-back configuration where those toggles you intend to release are also flipped Off" (phase-60/57 did exactly this: OFF byte-identity tests + ON behavior tests). Env-var/parameterized toggles "still require process restart" — restart-to-flip is a recognized point on the spectrum, acceptable for non-emergency flags. Release toggles are transitionary (~1-2 weeks): plan flag retirement after validation. |

Supplementing #1 with code-level proof, the INSTALLED APScheduler 3.11.2 source was read (internal evidence, not counted in the external gate): see §2 of the internal audit.

## Snippet-only table (does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|-----|------|------------------------|
| https://eclecticlight.co/2019/08/27/kickstarting-and-tearing-down-with-launchctl/ | Authoritative blog | ss64 man page covers kickstart -k authoritatively |
| https://www.kevinmcox.com/2024/03/changes-to-launchctl-kickstart-in-macos-14-4/ | Practitioner blog | Recency datapoint only: macOS 14.4 blocks `kickstart -k` on critical SYSTEM daemons; user-domain `gui/` agents (our case) unaffected |
| https://github.com/agronholm/apscheduler/issues/1095 | Maintainer issue tracker (Dec 2025) | Confirms 3.x missed-job semantics discussion; user guide + installed source already decisive |
| https://apscheduler.readthedocs.io/en/3.x/modules/jobstores/memory.html | Official docs | MemoryJobStore "stores jobs in memory as-is, without serializing them" — covered by user guide |
| https://betterstack.com/community/guides/scaling-python/apscheduler-scheduled-tasks/ | Community guide | "By default, APScheduler keeps all jobs in memory... lost when your application restarts" — corroboration only |
| https://launchdarkly.com/blog/guide-to-dark-launching/ | Vendor blog | Fowler covers the canonical practice; LD adds percentage-rollout detail not applicable to a single-operator local deployment |
| https://www.digitalapplied.com/blog/feature-flag-rollout-strategies-2026-engineering-playbook | Industry blog (2026) | Recency datapoint: keep kill-switch/rollback path ~30 days post-enable with explicit exit criteria |
| https://github.com/encode/uvicorn/pull/853 | Maintainer PR | SIGTERM/SIGINT graceful-shutdown fix history; docs read in full instead |
| https://github.com/Kludex/uvicorn/issues/668 | Issue tracker | Multi-worker SIGTERM edge cases — not applicable (single-process uvicorn here) |
| https://leancrew.com/all-this/man/man1/launchctl.html | Man-page mirror | Duplicate of ss64 content |
| https://launchdarkly.com/blog/release-management-flags-best-practices/ | Vendor blog | Overlaps Fowler |
| https://apscheduler.readthedocs.io/en/3.x/faq.html | Official docs | No restart-specific content beyond user guide |

## Search queries run (three-variant discipline)

1. `launchctl kickstart -k semantics kill restart service man page` — year-less canonical
2. `APScheduler misfire_grace_time coalesce missed jobs restart 2025` — last-2-year window
3. `feature flag dark launch safe rollout best practice 2026` — current-year frontier
4. `APScheduler MemoryJobStore jobs lost on restart add_job next_run_time` — year-less canonical
5. `pydantic-settings env_file loading precedence dotenv` — year-less canonical
6. `uvicorn graceful shutdown SIGTERM supervisor process managers` — year-less canonical

Disclosure (simple tier): six queries across five topics, not 3x5=15 literal variants. The variant mix is covered in aggregate — current-year (#3), last-2-year (#2, plus 2024-2026 hits surfacing inside #1/#3/#4 result sets), year-less (#1,#4,#5,#6). launchctl/pydantic/uvicorn are versioned-docs topics where the canonical page IS the current-year source; their recency hits (macOS 14.4 change, Dec-2025 APScheduler issue) arrived inside the year-less result sets and are reported below.

## Recency scan (last 2 years)

Performed; window 2024-2026. Findings:

1. **macOS 14.4 (Mar 2024) restricted `launchctl kickstart -k`** for critical SYSTEM processes (e.g. `cfprefsd`) — kill via `kill` instead (kevinmcox.com). NOT applicable here: `com.pyfinagent.backend`/`.frontend` are user `gui/` domain LaunchAgents, on macOS Darwin 25.5; the watchdog has exercised this exact command in production (scripts/launchd/backend_watchdog.sh:76) without issue.
2. **APScheduler issue #1095 (Dec 2025)**: "Missed jobs run at next fire time instead of immediately" — active maintainer discussion confirming 3.x does not eagerly re-fire missed cron occurrences in the cases users expected; consistent with the no-double-cycle conclusion.
3. **Feature-flag practice 2026** (digitalapplied 2026 playbook): classify the toggle (these three are release/ops hybrids) and keep the rollback path live ~30 days with explicit exit criteria — maps to keeping the flags revertible via `.env` edit + restart, and to the planned post-flag BQ evidence collection before flag retirement.
4. pydantic-settings docs (current 2025-2026 versions) unchanged on the load-at-instantiation + env-over-dotenv precedence semantics relied on here. No finding supersedes any canonical source used.

## Internal code audit

### 1. Flag definitions + reader sites

Definitions (all `bool = Field(False, ...)`, default OFF, in `backend/config/settings.py`):

| Flag (field name) | Definition | Env var (pydantic-settings case-insensitive field-name mapping; no alias) |
|---|---|---|
| `paper_swap_churn_fix_enabled` | settings.py:311-314 | `PAPER_SWAP_CHURN_FIX_ENABLED` |
| `paper_data_integrity_enabled` | settings.py:42-45 | `PAPER_DATA_INTEGRITY_ENABLED` |
| `paper_risk_judge_reject_binding` | settings.py:277-280 | `PAPER_RISK_JUDGE_REJECT_BINDING` |

Env-file wiring: `_ENV_FILE = Path(__file__).resolve().parent.parent / ".env"` (settings.py:12) -> `backend/.env`; `model_config = {"env_file": str(_ENV_FILE), "env_file_encoding": "utf-8", "extra": "ignore"}` (settings.py:536). No `case_sensitive` override, so `PAPER_..._ENABLED=true` maps to the lowercase field and `"true"` coerces to `True` for `bool` fields.

Reader sites (ALL use `getattr(settings, "<flag>", False)` — tolerant of older Settings objects):

| File:line | Flag | What it gates |
|---|---|---|
| `backend/services/portfolio_manager.py:194-212` (read at :196) | reject_binding | Binding REJECT gate at the candidate-build chokepoint (covers main BUY + swap paths); appends to `blocked_out`, `continue`s before `buy_candidates.append` |
| `backend/services/portfolio_manager.py:471` | churn_fix | `_churn_fix_on` — holding absent from same-cycle `holding_lookup` excluded from swap displacement |
| `backend/services/portfolio_manager.py:561` | churn_fix | `denom = max(abs(holding_score), 1.0 if _churn_fix_on else 0.01)` — kills the ~70,000% sentinel deltas |
| `backend/services/autonomous_loop.py:805` | churn_fix | Re-eval age hours-precise (`total_seconds()/86400`) vs truncated `.days` |
| `backend/services/autonomous_loop.py:1948` | data_integrity | `_di_enabled` in lite Claude analyzer — blocking flags -> `_data_integrity_blocked_analysis` pre-LLM |
| `backend/services/autonomous_loop.py:2228` | data_integrity | Same gate in the lite Gemini mirror path |
| `backend/services/data_integrity.py:17,114` | data_integrity | Pure functions; docstrings state `blocking=True` flags are enforced only when the flag is ON — the settings read itself lives in autonomous_loop (the module takes no settings object) |

**backend/.env duplicate check: NOT COMPLETABLE BY THIS AGENT.** Both `Bash grep` and `Read` on `backend/.env` are permission-denied in the researcher sandbox. Main MUST run, before editing:
`grep -nE "^(PAPER_SWAP_CHURN_FIX_ENABLED|PAPER_DATA_INTEGRITY_ENABLED|PAPER_RISK_JUDGE_REJECT_BINDING)=" backend/.env`
and expect zero hits. Indirect evidence none are set: phase-60.2/60.3/57.1 live_checks all recorded flag-OFF (byte-identical) behavior in the running system, which is impossible if `.env` already carried `=true`. Also note python-dotenv resolves duplicate keys last-occurrence-wins, so even an accidental duplicate is deterministic — but Main should still de-duplicate rather than append blindly. The launchd plist's `EnvironmentVariables` block (see §2) sets only `DEV_LOCALHOST_BYPASS`, `PATH`, `PYTHONUNBUFFERED` — no conflict with the three flags (real env vars would beat `.env` per pydantic-settings precedence; none exist here).

### 2. Restart safety (double-cycle risk) — THE critical question

**Verdict: a restart tonight CANNOT re-fire today's daily job. Code-level certainty, three independent reasons.**

Registration code, verbatim (`backend/api/paper_trading.py:1299-1322`):

```python
def _add_scheduler_job(settings):
    if not _scheduler:
        return
    _scheduler.add_job(
        _scheduled_run,
        "cron",
        hour=settings.paper_trading_hour,
        minute=0,
        day_of_week="mon-fri",
        timezone=ZoneInfo("America/New_York"),
        id=_scheduler_job_id,
        name="Paper trading daily run",  # phase-23.3.1: human-readable label
        replace_existing=True,
        # phase-44.2.X (2026-05-26): default APScheduler misfire_grace_time
        # is 1 second; on 2026-05-25 the cron fired at the right second but
        # event-loop contention from the ticket-queue interval job + polling
        # endpoints pushed dispatch 2.10s late, so APScheduler skipped the
        # run and advanced next_run to tomorrow. A daily job has no harm in
        # running a few seconds (or minutes) late, so we raise the grace
        # window to 1 hour. coalesce=True ensures if multiple windows are
        # missed (e.g. backend down for hours), we run ONCE, not N times.
        misfire_grace_time=3600,
        coalesce=True,
    )
```

And the lifespan wiring (`backend/main.py:264-272`): `scheduler = AsyncIOScheduler()` — constructed with **no jobstores argument**, i.e. the default **in-memory `MemoryJobStore`** — then `init_scheduler(scheduler)` (paper_trading.py:1289-1296, which only calls `_add_scheduler_job`) and `scheduler.start()`.

Reason 1 — **MemoryJobStore has no cross-restart state.** Misfire handling (`misfire_grace_time`, `coalesce`) applies only to run times the scheduler *knows were missed* — i.e. a `next_run_time` already recorded in a job store (persistent store across restarts, or a live process whose event loop stalled). A fresh process builds a fresh scheduler and calls `add_job` anew; APScheduler computes the job's first `next_run_time` as the next fire time **strictly after now**. Verbatim from the INSTALLED package (`.venv/lib/python3.14/site-packages/apscheduler-3.11.2`), `apscheduler/schedulers/base.py:1066-1068` inside `_real_add_job`:

```python
# Calculate the next run time if there is none defined
if not hasattr(job, "next_run_time"):
    now = datetime.now(self.timezone)
    replacements["next_run_time"] = job.trigger.get_next_fire_time(None, now)
```

and `apscheduler/triggers/cron/__init__.py:205-222` (`CronTrigger.get_next_fire_time`): with `previous_fire_time=None` — always the case for a newly added job — `start_date = now` (no trigger `start_date` configured at the call site) and the field search proceeds forward from `datetime_ceil(start_date)`; it can only return a time **at or after now**. A job added at process start therefore has zero past run times for `_process_jobs` to evaluate against `misfire_grace_time` — the misfire/coalesce machinery never even engages. There is no record that 2026-06-11 14:00 ET fired or didn't; the new scheduler's first scheduled fire is 2026-06-12 14:00 ET. No startup catch-up exists for in-memory stores (official docs: "If you always recreate your jobs at the start of your application, then you can probably go with the default (MemoryJobStore)").

Reason 2 — **even the hypothetical grace window is long past.** Today's fire time was 18:00 UTC (14:00 ET; the observed 18:00-19:10 UTC cycle implies `PAPER_TRADING_HOUR=14` in the operator env — settings.py:335 default is 10, ET-denominated). A restart at ~23:30 UTC 06-11 / ~01:30 Oslo 06-12 is >5.5h after the fire time, far outside `misfire_grace_time=3600` (1h) even if a persistent store existed.

Reason 3 — **no run-on-startup code path.** Exhaustive grep for `run_daily_cycle` callers in `backend/`: exactly three — `paper_trading.py:1031` (`dry_run=True` smoke endpoint), `paper_trading.py:1279` (`_run_cycle_background`, the operator-triggered run-now path), `paper_trading.py:1329` (`_scheduled_run`, the cron callback). The lifespan startup calls only `init_scheduler` + `scheduler.start()`; nothing invokes a cycle at boot. (Triple in-cycle guards — run-now 409 + `_running` + cycle_lock — exist independently per phase-49.2 memory.)

**Watchdog interaction (`com.pyfinagent.backend-watchdog`):** the watchdog (`scripts/launchd/backend_watchdog.sh`, StartInterval=60, RunAtLoad=true) restarts the backend with the *same* command — `launchctl kickstart -k "gui/$UID_NUM/com.pyfinagent.backend"` (line 76) — after 3 consecutive `/api/health` failures (lines 20,44-46). launchd enforces ONE instance per service label, so there is no topology in which watchdog + manual kickstart produce two backend processes. Worst case (backend down >3 min during the restart) the watchdog issues a redundant kickstart — another clean restart, not a double process, and per Reasons 1-3 not a double cycle. The watchdog resets its own failure counter after kickstarting (line 79) and on the first healthy check (line 35), so a normal seconds-long restart never reaches the threshold.

**Process-group teardown:** the backend plist (`~/Library/LaunchAgents/com.pyfinagent.backend.plist`) runs `caffeinate -i -s <venv>/uvicorn backend.main:app --host 0.0.0.0 --port 8000` with `KeepAlive=true`, `RunAtLoad=true`, `ThrottleInterval=5`. `kickstart -k` kills the running instance and restarts it (man-page semantics; external §below) — launchd tears down the job's process group (caffeinate parent + uvicorn child). Uvicorn here is **single-process** (no `--workers`, no `--reload` in ProgramArguments), so the CLAUDE.md "kill parent AND child workers" zombie rule reduces to the caffeinate->uvicorn pair, both inside the launchd job. Residual risk is uvicorn's own graceful-shutdown of in-flight requests (external §uvicorn); at ~23:30 UTC no cycle is in flight (today's finished ~19:10 UTC), so nothing is interrupted.

### 3. get_settings() lru_cache semantics

Verbatim (`backend/config/settings.py:539-541`):

```python
@lru_cache()
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]  # pydantic-settings loads from env/.env
```

- The cache is **per-process**: first `get_settings()` call in a new process instantiates `Settings()`, which reads `backend/.env` at that moment; every later call returns the same object. There is no TTL, no SIGHUP re-read, no file-watcher. **Restart is the only deterministic pickup** — confirmed.
- `_scheduled_run` (paper_trading.py:1327) calls `get_settings()` per invocation but receives the cached instance — irrelevant post-restart since the cache is rebuilt from the new env.
- Module-level snapshot sweep: `grep -rn "^settings = |^_settings = |^SETTINGS = " backend --include="*.py"` -> only `backend/agents/mcp_servers/data_server.py:28` (`_settings = None`, lazy init) plus two test-file path constants. **No eager module-level `Settings()` snapshot exists in backend/**; all three flag readers use `getattr(settings, ...)` on objects passed down from `get_settings()` at cycle/request time.
- Cross-process caveat (the only true stale-path class): the **Slack bot** is a separate long-lived process with its own lru_cache — it does NOT execute the trading path (digests/alerts only), so the three flags never matter there; restarting it tonight is optional. Cron wrapper scripts and the harness spawn fresh interpreters per run -> automatic pickup. Frontend never reads backend flags.

### 4. Frontend launchctl label

`~/Library/LaunchAgents/com.pyfinagent.frontend.plist` exists (ls verified, mode 600, dated Apr 8). `launchctl kickstart -k gui/$(id -u)/com.pyfinagent.frontend` is the documented stale-chunk remedy (CLAUDE.md npm-kickstart rule + auto-memory `feedback_npm_install_requires_launchctl_kickstart`: pkill races the launchd watchdog; a stale dev server serves 404 chunk bundles — exactly the reported stale `/login` chunk symptom). No `npm install` happens in 61.1, but the kickstart remedy is install-independent.

## Risks & gotchas (go/no-go)

**VERDICT: GO — restarting the backend tonight (~23:30 UTC 06-11 / 01:30 Oslo 06-12) is safe. The double-cycle risk is ZERO with code-level certainty** (internal audit §2: in-memory job store + forward-only `get_next_fire_time(None, now)` + no run-on-startup call site + restart instant >5.5h past the fire time vs a 1h grace window that could not apply anyway).

Conditions and residual gotchas, in execution order:

1. **Pre-edit .env check (MUST, blocking).** This agent is permission-blocked from `backend/.env`; Main must run `grep -nE "^(PAPER_SWAP_CHURN_FIX_ENABLED|PAPER_DATA_INTEGRITY_ENABLED|PAPER_RISK_JUDGE_REJECT_BINDING)=" backend/.env` and expect zero hits before appending the three `=true` lines (python-dotenv is last-wins on duplicates, but append-blind is sloppy and the 54.1 NoDecode comment shows cron wrappers `set -a; . backend/.env` bash-source this file — keep one line per key, no quotes needed for `true`).
2. **No cycle in flight at restart.** Today's cycle ended ~19:10 UTC; confirm via `curl -s localhost:8000/api/paper-trading/status` (loop status + `next_run`) before kickstart. A kickstart mid-cycle would SIGKILL a running cycle (watchdog comments, backend_watchdog.sh:56-58: kickstart -k bypasses Python finally blocks) — not tonight's situation, but check anyway.
3. **Watchdog cannot double-start** (same `kickstart -k` on the same label; launchd enforces one instance per label). Worst case it issues a redundant restart if `/api/health` is down 3 consecutive minutes — backend warm-up is seconds, so unreachable in practice.
4. **kickstart -k kills caffeinate + uvicorn together** (launchd job teardown; uvicorn is single-process here — no `--workers`, no `--reload` in the plist — so the zombie-children rule is satisfied by construction). `KeepAlive=true` + `ThrottleInterval=5` respawn the service; expect /api/health green within ~10-30s (BQ/macro preloads log after).
5. **Verify the new process postdates b0fe1983 (phase-60.4).** `ps -p $(pgrep -f "uvicorn backend.main" | head -1) -o lstart=` must show a start time after the .env edit; the phase-60.2/3/4 code is already on disk (committed), so process start time is the only deployment variable.
6. **Flag-load verification.** Log line `Paper trading scheduler active: daily at 14:00 ET` (paper_trading.py:1296) confirms init; for the flags themselves, `.venv/bin/python -c "from backend.config.settings import get_settings; s=get_settings(); print(s.paper_swap_churn_fix_enabled, s.paper_data_integrity_enabled, s.paper_risk_judge_reject_binding)"` proves the .env parses to `True True True` (fresh interpreter = same read path the backend takes at boot). For the RUNNING process, the next_run timestamp from /api/paper-trading/status must read 2026-06-12T14:00:00-04:00 — that simultaneously proves the no-double-cycle conclusion live.
7. **Frontend kickstart is independent and safe at any hour** (`com.pyfinagent.frontend` label confirmed on disk; documented stale-chunk remedy). Playwright-capture /login after it settles (per the UI-verification rule).
8. **Slack bot does NOT need a restart** for these flags (separate process, never executes the trading path). Do not touch it — it is crontab-monitored, not launchd (54.2 memory), and a pkill would be pure risk.
9. **First post-flag evidence is 2026-06-12 18:00 UTC.** Expect: swap-path logs using the 1.0-clamp denominator (portfolio_manager.py:561), any non-US blocking integrity flag producing `_data_integrity_blocked_analysis` rows (autonomous_loop.py:1948/2228), and any REJECT verdict appending to `blocked_out` instead of `buy_candidates` (portfolio_manager.py:196-212). Absence-of-trigger is also valid evidence (e.g. zero REJECTs that cycle) — record what fired and what had no occasion to.
10. **Fowler discipline note:** both flag states are already tested (phase-60/57 shipped OFF-byte-identity + ON-behavior tests), which is exactly the canonical "test the production-intended config AND the fall-back config" requirement — the flip itself is the residual untested surface, hence the 18:00 UTC evidence collection. Keep the rollback path (flags back to false + restart) live for the LaunchDarkly/2026-playbook ~30-day window before any flag-retirement refactor.

## Recommendations

1. Execute in this order: .env grep (item 1) -> append three lines -> backend kickstart -> health + process-age + flag-load checks (items 5-6) -> frontend kickstart -> Playwright /login capture -> wait for 2026-06-12 18:00 UTC cycle -> BQ evidence pull. No step requires waiting for another day.
2. Capture for `live_check_61.1.md`: the grep output, the `ps -o lstart=` line, the three-flag `True True True` print, the /status next_run JSON, the Playwright screenshot path, and (next day) the BQ rows. That converts every claim in this brief into operator-auditable artifacts.
3. The restart can happen tonight with zero scheduling risk; equally, there is no urgency-forcing reason it must (next cycle is 18.5h out). If the operator prefers daylight, tomorrow before ~17:30 UTC is equivalent — the only hard constraint is restarting BEFORE the 18:00 UTC cycle so the flags govern it.
4. Plan (not now) flag retirement per Fowler: once the flags have survived their validation window, fold the fixed behavior in as default and delete the dead OFF branches — release toggles "should generally not stick around much longer than a week or two"; these are operator-gated so a few weeks is fine, but do not let them become permanent inventory.

## JSON envelope

```json
{
  "tier": "simple",
  "external_sources_read_in_full": 5,
  "snippet_only_sources": 12,
  "urls_collected": 36,
  "recency_scan_performed": true,
  "internal_files_inspected": 12,
  "report_md": "handoff/current/research_brief.md",
  "gate_passed": true
}
```

Gate basis: 5 authoritative external sources fetched in full via WebFetch (2 official docs, 1 official man-page mirror, 1 official docs source via GitHub after the site refused connections, 1 canonical practice article); 36 unique URLs collected across 6 queries; recency scan performed with 4 reported findings; every internal claim carries file:line anchors. The one item this agent could not complete (backend/.env duplicate grep — sandbox permission denial on that file) is converted into a blocking pre-edit command for Main with indirect evidence already supporting the expected zero-hit result; it is an execution-step precondition, not an unresolved research question.
