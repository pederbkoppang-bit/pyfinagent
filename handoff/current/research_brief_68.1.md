# Research Brief — phase-68.1

**Step**: EXECUTION_BACKEND demonstrably reaches execution_router in the launchd process
**Tier**: moderate
**Date**: 2026-08-07
**Constraints honoured**: read-only; $0 metered; no service restarted; no code changed;
no credential value reproduced; `backend/.env` not read (sandbox-denied — see §1.3).

## Headline findings (read these three first)

1. **All three clauses of the step's `audit_basis` are CONFIRMED** — and one is *worse* than
   stated: there is no `execution_backend` field on `Settings` **at all**, so the settings
   leg must be *created*, not merely wired (§1.3).
2. **Criterion 4(b)'s "PKLIVE-class" discriminator rests on a false premise.** Alpaca's own
   documentation describes **no prefix or format difference whatsoever** between paper and
   live API keys — the separation is by *domain*, not by key shape (§2.2). The existing
   `_refuse_live_keys()` guard therefore protects against a string Alpaca does not document
   as existing. The criterion is immutable, so it must still be tested as written, but the
   brief recommends what to add so the leg is genuinely load-bearing.
3. **Criterion 1's live_check CANNOT be satisfied without a backend restart** (§1.7/§3.2) —
   this is structural, not a matter of effort. Measured restart outage is **2.455 s**, the
   watchdog cannot trip on it, and the only real hazard is a ~±2-minute window around the
   23:00 CEST digest (§3.2).

---

## Part 1 — Internal code inventory

### 1.1 The mode-resolution path (re-derived, NOT trusted from notes)

`backend/services/execution_router.py` is **329 lines**.

- `execution_router.py:37-39` — `BackendMode = Literal["bq_sim","alpaca_paper","shadow"]`;
  `VALID_MODES = ("bq_sim","alpaca_paper","shadow")`; `DEFAULT_MODE: BackendMode = "bq_sim"`.
  **There is no `alpaca_live` / `live` member at all.**
- `execution_router.py:65-71` — `_current_mode()`:
  ```python
  raw = (os.getenv("EXECUTION_BACKEND") or DEFAULT_MODE).strip().lower()
  if raw not in VALID_MODES:
      logger.warning("unknown EXECUTION_BACKEND=%r; falling back to %s", raw, DEFAULT_MODE)
      return DEFAULT_MODE
  return raw
  ```
  Reads **`os.getenv` ONLY** — never `settings`. Unknown value → WARN + `bq_sim`.
- `execution_router.py:268-269` — `ExecutionRouter.__init__`:
  `self.mode: BackendMode = mode or _current_mode()`.
  The caller's cited anchor is **CORRECT**: `__init__` opens at `:268`, the
  `mode or _current_mode()` assignment is at **:269**. Re-derived, confirmed.
- **Mode is resolved PER CONSTRUCTION, not at import.** The module docstring's claim at
  `:3-4` ("selected at import time by the EXECUTION_BACKEND env-var") is **STALE and WRONG**.
  This matters for the design: a single import-time banner would not reflect a later
  per-instance override, so the startup line must be explicit about what it is reporting.

**Is there an existing startup log of the resolved mode? NO.**
The router's only `logger` calls are `:68` (unknown value), `:172` (bad
`ALPACA_MAX_NOTIONAL_USD`), `:288` (shadow-alpaca failure), `:322` (`flip_to`), `:328`
(`rollback_to_bq_sim`). **Nothing logs the resolved mode on the happy path and nothing at
all is emitted at process start.** Criterion 1's log line does not exist in any form.

### 1.2 Every caller of the router

| Call site | Anchor | Notes |
|---|---|---|
| `paper_trader.execute_buy` | `backend/services/paper_trader.py:406` | `router = ExecutionRouter()` — no mode arg ⇒ `_current_mode()` |
| `paper_trader.execute_sell` | `backend/services/paper_trader.py:568` | same idiom (`:566` comment "mirrors execute_buy") |
| `AlpacaBroker.submit_order` | `backend/markets/alpaca_broker.py:89-100` | **bypasses `ExecutionRouter`**; calls `_alpaca_real_fill` / `_alpaca_mock_fill` directly, gated on `_has_creds()` |
| `broker_base` | `backend/markets/broker_base.py:25` | re-exports `FillResult` only |
| `rollback_to_bq_sim()` | `execution_router.py:325-329` | docstring says "used by circuit breaker" — **no in-repo caller found by grep** (dead helper) |
| tests (12 files) | e.g. `backend/tests/test_price_tolerance_gate.py:76` | all patch `backend.services.paper_trader.ExecutionRouter` |

The live money path is exactly the two `paper_trader` call sites. Both construct with no
mode ⇒ both resolve via `_current_mode()` ⇒ both are `bq_sim` in the launchd process today.

**Note for the design:** `AlpacaBroker` (`alpaca_broker.py:89-100`) is a *second* door to
the Alpaca fill functions that does **not** consult `EXECUTION_BACKEND` at all. It is not on
the live path today, but any "mode never escalates" claim should be scoped to
`ExecutionRouter`, or the claim over-reaches. (Class: *guards stop one seam short* — see
memory `feedback_guards_stop_one_seam_short`.)

### 1.3 The audit_basis claim, verified mechanically

All three clauses **CONFIRM**, with one refinement that makes the work larger:

1. *"pydantic env_file loads into settings without exporting to os.environ"* — **TRUE**
   (external confirmation in §2.1), **and there is no field to load into.**
   - `backend/config/settings.py:632` —
     `model_config = {"env_file": str(_ENV_FILE), "env_file_encoding": "utf-8", "extra": "ignore"}`
   - `backend/config/settings.py:16` — `class Settings(BaseSettings)`; `:8` imports
     `from pydantic_settings import BaseSettings, NoDecode`.
   - `grep -in execution backend/config/settings.py` returns **only 3 hits**, none of them a
     field: `:222` a `# --- Execution Mode ---` section header that actually introduces
     **Celery** (`use_celery` at `:223`) — a false friend; `:326` a phase-57.1 REJECT-gate
     description; `:582` a comment mentioning ExecutionRouter in the price-tolerance
     rationale. **There is NO `execution_backend` field.**
   - `extra: "ignore"` means an `EXECUTION_BACKEND=` line already in `backend/.env` is
     **silently dropped** today — no error, no warning (§2.1 quotes the docs on this).
   - No `load_dotenv` anywhere in the backend runtime: the only hits are
     `backend/backtest/spot_checks_harness.py:36-37` (standalone script) and test assertions
     in `backend/tests/test_phase_75_deploy_surface.py:554,560`. `backend/main.py` does not
     call it.
2. *"the launchd plist carries no EXECUTION_BACKEND key"* — **CONFIRMED**, §1.4.
3. *"execution_router silently defaults to bq_sim forever"* — **CONFIRMED** by 1 + 2.

**Researcher-sandbox gap Main must close:** `backend/.env` is denied to this sandbox (a Bash
call touching it was blocked mid-session), so I cannot say whether an `EXECUTION_BACKEND=`
line already exists there. Given `extra:"ignore"` + the router reading `os.environ` only, it
would be **inert either way** — but Main should run
`grep -c '^EXECUTION_BACKEND' backend/.env` and record the count (not the value) in the
live_check, because a pre-existing line changes the story an operator will tell about
criterion 2.

### 1.4 The launchd plist (read in full; key NAMES only)

`~/Library/LaunchAgents/com.pyfinagent.backend.plist` — `EnvironmentVariables` contains
**exactly four keys, by name**:

`CLAUDE_CODE_OAUTH_TOKEN`, `DEV_LOCALHOST_BYPASS`, `PATH`, `PYTHONUNBUFFERED`.

**`EXECUTION_BACKEND` is ABSENT. `ALPACA_API_KEY_ID` / `ALPACA_API_SECRET_KEY` /
`ALPACA_PAPER_TRADE` / `ALPACA_MAX_NOTIONAL_USD` are ALL ABSENT.**

**SECURITY NOTE (re-confirmed, value NOT reproduced):** the `CLAUDE_CODE_OAUTH_TOKEN` value
is stored as **plaintext** in this plist, and it also appears **malformed** (a doubled
`sk-ant-oat01-sk-ant-oat01-` prefix and an embedded line break). Out of scope for 68.1 — do
not fix it here — but it is a real finding and should be queued as its own step per
`feedback_queue_discovered_defects_in_masterplan`. The operational rule it implies for
*this* step: **do not add a second secret to this file.** `EXECUTION_BACKEND=bq_sim` is not
a secret, so plist carriage is fine here.

Other plist facts used in §3.2: `KeepAlive` = `<true/>`, `RunAtLoad` = `<true/>`,
`ThrottleInterval` = `5`, `ProcessType` = `Interactive`, `WorkingDirectory` = repo root,
stdout **and** stderr → `/Users/ford/.openclaw/workspace/pyfinagent/backend.log`, launched
via `/usr/bin/caffeinate -i -s <venv>/bin/uvicorn backend.main:app --host 0.0.0.0 --port 8000`.

### 1.5 The Alpaca creds path — does it "silently mock-fill"? YES, literally silently.

Creds are read from **`os.environ` only**, never from settings:

| Reader | Anchor |
|---|---|
| `_refuse_live_keys()` | `execution_router.py:75-76` |
| `_alpaca_real_fill()` | `execution_router.py:193-194` (`os.environ[...]` — KeyError if absent) |
| `submit_order` gate | `execution_router.py:277` |
| `shadow` gate | `execution_router.py:284` |
| `shadow_submit` gate | `execution_router.py:305` |
| `AlpacaBroker._has_creds` | `backend/markets/alpaca_broker.py:35-38` |
| `AlpacaBroker._trading_client` | `alpaca_broker.py:69-77` |

Settings **also** carries `alpaca_api_key_id` / `alpaca_api_secret_key` as `SecretStr` at
`backend/config/settings.py:132-133`, but their own descriptions scope them to
`data.alpaca.markets/v1beta1/news` — the **news-adapter channel**, a different consumer.
Because pydantic does not export to `os.environ`, a populated `backend/.env` leaves the
**router** creds-blind. Two disjoint credential channels for one vendor is a latent defect
worth flagging. **If anyone unifies them, beware the SecretStr truthiness trap**: a
non-empty `SecretStr` is truthy but `str()` yields `'**********'` — use an explicit unwrap
(memory `project_secretstr_dead_overlays`; this class already killed 4 alpha overlays).

**The silent mock-fill, verbatim** (`execution_router.py:276-280`):
```python
if self.mode == "alpaca_paper":
    if os.getenv("ALPACA_API_KEY_ID") and os.getenv("ALPACA_API_SECRET_KEY"):
        return _alpaca_real_fill(symbol, qty, side, client_order_id)
    return _alpaca_mock_fill(symbol, qty, side, client_order_id, close_price)
```
`_alpaca_mock_fill` (`:128-154`) contains **zero logger calls at any level**. So today, with
`mode=alpaca_paper` and no creds, the system fabricates a fill at a hardcoded 30 bps
slippage (`:139`) stamped `source="mock_alpaca"` (`:150`) and says **nothing**. Criterion
3's premise is **CONFIRMED exactly as written.**

Refinement: the one surviving trace is the `source` value `mock_alpaca` written to BQ via
`exec_source` (`paper_trader.py:412`). That is a good *detection* channel for a live_check
assertion, but it is post-hoc, not a startup signal.

### 1.6 Paper-only triple-enforcement — present/absent today

| Leg | Present today? | Anchor | Assessment |
|---|---|---|---|
| **(a) paper base URL pinned** | **PARTIAL — implicit, never asserted** | `execution_router.py:228` `TradingClient(key, secret, paper=True)`; `alpaca_broker.py:73-77` same | Pinned *by the SDK as a consequence of* `paper=True`; no explicit URL constant in the repo and **no test asserts the resulting URL**. Now provably assertable offline — see §3.3. The data-API call at `:205` uses `https://data.alpaca.markets/...`, which is **shared between paper and live** and is read-only (snapshot) — correctly not a trading endpoint, but a reader could mistake it for a leak, so the test should say so. |
| **(b) live-key rejection** | **PRESENT but the discriminator is unfounded — §2.2** | `execution_router.py:74-80` `_refuse_live_keys()`: `key.startswith("PKLIVE")` or `ALPACA_PAPER_TRADE=="false"` | Also **narrowly reachable**: called only from `_alpaca_real_fill:188`. It does NOT run at startup, NOT on the mock path, NOT when mode is `bq_sim`. So today it is unreachable in production. |
| **(c) mode never escalates** | **PRESENT by construction, untested, and inconsistent** | `:38` `VALID_MODES`, `:67-70` unknown→`bq_sim`, `:319-320` `flip_to` validates, `:290` `raise RuntimeError(f"unsupported mode: {self.mode}")` | Env-supplied escalation is genuinely blocked (`EXECUTION_BACKEND=alpaca_live` → WARN → `bq_sim`). **But `ExecutionRouter(mode="alpaca_live")` is NOT validated in `__init__`** (`:268-269` assigns whatever is passed); it fails only later at `:290` on submit. Fails closed, but via a late raise, not a constructor guard. **Zero tests cover any of the three legs.** |

### 1.7 Restart cost and dependants — MEASURED, not estimated

**(i) What a backend restart costs.** From `backend.log`, the most recent restart
(2026-08-06):

| Event | Timestamp |
|---|---|
| `Shutting down` | 13:46:15,324 |
| `Application shutdown complete` / `Finished server process [79058]` | 13:46:15,429 |
| `PyFinAgent backend starting` | 13:46:17,702 |
| `Application startup complete.` | 13:46:17,779 |

**Total outage = 2.455 s.** The earlier restart that day (13:24:51) shows the same shape.
Current process `89530` has uptime `01-06:48:39` (1 d 6 h 48 m), consistent with that
restart. `launchctl list` shows `com.pyfinagent.backend` PID `89530`, last exit `-15`
(SIGTERM — a clean prior `kickstart -k`, not a crash). `KeepAlive=true` + `RunAtLoad=true`
+ `ThrottleInterval=5` mean `launchctl kickstart -k gui/$(id -u)/com.pyfinagent.backend`
brings it straight back; measured 2.5 s startup sits under the 5 s throttle floor.

**(ii) Does anything alarm?**
- **`com.pyfinagent.backend-watchdog`** runs `scripts/launchd/backend_watchdog.sh` every
  60 s (`StartInterval 60`). Read in full: `FAILURE_THRESHOLD=3`, `TIMEOUT=5`, counter file
  `~/Library/Caches/com.pyfinagent.backend.watchdog.fails`, reset to 0 on any 200.
  **It needs 3 consecutive minute-spaced failures (~3 min) before acting, and it does not
  page — it dumps stacks via SIGUSR1 and kickstarts.** A 2.5 s outage **cannot** trip it.
  Verdict: the watchdog is a non-issue for this restart.
- **The Slack evening digest IS the real constraint, and the caller's premise is correct.**
  `backend/slack_bot/scheduler.py:239-248` schedules `_send_evening_digest` as
  `"cron", hour=settings.evening_digest_hour, minute=0, timezone=ZoneInfo("America/New_York")`.
  Default `evening_digest_hour = 17` (`backend/config/settings.py:627`) ⇒ **17:00 ET =
  23:00 CEST**, matching the caller's statement. It fetches
  `{_LOCAL_BACKEND_URL}/api/paper-trading/portfolio` at
  `scheduler.py:597`. Failure modes differ and this distinction matters:
  - non-200 response → `portfolio_data = {}` (`:598`), digest still posts but **empty** —
    silent degradation, no page;
  - **connection refused** (exactly what a mid-restart backend gives) → `httpx` raises →
    caught at `:636` → **`_route_exception_to_p1(exc, endpoint="evening_digest")` PAGES P1.**
  So a restart landing on 23:00:00 CEST both breaks the digest *and* fires a P1.
  `_send_evening_digest` also **skips entirely on non-US-trading days**
  (`scheduler.py:590-593`). **2026-08-07 is a Friday and a US trading day, so tonight's
  digest will fire.**
- The morning digest (`:227-235`) and a third job (`:263-267`) are on the same
  `America/New_York` timezone; the morning one is 08:00 ET = 14:00 CEST.
- Per memory `project_backend_restart_safety`, a restart **cannot** double-fire
  `paper_trading_daily`.

**(iii) Can criterion 1 be met WITHOUT a restart? NO.** There is:
- no existing startup line to grep (§1.1 — the router logs nothing at start);
- no endpoint exposing the mode — `grep -rn "execution_mode|current_mode|\"mode\""
  backend/api/` returns **zero hits**;
- no import-time resolution to observe (mode is per-construction, `:269`).

The line must first be *written* in GENERATE, and a newly-written log line cannot appear in
a process that started before the code existed. **The restart is structurally unavoidable.**
See §3.2 for the window.

### 1.8 Test precedent and tooling

- `pytest-timeout` **is installed** (imports cleanly in `.venv`), so `--timeout=120` in the
  immutable command works. Versions: `pytest 9.0.3`, `pydantic 2.12.5`,
  `pydantic-settings 2.13.1`, `alpaca-py 0.43.2`.
- `backend/tests/test_execution_backend_wiring.py` **does not exist** (confirmed) — it is a
  GENERATE deliverable, and the filename is fixed by the immutable command.
- Closest house analogues: `test_phase_37_2_default_alignment.py:49`
  (`..._settings_without_env_or_dotenv_resolves_to_...` — the exact shape criterion 2
  needs), `test_phase_50_6_settings_paper_markets.py`, `test_phase_39_1_autoresearch_env.py`,
  `test_phase_40_6_env_syntax_check.py`.
- **`get_settings()` is `@lru_cache()`** (`settings.py:635-637`), so any test that
  manipulates env **must** call `get_settings.cache_clear()`. Six existing test files already
  do this (e.g. `test_phase_75_bq_discipline.py`, `test_phase_82_13_preload_refusal_handling.py`)
  — follow that idiom, do not invent a new one.
- `backend/tests/conftest.py:21` sets `PYFINAGENT_TEST_NO_BQ=1`; there is no global env
  fixture, so use `monkeypatch.setenv` / `monkeypatch.delenv(..., raising=False)` per test.

---

## Part 2 — External research

### Search-query composition (three-variant discipline)

- **Year-less canonical**: "Alpaca API key prefix paper vs live account key format PK";
  "pydantic-settings env_file does not set os.environ dotenv precedence";
  "logging configuration provenance which source a config value came from".
- **Current-year (2026)**: "fail fast missing credentials startup validation trading system
  2026 safe mode guard"; "twelve-factor config environment variables criticism 2026 config
  provenance startup log".
- **Last-2-year (2025-2026)**: "pydantic settings 2025 2026 environment variable precedence
  changes best practices".

### Read in full (8; gate floor is 5)

| URL | Accessed | Kind | Fetched how | Key finding / quote |
|---|---|---|---|---|
| https://pydantic.dev/docs/validation/latest/concepts/pydantic_settings/ | 2026-08-07 | official docs | WebFetch (via 301 from docs.pydantic.dev) | Precedence, verbatim: "1. …CLI. 2. Arguments passed to the `Settings` class initialiser. 3. Environment variables… 4. Variables loaded from a dotenv (`.env`) file. 5. …secrets directory. 6. The default field values". And: "**environment variables will always take priority over values loaded from a dotenv file**". Dotenv feeds *model fields*, not `os.environ`. |
| https://docs.alpaca.markets/us/v1.1/docs/authentication-1 | 2026-08-07 | official docs | WebFetch | Headers `APCA-API-KEY-ID` / `APCA-API-SECRET-KEY`. **"There is no documented prefix or format that distinguishes paper-trading credentials from live-trading ones."** Live `https://api.alpaca.markets`; paper `https://paper-api.alpaca.markets`. |
| https://docs.alpaca.markets/us/docs/paper-trading | 2026-08-07 | official docs | WebFetch | "Your paper trading account will have a different API key from your live account." Paper base URL `https://paper-api.alpaca.markets`. **No prefix/format convention documented.** |
| https://alpaca.markets/learn/connect-to-alpaca-api | 2026-08-07 | official (vendor learn) | WebFetch | Keys generated identically regardless of account type; no example key format shown. "Set `paper=True` to place simulated trades in your paper account, or `paper=False` to send real orders to your live account." |
| https://12factor.net/config | 2026-08-07 | canonical methodology | WebFetch | "strict separation of config from code"; litmus test: "the codebase could be made open source at any moment, without compromising any credentials"; env vars as "granular controls, each fully orthogonal to other env vars"; argues against grouped named environments. |
| https://andrewlock.net/debugging-configuration-values-in-aspnetcore/ | 2026-08-07 | authoritative blog (named practitioner) | WebFetch | Prior art for config *provenance*: `GetDebugView()` / `GetValueAndProvider()` print each key **with its originating provider**. "being able to see exactly where a configuration value comes from is invaluable when things aren't working as you expect." |
| https://stuart.mchattie.net/posts/2026/03/07/pydantic-settings-safer-config/ | 2026-08-07 | blog, **2026-03-07** | WebFetch | Recency corroboration of the same precedence chain. "Fail fast by keeping required values required." Notably **does not** recommend logging resolved config — the provenance-logging idea is not yet mainstream pydantic advice. |
| https://forum.alpaca.markets/t/questions-1-build-scanner-2-paper-vs-live-api-key/7509 | 2026-08-07 | community (lowest tier) | WebFetch | **Negative evidence, deliberately included**: the thread contains **no** staff or user statement of any PK/AK/PKLIVE prefix convention. Recorded so the §2.2 conclusion is not resting on doc silence alone. |

### Identified but snippet-only (context; does NOT count toward the gate)

| URL | Kind | Why not fetched in full |
|---|---|---|
| https://deepwiki.com/pydantic/pydantic-settings/3.1-environment-variables-and-.env-files | derived doc | superseded by the official docs page |
| https://docs.pydantic.dev/1.10/usage/settings/ | official (v1) | wrong major version for this repo |
| https://pydantic.dev/docs/validation/2.9/concepts/pydantic_settings/ | official (2.9) | repo runs 2.13.1; `latest` read instead |
| https://itnext.io/the-12-factor-app-15-years-later-does-it-still-hold-up-in-2026-… | blog 2026 | Medium identity redirect blocked the fetch; snippet used for the recency scan |
| https://medium.com/lets-code-future/we-followed-the-12-factor-app-it-made-debugging-impossible-976cd0bc40e7 | blog | paywalled/redirect; snippet used |
| https://blog.doismellburning.co.uk/twelve-factor-config-misunderstandings-and-advice/ | blog | duplicate of the criticism already captured |
| https://allenap.me/posts/12-factor-app-config-in-the-environment-is-bad-advice | blog | ditto |
| https://gist.github.com/telent/9742059 | gist | ditto (env-var leakage argument) |
| https://github.com/node-config/node-config/issues/217 | issue tracker | prior art for provenance; andrewlock covers it better |
| https://en.wikipedia.org/wiki/Fail-fast_system | encyclopedia | definition only |
| https://docs.python.org/3/library/logging.config.html | official docs | about logging config, not config provenance |
| https://logging.apache.org/log4j/…/ConfigurationSource.html | official API | Java-specific, not applicable |
| https://oneuptime.com/blog/post/2026-02-20-twelve-factor-app-guide/view | blog 2026 | recency scan snippet |
| https://www.evnx.dev/blog/python-env-mastery-dotenv-pydantic | blog | duplicate of pydantic precedence |
| https://towardsdatascience.com/manage-environment-variables-with-pydantic/ | blog | ditto |
| https://apispine.com/alpaca/authentication | third-party | unofficial; official Alpaca pages preferred |
| ~25 further hits across the 6 searches | mixed | de-duplicated / lower tier |

**Unique URLs collected across all searches: 40+.**

### 2.1 pydantic-settings: the audit_basis is externally confirmed

The official docs give the precedence chain verbatim (table above). Two consequences bind
this step's design:

1. **The `.env` file is a *source for model fields*, not a writer to `os.environ`.** Nothing
   in the docs describes any export to the process environment. So a router that calls
   `os.getenv` is **unreachable from `backend/.env`** — exactly the audit_basis claim.
2. **`extra="ignore"` silently swallows unknown `.env` keys.** Verbatim: "if you set the
   `extra=forbid` (*default*) … and your dotenv file contains an entry for a field that is
   not defined in settings model, it will raise `ValidationError`." This repo sets
   `extra:"ignore"` (`settings.py:632`), so the error is suppressed. An operator who typed
   `EXECUTION_BACKEND=alpaca_paper` into `backend/.env` today would get **no feedback at
   all** — no error, no warning, no behaviour change. That is the strongest argument for
   criterion 1's provenance log line: it is the only mechanism that would tell that operator
   the truth.

The 2026 article corroborates the chain and adds "Fail fast by keeping required values
required" — relevant to criterion 3, though note it argues for *required fields*, which is
the wrong tool here (`execution_backend` must have a default of `bq_sim` for criterion 2).

### 2.2 The PKLIVE question — the criterion's discriminator is unfounded

The caller asked whether "PKLIVE-class" is even the right discriminator. **It is not, and
this should be disclosed rather than quietly implemented.**

Three official Alpaca sources, read in full, agree:
- Auth reference: "There is no documented prefix or format that distinguishes
  paper-trading credentials from live-trading ones. Instead, the difference is managed
  through separate credential sets."
- Paper-trading doc: keys differ between accounts, but "does not describe any specific
  naming patterns or prefixes."
- Vendor learn page: no example key format is published at all.
- The community thread adds no counter-evidence.

**Alpaca separates paper from live by DOMAIN, not by key shape**
(`https://paper-api.alpaca.markets` vs `https://api.alpaca.markets`). Therefore
`execution_router.py:76`'s `key.startswith("PKLIVE")` tests for a string Alpaca does not
document as existing. It is not *harmful* — a key that did start with `PKLIVE` would be
refused — but it provides **no assurance**, and a reader who trusts the docstring at `:11-13`
("Triple-enforced paper-only: … (2) this router refuses PKLIVE-prefix keys") will
over-estimate the protection. This is a documentation-honesty defect as much as a code one.
(Consistent with the phase-68.0 Q/A critique, which already used the phrase "PKLIVE-folklore
honesty" — this brief now supplies the primary-source basis for that judgement.)

**The genuinely load-bearing guard is the domain**, and it is now provably assertable
offline — see §3.3 leg (a).

### 2.3 Config provenance is established practice, and the repo already has the idiom

The ASP.NET Core prior art (`GetDebugView` / `GetValueAndProvider`) shows that printing a
resolved value **together with the provider that supplied it** is a mature, framework-level
practice, motivated exactly as criterion 1 motivates it: layered sources mean "developers
waste time troubleshooting seemingly mysterious configuration failures caused by unexpected
provider precedence." node-config issue #217 makes the same request for the Node ecosystem.

**Crucially, pyfinagent already does this — there is an in-repo precedent to copy rather
than invent.** `backend/main.py:181-184`:
```python
logging.info(
    "phase-31.1 model routing: settings.gemini_model='%s' -> standard-tier provider=%s",
    _std_model, _std_provider,
)
```
This logs a resolved value *and its consequence*, followed by a `logging.warning` at
`:186-193` when the resolution is suspicious. `_warn_if_allowlist_empty` (`main.py:130-150`)
is the second instance of the pattern. Criterion 1's line should be a third sibling in the
same `lifespan` block, with the same ASCII-only discipline (`.claude/rules/security.md`
forbids non-ASCII in logger messages — no arrows, no em dashes).

### 2.4 Recency scan (last 2 years, 2024-2026) — PERFORMED

**Result: no finding supersedes the canonical sources; two findings reinforce the design.**

- **pydantic-settings**: the March 2026 article states the identical precedence chain as the
  official docs. Repo runs `pydantic-settings 2.13.1`, `pydantic 2.12.5` — current. **No
  breaking change to env/dotenv precedence found in the 2024-2026 window.**
- **Alpaca**: no 2024-2026 source introduces a paper-vs-live key prefix. The domain split is
  unchanged. **The PKLIVE premise is not merely undocumented today — nothing in the recent
  window created it either.**
- **Twelve-factor (2026 retrospectives)** — this is the one place where recent work
  *qualifies* a canonical source, and it argues *for* criterion 1. The 2026 commentary
  ("The 12-Factor App — 15 Years later", ITNEXT; "We Followed the 12-Factor App. It Made
  Debugging Impossible.", Medium) reports that teams "spend significant time trying to figure
  out which of 40+ environment variables were misconfigured during production outages," and
  that the methodology is "not wrong, but incomplete." A further technical criticism: "Any
  code in an application can read or mutate environment variables, and they're not
  thread-safe."
  **Application to 68.1:** the 2026 critique is precisely the failure mode this step exists
  to fix — env-var config with no visibility into what was actually resolved. It also gently
  argues against making `os.environ` the *only* channel, which supports the
  env → settings → default chain over a pure-env design.
- **Fail-loud on missing credentials**: the 2026-scoped search surfaced no
  trading-specific literature; the canonical fail-fast framing ("immediately report any
  condition likely to indicate a failure … rather than attempt to continue a possibly flawed
  process") is the applicable principle and matches criterion 3's intent. **Reported as a
  genuine gap, not padded.**

---

## Part 3 — Application to pyfinagent

### 3.1 Per-criterion REMAINING-WORK table

| # | Criterion (abbrev.) | State today | Anchor | REMAINING WORK |
|---|---|---|---|---|
| **1** | `EXECUTION_BACKEND` reaches `execution_router`; startup log prints resolved mode **AND** source (env/.env/default) from the real launchd process | **NOT DONE.** Router reads `os.getenv` only; no settings field; plist lacks the key; **no startup log of any kind**; no endpoint exposes the mode | `execution_router.py:65-71,268-269`; `settings.py` (no field); plist `EnvironmentVariables` (4 keys); `grep` of `backend/api/` → 0 hits | (a) add `execution_backend: str = Field("bq_sim", ...)` to `Settings`; (b) change `_current_mode()` to resolve `os.environ` → `settings` → `DEFAULT_MODE` **and return the source**; (c) emit the banner in `main.py` `lifespan` next to `:181-184`, ASCII-only; (d) add `EXECUTION_BACKEND=bq_sim` to the plist (explicit, matches default ⇒ still byte-identical); (e) **restart** + capture the line (§3.2) |
| **2** | With `EXECUTION_BACKEND` set nowhere, behaviour byte-identical to today's `bq_sim` default (test-asserted) | **TRUE today but UNTESTED.** `DEFAULT_MODE="bq_sim"` at `:39`; zero tests exercise `_current_mode()` | `execution_router.py:39,65-71` | Test-assert: with `os.environ` cleared **and** no settings value, `_current_mode()` returns `"bq_sim"` and source is `"default"`; and `ExecutionRouter().mode == "bq_sim"`. Mirror `test_phase_37_2_default_alignment.py:49`. **Must call `get_settings.cache_clear()`** (`settings.py:635`) |
| **3** | Creds absent while `mode=alpaca_paper` logs LOUDLY (single unmissable startup error naming the missing keys) instead of silently mock-filling | **NOT DONE — premise CONFIRMED.** `:277-280` falls through to `_alpaca_mock_fill`, which has **zero logger calls** | `execution_router.py:276-280`, `:128-154` | Add a startup-time check in `lifespan`: if resolved mode is `alpaca_paper`/`shadow` and either `ALPACA_API_KEY_ID` or `ALPACA_API_SECRET_KEY` is missing from `os.environ`, emit **one** `logging.error` **naming the missing key(s) by name** (never values). Keep the per-order mock fallback (do not change behaviour — DARK), but add a `logger.warning` at the `:280` fallback so the post-hoc path is no longer silent. Test both the startup error and the fallback warning |
| **4a** | Paper base URL pinned | **PARTIAL — implicit, never asserted** | `execution_router.py:228`; `alpaca_broker.py:73-77` | Test-assert offline (§3.3) that `TradingClient(k,s,paper=True)._base_url` is `BaseURL.TRADING_PAPER` == `https://paper-api.alpaca.markets`, and that the repo never passes `url_override` |
| **4b** | Live-key pattern (PKLIVE-class) rejected | **PRESENT but discriminator unfounded (§2.2) and unreachable in prod** | `execution_router.py:74-80`, called only from `:188` | Test the guard **as written** (criterion is immutable): a `PKLIVE…` key raises, `ALPACA_PAPER_TRADE=false` raises. **AND disclose in `experiment_results.md`** that Alpaca documents no such prefix, so this leg is not the real protection — 4a is. Recommend correcting the misleading docstring at `:11-13` |
| **4c** | Mode never escalates beyond paper regardless of env values | **PRESENT by construction, untested, inconsistent** | `:38`, `:67-70`, `:290`, `:319-320` | Test-assert: `EXECUTION_BACKEND` ∈ {`alpaca_live`, `live`, `LIVE`, `""`, `"  "`, `"BQ_SIM"`, garbage} → resolves to a member of `VALID_MODES` and never to a live mode. Note the **`__init__` gap** (`:269` accepts an unvalidated mode) in the results file; fixing it is arguably in scope, but it is a behaviour change — recommend a constructor validation raise, flagged for Q/A |
| **5** | No trading-behaviour change (DARK); no scheduled cycle executes through any new path; fresh Q/A PASS with 67.1 gates | **Achievable.** Default stays `bq_sim`; plist value set to `bq_sim`; changes are a new field + resolution source + log lines + tests | — | Prove DARK: `git diff` shows no change to fill logic; assert `mock_alpaca` never becomes reachable at the default; confirm no cycle ran through a new path between the restart and the flip |

### 3.2 The restart recommendation (criterion 1's live_check)

**Can criterion 1 be satisfied without a restart? No** — §1.7(iii). Do not spend effort
looking for a workaround; there is no endpoint, no import-time line, and no existing log.

**Safest window, given the constraints measured today (2026-08-07, Friday, a US trading
day; local time at analysis 20:38 CEST / 14:38 ET):**

- The only hard exclusion is the evening digest at **23:00:00 CEST (17:00:00 ET)**, which
  hits `/api/paper-trading/portfolio` and **P1-pages on connection-refused**
  (`scheduler.py:597`, `:636`). With a measured 2.455 s outage, **avoid 22:58–23:02 CEST**.
- **Recommended: restart at or before ~22:45 CEST tonight**, which leaves a ~15-minute
  margin and lets the 23:00 digest run against a warm backend. If GENERATE slips past that,
  the next clean window is **after ~23:05 CEST**, once "Evening digest sent"
  (`scheduler.py:632`) appears in the bot log — that turns the constraint into an
  *observable* rather than a guess.
- The watchdog is **not** a constraint (needs ~3 min of failure; does not page) — §1.7(ii).
- Also avoid restarting inside an autonomous-cycle window; a restart cannot double-fire
  `paper_trading_daily` (memory `project_backend_restart_safety`), but a mid-cycle restart
  muddies the DARK evidence for criterion 5.
- Command: `launchctl kickstart -k gui/$(id -u)/com.pyfinagent.backend`
  (**not** `pkill` — it races the KeepAlive watchdog; see memory
  `feedback_npm_install_requires_launchctl_kickstart` for the analogous frontend rule).
- Evidence capture for the live_check: the plist sends both stdout and stderr to
  `/Users/ford/.openclaw/workspace/pyfinagent/backend.log`, so
  `grep -n "execution_backend=" backend.log | tail -3` plus the surrounding
  `Started server process [<new pid>]` / `Application startup complete` lines proves the line
  came from the **new launchd process**. Include the new PID from `launchctl list` so the
  artifact is auditable — a log line alone does not prove which process wrote it.

**One more precondition worth stating plainly:** because the plist change and the code change
must both be live for the banner to read `source=env`, sequence it as — code merged →
plist edited → `launchctl unload/load` (or `kickstart -k`, which re-reads the plist only
after a reload; a bare `kickstart -k` does **not** pick up plist edits). Main should verify
which of the two it needs before claiming `source=env`, or the captured line will say
`source=default` and the criterion will look failed when the code is fine.

### 3.3 Test-file design — `backend/tests/test_execution_backend_wiring.py`

Name fixed by the immutable command. House idiom: module docstring naming the phase,
`monkeypatch` for env, `get_settings.cache_clear()` around settings reads, no network.

Suggested structure (7 tests, one per criterion leg — keeps the mapping auditable):

1. `test_default_is_bq_sim_with_no_env_and_no_dotenv` — criterion 2. `monkeypatch.delenv("EXECUTION_BACKEND", raising=False)`, clear settings cache, assert `_current_mode()` → `("bq_sim", "default")` and `ExecutionRouter().mode == "bq_sim"`.
2. `test_env_var_wins_over_settings` — criterion 1. `monkeypatch.setenv("EXECUTION_BACKEND","shadow")`, assert mode `shadow`, source `env`. Mirrors the documented pydantic precedence (§2.1) so the two channels agree.
3. `test_settings_value_used_when_env_absent` — criterion 1. Env deleted, settings field patched to `alpaca_paper`; assert mode + source `settings`. **This is the test that would have caught the original bug.**
4. `test_unknown_mode_falls_back_to_bq_sim` — criterion 4c. Parametrize `["alpaca_live","live","LIVE","","  ","garbage"]`; assert result ∈ `VALID_MODES` and `!= ` any live-ish string. **Parametrize list must be non-empty** — an empty parametrize yields 0 tests and a false green (memory `project_non_forward_labels_82_16`).
5. `test_missing_creds_in_alpaca_mode_logs_loudly` — criterion 3. `caplog` at ERROR; delete both cred keys; assert exactly one ERROR record and that it **names both** `ALPACA_API_KEY_ID` and `ALPACA_API_SECRET_KEY`. Mutation check: with creds present, assert **zero** ERROR records — otherwise the test cannot fail.
6. `test_paper_base_url_is_pinned` — criterion 4a. **Verified feasible offline in this session**: `alpaca-py 0.43.2`, `TradingClient("dummy_key","dummy_secret",paper=True)` constructs with **no network**, and `client._base_url` is `BaseURL.TRADING_PAPER` == `"https://paper-api.alpaca.markets"` (measured). Assert that, assert `BaseURL.TRADING_LIVE == "https://api.alpaca.markets"` is **not** what the client holds, and assert the repo source never passes `url_override` (the SDK's documented escape hatch, `TradingClient.__init__` signature).
7. `test_live_key_prefix_is_refused` — criterion 4b. `PKLIVEXXXX` key → `_refuse_live_keys()` raises `RuntimeError`; `ALPACA_PAPER_TRADE=false` → raises. Add an explicit comment that the prefix is **not an Alpaca-documented format** (§2.2) so the next reader does not mistake the test for evidence that it is.

Cross-cutting cautions, all from prior incidents in this repo:
- Every guard must be **mutation-tested**: change the production call site, not a helper, and
  confirm the test goes red (`feedback_mutation_test_guards_and_fixtures`,
  `feedback_guards_stop_one_seam_short`).
- **No `pytest.skip` trapdoors** — a skipped test reads as green.
- `get_settings` is `lru_cache`d; a forgotten `cache_clear()` makes tests order-dependent and
  can produce a passing test that proves nothing.
- Assert the *behaviour*, not the source text — a test that greps `execution_router.py` for a
  string is not a wiring test.

### 3.4 Out-of-scope defects discovered (queue as their own steps)

Per `feedback_queue_discovered_defects_in_masterplan`, these are **not** for 68.1:

1. **Plaintext + malformed `CLAUDE_CODE_OAUTH_TOKEN` in the backend plist** (§1.4). A secret
   in a plist outside the repo, apparently double-prefixed and containing a line break.
2. **Two disjoint Alpaca credential channels** — `os.environ` (router) vs `SecretStr`
   settings (news adapter, `settings.py:132-133`). Unification needs the SecretStr-unwrap
   care noted in §1.5.
3. **Stale module docstring** at `execution_router.py:3-4` ("selected at import time") and
   the over-claiming triple-enforcement text at `:11-13` (§2.2).
4. **`rollback_to_bq_sim()` has no caller** (`:325-329`) despite the docstring claiming the
   circuit breaker uses it — either dead code or a missing wire.
5. **`AlpacaBroker` bypasses `ExecutionRouter`** (`alpaca_broker.py:89-100`), so it is not
   covered by any `EXECUTION_BACKEND` guarantee.

---

## Research Gate Checklist

Hard blockers:
- [x] ≥5 authoritative external sources READ IN FULL via WebFetch — **8**
- [x] 10+ unique URLs total (incl. snippet-only) — **40+**
- [x] Recency scan (last 2 years) performed + reported — §2.4
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every relevant module (router, settings, main, paper_trader, alpaca_broker, plist, watchdog, slack scheduler, tests)
- [x] Contradictions / consensus noted — §2.2 (criterion premise vs vendor docs), §2.4 (2026 twelve-factor critique qualifying the canonical source)
- [x] Claims cited per-claim
- [ ] **Gap disclosed**: `backend/.env` is unreadable from this sandbox (a Bash call touching it was denied), so the presence/absence of an `EXECUTION_BACKEND=` line there is UNVERIFIED — Main must run the count. Also `scripts/launchd/backend_watchdog.sh` was read only through line ~55 (enough to establish the threshold and that it does not page).

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 8,
  "snippet_only_sources": 16,
  "urls_collected": 40,
  "recency_scan_performed": true,
  "internal_files_inspected": 12,
  "coverage": {
    "audit_class": false,
    "rounds": 1,
    "dry_rounds": 0,
    "K_required": 2,
    "new_findings_last_round": 0,
    "dry": false
  },
  "summary": "All three audit_basis clauses CONFIRMED, one worse than stated: Settings has NO execution_backend field at all (settings.py grep yields only a Celery-related 'Execution Mode' header at :222), so the settings leg must be created, not wired. Router reads os.getenv only (execution_router.py:65-71); ExecutionRouter.__init__ resolves per-construction at :269 (module docstring's 'import time' claim is stale). Plist carries exactly 4 env keys, no EXECUTION_BACKEND. Criterion 3's premise verified literally: _alpaca_mock_fill (:128-154) has ZERO logger calls. HEADLINE EXTERNAL FINDING: criterion 4b's 'PKLIVE-class' discriminator is unfounded -- three official Alpaca sources read in full document NO prefix/format difference between paper and live keys; separation is by DOMAIN (paper-api.alpaca.markets vs api.alpaca.markets). The real guard is paper=True, now proven offline-assertable (alpaca-py 0.43.2, client._base_url == BaseURL.TRADING_PAPER, no network at construction). Criterion 1 CANNOT be met without a restart (no endpoint, no startup line, no import-time resolution). Restart outage MEASURED at 2.455s; watchdog needs ~3min so it cannot trip; the only hazard is the 23:00 CEST digest, which P1-pages on connection-refused -- recommend restarting before 22:45 CEST or after the 'Evening digest sent' line.",
  "brief_path": "handoff/current/research_brief_68.1.md",
  "gate_passed": true
}
```
