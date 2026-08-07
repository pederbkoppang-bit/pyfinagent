# Research Brief — Step 85.2 — Credential-free CI lane (e2e-smoke.yml collection failure)

Tier: **moderate** (caller-specified). Write-first; grown incrementally. Date: 2026-08-07.
Worktree used for the secretless simulation: detached HEAD `55df9006`.

Question: `.github/workflows/e2e-smoke.yml` fails at pytest COLLECTION on every scheduled
run. Design a repair that makes the lane execute tests **without weakening production
validation**.

---

## 1. HEADLINE: the step premise is INCOMPLETE — there are TWO independent blockers

The step names one root cause (pydantic `ValidationError` on four required settings fields).
Measured against the **actual CI log of the newest run 31154911052 (2026-08-07T06:42:47Z)**,
that accounts for **4 of the 7** collection errors. The other 3 are a *different* failure with
a *different* fix, and **the settings repair alone leaves the lane RED (exit 2)**.

Verbatim from `gh run view 31154911052 --log-failed`:

```
ERROR backend/tests/api/test_sovereign.py - pydantic_core._pydantic_core.ValidationError: 4 validation errors for Settings
ERROR backend/tests/test_agent_definitions_classification.py - pydantic_core._pydantic_core.ValidationError: 4 validation errors for Settings
ERROR backend/tests/test_claude_request_shapes.py - pydantic_core._pydantic_core.ValidationError: 4 validation errors for Settings
ERROR backend/tests/test_phase_51_3_digest_guard.py
ERROR backend/tests/test_phase_75_2_1_push_approval.py
ERROR backend/tests/test_phase_75_2_slack_control_plane.py
ERROR backend/tests/test_phase_80_2_error_response_contract.py - pydantic_core._pydantic_core.ValidationError: 4 validation errors for Settings
!!!!!!!!!!!!!!!!!!! Interrupted: 7 errors during collection !!!!!!!!!!!!!!!!!!!!
16 deselected, 2 warnings, 7 errors in 8.82s
##[error]Process completed with exit code 2.
```

The three un-annotated ERRORs are:

```
ImportError while importing test module '.../backend/tests/test_phase_51_3_digest_guard.py'.
E   ModuleNotFoundError: No module named 'aiohttp'
...
    from slack_bolt.async_app import AsyncApp
```

### Blocker B — root cause, pinned to the CI install log

```
Collecting slack-bolt>=1.18.0 (from slack-bolt[async]>=1.18.0->-r backend/requirements.txt (line 55))
  Downloading slack_bolt-1.30.0-py2.py3-none-any.whl.metadata (11 kB)
WARNING: slack-bolt 1.30.0 does not provide the extra 'async'
```

`backend/requirements.txt:55` is `slack-bolt[async]>=1.18.0`. **slack-bolt 1.30.0 dropped the
`async` extra**, so the unpinned `>=1.18.0` resolves to 1.30.0 and `aiohttp` is silently NOT
installed (pip emits only a WARNING, exit 0). `backend/slack_bot/*` does
`from slack_bolt.async_app import AsyncApp`, which needs `aiohttp`.

**Why the operator's Mac cannot see this**: `.venv` has `slack_bolt 1.27.0` + `aiohttp 3.13.3`
already resolved. This is pure version drift in an unpinned transitive extra — a time bomb that
detonated when slack-bolt 1.30.0 shipped. No `.py` file in `backend/` or `scripts/` does
`import aiohttp` directly, and `aiohttp` is absent from `backend/requirements.txt`
(`grep -rn -i aiohttp backend/requirements.txt` → no match, exit 1).

**Implication for criterion 7** (a green scheduled run closes the step): the step MUST fix
both. Fixing only the four settings fields moves 7 errors → 3 errors, still `exit 2`, still red.
Adding `aiohttp` (or pinning `slack-bolt<1.30` / replacing the dead extra) is a genuine missing
dependency, NOT "modifying a test to skip its way to green", so it is inside the non-scope fence.

---

## 2. Measured evidence (secretless simulation, this machine)

Method: `git worktree add --detach <scratch>/ci-sim HEAD` → a fresh checkout with **no
`backend/.env`** (only the tracked `backend/.env.example`, 615 bytes). Ran the CI's exact pytest
invocation under `env -i` (empty environment; `PATH`/`HOME`/`LANG` only, `HOME=/tmp/nohome-ci`
so gcloud ADC is also invisible), using the repo `.venv` interpreter by absolute path.

| Condition | Result |
|---|---|
| **BEFORE** — 4 vars unset, no `.env`, `--collect-only` | `2840/2856 tests collected (16 deselected), 4 errors in 8.39s` → **Interrupted, 0 executed** |
| **AFTER-proxy** — same, 4 vars injected as placeholders | `2884/2900 tests collected (16 deselected) in 4.01s` → **0 errors** |

Notes on the delta the step will have to record:
- The step quotes **7 errors** from run 31080041519. Locally the figure is **4**, because the
  local `.venv` already has `aiohttp`. **Both numbers are real and they measure different
  environments** — the honest "before" figure for the handoff is the CI one (7 errors,
  0 executed, 16 deselected), with the local 4 disclosed as the settings-only subset.
- Collected count rises 2856 → 2900 after the fix because the 4 erroring modules contribute
  their own tests once importable.
- `PYFINAGENT_TEST_NO_BQ=1` is set automatically at `backend/tests/conftest.py:19` via
  `os.environ.setdefault`, so the BQ write-guard is already active in a bare checkout.

(Full-run pass/fail population: see §3.)

---

## 3. Full credential-free run — the measured test population (THE decisive number)

Same scrubbed worktree, four vars injected as placeholders, CI's exact selection
(`-m "not requires_live"`), full run to completion. **Verbatim tail:**

```
46 failed, 2817 passed, 13 skipped, 16 deselected, 4 xfailed, 40 warnings, 4 errors in 371.37s (0:06:11)
```

**So clearing collection does NOT make the lane green.** 46 failures + 4 errors across **23
files** would still exit non-zero. Criterion 7 cannot be met by the settings + aiohttp fixes
alone — this is the single biggest risk to closing step 85.2, and the contract must say so
up front rather than discovering it on the first dispatch.

Failure census (by file, from the run's own `FAILED`/`ERROR` lines):

| Count | File | Apparent class |
|---|---|---|
| 8 | `backend/tests/test_paper_trading_v2.py` | local/tracked state (see below) |
| 6 | `backend/tests/test_phase_82_54_cost_budget_columns.py` | live BQ |
| 5 | `backend/tests/test_phase_82_39_outcome_rebuild_query.py` | live BQ |
| 4 | `backend/tests/test_phase_75_17_verification_paths.py` | environment/path |
| 4 (errors) | `backend/tests/test_phase_76_9_2_max_bridge.py` | environment (no `claude` rail) |
| 3 | `backend/tests/test_price_tolerance_gate.py` | **ADC** (`DefaultCredentialsError`) |
| 3 | `backend/tests/test_64_3_currency_path.py` | ADC/BQ |
| 2 | `backend/tests/test_phase_82_48_outcome_write_schema.py` | live BQ |
| 2 | `backend/tests/test_phase_70_4_gate_observability.py` | — |
| 1 each | `test_phase_83_1_1_gate_feasibility`, `test_phase_80_2_error_response_contract`, `test_phase_76_9_2_max_bridge`, `test_phase_75_sre_ops`, `test_phase_75_prompt_contracts`, `test_phase_75_19_preflight_calibration`, `test_phase_70_3_atomic_swap`, `test_phase_40_2_claude_code_v2_1_140_features`, `test_phase_23_2_4_pause_resume_no_deadlock_live`, `test_phase_23_2_14_no_reentrant_locks`, `test_dod4_tier1_coverage_investment`, `test_book_safety_69`, `test_64_4_multi_market_e2e` | mixed |

Two confirmed root causes, quoted from the run:

- **ADC absence** (faithful to CI — GitHub runners have no gcloud credentials):
  `google.auth.exceptions.DefaultCredentialsError: Your default credentials were not found.`
  This fires inside `test_price_tolerance_pass_1pct_deviation`, i.e. in tests that have nothing
  to do with credentials but import a path that constructs a BQ client.
- **Tracked mutable state**: the `test_paper_trading_v2` failures come from
  `paper_trader.py:282` → `kill_switch: REFUSING BUY ... the kill switch is PAUSED
  (pause_reason='manual')`. `handoff/kill_switch_audit.jsonl` **is a tracked file**
  (`git ls-files`/`git status` both show it), so a fresh clone — and therefore CI — inherits
  whatever kill-switch state the operator last committed. **The CI lane's verdict is coupled to
  a committed operations artifact.** Flag this to the operator; it is its own defect and its own
  step (the step's non-scope fence forbids widening 85.2 to absorb it).

Fidelity caveats on this simulation (state them in the handoff; do not over-claim):
- macOS, not `ubuntu-latest`; the local `.venv` (`slack_bolt 1.27.0`, `aiohttp 3.13.3`,
  `pydantic 2.13.4`) is NOT CI's freshly-resolved set (CI got `slack-bolt 1.30.0`,
  `pydantic 2.13.4`, `pytest 9.0.3`). Some of the 46 may pass or fail differently there.
- The worktree carries the committed tree, so tracked-state effects reproduce, but untracked
  local artifacts do not.
- **The only way to get the true CI number is a dispatch** — which is what criterion 7 asks for.
  Treat 46/2817 as an order-of-magnitude planning figure, not a promise.

**Recommended contract framing**: 85.2 should close on "collection is repaired and the lane
executes tests" with the delta `0 executed / 7 collection errors` → `2,900 collected, N
executed`, and the residual red **enumerated and queued as its own step(s)**. If the contract
instead promises a fully green workflow, the step is at high risk of the phase-81.0 failure
mode recorded in auto-memory `feedback_immutable_criteria_must_be_green_able` — a criterion
that is structurally uncloseable because it is bound to ~46 pre-existing unrelated failures.
Criterion 7 as written says "its conclusion recorded ... A green local test with a still-red
workflow does not close this step", which reads as requiring green; the executor should raise
this with the operator BEFORE generating, since the criteria are immutable.

---

## 4. Internal code inventory

| File | Anchor | Role | Status |
|---|---|---|---|
| `backend/config/settings.py` | `:26` `gcp_project_id: str = Field(...)` | required, no default | **blocker A** |
| `backend/config/settings.py` | `:56` `rag_data_store_id: str = Field(...)` | required, no default | **blocker A** |
| `backend/config/settings.py` | `:118` `ingestion_agent_url: str = Field(...)` | required, no default | **blocker A** |
| `backend/config/settings.py` | `:119` `quant_agent_url: str = Field(...)` | required, no default | **blocker A** |
| `backend/config/settings.py` | `:13` `_ENV_FILE = .../backend/.env` | absolute path, derived from `__file__` | cwd-independent |
| `backend/config/settings.py` | `:622` `model_config = {"env_file": str(_ENV_FILE), "env_file_encoding": "utf-8", "extra": "ignore"}` | single env-file, **not a tuple** | change site for layering |
| `backend/config/settings.py` | `:626-628` `@lru_cache() get_settings()` | process-wide singleton | first call fixes the failure |
| `backend/.env.example` | `:6,:12,:15,:16` | tracked; carries all four keys with placeholder values | usable CI template |
| `.github/workflows/e2e-smoke.yml` | `:78-86` pytest step | sets only the 3 paper_* flags; **no GCP env** | change site (option) |
| `.github/workflows/e2e-smoke.yml` | `:22` `cron: '17 6 * * *'` | nightly 06:17 UTC | trigger for live_check |
| `.github/workflows/e2e-smoke.yml` | `:60-63` `pip install -r backend/requirements.txt` | where the aiohttp gap opens | change site (blocker B) |
| `backend/requirements.txt` | `:55` `slack-bolt[async]>=1.18.0` | dead extra on 1.30.0 | **blocker B** |
| `backend/tests/conftest.py` | `:19` `os.environ.setdefault("PYFINAGENT_TEST_NO_BQ","1")` | import-time env bootstrap | **existing test-env seam** |
| `backend/tests/conftest.py` | `:47-58` slack-egress guard | import-time, same idiom | precedent for a settings bootstrap |
| `backend/tests/test_regime_detector.py` | `:150-153` | `os.environ.setdefault` for **all four** vars | **the existing in-repo idiom** |
| `tests/verify_phase_25_*.py` | e.g. `:28-29`, `:94-95` | same `setdefault` idiom (2 of 4 vars) | precedent, partial |

### Real consumers of the four fields (what a dummy value does)

| Field | Consumers | Behaviour with a placeholder |
|---|---|---|
| `gcp_project_id` | `backend/db/bigquery_client.py:45` `bigquery.Client(project=...)`, `:46,:47,:423,:449,:464,:513,:526` table FQNs; `backend/main.py:157` startup log; `backend/agents/_genai_client.py:49`; `backend/agents/multi_agent_orchestrator.py:272`; `backend/backtest/data_ingestion.py:49`; `backend/backtest/quant_optimizer.py:626`; `backend/agents/mcp_servers/{data,backtest}_server.py`; `backend/news/bq_writer.py:97`; `backend/models/{chronos,timesfm}_client.py`; `backend/autoresearch/rotation_runner.py:132` | Loud 403/404 at the first BQ/Vertex call. **Never silent** — every use is a network boundary. |
| `rag_data_store_id` | `backend/agents/orchestrator.py:599-600` (Vertex AI Search datastore path only) | Loud 404 at the RAG call. Single consumer. |
| `ingestion_agent_url` | `backend/agents/orchestrator.py:1115` `client.stream("POST", ...)` | Loud connection error at call time. Single consumer. |
| `quant_agent_url` | `backend/agents/orchestrator.py:1132` `client.stream("GET", ...)` | Loud connection error at call time. Single consumer. |

**None of the four is read at import time** other than `main.py:157`'s log line. All four are
consumed at a network boundary, so a placeholder produces a loud runtime error, not a wrong
answer. This is the key fact that makes lazy/boundary validation safe here.

### The "fail-fast" premise is already partly false in this repo

`os.getenv("GCP_PROJECT_ID", "sunny-might-477607-p8")` — i.e. the **real production project id
hardcoded as a fallback** — already appears in at least 12 modules:
`backend/meta_evolution/directive_review.py:51`, `directive_rewriter.py:52`,
`alpha_velocity.py:36`, `backend/agents/rag_agent_runtime.py:84`,
`backend/agents/skill_modification_review.py:55`, `backend/metrics/sortino.py:104`,
`backend/autoresearch/slot_accounting.py:41,132`, `backend/api/harness_autoresearch.py:190`,
`backend/api/sovereign_api.py:33`, `backend/services/observability/spend.py:158,224`,
`backend/slack_bot/jobs/_production_fns.py:42`, `backend/slack_bot/jobs/weekly_data_integrity.py:82`;
plus `backend/autonomous_loop.py:605` and `backend/backtest/learning_schema.py:75` default to
`"pyfinagent-prod"`. So "a missing GCP_PROJECT_ID is caught at startup" is **already untrue for
those paths** — they silently use a hardcoded project. Any claim in the handoff that the current
design gives end-to-end fail-fast on this variable must be qualified; the strictness that exists
today lives ONLY in `Settings`.

---

## 5. CI history (re-derived 2026-08-07)

`gh run list --workflow=e2e-smoke.yml --limit 15 --json ...` → **15/15 most recent runs
`"conclusion":"failure"`, all `"event":"schedule"`**, from 2026-07-24T07:10:16Z (30074575835)
through 2026-08-07T06:42:47Z (31154911052). So the streak is at least 15, longer than the
"twelve" the step cites; the newest run id to supersede 31080041519 in the handoff is
**31154911052**.

`gh auth status` → token scopes `'gist', 'read:org', 'repo', 'workflow'`. **`workflow` scope is
present**, so `gh workflow run e2e-smoke.yml` (workflow_dispatch) is available for the
live_check — the step does not have to wait for the 06:17 UTC cron.

Note the `pull_request` trigger is effectively dead for this project: CLAUDE.md mandates direct
pushes to `main` with no PRs, so only `schedule` and `workflow_dispatch` ever fire.

---

## 5b. Prior design records (why the lane broke on a known date)

- **phase-53.5** created the lane (`handoff/archive/phase-53.5/*`, `live_check_53.5.md`) as an
  advisory soft-launch with `continue-on-error: true`.
- **phase-75.15** (2026-07-24) flipped it enforcing and swapped the `--ignore` list for
  `-m "not requires_live"`. Its own contract records the key fact verbatim:
  *"step-level continue-on-error:true forces conclusion=success (the lane structurally CANNOT
  block)"* (`handoff/archive/phase-75.15/contract.md`).
- **`gh run list --limit 60` → 43 success / 15 failure; LAST SUCCESS = run `29987399582`,
  2026-07-23T07:09:38Z (schedule); 15 consecutive failures since.** The lane went red on the
  FIRST scheduled run after 75.15 removed `continue-on-error`.
- 75.15's green was captured **locally**, at a "CI-equivalent env" that overrode only the three
  `PAPER_*` booleans — with the operator's real `backend/.env` still on disk. That local
  environment could not possibly exercise the four missing GCP fields. **The pre-75.15 "green"
  runs are not evidence the tests passed; they are evidence the step could not fail the job.**
  The executor must not treat "it used to be green" as a baseline.

---

## 6. External research — read in full (7; >=5 required)

| # | URL | Accessed | Kind | Fetched how | Key finding / quote |
|---|---|---|---|---|---|
| 1 | https://pydantic.dev/docs/validation/latest/concepts/pydantic_settings/ | 2026-08-07 | official doc | WebFetch (full, after 301 from docs.pydantic.dev) | Source precedence, descending: **CLI args > init args > environment variables > dotenv file > secrets dir > "The default field values for the Settings model."** Multi-file: *"If you need to load multiple dotenv files, you can pass multiple file paths as a tuple or list. The files will be loaded in order, with each file overriding the previous one."* And: *"environment variables will always take priority over values loaded from a dotenv file."* |
| 2 | https://12factor.net/config | 2026-08-07 | canonical (year-less) | WebFetch (full) | Config belongs in the environment; config files risk that you *"mistakenly check in a config file to the repo"*. Litmus test: *"whether the codebase could be made open source at any moment, without compromising any credentials."* Also warns that named environment groups ("development"/"staging"/"production") produce a *"combinatorial explosion of config which makes managing deploys of the app very brittle."* |
| 3 | https://pydantic.dev/docs/validation/latest/concepts/validators/ | 2026-08-07 | official doc | WebFetch (full, after 301) | `@model_validator(mode='after')` is an **instance method**, a *"post-initialization hook"*, must **return the validated instance**; a plain `ValueError` raised inside surfaces as a `ValidationError`. This is the documented seam for conditional/cross-field requirements. |
| 4 | https://docs.pytest.org/en/stable/how-to/monkeypatch.html | 2026-08-07 | official doc | WebFetch (full) | `monkeypatch.delenv(NAME, raising=False)` is the documented way to assert behaviour when a var is ABSENT; `monkeypatch.setenv` for present. All changes auto-revert per test. Documented example asserts an exception under `delenv` — exactly the shape criterion 2 needs. |
| 5 | https://github.com/google-github-actions/auth | 2026-08-07 | official vendor doc | WebFetch (full) | *"Workload Identity Federation is recommended over Service Account Keys as it obviates the need to export a long-lived credential."* Service-account JSON keys are *"long-lived credentials and must be treated like a password."* Confirms that a CI lane which needs GCP should use OIDC/WIF, not a checked-in or secret-stored key — and by extension that a lane which needs NO cloud access should carry NO credential at all. |
| 6 | https://pypi.org/project/slack-bolt/ | 2026-08-07 | official package metadata | WebFetch (full) | Latest **1.30.0, released 2026-07-15**. No `async` extra is declared; the install instruction is now literally *"aiohttp is required — `pip install slack_bolt aiohttp`"*. Direct external confirmation of blocker B and its date. |
| 7 | https://reflectoring.io/validate-spring-boot-configuration-parameters-at-startup/ | 2026-08-07 | named-author eng. blog | WebFetch (full) | *"For some configuration parameters it makes sense to fail application startup if they're invalid."* Motivating incident: a mistyped address silently produced no error until customers stopped receiving reports. **Notably absent** from the article: any treatment of how tests boot without production config — the exact gap 85.2 has to fill. |

### Snippet-only (context; does NOT count toward the gate)

| URL | Kind | Why not fetched in full |
|---|---|---|
| https://github.com/pydantic/pydantic/issues/6463 | issue | Optional-vs-required semantics; superseded by doc #1 |
| https://github.com/pydantic/pydantic/issues/5481 | issue | "Optional fields are required by default" — v1→v2 history |
| https://github.com/pydantic/pydantic/issues/8006 | issue | duplicate of the above class |
| https://github.com/pydantic/pydantic/issues/10252 | issue | "Field required when another field has a specific value" — the conditional-required idiom; doc #3 is authoritative |
| https://docs.pydantic.dev/latest/api/config/ | doc | `validate_default` reference only |
| https://pydantic.dev/docs/validation/2.9/concepts/pydantic_settings/ | doc | older pinned version of #1 |
| https://en.wikipedia.org/wiki/Fail-fast_system | encyclopedia | community tier |
| https://dev.to/137foundry/how-to-validate-environment-variables-at-application-startup-5bb | blog | community tier; same argument as #7 |
| https://medium.com/@yegor-sychev/proactive-configuration-validation-in-net-... | blog | .NET-specific |
| https://startdebugging.net/2026/08/how-to-validate-options-at-startup-with-ivalidateoptions-in-dotnet-11/ | blog | 2026 .NET; cross-domain corroboration only |
| https://github.com/dotnet/AspNetCore.Docs/issues/16732 | issue | "Eager Validation (fail fast at startup) **with testing**" — the same tension, other ecosystem |
| https://github.com/pjlast/pytest-bigquery-mock | repo | BQ mocking; not needed (repo already has `PYFINAGENT_TEST_NO_BQ`) |
| https://pypi.org/project/pytest-bq/ | pkg | local BQ emulator; out of scope |
| https://github.com/ottogroup/bquest | repo | BQ query testing lib |
| https://github.com/googleapis/python-bigquery-sqlalchemy/issues/430 | issue | "unit tests require google application credentials" — same failure class, different project |
| https://hoop.dev/blog/the-simplest-way-to-make-bigquery-github-actions-work-like-it-should | blog | vendor marketing |
| https://aembit.io/blog/secretless-access-for-github-actions/ | blog | secretless CI / OIDC; tj-actions 2025 incident |
| https://blog.gitguardian.com/handle-secrets-in-ci-cd-pipelines/ | blog | secrets hygiene |
| https://infisical.com/blog/secrets-management-cicd | blog | vendor |
| https://oneuptime.com/blog/post/2026-01-24-secrets-cicd-pipelines/view | blog | 2026 secrets-in-CI overview |
| https://realpython.com/python-pydantic/ | tutorial | intro tier |
| https://tech-insider.org/pytest-tutorial-python-testing-ci-cd-2026/ | tutorial | intro tier |
| https://medium.com/google-cloud/your-agent-events-table-is-also-a-test-suite-999fbef885ed | blog | Apr 2026 BQ-in-CI, orthogonal topic |

**URLs collected: 30 unique** (7 read in full + 23 snippet-only).

### Search-query variants run (3-variant discipline)

- **Year-less canonical**: `pydantic-settings required field optional in tests validation profile pattern`; `fail fast configuration validation at startup versus lazy validation trade-offs`
- **Current-year (2026)**: `GitHub Actions CI test suite without credentials Google Cloud BigQuery pytest 2026`; `pydantic-settings 2026 conditional required field model_validator environment profile production`
- **Last-2-year window (2025)**: `secretless CI pipeline 2025 dummy placeholder environment variables tests import time configuration`

---

## 7. Recency scan (2024-2026)

Searched the 2025-2026 window explicitly (two of the five queries above were year-scoped).
**Result: 2 findings that COMPLEMENT, none that supersede, the canonical sources.**

1. **Secretless CI is now the mainstream position, and the 2025 `tj-actions/changed-files`
   supply-chain compromise (Mar 2025, ~23,000 repos' secrets exposed) is the reason.** The
   consequence for 85.2 is directive: do **not** solve this by adding a GitHub Actions secret
   holding a real project id or agent URL — the step's non-scope already says so, and the
   2025-2026 literature agrees the correct move for a lane that needs no cloud access is to
   carry no credential at all (google-github-actions/auth #5; aembit/gitguardian snippets).
2. **"Mock/placeholder values in the test stage, real secrets only in the deploy stage" is the
   named 2025 practice** (secrets-management surveys above). That is precisely the
   "placeholder default" mechanism — it is not a hack, it is the documented split.
3. Nothing in the 2024-2026 window changes pydantic-settings' source-precedence rule or
   introduces a first-class "profile" feature; `@model_validator(mode='after')` remains the
   documented seam for conditional requirements (pydantic issue #10252, 2024→ still open-idiom).

---

## 8. Key findings

1. **A default is the LOWEST-precedence source, so adding one cannot change production
   behaviour when `backend/.env` supplies the value.** Precedence is init args > env vars >
   dotenv > secrets dir > *"The default field values for the Settings model"*
   (https://pydantic.dev/docs/validation/latest/concepts/pydantic_settings/, 2026-08-07). This
   is the single most load-bearing fact for the do-no-harm constraint: with a real
   `backend/.env` present that defines all four keys, a default is **never consulted**, so
   `Settings()` is byte-identical. The executor can assert this directly.
2. **Env-file layering (`env_file=(_ENV_EXAMPLE, _ENV_FILE)`) works but is the WRONG mechanism
   here.** It is documented and would fix collection — *"The files will be loaded in order,
   with each file overriding the previous one"* (ibid.) — but it silently backfills a key that
   the real `.env` FORGOT, which is exactly the masking failure mode the step forbids. It also
   makes the repo's tracked example semantically load-bearing at runtime, contra 12factor's
   warning about checked-in config files (https://12factor.net/config).
3. **Strictness must be relocated, not deleted.** Criterion 2 needs a surviving loud failure.
   `@model_validator(mode='after')` is the documented seam: an instance-method post-init hook
   that must return `self` and whose `ValueError` surfaces as a `ValidationError`
   (https://pydantic.dev/docs/validation/latest/concepts/validators/). Gate it on an explicit
   marker so CI/import is permissive and the running application is strict.
4. **The four fields are safe to default because none is read at import time and every consumer
   is a network boundary** (see §4 table) — a placeholder yields a loud 403/404/connection
   error, never a wrong number. This is the empirical fact that makes boundary validation
   acceptable here, and it does NOT generalise to a field that silently changes arithmetic.
5. **Fail-fast-at-startup is the right default, and the literature does not resolve the test
   tension.** Reflectoring makes the case (*"it makes sense to fail application startup if
   they're invalid"*) but is silent on booting tests without prod config; the .NET ecosystem
   files the same tension as an open docs issue ("Eager Validation (fail fast at startup) with
   testing", dotnet/AspNetCore.Docs#16732). So there is **no canonical off-the-shelf answer** —
   which is why the repair must be justified from this repo's own consumer map, not by citing
   a pattern name.
6. **12factor argues AGAINST named profiles** (*"combinatorial explosion of config"*). Read
   strictly this is an argument against a `PYFINAGENT_ENV=ci|prod` profile enum. A single
   boolean "is this a real deployment" marker is a much smaller commitment than a profile
   namespace, and should be preferred.
7. **Blocker B is externally confirmed and dated**: slack-bolt 1.30.0 (2026-07-15) no longer
   ships the `async` extra; the vendor's own install line is now `pip install slack_bolt
   aiohttp` (https://pypi.org/project/slack-bolt/, 2026-08-07).

---

## 9. Per-criterion mechanism map (which criterion permits which repair)

| # | Criterion (compressed) | Mechanisms it PERMITS | Mechanisms it RULES OUT |
|---|---|---|---|
| 1 | Test constructs `Settings` with the 4 vars unset and `backend/.env` invisible, and it **succeeds**; test must not require `.env` to exist | Only a **`settings.py`-level** repair: defaults, env-file layering, or a `mode='before'` backfill | **Workflow-only env provisioning** (`cp backend/.env.example backend/.env`, or `env:` block in the YAML) — in-process construction would still raise. Also rules out the test creating `.env` itself (would clobber the operator's real file — forbidden) |
| 2 | Production strictness preserved; a genuinely-required missing value still fails loudly; **name the field**; "a repair that makes every field optional fails this criterion" | A **relocated strict check**: `@model_validator(mode='after')` (or an explicit `validate_production_config()` called from `main.py` lifespan) that raises naming e.g. `gcp_project_id` | Bare defaults with **no** replacement guard; env-file layering (backfills a forgotten prod key from the tracked example → masks it) |
| 3 | Repair named in the handoff with file + line + why it doesn't weaken prod validation | documentation duty | — |
| 4 | Verification command bounded to `backend/tests/test_phase_85_2_credential_free_lane.py`; no repo-wide checker | The command in the masterplan already satisfies this; do not widen it | Any `-m "not requires_live"` full-suite invocation as the verification command |
| 5 | Mutation: reverting the repair must make the command exit non-zero | Both halves must be mutation-covered: (a) restore `Field(...)`, (b) remove the prod guard | A test that only asserts construction succeeds (mutation (b) would survive) |
| 6 | Measured DELTA: collected-in-secretless before vs after, after captured verbatim from a real run | Requires the secretless simulation; see §2 for the harness | Asserting "it works now" without the two numbers |
| 7 | A real workflow run triggered; conclusion recorded with run id + URL; a green local test with a red workflow does NOT close it | **Requires blocker B fixed too** — otherwise 3 `aiohttp` collection errors keep exit code 2 | Closing on the local test alone |

**The decisive constraint is criterion 1 + criterion 2 read together**: 1 forbids a
workflow-only fix, 2 forbids an unguarded default. Only "defaults **plus** a relocated,
explicitly-gated strict check" satisfies both.

---

## 10. Recommended design (with literature basis)

**Mechanism: CI-safe placeholder defaults on the four fields + a gated production-strictness
validator, and an explicit `aiohttp` dependency.**

1. **`backend/config/settings.py:26,56,118,119`** — replace `Field(...)` with
   `Field("", description=...)` (empty-string default; keep the descriptions).
   *Basis*: defaults are the lowest-precedence source, so with a real `backend/.env` present
   nothing changes (pydantic-settings precedence list, source #1). Prefer `""` over a fake
   value like `"ci-placeholder-project"` so the guard in (2) has an unambiguous "unset"
   sentinel and no plausible-looking string can reach a BQ client.
2. **New `@model_validator(mode='after')` on `Settings`** (or a module-level
   `validate_production_config(settings)` invoked from `backend/main.py`'s startup path near
   the existing `:157` log line) that raises a `ValueError` naming each empty required field —
   **but only when an explicit deployment marker is present**. The cleanest marker given this
   repo: treat the presence of `backend/.env` (i.e. `_ENV_FILE.exists()`) as "this is a real
   deployment" and require the four to be non-empty then; a bare checkout has no `.env` and
   stays permissive. Optional belt-and-braces: skip when `PYFINAGENT_TEST_NO_BQ` is set (the
   existing seam at `backend/tests/conftest.py:19`).
   *Basis*: `mode='after'` is the documented post-init hook and `ValueError` → `ValidationError`
   (pydantic validators doc); relocating rather than deleting the check is what criterion 2
   demands; a single boolean marker rather than a `PYFINAGENT_ENV` enum respects 12factor's
   "combinatorial explosion" warning.
   *Why this specific marker is strong*: it makes the operator's Mac and the launchd services
   **strictly stricter than today is claimed to be** — today a typo'd `GCP_PROJECT_ID=` (empty)
   in `.env` PASSES `Field(...)` (empty string is a valid `str`); under the guard it fails.
   The executor should state this in the handoff: the repair **increases** production
   strictness on the empty-value case while removing the absent-file failure.
3. **`backend/requirements.txt:55`** — the `[async]` extra is dead on slack-bolt 1.30.0. Add an
   explicit `aiohttp>=3.9` line (vendor's own instruction) and drop or keep the now-inert
   extra. *Do not* pin `slack-bolt<1.30` as the primary fix — that freezes a security-relevant
   dep to dodge a one-line dependency declaration.
4. **`.github/workflows/e2e-smoke.yml`** — no env changes strictly required once (1) lands.
   Optional hardening: add a step asserting `python -c "import aiohttp, backend.config.settings"`
   right after install so a future extra-drift fails at a named step instead of inside pytest.

**Explicitly rejected alternatives** (record these in the contract so the choice is auditable):
- *Workflow copies `.env.example` → `.env`*: fails criterion 1, and makes CI depend on a file
  12factor warns about.
- *`env_file=(_ENV_EXAMPLE, _ENV_FILE)` layering*: works, documented, but backfills a key a real
  `.env` forgot → the exact masking failure the step forbids.
- *`PYFINAGENT_ENV=ci` profile enum*: more surface than needed; 12factor's combinatorial-explosion
  objection applies; and a fresh clone (a stated motivation) would not set it.
- *A GitHub Actions secret holding the real project id / agent URLs*: excluded by the step's
  non-scope and by the 2025-2026 secretless-CI consensus.

---

## 11. Fixture / mutation strategy

**Test file**: `backend/tests/test_phase_85_2_credential_free_lane.py`. The hard problem is that
`backend/.env` **exists on the operator's machine** and `_ENV_FILE` is an absolute path baked at
import (`settings.py:13`), so `monkeypatch.delenv` alone does not reproduce CI. Two safe ways to
hide it — never delete or rename the real file:

- **In-process (preferred for criterion 1)**: construct with the env-file explicitly disabled and
  the four vars removed:
  `Settings(_env_file=None)` after `monkeypatch.delenv("GCP_PROJECT_ID", raising=False)` (x4).
  `_env_file` as an init kwarg *"will override the value (if any) set on the `model_config`
  class"* (pydantic-settings doc). `raising=False` is the documented absent-var idiom
  (pytest monkeypatch doc). This runs identically on the Mac and on a bare runner.
  Note `get_settings()` is `@lru_cache`'d (`settings.py:626`) — the test must instantiate
  `Settings` directly or call `get_settings.cache_clear()`, or it will assert against a cached
  object built from the operator's `.env`.
- **Subprocess (belt-and-braces, optional)**: `subprocess.run([sys.executable, "-c", ...],
  env={minimal})` with `cwd` set to a `tmp_path` — proves the import-time path, not just the
  constructor. Keep it bounded; the in-process test is the one the criterion names.

**Required assertions**
- C1: `Settings(_env_file=None)` with all four deleted → **constructs**, and
  `s.gcp_project_id == ""`.
- C2: the production guard, invoked explicitly with the deployment marker forced on, **raises**
  and the message **contains `gcp_project_id`** (name the field, as the criterion demands).
  Assert on the field name, not just "an exception".
- Byte-identity: build a `Settings` with all four supplied via `monkeypatch.setenv` and assert
  the four attributes equal the supplied values — proves the default never wins when a value
  exists (this is the do-no-harm assertion, and it is cheap).
- Anti-vacuity: assert the guard **does not** raise when the four are populated — otherwise the
  guard test could pass against a guard that always raises.

**Mutation matrix (criterion 5 — run once, record verbatim exit code + tail, then restore)**
| # | Mutation | Must produce |
|---|---|---|
| M1 | Restore `gcp_project_id: str = Field(...)` at `settings.py:26` | command exits non-zero (C1 test errors at construction) |
| M2 | Delete the `@model_validator` / `validate_production_config` body (make it a no-op `return self`) | command exits non-zero (C2 test fails — **this is the mutation that catches "made everything optional"**) |
| M3 | Make the guard raise unconditionally (ignore the marker) | anti-vacuity test fails |
| M4 | Remove `aiohttp` from `backend/requirements.txt` | does NOT redden this step's bounded command (correctly — criterion 4 bounds it); cover B by the criterion-7 workflow run instead, and say so explicitly rather than pretending the local command covers it |

M2 is the one to run first — it is the guard most likely to be defended rather than tested
(auto-memory `feedback_mutation_test_guards_and_fixtures`).

**Do NOT**: touch `backend/.env`; add `PYFINAGENT_*` skips to existing tests; mark anything
`requires_live` to dodge a red.

---

## 12. live_check capture plan (criterion 7)

- `gh auth status` shows scopes `gist, read:org, repo, workflow` → **`workflow` scope is
  present**, so `gh workflow run e2e-smoke.yml --ref main` (workflow_dispatch) works; no need to
  wait for the 06:17 UTC cron. Capture with
  `gh run list --workflow=e2e-smoke.yml --limit 1 --json databaseId,conclusion,url`.
- The `pull_request` trigger never fires on this project (direct-to-main policy), so
  dispatch or cron are the only two paths.
- **Record in `handoff/current/live_check_85.2.md`**: the run id, the full URL
  (`https://github.com/<owner>/pyfinagent/actions/runs/<id>`), the conclusion, and the verbatim
  pytest tail from `gh run view <id> --log` showing `N passed` (the "after" half of criterion 6).
  The "before" half is run **31154911052** — `7 errors`, `16 deselected`, `exit code 2`.
- **Expect at least one dispatch to still fail** and budget for it: the settings + aiohttp fixes
  clear *collection*, after which ~2,900 tests actually execute in CI **for the first time**
  (they have never run to completion there — every prior "success" was `continue-on-error`).
  §3 measures how many fail locally in a secretless env; each one is a decision the executor
  must make honestly (fix, or `requires_live`-mark only where the ROOT CAUSE is genuinely live
  state — 75.15's own classification discipline), never a blanket skip.
- Non-pytest legs (`npx tsc`, `npm run test`, `npm run build`, harness dry-run, intel/phase-6
  e2e) have not executed since 2026-07-23 either; a dispatch may surface a failure in those too.
  Any such failure is a **separate step** per the step's own non-scope fence.


---

## 13. Consensus vs debate (external)

**Consensus**: config belongs in the environment (12factor); secrets never in the repo or in CI
for lanes that don't need them (google-github-actions/auth + 2025-2026 secretless-CI corpus);
fail-fast beats lazy diagnosis (reflectoring, dotnet).

**Genuine debate / unresolved**: how a fail-fast-at-startup application stays *importable* by its
own test suite. Reflectoring is silent on it; the .NET ecosystem has it as an open docs issue
(dotnet/AspNetCore.Docs#16732 "Eager Validation (fail fast at startup) **with testing**");
pydantic-settings offers no first-class "profile" feature. 12factor actively argues **against**
named environment profiles. There is therefore **no citable canonical answer** — the design must
be justified from this repo's own consumer map (§4), which is why §4's "every consumer is a
network boundary" table is the load-bearing evidence, not a pattern name.

## 14. Pitfalls (from literature + this repo)

1. **Env-file layering masks a forgotten prod key.** Documented and tempting; forbidden by the
   step's own do-no-harm rule.
2. **`@lru_cache` on `get_settings()` (`settings.py:626`)** — a test that calls `get_settings()`
   gets a cached object built from the operator's `.env`. Instantiate `Settings` directly or
   `cache_clear()`, or the test is a false green on this machine and a false red nowhere.
3. **An unpinned extra can vanish silently.** `slack-bolt[async]` produced only
   `WARNING: ... does not provide the extra 'async'` and `pip` exited 0. Any future
   `pkg[extra]` in `requirements.txt` carries the same risk; a post-install import assertion in
   the workflow is the cheap general guard.
4. **"It used to be green" is not a baseline** when the step was `continue-on-error: true`.
5. **A default of `""` is still a valid `str`** — so today's `Field(...)` already accepts an
   EMPTY `GCP_PROJECT_ID=` in a real `.env`. The proposed guard closes that hole; say so.
6. **Do not assert a count you did not measure** (auto-memory
   `feedback_measure_dont_assert_claims`): the "7 collection errors" and the "46 failures" are
   from two different environments and must be labelled as such.

## 15. Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch — **7**
- [x] 10+ unique URLs total (incl. snippet-only) — **30**
- [x] Recency scan (last 2 years) performed + reported — §7
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim — §4, §5b

Soft checks:
- [x] Internal exploration covered every relevant module (settings.py, e2e-smoke.yml,
      conftest.py, requirements.txt, consumers, prior handoff records)
- [x] Contradictions / consensus noted — §13 (12factor vs profiles; fail-fast vs testability)
- [x] All claims cited per-claim
- [ ] **Brief length exceeds the `moderate` <=700-word guidance.** Deliberate and disclosed: the
      caller required a per-criterion mechanism map, a measured test population, a
      fixture/mutation strategy and a live_check plan in one artifact. Depth, not scope creep.

## 16. JSON envelope

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 7,
  "snippet_only_sources": 23,
  "urls_collected": 30,
  "recency_scan_performed": true,
  "internal_files_inspected": 16,
  "coverage": {
    "audit_class": false,
    "rounds": 1,
    "dry_rounds": 0,
    "K_required": 2,
    "new_findings_last_round": 0,
    "dry": false
  },
  "summary": "Step premise INCOMPLETE: the CI lane has TWO independent blockers. Only 4 of the 7 collection errors are the pydantic ValidationError; the other 3 are ModuleNotFoundError: aiohttp, because slack-bolt 1.30.0 (2026-07-15) dropped the [async] extra that backend/requirements.txt:55 relies on. Fixing settings alone leaves exit code 2. Criteria 1+2 read together force one mechanism: CI-safe defaults on the four fields PLUS a relocated, marker-gated production-strictness validator (defaults are pydantic's LOWEST-precedence source, so prod with a real .env is byte-identical). Measured secretless run: before = 0 executed / 7 collection errors; after-proxy = 2884/2900 collected, 0 collection errors, but 46 failed / 2817 passed / 4 errors across 23 files -- so criterion 7 (green workflow) is NOT reachable from these two fixes. Last green run was 29987399582 (2026-07-23), the day before 75.15 removed continue-on-error; 15 consecutive failures since.",
  "brief_path": "handoff/current/research_brief_85.2.md",
  "gate_passed": true
}
```
