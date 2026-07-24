# Research Brief — Step 75.16

**Tier:** moderate · **Executor:** sonnet-4.6 / high · **BOUNDARY:** repo-only edits, NO gcloud/docker executed, $0 metered
**Step:** Audit75 S16 — Cloud Functions + Docker deploy-surface retirement/hardening + script bootstrap repair
**Gate:** PASSED (7 sources read in full, recency scan done, all internal claims file:line-anchored)

---

## THE LOAD-BEARING FINDING (live-path question)

The internal-audit note warned "8 of 8 prior gates found stale anchors." The single stale anchor here is the
**most important one**: `orchestrator.py:1061 quant_agent_url` is WRONG. Line 1061 today is inside
`_run_enrichment_batch`. The real live path is:

- `backend/config/settings.py:119` — `quant_agent_url: str = Field(..., ...)` → **REQUIRED** field (no default); the
  backend cannot boot without `QUANT_AGENT_URL` set.
- `backend/agents/orchestrator.py:1126` — `async def run_quant_agent(self, ticker)` ; **:1132** —
  `client.stream("GET", f"{self.settings.quant_agent_url}?ticker={ticker}")` ; **:1792** —
  `report["quant"] = await self.run_quant_agent(ticker)` (called in the analysis pipeline).
- **Measured operator env** (`backend/.env`): `QUANT_AGENT_URL=https://us-central1-sunny-might-477607-p8.cloudfunctions.net/...`
  → a real deployed Cloud Function, NOT localhost. **functions/quant/main.py is LIVE.**

**How orchestrator parses the quant stream (orchestrator.py:1134-1144) — this constrains leg (d):**
```python
async for line in r.aiter_lines():
    if line.startswith("FINAL_JSON:"):   final_json = json.loads(line.split("FINAL_JSON:",1)[1])
    elif line.startswith("ERROR:"):      raise RuntimeError(line)
    else:                                logger.info(f"Quant: {line}")
if final_json is None: raise RuntimeError("Quant Agent did not return final JSON data.")
```
=> **The sanitized error MUST keep the `ERROR:` line-prefix AND be a single line (no embedded `\n`).** Today the
multi-line traceback is split by `aiter_lines()`; only the first `ERROR: …` line reaches the `elif`, the rest are
logged as `Quant:` noise — so removing the traceback from the yield does NOT change what the orchestrator sees, it
only stops leaking the traceback to unauthenticated HTTP callers. Do NOT rename the `FINAL_JSON:`/`ERROR:` tokens.

**Are the other functions live? NO — both orphaned:**
- `functions/ingestion/main.py` (entry `ingest_market_data_el`, market-data E-L → BigQuery, returns a plain
  string) does NOT match the orchestrator's ingestion contract. `run_ingestion_agent` (orchestrator.py:1115) POSTs
  to `ingestion_agent_url` and expects a STREAM with `STREAM_COMPLETE`/`ERROR:` lines and per-ticker SEC filings.
  The deployed `INGESTION_AGENT_URL` function is a DIFFERENT service (its source was the now-deleted top-level
  `ingestion_agent/` dir — the `deploy_agents.sh:176` `--entry-point=ingestion_agent_http` matches the stale
  `cloudbuild.yaml`, NOT `functions/ingestion/`). **functions/ingestion/ is a never-deployed, half-built refactor.**
- `functions/earnings/main.py` is superseded by the in-backend service — the LIVE earnings path is
  `earnings_tone.get_earnings_tone(ticker, api_ninjas_key, bucket_name)` (orchestrator.py:1249, signals.py:91/151).
  **functions/earnings/ Cloud Function is orphaned.**

Consequence: leg (d) touches LIVE code (contract-preserving care required); legs (b)(c)(e) touch orphaned code
(still hardened per the immutable criteria, but zero production-contract risk).

---

## Internal code inventory (re-anchored file:line)

| File | Anchor | Current state (verbatim) | Status |
|------|--------|--------------------------|--------|
| scripts/deploy/deploy_agents.sh | :1 (`#!/bin/bash`, **no `set -e`**), cd :142 `quant-agent`, :166 `ingestion_agent`, :191 `pyfinagent-app/risk-management-agent`, :216 `earnings-ingestion-agent` | 4 `cd` targets **all MISSING** (verified). `--source=.` + `--allow-unauthenticated` on all 4 deploys (:156,:179,:205,:229) | DELETE (leg a) |
| functions/ingestion/cloudbuild.yaml | :9 `--runtime=python310`, :11 `--entry-point=ingestion_agent_http`, :13 `--allow-unauthenticated`, :14 `BUCKET_NAME=your-gcs-bucket-name,USER_AGENT_EMAIL=your-email@example.com` | entry-point does not exist (real fn is `ingest_market_data_el`); placeholder envs; runtime 310 (masterplan wants 311) | DELETE + note (leg b) |
| functions/ingestion/main.py | :25 `ingest_market_data_el`; :59 `fetch_raw_market_data(...)` (unwrapped); :72-77 BQ `except → status="Failure"`; :79 `return f"...Status: {status}..."` (implicit 200) | Returns 200 on BQ-load fail AND on fetch fail (fetcher swallows) | HARDEN → 500 (leg c) |
| functions/ingestion/utils/data_fetchers.py | broad `except Exception: logging.error(...); return pd.DataFrame()` (~:108-110) | **Swallows fetch errors → empty df**, so main.py can never see a fetch exception | must un-swallow (leg c mechanism) |
| functions/quant/main.py | SEC `requests.get` at **:64** (CIK map) + **:140** (companyfacts) — **no `timeout=`**; :252 `error_message = f"...{traceback.format_exc()}"`; **:255** `yield f"ERROR: {error_message}"` (streams full TB); :253 `logging.critical(...)` already keeps TB in Cloud Logging | LIVE fn; leaks TB to unauth callers | leg d |
| functions/earnings/main.py | **:139** `GenerativeModel("gemini-1.5-flash-001")`; :151-154 `except Exception → transcript_item['nlp_analysis']={"error":...}`; :168 `return (jsonify(data),200,headers)`; wildcard CORS :69 + :78; also :120 `requests.get(...)` **untimed** | orphaned; retired model; failure-as-200-data; `*` CORS | leg e |
| functions/{earnings,ingestion,quant}/requirements.txt | earnings: `requests>=2.25.0`,`functions-framework>=3.0.0` (**missing `vertexai`,`google-cloud-storage` that main.py imports**); ingestion/quant fully unpinned | all UNPINNED (`>=` or bare) | leg f |
| .github/workflows/pip-audit.yml | paths + `--requirement` cover only `backend/requirements.txt` + `.lock` + root `requirements.txt` | functions/*/requirements.txt NOT covered | leg f |
| backend/Dockerfile | :1 `FROM python:3.11-slim`; :5 `COPY requirements.txt .` (context-root pointer `-r backend/requirements.txt` → nested target absent in layer → build fails); :8 `COPY . .` | wrong python + broken COPY | leg g |
| frontend/Dockerfile | deps stage `COPY package.json ./` + `RUN npm install` (ignores committed `package-lock.json`, which EXISTS, 507 KB) | non-reproducible install | leg g |
| scripts/migrations/*.py (5) | migrate_bq_schema.py:18, migrate_agent_memories.py:17, migrate_signals_log.py:21, migrate_backtest_data.py:21, migrate_paper_trading.py:16 — `load_dotenv(Path(__file__).parent / "backend" / ".env")` | resolves to `scripts/migrations/backend/.env` (**nonexistent** → silent ADC fallback) | leg h |
| scripts/migrations/extend_historical_data.py | :25 `sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))` (inserts `scripts/migrations`, so `import backend.*` FAILS); :27 `load_dotenv("backend/.env")` (CWD-relative) | wrong sys.path + CWD .env | leg h |
| scripts/debug/*.py (4) | debug_ingestion.py:11, debug_sequential_updates.py:11, debug_db_update.py:11, debug_processor.py:12 — `backend_path = Path(__file__).parent / "backend"` | wrong path; **unreferenced by live code** (only an archived brief mentions debug_ingestion) | DELETE (leg h) |

Note: `scripts/migrations/add_round_trip_schema.py:24` already uses `.parents[2] / "backend"/".env"` (CORRECT — not one
of the broken five, and its `.parents[2]` does NOT match the forbidden `.parent / "backend"` substring).

---

## Decided defaults (the executor needs one path per leg)

- **(a) DELETE** `scripts/deploy/deploy_agents.sh`. Mechanics of the catastrophe: no `set -e` + 4 missing `cd`
  targets → CWD never changes → `git diff --quiet HEAD -- .` sees repo changes → `gcloud functions deploy … --source=.
  --allow-unauthenticated` uploads the **CWD tree** (repo root incl. `backend/.env`) as 4 public functions. Git
  history preserves the file.
- **(b) DELETE** `functions/ingestion/cloudbuild.yaml` **with a retirement note** (criteria explicitly allow deletion;
  verification accepts `ing==''`). Evidence for retirement over fix: orphaned E-L refactor, 1 commit only (April 13
  2026 restructure `fe5acdea`, untouched since), zero callers/schedulers, contract-mismatch with the live
  orchestrator, placeholder envs, half-built (`get_historical_universe` is a "CRITICAL TODO" placeholder). Put the
  note in the commit body + `experiment_results.md` (or a one-line `functions/ingestion/RETIRED.md`). **The live
  `INGESTION_AGENT_URL` is a separate already-deployed function — deleting this config does not touch it.**
- **(c) HARDEN** `functions/ingestion/main.py` regardless of (b) — criterion #2 requires it. Return explicit
  `(body, status)` tuples: `500` on fetch exception, `500` on BQ-load failure, `200` on empty-but-successful (no
  data), `200` on success. **Requires also editing `data_fetchers.py`** to stop swallowing (re-raise on genuine
  error; return empty df only for genuine no-data) — otherwise main.py can never observe a fetch exception. Refactor
  the status decision into a **pure helper** (`_response_for(df, load_ok)` or similar) so it is unit-testable without
  importing `functions_framework`/`google-cloud` (see BOUNDARY below).
- **(d)** In `functions/quant/main.py`: add `timeout=(5, 30)` to **both** `requests.get` (:64, :140). Split the
  traceback out: `tb = traceback.format_exc()` → `logging.critical(f"QuantAgent failed for {ticker_str}: {e}\n{tb}",
  exc_info=True)` (TB stays in Cloud Logging) → `yield f"ERROR: QuantAgent failed for {ticker_str}: {str(e)}"`
  (single line, `ERROR:` prefix, **no** `format_exc`). Preserve `FINAL_JSON:`/`ERROR:` tokens.
- **(e)** In `functions/earnings/main.py`: model id from env (e.g. `os.getenv("EARNINGS_NLP_MODEL", "<current
  non-retired gemini>")` — pick the model the live `earnings_tone` service uses; **note** `gemini-2.5-flash` retires
  2026-10-16, so choose a currently-supported id); make NLP failure distinguishable (non-200 status or an explicit
  `status`/`nlp_status` field, NOT `{'error':…}` stored as `data`); validate the parsed JSON against the 4 keys
  (`forward_sentiment_score`, `qa_confidence_summary`, `cyclical_catalysts_detected`, `key_quotes`); replace `*` CORS
  (:69, :78) with the localhost/Tailscale origin allowlist idiom used by the backend (`.claude/rules/security.md`
  "CORS").
- **(f)** Pin every non-comment line in all 3 `functions/*/requirements.txt` to `==` (inline `# comments` after a pin
  are fine — the assert skips only lines whose FIRST char is `#`). Then add the 3 files to `pip-audit.yml`'s `paths:`
  (push+PR) and add a `--requirement functions/<x>/requirements.txt` step for each. **Adjacent defect (flag, don't
  silently fix):** earnings/requirements.txt is missing `vertexai`/`google-cloud-storage` that main.py imports — the
  function is non-deployable as-is; queue a separate step per the queue-defects doctrine, or pin+add-missing only if
  Q/A agrees it's in-scope for "make requirements correct."
- **(g)** backend/Dockerfile: `FROM python:3.14-slim`; `COPY backend/requirements.txt .` + `RUN pip install -r
  requirements.txt` (assumes build context = repo root, which `COPY . .` already implies). frontend/Dockerfile deps
  stage: `COPY package.json package-lock.json ./` + `RUN npm ci`.
- **(h)** 5 migrations → `load_dotenv(Path(__file__).resolve().parents[2] / "backend" / ".env")`.
  extend_historical_data.py → `sys.path.insert(0, str(Path(__file__).resolve().parents[2]))` (add `from pathlib
  import Path`) and `load_dotenv(Path(__file__).resolve().parents[2] / "backend" / ".env")`. **DELETE** the 4 debug
  scripts (unreferenced; deletion satisfies the assert). Verify extend_historical_data.py with `python -m py_compile`
  + an import-path note (parents[2] = repo root ⇒ `import backend.*` resolves; do NOT actually run the import — it
  needs deps + .env, out of the $0 boundary).

---

## Test / behavioral legs BEYOND the assert (the assert is necessary-not-sufficient)

The immutable verification command is a **text-scan** `python3 -c` assert chain — it never imports the functions and
does NOT cover: leg (c) 500-return behavior, leg (d) "no TB in body" beyond a line-token check, leg (e) CORS/JSON/
status, or extend_historical_data's sys.path. Criteria #2/#3/#4/#6 demand more. Prescribed:

1. **(c)** Add `functions/ingestion/test_ingest_response.py` unit-testing the pure `_response_for(...)` helper:
   fetch-exception→`(_,500)`, load-failure→`(_,500)`, empty-df→`(_,200)`, rows→`(_,200)`. Pure helper ⇒ no
   `functions_framework`/BQ import needed ⇒ import-safe + `$0`.
2. **(d)** Q/A must read the ACTUAL `yield` lines and confirm the streamed body contains only `str(e)` (no
   `format_exc`, no `tb`), and that both `requests.get` carry a real `timeout=(5,30)` kwarg.
3. **(e)** Q/A verifies: env-var default (not a literal), a distinguishable failure signal, the 4-key validation, and
   no `*` in the response CORS header.

---

## Mutation matrix (standing doctrine — a guard that can't fail doesn't count; incl. the 75.15 OR/comment-token escape)

| Assert clause | Mutant that WRONGLY passes (escape hatch) | What Q/A must independently verify |
|---|---|---|
| `not os.path.exists(deploy_agents.sh)` | re-create the file → correctly FAILS (real guard) | file truly gone |
| `ing=='' or ('ingest_market_data_el' in ing and 'allow-unauthenticated' not in ing)` | keep real `--entry-point=ingestion_agent_http` but add a **comment** `# ingest_market_data_el` and drop allow-unauth → substring passes while config stays broken | the actual `--entry-point=` arg == `ingest_market_data_el` (or file deleted) |
| `q.count('timeout=')>=2` | put `timeout=` in a **comment/docstring** → count inflates | both `requests.get(:64,:140)` have a real `timeout=(5,30)` kwarg |
| `all('format_exc' not in ln for ln in q if 'yield'/'error_message' in ln)` | **rename `error_message`→`err_msg`** and keep `err_msg = f"...{traceback.format_exc()}"; yield f"ERROR:{err_msg}"` → the assign line has neither token, the yield line has no `format_exc` → **PASSES while still streaming the full TB** (the biggest hole) | trace what the `yield`/`Response` body actually contains end-to-end; only `str(e)` may reach the wire |
| `'gemini-1.5-flash-001' not in e` | hardcode a **different retired** model (`gemini-1.5-flash-002`) → passes | model comes from an env var AND the default is a currently-supported id |
| `all('==' in ln for ln in functions/*/requirements.txt if line and not startswith('#'))` | `pandas  # bump to == later` — unpinned dep with `==` in a **trailing comment** passes; also an **empty file** passes vacuously | every dep resolves to a concrete `name==x.y.z`; files not gutted; all imports covered |
| `'backend/requirements.txt' in b and '3.14' in b` | put both tokens in **comments** while real `COPY`/`FROM` stay wrong | the real `COPY` copies `backend/requirements.txt` and real `FROM` is `python:3.14-slim` |
| `'package-lock.json' in fr and 'npm ci' in fr` | tokens in comments | real `COPY … package-lock.json` + real `RUN npm ci` |
| `'.parent / "backend"' not in mig` (both quote styles) | a DIFFERENT CWD-relative form escapes — e.g. extend_historical_data's `load_dotenv("backend/.env")` is **not** matched | extend_historical_data.py + any bare-string `.env`/CWD bootstrap fixed too |

Mutate the STUB too: run the assert against the PRE-fix tree and confirm it FAILS on every leg (proves the guard
bites), then against the POST-fix tree.

---

## BOUNDARY confirmation

All changes are repo-file edits (delete/edit `.sh`/`.yaml`/`.py`/`Dockerfile`/`.txt`/`.yml`). No `gcloud`/`docker`/
network needed for verification: the immutable command only `open().read()`s files as text. Keep the functions'
`main.py` **import-safe without google-cloud deps** — do NOT add a test that imports `functions_framework`/`vertexai`/
`google.cloud` at module load (none are guaranteed in the venv); the leg-(c) test targets a pure helper. Dockerfile
correctness is by inspection only (no `docker build`). `pip-audit.yml` is edited but not run locally.

---

## External research — Read in full (7; clears the ≥5 floor)

| # | URL | Accessed | Kind | Key finding (quote) |
|---|-----|----------|------|---------------------|
| 1 | https://docs.cloud.google.com/run/docs/deploy-functions | 2026-07-24 | Google official | "--function … The value of this flag must be a function name … that exists in your source code." / "--allow-unauthenticated … assigns the Cloud Run IAM Invoker role to `allUser`, making your function publicly accessible. Omit this flag for authenticated or event-triggered functions." / base image now `python314` (Cloud Run functions form). |
| 2 | https://docs.cloud.google.com/functions/docs/securing/managing-access-iam | 2026-07-24 | Google official | `--allow-unauthenticated` grants `roles/run.invoker` to `allUsers`. "By default, entities that need to invoke an HTTP function must explicitly present authentication credentials." |
| 3 | https://docs.cloud.google.com/functions/docs/securing/authenticating | 2026-07-24 | Google official | Authenticated invocation = caller "Provide an ID token when it invokes the function" + holds "Cloud Run Invoker role (`roles/run.invoker`)" / perm `run.routes.invoke`. This is the OIDC pattern that replaces `--allow-unauthenticated` for scheduler→fn / fn→fn. |
| 4 | https://docs.cloud.google.com/scheduler/docs/reference/rpc/google.cloud.scheduler.v1 | 2026-07-24 | Google official | "The job is acknowledged by means of an HTTP response code in the range [200 - 299]." "A failure to receive a response constitutes a failed execution." RetryConfig: retry_count 0-5 (default 0), min_backoff 5s, max_backoff 1h, max_doublings 5. → **a 200-with-error-body reads as SUCCESS; you must return 500 for Scheduler to retry** (leg c). |
| 5 | https://cheatsheetseries.owasp.org/cheatsheets/Error_Handling_Cheat_Sheet.html | 2026-07-24 | OWASP official | "when an unexpected error occurs then a generic response is returned by the application but the error details are logged server side for investigation, and not returned to the user." "use 5xx to indicate errors … triggered on server side." → leg (d) sanitize-but-log; leg (c) 5xx. |
| 6 | https://owasp.org/Top10/2025/A10_2025-Mishandling_of_Exceptional_Conditions/ | 2026-07-24 | OWASP official (2025) | Maps **CWE-209** ("Generation of Error Message Containing Sensitive Information") + **CWE-550** + **CWE-636 "Failing Open"**. Scenario #2: "Sensitive data exposure via improper handling … reveals the full system error to the user." Recommends fail-closed + server-side logging. |
| 7 | https://requests.readthedocs.io/en/latest/user/advanced/ | 2026-07-24 | Requests official | "By default, requests do not time out unless a timeout value is set explicitly. Without a timeout, your code may hang for minutes or more." Tuple = `(connect, read)`: "Specify a tuple if you would like to set the values separately: `requests.get('https://github.com', timeout=(3.05, 27))`." → validates `timeout=(5,30)` (leg d). |

### Identified but snippet-only (context; does not count toward gate)
| URL | Kind | Why not fetched in full |
|-----|------|-------------------------|
| https://owasp.org/www-community/Improper_Error_Handling | OWASP | corroborates #5/#6; not needed in full |
| https://owasp.org/Top10/2021/A05_2021-Security_Misconfiguration/ | OWASP | prior-year canonical (superseded by A10:2025 for this topic) |
| https://cheatsheetseries.owasp.org/…/Error_Handling…#CWE-209 anchor | OWASP | same doc as #5 |
| https://registry.terraform.io/providers/hashicorp/google/latest/docs/resources/cloud_scheduler_job | vendor | RetryConfig field corroboration |
| https://cloud.google.com/scheduler/docs/reference/rpc/google.cloud.scheduler.v1beta1 | Google | beta mirror of #4 |
| https://docs.cloud.google.com/run/docs/authenticating/public | Google | "disable the Cloud Run Invoker IAM check" public-access alt |
| https://oneuptime.com/blog/post/2026-02-17-how-to-set-up-retry-policies-and-exponential-backoff-in-cloud-scheduler/view | blog (2026) | recency corroboration of retry semantics |
| https://oneuptime.com/blog/post/2026-02-17-how-to-troubleshoot-cloud-run-iam-invoker-permission-denied-errors/view | blog (2026) | recency corroboration of invoker role |
| https://requests.readthedocs.io/en/latest/user/quickstart/ | Requests | timeout quickstart mirror |
| https://github.com/psf/requests/issues/5227 | community | connect-vs-read clarification |
| https://codeql.github.com/codeql-query-help/java/java-stack-trace-exposure/ | CodeQL | CWE-209 detector rationale |

### Recency scan (2024-2026)
Performed (year-locked `2026`/`2025` + year-less canonical + explicit last-2yr passes). New/superseding findings:
(1) **OWASP Top 10:2025 A10 "Mishandling of Exceptional Conditions"** is the current home of stack-trace-exposure
guidance, mapping CWE-209/550/636 — it supersedes the 2021 A05 framing for this defect (source #6). (2) Google has
rebranded Cloud Functions → **"Cloud Run functions"**; the current deploy reference uses `--function` +
`--base-image python314` (source #1) — so the cloudbuild.yaml's `gcloud functions deploy --entry-point --runtime
python310` is the LEGACY form. This reinforces DELETE over fix for leg (b); if the pipeline is ever revived it should
be re-authored in the modern form. (3) requests timeout semantics unchanged (still no default timeout; tuple =
connect,read) as of the current 2.3x docs. No finding contradicts any leg's prescribed fix.

---

## Key findings (mapped to legs)
1. `--allow-unauthenticated` = `roles/run.invoker` to `allUsers`; the documented default is authenticated OIDC
   invocation (sources #1-3). Dropping it (leg b) + granting the caller SA `roles/run.invoker` is the sanctioned
   pattern — but here DELETE the whole stale config.
2. Cloud Scheduler treats **only 2xx as success and does not inspect the body** (#4) → the current ingestion
   200-on-failure silently hides failures from Scheduler; leg (c) 500s fix that.
3. Leaking tracebacks to callers is CWE-209/550 (A10:2025, #6); the fix is generic-client-message + server-side log
   (#5) — exactly leg (d) (keep `logging.critical`, sanitize the `yield`).
4. requests has **no default timeout** — an untimed `requests.get` "may hang for minutes or more" (#7); `timeout=(5,30)`
   = (connect 5s, read 30s), the correct tuple shape for leg (d).
5. `--entry-point` "must … exist in your source code" (#1) → cloudbuild's `ingestion_agent_http` would fail to deploy
   (only `ingest_market_data_el` exists) — confirms the leg (b) defect and the delete-or-fix requirement.

## Application to pyfinagent (external → file:line)
- #4 Scheduler-2xx → `functions/ingestion/main.py:72-79` + `data_fetchers.py:~108` (un-swallow).
- #5/#6 sanitize+log → `functions/quant/main.py:252-255` (keep :253 `logging.critical`).
- #7 timeouts → `functions/quant/main.py:64,140`.
- #1-3 OIDC/allow-unauth → `functions/ingestion/cloudbuild.yaml:13` (delete) + `deploy_agents.sh` (delete).

---

## Research Gate Checklist
Hard blockers (all satisfied ⇒ gate_passed true):
- [x] ≥5 authoritative external sources READ IN FULL via WebFetch (7)
- [x] 10+ unique URLs total (7 read + 11 snippet = 18)
- [x] Recency scan (last 2 years) performed + reported
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim

Soft:
- [x] Internal exploration covered every leg's module (a-h)
- [x] Consensus noted (OWASP + Google + requests all align; no contradiction)
- [x] Claims cited per-claim with URL + access date

---

## JSON envelope
```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 7,
  "snippet_only_sources": 11,
  "urls_collected": 18,
  "recency_scan_performed": true,
  "internal_files_inspected": 19,
  "coverage": {
    "audit_class": false,
    "rounds": 1,
    "dry_rounds": 0,
    "K_required": 2,
    "new_findings_last_round": 0,
    "dry": false
  },
  "summary": "Live-path resolved: only functions/quant/main.py is live (QUANT_AGENT_URL is a real cloudfunctions.net endpoint; orchestrator.py:1132/:1792 stream-parse FINAL_JSON:/ERROR: line-prefixes), so leg (d) must keep the ERROR: prefix + single-line and split format_exc into logging only. ingestion + earnings Cloud Functions are orphaned (contract-mismatch / superseded by in-backend earnings_tone) but still hardened per immutable criteria. Recommend DELETE deploy_agents.sh (a), DELETE cloudbuild.yaml+note (b), harden ingestion main.py to 500s which REQUIRES un-swallowing data_fetchers.py (c), timeout=(5,30)+sanitize quant (d), env-model+distinguishable-failure+no-* CORS earnings (e), ==-pin 3 requirements + pip-audit paths (f), python:3.14-slim/COPY backend/requirements.txt + npm ci/package-lock (g), parents[2] anchors in 5 migrations + extend_historical_data + DELETE 4 debug scripts (h). Stale anchor corrected: orchestrator.py:1061 is now :1132/:1792. Two mutation escape hatches flagged: the error_message->err_msg rename that still streams the TB, and the ==-in-a-comment requirements token. External (7 read): OWASP A10:2025/Error-Handling (CWE-209/550, sanitize+log), Cloud Scheduler 2xx-only success, requests no-default-timeout tuple, GCF --entry-point-must-exist + allow-unauth=roles/run.invoker-to-allUsers.",
  "brief_path": "handoff/current/research_brief_75.16.md",
  "gate_passed": true
}
```
