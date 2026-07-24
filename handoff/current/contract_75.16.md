# Contract -- Step 75.16: Cloud Functions + Docker deploy-surface retirement/hardening + script bootstrap repair

- **Step id**: 75.16 (phase-75, Audit75 S16) -- P1, sonnet-tier -> **Sonnet executor GENERATE** (7th delegated; Main reviews + re-measures).
- **Date**: 2026-07-24
- **BOUNDARY**: repo-only edits; NO gcloud/docker/network commands; $0 metered; functions main.py files stay import-safe without google-cloud deps (test pure helpers only).

## Research-gate summary (gate PASSED)

Workflow `wf_e787b99b-faf` (opus/max, moderate). Envelope: `7 read-in-full (GCF OIDC/unauthenticated, Cloud Scheduler v1 ack semantics, OWASP A10:2025/CWE-209, requests timeout docs, cloudbuild reference...), snippet=11, urls=18, recency=true, internal=19, gate_passed=true`. Brief: `research_brief_75.16.md`.

**Corrections adopted (binding):**
1. **THE LIVE PATH**: `quant_agent_url` consumer is orchestrator.py:1132 (client.stream GET) called from :1792; settings.py:119 REQUIRED field; **QUANT_AGENT_URL in the operator .env is a real cloudfunctions.net endpoint -> functions/quant/main.py IS LIVE.** The orchestrator parses the stream BY LINE PREFIX (:1134-1144: `FINAL_JSON:` -> json.loads; `ERROR:` -> raise). The leg-(d) sanitized error MUST: keep the `ERROR:` prefix, be a SINGLE line, and never rename `FINAL_JSON:`/`ERROR:`. (Today's traceback is already first-line-truncated by aiter_lines -- sanitizing changes nothing the orchestrator sees, only stops the leak to unauthenticated callers.)
2. **functions/earnings + functions/ingestion are ORPHANED** (live earnings = in-backend earnings_tone service; the ingestion function contract-mismatches the orchestrator's stream expectation). Hardened per criteria but zero production-contract risk. **Delete-over-fix default for cloudbuild.yaml** (orphaned half-built refactor, April-2026 single commit, zero callers/schedulers, entry-point mismatch; criteria accept deletion with a retirement note).
3. **Leg (c) needs data_fetchers.py too**: the broad `except Exception: return pd.DataFrame()` (~:108) swallows fetch errors -- main.py alone cannot 500 on fetch exception. Re-raise genuine errors; empty df only for genuine no-data. (Cloud Scheduler acks ANY 2xx and ignores the body -- the current 200-with-Failure-body reads as success; 500s are required for retries.)
4. cloudbuild currently pins python310 (not 311); moot under deletion.
5. **Two mutation escape hatches in the immutable assert** (executor must close via tests): the traceback check is defeated by renaming `error_message` -> `err_msg` (still streams the traceback); the `==` requirements check passes on `==` inside a trailing comment and vacuously on an empty file.

**Queued this step (discovered defects, own steps): 75.16.1** -- functions/earnings/requirements.txt missing vertexai + google-cloud-storage (function non-deployable) AND functions/earnings/main.py:120 untimed requests.get (out of leg-e scope).

## Plan steps (per research recommendations, verbatim adopted)

(a) DELETE scripts/deploy/deploy_agents.sh. (b) DELETE functions/ingestion/cloudbuild.yaml with a retirement note. (c) ingestion main.py explicit (body,status): 500 fetch-exception / 500 BQ-load-failure / 200 empty-success / 200 success, via a PURE status-decision helper + import-safe unit test; data_fetchers.py re-raises genuine errors. (d) quant main.py: timeout=(5,30) on both requests.get (:64,:140); traceback ONLY to logging.critical; `yield f'ERROR: QuantAgent failed for {ticker_str}: {str(e)}'` single-line; stream tokens untouched. (e) earnings main.py: model id from env var with a current default (2.5-flash retires 2026-10-16; prefer the live earnings_tone service's model); NLP failure distinguishable (non-200/status field, never {'error':...}-as-data); 4-key JSON validation; wildcard CORS -> the backend localhost/Tailscale allowlist idiom. (f) ==-pin all 3 functions requirements.txt (non-comment lines); add all 3 to pip-audit.yml paths + per-file --requirement steps. (g) backend/Dockerfile python:3.14-slim + real requirements install; frontend/Dockerfile deps via `npm ci` from the committed lockfile. (h) 5 migrations + extend_historical_data.py -> Path(__file__).resolve().parents[2] anchors (py_compile only -- no import execution); DELETE the 4 unreferenced scripts/debug/*.py.

**Tests** (new backend/tests/test_phase_75_deploy_surface.py or similar, import-safe): the pure ingestion status helper (all 4 outcomes); a quant-source scan that closes BOTH escape hatches (no `format_exc` reference reachable from any yield-adjacent assignment REGARDLESS of variable name -- e.g. assert `format_exc` appears only in the logging call's line/block; and single-line ERROR yield shape); requirements parsed-line pins (comments stripped FIRST, non-empty guard); Dockerfile content markers; migration anchor asserts.

**Mutation matrix** (each KILLED): restore the traceback into the yield UNDER A RENAMED variable (the escape hatch -- must die); un-pin one requirements line via a `==`-in-comment dodge (must die); revert one migration anchor; restore a 200-on-failure path (helper test dies); drop one timeout; restore wildcard CORS; run the IMMUTABLE assert against the PRE-fix tree (must FAIL on every leg -- proves it bites) and POST-fix (exit 0); STUB mutation on the helper test fixture.

**Verification**: immutable command verbatim exit 0; full backend suite vs the NEW 9-red baseline (tree_fails_75_15.txt is the comparator; CI-equivalent selection stays green); ruff derived-scope; py_compile on touched scripts; yaml.safe_load on pip-audit.yml; NO gcloud/docker/network calls anywhere.

## NOT in scope
Deploying/redeploying anything; the earnings missing-deps + untimed-get (queued 75.16.1); QUANT_AGENT_URL value changes; orchestrator.py stream-parsing changes.

## References
research_brief_75.16.md; audit_phase75/confirmed_findings.json (sre-ops-08, gap6-03/04/05/08...); the 75.13 pip-audit precedent; the 75.15 OR-escape-hatch lesson.
