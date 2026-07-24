# live_check — step 72.0 (P0 scoring-rail restoration audit)

Evidence class required by `verification.live_check`: verbatim backend.log/BQ `llm_call_log` lines pinning the degraded-scoring onset + failing provider/key, plus the operator `.env` grep output or an explicit note it was not provided. Collected 2026-07-18 by forensics workflow `wf_0542bf62-ffb` (3 read-only auditors + adversarial verifier; full JSON in the session task output, key rows reproduced verbatim below).

## Onset — verbatim log lines

ROOT onset (earliest credit-400 anywhere; `handoff/logs/backend.log.20260612T104931Z.gz`, date-anchored by adjacent APScheduler "scheduled at 2026-05-17 03:55:39.934229+02:00"):
```
03:55:44 E [orchestrator] Enrichment agent Options Flow failed: Error code: 400 - {.."message": "Your credit balance is too low to access the Anthropic API. Please go to Plans & Billing to upgrade or purchase credits.".. request_id req_011Cb7JtX5fXgpryDPiYpSxo}
```

Surface-B onset (first meta_scorer failure, 2026-05-22 22:32:06, same gz; SecretStr bug later fixed by phase-51.1 → failure mode became the credit-400 on the same direct path):
```
22:32:06 W [meta_scorer] meta_scorer LLM call failed: Header value must be str or bytes, not <class pydantic.types.SecretStr>
```

First observable guard markers (2026-06-11 — instrumentation deploy, ~3.5 weeks after root onset; note the alert path was dead at that moment):
```
20:01:22 W [autonomous_loop] Meta-scorer ran ENTIRELY on the no-LLM fallback for 10 candidates (conviction overlay degraded; damping leg inactive)
21:09:35 W [autonomous_loop] Degraded-scoring guard fired: 3/5 analyses scored 0/degraded
21:09:35 W [autonomous_loop] Degraded-scoring guard errored (non-fatal): No module named backend.services.alerting
```

Still failing at step close (current `backend.log`, 2026-07-17 evening):
```
20:00:46 W [macro_regime] Macro regime LLM call failed: Error code: 400 - {..credit balance is too low..}
20:00:51 W [news_screen] News screen LLM call failed: Error code: 400 - {..credit balance is too low..}
20:01:22 W [meta_scorer] meta_scorer LLM call failed: Error code: 400 - {.."message": "Your credit balance is too low to access the Anthropic API...".. request_id req_011Cd84sKfre3jyP2Y3eCKcB}
```

Current-period rail state (current `backend.log`):
```
20:09:10 W [alerting] alert bot-token fallback delivered=True source=claude_code_rail title=Claude Code rail breaker OPEN -- 20 consecutive failures; remaining rail calls skipped this cycle   (fired 07-10/13/14/15)
20:03:29 W [claude_code_client] claude_code_invoke: subprocess timeout after 120s prompt_len=6483   (305 occurrences)
20:02:45 I [claude_code_client] claude_code_invoke: success duration_ms=72721 input_tokens=3863 output_tokens=3274
```

## Failing provider/key — BQ `sunny-might-477607-p8.pyfinagent_data.llm_call_log`

Daily provider series (SQL: `SELECT DATE(ts) d,provider,COUNTIF(ok),COUNTIF(NOT ok) FROM llm_call_log WHERE ts>=TIMESTAMP('2026-05-01') GROUP BY d,provider`): anthropic direct = 0 successes every day **2026-06-15..2026-07-03**; partial recovery 07-07..07-09; re-blackout 07-13..07-15; ok rows 07-16..17.

Surface-A regression attribution (SQL: `SELECT agent,provider,model,COUNTIF(ok),COUNTIF(NOT ok) FROM llm_call_log GROUP BY agent,provider,model`):
```
agent='cc_rail'  provider='anthropic'  claude-sonnet-4-6: 515 ok / 2545 fail   (rail-tagged calls on the credit-dead DIRECT API)
agent='cc_rail'  provider='anthropic'  claude-opus-4-7:    45 ok /  451 fail
provider='claude-code' model='claude-code-cli': 19 rows TOTAL; last row 2026-07-09 19:17:17 (2 ok 07-08, 6 ok 07-09; silent since)
```

Surface-B telemetry blind spot (SQL: `SELECT agent,COUNT(*) FROM llm_call_log GROUP BY agent`): **zero** rows for any meta_scorer-like agent — the meta-scorer bypasses the rail AND the telemetry writer (`meta_scorer.py:220-225` direct ClaudeClient with `anthropic_api_key`).

Synthetic-fixture audit (verifier): anthropic "ok" rows on 05-19..06-10 are all exactly input_tok=1000/output_tok=50 (smoke fixtures); genuine varied-token successes cease after 2026-05-17.

Healthy leg (same daily series): gemini gemini-2.5-flash **229 ok / 0 fail** (06-11..07-17); gemini-2.0-flash 199 ok / 0 fail (05-16..06-01).

## Live host state (2026-07-18, read-only)

```
claude auth status -> {"loggedIn": true, "authMethod": "claude.ai", "apiProvider": "firstParty", "email": "peder.bkoppang@hotmail.no", "subscriptionType": "max"}   (OAuth ALIVE)
ps -axo pid,lstart,command -> 98681 ons. 8 jul. 23.24.28 2026 .venv/bin/uvicorn backend.main:app --host 0.0.0.0 --port 8000   (backend last restart 2026-07-08 23:24:28 CEST; launchctl runs=2, state running)
```
Implication: `paper_use_claude_code_route` is effectively True at runtime (rail invocations present post-restart) despite the code default False; any `.env` edit after 07-08 23:24 is NOT loaded into this process.

## Operator .env grep

**NOT provided as of step close 2026-07-18** (requested twice in-session; `backend/.env` is permission-blocked for both Main and subagents). Best-available substitutes recorded above: runtime behavior (rail invoked ⇒ route effectively ON) + `launchctl print` env block (no `PAPER_USE_CLAUDE_CODE_ROUTE` in the launchd env, so it comes from `.env` file load). The 72.1 token-reconciliation step re-requests the grep; criteria's escape hatch ("or an explicit note it was not provided") applies here.
