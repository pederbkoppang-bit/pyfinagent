# live_check_60.1 -- Deep-pipeline restoration + honest-degradation alarm (AW-4)

**Step:** 60.1 (phase-60, P0). **Date:** 2026-06-11. **All evidence below is from the LIVE stack** (operator's Mac, backend restarted via `launchctl kickstart -k gui/$(id -u)/com.pyfinagent.backend` after each fix iteration -- same deploy mechanism as the 56.x deploy this morning).

## A. Live smoke-call output for the replacement model (criterion 1, verbatim)

```
$ python scripts/debug/smoke_vertex_model.py gemini-3.1-flash-lite
[gemini-3.1-flash-lite] leg1 text_generation: FAIL ClientError: 404 NOT_FOUND. {'error': {'code': 404, 'message': 'Publisher Model `projects/sunny-might-477607-p8/locations/us-central1/publishers/google/models/gemini-3.1-flash-lite` was not found or your project does not have access to it. ...'}}
[gemini-3.1-flash-lite] leg2 structured_output: FAIL ClientError: 404 NOT_FOUND. ...
gemini-3.1-flash-lite: FAIL          (exit 1)

$ python scripts/debug/smoke_vertex_model.py gemini-2.5-flash gemini-3-flash-preview gemini-2.5-flash-lite
[gemini-2.5-flash] leg1 text_generation: PASS text='OK' tokens=26
[gemini-2.5-flash] leg2 structured_output: PASS parsed={'status': 'ok', 'score': 1}
[gemini-3-flash-preview] leg1/leg2: FAIL ClientError: 404 NOT_FOUND ...
[gemini-2.5-flash-lite] leg1 text_generation: PASS text='OK' tokens=8
[gemini-2.5-flash-lite] leg2 structured_output: PASS parsed={'status': 'ok', 'score': 1.0}
```

**Decision:** pin `gemini-2.5-flash` (PASS both legs, exit 0 when run alone). Google's named replacement `gemini-3.1-flash-lite` is NOT served in this project's region (us-central1) -- docs alone would have shipped another 404 pin; the smoke criterion did its job. `gemini-2.5-flash-lite` passed but was rejected (conflicting 2026-07-22 retirement capture; 6-week runway repeats AW-4).

**NOTE on criterion-1 line numbers:** the criterion text carries draft-time cites (model_tiers.py:63,73); the pins had drifted to :71,81 (now :84,:95 region with the GEMINI_WORKHORSE constant). The criterion's intent -- every retired pin migrated -- is satisfied by the repo-wide sweep; remaining `gemini-2.0-flash` strings are comments, the historical pricing row, the retired-label UI string, and an applied migration's SQL comment (verified by grep, list in experiment_results.md).

## B. BQ MCP evidence -- the away-week collapse baseline + the new pin live

Query (BQ MCP `execute_sql_readonly`, job_S8PTReIebVQj2SEkM2pNfQpm9tkV):
```sql
SELECT DATE(ts) AS day, COUNT(*) FROM pyfinagent_data.llm_call_log
WHERE model='gemini-2.0-flash' AND DATE(ts) BETWEEN '2026-05-25' AND '2026-06-11' GROUP BY day
-- 2026-05-26: 60 | 05-27: 88 | 05-28: 27 | 05-29: 14 | 06-01: 8 | 06-02 onward: ZERO ROWS
```
Exactly matches Google's documented discontinuation date (2026-06-01) and the AW-4 census.

New pin live (job_mJjQ5fa01mLMNAvLk9RKJIYVSAU_): `gemini-2.5-flash` rows present in `pyfinagent_data.llm_call_log` on 2026-06-11 (first call 10:35Z, from the live MU full-pipeline run's quant code-exec leg).

## C. Full-orchestrator US analysis end-to-end (criterion 1) -- THREE live failures found and fixed en route

The immutable criterion demanded a LIVE end-to-end run, and the live runs surfaced two additional AW-4 root-cause legs that code reading missed:

| Run | Result | Diagnosis (verbatim log) | Fix |
|---|---|---|---|
| MU c4686542 | FAILED at market | `Market timed out after 90s (attempt 1..3/3)` | gemini-2.5 thinks BY DEFAULT (2.0 did not); grounded+thinking blows 90s. `ThinkingConfig(thinking_budget=0)` now sent when caller didn't opt in (non-pro 2.5 only) |
| MU bae61830 | market PASSED ~30s; FAILED at competitor | `claude_code_invoke: success duration_ms=88862` logged at the second the step timed out | the 90s step budget RACES the Claude Code CLI rail (60-90s round trips, rail's own timeout 120s). `_resolve_step_timeout()`: grounded 90->180s; rails declare `recommended_step_timeout` (CC rail 150s); inner hang-guard 120->240s |
| MU 1556eebc | FAILED at competitor (pre-rail-fix) | same race, other coin-flip | (same fix, deployed after) |
| MU d1fbcc82 | **COMPLETED end-to-end** -- all 15 steps: market_intel, ingestion, quant, rag, market, competitor, data_enrichment, info_gap, enrichment_analysis, debate, macro, deep_dive, synthesis, bias_audit, risk_assessment | | |

**BQ MCP row evidence (criterion 1)** -- `financial_reports.analysis_results`, job_aDtkin1bGVIxY8XfonI6gcYGnpYt + job_eO5Zo5D4Tu_llvSpWuBFdfCXoeYX + job_A6YP2LvJKT2IvABn49UsRwUtl9Sv:

- MU 2026-06-11T11:37Z: `JSON_VALUE($._path) = "full"`, standard_model `claude-sonnet-4-6`, deep_think_model `claude-opus-4-7` (operator env), total_tokens 100,681, total_cost_usd 3.776 (cost_summary units).
- POPULATED enrichment/synthesis fields: top-level keys `alt_data, anomaly, competitor, debate, deep_dive, earnings_tone, insider, macro, market, nlp_sentiment, options, patent, quant, quant_model, rag, scenario, sector_analysis, social_sentiment, final_synthesis, _fact_ledger, _path, _session_context`; final_synthesis 46,661 chars; debate 3,541 chars; market.text 3,312 chars.
- CONTRAST (the away-week signature this step kills): the 2026-06-10 MU lite row has NONE of these keys (market/competitor/macro all NULL, no `_path`).
- DISCLOSED FLAKE (pre-existing class, NOT introduced by 60.1): `final_score=0.0 / recommendation N/A` on this row -- backend.log 12:09:22 `Synthesis-Final returned invalid JSON` (CLI-rail JSON-shape flake, fail-open to draft; same warning class existed pre-60.1, e.g. Critic invalid-JSON in away-week logs; full cycles 05-22..05-27 produced real scores on the same rail). In an autonomous CYCLE such a row now counts toward the 56.2 degraded-scoring guard (>=3 zero-scores alarm) -- i.e. the honest-degradation machinery this phase exists for makes the flake VISIBLE instead of silent. Root-causing the synthesis JSON shape on the CC rail is out of 60.1's immutable scope; routed to the cycle_block_summary candidate list.

## D. KR analysis with skipped-stage tags (criterion 2)

Live run 005930.KS (analysis 21e5868b, POST /api/analysis 2026-06-11): the three SEC-bound stages degraded to EXPLICIT tagged skips and the pipeline CONTINUED (away week: hard abort at the CIK stage -> silent lite fallback). Verbatim step log:

```
market_intel | completed | Got 10 articles from yfinance
ingestion    | completed | Skipped -- non-SEC market (no EDGAR filings)
quant        | completed | SEC quant skipped (non-US listing) -- yfinance fundamentals only
rag          | completed | Skipped -- non-SEC market (no EDGAR corpus)
market       | (continues on the market-agnostic agents...)
```

**BQ MCP row evidence (criterion 2)** -- run COMPLETED end-to-end; `financial_reports.analysis_results`, job_cYARriPuUcIVVpGB_tdmpKLLnfLu:

- 005930.KS 2026-06-11T12:04Z: `JSON_VALUE($._path) = "full"`, `$.skipped_stages` verbatim:
```json
[{"reason":"non-US listing: SEC EDGAR has no filings for this symbol","stage":"ingestion_sec"},
 {"reason":"non-US listing: not in SEC CIK mapping; fundamentals from yfinance only","stage":"quant_cf_sec"},
 {"reason":"non-US listing: RAG corpus is SEC 10-K/10-Q only","stage":"rag_sec_filings"}]
```
- quant built from yfinance: `$.quant.company_name = "Samsung Electronics Co., Ltd."`, `$.quant.part_1_financials.source = "yfinance only -- SEC EDGAR skipped (non-US listing, phase-60.1 KR-aware skip)"`.
- The market-agnostic agents RAN: final_synthesis 41,508 chars, market.text 3,336 chars. (Same disclosed 0-score synthesis-JSON flake class as the MU row -- see section C.)
- Away-week contrast: every .KS analysis aborted at the CIK stage and fell back to lite silently; this row is the first KR full-path row in the table.

Design choice (researcher-grounded, recorded in contract.md): honest tagged-skip NOW (`ingestion_sec`, `quant_cf_sec`, `rag_sec_filings` tags persisted in full_report_json; yfinance fundamentals via `_quant_from_yfinance`), OpenDART integration deferred (free key, ~10k req/day, corpCode.xml mapping -- a future ingestion surface, not a P0 restoration).

## E. Fallback-alarm unit-test output (criterion 3, verbatim)

```
$ python -m pytest backend/tests -k 'fallback_alarm or model_pin or 60_1' -q
22 passed, 780 deselected, 1 warning in 5.78s    (exit 0)
```
Includes `test_fallback_alarm_fires_on_away_week_100pct` -- 5/5 analyses tagged with the EXACT away-week failure shapes (gemini-2.0-flash 404 x2, 90s timeout, SEC-CIK abort x2) -> fire=True with per-ticker reasons named. Strict-> boundary: 2/4 quiet, 3/5 fires. Deliberate lite_mode: never fires.

## F. Provenance surfaces (criterion 4) + live UI capture (cycle-2 fix)

- Criterion-4's operator-visible surface is the DIGEST leg (the criterion's "and/or"); the reports UI renders unchanged, its API now ALSO carries `analysis_path` for a future UI step.
- **Cycle-2 correction (first Q/A flagged this):** the criterion-1 repin sweep DID change a UI surface -- `frontend/src/app/settings/page.tsx` (model-picker selectable set, fallback default, retired-label string) -- so the step's "Playwright capture if any UI surface changed" clause applies. An earlier draft of this section said "no UI surface was touched"; that wording was overbroad and is retracted.
- **Live Playwright MCP capture (2026-06-11 12:55 CEST):** `handoff/current/captures_60.1/settings-model-picker-60-1.png` (viewport screenshot) + `settings-snapshot.yml` (accessibility snapshot). Verified on the live page (browser_navigate + browser_snapshot + browser_take_screenshot):
  - The Standard AND Deep Think pickers list **"Gemini 2.5 Flash -- Gemini -- $0.3/2.5"** (the corrected pricing flowing live from the repinned `AVAILABLE_MODELS`).
  - **No "Gemini 2.0 Flash" entry anywhere in the rendered pickers** (removed from `_VALID_MODELS`/`AVAILABLE_MODELS`/`PRIMARY_MODEL_NAMES`; the retired model cannot be selected).
  - Saved values render their proper labels (Standard "Claude Sonnet 4.6", Deep Think "claude-opus-4-7") -- no blank/crash from the picker-set change; the SCREENSHOT shows the page rendered fully above the fold (sidebar, Analysis Mode, Debate Depth, Model Configuration with both pickers). Attribution note (Q/A-2): settings-snapshot.yml is an authentic but PRE-LOAD accessibility capture (version badge = the NEXT_PUBLIC_APP_VERSION initial, proving live origin) -- the section-rendering evidence lives in the screenshot, not the YAML; the below-fold Signal Stack section is not evidenced by either artifact (incidental to no immutable criterion).
  - Method per `.claude/rules/frontend.md` "Live-UI verification": second dev server `LIGHTHOUSE_SKIP_AUTH=1 npx next dev --port 3100` (operator's :3000 instance untouched -- verified 302-to-login after teardown); capture on the session-connected `@playwright/mcp` server (pinned 0.0.76 in .mcp.json; connected this session). :3100 killed after capture.
- Live API check (post-restart): `GET /api/reports/?limit=4` returns (verbatim): `005930.KS 0.0 -> 'full' | MU 0.0 -> 'full' | SNDK 7.0 -> None | HPE 7.0 -> None` -- new rows carry the tag end-to-end through BQ -> SQL JSON_VALUE -> ReportSummary -> API; pre-tag rows render unchanged, no invented provenance. The next morning digest renders `[full]` markers on these lines (formatter unit-tested; digest fires on its own schedule).
- Unit test `test_60_1_digest_renders_lite_full_markers`: `[lite]`/`[full]` markers render; pre-tag rows unchanged.

## G. Burn disclosure (58.1 spend ledger cross-ref)

The full-pipeline restoration this step performs is the ONE sanctioned live behavior change of the phase-60 goal (the $25 window approval contemplated full-mode cycles at $1.08-4.06/cycle). Metered burn from this step's live verification: Vertex gemini-2.5-flash calls (RAG/quant-exec/grounded legs of 4 MU runs + 1 KR run, $0.30/$2.50 per Mtok) + gemini-2.5-pro deep-think legs; the Claude legs ran on the $0 flat-fee CLI rail. Ledger row appended to live_check_58.1.md. ALSO NOTE: cost_tracker pricing for 2.5-flash corrected from $0.15/$0.60 to the verified $0.30/$2.50 -- future burn reports price the new pin correctly.

## H. Follow-up triggers recorded (operator visibility)

1. **2026-10-16: Gemini 2.5 family retirement** -- re-run `scripts/debug/smoke_vertex_model.py` against the then-served replacement and update `model_tiers.GEMINI_WORKHORSE` (one line) + `settings.deep_think_model` (gemini-2.5-pro retires the same day). Recorded at the constant and in cycle_block_summary.
2. Gemini 3.x on Vertex requires either a different region or the global endpoint for this project -- evaluate at the same time (google-genai SDK already in use, so no SDK blocker).
