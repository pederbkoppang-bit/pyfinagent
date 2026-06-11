# Contract -- 60.1 Deep-pipeline restoration + honest-degradation alarm (AW-4)

**Step:** 60.1 (phase-60, P0, harness_required). **Date:** 2026-06-11.
**Operator gate:** phase-60 installed per verbatim decision "Install minus 60.5 (Recommended)" (commit 7524e3cf).
**AW basis:** AW-4 (handoff/current/59.3-harness-free-output.md §3-4; cites = snapshot 70a8242b).

## Research-gate summary (researcher a8594338, tier=complex, gate_passed:true)

6 official-tier sources read in full, 31 URLs, 9 queries (3 variants x 3 topics), recency scan present. Brief: handoff/current/research_brief.md.

- **Root cause (external):** `gemini-2.0-flash-001`/`-lite-001` **discontinued on Vertex June 1, 2026** (Google model-lifecycle docs). The backend.log 404s starting 06-02 are the retirement landing, not an outage. llm_call_log collapse 88/day (05-27) -> 0 (06-02+) matches.
- **Migration target:** `gemini-3.1-flash-lite` (Google's named replacement, $0.25/$1.50 per Mtok) PRIMARY; `gemini-2.5-flash` ($0.30/$2.50, family already proven on this stack via the live `gemini-2.5-pro` deep-think rail) FALLBACK. The immutable live-smoke decides: Gemini 3.x is unproven on the repo's legacy `vertexai` SDK. Grounding + structured output survive on both families. LANDMINE: the 2.5 family retires 2026-10-16 -- if the fallback is chosen, that date is documented as a follow-up trigger.
- **KR design choice (criterion-2, recorded here):** **honest tagged-skip now; DART integration deferred.** Grounding: the single hard abort is the quant Cloud Function's SEC-CIK dependency (`functions/quant/main.py:88`, a hard step-2 dependency; ingestion is best-effort since 27.6.6); yfinance fundamentals already merge at `orchestrator.py:951-953`, so the 26+ market-agnostic agents can run without CIK. OpenDART (free key, ~10k req/day, corpCode.xml ticker->corp_code mapping, English portal since 2025-02) is a viable FUTURE ingestion path -- not a P0 build. Sources incl. FSS OpenDART official, dart-fss docs, XBRL.org KR English-disclosure expansion (>=5 incl. KRX/DART options -- floor met).
- **Internal corrections to the draft cites:** silent fallback is at `autonomous_loop.py:1529-1541` (draft's 1411-1419 is stale); pins drifted to `model_tiers.py:71,81` (draft said 63,73). Per-ticker failure reasons currently exist ONLY in a log line -- the alarm's missing input. 56.2 guard block at `autonomous_loop.py:898-925` (predicate :1669-1696) alerts via `backend/services/observability/alerting.py:119 raise_cron_alert` -- the new alarm wires THERE. Provenance `_path` is persisted (`_persist_analysis` :2180+) but NO surface displays it (digest `formatters.py:384-397` and `backend/api/reports.py` both blind). 90s timeout at `orchestrator.py:679`. No one-shot Vertex smoke script exists; GENERATE adds one. 6 test files assert the literal old pin and will false-red a correct fix -- they are updated in the same pass.

## Hypothesis

Repinning to a live-smoke-proven, currently-served Gemini model + a KR-aware tagged skip of the CIK-dependent quant stage + a fallback-rate alarm wired into the 56.2 guard path + lite/full provenance in the digest restores the intended full pipeline (the $25-window approval contemplated full-mode cycles) and makes the away-week class of silent degradation impossible to repeat unnoticed -- at burn within the approved envelope (smoke = pennies; one full US analysis ~$0.19-0.27 effective).

## Immutable success criteria (verbatim from .claude/masterplan.json step 60.1)

**Command:** `source .venv/bin/activate && python -m pytest backend/tests -k 'fallback_alarm or model_pin or 60_1' -q && test -f handoff/current/live_check_60.1.md`

1. "the retired gemini-2.0-flash pins (model_tiers.py:63,73; orchestrator.py:382 _GEMINI_FALLBACK; the hardcoded Gemini-locked roles noted at settings.py:35) are migrated to a researcher-validated, currently-served model (availability proven by a live smoke call exiting 0, not by docs alone), and one full-orchestrator analysis completes end-to-end on the live stack for >=1 US ticker with the full-path rail/model tags persisted -- evidenced by a BigQuery MCP row from financial_reports.analysis_results showing a non-lite row with populated enrichment/synthesis fields"
2. "KR (.KS) tickers no longer abort the whole full pipeline on the SEC-CIK stage: either the CIK-dependent stages (ingestion/RAG) degrade to an explicit, tagged KR-aware skip while the remaining agents run (proven by a full-path KR row with skipped-stage tags), or the KR limitation is documented and ALERTED per cycle -- silence is a FAIL; the chosen design is researcher-grounded (>=5 sources incl. KRX/DART filing-source options) and recorded in the contract"
3. "a fallback-rate alarm exists: any cycle whose full->lite fallback ratio exceeds a configurable threshold (default 50%) fires a Slack alert naming per-ticker failure reasons, covered by a unit test reproducing the away-week 100%-fallback case; the alarm is wired alongside the 56.2 degraded-scoring guard, not a parallel bespoke path"
4. "lite-fallback provenance is operator-visible: digest 'Recent Analyses' lines and/or the reports UI distinguish lite-path from full-path scores (the away week showed identical-looking 7.0/10 rows from a 2-call momentum wrapper); if the UI is touched, a Playwright MCP capture is REQUIRED in the live_check per the 59.2 binding rule"

**live_check:** "REQUIRED -- BQ MCP query + result rows showing post-fix full-path analyses (US and KR or the KR alert transcript), the fallback-alarm unit-test output, the live smoke-call output for the replacement model, and a Playwright capture if any UI surface changed."

NOTE on criterion-1's pin line numbers: the criterion text carries the draft-time cites (model_tiers.py:63,73). The pins have drifted to :71,81 (researcher-verified). The criterion's INTENT (every retired pin migrated) is satisfied by the full-inventory sweep; the live_check records the drift.

## Plan

1. **Smoke script** `scripts/smoke_vertex_model.py` -- one-shot generate_content against a CLI-given model id via the project's Vertex client; exit 0 on text response. Run LIVE for `gemini-3.1-flash-lite`; if it fails on the legacy SDK, run for `gemini-2.5-flash`. The exit-0 winner becomes the pin (capture output verbatim for live_check).
2. **Repin sweep** (every occurrence from the brief's inventory): `model_tiers.py:71,81`, `orchestrator.py:382` `_GEMINI_FALLBACK`, MAS fallback `:237-243`, `settings.py:35` Gemini-locked roles, harness/evaluator/cost-tracker/agent-map/`_inventory.json`/frontend/`.env.example`, + the 6 literal-pin test files. New pin goes through ONE settings-level constant where feasible to prevent recurrence.
3. **KR tagged-skip**: in the full-orchestrator path, detect non-SEC tickers (.KS) BEFORE the quant-CF CIK stage; record an explicit skipped-stage tag (e.g. `skipped_stages: ["quant_cik"]` with reason) and continue the market-agnostic agents. No Cloud Function redeploy (orchestrator-side guard only).
4. **Fallback-rate alarm**: capture per-ticker failure reason at the fallback site (`autonomous_loop.py:1529-1541`) into cycle state; at cycle end compute fallback ratio; if > `settings.fallback_alarm_threshold` (new, default 0.5) call `raise_cron_alert` with per-ticker reasons, wired in/abutting the 56.2 guard block (:898-925). Unit test reproduces the away-week 100% case.
5. **Provenance**: digest "Recent Analyses" lines gain a `[lite]`/`[full]` marker from the persisted `_path` tag (formatters.py:384-397). UI not touched in this step (digest satisfies the criterion's "and/or"); if that changes, Playwright capture per 59.2.
6. **Tests** named into the immutable -k net (`fallback_alarm or model_pin or 60_1`): model-pin sweep test (no retired id anywhere in backend config), alarm trigger/no-trigger boundary, KR-skip tag, digest marker, OFF-path identity where applicable.
7. **Live verification**: restart backend; run ONE full-orchestrator analysis for >=1 US ticker; BQ MCP row (non-lite, populated enrichment/synthesis) + KR run or alert transcript; write `live_check_60.1.md`; full pytest suite; spend-ledger row appended to live_check_58.1.md (burn disclosure per goal).
8. Fresh Q/A spawn -> harness_log append -> flip 60.1 done.

## Do-no-harm

The lite analyzer path and US momentum core are untouched except: (a) the model id constants (the retired id cannot be "kept" -- it no longer exists server-side), (b) additive alarm/provenance code. No flag flips; the alarm and skip-tags are observability. The repin itself is the goal's ONE sanctioned live behavior change (restores the intended pipeline).

## References

- handoff/current/research_brief.md (full citations + URL table)
- Google Cloud Vertex model lifecycle/deprecations docs; ai.google.dev pricing; FSS OpenDART (engopendart.fss.or.kr); SRE Workbook "Alerting on SLOs"; dart-fss docs; XBRL.org KR disclosure note
- 59.3 report AW-4; 56.2 guard (autonomous_loop.py:898-925); 27.6.6 best-effort ingestion precedent
