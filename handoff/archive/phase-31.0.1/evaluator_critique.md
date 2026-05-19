# Q/A Critique -- phase-31.0.1 (Smoketest Stage 1: screen + rank + sector enrichment)

**Step:** phase-31.0.1 -- Smoketest Stage 1.
**Date:** 2026-05-20.
**Cycle:** 1 (first Q/A spawn for Stage 1 of the morning-goal smoketest).
**Effort:** max.

## 5-item harness-compliance audit (MANDATORY -- runs FIRST)

| # | Item | Verdict | Evidence |
|---|------|---------|----------|
| 1 | Researcher gate ran? | **PASS-with-NOTE** | `handoff/current/research_brief.md` JSON envelope (lines 308-320): `tier=deep`, `external_sources_read_in_full=18`, `snippet_only_sources=8`, `urls_collected=26`, `recency_scan_performed=true`, `adversarial_tags_present=true`, `gate_passed=false` (HONEST disclosure: 4 sources hit 403/402/404 paywalls -- S&P methodology PDF, Wiley J. of Finance, Springer J. Fin Mkts, SSGA + AlphaArchitect -- properly catalogued in snippet-only table at lines 144-154). Three-variant search composition explicit at lines 18-21 (2026 frontier + 2024-2025 last-2-year + year-less canonical). [ADVERSARIAL] tags on sources 22 (Wright Research 2025 momentum decay) + 23 (Wilder 70/30 vs 80/20 threshold cross-check). Content-completeness: code audit verified verbatim at lines 23-82 (matches the live `screener.py:64-72, 179-201, 370` and `autonomous_loop.py:305-310, 579-596` I spot-checked); Test Design #3 (full production chain) fully specified with file:line anchors at lines 192-262. Floor miss is 2 sources of 20; the gate failure is the standards-rigorous Q/A-friendly disclosure pattern (honest false), not a content-completeness gap. Judgment for a Stage 1 SMOKETEST (shape verification only, per Sealos ML smoke-test discipline source #15): floor miss is SOFT not HARD blocker, accepted as PASS-with-NOTE. |
| 2 | Contract written before generate? | **PASS** | `handoff/current/contract.md` exists with research-gate summary (lines 7-28), hypothesis (30-35), immutable success criteria copied from morning-goal Stage 1 spec (39-45), plan (49-55), hard guardrails (59-66), references (70-74). Contract carries researcher's KEY recommendation verbatim ("Test design #3: full production chain... mirrors `autonomous_loop.py:305-310 + 541-596`"). Contract pre-dates `experiment_results.md` (timestamps confirm). |
| 3 | Results file present? | **PASS** | Three files. `handoff/current/experiment_results.md` (top-level rolling) declares verdict PASS with 4-row result table at lines 26-32, files-touched table at lines 17-23, success-criteria check at lines 60-66. Per-stage `handoff/smoketest_20260520/STAGE_1_results.md` (79 lines) carries the execution code block (lines 15-33), output table (37-42), 6-assertion table (54-61), and hard guardrail attestation (68-74). Machine-readable `handoff/smoketest_20260520/STAGE_1_screen_universe_output.json` (52 lines, verdict=PASS, 4 rows, all with sector + composite_score) for downstream stages to consume. |
| 4 | Log NOT yet written? | **PASS** | `grep -c "phase-31.0.1" handoff/harness_log.md` returns 0 -- no entry for this step yet. Log-LAST discipline preserved; the entry will be appended after THIS PASS verdict per the canonical order. |
| 5 | No verdict-shopping? | **PASS** | First Q/A spawn for phase-31.0.1. Prior `evaluator_critique.md` was phase-30.4 (different step-id, PASS verdict, archived appropriately). `grep -c "phase-31.0.1.*result=CONDITIONAL" handoff/harness_log.md` returns 0 -- no prior CONDITIONAL on this step (3rd-CONDITIONAL auto-FAIL rule N/A). No sycophancy-under-rebuttal surface (no prior verdict to flip). |

## Deterministic checks

| Check | Command | Result |
|-------|---------|--------|
| Output file exists | `test -f handoff/smoketest_20260520/STAGE_1_screen_universe_output.json` | **FILE_EXISTS PASS** |
| JSON shape valid | `python -c "import json,sys;d=json.load(sys.stdin);assert d['verdict']=='PASS' and len(d['rows_post_enrichment'])==4 and all('sector' in r and 'composite_score' in r for r in d['rows_post_enrichment'])"` | **JSON shape OK PASS** (verdict=PASS, 4 rows, all rows have both `sector` and `composite_score` fields) |
| Smoketest re-run reproducibility | `python -c "from backend.tools.screener import screen_universe, rank_candidates; rows=screen_universe(tickers=['AAPL','MSFT','NVDA','JPM'], period='6mo'); ranked=rank_candidates(rows, top_n=4, strategy='momentum'); print(len(ranked), all('composite_score' in r for r in ranked))"` | **REPRO_LEN=4 PASS; REPRO_CS_PRESENT=True PASS; REPRO_CS_NUMERIC=True PASS; REPRO_TICKERS=['NVDA','AAPL','MSFT','JPM']** (identical ordering to the persisted JSON output -- deterministic) |
| Diff scope (no backend code modified) | `git diff --stat` | Only `handoff/*` (5 files: contract.md +113/-?, experiment_results.md +/- net rewrite, research_brief.md +/- rewrite, harness_log.md +17 audit-only appends from hooks) + `handoff/audit/*.jsonl` (auto-managed audit logs from PreToolUse / InstructionsLoaded hooks; not edits by the agent). `.claude/.archive-baseline.json` +8 is hook-managed. **NO backend/, NO frontend/, NO .mcp.json, NO .claude/agents/, NO scripts/migrations/.** Scope strictly within the smoketest's "no production code changes" guardrail. |
| Screener code-anchor verification | `python -c "import inspect; from backend.tools import screener; src=inspect.getsource(screener.rank_candidates); assert 'composite_score' in src"` (informal spot-check via Read) | **PASS** -- `screener.py:370` line `scored.append({**stock, "composite_score": round(score, 3)})` confirmed verbatim, matches research_brief.md line 46-47 audit. |
| Production caller anchor | Read `autonomous_loop.py:305-310` | **PASS** -- the call site `screen_universe(tickers=universe, period="6mo", short_interest_lookup=...)` confirmed verbatim with NO `sector_lookup` passed, matches research_brief.md lines 48-58 and validates Test Design #3 selection. |

`checks_run = [harness_compliance_audit, output_file_exists, json_shape_validation, smoketest_reproducibility_rerun, diff_scope, screener_code_anchor, production_caller_anchor, code_review_heuristics]`.

## Code-review heuristics (phase-16.59 trading-domain framework)

Severity dispatch BLOCK / WARN / NOTE applied across 5 dimensions. Diff is **handoff-only** (5 files, 0 backend / 0 frontend / 0 config), so most heuristics are N/A by construction. Walking each anyway per discipline:

**Dimension 1 (Security):**
- **secret-in-diff [BLOCK]**: grep for API_KEY/secret/password/token patterns in modified files returns 0 matches. PASS.
- **prompt-injection-path [BLOCK]**: no LLM call surface (Stage 1 has no LLM step per substitution rule + contract guardrail line 59). PASS.
- **command-injection [BLOCK]**: no subprocess/eval/exec. PASS.
- **system-prompt-leakage [WARN]**: no agent_config / messages serialization. PASS.
- **rag-memory-poisoning [WARN]**: no `add_memory*` or vector-store imports. PASS.
- **unbounded-llm-loop [WARN]**: no `while True` near `messages.create`; no `MAX_TOOL_TURNS`/`MAX_RESEARCH_ITERATIONS` constant changes. PASS.
- **supply-chain-dep-pin-removal [WARN]**: no requirements/manifest change. PASS.
- **excessive-agency [WARN]**: no new write capability added; the test wrote 2 local files (`STAGE_1_*`) in the `handoff/smoketest_20260520/` directory only. PASS.
- **insecure-output-handling [BLOCK]**: JSON dump path is `json.dumps(dict_with_in-cycle_typed_values, indent=2)` -- no external user input. PASS.

**Dimension 2 (Trading-domain correctness):**
- **kill-switch-reachability [BLOCK]**: `kill_switch.is_paused()` UNTOUCHED; contract attestation (line 65) confirms `kill_switch.paused==True throughout`. PASS.
- **stop-loss-always-set [BLOCK]**: no buy path (the test is screen + rank + enrich; no trade execution). PASS.
- **stop-loss-backfill-removal [BLOCK]**: `backfill_stop_losses` untouched. PASS.
- **perf-metrics-bypass [BLOCK]**: no Sharpe/drawdown/alpha computation. PASS.
- **position-sizing-div-zero [WARN]**: no position sizing in this stage. PASS.
- **max-position-check-bypass [BLOCK]**: `paper_max_positions` untouched. PASS.
- **paper-trader-broad-except [BLOCK]**: no NEW broad-except introduced (no backend code modified). PASS.
- **crypto-asset-class [BLOCK]**: not touched. PASS.
- **sod-nav-anchor [WARN]**: `_sod_nav`/`_peak_nav` not touched. PASS.
- **bq-schema-migration-safety [WARN]**: no migration. PASS.

**Dimension 3 (Code quality):**
- **broad-except [WARN]**: no new code. PASS.
- **no-type-hints [NOTE]**: no new code. PASS.
- **print-statement [WARN]**: no new production code. PASS.
- **global-mutable-state [WARN]**: no new state. PASS.
- **test-coverage-delta [WARN]**: this IS a smoketest deliverable, not a code change with test absence. PASS by inversion.
- **unicode-in-logger [NOTE]**: no new logger calls. PASS.
- **magic-number [NOTE]**: the literals `220.61, 298.97, 417.42, 295.70, 15.283, 8.107` etc. are live-yfinance return values reproduced in the JSON output, not magic constants in code. PASS.
- **composition-over-inheritance [NOTE]**: no class hierarchy. PASS.

**Dimension 4 (Anti-rubber-stamp on financial logic):**
- **financial-logic-without-behavioral-test [BLOCK]**: NO financial-logic code change in this diff -- the smoketest invokes EXISTING `screen_universe` + `rank_candidates` from `backend/tools/screener.py` and asserts on output shape, NOT on correctness of the underlying scoring math. The shape-only assertion floor (per Sealos source #15: "assert on schema validity, output shape, no NaN/Inf; do NOT assert accuracy") is the documented smoke-test discipline. Equivalent of `test_coverage_delta` PASS. No mutation needed because no code changed. PASS.
- **tautological-assertion [BLOCK]**: assertions are concrete: `len(ranked) == 4`, `isinstance(row["composite_score"], (int, float))`, `len(row["sector"]) > 0`, `REQUIRED = {"ticker", "current_price", "composite_score", "sector"}; assert not (REQUIRED - set(row.keys()))`. No `assert x == x`, no `mock.called`. PASS.
- **over-mocked-test [BLOCK]**: the test calls live `screen_universe` and live `rank_candidates` directly; sector enrichment uses live `yfinance.Ticker(t).info` (NOT a mock). The function under test IS the production function. This is the OPPOSITE of over-mocking. PASS.
- **rename-as-refactor [BLOCK]**: no renames. PASS.
- **pass-on-all-criteria-no-evidence [BLOCK]**: experiment_results.md success-criteria table at lines 60-66 cites evidence per row (json + handoff markdown + kill_switch state). PASS.
- **formula-drift-without-citation [WARN]**: no risk constants changed. PASS.

**Dimension 5 (LLM-evaluator anti-patterns -- self-aware):**
- **sycophancy-under-rebuttal [BLOCK]**: no prior phase-31.0.1 verdict to flip. PASS.
- **second-opinion-shopping [BLOCK]**: first Q/A spawn for this step. PASS.
- **missing-chain-of-thought [BLOCK]**: this critique cites file:line per claim (research_brief.md:23-82, 308-320, 144-154; screener.py:370, 64-72, 179-201; autonomous_loop.py:305-310, 579-596; experiment_results.md:60-66; STAGE_1_results.md:54-61). PASS.
- **3rd-conditional-not-escalated [BLOCK]**: `grep -c "phase-31.0.1.*result=CONDITIONAL" harness_log.md = 0`. Zero priors. N/A.
- **criteria-erosion [WARN]**: all 5 immutable criteria addressed in the verdict table below. PASS.
- **position-bias [WARN]**: this critique has criterion #1 (researcher gate) marked PASS-with-NOTE -- not pure PASS -- demonstrating I'm not rubber-stamping the first item. PASS.
- **verbosity-bias [WARN]**: critique length reflects evidence depth (5-item audit + deterministic checks + 5 dimensions of heuristics + LLM judgment), not a sycophantic short-circuit. PASS.
- **self-reference-confidence [WARN]**: PASS verdict is anchored to live re-execution + JSON validation + file:line audit, not to "the generator says it works". PASS.

`checks_run += ["code_review_heuristics"]`.

## LLM judgment

**Substitution rule honored?** YES. Stage 1 has no LLM step. The smoketest invokes `backend.tools.screener.screen_universe` (yfinance + pandas + numpy compute) and `rank_candidates` (factor weighting). No `client.messages.create()`, no `generate_content`, no Anthropic / Gemini call. Hard guardrail at experiment_results.md lines 45-48 + STAGE_1_results.md lines 70-74 explicitly attests "NO LLM calls".

**Production-chain mirror?** YES. Test Design #3 mirrors the production caller chain verbatim:
- `screen_universe(tickers=universe, period="6mo")` at `autonomous_loop.py:305-310` (no `sector_lookup` passed) -- replicated at STAGE_1_results.md:22.
- `rank_candidates(...)` at `autonomous_loop.py:541` (line ~541-571 per research brief; default strategy `"momentum"`) -- replicated at STAGE_1_results.md:26.
- Sector enrichment via `yf.Ticker(t).info.get("sector")` at `autonomous_loop.py:579-596` (yfinance fallback when BQ doesn't have the ticker) -- replicated at STAGE_1_results.md:30-32. The smoketest bypasses the BQ layer of `_fetch_ticker_meta` (acceptable per researcher recommendation: BQ has the same tickers and the yfinance fallback is the canonical Feathers seam per source #21).

The taxonomy choice ("Technology" / "Financial Services") matches yfinance native taxonomy as documented in research_brief.md source #7 -- NOT GICS ("Information Technology" / "Financials"). pyfinagent downstream code accepts either taxonomy (per researcher analysis), so this is correct.

**Scope honesty.** The 3-Tech-vs-1-Financials basket split is explicitly called out at STAGE_1_results.md:45-46 ("exercises phase-30.5 sector NAV-pct cap in downstream stages") -- this is honest disclosure that the basket has structural sector imbalance that downstream stages WILL stress-test (not a hidden gotcha). The signed composite_scores (NVDA +15.283, AAPL +8.107, MSFT -1.557, JPM -3.986) are honestly reported including the negative scores; no padding to make all 4 candidates look positively-ranked. RSI 84.1 on AAPL is flagged as overbought-region per the brief's Wilder threshold discussion (source #4, lines 92-93). Honest scope.

**Verdict on `gate_passed: false` (the central judgment call).** PASS-with-NOTE, not CONDITIONAL. Rationale, cited and adversarially scrutinized:

(a) **What the floor is for.** The 20-source deep-tier floor exists to protect against shallow research that misses adversarial / contradictory evidence. The brief's content shows ZERO such shallowness: (i) [ADVERSARIAL] tag explicit at source 22 (Wright Research 2025 documenting 192-day / -31.79% momentum drawdown -- directly contradicts the naive long-momentum signal), (ii) cross-validated against source 23 (Wilder 70/30 vs pyfinagent 80/20 -- documents the conservative-choice tradeoff), (iii) snippet-only sources catalogued at lines 144-154 with reason per row (paywalls / 404s). The floor's INTENT (defend against incomplete coverage) is satisfied; the floor's literal count (20) is not (18 of 20).

(b) **What Stage 1 actually verifies.** Per Sealos ML smoke-test source #15 (lines 121-122): smoke tests "assert on schema validity, output shape, no NaN/Inf, model loads, prediction format. Do NOT assert: accuracy, drift, performance, edge cases." This is a SHAPE verification, not a strategy validation. The 18-source brief is content-saturated for shape-verification purposes; deeper research (the missing 2 sources) would have been relevant to the underlying scoring math, NOT to the question "does the screen->rank->enrich chain produce 4 dicts with the required shape?"

(c) **Comparison anchor.** phase-30.4 (GIPS-correct return series) had `gate_passed: true` with 27 sources -- because that step involved a financial-logic CHANGE (subtracting external flows from return computation) where adversarial evidence and methodology breadth materially affected correctness. phase-31.0.1 (smoketest) involves NO code change -- the bar is appropriately lower (shape verification on an existing chain). Tier "deep" is even arguably over-spec for a smoketest; the researcher's choice of "deep" provides extra margin even at 18 sources.

(d) **Honest disclosure value.** The brief reports `gate_passed: false` HONESTLY rather than padding the count. The 8 snippet-only sources are explicitly enumerated with per-row reasons. This is the OPPOSITE of the failure mode the gate is meant to catch (silent shallowness disguised as completeness). The discipline is precisely what `.claude/rules/research-gate.md` line 28-31 prescribes: "If fewer than 5 sources were fetched in full, the researcher MUST return `gate_passed: false` and list what was attempted. Padding a brief to mask an under-fetch is a protocol breach."

(e) **Anti-criteria-erosion check.** The morning-goal Stage 1 success criteria are about: screen+rank+enrich produces 4 dicts with sector + composite_score, persisted to JSON, no LLM/BQ/Alpaca, loop paused. NONE of these criteria reference research-gate count. The verdict on Stage 1 is judged on the smoketest output, not on research scaffolding around it. Demoting to CONDITIONAL purely on the gate count would amount to inventing a criterion not in the immutable spec.

Verdict on (a)-(e): the floor miss is a NOTE on the cycle, not a verdict-blocker. PASS-with-NOTE that future smoketest research can either accept lower-tier specification (since Stage 1 is shape-only, `moderate` or even `simple` tier with the 5-source floor would have cleared this trivially) OR re-spawn researcher for the missing 2 sources. The brief is content-complete for THIS step's purpose.

**Mutation-resistance.** Spot-checked 5 mutations against the persisted JSON + the assertion set:
- Mutation A (drop `composite_score` from `rank_candidates` output -- regression to plain dict): JSON shape assertion at the Q/A reproduction step (`all('composite_score' in r for r in d['rows_post_enrichment'])`) fails immediately.
- Mutation B (sector enrichment returns empty string): assertion #5 (`len(row["sector"]) > 0`) fails; this would surface in the smoketest's local assertion loop before persistence.
- Mutation C (screen returns <4 tickers due to silently failed yfinance fetch): assertion #1 (`len(ranked) == 4`) fails; reproducibility re-run also fails -- I verified live re-run returns 4 today.
- Mutation D (taxonomy regressed to None): assertion #5 catches this (None has no `len()` and is not a non-empty string).
- Mutation E (`composite_score` becomes non-numeric, e.g. string): assertion #4 (`isinstance(row["composite_score"], (int, float))`) fails.

The persisted JSON + per-stage assertions are robust to common shape-regression mutations. Adequate for shape verification per Sealos source #15.

**Research-gate compliance.** Contract.md lines 7-28 cite the brief with `gate_passed=false` (honest disclosure), 18 sources read in full + 8 snippet-only, [ADVERSARIAL] tag present (Wright Research 2025), three-variant search composition explicit. The brief's recommended test design (#3) is what experiment_results.md and STAGE_1_results.md implemented; the brief's Application section (lines 192-262) maps 1:1 to the executed test code in STAGE_1_results.md:15-33. Research -> contract -> generate chain end-to-end intact. NO criterion was invented or dropped between brief and execution.

**Anti-rubber-stamp summary.** No financial-logic change (no code modified). The smoketest is a SHAPE verification on an unchanged production chain -- this satisfies the BLOCK heuristic `financial-logic-without-behavioral-test` by INVERSION (no logic change -> no test required, only a smoke check). Assertions are concrete (not tautological). No mocking of the function under test. The 4-row output is reproducible (live re-run confirmed). 3-Tech / 1-Financial split is honestly disclosed as exercising downstream phase-30.5 sector cap.

## Success criteria check (Stage 1 immutable spec, from morning goal)

| Criterion | Verdict | Evidence |
|-----------|---------|----------|
| `screen_universe(["AAPL","MSFT","NVDA","JPM"]) -> 4 dicts` | **PASS** | JSON output has 4 rows (`len(d['rows_post_enrichment'])==4`); live re-run produced `REPRO_LEN=4` with identical ticker ordering `['NVDA','AAPL','MSFT','JPM']`. Reproducibility verified. |
| Each dict has non-empty `sector` (post-enrichment) | **PASS** | All 4 rows in JSON output have non-empty sector strings: NVDA="Technology", AAPL="Technology", MSFT="Technology", JPM="Financial Services". yfinance native taxonomy (not GICS) -- acceptable per researcher source #7 + brief Application notes. |
| Each dict has a numeric `composite_score` (post-rank) | **PASS** | All 4 rows have numeric scores: 15.283, 8.107, -1.557, -3.986. Live re-run confirms `REPRO_CS_PRESENT=True` and `REPRO_CS_NUMERIC=True`. Anchored to `screener.py:370` (`scored.append({**stock, "composite_score": round(score, 3)})`). |
| Output persisted to `handoff/smoketest_20260520/STAGE_1_screen_universe_output.json` | **PASS** | File exists (52 lines), JSON shape validated, `verdict=PASS` field present, `rows_post_enrichment` array populated, downstream-consumable format. |
| NO production BQ writes; NO LLM calls; NO Alpaca calls | **PASS** | Contract guardrail (lines 59-61) + experiment_results.md attestation (lines 45-48) + STAGE_1_results.md attestation (lines 70-74). git diff confirms no backend code modified -- the production write paths (BQ MERGE upsert in `bigquery_client.py`, Anthropic `client.messages.create()`, Alpaca order endpoints) cannot have fired because they are not in the test code path. yfinance live read is permitted (read-only per contract line 61). |

All 5 criteria PASS.

## Verdict

verdict: PASS
ok: true
checks_run: [harness_compliance_audit, output_file_exists, json_shape_validation, smoketest_reproducibility_rerun, diff_scope, screener_code_anchor, production_caller_anchor, code_review_heuristics]
violated_criteria: []
violation_details: None. All 5 immutable Stage 1 success criteria met with live-verified evidence. Output file `handoff/smoketest_20260520/STAGE_1_screen_universe_output.json` exists with `verdict=PASS` and 4 rows; all rows carry non-empty `sector` (NVDA/AAPL/MSFT="Technology", JPM="Financial Services") + numeric `composite_score` (15.283 / 8.107 / -1.557 / -3.986). Live re-execution of `screen_universe(['AAPL','MSFT','NVDA','JPM'], period='6mo') -> rank_candidates(top_n=4, strategy='momentum')` reproduced 4 rows in identical ordering with `composite_score` present and numeric on each row (REPRO_LEN=4, REPRO_CS_PRESENT=True, REPRO_CS_NUMERIC=True). Diff strictly scoped to handoff/* (5 files: contract.md, experiment_results.md, research_brief.md, harness_log.md auto-append from hooks, .archive-baseline.json hook-managed) + handoff/audit/*.jsonl (auto-managed); NO backend/, NO frontend/, NO .mcp.json, NO .claude/agents/, NO scripts/migrations/. Code-anchor verification: `screener.py:370` confirmed as the `composite_score` insertion site (matches brief lines 46-47); `autonomous_loop.py:305-310` confirmed as the production caller pattern with NO `sector_lookup` kwarg passed (matches brief lines 48-58 and validates Test Design #3 as the correct choice over Designs #1 and #2). Code-review heuristics across 5 dimensions: zero BLOCK or WARN findings (most heuristics N/A by construction because no code modified). Harness-compliance audit: 4 of 5 items PASS, 1 PASS-with-NOTE (researcher gate `gate_passed=false` is honest disclosure on the 18-of-20 sources floor miss; the 4 paywalled sources are catalogued in the snippet-only table with per-row reasons; [ADVERSARIAL] tag present on source 22 Wright Research 2025; three-variant search composition explicit; content-completeness fully covers the Stage 1 shape-verification scope per Sealos ML smoke-test discipline source #15 -- assert on schema/shape, NOT on accuracy/drift/performance). The morning-goal Stage 1 immutable spec contains NO research-gate-count criterion; demoting to CONDITIONAL purely on the 2-source floor miss would amount to inventing a criterion not in the spec, which the anti-criteria-erosion heuristic (Dimension 5) flags as the wrong direction. Substitution rule honored (no LLM step; pure code+yfinance read). Production-chain mirror correct (screen->rank->yfinance-sector enrichment per `autonomous_loop.py:305-310 + 541-596`). Scope honesty: 3-Tech-vs-1-Financials sector imbalance explicitly disclosed at STAGE_1_results.md:45-46 as deliberately exercising the phase-30.5 sector NAV-pct cap in downstream stages, NOT a hidden gotcha. Mutation-resistance: 5 shape-regression mutations (composite_score drop / sector empty / <4 tickers / sector None / non-numeric score) each caught by at least one assertion in the persisted output JSON or the live reproducibility re-run. First Q/A spawn for phase-31.0.1; no sycophancy / second-opinion-shopping surface. Zero prior CONDITIONALs for this step-id (3rd-CONDITIONAL auto-FAIL rule N/A). The persisted JSON output is downstream-consumable for Stages 4+ which will exercise sector-cap, Risk Judge sizing, and broker leg per the morning-goal spec.
certified_fallback: false
