# Research Brief -- step 61.2 (Decision-input integrity)

Tier: **complex**. Spawned as the mandatory research gate before contract.md.
Access date for all external sources: **2026-08-07**.
Status: IN PROGRESS (write-first; grown incrementally).

Central job (per caller): **per-sub-item liveness triage** of the six 61.2
sub-items -- the step was written ~2026-06, has ZERO immutable criteria and a
broad `-k` verification command. For each sub-item: is the defect still live,
was it fixed later (commit), or is it owned by a sharper queued step (72.0.1 /
72.0.2)?

## Six sub-items under triage

| # | Sub-item (verbatim from step) |
|---|-------------------------------|
| A | never persist synthetic 0.00/HOLD on synthesis failure (claimed "actively firing") |
| B | claude_code timeout 120 -> >=150s |
| C | company_name fallback for autonomous full-path rows |
| D | meta-scorer fallback rank-normalization + unavailability alert + root-cause |
| E | dead `signal_downgrade` exit path |
| F | RiskJudge portfolio context in advisory mode |

## HEADLINE: two caller premises REFUTED before triage begins

**Premise 1 REFUTED -- "it has ZERO immutable criteria".**
`.claude/masterplan.json` step 61.2 carries a full
`verification.success_criteria` array with **SIX** entries (one per sub-item,
A..F, in order) plus a `verification.live_check` string. They are immutable and
must be copied verbatim into the contract. The contract must NOT self-impose a
new criteria set -- that would be amending immutable criteria. (Evidence: the
masterplan JSON dump in the audit appendix below.)

**Premise 2 REFUTED -- "the step is a 2-month-old un-started defect".**
61.2 was **already BUILT** on 2026-07-08 in commit
`6186784c9a35cd7724c7d5ebd5e5bc034d8573db`
("phase-61.2: decision-input integrity DARK BUILD -- synthesis-error routing +
degraded NULL rows, retry-on-empty, 150s timeout, company_name fallback,
meta-scorer rank-normalization + streak WARN, signal_downgrade revival (2 flags
OFF); 33 new tests"). 20 source files + a 459-line test module were shipped.
The step is still `status: pending` because Q/A returned **CONDITIONAL** on
cycle 74 (commit `354eb6b4`, "designed intermediate state, live legs
deploy-gated"). So 61.2 is not a stale defect step -- it is a **DARK BUILD
awaiting flag promotion + a live_check**, with a post-build phase-76 audit
finding still open.

Consequence for the drain goal: the honest question is NOT "rescope or drop
because stale", it is "**what is left between the dark build and a closeable
step**". That is a much smaller, sharper contract.

## STATUS OF THE VERIFICATION COMMAND (measured, not asserted)

Ran verbatim, 2026-08-07:

```
python -m pytest backend/tests -k 'synthesis or persist or downgrade or meta_scorer or 61_2' -q
...
FAILED backend/tests/test_phase_50_2_multicurrency.py::test_krw_buy_row_persists_usd_total_value
FAILED backend/tests/test_phase_61_2_decision_integrity.py::TestSignalDowngrade::test_pos_row_stores_verdict_flag_on
FAILED backend/tests/test_phase_61_2_decision_integrity.py::TestSignalDowngrade::test_pos_row_stores_reason_flag_off
FAILED backend/tests/test_phase_83_0_news_corpus_persistence.py::test_c5_no_alphavantage_import_chain
4 failed, 67 passed, 2829 deselected, 1 warning in 19.22s
```

**The immutable verification command is RED today.** 71 tests collected.
Two of the four failures are OUTSIDE 61.2's blast radius (`test_phase_50_2`
matches on `persist`; `test_phase_83_0` matches on `persist`) -- the classic
broad-`-k` trap recorded in auto-memory
`feedback_immutable_criteria_must_be_green_able` (a step that ends in a checker
already red for unrelated reasons is structurally uncloseable). Two failures
ARE 61.2's own tests and are real regressions to be diagnosed (below).

## LIVE BQ EVIDENCE (read-only, bounded; run 2026-08-07)

`financial_reports.analysis_results`, last 40 days:

```
SELECT JSON_VALUE(full_report_json,'$._path') path, recommendation,
       JSON_VALUE(full_report_json,'$.final_synthesis.error') err,
       COUNT(*) n, MAX(analysis_date) last_seen
FROM `sunny-might-477607-p8.financial_reports.analysis_results`
WHERE analysis_date >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 40 DAY)
  AND final_score = 0.0 GROUP BY 1,2,3
-->
{'path':'full','recommendation':'HOLD','err':'Failed to parse final report.',
 'n':142,'last_seen':2026-08-06 19:20:56 UTC}

-- denominator
{'path':'full','n':156,'zero':142}      <-- 91% of full-path rows fabricated
{'path':'lite','n':14, 'zero':0}
```

Per-day (`zero_score`/`zero_hold`/`syn_err`/`null_name`/`n`):

| date | 0.0 score | 0.0+HOLD | synth error | NULL company_name | rows |
|---|---|---|---|---|---|
| 2026-08-06 | 2 | 2 | 2 | **0** | 5 |
| 2026-08-05 | 3 | 3 | 3 | **0** | 6 |
| 2026-08-04 | 4 | 4 | 4 | **0** | 5 |
| ... | | | | **0** | |
| 2026-07-09 | 7 | 7 | 7 | **0** | 10 |
| 2026-07-08 | 3 | 3 | 3 | **4** | 5 |
| 2026-07-07 | 5 | 5 | 5 | **5** | 5 |
| 2026-07-03 | 5 | 5 | 5 | **5** | 5 |

**Two decisive reads:**

1. **Sub-item A is ACTIVELY FIRING, daily, today.** 142 fabricated
   0.0/`HOLD` rows in 40 days, last one **yesterday 2026-08-06 19:20:56Z**.
   `zero_score == zero_hold == syn_err` EXACTLY every single day: the mapping
   from "synthesis failed" to "persisted 0.00/HOLD" is 1:1 with no leakage.
   Every one carries the SAME error string, `"Failed to parse final report."`
   (`backend/agents/orchestrator.py:1685`).
2. **Sub-item C is ALREADY FIXED AND LIVE-PROVEN.** `null_name` is 5/5/2/4 on
   2026-07-03..07-08 and **0 on every single day from 2026-07-09 to
   2026-08-06**. The ungated criterion-3 fix
   (`autonomous_loop.py:2926-2934`) went live at the 2026-07-08/09 backend
   restart and has held for a month. **This retires blocker #1 of the Q/A
   CONDITIONAL** -- the evidence the Q/A said could not exist yet now exists
   in BQ.

### Which writer produced the 142 rows (matters for scope)

`recommendation` is **`'HOLD'` (upper case)**. Only
`autonomous_loop.py:1942` emits upper-case `rec.get("action", "HOLD")`. The
two MANUAL save_report sites (`backend/api/analysis.py:211`,
`backend/tasks/analysis.py:212`) emit `rec_obj.get("action", "N/A")` -> would
appear as `'N/A'`. The 40-day group-by returned **exactly one group** -- no
`'N/A'` rows exist. So:

- All 142 fabricated rows are **autonomous full-path**, written by
  `_persist_analysis`. The 61.2 dark-build guard at
  `autonomous_loop.py:1932-1940` (`synthesis.get("error")`) **does** cover
  them -- flipping `paper_synthesis_integrity_enabled` ON routes all 142 to
  the lite fallback instead.
- The phase-76 `audit_note` in masterplan 61.2 is therefore **half right**:
  the manual `save_report` sites genuinely have NO integrity guard (latent,
  structural), but they are **not** the writers of the observed 0.00/HOLD
  population. The note's conclusion "or the 0.0/HOLD rows persist even after
  the LLM rail is restored" is **not supported by the measured data** -- the
  observed rows die when the autonomous flag flips. Treat the manual-path
  gap as a real but NOT-CURRENTLY-FIRING follow-on, not a 61.2 blocker.

### Root-cause note (bigger than 61.2's framing)

91% of full-path analyses currently fail with `"Failed to parse final
report."` -- the synthesis JSON is not parseable on the live rail. 61.2 does
not fix that; it stops the failure from being laundered into a tradeable
`HOLD`. Flipping the flag converts 142 fabricated rows into 142 REAL
lite-scored rows -- a material money-engine behaviour change, which is
exactly why it is dark and why the live_check matters.

## Per-sub-item triage

### A -- synthetic 0.00/HOLD on synthesis failure
**LIVE + BUILT-DARK.** Defect firing daily (142 rows/40d, last 2026-08-06).
Fix shipped in `6186784c`, gated behind `paper_synthesis_integrity_enabled`
(default `False`, `backend/config/settings.py:198`).
Fabrication chain, verified end-to-end:
`orchestrator.py:1683-1687` returns `{"error": "Failed to parse final
report."}` -> `orchestrator.py:2280` unconditionally stamps
`final_json["final_weighted_score"] = self.compute_weighted_score({})` =
**0.0** on that very error dict -> `orchestrator.py:2320`
`report["final_synthesis"] = final_json` -> `autonomous_loop.py:1942`
`rec.get("action", "HOLD")` -> `_persist_analysis` writes the row.
`orchestrator.py:2280` is the precise laundering line: it makes an error dict
indistinguishable from a scored one.
Guard (flag ON): `autonomous_loop.py:1932-1940` raises
`SynthesisDegradedError` inside the existing `try`, routing to the lite
fallback at `:1985-1998`; honest-degraded NULL row at `:2001-2020` +
`_persist_analysis` NULL coercion at `autonomous_loop.py:2935-2936`
(`final_score=None if _degraded`).
**In scope for 61.2: YES -- promotion + live_check only, no new code.**

### B -- claude_code timeout 120 -> >=150s
**ALREADY DONE AND LIVE (ungated).**
`backend/config/settings.py:186-191` `claude_code_timeout_s: int = Field(150,
ge=60, le=600)` with the description literally citing "phase-61.2 (criterion
2)". `backend/agents/claude_code_client.py:591` class attr
`recommended_step_timeout = 150`; `:593` `def __init__(self, model_name,
timeout_s: int = 150)`; `:600` `self.recommended_step_timeout = timeout_s +
30` (so the step budget always exceeds the subprocess timeout -- the race the
phase-60.1 comment warns about). Threaded from settings via
`llm_client.py make_client` (Q/A verified this by code read; still no unit
test on the threading -- registered test-debt).
**In scope: NO code work. Criterion 2 is already satisfiable by inspection.**

### C -- company_name fallback on autonomous full-path rows
**FIXED + LIVE-PROVEN (ungated).** `autonomous_loop.py:2926-2934`:
`company_name=(market_data.get("name") or (full_report.get("quant") or
{}).get("company_name") or None)` -- comment reads "phase-61.2 (criterion 3,
ungated pure fix)". Live BQ: **zero NULL company_name on every day from
2026-07-09 through 2026-08-06** vs 4-5/day before. This is the live evidence
the Q/A CONDITIONAL was waiting for.
**In scope: NO code work -- capture the BQ proof in live_check_61.2.md.**

### D -- meta-scorer fallback rank-normalization + alert + root-cause
**BUILT-DARK; and the 72.0.1 overlap is now MOSTLY OBSOLETE.**
- Rank-normalization: `meta_scorer.py:170-177` `_fallback_convictions`
  dispatcher (flag ON -> `_rank_normalized_convictions`, OFF ->
  legacy per-candidate `_fallback_conviction` clamp, byte-identical);
  tail path at `meta_scorer.py:296-300` normalizes over the **full**
  `head + tail` set so head and tail share one scale.
- Root cause: diagnosed and independently reproduced by the Q/A against live
  `llm_call_log` -- last genuine direct-API Anthropic success **2026-05-17**;
  the "06-03..06-10 window" was a sample of one continuous credit-death span;
  106 of the apparent successes were test-fixture pollution from
  `backend/tests/test_observability.py:230` writing to the PROD table.
- **72.0.1 overlap: the rail bypass it was written to fix NO LONGER EXISTS.**
  `meta_scorer.py:226-229` comment: *"phase-78.1: route through make_client so
  PAPER_USE_CLAUDE_CODE_ROUTE governs this call. It previously constructed
  ClaudeClient DIRECTLY and so could never see the rail -- the phase-72
  rail-bypass class implicated in the 97%-cash run."*
  `meta_scorer.py:242` now calls `make_client(...)`. `grep ClaudeClient
  backend/services/meta_scorer.py` returns **no construction site**. So
  72.0.1's premise ("meta_scorer.py:220-225 constructs ClaudeClient with
  anthropic_api_key directly") is REFUTED against the current tree; 72.0.1
  should be re-scoped or closed by whoever owns it. `anthropic_api_key` is
  still read at `meta_scorer.py:203` but only as a *presence check* for the
  no-key-and-no-rail early fallback at `:210-218`, not to construct a client.
- 72.0.2 (standard-tier fail-forward on rail-dead, `llm_client.py`
  provider-order seam) is **orthogonal**: it changes WHICH MODEL serves a
  standard-tier call when the rail is dead; 61.2-A changes WHAT IS PERSISTED
  when synthesis fails for any reason. They compose (72.0.2 reduces the
  frequency of the failure; 61.2 makes the residual failures honest) and
  neither subsumes the other.
- Only genuinely unresolved 61.2-D leg: the **"WARN after 2 consecutive
  all-fallback cycles"** alert. Built (`meta_scorer.py` streak state file per
  experiment_results) but flag-gated OFF and never live-fired.
**In scope: promotion + live_check; record 72.0.1 as superseded by 78.1.**

### E -- dead `signal_downgrade` exit path
**STILL STRUCTURALLY DEAD IN PRODUCTION; fix BUILT-DARK; and its TESTS ARE
NOW RED.**
- The rule: `portfolio_manager.py:156` emits `action="SELL",
  reason="signal_downgrade"`, reachable only when the position's stored
  `recommendation` fails the `_BUY_RECS` match (`portfolio_manager.py:50`).
  `paper_trader.py:443-450` stores the ANALYSIS recommendation only when
  `paper_position_recommendation_fix_enabled` is ON (`settings.py:202`,
  default `False`) -- otherwise the legacy trade-mechanism reason
  (`new_position` / `swap_...`). Flag is OFF -> **path still dead today**.
- Unsafe-combination guard exists: `portfolio_manager.py:114-121` logs a
  WARNING if this flag is ON while `paper_synthesis_integrity_enabled` is OFF
  (a fabricated HOLD would otherwise SELL a healthy position). **This makes
  E strictly downstream of A: E must never be promoted alone.**
- **NEW REGRESSION (found this session):** the two flag tests
  `TestSignalDowngrade::test_pos_row_stores_verdict_flag_on` and
  `::test_pos_row_stores_reason_flag_off` FAIL today. Cause is not 61.2:
  **phase-36.13 commit `3227347a` (2026-07-26)** added a kill-switch gate at
  `paper_trader.py:276-288` (`_kill_switch_refusal_for_buy`,
  `paper_trader.py:177-207`) that `execute_buy` now hits BEFORE building the
  position row. The helper falls back to the module singleton
  `kill_switch.get_state()` when `_injected_ks_state` is None, and that
  singleton replays the real on-disk audit (`kill_switch.py:254-272`), which
  is currently `paused=True, pause_reason='manual'`. The 61.2 tests, written
  2026-07-08, do not inject a kill-switch state, so **their outcome depends
  on the operator's live pause state** -- they will pass or fail depending on
  whether the book happens to be paused. That is a test-isolation defect, and
  it is the reason the immutable verification command is red.
**In scope: YES -- repair the two tests by injecting an unpaused
`_injected_ks_state` (a test fix, not a production change).**

### F -- RiskJudge portfolio context in advisory mode
**BUILT-DARK.** `autonomous_loop.py:1045-1052`:
```python
_rj_portfolio_ctx = ""
if getattr(settings, "paper_risk_judge_reject_binding", False) or getattr(
    settings, "paper_synthesis_integrity_enabled", False
):
    _rj_portfolio_ctx = _build_portfolio_sector_context(positions)
```
consumed at `autonomous_loop.py:1137`. Both flags OFF today -> context is
`""` -> **the judge still gets no portfolio context**, i.e. the defect is
live. Flag ON delivers advisory context without making rejects binding
(`paper_risk_judge_reject_binding` stays OFF; the binding legs at `:2147` and
`:2161` are untouched).
**In scope: promotion + live_check only.**

---

## External research

Search-query variants actually run (3-variant discipline, per
`.claude/rules/research-gate.md`):
- **frontier 2026**: "fail-closed vs fail-open degraded output LLM pipeline
  never persist fabricated default 2026"; "LLM agent orchestrator timeout
  tuning long-running inference retry budget 2026"
- **last-2-year**: "rank normalization percentile fusion heterogeneous
  scorers missing scores 2025"
- **year-less canonical**: "graceful degradation null object anti-pattern
  default value silent data corruption"; "Google SRE handling overload
  timeout budget deadline propagation cascading failures"

### Read in full (7; >=5 floor met)

| # | URL | Accessed | Kind | Fetched how | Key finding |
|---|-----|----------|------|-------------|-------------|
| 1 | https://docs.aws.amazon.com/wellarchitected/latest/reliability-pillar/rel_mitigate_interaction_failure_graceful_degradation.html | 2026-08-07 | official doc | WebFetch | Names as a **common anti-pattern**: *"Creating an inconsistent state when a transaction partially fails."* And: *"the pathways taken in case of component failure need to be tested and should be significantly simpler than the primary pathway. Generally, fallback strategies should be avoided."* |
| 2 | https://sre.google/sre-book/addressing-cascading-failures/ | 2026-08-07 | official doc (canonical) | WebFetch | *"It's usually wise to set a deadline... High deadlines can result in resource consumption in higher levels of the stack when lower levels of the stack are having problems. Short deadlines can cause some more expensive requests to fail consistently."* Deadline propagation: one absolute deadline set high in the stack, each hop **reduces** it. *"Limit retries per request."* + server-wide retry budget. |
| 3 | https://arxiv.org/html/2603.28886v1 | 2026-08-07 | preprint | WebFetch (`/html/`) | Percentile-rank / PIT = *"the empirical CDF, mapping scores to [0,1] uniform (equivalently, the 1D optimal transport map)"*. *"min-max normalization preserves the power-law spike in PPR scores, producing unequal marginals"*; PIT 53.8% vs min-max 52.7% LastHop@5, *"min-max is directionally worse than PIT on both splits"*. |
| 4 | https://zylos.ai/research/2026-02-20-graceful-degradation-ai-agent-systems/ | 2026-08-07 | industry research (2026-02-20) | WebFetch | *"Always be explicit about degraded state... (1) Acknowledge what is unavailable (2) Explain what you can still do (3) Annotate outputs with appropriate uncertainty."* Contrasts with *"silent degradation where users may not realize they are receiving inferior outputs."* Recommends a `model_quality_degradation` WARN alert at a 20% drop vs baseline. |
| 5 | https://arxiv.org/html/2606.01416v1 | 2026-08-07 | preprint (2026) | WebFetch (`/html/`) | Explicit recovery **budgets** per failure class rather than unbounded retries; on budget exhaustion *"return degraded response with uncertainty or missing-dependency notice"* rather than an unsupported answer. Verifier-guided self-healing reaches a **0.0% silent-failure rate** on wrong-but-plausible outputs; 97.3% vs 86.7% (retry-only) under high fault intensity. |
| 6 | https://opensearch.org/blog/introducing-reciprocal-rank-fusion-hybrid-search/ | 2026-08-07 | official doc | WebFetch | *"variations in score distributions can lead to unbalanced rankings. One method's scoring pattern may dominate."* *"Min-max and L2 normalization are sensitive to outliers."* RRF `1/(k+rank)`, k=60; missing items *"default to a score of 0.0, but this may not be optimal."* RRF trades 3.86% NDCG@10 for latency. |
| 7 | https://aipatternbook.com/fail-fast-and-loud | 2026-08-07 | practitioner pattern catalogue | WebFetch | *"Most damage in software happens not when something breaks, but when something breaks and execution continues."* Names exactly our defect: *"Functions returning plausible defaults instead of signaling failure."* Remedy: *"When something can't be done, throw an exception, return an explicit error, or panic."* |

### Identified but snippet-only (does NOT count toward the gate)

| URL | Kind | Why not fetched in full |
|---|---|---|
| https://aws.amazon.com/builders-library/avoiding-fallback-in-distributed-systems/ | official doc | **Attempted, FAILED.** 301 to `builder.aws.com`; the redirect target returned only the "AWS Builder Center" shell (JS-rendered). Same class as auto-memory `feedback_gcloud_docs_fetch`. Cited indirectly via source #1, which links it as the basis for "fallback strategies should be avoided". |
| https://builder.aws.com/content/3EuS9Sakq7L3VLQIF3qzfMfke1Y/avoiding-fallback-in-distributed-systems | official doc | Redirect target; empty body (see above) |
| https://martinfowler.com/bliki/CircuitBreaker.html | canonical blog | Already read in full and adopted by the ORIGINAL 61.2 brief (2026-07-08); not re-read |
| https://aws.amazon.com/builders-library/timeouts-retries-and-backoff-with-jitter/ | official doc | Same; full-jitter backoff already implemented at `orchestrator.py:877` |
| https://ieeexplore.ieee.org/document/7225284/ | peer-reviewed | Paywalled |
| https://www.emergentmind.com/topics/reciprocal-rank-fusion | aggregator | Superseded by #6 |
| https://apxml.com/courses/advanced-vector-search-llms/chapter-3-hybrid-search-approaches/rrf-fusion-algorithms | course | Superseded by #6 |
| https://arxiv.org/pdf/2409.01357 | preprint | Adjacent (legal-domain hybrid retrieval); no new mechanism |
| https://arxiv.org/pdf/2605.06110 | preprint | Constraint-driven resource allocation; budget angle covered by #5 |
| https://arxiv.org/html/2603.18897v1 | preprint | Speculative tool execution; off-topic |
| https://futureagi.com/blog/evaluating-llm-structured-output-modes-2026/ | blog | Notable snippet: schema-constrained decoding **collapsed to a safe default** on ~11% of tickets -- same fabrication class, but blog-tier |
| https://mixroute.ai/blog/handle-llm-api-failures/ | blog | Vendor content |
| https://devsecopsschool.com/blog/fail-closed/ | blog | Definitional only |
| https://nhimg.org/glossary/fail-closed/ | glossary | Definitional only |
| https://www.buildmvpfast.com/blog/building-with-unreliable-ai-error-handling-fallback-strategies-2026 | blog | Low tier |
| https://web-alert.io/blog/graceful-degradation-designing-resilient-systems | blog | Good "silent failure" framing, blog-tier |
| https://dev.to/young_gao/graceful-degradation-4b5p | community | Lowest tier |
| https://risknowlogy.com/articles/detail/17309/ | industry (IEC 61508) | Degraded-mode safety framing; overlaps auto-memory `project_kill_switch_36_9_armed_semantics` |
| https://abdulahd1996.medium.com/understanding-the-null-object-pattern-49d201d453d6 | community | Null Object pattern background |
| https://sre.google/sre-book/service-best-practices/ | official doc | Superseded by #2 |
| https://sre.google/sre-book/table-of-contents/ | official doc | Index page |
| https://highscalability.com/google-addressing-cascading-failures/ | blog | Summary of #2 |
| https://danluu.com/google-sre-book/ | blog | Commentary on #2 |
| https://www.spheron.network/blog/ai-agent-workflow-orchestration-temporal-inngest-restate-gpu-cloud/ | blog | Temporal/Inngest timeout config; useful but not adopted |
| https://www.truefoundry.com/blog/multi-agent-orchestration-frameworks | vendor blog | Framework survey |
| https://aisecurityguard.io/reports/secrets-of-llm-whisperer/8_retry_cost | industry report | Retry-cost amplification; 7.1x per-step error-rate claim unverified |
| https://arxiv.org/pdf/2605.19338 | preprint | Meta-strategic supervision; off-topic |
| https://arxiv.org/pdf/2605.00060 | preprint | Drilling-domain agent; off-topic |
| https://arxiv.org/pdf/2604.09591 | preprint | "Simplicity Scales"; off-topic |
| https://letsdatascience.com/blog/open-source-vs-closed-llms-choosing-the-right-model-in-2026 | blog | Irrelevant (query homonym on "closed") |

**URLs collected: 38 unique (7 read in full + 31 snippet-only).**

### Recency scan (2024-2026) -- performed

Searched the 2024-2026 window on all three topics. **Result: 3 new findings
that COMPLEMENT (do not supersede) the canonical sources.**

1. **arXiv:2606.01416 (2026)** is the strongest new result and it directly
   validates 61.2's design: bounded per-class recovery budgets + an explicit
   degraded marker on exhaustion, achieving a **0.0% silent-failure rate**.
   pyfinagent's `claude_code_empty_retry_max=2` + the `_degraded` NULL row are
   the same two mechanisms. This is *newer* evidence for a design already
   shipped in `6186784c` -- it strengthens the promotion case rather than
   changing it.
2. **arXiv:2603.28886 (2026)** supplies the missing quantitative backing for
   criterion 4's percentile-rank choice: PIT beats min-max on held-out data
   and min-max *preserves* the power-law spike -- exactly the saturation
   pathology the legacy `_fallback_conviction` clamp produced (constant 10s).
3. **Zylos 2026-02-20** supplies the alerting shape (a WARN on degraded-mode
   entry, not just on hard failure) that criterion 4's "2 consecutive
   all-fallback cycles" WARN implements.

**No 2024-2026 source argues FOR persisting a fabricated neutral verdict.**
The nearest thing to a counter-position is Google SRE's *"it is better to
allow some user-visible errors or lower-quality results to slip through than
try to fully serve every request"* -- but that endorses **lower-quality**
results, not **fabricated** ones, and pyfinagent's lite fallback is exactly
the "lower-quality real result" SRE means.

### Consensus vs debate

**Consensus (7/7 sources):** a failed component must not return a
plausible-looking default. AWS calls untested fallback paths an anti-pattern;
Google SRE endorses degraded-but-real results; Zylos and arXiv:2606.01416
both demand an explicit degraded/uncertainty marker; the Fail-Fast-and-Loud
catalogue names "functions returning plausible defaults instead of signaling
failure" as the core harm.

**Debate:** *how far* to fail forward. AWS says "generally, fallback
strategies should be avoided" (prefer making the primary path robust);
Google SRE and arXiv:2606.01416 endorse a **tested, simpler** fallback. 61.2
sits on the SRE/2606.01416 side -- but note AWS's caveat is load-bearing
here: the lite fallback path is currently exercised only ~9% of the time
(14 of 156 rows), so promoting the flag makes a rarely-exercised path carry
91% of traffic. That is the single biggest promotion risk and the live_check
must watch it.

Second debate: **rank normalization vs RRF.** Source #6 shows RRF is more
robust to missing scores and outliers than any score normalization, at a
3.86% NDCG cost. 61.2 chose percentile-rank (PIT). Source #3 justifies PIT
over min-max, but neither compares PIT to RRF on this shape. Not a blocker
-- the meta-scorer's tail is *ranked within one homogeneous composite*, not
fused across heterogeneous systems -- but it is an honest open question.

### Pitfalls (from literature), mapped to 61.2

1. **Untested fallback becomes the primary path** (AWS #1). Mitigation: the
   live_check must count lite-fallback rows post-promotion.
2. **Silent degraded mode** (Zylos #4, web-alert). Mitigation: the 60.1
   fallback-rate alarm + criterion-4 streak WARN must both be confirmed
   firing, not merely present.
3. **Retry amplification** (Google SRE #2). Mitigation: budget is bounded at
   `1 + claude_code_empty_retry_max` = 3 total, matching SRE's
   "limit retries per request"; `rail_guard_skipped` empties are structurally
   excluded (`orchestrator.py:871-874`) per Fowler's open-breaker rule.
4. **Deadline inversion** (Google SRE #2 -- each hop must *reduce* the
   deadline). pyfinagent inverts this deliberately: `recommended_step_timeout
   = timeout_s + 30` (`claude_code_client.py:600`) makes the OUTER budget
   LARGER so the inner subprocess timeout fires first and is retryable. That
   is a considered deviation, not a bug -- but the contract should state it,
   because it reads as an SRE violation to a fresh auditor.

## Application to pyfinagent

| External finding | pyfinagent anchor |
|---|---|
| "Functions returning plausible defaults instead of signaling failure" (#7) | `orchestrator.py:2280` stamps `final_weighted_score=0.0` onto the error dict -- the exact laundering line |
| Explicit degraded marker, never a fabricated answer (#4, #5) | `autonomous_loop.py:1932-1940` raise; `:2935-2936` NULL coercion; `$._degraded` |
| Bounded recovery budget (#5), limit retries per request (#2) | `settings.py:194` `claude_code_empty_retry_max=2`; `orchestrator.py:853-857` |
| PIT beats min-max; min-max preserves power-law spikes (#3) | `meta_scorer.py:170-177` + `:296-300` percentile-rank over the full `head+tail` set (replaces the saturating clamp) |
| WARN on degraded-mode ENTRY (#4) | criterion-4 "2 consecutive all-fallback cycles" WARN |
| Deadline sizing (#2) | `settings.py:186` `claude_code_timeout_s=150`; `claude_code_client.py:600` `+30` |
| Test the fallback path (#1) | live_check must measure the post-promotion lite-fallback share |

## Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (**7**)
- [x] 10+ unique URLs total (**38**)
- [x] Recency scan (2024-2026) performed + reported
- [x] Full pages read (not abstracts); arXiv via `/html/` per the PDF chain
- [x] file:line anchors on every internal claim

Soft checks:
- [x] Internal exploration covered every relevant module (13 files)
- [x] Contradictions / consensus noted (AWS-vs-SRE on fallback; PIT-vs-RRF)
- [x] Claims cited per-claim
- [ ] **Gap disclosed:** the AWS Builders' Library "Avoiding fallback"
  article could not be fetched (JS shell after redirect). Its thesis is
  carried indirectly by source #1, which cites it.
- [ ] **Gap disclosed:** no live Playwright UI capture taken in this session
  (Q/A blocker #3 remains open; it is a GENERATE/EVALUATE task, not research).

## JSON envelope

```json
{
  "tier": "complex",
  "external_sources_read_in_full": 7,
  "snippet_only_sources": 31,
  "urls_collected": 38,
  "recency_scan_performed": true,
  "internal_files_inspected": 13,
  "coverage": {
    "audit_class": false,
    "rounds": 1,
    "dry_rounds": 0,
    "K_required": 2,
    "new_findings_last_round": 0,
    "dry": false
  },
  "summary": "61.2 is NOT a stale un-started defect: it was fully BUILT 2026-07-08 (commit 6186784c) and Q/A returned CONDITIONAL only on live evidence. It also HAS six immutable success_criteria (caller's premise of zero is refuted). Live BQ proves sub-item A is firing daily -- 142 of 156 full-path analysis_results rows in 40d are final_score=0.0 + recommendation=HOLD + $.final_synthesis.error='Failed to parse final report.', last seen 2026-08-06 19:20:56Z -- and proves sub-item C is FIXED (zero NULL company_name every day since 2026-07-09), which retires the Q/A's main blocker. B is live/ungated. A, D, E, F are built but flag-dark. 72.0.1's premise is REFUTED: phase-78.1 already routed meta_scorer through make_client. The immutable pytest -k command is RED (4 failed/67 passed): 2 unrelated tests and 2 real 61.2 regressions caused by phase-36.13 (3227347a) adding a live-kill-switch gate to execute_buy.",
  "brief_path": "handoff/current/research_brief_61.2.md",
  "gate_passed": true
}
```
