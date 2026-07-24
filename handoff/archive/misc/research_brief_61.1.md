# Research Brief -- phase-61.1 (away-ops AM, 2026-06-15)

Tier: moderate (caller-stated). Research gate for the contract/PLAN
phase of the ONLY remaining criterion (criterion 4) of step 61.1.
Headless `claude -p` away session; model pinned Opus 4.8
(FABLE-HEADLESS workaround -- expected).

## The question

Criterion 4 is a NEGATIVE-ASSERTION (safety-guardrail) criterion:
with the flags ON, the first post-restart daily cycle must show
**zero** swap_for_higher_conviction SELLs of holdings lacking a
same-cycle analysis_results row (60.2) AND **zero** executed trades
with risk_judge_decision='REJECT' (57.1).

The first post-flag cycle (5f15fdbe, 2026-06-12 Fri) traded ZERO
times. So both negative assertions hold, but VACUOUSLY -- there
were no events of any kind for a guardrail to block. The genuine
open question Main asked me to research:

> How do you rigorously evidence a safety-guardrail / negative-
> assertion criterion ("zero bad events") when the period under test
> produced zero relevant (triggering) events? Is criterion 4
> closeable on (absence-in-prod + passing activation tests), or must
> it wait for a nonzero trading cycle?

**Bottom line (full reasoning in RECOMMENDATION):** the literature
is unusually direct here. A vacuous pass is real-but-weak evidence;
the documented remedy across formal verification (2001-2006) AND
modern AI-guardrail practice (2026) is identical -- pair the
absence-in-production observation with an explicit *witness /
activation test* proving the guardrail FIRES on a triggering input.
pyfinagent already HAS those witnesses (28 passing activation tests).
So criterion 4 is closeable as **PASS-with-caveat** (vacuous-in-prod
+ activation-tested), NOT a blind PASS and NOT a hard CONDITIONAL-
block -- provided the brief's framing + the witness evidence are
written into live_check_61.1.md. A defensible alternative is
CONDITIONAL pending the Mon 06-15 18:00Z cycle if Main/Q-A want
prod-side non-vacuous confirmation; the cost is one trading day and
the trade is not guaranteed to produce a triggering event either.

## Success criteria (verbatim from .claude/masterplan.json phase-61 -> 61.1)

1. "the operator's verbatim flag tokens (60.2 FLAG / 60.3 FLAG /
   57.1 FLAG, each ON or KEEP OFF) are recorded in
   handoff/current/live_check_61.1.md and backend/.env matches them
   exactly; no flag changed without its token"
2. "post-restart, the running uvicorn process start time is later
   than the phase-60.4 commit timestamp (ps -o lstart vs git log
   evidence pasted verbatim), proving phase-60.2/60.3/60.4 code is
   loaded"
3. "frontend kickstarted via launchctl; Playwright capture shows
   http://localhost:3000/login loads without ChunkLoadError"
4. "first post-restart daily-cycle evidence in live_check_61.1.md as
   verbatim BQ rows: if 60.2 FLAG: ON, zero swap_for_higher_conviction
   SELLs of holdings lacking a same-cycle analysis_results row; if
   57.1 FLAG: ON, zero executed trades with risk_judge_decision='REJECT'"
5. "handoff/harness_log.md cycle entry appended before the status flip"

Criteria 1-3 are COMPLETE (live_check_61.1.md A-D, independently
re-verified below for #2). #5 is a process step done at close. Only
**criterion 4** is open.

## External sources read in full (>=5 required for gate -- 7 achieved)

| # | URL | Accessed | Kind | Fetched how | Key finding |
|---|-----|----------|------|-------------|-------------|
| 1 | https://en.wikipedia.org/wiki/Evidence_of_absence | 2026-06-15 | reference (year-less canonical) | WebFetch full | "The expectation of evidence makes its absence significant." Absence of evidence becomes evidence of absence only when a positive result WOULD be expected if the effect were present; depends on "the detection power of the applied methods." |
| 2 | https://en.wikipedia.org/wiki/Vacuous_truth | 2026-06-15 | reference (year-less canonical) | WebFetch full | Vacuous truth = "a conditional... true because the antecedent cannot be satisfied"; such statements "do not really say anything" and can be "misleading." |
| 3 | http://www.cs.toronto.edu/~chechik/courses05/csc2108/beer01.pdf (Beer, Ben-David, Eisner, Rodeh, *Efficient Detection of Vacuity in Temporal Model Checking*, Formal Methods in System Design 18, 2001) | 2026-06-15 | peer-reviewed (canonical) | WebFetch->pdfplumber | "typically 20% of formulas are found to be trivially valid, and... trivial validity always points to a real problem in either the design or its specification or environment." Remedy = generate an "interesting witness: a trace which shows a non-trivial example of the validity of a formula. Examining a positive example provides some confidence that the formal specification accurately reflects the intent of the user." |
| 4 | https://www.cs.huji.ac.il/~ornak/publications/concur06b.pdf (Kupferman, *Sanity Checks in Formal Verification*, CONCUR 2006) | 2026-06-15 | peer-reviewed (canonical) | WebFetch->pdfplumber | Canonical req->grant: phi = AG(req -> AFgrant); "one should distinguish between satisfaction of phi in systems in which requests are never sent, and satisfaction in which phi's precondition is sometimes satisfied... the first type of satisfaction suggests some unexpected properties of the system, namely the absence of behaviors in which the precondition was expected to be satisfied." Sanity checks (vacuity + coverage) are needed because "a positive answer" alone can hide a modeling error. |
| 5 | https://www.promptfoo.dev/docs/guides/testing-guardrails/ | 2026-06-15 | official docs (practitioner) | WebFetch full | Guardrails need deliberate challenge testing, not just monitoring normal traffic; "Balance true and false positives"; measure False-Negative Rate (harmful query misclassified as safe). I.e., you must test that the guardrail FIRES. |
| 6 | https://arxiv.org/html/2412.14020 (Landscape of AI safety concerns -- LAISC methodology) | 2026-06-15 | peer-reviewed (recency 2024) | WebFetch full | "demonstrating the absence of AI-SCs... does not provide estimations for the system's probability of causing harm" on its own; requires Verifiable Requirements + Metrics&Mitigation as *affirmative* design-time evidence, "rather than waiting for failures." |
| 7 | https://www.getmaxim.ai/articles/the-complete-ai-guardrails-implementation-guide-for-2026/ | 2026-06-15 | industry (recency 2026) | WebFetch full | Production telemetry and adversarial testing are COMPLEMENTARY not interchangeable: "NIST AI RMF Measure 2.6: adversarial testing evidence from runtime guardrail telemetry." "Every blocked request... should land in observability... essential for... NIST AI RMF's Measure function." |

## Identified but snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched |
|-----|------|-----------------|
| https://arxiv.org/pdf/2510.03485 (Policy-Compliant Agents: Learning Guardrails for Policy Violation Detection) | peer-reviewed (2025) | WebFetch returned partial/compressed; confirmed it constructs explicit violation test cases (positive activation) + measures detection performance, but numbers not cleanly extractable. Corroborates #5 directionally. |
| https://dl.acm.org/doi/10.1007/s10703-014-0221-0 (Vacuity in practice: temporal antecedent failure, FMSD 2015) | peer-reviewed | HTTP 403 (paywall). Snippet: "temporal antecedent failure always indicates a problem in the model, environment or property." Superseded for our purposes by #3/#4 read in full. |
| https://link.springer.com/chapter/10.1007/3-540-48153-2_8 (Vacuity Detection in Temporal Model Checking, Kupferman & Vardi) | peer-reviewed | Springer paywall; content captured via #4's citations. |
| https://arxiv.org/pdf/2102.02625 (Safety Case Templates for Autonomous systems) | peer-reviewed | Snippet only; safety-case = "structured argument supported by a body of evidence." Background, not load-bearing. |
| https://accuknox.com/blog/runtime-ai-governance-security-platforms-llm-systems-2026 | industry (2026) | Snippet: monitoring vs enforcement modes; "log a warning" mode before enforcement. Corroborates #7. |
| https://www.esrb.europa.eu/pub/pdf/asc/esrb.ascreport202512_AIandsystemicrisk.en.pdf | official (Dec 2025) | Snippet only; autonomous-AI systemic-risk in finance; governance-as-evidence theme. Macro context. |
| https://arxiv.org/pdf/2506.00641 (AgentAuditor, NeurIPS 2025) | peer-reviewed (2025) | Snippet only; LLM-agent safety evaluation needs to catch step-by-step dangers rule-based evaluators miss. Recency context. |
| https://www.trmlabs.com/resources/blog/autonomous-ai-agents-and-financial-crime | industry | Snippet only; "governance architecture becomes evidence." Recency context. |
| https://en.wikipedia.org/wiki/Safety_case | reference | Snippet via search; safety-case definition. |
| https://arxiv.org/pdf/2507.06134 (OpenAgentSafety, ICLR 2026) | peer-reviewed | Snippet only; agent safety benchmark. Recency context. |

Total unique URLs collected: 17 (7 full + 10 snippet-only). >=10 floor met.

### Search-query variants run (3-variant discipline)

- Current-year frontier (2026): "LLM agent guardrail verification
  absence of violations production monitoring evidence 2026";
  "negative testing guardrail activation test positive assertion
  safety-critical 2026"; "vacuity detection safety assurance LLM
  agent 2025 2026 trivial satisfaction guardrail".
- Last-2-year window (2024-2025): "autonomous trading system go-live
  first cycle deployment risk control evidence discipline 2025".
- Year-less canonical: "safety case absence of evidence is not
  evidence of absence verification autonomous systems"; "vacuous
  truth verification temporal logic antecedent never satisfied";
  "vacuity detection model checking witness sanity check Beer
  Kupferman temporal specification passes". The year-less queries
  surfaced the keystone 2001/2006 prior art (Beer et al.,
  Kupferman) that a year-locked query would have buried.

## Recency scan (last 2 years, 2024-2026)

PERFORMED. Result: the last-2-year window CONFIRMS and MODERNIZES
the canonical 2001-2006 formal-verification finding rather than
superseding it. New findings:

1. **NIST AI RMF Measure 2.6 (live in 2026 guardrail practice)**
   explicitly requires "adversarial testing evidence from runtime
   guardrail telemetry" -- i.e., regulators/standards now demand the
   exact pairing (activation testing + production telemetry) that
   the 2001 vacuity literature called "interesting witness + the
   pass." (Maxim 2026 guide; AccuKnox 2026.)
2. **Monitor-mode-before-enforce-mode** is standard 2026 deployment
   practice ("run in monitoring mode before switching to
   enforcement"; "log a warning" mode). pyfinagent did the inverse-
   but-equivalent: it ran the OLD (broken) advisory behavior in
   prod, captured the failures (06-03 DELL, 06-09 066570.KS), THEN
   flipped to enforce -- so the "what does prod look like with the
   guardrail OFF" baseline already exists as evidence the antecedent
   CAN occur. (Maxim 2026; AccuKnox 2026.)
3. **AI-safety-assurance methodology (LAISC, arXiv:2412.14020, 2024)**
   states absence-of-failure is insufficient alone; demands
   affirmative design-time Verifiable-Requirement evidence -- the
   academic 2024 statement of "passing activation tests are
   required, not just a clean prod window."
4. **No new finding CONTRADICTS** closing a guardrail-activation gate
   on (absence-in-prod + passing activation tests). Across formal
   verification, AI-safety assurance, and 2026 guardrail practice the
   consensus is one-directional: a vacuous/absence observation is
   necessary but not sufficient; the sufficient complement is a
   witness/activation test. That complement EXISTS for 61.1.

## Internal code inventory (file:line anchors)

| File | Lines | Role | Status |
|------|-------|------|--------|
| .claude/masterplan.json | phase-61 -> 61.1 -> verification.success_criteria[0..4] | The 5 immutable criteria (quoted verbatim above) | VERIFIED -- criterion-4 interpretation correct: it is conditioned "if 60.2 FLAG: ON" / "if 57.1 FLAG: ON" and both flags ARE on |
| backend/services/portfolio_manager.py | 194-212 | 57.1 reject-binding gate (candidate-build chokepoint) | VERIFIED gated: `if _rj_decision=="REJECT" and getattr(settings,"paper_risk_judge_reject_binding",False): ... continue`. ON=>drop candidate (never enters buy_candidates); OFF=>passthrough (REDUCED/HEDGED stay advisory). Appends to blocked_out. |
| backend/services/portfolio_manager.py | 471, 478-507 | 60.2 churn-fix (swap displacement) | VERIFIED gated: `_churn_fix_on = bool(getattr(settings,"paper_swap_churn_fix_enabled",False))`. score is None + ON => `continue` (EXCLUDE holding from displacement). OFF => `score = 0.0` sentinel (byte-identical pre-60.2 churn engine). |
| backend/services/autonomous_loop.py | 784-788 | 57.1 per-cycle RiskJudge sector context | VERIFIED gated; built only when flag ON; OFF byte-identical (empty ctx). |
| backend/services/autonomous_loop.py | 805-808 | 60.2 hours-precise re-eval age | VERIFIED gated; ON => seconds/86400 (>=72h); OFF => truncated .days. |
| backend/services/autonomous_loop.py | 1728-1734, 1742-1752 | 57.1 RiskJudge prompt builders (system+template) | VERIFIED gated; "never bind on a blind judge"; OFF returns verbatim constant (byte-identity). |
| backend/services/autonomous_loop.py | 1948-1959, 2228-2239 | 60.3 data-integrity gate (Claude + Gemini lite analyzers) | VERIFIED gated: `_di_enabled = bool(getattr(settings,"paper_data_integrity_enabled",False))`; ON + blocking flag => `_data_integrity_blocked_analysis(...)` returns pre-LLM. Normalization/flagging ALWAYS runs (ungated observability); only the BLOCK is gated. |
| backend/services/autonomous_loop.py | 1162-1177 | RiskJudge-blocked surfaced on cycle summary | `summary["risk_judge_blocked"]=_rj_blocked` + `logger.warning("BINDING RiskJudge gate blocked %d BUY(s)...")`. IN-MEMORY summary + LOG ONLY. |
| backend/services/autonomous_loop.py | 1298-1329 | strategy_decisions heartbeat (phase-30.7) | Row fields = ts/cycle_id/decided_strategy/prior_strategy/trigger/decay_signal/decay_attribution/rationale. NO risk_judge_blocked field. |
| backend/services/cycle_health.py | 264-329 + autonomous_loop.py 1393-1402 | cycle_history.jsonl persistence | record_cycle_end persists status/n_trades/error_count/data_source_ages/bq_ingest_lag/meta_scorer_degraded ONLY. NO risk_judge_blocked. |
| backend/services/paper_trader.py | 119, 126, 270 (execute_buy); 348, 420 (execute_sell) ; backend/db/bigquery_client.py:99,210 | financial_reports.paper_trades writer | paper_trades has a `risk_judge_decision` column BUT only EXECUTED trades are written (execute_buy/execute_sell). A REJECT that is BLOCKED never calls execute_buy => no row. So a "REJECT that executed" is observable (the pre-flag bug); a "REJECT that was correctly blocked" is NOT a paper_trades row. |
| backend/api/paper_trading.py | 1299-1322 (`_add_scheduler_job`) | cron schedule | VERIFIED: APScheduler `"cron", hour=settings.paper_trading_hour, minute=0, day_of_week="mon-fri", ... coalesce=True`. Weekday-only; missed windows coalesce to ONE run. |
| backend/services/autonomous_loop.py | 355-394 | in-cycle calendar gate | INTL-only: `_open_today` returns True for US (ungated, byte-identical). The loop has NO internal US weekend gate -- weekend skipping is entirely the cron `day_of_week="mon-fri"`. |
| handoff/cycle_history.jsonl | tail (06-01..06-12) | live cycle log | VERIFIED weekday cadence: 06-05 Fri -> SKIP 06-06/06-07 Sat/Sun -> 06-08 Mon. Post-flag cycle 5f15fdbe = 06-12T18:00Z completed n_trades=0 meta_scorer_degraded=true. No 06-13/06-14 rows. |

Internal files inspected: 8 distinct source files (masterplan.json,
portfolio_manager.py, autonomous_loop.py, cycle_health.py,
paper_trader.py, bigquery_client.py, paper_trading.py,
cycle_history.jsonl).

## Key findings

1. **Criterion-4 interpretation is correct and both flags are ON.**
   The criterion is conditional ("if 60.2 FLAG: ON" / "if 57.1 FLAG:
   ON"); live_check A confirms both ON; section C confirms the
   running process booted with all three flags True. (masterplan.json
   phase-61->61.1; live_check_61.1.md A/C.)

2. **All three guardrails are correctly wired and flag-gated**, each
   on `getattr(settings,"<flag>",False)` with ON=>block /
   OFF=>byte-identical advisory passthrough. file:line anchors in the
   inventory. No wiring defect found.

3. **The live evidence for criterion 4 is NECESSARILY absence-in-
   paper_trades, because there is NO queryable "generated-but-
   blocked decisions" table.** A blocked REJECT (57.1) and an
   excluded-from-displacement holding (60.2) surface ONLY as (a) an
   in-memory `summary["risk_judge_blocked"]` that is NOT persisted to
   cycle_history.jsonl or any BQ table, and (b) a `logger.warning`
   line in backend.log. paper_trades has a `risk_judge_decision`
   column, but only EXECUTED trades are written -- so "REJECT that
   executed" is queryable (the pre-flag bug, 06-09 066570.KS) while
   "REJECT correctly blocked" is not a row. Positive prod evidence
   that the gate FIRED is therefore log-only, not BQ-queryable.
   (portfolio_manager.py:194-212; autonomous_loop.py:1162-1177;
   paper_trader.py:119/348; bigquery_client.py:99,210.)

4. **5f15fdbe is genuinely the only post-flag cycle.** The scheduler
   is `day_of_week="mon-fri"` (paper_trading.py:1299-1322) and
   cycle_history.jsonl shows the weekday-only cadence empirically
   (06-05 Fri -> skip weekend -> 06-08 Mon). Today is Mon 2026-06-15
   05:41 UTC; next cycle 18:00 UTC today. The 06-12 cycle traded 0
   times (n_trades=0) AND was meta_scorer_degraded=true, so it is
   doubly weak prod evidence -- zero trades means zero antecedents,
   and degraded scoring means the cycle wasn't a clean exemplar
   anyway.

5. **The vacuousness is a textbook "antecedent failure."** Beer et
   al. (2001) and Kupferman (2006) formalize EXACTLY this: a safety
   property like AG(req -> AFgrant) ["every request eventually
   granted"] is "satisfied vacuously in systems in which requests are
   never sent." Criterion 4 maps 1:1: "no executed trade is a REJECT"
   is vacuously satisfied in a cycle where no trade (no request)
   occurred. Beer's field data: "typically 20% of formulas are...
   trivially valid, and... trivial validity always points to a real
   problem." The pyfinagent twist: the vacuity here is NOT a property
   bug -- it is just a quiet trading day; but the EVIDENTIARY weakness
   is identical, and so is the prescribed remedy.

6. **The prescribed remedy -- in BOTH the 25-year-old formal
   literature AND 2026 guardrail practice -- is a WITNESS /
   ACTIVATION TEST, which pyfinagent already has.** Beer:
   "interesting witness: a trace which shows a non-trivial example of
   the validity of a formula. Examining a positive example provides
   some confidence that the formal specification accurately reflects
   the intent." Modern: NIST AI RMF Measure 2.6 wants "adversarial
   testing evidence from runtime guardrail telemetry" (Maxim 2026);
   Promptfoo: guardrails "require deliberate challenge testing, not
   just monitoring normal traffic." pyfinagent's 28 passing tests
   (test_phase_57_1_reject_binding.py, test_phase_60_2_churn_fix.py,
   test_phase_60_3_data_integrity.py) ARE those witnesses: they
   drive a REJECT input and assert the candidate is dropped, and
   drive a no-same-cycle-analysis holding and assert it is excluded.
   That is the positive-activation evidence the absence-in-prod
   observation needs to be non-vacuous.

7. **"Absence of evidence" becomes meaningful only "when a positive
   result would be expected."** (Evidence-of-absence, Wikipedia,
   citing the drug-trial logic.) On 06-12 a positive result (a
   triggering trade) was NOT expected -- zero trades. So the 06-12
   prod observation alone carries almost no evidentiary weight FOR OR
   AGAINST the guardrails. Its value is purely "the system ran
   post-restart without error," not "the guardrail works." This is
   the precise reason the activation tests, not the cycle, are the
   load-bearing evidence.

## Consensus vs debate (external)

Consensus (strong, one-directional): a vacuous / absence-of-violation
observation is NECESSARY but NOT SUFFICIENT evidence that a safety
guardrail works; it must be paired with a positive witness /
activation test proving the guardrail fires on a triggering input.
This holds across (a) formal verification 2001-2006 (Beer, Kupferman),
(b) AI-safety assurance 2024 (LAISC), and (c) industry guardrail
practice + NIST AI RMF 2026 (Maxim, Promptfoo, AccuKnox).

Debate / nuance: how MUCH prod-side non-vacuous confirmation is
required before a control is "trusted" is judgment-dependent. The
formal-methods view (witness suffices) is more permissive; the
phased-deployment view (monitor-mode -> enforce-mode with a real
observed firing) wants at least one live activation. No source
treats a single quiet prod cycle as itself sufficient -- so the
debate is only between "tests suffice now" and "wait for one live
firing," NEVER "the empty cycle closes it."

## Pitfalls (from literature + internal)

- **Treating the vacuous pass as a strong PASS.** Beer: trivial
  validity "always points to a real problem" in the verification
  context; the analogue risk here is recording "criterion 4 PASSED,
  zero REJECTs executed" without disclosing that zero trades occurred
  -- a future auditor reading only the BQ rows would over-trust it.
- **Looking for positive "binding fired" evidence in BQ and finding
  none, then concluding the guardrail is broken.** There is no
  generated-but-blocked decisions table; the absence of a "REJECT
  blocked" row is EXPECTED design, not a defect (finding 3).
- **Waiting for the Mon cycle and getting another zero-trade or
  degraded cycle.** n_trades has been 0 on 2 of the last 2 cycles
  (06-11, 06-12) and meta_scorer_degraded=true on 06-12; there is no
  guarantee 06-15 produces a triggering event. The wait may not
  resolve the vacuousness.
- **Test-isolation artifact masquerading as a guardrail regression.**
  The plain pytest run shows 4 failures because the tests read the
  live .env (flags now ON) and assert default-OFF/off-path behavior;
  neutralizing the env (PAPER_*=false) yields 28 passed. This is a
  test-harness .env-bleed issue, NOT a guardrail regression -- but it
  MUST be run with the env neutralized to count as a clean witness.

## Application to pyfinagent (mapping external findings to internal anchors)

- Beer/Kupferman "interesting witness" == the 28 activation tests
  (test_phase_57_1_reject_binding.py / 60_2_churn_fix.py /
  60_3_data_integrity.py), run with PAPER_* flags neutralized to
  exercise BOTH the ON-block and OFF-passthrough branches at
  portfolio_manager.py:194-212 / :471-507 and
  autonomous_loop.py:1948-1959/:2228-2239.
- "Absence becomes evidence only when a positive result is expected"
  == the 06-12 cycle had n_trades=0, so no positive result was
  expected; its evidentiary weight for criterion 4 is ~nil beyond
  "ran without error." (cycle_history.jsonl; autonomous_loop.py
  cycle-summary path.)
- NIST AI RMF Measure 2.6 "adversarial testing evidence from runtime
  telemetry" == backend.log warnings at autonomous_loop.py:1174 +
  portfolio_manager.py:199 are the runtime telemetry that WOULD carry
  a live firing; the tests are the adversarial/activation half. The
  pre-flag bug rows in paper_trades (06-03 DELL, 06-09 066570.KS) are
  the "antecedent CAN occur" baseline.

## RECOMMENDATION

**Criterion 4 is CLOSEABLE now as PASS-with-caveat (vacuous-in-prod +
activation-tested), and that is the better-supported call than a hard
CONDITIONAL-block.** Rationale, grounded in the sources:

1. The negative assertions are literally satisfied for cycle
   5f15fdbe: zero post-flag paper_trades rows since 2026-06-12 =>
   zero REJECT-executed trades AND zero swap_for_higher_conviction
   SELLs of unanalyzed holdings. (Verify with the two BQ queries in
   "Evidence to paste" below; expected: 0 rows each.)

2. That satisfaction is VACUOUS (antecedent failure: zero trades).
   The literature is unanimous that a vacuous pass is weak-but-real
   and must be DISCLOSED as such, not laundered into a clean PASS
   (Beer 2001; Kupferman 2006; Evidence-of-absence).

3. The disclosed weakness is FULLY remediated by the witness /
   activation evidence pyfinagent already possesses: 28 passing
   activation tests proving each guardrail FIRES on a triggering
   input (the exact "interesting witness" Beer prescribes and the
   "adversarial testing evidence" NIST AI RMF Measure 2.6 requires).
   This is the documented sufficient complement to absence-in-prod.

4. The pre-flag prod failures (06-03 DELL, 06-09 066570.KS REJECT-
   that-executed; the 9/10 churn swaps) are the baseline proving the
   antecedent CAN occur in this system -- so the activation tests are
   testing a real, observed failure mode, not a hypothetical one.

**Required framing for live_check_61.1.md (so the close is honest and
auditable):** criterion 4 must be recorded as
"PASS (vacuous-in-prod, activation-tested)" with ALL THREE of:
   (a) the two BQ queries + their 0-row outputs (the absence
       observation), explicitly noting n_trades=0 on 5f15fdbe so the
       vacuousness is on the record;
   (b) the 28-passed activation-test command output run with the env
       neutralized (`PAPER_RISK_JUDGE_REJECT_BINDING=false
       PAPER_DATA_INTEGRITY_ENABLED=false
       PAPER_SWAP_CHURN_FIX_ENABLED=false python -m pytest
       backend/tests/test_phase_60_2_churn_fix.py
       backend/tests/test_phase_57_1_reject_binding.py
       backend/tests/test_phase_60_3_data_integrity.py -q`) -- the
       witness;
   (c) the pre-flag contrast rows (06-09 066570.KS REJECT-executed)
       as the antecedent-can-occur baseline.

**Defensible alternative (if Main or Q/A want prod-side non-vacuous
confirmation):** mark criterion 4 CONDITIONAL pending the Mon
2026-06-15 18:00 UTC cycle and re-pull the BQ rows after it. Caveats:
(i) the cost is one trading day; (ii) there is NO guarantee the Mon
cycle produces a triggering event (last 2 cycles = 0 trades), so the
wait may not resolve the vacuousness and could recur indefinitely;
(iii) even a clean Mon cycle is still only one data point. Given (i)-
(iii) and the unanimous literature that the WITNESS (not the cycle) is
the load-bearing evidence, PASS-with-caveat is the stronger close.
Do NOT, in any case, close criterion 4 as an unqualified PASS that
hides the zero-trade vacuousness.

This is a Q/A judgment call; the research gate's job is to establish
that BOTH options are defensible and that the unqualified-PASS option
is NOT. The brief recommends PASS-with-caveat.

### Evidence to paste into live_check_61.1.md (BQ queries)

```sql
-- 57.1: zero executed trades with risk_judge_decision='REJECT' (post-flag)
SELECT ticker, action, reason, risk_judge_decision, created_at
FROM `sunny-might-477607-p8.financial_reports.paper_trades`
WHERE created_at >= '2026-06-12' AND risk_judge_decision = 'REJECT';
-- expected: 0 rows

-- 60.2: zero swap_for_higher_conviction SELLs of holdings lacking a
-- same-cycle analysis row (post-flag). The first clause alone should
-- already be empty for 5f15fdbe (no trades at all):
SELECT ticker, action, reason, created_at
FROM `sunny-might-477607-p8.financial_reports.paper_trades`
WHERE created_at >= '2026-06-12'
  AND action = 'SELL' AND reason = 'swap_for_higher_conviction';
-- expected: 0 rows (vacuous -- n_trades=0 on the only post-flag cycle)
```

## Research Gate Checklist

Hard blockers -- all satisfied:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (7: 2 Wikipedia canonical, 2 peer-reviewed PDFs via pdfplumber, Promptfoo docs, arXiv LAISC, Maxim 2026)
- [x] 10+ unique URLs total incl. snippet-only (17)
- [x] Recency scan (last 2 years) performed + reported (confirms, does not supersede; NIST AI RMF 2.6, monitor->enforce, LAISC 2024)
- [x] Full papers / pages read (not abstracts) for the read-in-full set (Beer & Kupferman extracted in full via pdfplumber after the /pdf binary-skip; finance/CS-text F1 high)
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every relevant module (8 files; wiring + persistence + scheduler + live cycle log)
- [x] Contradictions / consensus noted (consensus one-directional; only debate is tests-suffice vs wait-one-firing)
- [x] All claims cited per-claim

```json
{"tier":"moderate","external_sources_read_in_full":7,"snippet_only_sources":10,"urls_collected":17,"recency_scan_performed":true,"internal_files_inspected":8,"gate_passed":true}
```
