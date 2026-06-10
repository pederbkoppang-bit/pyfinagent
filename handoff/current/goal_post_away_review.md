# Goal Prompt -- goal-post-away-review

Set by operator 2026-06-10 (return from the 8-day away window 2026-06-01 -> 2026-06-10,
during which the autonomous paper-trading run continued unattended). Authored in a remote
Claude Code container on branch `claude/sweet-feynman-zhs8p3` from Slack-digest evidence,
a /paper-trading UI screenshot, and a read-only code trace -- NO live BQ/Alpaca credentials
at authoring time. Therefore: every number below tagged SECONDARY must be re-verified
against primary data in phase-55 before any conclusion is drawn from it.

Operator decisions baked into this goal (AskUserQuestion + plan review, 2026-06-10):
1. Direction: review -> fix -> go-live prep. The review completes BEFORE fix work starts.
2. LLM spend for live trading cycles is decided by the operator AFTER the review reports
   (explicit checkpoint with burn estimate + expected value).
3. (superseded by events, same day) 53.4 + 53.5 were to be finished first -- the local run
   shipped 53.5 on main on 2026-06-10 (closing the 2026-06-01 goal's autonomous scope)
   and the operator deferred 53.4 ("dropped it 2026-06-10, home"). Nothing to execute
   before phase-55. Do NOT resurrect 53.4 inside this goal without an operator ask.
4. The work is split into separate masterplan phases (review / fixes / improvement /
   go-live), not one mega-phase. Phase-57 (improvement) is NOT pre-installed -- it is
   authored BY the review (finetune-vs-features is an open question the review must answer).
5. UI claims are verified in the live UI via Playwright MCP from now on (standing rule;
   see the CLAUDE.md amendment payload below).

## North star linkage (N* = Profit - (Risk Exposure + Compute Burn))

- Profit: stop the quantified churn bleed (away week net ~-2.7pp, -4.2pp from peak;
  MU -6.3% one-day round trip; 000660.KS -9.9% round trip) and restore a trustworthy
  P&L readout (the KR value corruption makes Profit unmeasurable today).
- Risk: ~100% semis/memory concentration is unmeasured and uncapped; the kill switch
  did not trip on the -3.5% 06-05 day (verdict open); both get audited with evidence.
- Burn: phase-55 is $0 (BQ reads + local scripts + Playwright only). Any live LLM spend
  is explicitly budgeted and operator-approved at the checkpoint, never assumed.

## Objective (one sentence)

Forensically review the 8-day autonomous run from primary data, fix only what the review
proves broken (data correctness first), let the review itself select the next improvement
(finetune-lever vs feature), and resume the production-ready runway under an explicit
operator spend decision.

## Verified current state (2026-06-10)

PRIMARY (repo/code, verified this session):
- `backend/services/paper_trader.py:265` records trade `total_value` as
  `quantity * exec_price` in LOCAL currency -- missing `* _local_to_usd`. For KR trades
  this persists KRW magnitudes into `financial_reports.paper_trades.total_value` as if
  USD. CRITICAL, code-confirmed.
- `backend/services/paper_trader.py:386-414`: SELL `transaction_cost` is computed on the
  local-currency sell value and persisted unconverted. HIGH, code-confirmed.
- `frontend/src/components/paper-trading/trades-columns.tsx:11,107-122` renders
  `total_value`/`transaction_cost` verbatim; the line-11 comment falsely claims they are
  USD.
- `frontend/src/components/paper-trading/cockpit-helpers.tsx:197-218`: the "VS KOSPI"
  card shows filtered holdings return, NOT KOSPI (^KS11) excess -- the index is never
  fetched.
- `mark_to_market()` (`paper_trader.py:496-584`) converts position values with FX
  correctly; therefore the on-screen NAV inflation is NOT explained by the :265 bug
  alone. Open suspects: the FX-unavailable fallback at :512-520 ("keeping last-known
  market_value") and cash-credit paths. ROOT CAUSE OPEN -- 55.1 must trace it.
- Autonomous lite path (`backend/agents/orchestrator.py:1491-2069`): rag, earnings_tone,
  insider, patent, news/social DO run in lite mode; deep_dive, devil's-advocate,
  risk-assessment, and multi-round debate are SKIPPED. Ground truth of what actually
  fired is queryable from `pyfinagent_data.llm_call_log` (agent x ticker x cycle_id x
  cost columns).
- Masterplan (re-verified post-merge with origin/main, 2026-06-10): phase-54 done;
  53.1 done (no-trade rebalance band measured and REJECTED via the Ledoit-Wolf
  SR-difference gate -- binding precedent); 53.2/53.3 done; 53.5 DONE (e2e-smoke
  capstone shipped, closing the 2026-06-01 goal's autonomous scope); 53.4 DEFERRED
  (operator dropped it 2026-06-10). phase-53/43.0 parents intentionally stay pending
  (operator-gated closure). Phase ids 55-58 unused.
- Operator follow-ups batched in `handoff/current/cycle_block_summary.md` (mas-harness
  cron re-enable, phase-43.0 operator actions, 16 env-coupled test failures).

SECONDARY (Slack #ford-approvals digests + screenshot; reconcile in 55.1):
- NAV path: +21.9% (06-01) -> +22.5% (06-02) -> peak +23.4% (06-03) -> +22.8% (06-04)
  -> +19.3% (06-05; -$693 on an 8-trade mass rebalance) -> +19.4% (06-08) -> +19.2%
  (06-09).
- Round trips: MU bought 06-08 @ 954.38, sold 06-09 @ 894.53 (-6.3% in one day);
  000660.KS sold 06-02 @ 2,360,000, rebought 06-04 @ 2,298,000, sold 06-05 @ 2,070,000
  (-9.9%); DELL traded 4x in 9 days (467.62 -> 424.15 -> 400.15 -> 370.70).
- Signal flip example: SNDK 7.0 BUY (06-09 digest) vs 5.0 HOLD (06-08 digest).
- Concentration: every traded name is semis/memory/storage (MU, SNDK, WDC, STX, DELL,
  ON, INTC, HPE, AMD, 000660.KS, 005930.KS, 066570.KS, 009150.KS).
- Screenshot (/paper-trading, Trades tab, KR filter): Value column shows KRW-as-USD
  (1.47 x KRW 248,000 ~= USD 264 displayed as "364 175,06 USD"); NAV card 345,968.86
  USD on a "$10K virtual fund" (+1,629.84%); Cash 22,883.73 USD; fee $1,056.20 on one
  Samsung sell; "VS KOSPI +0.00%"; gate NOT ELIGIBLE 2/5; kill ACTIVE 0.0%/3.4%.
- Incidents: Slack "Approve" (06-01, x2) -> 'Missing API key for provider "anthropic"';
  backend-watchdog unreachable/ReadTimeout ~20:05-20:50 CEST on 05-27, 05-28, 06-04;
  05-28 morning digest scored ALL analyses 0.0/10 HOLD.
- Go-live gate NOT ELIGIBLE (1/5 on 06-01; 2/5 on the 06-10 screenshot); kill-switch
  thresholds daily -1.5%/4%, trail -0.1%/10%.

## CRITICAL constraints (violating any is an automatic FAIL for the violating step)

1. REVIEW BEFORE FIX. No phase-56+ step opens until phase-55 is `done`. No fix without a
   55.x finding ID (F-1, F-2, ... assigned in 55.3).
2. DO-NO-HARM. The working US pure-quant momentum core (+20% NAV) must not regress:
   every behavior change is config-gated + default-off + measured ON-vs-OFF before any
   live flip. No live flag flips inside this goal.
3. NO NAIVE NO-TRADE BAND. 53.1's Ledoit-Wolf REJECT is binding. Churn levers must be
   different mechanisms (hysteresis, min holding period, sector cap, turnover budget)
   and pass the SAME robustness gate.
4. OPERATOR-GATED: LLM API spend (live cycles), pip installs, BQ DROP / unqualified
   DELETE / historical-row backfill, launchctl changes. Phase-58 live cycles are blocked
   until the verbatim spend reply is on record.
5. 53.5 is done and 53.4 is deferred (2026-06-10) -- do not re-open either inside this
   goal without an explicit operator ask.
6. Install the payloads below byte-for-byte. If `.claude/masterplan.json` on main has
   drifted (a phase-55+ already exists, or statuses changed), STOP and ask the operator
   before renumbering.
7. Playwright-MCP UI verification: every UI claim in this goal is verified against the
   running app behind the NextAuth wall (browser_navigate + browser_snapshot /
   browser_take_screenshot); code reading alone is not UI evidence.
8. Full harness protocol per step (CLAUDE.md): researcher FIRST (>=5 sources read in
   full, recency scan), contract.md with criteria copied verbatim, ONE fresh qa after
   GENERATE, harness_log.md append BEFORE the masterplan flip. No self-evaluation, no
   verdict-shopping.

## Execution order

1. Install commit: phase-55/56/58 payload into `.claude/masterplan.json` + replace
   `handoff/current/active_goal.md` + add the CLAUDE.md bullet. Commit message:
   `feat(masterplan): add phase-55/56/58 post-away-review goal; refresh active_goal`.
2. 55.1 -> 55.2 -> 55.3 (review; $0; no fixes). (53.5 already shipped on main
   2026-06-10; 53.4 deferred -- nothing precedes phase-55.)
3. OPERATOR CHECKPOINT (see mechanics below). Phase-56 may start once phase-55 is done,
   in parallel with the operator's deliberation.
4. 56.1 -> 56.2 (fixes).
5. Phase-57 install (the variant the operator picked) -> execute it.
6. 58.1 (go-live runway; CLOSES this goal) -> refresh cycle_block_summary.md -> HARD STOP.

## Operator checkpoint mechanics (after 55.3)

55.3 posts a decision block to the operator Slack channel containing:
- LLM burn estimate: $/cycle derived from `pyfinagent_data.llm_call_log` cost columns
  (fallback: token counts x current model pricing), multiplied over the planned 1-2 week
  live window.
- Expected value: which of DoD-2/5/6/7/9 the window closes; projected go-live-gate delta
  (baseline 1/5 on 06-01).
- The finetune-vs-features recommendation + both candidate phase-57 payloads (summary).
Operator replies with BOTH lines (verbatim grammar):
- `LLM SPEND: APPROVED <budget>` or `LLM SPEND: DECLINED`
- `PHASE-57: LEVER` or `PHASE-57: FEATURE`
The replies are recorded verbatim (with dates) in `handoff/current/live_check_58.1.md`
(spend) and in the phase-57 install commit message (variant). Until then: phase-57 is
not installed and phase-58 runs no live cycle.

## Masterplan installation payload (canonical; install byte-for-byte)

Append these three objects to `phases` in `.claude/masterplan.json`. The
`success_criteria` arrays are the immutable acceptance criteria for each step -- copy
them verbatim into each step's `contract.md`; do NOT edit them.

```json
[
  {
    "id": "phase-55",
    "name": "Away-week forensic review (autonomous paper-trading run 2026-06-01 -> 2026-06-10): evidence-first post-mortem BEFORE any fix work. $0 phase family: BQ reads + local scripts + Playwright UI inspection only; NO code fixes, NO LLM trading-cycle spend. Outputs: primary-data reconciliation of the away week, confirmation/refutation of the code-traced FX defects, root cause of the NAV discrepancy, ops-incident triage, agent skill-coverage audit, ranked findings with stable IDs, a finetune-vs-features strategic recommendation with BOTH candidate phase-57 payloads, and the operator spend-decision block.",
    "status": "pending",
    "depends_on": ["phase-53"],
    "gate": null,
    "steps": [
      {
        "id": "55.1",
        "name": "Data-integrity + trading forensics -- PRIMARY-data post-mortem of the away week: reconcile the Slack-digest NAV path, quantify turnover/round-trips/concentration, confirm or refute the code-traced FX defects (paper_trader.py:265 total_value, :386-414 SELL fee) against live BQ rows, trace the on-screen NAV=345,968.86-on-$10K discrepancy to root cause, audit the VS-KOSPI readout and the 06-05 kill-switch non-trip. Live UI evidence via Playwright MCP.",
        "status": "pending",
        "harness_required": true,
        "priority": "P0",
        "depends_on_step": "53.5",
        "audit_basis": "Operator return 2026-06-10. Screenshot evidence: KR Value column shows KRW-as-USD (1.47 x KRW 248,000 ~= USD 264 displayed as 364,175.06 USD); NAV 345,968.86 USD on a $10K fund (+1,629.84%); fee 1,056.20 on one sell; VS KOSPI +0.00%. Slack #ford-approvals digests: +21.9% 06-01 -> peak +23.4% 06-03 -> +19.2% 06-09; MU -6.3% one-day round trip; 000660.KS -9.9%; DELL 4 trades in 9 days. Code trace 2026-06-10 (remote session): paper_trader.py:265 records total_value in local currency (missing * _local_to_usd); :386-414 SELL transaction_cost unconverted; mark_to_market :496-584 converts correctly so the NAV inflation root cause is OPEN (suspects: the :512-520 FX-unavailable fallback, cash-credit paths).",
        "verification": {
          "command": "cd /Users/ford/.openclaw/workspace/pyfinagent && test -f handoff/current/55.1-away-week-postmortem.md && test -f handoff/current/live_check_55.1.md",
          "success_criteria": [
            "the post-mortem (handoff/current/55.1-away-week-postmortem.md) is built from PRIMARY data (BQ financial_reports.paper_trades/paper_positions/paper_portfolio_snapshots + /api/paper-trading/* endpoints), reconciles the Slack-digest NAV path (+21.9% 06-01, +23.4% 06-03, +19.3% 06-05, +19.2% 06-09) to within +/-0.2pp or reports the divergence as its own finding, and quantifies: weekly turnover, per-round-trip realized P&L for MU (06-08 -> 06-09), 000660.KS (06-02 -> 06-05) and DELL (4 trades), plus win_rate/profit_factor/expectancy/median_holding_days (via /api/paper-trading/performance) and PSR/DSR/bootstrap-CI (via /api/paper-trading/metrics-v2)",
            "the code-traced FX defects are confirmed or refuted against live BQ rows (paper_trader.py:265 total_value; :386-414 SELL transaction_cost), the corruption scope is classified (stored-data vs display-only; affected row count + date range), and the NAV/Cash/'$10K fund' three-way discrepancy is traced to a root cause at file:line with the :512-520 FX-fallback suspect explicitly ruled in or out; the VS-KOSPI readout is audited against cockpit-helpers.tsx:197-218",
            "concentration is measured per snapshot day (sector weights + portfolio HHI) and the report cites the config/code path of any existing concentration limit or states NONE EXISTS; the kill-switch audit for 06-05 reads the configured thresholds from live config (file path cited), computes the daily P&L from snapshots, and renders SHOULD-HAVE-TRIPPED (defect traced to file:line) or CORRECTLY-DID-NOT-TRIP (arithmetic shown) -- presuming either verdict in advance is a FAIL",
            "live UI evidence is captured via Playwright MCP (the /paper-trading page behind the NextAuth wall: Value column, NAV/Cash cards, VS-KOSPI card) and embedded in live_check_55.1.md; the step performs NO fix work and NO LLM trading-cycle spend"
          ],
          "live_check": "REQUIRED -- the data-integrity section of the post-mortem + Playwright captures of the live /paper-trading page + the BQ row evidence (queries + result excerpts). No fixes in this step."
        },
        "retry_count": 0,
        "max_retries": 3
      },
      {
        "id": "55.2",
        "name": "Ops incidents + agent-quality audit -- root-cause the Slack approve-flow anthropic-key error, the nightly backend-watchdog ReadTimeouts, and the 05-28 all-0.0/10 scoring day; audit which agent skills actually fired in the autonomous lite path over the away week (rag, earnings_tone, insider, patent, news/social vs the lite-mode skip list); quantify signal stability (day-over-day action flips, |delta-score|).",
        "status": "pending",
        "harness_required": true,
        "priority": "P0",
        "depends_on_step": "55.1",
        "audit_basis": "Slack #ford-approvals 06-01: operator typed 'Approve' twice -> 'Missing API key for provider anthropic' -- the human-in-the-loop approval path is broken. Watchdog unreachable/ReadTimeout ~20:05-20:50 CEST on 05-27, 05-28, 06-04 (a pattern, not a one-off). 05-28 morning digest: ALL analyses 0.0/10 HOLD (degraded scoring passed silently). Code trace: orchestrator.py:1491-2069 shows lite mode runs rag/earnings_tone/insider/patent/news but skips deep_dive, devil's-advocate, risk-assessment, multi-round debate; pyfinagent_data.llm_call_log (agent x ticker x cycle_id x cost) is the ground truth of what fired. Operator question 2026-06-10: 'are the agents also using all the skills, rag, earnings tone, insider, patent, news'.",
        "verification": {
          "command": "cd /Users/ford/.openclaw/workspace/pyfinagent && test -f handoff/current/55.2-ops-skill-audit.md && test -f handoff/current/live_check_55.2.md",
          "success_criteria": [
            "incident triage with root cause or honest bounding for each: (a) the Slack 'Approve' -> 'Missing API key for provider anthropic' error (2026-06-01 x2) traced to file:line; (b) the nightly watchdog ReadTimeout pattern (~20:05-20:50 CEST on 05-27/05-28/06-04) characterized from logs with its trigger identified or honestly bounded; (c) the 05-28 all-analyses-0.0/10 day explained via pyfinagent_data.llm_call_log + strategy_decisions; each incident gets a severity and a stable finding ID for phase-56, or a WONTFIX rationale",
            "a per-skill fire-count table over 2026-06-01..2026-06-10 from llm_call_log (agent x cycle_id), explicitly covering rag, earnings_tone, insider, patent, news/social vs the lite-mode skip list (deep_dive, devil's-advocate, risk-assessment, multi-round debate), with the orchestrator.py:1491-2069 code paths cited; gaps between code expectation and observed fire counts are findings",
            "a reasoning-quality spot-check of >=3 stored analyses from the away week (the agent rationale behind at least one whipsaw trade among MU/000660.KS/DELL included), assessing whether the cited skills' outputs actually informed the decision",
            "signal stability is quantified across the week (count of day-over-day BUY/HOLD/SELL action flips and mean |delta-score| per ticker; the SNDK 7.0-BUY -> 5.0-HOLD flip reproduced from stored data), AND scripts/harness/paper_execution_parity.py + scripts/risk/tca_report.py outputs are included or their failure honestly reported; NO fix work, NO LLM trading-cycle spend"
          ],
          "live_check": "REQUIRED -- the ops+skills section of the post-mortem: llm_call_log query results (fire-count table), incident evidence excerpts, and the signal-stability table."
        },
        "retry_count": 0,
        "max_retries": 3
      },
      {
        "id": "55.3",
        "name": "Synthesis + operator checkpoint -- consolidate 55.1+55.2 into a ranked findings table with stable IDs; write the deep-research strategic chapter (finetune vs features); emit BOTH candidate phase-57 payloads (lever-variant and feature-variant, full step schema, install-ready); post the operator decision block (LLM burn estimate + expected DoD value + reply grammar) to Slack.",
        "status": "pending",
        "harness_required": true,
        "priority": "P0",
        "depends_on_step": "55.2",
        "audit_basis": "Operator directive 2026-06-10: 'use your knowledge and reasoning with deep research on the matter how to best consider whether our app is on the right track and needs more finetuning or we need more features'. Operator decisions: LLM spend decided after the review reports; phase-57 is NOT pre-installed because finetune-vs-features is an open question the review must answer (pre-installing a lever phase would presume the answer). Precedent: 53.1's honest REJECT shows the robustness-gate pattern the lever-variant must reuse.",
        "verification": {
          "command": "cd /Users/ford/.openclaw/workspace/pyfinagent && test -f handoff/current/55.3-synthesis-checkpoint.md && test -f handoff/current/live_check_55.3.md",
          "success_criteria": [
            "a ranked findings table (severity x N*-impact) consolidates 55.1 + 55.2, with each finding carrying a stable ID (F-1, F-2, ...) that phase-56 fixes must reference; the table separates CODE-CONFIRMED findings from DATA-INFERRED ones",
            "the strategic chapter passes the research gate (>=5 sources read in full + recency scan, per .claude/rules/research-gate.md) covering LLM-trading-agent evaluation, paper-trading statistical power (8 days of dailies cannot establish significance -- state the honest minimum window), and agent-skill ROI, and concludes finetune-vs-features with explicit reasoning grounded in the findings + literature",
            "the chapter emits TWO complete candidate phase-57 JSON payloads conforming to the masterplan step schema, install-ready: a LEVER variant (exactly ONE of score-hysteresis/persistence, minimum holding period, sector-concentration cap, turnover budget -- chosen from the evidence; measured ON-vs-OFF via the $0 replay/backtest on the production universe reporting Sharpe/return/turnover/maxDD; subject to the SAME Ledoit-Wolf SR-difference robustness gate as 52.3/53.1 (p<0.05 AND delta>=+0.05 AND CI_low>0); config-gated default-off; US momentum core byte-identical unless the flag is enabled; re-proposing the 53.1-rejected no-trade band in naive or renamed form is an automatic FAIL) and a FEATURE variant (the top capability gap from the strategic chapter, e.g. full-mode agents in the autonomous path, per-market benchmark fetches (^KS11), concentration limits as a feature -- with measurable acceptance criteria of the same rigor)",
            "an OPERATOR DECISION block is posted to the operator Slack channel: LLM burn estimate ($/cycle from llm_call_log cost columns, fallback token counts x current pricing, x planned cycles over a 1-2 week window), expected value (which of DoD-2/5/6/7/9 close; projected go-live-gate delta from baseline 1/5), the finetune-vs-features recommendation, and the verbatim reply grammar 'LLM SPEND: APPROVED <budget> | DECLINED' + 'PHASE-57: LEVER | FEATURE'; phase-56 may start once phase-55 is done, but phase-57 installation and any phase-58 live cycle are HARD-gated on the verbatim replies"
          ],
          "live_check": "REQUIRED -- the ranked findings table + the research-gate JSON envelope + the Slack message timestamp of the operator decision block."
        },
        "retry_count": 0,
        "max_retries": 3
      }
    ]
  },
  {
    "id": "phase-56",
    "name": "Data-correctness + ops fixes, gated on the phase-55 review (no fix without a 55.x finding ID). Restores a trustworthy P&L readout (FX/value/fee correctness for non-USD markets) and repairs the operator-facing ops defects surfaced by the away week (approve flow, watchdog, degraded-scoring guard, kill-switch verdict follow-up, test hygiene).",
    "status": "pending",
    "depends_on": ["phase-55"],
    "gate": null,
    "steps": [
      {
        "id": "56.1",
        "name": "FX/value/fee data-correctness fix -- persist total_value and SELL transaction_cost in USD for non-USD markets (paper_trader.py:265, :386-414), correct the trades-columns.tsx:11 comment and the VS-KOSPI handling per the 55.1 verdict, fix the NAV-discrepancy root cause identified by 55.1, and present any historical-row backfill as an operator-gated migration.",
        "status": "pending",
        "harness_required": true,
        "priority": "P0",
        "depends_on_step": "55.3",
        "audit_basis": "Code-confirmed defects (2026-06-10 trace): paper_trader.py:265 total_value missing * _local_to_usd; :386-414 SELL fee unconverted; trades-columns.tsx:11 comment false for non-USD; cockpit-helpers.tsx:197-218 VS-KOSPI is holdings return not index excess. Screenshot: NAV 345,968.86 USD on a $10K fund. The exact NAV fix target comes from the 55.1 root-cause trace; the criteria below are finding-ID-driven so they bind to evidence, not guesses.",
        "verification": {
          "command": "cd /Users/ford/.openclaw/workspace/pyfinagent && source .venv/bin/activate && python -m pytest backend/tests -k 'fx or paper_trader or krw' -q && test -f handoff/current/live_check_56.1.md",
          "success_criteria": [
            "total_value and SELL transaction_cost are persisted in USD for non-USD markets (the paper_trader.py:265 and :386-414 paths), covered by a unit test with a KRW fixture that FAILS on the pre-fix code and PASSES on the fixed code (regression-proof)",
            "the NAV-discrepancy root cause identified by 55.1 (finding ID cited) is fixed or, if it is data-only (no code defect), the correction path is specified; the live /paper-trading UI shows sane Value/Fee/NAV/Cash for KR rows, evidenced by a Playwright capture in live_check_56.1.md; the trades-columns.tsx:11 comment and the VS-KOSPI handling are corrected per the 55.1 verdict (true index excess via ^KS11 or an honest relabel)",
            "correction/backfill of historical corrupted BQ rows is executed ONLY as an operator-approved migration script under scripts/migrations/ (destructive ops are operator-gated); if the operator declines, the corrupted rows are flagged (not silently kept) and the audit-trail caveat is documented",
            "every change in this step cites a 55.x finding ID; fixing anything WITHOUT a finding ID is a FAIL; the finding-ID -> fix mapping is recorded in live_check_56.1.md; the US momentum core paths are untouched (do-no-harm)"
          ],
          "live_check": "REQUIRED -- the finding-ID -> fix map + the KRW-fixture test output + a Playwright capture of the corrected /paper-trading page."
        },
        "retry_count": 0,
        "max_retries": 3
      },
      {
        "id": "56.2",
        "name": "Ops fixes -- repair the Slack approve flow, add a degraded-scoring guard (all-0.0 cycle alerts instead of passing silently), apply the watchdog ReadTimeout fix per the 55.2 root cause, follow up the kill-switch verdict, and quarantine the 16 env-coupled backend test failures.",
        "status": "pending",
        "harness_required": true,
        "priority": "P1",
        "depends_on_step": "56.1",
        "audit_basis": "Slack 06-01: approve flow broken ('Missing API key for provider anthropic') -- the operator could not approve from his only away-week window. 05-28: degraded scoring (all 0.0/10) passed silently. Watchdog ReadTimeouts recurred 3 nights. cycle_block_summary.md: 16 env-coupled backend test failures (live-BQ probes + moved fixture-doc) to quarantine. Kill-switch follow-up depends on the 55.1 verdict (verdict-neutral here).",
        "verification": {
          "command": "cd /Users/ford/.openclaw/workspace/pyfinagent && source .venv/bin/activate && python -m pytest backend/tests -q && test -f handoff/current/live_check_56.2.md",
          "success_criteria": [
            "every finding ranked P0/P1 in the 55.3 table is either FIXED with a regression test or explicitly ESCALATED as operator-gated, with the finding-ID map recorded in live_check_56.2.md",
            "the Slack approval path is exercised end-to-end: typing 'Approve' in the operator channel no longer yields 'Missing API key for provider anthropic' (captured transcript in the live_check), or the residual is escalated with a one-line operator action",
            "a degraded-scoring guard exists: a cycle whose analyses all score 0.0 (or whose scoring backend is unavailable) is detected and alerted to Slack instead of passing silently, covered by a unit test; the watchdog ReadTimeout fix or a bounded escalation is applied per the 55.2 root cause",
            "the kill-switch defect is fixed with a unit test reproducing the 06-05 scenario IFF 55.1 ruled SHOULD-HAVE-TRIPPED; any threshold change is presented as an OPERATOR DECISION, never auto-applied; the 16 env-coupled backend test failures are quarantined (skip-markers + reason strings) and backend pytest is green"
          ],
          "live_check": "REQUIRED -- the approve-flow transcript + the degraded-scoring guard test output + the pytest summary line showing green-with-quarantine."
        },
        "retry_count": 0,
        "max_retries": 3
      }
    ]
  },
  {
    "id": "phase-58",
    "name": "Go-live runway resumption -- operator-spend-gated. Re-scores the live-blocked DoD criteria if the operator approves a budgeted live window, or honestly parks them and closes the $0 remainder if declined. Either branch produces an updated go-live-gate readout and a refreshed cycle_block_summary.md. CLOSES goal-post-away-review. (Phase-57, the evidence-selected improvement, is installed separately from the 55.3 payload the operator picks; 58 depends on phase-56, not on 57.)",
    "status": "pending",
    "depends_on": ["phase-56"],
    "gate": null,
    "steps": [
      {
        "id": "58.1",
        "name": "Go-live runway -- record the operator's verbatim spend decision; APPROVED: deploy fixes, verify kill switch, operator re-enables the mas-harness cron, run the budgeted live window, re-score DoD-2/5/6/7/9 with evidence; DECLINED: park live-blocked criteria honestly, close the $0 remainder (ablation exit=1 triage, autoresearch pip escalation note, test-quarantine verification). Either branch: go-live-gate readout + refreshed cycle_block_summary.md.",
        "status": "pending",
        "harness_required": true,
        "priority": "P2",
        "depends_on_step": "56.2",
        "audit_basis": "phase-43.0 audit (2026-06-01): NOT_PRODUCTION_READY, backend 8/14, UX 0/12, with DoD-2/5/6/7/9 LIVE-BLOCKED on operator LLM spend (cycle_block_summary.md). Go-live gate baseline 1/5 (06-01 digest). Operator decision 2026-06-10: spend decided after the review reports. mas-harness cron was booted during the away week (handoff-file collision) and must be re-enabled by the operator at HARD STOP: launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.pyfinagent.mas-harness.plist.",
        "verification": {
          "command": "cd /Users/ford/.openclaw/workspace/pyfinagent && test -f handoff/current/live_check_58.1.md && grep -E 'LLM SPEND: (APPROVED|DECLINED)' handoff/current/live_check_58.1.md",
          "success_criteria": [
            "the operator's verbatim spend reply ('LLM SPEND: APPROVED <budget>' or 'LLM SPEND: DECLINED', with date) is recorded in live_check_58.1.md BEFORE any live LLM trading cycle runs; running a live cycle without it is an automatic FAIL",
            "if APPROVED: the phase-56 fixes are deployed and the kill switch is verified ACTIVE before the window starts; the mas-harness cron re-enable is confirmed by the operator (command echoed in the live_check); each of DoD-2/5/6/7/9 is re-scored on its own criterion with evidence from the live window, and the DoD scorecard delta from the 2026-06-01 baseline (8/14 backend, 0/12 UX) is reported",
            "if DECLINED: DoD-2/5/6/7/9 remain honestly LIVE-BLOCKED (no synthetic substitutes); the $0-closable remainder is closed or escalated (ablation exit=1 triage, autoresearch pip-install escalation note, verification that the 56.2 test quarantine still holds)",
            "either branch: an updated go-live-gate eligibility readout (baseline 1/5 on 06-01) + a refreshed handoff/current/cycle_block_summary.md with a crisp operator ask list; this step CLOSES goal-post-away-review"
          ],
          "live_check": "REQUIRED -- the verbatim operator reply + the branch evidence (DoD re-scores or honest parking) + the refreshed cycle_block_summary.md."
        },
        "retry_count": 0,
        "max_retries": 3
      }
    ]
  }
]
```

## Phase-57 required shape (NOT installed now -- authored by 55.3)

55.3 emits two install-ready candidates; the operator picks ONE via `PHASE-57: LEVER` or
`PHASE-57: FEATURE`. Both candidates MUST:
- conform to the full masterplan step schema (id 57.1, all fields, retry_count 0,
  max_retries 3) with `"depends_on": ["phase-55"]` at phase level;
- carry immutable criteria of the same rigor as 53.1 (measure-first, ON-vs-OFF, the
  Ledoit-Wolf SR-difference robustness gate for the LEVER variant; measurable acceptance
  criteria + do-no-harm for the FEATURE variant);
- be config-gated and default-off; NO live flag flip inside phase-57;
- cite the 55.x finding IDs and research-brief sources that motivated them.
The install commit message records the operator's verbatim `PHASE-57:` reply.

## active_goal.md refresh payload (replace the file with this content at install)

```markdown
# Active Goal -- Post-away-week review -> evidence-gated fixes -> go-live runway

Set by operator 2026-06-10. Supersedes the 2026-06-01 goal, whose autonomous scope
completed on main the same day (53.5 e2e-smoke capstone shipped; 53.4 deferred by the
operator). Full specification: handoff/current/goal_post_away_review.md (the goal prompt).

## North star
(verbatim masterplan.json::goal) "Ship an Intelligence Engine trading system that
maximizes Net System Alpha = Profit - (Risk Exposure + Compute Burn) by dynamically
shifting capital to the highest-earning strategy, recursively self-improving under hard
risk caps, within a 15-slot daily Claude-routine budget." THIS GOAL'S LENS: you cannot
maximize what you cannot measure -- the away week showed the P&L readout itself is
corrupted for non-USD markets; measurement integrity comes first.

## Scope -- in order
1. phase-55 -- away-week forensic review (55.1 data/trading, 55.2 ops/skills,
   55.3 synthesis + operator checkpoint). $0; NO fixes.
2. OPERATOR CHECKPOINT -- replies: 'LLM SPEND: APPROVED <budget> | DECLINED' +
   'PHASE-57: LEVER | FEATURE'
3. phase-56 -- data-correctness + ops fixes (56.1 FX/value/fee + NAV root cause,
   56.2 approve flow / degraded-scoring guard / watchdog / kill-switch follow-up /
   test quarantine). No fix without a 55.x finding ID.
4. phase-57 -- evidence-selected improvement (installed from the 55.3 payload the
   operator picked; not pre-installed).
5. phase-58 -- go-live runway resumption (spend-gated; CLOSES this goal).

## Founding principles (non-negotiable)
- Full harness loop per step: researcher FIRST (>=5 sources read in full, recency
  scan) -> contract.md (criteria verbatim) -> GENERATE -> ONE fresh qa -> harness_log.md
  append -> masterplan flip. No self-evaluation; no verdict-shopping.
- REVIEW BEFORE FIX: phase-56+ steps cite 55.x finding IDs or they FAIL.
- DO-NO-HARM: the US pure-quant momentum core (+20% NAV) stays byte-identical unless a
  config flag is explicitly enabled; no live flag flips inside this goal.
- NO NAIVE NO-TRADE BAND: 53.1's Ledoit-Wolf REJECT is binding.
- UI claims are verified in the live UI via Playwright MCP (NextAuth wall); captures go
  in the live_checks.

## Effort policy
Main xhigh; Researcher + Q/A max (CLAUDE.md effort policy unchanged).

## Done-definition (HARD STOP)
- 58.1 closed (either spend branch) + phase-57 variant installed and executed or
  explicitly deferred by the operator + cycle_block_summary.md refreshed with a crisp
  operator ask list. Write the summary and stop.

## Constraints / gates
- OPERATOR-GATED: LLM API spend (live cycles), pip installs, BQ DROP / unqualified
  DELETE / historical-row backfill, launchctl changes.
- Local runs are main-based: merge claude/sweet-feynman-zhs8p3 before starting.

## Stop conditions
- SOFT STOP: 12 cycles elapsed OR a blocker needing the operator -> summary + crisp ask.

## Cycle ledger (this run)
- (appended per cycle)
```

## CLAUDE.md amendment payload (add ONE bullet under "Critical Rules" at install)

```markdown
- **UI verification via Playwright MCP (goal-post-away-review, 2026-06-10)** -- whenever
  the operator pastes a UI screenshot, or any step makes a claim about the UI, ALWAYS
  verify against the RUNNING app via the Playwright MCP (browser_navigate +
  browser_snapshot / browser_take_screenshot) behind the NextAuth wall. Code reading
  alone is not UI evidence. Every UI-touching live_check includes a Playwright capture.
```

## Review tooling (cite in the phase-55 contracts)

| Tool | Use |
|------|-----|
| GET /api/paper-trading/status, /portfolio, /trades, /snapshots | NAV path, positions, trade log (backend/api/paper_trading.py) |
| GET /api/paper-trading/performance | win_rate, profit_factor, expectancy, median_holding_days, mfe/mae |
| GET /api/paper-trading/metrics-v2 | PSR, DSR, Sortino, Calmar, bootstrap CI on rolling Sharpe |
| GET /api/paper-trading/round-trips, /attribution, /cycles/history | paired round trips, signal rationale, cycle outcomes |
| scripts/harness/paper_execution_parity.py | backtest <-> paper parity |
| scripts/risk/tca_report.py | trade-cost analysis |
| BQ financial_reports.paper_trades / paper_positions / paper_portfolio / paper_portfolio_snapshots | primary trade/NAV data (note: paper-trading tables live in financial_reports, NOT pyfinagent_pms) |
| BQ pyfinagent_data.llm_call_log, strategy_decisions | skill fire counts, burn estimate, cycle heartbeats |
| Playwright MCP | live UI evidence (/paper-trading behind NextAuth) |

## Files in scope

| File | Change |
|------|--------|
| .claude/masterplan.json | install phase-55/56/58 payload (commit 1); phase-57 at checkpoint |
| handoff/current/active_goal.md | replace with the refresh payload (commit 1) |
| CLAUDE.md | add the Playwright bullet (commit 1) |
| backend/services/paper_trader.py | 56.1 fix targets (:265, :386-414; NAV root cause per 55.1) |
| frontend/src/components/paper-trading/trades-columns.tsx, cockpit-helpers.tsx | 56.1 display/benchmark corrections |
| backend/slack_bot/* (approve path), watchdog module, scoring guard | 56.2 targets (exact files per 55.2 root causes) |
| scripts/migrations/ | operator-gated backfill migration (56.1, if approved) |
| handoff/current/cycle_block_summary.md | refreshed at 58.1 / HARD STOP |

## References (read before PLAN)

- CLAUDE.md (harness protocol, effort policy, BQ access, commit conventions)
- .claude/rules/research-gate.md (researcher floor: >=5 sources, 3-pass search, recency)
- docs/runbooks/per-step-protocol.md (the five-file protocol)
- handoff/archive/goal-market-filter-in-gate-bar/goal_market_filter_in_gate_bar.md
  (goal-prompt template precedent)
- handoff/current/live_check_53.1.md + commit 675e69d (Ledoit-Wolf robustness gate +
  honest-REJECT precedent the phase-57 LEVER variant must reuse)
- handoff/current/production_ready_audit_2026-06-01.md (DoD baseline 8/14, 0/12)
- handoff/current/cycle_block_summary.md (operator-gated follow-ups folded into 56/58)
- Slack #ford-approvals (C0ANTGNNK8D) digests 2026-05-27 .. 2026-06-10 (SECONDARY
  evidence; reconcile in 55.1)
