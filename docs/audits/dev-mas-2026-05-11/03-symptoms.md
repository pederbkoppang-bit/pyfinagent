# Phase 3 — Symptom Trace

Audit date: 2026-05-11. For each of the three application-visible
symptoms, the evidence trace runs **before** the verdict. Verdicts use
the prescribed labels: `SCOPING_GAP`, `PLAN_DEFECT`, `SPEC_DEFECT`,
`IMPLEMENTATION_DEFECT`, `VERIFICATION_DEFECT`, `INSUFFICIENT_EVIDENCE`.

Phase 1 reference: `01-roster.md`. Phase 2 reference: `02-per-agent.md`.

---

## Symptom 1 — "No automatic sell on threshold breach — feature never shipped"

### Evidence trace

#### 1.1 Masterplan history

Steps with status, name, and verification criteria, found by grepping
`.claude/masterplan.json` for "stop_loss / kill_switch / threshold /
flatten":

| Step | Status | Name | Source |
|---|---|---|---|
| 4.5.7 | `done` | "Kill-switch v2 — Pause / Resume / Flatten-all + daily loss + trailing DD limits" | `.claude/masterplan.json:812-825` |
| 3.7.3 | `done` | "New risk_server.py Risk Agent MCP (kill_switch + PBO veto)" | `.claude/masterplan.json:561` |
| 4.7.6 | `done` | "WCAG 2.1 AA + keyboard-only kill-switch workflow" | `.claude/masterplan.json:1184-1191` |
| 4.7.7 | `done` | "Virtual-fund learnings dashboard (reconciliation / kill-switch / regime / MFE-MAE)" | `.claude/masterplan.json:1200-1208` |
| 10.7 | `done` | "Rollback kill-switch wiring to phase-4.8.5" | `.claude/masterplan.json:2886` |
| 16.6 | `done` | "Kill switch + risk guards drill" | `.claude/masterplan.json:5278-5289` |
| 16.19 | `done` | "Trading mechanics drills (alpaca shadow + kill switch + zero-orders)" | `.claude/masterplan.json:5514-5528` |
| **23.1.8** | `done` | "Positions table reactivity (live MV+P&L) + **Stop Loss settings-driven default (8% O'Neil canon)**" | `.claude/masterplan.json` |
| 23.1.22 | `done` | "kill_switch reentrant-lock deadlock fix" | `.claude/masterplan.json:7160-7213` |
| **23.2.5** | **`pending`** | **"Verify kill-switch breach evaluation never falsely fired"** | `.claude/masterplan.json:7217-7221` |

Notable verification criteria already met:

- Step 4.5.7 (`masterplan.json:818-821`): criteria include
  `"flatten_all_endpoint_closes_every_open_position"` and
  `"breach_triggers_auto_flatten_and_pause"`. Both marked done.
- Step 16.6 (`masterplan.json:5280-5281`): criteria include
  `"kill_switch_test.py exits 0"` and `"pause -> flatten -> resume
  transitions all produce audit rows"`.
- Step 16.6 `notes` (`masterplan.json:5289`): "Kill switch state
  machine: pause+resume events captured in
  `handoff/kill_switch_audit.jsonl`. Current state `is_paused=False`."
- The PENDING step 23.2.5 is specifically about **verifying the
  inverse failure** ("never falsely fired") — not "fires when it should".

#### 1.2 Code present in `main`

`backend/services/portfolio_manager.py:80-91` implements per-position
stop-loss SELL:

```python
# Stop loss check (already priced in mark-to-market)
stop = pos.get("stop_loss_price")
current = pos.get("current_price", 0)
if stop and current and current <= stop:
    orders.append(TradeOrder(
        ticker=ticker, action="SELL", reason="stop_loss",
        price=current,
    ))
    continue
```

This runs every daily cycle, executed by
`backend/services/autonomous_loop.py` calling `decide_trades`.

`backend/services/portfolio_manager.py:97-99` implements the
explicit-SELL-signal path (recommendation downgrade → SELL):

```python
if rec in _SELL_RECS:
    orders.append(TradeOrder(
        ticker=ticker, action="SELL", reason="sell_signal", ...))
```

`backend/services/portfolio_manager.py:106-115` implements the
**downgrade** path (BUY → HOLD/SELL → auto-SELL with
`reason="signal_downgrade"`).

`backend/services/kill_switch.py` (referenced at
`_inventory.json:79`) implements the portfolio-level kill switch +
`flatten_all` + pause/resume state machine.

#### 1.3 Git log

Two relevant commits:

- `c383de65` "feat: add trailing stop loss to daily MTM loop (Phase
  1.5)" — trailing stop landed.
- `2960091d` "fix: disable vol targeting + trailing stops defaults,
  remove from optimizer search, reset best params to triple_barrier
  Sharpe 0.98" — trailing stops were intentionally **disabled by
  default** for the *backtest optimizer search* (not for paper
  trading), because they degraded Sharpe.
- `be3accb7` "Phase 4.3 Risk Management: size_position +
  check_stop_loss + track_drawdown" — first appearance of
  `check_stop_loss`.

#### 1.4 What the user's symptom likely refers to

Two plausible reads of "auto-sell on threshold breach":

- **Read A — per-position stop-loss**: the code at
  `portfolio_manager.py:80-91` IS this feature. Default 8% per
  23.1.8. **Shipped.**
- **Read B — portfolio-level kill-switch breach auto-flatten**:
  `kill_switch.py` + 4.5.7's `breach_triggers_auto_flatten_and_pause`
  criterion. **Shipped.**
- **Read C — RiskJudge decides "REJECT" on a held position during
  re-evaluation, triggering an exit**: NO direct evidence this is
  wired. `signal_attribution.py:295` ("elif agent == 'RiskJudge' or
  role == 'gate':") only labels, it doesn't trigger sells.
  `portfolio_manager.py:140-160` (Buy decisions) does extract
  `risk_judge_position_pct` for sizing on NEW buys, but there's no
  path where a re-evaluation's `risk_judge_decision == "REJECT"` →
  forced SELL. The downgrade path (line 106-109) handles `rec in
  _DOWNGRADE_RECS`, which is the *trade recommendation*, not the
  *risk judge decision*. Read C **may** not be shipped.

#### 1.5 Verification gap

No masterplan step in the active scope asserts "live-system: set
stop_loss_price, drop ticker price below it, observe SELL trade
emitted with reason='stop_loss', confirm trade row appears in
`paper_trades` BQ table." The closest is the kill-switch drill
(16.6) which exercises pause/flatten/resume but not the per-position
stop-loss path. 23.2.5 (PENDING) is the **inverse** check (no
false-fires), not the **forward** check (fires when it should).

### Verdict for Symptom 1: `INSUFFICIENT_EVIDENCE` with documented lean toward `VERIFICATION_DEFECT` + `SCOPING_GAP`

Reasoning:
- **Reads A and B are SHIPPED with masterplan steps marked `done`,
  code in `main`, and code-level criteria asserted.** The user's
  claim "feature never shipped" is empirically wrong for those reads.
- **Read C** (RiskJudge `REJECT` on re-eval → forced SELL) has no
  positive evidence of a code path. That is a true `SCOPING_GAP` if
  this is what the user meant.
- **End-to-end live verification of Reads A and B is absent** from
  the masterplan. Drill 16.6 tests the kill-switch machinery; no
  drill tests "real ticker drops below stop_loss_price → trade is
  recorded with `reason='stop_loss'`." That is a
  `VERIFICATION_DEFECT`.

The auditor cannot disambiguate Reads A/B/C without operator input
(which specific threshold? which UI surface?) and so returns
`INSUFFICIENT_EVIDENCE` for the headline verdict, with the two
contingent diagnoses spelled out.

**Repo-evidence citations**: `.claude/masterplan.json:812-825`
(step 4.5.7 done), `:7217-7221` (step 23.2.5 pending),
`backend/services/portfolio_manager.py:80-91` (stop-loss code),
`backend/services/kill_switch.py` (kill switch),
git commits `c383de65`, `2960091d`, `be3accb7`.

---

## Symptom 2 — "BUY card shows only 3 rationale rows (Quant, Trader, RiskJudge) despite 28 Layer-1 skills"

### Evidence trace

#### 2.1 The dev MAS already audited this symptom

The file `handoff/current/phase-23.2.A-agent-rationale-audit.md`
(generated 2026-04-29 by the Researcher subagent, internal-only
audit) reports the **identical observation** in section A and section
B:

- Section A enumerates **28 Layer-1 agents** in
  `orchestrator.py` (Quant, RAG, Market, Competitor, Insider,
  Options, Social, Patent, EarningsTone, Macro, AltData, Sector,
  NLP, Anomaly, Scenario, QuantModel, InfoGap, Bull, Bear, DevilsAdvocate,
  Moderator, Synthesis, Critic, BiasDetector, ConflictDetector,
  Aggressive, Conservative, Neutral, RiskJudge).
- Section B (sampled 3 live BUY trades from BQ on 2026-04-29):
  every BUY shows exactly **3 signal rows: Quant, Trader,
  RiskJudge**.

`handoff/harness_log.md:14468` records the audit conclusion verbatim:

> "25 of 28 Layer-1 agents leave no trace in drawer (the 11
> enrichment + 4 debate agents are in `enrichment_signals` but
> `signal_attribution.py` only reads 4 keys: analyst_summary,
> debate, trader_note, risk_assessment)."

#### 2.2 Root cause from the audit (section C1)

`backend/services/signal_attribution.py::extract_signals_from_analysis`
reads four keys from the `analysis` dict:

1. `analysis.get("analyst_summary") or analysis.get("synthesis")` →
   Analyst row.
2. `analysis.get("debate")` → Bull / Bear rows.
3. `analysis.get("recommendation")` + `analysis.get("trader_note")`
   → Trader row.
4. `analysis.get("risk_assessment")` → RiskJudge row.

The `analysis` dict passed in the **autonomous trading cycle** is
the **lite analysis output** produced by `autonomous_loop.py`, which
uses a Claude-based lite analyzer. That lite output has no
`analyst_summary`, no `debate`, only `trader_note` + `risk_assessment`.

So **only Trader + RiskJudge come from the analysis dict**. The Quant
row comes from `extract_quant_signals(candidate)` reading the screener
candidate dict directly (`signal_attribution.py:160-242`). Total: 3
rows. Matches observed behavior exactly.

#### 2.3 Why lite path is the default

`backend/agents/orchestrator.py:1014` reads `self.settings.lite_mode`
(boolean); `backend/api/settings_api.py:78` exposes
`lite_mode: bool`. Sections of the orchestrator pipeline are skipped
when `lite_mode=true`:

- `orchestrator.py:1443-1445`: "Skipping deep dive (lite mode)"
- `orchestrator.py:1530-1532`: "Skip risk assessment if data quality
  is below threshold or lite mode"

The autonomous paper-trading cycle uses the lite path by default for
cost reasons (auto-memory project context:
`local_only_deployment.md`, "Claude Max flat-fee"). The expensive
full pipeline (~$0.08-0.15/analysis) was de-scoped to the
on-demand "/analyze ticker" path, not the autonomous cycle.

#### 2.4 The user's expectation vs the design

The user expects **28 rationale rows** because the system documents
"28 Layer-1 agents" (CLAUDE.md "Analysis Pipeline (28 Gemini
agents)", ARCHITECTURE.md:345). The shipped design renders **a
progressive-disclosure 5-layer tree** (Analyst / Debate / Quant /
Trader / Risk) from TradingAgents (Xiao et al., 2024), where Layer-1
agents are AGGREGATED into those 5 layers, not surfaced individually.

This is a **documentation gap**: the rationale-drawer design and the
"28 agents" doc reference live in different paragraphs of the same
project. A reader who reads the "28 agents" doc and then opens the
drawer is set up to be confused.

#### 2.5 Plan, spec, implementation, verification trace

- **Plan**: `handoff/harness_log.md:948-952` records phase 4.5.5
  ("Agent-rationale drawer + per-trade signal attribution pipeline")
  with research citing TradingAgents (Xiao et al., 2024). Plan
  envisioned the 5-layer progressive-disclosure tree, NOT the
  28-row enumeration.
- **Spec**: phase-4.5.5 contract (archived) names
  `signal_attribution.py` and the `{agent, role, rationale, weight}`
  shape. The spec is consistent with the plan.
- **Implementation**: `signal_attribution.py` (277 lines) + frontend
  `AgentRationaleDrawer.tsx`. Implementation is consistent with the
  spec.
- **Verification**: tests in `tests/services/test_signal_attribution.py`
  pass for the 5-layer-tree shape. Tests do NOT assert "all 28
  agents are surfaced" because that was never the spec.
- **Production reality**: Lite path is the default, which collapses
  the 5-layer tree to 3 rows (Quant + Trader + RiskJudge); plan +
  spec + impl + verification all consistent with this.

### Verdict for Symptom 2: `SCOPING_GAP`

The dev MAS plan → spec → impl → verify chain is **internally
consistent**. The drawer was designed as a 5-layer progressive-disclosure
tree, not a 28-row enumeration. The "28 Layer-1 skills" framing in
ARCHITECTURE.md and CLAUDE.md is a separate doc-paragraph that an
operator can reasonably interpret as "I should see 28 rationale rows
in the BUY card."

The gap is therefore a **SCOPING_GAP at the documentation /
expectation boundary**, not a defect in any single plan or impl
stage. The dev MAS never scoped "surface all 28 Layer-1 enrichment
outputs to the operator UI" as a goal, because the drawer's
progressive-disclosure pattern is doing intentional aggregation. The
already-shipped phase-23.2.A audit explicitly flagged this as a gap
("Option C — route the full 28-agent Gemini pipeline for BUY
candidates above a composite_score threshold ... ~8h.").

**Cross-reference**: phase-23.2.A-audit.md proposed three options
A/B/C; **Option B shipped** (lite-path RiskJudge relabel — see
Symptom 3 below); **Option C deferred**. Operator's continued
surprise at "only 3 rows" is the predictable outcome of deferring
Option C.

**Repo-evidence citations**:
`handoff/current/phase-23.2.A-agent-rationale-audit.md:1-216` (the
audit itself), `handoff/harness_log.md:14468` (verbatim conclusion),
`backend/services/signal_attribution.py:57-157` (root-cause code),
`backend/agents/orchestrator.py:1014, 1443, 1530-1532` (lite-mode
branches), git commit `e7e09d77` "phase-23.1.7: capture full agent
rationale + signal stack into paper_trades.signals" (the
implementation commit).

---

## Symptom 3 — "RiskJudge rationale is verbatim identical to Trader's, weight 0.00, labeled '(gate)', shipped to a production-visible card"

### Evidence trace

#### 3.1 The dev MAS already FIXED this — for new trades

Commit `ad9d773c` (2026-05-04, "phase-23.2.2-fix + 23.2.A-fix: STX
cleanup + drawer Option B"). The 23.2.A-fix portion of the commit:

> "drawer 'Option B' — detect lite-path duplicate Risk Judge
> (weight=0.0 AND rationale equals Trader rationale). Replace with
> 'Lite-path: Risk Judge inherited Trader's reasoning; no independent
> risk debate ran' message + amber lite_path badge."

Code in `backend/services/signal_attribution.py:131-155`:

```python
# phase-23.2.A-fix: Option B -- detect lite-path duplicate.
trader_rationale_trimmed = _trim(trader_note) or f"Recommendation: {rec}"
is_lite_dup = (
    risk_weight == 0.0
    and risk_rationale == trader_rationale_trimmed
)
entry = {"agent": "RiskJudge", "role": "gate",
         "rationale": risk_rationale, "weight": risk_weight}
if is_lite_dup:
    entry["rationale"] = (
        "Lite-path: Risk Judge inherited Trader's reasoning; "
        "no independent risk debate ran for this analysis."
    )
    entry["lite_path"] = True
signals.append(entry)
```

Three new tests at `tests/services/test_risk_judge_lite_path.py`
(73 lines) cover (a) lite-path duplicate detection, (b) full-path
unchanged, (c) lite-path with distinct reasoning not flagged. All
pass.

Frontend renders the relabel:
`frontend/src/components/AgentRationaleDrawer.tsx:12-15, 168-182`:

```tsx
// phase-23.2.A-fix: when true, this Risk Judge row is the lite-path
// duplicate (no independent risk debate ran; reasoning inherited from
// Trader). Drawer renders an amber "lite-path" badge.
lite_path?: boolean;
... // at line 168:
{s.lite_path ? (<span ...>lite-path</span>) : ...}
```

#### 3.2 Why the symptom may still be visible to the user

`extract_signals_from_analysis` is called at **trade-write time**
(`portfolio_manager.py:96` and `109` in the SELL branch; the BUY
branch calls `extract_all_signals(analysis, candidate)` at line
~174, which calls `extract_signals_from_analysis` internally). The
`signals` JSON is persisted into the `paper_trades.signals` BQ column.

The rationale-drawer API endpoint at
`backend/api/paper_trading.py` returns the **stored signals**, not
re-extracted from the analysis. So **any paper-trade row created
before 2026-05-04 21:08 retains the OLD signals shape** (RiskJudge
rationale = Trader rationale verbatim, no `lite_path` flag, no
relabel). A trade row from before that date will still show the
exact symptom the user described.

#### 3.3 Plan, spec, implementation, verification trace

- **Plan**: identified by the Researcher subagent in
  `handoff/current/phase-23.2.A-agent-rationale-audit.md` (2026-04-29).
  Section D's verdict: "`sufficient_as_designed: false`." Section
  "Proposed Phase-2 Fix" listed Options A/B/C with effort estimates.
- **Spec / decision**: commit `ad9d773c` (2026-05-04) chose Option B
  ("detect lite-path duplicate ... replace with 'Lite-path: Risk
  Judge inherited Trader's reasoning ... ' message + amber lite_path
  badge"). Spec evident in the commit body and the new test file's
  docstrings.
- **Implementation**: `signal_attribution.py:131-155` +
  `AgentRationaleDrawer.tsx:12-15, 168-182`. ~29 lines added in
  signal_attribution.py + ~16 lines in AgentRationaleDrawer.tsx
  (commit stat).
- **Verification**: three unit tests in
  `test_risk_judge_lite_path.py` + frontend `tsc --noEmit` clean.
  Q/A did not appear to verify against the production BQ table
  (no "I checked the next 5 paper trades in BQ and confirmed the
  relabel" entry in the cycle's harness_log).
- **Production reality**: NEW trades created after 2026-05-04 21:08
  CET should show the relabeled RiskJudge row. OLD trades retain
  the duplicate. If the user sampled OLD trades (likely — paper
  trading shows historical rows), they see the unfixed symptom.

### Verdict for Symptom 3: `VERIFICATION_DEFECT`

The dev MAS plan → spec → impl → verify chain is **mostly complete**:
the audit found the bug, the fix landed, unit tests cover it. The
defect is in the verification stage: **no live-production check**
that NEW trades show the relabel, and **no back-migration / replay
path** for OLD trades. The auditor would expect a Q/A cycle to
include:

1. After fix lands, force one paper-trade cycle and confirm new
   trade's `signals` JSON contains `lite_path: true` for the
   RiskJudge row.
2. Decide and document whether to back-migrate the BQ
   `paper_trades.signals` column, OR to expose a "re-render
   rationale" endpoint that re-extracts from stored `analysis_json`.

Neither happened. The unit tests verify the function on synthetic
inputs only.

This is a **classic harness self-evaluation symptom** per
HARNESS-DOC: "agents tend to respond by confidently praising the
work — even when, to a human observer, the quality is obviously
mediocre." Three unit tests + tsc clean = "PASS"; the live BQ
sample never gets opened.

**Repo-evidence citations**:
`handoff/current/phase-23.2.A-agent-rationale-audit.md` (audit),
git commit `ad9d773c` 2026-05-04 (fix),
`backend/services/signal_attribution.py:131-155` (fix code),
`tests/services/test_risk_judge_lite_path.py:1-74` (tests),
`frontend/src/components/AgentRationaleDrawer.tsx:12-15, 168-182`
(rendering), absence of a "BQ replay verified" line in
`handoff/harness_log.md` for cycle bordering 2026-05-04.

---

## Systemic pattern across the three symptoms

A common pattern emerges:

| Symptom | Stage that failed | Stage(s) that passed |
|---|---|---|
| 1 — Auto-sell on threshold breach | Verification (no live-system end-to-end check) + Scoping (Read C unclear) | Plan, spec, implementation (for reads A, B) |
| 2 — 3-of-28 rationale rows | Scoping (drawer design vs "28 agents" doc never reconciled) | Plan, spec, implementation, unit verification |
| 3 — RiskJudge=Trader duplicate | Verification (unit-tests pass; live BQ replay never run) | Plan, spec, implementation |

**The dominant failure mode is VERIFICATION_DEFECT, specifically: unit
tests pass on synthetic inputs, but no step in the harness cycle opens
the production data + UI to confirm the user-visible behavior.**

Three contributing root causes from the per-agent audit (Phase 2):

- **Q-5**: Q/A's "Anti-rubber-stamp ... mutation-resistance test" is
  documented but not hook-enforced; Main can skip it under pressure.
- **H-1, H-2**: TaskCompleted hook is a redundant evaluator with
  unconstrained tool access; its `ok:true` returns can short-circuit
  Main's perception of completion before Q/A's rigorous LLM-judgment
  leg runs.
- **C-A4**: Research-gate is behavioral, not hook-enforced. When
  Main skips researcher under pressure, the resulting contract.md
  lacks the citations that would surface "did we verify on real
  data?" as a contract criterion.

A secondary pattern: **Documentation surface drift.** The "28 Layer-1
agents" framing in CLAUDE.md and ARCHITECTURE.md sets an operator
expectation that the 5-layer progressive-disclosure drawer cannot
satisfy. The dev MAS never had a single canonical paragraph that
explained "you see aggregated layers, not 28 rows, AND in lite mode
even fewer." This makes Symptom 2 visible as a "bug" when it is, by
implementation choice, a deliberate design — and visible as such
documentation gap, not a code gap.

A tertiary pattern (cross-cutting Layer-2 vs Layer-3): **the
in-app Layer-2 dev agents (Ford, in-app Researcher, in-app Q&A) share
labels with the Layer-3 harness subagents (Main, researcher.md,
qa.md)**. This is Phase 1's namespace-collision finding (F-1, C-2).
For symptom traces specifically, the implication is: when a user
says "the Risk Judge said X," it's ambiguous whether they mean the
Layer-1 risk_debate.py RiskJudge, the Layer-2 in-app risk role, or
the RiskJudge label in the rationale drawer. All three exist; only
the drawer label is what the user typically sees.

## Self-bias check (Phase 3)

1. **Pro-fix bias on Symptom 3.** Symptom 3 is the clearest "the fix
   exists; the verification is the gap" case. My natural instinct
   was to label it `VERIFICATION_DEFECT` quickly because the
   evidence stack is clean. Counter-check: I read the actual fix
   code (signal_attribution.py:131-155) line by line to confirm the
   detection logic is sound, AND I traced the persistence path
   (trade-write time) to confirm OLD trades retain OLD shape. The
   `VERIFICATION_DEFECT` label is now grounded in the path tracing,
   not just optimism.
2. **Charitable inference resistance on Symptom 1.** The user said
   "feature never shipped." The harsh reading would be "the user is
   wrong, features A and B are shipped." The lenient reading would
   be "the user is right, something is unshipped." Per the
   anti-leniency protocol I went harder: I named Read C (the most
   plausible "unshipped" interpretation) and returned
   `INSUFFICIENT_EVIDENCE`. I refused to charitably guess what the
   user meant.
3. **Symptom 2 reframing risk.** It would be lenient to call
   Symptom 2 a "doc problem" and move on. The harsh reading is that
   the dev MAS made a SCOPING decision (Option C deferred at the
   2026-05-04 cycle) that the operator still finds surprising one
   week later. I labeled it `SCOPING_GAP` to keep that decision
   visible, not "doc gap" which would understate the trade-off.
4. **Auditor-as-Main risk on the systemic pattern.** I am Claude
   Code Main of the Layer-3 MAS; the "dominant failure mode is
   verification" is a claim about my own past behavior (or the
   behavior of prior Main instances). This is the exact bias the
   audit prompt warned about. Counter-check: the
   `feedback_qa_harness_compliance_first.md` and
   `feedback_log_last.md` auto-memories (loaded in this session)
   independently document Main's empirical slippage patterns. The
   "verification skips" claim is corroborated by those memories,
   not just self-deduction.

## Done criteria check

- [x] Each symptom has an evidence trace section **before** its
  verdict. Sections 1.1-1.5, 2.1-2.5, 3.1-3.3 precede the verdicts.
- [x] Each verdict cites at least one piece of repo evidence
  (commit, handoff file, or masterplan entry) OR explicitly states
  "no evidence found." Citation blocks at the end of each symptom.
- [x] The systemic pattern (if any) across the three symptoms is
  named in a closing subsection. "The dominant failure mode is
  VERIFICATION_DEFECT, specifically..." above.
