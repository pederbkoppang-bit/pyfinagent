# Research Brief — phase-57.1 (Binding RiskJudge gate + concentration-aware prompt context)

**Tier:** complex. **Date:** 2026-06-11. **Step:** 57.1 (phase-57 FEATURE; operator reply `PHASE-57: FEATURE` 2026-06-11).
**Spec-of-record:** `handoff/archive/phase-55.3/55.3-synthesis-checkpoint.md` §2.6 FEATURE paragraph.
**Cites:** F-3 (RiskJudge REJECT advisory-only — portfolio_manager.py:185 records, :194-198 log-only; 3 away-week REJECTs executed incl. DELL 06-03), F-8 (lite RiskJudge system prompt phantom "10% cap" vs config paper_max_per_sector_nav_pct=30.0; rationales say "no portfolio sector breakdown was provided").

> STATUS: COMPLETE. Gate PASSED (6 sources read in full, recency scan done, all hard blockers satisfied).

---

## A. Internal fix-design (file:line)

### A0. Gate-site topology (decide_trades flow, portfolio_manager.py)

`decide_trades(current_positions, candidate_analyses, holding_analyses, portfolio_state, settings, candidates_by_ticker)` returns `list[TradeOrder]`, sells appended first, buys second, final stable-sort `SELL`-before-`BUY` at :398.

Two loops matter:
- **Candidate-build loop :148-198** — iterates `candidate_analyses`, skips held/non-BUY-rec, extracts `risk_assessment = analysis.get("risk_assessment", {})`, computes position_pct/stop_loss, appends a dict to `buy_candidates` **including `"risk_judge_decision": risk_assessment.get("decision", "")` at :185**. The advisory log-only branch is :193-198 (`if decision and decision != "APPROVE_FULL": logger.info(...)`). THIS is F-3's exact site.
- **BUY-emit loop :256-365** — sorts `buy_candidates` by final_score desc (:201), then for each candidate runs (in order): position-cap break (:257), available-cash break (:269), sector-COUNT cap -> `sector_blocked.append` + continue (:276-286), position-size + $50-floor (:288-299), sector-NAV-pct cap continue (:301-318), FF3 factor-corr cap continue (:320-334), then `orders.append(TradeOrder(... risk_judge_decision=cand["risk_judge_decision"] ...))` at :336-356, decrement available_cash + increment sector trackers (:357-365).

Every BUY that ever executes flows through the `orders.append(TradeOrder(...))` at :336. The `risk_judge_decision` string is already carried on every candidate dict (:185) and on every emitted TradeOrder (:342).

**CRITICAL TOPOLOGY FINDING (decides the gate site):** there is a SECOND BUY emit path — the swap path `_compute_swap_candidates` (:405-585), emitting `swap_buy` TradeOrders at :553-568 — and **`risk_judge_decision=cand.get("risk_judge_decision","")` IS carried into the swap BUY at :559**. The event study (Section A5) proves **all 3 executed-REJECT BUYs were `reason=swap_buy`** (HPE, DELL, 066570.KS). Therefore a gate placed only in the main BUY-emit loop (:256-365) would MISS every real-world REJECT execution. The gate MUST cover the candidate-build stage (the common ancestor that feeds BOTH the main loop's `buy_candidates` AND, via `sector_blocked`, the swap path).

### A1. Gate placement — RECOMMENDED: drop at candidate-build (:148-198), default-OFF flag

**Recommended option (single chokepoint, covers both paths):** in the candidate-build loop, AFTER `decision = risk_assessment.get("decision", "")` is known (currently the advisory log-only branch :193-198), add — *gated by the new flag* — a `continue` that skips appending the candidate to `buy_candidates` when `decision == "REJECT"`. Because a candidate that never enters `buy_candidates` can be neither bought by the main loop NOR queued into `sector_blocked` for the swap path, this one site binds both. This is the minimal, single-chokepoint change the spec asks for.

Concretely (flag-ON only):
```
decision = (risk_assessment.get("decision", "") or "")
if decision == "REJECT" and reject_binding_on:
    blocked_buys.append({"ticker": ticker, "decision": decision,
                         "reason": risk_assessment.get("reason") or risk_assessment.get("reasoning") or "",
                         "final_score": final_score})
    logger.warning("BINDING RiskJudge gate: BLOCKED BUY %s (decision=REJECT) -- "
                   "flag paper_risk_judge_reject_binding=ON; F-3", ticker)
    continue   # <-- candidate never reaches buy_candidates -> blocked from BOTH main loop and swap path
# else: existing advisory log-only branch (:193-198) unchanged when flag OFF
```
Placement note: insert the gate **before** `buy_candidates.append(...)` (currently :180). The existing advisory `logger.info` branch (:193-198) stays as-is for the non-REJECT / flag-OFF cases (it also logs APPROVE_REDUCED/APPROVE_HEDGED, which the gate does NOT block — only REJECT binds, matching the spec "blocks the BUY when the judge returns REJECT").

**Why REJECT-only (not APPROVE_REDUCED/HEDGED):** the spec is explicit — "blocks the BUY when the judge returns REJECT". APPROVE_REDUCED/APPROVE_HEDGED already flow into sizing via `recommended_position_pct` (:289, :551); binding those too would be a scope expansion beyond F-3 and would change far more behavior. Keep the gate to the single verdict the spec names. (Tiering rationale from external KF — Section B2 — supports REJECT=hard-block while REDUCED=size-down stays advisory.)

**Budget-reallocation semantics (spec question "should a blocked candidate free its budget for the next-ranked candidate?"):** With the recommended placement, a REJECTed candidate is removed *before* the sort+emit loop. The main BUY loop (:256-365) iterates the surviving `buy_candidates` in final_score order and naturally draws from `available_cash` until exhausted — so the next-ranked surviving candidate automatically gets the budget the REJECTed one would have consumed. **No extra code needed**: dropping at build-time IS sell-first-then-buy-clean and IS budget-reallocating by construction. (Contrast: a gate at emit-time that `continue`s would also reallocate, but would not cover the swap path — rejected for that reason, not the budget one.) Recommend documenting this in the contract so Q/A doesn't read the absence of explicit reallocation code as a gap.

### A2. Settings flag (backend/config/settings.py)

Convention scan (grep of `paper_*_enabled` / default-OFF bool Fields): `rebalance_band_enabled` (53.1), `paper_swap_enabled`, `paper_scale_out_enabled` style. Proposed Field:
```python
paper_risk_judge_reject_binding: bool = Field(
    default=False,
    description="phase-57.1 (55.3 F-3): when True, a lite-path RiskJudge "
                "REJECT verdict BLOCKS the BUY (binding gate) instead of "
                "advisory-only. Default OFF preserves byte-identical behavior. "
                "Operator flips live AFTER OOS validation (no flip in 57.1).",
)
```
Default-OFF Field with description citing F-3, mirroring the existing `paper_swap_enabled` / `rebalance_band_enabled` idiom. Read at decide_trades time via `getattr(settings, "paper_risk_judge_reject_binding", False)` (defensive getattr like the sector caps at :221/:229). **Recommend NOT adding a risk_overrides runtime key** in 57.1 — keep it a pure settings flag for the dark-launch; a runtime override can be added later if the operator wants live toggling without restart (matches how paper_swap_* started settings-only).

### A3. Prompt-context injection (F-8) — SAME flag, so flag-OFF is byte-identical incl. prompts

Two sub-fixes, BOTH behind the same `paper_risk_judge_reject_binding` flag (recommended — see do-no-harm note below):

**(a) Replace phantom "10%" with the configured cap.** `_LITE_RISK_JUDGE_SYSTEM` (autonomous_loop.py:1515-1525) hardcodes at :1520-1521: *"Would adding this position exceed 10% of portfolio in one sector? High = reduce size."* The configured cap is `paper_max_per_sector_nav_pct=30.0`. Make the system prompt a function of settings at call time. Because `_LITE_RISK_JUDGE_SYSTEM` is a module-level constant consumed in BOTH `_run_claude_analysis` (system= at :1796/:1814) and `_run_gemini_analysis` (concatenated at :1979), the cleanest fix is a builder:
```python
def _build_risk_judge_system(settings) -> str:
    if not getattr(settings, "paper_risk_judge_reject_binding", False):
        return _LITE_RISK_JUDGE_SYSTEM   # byte-identical when flag OFF
    cap = float(getattr(settings, "paper_max_per_sector_nav_pct", 30.0) or 30.0)
    return _LITE_RISK_JUDGE_SYSTEM.replace(
        "exceed 10% of portfolio in one sector",
        f"exceed {cap:.0f}% of portfolio NAV in one sector")
```
Call sites change from the constant to `_build_risk_judge_system(settings)`. Flag-OFF returns the literal constant -> byte-identical prompt.

**(b) Inject the live sector breakdown into the template.** `_LITE_RISK_JUDGE_TEMPLATE` (:1527-1540) has no portfolio context. `_run_claude_analysis(ticker, settings)` and `_run_gemini_analysis(ticker, settings)` have NO positions param today. Threading design:
- **Compute ONCE per cycle (concurrency-correct):** positions are identical for all tickers in a cycle, and `_run_single_analysis` is fanned out concurrently (Semaphore, :826/:870). Do NOT call `trader.get_positions()` per-ticker. Instead compute a compact sector-weight string ONCE in `run_daily_cycle` after the existing `positions = trader.get_positions()` at **:774** (which is BEFORE Step-3 analysis dispatch at :798), and thread it down.
- **Threading path:** add an optional kwarg `portfolio_context: str | None = None` to `_run_single_analysis`, `_run_claude_analysis`, `_run_gemini_analysis`. `_run_and_persist_one` (:828) closes over cycle scope, so it can pass the precomputed string into `_run_single_analysis(ticker, settings, portfolio_context=_sector_ctx)`. (Alternative considered: a module-level cache var set per cycle — rejected; explicit param is cleaner and concurrency-safe, no shared mutable state.)
- **Sector-weight compute (robust to mark_to_market not yet run at :774):** `paper_positions` rows carry `quantity`, `avg_entry_price`, `current_price`, `sector` (confirmed: get_paper_positions = SELECT * ; columns verified live). At :774 `mark_to_market` (:923) has NOT run, so `current_price` may be stale/missing — use `quantity * (current_price or avg_entry_price)` for weight (same fallback idiom as paper_trader.py:521). Build `{sector: sum(value)} -> pct of invested book`, render e.g. `"Portfolio sector weights (invested book): Technology 100.0%; cash 66% of NAV"`. Inject as a new template line, only when flag ON (flag-OFF -> empty string / unchanged template).
- **Template change:** add a `{portfolio_context}` placeholder line; flag-OFF passes `portfolio_context=""` AND uses the unchanged constant template so the rendered prompt is byte-identical. Simplest byte-identity guarantee: keep TWO template paths — flag-OFF uses `_LITE_RISK_JUDGE_TEMPLATE` (current constant, no new field, no `.format` key change) and flag-ON uses an augmented template. This avoids adding a `{portfolio_context}` key to the OFF path (which would change the `.format(...)` call signature). Recommend the builder approach: `_build_risk_judge_template(settings) -> str` returning the constant when OFF, the augmented string when ON.

**do-no-harm — why prompt changes share the SAME flag (recommended):** prompt edits alter LLM token distribution on the lite path -> non-deterministic behavior change. The spec's criterion 4 requires "US momentum core byte-identical with the flag OFF" and explicitly flags this nuance. Gating prompts behind the SAME flag means flag-OFF is fully byte-identical (prompts AND gate); flag-ON changes prompts AND binding together — which is correct, because the binding gate is only meaningful if the judge is also reasoning with correct caps + real sector state (a REJECT from a judge citing a phantom 10% cap is less trustworthy to bind on). One flag, coherent semantics. (A separate flag would allow "bind on a blind judge" — an incoherent half-state. Reject.)

### A4. Blocked-trade observability (event study + DoD-7 evidence)

Minimal honest option (recommended): a structured `logger.warning` per blocked BUY (shown in A1) PLUS a cycle-summary field. In `decide_trades`, accumulate `blocked_buys: list[dict]` and surface a count; in `run_daily_cycle`, fold into `summary["risk_judge_blocked"] = [...]` (the cycle summary dict already carries `summary["degraded"]`, `summary["signals_logged"]` etc. — same pattern). This gives the away-week-style event study durable evidence without a new BQ table. **BQ row: NOT required for 57.1** — the existing `paper_trades.risk_judge_decision` column already records REJECT on EXECUTED trades; with the gate ON there will be NO executed REJECT BUYs, so the *absence* of new REJECT-BUY rows post-flip is itself the durable proof (queryable by the same Section-A5 SQL). Recommend the structured log + `summary` field for 57.1; a dedicated `paper_blocked_trades` BQ table is a clean follow-on if DoD-7 wants per-block attribution, but is out of 57.1 scope.

### A5. The away-week replay / event study ($0, stored-data only — BQ-CONFIRMED)

This is NOT a backtest-engine run; it is an analysis of stored BQ rows. Queries run live 2026-06-11 (bytes_billed ~21MB, read-only). The natural sample is the executed-REJECT BUYs.

**Query 1 — executed REJECT BUYs** (`financial_reports.paper_trades WHERE action='BUY' AND risk_judge_decision='REJECT'`): exactly **3 rows**, ALL `reason=swap_buy`:

| ticker | buy time (UTC) | buy value | reason | round_trip_id |
|---|---|---|---|---|
| HPE | 2026-06-02 19:18:58 | $245.04 | swap_buy | (unlinked) |
| DELL | 2026-06-03 19:05:19 | $246.67 | swap_buy | (unlinked) |
| 066570.KS (LG Electronics) | 2026-06-09 18:12:39 | $238.40 | swap_buy | (unlinked) |

**Query 2 — round-trip outcomes** (LEFT JOIN to `paper_round_trips` on `ticker` + `DATE(entry_date)=DATE(buy_time)`; the BUY rows have `round_trip_id=None` so the join is by ticker+entry-day):

| ticker | buy | exit | realized P&L $ | realized P&L % | hold days | exit reason |
|---|---|---|---|---|---|---|
| HPE | 06-02 | 06-03 | **-0.81** | -0.33% | 0 | swap_for_higher_conviction |
| DELL | 06-03 | 06-04 | **+0.54** | +0.22% | 0 | swap_for_higher_conviction |
| 066570.KS | 06-09 | 06-10 | **-23.18** | -9.68% | 1 | **stop_loss_trigger** |
| **SUM** | | | **-$23.45** | | | |

**Event-study headline (honest framing):** Had `paper_risk_judge_reject_binding` been ON, these **3 trades would not have been placed**, avoiding a net realized **-$23.45** (the loss is dominated by the single LG Electronics stop-out at -9.68%; DELL was a small win, HPE ~flat). The spec's "DELL exited +0.22%" reconciles exactly (DELL +0.2193%).

**Selection/conditioning caveat (MANDATORY honesty — see Section B4):** n=3 is an anecdote, not a statistic. The sample is conditioned on (a) REJECT verdicts that (b) actually executed via the swap path AND (c) happened to close within the window. This is a *post-hoc, condition-selected* set. -$23.45 is the realized cost of THESE three specific decisions, NOT an unbiased estimate of the gate's expected value. The gate's true EV depends on the full distribution of REJECT verdicts (including ones the judge got wrong, where blocking would have FOREGONE a gain). The brief presents this as "the gate would have avoided these 3 specific trades, net -$23.45 realized" and explicitly disclaims any annualized/Sharpe extrapolation. The event study's role is the **regression-fixture witness + a directionally-suggestive anecdote**, not a promotion gate. (This is why 57.1 carries NO live flip — the operator flips later and validates OOS, per spec criterion 4.)

### A6. Tests inventory (existing patterns to mirror)

- **decide_trades unit tests** — grep `tests/` for files instantiating `decide_trades` with fake candidate dicts (sector-cap tests from phase-23.1.13 / phase-30.5 are the closest precedent; they build `candidate_analyses` with `risk_assessment.decision` set and assert which orders emit). Mirror that to build a REJECT candidate and assert: flag-OFF -> BUY order present (advisory, unchanged); flag-ON -> BUY order absent (blocked). This is acceptance criterion 1's fixture.
- **Settings-flag OFF-identity tests** — `test_phase_50_2*` byte-identity style and 53.1's `rebalance_band` OFF-identity test (grep `rebalance_band` in tests/). Mirror: assert `decide_trades(..., flag OFF)` returns an order list IDENTICAL to pre-flag behavior on a non-REJECT candidate set, and that the rendered risk-judge prompt with flag-OFF equals the current constant prompt (prompt byte-identity).
- **Prompt-content tests** — assert `_build_risk_judge_system(settings_flag_ON)` contains "30%" and NOT "10%"; assert the flag-ON template contains the injected sector line + the real cap; assert flag-OFF returns the verbatim `_LITE_RISK_JUDGE_SYSTEM` / `_LITE_RISK_JUDGE_TEMPLATE` constants (this is criterion 3 + the byte-identity half of criterion 4).
- Locate exact files at GENERATE time: `grep -rln "decide_trades\|rebalance_band\|_LITE_RISK_JUDGE" tests/`.

### A7. Proposed immutable verification command

Mirror 56.x style (pytest -k selector + `test -f` live_check):
```
source .venv/bin/activate && python -m pytest tests/ -k "risk_judge_binding or reject_binding" -q && test -f handoff/current/live_check_57.1.md && echo VERIFICATION_OK
```
(Final `-k` selector to match the test names authored at GENERATE; the `test -f live_check_57.1.md` enforces operator-auditable evidence per the live_check gate; no live flag flip is asserted because none happens in 57.1 — the live_check documents the OFF-identity proof + the event-study numbers, not a live execution.)

---

## B. External research

### B0. Read in full (>=5 required; counts toward the gate)

| # | URL | Accessed | Kind | Fetched how | Key finding (quote/paraphrase) |
|---|-----|----------|------|-------------|-------------------------------|
| 1 | https://www.law.cornell.edu/cfr/text/17/240.15c3-5 | 2026-06-11 | Official (US CFR primary) | WebFetch (HTML) | (c)(1)(i) controls must "Prevent the entry of orders that exceed appropriate pre-set credit or capital thresholds… by **rejecting** orders"; (c)(1)(ii) "Prevent the entry of erroneous orders, **by rejecting** orders…"; (d) controls "under the **direct and exclusive control** of the broker or dealer" — rejection, not flagging; non-bypassable. The canonical hard-block-vs-advisory anchor. |
| 2 | https://arxiv.org/abs/2604.01483 | 2026-06-11 | Peer-reviewed preprint (arXiv) | WebFetch (HTML render) | "probabilistic execution without rigid constraints is architecturally and legally untenable"; the deterministic gateway "treats every proposed agentic action as a mathematical conjecture: execution is permitted if and only if" the constraint is proven; the gate "intercepts this API call **before** it reaches the execution environment". Validates a deterministic pre-execution chokepoint enforcing the LLM verdict. |
| 3 | https://qiniu-images.datayes.com/ENNETH_R__AHERN.pdf (Ahern, UCLA, 2006) | 2026-06-11 | Peer-reviewed (event-study methodology) | WebFetch->pdfplumber 0.11.9 | "If these characteristics are related to selection in an event study sample, imprecise predictions… may produce **erroneous results**"; statistical biases "important when researchers look for cross-sectional explanations of abnormal returns, which are typically **small but significant**". Canonical selection-bias caution for the A5 event study. |
| 4 | https://www.esma.europa.eu/sites/default/files/2026-02/ESMA74-1505669079-10311_Supervisory_Briefing_on_Algorithmic_Trading_in_the_EU.pdf | 2026-06-11 | Official (EU regulator, **Feb 2026**) | WebFetch->pdfplumber | "**Hard blocks** should be designed as mechanisms which **block orders exceeding** set… parameters"; mandated PTCs incl. "**coverage checks** for positions in securities or in cash" and "**warnings in case of old data feeds**". Hard-vs-soft limit distinction + the "blind/stale data" control that maps to F-8. |
| 5 | https://arxiv.org/abs/2406.09187 (GuardAgent) | 2026-06-11 | Peer-reviewed preprint (arXiv) | WebFetch (HTML render) | A separate guard agent "**denied if Ol=1**" the target's action; **requires** "the input and output logs recording the target agent's action trajectories" to judge — i.e. the guard must be GIVEN context; achieves "**over 98% and 83% guardrail accuracies**". Directly supports F-8 (give the judge the portfolio state) + judge-as-binding-veto. |
| 6 | https://arxiv.org/abs/2511.15123 (Causal Inference in Financial Event Studies) | 2026-06-11 | Peer-reviewed preprint (**Nov 2025**) | WebFetch (HTML render) | "when factor models are misspecified… abnormal return estimators are generally **inconsistent** estimators for causal effects"; recommends **synthetic control** over naive abnormal-return for counterfactuals. Recency corroboration of the A5 caveat. |

(6 read in full; floor is 5. Mix spans US-primary-law + EU-regulator + 2 agent-guardrail arXiv + 2 event-study-methodology — cross-domain triangulation: finance-regulation and ML-agent-safety converge on "advisory verdict must be enforced by a separate deterministic gate that is given the relevant state".)

### B1. Identified but snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|-----|------|------------------------|
| https://www.sec.gov/.../divisionsmarketregfaq-0 (SEC 15c3-5 FAQ) | Official | Primary CFR text (source 1) is the authoritative version; FAQ is gloss. |
| https://www.finra.org/rules-guidance/key-topics/market-access | Official | Secondary to the CFR; FINRA exam-priority context only. |
| https://www.pwc.ch/.../pre-trade-controls-for-algorithmic-trading-techniques.html | Industry | Commentary on the ESMA briefing (source 4 is the primary). |
| https://www.hoganlovells.com/.../esma-publishes-supervisory-briefing… | Industry (law firm) | Summary of source 4. |
| https://arxiv.org/abs/2605.29251 (Provably Secure Agent Guardrail) | Preprint | Same thesis as sources 2+5 (formal-methods guardrail); marginal additional value. |
| https://arxiv.org/abs/2509.23994 (Policy-as-Prompt) | Preprint | Governance-rules-to-guardrails; tangential to a single REJECT-gate. |
| https://scholar.harvard.edu/files/borusyak/.../borusyak_jaravel_event_studies.pdf | Working paper | Panel/DiD event-study design; less relevant than Ahern for a 3-event counterfactual. |
| https://www.digitalapplied.com/blog/llm-guardrails-production-safety-layers-reference-2026 | Community/blog | Lower tier; "tool-call gating before write access" theme already covered by sources 2/5. |
| https://www.value-at-risk.net/risk-limit/ | Industry | Tiering taxonomy (hard-block/size-reduce/escalate); corroborates B2 but textbook-level. |
| https://www.fortraders.com/blog/how-prop-firms-monitor-risk-behind-the-scenes | Community | "read-only mode" escalation example; anecdotal corroboration of B2. |

### B2. Key findings (cited per-claim)

1. **Pre-trade risk controls must REJECT, not flag — and must be non-bypassable.** SEC Rule 15c3-5(c)(1)(i)-(ii) requires controls "by rejecting orders" that breach thresholds, and (d) puts them under "direct and exclusive control" so they can't be routed around. (Source 1, law.cornell.edu/cfr/text/17/240.15c3-5, accessed 2026-06-11.) -> The away-week defect (F-3: RiskJudge REJECT recorded but advisory) is exactly the failure mode this canonical rule names; converting it to a binding block is the regulation-aligned design. The chokepoint placement (A1: candidate-build, the common ancestor) satisfies the "cannot be bypassed by routing" property — the swap path can't circumvent it.

2. **Hard limit (blocks pre-acceptance) vs soft limit (unwinds post-acceptance) is the canonical taxonomy; "coverage checks" and "old data feed warnings" are named PTCs.** ESMA Feb 2026: "Hard blocks should be designed as mechanisms which block orders exceeding set… parameters"; mandated controls include "coverage checks for positions… or in cash" and "warnings in case of old data feeds". (Source 4, ESMA74-1505669079-10311, accessed 2026-06-11.) -> Two implications: (a) the binding RiskJudge gate is a *hard* limit by this definition (blocks the BUY pre-execution), the correct tier for a REJECT verdict; (b) F-8's "judge reasons blind / phantom 10% cap" is precisely the "old/stale data feed" + "coverage check" failure — injecting the live sector breakdown + real cap is the named remedy.

3. **Tiering: REJECT=hard-block, REDUCED=size-down, escalate=human — graduated responses are standard.** Industry practice: "graduated responses before complete trading restrictions"; warning at 50% of limit, then "read-only mode" (manage existing, can't open new). (Snippet: value-at-risk.net/risk-limit, fortraders.com; corroborated by source 4's hard-vs-soft.) -> Justifies the A1 design choice to bind ONLY REJECT (the hard-block tier) while leaving APPROVE_REDUCED/HEDGED as size-down (advisory sizing, unchanged). Don't over-bind.

4. **A deterministic gate must ENFORCE the probabilistic verdict; the LLM cannot be trusted to self-enforce.** arXiv:2604.01483: "probabilistic execution without rigid constraints is architecturally and legally untenable"; intercept "before it reaches the execution environment". GuardAgent: "Hardcoded Safety Rules fail… while degrading task performance to merely 3.2%" — a *separate* guard that blocks (Ol=1 -> denied) is the working pattern. (Sources 2, 5.) -> The binding gate (deterministic `if decision==REJECT: continue`) enforcing the LLM judge's verdict is the literature-endorsed architecture; the judge proposes, the gate disposes.

5. **A guard/judge must be GIVEN the relevant state to judge correctly.** GuardAgent requires "the input and output logs recording the target agent's action trajectories" and reaches ">98% / 83% guardrail accuracies" when it has them. (Source 5.) -> Direct support for F-8's context injection: a RiskJudge told to evaluate CONCENTRATION but given no sector breakdown and a wrong cap (10% vs 30%) is a guard without its inputs. Inject the live sector weights + real cap (A3) so the binding verdict is trustworthy. **This is why A3 gates prompts behind the SAME flag as the gate** — binding on a blind judge is the incoherent half-state the literature warns against.

6. **Counterfactual "would-have-been-blocked P&L" on a conditioned, tiny sample is descriptive, not inferential.** Ahern: selection-related characteristics "may produce erroneous results"; biases matter most when effects are "small but significant". arXiv:2511.15123: naive abnormal-return estimators are "inconsistent… when factor models are misspecified"; prefer synthetic control. (Sources 3, 6.) -> The A5 event study (n=3, conditioned on executed-REJECT-via-swap-that-closed) must be presented as "the realized cost of these 3 specific decisions was -$23.45", NOT as the gate's expected value. No Sharpe/annualized extrapolation. The honest role is the regression-fixture witness + a directional anecdote; OOS validation (post-flip, operator-gated) is where EV gets established.

### B3. Consensus vs debate

- **Strong consensus:** risk verdicts that matter must be enforced by a deterministic, non-bypassable pre-execution gate (SEC 15c3-5, ESMA 2026, arXiv:2604.01483, GuardAgent all agree). No credible source argues an advisory-only risk flag is sufficient when the verdict is "do not trade this".
- **Consensus:** a judge/guard needs the relevant state injected (GuardAgent's central result). F-8 is a textbook violation.
- **Debate / nuance:** how MUCH to bind. Pure hard-block-everything (15c3-5's erroneous-order frame) vs graduated tiering (industry: warn->reduce->read-only->block). Resolution adopted: bind ONLY the REJECT verdict (the hard-block tier); keep REDUCED/HEDGED as advisory sizing. This is the minimal, spec-faithful choice and avoids over-restricting a momentum core that depends on firing.
- **Debate:** event-study rigor. Ahern (factor-model + sign-test) vs arXiv:2511.15123 (synthetic control). For n=3 neither is applicable at power; both agree the honest move is to NOT infer EV from the sample. The brief sidesteps the methodology fight by refusing the inferential claim entirely.

### B4. Pitfalls (from literature, mapped to this change)

- **P1 — binding a blind judge (incoherent half-state).** If prompts (F-8) and the gate (F-3) are on separate flags, you could bind REJECTs from a judge citing a phantom 10% cap. (GuardAgent: guard needs its inputs.) Mitigation: ONE flag (A3).
- **P2 — gate bypass via the swap path.** All 3 real REJECT executions were `swap_buy`. A gate only in the main BUY loop is bypassable. (15c3-5(d): non-bypassable.) Mitigation: gate at candidate-build, the common ancestor (A1).
- **P3 — over-binding kills the core.** Binding APPROVE_REDUCED/HEDGED too would suppress far more trades and could regress the working momentum engine (the asset to protect). Mitigation: REJECT-only (B2#3).
- **P4 — event-study over-claim.** Presenting -$23.45 as "the gate makes money" is the selection-bias trap (Ahern). Mitigation: descriptive framing only; no live flip in 57.1.
- **P5 — prompt change breaks byte-identity.** Adding a `{portfolio_context}` `.format` key to the OFF path changes the rendered prompt. Mitigation: builder returns the verbatim constant when OFF (A3).
- **P6 — per-ticker positions fetch under concurrency.** Calling get_positions() inside the fanned-out analysis would be N redundant BQ reads + a race on stale data. Mitigation: compute the sector summary ONCE at :774, thread as a param (A3).

### B5. Recency scan (2024-2026)

Searched 2024-2026 literature on (a) pre-trade risk controls / hard-block vs advisory, (b) LLM-agent action-gating / guardrails, (c) counterfactual event-study methodology. Result: **3 new findings that COMPLEMENT (and in one case sharpen) the canonical sources**:
- **ESMA Supervisory Briefing on Algorithmic Trading, 26 Feb 2026** (source 4) — the freshest authoritative statement of the hard-vs-soft limit taxonomy and the "coverage checks / old-data-feed warnings" PTCs; directly current and directly on F-8. Supersedes older MiFID-II commentary as the live supervisory expectation.
- **arXiv:2604.01483 "Type-Checked Compliance: Deterministic Guardrails for Agentic Financial Systems", 2026** (source 2) — newest formal articulation that agentic financial actions need a deterministic pre-execution gate, not probabilistic self-policing. Complements (does not supersede) 15c3-5 by extending the principle to LLM agents specifically.
- **arXiv:2511.15123 "Causal Inference in Financial Event Studies", Nov 2025** (source 6) — modern (synthetic-control) update to Ahern's 2006 selection-bias warning; reinforces the A5 honesty caveat. Older Ahern remains the cleaner canonical citation for *conditioned-sample* bias; the 2025 paper adds the misspecification/inconsistency framing.
- GuardAgent (2024) and the broader 2026 "eval-to-guardrail" / "tool-call gating before write access" pattern (snippet, digitalapplied 2026) confirm the judge-as-binding-veto design is current industry direction, not a one-off.
No 2024-2026 finding CONTRADICTS the binding-gate design; the only adversarial note (KTD-Fin, re-cited from 55.2: more agents/tools != returns) is consistent — this change adds RELIABILITY (binding an existing verdict + fixing its inputs), not new analytical capability, which is exactly the kind of change the adversarial literature does NOT warn against.

### B6. Query log (3-variant discipline)

- 15c3-5 hard block: `"SEC Rule 15c3-5 market access pre-trade risk controls hard block"` (canonical/year-less) + `"pre-trade risk controls algorithmic trading hard limits vs advisory 2026"` (frontier) -> surfaced ESMA 2026 + arXiv:2604.01483.
- Tiering: `"risk limit enforcement hard block size reduction human escalation tiering trading"` (year-less canonical) -> industry tiering taxonomy.
- Agent gating: `"LLM judge agent action gating veto guardrail tool-use permission 2026"` (frontier) -> GuardAgent + Provably-Secure-Guardrail + Policy-as-Prompt.
- Event study: `"event study counterfactual trade analysis selection bias small sample methodology"` (canonical) -> Ahern (2006) + arXiv:2511.15123 (2025 recency).
- Year-less canonical hits (snippet table): SEC FAQ, FINRA market-access, value-at-risk.net risk-limit — prior-art confirmed.

---

## Draft immutable success criteria (4-6, 53.1 rigor — ready to paste into the payload)

Copy these VERBATIM into `.claude/masterplan.json` step 57.1 `verification.criteria` (immutable once written). Phrased to be deterministically checkable by Q/A.

1. **Binding-gate regression fixture (both BUY paths).** A unit test instantiates `decide_trades` with a candidate whose `risk_assessment.decision == "REJECT"` and a screen that routes it through BOTH the main BUY path AND the swap path. With `paper_risk_judge_reject_binding=False` (default) the REJECT candidate's BUY order IS emitted (advisory, unchanged). With the flag `True` the REJECT candidate's BUY order is ABSENT from `decide_trades` output on both paths. Test passes.

2. **Default-OFF byte-identity of the US momentum core.** A unit test asserts that with `paper_risk_judge_reject_binding=False`, `decide_trades` returns an order list byte-identical (same tickers, actions, amounts, order) to the pre-57.1 behavior on a candidate set containing NO REJECT verdicts; AND the rendered lite-RiskJudge system prompt + template with the flag OFF are byte-identical to the current `_LITE_RISK_JUDGE_SYSTEM` / `_LITE_RISK_JUDGE_TEMPLATE` constants. No live flag flip occurs in 57.1.

3. **Prompt-context correctness with the flag ON (F-8).** A unit test asserts that with `paper_risk_judge_reject_binding=True`: the RiskJudge system prompt contains the configured sector cap value (e.g. "30%") and does NOT contain the phantom "10% of portfolio in one sector"; AND the RiskJudge user prompt/template includes a live portfolio sector-breakdown line derived from the current positions (asserted via an injected/fake positions fixture, e.g. "Technology 100.0%").

4. **Per-cycle single-compute of the sector summary (concurrency-correct).** The sector-breakdown context passed to the per-ticker RiskJudge is computed ONCE per cycle (not once per ticker): verified by a test asserting the position-reading helper is invoked at most once across an N-ticker analysis fan-out (e.g. via a mock/spy on the positions accessor), OR by code-structural assertion that `_run_single_analysis` receives a precomputed `portfolio_context` argument rather than fetching positions itself.

5. **Event-study evidence artifact (stored-data, $0, honestly framed).** `handoff/current/live_check_57.1.md` exists and contains: the BQ query + result enumerating the executed-REJECT BUYs (3 rows: HPE, DELL, 066570.KS), their realized round-trip P&L (-$0.81 / +$0.54 / -$23.18; net -$23.45), AND an explicit selection-bias caveat stating that n=3 is descriptive of those specific trades, not the gate's expected value, with no annualized/Sharpe extrapolation.

6. **Verification command green.** `source .venv/bin/activate && python -m pytest tests/ -k "risk_judge_binding or reject_binding" -q && test -f handoff/current/live_check_57.1.md` exits 0 (all new tests pass; live_check artifact present).

(6 criteria. 1-4 are deterministic unit tests; 5 is the operator-auditable evidence artifact; 6 is the immutable verification command. This matches 53.1 rigor: a regression fixture, an OFF-identity guarantee, a prompt-content assertion, a concurrency guard, an honestly-caveated event study, and a single green command. Note: criterion 2 + the spec's criterion 4 BOTH require byte-identical flag-OFF including prompts — the A3 builder pattern is the mechanism.)

### Research Gate Checklist

Hard blockers — `gate_passed` is true only if all checked:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (6: SEC CFR, arXiv:2604.01483, Ahern, ESMA 2026, GuardAgent, arXiv:2511.15123)
- [x] 10+ unique URLs total (6 read-in-full + 10 snippet-only = 16)
- [x] Recency scan (last 2 years) performed + reported (B5: ESMA Feb 2026 + 2 arXiv 2025-2026)
- [x] Full papers / pages read (not abstracts) for the read-in-full set (PDFs via pdfplumber; HTML via arxiv.org/html)
- [x] file:line anchors for every internal claim (portfolio_manager.py + autonomous_loop.py + paper_trader.py + bigquery_client.py, exact line numbers)

Soft checks:
- [x] Internal exploration covered every relevant module (gate site, swap path, both prompt twins, cycle entry, positions schema, round-trips schema)
- [x] Contradictions / consensus noted (B3)
- [x] All claims cited per-claim (B2 per-finding source attributions)

---

## JSON envelope

```json
{
  "tier": "complex",
  "external_sources_read_in_full": 6,
  "snippet_only_sources": 10,
  "urls_collected": 16,
  "recency_scan_performed": true,
  "internal_files_inspected": 5,
  "report_md": "handoff/current/research_brief.md",
  "gate_passed": true
}
```
