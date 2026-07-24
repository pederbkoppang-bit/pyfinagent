# Research Brief — phase-70.4 (S3: un-gate throughput; make silent BUY-blockers visible + tunable)

- **Step:** 70.4 (phase-70) — observability-first, $0, always-on logging; any
  TRADING-THRESHOLD change flag-gated default-OFF.
- **Researcher tier:** complex (3 distinct systemic-observability topics + a
  multi-file internal audit).
- **HEAD at anchoring:** `d0efa50dc2973937b85fe1f4d6471808fa5df738`
- **Builds on:** `handoff/current/design_trade_diversity_70.md` §(c) BUY-gate
  observability + `research_brief_70.0.md`.
- **Binding constraints:** observability additions always-on + $0; any
  trading-threshold change flag-gated default-OFF; NO risk threshold moved;
  historical_macro FROZEN; fail-safe.

Status: WRITE-FIRST — skeleton created before external reads; source rows,
citations, recency scan, and design recommendations appended incrementally.

---

## Three gates in scope (task framing)

1. **Hidden per-cycle session budget** `_SESSION_BUDGET_USD=$1.00` is HALF the
   operator-visible `paper_max_daily_cost_usd=$2.00`; the breach raises with no
   log and is swallowed under `gather(return_exceptions=True)` → cost-truncated
   cycle is SILENT. Reconcile + surface/log on breach.
2. **5% price-tolerance gate** rejects BUYs with WARN-only `return None`; not
   surfaced/counted → 0-trade cycles un-diagnosable. (Tunability already exists.)
3. **Lite-analyzer parse/rail failure** defaults `score=5` and masquerades as a
   legitimate HOLD → silently suppresses BUYs and EVADES the degraded guard.

---

## Internal RE-ANCHOR (exact HEAD lines — verified read)

**File note:** there are TWO `autonomous_loop.py`. The task's line numbers
(`:90`, `:1037-1099`, `:2250`) map to **`backend/services/autonomous_loop.py`
(3061 lines)**, NOT `backend/autonomous_loop.py` (619 lines). All line refs
below are the services/ file unless stated.

### Gate 1 — session vs daily budget

- **`backend/services/autonomous_loop.py:90`** — `_SESSION_BUDGET_USD: float =
  float(os.getenv("PYFINAGENT_SESSION_BUDGET_USD", "1.0"))`. Module-level
  constant; **env-override ONLY**; confirmed NOT in `backend/api/settings_api.py`
  (grep empty) → NOT operator-visible. Half of the $2.00 daily cap.
- **`:91`** — `_session_cost: float = 0.0` (module accumulator; reset to 0.0 at
  cycle start, `:321`).
- **`:95-105`** — `_check_session_budget(stage)`: `if _session_cost >=
  _SESSION_BUDGET_USD:` → `raise BudgetBreachError(...)`. **NO `logger` call on
  the raise path** — the breach is silent at the source.
- **`:108-111`** — `_add_session_cost(usd)` mutates `_session_cost` (called at
  `:1072` after each analysis).
- **`:335`** — `summary["session_budget_usd"] = _SESSION_BUDGET_USD` (the $1.00
  is recorded into the summary at cycle start, but no *breach* flag is ever set).
- **`:1037-1087`** — `async def _run_and_persist_one(ticker, kind)`: at **`:1049`**
  `_check_session_budget(f"pre_analysis_{kind}")` inside a try whose except
  (`:1050-1052`) **re-raises** (comment `:1051` claims "propagate to the
  cycle-level catch" — this is WRONG, see below). Separately at **`:1053-1058`**
  the *daily* cap `if total_analysis_cost >= settings.paper_max_daily_cost_usd:`
  does `logger.warning(...)` + `return None` (a CLEAN, LOGGED per-ticker skip —
  contrast with the silent session raise). `_run_single_analysis` exceptions are
  caught at `:1065-1067` (`logger.error` + `return None`).
- **`:1090-1094`** — `candidate_results = await asyncio.gather(*[...], `
  **`return_exceptions=True`** `)`; then `candidate_analyses = [r for r in
  candidate_results if isinstance(r, dict)]`. The `BudgetBreachError` raised at
  `:1049` is CAPTURED as a result element by `return_exceptions=True` and then
  **silently dropped** by the `isinstance(r, dict)` filter → does NOT reach the
  cycle-level catch.
- **`:1097-1101`** — identical pattern for the re-eval gather.
- **`:1585-1602`** — cycle-level `except Exception as e:` with `if
  type(e).__name__ == "BudgetBreachError":` → sets `status="budget_breach"`,
  `budget_tripped=True`. **This is the intended clean cycle-halt — but it is
  UNREACHABLE for the session budget**, because the only `_check_session_budget`
  caller (`:1049`) sits inside the `return_exceptions=True` gather. (This branch
  DOES catch the llm_client daily/monthly `_check_cost_budget` breach that
  propagates through other paths.)
- **Settings that already exist:** `backend/config/settings.py:371`
  `paper_max_daily_cost_usd: float = Field(2.0, ...)` (operator-visible via
  `settings_api.py:105/159/298/366`); `:378` `cost_budget_daily_usd=25.0` and
  `:379` `cost_budget_monthly_usd=300.0` (llm_client global caps, distinct
  system). The session $1.00 has NO settings field.

### Gate 2 — price-tolerance gate

- **`backend/services/paper_trader.py:169-193`** — the phase-30.6 pre-trade gate
  inside `execute_buy`. `:177-179`
  `price_tolerance_pct = float(getattr(self.settings,
  "paper_price_tolerance_pct", 0.0) or 0.0)`; `:180-185` guard (`>0`,
  `price_at_analysis not None/>0`, `price>0`); `:186`
  `divergence_pct = abs(price - price_at_analysis)/price_at_analysis*100`;
  `:187-193` `if divergence_pct > price_tolerance_pct:` → `logger.warning(...)`
  (already logs ticker, live price, drift%, analysis price, tolerance) →
  **`return None`**.
- **`backend/config/settings.py:557-562`** — `paper_price_tolerance_pct: float =
  Field(5.0, ge=0.0, le=50.0, ...)`. **Tunability ALREADY EXISTS** (operator can
  set 0-50; 0 disables). The task's "make tunable" is already satisfied — do NOT
  redo it.
- **Consumer / surfacing point:** `autonomous_loop.py:1445-1469` — cycle calls
  `trader.execute_buy(...)`; `:1468 if trade: trades_executed += 1`. The `None`
  return (price-tolerance OR insufficient-cash `:202-207` OR max-positions
  `:212-214` OR FX-unavailable `:216+`) is a **silent no-op else** — nothing is
  counted, nothing reaches `summary`. A 0-trade cycle cannot be attributed to a
  cause at the summary/BQ level.

### Gate 3 — lite-analyzer parse-fail masquerade

- **`backend/services/autonomous_loop.py:2393-2399`** — after the LLM call,
  `json_match = re.search(r'\{[^}]+\}', text)`; if match → parse; **`:2399 else:
  analysis = {"action": "HOLD", "confidence": 0, "score": 5, "reason": "Could
  not parse analysis"}`**. The `else` branch has **NO log of its own**.
- **`:2401`** — `logger.info(f"Claude analysis for {ticker}: {analysis['action']}
  (confidence=..., score=...)")` — logs the FABRICATED default as if it were a
  real analysis (INFO, not WARN).
- **`:2493-2504`** — the returned lite dict: `"_path": "lite"`,
  **`"recommendation": analysis["action"]`** (→ `"HOLD"`),
  **`"final_score": analysis["score"]`** (→ `5`), `"risk_assessment": {...}`.
  **The returned dict has NO top-level `confidence` key.**
- **`:2108-2135` `_degraded_scoring_check(analyses)`** — the guard. Degraded iff
  `float(a.get("final_score") or a.get("score") or 0) == 0.0` (`:2119`) OR
  (`_conf_raw is not None and float==0.0 and rec.isupper() and bool(rec)`)
  (`:2123-2129`). **For the parse-fail dict:** `final_score=5` → score-leg False;
  `a.get("confidence")` is **`None`** (key absent from the returned dict) →
  conf-leg False. **⇒ parse-fail EVADES the guard.** Confirmed by read.
- **Guard is consumed at `:1109-1131`** — fires a P1 `raise_cron_alert` when ALL
  degraded or ≥3 degraded. The parse-fail never counts toward `n_degraded`.
- **Contrast — the honest degraded path exists already:** `:1886-1910`
  ("both full and lite paths failed") returns a dict with `_degraded=True`,
  `_path="degraded"`, `recommendation=None`, `final_score=None` **when
  `paper_synthesis_integrity_enabled` is ON** (`:1894`); `:1080` in
  `_run_and_persist_one` UNCONDITIONALLY does `if analysis.get("_degraded"):
  return None` (drops it from candidate/holding but persists an honest NULL row
  via `:2748-2772`). The parse-fail path does NOT use this marker — that is the
  gap.

**internal_files_inspected:** `backend/services/autonomous_loop.py`,
`backend/services/paper_trader.py`, `backend/config/settings.py`,
`backend/api/settings_api.py`, `backend/services/portfolio_manager.py`,
`backend/services/autonomous_loop.py` (619-line sibling ruled out),
`backend/tests/test_price_tolerance_gate.py` (existing gate test). = 6 distinct
source files inspected.

---

## Search-query composition (3-variant discipline — disclosed)

Three topics, each run current-year / last-2-year / year-less canonical:

- **T1 budget observability:** (frontier) `LLM agent cost budget observability
  surface truncation logging 2026`; (last-2-yr) `autonomous agent token budget
  cap silent failure alerting 2025`; (year-less) `AI agent budget guardrail cost
  ceiling design pattern graceful stop`.
- **T2 pre-trade price-tolerance / rejection audit:** (year-less canonical)
  `pre-trade risk controls price collar tolerance rejection audit SEC 15c3-5`;
  (frontier) `algorithmic trading order rejection reason logging transparency
  slippage control 2026`; the SEC/CFR corpus (2010 rule) is the canonical prior
  art, the arXiv 2603.07752 (2026) + slippage guides are the frontier.
- **T3 fail-open/degraded/parse-fail signaling:** (year-less canonical)
  `fail-open versus fail-closed fail-safe defaults design principle` +
  `graceful degradation observability degraded mode signal SRE never silent
  fallback`; (frontier) `LLM structured output JSON parse failure default value
  validation guardrail distinguish neutral 2026`.

Source table below carries a mix of current-year (2026), last-2-year, and
year-less/canonical (SEC 2010 rule, fail-open first-principles) hits.

## Source table (fetched IN FULL via WebFetch) — 8 sources

| # | Source | Tier | Topic | Load-bearing takeaway |
|---|--------|------|-------|-----------------------|
| 1 | [SEC Rule 15c3-5, 17 CFR §240.15c3-5 (Cornell LII)](https://www.law.cornell.edu/cfr/text/17/240.15c3-5) | 2 (regulatory) | T2 | (c)(1)(ii) reject erroneous orders by price/size params **pre-trade, order-by-order**; (d) controls under **exclusive control** (non-bypassable); (b) **preserve a written description** of controls; (e)(1) **review effectiveness no less than annually** + document; (e)(2) CEO cert. Rejections are auditable, first-class. |
| 2 | [arXiv 2603.07752 — Dynamic slippage control & rejection feedback (spot FX MM)](https://arxiv.org/html/2603.07752) | 1 (peer-reviewed) | T2 | Rejection tolerance ϵ is **endogenously tuned**; rejection is a **first-class measured state** (EMA reputation, rejection-rate rows in Table 1); silent/opaque rejections **reduce future arrival intensity → hurt throughput**; dealers incentivized to keep rejection rates transparent + measured. |
| 3 | [Braintrust — How to track LLM costs (2026)](https://www.braintrust.dev/articles/how-to-track-llm-costs-2026) | 4 (practitioner) | T1 | "**Kill switches on agent runs** … stop while it is still executing"; enforce at proxy/middleware "**because blocking at the LLM call is too late once input tokens have already been billed**"; "**Webhook destinations should route cost … alerts** … rather than remaining hidden in logs"; per-agent-run rollup (median vs p99). |
| 4 | [nexgismo — AI Agent Budget Guards](https://www.nexgismo.com/blog/ai-agent-budget-guards-stop-runaway-api-costs) | 4 (practitioner) | T1 | Budget guard = "**a hard limit enforced at the code level — not a billing email**"; "**a gateway-level hard cap cannot be circumvented by agent logic**"; silent runaway incident ($6,531 before human intervention) — nobody told it to stop. |
| 5 | [dev.to/mudassirworks — Why LLM Agents Fail Silently](https://dev.to/mudassirworks/why-llm-agents-fail-silently-and-how-to-debug-them-251l) | 4 (practitioner) | T1+T3 | On budget-truncated output: "**treat it as a hard failure, not a graceful noop**"; swallowed exceptions → "**reraise … or … return a structured error payload instead of propagating `None`**"; Pydantic-validate after each call; **structured logging counters** (empty responses, validation failures, finish_reason distribution). |
| 6 | [SRE School — Graceful Degradation (2026 guide)](https://sreschool.com/blog/graceful-degradation/) | 4 (practitioner) | T3+T1 | "**Alert on both degraded state and root metric**"; "**Ticket for degraded-mode activation**"; **degraded-mode fraction** metric (<5% baseline); "**Instrument fallbacks and tag telemetry**" / "**Emit degraded-mode telemetry for each path**" so degraded is distinguishable from healthy. |
| 7 | [FutureAGI — LLM Input/Output Validation (2026)](https://futureagi.com/blog/what-is-llm-input-output-validation-2026/) | 4 (practitioner) | T3 | 3 layers (schema/structural/content); on parse fail: "**Catching `ValidationError` and silently passing** … is a regression you will not see"; retry-with-error-feedback (cap 3), **not** a silent default; schema-valid-but-wrong ≠ genuine; "validation pass-rate, retry rate, and **per-failure-mode breakdown as span attributes**"; **fail-closed** ("Reject on structural impossibility"). |
| 8 | [AuthZed — Failed Open / Fail Closed in software engineering](https://authzed.com/blog/fail-open) | 3–4 (practitioner) | T3+G1 | Fail-closed "**essential … in financial transactions**"; the masquerade danger: when a failure path returns a response anyway the "**failure invisible while granting unintended access**" — the software analog of `score=5` masquerading as a real HOLD. |

## Snippet-only sources (evaluated, not read in full)

| # | Source | Why snippet-only |
|---|--------|------------------|
| s1 | SEC.gov 15c3-5 FAQ (divisionsmarketregfaq-0) | HTTP 403 on WebFetch; substituted the Cornell LII rule text (#1). |
| s2 | earezki.com — Why LLM Agents Fail Silently | HTTP 403; read the dev.to mirror (#5) with identical content. |
| s3 | gtngroup.com — SEC 15c3-5 Market Access Controls | Fetched but thin (disclosure page, no mechanism detail). |
| s4 | nasdaqtrader.com — Understanding the SEC Market Access Rule (PDF) | Redundant with #1 regulatory text. |
| s5 | federalregister.gov — Risk Mgmt Controls for Market Access (2010-28303) | Rule adopting release; #1 covers the operative text. |
| s6 | wilmerhale.com — SEC Staff FAQs on Rule 15c3-5 | Law-firm summary; regulatory primary (#1) preferred. |
| s7 | finra.org — Market Access Rule exam report | Exam-priority note; not mechanism-level. |
| s8 | Braintrust — Agent observability complete guide (2026) | Overlaps #3; reasoning/state-transition spans. |
| s9 | Braintrust — Best tools tracking LLM costs (2026) | Tool roundup, not design guidance. |
| s10 | aicostboard.com — Complete Guide to LLM Observability 2026 | Request-logging fundamentals; general. |
| s11 | MLflow — Top LLM Observability Tools 2026 | Tool comparison. |
| s12 | openobserve.ai — LLM Cost Monitoring | Vendor how-to. |
| s13 | tentoro.ai — AI Agent Token Sprawl: Silent Budget Killer | Framing overlaps #4. |
| s14 | waxell.ai — AI Agent Token Budget Enforcement 2026 | Enforcement overview; #3/#4 stronger. |
| s15 | mindstudio.ai — AI Agent Token Budget Mgmt (Claude Code) | Vendor angle. |
| s16 | medium/Micheal-Lanham — Cost Guardrails for Agent Fleets | Snippet: "enforcement in the orchestrator … returns a **structured signal** … agent reacts gracefully" — corroborates G1 surfacing. |
| s17 | blogs.oracle.com — Runtime Budget Guardrails for Agentic AI | Corroborates pre-call enforcement. |
| s18 | dev.to/thedailyagent — Stop AI Agent Cost Spirals | Snippet: per-session budgets cap blast radius; structured signal on spend. |
| s19 | devsecopsschool — Fail-Safe Defaults (2026) | Canonical fail-safe framing; #8 preferred. |
| s20 | arXiv 2510.11837 — Countermind (multi-layer LLM security) | Non-bypassable-invariants adjacent; not trade-specific. |
| s21 | dev.to/pockit + pockit.tools — LLM Structured Output 2026 | "8% unparseable, 5% wrong-type"; overlaps #7. |
| s22 | medium/rosgluk — Your LLM JSON Is Valid — And Still Wrong | Schema-valid-but-wrong; overlaps #7. |
| s23 | arXiv 2601.06151 — PromptPort (cross-model structured extraction) | Reliability layer; general. |
| s24 | daytrading.com — Ultimate Guide to Trade Slippage 2026 | "platforms let users set a slippage tolerance, auto-rejecting orders that exceed it" — corroborates G2 tunability. |
| s25 | quantvps / nordfx — slippage in automated trading | Retail slippage explainers. |
| s26 | agentmarketcap — AI Agent Token Consumption Gap | Cost-scale context. |

Plus ~8 further candidates surfaced (confident-ai, augmentcode, isimplifyme,
cosmicjs, pragmaticstack, blog.alephant, trainingcamp, cisco-community) pruned
as lower-tier/redundant. **urls_collected = 34** (8 read-in-full + 26 tabled
snippet-only).

## Recency scan (last 2 years)

- **New (2026):** dev.to "Fail Silently" (2026-06-27) gives the exact idiom for
  G1/G3 — `finish_reason=="length"` / unparsed output "**treat as a hard
  failure, not a graceful noop**." FutureAGI (2026) codifies "silently passing a
  ValidationError is a regression you will not see" — direct support for G3's
  parse-fail-must-be-visible. SRE School's 2026 graceful-degradation guide
  supplies concrete degraded-mode metrics ("alert on degraded, not just down";
  degraded-mode fraction <5%) that map onto pyfinagent's *existing*
  `_degraded_scoring_check` + `_fallback_rate_check` alarms (autonomous_loop.py
  :1103-1175) — the 70.4 gate-3 fix extends that same, already-blessed pattern.
  arXiv 2603.07752 (2026) is a peer-reviewed treatment of rejection-tolerance as
  a **tunable, measured** control — supersedes generic slippage blogs for G2.
- **Still-canonical (older, not superseded):** SEC Rule 15c3-5 (2010) remains
  the authoritative pre-trade-control + audit reference the code already cites
  (paper_trader.py:169-176, settings.py:551-556); the FIA-WP/LULD anchors in the
  code are unchanged. Fail-open/fail-closed is a first-principles security axiom
  (no recency dependence). **No last-2-year finding overturns the existing
  design anchors; the new work sharpens the *observability* layer, which is
  exactly 70.4's scope.**

---

## Design recommendations

Legend: **[OBS]** = observability, always-on, $0, no threshold moved.
**[FLAG]** = behavior change, new/existing flag, default-OFF, byte-identical when OFF.

### Gate 1 — reconcile + surface the hidden session budget

Root cause (verified): session ceiling `$1.00` (autonomous_loop.py:90) is HALF
the operator-visible daily `$2.00` (settings.py:371) and fires first;
`_check_session_budget` (:95-105) **raises with no log**; the raise is the only
caller-site inside a `return_exceptions=True` gather (:1090-1094 / :1097-1101),
so it is captured as a result element and then dropped by `isinstance(r, dict)`
→ never reaches the clean `budget_breach` halt at :1591-1597. **Silent
truncation.** (Contrast: the *daily* cap at :1053-1058 logs + skips cleanly.)

- **G1-A [OBS]** — Log at the source. In `_check_session_budget`
  (autonomous_loop.py:99, immediately before `raise`), add
  `logger.warning("session_budget_breach: cumulative $%.4f >= ceiling $%.4f (stage=%s cycle=%s)", ...)`.
  Grounds: dev.to (#5) "hard failure, not a graceful noop"; nexgismo (#4).
- **G1-B [OBS]** — Catch the swallowed breach in the gather post-processing.
  After :1094 and :1101, **before** the `isinstance(r, dict)` filter, scan the
  raw results for `type(r).__name__ == "BudgetBreachError"` (reuses the module's
  own name-check idiom at :1591). If any found, set
  `summary["session_budget_breach"]=True`,
  `summary["session_budget_ceiling_usd"]=_effective_session_budget`,
  `summary["session_cost_at_breach_usd"]=get_session_cost_usd()`,
  `summary["analyses_skipped_by_budget"]=<count>`, and `logger.warning(...)`.
  The `isinstance(r, dict)` filter stays (correctly excludes exceptions from
  analyses). Grounds: Braintrust (#3) "route cost alerts … rather than remaining
  hidden in logs"; SRE (#6) "the fallback firing is itself an event";
  Micheal-Lanham (s16) "returns a structured signal."
- **G1-C [FLAG]** — Single-knob reconcile. New flag
  `paper_session_budget_reconcile_enabled: bool = Field(False, ...)` in
  settings.py. At cycle start (near autonomous_loop.py:321/335) compute
  `_effective_session_budget = settings.paper_max_daily_cost_usd if
  settings.paper_session_budget_reconcile_enabled else _SESSION_BUDGET_USD` and
  thread it into `_check_session_budget(stage, ceiling=_effective_session_budget)`
  (and into `summary["session_budget_usd"]` at :335). When **OFF** → resolves to
  the current `$1.00` env-default (byte-identical). When **ON** → session ceiling
  == operator-visible daily `$2.00` (single knob; still env-overridable). This
  RAISES the truncation point → behavior change → flag-gated. **Cost knob only;
  NO risk threshold moved.** (Fallback design if a merged knob is undesired:
  promote the session budget to an operator-visible settings field
  `paper_session_budget_usd` default `1.0` so it is at least no longer *hidden*.)

### Gate 2 — surface + count price-tolerance rejections (tunability already exists)

**Already satisfied — do NOT redo:** the gate already logs ticker+drift
(paper_trader.py:188-192) and the tolerance is already tunable via
`paper_price_tolerance_pct` (settings.py:557, `ge=0 le=50`, `0` disables). The
only gap is that the `return None` (:193) is un-counted → a 0-trade cycle can't
be attributed at the summary/BQ layer (the caller at autonomous_loop.py:1468
only does `if trade: trades_executed += 1`, silent no-op else).

- **G2-A [OBS]** — Rejection accumulator on `PaperTrader`. In `__init__` add
  `self.buy_rejections: list[dict] = []`. At paper_trader.py:193 (price-tolerance)
  append `{"ticker":ticker, "reason":"price_tolerance", "divergence_pct":round(...),
  "tolerance_pct":price_tolerance_pct, "live":price, "analysis":price_at_analysis}`
  before `return None`. Recommended: also tag the other `execute_buy` None-exits
  (insufficient-cash :207, max-positions :214, FX-unavailable :216+) with their
  reason so *every* skipped BUY is attributable. Then in autonomous_loop.py after
  the trade loop (~:1470) fold `trader.buy_rejections` into
  `summary["buy_rejections"]` + `summary["buy_rejections_by_reason"]` (a Counter).
  Grounds: SEC 15c3-5 (#1) (e)(1) review effectiveness + preserve control records;
  arXiv 2603.07752 (#2) rejection as a measured first-class state; daytrading
  (s24) slippage logging reveals patterns.
- **G2 [FLAG]:** none. Surfacing is pure observability; tolerance already tunable;
  default stays `5.0` (SEC LULD Tier-1 anchor). **NO threshold moved.** (Optional,
  separate: expose `paper_price_tolerance_pct` in settings_api for UI visibility —
  it is currently a Settings field only. Cosmetic, not required by 70.4.)

### Gate 3 — make the lite parse-fail a DEGRADED signal (kill the score=5 masquerade)

Root cause (verified): parse-fail else-branch (autonomous_loop.py:2399) defaults
`{"action":"HOLD","confidence":0,"score":5,...}` with **no log**; :2401 logs it
as a normal INFO analysis; the returned lite dict (:2493-2500) carries
`recommendation="HOLD"`, `final_score=5`, **and no top-level `confidence` key**.
`_degraded_scoring_check` (:2119-2129): score-leg sees `5≠0` → False; conf-leg
sees `a.get("confidence") is None` → False → **evades the guard.** A parse-fail
is indistinguishable from a genuine HOLD and silently suppresses a BUY.

- **G3-A [OBS]** — Log + mark the parse-fail. At :2399 change the else to emit
  `logger.warning("lite parse-fail for %s: no JSON in %d-char response; marking degraded", ticker, len(text))` and add a marker:
  `{"action":"HOLD","confidence":0,"score":5,"reason":"Could not parse analysis","_parse_failed":True}`.
  Propagate `"_parse_failed": True` (and `"_degraded_reason":"lite_parse_fail"`)
  onto the RETURNED dict at :2493-2504. Fix :2401 so a `_parse_failed` analysis
  is not logged as a normal decision. Grounds: dev.to (#5); FutureAGI (#7)
  "silently passing … a regression you will not see."
- **G3-B [OBS]** — Make the guard catch it. Extend `_degraded_scoring_check`
  (:2117-2131) to also count `n_degraded += 1` when
  `a.get("_parse_failed")` or `a.get("_degraded")` is truthy — so `score=5` can
  no longer evade it. This only affects the P1 degraded-scoring **alert**
  (:1113-1131) — it pages the operator, it does **not** change any trade. Add
  `summary["lite_parse_failures"] = <count>` so even a single parse-fail (below
  the ≥3 guard threshold) is visible. Grounds: SRE (#6) "alert on degraded, not
  just down"; FutureAGI (#7) per-failure-mode breakdown.
- **G3-C [FLAG]** — Drop the parse-fail from decide_trades input (fail-safe).
  Gate behind the **existing** `paper_synthesis_integrity_enabled` flag (same
  flag as the honest-degraded path at :1894). When **ON**, set `"_degraded":True`
  on the parse-fail dict so the **unconditional** guard at :1080 (`if
  analysis.get("_degraded"): return None`) drops it from candidate/holding and
  persists an honest NULL-score row (via :2748-2772) — matching phase-61.2
  semantics. When **OFF** → legacy HOLD-with-score-5 behavior is preserved BUT
  still logged + counted + guard-caught (G3-A/B are always-on regardless of the
  flag). **Fail-safe note:** dropping a parse-fail HOLD can only *remove* a
  spurious neutral — it can never create a BUY, so it cannot cause an unsafe
  trade. A GENUINE parsed HOLD is untouched (only the `else` branch is marked),
  preserving "distinguishable from a real HOLD." **NO risk threshold moved.**

### Cross-cutting invariants honored

- Observability (G1-A/B, G2-A, G3-A/B) is **always-on + $0** (log lines, dict
  markers, summary counters, a predicate extension — no new LLM calls).
- Every behavior change (G1-C ceiling reconcile, G3-C drop-on-parse-fail) is
  **flag-gated, default-OFF, byte-identical when OFF**.
- **NO risk threshold moved** (stops, sector caps, PBO/DSR, kill-switch all
  untouched); **historical_macro FROZEN** (no optimizer path touched); all
  additions **fail-safe** (a bad log/marker/counter can never open a trade).

---

## Gate envelope

```json
{
  "tier": "complex",
  "external_sources_read_in_full": 8,
  "snippet_only_sources": 26,
  "urls_collected": 34,
  "recency_scan_performed": true,
  "internal_files_inspected": 6,
  "gate_passed": true
}
```

`gate_passed: true` — 8 ≥ 5 sources read in full (incl. tier-1 arXiv + tier-2
regulatory), recency scan performed, 3-variant queries disclosed, brief written
incrementally, internal RE-ANCHOR complete with exact HEAD lines.
