# Money Diagnosis — phase-72 (write-first draft, started 2026-07-18)

Baseline: `handoff/current/money_recon_2026-07-18.md` (adversarially verified). Window segmentation (binding for this document — no single-root-cause framing):
1. **Late May (05-15..06-03)**: real gains booked (+$3,194.68 realized across the window's round trips), but the headline "+14% alpha" is partly a benchmark discontinuity — see P2.
2. **June (06-03..07-02)**: profit-taking-into-cash — winners exited (+$ realized), redeployment tapered to ~1 trade/week, zero-trade week Jun 15-21.
3. **July (07-03..present)**: degraded-scoring stall — 100% cash 07-03..07-08, brief AMD+MU re-entry 07-09, nothing since 07-13; book 97.2% cash.

---

## P0 — Scoring-rail root cause (step 72.0, in progress)

### Mechanism (verified, code-level)

The funnel converts LLM degradation into zero trades with no observable decision-seam log:

1. Rail/LLM failure → analyses come back `_degraded` or lite-HOLD.
2. Degraded analyses are dropped before decision (`backend/services/autonomous_loop.py:1103-1109`).
3. `decide_trades` emits a BUY only for `{BUY, STRONG_BUY}` and silently `continue`s on everything else — **no log line on that branch** (`backend/services/portfolio_manager.py:63,182-189`).
4. Result: 0 BUYs *and* 0 sells (frozen book), observed as "Executing 0 trades" in 8 of 9 cycles in the current log rotation.

**Two independent failure surfaces (research gate, `handoff/current/research_brief.md`):**

- **Surface A — pipeline rail.** Standard tier = `claude-sonnet-4-6` (`backend/config/settings.py:30`). It reaches the flat-fee Claude Code Max rail only when `paper_use_claude_code_route=True` (`settings.py:175`, **code default False**) AND the host OAuth session is alive (`claude_code_health_probe` runs `claude auth status`, `backend/agents/claude_code_client.py:380-425`). Otherwise every claude-* call silently regresses to the Anthropic direct API, whose credits are exhausted (phase-66 finding). The rail path scrubs `ANTHROPIC_API_KEY` from the subprocess env (`claude_code_client.py:287-297`) — but only on the rail path.
- **Surface B — meta-scorer.** `backend/services/meta_scorer.py:220-225` constructs its ClaudeClient with `anthropic_api_key` **directly**, bypassing `make_client()` and the route flag entirely. The conviction overlay is therefore credit-dead even when the rail is healthy, collapsing to flat conviction=10 and erasing ranking. This is the open "Meta-scorer LLM-leg repair (credit-exhaustion class)" follow-up (`cycle_block_summary.md:27`).

**Error-semantics context (external, cited in research brief):** a $0 Anthropic credit balance returns a non-retryable 402 billing error, but "sometimes returns a rate limit error instead" — so 429s in our logs cannot be assumed to be rate-limiting; disambiguation requires balance/headers, not status codes. The SDK auto-retries only connection/429/5xx, so a credit-dead key can also present as a retry-then-fail pattern.

### Onset + failing-provider attribution (forensics `wf_0542bf62-ffb`, adversarially verified; verbatim lines in `live_check_72.0.md`)

- **ROOT onset 2026-05-17 03:55:44 CEST** — first HTTP-400 `invalid_request_error` "Your credit balance is too low to access the Anthropic API" (`req_011Cb7JtX5fXgpryDPiYpSxo`, orchestrator Enrichment), date-anchored via the adjacent APScheduler "scheduled at 2026-05-17" lines. Genuine Anthropic-direct successes **cease after 05-17**: the BQ token audit shows every later "ok" day (05-19..06-10) is exactly input=1000/output=50 — synthetic smoke fixtures, not live scoring.
- **Surface B onset 2026-05-22 22:32:06** — first "meta_scorer LLM call failed" (initially the SecretStr header TypeError, fixed by phase-51.1; thereafter the credit-400 on the same direct-API path). The meta-scorer LLM leg then failed **every single trading day 2026-05-22 → 2026-07-17** (all gaps are weekends/05-25 holiday — no recovery window ever), and writes **zero rows to `llm_call_log`** (it bypasses the rail AND the telemetry writer).
- **Observable trade-freeze onset 2026-06-11/15** — first zero-score cluster in `analysis_results` and the first "no-LLM fallback" (20:01:22) / "Degraded-scoring guard fired" (21:09:35) markers on 06-11; 100% HOLD / final_score=0.0 from 06-15; last normal BUY 2026-06-10. The 06-11 guard markers are **instrumentation deployed ~3.5 weeks after the root onset**, not the onset. NAV flat since 05-29 (late-May flatness = churn + meta-scorer ranking loss; hard freeze from 06-11/15).
- **Failing provider/key**: the Anthropic **direct API on the metered key — credit exhaustion** (HTTP 400 "credit balance too low", dominant class), plus HTTP 401 "invalid x-api-key" during the away window only (= the cc_rail credential death of 06-15). **Not 402/429** as the research brief's external error-taxonomy suggested. Direct-API daily series: blackout 06-15..07-03, partial recovery 07-07..07-09, re-blackout 07-13..07-15, ok-rows again 07-16..17 *while the meta-scorer still 400s the same evening* — balance/keys appear intermittent or per-path; flagged as an open question for the operator (only the console balance can settle it).
- **Surface A verdict — contract hypothesis PARTIALLY REFUTED (recorded honestly):** OAuth is **ALIVE** (`claude auth status` → loggedIn:true, max plan) and the route is **effectively ON at runtime** (the running backend, pid 98681 since 07-08 23:24, invokes the rail; the probe never logged FAILED). Today's Surface-A failure is instead: (a) rail-tagged calls **regressing to the credit-dead direct API** (BQ: `agent='cc_rail'` but `provider='anthropic'` — 560 ok / 2,996 fail), (b) the genuine flat-fee CLI rail barely used (19 rows ever, **silent after 07-09 19:17**), and (c) **305 subprocess timeouts @120s** tripping the rail breaker OPEN (bot-token pages delivered 07-10/13/14/15).

### Away-posture vs live defect (verified per window)

- **BEFORE 06-12 — healthy engine, latent defect.** BUYs executed every day 06-01..06-10 (last normal BUY 06-10); analyses scored 4-of-5 BUY recs at avg final_score ~6 with zero fabricated 0.0s. The only degradation was the meta-scorer conviction overlay (credit-dead since 05-17/22) — ranking-only, non-trade-blocking. Caveat: the full pipeline had been on a silent lite fallback since 06-01 until the 60.1 repin.
- **AWAY 06-12..07-06 — DEFECT, not posture.** `away-ops-rules.md` rail 4 binds only *dev sessions* and explicitly exempts the "existing Gemini pipeline" — no away rail disabled the pipeline's own LLM calls. The loop kept running (5-8 analyses/day) but fabricated HOLD/0.0 (the 61.2 synthesis defect) because the cc_rail credentials died 06-15 (ECONNRESET → 401) while `backend/services/alerting.py` **did not exist** (4 P1 import sites swallowed ModuleNotFoundError) — 34 consecutive sessions failed with zero pages.
- **AFTER 07-06 — DEFECT persists.** Phase-66 fixed the *plumbing* (probe-gate, breaker, paging, OAuth restored) but not the scoring defect: the 07-09 AMD+MU BUYs came via the pre-existing **Gemini LITE fallback** (`_path=lite`, APPROVE_REDUCED) — a one-off catch, not the restored rail; the operator-approved synthesis-integrity + RJ-shape flags were **never written to the agent-locked `backend/.env`** (harness_log: "agent can't write the agent-locked backend/.env; NOW OPTIONAL"); the meta-scorer was left degraded ("Leave degraded for now", 07-08); cycles 78-111 were all harness/audit dev work. 07-10..07-17: all 0-buy / 100% HOLD / ~100% final_score=0.0.
- **The Gemini/Vertex leg had ZERO failures across the entire window** — it is the only fully healthy provider, and it produced the only post-return BUYs.

### Restoration plan (finalized on forensics — priority order corrected by evidence)

**Correction from forensics:** the top lever is NOT the route flag or OAuth (both fine). The stack rank is now: (1) the **operator decision on Anthropic direct-API credits** — top up or formally abandon the direct API (dead since 05-17; every claude-* leg regresses to it); (2) **promote the already-approved synthesis-integrity + RJ-shape flags into `.env` + restart** — the 61.2 fabrication is what converts rail hiccups into HOLD/0.0, and the flags were approved 07-09 but never written (agent-locked `.env`); (3) **R1 meta-scorer decoupling** (credit-dead every trading day since 05-22, telemetry-blind); (4) **R2 fail-forward to Vertex-Gemini** (zero failures all window — the only healthy leg, and the source of the only post-return BUYs); (5) **rail throughput** — the genuine CLI rail is near-unused (19 rows, silent since 07-09) with 305 subprocess timeouts @120s; why rail-tagged calls regress to the direct API belongs to R2's routing work.

Split per session rules: **code-side levers become executor-tagged masterplan steps** (cheaper sessions implement); **.env/runtime-side levers are operator actions** → `operator_decision_sheet_72.md` (recommend-only).

Code-side (to be appended as pending masterplan steps 72.0.R*):
- **R1**: Route the meta-scorer through `make_client()` / the claude-code route (or repin `meta_scorer_model` to a Vertex-Gemini id) so Surface B stops depending on direct-API credits (`meta_scorer.py:203-225`). [executor: sonnet-4.6/high]
- **R2**: Fail-forward for the standard tier: when the cc_rail probe is dead, fall to Vertex-Gemini instead of lite→HOLD (provider order seam `backend/agents/llm_client.py:1983-2042`; phase-37.2 precedent moved `deep_think_model` to Vertex for exactly this failure class, `settings.py:31`). [executor: opus-4.8/xhigh — routing judgment]
- **R3**: Observability: log line on the silent non-BUY drop (`portfolio_manager.py:188`) so a degraded cycle is visible at the decision seam. [executor: sonnet-4.6/high]
- **R4**: Verify the existing degraded-alert seams page live via the bot-token path (`autonomous_loop.py:360-398,902-958`). [executor: sonnet-4.6/high]

Operator-side (decision sheet; corrected by forensics — route flag + OAuth are FINE, do not touch):
- **Decide the Anthropic direct-API credit question**: top up the metered key (dead since 2026-05-17) OR formally abandon Anthropic-direct and let R1/R2 move the affected legs to the rail/Vertex. Check the console balance — logs show 400 "credit too low" on 07-17 evening while other calls succeeded ok the same hour; only the console can disambiguate.
- **Write the 2026-07-09-approved flags into the agent-locked `backend/.env`** (`PAPER_SYNTHESIS_INTEGRITY_ENABLED=true`, `PAPER_RISK_JUDGE_SHAPE_FIX_ENABLED=true`) **and restart the backend** — the running process (pid 98681) started 2026-07-08 23:24 and any later `.env` edit is not loaded. This was approved but never applied (harness_log 07-09: "agent can't write the agent-locked backend/.env").
- Provide the pending `.env` grep so 72.1 can reconcile every token line against live values.

Every restoration step carries an immutable live_check: **a post-restoration cycle log line showing non-degraded scoring** (meta-scorer LLM leg active, no degraded-guard fire, ≥1 non-HOLD recommendation or an honest market-driven HOLD).

---

## P1 — Approved-but-unapplied tokens (step 72.1 — reconciled 2026-07-18)

Full 15-row reconciliation in `operator_decision_sheet_72.md` §P1 (researcher `wf_ce9e1cac-e72`, gate passed). Headline: **exactly one true approval-to-deployment gap** — the 2026-07-09 PROMOTE SYNTHESIS-INTEGRITY + RJ-SHAPE bundle (settings.py:197/:311, default False), double-blocked: never written to the agent-locked `.env` AND the backend process (pid 98681, up since 07-08 23:24) predates the approval. The 06-11 keystroke batch (swap-churn-fix / data-integrity / RJ-binding) IS applied and loaded (runtime-corroborated for 60.2). The phase-69 tokens (KS-PEAK-RESET, sign_safe_overlays, regime_net_liquidity) are owed-not-approved — correctly dark, ranked in P3, not deployment gaps. Structural root: no auto-apply path (`operator_tokens.py:52-61` maps only AWAY DRILL) + one-directional sentinel (`sentinel.sh:102-126`, blind to approved-but-unapplied); report-only reverse-leg reconciliation queued as step 72.1.1 [executor: sonnet-4.6/high]. Live `.env` values remain UNCONFIRMED-marked pending the operator grep (ACT-NOW #4).
## P2 — Measurement integrity (step 72.2 — pending)
## P3 — Earning-capacity levers (step 72.3 — pending; output in operator_decision_sheet_72.md)
## P4 — Regime deployment policy (step 72.4 — pending)
