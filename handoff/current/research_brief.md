# Research Brief — phase-72.0 P0 SCORING-RAIL RESTORATION AUDIT

**Step:** phase-72.0 · **Tier:** complex · **Audit-class:** false
**Researcher spawn:** 2026-07-18 · **Brief path:** handoff/current/research_brief.md

> WRITE-FIRST: created before any source was read; grown incrementally.

## Objective

Root-cause the degraded LLM scoring rail holding the paper book at ~97% cash /
0-trade cycles since early July 2026, and surface the exact config/code seams a
later RESTORATION masterplan step would touch. THIS session does not edit
product code.

---

## Queries run (3-variant discipline: current-year / last-2-year / year-less)

**Topic 1 — Anthropic API credit-exhaustion failure semantics**
- Current-year: `Anthropic API error 429 rate_limit vs credit balance too low 400 error retry-after 2026`
- Last-2-year: `Anthropic Claude subscription OAuth rail credit exhaustion outage 2025 2026`
- Year-less: `Anthropic Claude API HTTP error codes rate limit billing retry`

**Topic 2 — Claude Code subscription rail vs metered API (headless)**
- Current-year: `Claude Code subscription usage limits vs API billing headless programmatic 2026`
- Last-2-year: `Claude Code Agent SDK headless authentication subscription 2025`
- Year-less: `Claude Code Pro Max plan authentication ANTHROPIC_API_KEY`

**Topic 3 — LLM decision-system fallback under outage/degradation**
- Current-year: `LLM-dependent trading decision system model outage fail-closed fallback design 2026`
- Last-2-year: `LLM fallback routing provider outage graceful degradation 2025`
- Year-less: `circuit breaker pattern fail-fast graceful degradation dependency unavailable` (Fowler / Nygard canonical)

---

## Read in full (>=5 required; counts toward gate)

| # | URL | Accessed | Kind | Fetched how | Key finding |
|---|-----|----------|------|-------------|-------------|
| 1 | https://platform.claude.com/docs/en/api/errors | 2026-07-18 | Official doc (T2) | WebFetch full | 402=`billing_error` (billing/payment); 429=`rate_limit_error`; SDKs auto-retry **only** connection errors, rate limits, 5xx (exp backoff, 2x default, honor `retry-after`) — **402 billing is NOT in the auto-retry set**. |
| 2 | https://support.claude.com/en/articles/11145838-use-claude-code-with-your-pro-or-max-plan | 2026-07-18 | Official doc (T2) | WebFetch full | "If you have an ANTHROPIC_API_KEY environment variable set... Claude Code will use this API key... instead of your Claude subscription, resulting in API usage charges." No documented headless/cron guidance for subscription auth (gap). |
| 3 | https://zed.dev/blog/anthropic-subscription-changes | 2026-07-18 | Authoritative blog (T3) | WebFetch full | June 15 2026 split: first-party tools (chat, **official Claude Code CLI**) use subscription limits; third-party/ACP/`claude -p`/Agent-SDK draw a **separate monthly Agent-SDK credit** billed at API rates. Once credits deplete, "requests stop until credit resets" unless extra usage enabled. |
| 4 | https://claudestatus.com/blog/claude-api-error-codes | 2026-07-18 | Community (T5) | WebFetch full | 429="you specifically used your quota" → backoff w/ jitter + honor `retry-after`; 529="system overloaded for everyone". Escalate to on-call on "429 with quotas you did not expect". (402 not covered — confirms 402 is a distinct billing class.) |
| 5 | https://futureagi.com/blog/what-is-llm-fallback-strategy-2026/ | 2026-07-18 | Industry (T4) | WebFetch full | 5 fallback strategies (provider-rotation / model-downgrade / retry-then-fallback / cache-on-failure / manual-route). "A chain without a held-out quality floor silently degrades output" — recommends conservative degradation **with a quality floor**, not silent continuation. |
| 6 | https://www.truefoundry.com/blog/llm-failover-load-balancing-provider-outages | 2026-07-18 | Industry (T4) | WebFetch full | Single-gateway pattern; ordered fallback lists per status code. **Transient (retry/failover): 5xx, 429, timeouts. Non-transient (surface, don't failover): content-filter / policy / billing — "a property of the request, not a transient fault. Retrying it wastes time and money."** Circuit breaker = closed/half-open/open, fail-fast so a down provider doesn't cascade. |
| 7 | https://martinfowler.com/bliki/CircuitBreaker.html | 2026-07-18 | Authoritative canonical (T3) | WebFetch full | Closed/open/half-open. "Once failures reach a threshold, the circuit breaker trips, and all further calls return with an error, without the protected call being made at all." "Breakers on their own are valuable, but clients using them need to react to breaker failures" (queue / stale-data fallback). "Any change in breaker state should be logged." |
| 8 | https://hidekazu-konishi.com/entry/anthropic_claude_api_errors_reference.html | 2026-07-18 | Technical blog (T4) | WebFetch full | **402 `billing_error` = NON-retryable** ("account state, not a transient condition"; must be resolved at account level). **429 `rate_limit_error` = retryable** (honor `retry-after`, read `anthropic-ratelimit-*`). Crisp retryable/non-retryable split. |
| 9 | https://arxiv.org/html/2606.14589 | 2026-07-18 | Preprint / peer-tier (T1) | WebFetch full | Longitudinal taxonomy of silent failures in a production LLM-agent runtime. 5 classes; **Class C error-swallowing** + **Class D fail-plausible fabrication** ("transforms an internal error into coherent, contextually appropriate, and false output"). "time-to-detect dominates time-to-repair by one to two orders of magnitude"; "the longest-latency failures lived in seams... between declared state and runtime state"; unit tests reached 0% detection for this class. |

---

## Identified but snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|-----|------|-------------------------|
| https://tygartmedia.com/claude-code-billing-credit-pool-2026/ | Blog | Redundant with zed.dev on the June-15 split |
| https://www.developersdigest.tech/blog/claude-code-usage-limits-playbook-2026 | Blog | Usage-limit playbook; covered by official doc |
| https://www.pravinkumar.co/blog/claude-june-15-billing-change-explained-2026 | Blog | Duplicate of the billing-change narrative |
| https://hidekazu-konishi.com/entry/anthropic_claude_api_errors_reference.html | Technical blog | Candidate for read-in-full (topic 1 secondary) |
| https://buttondown.com/redpen/archive/http-error-codes-429-529-and-400-what-they-mean/ | Blog | 429/529/400 explainer; official doc supersedes |
| https://www.getmaxim.ai/articles/best-llm-gateway-to-design-reliable-fallback-systems-for-ai-apps/ | Industry | Gateway roundup; truefoundry covers the pattern |
| https://theroadtoenterprise.com/blog/model-agnostic-ai-layer-fallbacks | Blog | Model-agnostic layer; overlaps futureagi/truefoundry |
| https://www.buildmvpfast.com/blog/building-with-unreliable-ai-error-handling-fallback-strategies-2026 | Blog | Error-handling overview; overlaps |
| https://platform.claude.com/docs/en/api/rate-limits | Official doc | Rate-limit tiers; 429 mechanics already captured |
| https://arxiv.org/html/2606.08162 | Preprint | Companion silent-failure paper (Entropy Principle); #2606.14589 read instead |
| https://earezki.com/ai-news/2026-06-27-why-llm-agents-fail-silently-and-how-to-debug-them/ | Blog | Token-budget/schema-drift/swallowed-exception silent failures; academic paper preferred |
| https://medium.com/data-science-collective/why-ai-agents-keep-failing-in-production-cdd335b22219 | Blog | AI-agent production failure survey; overlaps sources #5/#9 |

---

## Recency scan (2025-2026)

Searched 2025-2026 literature on all three topics (dedicated year-locked +
year-less passes). **Four new findings** that complement/supersede older canon:

1. **June 15 2026 Anthropic billing split** (Source #3, zed.dev) — first-party
   Claude Code CLI stays on the subscription pool; Agent-SDK / `claude -p` /
   third-party apps move to a separate metered Agent-SDK credit. Subscribers
   were told (WebSearch topic-2) it "is not taking effect yet" and `claude -p`
   still works on subscriptions "exactly as before." **Material** — it is the
   live policy governing which pool pyfinagent's `claude --print` subprocess
   draws from; a future enforcement could change the rail's flat-fee assumption.
2. **2026 practitioner convergence on error-class-aware fallback** (Sources
   #5/#6) — transient (5xx/429/timeout) → failover; billing/policy → surface,
   don't retry. Supersedes the older "just add exponential backoff to
   everything" advice; validates pyfinagent's retry suppression on
   rail_guard_skipped empties (settings.py:191-196).
3. **arXiv:2606.14589 (June 2026)** — production LLM-agent silent-failure
   taxonomy. **Directly on-point recent academic work**: Class C (error
   swallowing) + Class D (fail-plausible fabrication) are the exact failure this
   audit concerns; "time-to-detect dominates time-to-repair by 1-2 orders of
   magnitude" and "failures live in seams between declared state and runtime
   state" describe pyfinagent's 3-week silent away-week outage precisely. New
   framing (fail-plausible) that older reliability canon (Fowler/Nygard) lacks.
4. **LangChain May-2025 postmortem** (WebSearch topic-3) — a cert/credential
   expiry incident whose remediation list led with "proactive certificate status
   monitoring + automated expiry alerts." Maps onto the OAuth-session-expiry
   class that pyfinagent's `claude auth status` probe (claude_code_client.py:380)
   is designed to catch. Complements, does not supersede, the health-probe canon.

The older canon (Fowler CircuitBreaker, Source #7) remains valid and is directly
implemented by pyfinagent's phase-66.1 rail guard — the 2026 work adds the
LLM-specific fail-plausible class on top, not a replacement.

---

## Key findings (external)

1. **A $0 credit balance is a 402 `billing_error`, NOT a 429, and is
   NON-retryable.** Official Anthropic docs list 402=`billing_error` ("issue
   with your billing or payment information") separately from 429=`rate_limit_error`;
   the SDK auto-retry set is explicitly connection-errors + rate-limits + 5xx
   only — billing is excluded (Source #1, platform.claude.com/api/errors,
   2026-07-18). Practitioner guidance agrees: billing/policy errors are "a
   property of the request, not a transient fault. Retrying it wastes time and
   money" (Source #6, truefoundry). **Implication for pyfinagent:** a
   credit-exhausted direct-API key must be treated as a hard route-switch
   signal, not a retry loop.
   - Caveat (Source #1 warning box): "When your credit balance is $0, Anthropic
     sometimes returns a rate limit error instead of a billing error" — so a 429
     can *masquerade* as a rate limit when the true cause is $0 credit. Disambiguate
     via the console balance / `anthropic-ratelimit-*` headers, not the code alone.

2. **`ANTHROPIC_API_KEY` in the environment silently overrides subscription
   auth** and routes to metered API billing (Source #2, official). pyfinagent's
   rail already counters this: `claude_code_invoke` scrubs `ANTHROPIC_API_KEY` +
   `ANTHROPIC_AUTH_TOKEN` from the subprocess env so the CLI uses the ~/.claude
   OAuth (Max) path (internal: claude_code_client.py:287-297).

3. **The official Claude Code CLI still draws from the subscription pool; the
   June-15-2026 split targets Agent-SDK / `claude -p` / third-party apps**
   (Source #3, zed.dev). pyfinagent invokes `claude --print` (headless print
   mode), which is the boundary case: Anthropic's own subscribers were told the
   split "is not taking effect yet" and `claude -p` "continue[s] to work with
   Claude subscriptions exactly as before" for now (Source, WebSearch topic-2).
   **Risk:** the rail's cost assumption (flat-fee Max) is contingent on a policy
   Anthropic has signalled it may change with notice — a restoration plan should
   record this as a watch-item, not a guarantee.

4. **Subscription-authenticated Claude Code has NO documented headless/cron
   behavior; the failure mode in unattended use is an expired OAuth session**
   (Source #2 gap + internal probe rationale). This matches pyfinagent's own
   post-mortem: the 2026-06-01..06-09 away week "ran with this rail silently down
   (expired OAuth session in unattended mode)" (claude_code_client.py:389-392).

5. **Practitioner + canonical consensus: fail-fast on a dead dependency, degrade
   to a floor — do not silently continue on a degraded signal.** 2026 fallback
   taxonomy centres on a single gateway boundary + ordered, error-class-aware
   fallback chains + a circuit breaker (closed/half-open/open) (Sources #5,#6).
   The named anti-pattern is "model-downgrade without a quality floor degrades
   output silently on hard prompts" (Source #5). For a scoring/decision model
   specifically, the recommended posture is **conservative degradation with an
   explicit held-out quality floor** — i.e. a *known* degraded state that gates
   action, not a fabricated score. pyfinagent's degraded-scoring guard +
   `_degraded` marker (return honest None, never a fabricated 0.0/HOLD) is the
   correct application of this pattern; the defect is upstream (the rail being
   down at all), not the guard.

6. **The canonical Circuit Breaker pattern (Fowler, Source #7) prescribes
   exactly pyfinagent's rail guard — with one gap the literature highlights.**
   Fowler: trip after a threshold, reject calls without making them, and "any
   change in breaker state should be logged." pyfinagent's phase-66.1 rail guard
   (claude_code_client.py:82-211, `claude_rail_breaker_threshold=20`) implements
   this. But Fowler also stresses "clients using them need to react to breaker
   failures" with a *fallback action* (queue-for-later / stale-data). pyfinagent's
   current reaction is **degrade-to-lite → HOLD → 0 trades** (fail-closed, earns
   $0), where the literature would prefer a fail-forward to an alternate provider
   under a quality floor. The **2026 academic anchor (arXiv:2606.14589, Source
   #9)** adds the LLM-specific twist: the most dangerous reaction is **Class D
   fail-plausible fabrication** — a coherent but false score. pyfinagent's
   `_degraded` marker is the documented countermeasure, and the paper's headline
   ("time-to-detect dominates time-to-repair by 1-2 orders of magnitude; failures
   live in seams between declared state and runtime state") is precisely why the
   restoration step must produce **live-host evidence** (green `claude auth
   status` + a confirmed `.env` route value), not just a code read — the seam
   between the committed default (`paper_use_claude_code_route=False`) and the
   live runtime state is exactly where this failure class hides.

---

## Internal code inventory

| File:line | Role | Status / finding |
|-----------|------|------------------|
| `backend/config/settings.py:30` | `gemini_model="claude-sonnet-4-6"` — the STANDARD tier (enrichment+debate) | Claude-routed. If cc_rail off/down → Anthropic-direct (credit-dead). |
| `backend/config/settings.py:31` | `deep_think_model="gemini-2.5-pro"` | **Already documents the credit-exhaustion class**: "Previously claude-opus-4-7 -- caused silent regression to Anthropic credit-exhaustion on fresh checkout/restart." Deep-think tier deliberately moved OFF Anthropic to Vertex Gemini in phase-37.2. |
| `backend/config/settings.py:175-178` | `paper_use_claude_code_route` (**default False**) | THE rail switch. False → every claude-* call bills api.anthropic.com direct. "flip to False before flipping real_capital_enabled to True." Live .env value UNCONFIRMED (permission-blocked). |
| `backend/config/settings.py:179-184` | `claude_rail_breaker_threshold=20` | Consecutive cc_rail failures that trip the breaker for the cycle. |
| `backend/config/settings.py:185-190` | `claude_code_timeout_s=150` | cc_rail subprocess timeout. |
| `backend/config/settings.py:191-196` | `claude_code_empty_retry_max=2` | Extra retries on errored-empty; **only effective when `paper_synthesis_integrity_enabled=True`**. rail_guard_skipped empties never retried (open breaker/probe-dead). |
| `backend/config/settings.py:197-200` | `paper_synthesis_integrity_enabled` (**default False**) | BUY-survival-under-rail-degradation umbrella (SynthesisDegradedError routing + honest degraded row + retry-on-empty + meta-scorer rank-normalize). Operator-APPROVED 2026-07-09 per money_recon; live state UNCONFIRMED. |
| `backend/config/settings.py:201-204` | `paper_position_recommendation_fix_enabled` (default False) | Revives signal_downgrade SELL; unsafe-combination guard warns if ON while synthesis-integrity OFF. |
| `backend/config/settings.py:402` | `meta_scorer_enabled` (default False) | Gates the conviction overlay call. |
| `backend/agents/llm_client.py:1983-2002` | cc_rail routing branch | `if model.startswith("claude-") and paper_use_claude_code_route:` → `ClaudeCodeClient` subprocess (Max OAuth). |
| `backend/agents/llm_client.py:2005-2023` | Anthropic-direct branch + **routing-breach guard** | If route=True but ClaudeCodeClient import failed → raises `ValueError("Routing breach...")` so it never silently bills the credit-dead API. If route=False + key present → direct (credit-dead). |
| `backend/agents/llm_client.py:2047-2051` | No-key ValueError | If route=False AND no ANTHROPIC_API_KEY → "Model has no compatible key." |
| `backend/agents/claude_code_client.py:82-211` | Rail guard: probe-gate + circuit breaker (phase-66.1) | `_RailGuardState`, `rail_guard_reset/disable/status`, `_rail_guard_record_failure` trips at threshold → ONE P1 page. |
| `backend/agents/claude_code_client.py:214-377` | `claude_code_invoke` | Runs `claude --print --output-format json` via stdin, scrubbed env (no API key), success = `subtype=="success"`. Non-zero exit logs BOTH stdout+stderr (07-07 quota burst logged empty stderr). |
| `backend/agents/claude_code_client.py:287-297` | env scrub | Removes ANTHROPIC_API_KEY/ANTHROPIC_AUTH_TOKEN so CLI uses OAuth not credit-dead key. |
| `backend/agents/claude_code_client.py:380-425` | `claude_code_health_probe` | Runs `claude auth status` (token-less, scrubbed env). Returns (ok, detail). Never raises. This is the "rail health probe FAILED" source. |
| `backend/services/autonomous_loop.py:360-398` | Per-cycle probe call | `rail_guard_reset` → `claude_code_health_probe` in a thread; FAIL → `rail_guard_disable` + P1 "claude-code rail health probe FAILED: ..." + "lite analyzer + conviction overlay will run degraded fallbacks". |
| `backend/services/autonomous_loop.py:902-958` | Meta-scorer degraded detection | `meta_scorer_enabled` → `meta_score_candidates`; sets `meta_scorer_degraded=True` when ALL on no-LLM fallback; streak ≥2 → P1 "root_cause_hint: direct-API Anthropic credit/key (live_check_66.2.md 5d)". |
| `backend/services/autonomous_loop.py:1096-1109` | Degraded-analysis drop | `_path in (lite,full,degraded)` persists honest BQ row; `if _degraded: return None` → degraded analyses excluded from candidate/holding lists. |
| `backend/services/autonomous_loop.py:1149-1174` | Degraded-scoring guard (phase-56.2 F-5) | Fires P1 "Degraded-scoring guard fired: N/N analyses scored 0/degraded"; stamps cycle `degraded`. |
| `backend/services/meta_scorer.py:203-225` | **Meta-scorer bypasses the cc_rail** | Reads `anthropic_api_key` directly, constructs `ClaudeClient(model=meta_scorer_model default claude-haiku-4-5, api_key=anthropic_key)` — i.e. api.anthropic.com DIRECT, NOT `make_client()` and NOT the subscription rail. No key / call-fail → `_fallback_all`. |
| `backend/services/meta_scorer.py:138-177` | `_fallback_conviction` | Constant **10** ("live composites run 78-163 → every fallback cycle emitted conviction 10.00 for all candidates, erasing the ranking"). |
| `backend/services/portfolio_manager.py:63` | `_BUY_RECS={"BUY","STRONG_BUY"}` | Only these trigger a buy. |
| `backend/services/portfolio_manager.py:182` | `rec=(analysis.get("recommendation") or "HOLD").upper()` | Degraded/lite-null rec defaults to HOLD. |
| `backend/services/portfolio_manager.py:188-189` | `if rec not in _BUY_RECS: continue` | **Silent non-BUY drop — NO log line.** A degraded cycle produces HOLDs → all continue → 0 buys → 0 trades. |
| `backend/config/model_tiers.py:55-103` | `_BUILD_TIER` role→model | mas_communication/mas_research/autoresearch_smart=sonnet-4-6; mas_main/mas_qa/autoresearch_strategic=opus-4-8; gemini roles=Vertex. |

---

## Restoration seams (file:line — the exact seams a restoration step would touch)

1. **`backend/config/settings.py:175` `paper_use_claude_code_route`** — the
   master rail switch. Restoration must confirm the LIVE `.env`
   `PAPER_USE_CLAUDE_CODE_ROUTE` value and set it True (or migrate the pipeline
   off Anthropic-direct). Default-False means a fresh env silently regresses to
   the credit-dead direct API.
2. **`backend/agents/claude_code_health_probe` (claude_code_client.py:380) +
   the host `claude auth status`** — the probe endpoint. Restoration's live
   evidence is a green `claude auth status` on the trading host (OAuth session
   not expired), captured while unattended. This is the #1 root-cause candidate
   for July's degraded cycles.
3. **`backend/services/meta_scorer.py:220-225` (ClaudeClient direct
   construction)** — the seam that makes the meta-scorer credit-fragile.
   Restoration option: route the meta-scorer through `make_client()` /
   `paper_use_claude_code_route` so it uses the subscription rail like the rest
   of the pipeline, OR set `meta_scorer_model` to a Vertex-Gemini id (mirroring
   the phase-37.2 deep_think move at settings.py:31). Either removes the
   direct-API `ANTHROPIC_API_KEY` dependency.
4. **`backend/config/settings.py:197` `paper_synthesis_integrity_enabled`** —
   the operator-approved (2026-07-09) BUY-survival flag; restoration must confirm
   its LIVE value (approved ≠ deployed). Gates the retry-on-empty
   (`claude_code_empty_retry_max`) and the meta-scorer rank-normalization.
5. **Provider fallback ORDER in `backend/agents/llm_client.py:1983→2005→2030→2042`**
   — the routing chain is cc_rail → Anthropic-direct → GitHub Models → Vertex.
   A restoration could add a Vertex-Gemini fail-forward for standard-tier claude
   models when the cc_rail is probe-dead (today it hard-fails or bills the
   credit-dead key), giving the pipeline a non-degraded path instead of
   dropping to lite. (External sources #5/#6: ordered, error-class-aware
   fallback chain terminating in a self-hosted/alt-provider leg.)
6. **`backend/config/settings.py:30` `gemini_model="claude-sonnet-4-6"`** — the
   standard-tier model pin. A restoration lever is repinning standard tier to a
   Vertex-Gemini workhorse (the phase-37.2 precedent for deep_think at :31),
   removing the Anthropic dependency from the highest-volume tier entirely.
7. **`backend/services/portfolio_manager.py:188`** — add a log line on the
   silent non-BUY drop so a degraded cycle is observable at the decision seam
   (money_recon flagged the no-log continue). Not a root-cause fix but a
   diagnosability seam.

---

## Application to pyfinagent

The literature maps cleanly onto a **two-surface root cause**:

- **Surface A — the pipeline rail (cc_rail / OAuth):** the standard tier
  (settings.py:30) is `claude-sonnet-4-6`. It only reaches the flat-fee
  subscription when `paper_use_claude_code_route=True` (settings.py:175, default
  **False**) AND the host OAuth session is alive (claude_code_health_probe /
  `claude auth status`). If either is false, calls either hard-fail (routing
  breach ValueError, llm_client.py:2013) or bill the credit-exhausted direct API
  (a 402 `billing_error` per Source #1, which is NON-retryable per Sources
  #1/#6). The probe gate then disables the rail → lite analyzers run degraded →
  analyses score 0/HOLD/degraded → dropped (autonomous_loop.py:1103) → 0 BUYs
  (portfolio_manager.py:188).

- **Surface B — the meta-scorer (independent, direct-API):**
  meta_scorer.py:203-225 constructs `ClaudeClient` with `anthropic_api_key`
  directly, bypassing the subscription rail entirely. So even a perfectly healthy
  cc_rail leaves the conviction overlay credit-dead → `_fallback_conviction`
  returns a flat 10 → ranking erased. This is the "Meta-scorer LLM-leg repair
  (credit-exhaustion class)" open follow-up (cycle_block_summary.md:27). It is a
  RANKING degradation, not the BUY/HOLD decision itself — necessary to fix for
  signal quality but not the sole cause of 0 trades.

**Consensus vs debate:** External sources agree a dead-credit/billing error is
non-transient and must trigger a route-switch, not a retry — pyfinagent already
suppresses retries on rail_guard_skipped empties (settings.py:191-196), which is
correct. Where the literature would push further: pyfinagent's fallback chain
today *degrades* (to lite) rather than *fails forward* to a live alternate
provider (Vertex Gemini is already wired for deep_think). The debate is
fail-closed (HOLD, current behavior — safe, but earns $0) vs fail-forward
(route standard tier to Vertex when the cc_rail is probe-dead). Sources #5/#6
favor an ordered fallback that terminates in a non-degraded alternate; the
pyfinagent-specific counter is that a *degraded* analysis feeding a live BUY is
the exact bug phase-60.3/61.2 fixed — so any fail-forward must preserve the
quality floor (the `_degraded` marker), not bypass it.

**Pitfalls (from literature):** (a) silent model-downgrade without a quality
floor (Source #5) — pyfinagent avoids this via `_degraded`; (b) retrying a
billing error (Sources #1/#6) — pyfinagent avoids this via retry suppression;
(c) treating a $0-balance 429-masquerade as a rate limit (Source #1 warning) —
pyfinagent's OAuth-scrub sidesteps the direct API for the rail but NOT for the
meta-scorer (Surface B).

---

## Research Gate Checklist

- [x] >=5 authoritative external sources READ IN FULL via WebFetch (**9** read in full)
- [x] 10+ unique URLs total (incl. snippet-only) — 21 collected
- [x] Recency scan (last 2 years) performed + reported (4 findings; see section)
- [x] Full papers / pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim

Source-quality mix (hierarchy satisfied): 1 preprint (T1), 2 official docs (T2),
2 authoritative blogs (T3, incl. Fowler canonical), 2 industry (T4), 1 technical
blog (T4), 1 community (T5). Not a community-heavy set.

---

## JSON envelope

```json
{
  "tier": "complex",
  "external_sources_read_in_full": 9,
  "snippet_only_sources": 12,
  "urls_collected": 21,
  "recency_scan_performed": true,
  "internal_files_inspected": 7,
  "coverage": {
    "audit_class": false,
    "rounds": 1,
    "dry_rounds": 0,
    "K_required": 2,
    "new_findings_last_round": 0,
    "dry": false
  },
  "summary": "Root cause is a two-surface degraded scoring rail. Surface A (pipeline): standard tier is claude-sonnet-4-6 (settings.py:30) which only reaches the flat-fee Max subscription when paper_use_claude_code_route=True (settings.py:175, DEFAULT FALSE) AND the host OAuth session is alive (claude_code_health_probe / `claude auth status`, claude_code_client.py:380). If either is false, calls hard-fail (routing-breach ValueError, llm_client.py:2013) or bill the credit-exhausted direct API (402 billing_error, NON-retryable per official docs); the probe gate disables the rail -> lite/degraded analyses -> dropped (autonomous_loop.py:1103) -> 0 BUYs (portfolio_manager.py:188 silent non-BUY continue). Surface B (meta-scorer): meta_scorer.py:220-225 builds ClaudeClient with anthropic_api_key DIRECTLY, bypassing the subscription rail, so the conviction overlay is credit-dead even when the cc_rail is healthy -> flat conviction=10 -> ranking erased (the open 'credit-exhaustion class' follow-up). External consensus: a $0-credit/billing error is non-transient -> route-switch, never retry; degrade under a quality floor, never fabricate a score (arXiv:2606.14589 Class D). Restoration seams enumerated at settings.py:30/175/197, meta_scorer.py:220-225, claude_code_client.py:380, llm_client.py:1983-2044, portfolio_manager.py:188. THIS session made no code changes.",
  "brief_path": "handoff/current/research_brief.md",
  "gate_passed": true
}
```
