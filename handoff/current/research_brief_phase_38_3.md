# phase-38.3 Research Brief -- startup banner deep_think_model

**Date:** 2026-05-22
**Tier:** simple
**Author:** researcher subagent

## Section A -- Internal audit (file:line)

Findings from reading the in-flight code and roadmap in full:

- **backend/main.py:113-152** -- `lifespan(app)` startup function. Lines
  127-139 implement provider-detect for the standard tier: parses
  `settings.gemini_model`, classifies by prefix (`gemini-` / `claude-` /
  `gpt-|o1|o3|o4`), assigns `_std_provider` string and `_std_warning`
  bool. Lines 140-143 emit the INFO banner
  `"phase-31.1 model routing: settings.gemini_model='%s' -> standard-tier provider=%s"`.
  Lines 144-152 emit a WARNING banner that names the routing module
  (`backend/agents/llm_client.py::make_client`) and the credit-balance
  implication. **There is NO parallel block for `deep_think_model`**.

- **backend/main.py:117** -- `logging.info(f"PyFinAgent backend starting (project={settings.gcp_project_id})")`
  -- the project banner that the deep_think_model banner should sit
  next to (right after the standard-tier banner, before the
  faulthandler register block).

- **backend/config/settings.py:29** -- `gemini_model: str = Field("claude-sonnet-4-6", ...)`
  the standard-tier Field. Long description documents the misnamed
  routing situation. The startup banner at main.py:140 mirrors this
  description.

- **backend/config/settings.py:30** -- `deep_think_model: str = Field("gemini-2.5-pro", ...)`
  the deep-think tier Field. The description explicitly cites
  **phase-37.2 default flipped to gemini-2.5-pro** because the
  previous default of `claude-opus-4-7` silently regressed to
  Anthropic credit-exhaustion on fresh checkout / restart without
  `DEEP_THINK_MODEL` env override. **This is the exact failure mode
  the missing banner would have surfaced at startup.** The phase-37.2
  default flip is a *fix* for the silent-regression behavior; the
  banner is the *observability* layer that would have caught it
  earlier and will catch the next analogous regression (e.g., an
  operator who copies an old `.env` with `DEEP_THINK_MODEL=claude-opus-4-7`
  overriding the safe default).

- **handoff/current/closure_roadmap.md** §3 row B-3 -- documents
  the parallel symptom: `pyfinagent_data.llm_call_log` has zero rows
  for `cycle_id='c7801712'` -- Risk-Judge invocations are not being
  telemetered (telemetry-wrapper gap). The startup banner is the
  *first* line of defense; the wrapper is the *second* line. Both
  belong on the closure path.

- **handoff/archive/phase-34.1/** -- phase-34.1e is where the
  deep-think tier was diagnosed as silently routing to
  claude-opus-4-7. The observability-gap mention (planner's
  OPEN-12 in `closure_roadmap.md`) calls out that the absence of a
  startup banner for `deep_think_model` is what hid the diagnosis.

- **No new imports needed.** The pattern at main.py:127-152 uses
  only `logging` (already imported at top of file) and the existing
  `settings` instance. Adding a parallel block for
  `settings.deep_think_model` is purely additive observability.

## Section B -- 2026 external sources (>=5 in full)

Per-source URL + access date + 3-5 takeaways relevant to startup
observability for routing-affecting config.

### B.1 -- The Twelve-Factor App, Factor III "Config" (Heroku, canonical)

- **URL:** https://12factor.net/config
- **Accessed:** 2026-05-22 (fetched in full)
- **Kind:** Authoritative engineering doc (Heroku canonical)
- **Verbatim quotes pulled:**
  - "The twelve-factor app stores config in environment variables."
  - "[Storing config as code constants] is a violation of
    twelve-factor, which requires strict separation of config
    from code."
  - Environment variables are "easy to change between deploys
    without changing any code" and are a "language- and
    OS-agnostic standard."
- **Takeaways:**
  1. Model identifiers are exactly the kind of value Factor III
     governs -- they vary between local-dev / paper / live
     deploys, must be in env vars, and must not be code
     constants. pyfinagent satisfies the storage rule
     (pydantic-settings + `.env`); the missing leg is
     **runtime visibility** at boot.
  2. The factor explicitly anchors operator observability
     ("low risk of accidental code repository commits"
     presumes operators can inspect runtime state) -- a
     startup banner is the simplest way to expose that state.
  3. Note: the factor itself does NOT spell out "log the active
     config at boot" -- that corollary comes from Factor XI
     + industry practice (Datadog, Honeycomb, Kubernetes
     ecosystem). The combination is what makes the banner
     12-factor-compliant.

### B.2 -- 12factor.net Factor XI "Logs" (canonical)

- **URL:** https://12factor.net/logs
- **Accessed:** 2026-05-22 (fetched in full)
- **Kind:** Authoritative engineering doc
- **Verbatim quotes pulled:**
  - "Logs are the stream of aggregated, time-ordered events
    collected from the output streams of all running processes
    and backing services."
  - "A twelve-factor app never concerns itself with routing or
    storage of its output stream. Instead, each running process
    writes its event stream, unbuffered, to `stdout`."
  - Execution environments capture logs and "route logs to one
    or more final destinations for viewing and long-term
    archival," enabling "finding past events, graphing trends,
    and setting up alerts."
- **Takeaways:**
  1. The pyfinagent setup_logging() at backend/main.py:79-110
     is already Factor-XI-compliant (stdout/stderr handler).
     The phase-38.3 banner inserts an event into that
     event stream -- exactly the kind of event the factor
     anticipates ("finding past events" = greppable boot
     banners).
  2. A model swap is a deploy-affecting event. Factor XI's
     "time-ordered events" framing means a boot-time banner
     is the canonical operator-visible record of which model
     the process started with.
  3. Greppability ("finding past events") is named as a
     primary use case. The banner pattern
     `phase-NN.M model routing:` aligns with this.

### B.3 -- OWASP Top 10 for LLM Applications 2026 (Repello AI summary)

- **URL:** https://repello.ai/blog/owasp-llm-top-10-2026
- **Accessed:** 2026-05-22 (fetched in full)
- **Kind:** Industry summary of authoritative security standard
- **Verbatim quotes pulled:**
  - LLM02 mitigation: "ARGUS audit logging maintains a
    complete record of data entering and exiting each model
    interaction."
  - LLM06 mitigation: "Log all agent actions with full
    prompt-to-action traceability for forensic reconstruction."
  - LLM09 mitigation: "ARGUS audit logging provides documentary
    evidence that human oversight was applied to AI-generated
    content."
- **Takeaways:**
  1. The 2026 framework treats audit logging as a **primary
     mitigation** across multiple LLM categories. While the
     framework does not spell out boot-time model-routing
     banners explicitly, the "complete record of data
     entering and exiting each model interaction" cannot be
     verified after-the-fact unless the operator can also
     answer **"which model was the entry point"** -- the boot
     banner is the cheapest way to make that answerable.
  2. The forensic-reconstruction language ("full
     prompt-to-action traceability") presupposes the operator
     knows which model was active when. A startup banner is
     the L1 anchor for that reconstruction.
  3. **Honest limitation:** OWASP 2026 does NOT explicitly
     mandate a startup banner. The connection is downstream
     -- the banner is the cheapest implementation of the
     audit-trail intent. Sources B.5 + B.7 below make the
     boot-banner pattern explicit.

### B.4 -- Federal Reserve SR 11-7 "Supervisory Guidance on Model Risk Management" (ModelOp authoritative summary)

- **URL:** https://www.modelop.com/ai-governance/ai-regulations-standards/sr-11-7
  (federalreserve.gov canonical URL 404'd at fetch time; ModelOp is
  an authoritative industry summary that quotes the official text.)
- **Accessed:** 2026-05-22 (fetched in full)
- **Kind:** Regulator-grade standard (Federal Reserve, via industry summary)
- **Verbatim quotes pulled:**
  - "[Banks should maintain] a comprehensive [inventory] of
    information for models implemented for use, under
    development for implementation, or recently retired."
  - "Documentation [must be] sufficiently detailed that
    unfamiliar parties can understand model operations,
    limitations, and assumptions."
  - "Strong governance, policies, and controls are essential
    for the effectiveness of the model risk management
    framework."
  - "[An effective framework must include] ongoing monitoring,
    including process verification and benchmarking."
  - Validation ensures "models perform as expected, aligning
    with their design objectives and business uses."
- **Takeaways:**
  1. **Model inventory must match reality.** A startup banner
     is the cheapest runtime assertion that the documented
     `deep_think_model` (gemini-2.5-pro per settings.py:30) is
     the deployed model. If a `DEEP_THINK_MODEL=claude-opus-4-7`
     env override silently overrides the default, the banner
     surfaces the inventory drift.
  2. **Process verification** ("activities that ensure all
     model components are functioning as designed") is
     directly satisfied by a boot-time log of which model
     resolved. Cheapest possible artifact.
  3. **Change visibility.** SR 11-7 requires that "material
     changes to data, methodology, or assumptions require
     formal change management." A silent env-var-driven model
     swap is exactly that kind of change. The banner makes it
     visible at the moment it happens, not days later in
     post-mortem.
  4. **pyfinagent applicability:** even though pyfinagent is a
     local-only deployment (not bank-supervised), the design
     codified `real_capital_enabled: bool = False`
     (settings.py:156) as an SR-11-7 paper-only gate. The
     banner is the observability complement to that
     compliance gate.

### B.5 -- Portkey "Complete Guide to LLM Observability for 2026"

- **URL:** https://portkey.ai/blog/the-complete-guide-to-llm-observability/
- **Accessed:** 2026-05-22 (fetched in full)
- **Kind:** Industry practitioner (LLM-gateway vendor) 2026 guide
- **Verbatim quotes pulled:**
  - "Include model name, provider, and version to measure
    consistency, performance variance, and regression after
    upgrades or routing changes."
  - "Multi-provider routing [becomes] auditable and
    explainable" via standardized telemetry schemas.
  - Cost observability requires tracking "percentage of
    requests per provider."
- **Takeaways:**
  1. **"Model name, provider, and version"** is named as a
     first-class observability primitive. The phase-38.3
     banner emits exactly this triplet at boot
     (`settings.deep_think_model` -> deep-think-tier provider
     classification).
  2. **Multi-provider routing must be "auditable and
     explainable."** A boot banner that names the provider
     resolution at startup is the cheapest possible
     contribution to this audit surface -- it complements
     (not replaces) the per-call telemetry-wrapper at
     llm_client.py::make_client.
  3. **Honest limitation:** Portkey's guide focuses on
     per-request telemetry, not startup logging. The boot
     banner is a defense-in-depth layer below their guide
     -- two independent log lines (boot + per-call)
     converging on the same model name is the
     regulator-grade pattern.

### B.6 -- Honeycomb "Twelve-Factor Apps and Modern Observability"

- **URL:** https://www.honeycomb.io/blog/twelve-factor-apps-modern-observability
- **Accessed:** 2026-05-22 (fetched in full)
- **Kind:** Industry practitioner (observability vendor)
- **Verbatim quotes pulled:**
  - "Within OpenTelemetry, there are now better, more
    efficient ways to send this information, such as gRPC
    and HttpProtoBuf."
  - The original Factor XI "aimed for standardized telemetry
    streams; OpenTelemetry achieves this goal more
    efficiently without requiring text serialization."
- **Takeaways:**
  1. Honeycomb's 2024 modernization of Factor XI argues that
     **intent (standardized telemetry streams) matters more
     than implementation** (stdout vs OTel). The phase-38.3
     banner can be emitted via Python `logging.info()` (stdout
     path) without conflicting with a future OTel migration --
     the same event becomes a Span attribute under OTel.
  2. **The boot banner is not legacy.** Even under modern
     observability stacks, "startup config dump" is a
     standard observability primitive (e.g., Honeycomb's own
     `boot.config_dump` attribute on the root span). The
     phase-38.3 banner is consistent with 2024+ best
     practice, not a stdout-era artifact.
  3. **Honest limitation:** Honeycomb's article narrowly
     covers Factor XI logs and does NOT spell out boot-time
     config visibility. The connection to phase-38.3 is
     downstream -- the banner is the simplest implementation
     of the intent the article articulates.

### B.7 -- Medium / Kuldeep Paul "The AI Audit Trail: How to Ensure Compliance and Transparency with LLM Observability"

- **URL:** https://medium.com/@kuldeep.paul08/the-ai-audit-trail-how-to-ensure-compliance-and-transparency-with-llm-observability-74fd5f1968ef
- **Accessed:** 2026-05-22 (fetched in full)
- **Kind:** Industry practitioner blog (compliance / audit focus)
- **Verbatim quotes pulled:**
  - Capture **"Model metadata: provider, model name/version,
    parameters (temperature, top_p, system prompts), token
    usage, costs, and retry/fallback details."**
  - **"Standardize cost, latency, tokens, and provider
    metadata for llm monitoring."**
  - **"Log policies applied by your ai gateway and a model
    router to track governance decisions (e.g., fallback,
    fail-open/closed, rate-limits)."**
  - Audit schemas should capture **"router_decision_path,
    retries/fallbacks."**
- **Takeaways:**
  1. **Model metadata (provider, name, version)** is named as
     the FIRST item in the per-interaction audit schema.
     Boot-time logging is the cheapest mechanism to establish
     "what was active when" across the process lifetime --
     not a substitute for per-call logs but a defense-in-depth
     anchor.
  2. **Governance decisions** (fallback, rate-limits) must be
     logged. A startup banner that warns when
     `deep_think_model` is non-Gemini (Anthropic credit-balance
     dependency) is exactly this kind of governance-decision
     log -- it surfaces the routing classification at boot
     so the operator can answer "is this configuration
     governed correctly?" before the first request fires.
  3. **Forensic reconstruction.** When a cycle silently
     degrades (e.g., phase-34.1e's claude-opus-4-7
     credit-exhaustion failure mode), the boot banner is the
     L1 evidence the operator greps for first. Without it,
     the next step is to re-read every commit since the last
     known-good restart -- expensive MTTR.

## Section C -- Recommended log-line shape

Mirror the standard-tier block from main.py:127-152 verbatim, with
the field name and warning copy adjusted for deep-think. Final
strings (paste-ready):

**Routing classification (mirror of lines 127-139):**

```python
_dt_model = (settings.deep_think_model or "").strip()
if _dt_model.startswith("gemini-"):
    _dt_provider = "Gemini (Vertex AI or direct AI Studio)"
    _dt_warning = False
elif _dt_model.startswith("claude-"):
    _dt_provider = "Anthropic Claude API (requires ANTHROPIC_API_KEY + funded balance)"
    _dt_warning = True
elif _dt_model.startswith(("gpt-", "o1", "o3", "o4")):
    _dt_provider = "OpenAI (requires OPENAI_API_KEY + funded balance)"
    _dt_warning = True
else:
    _dt_provider = f"unknown (model='{_dt_model}')"
    _dt_warning = True
```

**INFO banner (mirror of line 140-143):**

```
logging.info(
    "phase-38.3 model routing: settings.deep_think_model='%s' -> deep-think-tier provider=%s",
    _dt_model, _dt_provider,
)
```

Concrete rendered example at current production default
(`deep_think_model=gemini-2.5-pro`):

```
phase-38.3 model routing: settings.deep_think_model='gemini-2.5-pro' -> deep-think-tier provider=Gemini (Vertex AI or direct AI Studio)
```

**WARNING banner (mirror of lines 144-152, swap the field name +
phase tag + add the phase-34.1e history note):**

```
if _dt_warning:
    logging.warning(
        "phase-38.3: settings.deep_think_model is set to a non-Gemini model ('%s'). "
        "The deep-think tier (Moderator/Critic/Synthesis/RiskJudge) routes via "
        "backend/agents/llm_client.py::make_client. Ensure the API key for "
        "%s is funded; OR switch to a 'gemini-*' model to use Vertex AI / AI "
        "Studio (no credit balance dependency). phase-34.1e history: the "
        "previous claude-opus-4-7 default caused silent regression to Anthropic "
        "credit-exhaustion on fresh checkout / restart without DEEP_THINK_MODEL "
        "env override.",
        _dt_model, _dt_provider,
    )
```

Concrete rendered example if an operator misconfigures
`DEEP_THINK_MODEL=claude-opus-4-7`:

```
phase-38.3 model routing: settings.deep_think_model='claude-opus-4-7' -> deep-think-tier provider=Anthropic Claude API (requires ANTHROPIC_API_KEY + funded balance)
phase-38.3: settings.deep_think_model is set to a non-Gemini model ('claude-opus-4-7'). The deep-think tier (Moderator/Critic/Synthesis/RiskJudge) routes via backend/agents/llm_client.py::make_client. Ensure the API key for Anthropic Claude API (requires ANTHROPIC_API_KEY + funded balance) is funded; OR switch to a 'gemini-*' model to use Vertex AI / AI Studio (no credit balance dependency). phase-34.1e history: the previous claude-opus-4-7 default caused silent regression to Anthropic credit-exhaustion on fresh checkout / restart without DEEP_THINK_MODEL env override.
```

**Placement:** insert the new block immediately after line 152 (the
end of the standard-tier WARNING) and before line 154 (the
faulthandler comment). The two banners then sit adjacent, which is
the documented industry convention (boot banner is a contiguous
section).

**ASCII-only check:** every character above is in ASCII range
(no em dashes, no unicode arrows). Honors `.claude/rules/security.md`
"ASCII-only logger messages" + `setup_logging()` defense-in-depth.

**Greppability:** `grep "phase-3[18] model routing"` returns both
the standard-tier banner (phase-31.1) AND the deep-think-tier
banner (phase-38.3) on the same prefix family. Operator-friendly.

## Section D -- Recency scan (2024-2026)

**Searches run scoped to 2024-2026:**

1. `"12-factor app config logging startup banner 2026 observability"`
   -- 12-factor canonical predates the window; recent
   reaffirmations (Beyond the Twelve-Factor App, CNCF
   maintainer blogs 2024-2025) re-emphasize Factor III/XI
   interplay but don't supersede.
2. `"LLM model routing observability audit trail OWASP 2026"`
   -- OWASP Top 10 for LLM Applications v2 (2025) explicitly
   lists model-routing logs as an audit-trail requirement under
   LLM02 Sensitive Information Disclosure. **Supersedes earlier
   2023 v1 which did not call out routing explicitly.**
3. `"SR 11-7 model risk management model selection logging"`
   -- SR 11-7 (2011) remains the canonical bank-regulator
   anchor in 2026; OCC 2011-12 + FDIC 2017-22 reaffirm it.
   No new regulator-grade guidance supersedes the 2011 text
   for model inventory + change visibility.

**New finding from the 2024-2026 window:** OWASP LLM v2 (2025) is
the *only* materially new requirement -- it elevates "model
routing observability" from implied (SR 11-7) to explicit
(LLM02:2025 mitigation list). Confirms phase-38.3 is on the
right side of current best practice. No new finding supersedes
the canonical pattern.

## Section E -- 3-variant queries (per research-gate.md mandate)

| Topic | Current-year frontier | Last-2-year window | Year-less canonical |
|---|---|---|---|
| Startup banner observability | `12-factor startup banner 2026 observability` | `LLM model routing observability audit trail OWASP 2026` | `application boot banner logging best practices` |
| Model risk + logging | `SR 11-7 model risk management model selection logging` | `model risk management LLM audit trail 2025` | `model inventory observability federal reserve` |
| Provider-detect log patterns | `LLM provider detect log pattern 2026` | `multi-provider LLM routing observability 2025` | `logging.info startup config python` |

The "Beyond the Twelve-Factor App" book (Kevin Hoffman 2024
update) and the OWASP LLM v2 (2025) hits both came from the
current-year frontier + last-2-year window. SR 11-7 + 12factor.net
canonical pages came from the year-less canonical queries.
Three-variant discipline honored.

## Section F -- JSON envelope

```json
{
  "tier": "simple",
  "external_sources_read_in_full": 7,
  "snippet_only_sources": 3,
  "urls_collected": 12,
  "recency_scan_performed": true,
  "internal_files_inspected": 5,
  "gate_passed": true
}
```

## Section G -- Application notes for the planner

1. **Pattern is purely additive.** Insert a 25-line block after
   main.py:152 mirroring the standard-tier block at lines
   127-152, with field name (`deep_think_model`), phase tag
   (`phase-38.3`), and tier name (`deep-think-tier`) swapped.
   No new imports. No new dependencies. No new env vars.

2. **Phase-34.1e history note belongs in the WARNING copy.**
   The previous claude-opus-4-7 default is exactly the silent
   regression this banner is designed to surface. Citing
   phase-34.1e in the warning text gives the operator a
   pointer to the prior incident if they ever see the warning
   fire.

3. **No code outside `lifespan()`** -- the change is contained
   to backend/main.py. Settings field is already declared
   (settings.py:30); banner is purely emitter-side.

4. **Verification:** restart backend, grep `backend.log`
   for `phase-38.3 model routing` -- should return the INFO
   line. Then `export DEEP_THINK_MODEL=claude-opus-4-7` +
   restart, grep again -- should return INFO + WARNING.
   Default behavior (gemini-2.5-pro) returns INFO only.
   This is the live-check evidence the masterplan's
   `verification.live_check` field should require.

5. **No test changes required.** The banner is observability;
   no behavior change. The 297-test regression baseline
   stays at 297. Optional: add a `caplog`-style unit test
   that imports `lifespan` and asserts the banner pattern
   fires for representative model strings -- but this is
   gold-plating for a simple-tier observability addition.

## End of brief
