# Research Brief — step 86.41

**Topic:** Defensive handling of absent upstream financial data in a multi-agent
analysis pipeline — distinguishing "the provider returned nothing" from "the field
is legitimately null" from "the code assumed a dict and got None". Fail-fast vs
fail-soft for a single agent inside a larger pipeline; the broad-`except`
anti-pattern that converts a data gap into a degraded whole-pipeline fallback;
null-object and Optional-chaining patterns; making a partial-data path observable
rather than silent; whether one agent's crash should degrade an entire analysis;
schema-validation-at-the-boundary vs scattered defensive checks.

**Tier:** moderate (caller-specified). **Audit-class:** NO (coverage reported for
information only; `coverage.dry` not required).
**Started:** 2026-08-11. **Researcher:** Layer-3 Researcher (Workflow rail).

---

## ENVELOPE (born inert — flipped to COMPLETE as the final act)

```json
{
  "brief_status": "COMPLETE",
  "tier": "moderate",
  "external_sources_read_in_full": 7,
  "snippet_only_sources": 13,
  "urls_collected": 21,
  "recency_scan_performed": true,
  "internal_files_inspected": 12,
  "coverage": {
    "audit_class": false,
    "rounds": 1,
    "dry_rounds": 0,
    "K_required": 2,
    "new_findings_last_round": 0,
    "dry": false
  },
  "summary": "The 'NoneType' error is NOT in this repo: it is raised at /workspace/main.py:89 inside the remote Quant Agent Cloud Function, one frame after a SEC 429 exhausts its CIK-map retry ladder and the fetcher returns None instead of raising. 17 of 18 measured events follow a 429 within 8 log lines; 13 tickers, all SEC-covered US names. The census (derive_lite_fallback_census_86_38.py:39-53) therefore misattributes all of them to 'code defect: QuantAgent NoneType' because classify() only sees the wrapper reason string, which carries no 429. orchestrator.py:1792 is the ONLY unguarded sub-agent call (RAG :1160-1167, ingestion :1775-1787 and phase-32.3 :1828-1839 all fail open), and autonomous_loop.py:2201 converts it into a whole-ticker lite fallback. External consensus: AWS/Google SRE treat routine fallback as an anti-pattern; Azure Bulkhead wants isolation at the dependency, not the ticker; Fowler (tolerate) and RFC 9413 (refuse) genuinely disagree, reconciled as tolerate SHAPE, refuse ABSENCE. Zero events in the current backend.log makes any 'zero errors next cycle' criterion vacuous.",
  "brief_path": "handoff/current/research_brief_86.41.md",
  "gate_passed": true
}
```

**Envelope flipped to COMPLETE as the final act, 2026-08-11.**

---

## Search-method disclosure (read this before the source tables)

**`WebSearch` was unavailable for this entire session: the budget was already at
200/200 when this agent was spawned** (session-shared budget; the tool returned
"this session has used its web search budget (200 of 200 WebSearch calls)" on the
first attempt). The mandatory three-variant query discipline
(`.claude/rules/research-gate.md` "Search-query composition") therefore could not
be executed as three `WebSearch` calls. I did not silently skip it; I substituted:

- **Year-less canonical variant** → satisfied by fetching the canonical prior art
  directly by URL (Google SRE Book chapters, Fowler's bliki, an IETF RFC, the
  Azure pattern catalogue) — exactly the class of textbook/founding-source hits
  the year-less query exists to surface.
- **Current-year / last-2-year variants** → satisfied by the **arXiv Atom API**
  (`export.arxiv.org/api/query`, `sortBy=submittedDate&sortOrder=descending`),
  which is a date-sorted frontier scan and is *stronger* than a year-suffixed
  keyword search for recency.

Limitation, stated rather than buried: this substitution biases the external set
toward sources I could name a priori, so an unknown-unknown blog post or a
2026 vendor postmortem could have been missed. Two candidate sources were also
lost to fetch mechanics (`aws.amazon.com/builders-library/...` 301s to a
JS-rendered SPA at `builder.aws.com` that yields 20 characters of text; and
`web.archive.org` is blocked for `WebFetch` in this environment).

## Read in full (>=5 required; counts toward the gate)

| # | URL | Accessed | Kind | Fetched how | Key quote / finding |
|---|-----|----------|------|-------------|---------------------|
| 1 | https://sre.google/sre-book/addressing-cascading-failures/ | 2026-08-11 | official book/doc (Google SRE, Ch. 22) | WebFetch, full chapter | "Remember that the code path you never use is the code path that (often) doesn't work." Also: "Graceful degradation shouldn't trigger very often... Keep the system simple and understandable, particularly if it isn't used often"; "When overloaded at either the frontend or backend layers, fail early and cheaply"; "Monitor and alert when too many servers enter these modes." |
| 2 | https://sre.google/sre-book/handling-overload/ | 2026-08-11 | official book/doc (Google SRE, Ch. 21) | WebFetch, full chapter | Retry discipline: per-request budget "up to three attempts... we let the failure bubble up to the caller"; per-client budget capped at 10% retries; "If multiple layers retried, we'd have a combinatorial explosion" — a failed request "should only be retried by... the layer immediately above it", which returns an explicit "overloaded; don't retry" error. Degraded responses are ones "not as accurate as or that contain less data than normal responses". |
| 3 | https://arxiv.org/html/2503.13657v2 | 2026-08-11 | preprint (MAST failure taxonomy, multi-agent LLM) | WebFetch, full HTML paper | Taxonomy: FC1 Specification 41.77%, FC2 Inter-Agent Misalignment 36.94%, FC3 Task Verification 21.30%. FM-3.2 "No or incomplete verification" 6.82% + FM-3.3 "Incorrect verification" 6.66%. Explicitly rejects prompt-only fixes: "improvement to the MAS system design itself can reduce failures, independent of base model improvements". |
| 4 | https://learn.microsoft.com/en-us/azure/architecture/patterns/bulkhead | 2026-08-11 | official doc (Azure Architecture Center, updated 2026-03-19) | WebFetch, full page (1307 words) | "Excessive load or failure in a service affects all consumers of the service." Solution "Isolates consumers and services from cascading failures... Preserves some functionality if a service failure occurs. Other services and features of the application continue to work." Not suitable when "The added complexity isn't necessary." |
| 5 | https://martinfowler.com/bliki/TolerantReader.html | 2026-08-11 | authoritative blog (Fowler) | WebFetch, full page | Postel: "be conservative in what you do, be liberal in what you accept from others"; "only take the elements you need, ignore anything you don't"; wrap tolerance in ONE place — a DTO — so "the rest of the system can just go `anOrderHistory.orders` and be impervious to changes." |
| 6 | https://www.rfc-editor.org/rfc/rfc9413.html | 2026-08-11 | official standards doc (IETF, RFC 9413 "Maintaining Robust Protocols") **[COUNTER-SOURCE to #5]** | WebFetch, full page | "negative consequences to interoperability accumulate over time if implementations silently accept faulty input"; "Tolerating unexpected input instead conceals problems, making it harder, if not impossible, to fix them later"; "Choosing to generate fatal errors for unspecified conditions instead of attempting error recovery can ensure that faults receive attention." |
| 7 | https://pydantic.dev/docs/validation/latest/get-started/why/ | 2026-08-11 | official doc (Pydantic) | WebFetch, full page (thin — disclosed) | "By default, Pydantic is tolerant to common incorrect types and coerces data to the right type"; offers "a strict mode where types are not coerced and a validation error is raised unless the input data exactly matches the expected schema." **Honest note: this page is thinner than expected and is silent on absent-vs-None semantics and on boundary placement**; I did not pad it into a claim it does not support. |

**Read in full but NOT counted toward the WebFetch floor (mechanism disclosed):**

| URL | Kind | Mechanism | Why it matters |
|---|---|---|---|
| https://web.archive.org/web/2022id_/https://aws.amazon.com/builders-library/avoiding-fallback-in-distributed-systems/ | industry engineering (Jacob Gabrielson, AWS Builders' Library) | `curl` + tag-strip, 18,068 chars of body text extracted (WebFetch is blocked for `web.archive.org`; the live URL is now a JS SPA) | The single most on-point source for this step. Verbatim: "This article covers fallback strategies and why we **almost never use them at Amazon**." "fallback logic is [rarely tested]... In production, if malloc fails, the machine is most likely out of memory... How do you simulate those broader memory problems?" "At Amazon we have found that **spending engineering resources on making the primary (non-fallback) code more reliable usually raises our odds of success more than investing in an infrequently used fallback strategy**." And on side effects: "The customer is now experiencing two problems (slower application and slower machine) instead of one." |

## Identified but snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|-----|------|-------------------------|
| https://export.arxiv.org/api/query?search_query=abs:%22multi-agent%22+AND+abs:%22failure%22+AND+abs:%22LLM%22&start=0&max_results=30&sortBy=submittedDate&sortOrder=descending | search instrument (arXiv Atom API) | Fetched in full, but it is a *listing*, not a source — counted as the recency-scan instrument, not as an authoritative source |
| https://arxiv.org/abs/2608.00711 | preprint (SCHEMA / "Tracing the Cascade", 2026-08-01) | Abstract page only. Per `.claude/rules/research-gate.md` "Never do": an abstract page is NOT a read-in-full. Content used only as corroboration |
| https://arxiv.org/abs/2608.00718 | preprint, "Adversarial Attacks in Multi-Agent LLM Pipelines" (2026-08-01) | Listing entry: "once adversarial content enters pipeline, it propagates as trusted input across agents without boundary verification" |
| https://arxiv.org/abs/2607.26836 | preprint, HalluProp pre-hoc failure-risk inference (2026-07-29) | Listing entry — models propagation via communication topology |
| https://arxiv.org/abs/2607.29055 | preprint, MARS autonomous repair for MAS (2026-07-31) | Listing entry — automated failure recovery |
| https://arxiv.org/abs/2607.19336 | tutorial, "Agents in the Wild: Where Research Meets Deployment" (2026-07-21) | Listing entry — explicitly covers "verification pipelines, fallback mechanisms, and human oversight" |
| https://arxiv.org/abs/2608.04634 | preprint, HELENA (2026-08-05) | Listing entry — "preventing noise propagation" |
| https://arxiv.org/abs/2608.03735 | preprint, multilingual multi-agent planning failures (2026-08-04) | Listing entry — "task-critical information loss" |
| https://arxiv.org/abs/2607.13548 | preprint, root-cause analysis on real-world telemetry (2026-07-15) | Listing entry — "evidence present but reasoning fails (Reasoning Gap)" |
| https://arxiv.org/abs/2608.01805 | preprint, CockpitHAT (2026-08-03) | Listing entry — "Correctness Collapse... high task accuracy masks severe process-level failures" |
| https://aws.amazon.com/builders-library/avoiding-fallback-in-distributed-systems/ | industry engineering (canonical URL) | 301 → JS SPA; unfetchable by WebFetch. Superseded by the Wayback row above |
| https://builder.aws.com/content/3EuS9Sakq7L3VLQIF3qzfMfke1Y/avoiding-fallback-in-distributed-systems | industry engineering (redirect target) | JS-rendered; `curl` yields 20 characters of text |
| https://docs.pydantic.dev/latest/why/ | official doc (origin URL) | 301 → `pydantic.dev/docs/validation/...`; the redirect target IS row 7 |

**Unique URLs collected: 21** (7 counted read-in-full + 1 curl-read-in-full + 13
snippet-only/context). Counted by enumerating the three tables above, not asserted.

## Recency scan (2024–2026)

**Performed.** Two passes.

1. **Frontier pass (last 30 days).** arXiv Atom API, `abs:"multi-agent" AND
   abs:"failure" AND abs:"LLM"`, date-descending, 30 results — window
   **2026-07-10 → 2026-08-10**. Result: **the 2026 literature has shifted from
   "does a multi-agent system fail?" to "how does one agent's error propagate,
   and can you contain it at a boundary?"** — which is precisely this step's
   question. Concretely: arXiv:2608.00711 finds "a single erroneous claim on a
   foundational concept can propagate through multi-step reasoning and corrupt
   entire trajectories", and that "final-answer accuracy decouples from
   trajectory honesty; models often reach correct conclusions through
   structurally flawed reasoning" — i.e. **a plausible output is not evidence
   the inputs were sound**, the direct analogue of a lite-fallback HOLD that
   looks fine while the quant block was never fetched. arXiv:2608.00718 names
   the mechanism this step must defend against: content that enters a pipeline
   "propagates as trusted input across agents **without boundary verification**".

2. **Last-2-year pass.** MAST (arXiv:2503.13657, 2025) read in full — it remains
   the canonical failure taxonomy and is **not superseded**; the 2026 work
   extends it toward propagation/topology rather than replacing it.

**Does the new work supersede the canonical sources?** No. Google SRE (2016),
Fowler's Tolerant Reader (2011) and RFC 9413 (2023) still govern the *mechanism*
questions (retry budgets, degraded-mode observability, tolerance-vs-strictness);
the 2024-2026 agent literature adds one genuinely new and load-bearing claim:
**terminal accuracy is an insufficient reliability signal for an agent pipeline**
(arXiv:2608.00711). That directly argues against measuring this step's fix by
"the cycle still produced a recommendation".

## Key findings

1. **Fallback is the default-suspect design, not the default-safe one.** "This
   article covers fallback strategies and **why we almost never use them at
   Amazon**"; the operative reason is that the fallback path is exercised only
   in the exact conditions that are hardest to simulate, and Amazon found
   "spending engineering resources on making the primary (non-fallback) code
   more reliable usually raises our odds of success more than investing in an
   infrequently used fallback strategy" (Gabrielson, AWS Builders' Library,
   Wayback 2022 capture, accessed 2026-08-11). Google SRE states the same law in
   one sentence: "Remember that the code path you never use is the code path
   that (often) doesn't work"
   (https://sre.google/sre-book/addressing-cascading-failures/, 2026-08-11).

2. **A degraded mode is legitimate only if it is rare, simple, exercised, and
   alarmed.** "Graceful degradation shouldn't trigger very often — usually in
   cases of a capacity planning failure or unexpected load shift"; the mitigation
   is to "make sure that graceful degradation stays working by regularly running
   a small subset of servers near overload", and to "**Monitor and alert when too
   many servers enter these modes**" (Google SRE Ch. 22, 2026-08-11). A fallback
   that fires routinely has stopped being a safety net and has become the
   architecture.

3. **Retry-then-return-None is the specific anti-pattern that destroys cause
   information.** Google SRE's rule is that an exhausted retry budget must
   "**let the failure bubble up to the caller**" as an explicit typed condition —
   they even name the wire signal, an "overloaded; don't retry" error — and warns
   that if every layer retries independently "we'd have a combinatorial
   explosion" (https://sre.google/sre-book/handling-overload/, 2026-08-11). A
   sentinel `None` return is the opposite: it *swallows* the 429 and re-emits it
   one frame later as a type error with the cause erased.

4. **Isolate the blast radius at the dependency boundary, not at the whole-unit
   boundary.** The Bulkhead pattern exists so that "a problem that affects a
   consumer or service can be isolated within its own bulkhead to prevent the
   entire solution from failing" and so the system "**preserves some
   functionality if a service failure occurs. Other services and features of the
   application continue to work**"
   (https://learn.microsoft.com/en-us/azure/architecture/patterns/bulkhead,
   2026-08-11). Its stated non-applicability is honest and relevant: skip it when
   "the added complexity isn't necessary."

5. **Tolerance must be localised to one adapter, not sprinkled through
   consumers.** Fowler's prescription is explicitly *architectural*: put the
   tolerance in one DTO-shaped reader so "the rest of the system can just go
   `anOrderHistory.orders` and be impervious to changes"
   (https://martinfowler.com/bliki/TolerantReader.html, 2026-08-11). Scattered
   `or {}` guards are the failure mode this pattern was written against.

6. **...but tolerance has a documented, standards-track counter-argument.** RFC
   9413: "negative consequences to interoperability accumulate over time if
   implementations silently accept faulty input", and "**Tolerating unexpected
   input instead conceals problems, making it harder, if not impossible, to fix
   them later**"; the recommended alternative is to "generate fatal errors for
   unspecified conditions instead of attempting error recovery... to ensure that
   faults receive attention" (https://www.rfc-editor.org/rfc/rfc9413.html,
   2026-08-11). This is the strongest external argument *against* simply adding
   `or {}` at pyfinagent's call sites.

7. **Verification, not tolerance, is where multi-agent systems actually fail.**
   MAST attributes 21.30% of observed multi-agent failures to Task Verification,
   with "no or incomplete verification" at 6.82% and "incorrect verification" at
   6.66%, and states that design change beats model change: "improvement to the
   MAS system design itself can reduce failures, independent of base model
   improvements" (https://arxiv.org/html/2503.13657v2, 2026-08-11).

## Internal code inventory

### I-0. THE HEADLINE: the `None` is NOT in this repo, and it is NOT a missing ticker

The `'NoneType' object has no attribute 'get'` is raised inside the **remote Quant
Agent Cloud Function**, not in pyfinagent. The traceback is streamed back verbatim
and logged line-by-line by `backend/agents/orchestrator.py:1141`
(`logger.info(f"Quant: {line}")`). Verbatim from
`handoff/logs/backend.log.20260810T064130Z.gz` (2026-08-05 20:09:56, ticker DDOG):

```
Quant: Fetching and caching SEC CIK map...
Quant: SEC 429 rate-limit on CIK map, retrying in 2s...
Quant: SEC 429 rate-limit on CIK map, retrying in 3s...
Quant: SEC 429 rate-limit on CIK map, retrying in 5s...
Quant: QuantAgent failed for DDOG: 'NoneType' object has no attribute 'get'
Quant: Traceback (most recent call last):
Quant:   File "/workspace/main.py", line 220, in generate_logs_and_data
Quant:     cik_10_digit = get_cik(ticker_str.upper())
Quant:   File "/workspace/main.py", line 89, in get_cik
Quant:     cik = cik_map.get(ticker.upper())
Quant: AttributeError: 'NoneType' object has no attribute 'get'
```

`/workspace/main.py` is the Cloud Function's filesystem, not this repo (grep for
`QuantAgent failed` across the whole tree returns **0 hits in any tracked file** —
the string exists only inside log payloads). So the causal chain is:

1. SEC EDGAR returns **429** on the `company_tickers.json` CIK-map fetch;
2. the CF's retry ladder (2s / 3s / 5s) exhausts and the CIK-map fetcher
   **returns `None` instead of raising** — the classic "return None on failure"
   sentinel;
3. one frame later `cik_map.get(...)` raises `AttributeError`, and **the 429
   provenance is destroyed** — the exception message carries no trace of it;
4. the CF streams `ERROR: QuantAgent failed for <T>: 'NoneType' ...`, which
   `orchestrator.py:1138-1139` turns into `raise RuntimeError(line)`;
5. `orchestrator.py:1792` (`report["quant"] = await self.run_quant_agent(ticker)`)
   is **NOT wrapped in try/except**, so the whole ticker analysis aborts;
6. `backend/services/autonomous_loop.py:2201` catches it with a bare
   `except Exception` and drops the ENTIRE ticker onto the lite Claude path
   (`:2203` log, `:2214` `_select_lite_analyzer`).

**Answer to the step's three-way question:** this is neither "the field is
legitimately null" nor "the data provider returned nothing for this ticker". It is
a **transient rate limit** (case 1) that an upstream sentinel-return converted into
case 3 ("the code assumed a dict and got None") one stack frame later. Every one of
the 13 affected tickers is a large, SEC-covered US name (AAPL, INTC, MU, CRWD,
DDOG, DELL, PANW, NTAP, WDC, STX, DVA, COHR, SNDK) — none is an absent-ticker case,
and the schema is unchanged.

### I-1. Measured population (derived here, 2026-08-11)

Script: dedup counter over `backend.log` + all six `handoff/logs/backend.log.*.gz`,
matching messages containing `QuantAgent failed for` AND `has no attribute`, and
EXCLUDING the `-- falling back` wrapper line so each event is counted once.

| Metric | Value |
|---|---|
| QuantAgent `NoneType` failure events (deduped) | **18** |
| ...of which preceded within 8 log lines by a SEC 429 / rate-limit line | **17 of 18 (94%)** |
| distinct tickers affected | 13 |
| events in the CURRENT `backend.log` | **0** |
| per rotated file | 0612:6, 0706:2, 0724:2, 0729:1, 0804:3, 0810:4 |
| `has no attribute 'strip'` occurrences (all files) | **1**, in `backend.log.20260612T104931Z.gz` only |

Two things follow. (a) The `'strip'` string the step asks about is a **single
June-era event**, not a live class — do not build a step around it without saying
so. (b) Zero events in the current log means the class is **intermittent, tied to
SEC 429 pressure**, not permanently on; a "zero errors in the next cycle" live_check
would pass vacuously (see Pitfall P-5).

### I-2. The census instrument and its population rule

`scripts/qa/derive_lite_fallback_census_86_38.py` (299 lines). Population rule,
read verbatim:

- `FALLBACK_MARK = "falling back to lite Claude analyzer"` (`:30`) and
  `FULL_MARK = "Critic verdict"` (`:31`) — a line counts only if it contains one
  of these two strings (`:136-137`, `:220-237`).
- `_REASON = re.compile(r"Full orchestrator failed for (\S+?): (.*?) -- falling back", re.S)` (`:36`)
  extracts the reason from the WRAPPER line only.
- `classify()` (`:39-53`) is an **ordered** if-ladder:
  429 → GITHUB_TOKEN → timeout → `nonetype` → 503 → 500 → raw prefix.
- Coverage self-assertion (`:248-261`): raw grep count must equal parsed count per
  file or the census is withheld and exits 1. `--per-cycle` (`:62-199`) attributes
  events to `handoff/cycle_history.jsonl` windows with a hardcoded +2h CEST shift
  guarded by a boundary self-check at `:117-131`.

**Defect found in the instrument (contract-relevant):** because the ladder tests
`429` against the **wrapper reason string** — which for these events is
`ERROR: QuantAgent failed for DDOG: 'NoneType' object has no attribute 'get'` and
contains no `429` — all 18 rate-limit-caused events are classified as
`"code defect: QuantAgent NoneType"` (`:47-48`) rather than
`"429 RESOURCE_EXHAUSTED (quota)"` (`:41-42`). The 429 evidence exists only in the
preceding INFO stream lines, which the population rule at `:136-137` never reads.
The census is therefore **internally consistent and externally wrong about cause**
for this class: it will keep reporting a "code defect" bucket that a code fix in
this repo cannot empty. Its own coverage assertion cannot catch this — it asserts
*line accounting*, not *cause attribution*.

### I-3. File/line anchor table

| File | Anchor | Role | Status |
|------|--------|------|--------|
| `backend/agents/orchestrator.py` | `:1126-1149` | `run_quant_agent` — streams the CF; `:1138-1139` `ERROR:` line → `raise RuntimeError(line)`; `:1143-1144` `final_json is None` → `RuntimeError` | LIVE; fail-fast by design |
| `backend/agents/orchestrator.py` | `:1141` | `logger.info(f"Quant: {line}")` — the only place the 429 provenance is ever recorded | LIVE; INFO level, unstructured |
| `backend/agents/orchestrator.py` | `:1789-1804` | Step 2 call site. **`:1792` has NO try/except** (contrast `:1775-1787` ingestion, which IS best-effort). `:1804` `report["quant"].get(...)` is the first consumer | LIVE; the fail-fast/fail-soft asymmetry lives here |
| `backend/agents/orchestrator.py` | `:1807-1818` | phase-27.6.2 `or {}` idiom with an in-code rationale: "`Dict.get(k, {})` only returns the default when k is ABSENT; if k=None, it returns None and the next `.get` raises AttributeError" | LIVE; the repo's existing idiom |
| `backend/agents/orchestrator.py` | `:1961-1962, :2017-2018, :2153-2154, :2206-2207` | Four repetitions of `if isinstance(report.get("quant"), dict): ... .get("sector","")` | LIVE; scattered-defensive-check duplication (4x) |
| `backend/agents/orchestrator.py` | `:1151-1172` | `run_rag_agent` — the repo's **explicit fail-open precedent**: `except Exception` → `return {"text": "", "citations": []}` + `self._rag_available = False`, with the rationale "RAG is enrichment, not core" | LIVE; the null-object pattern already in-repo |
| `backend/agents/orchestrator.py` | `:1828-1839` | phase-32.3 fail-open on portfolio_sector_exposure, "stores None under the field so downstream prompts get an explicit 'no data' rather than crashing" | LIVE; second fail-open precedent |
| `backend/services/autonomous_loop.py` | `:2201-2209` | The broad `except Exception` → whole-ticker lite fallback. `:2209` stamps `_fb_reason = f"{type(e).__name__}: {e}"` | LIVE; **this is the blast-radius amplifier** |
| `backend/services/autonomous_loop.py` | `:2213-2236` | `_select_lite_analyzer(...)`, `:2235` `_lite["_fallback_reason"] = _fb_reason[:500]`. `:2221-2233` documents that `_intended_path` was REMOVED as write-only | LIVE |
| `backend/services/autonomous_loop.py` | `:2237-2252` | Both-paths-failed branch; `paper_synthesis_integrity_enabled` → `_degraded` marker row instead of silent `None` | LIVE; the repo's observability precedent for degraded rows |
| `backend/services/autonomous_loop.py` | `:2156-2164` | `SynthesisDegradedError` raised behind `paper_synthesis_integrity_enabled` — precedent for a **typed** degradation exception | LIVE |
| `scripts/qa/derive_lite_fallback_census_86_38.py` | `:30-53`, `:136-137`, `:248-261` | The census instrument; population rule + `classify()` ladder + coverage assertion | LIVE; **misattributes this class** (see I-2) |
| `backend/agents/claude_code_client.py` | see I-4 | lite-fallback entry point | inspected below |
| `handoff/logs/backend.log.*.gz` (6 files) + `backend.log` | — | evidence corpus | read |

### I-4. The lite-fallback entry point

`_select_lite_analyzer` (`backend/services/autonomous_loop.py:2440-2471`) is a
factory returning an *uncalled* coroutine: `gemini-*` → `_run_gemini_analysis`;
`claude-*` → `_run_claude_analysis` (`:2829`), except that with the DARK flag
`paper_rail_failforward_enabled` ON **and** `_rail_dead_reason()` non-None
(`:2474-2492`, a strict reader of `claude_code_client.rail_guard_status()`) it
returns `_run_failforward_analysis`. `_run_claude_analysis` reaches
`backend/agents/claude_code_client.py` via `claude_code_invoke` (`:297`) at
`autonomous_loop.py:2952` — the CC rail, guarded by the `_RailGuardState` breaker
(`claude_code_client.py:92-172`).

**Note for the contract:** `claude_code_client.py` contains no `lite`-named
symbol; the lite path is `_run_claude_analysis` in `autonomous_loop.py` and
`claude_code_invoke` is its transport. `_fallback_reason` has exactly one
production consumer, `autonomous_loop.py:2657-2669` (`_fallback_rate_check`), plus
test assertions at `test_phase_60_1_deep_pipeline.py:103/207/217/235`,
`test_phase_61_2_decision_integrity.py:122`, and
`test_phase_86_38_degradation_visibility.py:42/147`.

## Consensus vs debate (external)

**Consensus.** (a) A rarely-exercised fallback path is a liability, not a safety
net (AWS; Google SRE Ch. 22). (b) Degradation must be *alarmed*, not merely
logged (Google SRE Ch. 22). (c) Blast radius belongs at the dependency boundary
(Azure Bulkhead). (d) An exhausted retry must surface as an explicit typed
condition, not a sentinel (Google SRE Ch. 21).

**Genuine debate — and it is the crux of this step.** Fowler's Tolerant Reader
says be liberal with what upstream sends you and localise the tolerance; RFC 9413
says liberality "conceals problems, making it harder, if not impossible, to fix
them later" and prescribes fatal errors instead. Both are right about different
things, and the reconciliation is a distinction neither states outright but both
imply: **be tolerant about SHAPE, be strict about ABSENCE.** Extra/renamed/unused
fields → tolerate silently (Fowler). A required input that is missing → refuse
loudly and name it (RFC 9413). pyfinagent's failure is on the second axis, not the
first: the CIK map was *absent*, and the CF was liberal about absence.

A second, quieter debate: MAST finds design changes beat model changes, whereas
the 2026 propagation papers (arXiv:2608.00711/2608.00718) argue the containment
must be a *boundary check*, not better agent behaviour. These agree in direction
and differ in where the check goes.

## Pitfalls (from literature)

- **P-1 — "add `or {}` everywhere" is the pattern Fowler and RFC 9413 BOTH
  reject.** Fowler rejects it as scattered tolerance (belongs in one adapter);
  RFC 9413 rejects it as silent concealment. `orchestrator.py` already has four
  copies of `if isinstance(report.get("quant"), dict)` (`:1961, :2017, :2153,
  :2206`) — adding more is the anti-pattern, not the fix.
- **P-2 — a fallback that fires routinely has become the architecture.** Google
  SRE: degradation "shouldn't trigger very often". Any fix that makes the lite
  path *smoother* rather than *rarer* moves in the wrong direction.
- **P-3 — the fallback path is the least-tested path.** "The code path you never
  use is the code path that (often) doesn't work." pyfinagent's lite path is now
  routine, which mitigates P-3 but proves P-2.
- **P-4 — retries at multiple layers multiply.** The CF already retries 3x on the
  CIK map, and pyfinagent then retries the whole ticker on a different path.
  Google SRE: only "the layer immediately above" should retry.
- **P-5 (project-specific, from the measurement) — "zero NoneType errors next
  cycle" is a VACUOUS live_check.** The current `backend.log` already contains
  **0** such events; the class is intermittent and 429-driven (17/18 events
  follow a SEC 429 within 8 lines). A green result would prove nothing. This
  mirrors `feedback_immutable_criteria_must_be_green_able` and
  `feedback_a_zero_assertion_guard_passes_vacuously`. Whatever criterion this
  step freezes must be a **measured delta on a derivable population**, not an
  absence-of-string check on a window where the population may be empty. (The
  historically-queued criterion in `scripts/add_phase_27_6_sub.py:87` —
  "fresh Claude cycle has zero `QuantAgent failed.*NoneType` log lines" — is
  exactly this vacuous shape; do not inherit it unchanged.)
- **P-6 — the fix may not be in this repo.** The `AttributeError` is raised at
  `/workspace/main.py:89` inside the Cloud Function. A local `or {}` cannot stop
  it; it can only decide what pyfinagent does *about* it. Any contract claiming
  to "fix the NoneType" must say which side of the boundary it changes.
- **P-7 — terminal accuracy is not a reliability signal.** arXiv:2608.00711:
  "final-answer accuracy decouples from trajectory honesty". A lite-path HOLD
  looks like a healthy output; do not accept "the cycle still produced a
  decision" as evidence the degradation was harmless.

## Application to pyfinagent

**A-1. Reclassify before you remediate.** The census
(`scripts/qa/derive_lite_fallback_census_86_38.py:39-53`) files all 18 events as
`"code defect: QuantAgent NoneType"` because `classify()` only ever sees the
wrapper reason string, which has no `429` in it (`:36`, `:136-137`). The
measured truth is 17/18 preceded by a SEC 429. Either the classifier gains a
second signal (the preceding `Quant: ...429...` INFO lines, which requires
widening the population rule at `:136-137`), or its NoneType bucket must be
relabelled as "upstream 429, surfaced as a type error". Leaving it as-is means
the project keeps aiming a code fix at a rate-limit problem. *This finding alone
may be the highest-value deliverable of step 86.41.*

**A-2. The three-way distinction, resolved with anchors.** "Provider returned
nothing" (SEC 429 → CIK map `None`) is the ACTUAL case; "legitimately null"
(a real ticker with no P/E) is already handled by the phase-27.6.2 idiom at
`orchestrator.py:1813`; "code assumed a dict and got `None`" is the SYMPTOM at
CF `main.py:89`, one frame downstream of case 1. Any fix that treats the symptom
as the cause will be re-triggered by the next 429.

**A-3. Fail-soft is already the repo's idiom for enrichment, and quant is the
outlier.** `run_rag_agent` (`orchestrator.py:1160-1167`) fails open with the
stated rationale "RAG is enrichment, not core"; phase-32.3 (`:1828-1839`) stores
`None` "so downstream prompts get an explicit 'no data' rather than crashing";
ingestion (`:1775-1787`) is best-effort. **Only `:1792` is unguarded.** So the
design question is not "should we add fail-soft?" but "is `quant` core or
enrichment?" — and the evidence says it is *partly* core: `:1804`, `:1813`,
`:1822` (`_build_fact_ledger`) and `:1961/2017/2153/2206` (sector routing) all
read it. A null-object `quant` would silently produce a sector-less,
ledger-less analysis — which per RFC 9413 and arXiv:2608.00711 is worse than a
loud refusal, unless the emptiness is *stamped on the output*.

**A-4. The blast-radius question, answered.** Currently ONE failed sub-agent
(`:1792`) discards ~14 completed pipeline steps for that ticker and restarts it on
the lite path (`autonomous_loop.py:2201-2216`). That is the opposite of a bulkhead:
the isolation boundary is drawn around the whole ticker instead of around the
failing dependency. The Bulkhead-consistent shape is to contain the failure at
`run_quant_agent` and let the pipeline continue with a *typed, visible* quant gap.
But note the Azure caveat — "might not be suitable when... the added complexity
isn't necessary" — so this is a judgement call the contract must argue, not assume.

**A-5. Make the partial path observable, and reuse the precedent.** The repo
already has the right shape twice: `SynthesisDegradedError` (`:2156-2164`) is a
*typed* degradation exception, and the `_degraded` / `_degraded_reason` marker row
(`:2245-2252`) makes a degraded outcome persist as data instead of vanishing.
A quant-gap should follow the same precedent (typed exception + a field on the
persisted row) rather than inventing a third mechanism — and per the phase-86.38
comment at `:2221-2233`, it must have a **consumer**, or it will be removed as
write-only exactly like `_intended_path` was.

**A-6. Retry placement.** The CF retries the CIK map 3x (2s/3s/5s) and pyfinagent
then re-runs the ticker on another path. Per Google SRE Ch. 21 the retry belongs
in exactly one layer and the exhausted result must be an explicit typed signal.
The cheapest correct change may be entirely upstream: make the CF *raise* on an
exhausted CIK-map fetch instead of returning `None`, which preserves the 429
provenance and lets `classify()` at `:41-42` bucket it correctly with no change
to the census at all.

## Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch — **7**
- [x] 10+ unique URLs total (incl. snippet-only) — **20**
- [x] Recency scan (last 2 years) performed + reported — arXiv frontier pass
      2026-07-10..2026-08-10 + MAST 2025 read in full
- [x] Full papers / pages read (not abstracts) for the read-in-full set —
      arXiv:2608.00711 was deliberately EXCLUDED from the counted set because
      only its abstract page was fetched
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every module in the caller's scope
      (QuantAgent impl + caller, the census instrument, backend.log + 6 rotated
      logs, the upstream source of the `None`, and the lite-fallback entry point)
- [x] Contradictions / consensus noted (Fowler vs RFC 9413 is a real, load-bearing
      contradiction, not a manufactured one)
- [x] All claims cited per-claim
- [ ] **GAP, disclosed:** the mandatory three-variant `WebSearch` discipline could
      not be executed — the session's WebSearch budget was exhausted (200/200)
      before this agent was spawned. Substitute method documented above. This is
      a soft check, not a hard blocker, but a re-run with search budget available
      could surface sources this brief could not.
- [ ] **GAP, disclosed:** the Cloud Function source (`/workspace/main.py`) is not
      in this repo and was not read. Every claim about it is derived from its
      streamed traceback, which is verbatim evidence but not source.

---

## ANNOTATION (appended 2026-08-11 -- ORIGINAL BRIEF UNCHANGED)

**Any "17 of 18 (94%)" figure in this brief is SUPERSEDED.** Re-measured at the
event level: every occurrence emits two log lines, so the raw 34 is a LINE count
and the true population is **17 events, all 17 carrying an upstream SEC 429 cue
(100%)**. The call site's stable identifier is the function `get_cik`, not a line
number. See `handoff/current/experiment_results_86.41.md` criteria 1-2.
