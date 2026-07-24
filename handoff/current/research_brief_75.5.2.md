# Research Brief — Step 75.5.2

**Step:** 75.5.2 — Route remaining hardcoded gemini-2.5 BEHAVIOURAL pins through
`GEMINI_WORKHORSE` / `GEMINI_DEEP_THINK` in `backend/config/model_tiers.py`.
**Tier:** moderate | **audit_class:** TRUE (loop-until-dry, K=2) | **Executor:** sonnet-4.6/high
**Boundary:** literals → constants ONLY. NO tier pin VALUE changed.
**Deadline:** gemini-2.5 family shuts down 2026-10-16; the 2.0-flash retirement under a
hardcoded pin cost this project 9 silent days.

_Status: internal census COMPLETE + coverage dry (rounds 4 & 5). External research below._

---

## 1. Classification rule (VERBATIM — the discriminator, checked into the test)

> A `gemini-2.5` string literal is a **BEHAVIOURAL PIN** iff it SELECTS the model for a
> runtime call — i.e. the literal flows into a `model=` / `model_name=` /
> `evaluator_model=` parameter of an LLM call or client constructor, OR it is the value a
> model-selection variable / module-constant / `or`-fallback resolves to and is then
> passed to such a call.
>
> It is **NON-BEHAVIOURAL** if it is any of: (a) a **pricing / capability lookup-table
> KEY** (`MODEL_PRICING`, `MODEL_CONTEXT_WINDOWS`, away-ops pricing dict, settings_api
> `_VALID_MODELS` whitelist, pricing dropdown) — the string is a dict KEY/DATA, not a
> selector; (b) a **docstring / comment** describing behaviour; (c) a **roster / display
> metadata** value (`_inventory.json` `"model"` fields, agent-map display); (d) a
> **historical / sample record** (compliance audit trace rows, masterplan migration-text
> payloads); (e) a **sample env template** (`.env.example`); (f) a **test fixture**
> (`backend/tests/**`).
>
> **A separate class — "family guard" — is behavioural but NOT a pin:** a
> `startswith("gemini-2.5")` / family-membership TEST branches on the model name (it does
> not select a model). `backend/agents/llm_client.py:985` is the sole instance. Handled
> in the C1 scan (literal removal via a prefix constant), NOT counted among the routed
> "resolved-value" pins for criteria 2/3.

**75.5 precedent (criterion 1 says the scan is "read strictly, not reinterpreted"):**
75.5's scan (`backend/tests/test_phase_75_llm_rail.py:396-399`) is a **per-file raw
substring check** — `assert "gemini-2.5" not in text` over an explicit `CRITERION_4_FILES`
list — with **docstrings and comments included**. It did NOT try to classify
behavioural-vs-comment within a file; it removed the substring entirely from every
in-scope file. "docstring prose included" means: if a file is in-scope, you may NOT leave
a `gemini-2.5` in its docstring/comment and argue the scan should skip it. (Confirmed: the
5 files 75.5 cleaned — settings.py, evaluator_agent.py, rag_agent_runtime.py,
skill_modification_review.py, backend/autonomous_loop.py — now have ZERO `gemini-2.5`,
comments included. rag_agent_runtime.py:55 was reworded to `2.5-pro, 2.5-flash` — the
`gemini-` prefix stripped from the comment so it no longer matches the substring.)

Note the substring is exactly `"gemini-2.5"` (hyphen). It does NOT match `2.5-flash`
(bare), `models/gemini-embedding-2`, or `gemini_2_5` (underscore). The census confirms
**no behavioural underscore/dotted/bare forms exist outside tests** — the hyphen substring
is a sufficient behavioural discriminator here.

---

## 2. FULL DERIVED CENSUS (tree-wide, all patterns: `gemini[-_.]?2[._]5`, `2.5-flash|pro`, `models/gemini`, model= kwargs)

### 2a. BEHAVIOURAL PINS — the fix targets (8 named + 1 discovered runtime-script)

| # | file:line (RE-DERIVED) | literal / shape | verdict | target constant | pre-change RESOLVED value |
|---|---|---|---|---|---|
| 1 | `backend/meta_evolution/directive_review.py:159` | `model="gemini-2.5-flash"` arg to `g_client.models.generate_content()` | PIN (Vertex call) | `GEMINI_WORKHORSE` | `gemini-2.5-flash` |
| 2 | `backend/meta_evolution/directive_rewriter.py:202` | `model="gemini-2.5-flash"` arg to `g_client.models.generate_content()` | PIN (Vertex call) | `GEMINI_WORKHORSE` | `gemini-2.5-flash` |
| 3 | `backend/news/sentiment.py:81` | `SCORER_MODEL_GEMINI_FLASH = "gemini-2.5-flash"` module const, consumed as the flash-scorer model | PIN | `GEMINI_WORKHORSE` | `gemini-2.5-flash` |
| 4 | `backend/agents/harness_memory.py:322` | `model_name: str = "gemini-2.5-flash"` default param of `ObservationMasker.__init__` | PIN (→ `get_context_window`) | `GEMINI_WORKHORSE` | `gemini-2.5-flash` |
| 5 | `backend/agents/harness_memory.py:503` | `model_name: str = "gemini-2.5-flash"` default param of `create_masker()` | PIN | `GEMINI_WORKHORSE` | `gemini-2.5-flash` |
| 6 | `backend/services/autonomous_loop.py:2670` | `(settings.gemini_model or "gemini-2.5-flash").strip()` — `_model_for_block` fallback | PIN (fallback selector) | `GEMINI_WORKHORSE` | `gemini-2.5-flash` |
| 7 | `backend/services/autonomous_loop.py:2685` | `(settings.gemini_model or "gemini-2.5-flash").strip()` — `model_name` fallback → `make_client()` | PIN (fallback selector) | `GEMINI_WORKHORSE` | `gemini-2.5-flash` |
| 8 | `backend/api/agent_map.py:132` | `out["live_model"] = node.get("model") or "gemini-2.5-flash"` (gemini_locked branch) | PIN (resolved-model fallback) | `GEMINI_WORKHORSE` | `gemini-2.5-flash` |
| 9* | `scripts/harness/run_autonomous_loop.py:74` | `evaluator_model="gemini-2.5-flash"` arg to `AutonomousLoopOrchestrator(...)` | PIN (runtime harness entrypoint) | `GEMINI_WORKHORSE` | `gemini-2.5-flash` |

**All 9 resolve to `gemini-2.5-flash` → `GEMINI_WORKHORSE`. NONE routes to `GEMINI_DEEP_THINK`.**
(No behavioural `gemini-2.5-pro` PIN exists outside model_tiers.py; the only pro-pins are
`_BUILD_TIER["gemini_deep_think"]` and `settings.deep_think_model`, both already handled
inside/by 75.5.)

**Status of the 8 named sites since 75.5 (internal-audit item 4):** all 8 still present,
none deleted/fixed. Line numbers: 1-5 & 8 STABLE; sites 6-7 MOVED (`services/autonomous_loop.py`
`2648→2670`, `2663→2685`, +22 lines). No NEW *backend* pin appeared. **Site 9 is the one
discovered addition** (a runtime-capable `scripts/harness` entrypoint; see §6 scope note).

### 2b. FAMILY GUARD — behavioural but NOT a pin (C1 cleanup only)

| file:line | shape | handling |
|---|---|---|
| `backend/agents/llm_client.py:985` | `bundle.model_name.startswith("gemini-2.5")` (+ `"-pro" not in ...` :986) — disables thinking-budget for 2.5-flash family | Route the literal through a NEW `GEMINI_2_5_FAMILY_PREFIX = "gemini-2.5"` constant in model_tiers.py: `startswith(GEMINI_2_5_FAMILY_PREFIX)`. Byte-identical behaviour. NOT a "routed pin" for C2/C3. |

### 2c. NON-BEHAVIOURAL — EXCLUDED from the "zero literals" assertion (with reason)

| file:line(s) | class | why excluded |
|---|---|---|
| `backend/config/model_tiers.py` (52,60,65-84,129,133) | constants HOME | criterion says "outside model_tiers.py"; these literals ARE the single source of truth |
| `backend/agents/cost_tracker.py:23,26,27,149` | pricing table `MODEL_PRICING` + docstring | step-prompt names this non-behavioural; dict KEYS/DATA |
| `backend/agents/harness_memory.py:49,50` | `MODEL_CONTEXT_WINDOWS` capability-table KEYS | co-located in an IN-SCOPE file (see §6 — must be converted, not excluded) |
| `backend/api/settings_api.py:26,196,198,199` | `_VALID_MODELS` whitelist + UI pricing dropdown | input-validation catalog + display data |
| `backend/agents/llm_client.py:797,921,976` | docstring + comments | prose (file IS in-scope → reword; see §6) |
| `backend/agents/orchestrator.py:384` | docstring | prose (file IS in-scope → reword; see §6) |
| `backend/agents/rag_agent_runtime.py:11,53,55` | `models/gemini-embedding-2` + `2.5-flash` comment | does NOT contain substring `gemini-2.5`; passes naturally |
| `backend/news/sentiment.py:30` | module-docstring scorer-enum list | co-located in IN-SCOPE file → reword |
| `backend/api/agent_map.py:120` | docstring | co-located in IN-SCOPE file → reword |
| `backend/services/autonomous_loop.py:2174` | narrative comment | co-located in IN-SCOPE file → reword |
| `backend/agents/_inventory.json` (~20 nodes) | roster DISPLAY metadata | JSON data; `node["model"]` used ONLY for `agent_map` `live_model` DISPLAY (agent_map.py:132,146), NEVER as a call selector; not routable to a Py constant |
| `backend/.env.example:11` | sample env template | documentation; not routable |
| `scripts/away_ops/metered_spend.py:35,40,44,45` | away-ops pricing dict + comments | pricing catalog (like MODEL_PRICING) |
| `scripts/compliance/finra_rationale_audit.py:50` | `"model":"gemini-2.5-pro"` in a synthetic audit-trace dict | recorded metadata in a manually-run compliance backfill; not a call selector |
| `scripts/add_phase_27*.py` (several) | masterplan migration-TEXT payloads | historical narrative in one-off migration scripts |
| `scripts/debug/smoke_vertex_model.py:17,18` | CLI usage examples in help text | debug script docstring |
| `backend/tests/**` (many) | test fixtures | step-prompt names test fixtures non-behavioural |

---

## 3. model_tiers.py constants + `gemini_retirement_warning()` mechanics (internal-audit item 2)

- `GEMINI_WORKHORSE = "gemini-2.5-flash"` (`model_tiers.py:52`). Confirmed.
- `GEMINI_DEEP_THINK = "gemini-2.5-pro"` (`model_tiers.py:60`). Confirmed — created by 75.5
  (llmeng-06). (Minor: `_BUILD_TIER["gemini_deep_think"]` :133 still uses the literal
  `"gemini-2.5-pro"`, not the constant — INSIDE model_tiers.py so excluded/out-of-scope; note only.)
- `GEMINI_2_5_RETIREMENT_DATE = _date(2026, 10, 16)`; `GEMINI_2_5_WARN_FROM = _date(2026, 9, 15)` (:65-66).
- `gemini_retirement_warning(model, today=None) -> str | None` (:69-86): returns `None` if
  `model` is falsy or `"gemini-2.5" not in str(model)`; `None` if `today < 2026-09-15`;
  else a warning string containing the ISO shutdown date. **Pure + date-injectable.**
- **HOW SITES GET "COVERED" (internal-audit item 2):** `gemini_retirement_warning` has
  **ZERO runtime callers** — it is only DEFINED and exercised by the 75.5 test
  (`test_phase_75_llm_rail.py:413-425`). So "coverage" is the **test-time relationship**:
  each routed site's resolved model (all == `GEMINI_WORKHORSE == "gemini-2.5-flash"`, which
  contains `"gemini-2.5"`) → the function fires after the warn date. **Criterion 3 needs NO
  runtime wiring** — the C3 test passes each resolved constant to the function, exactly as
  75.5 did. (Observation, out-of-scope: the tripwire is never emitted at startup in prod —
  a candidate follow-up, NOT this step.)

**75.5 test patterns C2/C3 MUST mirror (`test_phase_75_llm_rail.py`):**
- C1 scan (`:386-399`): `CRITERION_4_FILES` list + `@pytest.mark.parametrize("rel", ...)` →
  per-file `assert "gemini-2.5" not in text` (names the offending file on failure).
- C2 (`:402-410`): import the routed symbol, assert `default == GEMINI_WORKHORSE` via
  `inspect.signature(...).parameters[...].default`; assert `GEMINI_DEEP_THINK` truthy.
- C3 (`:413-425`): `assert gemini_retirement_warning("gemini-2.5-pro", date(2026,9,15))` and
  `"2026-10-16" in warn`; two negative controls — `date(2026,9,14)) is None` (before) and
  `gemini_retirement_warning("gemini-3.5-flash", date(2026,9,15)) is None` (off-family).
- Doctrine header (`:23-27`): "a guard that cannot fail does not count … assert BEHAVIOUR
  … rather than the presence of a substring in source, **except where a criterion
  explicitly demands a scan**." (Criterion 1 DOES demand a scan.)

**Import ergonomics (internal-audit item 3, precedent):** 75.5 already added
`from backend.config.model_tiers import GEMINI_WORKHORSE  # phase-75.5 (llmeng-06)` to
`evaluator_agent.py:49`, `rag_agent_runtime.py:45`, `skill_modification_review.py:39`,
`orchestrator.py:53`, `backend/autonomous_loop.py:21`, and locally in
`multi_agent_orchestrator.py:276`; `settings.py:11` imports `GEMINI_DEEP_THINK`. **No
circular-import risk** — model_tiers' only module-level imports are `datetime`/`typing`
(it imports `settings` lazily INSIDE `resolve_model`). `agent_map.py` already does a local
`from backend.config.model_tiers import resolve_model` (:142). The 6 backend pin-files +
the script currently do NOT import the constants (add them).

---

## 4. Coverage rounds log (audit-class, K=2)

| Round | Scope / patterns | New read-in findings | Dry? |
|---|---|---|---|
| 1 | `backend/**/*.py`, multi-pattern (`gemini[-_.]?2[._]5`, `2.5-flash\|pro`, `models/gemini`) — full classification | 8 backend pins + all non-behavioural classes | no (baseline) |
| 2 | `backend/**/*.py` alt patterns (bare `2.5-flash/pro`; `model(_name/_id)=['\"]gemini`) | 0 new behavioural | **DRY** |
| 3 | `scripts/**/*.py` + `backend/` non-py (json/yaml/env) | 1 new: `run_autonomous_loop.py:74` | no |
| 4 | model-passing kwargs across `backend/`+`scripts/` (`evaluator_model`/`planner_model`/`vertex_model`/`standard`/`scorer_model`=gemini-2.5) | 0 new (only re-hit :74) | **DRY** |
| 5 | broad catch-all `gemini-2.5` over `backend/` + `scripts/harness` + `scripts/away_ops`, minus known | 0 new (re-hit known pin + pricing comments) | **DRY** |

**2 consecutive dry rounds (4 & 5) → `coverage.dry = true`.**

---

## 5. External research — key findings (per-claim cited; full table §8)

1. **gemini-2.5-flash shutdown = 2026-10-16, successor `gemini-3.6-flash`; gemini-2.5-pro
   shutdown = 2026-10-16, successor `gemini-3.1-pro-preview`** — official. "The shutdown
   dates listed in the table indicate the _earliest possible dates_ on which a model might
   be retired." (ai.google.dev/gemini-api/docs/deprecations, 2026-07-24). **This CONFIRMS
   the project's `GEMINI_2_5_RETIREMENT_DATE = 2026-10-16`.**
2. **Current stable flash workhorse has advanced to `gemini-3.6-flash`** (catalog now:
   3.6-flash, 3.5-flash, 3.5-flash-lite, 3.1-flash-lite, 2.5-flash, 2.5-flash-lite,
   2.5-pro). (ai.google.dev/gemini-api/docs/models.) → For the eventual Oct-2026 migration
   the WORKHORSE target is now `gemini-3.6-flash`, NEWER than the `gemini-3.5-flash`
   named in model_tiers.py's comments. **Not actionable for 75.5.2 (no value changes) —
   migration-step context only.**
3. **"Same model, different retirement dates per platform" is a real operational risk;
   always confirm dates on official pages.** Confirms 2.5 → 2026-10-16. (hidekazu-konishi
   AI deprecation/lifecycle calendar.) — the cross-domain case FOR a single greppable
   retirement target + tripwire.
4. **Replace Magic Literal / Symbolic Constant (settled prior art):** benefits are
   *single point of change*, *self-documenting*, *reduced duplication* — "much easier to
   change the value of a constant than to search for this number throughout the entire
   codebase, without the risk of accidentally changing the same number used elsewhere."
   Caveat: "If the purpose of a number is obvious, there's no need to replace it."
   (Fowler catalog; sourcemaking; refactoring.guru.) → validates routing the pins to
   `GEMINI_WORKHORSE`; the caveat is why non-behavioural DATA keys (pricing/roster) are
   legitimately left as literals.
5. **Gemini-2.5 retirement migration playbook: test early, don't wait for the last
   month.** (gcpstudyhub, 2026.) — supports the value-pin tripwire as a deliberate
   migration forcing-function.

## 6. C1 SCAN DESIGN (exclusion rule + anti-vacuous — the crux deliverable)

**Design:** tree-wide `(REPO/"backend").rglob("*.py")` **plus** the one runtime harness
script `scripts/harness/run_autonomous_loop.py`, minus an EXPLICIT exclusion set;
`@pytest.mark.parametrize` per file → `assert "gemini-2.5" not in text` (names the file on
failure). This is the **exact 75.5 idiom** (`test_phase_75_llm_rail.py:396-399`), raised
from a hand-picked list to a glob-derived tree with justified exclusions.

**EXCLUSION set (each a non-behavioural DATA/catalog home; justify inline):**
- `backend/config/model_tiers.py` — constants home (criterion: "outside model_tiers.py").
- `backend/agents/cost_tracker.py` — `MODEL_PRICING` (pricing-table data keys).
- `backend/api/settings_api.py` — `_VALID_MODELS` whitelist + UI pricing dropdown (data).
- `backend/tests/**` — fixtures.
(No exclusion needed for `_inventory.json`/`.env.example` — not matched by a `*.py` glob.)

**Why raw-substring-per-in-scope-file, NOT a comment/docstring-skipping classifier:**
criterion 1 says "docstring prose included … read strictly, not reinterpreted." A
classifier that exempts comments/docstrings would be *reinterpreting the scan* (forbidden)
and would let a future pin hide in a commented-out line. So every IN-SCOPE file is cleaned
to zero `gemini-2.5` — **prose included**.

**Anti-vacuous / anti-leak guarantees (attacks the vacuous-guard family directly):**
- The exclusion set is 3 py files + tests, each a genuine data catalog. It EXCLUDES **no**
  file containing a behavioural pin. It does NOT exclude `llm_client.py` / `orchestrator.py`
  (the most LLM-central files) — excluding them would create the exact blind spot the step
  warns against (a future `model="gemini-2.5-flash"` there would pass unseen).
- **Add a self-test** (`test_scan_is_non_vacuous`): assert the derived in-scope list is
  non-empty (`len > 50`) AND is a superset of the known pin-files — mirrors the 75.5.8
  doctrine "a scan that cannot locate its own known members fails." Catches a glob that
  silently returns nothing OR an over-broad exclusion.
- Mutation M1 (restore a literal at any of the 9 sites) → that file's parametrized case
  fails. Mutation M3 (add a pin-file to EXCLUDE) → the self-test fails.

**Consequence — co-located non-behavioural literals in IN-SCOPE files MUST be cleaned too**
(this is where "docstring prose included" bites):
- `harness_memory.py:49-50` context-window table keys → `GEMINI_WORKHORSE` / `GEMINI_DEEP_THINK`.
- `sentiment.py:30`, `agent_map.py:120`, `autonomous_loop.py:2174`, `orchestrator.py:384`,
  `llm_client.py:797/921/976` prose → reword to drop the `gemini-2.5` token.
- `llm_client.py:985` family-guard → `startswith(GEMINI_2_5_FAMILY_PREFIX)` (new constant).

**Scope note (scripts/):** criterion 1 says "outside **model_tiers.py**" (not "outside
backend/"), so a strict reading includes scripts/. Only ONE scripts/ file has a behavioural
pin (`run_autonomous_loop.py:74`). RECOMMENDATION: include that one file + route it (do
NOT sweep all of scripts/ — its other hits are pricing/migration-text/debug DATA). If the
executor's contract instead scopes the scan strictly to backend/, still route :74 and add
a one-file scan leg — leaving a live behavioural gemini-2.5 pin unrouted contradicts the
step's purpose.

## 7. PLAN recommendations (sized for a SONNET executor — exact edits + test design)

### 7a. `model_tiers.py` — add ONE constant (no VALUE change)
```python
# phase-75.5.2: family-membership prefix for the llm_client thinking-budget guard.
# Value == the substring the guard already used; routing it here makes the family
# check a named, retirement-visible target. NOT a model pin.
GEMINI_2_5_FAMILY_PREFIX = "gemini-2.5"
```

### 7b. The 9 PIN sites (route literal → `GEMINI_WORKHORSE`; add the import)
Import line (75.5 precedent, verbatim): `from backend.config.model_tiers import GEMINI_WORKHORSE  # phase-75.5.2`
| site | edit |
|---|---|
| directive_review.py:159 | `model=GEMINI_WORKHORSE,  # phase-60.1: 2.0-flash discontinued 2026-06-01` |
| directive_rewriter.py:202 | `model=GEMINI_WORKHORSE, ...` |
| sentiment.py:81 | `SCORER_MODEL_GEMINI_FLASH = GEMINI_WORKHORSE  # phase-60.1: ...` |
| harness_memory.py:322, :503 | `model_name: str = GEMINI_WORKHORSE,` |
| services/autonomous_loop.py:2670, :2685 | `(settings.gemini_model or GEMINI_WORKHORSE).strip()` |
| api/agent_map.py:132 | `out["live_model"] = node.get("model") or GEMINI_WORKHORSE` (add `GEMINI_WORKHORSE` to the existing local `model_tiers` import at :142 or add module-level) |
| scripts/harness/run_autonomous_loop.py:74 | `evaluator_model=GEMINI_WORKHORSE,  # phase-60.1: 2.0-flash discontinued` |

### 7c. Co-located cleanups in IN-SCOPE files (so the strict scan passes)
- **harness_memory.py:49-50** → `GEMINI_WORKHORSE: 1_048_576,` and `GEMINI_DEEP_THINK: 1_048_576,`
  (import both; LEAVE `:48 gemini-2.0-flash` — no constant, no `gemini-2.5` substring;
  value-preserving because `get_context_window(GEMINI_WORKHORSE)` hits the same entry).
- **Prose rewords (drop the exact `gemini-2.5` token — capitalised "Gemini 2.5" / space form
  or reference the constant NAME both avoid the lowercase-hyphen substring):**
  sentiment.py:30, agent_map.py:120, autonomous_loop.py:2174, orchestrator.py:384
  (orchestrator already imports GEMINI_WORKHORSE at :53), llm_client.py:797/921/976.
- **llm_client.py:985** → `and bundle.model_name.startswith(GEMINI_2_5_FAMILY_PREFIX)`
  (import the new constant). Byte-identical behaviour.
- DO NOT touch the migration/DB enum in `scripts/migrations/add_news_sentiment_schema.py`
  (out of scope — a persisted enum VALUE, not a pin). sentiment.py:30's docstring may keep
  the cross-reference `…:24-36` without repeating the literal.

### 7d. Test design — `backend/tests/test_phase_75_5_2_model_pins.py` (mirror `test_phase_75_llm_rail.py`)
- **C1:** glob-derived `IN_SCOPE` (backend/*.py − EXCLUDE − tests, + run_autonomous_loop.py);
  `test_scan_is_non_vacuous` (non-empty + superset of known pin-files); parametrized
  `assert "gemini-2.5" not in text`.
- **C2:** `assert GEMINI_WORKHORSE == "gemini-2.5-flash"` and `GEMINI_DEEP_THINK == "gemini-2.5-pro"`
  (the value-pin — also satisfies mutation "changed value fails"); introspect module-const /
  param-default sites (`sentiment.SCORER_MODEL_GEMINI_FLASH`, `ObservationMasker.__init__` &
  `create_masker` defaults) `== GEMINI_WORKHORSE`; behavioural for agent_map
  (`_inject_live_model({"gemini_locked":True})["live_model"] == GEMINI_WORKHORSE`);
  behavioural for directive_review/rewriter (patch `genai.Client`, force Anthropic-fail,
  capture the `model=` kwarg == `"gemini-2.5-flash"`) — GOLD; AST fallback (model arg is
  `ast.Name id="GEMINI_WORKHORSE"`) for the two deep autonomous_loop fallbacks + the script.
  **Include a MISROUTE guard (mutation M4):** assert each inline site references
  `GEMINI_WORKHORSE` specifically (not `GEMINI_DEEP_THINK`) — proves C2 is not redundant
  with C1.
- **C3:** for `{GEMINI_WORKHORSE, GEMINI_DEEP_THINK}` assert `gemini_retirement_warning(m, date(2026,9,15))`
  is truthy and contains `"2026-10-16"`; negatives — `date(2026,9,14)) is None` (before) and
  `gemini_retirement_warning("gemini-3.6-flash", date(2026,9,15)) is None` (off-family successor).
- **C4 (mutation matrix → experiment_results.md, prose, not code):** M1 restore literal at
  ONE of the 9 sites → that file's C1 case fails; M2 change `GEMINI_WORKHORSE` value → C2
  value-pin fails; M3 over-broad exclusion → self-test fails; M4 route a site to
  `GEMINI_DEEP_THINK` → C2 misroute-guard fails while C1 still passes (proves C2 independent).

### 7e. Pitfalls the executor must avoid
- The `GEMINI_WORKHORSE == "gemini-2.5-flash"` value-pin is an INTENTIONAL tripwire that a
  legitimate Oct-2026 migration will trip — do NOT loosen it to "fix" a future failure.
- A careless prose reword that leaves the exact `gemini-2.5` token still fails C1 (use the
  space/capital form or the constant name).
- `services/autonomous_loop.py` is 167 KB — if a module-top import risks load-order issues,
  a local import inside the function is acceptable (agent_map.py:142 precedent).
- `gemini_retirement_warning` has NO runtime caller — do NOT try to wire a startup hook
  (out of scope; criterion 3 is the test-time relationship only).
- Adjacent OUT-OF-SCOPE observations (do NOT fix here; candidates to QUEUE per the
  discovered-defects doctrine): (i) the tripwire is never emitted at runtime;
  (ii) `_inventory.json` (~20 nodes) + `.env.example` carry stale-able `gemini-2.5-flash`
  DISPLAY/sample strings a JSON-consistency test could pin; (iii) `_BUILD_TIER["gemini_deep_think"]`
  still uses the literal `"gemini-2.5-pro"` not `GEMINI_DEEP_THINK` (inside model_tiers.py).

## 8. External sources

### Read in full (>=5 required; counts toward the gate)
| # | URL | Accessed | Kind | Fetched | Key finding |
|---|-----|----------|------|---------|-------------|
| 1 | https://ai.google.dev/gemini-api/docs/deprecations | 2026-07-24 | Official doc | WebFetch full | 2.5-flash & 2.5-pro shutdown 2026-10-16; successors gemini-3.6-flash / gemini-3.1-pro-preview; "earliest possible dates" |
| 2 | https://ai.google.dev/gemini-api/docs/models | 2026-07-24 | Official doc | WebFetch full | Current catalog; gemini-3.6-flash is latest flash workhorse; 2.5-flash/pro still GA |
| 3 | https://sourcemaking.com/refactoring/replace-magic-number-with-symbolic-constant | 2026-07-24 | Authoritative (refactoring) | WebFetch full | single point of change / self-doc / reduced duplication; caveat "obvious number needs no constant" |
| 4 | https://refactoring.guru/replace-magic-number-with-symbolic-constant | 2026-07-24 | Authoritative (refactoring) | WebFetch full | "single source of truth"; when NOT to apply |
| 5 | https://hidekazu-konishi.com/entry/ai_model_deprecation_and_lifecycle_calendar.html | 2026-07-24 | Industry (cross-provider) | WebFetch full | same model different retire dates per platform = operational risk; 2.5 → 2026-10-16 |
| 6 | https://gcpstudyhub.com/blog/google-is-retiring-gemini-2-5-on-agent-platform-what-you-need-to-know-and-do-before-october-2026 | 2026-07-24 | Industry | WebFetch full | retirement "no earlier than 2026-10-16"; test-early migration playbook |

### Snippet-only (context; does NOT count toward gate)
| URL | Kind | Why not read in full |
|-----|------|----------------------|
| https://refactoring.com/catalog/replaceMagicLiteral.html | Fowler catalog | JS-heavy; got principle + example only (partial) |
| https://realpython.com/ref/best-practices/constants/ | Blog | HTTP 403 |
| https://github.blog/changelog/2026-07-02-upcoming-deprecation-of-gemini-2-5-pro-and-gemini-3-flash/ | Vendor changelog | real-world deprecation-forcing-migration (snippet) |
| https://discuss.ai.google.dev/t/gemini-2-5-flash-deprecated-without-warning-earlier-than-shutdown-date/174217 | Forum | "deprecated without warning earlier than shutdown" — reinforces tripwire value (snippet) |
| https://ai.google.dev/gemini-api/docs/changelog | Official | release-notes index (snippet) |
| https://deepmind.google/models/gemini/ | Vendor | Gemini 3.5/3.6 lineup (snippet) |
| https://itnext.io/magic-numbers-are-problematic-5b12cfe5f31a | Blog | magic-number prior art (snippet) |
| https://newsletter.shiftelevate.dev/p/magic-numbers-replace-magic-number-with-named-constant-clean-code | Blog | Clean Code magic-number (snippet) |
| https://codeclaritylab.com/glossary/magic_strings | Glossary | magic-strings "no single source of truth to validate against" (snippet) |
| https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/model-versions | Official | model versions & lifecycle (snippet) |
| https://x.com/GHchangelog/status/2072828939777560672 | Social | Copilot 2.5-pro deprecation 2026-07-31 (snippet) |
| https://aiweekly.co/alerts/google-retires-gemini-20-flash-001-replace-with-25-flash | Industry | the 2.0-flash retirement that cost 9 days (snippet) |

**urls_collected: ~20 unique | read_in_full: 6 | snippet_only: 12+**

## 9. Recency scan (last 2 years, 2024-2026)

Searched 2026 / 2025 / year-less variants. **Findings (NEW vs the project's current state):**
- The Gemini deprecation record (2.5 → 2026-10-16) is 2025-2026 material and **confirms**
  the project's pinned `GEMINI_2_5_RETIREMENT_DATE`. No supersession — the date holds.
- **NEW:** the successor catalog has advanced — the official replacement for gemini-2.5-flash
  is now **gemini-3.6-flash** (model_tiers.py comments still say "gemini-3.5-flash"). This
  is migration-step context, NOT actionable for 75.5.2 (which changes no values); worth
  carrying into the eventual Oct-2026 migration step.
- The Replace-Magic-Literal / single-source-of-truth prior art is **settled/canonical** —
  no recency change; newer blog restatements add nothing over Fowler.
- A 2026 forum report ("2.5-flash deprecated without warning earlier than shutdown date")
  and cross-provider evidence ("same model, different dates per platform") both **strengthen**
  the case for a single greppable target + tripwire — the exact thing this step advances.

## 10. Queries run (3-variant discipline)
- Current-year frontier: `Gemini 2.5 Flash Pro deprecation retirement date 2026 shutdown`
- Year-less canonical: `Gemini API model deprecations lifecycle successor versions`
- Last-2-year window (2025): `Google Gemini model catalog 2025 gemini-2.5-flash replacement gemini-3`
- Year-less canonical (prior art): `replace magic literal with named constant refactoring single source of truth`

## 11. JSON envelope
```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 6,
  "snippet_only_sources": 12,
  "urls_collected": 20,
  "recency_scan_performed": true,
  "internal_files_inspected": 15,
  "coverage": {"audit_class": true, "rounds": 5, "dry_rounds": 2, "K_required": 2, "new_findings_last_round": 0, "dry": true},
  "summary": "Re-derived the gemini-2.5 census tree-wide: exactly the 8 named backend behavioural PINS survive (lines 6-7 moved 2648/2663->2670/2685), all resolve to gemini-2.5-flash -> GEMINI_WORKHORSE; NONE routes to GEMINI_DEEP_THINK. Discovered a 9th runtime pin (scripts/harness/run_autonomous_loop.py:74). One family-guard (llm_client.py:985 startswith) is behavioural-but-not-a-pin -> route via a new GEMINI_2_5_FAMILY_PREFIX constant. Classification rule = SELECTS-a-model (pin) vs pricing/capability KEY / prose / roster-display / sample-record / fixture (non-behavioural). C1 = tree-wide raw-substring scan (75.5 idiom) with a 3-file+tests exclusion set + a non-vacuous self-test; docstring prose in in-scope files must be cleaned (criterion: read strictly). Retirement date 2026-10-16 triple-confirmed official; successor now gemini-3.6-flash (migration context only). Coverage dry after rounds 4&5.",
  "brief_path": "handoff/current/research_brief_75.5.2.md",
  "gate_passed": true
}
```
