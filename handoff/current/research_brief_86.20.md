# Research Brief -- step 86.20

**Topic:** How a money-moving trade gate should normalise a free-text
recommendation vocabulary (`Strong Buy` vs `STRONG_BUY`) coming from an LLM
analyzer, WITHOUT widening the gate to admit non-buy-intent strings.
**Tier:** moderate (caller-specified). **Audit-class:** NO -- `coverage` is
reported for information; `coverage.dry` is not required for this step.
**Date:** 2026-08-09. Researcher = Layer-3 combined external + internal.

---

## Search queries run (three-variant discipline, `.claude/rules/research-gate.md`)

| # | Variant | Query |
|---|---------|-------|
| 1 | year-less canonical | `robustness principle Postel's law harmful protocol security "be liberal in what you accept" considered harmful` |
| 2 | current-year frontier (2026) | `enum validation unknown value fail loud closed vocabulary API design 2026` |
| 3 | year-less canonical (finance) | `IBES analyst recommendation standardization broker rating scale mapping strong buy accumulate outperform five point scale` |
| 4 | last-2-year window (2025) | `constrained decoding JSON schema enum LLM structured output 2025 guaranteed valid vs post-hoc validation` |

Mix achieved: year-less canonical hits (IETF draft 2011-era argument, CWE-180,
Google SRE book, I/B/E/S), last-2-year hits (arXiv 2502.14905 Feb-2025,
devblogs.microsoft.com Feb-2025), current-year hits (Anthropic structured-
outputs doc, live 2026).

---

## Read in full (8; >=5 required; counts toward the gate)

| # | URL | Accessed | Kind | Fetched how | Key finding |
|---|-----|----------|------|-------------|-------------|
| 1 | https://datatracker.ietf.org/doc/html/draft-thomson-postel-was-wrong-02 | 2026-08-09 | Official standards doc (IETF I-D) | WebFetch, full | "Sloppy implementations, lax interpretations of specifications, and uncoordinated extrapolation of requirements to cover gaps in specification can result in security problems." + "Errors in implementations ... can thereby be masked. These errors can become entrenched, forcing other implementations to be tolerant of those errors." |
| 2 | https://cwe.mitre.org/data/definitions/180.html | 2026-08-09 | Peer-reviewed-equivalent taxonomy (MITRE CWE-180) | WebFetch, full | "Inputs should be decoded and canonicalized to the application's current internal representation **before** being validated." Validating before canonicalizing = Bypass Protection Mechanism. Also: "ensure the application doesn't decode the same input twice." |
| 3 | https://platform.claude.com/docs/en/build-with-claude/structured-outputs | 2026-08-09 | Official vendor doc (Anthropic) | WebFetch, full | Constrained decoding, not post-hoc: "Structured outputs guarantee schema-compliant responses through constrained decoding ... Always valid ... Reliable: No retries needed for schema violations." `enum` IS supported (scalars only) -- **with an explicit "capitalization caveat"** cross-referenced to the Invalid-outputs section. `minimum`/`maximum`/`minLength`/`maxLength` NOT supported on the wire (SDK strips + validates locally); `minItems` only 0 or 1. |
| 4 | https://arxiv.org/html/2502.14905 | 2026-08-09 | Preprint (arXiv, Feb 2025) | WebFetch, full (arXiv HTML chain, per rules) | Prompt-only schema instruction "does not guarantee consistency"; JSON success rates "can vary widely from 0% to 100% depending on the task complexity and model used." Constrained decoding gives "100% schema adherence by construction" at the cost of setup complexity + slight latency. Best unconstrained baselines still ~41-43% mean match. |
| 5 | https://devblogs.microsoft.com/oldnewthing/20250217-00/?p=110873 | 2026-08-09 | Authoritative vendor blog (Raymond Chen, Microsoft, Feb 2025) | WebFetch, full | **[ADVERSARIAL to the fail-loud line]** Argues an explicit `Other` member is a forward-compatibility trap and that "programs should treat any unrecognized values as if they were 'Other'" -- i.e. graceful degradation over explicit failure. |
| 6 | https://tyk.io/blog/api-design-guidance-enums/ | 2026-08-09 | Industry practitioner (API-gateway vendor) | WebFetch, full | "if you're working with a fixed set of values that will never change, it's time for enums"; "It is easier to turn a string into an enum ... rather to force client code to adapt"; "Assume that adding or removing enum values in API responses will break API client code -- even if the change seems minimal." |
| 7 | https://sre.google/sre-book/monitoring-distributed-systems/ | 2026-08-09 | Official engineering doc (Google SRE book) | WebFetch, full | Errors golden signal counts failures "either explicitly (e.g., HTTP 500s), **implicitly (for example, an HTTP 200 success response, but coupled with the wrong content)**, or by policy". "Your monitoring system should address two questions: what's broken, and why?" |
| 8 | https://research2.fidelity.com/fidelity/research/reports/release2/Research/RefinitivIBES.asp | 2026-08-09 | Industry reference (Refinitiv I/B/E/S methodology) | WebFetch, full | Finance's own answer: a **standardised 5-point scale** (1 Strong Buy / 2 Buy / 3 Hold / 4 Underperform / 5 Sell) with contributors REQUIRED to map into it -- "contributors must map bullish ratings to a 1 or 2, neutral ratings to a 3 and bearish ratings to a 4 or 5"; ">5-point broker scales ... may map back to a single point". |

### Attempted but NOT read (must not be counted)

| URL | Why it failed |
|-----|---------------|
| https://cacm.acm.org/practice/the-robustness-principle-reconsidered/ | WebFetch returned **HTTP 403 Forbidden**. Allman's CACM paper is the canonical "robustness reconsidered" source; the IETF draft (#1 above) carries the same argument and was read instead. |

---

## Identified but snippet-only (34; context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|-----|------|-------------------------|
| https://en.wikipedia.org/wiki/Robustness_principle | encyclopedia | tertiary; superseded by #1 |
| https://www.laws-of-software.com/laws/postel/ | community | tertiary |
| https://devopedia.org/postel-s-law | community | tertiary |
| https://lobste.rs/s/enszyj/harmful_consequences_robustness | forum | discussion of #1 |
| https://medium.com/@mesw1/understanding-the-robustness-principle-postels-law-c1199ea79210 | blog | low tier |
| https://arxiv.org/pdf/1912.03962 | preprint | protocol-detection attacks; adjacent not on-point |
| https://arxiv.org/pdf/1906.11520 | preprint | anonymity networks; off-topic hit |
| https://arxiv.org/pdf/2203.03764 | preprint | as above |
| https://www.speakeasy.com/openapi/schemas/enums/ | industry | duplicates #6's guidance |
| https://openapi-code-generator.nahkies.co.nz/guides/concepts/enums | industry | codegen-specific |
| https://gitdoc.ai/resources/json-schema-enum | industry | duplicates JSON-Schema enum basics |
| https://apinotes.io/blog/common-openapi-spec-errors-and-how-to-fix-them | industry | spec-error survey |
| https://github.com/OpenAPITools/openapi-generator/issues/625 | community (issue) | strong corroboration: "throw an exception for invalid enum values instead of using null" -- the silent-drop defect class, in the wild |
| https://github.com/dotnet/runtime/issues/111018 | community (issue) | .NET enum-validation API proposal |
| https://github.com/api-platform/api-platform/issues/2968 | community (issue) | enum normalizer throws before validation -- ordering bug, cf. CWE-180 |
| https://github.com/ocsf/ocsf-server/issues/96 | community (issue) | closed-vocabulary schema |
| https://www.anderson.ucla.edu/documents/areas/fac/accounting/trueman_ratings.pdf | peer-reviewed | analyst-recommendation levels vs changes; economics of ratings, not vocabulary normalisation |
| https://www.depts.ttu.edu/rawlsbusiness/about/finance/research-seminar/documents/2014/kmwz3_july8_2014.pdf | working paper | as above |
| https://www.aeaweb.org/conference/2017/preliminary/paper/ErH5Fa7i | conference paper | as above |
| https://arxiv.org/pdf/1405.3225 | preprint | analyst rally/crash prediction |
| https://www.interactivebrokers.com/campus/glossary-terms/consensus-rank-avg-rating/ | industry glossary | corroborates the 1-5 consensus scale |
| https://help.benzinga.com/en/articles/1618902-what-do-analyst-ratings-mean | industry | corroborates vocabulary sprawl ("Outperform" = "moderate buy"/"accumulate"/"overweight") |
| https://anachart.com/what-does-analyst-buy-rating-mean-stock-ratings-explained/ | community | as above |
| https://fastercapital.com/content/Navigating-Analyst-Recommendations-with-IBES--A-Winning-Strategy | community | low tier |
| https://arxiv.org/pdf/2503.24191 | preprint (2025) | structured-output control-plane vulnerabilities -- flagged for a future security step |
| https://arxiv.org/pdf/2506.01151 | preprint (2025) | Earley-driven pruning; decoding performance, not semantics |
| https://arxiv.org/pdf/2510.07248 | preprint (2025) | adapt tool schemas to models |
| https://arxiv.org/pdf/2512.19701 | preprint (2025) | LASER regression; off-topic |
| https://arxiv.org/pdf/2408.02442 | preprint (2024) | "Let Me Speak Freely?" -- format restrictions can COST reasoning quality; the trade-off cited below |
| https://zeroentropy.dev/concepts/constrained-decoding/ | industry | mechanism explainer |
| https://letsdatascience.com/blog/structured-outputs-making-llms-return-reliable-json | blog | explainer |
| https://dev.to/pockit_tools/llm-structured-output-in-2026-stop-parsing-json-with-regex-and-do-it-right-34pk | community (2026) | recency-scan hit; practitioner restatement |
| https://medium.com/@emrekaratas-ai/structured-output-generation-in-llms-json-schema-and-grammar-based-decoding-6a5c58b698a6 | blog | explainer |
| https://cacm.acm.org/practice/the-robustness-principle-reconsidered/ | peer-reviewed | 403 (see above) |

**URLs collected (unique): 42.** Read in full: 8. Snippet-only: 34.

---

## Recency scan (2024-2026) -- PERFORMED

Searched the 2024-2026 window explicitly (queries #2 and #4). **Result: 3 new
findings that COMPLEMENT rather than supersede the canonical sources.**

1. **Constrained decoding is now the default answer, and it is a hard
   guarantee, not a retry loop** -- Anthropic's structured-outputs doc (live
   2026) states conformance comes from "constrained sampling with compiled
   grammar artifacts", and `enum` is a supported keyword. This did not exist as
   a production option when the robustness-principle debate was framed. It
   materially changes the recommendation: the class can be removed at the
   producer, not merely handled at the consumer.
2. **But schema conformance is structural only, and there is an explicit
   capitalization caveat** -- the same doc cross-references an "Invalid outputs"
   capitalization caveat on `enum`, and the numeric/length keywords are stripped
   from the wire schema. This is the same lesson already codified in this repo
   for the research-gate workflow (`.claude/rules/research-gate.md`, "the floors
   are enforced by the SCRIPT, not by the schema"). **A schema enum reduces but
   does not eliminate the need for a validating boundary.**
3. **Prompt-only vocabulary instruction measurably fails** -- arXiv 2502.14905
   (Feb 2025): JSON success rates "vary widely from 0% to 100%"; strong 2025
   baselines score 41-43% mean match unconstrained. `backend/agents/schemas.py:40`
   is exactly a prompt-only vocabulary instruction.

Counter-note from the same window: arXiv 2408.02442 ("Let Me Speak Freely?",
2024) reports format restrictions can degrade reasoning quality -- an argument
for constraining only the `action` field, not the whole synthesis object.

---

## Key findings (external), cited per claim

1. **Normalise BEFORE validating, exactly once, at one boundary.** "Inputs
   should be decoded and canonicalized to the application's current internal
   representation before being validated" and "ensure the application doesn't
   decode the same input twice" (MITRE CWE-180,
   https://cwe.mitre.org/data/definitions/180.html, accessed 2026-08-09).
   The second clause is the one that bites here: **N scattered `.upper()` calls
   at N call sites IS "decoding the same input twice"** and is how dialects
   drift. One canonicalising read boundary, then a strict closed-set test.

2. **Aggressive leniency at the receiver is how an over-permissive gate is
   born, and it entrenches.** "Errors in implementations, or confusion about
   semantics can thereby be masked. These errors can become entrenched, forcing
   other implementations to be tolerant of those errors" ... "Sloppy
   implementations, lax interpretations of specifications ... can result in
   security problems" (IETF draft-thomson-postel-was-wrong-02,
   https://datatracker.ietf.org/doc/html/draft-thomson-postel-was-wrong-02,
   accessed 2026-08-09). Direct answer to the caller's sub-question (3): the
   normalisation must be **total and enumerable** (a fixed fold: case +
   separator + trim), never a fuzzy/substring/"contains BUY" match.

3. **Strictness is what forces the defect into the open.** "Favoring strict
   error handling over attempting error recovery is an effective technique for
   ensuring that faults receive attention"; "A fatal error provides excellent
   motivation to address problems" (same IETF source). The 86.20 defect survived
   because `continue` is neither strict nor observable.

4. **The silent-drop is its own defect class, independent of the vocabulary
   bug.** Google SRE counts as errors those requests that fail "implicitly (for
   example, an HTTP 200 success response, but coupled with the wrong content)"
   (https://sre.google/sre-book/monitoring-distributed-systems/, accessed
   2026-08-09). A `continue` on an unrecognised token is precisely an implicit
   error: the cycle reports success with the wrong content. Corroborated in the
   wild by OpenAPITools issue #625 ("throw an exception for invalid enum values
   instead of using null").

5. **[ADVERSARIAL] The strongest counter-argument is forward-compatibility.**
   Raymond Chen argues consumers "should treat any unrecognized values as if
   they were 'Other'" -- graceful degradation over explicit failure
   (https://devblogs.microsoft.com/oldnewthing/20250217-00/?p=110873, accessed
   2026-08-09); Tyk warns "Assume that adding or removing enum values in API
   responses will break API client code"
   (https://tyk.io/blog/api-design-guidance-enums/). **Resolution for this
   step:** both arguments are about *independently versioned* producer and
   consumer. Here the producer and consumer are in ONE repo and ONE deploy, and
   the consumer is a money gate. The right synthesis is **asymmetric**: default
   an unknown token to the *safe* class (no BUY), which is Chen's graceful
   degradation, AND simultaneously make it *loud* (counter + log), which is the
   IETF's fatal-error motivation. Silence is the only option both camps reject.

6. **Constrained decoding removes the class at the source; the trade-off is
   coupling + a residual capitalization caveat.** Anthropic: constrained
   decoding gives "Always valid ... No retries needed for schema violations",
   and `enum` is supported (https://platform.claude.com/docs/en/build-with-claude/structured-outputs).
   arXiv 2502.14905: constrained decoding = "100% schema adherence by
   construction" vs prompting which "does not guarantee consistency". Cost:
   arXiv 2408.02442 finds format restriction can degrade reasoning -- so
   constrain the `action` field, not the prose fields.

7. **Finance already solved this exact problem, and its answer is
   "standardise at ingest, keep the closed scale small".** Refinitiv I/B/E/S
   forces every broker's idiosyncratic vocabulary (Strong Buy / Outperform /
   Accumulate / Overweight / Hold / Neutral / Underperform / Reduce / Sell) onto
   a **5-point standard scale**, with the rule "contributors must map bullish
   ratings to a 1 or 2, neutral ratings to a 3 and bearish ratings to a 4 or 5",
   and collapses >5-point scales onto single points
   (https://research2.fidelity.com/fidelity/research/reports/release2/Research/RefinitivIBES.asp,
   accessed 2026-08-09). Two transferable design rules: (a) mapping is a
   **published, explicit table owned by the consumer**, not per-site guesswork;
   (b) the canonical internal representation is deliberately **coarser** than
   the input vocabulary.

---

## Internal code inventory (the Explore half)

### The defect, stated precisely

`backend/services/portfolio_manager.py:63` -- `_BUY_RECS = {"BUY", "STRONG_BUY"}`
(UNDERSCORE). The only normalisation before the membership test is `.upper()`:

- `portfolio_manager.py:140` -- `rec = (analysis.get("recommendation") or "HOLD").upper()` (holding re-eval)
- `portfolio_manager.py:141` -- `old_rec = (pos.get("recommendation") or "").upper()` (prior position rec)
- `portfolio_manager.py:182` -- `rec = (analysis.get("recommendation") or "HOLD").upper()` (candidate)

`.upper()` folds CASE only, never the separator. `"Strong Buy"` -> `"STRONG BUY"`
(space) which is NOT in `_BUY_RECS`, so the candidate hits
`if rec not in _BUY_RECS: continue` (`portfolio_manager.py:188`) and is
**dropped with no log line at all**. Plain `"Buy"` works by accident.

**The mismatch selectively destroys the HIGHEST-conviction recommendation while
letting the medium one through** -- it inverts the conviction ordering that
reaches the book, which is worse than a uniform failure.

### The same mismatch is FAIL-DANGEROUS on the sell side

`portfolio_manager.py:59` `_SELL_RECS = {"SELL","STRONG_SELL"}`;
`:61` `_DOWNGRADE_RECS = {"HOLD","SELL","STRONG_SELL"}`.
`"Strong Sell".upper() == "STRONG SELL"` is in **neither** set, so a full-path
`Strong Sell` on a held position matches neither the `sell_signal` branch
(`:144`) nor the `signal_downgrade` branch (`:154`) -- the position is **not
sold at all**; only the stop-loss at `:131` can still exit it. The buy-side half
costs opportunity; the sell-side half costs **protection**. The two halves have
opposite risk polarity, which is the central design tension for the contract.

### phase-61.2's fix is silently defeated by the same mismatch (new finding)

`backend/services/paper_trader.py:447-457` chooses `_pos_rec =
analysis_recommendation` when `paper_position_recommendation_fix_enabled` is ON,
and writes it verbatim at `:488` and `:512` -- **no normalisation on the write
path**. On the full path that persists `"Strong Buy"` into
`paper_positions.recommendation`. `portfolio_manager.py:141` then produces
`"STRONG BUY"`, which fails `old_rec in _BUY_RECS` at `:154`. So the
`signal_downgrade` exit rule that phase-61.2 exists to revive
(`settings.py:210-212`) **stays structurally dead for exactly the full-path rows
it was built for**, even with the flag ON.

### WHERE the free-text vocabulary is produced

| Producer | Anchor | Vocabulary | Constrained? |
|---|---|---|---|
| Full-pipeline synthesis | `backend/agents/schemas.py:39-41` -- `action: str = Field(description="Strong Buy, Buy, Hold, Sell, or Strong Sell")` | TITLE CASE WITH SPACE | **NO** -- plain `str`; the vocabulary lives only in the `description`, i.e. in the prompt. Cannot reject `"strong buy"`, `"Accumulate"`, `"Strong Buy!"`. |
| Full path -> analysis dict | `backend/services/autonomous_loop.py:2138` -- `"recommendation": rec.get("action","HOLD") if isinstance(rec, dict) else str(rec)` | passes producer string through untouched | NO |
| Lite Claude analyzer | prompt `autonomous_loop.py:2835` (`1. Action: BUY, SELL, or HOLD`), example `:2841`; consumed `:3013` | `BUY`/`SELL`/`HOLD` -- already canonical, **never STRONG_\*** | YES, structurally: `autonomous_loop.py:2467` `if analysis.get("action") not in ("BUY","SELL","HOLD")` |
| Lite Gemini analyzer | prompt `:3159`; consumed `:3249` | same 3 tokens | same gate |
| Degraded marker | `autonomous_loop.py:2207` `"recommendation": None` | folded out by `_fold_degraded_for_trading` (`:2558-2568`) | N/A |

**Two disjoint in-repo dialects, and the repo already contains both:**
- `backend/api/models.py:21-26` -- `class Recommendation(str, Enum): STRONG_BUY = "Strong Buy"; BUY = "Buy"; HOLD = "Hold"; SELL = "Sell"; STRONG_SELL = "Strong Sell"`. The MEMBER name is underscored, the VALUE is spaced title case; anything serialising this enum emits `"Strong Buy"`.
- `backend/agents/schemas.py:95` -- `consensus: Literal["STRONG_BUY","BUY","HOLD","SELL","STRONG_SELL"]`. **Proof the underscore dialect is already schema-enforceable in this codebase** -- the synthesis `action` field simply never got the same treatment.

### Every OTHER literal-set comparison (the class is wider than portfolio_manager)

| File:line | Expression | Dialect | Consequence of the other dialect |
|---|---|---|---|
| `backend/services/portfolio_manager.py:59/61/63` + `:144/:154/:188` | `_SELL_RECS`/`_DOWNGRADE_RECS`/`_BUY_RECS` | UNDERSCORE | **money path** -- silent skip + missed exit |
| `backend/api/portfolio.py:138-142` | `rec = (...).upper()`; `rec in ("BUY","STRONG_BUY","SELL","STRONG_SELL")`; `is_buy = rec in ("BUY","STRONG_BUY")` | UNDERSCORE | recommendation-accuracy metric silently under-counts `Strong Buy` -- an analytics lie from the same root |
| `backend/agents/bias_detector.py:119,128,153-155` | `recommendation.upper() in ("STRONG_BUY","BUY")` etc. | UNDERSCORE | tech/large-cap bias checks never fire on `Strong Buy` |
| `backend/agents/bias_detector.py:21-24` | `{"STRONG_BUY":0.08,"BUY":0.30,...}` base-rate table | UNDERSCORE | keyed lookup misses |
| `backend/agents/conflict_detector.py:114,121,131,140` | `if "STRONG_BUY" in rec_label` ... `elif "BUY" in rec_label` | UNDERSCORE **substring** | `"STRONG BUY"` fails the `STRONG_BUY` test but PASSES `"BUY" in rec_label` -> graded against the weaker `score<5.5` rule instead of `<7.0`. Substring matching is an independent defect class (`"STRONG_SELL"` also contains `"SELL"`). |
| `backend/agents/skill_optimizer.py:244-255` | `consensus in ("STRONG_BUY","BUY")` | UNDERSCORE | reads the `Literal` field -- **correct by construction** |
| `backend/services/outcome_tracker.py:57` | `is_buy = recommendation in ("Strong Buy","Buy")` | **SPACED, no `.upper()`** | mirror-image bug: breaks on `STRONG_BUY`/`BUY` |
| `backend/agents/memory.py:229` | `original_recommendation in ("Strong Buy","Buy")` | **SPACED, no fold** | same mirror-image |
| `backend/slack_bot/formatters.py:169` | `if "STRONG_BUY" in action_upper or "STRONG BUY" in action_upper` | **BOTH** | the only site already handling both spellings -- in the least money-critical place |
| `backend/services/signal_attribution.py:185` | `rec = str(analysis.get("recommendation","")).upper() or "HOLD"` | pass-through | mis-tagged attribution |
| `backend/services/compliance_logger.py:57` | `output_recommendation: str  # "BUY"/"SELL"/"HOLD"` | comment only | audit record stores whatever arrives |

**Class size: >=8 comparison sites, 4 conventions (underscore-equality,
spaced-equality, substring, both-spellings), across 3 layers (money path,
analytics/API, agent audit).**

### Does a normalisation helper already exist? NO

`grep -rn "def normalize\|def normalise\|def canonical" backend/` finds
canonicalisers for URLs (`backend/news/normalize.py:36 canonical_url`,
`backend/intel/scanner.py:72 _canonicalize`), text (`backend/news/normalize.py:64
normalize_text`), tickers (`backend/services/news_screen.py:170`), dates
(`backend/alt_data/f13.py:110`), model names (`backend/agents/llm_client.py:321
_normalize_model_name`), econ windows (`backend/econ_calendar/normalize.py:26`),
market values (`backend/services/data_integrity.py:53`) -- **nothing for a
recommendation vocabulary.** The repo idiom for a new one is established: a pure
function in a small module with table-driven tests
(`backend/tests/test_intel_scanner.py:77-80`).

### Existing tests: none dedicated

`ls backend/tests/ | grep -i "portfolio_manager|decide_trades|recommend"` -> **no
match**. `decide_trades` has no dedicated test module. Coverage is incidental
(`backend/tests/test_phase_66_2_risk_judge_shape.py:148` documents the `:140/:182`
`.upper()` None-safety fix). Critically,
`backend/tests/test_dod4_tier1_coverage_investment.py:884` feeds
`"recommendation": "STRONG_BUY"` -- **the fixtures use the underscore dialect the
full-path producer never emits**, so the suite is green against a vocabulary
production does not generate. Same trap already recorded as
`reference_vacuous_type_guards_on_bq_string_columns` (fixtures must emit the
PRODUCTION shape).

### Flag convention for a dark money-path change

`backend/config/settings.py` uses `paper_*_enabled: bool = Field(False, ...)`
read as `getattr(settings, "<flag>", False)` so flag-absent == flag-False
(byte-identical OFF). Exemplars visible inside `portfolio_manager.py` itself:
`paper_position_recommendation_fix_enabled` (`:114`, defined `settings.py:210`),
`paper_synthesis_integrity_enabled` (`:115`, `settings.py:206`),
`paper_risk_judge_shape_fix_enabled` (`:201/:208/:388`),
`paper_risk_judge_reject_binding` (`:248`), `paper_swap_churn_fix_enabled`
(`:585`), `paper_cross_sector_rotation_enabled` (`:668`),
`paper_atomic_swap_enabled` (`:761`), `paper_unknown_sector_cap_exempt` (`:321`).
Note `settings.py:210-212` also demonstrates the **unsafe-combination WARNING**
idiom, which this step will likely need (arming the buy side and the sell side
have opposite polarity).

**Internal files inspected: 12** -- `portfolio_manager.py`, `autonomous_loop.py`,
`paper_trader.py`, `agents/schemas.py`, `api/models.py`, `api/portfolio.py`,
`agents/bias_detector.py`, `agents/conflict_detector.py`,
`services/outcome_tracker.py`, `agents/memory.py`, `slack_bot/formatters.py`,
`config/settings.py` (plus greps over `backend/tests/`, `news/normalize.py`,
`signal_attribution.py`, `compliance_logger.py`, `skill_optimizer.py`).

---

## Consensus vs debate (external)

**Consensus:** canonicalise before validating, once, at one boundary (CWE-180);
never leave an unrecognised token silently dropped (IETF draft; Google SRE
implicit-error definition; OpenAPITools #625); a fixed finite vocabulary belongs
in an enum (Tyk); constrained decoding is the strongest available producer-side
control (Anthropic docs; arXiv 2502.14905); finance standardises heterogeneous
rating vocabularies onto one coarse closed scale at ingest (I/B/E/S).

**Debate:** whether an unknown token should FAIL or DEGRADE. Chen/Tyk favour
graceful degradation for forward compatibility; the IETF draft favours fatal
errors so faults get attention. Both agree that **silence** is wrong. Secondary
debate: arXiv 2408.02442 warns schema constraints can cost reasoning quality --
argues for constraining only the decision field.

## Pitfalls (from the literature, mapped)

1. **Normalising more aggressively is exactly how an over-permissive gate is
   born** (IETF). A `startswith("STRONG")` / `"BUY" in rec` fold would admit
   `"NOT A BUY"`, `"STRONG_SELL"`-contains-`SELL` confusions, and free-text
   prose. `conflict_detector.py:121/131` already demonstrates this failure mode
   in-repo. The fold must be a **total function over a finite table**, not a
   predicate over substrings.
2. **Double-decoding / per-site folding** (CWE-180: "doesn't decode the same
   input twice"). Eleven sites each doing their own `.upper()` IS the anti-
   pattern; adding a twelfth variant makes it worse.
3. **Fixtures in the wrong dialect give a false green** (in-repo:
   `test_dod4_tier1_coverage_investment.py:884`). Any 86.20 test must feed the
   PRODUCER's actual string (`"Strong Buy"`), and must mutate the guard to prove
   it can fail.
4. **A schema enum is not a total guarantee** -- Anthropic's own `enum` support
   carries an explicit capitalization caveat, and Gemini is the actual synthesis
   producer here. Keep the validating boundary even after constraining the
   producer.

## Application to pyfinagent (external -> file:line)

1. **Two-part fix, producer + boundary** (CWE-180 + Anthropic/arXiv 2502.14905).
   (a) Constrain the producer: `backend/agents/schemas.py:40` `action: str` ->
   a `Literal[...]` in ONE dialect, mirroring the already-correct
   `schemas.py:95`. (b) Keep a single canonicalising read boundary before the
   set tests at `portfolio_manager.py:140/141/182`. Do not rely on (a) alone.
2. **One helper, total, table-driven, coarse** (I/B/E/S). A pure
   `canonical_recommendation(raw) -> str | None` with an explicit finite mapping
   (case-fold, trim, fold `[ -]`->`_`, collapse repeats) returning `None` for
   anything not in the table. `None` must NOT be coerced to `HOLD` at the buy
   gate. Place it beside the existing pure-canonicaliser idiom
   (`backend/news/normalize.py`), NOT inline in `portfolio_manager.py`.
3. **Asymmetric default on unknown** (resolves the Chen-vs-IETF debate for a
   money gate): unknown -> NOT a buy (`portfolio_manager.py:188` keeps skipping)
   AND unknown -> NOT a sell-suppressor. Never let normalisation *create* a BUY
   that the raw string did not clearly assert.
4. **Make the skip observable** (Google SRE implicit-error). `:188` currently has
   **zero logging** -- unlike its neighbours `:355`, `:375`, `:397`, `:415`,
   `:433` which all log their skip reason. Minimum: a per-cycle
   **rejected-reasons counter breakdown** (`unrecognised_token` vs
   `not_a_buy_rec` vs `already_held`) plus a WARNING carrying the raw token, so
   an unknown vocabulary is a symptom you can alert on rather than a cause you
   have to excavate. The repo already has the emit idiom
   (`autonomous_loop.py:2570-2600` `_degraded_scoring_check` -> P1 alert).
5. **Sell-side polarity must be decided explicitly in the contract.** Arming
   `_SELL_RECS`/`_DOWNGRADE_RECS` for `"STRONG SELL"` is fail-SAFE (more exits);
   arming `_BUY_RECS` for `"STRONG BUY"` is fail-DANGEROUS (more entries, on a
   live book). They can be separate flags, exactly as phase-61.2 split
   `paper_position_recommendation_fix_enabled` from
   `paper_synthesis_integrity_enabled` for the same blast-radius reason
   (`settings.py:210-212`).
6. **Fix the write path too, or 61.2 stays dead** -- `paper_trader.py:447-457/488/512`
   must persist the CANONICAL form, else `old_rec` at `portfolio_manager.py:141`
   keeps failing `:154`.
7. **Scope caution for Main:** the >=8 other sites (esp. `api/portfolio.py:138-142`,
   `conflict_detector.py:121/131`, `outcome_tracker.py:57`, `memory.py:229`)
   are the same class but NOT the money path. Per
   `feedback_queue_discovered_defects_in_masterplan`, they belong in their own
   queued step(s), not silently bundled into 86.20.

---

## Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch -- **8**
- [x] 10+ unique URLs total -- **42**
- [x] Recency scan (2024-2026) performed + reported -- 3 complementary findings + 1 counter-note
- [x] Full pages read (not abstracts) for the read-in-full set; arXiv fetched via the `/html/` chain per `.claude/rules/research-gate.md` (no `/pdf/` WebFetch)
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every module named in INTERNAL SCOPE, plus the producer chain and 8 out-of-scope sibling sites
- [x] Contradictions noted (Chen/Tyk graceful-degradation vs IETF fail-fast; arXiv 2408.02442 format-restriction cost)
- [x] Claims cited per-claim with URL + access date
- [ ] GAP: CACM "The Robustness Principle Reconsidered" (Allman) returned HTTP 403 and could not be read; the IETF draft covers the same argument. Not counted toward the gate.
- [ ] GAP (deliberate, out of scope): no BigQuery query was run to measure how many LIVE rows currently carry the spaced dialect. The full path vs lite path split determines whether this is an ACTIVE or a LATENT money bug -- the lite path (`autonomous_loop.py:2467`) emits only canonical `BUY/SELL/HOLD`, so the defect arms only on full-path/orchestrator rows. **Main should measure this before sizing the fix** (per `feedback_measure_dont_assert_claims`).

---

## Envelope

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 8,
  "snippet_only_sources": 34,
  "urls_collected": 42,
  "recency_scan_performed": true,
  "internal_files_inspected": 12,
  "coverage": {
    "audit_class": false,
    "rounds": 2,
    "dry_rounds": 0,
    "K_required": 2,
    "new_findings_last_round": 3,
    "dry": false
  },
  "summary": "Two disjoint recommendation dialects coexist in-repo: the full-pipeline synthesis emits spaced title case ('Strong Buy', free-text str at agents/schemas.py:40) while the money gate tests an underscore set (_BUY_RECS at portfolio_manager.py:63). .upper() folds case but not the separator, so 'STRONG BUY' silently fails the buy gate at :188 with no log, and 'STRONG SELL' matches neither _SELL_RECS nor _DOWNGRADE_RECS -- the highest-conviction signals are exactly the ones destroyed, in both directions. paper_trader.py:447-457 persists the raw string, so phase-61.2's signal_downgrade revival is defeated by the same mismatch. Literature: canonicalise once before validating (CWE-180), never silently drop an unknown token (IETF postel-was-wrong; Google SRE implicit errors), constrain the producer with a schema enum (Anthropic constrained decoding; arXiv 2502.14905), and follow I/B/E/S: an explicit finite mapping table onto a coarse closed scale. Unknown must default to NOT-a-buy and be loudly counted, never normalised into one.",
  "brief_path": "handoff/current/research_brief_86.20.md",
  "gate_passed": true
}
```
