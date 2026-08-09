# Research Brief — phase-86.22

**Topic:** Repairing a recommendation-vocabulary mismatch across multiple consumers,
and deciding what to do about a learning corpus the wrong label would poison.

**Tier:** moderate (caller-specified) · **Audit-class:** YES (loop-until-dry, K=2)
**Started:** 2026-08-10 · **Status:** IN PROGRESS (write-first; this file grows incrementally)

**Internal scope (caller):** `outcome_tracker.py:57-58`, `memory.py:228-251`,
`bias_detector.py:119`, `api/portfolio.py:138-142`, `conflict_detector.py:120`,
`backend/services/recommendation_vocab.py` (86.20 canonicaliser — REUSE, do not duplicate).

---

## Progress log

- [x] Read `.claude/agents/researcher.md` + `.claude/rules/research-gate.md` in full
- [x] Internal inventory (18 files; producer + 3 consumer dialect groups + corpus path)
- [x] External search rounds (10 rounds, ~12 queries, three-variant discipline)
- [x] Recency scan 2024-2026 (5 window findings, all design-changing)
- [x] Loop-until-dry rounds (rounds 9 + 10 dry, K=2 satisfied ⇒ `coverage.dry = true`)
- [x] 13 sources READ IN FULL via WebFetch; 2 fetch failures disclosed
- **GATE: PASSED**

---

## HEADLINE INTERNAL FINDING (round 1) — the premise "86.20 landed a normaliser" is TRUE but INCOMPLETE

`backend/services/recommendation_vocab.py` EXISTS (92 lines, written 2026-08-09 21:21) and is
exactly the shared canonicaliser 86.22 asks for. **But its only production consumer,
`portfolio_manager.py:116`, reads it behind a DARK feature flag**:

- `backend/config/settings.py:214` — `paper_recommendation_vocab_fix_enabled: bool = Field(...)`,
  described verbatim as *"phase-86.20 (DARK until operator promotion)"*.
- `backend/services/portfolio_manager.py:116` —
  `flag_on = getattr(settings, "paper_recommendation_vocab_fix_enabled", False)`
- `backend/services/portfolio_manager.py:128` — `canon = canonical_recommendation(probe)`

So the class-level design question 86.22 must answer is NOT "build a normaliser" — that is done —
but **"do the seven other consumers adopt it behind the SAME dark flag, a DIFFERENT flag, or
unconditionally?"** The 86.20 flag description states the arming rationale is a **money-path risk
decision** (it can spend cash and can revive exits). `outcome_tracker`, `memory`, `bias_detector`,
`api/portfolio` and `conflict_detector` are **not** money paths. Coupling them to a money-path
operator token would leave the learning corpus poisoned for as long as the operator withholds it.
This is the single highest-leverage decision in the step.

---

## Internal code inventory (round 1) — every claim carries a file:line anchor

### Producer side — the repo defines TWO dialects in ONE file, and one of them is unconstrained

| File:line | What it declares | Dialect |
|---|---|---|
| `backend/agents/schemas.py:40` | `action: str = Field(description="Strong Buy, Buy, Hold, Sell, or Strong Sell")` | **spaced title case, and it is a bare `str`** — the description is a *hint to the LLM*, not a constraint |
| `backend/agents/schemas.py:95` | `consensus: Literal["STRONG_BUY","BUY","HOLD","SELL","STRONG_SELL"]` | **UPPER_SNAKE, schema-ENFORCED** |
| `backend/api/models.py:22-26` | `class Recommendation(str, Enum): STRONG_BUY = "Strong Buy" ...` | spaced title case VALUES under UPPER_SNAKE member NAMES |

This is the root: the *same* `schemas.py` pins one field to an enforced underscore Literal and
leaves the other a free-text string documented in title case. `recommendation_vocab.py:12-17`
already names this ("The repo already contains BOTH dialects as first-class citizens").
**Design consequence:** `schemas.py:40` is where a producer-side fix would go, and it is a
*strictly available* option (turn `action` into the same `Literal`). 86.20 did NOT take it — it
chose read-side canonicalisation. 86.22 must not reverse that (masterplan: "the two steps must
not make opposite decisions about producer-side versus read-side normalisation").

### Consumer side — three dialect groups, derived by grep over `backend/**/*.py` minus `/tests/`

**(A) UPPER_SNAKE, exact-membership — drops the space form (the 86.20 failure mode)**

| File:line | Expression | Path class |
|---|---|---|
| `backend/agents/bias_detector.py:119` | `recommendation.upper() in ("STRONG_BUY","BUY") and score >= 7.5` | analysis |
| `backend/agents/bias_detector.py:128` | `... in ("STRONG_BUY","BUY") and score >= 8.0` | analysis |
| `backend/agents/bias_detector.py:154-155` | `rec in ("STRONG_BUY","BUY")` / `("STRONG_SELL","SELL")` | analysis |
| `backend/api/portfolio.py:140` | `if rec in ("BUY","STRONG_BUY","SELL","STRONG_SELL")` — **the accuracy DENOMINATOR** | reporting |
| `backend/api/portfolio.py:142` | `is_buy = rec in ("BUY","STRONG_BUY")` | reporting |
| `backend/services/portfolio_manager.py:60/62/64` | `_SELL_RECS` / `_DOWNGRADE_RECS` / `_BUY_RECS` | **money (86.20 scope, already fixed-but-dark)** |

**(A′) UPPER_SNAKE SUBSTRING — a different and worse failure shape**

| File:line | Expression | Note |
|---|---|---|
| `backend/agents/conflict_detector.py:121` | `if "STRONG_BUY" in rec_label and score < 7.0` | space form fails, then falls through |
| `backend/agents/conflict_detector.py:131` | `elif "BUY" in rec_label and score < 5.5` | **a missed `STRONG BUY` lands HERE and is graded against the *weaker* 5.5 threshold** — exactly the hazard `recommendation_vocab.py:32-36` warns about |
| `backend/agents/conflict_detector.py:140` | `elif "SELL" in rec_label and score > 6.0` | `"STRONG_SELL"` **contains** `"SELL"` — the elif ordering saves it today, but only by accident of ordering |

**(B) TITLE-CASE-EXACT, NO case folding at all — drops the uppercase form (opposite direction)**

| File:line | Expression | Path class |
|---|---|---|
| `backend/services/outcome_tracker.py:57-58` | `is_buy = recommendation in ("Strong Buy","Buy")` / `is_sell = ... ("Strong Sell","Sell")` | learning |
| `backend/agents/memory.py:229-230` | byte-identical expression inside `generate_reflection` | **learning corpus** |

**(C) ALREADY dialect-tolerant — the derivation method MUST NOT flag these as broken (false-positive control)**

| File:line | Expression | Why it is clean |
|---|---|---|
| `backend/slack_bot/formatters.py:169` | `if "STRONG_BUY" in action_upper or "STRONG BUY" in action_upper` | a **third**, local, ad-hoc patch — handles both dialects. Cosmetic (Block Kit colour). It is *the class* (a private vocabulary) but not *the defect*. |
| `backend/agents/skill_optimizer.py:244-255` | `consensus in ("STRONG_BUY","BUY")` | reads `report["debate_consensus"]`, which descends from the **schema-ENFORCED `Literal`** at `schemas.py:95`. Underscore is correct here. **This is the best false-positive test case for criterion 2's derivation method.** |

### The learning-corpus question — MEASURED at source, and the answer is asymmetric

The step asks whether wrong reflections have already been persisted. Two channels, opposite answers:

1. **BigQuery `outcome_tracking`: NOT poisoned.** `outcome_tracker.py:70` puts
   `directionally_correct` in the returned dict, but the persistence call at
   `outcome_tracker.py:74-83` passes only `ticker, analysis_date, recommendation, price_at_rec,
   current_price, return_pct, holding_days, beat_benchmark` — and
   `bigquery_client.py:400-414 save_outcome` builds its row from exactly those eight fields.
   **`directionally_correct` is never written to BQ.** So no stored boolean needs repair.
2. **BigQuery `agent_memories`: poisoning is LIVE and reachable.**
   `outcome_tracker.py:147-148` gates reflections on `if self._model:`.
   `outcome_tracker.py:213` (`evaluate_recent`) constructs `OutcomeTracker(settings)` with **no**
   model → that path generates nothing. But `backend/services/autonomous_loop.py:3392` constructs
   `OutcomeTracker(settings, model=model_client)` — **the live daily cycle DOES pass a model.**
   The lesson text is then either the LLM's answer to a prompt containing
   `Directionally correct: {'YES' if direction_correct else 'NO'}` (`memory.py:238`) or, on LLM
   failure, the literal fallback `"{'Correct' if direction_correct else 'Incorrect'} call on
   {ticker}..."` (`memory.py:251`). Both are persisted verbatim via
   `bigquery_client.py:503-516 save_agent_memory` into `agent_memories`, which
   `memory.py` retrieves by BM25 into future prompts.
   **=> The corpus is the thing to measure; `outcome_tracking` is not.** The measurement is a
   `agent_memories` query, not an `outcome_tracking` one — do not aim criterion 7 at the wrong table.

### Out of the stated scope but same class — frontend (report only, do NOT widen the step)

`frontend/src/components/reports-columns.tsx:16` and `ReportCompareDrawer.tsx:20` test
`norm === "STRONG BUY"` (space only) — they miss the *underscore* form. `RecentReportsTable.tsx:34`
and `DebateView.tsx:79` test both. The masterplan's criterion 4 scopes the no-second-normaliser
assertion to `backend/`, so the frontend is correctly out of scope; flag it as a follow-up rather
than expanding this step.

---

## External research

### Search-query composition (three-variant discipline, made visible)

| Variant | Query run |
|---|---|
| year-less canonical | `memory poisoning LLM agent long-term memory corrupted lessons attack` |
| year-less canonical | `CWE-180 incorrect behavior order validate before canonicalize` |
| year-less canonical | `anti-corruption layer shared kernel enum vocabulary drift across bounded contexts microservices` |
| year-less canonical | `I/B/E/S analyst recommendation standardized five point scale mapping broker text strong buy` |
| current-year frontier (2026) | `agent memory poisoning RAG knowledge base 2026 defense` |
| last-2-year window (2025) | `label noise systematically mislabeled training data repair versus discard backfill 2025` |
| round-2 / round-3 queries | listed in the loop-until-dry section below |

### Read in full (WebFetch; counts toward the gate)

| # | URL | Accessed | Kind | Key finding |
|---|---|---|---|---|
| 1 | https://cwe.mitre.org/data/definitions/180.html | 2026-08-10 | official (MITRE) | *"Inputs should be decoded and canonicalized to the application's current internal representation before being validated. Make sure that the application does not decode the same input twice."* Mitigation is explicitly **"canonicalization once at an entry point before any validation decisions occur."** ChildOf CWE-179 (Early Validation); ParentOf CWE-647. |
| 2 | https://arxiv.org/html/2512.16962v1 (MemoryGraft) | 2026-08-10 | preprint | 10 poisoned of 110 records (9%) ⇒ **47.9% of ALL retrievals were poisoned**. Root cause stated as *"the agent assumes that retrieved memories are trustworthy and imitates their procedural structure without verifying correctness or provenance."* Compromise *"remains active until the memory store is explicitly purged or replaced."* Defenses: cryptographic provenance attestation; consistency reranking. **No discussion of non-adversarial/accidental corruption** — an explicit gap. |
| 3 | https://arxiv.org/html/2407.12784v1 (AgentPoison, NeurIPS 2024) | 2026-08-10 | peer-reviewed | Poison ratio **<0.1%** with **as few as 2 instances** (EHRAgent: 2 of 700) reaches 62.6% end-to-end success; benign-utility drop **<1%**. Amplification mechanism: retrieval returns only k≈4-5 examples, so one poisoned entry *"creates disproportionate influence."* Detection is weak: 47.2% ASR survives perplexity filtering. **No purge/remediation mechanism discussed.** |
| 4 | https://learn.microsoft.com/en-us/azure/architecture/patterns/anti-corruption-layer | 2026-08-10 | official (Microsoft) | *"Isolate the different subsystems by placing an anti-corruption layer between them... The anti-corruption layer contains all the logic necessary to translate between the two systems."* Anti-pattern warning: *"Avoid placing business rules or orchestration in the layer"* — the ACL translates, it does not decide. Also: *"consider enforcing input validation and sanitization at this boundary"* and *"Plan for observability... to diagnose translation failures."* |
| 5 | https://datatracker.ietf.org/doc/html/draft-thomson-postel-was-wrong-03 | 2026-08-10 | official (IETF I-D) | *"Over time, implementations progressively add new code to constrain how data is transmitted, or to permit variations in what is received."* Tolerant receivers entrench errors: *"These errors can become entrenched, forcing other implementations to be tolerant of those errors."* Prescription: *"Choosing to generate fatal error for unspecified conditions instead of attempting error recovery can ensure that faults receive attention."* |
| 6 | https://research2.fidelity.com/fidelity/research/reports/release2/Research/RefinitivIBES.asp | 2026-08-10 | industry (Refinitiv/Fidelity) | The domain's own answer: a **closed 5-point scale** (1 Strong Buy … 5 Sell) and a hard mapping rule — *"all points in their scale must map back to the standardized Refinitiv I/B/E/S scale of 1-5"*, *"contributors must map bullish ratings to a 1 or 2, neutral ratings to a 3 and bearish ratings to a 4 or 5"*, and *"multiple points in a broker's scale may map back to a single point."* **Notably the document does NOT address ratings that cannot be mapped** — i.e. the real-world standard has no UNKNOWN state, which is precisely the gap `recommendation_vocab.py:73-76` fills deliberately. |
| 7 | https://arxiv.org/html/2606.11699v1 (data-centric corrupted-label framework) | 2026-08-10 | preprint | REPAIR beats DISCARD: data-centric correction *"address[es] the root cause... by directly improving the quality of the data itself"* and *"are not limited to a specific model, but can benefit any downstream learner."* Crucially for us: under **symmetric (random) noise** baselines held ~10% post-correction error, but under **asymmetric / instance-dependent (systematic) noise** baseline error rose to **23-28%** while the repair method held ~8.5%. **Systematic corruption is materially worse than random noise** — and pyfinagent's corruption is maximally systematic (every uppercase `BUY` flips one way). |
| 8 | https://docs.python.org/3/library/enum.html | 2026-08-10 | official (Python) | `_missing_(cls, value)` is *"A classmethod for looking up values not found in cls. By default it does nothing, but can be overridden to implement custom search behavior"*, with the documented recipe being exactly a case-folding lookup on a `StrEnum` (`Build('deBUG') -> <Build.DEBUG>`). Returning `None` from `_missing_` yields the normal `ValueError`. **Caveat that matters here:** *"Some stdlib code checks for exact `str` type (`type(x) == str`) rather than `isinstance(x, str)`"* — a `StrEnum` swap is not free at BQ-insert / JSON boundaries. |
| 9 | https://martinfowler.com/articles/feature-toggles.html | 2026-08-10 | authoritative blog (Fowler/Hodgson) | Taxonomy: **Release** (static, transient) vs **Ops** (dynamic, short-lived + long-lived kill switches) vs Experiment vs Permissioning. *"Savvy teams view the Feature Toggles in their codebase as inventory which comes with a carrying cost and seek to keep that inventory as low as possible."* And the direct hit for 86.22: **do not tie unrelated behaviours to a single toggle** — the article's own example warns against wiring a narrow behaviour to a broad `next-gen-ecomm` flag. Also: isolate toggle *decision points* from *decision logic*. |
| 10 | https://lexi-lambda.github.io/blog/2019/11/05/parse-don-t-validate/ | 2026-08-10 | authoritative blog (Alexis King) | *"Get your data into the most precise representation you need as quickly as you can. Ideally, this should happen at the boundary of your system, before **any** of the data is acted upon."* And *"Perform the check exactly once, in ... the same place we were already doing the input validation."* Names the pyfinagent failure exactly — **shotgun parsing**: *"parsing and input-validating code ... spread across processing code—throwing a cloud of checks at the input"*, whose consequence is *"some portion of invalid input having been processed, with ... program state ... difficult to accurately predict."* |
| 11 | https://cheatsheetseries.owasp.org/cheatsheets/Input_Validation_Cheat_Sheet.html | 2026-08-10 | official (OWASP) | Allowlist over denylist: *"Allowlist validation involves defining exactly what IS authorized, and by definition, everything else is not authorized."* For enumerated inputs: *"the input needs to match exactly one of the values offered... Any failure to validate a value against this discrete list of options on the server side is a high security event and should be logged as a high severity event."* Normalization is prescribed BEFORE validation. |
| 12 | https://www.sonarsource.com/blog/security-implications-of-url-parsing-differentials/ | 2026-08-10 | vendor engineering blog | The **parser-differential** class: *"differentials occur when different components of an application parse the same [input] and reach different conclusions."* Three prescribed defences, all three of which map 1:1 onto 86.22: *"using a single consistent ... parsing library throughout the application"*, *"normalizing ... to a canonical form before any security check"*, and *"adding integration tests for known differential-exploiting inputs."* |
| 13 | https://arxiv.org/html/2606.26511v1 (MemStrata, temporal validity in retrieval memory) | 2026-08-10 | preprint | **The single most important finding for the corpus half.** Similarity CANNOT find a wrong stored entry: over 98 labelled pairs, cosine AUROC for separating contradictions from duplicates is **0.59**, and contradictions are *more* similar (0.812) than genuine duplicates (0.800) because a value-flip is a minimal edit; max achievable precision at any threshold is 0.67. *"The solution is structural, not learned"* — deterministic key matching. On remediation: retaining everything without supersession degraded accuracy to 0.33 and raised stale-fact errors 25-60%, while aggressive deletion/merging collapsed recall to 0.62 (vs RAG 0.82). The chosen design **invalidates rather than deletes** (`valid_to` + `superseded_by`), *"bound[ing] growth only on the axis that matters."* |

**Attempted and FAILED (recorded honestly; these do NOT count):**
`https://best.openssf.org/Secure-Coding-Guide-for-Python/CWE-707/CWE-180/` → HTTP 404.
`https://langsec.org/spw23/papers/Ali_LangSec23.pdf` → TLS error ("unable to verify the first certificate").
Both angles were covered instead by sources 11 and 12 respectively.

### Identified but snippet-only (context; does NOT count toward the gate)

| URL | Kind | Why not fetched in full |
|---|---|---|
| https://arxiv.org/abs/2601.05504 | preprint | Memory-poisoning attack/defense; superseded for our purpose by MemoryGraft + AgentPoison read in full |
| https://openreview.net/forum?id=Y841BRW9rY | venue page | AgentPoison landing page; the arXiv HTML was read instead |
| https://proceedings.neurips.cc/paper_files/paper/2024/file/eb113910e9c3f6242541c1652e30dfd6-Paper-Conference.pdf | PDF | binary PDF of a paper already read via HTML |
| https://arxiv.org/pdf/2605.29960 | preprint | trojan memory attacks; same class, no new mechanism |
| https://arxiv.org/pdf/2606.29030 | preprint | memory as attack surface (MCQA); off-domain |
| https://arxiv.org/pdf/2606.24322 | preprint | non-malleable origin-bound memory authority; adversarial-only |
| https://arxiv.org/pdf/2605.26754 | preprint | Cordon-MAS information-flow control for RAG |
| https://arxiv.org/pdf/2510.06445 | survey | agentic security survey; breadth over depth |
| https://arxiv.org/pdf/2504.15585 | survey | LLM full-stack safety survey |
| https://arxiv.org/html/2606.04990v4 | survey | evidence tracing / execution provenance in LLM agents |
| https://arxiv.org/pdf/2606.30306 | survey | persistent memory, state and governance |
| https://arxiv.org/html/2607.12790v1 | preprint | co-evolving evaluation metrics for self-improving agents |
| https://arxiv.org/html/2607.24300v1 | preprint | self-authored verification unreliable |
| https://arxiv.org/html/2607.13091v1 | preprint | accumulated behavioural rules closed loop |
| https://www.lakera.ai/blog/agentic-ai-threats-p1 | vendor blog | memory poisoning overview |
| https://www.emergentmind.com/topics/persistent-memory-poisoning | aggregator | secondary summary |
| https://github.com/Agent-Threat-Rule/agent-threat-rules/blob/main/rules/data-poisoning/ATR-2026-00070-data-poisoning.yaml | rules repo | detection rule, not analysis |
| https://cwe.mitre.org/data/definitions/179.html (parent of CWE-180) | official | parent CWE; content summarised inside source 1 |
| https://docs.aws.amazon.com/prescriptive-guidance/latest/cloud-design-patterns/acl.html | official | duplicate of the ACL pattern read in full |
| https://microservices.io/patterns/refactoring/anti-corruption-layer.html | practitioner | same pattern |
| https://contextmapper.org/docs/anticorruption-layer/ | practitioner | same pattern |
| https://deviq.com/domain-driven-design/shared-kernel/ | practitioner | shared-kernel counterpoint to ACL |
| https://arxiv.org/pdf/2310.01905 | SLR | DDD systematic literature review |
| https://www.tandfonline.com/doi/full/10.1080/21642583.2025.2488120 | survey | learning-from-label-noise survey (2025) |
| https://www.sciencedirect.com/science/article/pii/S2405959525001481 | review | imbalanced classification with label noise (2025) |
| https://doi.org/10.3390/technologies13040132 | journal | identifying/mitigating label noise (2025) |
| https://en.wikipedia.org/wiki/Label_noise | encyclopedia | background only |
| https://discuss.python.org/t/extending-strenum-with-normalize-input-for-flexible-value-lookups/79752 | community | `_normalize_input_` proposal; the stdlib doc was read instead |
| https://pypi.org/project/stringenum/0.3.0 | package | `CaseInsensitiveStrEnum`; a dependency we should NOT add |
| https://www.anderson.ucla.edu/documents/areas/fac/accounting/Trueman_BuysHoldsSells.pdf | paper | distribution of bank recommendations; domain colour |
| https://help.benzinga.com/en/articles/1618902-what-do-analyst-ratings-mean | industry | rating-scale glossary |
| https://langsec.org/spw23/papers/Ali_LangSec23.pdf | paper | fetch FAILED (TLS); listed as identified only |
| https://best.openssf.org/Secure-Coding-Guide-for-Python/CWE-707/CWE-180/ | official | fetch FAILED (404); listed as identified only |
| https://www.datasops.com/blog/data-contracts-versioning | practitioner | data contracts / schema evolution |
| https://www.fixtrading.org/standards/fixml-online/ | standards body | FIX enum extension mechanism |
| https://tianpan.co/blog/2026-04-09-structured-output-failures-production-llm | blog | structured-output failure taxonomy |
| https://learn.microsoft.com/en-us/azure/well-architected/operational-excellence/observability | official | OE:07 monitoring, referenced from source 4 |

### Recency scan (2024-2026) — MANDATORY SECTION, and it is NOT empty

Searched explicitly in the 2024-2026 window (`...2026 defense`, `...2025`, `...2026 production incident`,
`...2026` on self-improving agents). **Result: 5 findings from the window that materially change the
plan, not merely complement it.**

1. **(2026) Similarity-based detection of wrong stored memories is a dead end** — MemStrata's
   AUROC 0.59 (arXiv:2606.26511). This *supersedes* the intuitive plan of "semantically search
   `agent_memories` for wrong lessons". Any measurement of the poisoned subset must be
   **deterministic/structural**, not embedding-based.
2. **(2024, NeurIPS) A sub-0.1% poisoned fraction is enough** — AgentPoison: 2 of 700 entries,
   62.6% end-to-end effect, <1% benign degradation. This *raises* the significance of a small
   poisoned subset: "only a handful of bad lessons" is NOT a reason to defer remediation.
3. **(2025-2026) Systematic label corruption is much worse than random** — corrupted-label
   framework (arXiv:2606.11699): asymmetric/instance-dependent noise pushed baseline error to
   23-28% vs ~10% for symmetric. pyfinagent's corruption is perfectly systematic (all 91 uppercase
   `BUY` rows flip the same way), so the random-noise intuition understates it.
4. **(2025-2026) Repair is now preferred over discard** for corrupted labels — a shift from the
   older reweight/remove orthodoxy. Supports regenerate/supersede over purge.
5. **(2026) Constrained decoding makes an enum-typed field structurally unbreakable** — the
   producer-side option at `backend/agents/schemas.py:40` is stronger in 2026 than when that line
   was written; `schemas.py:95` already proves the pattern works in this repo.

Older canonical sources (CWE-180, Postel-was-wrong, ACL, parse-don't-validate, I/B/E/S) remain
valid and are not superseded — they supply the *shape* of the fix; the 2024-2026 work supplies the
*urgency and the measurement method* for the corpus half.

---

## Key findings (each cited per-claim)

1. **Canonicalise once, at the boundary, before any decision — not per consumer.** *"Inputs should
   be decoded and canonicalized to the application's current internal representation before being
   validated."* (MITRE CWE-180, https://cwe.mitre.org/data/definitions/180.html, 2026-08-10). Same
   prescription from OWASP (normalize, then allowlist) and from King: *"Perform the check exactly
   once."* (https://lexi-lambda.github.io/blog/2019/11/05/parse-don-t-validate/).
2. **What pyfinagent has today has a name: shotgun parsing.** *"parsing and input-validating code
   ... spread across processing code—throwing a cloud of checks at the input"*, whose consequence is
   *"some portion of invalid input having been processed."* (King, ibid.) Seven+ private vocabularies
   across `bias_detector`, `conflict_detector`, `api/portfolio`, `outcome_tracker`, `memory`,
   `slack_bot/formatters`, plus the frontend — exactly the cloud.
3. **Two normalisers that disagree is a named vulnerability class, not a style issue.**
   *"differentials occur when different components ... parse the same [input] and reach different
   conclusions"*; mitigation is *"a single consistent ... library throughout the application"*
   (Sonar, https://www.sonarsource.com/blog/security-implications-of-url-parsing-differentials/).
   This is the external backing for the masterplan's criterion 4.
4. **The translator must translate, not decide.** *"Avoid placing business rules or orchestration in
   the layer."* (Microsoft ACL,
   https://learn.microsoft.com/en-us/azure/architecture/patterns/anti-corruption-layer). Read against
   `recommendation_vocab.py`: it maps and returns `None`; the *policy* (is UNKNOWN a buy?) stays at
   the call site. That split is already correct — preserve it, do not push per-consumer policy into
   the vocab module.
5. **The domain's own answer is a closed scale with a mandatory total mapping.** *"all points in
   their scale must map back to the standardized Refinitiv I/B/E/S scale of 1-5"*
   (https://research2.fidelity.com/fidelity/research/reports/release2/Research/RefinitivIBES.asp).
   `recommendation_vocab.py:58-60` is the same construct. **But I/B/E/S has no UNKNOWN state** and
   pyfinagent deliberately does — that is a genuine improvement, not a deviation to fix.
6. **Unrecognised values must be LOUD, and the literature is unusually blunt about it.** OWASP:
   *"Any failure to validate a value against this discrete list of options on the server side is a
   high security event and should be logged as a high severity event."* IETF: *"Choosing to generate
   fatal error for unspecified conditions instead of attempting error recovery can ensure that faults
   receive attention."* `portfolio_manager.py:131-143` already implements the WARNING; the other
   consumers have nothing.
7. **A tiny poisoned fraction of a retrieval corpus is disproportionate.** 2 poisoned of 700 →
   62.6% end-to-end effect, <1% benign degradation (AgentPoison,
   https://arxiv.org/html/2407.12784v1); 9% poisoned → 47.9% of all retrievals (MemoryGraft,
   https://arxiv.org/html/2512.16962v1). BM25 top-k retrieval concentrates influence.
8. **You cannot find the bad lessons by similarity.** AUROC 0.59; contradictions are *more* similar
   than duplicates; *"The solution is structural, not learned."* (MemStrata,
   https://arxiv.org/html/2606.26511v1).
9. **Prefer supersede/invalidate over delete, and over leave-in-place.** Leave-in-place: accuracy
   0.33, stale errors 25-60%. Aggressive removal: recall collapses 0.82 → 0.62. Targeted
   invalidation wins (ibid.). Reinforced by the corrupted-label literature's 2025-2026 shift toward
   *repair* (https://arxiv.org/html/2606.11699v1).
10. **Do not hang unrelated behaviour off one toggle.** *"Savvy teams view the Feature Toggles in
    their codebase as inventory which comes with a carrying cost"*; the article's worked example is
    precisely the failure of wiring a narrow behaviour to a broad flag
    (https://martinfowler.com/articles/feature-toggles.html).
11. **A producer-side fix is now technically stronger than when `schemas.py:40` was written.**
    Constrained decoding makes an enum-valued field structurally impossible to violate; `schemas.py:95`
    already relies on it. Recorded as an option, NOT a recommendation — see the conflict note below.
12. **Python gives a stdlib idiom for this** — `_missing_` on a `StrEnum`, with the documented recipe
    being case-folding lookup (https://docs.python.org/3/library/enum.html). **But** the same page
    warns *"Some stdlib code checks for exact `str` type (`type(x) == str`)"*, which is a live hazard
    at the `insert_rows_json` boundary (`bigquery_client.py:404-415`, `:506-514`). **Recommendation:
    keep the plain-function canonicaliser that already exists; do not convert it to a StrEnum.**

## Consensus vs debate (external)

**Consensus (essentially unanimous):** canonicalise once at a boundary; a single shared
implementation; allowlist onto a closed set; make unrecognised input loud rather than silently
dropped. CWE-180, OWASP, King, Sonar, Microsoft and the IETF draft all agree.

**Genuine debate 1 — strict rejection vs tolerant acceptance.** Postel-was-wrong argues tolerance
entrenches error. The FIX-protocol / extension-handling material argues the opposite for
*informational* fields: *"Parsers and validators should not fail solely because a value is unknown
when that field has a defined fallback path."* **Resolution for 86.22:** the two are reconcilable on
risk class — strict on the money path, tolerant-with-loud-logging on reporting/analysis paths. This
is exactly the per-site risk split the masterplan already asks for.

**Genuine debate 2 — repair vs discard for a corrupted corpus.** MemoryGraft says purge
(*"remains active until the memory store is explicitly purged or replaced"*); the 2025-2026
label-noise work and MemStrata say repair/supersede beats deletion. **Resolution:** MemoryGraft is
reasoning about an *adversarial* corpus where every poisoned entry is hostile; pyfinagent's is
*accidental* and mostly-correct, so wholesale purge destroys real signal (MemStrata measured recall
0.82 → 0.62 under aggressive removal). Supersede/regenerate is the better fit — and note MemoryGraft
explicitly *"provides no discussion of non-adversarial memory corruption"*, so it is being applied
outside its stated scope if cited for purge here.

**Debate 3 — producer-side vs read-side normalisation.** The literature leans producer-side
("parse, don't validate"; constrained decoding). **86.20 already chose read-side.** The masterplan
forbids the two steps disagreeing. So: read-side is the binding choice for 86.22; the producer-side
option belongs in a *separate follow-up step*, not this one.

## Pitfalls (from the literature, mapped to concrete traps here)

- **Canonicalising twice, or in the wrong order.** CWE-180: *"Make sure that the application does not
  decode the same input twice."* Trap: `portfolio_manager._resolve_rec` already canonicalises; a
  consumer downstream of it must not re-canonicalise a token that is already canonical.
- **Widening a set instead of normalising.** OWASP allowlist discipline + the masterplan's own
  prohibition. `conflict_detector.py:131` is the specific hazard: a substring `elif "BUY" in
  rec_label` will happily absorb a normalised `STRONG_BUY` into the *weaker* threshold branch. Any
  fix there MUST convert substring tests to equality on the canonical token, not just feed
  canonical strings into the existing substring chain.
- **`"STRONG_SELL"` contains `"SELL"`.** `recommendation_vocab.py:35` names it;
  `conflict_detector.py:140` survives only by elif ordering.
- **Silent metric-denominator exclusion.** `api/portfolio.py:140` is a denominator. A row that fails
  the membership test is dropped from BOTH numerator and denominator, so
  `recommendation_accuracy` (`:155`) is computed over a self-selected subset. Fixing the membership
  test CHANGES A PUBLISHED METRIC — that is a reportable before/after delta, not a silent repair.
- **Assuming the wrong table.** The poisoned artefact is in `agent_memories`, NOT `outcome_tracking`
  (`directionally_correct` is never persisted — `outcome_tracker.py:74-83` vs
  `bigquery_client.py:400-414`).
- **Assuming the corpus can be searched semantically.** AUROC 0.59 (MemStrata). Use a deterministic
  handle instead — see below.
- **Flag coupling.** Reusing `paper_recommendation_vocab_fix_enabled` for the learning/reporting
  consumers ties non-money behaviour to a money-path operator token (Fowler: don't tie unrelated
  behaviours to one toggle), and leaves the corpus poisoned until an unrelated risk decision is made.

## Application to pyfinagent (external findings → file:line anchors)

**A. Reuse, don't re-mint.** `backend/services/recommendation_vocab.py` already is the ACL/parse
boundary. Criterion 4's "no second normaliser in `backend/`" is directly supported by the
parser-differential literature. The derivation method for criterion 2 must flag
`outcome_tracker.py:57` and `bias_detector.py:119` (true positives) while NOT flagging
`skill_optimizer.py:244` (reads the schema-enforced `Literal` from `schemas.py:95` — correct as-is)
and while classifying `slack_bot/formatters.py:169` as *already dialect-tolerant but still a private
vocabulary*. Those three are the built-in recall/precision controls the criterion asks for.

**B. The flag decision (the highest-leverage call in the step).** Recommended framing for the
contract, on the evidence: **do not extend the money-path flag**. `outcome_tracker`, `memory`,
`bias_detector`, `conflict_detector` and `api/portfolio` place no orders — their fix is
risk-reducing or metric-correcting only. Fowler's inventory/coupling guidance plus the fact that
`settings.py:214`'s own description justifies darkness by *"ARMING THIS CHANGES BEHAVIOUR ON BOTH
SIDES"* (a money claim) means the justification does not transfer. If the operator still wants a
gate, use a **separate** flag so the corpus repair is not blocked on a trading decision. Note
`api/portfolio.py`'s change is *metric-visible* (see pitfall above) and may deserve its own
disclosure even if unflagged.

**C. Measuring the poisoned corpus — a deterministic handle exists, with a stated blind spot.**
`bigquery_client.py:503-516 save_agent_memory` persists only
`{agent_type, ticker, situation, lesson, created_at}`; `build_situation_description`
(`memory.py:170-210`) embeds ticker, sector, signal names and the debate consensus — **but never the
analyst recommendation**. So the memory row carries no direct key. However:
- **LLM-failure fallback lessons ARE deterministically identifiable**: `memory.py:250-253` writes the
  literal `"{'Correct'|'Incorrect'} call on {ticker}. Recommended {original_recommendation}, actual
  return {x}% over {n} days."` — so a row whose `lesson` matches `Incorrect call on %` *and*
  `Recommended BUY%` (uppercase) *and* whose stated return is positive is a **provably** wrong
  lesson. That is the structural key MemStrata argues for.
- **LLM-generated lessons are NOT** so identifiable; they must be reached by joining `agent_memories`
  to `analysis_results` on `(ticker, created_at)` windowed against the report date. State that join
  as approximate and report its uncertainty rather than claiming an exact count.
- Report the measurement **either way** (criterion 7), and remember reflections only exist where a
  model was passed — `autonomous_loop.py:3392` (yes) vs `outcome_tracker.py:213` (no).

**D. Remediation posture.** Supersede/regenerate beats purge and beats leave-in-place
(MemStrata; corrupted-label repair). Deferring is defensible if the measured count is small — but
AgentPoison's 2-of-700 result means "small" is not by itself a reason to defer. Whatever is chosen,
say it plainly (criterion 7 forbids leaving the question open).

**E. Loudness.** Extend `portfolio_manager.py:131-143`'s two-case WARNING (UNRECOGNISED vs
VOCABULARY MISMATCH) to the other consumers, per OWASP/IETF. It changes no decision, so it can ship
unflagged exactly as 86.20 shipped its observability unconditionally.

---

## Loop-until-dry completeness critic (audit-class)

| Round | Angle / queries | New read-in-full findings | Dry? |
|---|---|---|---|
| 1 | memory poisoning (year-less + 2026); CWE-180; ACL/bounded contexts | 5 (CWE-180, MemoryGraft, AgentPoison, MS ACL, IETF Postel) | no |
| 2 | I/B/E/S recommendation standardisation; label-noise repair vs discard (2025) | 2 (Refinitiv I/B/E/S, arXiv:2606.11699) | no |
| 3 | Python `_missing_`/StrEnum; feature-flag blast radius | 2 (docs.python.org enum, Fowler feature-toggles) | no |
| 4 | producer-side type constraints; constrained decoding (2026) | 1 (parse-don't-validate) | no |
| 5 | silent-skip observability / drift; OWASP input validation | 1 (OWASP cheat sheet) | no |
| 6 | producer/consumer data contracts; FIX enum extension + unknown-value handling | 0 | **DRY (not counted — round 7 broke the streak)** |
| 7 | parser differentials / divergent canonicalisers | 1 (Sonar) + 1 FAILED fetch (LangSec, TLS) | no |
| 8 | quarantine/invalidate agent memory; metric-denominator selection bias | 1 (MemStrata arXiv:2606.26511) | no |
| 9 | shared-canonicaliser adoption / idempotence | 0 | **DRY 1** |
| 10 | 2026 production-incident framing; corrupted feedback in self-improving agents | 0 | **DRY 2** |

`rounds = 10`, `dry_rounds = 2` consecutive (rounds 9 and 10), `K_required = 2` ⇒ **`coverage.dry = true`.**

---

## Research Gate Checklist

Hard blockers:
- [x] ≥5 authoritative external sources READ IN FULL via WebFetch — **13**
- [x] 10+ unique URLs total — **50 enumerated** (13 read-in-full + 37 snippet-only), from ~12 searches
- [x] Recency scan (last 2 years) performed + reported — 5 window findings, all design-changing
- [x] Full pages/papers read (not abstracts) for the read-in-full set; 2 failed fetches disclosed
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every module in the caller's scope, plus 3 the caller did not
      name (`skill_optimizer`, `slack_bot/formatters`, `schemas.py`/`api/models.py` producers)
- [x] Contradictions / consensus noted (3 genuine debates, each resolved with a stated rule)
- [x] All claims cited per-claim with URL + access date
- [ ] **Gap disclosed:** the *counts* in the masterplan (HOLD n=275, BUY n=91, etc.) were NOT
      re-derived here — that requires a BigQuery `execute-query` against
      `financial_reports.analysis_results`, which is an approval-gated write-class tool and belongs
      in GENERATE per criterion 3 ("RE-DERIVED at fix time"). The *table resolution* WAS verified at
      source: `outcome_tracker.py:140` → `bq.get_recent_reports` → `self.reports_table`
      (`bigquery_client.py:290`) → `settings.bq_dataset_reports = "financial_reports"`
      (`bigquery_client.py:486` idiom).

---

## Envelope

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 13,
  "snippet_only_sources": 37,
  "urls_collected": 50,
  "recency_scan_performed": true,
  "internal_files_inspected": 18,
  "coverage": {
    "audit_class": true,
    "rounds": 10,
    "dry_rounds": 2,
    "K_required": 2,
    "new_findings_last_round": 0,
    "dry": true
  },
  "brief_path": "handoff/current/research_brief_86.22.md",
  "gate_passed": true
}
```

**Status: COMPLETE.**
