# Research Brief -- step 86.63

**Topic:** ONE guard at the recommendation-vocabulary boundary instead of a sixth patch at a sixth site.
**Tier:** moderate. **Audit-class:** YES (loop-until-dry, K=2).
**Researcher:** Layer-3 researcher (Workflow rail). **Started:** 2026-08-13.

## Status envelope (born inert -- phase-86.37; flipped to COMPLETE as the final act)

This block was written INCOMPLETE with zeroed counts in the first tool call of the session and
updated in place at the end. It is **byte-identical** to the "Status envelope -- FINAL" block at
the foot of this brief -- deliberately, so a parser that finds either one reads the same answer
and a torn brief can never be ambiguous (auto-memory
`feedback_matching_totals_hide_contradictory_content`: two copies of one fact that disagree are
worse than one copy).

```json
{
  "brief_status": "COMPLETE",
  "tier": "moderate",
  "external_sources_read_in_full": 10,
  "snippet_only_sources": 29,
  "urls_collected": 39,
  "recency_scan_performed": true,
  "internal_files_inspected": 24,
  "coverage": {
    "audit_class": true,
    "rounds": 7,
    "dry_rounds": 2,
    "K_required": 2,
    "new_findings_last_round": 0,
    "dry": true
  },
  "brief_path": "handoff/current/research_brief_86.63.md",
  "gate_passed": true
}
```

---

## Work log (append-only)

- 2026-08-13: brief created, envelope born inert. Read `.claude/agents/researcher.md` +
  `.claude/rules/research-gate.md` in full.

---

## Read in full (>=5 required; counts toward the gate)

| # | URL | Accessed | Kind | Fetched how | Key quote or finding |
|---|-----|----------|------|-------------|----------------------|
| 1 | https://lexi-lambda.github.io/blog/2019/11/05/parse-don-t-validate/ | 2026-08-13 | Authoritative blog (Alexis King, canonical, year-less) | WebFetch full | "the difference between validation and parsing lies almost entirely in how information is preserved". "Get your data into the most precise representation you need as quickly as you can. Ideally, this should happen at the boundary of your system, **before any of the data is acted upon**." Names SHOTGUN PARSING: "Shotgun parsing necessarily deprives the program of the ability to reject invalid input instead of processing it." On re-inlining: scattered validators "allow inconsistency: modifications in one place needn't update others, enabling bugs to slip past undetected", whereas a parse changes the RETURN TYPE so a divergence is caught "before we even run the program". |
| 2 | https://learn.microsoft.com/en-us/azure/architecture/patterns/anti-corruption-layer | 2026-08-13 | Official vendor doc (Microsoft, `ms.date` 2026-05-28 -- current-year) | WebFetch full | ACL = "a facade or adapter layer between different subsystems that don't share the same semantics". Placement is explicit and directional: "Communication between subsystem A and the anti-corruption layer always uses the data model and architecture of subsystem A. Calls from the anti-corruption layer to subsystem B conform to that subsystem's data model". Two considerations that bear directly on 86.63: "**Because the anti-corruption layer mediates systems that might have different trust levels, consider enforcing input validation and sanitization at this boundary**" and "**Plan for observability, including correlation IDs and structured logging, to diagnose translation failures**". Anti-scope warning: "it's important to focus the anti-corruption layer on translation logic. **Avoid placing business rules or orchestration in the layer.**" |
| 6 | https://cwe.mitre.org/data/definitions/180.html | 2026-08-13 | Official standards body (MITRE CWE-180), year-less canonical -- **cited by `recommendation_vocab.py:38`** | WebFetch full | "The product **validates input before it is canonicalized**, which prevents the product from detecting data that becomes invalid after the canonicalization step." Mitigation: "**Inputs should be decoded and canonicalized to the application's current internal representation before being validated.**" Confirms the internal module's stated ordering is the CWE-sanctioned one -- and note the corollary: a site that hand-writes `{"SELL","STRONG_SELL"}` and tests membership on a raw string is validating WITHOUT canonicalising at all, which is the CWE-180 shape's degenerate case. |
| 7 | https://peps.python.org/pep-0661/ | 2026-08-13 | Official language doc (PEP 661), year-less canonical -- **cited by `recommendation_vocab.py:157`** | WebFetch full | Corroborates 86.25's `UNKNOWN_RECOMMENDATION` design: a distinct sentinel is needed "when it needs to be **distinct from `None`** since `None` is a valid value in that context". Requirements that map onto the repo: identity semantics ("always be considered identical to itself but never to any other object"), "a clear and short repr", and "It should be possible to use **clear type signatures** for sentinels". NOTE a real caveat for pyfinagent: PEP 661 sentinels are OBJECTS; `outcome_tracking.recommendation` is a REQUIRED BQ STRING, so the repo's string-typed `"UNKNOWN"` is a necessary deviation -- its safety rests on `"UNKNOWN" not in CANONICAL_RECOMMENDATIONS`, which is a **runtime** property, not a type-system one. |
| 8 | https://langsec.org/papers/langsec-cwes-secdev2016.pdf | 2026-08-13 | **Peer-reviewed** (Momot, Bratus, Hallberg, Patterson -- IEEE SecDev 2016), year-less canonical | `curl` + `pypdf` extract (8 pp, 49,350 chars); every quote below regex-verified against the extracted text, NOT taken from a summary | Defines the anti-pattern by name: "**Shotgun parsing** (ad-hoc validation during processing)". The one-line statement of the whole class: "**Input-driven exploitation** There is a way in which all input-driven vulnerabilities are alike and exploited alike: **invalid input is processed instead of being rejected.**" On recognisers: "a well-constructed parser must plainly follow this grammar, and **must reject non-conforming inputs without operating on them any further**." Names the multi-site failure directly: "We identify this theoretical result as a leading root cause for **parser differentials**", citing Kaminsky/Sassaman/Patterson "demonstrating over 20 of these between the different libraries used by the X.[509 PKI]". Also faults CWE-20 for being too weak: "the closest it comes to directing developers specifically to avoid writing shotgun parsers ... is to call for input canonicalization". |
| 9 | https://www.cs.ru.nl/~erikpoll/papers/2018_langsec.pdf | 2026-08-13 | **Peer-reviewed** (Erik Poll, Radboud Univ., LangSec/SPW 2018), last-2-year-agnostic canonical | `curl` + `pypdf` extract (6 pp, 32,434 chars); quotes regex-verified | The most directly transferable source in the brief, because pyfinagent's defect is a **forwarding** flaw, not a processing flaw. Poll splits input flaws into "(1) flaws in **processing** input ... (2)" flaws in **forwarding** input, and observes: "the LangSec anti-pattern of shotgun parsing is present in **forwarding flaws** ... some of the parsing **is not done in the main application but in the external back-end that it relies on**." Also: "A validation routine can simply **filter out the invalid inputs from valid ones, rejecting the invalid ones**, but it can also try to **sanitise** data" -- the reject-vs-coerce fork, named. And: "nearly all the work presented at the annual LangSec workshop focus on the first category" -- i.e. the forwarding class (pyfinagent's) is the **under-studied** one. |
| 4 | https://www.rfc-editor.org/rfc/rfc9413.html | 2026-08-13 | IETF/IAB standards-track document (RFC 9413, "Maintaining Robust Protocols"), year-less canonical | WebFetch full | **The strongest evidence in the brief, and it is a measured case, not an opinion.** Names the exact mechanism 86.63 is fighting: "An implementation that reacts to variations in the manner recommended in the robustness principle enters a **pathological feedback cycle**. Over time: Implementations progressively add logic to constrain how data is transmitted or to permit variations in what is received. Errors in implementations or confusion about semantics are permitted or ignored. **These errors can become entrenched, forcing other implementations to be tolerant of those errors.**" And: "A flaw can become **entrenched as a de facto standard**. Any implementation of the protocol is required to replicate the aberrant behavior, or it is not interoperable." Worked example: TLS ClientHello servers that broke on a trailing empty extension -- "client implementations were required to be aware of this bug". Prescription is fail-loud **with feedback to the producer**: "**Choosing to generate fatal errors for unspecified conditions instead of attempting error recovery can ensure that faults receive attention.**" "A notification for a fatal error is best sent as explicit error messages to the entity that made the error." |
| 5 | https://martinfowler.com/bliki/TolerantReader.html | 2026-08-13 | Authoritative blog (Martin Fowler), year-less canonical -- **[COUNTERPOINT]** | WebFetch full | The strongest published argument AGAINST fail-loud reading, deliberately sought: "only take the elements you need, **ignore anything you don't**"; "Your aim should be to allow the provider to make any change that ought not to break your code." BUT its own caveats cut FOR 86.63, and this is the load-bearing part: "**make sure there's only one bit of code that reads data payloads**" -- isolate the tolerance behind a single DTO -- and providers "should receive reader code and tests to detect actual breakages". So even the tolerant-reader position demands ONE reader, not a re-inlined predicate per site. See "Consensus vs debate" for how the two reconcile. |
| 3 | https://docs.aws.amazon.com/prescriptive-guidance/latest/cloud-design-patterns/acl.html | 2026-08-13 | Official vendor doc (AWS Prescriptive Guidance, year-less canonical) | WebFetch full | "The ACL pattern acts as a mediation layer that translates domain model semantics from one system to another system." Its worked sample code FAILS LOUD on an unmappable token rather than coercing -- `if (Int32.TryParse(...)) {...} else { Console.WriteLine("String could not be parsed."); return HttpStatusCode.BadRequest; }`. Considerations name "**Single point of failure:** Any failures in the ACL can make the target service unreachable" and "**Service-specific or shared implementation:** You can design ACL as a shared object ... or service-specific classes" -- i.e. the shared-vs-per-site choice is an explicit design axis, not an accident. |

**Fetch-method disclosure (do not skip when auditing this count).** 10 sources were read in
full. **8 via `WebFetch`** (#1-7, #10) and **2 via `curl` + `pypdf` text extraction** (#8, #9),
which is the sanctioned PDF chain in `.claude/rules/research-gate.md` step 3. Under the very
strictest reading -- "only a full WebFetch counts" -- the floor of 5 is still cleared with 8.
The pypdf route was chosen deliberately: auto-memory
`reference_webfetch_pdf_summaries_fabricate_quotes` records that WebFetch PDF summaries have
fabricated quotes TWICE on this project, so every quote from #8/#9 was **regex-verified against
the extracted text**, not lifted from a summary.

## Identified but snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|-----|------|-------------------------|
| https://deviq.com/principles/parse-dont-validate/ | Community wiki | Restates #1 with less depth |
| https://github.com/andrewbanchich/parse_dont_validate_rs | Code repo | Rust port, no new argument |
| https://medium.com/@miggo-engineering/parse-dont-validate-in-practice-4b1a10177759 | Blog | Community tier, superseded by #1 |
| https://lobste.rs/s/uon7sc/parse_don_t_validate_2019 | Forum | Lowest tier |
| https://cekrem.github.io/posts/parse-dont-validate-typescript/ | Blog | TS-specific; relevant to the frontend half but community tier |
| https://kb.evryg.com/en/advanced-software-engineering/type-design/parse-dont-validate | Wiki | Derivative of #1 |
| https://rednegra.net/blog/20250810-parse-dont-validate/ | Blog (2025) | Recency-scan hit; derivative of #1 |
| https://github.com/eclipse-langium/langium/discussions/1777 | Forum (2025) | Recency-scan hit; no measurements |
| https://deviq.com/domain-driven-design/anti-corruption-layer/ | Community wiki | Superseded by #2/#3 |
| https://ddd-practitioners.com/.../anticorruption-layer/ | Community wiki | Superseded by #2/#3 |
| https://hosseinnejati.medium.com/the-anti-corruption-layer-protecting-your-domain-from-legacy-systems-6da58fc5f462 | Blog | Community tier |
| https://oneuptime.com/blog/post/2026-01-30-anti-corruption-layer-pattern/view | Blog (2026-01) | Recency-scan hit; opinion, no measurements |
| https://codeopinion.com/anti-corruption-layer-for-mapping-between-boundaries/ | Blog | Community tier |
| https://arxiv.org/pdf/2310.01905 | **Peer-reviewed SLR** (DDD systematic literature review) | Strong candidate; budget-capped after #10 already covered the boundary-placement question with measurements |
| https://langsec.org/brucon/ShotgunParsersBruCON.pdf | Conference slides (Bratus/Patterson) | Slide deck; #8 is the paper form |
| http://langsec.org/ShotgunParsersShmoo.pdf | Conference slides | Same |
| https://www.usenix.org/system/files/login/articles/login_spring17_08_bratus.pdf | USENIX ;login: | Overlaps #8 |
| http://spw16.langsec.org/papers/underwood-android-shotgun-parsers.pdf | Peer-reviewed (SPW16) | Android-specific corpus study; adjacent |
| https://www.semanticscholar.org/paper/The-Seven-Turrets-of-Babel...ae4e54c65d5139c21b2a9499d1f24c7e3e14af05 | Index page | Metadata for #8 |
| https://www.manning.com/books/secure-by-design | Book (paywalled) | Domain-primitives source; not fetchable in full |
| https://livebook.manning.com/book/secure-by-design/chapter-5 | Book ch. 5 (paywalled) | "Domain primitives" chapter; paywall |
| https://freecontent.manning.com/domain-primitives-what-they-are-and-how-you-can-use-them-to-make-more-secure-software/ | Publisher excerpt | Industry tier; #10 covers the same ground with data |
| https://hj.diva-portal.org/smash/get/diva2:1892374/FULLTEXT01.pdf | Thesis | "From Insecure to Secure by Design Using Domain Primitives"; student thesis, lower weight |
| https://owasp.org/www-project-secure-by-design-framework/ | Official (OWASP) | Process framework, not a boundary-placement finding |
| https://functional-architecture.org/make_illegal_states_unrepresentable/ | Blog | Recency-scan hit; no measurements |
| https://deviq.com/principles/make-illegal-states-unrepresentable/ | Community wiki | Same |
| https://arxiv.org/pdf/1811.00820 | Peer-reviewed | "Too Trivial To Test?" -- defect prediction; off-topic on inspection |
| https://www.baeldung.com/cs/enums-vs-constants | Community | Enum-vs-constant; opinion only, no measurements |
| https://blog.logrocket.com/typescript-enum/ | Blog | TS enums; community tier |

**URLs collected: 39** (10 read in full + 29 snippet-only).

## Recency scan (2024-2026)

**Performed.** Queries run (the three-variant discipline is visible in the tables above --
the read-in-full set mixes year-less canonical (#1, #2 rev-2026, #3, #4, #5, #6, #7),
last-2-year (#9 is 2018 but the SPW line continues; #8 is 2016) and current-year (#10, 2026)):

1. *Current-year / last-2-year frontier:* `"secure by design domain primitives validation at construction boundary 2025 2026"`; `"2025 2026 research validation at write boundary versus read boundary schema enforcement measured study"`; `"parse don't validate" OR "make illegal states unrepresentable" 2026 production adoption results`; `"empirical study string constants versus enum type safety defects measured"`.
2. *Year-less canonical:* `"parse don't validate boundary parsing type-driven design"`; `"anti-corruption layer bounded context vocabulary translation domain-driven design"`; `"LangSec shotgun parsing input handling failures Sassaman Bratus Patterson paper"`.

**Result: ONE genuinely new finding in the 2024-2026 window, and it is the single most
transferable source in this brief** -- **arXiv:2607.01711v1 (2026), "Trust Boundary Semantic
Gaps"** (#10). It does not supersede the canonical sources; it **complements them with
measurements they lack**, and it answers the caller's explicit "write side vs read side"
question with data rather than opinion. Details in Key Findings #4.

Two negative results worth recording, because a null is a finding:

- **No empirical study measuring string-constant vs enum defect rates was found.** The only
  quantitative claim surfaced was an unattributed "over 40% fewer bugs" figure in a search
  snippet with no locatable primary source. **I did not use it and it is not cited above.**
- **No 2024-2026 work supersedes "parse, don't validate" (2019) or RFC 9413 (2023).** The
  2025-2026 hits (`rednegra.net`, `oneuptime.com`, `functional-architecture.org`, the Langium
  discussion) are restatements without new measurements.

## Key findings

1. **Parse at the boundary, once, and let the TYPE carry the proof.** "Get your data into the
   most precise representation you need as quickly as you can. Ideally, this should happen at
   the boundary of your system, **before any of the data is acted upon**" (King 2019,
   https://lexi-lambda.github.io/blog/2019/11/05/parse-don-t-validate/, accessed 2026-08-13).
   The failure mode of *not* doing this is named: scattered validators "allow inconsistency:
   modifications in one place needn't update others, enabling bugs to slip past undetected."
   **This is a literal description of `portfolio_manager.py:60-64`** -- a caller that unwrapped
   the shared sets back into hand-written literals, exactly as `recommendation_vocab.py:104-105`
   predicted it would.

2. **Silent tolerance of an unknown token is not neutral -- it entrenches the producer's
   defect.** "An implementation that reacts to variations in the manner recommended in the
   robustness principle enters a **pathological feedback cycle** ... These errors can become
   entrenched, forcing other implementations to be tolerant of those errors" and "A flaw can
   become entrenched as a **de facto standard**" (IAB, RFC 9413,
   https://www.rfc-editor.org/rfc/rfc9413.html, accessed 2026-08-13). The prescription is
   explicitly fail-loud **with feedback to the producer**: "Choosing to generate fatal errors
   for unspecified conditions instead of attempting error recovery **can ensure that faults
   receive attention**." This is the strongest published warrant for the caller's constraint
   that the phase-86.20 `UNRECOGNISED` log line must stay loud -- **its loudness is the feedback
   channel RFC 9413 requires**, and quieting it would restart the cycle.

3. **The one-line statement of the whole defect class.** "There is a way in which all
   input-driven vulnerabilities are alike and exploited alike: **invalid input is processed
   instead of being rejected**"; a well-formed recogniser "must **reject non-conforming inputs
   without operating on them any further**" (Momot, Bratus, Hallberg, Patterson, IEEE SecDev
   2016, https://langsec.org/papers/langsec-cwes-secdev2016.pdf, accessed 2026-08-13). The
   paper also names what happens when several sites each re-derive the rule: **parser
   differentials** -- Kaminsky/Sassaman/Patterson "demonstrating over 20 of these between the
   different libraries used by the X.[509 PKI]". `_BUY_RECS` at `portfolio_manager.py:64` vs
   `BUY_INTENT` at `recommendation_vocab.py:111` is a parser differential in miniature: two
   definitions of the same predicate that agree today by coincidence, not by construction.

4. **WRITE side vs READ side -- the measured answer, and it favours the WRITE side.** The 2026
   TBSG study (https://arxiv.org/html/2607.01711v1, accessed 2026-08-13) analysed **75 publicly
   reported incidents from 2014-2025**, keeping only cases where "the receiving domain accepted
   it after **syntactic validation passed**" -- i.e. precisely pyfinagent's shape, where a
   `str` is a perfectly valid `str` and the damage is semantic. Measured dimension prevalence:
   Identity 64/75 (85.3%), Spatial 62/75 (82.7%), **Interpretation 59/75 (78.7%)**, Temporal
   40/75 (53.3%). The mitigation is stated as a placement rule: **"placing controls at the
   boundary where gaps originate, not merely where they become visible"**, with a worked
   example -- for SUNBURST, attestation belongs "at the build-output boundary (TB2), not
   repeated at customer validation (TB3)". **Mapped to 86.63: the gap ORIGINATES at
   `paper_trader.py:452` (`_pos_rec = reason`) and becomes VISIBLE at `portfolio_manager.py:264`
   (the dead `signal_downgrade`). The control belongs at the write.** The paper's "P1/P2/P3
   prioritization to identify root gaps versus propagated ones" is the same distinction the
   masterplan has been re-discovering one step at a time.

5. **Forwarding flaws, not processing flaws -- and they are the under-studied half.** Poll
   splits input flaws into "(1) flaws in **processing** input" and (2) flaws in **forwarding**
   it, and notes "the LangSec anti-pattern of shotgun parsing is present in **forwarding
   flaws** ... some of the parsing is **not done in the main application but in the external
   back-end** that it relies on" (https://www.cs.ru.nl/~erikpoll/papers/2018_langsec.pdf,
   accessed 2026-08-13). pyfinagent's defect is squarely category (2): `paper_trader` forwards
   an order-reason string to BigQuery, and the "parsing" happens later, in whichever consumer
   happens to test it. Poll also names the exact fork the caller asked about: "A validation
   routine can simply **filter out the invalid inputs from valid ones, rejecting the invalid
   ones**, but it can also try to **sanitise** data" -- reject vs coerce. And "nearly all the
   work presented at the annual LangSec workshop focus on the first category", so the class
   pyfinagent is in has **less prior art than it looks** -- a reason to lean on the measured
   TBSG placement rule rather than on folklore.

6. **A single translation point is the ACL's defining property -- and it is a stated design
   axis, not an accident.** AWS lists "**Service-specific or shared implementation:** You can
   design ACL as a shared object to convert and redirect calls to multiple services **or**
   service-specific classes" (https://docs.aws.amazon.com/prescriptive-guidance/latest/cloud-design-patterns/acl.html,
   accessed 2026-08-13); its worked sample **fails loud on an unmappable token** rather than
   coercing (`else { ... return HttpStatusCode.BadRequest; }`). Microsoft adds the two
   properties 86.63 needs most: "consider **enforcing input validation and sanitization at this
   boundary**" and "**Plan for observability** ... to diagnose translation failures"
   (https://learn.microsoft.com/en-us/azure/architecture/patterns/anti-corruption-layer,
   accessed 2026-08-13). Microsoft also draws the scope line the caller will want enforced:
   "focus the anti-corruption layer on **translation logic. Avoid placing business rules or
   orchestration in the layer.**" -- i.e. the guard decides *parses / does not parse*, never
   *buy / sell*.

7. **Canonicalise BEFORE validating -- and a hand-written literal set does neither.**
   "Inputs should be **decoded and canonicalized** to the application's current internal
   representation **before being validated**" (MITRE CWE-180,
   https://cwe.mitre.org/data/definitions/180.html, accessed 2026-08-13). The internal module
   already cites this at `recommendation_vocab.py:38` and complies. `portfolio_manager.py:254`
   (`rec in _SELL_RECS`) is the degenerate case: it validates against a literal set with **no
   canonicalisation step in the same expression** -- correct today only because `_resolve_rec`
   happens to have canonicalised upstream, and only when the flag is ON.

8. **A sentinel must be provably outside the domain, and PEP 661's guarantees are weaker here
   than they look.** PEP 661 (https://peps.python.org/pep-0661/, accessed 2026-08-13) requires
   a sentinel that "should always be considered identical to itself but **never to any other
   object**" and that supports "clear type signatures". **The repo cannot have that**:
   `outcome_tracking.recommendation` is `STRING, mode="REQUIRED"`
   (`scripts/migrations/migrate_bq_schema.py:126`), so the sentinel must be a *string*. Its
   safety therefore rests on the **runtime** property `"UNKNOWN" not in
   CANONICAL_RECOMMENDATIONS`, not on a type. That is a real, load-bearing caveat for any new
   guard that adds another sentinel -- and see Pitfall P5, because `paper_positions` differs.

## Internal code inventory

### The enumeration command (pin the binary -- the shell `grep` is a ugrep shim)

`which grep` resolves to a **shell function** that re-execs the Claude binary as `ugrep`
(verified this session; `/usr/bin/grep --version` -> `grep (BSD grep, GNU compatible) 2.6.0-FreeBSD`).
Every command below therefore calls `/usr/bin/grep` by absolute path. The ugrep shim also
applies `--ignore-files` (i.e. `.gitignore` semantics), so it and BSD grep can legitimately
return **different denominators** -- which is exactly the failure mode this audit is about.

```bash
cd /Users/ford/.openclaw/workspace/pyfinagent

# E1 -- distinct rec-ish FIELD NAMES appearing as quoted keys
/usr/bin/grep -rnoE '"(recommendation|analyst_recommendation|analysis_recommendation|consensus|rec|verdict|decision|risk_judge_decision|action|signal|direction)"' \
  --include='*.py' backend/ | /usr/bin/grep -v '/tests\?/' | awk -F: '{print $NF}' | sort | uniq -c | sort -rn

# E2 -- WRITE seams (a value assigned INTO a rec-ish field)
/usr/bin/grep -rnE '("(recommendation|analyst_recommendation|analysis_recommendation|consensus|rec)"[[:space:]]*:)|(\.(recommendation|analyst_recommendation|analysis_recommendation|consensus)[[:space:]]*=[^=])|((^|[^_a-zA-Z])(recommendation|analyst_recommendation|analysis_recommendation)[[:space:]]*=[^=])' \
  --include='*.py' backend/ | /usr/bin/grep -vE '/tests?/'

# E3 -- READ seams that go THROUGH the vocab module
/usr/bin/grep -rnE '(_BUY_RECS|_SELL_RECS|_DOWNGRADE_RECS|BUY_INTENT|SELL_INTENT|is_buy_intent|is_sell_intent|is_directional|canonical_recommendation|is_recognised|resolve_outcome_recommendation|CANONICAL_RECOMMENDATIONS|UNKNOWN_RECOMMENDATION)' \
  --include='*.py' backend/ scripts/ | /usr/bin/grep -vE '/tests?/'

# E3b -- HAND-WRITTEN rec literals that BYPASS the vocab module (incl. TypeScript)
/usr/bin/grep -rnE '"(STRONG_BUY|STRONG BUY|Strong Buy|STRONG_SELL|STRONG SELL|Strong Sell)"' \
  --include='*.py' --include='*.ts' --include='*.tsx' backend/ frontend/src/ scripts/ \
  | /usr/bin/grep -vE '/tests?/|\.test\.|__tests__'
```

**Positive control for E1-E3** (a zero must be paired with a control): the four seams the
caller named were asserted to appear, by printing the literal line at each anchor.
All four resolved and all four matched what the caller stated -- `portfolio_manager.py:16`
= `from backend.services.recommendation_vocab import canonical_recommendation  # phase-86.20`;
`:60` = `_SELL_RECS = {"SELL", "STRONG_SELL"}`; `paper_trader.py:452` = `_pos_rec = reason`;
`bigquery_client.py:638` = `if "ticker" not in row:`. **No caller-supplied anchor was stale
in this brief** (a departure from 86.58, where `:127` had drifted to `:264` -- see below).

### E1 result -- field-name frequency (backend/, non-test)

| Field name (as a quoted key) | Occurrences | Vocabulary it is supposed to carry |
|---|---|---|
| `"signal"` | 144 | signal-attribution, NOT a recommendation |
| `"action"` | 60 | **trade action** (BUY/SELL) -- a DIFFERENT closed set |
| `"recommendation"` | 57 | the analyst scale |
| `"decision"` | 36 | mixed |
| `"verdict"` | 21 | Q/A + risk vocabularies |
| `"consensus"` | 10 | the analyst scale, underscored `Literal` |
| `"risk_judge_decision"` | 8 | **approval** vocabulary (APPROVE_REDUCED/REJECT/...) |
| `"analyst_recommendation"` | 2 | the analyst scale |
| `"analysis_recommendation"` | 2 | the analyst scale |
| `"rec"` | 1 | the analyst scale |

### The count is NOT five. Derived denominators

**Filed masterplan steps in this class = SIX, not five** (`.claude/masterplan.json`, read via
`json.load` + id filter). The caller's prompt lists 86.22 / 86.25 / 86.40 / 86.52 / 86.58 and
omits **86.20**, which is the founding instance and the one that created
`recommendation_vocab.py`. Statuses as of 2026-08-13: 86.20 `done`, 86.22 `done`,
86.25 `done`, 86.40 `pending`, 86.52 `pending`, 86.58 `done`, 86.63 `pending`.
So 86.63 is the **seventh** filed member, not the sixth.

**Code seams are ~37 write sites + ~25 read sites**, spanning **Python AND TypeScript** --
the surface is materially larger than six steps implies, and the TypeScript half is
structurally unreachable from `recommendation_vocab.py`.

### WRITE seams -- every producer that puts a string into a rec-ish field

| # | File:line | What it writes | Vocabulary actually written | Guarded? |
|---|---|---|---|---|
| W1 | `backend/services/paper_trader.py:452` | `_pos_rec = reason` | **order-reason** (`new_buy_signal`, `swap_buy`) | **NO** -- the named seam |
| W2 | `backend/services/paper_trader.py:457` | `_pos_rec = analysis_recommendation` | analyst scale | flag-gated only |
| W3 | `backend/services/paper_trader.py:488` | `"recommendation": _pos_rec` (add-on lot) | whichever of W1/W2 won | **NO** |
| W4 | `backend/services/paper_trader.py:512` | `"recommendation": _pos_rec` (new lot) | whichever of W1/W2 won | **NO** |
| W5 | `backend/services/paper_trader.py:676` | `"recommendation": position.get("recommendation","")` | re-emits whatever W3/W4 stored | **NO** |
| W6 | `backend/services/autonomous_loop.py:2172` | `"recommendation": rec.get("action","HOLD")` | **ACTION** vocabulary | **NO** |
| W7 | `backend/services/autonomous_loop.py:3119` | `"recommendation": analysis["action"]` | **ACTION** vocabulary | **NO** |
| W8 | `backend/services/autonomous_loop.py:3355` | `"recommendation": analysis["action"]` | **ACTION** vocabulary | **NO** |
| W9 | `backend/services/autonomous_loop.py:3427` | `... or "Hold"` | analyst scale in **TITLE CASE** | **NO** |
| W10 | `backend/services/autonomous_loop.py:2351` | `"recommendation": "HOLD"` (lite path) | analyst scale, UPPER | **NO** |
| W11 | `backend/services/autonomous_loop.py:2254` | `"recommendation": None` | absent | **NO** |
| W12 | `backend/services/autonomous_loop.py:3554` | `resolve_outcome_recommendation(...)` | canonical or `UNKNOWN` | **YES** (86.25) |
| W13 | `backend/tasks/analysis.py:214` | `recommendation=rec_obj.get("action","N/A")` | **ACTION**, default `"N/A"` | **NO** |
| W14 | `backend/api/analysis.py:214` | byte-identical duplicate of W13 | **ACTION**, default `"N/A"` | **NO** |
| W15 | `backend/agents/orchestrator.py:2386` | `final_json["recommendation"]["recommendation"]` default `"HOLD"` | LLM-emitted, unvalidated | **NO** |
| W16 | `backend/agents/debate.py:347` / `orchestrator.py:2197` | `"consensus": "HOLD"` | analyst scale (fallback) | **NO** |
| W17 | `backend/agents/risk_debate.py:220` | `"consensus": debate_result.get("consensus")` | pass-through | **NO** |
| W18 | `backend/agents/compaction.py:157,160` | re-nests `recommendation` | pass-through | **NO** |
| W19 | `backend/api/portfolio.py:83` | `"recommendation": body.recommendation` | **request body** (external!) | **NO** |
| W20 | `backend/db/bigquery_client.py:163,:407` | `"recommendation": recommendation` | pass-through to BQ | **NO** |
| W21 | `backend/services/portfolio_manager.py:383` | `"recommendation": rec` | output of `_resolve_rec` | partial |
| W22 | `backend/services/portfolio_manager.py:578,:918` | `analysis_recommendation=cand.get("recommendation","")` | pass-through | **NO** |
| W23 | `backend/slack_bot/jobs/nightly_outcome_rebuild.py:111` | `resolve_outcome_recommendation(...)` | canonical or `UNKNOWN` | **YES** (86.25) |
| W24 | `backend/slack_bot/jobs/_production_fns.py:406` | `"recommendation": outcome.get("recommendation")` | pass-through | **NO** |
| W25 | `backend/services/outcome_tracker.py:71,:84,:147,:168` | `recommendation=` pass-through | pass-through | **NO** |

**2 of ~25 write seams are guarded** (W12, W23 -- both are `resolve_outcome_recommendation`,
both added by 86.25). Every other producer writes an unparsed string.

### READ seams -- who tests a rec value, and against which vocabulary

| # | File:line | Vocabulary tested against | Via the module? |
|---|---|---|---|
| R1 | `portfolio_manager.py:254` `rec in _SELL_RECS` | hand-written `{"SELL","STRONG_SELL"}` at `:60` | **NO -- re-inlined** |
| R2 | `portfolio_manager.py:264` `old_rec in _BUY_RECS and rec in _DOWNGRADE_RECS` | hand-written at `:62`/`:64` | **NO -- re-inlined** |
| R3 | `portfolio_manager.py:304` `rec not in _BUY_RECS` | hand-written at `:64` | **NO -- re-inlined** |
| R4 | `outcome_tracker.py:64-65` | `is_buy_intent` / `is_sell_intent` | YES (86.22) |
| R5 | `agents/memory.py:235-236` | `is_buy_intent` / `is_sell_intent` | YES (86.22) |
| R6 | `agents/bias_detector.py:122,131,157,158` | `is_buy_intent` / `is_sell_intent` | YES (86.22) |
| R7 | `agents/skill_optimizer.py:263-264` | `is_buy_intent` / `is_sell_intent` | YES (86.22) |
| R8 | `agents/conflict_detector.py:120` | `canonical_recommendation` | YES (86.22) |
| R9 | `api/portfolio.py:145,147` | `is_directional` / `is_buy_intent` | YES (86.22) |
| R10 | `slack_bot/formatters.py:189` | `canonical_recommendation` | YES (86.22) |
| R11 | `agents/bias_detector.py:21-25` | dict keyed `"STRONG_BUY"..."STRONG_SELL"` (underscored literals) | **NO -- re-inlined** |

### The TWO live enums nothing maps between (the root vocabulary split)

- `backend/api/models.py:22,:26` -- `Recommendation.STRONG_BUY = "Strong Buy"`, `STRONG_SELL = "Strong Sell"` (**SPACED TITLE CASE**)
- `backend/agents/schemas.py:95` -- `consensus: Literal["STRONG_BUY","BUY","HOLD","SELL","STRONG_SELL"]` (**UNDERSCORED UPPER**)

Both are first-class and both are live. `recommendation_vocab.py:12-17` already says this and
says it "is meant to be the ONLY one".

### The surface `recommendation_vocab.py` structurally CANNOT reach: the frontend

A Python-side boundary guard cannot cover these, and three of them use the exact **substring**
shape that `recommendation_vocab.py:32-36` files as a defect class:

| File:line | Shape | Note |
|---|---|---|
| `frontend/src/components/DebateView.tsx:79,81` | `c.includes("STRONG_BUY") \|\| c.includes("BUY")` | **substring** -- the R3 shape |
| `frontend/src/components/RecentReportsTable.tsx:34,36` | `r.includes("STRONG_BUY") \|\| r.includes("STRONG BUY")` | **substring**, both dialects hand-listed |
| `frontend/src/components/reports-columns.tsx:16-17` | `norm === "STRONG BUY" \|\| norm === "BUY"` | exact-match, **SPACED dialect only** |
| `frontend/src/components/ReportCompareDrawer.tsx:20-21` | `norm === "STRONG BUY" \|\| norm === "BUY"` | duplicate of the above |

### THE HEADLINE FINDING (round 2-4): the split is authored UPSTREAM, in the LLM prompts

The two dialects are not an accident of Python code -- **they are instructed, in two different
skill prompt files**, and neither prompt is reachable by any Python-side guard:

- `backend/agents/skills/synthesis_agent.md:19` -- "Recommendation values: **Strong Buy / Buy / Hold / Sell / Strong Sell**" (SPACED)
- `backend/agents/skills/moderator_agent.md:18,:101` -- "Consensus values: **STRONG_BUY / BUY / HOLD / SELL / STRONG_SELL**" (UNDERSCORED)

Worse, `synthesis_agent.md` puts the RECOMMENDATION scale into a field literally named
**`action`**:

- `synthesis_agent.md:82` -- `"action": "<Strong Buy|Buy|Hold|Sell|Strong Sell>"`
- `synthesis_agent.md:163` -- `"action": "<string, one of: 'Strong Buy', 'Buy', 'Hold', 'Sell', 'Strong Sell'>"`

**That is the field conflation at its origin**, and it is why so many Python sites can write
`"recommendation": analysis["action"]` (`autonomous_loop.py:3119`, `:3355`) and look locally
reasonable.

### A SOURCE-LEVEL CONTRADICTION that 86.63 must DRIVE, not assume

`backend/services/autonomous_loop.py:2514` is a structural gate:

```python
if analysis.get("action") not in ("BUY", "SELL", "HOLD"):
    return False
```

That set is the **trade-action** vocabulary. **None of the five values `synthesis_agent.md`
instructs** (`Strong Buy`, `Buy`, `Hold`, `Sell`, `Strong Sell`) is a member of
`("BUY","SELL","HOLD")` -- not even by case. So as written, a synthesis-shaped analysis fails
this gate outright.

**I am deliberately NOT claiming that is the live behaviour.** I could not locate a call site
for the enclosing function by name in this session (round 4a returned zero, and I have no
positive control proving that grep could have found it, so treat 4a as UNRESOLVED, not as a
zero). Which producer actually fills `analysis["action"]` at that call site is a **behavioural**
question and must be settled by DRIVING the code, not by reading it. Recorded here as the
single highest-value open question for the contract.

### Additional seams found in rounds 3-4 (the loop was not dry until round 5)

| Seam | Finding |
|---|---|
| `autonomous_loop.py:2630` | `rec = str(a.get("recommendation") or a.get("action") or "")` -- **the two vocabularies are merged into one variable by a fallback chain**, then `str()`-coerced. A read-side seam that no write-side guard can catch. |
| `autonomous_loop.py:3028`, `:3305` | `trader_action=analysis["action"]` -- the SAME value that feeds `"recommendation"` at `:3119`/`:3355` also feeds a field named `trader_action`. One value, two field names, two assumed vocabularies. |
| `backend/tasks/analysis.py:205-220` vs `backend/api/analysis.py:205-220` | **BYTE-IDENTICAL** (verified with `/usr/bin/diff`, exit 0). The `recommendation=rec_obj.get("action","N/A")` defect exists in duplicate. Fixing one and not the other is a live risk for this step. |
| `frontend/src/lib/types.ts:126,:152,:283,:653` | `recommendation: string \| null`, `recommendation?: string`, `consensus: string` -- **the TypeScript boundary is `string`, not a union**. TS provides ZERO vocabulary enforcement today. |
| `backend/api/models.py:21` `Recommendation(str, Enum)` | Consumed only by `RecommendationDetail` at `:39`/`:46` within the same file. The SPACED dialect is a near-dead enum as an *enforcement* mechanism, even though it is the dialect the producer prompt emits. |
| `bigquery_client.py:663` | `params.append(... "STRING", str(v))` -- `save_paper_position` **`str()`-coerces every non-float/int value**. A non-string recommendation is silently stringified at the persistence boundary; `canonical_recommendation` deliberately refuses to do this (`recommendation_vocab.py:80-84`). |
| `backend/agents/bias_detector.py:21-25` | A dict keyed by underscored literals (`"STRONG_BUY": 0.08 ... "STRONG_SELL": 0.07`) -- a **re-inlined** membership set surviving inside a file that otherwise imports the shared predicates at `:15`. Exactly the "unwrapped back into a literal set" failure `recommendation_vocab.py:104-105` predicts. |

### Round 3a: there is NO rival canonicaliser (a genuine negative, with control)

A sweep for `def *(canonical|normalise|normalize|_rec|to_rec|parse_rec)*` across `backend/` +
`scripts/` returned **no second recommendation canonicaliser**. The only rec-adjacent
definitions are `portfolio_manager.py:74 _resolve_rec` (a wrapper that CALLS the module) and
`slack_bot/formatters.py:175 _rec_color` (which calls `canonical_recommendation` at `:189`).
Positive control: the same command found `def canonical_recommendation` (count = 1), so the
grep could have found a rival had one existed. **`recommendation_vocab.py` really is the only
canonicaliser** -- the problem is coverage, not competition.

### Existing test coverage a new guard must not merely duplicate

`backend/tests/test_phase_86_20_recommendation_vocabulary.py`,
`test_phase_86_20_portfolio_manager_recommendation_vocabulary.py`,
`test_phase_86_22_outcome_tracker_bias_detector_vocabulary.py`,
`test_phase_86_25_outcome_tracker_vocabulary_boundary.py`,
`test_phase_35_1_learn_loop_writer.py`; plus mutation matrices
`scripts/qa/mutation_matrix_86_20.py`, `mutation_matrix_86_22.py` and the driver
`scripts/qa/drive_86_58_dead_downgrade.py`.

### Disagreement with the 86.58 brief: NONE on substance, ONE addition

The caller's summary of 86.58 is confirmed line-by-line against the source this session
(all four anchors printed above). One thing to add: `paper_trader.py:452` is the seam, but
the value it writes is consumed **three lines apart at `:488` and `:512`** and then
**re-emitted at `:676`** -- so a guard placed at `:452` alone still leaves `:676` able to
launder a stored bad value back out. A guard at `save_paper_position` covers `:488`/`:512`
but NOT `:676` (which is a trade row, not a position row).

## Consensus vs debate (external)

**Consensus (5 of 5 non-adversarial sources agree):** translation between two vocabularies
belongs at **one** place, that place should be **as close to the origin as possible**, and an
**unmappable token should be rejected, not guessed**. King, RFC 9413, LangSec/SecDev, Poll and
the 2026 TBSG study reach this from five independent directions (type theory, protocol
maintenance, formal-language security, input-flaw taxonomy, incident forensics).

**The debate, stated fairly.** Fowler's *TolerantReader*
(https://martinfowler.com/bliki/TolerantReader.html, accessed 2026-08-13) is the strongest
published position AGAINST: "only take the elements you need, **ignore anything you don't**",
because "Your aim should be to allow the provider to make any change that ought not to break
your code." RFC 9413 is a direct rebuttal at the protocol layer -- that tolerance is what
starts the entrenchment cycle.

**How they reconcile, and it matters for the design.** Fowler is arguing about **unknown
FIELDS** (schema evolution -- a new key appearing); RFC 9413 and LangSec are arguing about
**unknown VALUES in a known field** (a closed scale). 86.63 is unambiguously the second.
Decisive for this step: **even Fowler's own caveat lands on 86.63's side** -- "make sure
there's **only one bit of code that reads data payloads**", isolated behind a single DTO. Both
sides of the debate therefore agree that the predicate must live in ONE place. **What they
disagree about is only whether an unrecognised value should be dropped quietly or reported
loudly** -- and there, the money path plus the caller's constraint plus RFC 9413's measured
entrenchment argument all point the same way: loudly.

**One honest limit on the external evidence.** None of these sources measured *this* question
in *this* setting. The TBSG study is the only one with a quantified corpus, and it is a
**security-incident** corpus, not a trading-system one; its 78.7% Interpretation figure
describes where incidents cluster, not the probability that pyfinagent's next vocabulary drift
causes a loss. Treat it as a **placement** argument, not a risk estimate.

## Pitfalls (from literature + this repo's own history)

- **P1 -- A boundary that accepts the wrong vocabulary is a boundary callers can misuse.**
  `recommendation_vocab.py:196-202` already states this: `resolve_outcome_recommendation`
  "deliberately does not accept a `risk_judge_decision` or an `action` argument". A new write
  guard must take **only** a recommendation candidate. Adding a `reason=` parameter "for
  convenience" would rebuild the 86.25 defect.
- **P2 -- The guard will be re-inlined.** That is not a hypothetical; it is what happened.
  `portfolio_manager.py:60-64` and `bias_detector.py:21-25` both hold hand-written literal sets
  in files that ALSO import the shared module. Any 86.63 design needs a **mechanical check that
  no new literal set appears**, or P2 recurs as 86.7x.
- **P3 -- Microsoft's scope warning.** "Avoid placing business rules or orchestration in the
  layer." A guard that decides *whether a value parses* is in scope; one that decides *whether
  to trade* is not.
- **P4 -- A guard at `save_paper_position` does not cover `paper_trader.py:676`.** That line
  re-emits `position.get("recommendation","")` onto a **trade** row, not a position row, so it
  bypasses a position-write precondition entirely. Scope the guard to the class, not to one
  function -- and derive the class, per auto-memory `feedback_count_the_class_not_your_list`.
- **P5 -- `paper_positions.recommendation` is NULLABLE; `outcome_tracking.recommendation` is
  REQUIRED.** `scripts/migrations/migrate_paper_trading.py:51` = `mode="NULLABLE"`;
  `scripts/migrations/migrate_bq_schema.py:126` = `mode="REQUIRED"`. **86.25's core argument
  ("SQL NULL is unavailable, so the marker must be a string") DOES NOT TRANSFER to
  `paper_positions`** -- a real NULL is available there. Copying the `"UNKNOWN"` string
  sentinel across by analogy would be reasoning from the wrong table.
- **P6 -- `bigquery_client.py:663` `str()`-coerces every non-numeric value.** The persistence
  layer will happily stringify a dict or an enum member into a plausible-looking token --
  precisely what `canonical_recommendation` refuses to do at `recommendation_vocab.py:80-84`.
  A guard placed *above* this line is undone by it if it only checks `isinstance(v, str)`.
- **P7 -- The mutation-test trap this repo keeps hitting.** Per
  `feedback_mutation_probe_must_discriminate` and `feedback_negatives_must_reach_the_second_condition`:
  a fail-loud guard's negative cases must reach the *second* condition, and a disabled-state
  ("provably inert") proof needs an **oracle comparison of outcomes**, not a source diff --
  the `_resolve_rec` legacy-parity oracle at `portfolio_manager.py:97-106` is the pattern that
  already works here, including the "a raised AttributeError must match a raised
  AttributeError" detail.
- **P8 -- Do not treat round 4a as a zero.** I could not find a call site for the function
  enclosing `autonomous_loop.py:2514` and I have **no positive control** proving the search
  could have found one. Per `feedback_suspect_the_clean_check`, that is UNRESOLVED, not absent.

## Application to pyfinagent

**The evidence converges on placing ONE guard on the WRITE side, at the origin, failing loud.**

| External finding | pyfinagent anchor |
|---|---|
| Control belongs "at the boundary where gaps originate, not merely where they become visible" (TBSG 2026, 78.7% Interpretation over 75 incidents) | Origin = `backend/services/paper_trader.py:452` `_pos_rec = reason`. Visible = `backend/services/portfolio_manager.py:264` dead `signal_downgrade`. **Guard the write.** |
| "invalid input is processed instead of being rejected" (SecDev 2016) | `backend/db/bigquery_client.py:663` `str(v)` -- the last point at which an unparsed token becomes a persisted fact |
| Fail loud + feed the producer back (RFC 9413) | The existing `phase-86.20: UNRECOGNISED` warning at `backend/services/portfolio_manager.py:132-137` **is** that feedback channel. Caller's constraint to keep it loud is externally warranted, not just stylistic. |
| Precedent for a boundary precondition at exactly this seam | `backend/db/bigquery_client.py:638-639` already raises `ValueError` on a missing `ticker` for the MERGE key -- same function, same shape, already accepted in this codebase |
| "only one bit of code that reads data payloads" (even the tolerant-reader side) | `backend/services/recommendation_vocab.py` exists and is provably the **only** canonicaliser (round 3a, with positive control). The work is COVERAGE, not a new module. |
| ACL = translation only, no business rules | The guard answers *parses / does not parse*. `is_buy_intent` etc. stay where they are. |

**Highest-value target, derived not assumed:** of ~25 write seams, **2 are guarded** (both
`resolve_outcome_recommendation`, both from 86.25). The single seam that reaches the money path
is `paper_trader.py:452 -> :488/:512 -> save_paper_position`, with `:676` as a separate
laundering path that a position-write guard would miss (P4).

**Second-highest, and it is the actual root:** the split is **authored in the prompts** --
`backend/agents/skills/synthesis_agent.md:19,:82,:163` (SPACED, in a field named `action`) vs
`backend/agents/skills/moderator_agent.md:18,:101` (UNDERSCORED). No Python-side guard can
reach a `.md` prompt. A guard at the write seam will **detect** this drift loudly; it cannot
prevent it. Worth queueing as its own step per
`feedback_queue_discovered_defects_in_masterplan`.

**Out of scope but recorded:** the frontend re-implements the vocabulary in 4 files with
`string`-typed fields (`frontend/src/lib/types.ts:126,:152,:283,:653`) and two **substring**
tests (`DebateView.tsx:79,81`, `RecentReportsTable.tsx:34,36`) of the exact shape
`recommendation_vocab.py:32-36` files as a defect class. A Python boundary guard is
**structurally incapable** of covering these. Naming this honestly in the contract prevents a
"the class is now closed" over-claim.

**Constraint check (all caller constraints satisfied by the above):** nothing here proposes
adding `new_buy_signal` to the vocabulary (it argues the opposite -- the field conflation is
the defect); nothing proposes quieting the UNRECOGNISED line (finding #2 argues it is
load-bearing); nothing proposes promoting a flag; the recommended guard fails loud and never
coerces; and the inert-when-disabled requirement has a working precedent in the `_resolve_rec`
legacy-parity oracle (P7). Research and report only -- **no production code was modified in
this session.**

## Audit-class coverage loop (loop-until-dry, K=2)

| Round | Scope | New read-in-full findings | Dry? |
|---|---|---|---|
| 1 | E1-E3b field/write/read enumeration + masterplan denominators | Many: ~25 write seams, ~11 read seams, 2 live enums, 4 frontend files, count is 6 filed steps not 5 | No |
| 2 | Non-Python surfaces: SQL/JSON/MD, skill prompts, trade-action vocabulary | **Root cause found** -- the split is authored in `synthesis_agent.md` / `moderator_agent.md`; `action` field carries recommendation VALUES | No |
| 3 | Rival canonicalisers, the `:2514` gate, duplicate files, existing tests | Source-level contradiction at `:2514`; `tasks/analysis.py` == `api/analysis.py` byte-identical | No |
| 4 | Producer trace, frontend typing, enum consumers | `:2630` fallback chain merges both vocabularies; frontend typed `string`; the SPACED enum is near-dead as enforcement | No |
| 5 | BQ schema modes, sentinel reachability, API request models, whole-`scripts/` sweep | `paper_positions` is NULLABLE vs `outcome_tracking` REQUIRED (P5); `api/portfolio.py:38` external ingress | No |
| 6 | Close the sentinel-to-disk risk (`portfolio_manager.py:294->304->383`) | **Zero.** Resolved the 5b risk to "cannot reach disk" (`:304 continue` filters `__UNRECOGNISED__` before `:383`). No new seam, no new source. | **Yes** |
| 7 | Whole-repo all-extension literal sweep (with control) + orchestrator producer + Pydantic/Literal constraints; external re-search | **Zero.** All 22 files returned were already inventoried (control: 27 without exclusions). Only refinement: `backend/agents/schemas.py:46 recommendation: Recommendation` -- an anchor for the already-recorded two-enum finding, not a new class member. External round returned no new read-in-full source. | **Yes** |

`dry_rounds = 2`, `K_required = 2` -> **`coverage.dry = true`**. Recorded transparently so a
reader can disagree with the round-7 judgement: the one thing round 7 added
(`schemas.py:46`) is stated above rather than omitted.

## Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL -- **10** (8 `WebFetch` + 2 `curl`+`pypdf`; floor cleared on `WebFetch` alone)
- [x] 10+ unique URLs total (incl. snippet-only) -- **39**
- [x] Recency scan (last 2 years) performed + reported -- 1 new finding (arXiv:2607.01711v1) + 2 recorded nulls
- [x] Full papers / pages read (not abstracts); PDF quotes regex-verified against extracted text
- [x] file:line anchors for every internal claim; caller-supplied anchors independently re-printed as positive controls
- [x] `coverage.dry == true` (audit-class) -- rounds 6 and 7 both dry, table above

Soft checks:
- [x] Internal exploration covered every relevant module in the caller's scope, plus the prompt/frontend/migration surfaces the scope did not name
- [x] Contradictions / consensus noted -- incl. a deliberately sought counterpoint (#5) and one source-level contradiction flagged as UNRESOLVED rather than asserted (P8)
- [x] All claims cited per-claim with URL + access date

## Status envelope -- FINAL

```json
{
  "brief_status": "COMPLETE",
  "tier": "moderate",
  "external_sources_read_in_full": 10,
  "snippet_only_sources": 29,
  "urls_collected": 39,
  "recency_scan_performed": true,
  "internal_files_inspected": 24,
  "coverage": {
    "audit_class": true,
    "rounds": 7,
    "dry_rounds": 2,
    "K_required": 2,
    "new_findings_last_round": 0,
    "dry": true
  },
  "brief_path": "handoff/current/research_brief_86.63.md",
  "gate_passed": true
}
```
