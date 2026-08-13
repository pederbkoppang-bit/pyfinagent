# Research Brief -- step 86.58: a SELL rule that cannot fire

**Tier:** moderate (caller-specified). **Audit-class:** NO (coverage reported for information only).
**Researcher:** Layer-3 Researcher (Workflow rail). **Started:** 2026-08-13.

## Envelope (born inert -- flipped to COMPLETE as the final act)

```json
{
  "brief_status": "COMPLETE",
  "tier": "moderate",
  "external_sources_read_in_full": 6,
  "snippet_only_sources": 32,
  "urls_collected": 38,
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
  "gate_passed": true
}
```

## Objective

Two questions:

1. **EXTERNAL** -- how do practitioners keep a value from one vocabulary out of a field whose
   readers assume another? Boundary-validation patterns (closed enums / sealed types at module
   boundaries, parse-don't-validate, contract/schema checks at persistence boundaries, tagged
   unions), and **where the guard belongs** (one boundary guard vs per-site patches). This project
   has fixed five instances of this class at five separate sites and a sixth keeps appearing --
   evidence on single-boundary enforcement vs per-site patching is the highest-value finding.
2. **INTERNAL** -- what writes `paper_positions.recommendation`, what reads it, what the two dark
   flags change, and what the `:208-218` interaction warning implies for blast radius.

Hard constraints carried from the spawn prompt: paper only; must NOT promote either flag
(operator-gated, ask 06-8); must NOT add `'new_buy_signal'` to the recommendation vocabulary;
must NOT weaken/quiet the phase-86.20 UNRECOGNISED log line; reading `backend/.env` is DENIED
(live flag values must come from the running process via `GET /api/settings/`).

---

## INTERNAL: the mechanism, precisely

### The write path -- what puts a trade reason into an analysis-verdict field

`backend/services/paper_trader.py:447-457` is the single site that chooses what
`paper_positions.recommendation` stores:

```python
_pos_rec = reason
if (
    getattr(self.settings, "paper_position_recommendation_fix_enabled", False)
    and analysis_recommendation
):
    _pos_rec = analysis_recommendation
```

`_pos_rec` is then written into the position row at **two** places -- the ADD-ON branch
(`paper_trader.py:488`) and the NEW-POSITION branch (`paper_trader.py:512`), both keyed
`"recommendation"`, both handed to `self.bq.save_paper_position(pos_row)` (`:498`, `:519`).
So with the flag OFF (its default), **both** branches persist `reason` -- the ORDER
MECHANISM (`'new_buy_signal'`, `'swap_buy'`) -- into a column every downstream reader
treats as an ANALYST VERDICT. `analysis_recommendation` reaches `execute_buy` as a keyword
arg declared at `paper_trader.py:256-260`, supplied by the two live call sites in
`backend/services/autonomous_loop.py:251` and `:1768`, both via
`getattr(order, "analysis_recommendation", "")`.

The producing field is `TradeOrder.analysis_recommendation` (`portfolio_manager.py:56`),
and its own docstring (`portfolio_manager.py:49-56`) already names the defect verbatim:

> "the ANALYSIS recommendation (BUY/STRONG_BUY) behind this order, distinct from `reason`
> (the trade mechanism, e.g. "new_buy_signal"/"swap_buy"). paper_trader historically wrote
> `reason` into paper_positions.recommendation, so the signal_downgrade SELL rule
> (old_rec in _BUY_RECS at :127) could never match -- structurally dead."

NOTE a stale anchor inside that docstring and in the settings description: both cite
`portfolio_manager.py:127` for the rule. The rule is at **`:264`** in the current file
(999 lines). Re-derive before citing.

### The read path -- who consumes the field

- **`portfolio_manager.py:247-251`** -- `old_rec = _resolve_rec(pos.get("recommendation"), ...,
  default="", site="held position row")`. This is the ONLY consumer that makes a MONEY
  decision from the column.
- **`portfolio_manager.py:264`** -- the rule itself:
  `if old_rec in _BUY_RECS and rec in _DOWNGRADE_RECS:` -> SELL, `reason="signal_downgrade"`.
  With `_BUY_RECS = {"BUY", "STRONG_BUY"}` (`:64`) and a stored `'new_buy_signal'`,
  `_resolve_rec` returns `'NEW_BUY_SIGNAL'` on the OFF path (the literal
  `(raw or default).upper()` at `:159`), which is in none of the three sets. The left
  conjunct is therefore **false on every held row**, and the rule is dead by construction --
  no HOLD/SELL downgrade can ever reach the SELL branch through it.
- **`paper_trader.py:676`** -- `"recommendation": position.get("recommendation", "")`, a
  pass-through into a returned/serialised position payload (reporting, not a decision).

### The two dark flags (defaults FALSE at `settings.py:210` and `:214`)

| Flag | settings.py | What it changes | Blast radius |
|---|---|---|---|
| `paper_position_recommendation_fix_enabled` | `:210` (desc `:212`) | `paper_trader` stores the ANALYSIS verdict instead of `reason` in the position row | Revives `signal_downgrade` **for NEW buys only** -- "old rows keep trade reasons and never match; no backfill" |
| `paper_recommendation_vocab_fix_enabled` | `:214` (desc `:216`) | canonicalises onto the closed scale `{STRONG_BUY,BUY,HOLD,SELL,STRONG_SELL}` before the membership gates | BOTH sides, OPPOSITE polarity: BUY side spends cash ('Strong Buy' starts reaching the buy stage); SELL side is risk-REDUCING ('Strong Sell' currently matches NEITHER `_SELL_RECS` NOR `_DOWNGRADE_RECS`) |

The two flags are **entangled**, and the entanglement is already documented at
`settings.py:216`: the vocab flag "ALSO un-defeats phase-61.2: paper_trader persists the
recommendation verbatim, so the full path stores 'Strong Buy' and the signal_downgrade rule
61.2 revives could never match it even with paper_position_recommendation_fix_enabled ON."
That is the sixth instance of the class: fixing the FIELD (61.2) still leaves the VALUE in a
foreign spelling, so **promoting flag 1 alone does not revive the rule for full-path buys.**

### The `:208-218` interaction warning and its blast radius

`portfolio_manager.py:208-220` fires once per `decide_trades` call when
`paper_position_recommendation_fix_enabled` is ON while `paper_synthesis_integrity_enabled`
is OFF:

> "rail-failure synthetic HOLDs can trigger signal_downgrade SELLs of healthy positions.
> Enable the integrity flag first (phase-61.2 interaction hazard)."

Mechanism: with the field fixed, `old_rec` becomes a real `BUY`/`STRONG_BUY`, so the left
conjunct finally passes. The right conjunct is `rec in _DOWNGRADE_RECS`, and
`_DOWNGRADE_RECS` **includes `HOLD`** (`:62`). `rec` defaults to `"HOLD"` when the re-eval
carries no recommendation (`:242-246`, `default="HOLD"`). So any re-eval that DEGRADES to a
synthetic/absent verdict -- exactly what `paper_synthesis_integrity_enabled` exists to stop
(`settings.py:208`: synthesis errors currently "persist synthetic 0.0/HOLD") -- produces
`HOLD` and SELLS a healthy position. **Blast radius = every held position whose re-eval
fails on that cycle, not just genuinely downgraded ones.** It is a transient-rail-failure
liquidation risk, which is why it is a WARNING and why the step must not promote the flag.

### Live flag values -- READ FROM THE RUNNING PROCESS (a stated absence, proven)

`GET http://127.0.0.1:8000/api/settings/` returns **45 keys**. Enumerated in full, ZERO of
them contain `fix`, `vocab`, `integrity`, or `recommend`. All three flags
(`paper_position_recommendation_fix_enabled`, `paper_recommendation_vocab_fix_enabled`,
`paper_synthesis_integrity_enabled`) are **ABSENT from the read surface**. Per the spawn
constraint I therefore do NOT substitute a `settings.py` default as "the live value".

This absence is STRUCTURAL, not incidental: the read model `FullSettings`
(`backend/api/settings_api.py:101-123`) simply has no field for them. Yet the WRITE map
`_FIELD_TO_ENV` (`settings_api.py:261-266`) DOES carry
`paper_synthesis_integrity_enabled` and `paper_position_recommendation_fix_enabled` under a
comment claiming they are "operator-visible in the Settings UI rather than
manual-.env-only -- the 61.1 lesson". **They are writable but not readable** -- a real
observability gap. `paper_recommendation_vocab_fix_enabled` (phase-86.20) is in NEITHER map.

---

## INTERNAL (cont.): the boundary module ALREADY EXISTS -- and is installed on the wrong side

`backend/services/recommendation_vocab.py` (209 lines, phase-86.20) is already the
single-canonicaliser this class calls for. Its own docstring states the intent
(`recommendation_vocab.py:15-17`):

> "This module is that mapping, and it is meant to be the ONLY one -- two canonicalisers
> that disagree would be this same defect wearing a different hat."

It exports the closed scale `CANONICAL_RECOMMENDATIONS` (`:58-60`), `canonical_recommendation()`
(`:67`), `is_recognised()` (`:89`), and -- added by phase-86.22 -- the SHARED INTENT PREDICATES
`BUY_INTENT`/`SELL_INTENT` (`:111-112`) and `is_buy_intent()`/`is_sell_intent()`/`is_directional()`
(`:115`/`:125`/`:130`). The phase-86.22 comment block (`:95-105`) counts the prior instances of
this class explicitly -- this is the caller's "five instances" measured in-repo:

> "phase-86.20 gave the repo one canonicaliser but left every consumer to decide for itself what
> counts as a buy. Measured then: FIVE consumers, TWO mutually incompatible dialects, and the sets
> written out by hand at each site -- `("Strong Buy","Buy")` in one file, `("BUY","STRONG_BUY")` in
> another, and a SUBSTRING test in a third. Re-deriving membership per call site is how the two
> dialects drifted apart in the first place, so the sets live HERE, once."

and it names the exact failure mode the sixth instance exhibits (`:102-105`):

> "A caller that unwraps them back into a literal set has undone the point."

**That is precisely what `portfolio_manager.py` still does.** It imports ONLY
`canonical_recommendation` (`portfolio_manager.py:16`) and re-declares the membership sets by hand
at `:60`, `:62`, `:64` -- `_SELL_RECS`, `_DOWNGRADE_RECS`, `_BUY_RECS` -- while `BUY_INTENT` and
`is_buy_intent()` sit unused in the shared module. The file that owns the dead rule is the one file
that opted out of the shared predicates.

**And the guard is on the READ side only.** The canonicaliser is consulted at
`portfolio_manager.py:128` (a reader). The WRITE side has no parse step at all:
`paper_trader.py:452` assigns `_pos_rec = reason` unconditionally, and the persistence choke point
`bigquery_client.py::save_paper_position` (`:626`) accepts an arbitrary `dict` and MERGEs whatever
keys it is handed. It DOES already carry a boundary precondition -- `:638-639` raises
`ValueError("save_paper_position requires 'ticker' field for MERGE key")` -- so a boundary
assertion has precedent there. Note also `:637` drops `None` values before the MERGE, so a `None`
recommendation silently leaves the existing column untouched rather than nulling it.

Consumers of the shared module today: `portfolio_manager.py:16`, `agents/conflict_detector.py:10`
(86.22), `slack_bot/formatters.py:14`, `slack_bot/jobs/nightly_outcome_rebuild.py:8` (86.25),
`services/autonomous_loop.py:26` (86.25). Note `resolve_outcome_recommendation` (`:195`) resolves
an unparseable value to `UNKNOWN_RECOMMENDATION = "UNKNOWN"` (`:192`) -- an absence-as-value
convention distinct from `portfolio_manager`'s `_UNRECOGNISED_REC = "__UNRECOGNISED__"` (`:71`).
Two sentinel spellings for the same fact already exist.

---

## EXTERNAL: read in full (6; counts toward the gate)

| # | URL | Accessed | Kind | Fetched how | Key finding |
|---|---|---|---|---|---|
| 1 | https://lexi-lambda.github.io/blog/2019/11/05/parse-don-t-validate/ | 2026-08-13 | authoritative blog (canonical, year-less) | WebFetch full | "Get your data into the most precise representation you need as quickly as you can. Ideally, this should happen at the boundary of your system, before _any_ of the data is acted upon." Names the anti-pattern: "shotgun parsing" -- "parsing and input-validating code is mixed with and spread across processing code -- throwing a cloud of checks at the input". |
| 2 | https://www.rfc-editor.org/rfc/rfc9413.html | 2026-08-13 | IETF standards-track doc (official) | WebFetch full | "It is far more efficient in the long term to fix one isolated bug than it is to deal with the consequences of workarounds." Describes the "pathological feedback cycle" where tolerated errors "become entrenched, forcing other implementations to be tolerant of those errors", and recommends "generat[ing] fatal errors for unspecified conditions instead of attempting error recovery ... to ensure that faults receive attention". |
| 3 | https://learn.microsoft.com/en-us/azure/architecture/patterns/anti-corruption-layer | 2026-08-13 | official vendor architecture doc | WebFetch full | ACL = one translation membrane between subsystems that "don't share the same semantics"; "The anti-corruption layer contains all the logic necessary to translate between the two systems." Warns: "focus the anti-corruption layer on translation logic. Avoid placing business rules or orchestration in the layer," and "consider enforcing input validation and sanitization at this boundary." |
| 4 | https://martinfowler.com/bliki/TolerantReader.html | 2026-08-13 | authoritative blog **[COUNTER-POSITION]** | WebFetch full | Argues the opposite reflex -- "be conservative in what you do, be liberal in what you accept" -- "only take the elements you need, ignore anything you don't". Crucially he still centralises: tolerance belongs in "one bit of code that reads data payloads like this" (a DTO), not sprinkled. Tolerance is scoped to *additive* change, not to contract violations. |
| 5 | https://cheatsheetseries.owasp.org/cheatsheets/Input_Validation_Cheat_Sheet.html | 2026-08-13 | official security guidance | WebFetch full | "Input validation should happen as early as possible in the data flow, preferably as soon as the data is received from the external party." Allow-list over deny-list: "Allowlist validation involves defining exactly what IS authorized, and by definition, everything else is not authorized." For fixed option sets: "the input needs to match exactly one of the values offered". |
| 6 | https://arxiv.org/html/2607.13206v1 | 2026-08-13 | peer-reviewed preprint (2026, recency) | WebFetch full (arXiv HTML per the chain) | 1,646 multi-patch fixes, 1999-2025, mean 2.55 patches/CVE. **Category C1 = 641 incomplete fixes** "leaving vulnerabilities unresolved"; **Category A = 860 multi-location** cases where "similar vulnerable code appears in multiple methods, branches, or even separate projects". 31.7% of multi-patch fixes span >1 day between patches. |

### Search-query variants run (three-variant discipline)

- **Year-less canonical:** `parse don't validate make illegal states unrepresentable boundary validation`; `validate at the boundary once single choke point versus scattered validation defensive programming`; `anti-corruption layer bounded context translate vocabulary between models Evans DDD`.
- **Last-2-year / current-year:** `enum string mismatch bug database persisted value schema validation 2025 2026 postgres check constraint enum drift`; `"parse, don't validate" 2025 2026 boundary type safety adoption critique`; `recurring defect same class multiple sites fix root cause once systemic remediation empirical study 2025`.

### Identified but snippet-only (32; does NOT count toward the gate)

| URL | Kind | Why not fetched in full |
|---|---|---|
| https://deviq.com/principles/parse-dont-validate/ | wiki | secondary restatement of source 1 |
| https://deviq.com/principles/make-illegal-states-unrepresentable/ | wiki | secondary restatement |
| https://aipatternbook.com/make-illegal-states-unrepresentable | wiki | secondary restatement |
| https://kb.evryg.com/en/advanced-software-engineering/type-design/parse-dont-validate | wiki | secondary restatement |
| https://lobste.rs/s/uon7sc/parse_don_t_validate_2019 | forum | community tier |
| https://news.ycombinator.com/item?id=47103931 | forum | community tier |
| https://medium.com/@trinitietp/parse-dont-validate-practical-lessons-df352da8a154 | blog | low tier, no new claim |
| https://rednegra.net/blog/20250810-parse-dont-validate/ | blog | 2025 restatement |
| https://contemplating.dev/posts/parse-dont-validate/ | blog | 2025 restatement |
| https://cekrem.github.io/posts/parse-dont-validate-typescript/ | blog | language-specific |
| https://cekrem.github.io/posts/arktype-parse-dont-validate-sequel/ | blog | library-specific |
| https://www.ricardodecal.com/opinions/parse-don-t-validate-in-python/ | blog | Python framing, no new claim |
| https://www.beyondthesemicolon.com/parse-dont-validate-level-up-your-c-code-by-making-illegal-states-impossible/ | blog | C#-specific |
| https://www.lelanthran.com/chap13/content.html | blog | C-specific |
| https://henko.net/blog/barricade-choke-point/ | blog | supports choke-point framing |
| https://blog.securityinnovation.com/blog/2011/09/validating-trust-boundaries.html | industry | centralised-validation argument, superseded by source 5 |
| https://devsecopsschool.com/blog/trust-boundary/ | blog | definitional |
| https://medium.com/@dykyi.roman/understanding-validation-levels-74d0adecda5e | blog | taxonomy only |
| https://hosseinnejati.medium.com/the-anti-corruption-layer-protecting-your-domain-from-legacy-systems-6da58fc5f462 | blog | secondary to source 3 |
| https://synchronium.github.io/software-architecture-wiki/patterns/anti-corruption-layer.html | wiki | secondary to source 3 |
| https://www.sysdesai.com/learn/decomposition-integration/anti-corruption-layer | blog | secondary to source 3 |
| https://ddd-practitioners.com/home/glossary/bounded-context/bounded-context-relationship/anticorruption-layer/ | glossary | secondary to source 3 |
| https://learnixo.io/blog/ddd-bounded-contexts | blog | secondary |
| https://softwarepatternslexicon.com/mastering-design-patterns/domain-driven-design-ddd-patterns/bounded-contexts/ | wiki | secondary |
| https://medium.com/@malotor/anticorruption-layer-a-effective-shield-caa4d5ba548c | blog | secondary |
| https://thoughtbot.com/blog/enum-validations-and-database-constraints-in-rails-7-1 | industry blog | framework-specific enum+DB constraint pairing |
| https://github.com/rails/rails/issues/52279 | issue tracker | anecdote |
| https://github.com/mikro-orm/mikro-orm/issues/7395 | issue tracker | anecdote |
| https://github.com/spring-projects/spring-data-jpa/issues/3699 | issue tracker | anecdote |
| https://www.postgresql.org/message-id/17271-8b4317357c4991e2@postgresql.org | mailing list | enum-constraint mechanics, not design guidance |
| https://arxiv.org/html/2510.19593v1 | preprint | RCA survey, adjacent not on-point |
| https://arxiv.org/pdf/2411.13017 | preprint | RCA-with-GenAI, adjacent; PDF URL not fetched per the arXiv chain |

### Recency scan (2024-2026) -- PERFORMED

Searched the 2024-2026 window on three axes: enum/vocabulary drift at persistence boundaries;
"parse, don't validate" adoption and critique; and empirical work on repeat-site defect
remediation. **Result: two findings that MATERIALLY change the recommendation, and they point in
opposite directions.**

1. **NEW, and it is the strongest evidence in this brief.** arXiv:2607.13206v1 (2026) supplies the
   first *quantitative* answer to the caller's question. In 1,646 multi-patch CVE fixes, **641
   (38.9%) were Category C1 incomplete fixes** -- a first patch that did not finish the job -- and
   **860 were Category A multi-location** cases arising precisely because "similar vulnerable code
   appears in multiple methods, branches, or even separate projects". Per-site patching is not
   merely inelegant; in the largest corpus available it is the modal route to an incomplete fix.
   This supersedes the purely argumentative case made by the older canonical sources.
2. **NEW, and it QUALIFIES source 1.** Alexis King (the author of "Parse, Don't Validate")
   published a follow-up, *"Names are not type safety"*, clarifying that the newtype/wrapper
   pattern gives **weaker guarantees than correctness by construction**. Directly relevant: merely
   *naming* a field `analysis_recommendation` (as `TradeOrder:56` already does) buys nothing on its
   own -- `portfolio_manager` still hand-writes literal sets at `:60-64`. Surfaced as a snippet
   only; not read in full, so it is reported as a qualifier, not a load-bearing claim.

No 2024-2026 source contradicted the boundary-enforcement consensus.

---

## Key findings

1. **Put the parse at the boundary, once, and make the downstream type the proof.** "Get your data
   into the most precise representation you need as quickly as you can. Ideally, this should happen
   at the boundary of your system, before _any_ of the data is acted upon" (King 2019,
   https://lexi-lambda.github.io/blog/2019/11/05/parse-don-t-validate/). The scattered alternative
   has a name -- *shotgun parsing* -- and the stated consequence is that "some portion of invalid
   input [has] been processed, with the consequence that program state is difficult to accurately
   predict". pyfinagent's live symptom is exactly that: an order-mechanism string is already
   persisted and already being read by a money gate.
2. **Per-site patching is empirically the route to incomplete fixes.** 641 of 1,646 multi-patch
   fixes (38.9%) were incomplete first attempts; 860 arose from the same defective code recurring
   in multiple locations (arXiv:2607.13206v1, https://arxiv.org/html/2607.13206v1). This is the
   direct answer to the caller's highest-value question, and it favours single-boundary
   enforcement over a sixth per-site patch.
3. **Tolerating the wrong value entrenches it.** "Errors in implementations or confusion about
   semantics are permitted or ignored. These errors can become entrenched" and the flaw "can become
   entrenched as a de facto standard" (RFC 9413, https://www.rfc-editor.org/rfc/rfc9413.html).
   Applied here: adding `'new_buy_signal'` to `_BUY_RECS` -- explicitly forbidden by the spawn
   constraints -- is the textbook entrenchment move. The RFC's own remedy is the one this repo
   already chose: "It is far more efficient in the long term to fix one isolated bug than it is to
   deal with the consequences of workarounds."
4. **Loudness is the mechanism that makes the boundary self-correcting.** "Intolerance toward
   violations of specification improves feedback ... it receives strong feedback that allows the
   problem to be discovered quickly" (RFC 9413, ibid.). This is a literature-backed argument for
   the spawn constraint "do not weaken or quiet the phase-86.20 UNRECOGNISED log line" --
   `portfolio_manager.py:132-137`, which is unconditional by design (`:107-109`).
5. **The translation layer must be single and must not carry business rules.** "The anti-corruption
   layer contains all the logic necessary to translate between the two systems" and "focus the
   anti-corruption layer on translation logic. Avoid placing business rules or orchestration in the
   layer" (Microsoft Learn, ACL pattern). `recommendation_vocab.py` already satisfies both halves;
   the deviation is that `portfolio_manager` bypasses its predicates.
6. **Fixed option sets get exact-match allow-listing at entry.** "If the input field comes from a
   fixed set of options ... the input needs to match exactly one of the values offered" and
   validation "should happen as early as possible in the data flow" (OWASP Input Validation Cheat
   Sheet). `CANONICAL_RECOMMENDATIONS` (`recommendation_vocab.py:58`) is that allow-list; nothing
   applies it on the WRITE path.
7. **The counter-position exists and is narrow.** Fowler's Tolerant Reader
   (https://martinfowler.com/bliki/TolerantReader.html) argues for liberal acceptance -- but scoped
   to *additive* schema change from a provider you don't control, and even then he localises the
   tolerance in "one bit of code". It does not license a money gate silently accepting a foreign
   vocabulary. RFC 9413 is the standards-body rebuttal of the same Postel framing. The repo's own
   docstring already cites the pre-publication draft of that RFC
   (`recommendation_vocab.py:41`, "IETF `draft-thomson-postel-was-wrong`").

## Consensus vs debate

**Consensus (5 of 6 read-in-full sources):** validate/parse ONCE, at the boundary where data
enters, against an explicit closed set; propagate a value that cannot be wrong downstream; fail
loudly on the unexpected. **Debate:** Fowler's Tolerant Reader is the live counter-position and RFC
9413 is its direct rebuttal; both agree, however, that whatever policy is chosen belongs in ONE
place. No source in any tier advocates re-deriving a membership set per call site -- and the repo's
own `recommendation_vocab.py:95-105` independently reached the same conclusion after measuring
five drifted consumers.

## Pitfalls from the literature

- **Canonicalise before validating, never after** -- CWE-180, already cited at
  `recommendation_vocab.py:38-39`. Validating first and canonicalising second is its own bug class.
- **Substring/prefix matching on a closed scale** -- `"STRONG_SELL"` contains `"SELL"`; already
  filed as phase-86.22 (`recommendation_vocab.py:32-36`).
- **A centralised routine's bug "will manifest itself in hundreds of ways"** (Security Innovation,
  snippet) -- the cost of centralising is a single high-blast-radius component, which argues for
  mutation coverage on the boundary rather than against the boundary.
- **Names are not type safety** (King follow-up, snippet) -- a well-named field is not a guard.
- **Incomplete first fixes are the norm, not the exception** (38.9%, arXiv:2607.13206v1) -- expect
  to have to prove completeness, not assert it.

## Application to pyfinagent

- The defect is a **write-side** vocabulary breach with a **read-side** guard.
  `paper_trader.py:452` (`_pos_rec = reason`) is the origin; `portfolio_manager.py:264`
  (`old_rec in _BUY_RECS`) is where it manifests. Sources 1/5/6 all locate the correct guard at the
  ORIGIN, i.e. between `execute_buy` and `bigquery_client.py:626 save_paper_position`.
- **A single-boundary option already has a home.** `save_paper_position` (`bigquery_client.py:626`)
  is the sole persistence choke point for `paper_positions` and already raises on a missing
  `ticker` (`:638-639`) -- precedent for a boundary precondition. Caveat from `:637`: `None` values
  are dropped before the MERGE, so "write nothing" and "write NULL" are indistinguishable there.
- **A cheaper single-boundary option:** have `portfolio_manager` consume the shared predicates
  (`is_buy_intent`, `recommendation_vocab.py:115`) instead of `_BUY_RECS` (`:64`), retiring the
  hand-written sets at `:60-64`. That closes the "unwrapped back into a literal set" deviation the
  vocab module itself names at `:102-105`. NOTE this alone does NOT fix the defect -- the stored
  value is `'new_buy_signal'`, which is not a buy under any spelling -- so it is a
  contributing hardening, not the fix.
- **Blast-radius constraint is real and is the reason this is research-only.** Reviving the rule
  makes `_DOWNGRADE_RECS` (which contains `HOLD`, `:62`) the right conjunct against a `rec` that
  DEFAULTS to `"HOLD"` (`:242-246`). Any re-eval degradation sells a healthy position
  (`:208-220`). Any proposal must keep both flags dark and must state what happens under a
  rail-failure cycle.
- **Observability gap worth queueing separately:** all three flags are writable via
  `settings_api.py:261-266` but absent from the `FullSettings` read model
  (`settings_api.py:101-123`), so `GET /api/settings/` (45 keys, enumerated) cannot report their
  live state. `paper_recommendation_vocab_fix_enabled` is in NEITHER map.
- **Stale anchor to correct wherever it is copied:** `portfolio_manager.py:53` and
  `settings.py:212` both cite the rule at `portfolio_manager.py:127`; it is at `:264`.

## Internal code inventory

| File | Lines / anchors | Role | Status |
|---|---|---|---|
| `backend/services/portfolio_manager.py` | 999; `:49-56`, `:60-64`, `:71`, `:74-161`, `:208-220`, `:242-271` | Owns `_BUY_RECS`/`_DOWNGRADE_RECS`/`_SELL_RECS`, `_resolve_rec`, the dead `signal_downgrade` rule, the interaction warning | LIVE; rule dead by construction; hand-written sets duplicate `recommendation_vocab` |
| `backend/services/paper_trader.py` | `:256-260`, `:447-457`, `:488`, `:512`, `:676` | The write path; chooses `reason` vs `analysis_recommendation` | LIVE; flag OFF -> writes the order reason |
| `backend/services/recommendation_vocab.py` | 209; `:52-60`, `:67`, `:95-105`, `:111-137`, `:192-208` | The existing single canonicaliser + shared intent predicates | LIVE; predicates UNUSED by `portfolio_manager` |
| `backend/config/settings.py` | `:210`/`:212`, `:214`/`:216`, `:206`/`:208` | The three flag definitions, all default `False` | LIVE, all DARK |
| `backend/api/settings_api.py` | `:101-123` (read model), `:261-266` (write map) | Settings surface | LIVE; write-without-read asymmetry |
| `backend/db/bigquery_client.py` | `:626-656` | `save_paper_position` MERGE -- the persistence choke point | LIVE; no vocabulary precondition |
| `backend/services/autonomous_loop.py` | `:251`, `:1768`, `:26` | The two `execute_buy` call sites that pass `analysis_recommendation` | LIVE (use `backend/services/`, NOT the other `autonomous_loop.py`) |

## Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch -- **6**
- [x] 10+ unique URLs total (incl. snippet-only) -- **38** (6 read-in-full + 32 snippet-only)
- [x] Recency scan (last 2 years) performed + reported -- 2 material findings, both reported
- [x] Full papers / pages read (not abstracts) -- arXiv read via `/html/` per the chain; no
      `arxiv.org/pdf/` URL was WebFetched
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every module in the stated scope
- [x] Contradictions / consensus noted (Tolerant Reader vs RFC 9413, explicitly)
- [x] All claims cited per-claim
- [~] GAP, stated honestly: the live values of the three flags could NOT be read from
      `GET /api/settings/` because the surface does not expose them. Per the spawn constraint I did
      not substitute a `settings.py` default. Their live state is therefore **UNKNOWN from the
      permitted surface**; `settings.py` shows the DEFAULT is `False` for all three, which is not
      the same claim.

## Envelope (final)

```json
{
  "brief_status": "COMPLETE",
  "tier": "moderate",
  "external_sources_read_in_full": 6,
  "snippet_only_sources": 32,
  "urls_collected": 38,
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
  "brief_path": "handoff/current/research_brief_86.58.md",
  "gate_passed": true
}
```
