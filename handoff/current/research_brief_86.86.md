# Research Brief -- phase-86.86

**Topic:** Eliminating the falsy-zero / three-state-Optional collapse at data-INGRESS
seams -- routing parallel dict-construction sites through ONE resolver; sentinel vs
NamedTuple vs Optional-triple-state in Python; `or`-default anti-patterns; AST-based
defect-CLASS enumeration; mutation-testing a guard so completeness is derived not asserted.

**Tier:** moderate (caller-stated). **Audit-class:** YES (`coverage.audit_class = true`,
K_required = 2 consecutive dry rounds).
**Researcher:** Layer-3 pyfinagent researcher, Workflow rail.
**Started:** 2026-08-15.

---

## ENVELOPE (born inert -- phase-86.37)

```json
{
  "brief_status": "COMPLETE",
  "tier": "moderate",
  "external_sources_read_in_full": 16,
  "snippet_only_sources": 23,
  "urls_collected": 39,
  "recency_scan_performed": true,
  "internal_files_inspected": 8,
  "coverage": {
    "audit_class": true,
    "rounds": 12,
    "dry_rounds": 2,
    "K_required": 2,
    "new_findings_last_round": 0,
    "dry": true
  },
  "summary": "The falsy-zero collapse is at the PRODUCER (autonomous_loop.py:3092 and its Gemini twin :3338), upstream of the 86.74 three-state resolver, so a 0% REJECT is rewritten to 3.0 before portfolio_manager.py:1102 ever sees it. Ten `or`-default sites in the two lite constructions; 3 harmful, 2 benign, reasons per site. paper_risk_judge_shape_fix_enabled has ZERO production readers. Prior art converges hard: carry a presence bit, do not tune the default (protobuf field presence, PEP 661 truthy sentinels, dataclasses.MISSING, RFC 7396, Codd's two-NULL model). Fix at the boundary in ONE resolver reused from PositionVerdict. The existing mutation matrix is complete only over the subject it names and has no producer cell -- and a test must drive the producer first or the cell is UNSCORABLE.",
  "brief_path": "handoff/current/research_brief_86.86.md",
  "gate_passed": true
}
```

Gate arithmetic: `external_sources_read_in_full` 16 >= 5; `recency_scan_performed`
true; all hard blockers checked; `coverage.audit_class` true AND `coverage.dry`
true (rounds 11 and 12 consecutive dry). Envelope verified against the brief:
all 16 claimed URLs literally present (`grep -F`, 0 missing).

---

## Read in full (>=5 required; counts toward the gate)

All 16 fetched with the full page/paper text read. Source-quality tiers per
`.claude/rules/research-gate.md`: 1 peer-reviewed, 2 official docs/standards,
3 authoritative blogs, 4 industry practitioner.

| # | URL | Accessed | Kind | Tier | Fetched how | Key finding |
|---|-----|----------|------|------|-------------|-------------|
| 1 | https://peps.python.org/pep-0661/ | 2026-08-15 | PEP (Final, 3.15) | 2 | WebFetch | `sentinel()` builtin; sentinels are **truthy**, so an `or`-default cannot destroy one |
| 2 | https://protobuf.dev/programming-guides/field_presence/ | 2026-08-15 | official doc | 2 | WebFetch | Implicit presence makes "set to 0" indistinguishable from "unset"; Google's fix was to RESTORE presence, not tune defaults |
| 3 | https://www.rfc-editor.org/rfc/rfc7396 | 2026-08-15 | IETF RFC | 2 | WebFetch | Absent vs null are the two most different states; overloading one channel makes a meaning unexpressible |
| 4 | https://docs.python.org/3/library/dataclasses.html | 2026-08-15 | official doc | 2 | WebFetch | `MISSING` exists *"because `None` is a valid value... with a distinct meaning"* |
| 5 | https://cwe.mitre.org/data/definitions/1188.html | 2026-08-15 | MITRE standard | 2 | WebFetch | Permissive default = CWE-1188 (parent CWE-665); remedy is explicit initialise-to-denied |
| 6 | https://www.law.cornell.edu/cfr/text/17/240.15c3-5 | 2026-08-15 | regulation | 2 | WebFetch | Controls are written in PREVENT form: *"Prevent the entry of orders..."* |
| 7 | https://research.google.com/pubs/archive/46584.pdf | 2026-08-15 | paper (ICSE-SEIP'18) | 1 | curl + pypdf (9pp, 41,213 ch) | Coverage misleads when *"statements are covered but their consequences not asserted upon"*; AST-based arid-node suppression |
| 8 | https://arxiv.org/html/2604.01483v1 | 2026-08-15 | preprint (2026-04-01) | 1 | WebFetch | **[ADVERSARIAL]** calls Pydantic-style shape validation *"logically shallow"* -- but ships no data and no proof-timeout semantics |
| 9 | https://engineering.fb.com/2025/09/30/security/llms-are-the-key-to-mutation-testing-and-better-compliance/ | 2026-08-15 | vendor eng blog | 2 | WebFetch | Generic mutation operators *"do not represent faults developers would realistically introduce"*; equivalent-mutant detection needs an agent |
| 10 | https://docs.astral.sh/ruff/rules/if-exp-instead-of-or-operator/ | 2026-08-15 | official tool doc | 2 | WebFetch | **[ADVERSARIAL]** FURB110 pushes code TOWARD `x or y`, claiming *"the same functionality"* |
| 11 | https://inside.java/2024/06/03/dop-v1-1-illegal-states/ | 2026-08-15 | vendor eng blog | 2 | WebFetch | Sealed interface over nullable fields; runtime checks are the LAST resort |
| 12 | https://lexi-lambda.github.io/blog/2019/11/05/parse-don-t-validate/ | 2026-08-15 | authoritative blog | 3 | WebFetch | *"Push the burden of proof upward as far as possible"*; names shotgun parsing |
| 13 | https://www.red-gate.com/simple-talk/databases/sql-server/learn/sql-and-the-snare-of-three-valued-logic/ | 2026-08-15 | practitioner journal | 3 | WebFetch | Codd's SECOND model has two NULLs (unknown-value vs missing-attribute) = SIZE/UNPARSEABLE/ABSENT |
| 14 | https://prmia.org/common/Uploaded%20files/eAI/PRMIA%20Case%20study%20-%20Knight%20Trading.pdf | 2026-08-15 | industry case study | 4 | curl + pypdf (8pp, 20,416 ch) | 7-of-8 servers updated -> $440-460M in 45 min; the parallel-twin geometry |
| 15 | https://docs.aws.amazon.com/codeguru/detector-library/python/dict-get-method/ | 2026-08-15 | vendor detector doc | 2 | WebFetch | **Refutes its own snippet**: about `KeyError`, NOT falsiness; Severity Low, CWE unmapped |
| 16 | https://docs.quantifiedcode.com/python-anti-patterns/correctness/not_using_get_to_return_a_default_value_from_a_dictionary.html | 2026-08-15 | anti-pattern catalogue | 5 | WebFetch | Also about `in`-check vs `.get`, not falsiness -- confirms the prior-art gap |

---

## Internal code inventory

_(in progress)_

---

## Work log

- Round 0: read `.claude/agents/researcher.md` + `.claude/rules/research-gate.md` in full;
  wrote this born-inert brief.

---

## Internal code inventory (round 1)

### A. The AST-enumerated CLASS of `or <default>` (objective item c)

Method: `ast.walk` over the two target files, collecting every
`ast.BoolOp(op=ast.Or)` and unparsing it. This DERIVES the class instead of
hand-listing it. Script: scratchpad `enum_or_default.py` (research-time only).
**Result: 184 `or` BoolOp nodes** across
`backend/services/autonomous_loop.py` + `backend/services/portfolio_manager.py`.
Ten of them sit in the two lite risk-judge `risk_assessment` constructions.

### B. The ten lite-path sites, classified harmful vs benign

Claude lite judge (`backend/services/autonomous_loop.py:3084-3096`) and the
Gemini lite judge (`:3332-3342`) are **textually parallel** -- same five idioms,
duplicated. Falsy trigger is the value that silently becomes the default.

| # | Site | Idiom | Falsy trigger | Verdict |
|---|------|-------|---------------|---------|
| 1 | `autonomous_loop.py:3084` / `:3332` | `risk_dict.get("reasoning") or _LITE_RISK_DEFAULT["reasoning"]` | `""` | **BENIGN-OBSERVABILITY.** An empty reasoning becomes the literal string "risk-judge parse failed; falling back to conservative default sizing" -- which is a FALSE statement when the parse succeeded and only the reasoning was blank. No decision reads it; it is persisted as the `reason` summary column. Harm is audit-trail fabrication, not sizing. |
| 2 | `autonomous_loop.py:3086` / `:3334` | `risk_dict.get("decision") or _LITE_RISK_DEFAULT["decision"]` | `""` | **HARMFUL (fail-open).** `_LITE_RISK_DEFAULT["decision"] == "APPROVE_REDUCED"` (`:2298`). An empty-string decision is converted into an APPROVE. Same shape as the sizing defect one field over: absent and empty collapse, and the collapse resolves toward permission. |
| 3 | `autonomous_loop.py:3092-3094` / `:3338-3340` | `float(risk_dict.get("recommended_position_pct") or _LITE_RISK_DEFAULT["recommended_position_pct"])` | **`0.0`** | **HARMFUL -- THE DEFECT.** A judge that returns `recommended_position_pct: 0.0` (the strongest possible risk signal) has that zero overwritten with **3.0** at INGRESS. |
| 4 | `autonomous_loop.py:3095` / `:3341` | `risk_dict.get("risk_level") or _LITE_RISK_DEFAULT["risk_level"]` | `""` | **HARMFUL-MILD.** Empty risk_level becomes `"MODERATE"` -- an unknown risk is recorded as a specific, middling, reassuring one. Fabricates a risk grade rather than sizing. |
| 5 | `autonomous_loop.py:3096` / `:3342` | `dict(risk_dict.get("risk_limits") or _LITE_RISK_DEFAULT["risk_limits"])` | `{}` | **BENIGN-CONSERVATIVE, with a caveat.** An empty `risk_limits` becomes `{stop_loss_pct: 10.0, max_drawdown_pct: 15.0}`. A stop is INSTALLED where none existed, so the substitution is protective. Caveat: a judge asking for a *tighter-than-default* stop by sending `{}` cannot; and a **non-empty** `{"stop_loss_pct": 0.0}` passes the `or` untouched, so this idiom is not where a zero-stop would be lost. |

### C. Asymmetry worth recording

At site 3 the value is `float(... or ...)`, so the falsy test runs on the RAW
value **before** coercion. A judge emitting the JSON **string** `"0"` is TRUTHY
and survives to `float("0") == 0.0`; a judge emitting the **number** `0` is falsy
and is destroyed. The defect is therefore serialisation-dependent -- two judges
saying the same thing get opposite treatment.


### D. The chain: where the zero dies, and why 86.74 cannot see it

1. Lite judge parses JSON -> `risk_dict` (`autonomous_loop.py:3074-3082` /
   `:3320-3328`).
2. **`risk_assessment` is CONSTRUCTED at `:3085-3097` / `:3333-3343`. The `or`
   collapses `0.0` -> `3.0` HERE.**
3. Returned in the analysis dict as `"risk_assessment": risk_assessment`
   (`:3121` / `:3357`).
4. `decide_trades` -> `portfolio_manager.py:340` calls
   `_resolve_position_pct(_rj_view, analysis)` which reads
   `risk_assessment.get("recommended_position_pct")` (`:1102`).
5. It sees **3.0**, returns `PositionVerdict(SIZE, 3.0)`, `blocks_buy` is False,
   `position_pct_state = "SIZE"` is written at `:409`, and `_sizing_pct`
   (`:1022`) correctly returns 3.0 for a verdict that was really zero.

**This is the headline. phase-86.74 hardened the CONSUMER; the destruction
happens at the PRODUCER, upstream of it.** The three-state machinery is not
wrong -- it is *blind*, because the information it exists to preserve was
already erased two files earlier. A resolver can only distinguish states that
still reach it.

Corollary: the 86.74 fix IS load-bearing on the FULL path (where
`risk_debate.py:152` writes a real `0`, and `orchestrator.py:2415` likewise),
and on the deterministic `_data_integrity_blocked_analysis` path
(`autonomous_loop.py:2358` writes `"recommended_position_pct": 0.0` as a plain
dict literal with NO `or`, so that zero survives). It is defeated **only** on the
two LLM lite paths.

### E. Flag state (measured, not assumed)

| Flag | `settings.py` | Production readers | Status |
|---|---|---|---|
| `paper_risk_judge_reject_binding` | `:342`, default `False` | exactly ONE: `portfolio_manager.py:385` | LIVE-but-DARK; gates the lite REJECT block |
| `paper_risk_judge_parse_fail_reject` | `:346`, default `False` | one: `risk_debate.py:138` (FULL path only) | DARK; does not touch the lite path |
| `paper_risk_judge_shape_fix_enabled` | `:350`, default `False` | **ZERO** -- only tests reference it | **DEAD FLAG.** 86.74 made the fix unconditional; nothing in `backend/` or `scripts/` reads it any more. |

Method: `grep -rn --include="*.py" <flag> backend/ scripts/ | grep -v settings.py`.
The `shape_fix` flag being dead matters for scoping: a 86.86 fix must NOT be hung
on it, and its `settings.py:352` description still claims it governs behaviour.

### F. Second-order consequence: the persisted audit trail

`signal_attribution.py:229-247` reads `risk.get("recommended_position_pct")` and
its phase-86.74 comment records that **DELL on 2026-08-13 carried
`recommended_position_pct = 0`** read from
`$.final_synthesis.risk_assessment.judge.recommended_position_pct`. On the lite
path that zero would never have reached BQ at all -- the persisted row would say
3.0. So the ingress collapse also destroys the forensic record: an auditor
querying `analysis_results` cannot reconstruct that the judge said zero.

`_persist_analysis` (`autonomous_loop.py:3379`, generalised from
`_persist_lite_analysis`) writes the already-collapsed dict; it introduces no new
defaulting of this field.

---

## External findings (rounds 1-3)

**Search-query variants run** (three-variant discipline): current-year frontier
("...2026"), last-2-year ("...2025"/"2024"), and year-less canonical
("PEP 661 sentinel values Python standard library", "Python or default operator
falsy zero anti-pattern", "mutation testing State of Mutation Testing at Google").

1. **PEP 661 is FINAL, resolved 2026-04-23, landing in Python 3.15** -- a
   `sentinel()` builtin. Design criterion 1: *"When compared using the `is`
   operator, it should always be considered identical to itself but never to any
   other object."* Decisive detail for this step: *"Sentinel objects are
   'truthy', i.e. boolean evaluation will result in `True`. This parallels the
   default for arbitrary classes... This is unlike `None`, which is 'falsy'."*
   A PEP-661 sentinel therefore **cannot be destroyed by an `or`-default**,
   whereas `None` and `0.0` both can. (https://peps.python.org/pep-0661/,
   accessed 2026-08-15)
   PEP 661 also rejects the single-valued-Enum alternative -- *"repr is overly
   long"* -- and rejects a single global `MISSING` because it *"could be valid in
   some contexts; lacks contextual meaning"*. That argues against a
   project-global sentinel and for a domain-named one, which is what
   `portfolio_manager.py:994-1013` already does with `SIZE`/`ABSENT`/`UNPARSEABLE`.

2. **[ADVERSARIAL] Ruff FURB110 actively pushes code TOWARD the defective
   idiom.** The rule `if-exp-instead-of-or-operator` says: *"Checks for ternary
   `if` expressions that can be replaced with the `or` operator"*, because
   *"Ternary `if` expressions are more verbose than `or` expressions while
   providing the same functionality."* **"The same functionality" is false for
   falsy-but-valid values**, which is exactly this defect class. Ruff's only
   stated safety caveat is double-evaluation side effects, not falsy semantics,
   and *"Fix is always available"*. So a `x if x is not None else y` rewrite --
   the correct fix -- is not what FURB110 targets (it targets `x if x else y`),
   but the rule's framing legitimises `or` as the idiomatic default and a team
   enabling refurb rules will drift toward it.
   (https://docs.astral.sh/ruff/rules/if-exp-instead-of-or-operator/, accessed
   2026-08-15)

3. **The two "dict.get" prior-art sources DO NOT support the falsy claim -- read
   in full, they are about `KeyError`, not falsiness.** AWS CodeGuru's
   `python/dict-get-method@v1.0` (Severity **Low**, Category **Code Quality**,
   CWE **unmapped**, tags `#availability`, `#maintainability`) flags
   `try: mydict[key] except KeyError` and recommends `mydict.get(key, 0)`. The
   quantifiedcode anti-pattern likewise contrasts `if "k" in d` with
   `d.get("k", "")` and justifies it as *"more concise"* and avoiding querying
   *"the dictionary twice"*. **Neither mentions the falsy-value trap.** This is a
   genuine gap in the prior art: the search snippets characterised these as
   falsy-default sources and reading them in full refuted that. Recording it
   because it changes the design argument -- there is no authoritative
   lint/detector backing for "`or`-default is a correctness defect", so this
   project cannot cite one and must carry its own guard.
   (https://docs.aws.amazon.com/codeguru/detector-library/python/dict-get-method/
   and
   https://docs.quantifiedcode.com/python-anti-patterns/correctness/not_using_get_to_return_a_default_value_from_a_dictionary.html,
   both accessed 2026-08-15)

4. **CWE-1188, "Initialization of a Resource with an Insecure Default"** --
   *"The product initializes or sets a resource with a default that is intended
   to be changed by the product's installer, administrator, or maintainer, but
   the default is not secure."* Parents: **CWE-665 Improper Initialization**,
   CWE-344 Use of Invariant Value in Dynamically Changing Context. Its
   demonstrative example is an uninitialised `$authorized` where the fix is to
   *initialise `$authorized = false` explicitly*. That is precisely the shape
   here: `_LITE_RISK_DEFAULT["decision"] = "APPROVE_REDUCED"` is an
   initialise-to-permitted default. The CWE-standard remedy is
   initialise-to-denied. (https://cwe.mitre.org/data/definitions/1188.html,
   accessed 2026-08-15)

5. **Mutation testing (Petrovic & Ivankovic, ICSE-SEIP 2018, Google)** -- read in
   full via `pypdf` (9 pages, 41,213 chars extracted; WebFetch on PDFs is barred
   in this project because its summaries have twice fabricated quotes).
   *"Mutation testing assesses test suite efficacy by inserting small faults into
   programs and measuring the ability of the test suite to detect them. It is
   widely considered the strongest test criterion in terms of finding the most
   faults and it subsumes a number of other coverage criteria."* And the reason
   coverage is not enough: *"coverage alone might be misleading, as in many cases
   where statements are covered but their consequences not asserted upon"*.
   *"Mutation score is the ratio of killed mutants to the total number of mutants
   and is a measure of this efficacy."*
   Two mechanisms transfer directly to this step:
   - **AST-based arid-node suppression.** *"we describe a method of transitive
     mutation suppression of uninteresting, arid lines based on developer
     feedback and program's AST"* -- i.e. Google enumerates and classifies
     mutation targets **from the AST**, the same technique this brief used to
     derive the 184-node class. Enumerate from the tree, then suppress by rule.
   - **Equivalent mutants are the trap.** Google excludes memory-reserving calls
     because mutating them *"usually leads to slower but equivalent mutants"* --
     a mutant that changes source but not observable behaviour. A guard-mutation
     cell that produces an equivalent mutant LOOKS like a surviving mutant
     (test still green) but indicts nothing.
   (https://research.google.com/pubs/archive/46584.pdf, accessed 2026-08-15)


6. **Sum types beat nullable fields -- Oracle, Data-Oriented Programming v1.1
   (2024).** *"a guiding principle of data-oriented programming is to make
   illegal states unrepresentable."* Its three-tier recipe, in priority order:
   (1) *"Use precisely modeled types (usually records) to describe the data."*
   (2) instead of multiple conditional fields, *"create a sealed interface to
   model the alternatives and use it as the type for a mandatory field."*
   (3) *"Only if these design techniques... are not sufficient, resort to
   run-time checks in the constructor."* The worked example is exactly this
   step's shape: a nullable `String email` makes the invariant *"implicit at
   best but no longer enforced"*, and the fix is `sealed interface User permits
   UnregisteredUser, RegisteredUser`.
   Mapping: `PositionVerdict` (`portfolio_manager.py:998`) is tier-1/2 done
   right -- a tagged union in NamedTuple clothing. The lite paths are still at
   tier-0 (a bare `float` in a `dict`).
   (https://inside.java/2024/06/03/dop-v1-1-illegal-states/, accessed 2026-08-15)

7. **17 CFR 240.15c3-5 is written in PREVENT form, not ADVISE form.** (c)(1)(i):
   *"Prevent the entry of orders that exceed appropriate pre-set credit or
   capital thresholds in the aggregate for each customer and the broker or
   dealer"*; (c)(1)(ii): *"Prevent the entry of erroneous orders, by rejecting
   orders that exceed appropriate price or size parameters"*; (b) requires
   controls *"reasonably designed to manage the financial, regulatory, and other
   risks"*. **Honest limitation of this source:** the fetched text does not
   explicitly mandate automated enforcement or state that controls cannot be
   bypassed, so this brief does NOT claim it does. What it does establish is the
   verb: a risk control is a thing that *prevents*, and a control whose absent
   value is silently replaced by a permissive default has not prevented
   anything. (https://www.law.cornell.edu/cfr/text/17/240.15c3-5, accessed
   2026-08-15)

---

## Recency scan (last 2 years, 2024-2026)

Searched: `PEP 661 sentinel values Python standard library`; `null object pattern
versus Optional three-state modelling 2026 "make illegal states
unrepresentable"`; `algorithmic trading risk limit default fail-open position
sizing zero value bug 2025`; `equivalent mutant problem mutation testing
undecidable survived mutant false alarm 2025`; `ruff FURB110 ... falsy default`.

**Result: FOUR new findings in the window, and one of them changes the design.**

- **PEP 661 moved to Final with a resolution date of 2026-04-23, targeting Python
  3.15** -- so a first-class `sentinel()` is now a *standard* answer, not a
  folklore one. Materially: PEP 661 sentinels are **truthy**, which is exactly
  the property a falsy-zero-safe marker needs. This SUPERSEDES the usual
  `_MISSING = object()` advice.
- **Ruff FURB110** (current) is live tooling that nudges toward `x or y`. New
  since the older anti-pattern literature and pointing the opposite way.
- **Oracle DoP v1.1 (2024-06-03)** restates make-illegal-states-unrepresentable
  with a concrete nullable-field-to-sealed-interface recipe.
- **Equivalent-mutant research 2025-2026** (Meta engineering 2025-09-30;
  arXiv 2607.00511 multi-lingual equivalent-mutant detection) -- the field's
  live problem is telling a SURVIVED mutant from an EQUIVALENT one, which is
  undecidable in general and reported at **4%-39%** of mutants in real projects.

**No relevant new finding** in the financial-domain search: the 2025-2026 hits
(`nurp.com`, `3commas.io`, `luxalgo.com`, `algobulls.com`) are all community-tier
marketing blogs with no incident analysis of a falsy-default risk-limit bug.
Recorded as a genuine gap, not padded into the read-in-full table.

---

## Internal inventory (round 2): the existing mutation-testing discipline

`scripts/qa/mutation_matrix_86_74.py` (260 lines) already implements objective
item (d) and is the pattern to reuse rather than re-invent. Its docstring
(`:12-24`) states four rules, each with a stated provenance:

- **Control green FIRST** (`:14-16`): *"A cell whose control was already red
  proves nothing -- the mutant 'failing' is then indistinguishable from the suite
  being broken. Such a cell is scored UNSCORABLE, never KILLED."* Enforced at
  `:203-205`.
- **Byte-identical restore, verified by sha256** (`:17`): *"Not 'looks
  restored'."* Enforced in the `finally` at `:239-244` (per-cell) and again
  globally at `:246`, returning exit 3 on mismatch.
- **The probe must DISCRIMINATE** (`:18-21`): a mutation whose target text is
  absent is scored `NOT_APPLIED`, because *"a no-match `str.replace` looks
  exactly like success"* (`:208-212`).
- **Mutate the SUBJECT, not a copy** (`:22-23`).

Plus a fifth rule added by the 86.74 Q/A -- `selected()` at `:49-70`: pytest
**exits 5 when `-k` selects nothing**, and cells are scored `killed = rc != 0`,
so a typo'd selector would score a zero-assertion run as KILLED. Every cell now
proves its selector is live first (`:218-222`), and the kill test was tightened
to `rc_m == 1` (`:233`) so exit-5 can never read as a kill.

**Gap for 86.86:** `SUBJECTS`/`MUTATIONS` target `portfolio_manager.py` only
(`:32-34`, `SUITE = backend/tests/test_phase_66_2_risk_judge_shape.py`). There is
no cell that restores `or _LITE_RISK_DEFAULT[...]` in `autonomous_loop.py`,
because no test drives the lite ingress construction. `test_phase_66_2_risk_judge_shape.py`
(632 lines) builds its lite fixture as a **hand-written flat dict**
(`_lite_path_analysis`, `:53-60`) -- it never calls the real lite judge, so the
`or` at `:3092` is not exercised by any test in that suite. That is why the
defect survived 86.74's mutation matrix: **the matrix was complete over the
subject it named, and the subject did not include the producer.**


---

## External findings (rounds 4-6)

8. **Meta engineering (2025-09-30): mutants must be REALISTIC, and the
   equivalent-mutant problem is the live obstacle.** *"Where statement or branch
   coverage might still fail to detect a bug if a line still runs, mutation
   testing reveals whether a test fails after inserting a mutation."* Meta
   explicitly rejects generic operators -- traditional *"predefined, rule-based
   mutation operators that apply generic syntactic changes to code"* create
   mutants that *"do not represent faults that developers would realistically
   introduce."* Their ACH tool instead generates problem-specific mutants and
   runs an *"LLM-based Equivalence Detector agent"* (precision 0.79 / recall
   0.47, rising to 0.95 / 0.96 with preprocessing). Trial Oct-Dec 2024:
   *"privacy engineers at Meta accepted 73% of the generated tests."*
   Combined with the survey figure that equivalent mutants are **4%-39%** of all
   mutants and that mutant equivalence is **undecidable in general**, this is the
   caveat for 86.86's matrix: a SURVIVED cell is ambiguous between "the guard is
   not load-bearing" and "the mutant was equivalent". The existing matrix
   sidesteps this by using *restore-the-old-defect* mutants -- the old code is
   known non-equivalent because it produced a different production outcome --
   which is a stronger design than generic operators and should be kept.
   (https://engineering.fb.com/2025/09/30/security/llms-are-the-key-to-mutation-testing-and-better-compliance/,
   accessed 2026-08-15)

9. **"Parse, don't validate" (Alexis King, 2019) -- the canonical answer to
   "route N construction sites through ONE resolver".** Year-less canonical
   query leg. Thesis: design functions to *return refined types that preserve
   the information gained during the check*, rather than returning nothing and
   discarding it. *"Both of these functions check the same thing, but
   `parseNonEmpty` gives the caller access to the information it learned, while
   `validateNonEmpty` just throws it away."* The failure mode it names is
   **exactly** this defect: *"Shotgun parsing is a programming antipattern
   whereby parsing and input-validating code is mixed with and spread across
   processing code"*, with the consequence that *"Late-discovered errors in an
   input stream will result in some portion of invalid input having been
   processed, with the consequence that program state is difficult to accurately
   predict."* Prescription: *"Push the burden of proof upward as far as possible,
   but no further"* -- validate *"at the boundary of your system, before any of
   the data is acted upon."* Takeaway 1 is *"Use a data structure that makes
   illegal states unrepresentable."*
   (https://lexi-lambda.github.io/blog/2019/11/05/parse-don-t-validate/, accessed
   2026-08-15)

10. **The stdlib already ships this exact three-state pattern, and says why.**
    `dataclasses.MISSING` is *"A sentinel value signifying a missing default or
    default_factory"*, and the docs give the rationale verbatim: *"the MISSING
    value is a sentinel object used to detect if some parameters are provided by
    the user. This sentinel is used because `None` is a valid value for some
    parameters with a distinct meaning."* Substitute "0.0" for "None" and that is
    this step in one sentence. The docs also warn *"No code should directly use
    the MISSING value"* -- i.e. the sentinel is an INTERNAL protocol between
    producer and resolver, never part of the public surface. Good precedent for
    keeping `SIZE`/`ABSENT`/`UNPARSEABLE` internal to the sizing seam.
    (https://docs.python.org/3/library/dataclasses.html, accessed 2026-08-15)


## External findings (rounds 7-8)

11. **RFC 7396 (IETF, JSON Merge Patch) treats absent and null as the two most
    different things in the format.** *"If the provided merge patch contains
    members that do not appear within the target, those members are added. If the
    target does contain the member, the value is replaced."* And: *"Null values
    in the merge patch are given special meaning to indicate the removal of
    existing values in the target."* An absent member leaves the target
    unchanged; a null member DELETES it. The RFC then documents the cost of
    having only two states: because null is overloaded as "delete", *"it is not
    possible to patch part of a target that is not an object"* and a value cannot
    be SET to null at all. That is the exact tax this codebase pays -- overload
    one channel with two meanings and one meaning becomes unexpressible.
    (https://www.rfc-editor.org/rfc/rfc7396, accessed 2026-08-15)

12. **Codd himself concluded ONE null is not enough -- and SQL shipped one
    anyway.** *"One kind of NULL marks values which are missing because the value
    is unknown, and the other kind marks values that are missing because the
    attribute is missing."* Illustrated as: *"my hat size exists (I have a head)
    but might not be known so it is the first kind of NULL; my hair color does
    not exist since I am bald as a cue ball, so it is the second kind."* Standard
    SQL implemented only the first model, and *"FirstSQL is the only product
    which implemented Dr. Codd's second model."*
    **This maps 1:1 onto `PositionVerdict`:** unknown-but-exists = `UNPARSEABLE`
    (a verdict was given, we cannot read it), attribute-missing = `ABSENT` (no
    verdict was given), value-present = `SIZE`. The project independently
    re-derived Codd's four-valued (value + 2 nulls) model. The article's remedy
    is prevention -- *"not make a column NULL-able unless it makes sense in the
    data model"* -- and it notes the asymmetry that DDL treats
    `{TRUE, UNKNOWN}` alike while DML treats `{FALSE, UNKNOWN}` alike, i.e. the
    same unknown fails OPEN in one layer and CLOSED in another. Precisely the
    lite-vs-full asymmetry in this codebase.
    (https://www.red-gate.com/simple-talk/databases/sql-server/learn/sql-and-the-snare-of-three-valued-logic/,
    accessed 2026-08-15)

13. **Protocol Buffers: the closest industrial precedent, including the FIX.**
    Google's field-presence guide defines explicit presence as when *"the API
    also stores whether or not a field has been set"* versus implicit presence
    where *"the generated message API stores field values (only)"*. Under
    implicit presence, *"Default values are not serialized"* and *"To 'clear' a
    field, it is set to its default value"* -- so *"the field was explicitly set
    to its default value"* is indistinguishable from *"the field was never set"*.
    The documented consequence is **information loss on round-trip**: with Client
    A setting a field to 0, *"Client A requires (by assert) that the field is
    present; even without any modifications through the API, that requirement
    fails."*
    **Google's fix was not a better default -- it was to restore presence
    tracking.** `optional` was reintroduced for proto3 scalars (*"enabled by
    default since v3.15.0"*), and the standing recommendation is now: *"We
    recommend always adding the `optional` label for proto3 basic types."*
    A singular numeric field has NO presence in bare proto3 and DOES with
    `optional` -- the same field, the same wire, one extra bit of state.
    That is the design verdict for 86.86: carry the presence bit, do not tune the
    default. (https://protobuf.dev/programming-guides/field_presence/, accessed
    2026-08-15)

---

## Consensus vs debate (external)

**Consensus is unusually strong and cross-domain.** Five independent
communities -- Python (PEP 661, `dataclasses.MISSING`), IETF (RFC 7396), Google
protobuf (field presence), relational theory (Codd's second model), and typed FP
(parse-don't-validate, DoP sealed interfaces) -- all converge on: *carry an
explicit presence/kind marker; never infer absence from a value's falsiness*.
The stdlib states the rationale in one line: *"This sentinel is used because
`None` is a valid value for some parameters with a distinct meaning."*

**Where they debate:**
- *Sentinel vs sum type.* PEP 661 delivers a truthy singleton for the
  "one field, one extra state" case; DoP and parse-don't-validate argue for a
  tagged union when there are >=3 states. The project's `PositionVerdict`
  NamedTuple sits with the latter, correctly -- there are three states, and one
  of them carries a payload.
- *[ADVERSARIAL] Ruff FURB110 argues the opposite of everyone else*, telling
  authors that `x if x else y` and `x or y` provide *"the same functionality"*.
  They do not, for falsy-valid values.
- *Prior art gap.* The two mainstream "dict.get" sources (AWS CodeGuru,
  quantifiedcode) are about `KeyError`, not falsiness -- there is no
  authoritative linter or detector for this defect class in Python. Compare
  JavaScript, where the language itself grew `??` because `||` had this bug.
- *Mutation testing's own limit.* Equivalent mutants are undecidable in general
  and 4%-39% of mutants in practice; Meta needed an LLM agent to triage them.
  A SURVIVED cell is therefore not automatically an indictment.

## Pitfalls (from the literature, applied)

1. **Fixing the consumer while the producer still destroys the data.** Protobuf's
   round-trip example is this exact failure: the receiving code is correct and
   the information is already gone.
2. **Overloading one channel with two meanings makes one unexpressible**
   (RFC 7396: you cannot set a value to null).
3. **Shotgun parsing** -- spreading the fix across many sites. King: *"Push the
   burden of proof upward as far as possible, but no further."* Two textually
   parallel lite constructions (`:3085` and `:3333`) are two chances to drift.
4. **A permissive default is a CWE.** CWE-1188/CWE-665; the remedy is to
   *initialise explicitly to the denied state*.
5. **A mutation cell can be vacuous.** Control not green -> UNSCORABLE; target
   text absent -> NOT_APPLIED; `-k` selecting zero tests -> pytest exit 5 scored
   as a KILL. All three already guarded in `mutation_matrix_86_74.py`.
6. **A matrix is only complete over the SUBJECT it names.** 86.74's matrix
   targets `portfolio_manager.py`; the producer was never in scope.

## Application to pyfinagent

- **The fix belongs at the PRODUCER**, `autonomous_loop.py:3085-3097` and
  `:3333-3343` -- upstream of `portfolio_manager.py:1102`, which is where the
  86.74 resolver reads. Anything downstream is already too late (protobuf
  round-trip; parse-don't-validate boundary rule).
- **ONE resolver, not a second idiom.** The two lite constructions are textually
  parallel; a helper (e.g. `_lite_risk_assessment(risk_dict)`) called from both
  is the parse-don't-validate move and removes the drift surface. The project has
  the precedent: 86.74 collapsed four sizing sites into `_sizing_pct`
  (`portfolio_manager.py:1022`) precisely so *"which paths can reach the 10%
  default?" is answerable by reading one function"* (`:1017-1018`).
- **Reuse `PositionVerdict`, don't invent a parallel type.** It already encodes
  SIZE/ABSENT/UNPARSEABLE (`portfolio_manager.py:994-1013`) and `_coerce_pct`
  (`:1078`) already does `raw is None` rather than `if raw:`. The producer should
  emit the state, not a second sentinel vocabulary. (PEP 661 rejects a single
  global `MISSING` for lacking *"contextual meaning"*; Codd's two-null model
  argues the ABSENT/UNPARSEABLE split is right.)
- **Per-site remedy is NOT uniform** -- see table B. Site 3 (`recommended_position_pct`)
  needs three-state treatment. Site 2 (`decision`) needs a fail-CLOSED default
  (CWE-1188): an empty decision must not become `APPROVE_REDUCED`. Sites 1 and 4
  are audit-trail fabrication -- prefer an honest empty/UNKNOWN over a fabricated
  "MODERATE" or a false "parse failed" reason. Site 5 is protective; leaving it is
  defensible, and the brief says so rather than proposing a uniform sweep.
- **Do NOT hang the fix on `paper_risk_judge_shape_fix_enabled`** -- it has zero
  production readers (section E). 86.74's precedent was to make the fix
  unconditional because *"OFF is the shipped production state and OFF is the
  broken one"* (`portfolio_manager.py:1116-1118`).
- **Mutation matrix extension.** Add a cell that restores
  `or _LITE_RISK_DEFAULT["recommended_position_pct"]` at `autonomous_loop.py:3092`
  (and the Gemini twin at `:3338`) -- but that cell is UNSCORABLE until a test
  actually drives the lite ingress. Today `test_phase_66_2_risk_judge_shape.py`
  hand-builds its lite fixture (`:53-60`) and never calls the producer. The test
  must come first, or the cell scores NOT_APPLIED/UNSCORABLE and proves nothing.
  Use restore-the-old-defect mutants (known non-equivalent), not generic
  operators -- per Meta, generic operators *"do not represent faults that
  developers would realistically introduce."*

## External findings (rounds 9-11)

14. **[ADVERSARIAL] arXiv:2604.01483 (2026-04-01), "Type-Checked Compliance:
    Deterministic Guardrails for Agentic Financial Systems Using Lean 4"** --
    argues that shape/schema fixes of the kind 86.86 proposes are *not enough*.
    It names the exact remedy this project uses and calls it **"logically
    shallow"**: Guardrails AI *"focuses on syntactic validation, utilizing the
    RAIL specification and Python-based libraries (like Pydantic) to enforce
    structural consistency"*, but *"Pydantic can guarantee that an agent's
    proposed trade volume is an integer, but it cannot dynamically prove that the
    specific integer, when combined with historical margin usage and real-time
    capital constraints, adheres to a multi-tiered regulatory policy."* Its thesis
    is permit-iff-proven: *"If the kernel cannot verify the proof... the action is
    definitively blocked."*
    **But the adversary is itself weak, and that is the useful part.** The paper
    has **no experimental results, no failure-rate data, no adversarial
    robustness testing**, no limitations section, and -- decisively for a
    fail-closed argument -- **no treatment of what happens when a proof times out
    or cannot be mechanically completed**, nor of `sorry` proof-holes in the
    policy environment. So the correct reading is: a three-state verdict is
    necessary but not sufficient, AND adopting a heavyweight verification
    framework does not by itself buy fail-closed semantics. (This matches the
    prior finding recorded for 86.74 that AgentSpec, arXiv:2503.18666, is
    fail-OPEN on an indeterminate predicate.) The cheap, verifiable win --
    preserve the presence bit at ingress -- is the one actually supported by
    evidence. (https://arxiv.org/html/2604.01483v1, accessed 2026-08-15)

15. **Knight Capital (PRMIA case study, 8pp, read via `pypdf`)** -- $440-460M in
    **45 minutes** on 2012-08-01. Mechanism: *"while the software was installed
    on the majority of the firm's production servers, one of the eight servers
    used in Knight's trading environment did not receive the updated code"*, so
    that server *"activated obsolete legacy software... known as 'Power Peg'"*,
    code that *"had not been properly removed from the production environment and
    had remained dormant within Knight's systems for several years."*
    **The transferable lesson is coverage, not deployment:** a change applied to
    7 of 8 identical sites is the same failure shape as a fix applied to one of
    two identical lite constructions. `autonomous_loop.py:3085` and `:3333` are
    textually parallel twins; patching one is the Knight geometry. Also: *"Knight's
    systems reportedly generated warning messages before and during the event, yet
    these were not acted upon"* -- and here the lite judge DOES log
    `position_pct=%.1f` at `:3100-3104` / `:3346-3350`, so a 0%-verdict-shown-as-3.0
    is already visible in the logs and unactioned.
    NOTE: the primary SEC administrative order (34-70694) was ATTEMPTED and
    **not** fetched -- sec.gov returned `SEC.gov | Request Rate Threshold
    Exceeded`. The PRMIA case study is industry-practitioner tier (hierarchy
    rank 4) and is labelled as such rather than passed off as the SEC source.
    (https://prmia.org/common/Uploaded%20files/eAI/PRMIA%20Case%20study%20-%20Knight%20Trading.pdf,
    accessed 2026-08-15)

---

## Identified but snippet-only (context; does NOT count toward the gate)

| URL | Kind | Why not fetched in full |
|-----|------|-------------------------|
| https://github.com/tc39/proposal-nullish-coalescing | proposal | JS grew `??` because `\|\|` has exactly this bug -- pure cross-language confirmation of finding 1/13, de-dup |
| https://www.sec.gov/files/litigation/admin/2013/34-70694.pdf | regulator | ATTEMPTED, blocked: "SEC.gov \| Request Rate Threshold Exceeded" |
| https://www.ecfr.gov/current/title-17/part-240/section-240.15c3-5 | regulator | ATTEMPTED, 302 to unblock.federalregister.gov; used Cornell LII instead |
| https://github.com/Instagram/LibCST | tool | codemod/CST tooling; stdlib `ast` already derived the class here |
| http://astgrep.com/catalog/python/ | tool | structural search; same, no new finding |
| https://docs.semgrep.dev/writing-rules/pattern-syntax | tool doc | AST pattern matching; Semgrep is not a project dep |
| https://docs.pydantic.dev/latest/concepts/serialization/ | official doc | `model_fields_set`/`exclude_unset` = same presence-tracking idea as `dataclasses.MISSING`, de-dup |
| https://dl.acm.org/doi/10.1145/3183519.3183521 | paper | ACM landing for the Google paper already read via PDF |
| https://arxiv.org/pdf/2607.00511 | paper | multi-lingual equivalent-mutant detection; equivalent-mutant point already made by Meta + survey |
| https://www.sciencedirect.com/science/article/abs/pii/S0950584920300690 | paper | equivalent-mutant evolutionary study, paywalled, de-dup |
| https://web.eecs.umich.edu/~weimerw/2022-481F/readings/mutation-testing.pdf | survey | Papadakis mutation survey; 4%-39% figure already captured |
| https://en.wikipedia.org/wiki/Offensive_programming | encyclopedia | community tier |
| https://functional-architecture.org/make_illegal_states_unrepresentable/ | blog | de-dup of Oracle DoP v1.1 |
| https://khalilstemmler.com/articles/typescript-domain-driven-design/make-illegal-states-unrepresentable/ | blog | de-dup |
| https://nurp.com/algorithmic-trading-blog/7-risk-management-strategies-for-algorithmic-trading/ | marketing blog | community tier, no incident analysis |
| https://3commas.io/blog/ai-trading-bot-risk-management-guide-2025 | marketing blog | community tier |
| https://www.luxalgo.com/blog/risk-management-strategies-for-algo-trading/ | marketing blog | community tier |
| https://pypi.org/project/flake8-boolean-trap/0.1.0 | tool | confirms NO Python detector exists for this class -- de-dup of finding 3 |
| https://www.finra.org/rules-guidance/notices/15-09 | regulator | 15c3-5 language already obtained via Cornell |
| https://discuss.python.org/t/pep-661-sentinel-values/9126 | forum | PEP 661 discussion thread; PEP itself read in full |
| https://death.andgravity.com/sentinels | blog | sentinel typing patterns; superseded by PEP 661 Final |
| https://engineering.instawork.com/refactoring-a-python-codebase-with-libcst-fc645ecc1f09 | blog | codemod practice, no new finding |
| https://www.rfc-editor.org/info/rfc7396/ | metadata | info page for the RFC read in full |

**URLs collected: 39 unique** (16 read in full + 23 snippet-only).

---

## Internal code inventory

| File | Lines | Role | Status |
|---|---|---|---|
| `backend/services/autonomous_loop.py` | `:2297-2303` `_LITE_RISK_DEFAULT` | the 5-key default dict; `decision=APPROVE_REDUCED`, `pct=3.0` | **permissive default (CWE-1188 shape)** |
| " | `:3074-3097` Claude lite judge | parses judge JSON, builds `risk_assessment` | **DEFECT SITE (producer)** |
| " | `:3320-3343` Gemini lite judge | textually parallel twin | **DEFECT SITE (producer) -- the Knight geometry** |
| " | `:2346-2371` `_data_integrity_blocked_analysis` | writes `recommended_position_pct: 0.0` as a plain literal, no `or` | **SAFE -- the zero survives** |
| " | `:3379` `_persist_analysis` | persists the already-collapsed dict | no new defaulting; inherits the loss |
| " | `:3431-3435` | `analysis.get("risk_assessment") or {}` + nested `judge` resolve | benign (`{}` vs absent are equivalent here) |
| `backend/services/portfolio_manager.py` | `:994-1013` `SIZE`/`ABSENT`/`UNPARSEABLE` + `PositionVerdict` | the three-state type | correct; REUSE, do not duplicate |
| " | `:1019` `DEFAULT_POSITION_PCT = 10.0` | named constant | fine |
| " | `:1022-1091` `_sizing_pct` | THE single 10%-default seam | correct; 4 call sites `:531/:824/:877/:902` |
| " | `:1078-1091` `_coerce_pct` | `raw is None` not `if raw:` | correct |
| " | `:1093-1109` `_resolve_position_pct` | reads `recommended_position_pct` | **reads the ALREADY-COLLAPSED value** |
| " | `:1111-1124` `_extract_position_pct` | legacy `Optional[float]` shim | fixed unconditionally |
| " | `:340-350`, `:384-400`, `:409` | call site; REJECT gate; `position_pct_state` write | the single state-write site |
| `backend/services/signal_attribution.py` | `:229-247` | persists the judge verdict to `factors_json` | guard already `pos_pct is not None` |
| `backend/config/settings.py` | `:342/:346/:350` | the three flags | `shape_fix` has **ZERO** prod readers |
| `backend/tests/test_phase_66_2_risk_judge_shape.py` | 632 | 66.2/86.74 suite | lite fixture is hand-built (`:53-60`); **never drives the producer** |
| `scripts/qa/mutation_matrix_86_74.py` | 260 | mutation matrix | subject = `portfolio_manager.py` only; **no producer cell** |
| `backend/agents/risk_debate.py` | `:132-162` | FULL-path parse-fail fallback | writes a real `0`; 86.74 fix binds here |

**Internal files inspected: 8.**

---

## Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch/pypdf -- **16**
- [x] 10+ unique URLs total -- **39**
- [x] Recency scan (last 2 years) performed + reported -- 4 findings in window
- [x] Full papers / pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every module in the caller's scope
- [x] Contradictions / consensus noted (FURB110 adversarial; arXiv:2604.01483
      adversarial; the CodeGuru/quantifiedcode prior-art gap)
- [x] All claims cited per-claim with URL + access date

## Adaptive coverage log (audit-class, K=2)

| Round | Angle | New read-in-full findings |
|---|---|---|
| 1 | PEP 661; `or`-default anti-pattern | 4 |
| 2 | mutation testing; Ruff | 2 |
| 3 | illegal-states; financial regulation | 2 |
| 4 | equivalent mutants (Meta) | 1 |
| 5 | one-resolver / boundary parsing | 1 |
| 6 | stdlib sentinel prior art | 1 |
| 7 | absent-vs-null (IETF); three-valued logic | 2 |
| 8 | protobuf field presence | 1 |
| 9 | LLM structured output; pydantic presence | **0 (DRY 1)** |
| 10 | Python linters; SEC/FINRA incidents | 2 |
| 11 | LibCST / ast-grep tooling | **0 (DRY 1)** |
| 12 | absent-treated-as-zero incident sweep | **0 (DRY 2)** |

Rounds 9 and 11 were dry but non-consecutive (round 10 surfaced arXiv:2604.01483
and the Knight case study, so the loop correctly did NOT stop there). **Rounds 11
and 12 are consecutive dry rounds -> `coverage.dry = true`.**
