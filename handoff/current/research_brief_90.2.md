# Research Brief -- step 90.2

**Tier:** moderate (caller-specified). **Audit-class:** NO (coverage reported for information only).
**Topic:** Caller-side routing of severity that an LLM judge already emits in free text --
returning a routing decision ALONGSIDE a judge's structured object without mutating it,
sibling-leak invariants, replay/confusion-table validation over historical returns, and why a
caller must READ the judge's own classification rather than RE-DERIVE severity itself.

**Status marker (phase-86.37).** Born inert as `INCOMPLETE` in the first tool call of this session and
flipped to `COMPLETE` as the final act. `urls_collected` is re-derived from this file, not carried:
18 distinct URL strings appear below, of which `rfc9413.txt` and `rfc9413.html` are the SAME document,
so the de-duped count is **17** -- the lower of the two is claimed.

```json
{
  "brief_status": "COMPLETE",
  "tier": "moderate",
  "external_sources_read_in_full": 7,
  "snippet_only_sources": 10,
  "urls_collected": 17,
  "recency_scan_performed": true,
  "internal_files_inspected": 9,
  "coverage": {
    "audit_class": false,
    "rounds": 1,
    "dry_rounds": 0,
    "K_required": 2,
    "new_findings_last_round": 7,
    "dry": false
  },
  "gate_passed": true
}
```

---

## Search-query composition (three-variant discipline, per .claude/rules/research-gate.md)

Year-less canonical: "OpenTelemetry logs data model SeverityText SeverityNumber original severity preserved";
"SARIF result level property severity"; "robustness principle harmful explicit signalling";
"JSON Schema additionalProperties false closed schema extension"; "confusion matrix shadow deployment
replay validation classifier".
Current-year frontier (2026) + last-2-year window (2024-2025): see the Recency scan section.

## Read in full (>=5 required; counts toward the gate)

| # | URL | Accessed | Kind | Fetched how | Key finding |
|---|-----|----------|------|-------------|-------------|
| S1 | https://opentelemetry.io/docs/specs/otel/logs/data-model/ | 2026-08-20 | official spec | WebFetch (full) | Severity carried on TWO sibling fields: `SeverityText` preserves the source's own string; `SeverityNumber` is the consumer's normalization. Additive, never destructive. |
| S2 | https://docs.oasis-open.org/sarif/sarif/v2.1.0/errata01/os/sarif-v2.1.0-errata01-os-complete.html | 2026-08-20 | OASIS standard | WebFetch (full) + curl/grep verbatim verification | `level` (producer severity, closed enum none/note/warning/error) is a SEPARATE field from `kind` (outcome) and from `rank` (consumer priority); absent severity resolves by a WRITTEN default ladder. The summariser's "consumer SHALL NOT re-derive" quote is FABRICATED -- see S2 note. |
| S3 | https://www.rfc-editor.org/rfc/rfc9413.html | 2026-08-20 | IETF/IAB RFC | WebFetch (full) + curl/grep of the .txt, 5/5 quotes verified | "inferred intent of the sender" is named as the anti-pattern; tolerating unspecified input starts a "pathological feedback cycle"; the remedy is Virtuous Intolerance -- fatal errors over error recovery. |
| S4 | https://json-schema.org/understanding-json-schema/reference/object | 2026-08-20 | official docs | WebFetch (full) | `additionalProperties:false` closes the object and "only recognizes properties declared in the same subschema", so it "can restrict you from 'extending' a schema using combining keywords such as allOf". |
| S5 | https://arxiv.org/html/2604.16706 | 2026-08-20 | preprint (2026) | WebFetch (full) + curl/grep verbatim verification of every kappa | **Substring-heuristic judging agrees with humans "only at chance level (Cohen's kappa = 0.049)"**, vs 0.432 (3-LLM ensemble), 0.567 (single judge), 0.835 (human-human). Propagation cascade = the modal silent failure. |
| S6 | https://arxiv.org/html/2603.05399 | 2026-08-20 | preprint (2026) | WebFetch (full) | Judge Reliability Harness: formatting perturbations hurt more than semantic ones; multi-level ORDINAL scoring degrades "substantially" vs binary; validate the judge BEFORE deployment, not by post-hoc adjustment of its output. |
| S7 | https://www.anthropic.com/engineering/building-effective-agents | 2026-08-20 | official vendor eng. blog | WebFetch (full) | "Routing classifies an input and directs it to a specialized followup task" -- the benefit is "separation of concerns"; evaluator-optimizer keeps the evaluator separate from the actor; avoid abstraction that obscures prompts/responses. |

## Identified but snippet-only (context; does NOT count toward the gate)

| URL | Kind | Why not fetched in full |
|-----|------|-------------------------|
| https://opentelemetry-python.readthedocs.io/en/latest/api/_logs.severity.html | impl. docs | Implementation of S1; spec governs |
| https://github.com/open-telemetry/opentelemetry-collector-contrib/blob/main/pkg/stanza/docs/types/severity.md | impl. docs | Collector-side mapping detail; S1 is the normative text |
| https://github.com/open-telemetry/opentelemetry-specification/blob/v1.59.0/oteps/logs/0097-log-data-model.md | OTEP | Superseded by the published spec (S1) |
| https://www.dash0.com/knowledge/opentelemetry-logging-explained | vendor blog | Tier-4; restates S1 |
| https://uptrace.dev/opentelemetry/logs | vendor blog | Tier-4; restates S1 |
| https://arxiv.org/html/2606.01629v1 | preprint (2026) | Long-form judge benchmarking; adjacent, not about caller-side routing |
| https://www.openlayer.com/blog/llm-as-judge-evaluation-guide | vendor blog | Tier-4; its Cohen's-kappa calibration advice is already covered by S5 |
| https://futureagi.com/blog/llm-as-a-judge/ | vendor blog | Tier-4 |
| https://labelyourdata.com/articles/llm-as-a-judge | vendor blog | Tier-4 |
| https://www.mlaidigital.com/blogs/the-ultimate-guide-to-llm-as-a-judge-in-2026 | vendor blog | Tier-4 |

## Recency scan (2024-2026)

(pending)

## Key findings

(pending)

## Internal code inventory

| File | Lines | Role | Status |
|------|-------|------|--------|


---

## S1 -- OpenTelemetry Logs Data Model (official spec) -- READ IN FULL

URL: https://opentelemetry.io/docs/specs/otel/logs/data-model/ (accessed 2026-08-20, WebFetch, official spec)

The single closest prior-art match to this step's design question. OTel carries severity on **two
sibling fields, not one**:

- `SeverityText` -- "the original string representation of the severity **as it is known at the
  source**" (Severity Fields section). The emitter's own words are PRESERVED, never overwritten.
- `SeverityNumber` -- "numerical value of the severity, **normalized** to values described in this
  document", a 1-24 scale bucketed TRACE 1-4 / DEBUG 5-8 / INFO 9-12 / WARN 13-16 / ERROR 17-20 /
  FATAL 21-24.

Design consequences the spec states directly:
1. **Normalisation is additive, not destructive.** The consumer computes a comparable number and
   stores it BESIDE the source's own string; both survive into the record.
2. **Ordering is preserved within a band** ("If the source format has more than one severity that
   matches a single range ... the severities of the source format must be assigned numerical values
   from that range according to how severe (important) the source severity is"), so a 3-value
   producer taxonomy maps into the normalized scale without collapsing its internal order.
3. **Single-severity sources get the smallest value of the range** -- an explicit, deterministic
   tie-break rather than an inferred one.
4. **Both are displayed** -- "A recommended combined string ... begins with the short name followed
   by `SeverityText` in parenthesis" (e.g. `INFO (Informational)`), i.e. the audit trail shows the
   caller's normalisation AND the emitter's original, so a mis-normalisation is visible.
5. **Unmappable values degrade to nearest-in-range**, not to a silent drop.

## S2 -- OASIS SARIF v2.1.0 (errata01) result object, sections 3.27.9 / 3.27.10 / 3.27.25 -- READ IN FULL

URL: https://docs.oasis-open.org/sarif/sarif/v2.1.0/errata01/os/sarif-v2.1.0-errata01-os-complete.html
(accessed 2026-08-20; WebFetch + `curl` + tag-stripped grep of the 1,630,711-byte source to verify
quotes verbatim)

**HONESTY NOTE, and it is itself a finding for this step.** The WebFetch summariser returned two
authoritative-sounding sentences -- *"A SARIF consumer SHALL NOT override a level value that a SARIF
producer has explicitly specified. A SARIF consumer SHALL NOT re-derive the level value."* -- attributed
to section 3.27.10. **Neither sentence exists.** The string `re-derive` does not appear anywhere in the
1.6 MB specification, and the only `SHALL NOT override` in the document is section 3.58.6 (notification
objects), reading: *"Because a notification whose level property is "error" describes a failed run, an
analysis tool SHALL NOT override the severity of such a notification."* A summariser asked to
characterise someone else's classification **re-derived it and invented the sentence the reader wanted**
-- a live instance of the exact failure mode this step's criterion 4 is about. (Consistent with the
standing memory `reference_webfetch_pdf_summaries_fabricate_quotes`; it is not PDF-specific.)

Verbatim spec text (verified by grep):

- **3.27.10 level property** -- "A result object MAY contain a property named `level` whose value is
  one of a fixed set of strings that specify the severity level of the result": `"warning"` (a problem
  was found), `"error"` (a serious problem), `"note"` ("a minor problem or an opportunity to improve
  the code"), `"none"` ("The concept of severity does not apply to this result because the `kind`
  property has a value other than `"fail"`").
- **3.27.9 kind property** -- a SEPARATE axis: `"pass"` / `"open"` / `"informational"` /
  `"notApplicable"` / `"fail"`. Severity and outcome-kind are deliberately two orthogonal fields.
- **Cross-field invariant** -- "If `kind` has any value other than `"fail"`, then if `level` is absent,
  it SHALL default to `"none"`, and if it is present, it SHALL have the value `"none"`." A sibling pair
  is kept CONSISTENT by an explicit rule, not by hope.
- **Absent-severity resolution is a deterministic ladder, not an inference** -- when `kind == "fail"`
  and `level` is absent, the spec prescribes: configurationOverride level, else
  `theDescriptor.defaultConfiguration.level`, else "IF level has not yet been set THEN SET level to
  `"warning"`". A *specified default*, never an ad-hoc guess from the message text.
- **3.27.25 rank property** -- the consumer-facing priority axis, explicitly distinguished: "rank is
  only meaningful if `kind` has the value `"fail"`", absent defaults to `-1.0` "which indicates that
  the value is unknown (not set)", with the NOTE that "rank values produced by different tools are in
  general not commensurable."

So the correct, *sourceable* SARIF lesson is NOT a no-re-derive commandment. It is four structural
ones: (a) severity and outcome are separate fields; (b) severity's enum is closed and defined per
value; (c) when the producer omits severity the consumer follows a WRITTEN default ladder ending in a
named constant, and (d) the consumer's own priority ordering (`rank`) lives in a DIFFERENT field from
the producer's severity (`level`), with an explicit "unknown" sentinel (`-1.0`) distinct from "lowest".

## S3 -- IAB RFC 9413, "Maintaining Robust Protocols" (2023) -- READ IN FULL

URL: https://www.rfc-editor.org/rfc/rfc9413.html (accessed 2026-08-20, WebFetch; all quotes below
re-verified verbatim against https://www.rfc-editor.org/rfc/rfc9413.txt, 36,103 bytes, 5/5 found)

The canonical modern statement of *"do not infer what the peer meant; make the peer say it."*

- 2.3: "Some interpretations even suggest that a faulty or ambiguous message be processed according
  to the **inferred intent of the sender**" -- and the RFC's position is that "an interpretation
  that advocates for tolerating unexpected inputs is no longer considered best [practice]".
- 4.1 (Protocol Decay): "An implementation that reacts to variations in the manner recommended in
  the robustness principle enters a **pathological feedback cycle**. Over time: Implementations
  progressively add logic to constrain how data is transmitted or to permit variations in what is
  received. Errors ... are permitted or ignored. These errors can become entrenched, forcing other
  implementations to be [bug-compatible]."
- 4.2 (Ecosystem Effects): "if non-compliance is tolerated by existing implementations,
  non-compliant implementations can be deployed successfully. Newer implementations then have a
  strong incentive to tolerate any existing non-compliance."
- 5.1 (**Virtuous Intolerance**): "A well-specified protocol includes rules for consistent handling
  of aberrant conditions. ... **Choosing to generate fatal errors for unspecified conditions instead
  of attempting error recovery can ensure that faults receive attention.**"

Mapped onto this step: a caller that parses the judge's prose for severity IS "processing according
to the inferred intent of the sender". Each parser fix (add a negation guard, add a synonym) is one
turn of the 4.1 decay cycle. The RFC-compliant design is (a) a field the judge fills explicitly,
(b) a caller that reports `unparseable` / `not_supplied` loudly rather than recovering, which is
exactly `enforceEscalation`'s existing `sequence_status` ladder
(`.claude/workflows/qa-verdict.js:621-630`).

## S4 -- JSON Schema, "Objects" reference (official docs) -- READ IN FULL

URL: https://json-schema.org/understanding-json-schema/reference/object (accessed 2026-08-20, WebFetch)

- "Setting the `additionalProperties` schema to `false` means no additional properties will be
  allowed." -- so with `VERDICT_SCHEMA.additionalProperties === false`
  (`.claude/workflows/qa-verdict.js:441`) and `violation_details.items.additionalProperties === false`
  (`:452`), a severity key CANNOT be smuggled in at runtime. It is a schema edit or nothing.
- "`additionalProperties` only recognizes properties declared in the **same subschema** as itself",
  and therefore "`additionalProperties` can restrict you from 'extending' a schema using combining
  keywords such as `allOf`". A closed schema is deliberately hostile to bolt-on extension.
- `unevaluatedProperties` "is similar to `additionalProperties` except that it can recognize
  properties declared in subschemas" -- the standard escape hatch. **Not usable here**: Anthropic
  structured outputs constrains which keywords survive to the wire, and the project has already
  measured that constraint keywords are stripped (`.claude/rules/research-gate.md`, "Launch" section).

Consequence for design: the routing decision has exactly two legal homes -- (1) a NEW optional
property inside `VERDICT_SCHEMA` that the JUDGE fills, or (2) a caller-computed SIBLING key on the
returned object, outside the schema entirely. The shipped precedent uses BOTH, split by authorship:
phase-86.72 put `research_needed` + `research_brief_spec` INSIDE the schema as OPTIONAL (absent from
`required`, `:466-486`) because the JUDGE authors them, and put the derived routing OUTSIDE as the
sibling `research_routing` (`:834-835`) because the CALLER computes it.

---

## Internal measurement: the replay corpus and the confusion tables

Corpus: `~/.claude/projects/-Users-ford--openclaw-workspace-pyfinagent/*/workflows/wf_*.json`.

| Quantity | Measured value |
|---|---|
| Total workflow run records on this machine | 618 |
| `workflowName == 'qa-verdict'` (exact) | **436** |
| `workflowName.startsWith('qa-verdict')` (adds `qa-verdict-writefirst-82-5` x3, `-82-7` x2) | **441** |
| ...of those, with a parseable dict `result` | 398 (9.8% carry none -- drops/errors) |
| ...with a non-null `verdict` (**the replay denominator**) | **397** |
| Verdict base rates | CONDITIONAL 221 (55.7%), PASS 109 (27.5%), FAIL 67 (16.9%) |
| `violation_details` rows total | 969 |
| ...carrying an explicit `severity` KEY | **0** (the closed schema is airtight) |
| ...whose free text mentions a severity word | 703 |
| Records with an UPPERCASE BLOCK/WARN/NOTE token anywhere | 365 / 398 |
| Uppercase token counts | WARN 931, NOTE 551, BLOCK 310 |

**The caller's "441" is only reproducible with a `startsWith` filter; an exact-match filter yields
436, and the replay denominator is 397, not 441.** State the predicate in the checker.

### Confusion table A -- re-deriving severity by TOKEN PRESENCE (worst token wins)

Signal present on 360/397. Rows = re-derived severity, columns = the judge's actual verdict.

| re-derived | FAIL | CONDITIONAL | PASS |
|---|---|---|---|
| BLOCK (expects FAIL) | 32 | **116** | **35** |
| WARN (expects CONDITIONAL) | 25 | 70 | 16 |
| NOTE (expects PASS) | 6 | 18 | 42 |
| (no token) | 4 | 17 | 16 |

- Agreement with the SKILL.md dispatch: **144/360 = 40.0%**.
- Majority-class baseline on the same 360 (always guess CONDITIONAL): **204/360 = 56.7%**.
  **The re-derivation is 16.7 points WORSE than a constant.**
- Cohen's kappa = **0.129** (po=0.400, pe=0.311) -- "slight" agreement.
- **Mechanism:** of the 183 records containing the token `BLOCK`, **130 (71.0%) contain a NEGATED
  occurrence** within 25 characters ("no BLOCK, no WARN", "no BLOCK or WARN fired", "this is
  NOTE/WARN, not BLOCK"). The dominant use of the token is the judge reporting the heuristic did
  **not** fire. A presence-based extractor reads the negation as the assertion.

### Confusion table B -- re-deriving from the LITERAL `severity=` / `severity:` form

Signal present on only 63/397 (15.9%).

| re-derived | FAIL | CONDITIONAL | PASS |
|---|---|---|---|
| BLOCK | 7 | 4 | 0 |
| WARN | 2 | 43 | 2 |
| NOTE | 0 | 3 | 2 |

- Agreement **52/63 = 82.5%** -- high precision, but **84.1% of the corpus has no such signal at all**.
- The same regex also captures 8 junk values out of 167 total captures (4.8%): `P`, `BLOCKING`,
  `CAPPING`, `ONE`, `THE`, `SOLE`, `A`, `_QA_SEV`.

**Design-deciding conclusion:** neither extractor is fit to route on. Free-text re-derivation is
either recall-complete and worse-than-constant (40.0%, kappa 0.129), or precise and blind to 84% of
runs. The judge must EMIT the classification into a dedicated field; the caller routes on that field
and reports an explicit ABSENT state for the ~9.3% of runs that carry no severity signal at all.

## S5 -- AgentProp-Bench (arXiv 2604.16706, 2026) -- READ IN FULL

URL: https://arxiv.org/html/2604.16706 (accessed 2026-08-20, WebFetch; **all four kappa figures
re-verified verbatim** by curl + tag-strip grep of the HTML, 45,260 chars of extracted text)

Verbatim abstract text: *"substring-heuristic judging of agent outputs agrees with human annotation
**only at chance level (Cohen's kappa = 0.049** against each of two annotators), while a three-LLM
ensemble reaches moderate agreement (kappa = 0.432) and a single GPT-4o-mini judge is in fact the
strongest (kappa = 0.567); dual-annotator agreement is almost perfect (kappa = 0.835)."*

- **This is the external corroboration of Confusion table A.** A substring heuristic over free text
  is at chance (0.049) where an actual classifier is 0.43-0.57 and humans are 0.835. This repo's
  token-presence re-derivation lands at kappa=0.129 -- the same regime, one order of magnitude below
  the classifier it is trying to substitute for. n=14,750 traces across 13 agents, 4 domains.
- **"Propagation cascade (the modal failure)"** (Error analysis): the corrupted value "executes and
  the agent reports Manchester's conditions as London's -- an S1->S2->S3 chain **with no visible
  uncertainty**". A mis-routed severity has exactly this shape: silent, confident, downstream.
- Methodology to copy: an **ensemble-vs-human confusion matrix (Table 3, n=100)**, with the dominant
  disagreement cell named; stage probabilities computed from *independent data columns* to avoid
  circular validation; and a **concurrent no-interceptor control** arm.
- Honest limit, stated by the paper: it validates the judge against humans, **not** whether a
  downstream consumer correctly re-reads the judge's own label. That specific gap is what step 90.2
  measures, so this repo's confusion tables are net-new evidence, not a replication.

## S6 -- Judge Reliability Harness (arXiv 2603.05399, 2026) -- READ IN FULL

URL: https://arxiv.org/html/2603.05399 (accessed 2026-08-20, WebFetch)

- **Formatting perturbations degrade judges more than semantic ones**; "judges that are brittle to
  such differences risk embedding instability into downstream model comparisons". Directly argues
  against making the routing depend on the judge's *prose formatting*.
- **Multi-level ordinal scoring performs "substantially" worse than binary classification.** A
  3-level BLOCK/WARN/NOTE ordinal is therefore the harder regime, and the replay validation should
  report per-level cells, not a single accuracy number.
- Recommendation: validate the judge configuration **before deployment** rather than "post-hoc
  adjustment of judge outputs" -- i.e. fix the emission, do not patch the parse.
- Caveat recorded for honesty: this paper reports **accuracy without a majority-class baseline**, so
  it cannot be cited for the baseline argument. The baseline comparison in this brief is computed
  locally (56.7% constant vs 40.0% re-derived).

## S7 -- Anthropic, "Building Effective Agents" -- READ IN FULL

URL: https://www.anthropic.com/engineering/building-effective-agents (accessed 2026-08-20, WebFetch)

- **"Routing classifies an input and directs it to a specialized followup task"**, and the benefit is
  "separation of concerns, and building more specialized prompts". Use it "where there are distinct
  categories that are better handled separately, **and where classification can be handled
  accurately**" -- the accuracy precondition is exactly what Confusion table A fails and what a
  judge-emitted field would satisfy.
- Evaluator-optimizer: "One LLM call generates a response while another provides evaluation and
  feedback in a loop" -- the evaluator is structurally separate from what acts on the evaluation.
- Avoid abstractions that "obscure the underlying prompts and responses, making them harder to
  debug" -- an argument for echoing the raw judge-supplied severity back in the routing object
  (as `enforceEscalation` already does with `sequence_supplied`, `.claude/workflows/qa-verdict.js:599`).

---

## Recency scan (2024-2026) -- PERFORMED

Queries: "LLM-as-a-judge structured output severity label versus free-text rationale parsing
reliability **2026**"; the year-less canonical variants listed at the top; plus the year-less
standards searches (OTel / SARIF / robustness principle) that surfaced S1-S4.

**Result: 2 new findings that materially complement -- and one that partially supersedes -- the
canonical sources.**

1. **S5 (2026) supersedes the intuition, not just complements it.** Before this paper the claim "a
   caller should not re-derive severity from prose" rested on standards-body design preference
   (S1-S3). S5 supplies a *measured* number for the substring heuristic (kappa=0.049, chance) on
   n=14,750 traces. The argument is now empirical, not aesthetic.
2. **S6 (2026)** adds the ordinal-vs-binary caution: a 3-level severity is measurably the harder
   judging regime, which argues for per-level replay cells and against a single accuracy headline.
3. **No 2024-2026 source was found that studies caller-side re-reading of a judge's own severity
   label** -- S5 explicitly does not measure it. The nearest prior art remains the standards
   (S1/S2) and the in-repo sibling rails. Treat step 90.2's confusion tables as the primary
   evidence, not as a replication of published work.

The canonical/year-less sources (S1 OTel spec, S2 SARIF 2019/errata, S3 RFC 9413 2023, S4 JSON
Schema, S7 2024) are NOT superseded: all four standards remain current and none has a competing
successor for the two-field severity pattern.

---

## Internal code inventory (every claim carries a file:line anchor)

| File | Anchor | Role | Status |
|---|---|---|---|
| `.claude/workflows/qa-verdict.js` | 858 lines | The rail. Prompt build, `VERDICT_SCHEMA`, caller-side enforcement, sibling merge | LIVE |
| " | `:441` | `VERDICT_SCHEMA.additionalProperties: false` | closed -- no severity key can appear at runtime |
| " | `:452`, `:455` | `violation_details.items.additionalProperties:false`; closed 7-value `violation_type` enum, **no severity member** | measured: **0 of 969** historical rows carry a `severity` key |
| " | `:466-486` | phase-86.72 precedent: `research_needed` + `research_brief_spec` added as **OPTIONAL** (absent from `required` at `:442`), authored by the JUDGE | the pattern a severity field should copy |
| " | `:593-649` | `enforceEscalation(verdict, sequence, opts)` -- pure, caller-side | the pattern criterion 1 points at |
| " | `:597-619` | Output shape: echoes its input (`sequence_supplied`), an explicit `sequence_status`, `burden_on`, `override`/`override_reason` = null, and a COMPUTED `judge_was_told_consequence` | idiom: echo the input, never re-derive it |
| " | `:621-630` | Status ladder `not_supplied` / `unusable` / `unparseable`, each returning **null, never 0** ("FAILS CLOSED", `:572-573`) | the ABSENT-state idiom to reuse |
| " | `:779-798` | `enforceResearchRouting(verdict)` -- reads the judge's OPTIONAL field and emits caller-facing guidance; "can neither author the signal nor alter the verdict" (`:773-774`) | the closest structural template for severity routing |
| " | `:835` | `const merged = { ...verdict, escalation, research_routing }` | siblings, never merged into the verdict |
| " | `:836-843` | **Leak invariant #1**: `Object.keys(escalation).filter(k => k !== 'escalation' && k in merged)` -> throws | runtime, not just a test (`:818-823`: flattening SURVIVED the whole checker when it was prose-only) |
| " | `:844-854` | **Leak invariant #2** for `research_routing`, with an explicit allow-list carve-out for `research_needed` / `research_brief_spec` because "the judge authored them there" | the exact precedent for a judge-authored severity field |
| " | `:855-858` | `verdict_unmodified` = `Object.keys(verdict).every(k => merged[k] === verdict[k])` -- **computed, not attested** | extend this, do not duplicate it |
| " | `:809-815` | A null/non-object return is returned bare: "A DROPPED RAIL IS NO VERDICT, NEVER PASS" | any new sibling must sit AFTER this guard |
| " | `:335-337` | `KNOWN_ARG_KEYS` + `UNKNOWN_ARG_KEYS` warning (`:495-499`) | a new `args` key must be registered here or it reaches nothing |
| `.claude/skills/code-review-trading-domain/SKILL.md` | `:24-32` | The severity dispatch table: BLOCK->Auto-FAIL, WARN->Force CONDITIONAL, NOTE->PASS-with-flag; `:32` "verdict = worst severity hit" | LIVE, preloaded at spawn |
| " | `:29` | Already instructs "severity=`WARN` in details" | **followed in only 63/397 (15.9%) of historical returns** |
| `.claude/agents/qa.md` | `:499-503` | A SECOND severity vocabulary: "BLOCKING violation" / "WARN-level finding" | vocabulary drift risk vs SKILL.md's BLOCK/WARN/NOTE |
| " | `:845-854` | Records that the skill is "preloaded into this Q/A subagent's context at spawn via the `skills:` frontmatter entry" | the judge does receive the dispatch rule |
| `.claude/workflows/research-gate.js` | `:568` | `brief_status_in_brief` enum `['COMPLETE','INCOMPLETE','ABSENT']` -- a three-valued read of the emitter's own marker with an explicit ABSENT member | the sibling rail's "unknown" sentinel |
| " | `:579`, `:769-770` | `enforceGate` returns `{gate_passed, violations[], checks[], agent_self_reported_gate_passed, self_report_disagreed}` | **the canonical "read the emitter's claim, recompute, and REPORT the disagreement" shape** |
| `scripts/qa/verify_escalation_86_78.mjs` | 24,807 bytes | Checker template: drives the REAL function via temp `export {...}` re-export, `EXPECTED_CHECKS = 49` cardinality floor, `sourceOverride` mutation seam, brace-matching `extractFn` | model `verify_severity_routing_90_2.mjs` on this |
| `handoff/verdict_ledger.jsonl` | 138 rows | `{step_id, cycle, verdict, run_id, recorded_by, date}` -- **no severity column** | a severity field would need its own decision about whether it enters the ledger |
| run-record corpus | 618 files | `*/workflows/wf_*.json`; keys incl. `workflowName`, `args`, `result`, `error`, `status` | the replay input; see the measurement section above |

---

## Consensus vs debate (external)

**Consensus (4 independent standards + 1 vendor + 1 preprint):** severity belongs in its own
explicitly-typed field emitted by the party that determined it; the consumer normalises/routes in a
*separate* field; an absent value gets a *named* sentinel rather than an inferred one. OTel
(SeverityText/SeverityNumber), SARIF (level/kind/rank + the `-1.0` unknown), RFC 9413 (do not infer
sender intent), Anthropic (routing = classify, then direct), and this repo's own `research-gate.js`
(`self_report_disagreed`) all land on the same shape independently.

**Debate / open questions:**
1. **Does the consumer ever get to override?** SARIF forbids overriding only in one narrow case
   (3.58.6 notifications whose `level` is "error"); OTel is silent; RFC 9413's "virtuous intolerance"
   implies *reject*, not *override*. So "never override" is a defensible project choice but is NOT
   a quotable standards mandate -- and the fabricated SARIF quote in S2 shows how easy it is to
   believe otherwise.
2. **Where does a missing severity default?** SARIF defaults to `"warning"` (a real severity);
   OTel maps to nearest-in-range; this repo's `enforceEscalation` returns `null` and fails closed.
   These conflict. `null` + a `status` string is the more conservative choice and matches the
   in-repo idiom, but it is a CHOICE and should be stated as one.
3. **Ordinal vs binary** (S6): the 3-level BLOCK/WARN/NOTE ordinal is the measurably harder regime.

## Pitfalls (from the literature + the corpus)

1. **Negation inversion.** 130 of 183 records mentioning `BLOCK` (71.0%) contain a negated mention.
   Any presence-based extractor reads "no BLOCK fired" as BLOCK.
2. **Worse than a constant.** 40.0% vs a 56.7% majority-class baseline -- always report the baseline
   beside the accuracy, or a bad extractor looks acceptable.
3. **Vocabulary drift.** BLOCK vs BLOCKING vs "NOTE-LEVEL" vs "NOTE only" appear in the corpus
   (`BLOCKING` twice as a literal severity value); `.upper()` folds case, not separators or suffixes.
4. **A summariser will invent the rule you were hoping for** -- demonstrated in S2 on a 1.6 MB
   normative spec. Verify load-bearing quotes at source.
5. **Coverage illusion.** The precise extractor (82.5%) covers 15.9% of runs; reporting only its
   accuracy would hide that 84.1% of the corpus is silent.
6. **Denominator drift.** 441 vs 436 vs 398 vs 397 are four different, all-defensible counts of "the
   qa-verdict corpus". Pin the predicate in the checker or the replay is unreproducible.
7. **Sibling leak.** Flattening the routing object survived an entire checker once
   (`qa-verdict.js:818-823`); the invariant must be a RUNTIME throw, and the new one must be added
   to the same guard family rather than replacing it.
8. **A single global accuracy hides per-level failure** (S6's ordinal finding + S5's Table-3
   "dominant disagreement cell" methodology).

## Application to pyfinagent (step 90.2)

1. **Have the judge EMIT, do not have the caller PARSE.** Add an OPTIONAL severity field to
   `VERDICT_SCHEMA` exactly as phase-86.72 added `research_needed` (`qa-verdict.js:466-486`): absent
   from `required` (`:442`), so absence is a normal verdict and no historical shape breaks. This is
   the only mechanism available -- `additionalProperties:false` (`:441`, S4) makes any runtime-added
   key impossible, so "the judge already says WARN in prose" cannot be harvested without a schema edit.
2. **Route in a pure caller-side sibling.** `enforceSeverityRouting(verdict)` modelled on
   `enforceResearchRouting` (`:779-798`): pure (no filesystem in the Workflow runtime, `:563-566`),
   not exported, driven by the checker via temp re-export.
3. **Return it as a sibling, extend the existing invariants -- do not write a third bespoke one.**
   Add to `merged` at `:835`, then extend the leak filters at `:839` / `:848` including the
   allow-list carve-out shape at `:849-850` for judge-authored fields, and keep
   `verdict_unmodified` (`:857`) computed over the unchanged `verdict` keys.
4. **Echo + disagree, never override.** Mirror `research-gate.js:769-770`: report the judge's own
   value AND the caller's derived routing AND an explicit `disagreed` flag. That satisfies both
   OTel's "keep the source's own string" and Anthropic's "don't obscure the underlying response".
5. **Give ABSENT a name.** Reuse the `sequence_status` ladder idiom (`:621-630`, null-never-zero) and
   the sibling rail's three-valued `['...','...','ABSENT']` enum (`research-gate.js:568`). Measured:
   9.3% of historical returns carry no severity signal at all, so ABSENT is a real state, not a corner.
6. **Validate by replay with the baseline attached.** Denominator = 397 parseable verdicts from the
   441-record `startsWith('qa-verdict')` corpus. Report the full 3x3 confusion table, Cohen's kappa,
   AND the majority-class baseline (56.7%) -- per S5's Table-3 methodology and S6's ordinal caution.
   Expect the replay to demonstrate that re-derivation FAILS (kappa 0.129), which is the evidence
   for the design, not an obstacle to it.
7. **Checker.** `scripts/qa/verify_severity_routing_90_2.mjs` modelled on
   `verify_escalation_86_78.mjs`: real-function-via-temp-re-export, an `EXPECTED_CHECKS` cardinality
   floor (a checker whose loop covers nothing exits 0), a `sourceOverride` mutation seam, and a RED
   cell proving the flattened-sibling mutation is caught.

## Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch -- **7** (S1-S7)
- [x] 10+ unique URLs total -- **17** (7 read in full + 10 snippet-only)
- [x] Recency scan (last 2 years) performed + reported -- section above, 3 findings
- [x] Full pages read, not abstracts; arXiv fetched via `/html/`, never `/pdf/`; the two
      load-bearing quote sets (SARIF, RFC 9413) and all four AgentProp-Bench kappas verified
      verbatim at source
- [x] file:line anchors for every internal claim -- inventory table above

Soft checks:
- [x] Internal exploration covered every module in the caller's INTERNAL SCOPE (qa-verdict.js
      incl. `enforceEscalation` + the leak invariants, qa.md, SKILL.md:24-32, `VERDICT_SCHEMA`,
      scripts/qa/ checker patterns, verdict_ledger.jsonl, the 441-record run corpus)
- [x] Contradictions noted -- the SARIF fabrication, the default-value conflict (SARIF "warning"
      vs OTel nearest-in-range vs this repo's fail-closed null), and S6's missing baseline
- [x] Per-claim citation with URL + access date

---

## Key findings (per-claim citations)

1. **A caller must read the emitter's classification because re-deriving it measures at chance.**
   "substring-heuristic judging of agent outputs agrees with human annotation only at chance level
   (Cohen's kappa = 0.049)" vs 0.432 / 0.567 for real classifiers (S5, arXiv 2604.16706,
   https://arxiv.org/html/2604.16706, accessed 2026-08-20). Reproduced in-repo: token-presence
   re-derivation over 360 historical qa-verdict returns scores **40.0% / kappa 0.129**, against a
   **56.7% majority-class baseline** -- i.e. 16.7 points WORSE than a constant.
2. **The mechanism is negation, and it is measurable.** 130 of the 183 records containing `BLOCK`
   (71.0%) contain a NEGATED occurrence within 25 characters ("no BLOCK, no WARN"); the judge's
   dominant use of the token is to report the heuristic did *not* fire (measured, run-record corpus).
3. **Severity must be its own field; the closed schema makes this a schema edit or nothing.**
   `additionalProperties:false` "means no additional properties will be allowed" and "only recognizes
   properties declared in the same subschema" (S4, https://json-schema.org/understanding-json-schema/reference/object).
   Confirmed empirically: **0 of 969** historical `violation_details` rows carry a `severity` key,
   despite `SKILL.md:29` instructing the judge to write one.
4. **The two-field pattern is standards consensus.** OTel keeps `SeverityText` = "the original string
   representation of the severity as it is known at the source" beside a normalized `SeverityNumber`
   (S1, https://opentelemetry.io/docs/specs/otel/logs/data-model/); SARIF separates producer `level`
   (3.27.10) from outcome `kind` (3.27.9) from consumer `rank` (3.27.25)
   (S2, https://docs.oasis-open.org/sarif/sarif/v2.1.0/errata01/os/sarif-v2.1.0-errata01-os-complete.html).
5. **Inferring the sender's intent is the named anti-pattern**, and tolerating it "enters a
   pathological feedback cycle"; the prescribed remedy is Virtuous Intolerance -- "generate fatal
   errors for unspecified conditions instead of attempting error recovery"
   (S3, RFC 9413 sections 2.3 / 4.1 / 5.1, https://www.rfc-editor.org/rfc/rfc9413.html, quotes
   verified verbatim against the .txt).
6. **Routing is a classify-then-direct pattern whose stated precondition is accurate classification**
   -- "Routing classifies an input and directs it to a specialized followup task ... where
   classification can be handled accurately"
   (S7, https://www.anthropic.com/engineering/building-effective-agents).
7. **A 3-level ordinal is the harder judging regime** and formatting perturbations hurt more than
   semantic ones (S6, https://arxiv.org/html/2603.05399) -- so validate per-level, before deployment,
   and never by post-hoc patching of the parse.
8. **The repo already ships the exact idiom**: `research-gate.js:769-770` returns
   `agent_self_reported_gate_passed` + `self_report_disagreed` -- read the emitter's own claim,
   recompute, report the disagreement, let the enforced value govern. `enforceResearchRouting`
   (`qa-verdict.js:779-798`) + the sibling-leak throws (`:839`, `:848`) + computed
   `verdict_unmodified` (`:857`) are the structural template; extend them rather than adding a
   parallel mechanism.
9. **Watch the denominator**: 441 records only under `startsWith('qa-verdict')` (436 exact-match),
   398 with a parseable result, **397 with a verdict** -- the replay denominator.
