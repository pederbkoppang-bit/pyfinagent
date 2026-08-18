# Research Brief -- phase-86.96

**Topic:** Defensive argument marshalling at an LLM-caller/script boundary --
why hand-composed JSON strings fail re-parse (unescaped interior quotes, invalid
escapes, delimiter errors mid-payload), the string-vs-object contract for Claude
Code Workflow `args`, byte-verbatim round-trip testing of adversarial payloads
(backticks, double quotes, newlines, non-ASCII), and fail-fast vs repair
strategies for malformed structured input.

**Tier:** moderate (caller-specified). **Audit-class:** NO (coverage reported for
information only; `coverage.dry` not required).
**Researcher:** Layer-3 Workflow rail. **Date:** 2026-08-17.

---

## STATUS ENVELOPE (born inert -- phase-86.37)

```json
{
  "brief_status": "COMPLETE",
  "tier": "moderate",
  "external_sources_read_in_full": 8,
  "snippet_only_sources": 22,
  "urls_collected": 30,
  "recency_scan_performed": true,
  "internal_files_inspected": 13,
  "coverage": {
    "audit_class": false,
    "rounds": 2,
    "dry_rounds": 0,
    "K_required": 2,
    "new_findings_last_round": 3,
    "dry": false
  },
  "summary": "See the FINAL envelope at the tail of this brief.",
  "brief_path": "handoff/current/research_brief_86.96.md",
  "gate_passed": true
}
```

---

## Sections (filled incrementally)

- [ ] Internal reproduction of the two failing payloads
- [ ] Internal code inventory (workflow scripts, guards, run records)
- [ ] Read in full (>=5)
- [ ] Snippet-only
- [ ] Recency scan (2024-2026)
- [ ] Key findings
- [ ] Application to pyfinagent
- [ ] Research Gate Checklist


---

## 1. Internal reproduction -- the minimal failing input is ONE CHARACTER

Reproduced 2026-08-17 with stock `python3 json.loads` on both saved payloads.

| Payload | bytes | `json.loads` error | pos | char AT pos |
|---|---|---|---|---|
| `failing_args_wf_1f6b0398-020.txt` (step 86.90) | 5481 | `Expecting ',' delimiter: line 1 column 4940` | 4939 | `}` |
| `failing_args_wf_88302c2a-d20.txt` (step 86.91) | 6090 | `Expecting ',' delimiter: line 1 column 5537` | 5536 | `}` |

**Container-stack recovery at the failing offset** (hand-rolled scanner, string/escape
aware, reporting the stack BEFORE the offending char is consumed):

```
86.90: [('{',0), ('{',2093 = "extra"), ('[',3865 = "judge_these_specifically")]  -> closed with '}'
86.91: [('{',0), ('{',2637 = "extra"), ('[',4533 = "judge_these_specifically")]  -> closed with '}'
```

**The innermost open container is an ARRAY `[`, closed with `}`.** Python's C array
parser raises `Expecting ',' delimiter` (not `Expecting ']'`) because after a value it
accepts only `,` or `]`; any other char -- including `}` -- takes the comma branch. So
the error MESSAGE names a comma while the DEFECT is a bracket. That mis-direction is the
single most important fact for whoever debugs this next: **do not go looking for a
missing comma.**

**Bisection result -- minimal edit distance = 1 byte.** Substituting `]` for the single
`}` at the failing offset makes BOTH payloads parse:

```
86.90: SINGLE-CHAR FIX ('}'->']') at 4939 : PARSES OK   -> 8 top keys, extra{4}, judge_these_specifically = list[4]
86.91: SINGLE-CHAR FIX ('}'->']') at 5536 : PARSES OK   -> 8 top keys, extra{4}, judge_these_specifically = list[4]
```

There is **no second defect**: one substitution yields a fully valid document in both
cases. Head and tail are intact exactly as the step description says -- the corruption is
neither a truncation nor an encoding fault.

**Composition-provenance evidence (new, not in the step description).** For BOTH fixed
documents:

```python
json.dumps(obj, ensure_ascii=False, separators=(',',':')) == fixed   # True
```

Byte-identical to a *compact* serialiser's output. A real serialiser cannot emit `}` to
close a `[` -- the mismatch is structurally unreachable from `json.dumps`. So the payload
was **hand-composed in compact style by the model**, not produced by a serialiser and
then damaged in transit. This localises the defect to **layer 1 (caller serialisation)**
and exonerates layers 2 and 3 before any of them is inspected.

**Interior escaping was NOT the failure mode here.** 86.91 carries
`` `s in (\"86.86\",\"12.7\")` `` -- backticks plus correctly `\"`-escaped interior
quotes -- and both payloads carry apostrophes, `--`, `->` and nested single-quoted shell
in `verification_command` (`bash -c 'source .venv/bin/activate && ...'`). All of that
survived. The model got the hard escaping right and the easy bracket wrong.

## 2. Per-layer localisation -- layer 1 is guilty, layers 2 and 3 are clean

The step asks for three-layer localisation (caller serialisation / Workflow
marshalling / script `JSON.parse`). All three are now MEASURED, not argued.

**Layer 2 (Workflow marshalling) -- EXONERATED, byte-verbatim.** The run records
`~/.claude/projects/-Users-ford--openclaw-workspace-pyfinagent/4875bd0a-.../workflows/wf_1f6b0398-020.json`
and `wf_88302c2a-d20.json` store `args` as a **string** whose SHA-1 matches the
scratchpad copy exactly:

```
wf_1f6b0398-020  record args  sha1 f678987c273a60bb == scratchpad f678987c273a60bb   (5481 chars, identical: True)
wf_88302c2a-d20  record args  sha1 151e86ef71d17910 == scratchpad 151e86ef71d17910   (6090 chars, identical: True)
```

Both payloads are **pure ASCII** (no non-ASCII code points at all), contain **zero**
control characters below 0x20, **zero** literal newlines, and carry 6 and 16 backticks
plus 4 and 8 correctly-escaped `\"` sequences respectively. Nothing was re-encoded,
normalised, or line-wrapped in transit. The marshalling layer is a faithful pipe.

**Layer 3 (script `JSON.parse`) -- CORRECT, and it already fails fast.** The record's
`error` field is the verbatim kill and it names the script's own guard:

```
Error: qa-verdict: args are PRESENT but not parseable as JSON (typeof=string isArray=false
len=5481 preview="{\"step_id\":\"86.90\",\"verdict_sequence\":[\"CONDITIONAL\",\"CONDITIONAL\"],\"attempt_num")
-- pass a plain object (or valid JSON) carrying step_id, or omit args entirely for a dry run.
    at fail (workflow.js:78:11)
    at classifyArgs (workflow.js:87:48)
```

That is `classifyArgs` at `.claude/workflows/qa-verdict.js:75-99`, specifically the
`try { v = JSON.parse(v) } catch (_e) { fail(...) }` at **`:93`**. **The guard worked.**
86.96 is therefore NOT "add a guard" -- the fail-fast behaviour is already shipped,
already correct, and already produced a diagnostic that names typeof, isArray, length
and an 80-char preview. What it does NOT do is tell the caller **WHERE** the payload
broke, which is the only reason two 5-6 KB payloads had to be exported to a scratchpad
and bisected by hand.

**Layer 1 (caller serialisation) -- GUILTY.** Proven twice over: (a) the byte-identical
compact-`json.dumps` round-trip above, which a serialiser could produce and a
bracket-mismatch could not; (b) the sha1 identity, which removes every downstream
suspect.

## 3. Execution-time args-shape census (580 run records, 2026-08-17)

Re-derived independently rather than carried forward from 86.90's `31 objects + 409
strings` baseline. Source: every `*/workflows/wf_*.json` under the project's Claude
projects dir.

| args shape | count | share of parameterised launches |
|---|---|---|
| string, parses | 390 | 80.6% |
| string, FAILS to parse | **4** | **0.8%** |
| plain object (`dict`) | 90 | 18.6% |
| absent / null (legal dry run) | 96 | -- |
| **total records** | **580** | |

String-arg length distribution: n=394, min=327, p50=3,942, p90=5,473, max=7,358.

Two facts the contract should turn on:

1. **The string path is the DOMINANT path, not an edge case** -- 394 of 484
   parameterised launches (81.4%). Any remedy that assumes "callers normally pass an
   object" is designing for the minority.
2. **The failure is exclusive to the string path.** 4 of 394 string launches fail
   (1.02%); **0 of 90** object launches fail (0%). The string-vs-object contract is not
   stylistic -- it is the difference between a 1% loss rate and a measured zero.

## 4. THE CLASS IS FOUR EVENTS, NOT TWO -- and the leading hypothesis is wrong

The step description names two failing payloads. The census finds **four**, spanning
**both** Layer-3 scripts and **two distinct defect shapes** -- and *neither shape is an
escaping defect*.

| run | date | script | len | error | shape |
|---|---|---|---|---|---|
| `wf_1f6b0398-020` | 2026-08-16 | qa-verdict.js | 5481 | `Expecting ',' delimiter` @4939 | **bracket mismatch** `}` closes `[` |
| `wf_88302c2a-d20` | 2026-08-16 | qa-verdict.js | 6090 | `Expecting ',' delimiter` @5536 | **bracket mismatch** `}` closes `[` |
| `wf_8375665b-f5a` | 2026-08-09 | research-gate.js | 4911 | `Expecting ',' delimiter` @4911 = EOF | **TRUNCATION** (mid-document EOF) |
| `wf_b098cab6-87b` | 2026-08-06 | research-gate.js | 3201 | `Unterminated string starting at` @123 | **TRUNCATION** (mid-string EOF) |

- `wf_8375665b-f5a` has `status: killed` / `error: Workflow aborted`; its args end
  `...that materially changes the severity."` with no closing `}`.
- `wf_b098cab6-87b` ends `...the audit-class \`coverage\` object.` -- **no closing quote,
  no closing brace**; 0 unescaped quotes remain after the opening quote at char 123,
  which is the signature of an EOF cut rather than a stray quote.

**ZERO of the four are unescaped-interior-quote or invalid-escape defects.** The step
objective leads with that hypothesis; the evidence refutes it for this corpus. Both
observed shapes are **composition/streaming** faults: the model either mis-typed one
bracket in a hand-built 5 KB literal, or its tool-input stream was cut mid-flight. A
regression guard built only from the two named payloads would test one of the two live
shapes and miss the other entirely (cf. the standing lesson "a guard from the instance
is not a guard against the class").

## 5. The pre-hardening SILENT-DEGRADATION event (the one that did not fail)

`wf_b098cab6-87b` (2026-08-06, `research-gate.js`) has **`status: completed`, an empty
`error`, and a full result envelope** -- while its `args` string is unparseable. It ran
**BLIND** on the old fallback-to-`{}` path that `qa-verdict.js:44-56` documents and
phase-86.17 replaced. The damage is measurable in the receipt:

```
caller passed : "audit_class": true        (visible in the truncated args head)
run returned  : coverage.audit_class: False, coverage.dry: False, gate_passed reported on 7 sources
```

The **audit-class loop-until-dry requirement was silently dropped**, and the run
returned a gate result anyway. This is the strongest single argument in the corpus for
fail-fast over repair-or-default: the two 2026-08-16 payloads cost one retry each and
were fully diagnosable; this one produced a **plausible, confident, wrong** gate result
that nothing flagged.

## 6. The bracket is WRONG, not MISSING -- and the mechanism is idiom priming

A one-character defect has two candidate mechanisms, and they demand different
remedies: a character **dropped in transit** (a streaming/delta-boundary shear, which
would indict the runtime) versus a character **mis-emitted at composition** (which
indicts the caller). They are distinguishable by whether the document repairs under
INSERTION or under SUBSTITUTION.

```
wf_1f6b0398-020   SUBSTITUTE '}'->']'  -> PARSES (8 root keys, extra{4} intact)
                  INSERT     ']'       -> FAILS: Extra data: line 1 column 5482
wf_88302c2a-d20   SUBSTITUTE '}'->']'  -> PARSES (8 root keys, extra{4} intact)
                  INSERT     ']'       -> FAILS: Extra data: line 1 column 6091
```

Insertion leaves a surplus `}` at the tail. **The character is WRONG, not MISSING** --
which rules out the delta-boundary token-dropout mechanism (claude-code #67765 / #69085,
read in full below) for these two payloads and pins them on composition.

**The mechanism, from the recovered structure.** In BOTH payloads, `extra` has the same
four fields in the same order and the same types:

```
extra = { cycle: str,
          what_changed_since_the_cycle_2_verdict: dict,   <-- closes with the idiom  "},
          judge_these_specifically: list,                 <-- must close with        "],
          known_weak_points_main_is_flagging: list }
```

The field IMMEDIATELY BEFORE the failing one is a **dict**, whose close idiom is `"},`.
The failing field is a **list**, and it was closed with that same `"},`. This is
local-idiom repetition (priming) at a container boundary, and it fired **identically on
two independent spawns on the same day, at the same key, in the same direction**. The
defect is not a random typo: it is **shape-triggered** by an object-valued field
immediately followed by an array-valued field, both carrying long free-text strings that
push the opening bracket thousands of characters away from its close. Note both failures
land at ~4.9-5.5 KB in payloads whose p90 is 5,473 -- deep in the long tail where the
opener is furthest out of view.

**Design consequence:** a fix that only teaches the caller "escape your quotes" addresses
a class that has never fired here. A fix that removes the hand-composition step, or that
reports WHERE the payload broke, addresses both live shapes.

## 7. Internal code inventory

| File | Lines | Role | Status |
|---|---|---|---|
| `.claude/workflows/qa-verdict.js` | 746 | `classifyArgs` at `:75-99`; the `JSON.parse` fail-fast at **`:93`**; render boundary `:101-292` | **Correct.** Fired exactly as designed on both 2026-08-16 payloads |
| `.claude/workflows/research-gate.js` | 1101 | byte-identical `classifyArgs` + render block (runtime forbids imports, so duplication is deliberate) | Same guard; owns the other 2 failures |
| `scripts/qa/verify_workflow_args_boundary.mjs` | 601 | drives BOTH scripts over a 10-shape `SHAPES` fixture at `:141-151` | **GREEN: 96 passed, 0 failed** (re-run 2026-08-17) |
| `scripts/qa/verify_prompt_render_86_90.mjs` | 448 | asserts the two render-block copies have not drifted | **GREEN: 95 passed, 0 failed** |
| `handoff/current/live_check_86.90.md` §6 | -- | the `31 objects + 409 strings` census baseline | Superseded by §3 above (580 records) |

**PREMISE CORRECTION -- the step description is stale on one point.** It states
`verify_workflow_args_boundary.mjs` "is RED 84/3 for an unrelated filed reason, 86.92."
It is **not red**: commits `687109bb -> b46f0e17 -> 45b74291 -> e45c1bf6`
("phase-86.92: PASS -- close the restored args-boundary gate") landed the fix, and a
fresh run today prints `ALL GREEN: 96 passed, 0 failed`. The contract must not budget
work for repairing a guard that is already repaired.

**The real gap in that guard.** Its `SHAPES` corpus already covers
`malformed-json-string` (`{"step_id": "86.17"` -- a truncation) and
`json-string-raw-newline`. It does **not** cover:

1. a **bracket-type mismatch deep inside a large payload** -- i.e. the exact shape of the
   two live 2026-08-16 failures;
2. **scale** -- every fixture is 20-45 chars; live string args have p50 3,942 and p90
   5,473, and both failures landed at 4.9 KB and 6.1 KB. A 40-char fixture cannot
   exercise a defect whose mechanism is "the opening bracket is 1,000+ chars out of
   view";
3. any assertion that the thrown message **localises WHERE** the parse broke. Today it
   reports typeof/isArray/len plus the first 80 chars -- all four real payloads have an
   intact head, so the preview is identical to a healthy one and carries zero signal.
4. **byte-verbatim round-trip** of adversarial content (backticks, `\"`, non-ASCII,
   newlines) through the string path.

Items 1-4 are the natural content of criterion "mutation-tested regression guard" and
"byte-verbatim criteria round-trip".

---

## 8. External research

### Search-query composition (three-variant discipline)

| Variant | Query run |
|---|---|
| year-less canonical | `LLM generated JSON parse errors unescaped quotes invalid escape repair vs fail-fast` |
| year-less canonical | `Anthropic tool use streaming partial JSON input_json_delta truncated arguments max_tokens` |
| current-year frontier (2026) | `arXiv 2026 constrained decoding structured output JSON schema compliance LLM agent tool call reliability` |
| last-2-year window (2025) | `JSON round-trip property fuzzing adversarial payload unicode backtick newline test corpus 2025` |

### Read in full (8; counts toward the gate)

| URL | Accessed | Kind | Fetched how | Key quote / finding |
|---|---|---|---|---|
| https://platform.claude.com/docs/en/agents-and-tools/tool-use/fine-grained-tool-streaming | 2026-08-17 | official doc | WebFetch | "Because the API does not buffer or validate a tool's input before streaming it, you might receive partial or invalid JSON. A response that ends with the stop reason `max_tokens` can also cut a parameter off midway. **Accumulate the fragments, guard the parse**". Prescribes returning `{"INVALID_JSON": "<the unparseable input you received>"}` with `is_error: true` -- **preserve the original, never mutate it**. And: "Build the wrapper with your JSON library rather than by concatenating strings, so quotes and other special characters in the invalid input are escaped correctly." |
| https://www.rfc-editor.org/rfc/rfc9413.html | 2026-08-17 | official (IETF) | WebFetch | s5.1: "Choosing to generate **fatal errors** for unspecified conditions instead of attempting error recovery can ensure that faults receive attention." s4.1: tolerance produces a "pathological feedback cycle" where "errors ... become entrenched, forcing other implementations to be tolerant of those errors." |
| https://www.rfc-editor.org/rfc/rfc8259.html | 2026-08-17 | official (IETF) | WebFetch | s5: `array = begin-array [ value *( value-separator value ) ] end-array`, with `end-array = ws %x5D ws`. The grammar is **type-matched**: `%x7D` (`}`) cannot terminate an array. s7: only quotation mark, reverse solidus and U+0000-U+001F MUST be escaped. s9: "A JSON parser MAY accept non-JSON forms or extensions" -- i.e. leniency is permitted by the spec, which is exactly why it must be a local policy decision. |
| https://github.com/anthropics/claude-code/issues/69085 | 2026-08-17 | vendor issue tracker | WebFetch | [ADVERSARIAL to my layer-1 conclusion] A real client-side truncation bug: "the truncation is introduced **before the bytes leave the client**, in the streamed-`input_json_delta` accumulation." Symptoms: **tail cut**, size-correlated >4 KB, surfacing as the SAME misleading `Expecting ',' delimiter`. Deterministic repro: "Press Escape mid-stream on a >4 KB tool call." CLI 2.1.179; closed as duplicate of #67765 (`VH1` tokenizer string-token dropout at delta boundaries). |
| https://arxiv.org/html/2501.10868 | 2026-08-17 | preprint (JSONSchemaBench) | WebFetch | 10K real-world schemas. Unconstrained "LM-only approaches" are the **lowest-reliability** tier at "13-90%" compliance; constrained engines reach "88-100%". Also: "Constrained decoding can speed up generation by 50% compared to unconstrained decoding" via token fast-forwarding. |
| https://arxiv.org/html/2605.02363 | 2026-08-17 | preprint, 2026 | WebFetch | [ADVERSARIAL] "A response that solves the task but violates the output schema is as unusable as one that is simply wrong." Models at 77-85% task accuracy scored **0% output accuracy** under strict JSON compliance. The authors **reject** fail-fast and prefer prompt optimisation (AloLab, 84-87%), citing constrained decoding's "3.6x-8.2x latency overhead". Documents an escaping class we did NOT see: unescaped backslashes in LaTeX. |
| https://fsharpforfunandprofit.com/posts/property-based-testing-2/ | 2026-08-17 | authoritative blog (canonical) | WebFetch | The "There and back again" property: "combining an operation with its inverse, ending up with the same value you started with", applied to "serialization/deserialization". Caveat: a round-trip proves only *consistency between the operations*, not that the intermediate form is correct -- so a round-trip test must be paired with an independent validity check. |
| https://dasroot.net/posts/2026/05/structured-output-llms-json-breaks-analyzed/ | 2026-08-17 | community | WebFetch | Claims "incorrectly quoted strings ... 18% of all JSON failures". **Downweighted deliberately**: its three headline figures have incommensurable denominators (18% of failures, a "23% increase in error logs", "12% of users"), so the taxonomy is not usable as evidence. Recorded because it was read, not because it is load-bearing. |

### Identified but snippet-only (context; does NOT count toward the gate)

| URL | Kind | Why not fetched in full |
|---|---|---|
| https://github.com/anthropics/claude-code/issues/67765 | vendor issue | named as the root-cause duplicate by #69085, already characterised there |
| https://github.com/anthropics/claude-code/issues/66247 | vendor issue | regression-window reference only |
| https://github.com/block/goose/issues/2892 | community issue | same class (raw control chars in tool-call args); our payloads have 0 control chars |
| https://dl.acm.org/doi/10.1145/3634737.3657003 | peer-reviewed (ACM AsiaCCS) | differential testing of JSON parsers; paywalled, and cross-parser divergence is out of scope (one parser here) |
| https://arxiv.org/pdf/2408.02442 | preprint ("Let Me Speak Freely?") | format-restriction cost; superseded for our purposes by 2605.02363 |
| https://arxiv.org/pdf/2603.03305 | preprint | "Hidden Cost of Structured Generation" -- same debate axis, already represented |
| https://arxiv.org/html/2505.04016v1 | preprint (SLOT) | post-hoc schema-conforming repair model; repair-side, represented by 2605.02363 |
| https://arxiv.org/pdf/2606.01926 | preprint | bias in locally constrained decoding; not applicable at this boundary |
| https://arxiv.org/pdf/2601.17717 | preprint survey | LLM-generated-data trustworthiness; too general |
| https://arxiv.org/pdf/2605.26539 | preprint (FuzzPilot) | structured-text fuzzing recipes; useful later if a fuzz corpus is built |
| https://github.com/nlohmann/json/blob/develop/tests/fuzzing.md | official lib docs | OSS-Fuzz corpus practice; C++-specific |
| https://github.com/golang/go/issues/31309 | official lib issue | canonical `encoding/json` round-trip fuzz target |
| https://blog.asymmetric.re/finding-fractures-an-intro-to-differential-fuzzing-in-rust/ | blog | differential fuzzing method, out of scope |
| https://medium.com/@yanxingyang/tutorial-on-using-json-repair-in-python-easily-fix-invalid-json-returned-by-llm-8e43e6c01fa0 | community | `json_repair` tutorial -- the repair camp; low tier |
| https://medium.com/@lilianli1922/leveraging-llms-for-automated-correction-of-malformed-json-e3c1f8b789a6 | community | LLM-based JSON correction; low tier |
| https://aijsonmedic.com/blog/json-unterminated-string-error-fix | community/SEO | unterminated-string class; no primary data |
| https://github.com/oracle/langchain-oracle/pull/56 | code PR | escaping fix in a connector; incidental |
| https://dev.to/gabrielanhaia/streaming-tool-calls-parse-anthropic-sse-without-loading-the-whole-message-2on | blog | SSE accumulation how-to; superseded by the official doc |
| https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-anthropic-claude-messages-tool-use.html | official doc (AWS) | Bedrock mirror of the same tool-use contract |
| https://mbrenndoerfer.com/writing/constrained-decoding-structured-llm-output | blog | constrained-decoding explainer |
| https://niteagent.com/blog/llm-structured-outputs-2026/ | blog | 2026 vendor-compliance figures (OpenAI 100%, Claude ~99%, Gemini ~98%) -- unverified |
| https://www.we-fuzz.io/blog/attacking-apis-with-json-injection-a-technical-deep-dive | blog | JSON injection / unicode surrogate tricks; security angle, adversarial-payload ideas |

**URLs collected: 30** (8 read in full + 22 snippet-only).

### Recency scan (2024-2026) -- PERFORMED

Searched the 2024-2026 window explicitly (queries 3 and 4 above, plus the vendor-issue
trail). **Result: 3 findings that materially complement the canonical sources.**

1. **(2026, vendor)** Fine-grained tool streaming is now a **per-tool
   `eager_input_streaming` field**, replacing the `fine-grained-tool-streaming-2025-05-14`
   beta header; with it on, "the API does not buffer or validate a tool's input". This is
   a live, current mechanism by which a caller can receive invalid JSON through no fault
   of the model's composition -- and it did not exist in this form in the older literature.
2. **(2026, vendor issue #69085 / #67765)** A *measured* client-side truncation bug in
   Claude Code >= 2.1.165 that manifests as the identical `Expecting ',' delimiter`
   message. This is the strongest new caution in the window: **the error string alone
   cannot distinguish a caller defect from a runtime defect.** It is why §6's
   insertion-vs-substitution test matters -- that test, not the message, is what
   separated them here.
3. **(2026, arXiv 2605.02363)** An explicitly anti-fail-fast position with numbers,
   published this year. It does not overturn RFC 9413 for this boundary (see §9) but it
   is the current best statement of the opposing case.

Canonical prior art (RFC 8259 2017, RFC 9413 2023, the property-testing taxonomy)
remains valid and is **not** superseded.

---

## 9. Consensus vs debate

**Consensus:** never silently mutate a payload in transit. The two camps disagree about
what to do INSTEAD, not about that.

- **Fail-fast camp (RFC 9413 s5.1; Anthropic tool-use doc; RFC 8259's type-matched
  grammar).** Generate a fatal error; preserve the original bytes; hand the failure back
  to the producer. Anthropic's own prescription is *literally* preserve-and-report:
  return the unparseable input verbatim inside `{"INVALID_JSON": ...}` with
  `is_error: true`, built with a JSON library rather than string concatenation.
- **Repair camp (arXiv 2605.02363; `json_repair`; SLOT).** A structurally-invalid
  response may still contain a correct answer, so discarding it wastes a completion; fix
  the format instead. Its strongest argument is cost -- constrained decoding carries
  "3.6x-8.2x latency overhead" and can degrade task quality.

**Resolution for this boundary, on our own evidence -- fail-fast wins, and it is not
close.** The repair camp's cost argument is about *response* payloads, where a discarded
generation is a real loss. This is an *argument* boundary: the payload is 5 KB of
evaluation context, the retry is cheap, and §5 is a measured counterexample --
`wf_b098cab6-87b` took the repair-adjacent path (fall back to `{}`), **completed**, and
returned a gate result with `coverage.audit_class: False` when the caller had passed
`true`. A repaired-or-defaulted argument produces a **confident wrong answer that nothing
flags**; a rejected argument produces a retry. Note also that guessing the repair here is
genuinely unsafe: §6 shows the correct fix was SUBSTITUTION, while the visually obvious
"a bracket is missing" reading (INSERTION) yields a *different* and still-invalid
document. A repairer that guessed would have guessed wrong.

## 10. Pitfalls (from the literature + this corpus)

1. **The parser's error message names the wrong token.** `Expecting ',' delimiter` is
   what Python's array parser emits for *any* non-`,`/`]` char, including `}` and EOF.
   Three of our four payloads report it; the causes are a bracket, a bracket, and a
   truncation. (RFC 8259 s5; reproduced in §1.)
2. **The same message spans caller-defect and runtime-defect.** #69085 shows a genuine
   client-side shear producing it. Distinguish by insertion-vs-substitution and by
   whether the tail is intact -- never by the message.
3. **A truncation and a bracket error look identical at the head.** All four payloads
   have a clean head, so an 80-char preview is a null diagnostic
   (`qa-verdict.js:81`).
4. **Round-trip alone is not a validity proof.** A `parse -> serialise -> parse` identity
   can hold for a form that is not the one you meant (fsharpforfunandprofit). Pair it
   with an independent byte-equality assertion against the *original* input.
5. **`JSON.stringify` on a value you did not construct is a silent-loss operation** --
   already the standing lesson at `qa-verdict.js:117-121` (dropped `undefined` keys,
   Map/Set to `{}`, NaN to `null`, arrays to `a,b` with no marker).
6. **Building the error report by concatenation re-introduces the bug you are
   reporting.** Anthropic states this explicitly: build the wrapper with a JSON library.
7. **A fixture two orders of magnitude smaller than production cannot exercise a
   distance-dependent defect** (§7 item 2).

## 11. Application to pyfinagent (external findings -> file:line anchors)

| Finding | Anchor | Implication for the 86.96 contract |
|---|---|---|
| Fail-fast is already correct and already shipped | `.claude/workflows/qa-verdict.js:93` (`catch (_e) { fail('are PRESENT but not parseable as JSON', raw) }`) | Do **not** re-litigate fail-fast; RFC 9413 s5.1 backs the existing behaviour. The gap is diagnosis, not policy. |
| The preview is a null diagnostic on every real failure | `qa-verdict.js:76-82` (`describe`) | Cheapest high-value change: report the parse **position** and a window AROUND it, plus the enclosing container stack -- exactly the bisection §1 did by hand. Must be duplicated byte-identically into `research-gate.js` (the runtime forbids imports; `verify_prompt_render_86_90.mjs` already enforces non-drift for the render block). |
| Preserve, never mutate | Anthropic tool-use doc | The failing `args` string is **already** persisted verbatim in the run record (sha1-verified, §2). Any new diagnostic must read it, not rewrite it. |
| The string path is 81.4% of launches and carries 100% of the failures | §3 census | The durable fix is at the CALLER: pass a plain object. 0/90 object launches have ever failed. A doc/prompt change alone is unenforceable; consider whether the boundary can reject the string form outright once callers are migrated -- but note `classifyArgs` accepts strings deliberately today, so that is a policy change, not a bug fix. |
| The defect is shape-triggered (dict-field then list-field) | §6 | The regression fixture must reproduce **that shape at production scale**, not a 40-char stub. |
| Guard corpus is scale- and shape-blind | `scripts/qa/verify_workflow_args_boundary.mjs:141-151` | Extend `SHAPES` with: bracket-mismatch-at-depth in a >=5 KB payload; the two real payloads as golden fixtures; a byte-verbatim round-trip over an adversarial string (backticks, `\"`, raw newline, non-ASCII, lone surrogate). Mutation cells must kill each new assertion individually -- the file's own `[4] MUTATION` convention at `:22`. |
| 86.92 is closed | commit `e45c1bf6`; `ALL GREEN: 96 passed, 0 failed` | Correct the step's stale "RED 84/3" premise in the contract. |

## 12. Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (**8**)
- [x] 10+ unique URLs total incl. snippet-only (**30**)
- [x] Recency scan (last 2 years) performed + reported (§8, 3 findings)
- [x] Full pages read, not abstracts, for the read-in-full set (arXiv via `/html/`; no `/pdf/` fetched)
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every module in the caller's scope
- [x] Contradictions noted (§9; two `[ADVERSARIAL]`-tagged sources)
- [x] Claims cited per-claim
- [ ] **Gap, stated:** `research-gate.js` was inspected via `wc`, targeted grep, the census
  and the two checkers that assert its byte-identity with `qa-verdict.js` -- it was not
  read line-by-line end to end. Its `classifyArgs` and render block are asserted identical
  by `verify_prompt_render_86_90.mjs`, so the claims above hold; a claim about its
  non-duplicated regions would not be supported by this brief.

---

## STATUS ENVELOPE -- FINAL

```json
{
  "brief_status": "COMPLETE",
  "tier": "moderate",
  "external_sources_read_in_full": 8,
  "snippet_only_sources": 22,
  "urls_collected": 30,
  "recency_scan_performed": true,
  "internal_files_inspected": 13,
  "coverage": {
    "audit_class": false,
    "rounds": 2,
    "dry_rounds": 0,
    "K_required": 2,
    "new_findings_last_round": 3,
    "dry": false
  },
  "summary": "Minimal failing input is ONE character: '}' closing a '[' at extra.judge_these_specifically. Repairs under SUBSTITUTION, not INSERTION -- so the bracket is wrong, not dropped; layer 1 (caller composition) is guilty. Run-record args are sha1-identical to the payloads, exonerating Workflow marshalling; classifyArgs at qa-verdict.js:93 already fails fast correctly. Census of 580 records finds FOUR failures, not two, in two shapes (2 bracket, 2 truncation), all on the string path (4/394) vs 0/90 objects. One pre-hardening failure ran BLIND and silently dropped audit_class:true. verify_workflow_args_boundary.mjs is GREEN 96/0, not RED.",
  "brief_path": "handoff/current/research_brief_86.96.md",
  "gate_passed": true
}
```
