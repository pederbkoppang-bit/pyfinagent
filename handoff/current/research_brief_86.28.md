# Research Brief -- step 86.28

**Tier:** moderate (caller-specified). **Audit-class:** NO (coverage reported for information only).
**Date:** 2026-08-10. **Status:** COMPLETE. Written incrementally (write-first), not flushed at the end.

## Research: silent capability degradation, artifact-grounded verification of self-reported process claims, and mutation testing of guard code

Objective (restated):
1. **SILENT DEGRADATION** -- caller requests a stricter capability tier than the gate implements:
   fail closed, report-and-proceed, or silently substitute? Key distinction: **ABSENT** parameter
   (defaulting legitimate) vs **UNSUPPORTED** parameter (caller named something real this
   implementation does not provide).
2. **ARTIFACT-GROUNDED VERIFICATION** -- post-hoc artifact checks vs prompt-level self-reflection;
   which claim classes are checkable from an artifact (CONTENT) vs not (PROCESS); avoiding false
   assurance from a syntactic-only check.
3. **MUTATION TESTING of guard code** -- proving a new check can actually FAIL; vacuous-pass modes.

---

## Search queries run (three-variant discipline, `.claude/rules/research-gate.md` "Search-query composition")

| Variant | Query | Purpose |
|---|---|---|
| year-less canonical | `silent downgrade unsupported protocol option fail closed capability negotiation RFC` | downgrade as a defect class |
| year-less canonical | `LDAP control criticality TRUE unavailableCriticalExtension server MUST NOT perform operation unrecognized control` | the canonical "caller named an unsupported capability" rule |
| year-less canonical | `vacuity detection temporal logic antecedent failure property passes for the wrong reason` | formal-methods name for a vacuous guard |
| last-2-year | `mutation testing vacuous tests guards that cannot fail 2025 2026` | recency, sub-question 3 |
| last-2-year / frontier | `EviBound evidence-bound execution agent false completion claims verification gate artifact` | recency, sub-question 2 |
| current-year frontier | `agent self-reported process claims unverifiable artifact grounding verification 2026` | recency scan |

---

## Read in full (>=5 required; counts toward the gate) -- 7 fetched

| # | URL | Accessed | Kind | Fetched how | Key finding (verbatim where quoted) |
|---|-----|----------|------|-------------|--------------------------------------|
| 1 | https://www.rfc-editor.org/rfc/rfc9413.html | 2026-08-10 | official std (IETF BCP) | WebFetch, full HTML | §5.1: *"Choosing to generate fatal errors for unspecified conditions instead of attempting error recovery can ensure that faults receive attention."* §5.1: *"Intolerance toward violations of specification improves feedback for new implementations in particular."* §6: *"Hiding the consequences of protocol variations encourages the hiding of issues, which can conceal bugs and make them difficult to discover."* §4.1: *"These errors can become entrenched, forcing other implementations to be tolerant of those errors."* |
| 2 | https://www.rfc-editor.org/rfc/rfc7240.html | 2026-08-10 | official std (IETF) | WebFetch, full HTML | The **report-and-proceed** archetype. §2: *"A server that does not recognize or is unable to comply with particular preference tokens ... MUST ignore those tokens and continue processing instead of signaling an error."* Legal **only because** §3 supplies `Preference-Applied`, needed since §3: *"a client application might not be capable of reliably determining if the preference was (or was not) applied simply by examining the payload of the response."* §1 contrasts `Expect`: *"intermediaries and servers are required to reject any request that states unrecognized or unsupported expectations."* §2: *"preferences cannot be used as expectations."* |
| 3 | https://www.rfc-editor.org/rfc/rfc7507.html | 2026-08-10 | official std (IETF) | WebFetch, full HTML | Downgrade must be **detectable**, never silent. §1: *"there's a risk that active attackers could exploit the downgrade strategy to weaken the cryptographic security of connections"*; *"All unnecessary protocol downgrades are undesirable"*. §3: with the fallback signal present and a higher version available, the server *"MUST respond with a fatal inappropriate_fallback alert"*. §1 on why the silent-retry heuristic is unsafe: *"handshake errors due to network glitches could similarly be misinterpreted as interaction with a legacy server and result in a protocol downgrade."* |
| 4 | https://arxiv.org/html/2511.05524 | 2026-08-10 | preprint (arXiv, Nov 2025) | WebFetch, native arXiv HTML | EviBound. Abstract: Baseline A (prompt-level self-reflection only) = **"100% hallucination (8/8 claimed, 0/8 verified)"**; Baseline B (verification-only) = **25%**; dual gates = **"0% hallucination: 7/8 tasks verified and 1 task correctly blocked"**. §1.1: *"Prompt-level techniques like self-reflection and critique help with factual errors, but they can't guarantee artifacts actually exist"*; *"A system can still claim 'training converged' without ever creating a run_id or artifact file. Detection strategies alone won't close the gap."* Verification Gate = run_id queryable + artifacts present + status FINISHED. Overhead: **"≈8.3% execution time vs. Baseline A"** (§4.5). Limits: §5.3.1 cross-run checks out of scope; §5.3.2 binary, not a *"claim accuracy spectrum"*. |
| 5 | https://arxiv.org/html/2607.14890 | 2026-08-10 | preprint (arXiv, Jul 2026) | WebFetch, native arXiv HTML | Proof-or-Stop. §1: *"A self-report is not evidence; a log line saying 'All tests passed' is not evidence that the tests correspond to the code about to be merged"*; *"agent outputs may propose lifecycle claims, but do not themselves constitute lifecycle state."* Fail-closed: *"block DONE on stale, missing, or command-set-drifted proof"* (§4). Prose is out of gate scope: §3 *"Ordinary developer notes, design rationale, and documentation are advisory: they inform attention, never a gate"*; §4 *"a reviewer response saying 'LGTM' is not, by itself, an artifact that a later gate can re-check."* Measured: amplification 31/1800 -> 2/1800 (§5.2); cost ~+1.2x tokens, +49% wall-time. |
| 6 | https://arxiv.org/html/2607.25364v2 | 2026-08-10 | preprint (arXiv, Jul 2026) | WebFetch, native arXiv HTML | EBTE. §I: model rationales are *"neither authorization nor reliable introspection."* §IV-B splits claims into **authorizing** (tool identity, operation, resource, effect bound, destination -- checkable against authoritative state) and advisory: *"The intentSummary, toolReasonCode, instructionInfluence, and low-uncertainty declarations are non-authorizing"*, with reason codes *"never used to widen a decision."* §IV-C / §XVI is the syntactic-vs-semantic warning: the artifact *"scans for a fixed synthetic marker"* yet *"semantic detection of novel sensitive content remain[s] outside these structural checks."* |
| 7 | https://ar5iv.labs.arxiv.org/html/2103.07189 | 2026-08-10 | peer-reviewed (ICSE 2021, via ar5iv) | WebFetch, ar5iv HTML (pre-Dec-2023 paper) | Mutation testing at Google. §II: *"Any surviving mutant that is not detected by the test suite constitutes a concrete test goal"*; suppression rules *"filter out code that cannot result in productive mutants (e.g., logging statements)."* §II: *"Only one mutant is generated per line ... no more than seven mutants are reported per file in a changelist."* §IV-D: *"More than 90% of all lines have a mutant majority fate of 100% ... generating and reporting at most a single mutant per line is a valid optimization."* §IV-C1: *"For 1043 (70%) of the bugs, mutation testing would have reported a fault-coupled mutant."* Behaviour: exposure vs changed test hunks r_s=0.9 (p<.001); exposure vs survivability r_s=-0.50 (p<.001). |

Source-hierarchy note: the read-in-full set is 3x IETF standards-track + 3x arXiv preprints + 1x
peer-reviewed ICSE paper. **Zero community-tier sources are counted toward the gate**
(`.claude/rules/research-gate.md:62-71`).

---

## Identified but snippet-only (context; does NOT count toward gate) -- 27

| URL | Kind | Why not fetched in full |
|-----|------|-------------------------|
| https://arxiv.org/abs/2511.05524 | abstract page | superseded by the full HTML (row 4) |
| https://arxiv.org/html/2605.17998 | preprint (Verify-Gated Completion as Admission Control) | same finding class as rows 4-5; budget |
| https://arxiv.org/pdf/2603.25097 | preprint (ElephantBroker) | PDF-only; adjacent, not load-bearing |
| https://arxiv.org/pdf/2605.20312 | preprint (Pramana: protocol-layer claim verification in agent networks) | PDF-only; corroborates the attestation direction |
| https://arxiv.org/pdf/2607.05463 | preprint (Governable Individuals) | PDF-only; identity layer, tangential |
| https://arxiv.org/pdf/2604.24658 | preprint (agent-native research artifacts) | PDF-only; tangential |
| https://arxiv.org/pdf/2405.16205 | preprint (GeneAgent self-verification vs domain DBs) | PDF-only; cross-domain corroboration |
| https://ldap.com/ldap-result-code-reference-core-ldapv3-result-codes/ | industry reference | RFC 4511 §4.1.11 is the primary; snippet carried the rule |
| https://docs.ldap.com/ldap-sdk/docs/result-code-reference/core-ldapv3-result-codes.html | vendor SDK docs | duplicate of the above |
| https://learn.microsoft.com/en-us/openspecs/windows_protocols/ms-adts/3c5e87db-4728-4f29-b164-01dd7d7391ea | vendor spec (MS-ADTS LDAP controls) | corroborates criticality only |
| https://www.openldap.org/lists/ietf-ldapbis/200312/msg00052.html | mailing list | community tier; drafting history |
| https://datatracker.ietf.org/doc/html/rfc5939 | official std (SDP capability negotiation) | negotiation-mechanism design, tangential |
| https://www.rfc-editor.org/rfc/rfc7006.txt | official std | false hit |
| https://www.certguard.app/blog/tls-13-downgrade-attack-prevention | blog | community tier; RFC 7507 is primary |
| https://github.com/vercel/ai/issues/14413 | issue tracker (2026) | live instance of the class; community tier |
| https://github.com/grpc/grpc-java/issues/7774 | issue tracker | false hit |
| https://www.cs.toronto.edu/~chechik/courses05/csc2108/beer01.pdf | peer-reviewed (Beer et al., vacuity detection) | non-arXiv PDF; concept captured from snippet, cited AS snippet |
| https://link.springer.com/article/10.1007/s10703-014-0221-0 | peer-reviewed (Vacuity in practice: temporal antecedent failure) | paywalled |
| https://dl.acm.org/doi/10.1007/s10703-014-0221-0 | peer-reviewed (same, ACM DL) | paywalled |
| https://link.springer.com/chapter/10.1007/978-3-540-74407-8_33 | peer-reviewed (Temporal Antecedent Failure: Refining Vacuity) | paywalled |
| https://link.springer.com/article/10.1023/A:1008779610539 | peer-reviewed (Efficient Detection of Vacuity, FMSD) | paywalled |
| https://research.ibm.com/publications/vacuity-in-practice-temporal-antecedent-failure | vendor research index | pointer only |
| https://en.wikipedia.org/wiki/Mutation_testing | encyclopedia | community tier; equivalent-mutant definition only |
| https://arxiv.org/pdf/2103.01341 | preprint (What Are We Really Testing in Mutation Testing for ML?) | PDF; adjacent domain |
| https://arxiv.org/pdf/2104.11767 | preprint (mutation vs branch coverage, industrial) | PDF; corroborating |
| https://arxiv.org/pdf/1506.07330 | preprint (mutation testing as safety net for test refactoring) | PDF; corroborating |
| https://arxiv.org/pdf/2210.17215 | preprint (mutation testing optimisations, Clang) | PDF; tooling, not semantics |
| https://accu.org/journals/overload/20/108/vanlaenen_1929/ | practitioner journal | community tier |
| https://rareskills.io/post/solidity-mutation-testing | blog | community tier |
| https://paradigma-digital.medium.com/improving-test-quality-with-mutation-testing-af5eee5fc214 | blog | community tier (lowest) |
| https://medium.com/@pvginkel/my-ai-workflow-part-5-grounding-cite-or-dont-claim-8ee3f438ce49 | blog (2026) | community tier; "cite or don't claim" framing only |
| https://medium.com/@Micheal-Lanham/your-agent-will-lie-heres-how-to-stop-it-aa60fb1911f0 | blog (2026) | community tier |

**Unique URLs collected: 34** (7 read in full + 27 snippet-only).

---

## Recency scan (2024-2026) -- PERFORMED

Scoped passes were run for the 2024-2026 window on all three sub-questions.
**Result: 3 new findings that SUPERSEDE the prior canonical answer on sub-question 2; 0 that
supersede sub-questions 1 or 3.**

- **Sub-question 2 -- SUPERSEDED.** The pre-2024 answer was prompt-level self-critique
  (Reflexion / self-refine style). The 2025-2026 window overturns it: EviBound (arXiv 2511.05524,
  Nov 2025) measures prompt-level-only self-reflection at **100% false-completion claims
  (8/8 claimed, 0/8 verified)**, falling to **0%** only with a post-hoc gate that queries the
  artifact store; Proof-or-Stop (arXiv 2607.14890, Jul 2026) and EBTE (arXiv 2607.25364v2,
  Jul 2026) reach the same conclusion independently from lifecycle-control and tool-execution
  angles. Three independent in-window sources, no dissent found. This supersedes any design that
  asks the agent to self-assess.
- **Sub-question 1 -- NOT superseded.** RFC 9413 (2023, BCP) remains the current IETF position and
  nothing in 2024-2026 revises it. In-window hits are *instances* of the class rather than new
  theory (e.g. the 2026 vercel/ai issue #14413: a client that keeps emitting an un-negotiated
  protocol version). The canonical prior art -- LDAP criticality (RFC 4511 §4.1.11), `Expect`/417,
  TLS_FALLBACK_SCSV (RFC 7507) -- is stable and still the reference.
- **Sub-question 3 -- NOT superseded.** The Google industrial mutation-testing line (Petrovic et
  al., ICSE 2021 / arXiv 2103.07189) is still the reference account; in-window work concentrates on
  LLM-assisted mutant *generation*, which does not change the kill/survive semantics this step
  needs. The older formal-methods framing (vacuity / antecedent failure, Beer et al.) is likewise
  unrevised and remains the sharper vocabulary for "a check that passed for the wrong reason".

---

## Key findings

### A. Silent degradation -- the disposition turns on ABSENT vs UNSUPPORTED, and no source endorses silent substitution

1. **The two cases differ in kind, and protocol design keeps separate machinery for each.**
   An ABSENT parameter has no referent to violate, so defaulting is legitimate. An UNSUPPORTED
   parameter means the caller named a real capability this implementation lacks -- there IS a
   referent, and quietly substituting a weaker one yields a result the caller will mis-read.
   LDAP encodes this as a per-request *criticality* flag: an unrecognised control with criticality
   TRUE means the server must not perform the operation and returns `unavailableCriticalExtension`,
   while a non-critical one is ignored (RFC 4511 §4.1.11; corroborated from snippet at
   https://ldap.com/ldap-result-code-reference-core-ldapv3-result-codes/ and
   https://learn.microsoft.com/en-us/openspecs/windows_protocols/ms-adts/3c5e87db-4728-4f29-b164-01dd7d7391ea,
   accessed 2026-08-10 -- **snippet-tier, not counted toward the gate**). HTTP splits the same way:
   `Expect` -- *"intermediaries and servers are required to reject any request that states
   unrecognized or unsupported expectations"* -- versus `Prefer`, which MUST be ignored rather than
   erroring (RFC 7240 §1-§2, https://www.rfc-editor.org/rfc/rfc7240.html, accessed 2026-08-10).

2. **"Report-and-proceed" is legitimate ONLY when the report reaches the caller in the response.**
   RFC 7240 permits ignoring a preference and, in the same document, defines `Preference-Applied`
   because *"a client application might not be capable of reliably determining if the preference was
   (or was not) applied simply by examining the payload of the response"* (§3). The signal is what
   makes ignoring safe. A degradation disclosed only inside the artifact the caller is trying to
   assess is NOT this pattern -- it is the payload, not the response header.

3. **A downgrade that is not detectable is treated as a security defect, not a convenience.**
   RFC 7507 exists solely so an unintended TLS downgrade becomes visible; the server *"MUST respond
   with a fatal inappropriate_fallback alert"* (§3), and §1 records that the silent-retry heuristic
   misfires on unrelated inputs -- *"handshake errors due to network glitches could similarly be
   misinterpreted as interaction with a legacy server and result in a protocol downgrade"*
   (https://www.rfc-editor.org/rfc/rfc7507.html, accessed 2026-08-10). Directionality matters: the
   harm is specifically degrading a request for a STRICTER standard.

4. **IETF BCP prefers erroring over silent repair and names hiding as the harm.** RFC 9413 §5.1:
   *"Choosing to generate fatal errors for unspecified conditions instead of attempting error
   recovery can ensure that faults receive attention"*; §6: *"Hiding the consequences of protocol
   variations encourages the hiding of issues, which can conceal bugs and make them difficult to
   discover"*; §4.1: *"These errors can become entrenched, forcing other implementations to be
   tolerant of those errors."* (https://www.rfc-editor.org/rfc/rfc9413.html, accessed 2026-08-10.)

**Consensus vs debate.** Unanimous: silent substitution is never endorsed. The genuine debate is
only between *fail closed* (LDAP-critical, `Expect`/417, TLS `inappropriate_fallback`) and
*proceed-with-a-machine-readable-signal* (`Prefer`/`Preference-Applied`). The deciding variable in
every case is **whether the caller can detect the substitution from the response** -- not how severe
the substitution is. Both accepted options put the disclosure in the RESPONSE; neither buries it in
the payload.

### B. Artifact-grounded verification -- what is checkable, and where a check only looks like one

5. **Post-hoc artifact checks beat prompt-level self-reflection, measured, and the gap is total.**
   EviBound: prompt-level-only = *"100% hallucination (8/8 claimed, 0/8 verified)"*; dual gates =
   *"0% hallucination"*; overhead *"≈8.3% execution time"* (§4.5). §1.1: *"Prompt-level techniques
   like self-reflection and critique help with factual errors, but they can't guarantee artifacts
   actually exist."* (https://arxiv.org/html/2511.05524, accessed 2026-08-10.) Proof-or-Stop
   corroborates from lifecycle control -- *"A self-report is not evidence"* (§1) -- reducing
   amplified visible-pass/hidden-fail cases from 31/1800 to 2/1800 (§5.2)
   (https://arxiv.org/html/2607.14890, accessed 2026-08-10).

6. **The checkable/uncheckable line is CONTENT vs PROCESS, and mature systems draw it explicitly
   instead of pretending.** EBTE §IV-B keeps an authorizing set (tool identity, operation, resource,
   effect bound, destination -- comparable against authoritative state) and demotes the rest:
   *"The intentSummary, toolReasonCode, instructionInfluence, and low-uncertainty declarations are
   non-authorizing"* (https://arxiv.org/html/2607.25364v2, accessed 2026-08-10). Proof-or-Stop §3
   does the same: *"Ordinary developer notes, design rationale, and documentation are advisory: they
   inform attention, never a gate."* **The design move is not to make process claims checkable -- it
   is to label them non-authorizing so they cannot carry a pass.**

7. **Syntactic corroboration is the specific false-assurance trap, and it is documented as such.**
   EBTE ships a structural check (*"scans for a fixed synthetic marker"*) while stating that
   *"semantic detection of novel sensitive content remain[s] outside these structural checks"*
   (§IV-C, §XVI). A substring / field-presence check answers "is the token there", never "is the
   claim true". Proof-or-Stop puts it in artifact terms: *"a reviewer response saying 'LGTM' is not,
   by itself, an artifact that a later gate can re-check"* (§4). EviBound concedes its own residue:
   verification is per-run (*"Cross-run consistency checks ... are out of scope"*, §5.3.1) and binary
   rather than a *"claim accuracy spectrum"* (§5.3.2). **Corollary for design: a new syntactic check
   is still worth adding -- it is necessary-but-not-sufficient -- provided it is described as
   presence, not as proof of the process behind it.**

### C. Mutation testing of guard code

8. **A surviving mutant is the operational definition of a guard that is not load-bearing.**
   Google: *"Any surviving mutant that is not detected by the test suite constitutes a concrete test
   goal"* (§II). Coverage is no substitute -- a mutant on an unexecuted line can never be killed, so
   an unreached guard is indistinguishable from a correct one until mutated
   (https://ar5iv.labs.arxiv.org/html/2103.07189, accessed 2026-08-10).
9. **Two vacuity modes must be separated.** (i) *Equivalent / arid* mutants -- behaviour-preserving,
   so survival is not a test defect; Google suppresses these by rule (*"filter out code that cannot
   result in productive mutants"*, §II). (ii) A guard whose **precondition never holds in the
   fixture** -- the check passes for the wrong reason. Formal methods names (ii) *antecedent
   failure* / *vacuity*: the classic illustration is that *"every request is eventually followed by
   an acknowledgment" is satisfied vacuously by a system that never generates any requests*, and
   such passes are described as making a successful verification "meaningless" (Beer, Ben-David,
   Eisner, Rodeh -- **snippet-tier only**, https://www.cs.toronto.edu/~chechik/courses05/csc2108/beer01.pdf
   and https://link.springer.com/article/10.1007/s10703-014-0221-0, accessed 2026-08-10; NOT counted
   toward the gate). Only (ii) is a defect, and only mutation plus a positive control distinguishes
   them.
10. **Volume discipline.** *"Only one mutant is generated per line ... no more than seven mutants are
    reported per file in a changelist"* (§II), justified by *"More than 90% of all lines have a
    mutant majority fate of 100%"* (§IV-D). One well-chosen mutant per guard is the evidenced ratio,
    not a full matrix.

---

## Internal code inventory

| File | Lines | Role | Status |
|------|-------|------|--------|
| `.claude/workflows/research-gate.js` | 505 | the gate under study | LIVE; carries defects 1-3 |
| `.claude/workflows/research-gate.js:107-139` | `classifyArgs` | ABSENT / UNUSABLE / INCOMPLETE trichotomy at the args boundary | LIVE; **the in-house precedent to mirror** |
| `.claude/workflows/research-gate.js:141-155` | arg unpacking incl. tier | where the tier is silently substituted | LIVE; **defect-1 site** |
| `.claude/workflows/research-gate.js:157-159` | `FLOOR_SOURCES=5 / FLOOR_URLS=10 / K_REQUIRED=2` | the floors | LIVE; tier-independent |
| `.claude/workflows/research-gate.js:205-238` | `ENVELOPE_SCHEMA`; `tier: {enum: VALID_TIERS}` at :214 | wire contract | LIVE; **makes a `deep` self-report unrepresentable** |
| `.claude/workflows/research-gate.js:268-363` | `enforceGate` (pure) | recomputes the gate | LIVE; `opts.floors` (:269) has **no caller** |
| `.claude/workflows/research-gate.js:307,:310` | `urls_collected`, `recency_scan_performed` checks | floors compared against the envelope only | LIVE; **defect-2 sites** |
| `.claude/workflows/research-gate.js:321-352` | artifact cross-check | over-claim + URL-presence | LIVE; the only corroborated field |
| `.claude/workflows/research-gate.js:424-467` | stage-2 verifier + `BRIEF_VERIFICATION_SCHEMA` (:242-254) | independent artifact read | LIVE; **substring-only, and on self-selected URLs** |
| `.claude/workflows/research-gate.js:490-504` | return shape incl. `input_health` (:500) | degradation reported as its own field | LIVE; **the pattern to copy for tier** |
| `.claude/workflows/qa-verdict.js:52-76,:171-182` | sibling rail | same three-class args boundary; blind run returns NO VERDICT | LIVE; deliberate duplicate (runtime forbids imports, :35-37) |
| `.claude/agents/researcher.md:199-204` | tier table incl. the `deep` row | role definition | LIVE; **defines `deep`** |
| `.claude/agents/researcher.md:206-276` | `deep` requirements + gate check | >=20 sources, >=1 `[ADVERSARIAL]`, multi-pass, fork | LIVE; **unimplemented by the rail** |
| `.claude/agents/researcher.md:248-263` | multi-subagent fork | *"confirm with caller before forking"*, *"~1 Claude Max 5-hour rolling window per subagent"* | LIVE; why `deep` is not a drop-in |
| `.claude/agents/researcher.md:75` | launch doc says `agentType:'general-purpose'` | doc | **WRONG -- defect 3** |
| `.claude/rules/research-gate.md:20-31` | mandatory dedicated "Recency scan (last 2 years)" section | artifact proxy for a process claim | LIVE |
| `.claude/rules/research-gate.md:157-162` | URL-collection floor (10+) | floor | LIVE |
| `scripts/qa/verify_research_gate_workflow.mjs` | 282 | re-runnable checker | LIVE; **measured 40 passed / 0 failed, exit 0, 2026-08-10** |
| `scripts/qa/verify_research_gate_workflow.mjs:215-244` | mutation harness, 6 mutants | criterion-6 evidence | LIVE; **anchors on `const FLOOR_*` source text** |
| `scripts/qa/verify_research_gate_workflow.mjs:246-278` | structural assertions | schema / import / rider guards; :271 asserts `agentType:'researcher'` | LIVE |
| `.claude/masterplan.json` step `86.28` | -- | step definition, audit_basis, 9 criteria | read; scope + NON-SCOPE recorded below |

### Internal finding 1 -- the tier substitution is exactly the UNSUPPORTED case, and it is silent on every channel

`research-gate.js:147-150` (measured via grep, 2026-08-10):

```js
const VALID_TIERS = ['simple', 'moderate', 'complex']
const tierRaw = a.tier || 'moderate'
const tier = VALID_TIERS.includes(tierRaw) ? tierRaw : 'moderate'
const tierDefaulted = !a.tier || !VALID_TIERS.includes(tierRaw)
```

`tierDefaulted` is ONE boolean spanning **two different classes**, and the string it drives at `:173`
is factually false for one of them:

```js
'TIER: ' + tier + (tierDefaulted ? '  (NOT passed by the caller -- defaulted to moderate; state this assumption in the brief)' : ''),
```

- ABSENT (`a.tier` missing): *"NOT passed by the caller"* is **true**; defaulting is legitimate
  (the RFC 7240 class).
- UNSUPPORTED (`a.tier === 'deep'`): the caller **did** pass a tier, and named one that
  `.claude/agents/researcher.md:204` defines and `:206-276` specifies -- >=20 sources read in full,
  >=1 `[ADVERSARIAL]` source, an explicit pass-1/pass-2/pass-3 structure. The rail substitutes
  `moderate`, leaves `FLOOR_SOURCES = 5` (`:157`) untouched, and tells the agent *"NOT passed by the
  caller"* -- a statement the caller can verify is untrue. **Measured: `grep -c -i deep
  .claude/workflows/research-gate.js` returns 0** -- the rail has no concept of the tier at all.

Three independent invisibilities compound: (a) the returned object (`:490-504`) has **no** field
disclosing the substitution -- contrast `input_health` at `:500`, added by phase-86.17 precisely so
*"a caller must be able to tell 'failed the floors' apart from 'never had a subject'"* (`:497-499`);
(b) `ENVELOPE_SCHEMA.tier` is `enum: VALID_TIERS` (`:214`) with `additionalProperties:false`, so the
agent **cannot** report `tier:'deep'` even if it tried -- and unlike `minimum`/`minItems`, `enum` is
*not* stripped on the wire, so this constraint really binds; (c) the only trace is prose in the
agent's prompt, i.e. the payload, not the response (RFC 7240 §3).

It also contradicts the file's own doctrine 40 lines earlier (`:101-106`): *"DO NOT REPAIR
NEAR-MISSES (the house idiom, and RFC 9413's anti-workaround argument) ... Throwing costs the run at
this line, before a single token is spent; silently defaulting costs a full max-effort session AND
deposits a misfiled artifact."* Substituting `moderate` for `deep` is the same repair-a-near-miss
move with the same cost profile -- a full max-effort session producing a brief the caller will read
as meeting a 20-source standard that was never applied.

### Internal finding 2 -- three of four self-reported fields have no corroboration; one of them cannot have any

The stage-2 verifier prompt (`:437-456`) asks only whether each URL *"appear[s] as a literal
substring of the file"*; `BRIEF_VERIFICATION_SCHEMA` (`:242-254`) returns `brief_exists`,
`brief_non_empty`, `char_count`, `urls_checked`, `urls_present`, `urls_missing`. Mapping the
envelope against what that can actually establish:

| Envelope field | Claim class | Corroborated today? | Available artifact proxy |
|---|---|---|---|
| `sources_read_in_full[]` | CONTENT (URL is in the brief) | YES -- substring (`:347-352`) | already checked; **presence != "read in full"** (EBTE §XVI class) |
| `external_sources_read_in_full` | mixed | PARTIAL -- over-claim check at `:324` | count of distinct URLs in the read-in-full table |
| `urls_collected` (`:307`) | CONTENT-ish | **NO** -- compared against itself | count of distinct URLs anywhere in the brief; cheap, stage 2 already has the file |
| `recency_scan_performed` (`:310`) | **PROCESS** ("I ran a scoped search") | **NO** -- bare boolean | presence of the dedicated "Recency scan" section mandated by `.claude/rules/research-gate.md:20-31` (necessary, not sufficient) |
| `coverage.dry_rounds` / `dry` (`:315`) | **PROCESS** | **NO** | **none exists even in principle** -- "dry" is a property of executed discovery rounds, not of a file |
| `internal_files_inspected` | **PROCESS** | **NO** | count of `path:line` anchors in the brief (weak) |
| `tier` | input echo | **NO** | see finding 1 |

Two sharpenings the external literature supplies. First, the stage-2 check runs on URLs the
**researcher itself supplied**, so it is a *consistency* check, not an independent one: it proves the
agent wrote down what it said it wrote down. Strengthening means adding a claim the agent did not
choose -- e.g. "does a section titled *Recency scan* exist", "how many distinct URLs are in the
file". Second, `coverage.dry` belongs in EBTE's *non-authorizing* bucket (§IV-B) / Proof-or-Stop's
*advisory* bucket (§3): the honest disposition is to label it, not to fake a check for it. This
matches the step's own recorded out-of-scope reasoning.

### Internal finding 3 -- doc drift on `agentType` (measured)

`.claude/agents/researcher.md:75` states the gate spawns with `agentType:'general-purpose'`, while
the shipped code pins `agentType: 'researcher'` at `research-gate.js:419` and the checker ASSERTS
that pin at `scripts/qa/verify_research_gate_workflow.mjs:271` (*"agentType is 'researcher' (needs
Write for write-first)"*, currently GREEN). The pin is load-bearing: write-first requires `Write`,
and `research-gate.js:42-48` records that the Q/A rail's restrict-the-surface precedent deliberately
does not carry here. The code is correct; the doc is wrong. (Note `:461` pins `agentType: 'Explore'`
for the stage-2 verifier -- a different, intentional pin.)

### Internal finding 4 -- the checker already implements four anti-vacuity countermeasures worth preserving

`verify_research_gate_workflow.mjs:233-235` fails a mutant whose anchor string is missing (`anchor
not found`) and again when `mutated === src` (`replace was a no-op`) -- the harness refuses to score
a mutant it did not actually apply, which is the direct countermeasure to a `str.replace` that
silently matches nothing. `mutantKilled` (`:126-129`) counts a **throw** as a kill, on the stated
ground that the removed guard was the only thing preventing a crash. `[1]` at `:134-135` is the
positive control -- a compliant envelope must still pass -- which is what stops a new guard from
being "killed" by simply rejecting everything. `[6]` at `:209-213` asserts an honestly-false
self-report still passes when the floors hold, keeping the `const:true` trap representable.
**Measured baseline 2026-08-10, before any edit: `ALL GREEN: 40 passed, 0 failed`, exit 0**, with all
six existing mutants KILLED (five `let-a-bad-envelope-through`, one `threw: Cannot read properties of
null (reading 'brief_exists')`).

---

## Application to pyfinagent

1. **Classify the tier the way `classifyArgs` already classifies args** (`research-gate.js:107-139`).
   Three cases, not two: ABSENT -> default to moderate silently, exactly as today (RFC 7240 §2 class);
   SUPPORTED -> use it; UNSUPPORTED-but-real (`deep`) -> the LDAP-critical / `Expect`-417 class.
   Pick one of the two dispositions the literature endorses, never the third:
   - **fail closed** at the boundary, consistent with `classifyArgs`'s own class-B/C throws and its
     cost argument at `:101-106`; or
   - **proceed with a machine-readable signal** -- the `Preference-Applied` pattern -- which here
     means a distinct field in the RETURN VALUE, mirroring `input_health` at `:500` (requested tier,
     effective tier, reason), so the caller detects it without reading the brief.
   Either way `tierDefaulted` must be split: one boolean cannot carry two dispositions, and the
   `:173` string is false in the UNSUPPORTED branch. Note the step's criterion 2 requires the fact in
   the **return value**, "not only in the agent prompt" -- i.e. it has already chosen the
   response-channel requirement that RFC 7240 §3 supplies the rationale for.
2. **Do NOT enable `deep`, and say why in the handoff.** The step's NON-SCOPE is explicit ("do not
   add 'deep' to VALID_TIERS; do not implement producer fan-out in any form"), and the internal
   evidence supports it independently: `researcher.md:248-263` makes the fork a caller-gated,
   ~1-Max-window-per-subagent decision, while the rail is N=1 by construction -- one brief path
   (`:155`, `:182`), one stage-2 verifier bound to one path (`:430-441`), no cross-branch URL
   de-duplication. Enabling the tier name without raising the floor would be the *worst* option:
   it would look honoured and would not be. The correct output is disclosure for an operator
   decision (criterion 3).
3. **Leave `opts.floors` and `coverage.dry` alone -- and record the reason.** `opts.floors` (`:269`)
   has no caller (measured: the only production call is `enforceGate(envelope, verification,
   { inputHealth })` at `:469`; the checker calls the two-arg form at `:119` and `:198`). Its only
   consumer would be tier-aware floors, which depend on the `deep` decision that this step does not
   make -- wiring it now is speculative generality. `coverage.dry` is not establishable from a file
   even in principle (it asserts that executed search rounds happened and surfaced nothing new), so
   no read-only check can corroborate it; the literature's answer is to label it non-authorizing
   (EBTE §IV-B, Proof-or-Stop §3), not to fake a proxy.
4. **Corroborate the two fields that CAN be corroborated, and describe them accurately.** Extend
   `BRIEF_VERIFICATION_SCHEMA` (`:242-254`) with claims the researcher did not self-select: presence
   of the "Recency scan" section mandated by `.claude/rules/research-gate.md:20-31`, and a count of
   distinct URLs in the brief to check `urls_collected` against `FLOOR_URLS`. Preserve the
   fail-closed branch at `:336-340` unchanged (criterion 4). Frame both as **presence checks, not
   proof of process** -- EBTE §XVI is explicit that structural conformance is not semantic
   verification, and over-describing them would reproduce the very over-claim this step exists to
   remove.
5. **Every new check needs its own mutant, plus the harness's two existing safety rails and a
   positive control.** Follow `verify_research_gate_workflow.mjs:215-244`: anchor-present assertion,
   no-op-replace assertion, and a probe that PASSED before the mutation and FAILS after (criterion
   5 also requires the mutation output recorded verbatim). One mutant per guard is the evidenced
   ratio (Google §II, §IV-D), not a matrix. Add a **false-positive** case alongside each -- a
   legitimately-`moderate` or absent-tier call must NOT be flagged as degraded, and a brief that
   genuinely contains the recency section must still pass -- otherwise a guard that rejects
   everything scores as "killed" while being useless. This is the antecedent-failure mode: a check
   whose precondition never holds in the fixture passes for the wrong reason. Also note that any
   floor routed through `opts.floors` rather than the `const FLOOR_*` literals would be invisible to
   the existing source-text mutants -- a further reason not to move it.
6. **Watch the count, not just the colour.** Criterion 1 requires strictly more than 40 passing
   checks with none deleted or weakened; the pre-change baseline measured here is `40 passed, 0
   failed` (exit 0, 2026-08-10). EviBound's overhead figure (*"≈8.3% execution time"*, §4.5) is the
   evidence that a second artifact-reading pass is cheap relative to what it prevents -- stage 2
   already reads the file, so the added corroboration costs one prompt, not one agent.

---

## Pitfalls (from the literature, mapped)

- **Entrenchment.** RFC 9413 §4.1: *"These errors can become entrenched, forcing other
  implementations to be tolerant of those errors."* Every step that silently ran at `moderate` while
  asking for `deep` becomes precedent for the next.
- **The heuristic misfires on unrelated inputs.** RFC 7507 §1 (glitches misread as legacy servers):
  a catch-all `else -> moderate` cannot tell a typo from a deliberate stricter request. Any fix must
  distinguish "unknown string" from "known-but-unimplemented tier", or it just relocates the
  conflation.
- **Structural conformance mistaken for truth.** EBTE §XVI; already stated at
  `research-gate.js:18-21` -- the tier path is the one place the file does not follow its own rule.
- **Equivalent / arid mutants.** Google §II: a survivor may be behaviour-preserving rather than a
  test hole. Expect to justify, not merely record, any survivor.
- **Vacuous pass / antecedent failure.** A guard whose precondition never holds in the fixture
  passes meaninglessly (Beer et al., snippet-tier). The positive control and the false-positive case
  are the countermeasures, not the mutant alone.
- **Binary verification hides partial failure.** EviBound §5.3.2: present/absent, not a *"claim
  accuracy spectrum"* -- a brief can satisfy every count and still be thin. Corroborating
  `urls_collected` raises the floor; it does not make the brief good.

---

## Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch -- **7** (3x IETF standards-track,
      3x arXiv preprints, 1x ICSE peer-reviewed via ar5iv); zero community-tier in the counted set
- [x] 10+ unique URLs total -- **34**
- [x] Recency scan (last 2 years) performed + reported -- dedicated section above with an explicit
      supersede / no-supersede verdict per sub-question
- [x] Full pages read, not abstracts -- all 7 via WebFetch on full HTML; the arXiv chain was honoured
      (native `/html/` for the post-Dec-2023 papers, ar5iv for the 2021 paper; **no `arxiv.org/pdf/`
      URL was fetched**)
- [x] file:line anchors for every internal claim, and every measured claim re-derived in-session
      (`grep -c -i deep` = 0; checker baseline 40/0; `enforceGate(` call sites; `agentType` sites)

Soft checks:
- [x] Internal exploration covered the caller's stated scope: `research-gate.js` (`enforceGate`
      :268-363, args boundary :77-139, stage-2 :424-467, `VALID_TIERS`/`tierDefaulted`
      :147-150/:173), `researcher.md` (tier table :199-204, deep :206-276, fork :248-263),
      `research-gate.md` (recency :20-31, URL floor :157-162),
      `verify_research_gate_workflow.mjs` (40 checks + mutation harness :215-244), `qa-verdict.js`
      (sibling idiom), plus the masterplan step entry
- [x] Contradictions / consensus noted -- no source endorses silent substitution; the live debate is
      fail-closed vs signalled-proceed; the in-file contradiction between `:101-106` and `:149` is
      recorded, as is the researcher.md-vs-code `agentType` drift
- [x] All claims cited per-claim with URL + access date or file:line; snippet-tier evidence is
      labelled as such wherever it is used (LDAP criticality, vacuity/antecedent failure)
- [ ] **Disclosed gaps.** (a) RFC 4511 §4.1.11 was corroborated from vendor/industry snippets, not
      read in full -- it is NOT counted toward the gate; the LDAP argument would be stronger with the
      RFC read in full. (b) The vacuity/antecedent-failure literature (Beer et al.) is paywalled or
      non-arXiv PDF; used as snippet-tier only. (c) Tool-call budget exceeded the `moderate`
      guidance of <=18 (actual 21) to hold the >=5 read-in-full floor at 7 sources -- the floor
      governs over the budget per `.claude/agents/researcher.md:271-276`. (d) Brief length exceeds
      the `moderate` <=700-word target: the caller posed three separable sub-questions plus a
      five-file internal scope; the prose is dense but longer than tier guidance.

---

## JSON envelope

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 7,
  "sources_read_in_full": [
    "https://www.rfc-editor.org/rfc/rfc9413.html",
    "https://www.rfc-editor.org/rfc/rfc7240.html",
    "https://www.rfc-editor.org/rfc/rfc7507.html",
    "https://arxiv.org/html/2511.05524",
    "https://arxiv.org/html/2607.14890",
    "https://arxiv.org/html/2607.25364v2",
    "https://ar5iv.labs.arxiv.org/html/2103.07189"
  ],
  "snippet_only_sources": 27,
  "urls_collected": 34,
  "recency_scan_performed": true,
  "internal_files_inspected": 6,
  "coverage": {
    "audit_class": false,
    "rounds": 2,
    "dry_rounds": 0,
    "K_required": 2,
    "new_findings_last_round": 4,
    "dry": false
  },
  "brief_path": "handoff/current/research_brief_86.28.md",
  "gate_passed": true
}
```
