# Research Brief -- step 86.19

**Topic:** Identifier collisions in a hierarchical config document (`.claude/masterplan.json`),
and where to fix them: data vs resolver; type-blind depth-first `find_step`; fail-open gate design.
**Tier:** simple (caller-stated). **Audit-class:** NO (coverage reported for information only).
**Started:** 2026-08-10.

```json
{
  "brief_status": "COMPLETE",
  "tier": "simple",
  "external_sources_read_in_full": 8,
  "snippet_only_sources": 20,
  "urls_collected": 28,
  "recency_scan_performed": true,
  "internal_files_inspected": 8,
  "coverage": {
    "audit_class": false,
    "rounds": 2,
    "dry_rounds": 0,
    "K_required": 2,
    "new_findings_last_round": 2,
    "dry": false
  },
  "summary": "MEASURED: .claude/masterplan.json has 4 duplicate ids across 1348 id-bearing nodes -- 5.1/5.2/5.3 (live steps[] vs archived_legacy_steps[] in phases[36]) and phase-6.5 (a PHASE at /phases[13] vs a STEP at /phases[12]/steps[4]). live_check_gate.py:34-48 find_step is depth-first first-match over node.values(), so the winner is decided by JSON key-insertion order: archived_legacy_steps precedes steps, so the ARCHIVED twin wins 5.1 (status=pending) over the live done step. No colliding id currently carries a live_check, so all four return proceed today -- the defect is latent and armed. Literature is unanimous: duplicate ids in one document are a defect (W3C XML document-wide constraint; RFC 8259 'unpredictable'); 'first match wins' is not among RFC 8259's three recognised behaviours; ambiguity should be refused (PEP 20, JSON Schema SHOULD raise, C# CS0121); scoping the resolver (k8s namespaces) is the remedy that preserves provenance. Saltzer & Schroeder: an exclusion-shaped mechanism fails by permitting, unnoticed. Recommend scope+type-tag the resolver via ONE shared exclusion set (the drift seam is already open at test_phase_75_19:151-152), a distinct ambiguous token (never proceed), and a load-time uniqueness assertion -- per-type is green today (1230/114, 0 dups), cross-type fires exactly once on phase-6.5. Do NOT renumber the archive.",
  "brief_path": "handoff/current/research_brief_86.19.md",
  "gate_passed": true
}
```

> Envelope was born inert (`INCOMPLETE`, zeroed counts) at file creation per phase-86.37 and
> flipped to `COMPLETE` as the final act of the session.

---

## Sections (filled incrementally)

### Search queries run (three-variant discipline, `.claude/rules/research-gate.md`)
1. **Year-less canonical:** `duplicate identifier collision hierarchical document resolver scope vs data fix`
2. **Year-less canonical:** `soft delete tombstone archived records reuse identifier provenance immutability audit`
3. **Current-year frontier (2026):** `configuration file duplicate keys silent misconfiguration detection 2026`
4. **Last-2-year window (2025):** `ambiguous name resolution error rather than first match wins 2025 language design`

### Read in full (>=5 required; counts toward the gate)

| # | URL | Accessed | Kind | Fetched how | Key quote / finding |
|---|-----|----------|------|-------------|---------------------|
| 1 | https://www.rfc-editor.org/rfc/rfc8259.txt | 2026-08-10 | standard (IETF STD 90) | WebFetch, **quote re-verified verbatim via `curl`** | "The names within an object SHOULD be unique." / "An object whose names are all unique is interoperable in the sense that all software implementations receiving that object will agree on the name-value mappings." / "When the names within an object are not unique, the behavior of software that receives such an object is unpredictable." / "Many implementations report the last name/value pair only. Other implementations report an error or fail to parse the object, and some implementations report all of the name/value pairs, including duplicates." |
| 2 | https://www.w3.org/TR/xml/ | 2026-08-10 | W3C Recommendation | WebFetch, **quote re-verified verbatim via `curl`** | *Validity constraint: ID* -- "A name MUST NOT appear more than once in an XML document as a value of this type; i.e., ID values MUST uniquely identify the elements which bear them." *Validity constraint: One ID per Element Type* -- "An element type MUST NOT have more than one ID attribute specified." Uniqueness scope is **the whole document**, not a container. |
| 3 | https://web.mit.edu/Saltzer/www/publications/protection/Basic.html | 2026-08-10 | peer-reviewed (Saltzer & Schroeder, Proc. IEEE 1975) | WebFetch, **quote re-verified verbatim via `curl`** | *Fail-safe defaults* -- "Base access decisions on permission rather than exclusion... the default situation is lack of access". And the decisive sentence: "A design or implementation mistake in a mechanism that gives explicit permission tends to fail by refusing permission, a safe situation, since it will be quickly detected. On the other hand, a design or implementation mistake in a mechanism that explicitly excludes access tends to fail by allowing access, a failure which may go unnoticed in normal use." *Economy of mechanism* -- "errors that result in unwanted access paths will not be noticed during normal use". |
| 4 | https://peps.python.org/pep-0020/ | 2026-08-10 | official (PEP 20, Zen of Python) | WebFetch | "In the face of ambiguity, refuse the temptation to guess." / "Errors should never pass silently. / Unless explicitly silenced." / "Explicit is better than implicit." |
| 5 | https://kubernetes.io/docs/concepts/overview/working-with-objects/namespaces/ | 2026-08-10 | official docs | WebFetch | "Names of resources need to be unique within a namespace, but not across namespaces." And: "Namespace-based scoping is applicable only for namespaced objects ... and not for cluster-wide objects" -- i.e. the same system carries BOTH scoped and unscoped kinds and keeps them in separate lookup surfaces (`kubectl api-resources --namespaced=true/false`). |
| 6 | https://json-schema.org/draft/2020-12/json-schema-core | 2026-08-10 | specification | WebFetch | §9.1.2: "A schema MAY (and likely will) have multiple URIs, but there is no way for a URI to identify more than one schema. When multiple schemas try to identify as the same URI, validators SHOULD raise an error condition." §8.2.2 on repeated anchors: "The effect of specifying the same fragment name multiple times within the same resource ... is undefined. Implementations MAY raise an error if such usage is detected." |
| 7 | https://learn.microsoft.com/en-us/dotnet/csharp/language-reference/compiler-messages/overload-resolution | 2026-08-10 | official docs (ms.date 2024-09-10) | WebFetch | CS0121 *"The call is ambiguous between the following methods or properties"*; "These errors indicate there isn't one better overload than others." Remedy is **explicit disambiguation by the caller**: "In most of these cases, adding an explicit cast can specify which overload should be chosen." A mainstream compiler **refuses to pick** rather than taking the first candidate. |
| 8 | https://oneuptime.com/blog/post/2026-02-21-fix-ansible-duplicate-key-yaml/view | 2026-08-10 | industry blog (pub. **2026-02-21**) | WebFetch | Duplicate YAML keys: "it either raises an error or silently uses the last value, depending on your Ansible version and configuration"; remedy is `duplicate_dict_key = error` in `ansible.cfg` plus a lint rule (`yamllint -d "{rules: {key-duplicates: enable}}"`) so the defect is caught **before runtime**, not resolved arbitrarily at runtime. |

### Identified but snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|---|---|---|
| https://web.mit.edu/Saltzer/www/publications/protection/ | paper index | **Attempted WebFetch; returned abstract + TOC only** (nav-only page). Superseded by `/Basic.html` (row 3), which was read in full. |
| https://handwiki.org/wiki/Naming_collision | encyclopedia | community tier; concept already covered by rows 1-2 |
| https://en.wikipedia.org/wiki/Ambiguous_name_resolution | encyclopedia | community tier; ANR is a *deliberate* fuzzy-match feature, not a lookup contract |
| https://ldapwiki.com/wiki/Wiki.jsp?page=Ambiguous+Name+Resolution | community wiki | lowest tier; interesting counter-case (LDAP simple bind authenticates as the *first* object on a UPN collision -- a real-world "first match wins" hazard) |
| https://www.learncpp.com/cpp-tutorial/function-overload-resolution-and-ambiguous-matches/ | tutorial | superseded by row 7 (official docs) |
| https://thunderseethe.dev/posts/nameres-base/ | dev blog (Jan 2026) | name-resolution-as-a-pass framing; recency evidence, not load-bearing |
| https://github.com/microsoft/TypeScript/issues/62767 | issue tracker | anecdote about ambiguous error text |
| https://typescript.page/fix-duplicate-identifier-errors-typescript | tutorial | community tier |
| https://deverrors.com/errors/ts-duplicate-identifier | tutorial | community tier |
| https://labex.io/tutorials/cpp-how-to-fix-duplicate-identifier-errors-420667 | tutorial | community tier |
| https://arxiv.org/pdf/cs/0205071 | paper (OAI harvesting) | directly on-point (hierarchical harvest -> same record reachable twice -> collision policy) but PDF-only and pre-2003; snippet retained as prior art |
| https://www.dev-toolbox.tech/tools/yaml-formatter/examples/yaml-duplicate-keys | tool docs | duplicate-key linting, superseded by row 8 |
| https://abacktools.com/blog/yaml-cpp-duplicate-keys-behavior | vendor blog | "silent duplicates ... the wrong value is used in production with no error logged" -- same claim as row 8 |
| https://community.home-assistant.io/t/duplicate-key-error-in-configuration-yaml/685255 | forum | lowest tier |
| https://www.gomboc.ai/blog/5-common-iac-misconfigurations-to-avoid-in-2026 | vendor blog | 2026 recency corroboration only |
| https://www.paloaltonetworks.com/cyberpedia/security-misconfiguration-api8 | vendor | OWASP A05 framing; not specific to id collisions |
| https://www.nilus.be/blog/soft_deletes_vs_hard_deletes_in_data_architecture/ | consultancy blog | soft-delete/provenance argument; covered by rows 5-6 reasoning |
| https://streamkap.com/resources-and-guides/cdc-soft-deletes-tombstones | vendor | Kafka tombstone semantics (delete marker keeps the key) |
| https://arxiv.org/pdf/2606.30306 | survey (2026) | states the tension directly ("immutability and deletion are in direct tension"); PDF-only, off-domain (LLM agent memory) |
| https://dredyson.com/how-i-resolved-duplicate-document-ids-in-legaltech-e-discovery-platforms-a-definitive-step-by-step-guide-for-beginners-and-beyond-using-the-same-naming-collision-principles-from-game-develo/ | personal blog | lowest tier |

**URLs collected: 28** (8 read in full + 20 snippet-only).

### Recency scan (2024-2026)

Searched the 2024-2026 window explicitly (queries 3 and 4 above). Result: **2 new findings that
COMPLEMENT, and 0 that supersede, the canonical sources.**

1. **oneuptime, 2026-02-21** (read in full, row 8) -- the practitioner consensus has moved from
   "duplicate keys are a parser quirk" to "make the parser fail and lint it in CI":
   `duplicate_dict_key = error`, `yamllint key-duplicates`. This is the modern restatement of RFC
   8259's 2017 "behavior ... is unpredictable" -- same diagnosis, newer prescription (**detect at
   load, not at lookup**).
2. **Microsoft Learn, ms.date 2024-09-10** (read in full, row 7) -- C# 13's
   `OverloadResolutionPriorityAttribute` is a 2024 addition, and it is instructive: even when a
   language *does* let you break a tie, it makes the tiebreak an **explicit, declared priority**,
   never a positional accident. No 2024-2026 source anywhere in this scan advocates
   "first match wins" for identifier lookup.

Nothing in the window supersedes Saltzer & Schroeder (1975), W3C XML (2008), or RFC 8259 (2017);
they remain the canonical statements of fail-safe defaults and document-wide id uniqueness.

### Internal code inventory (MEASURED 2026-08-10)

| File / anchor | Role | Status |
|---|---|---|
| `.claude/hooks/lib/live_check_gate.py:34-48` `find_step` | depth-first, type-blind, **first-match-wins** walk over the whole masterplan dict | DEFECTIVE (returns the archived twin -- measured below) |
| `.claude/hooks/lib/live_check_gate.py:51-72` `gate_decision` | reads `step["verification"]["live_check"]`; `proceed` / `passed` / `skip` | fail-open by design (docstring :17-19: "NEVER raises -- argument / parse errors fail-open") |
| `scripts/meta/preflight_verify_masterplan.py:89-90` | `LIVE_STEP_CONTAINERS = frozenset({"steps","subphases"})`, `ARCHIVE_CONTAINERS = frozenset({"archived_legacy_steps","archived_dropped_steps"})` | REFERENCE exclusion set; **the only definition in the repo** |
| `scripts/meta/preflight_verify_masterplan.py:132-160` `iter_steps` | container-explicit walk; archive tag is **sticky** (`:149-150`), `superseded_record` never descended (`:147-148`) | CORRECT pattern to copy |
| `.claude/hooks/auto-commit-and-push.sh:70-91` `load_done_ids` | second whole-tree walker (embedded python heredoc); `{id: name}` for every `status=="done"` **anywhere in the tree** | NO container exclusion -> would count an archived `done` twin as a real step; dict-keying silently collapses the collision |
| `backend/tests/test_phase_75_19_preflight_calibration.py:151-152` | `@pytest.mark.parametrize("container", ["archived_legacy_steps","archived_dropped_steps"])` | **DRIFT SEAM ALREADY OPEN**: re-declares the two names as string literals instead of importing `ARCHIVE_CONTAINERS` |
| `.claude/hooks/archive-handoff.sh:102-103,124-128` | archive dir naming `phase-${sid#phase-}` | the archive namespace inherits the same collision (below) |

**The collision, measured** (`python3` walk over `.claude/masterplan.json`, 1348 id-bearing nodes):

| id | twin A | twin B | what `find_step` returns |
|---|---|---|---|
| `5.1` | `/phases[36]/steps[0]` status=**done** | `/phases[36]/archived_legacy_steps[0]` status=**pending** | **the ARCHIVED one** (status=pending) |
| `5.2` | `/phases[36]/steps[1]` pending | `/phases[36]/archived_legacy_steps[1]` pending | the archived one |
| `5.3` | `/phases[36]/steps[2]` pending | `/phases[36]/archived_legacy_steps[2]` pending | the archived one |
| `phase-6.5` | `/phases[12]/steps[4]` **(CHILD/step)** done | `/phases[13]` **(PARENT/phase)** done | **the STEP**, because `phases[12]` is walked before `phases[13]` |

Why the archive wins: `phases[36]` key insertion order is
`['id','name','status','notes','depends_on','gate','path_decision','open_issues','archived_legacy_steps','steps','_comments','archived_dropped_steps']`
-- `archived_legacy_steps` precedes `steps`, and `find_step` iterates `node.values()` in
insertion order (`live_check_gate.py:38-42`). **The winner is decided by JSON key order,
i.e. by whoever last hand-edited the file.**

**Blast radius today:** none of the 4 colliding ids currently carries a `live_check`, so
`gate_decision(...)` returns `proceed` for all four (measured). The defect is **latent and
armed**: the moment live step `5.1` (status=done) is given a `live_check`, the gate reads the
archived twin -- which has none -- and returns `proceed`, silently disarming the gate for a
step that asked for it. That is the exact fail-open shape described in (c): a confident
decision about the **wrong subject**.

**Archive namespace collision:** `handoff/archive/phase-5.1/` and `handoff/archive/phase-6.5/`
both EXIST on disk (`phase-5.2/`, `phase-5.3/` do not). `archive-handoff.sh:127` computes
`phase-${sid#phase-}`, so live step `5.1` and archived legacy `5.1` -- and step `phase-6.5`
and phase `phase-6.5` -- would share one directory. The id collision is not confined to the
JSON; it propagates into the on-disk artifact namespace.

**Green-ability of a uniqueness assertion (measured, same walk):**

| id-space | count | duplicates |
|---|---|---|
| live steps (`steps` + `subphases`, archives excluded) | 1230 | **0** |
| phases | 114 | **0** |
| cross-type (`phases` ∪ live steps) | 1344 | **1** -- `phase-6.5` |

So a **per-type** uniqueness assertion is green *today* (this matters: an immutable criterion
must be green-able -- `feedback_immutable_criteria_must_be_green_able`). A **cross-type**
assertion would fire exactly once, on `phase-6.5`, and is therefore only usable if 86.19 also
resolves that one, or scopes the assertion per type.

---

## Key findings (external, cited per claim)

1. **A duplicate id inside one document is a data defect by every relevant specification --
   uniqueness is document-wide, not container-wide.** W3C XML states the constraint at document
   scope: *"A name MUST NOT appear more than once in an XML document as a value of this type;
   i.e., ID values MUST uniquely identify the elements which bear them."*
   (https://www.w3.org/TR/xml/, accessed 2026-08-10). JSON says the same softer:
   *"The names within an object SHOULD be unique."*
   (https://www.rfc-editor.org/rfc/rfc8259.txt, accessed 2026-08-10).

2. **"First match wins" is not one of the recognised duplicate behaviours.** RFC 8259 enumerates
   exactly three observed ones: *"Many implementations report the last name/value pair only.
   Other implementations report an error or fail to parse the object, and some implementations
   report all of the name/value pairs, including duplicates."*
   (https://www.rfc-editor.org/rfc/rfc8259.txt). `find_step` implements a **fourth, unnamed**
   behaviour -- *first in JSON key-insertion order wins* -- which is the only one whose outcome
   is decided by whoever last hand-edited the file.

3. **Ambiguity should be refused, not guessed.** *"In the face of ambiguity, refuse the
   temptation to guess."* + *"Errors should never pass silently. / Unless explicitly silenced."*
   (https://peps.python.org/pep-0020/). JSON Schema makes it normative: *"When multiple schemas
   try to identify as the same URI, validators SHOULD raise an error condition."*
   (https://json-schema.org/draft/2020-12/json-schema-core §9.1.2). A production compiler does
   the same rather than picking a candidate: CS0121 *"The call is ambiguous between the following
   methods or properties"*, and the fix is caller-side disambiguation -- *"adding an explicit cast
   can specify which overload should be chosen"*
   (https://learn.microsoft.com/en-us/dotnet/csharp/language-reference/compiler-messages/overload-resolution).

4. **Scoping the RESOLVER is the standard remedy that also preserves the archive.** Kubernetes
   ships exactly this contract: *"Names of resources need to be unique within a namespace, but not
   across namespaces."* -- and it keeps namespaced and cluster-wide kinds in **separate lookup
   surfaces** (`kubectl api-resources --namespaced=true` / `=false`)
   (https://kubernetes.io/docs/concepts/overview/working-with-objects/namespaces/). Applied here:
   the archive is a different namespace from the live plan, and phases are a different *kind* from
   steps. Neither requires mutating history.

5. **A fail-open mistake in an EXCLUSION-shaped mechanism goes unnoticed; that is the whole
   hazard of (c).** Saltzer & Schroeder: *"A design or implementation mistake in a mechanism that
   gives explicit permission tends to fail by refusing permission, a safe situation, since it will
   be quickly detected. On the other hand, a design or implementation mistake in a mechanism that
   explicitly excludes access tends to fail by allowing access, a failure which may go unnoticed in
   normal use."* (https://web.mit.edu/Saltzer/www/publications/protection/Basic.html). The
   live_check gate is an *exclusion* mechanism (it holds a commit); a wrong-node lookup makes it
   permit, and permitting is the silent direction.

6. **Detect at LOAD, not at LOOKUP.** The 2026 practitioner prescription for exactly this class is
   to make the parser fail and lint it in CI: `duplicate_dict_key = error`, plus
   `yamllint -d "{rules: {key-duplicates: enable}}"`
   (https://oneuptime.com/blog/post/2026-02-21-fix-ansible-duplicate-key-yaml/view, pub.
   2026-02-21). Same shape as JSON Schema's "validators SHOULD raise". A load-time assertion is
   O(1) per file and catches BOTH collision kinds; a lookup-time fix only protects the one caller
   you patched.

## Consensus vs debate (external)

**Consensus (5 of 5 spec-tier sources agree):** duplicate identifiers in a single document are a
defect; the resolution must be *declared* (error, last-wins, or explicit priority), never
positional; and namespacing is the accepted way to let two same-named things coexist.

**Genuine debate -- data vs resolver.** The specs are about an **id space**, not about whether a
record may be retained, and that is where the two arguments in the objective actually meet:
- *Renumber/remove the archived twins* satisfies XML's document-wide reading literally, but XML
  itself shows the cost: the IDREF constraint requires *"each Name MUST match the value of an ID
  attribute on some element"* -- i.e. changing an id breaks every reference to it. Here the
  references are real and external to the JSON: `handoff/archive/phase-5.1/`, `handoff/harness_log.md`
  entries, commit subjects, and step `depends_on` edges.
- *Scope the resolver* is what Kubernetes and every module system actually do, and it is the only
  option that leaves provenance byte-identical. The tombstone/soft-delete literature agrees in
  spirit: a tombstone **keeps** the key and marks it dead rather than renumbering it
  (streamkap CDC, snippet-only), and the 2026 agent-memory survey states the underlying tension
  outright -- immutability and deletion are in direct tension (arXiv 2606.30306, snippet-only).

**No source found advocating "first match wins"** for identifier lookup. The one real-world
instance surfaced (LDAP simple bind authenticating as the *first* object on a UPN collision,
ldapwiki, snippet-only) is cited as a **hazard**, not a design.

## Pitfalls (from literature + measured here)

1. **Renumbering breaks dangling references** (W3C IDREF constraint) -- and `handoff/archive/phase-5.1/`
   already exists on disk, so the id is load-bearing outside the JSON.
2. **Two consumers drifting is not hypothetical here -- the seam is already open.**
   `ARCHIVE_CONTAINERS` is defined once (`preflight_verify_masterplan.py:90`) but
   `test_phase_75_19_preflight_calibration.py:151-152` **re-declares the two names as string
   literals** instead of importing the frozenset, and neither `live_check_gate.py` nor
   `auto-commit-and-push.sh:70-91` knows about the concept at all. Any fix that adds a *second*
   literal copy makes it three.
3. **Fail-open on ambiguity is indistinguishable from fail-open on "no gate".** Both make
   `gate_decision` print `proceed` (`live_check_gate.py:60,63,66,70`). An operator reading the log
   cannot tell "this step has no live_check" from "I could not tell which node you meant". The
   ambiguity signal must be a **distinct token**, not a shared one.
4. **Archive tagging must be sticky.** `iter_steps` carries `_container == "archived"` down through
   nested values (`preflight_verify_masterplan.py:149-150`). A naive one-level `if key in
   ARCHIVE_CONTAINERS: skip` misses anything nested deeper.
5. **An archive-exclusion fix does NOT close `phase-6.5`.** Both twins are LIVE (a phase at
   `/phases[13]` and a step at `/phases[12]/steps[4]`). That one needs **type-tagged** lookup, i.e.
   the `LIVE_STEP_CONTAINERS` half of the same idiom -- not the archive half.
6. **The hook cannot raise.** `live_check_gate.py:17-19` promises it "NEVER raises", and PostToolUse
   hooks cannot block anyway (`reference_claude_code_hooks_run_in_parallel`). So "make it loud"
   here means a distinct fail-**closed-ish** return token + a visible WARN, not an exception.

## Application to pyfinagent (external findings -> file:line anchors)

| Remedy | Anchor | Note |
|---|---|---|
| **R1 -- scope the resolver, don't mutate the archive** (finding 4) | rewrite `live_check_gate.py:34-48` to walk with container semantics, mirroring `preflight_verify_masterplan.py:132-160` | Answers (a): the archive stays byte-identical; provenance intact; `handoff/archive/phase-5.1/` keeps its meaning. |
| **R2 -- ONE shared exclusion set** (finding 4, pitfall 2) | export `ARCHIVE_CONTAINERS` / `LIVE_STEP_CONTAINERS` from a single module; make `preflight_verify_masterplan.py:89-90`, `live_check_gate.py`, `auto-commit-and-push.sh:70-91` and `test_phase_75_19_preflight_calibration.py:151-152` all consume it | The test's literal re-declaration is the drift seam to close, and closing it is *observable* (the parametrize becomes `ARCHIVE_CONTAINERS`). |
| **R3 -- type-tagged lookup** (finding 4, pitfall 5) | only nodes reached via `LIVE_STEP_CONTAINERS` may answer a *step* lookup | Kills `phase-6.5`: the phase at `/phases[13]` can no longer answer. Answers (b). |
| **R4 -- loud ambiguity, not arbitrary** (findings 2,3,5) | collect **all** matches in `find_step`; 0 -> `proceed` (unchanged: "no such step"), 1 -> decide, >1 -> a **distinct** ambiguous token that `auto-commit-and-push.sh` logs as WARN and treats as `skip` | Answers (c). `skip` is the safe direction per Saltzer: hold the commit and be detected, rather than permit and go unnoticed. Must NOT reuse `proceed` (pitfall 3). |
| **R5 -- assert uniqueness at load** (finding 6) | add a duplicate-id assertion to `preflight_verify_masterplan.py` (it already has the correct walk) | Per-type assertion is **green today** (1230 live steps / 114 phases, 0 dups each -- measured). A cross-type assertion fires exactly once (`phase-6.5`) and must therefore be scoped or sequenced after R3. |

**Recommended ordering:** R4 (loud) and R3 (type-tagged) are the two that change a *decision*;
R1/R2 are the structural way to express them without a second literal copy; R5 is the cheap
regression guard that stops the next twin from being added silently. **Data mutation (renumbering
the archived 5.1/5.2/5.3) is not recommended** -- it is the only option that destroys provenance
and breaks live on-disk references, and no source in this scan requires it once the resolver is
scoped.

---

## Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch -- **8** (spec/peer-reviewed/official-doc tier; 3 quote sets independently re-verified verbatim via `curl`)
- [x] 10+ unique URLs total -- **28** (8 full + 20 snippet-only)
- [x] Recency scan (2024-2026) performed + reported -- 2 complementary findings, 0 superseding
- [x] Full pages read (not abstracts) for the read-in-full set -- the Saltzer index page returned TOC-only and was **demoted to snippet-only**; `/Basic.html` was read instead
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every module in the caller's scope (`find_step`, `gate_decision`, `preflight_verify_masterplan.py:23/:90`, `load_done_ids`, `archive-handoff.sh` naming, archive dirs on disk, test `:151`)
- [x] Contradictions / consensus noted (data-vs-resolver debate is genuine; "first match wins" has no advocates)
- [x] All claims cited per-claim with URL + access date

**Note on quote integrity:** per `reference_webfetch_pdf_summaries_fabricate_quotes`, the three
load-bearing quote sets (RFC 8259, W3C XML, Saltzer & Schroeder) were re-extracted from raw source
via `curl | sed 's/<[^>]*>//g' | grep` and matched verbatim. The Kubernetes, PEP 20, JSON Schema,
MS Learn and oneuptime quotes are WebFetch-derived and not independently re-verified.
