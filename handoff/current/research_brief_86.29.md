# Research Brief -- phase-86.29

> ## GATE FAILED -- THIS BRIEF IS PARTIAL. DO NOT PROCEED TO A CONTRACT ON IT.
>
> The research-gate rail **DROPPED** on 2026-08-10: run `wf_f23b7949-ea3`,
> `agent({schema}): subagent completed without calling StructuredOutput`, after
> **181,082 subagent tokens and 68 tool uses**.
>
> `.claude/rules/research-gate.md` is explicit: the script "treats an empty or
> errored return as a FAILED gate, never as `gate_passed`". So **the gate has
> NOT passed for 86.29**, and no contract may be written from this file.
>
> This is confirmed by the artifact itself, not just the rail: it carries **no
> JSON envelope** and **no `STATUS: COMPLETE`**, and it stops mid-way through
> "External research -- rounds 5-6 (audit-class loop continues)". The researcher
> was still looping.
>
> **Main did NOT fabricate an envelope and did NOT infer a pass** from the fact
> that 15 sources happen to be present. A count is not a gate.
>
> **What survived is still valuable** -- this is the researcher rail's write-first
> discipline working exactly as designed, the same property phase-86.31 gave the
> Q/A rail this morning. The next session should re-run the gate and may use this
> as a head start, but must let the re-run stand on its own envelope.
>
> Preserved verbatim below; nothing was edited.

**Topic:** Archiving + provenance for append-only audit trails when a naming
convention and its consumer silently disagree.
**Tier:** moderate (caller-stated). **Audit-class:** YES (loop-until-dry, K=2).
**Started:** 2026-08-10. Researcher = Layer-3 combined external + internal.

Sub-questions from the caller:
- (a) silent no-match as a defect class: a glob matches ZERO files and the code
  falls back to a different source instead of erroring. Fail-fast vs fail-silent
  for provenance systems.
- (b) classifying an already-written corpus of archive dirs when the classifier
  itself may be incomplete: validating a detection method's RECALL against known
  positives BEFORE trusting its census; reporting the UNCLASSIFIED remainder
  honestly instead of folding it into "clean".
- (c) remediating already-wrong historical archive entries: repair-in-place vs
  leave-and-annotate vs regenerate; what audit/provenance literature says about
  mutating an audit trail after the fact (immutability, tamper-evidence, WORM).
- (d) deriving destination filenames from an identifier instead of two
  independent sites agreeing on a convention by hand; detecting producer/consumer
  naming drift.

STATUS: IN PROGRESS -- written incrementally, write-first per
`.claude/agents/researcher.md`.

---

## Search queries run (three-variant discipline)

(filled in below as rounds complete)

---
## Internal inventory -- PART 1: the defect, measured

### The two branches of `.claude/hooks/archive-handoff.sh`

| Branch | Lines | Verb | Pattern it uses |
|---|---|---|---|
| Rolling phase-level files | `:146-152` | **COPY** | fixed literal list `contract.md experiment_results.md evaluator_critique.md research.md research_brief.md` (UNSUFFIXED) |
| Step-specific files | `:160-169` | **MOVE** (`git mv`, fallback `mv`) | `"$CURRENT_DIR/${sid}-"*.md` and `"$CURRENT_DIR/phase-${sid}-"*.md` (**PREFIX** form) |

Verbatim, `archive-handoff.sh:146`:
```bash
for f in contract.md experiment_results.md evaluator_critique.md research.md research_brief.md; do
```
Verbatim, `archive-handoff.sh:160`:
```bash
for f in "$CURRENT_DIR/${sid}-"*.md "$CURRENT_DIR/phase-${sid}-"*.md; do
```

### MEASURED: the prefix globs match ZERO files today, and have since 2026-04-24

- `ls handoff/current/ | grep -cE '^(phase-)?[0-9]+(\.[0-9]+)*-.*\.md$'` -> **0**
- `ls handoff/current/ | grep -cE '_[0-9]+(\.[0-9]+)*\.md$'` -> **385** (the live
  convention is the **SUFFIX** form `contract_86.31.md`, `research_brief_86.29.md`)
- The prefix form is NOT a strawman -- it *was* real: 283 prefix-form files were
  added under `handoff/current/` across git history, and 698 prefix-form `.md`
  files sit in `handoff/archive/` today. So the glob was live once.
- Last prefix-form add: commit `1122a021` (2026-04-24). First suffix-form add:
  `fc1685d1` (2026-05-27). **The producer's convention changed and the
  consumer's glob did not.** The MOVE branch has been dead ~3.5 months.

### The silent fallback (this is the actual harm, not the dead glob)

The COPY branch does not error when the step-specific files are missing --
it copies whatever unsuffixed rolling file happens to exist. Those rolling
files are stale and **mutually inconsistent**:

| Rolling file | mtime | Step it declares in line 1 |
|---|---|---|
| `handoff/current/contract.md` | 2026-08-06 13:05 | `# Contract -- phase-82.54` |
| `handoff/current/experiment_results.md` | 2026-08-06 20:40 | `phase-82.6` |
| `handoff/current/evaluator_critique.md` | 2026-08-06 20:40 | `phase-82.6` |
| `handoff/current/research_brief.md` | 2026-07-25 17:58 | `phase-80.2` |
| `handoff/current/research.md` | ABSENT | (never existed under this name) |

So every archive dir minted after 2026-08-06 receives a `contract.md` that is
phase-82.54's, filed under a different step's directory name. Confirmed:
`handoff/archive/phase-86.2/contract.md` and
`handoff/archive/phase-85.4/contract.md` **both** begin `# Contract -- phase-82.54`.
That is a provenance forgery by omission: the artifact asserts a step it does
not belong to, and nothing in the pipeline said so.

### Archive census (raw)
- `handoff/archive/phase-*/` directories: **818**
- with a `contract.md`: **794**
- without a `contract.md`: **24**

---

## Internal inventory -- PART 2: the census, and how far it can be trusted

Classifier (scratchpad `census.py`): for each `handoff/archive/phase-<sid>/`,
extract the step id **declared inside** `contract.md` (six regex variants over
the first 45 lines: H1 `# Contract -- phase-N` / `# Sprint Contract -- ...`,
`**Step:**`, `step_id:`, `# phase-N`, table `| Step | N`), compare to `<sid>`.

| Class | Count |
|---|---|
| MATCH (declared id == dir id) | 373 |
| **MISMATCH** (declared id != dir id) | **196** |
| **UNCLASSIFIED** (no id extractable) | **225** |
| NO_CONTRACT (dir has no contract.md) | 24 |
| TOTAL dirs | 818 |

### Recall validation BEFORE trusting the census (the caller's requirement)

Known-positive set derived **independently of the classifier**: 69 distinct
`contract.md` blobs are byte-identical duplicates spanning 348 directories. A
duplicate blob covering N dirs means at least N-1 are misfiled. For every dup
group whose blob declares an id, each member dir whose `<sid>` != that id is a
KNOWN positive.

- known positives: **140**
- caught by the classifier: **140**
- missed: **0**
- **recall = 140/140 = 1.000** on the duplicate-derived population.

### What that recall number does NOT license

Recall was measured only on the population the classifier can read. It says
**nothing** about the 225 UNCLASSIFIED. 206 of those begin
`# Sprint Contract -- Cycle <N>` -- a **cycle** number, not a step id, so they
are *structurally* unclassifiable by any id-extraction method, not merely
missed by mine. Pattern-usage telemetry confirms the classifier is thin:
patterns #0 (514 hits) and #4 (54) carry it, #1 fired once, #2/#3/#5 never
fired. **The honest report is 4 buckets, not 2.** Folding UNCLASSIFIED into
"clean" would turn a 373/818 = 45.6% verified-clean rate into a fabricated
598/818 = 73.1%.

Largest duplicate groups (the smoking guns):

| md5 (first 10) | dirs | sample members |
|---|---|---|
| `9457c44682` | **32** | phase-36.17, 36.27, 4000.2, 4000.3, 62.1, 82.51, 82.54, 82.58, 82.59, 82.6, 83.0, 83.0.1, 83.0.3, 83.1, 83.1.1, 84.1, 85.3, 85.4, 85.5, 85.5.1, 85.6, 86.1, 86.12, 86.17, 86.2, 86.20, 86.22, 86.24, 86.26, 86.27, 86.3, 86.6 |
| `7fbb6eaab6` | 26 | phase-8.3, 8.4, 8.5.0, 10.0, 10.1, 10.2 ... |
| `f5084a5990` | 19 | phase-6.5.1, 6.5.2, 6.5.7, 6.5.9, 7.0, 7.1 ... |

The 32-dir group is **the live rolling `handoff/current/contract.md`**
(md5 `9457c44682beded6d94db1d017eb3df9`), i.e. phase-82.54's contract. 31 of
those 32 archive dirs carry a contract that is not theirs, including every
phase-86 step closed so far. This is the defect actively in flight.

### Independent corroboration of MISMATCH (not classifier-internal)

Spot samples with the wrong content plainly visible in line 1:
- `handoff/archive/phase-17.4/contract.md` -> `# Contract -- phase-62.2: Inbound operator-token handler (Socket-Mode bot)`
- `handoff/archive/phase-16/contract.md` -> `# phase-45.0 -- CLOSURE Re-Audit + Master Sequencing Plan`
- `handoff/archive/phase-11.4/contract.md` -> `# Sprint Contract -- phase-11.3 Migrate Complex Vertex Callers`

---

## Internal inventory -- PART 3: the SAME prefix assumption exists at 3 sites

| Site | Anchor | Pattern | Form |
|---|---|---|---|
| archive hook, MOVE branch | `.claude/hooks/archive-handoff.sh:160` | `${sid}-*.md`, `phase-${sid}-*.md` | PREFIX |
| layout verifier | `scripts/housekeeping/verify_handoff_layout.py:56` | `STEP_ID_RE = ^(?:phase-)?([0-9]+(?:\.[0-9]+)*)[-.].*\.md$` | PREFIX |
| backfill | `scripts/housekeeping/backfill_handoff_archive.py` (same `STEP_ID_RE`) | same | PREFIX |

`contract_86.29.md` (the live suffix convention) matches **none** of them.
`backfill_handoff_archive.py`'s docstring is explicit that non-conforming files
go to `handoff/archive/misc/` -- so the backfill's remedy for the *current*
convention is to sweep it into a junk drawer. This exact class already burned
the project once: phase-81.0 -- "STEP_ID_RE matched 0/127 so the gate lost its
input" (verdict gate dark for 13 consecutive step closes). Both scripts also
carry hand-maintained "keep byte-identical to the other script" comments --
i.e. the codebase already knows duplicated conventions drift, and has been
mitigating with comments rather than a shared derivation.

Hook wiring: `.claude/settings.json:90` and `:116` -- `archive-handoff.sh` is
registered under **two** PostToolUse matchers (Write and Edit on the
masterplan). Per `reference_claude_code_hooks_run_in_parallel`, hooks under one
matcher run in PARALLEL, so no ordering guarantee vs `live_check_gate` /
`auto-commit-and-push`.

---

## External research

### Search queries run (three-variant discipline)

| # | Query | Variant |
|---|---|---|
| 1 | audit trail immutability append-only tamper-evident annotate instead of rewrite | year-less canonical |
| 2 | bash nullglob failglob glob matches no files silent failure shell script | year-less canonical |
| 3 | GxP data integrity ALCOA audit trail must not obscure previously recorded information | year-less canonical |
| 4 | quasi-gold standard search recall validation systematic review software engineering | year-less canonical |
| 5 | silent failure fail-fast provenance pipeline 2026 | current-year frontier |
| 6 | audit log correction compensating entry immutable 2025 | last-2-year window |

### Sources READ IN FULL (WebFetch / curl+tag-strip)

| # | URL | Accessed | Kind | Fetched how | Key finding |
|---|---|---|---|---|---|
| 1 | https://www.gnu.org/software/bash/manual/html_node/The-Shopt-Builtin.html | 2026-08-10 | official doc | WebFetch | `nullglob`: "filename expansion patterns which match no files ... expand to nothing and are removed, rather than expanding to themselves." `failglob`: "patterns which fail to match filenames during filename expansion result in an expansion error." The manual does NOT state the default, nor the both-set precedence -- so the "failglob wins" claim seen in search snippets is NOT corroborated by the manual and I do not assert it. |
| 2 | https://mywiki.wooledge.org/glob | 2026-08-10 | authoritative community ref | WebFetch | Default is literal pass-through, and that is the hazard: "`tar xvf *.tar` without matching files becomes `tar xvf *.tar`, which may create a file literally named `*.tar` rather than signaling an error." Recommended POSIX idiom for maybe-empty loops is an explicit existence check: `[ -e "$x" ] || break`. |
| 3 | https://martinfowler.com/eaaDev/EventSourcing.html | 2026-08-10 | authoritative blog (named practitioner) | WebFetch | The canonical answer to "a past record is wrong": **reverse, do not edit.** "If we find a past event was incorrect, we can compute the consequences by reversing it and later events and then replaying the new event and later events." And a warning against the bi-temporal rabbit hole: "Clearly this stuff can get very messy, don't go down this path unless you really need to." |
| 4 | https://www.ecfr.gov/api/renderer/v1/content/enhanced/current/title-21?chapter=I&subchapter=A&part=11&subpart=B&section=11.10 | 2026-08-10 | official regulation (eCFR API, curl + tag-strip; the HTML page 302s to an unblock interstitial) | curl+strip, full section | 21 CFR 11.10(e) verbatim: "Use of secure, computer-generated, time-stamped audit trails to independently record the date and time of operator entries and actions that create, modify, or delete electronic records. **Record changes shall not obscure previously recorded information.**" Also 11.10(a): validation must ensure "the ability to discern invalid or altered records" -- i.e. detectability of a bad record is itself a required control, and 11.10(f): "operational system checks to enforce permitted sequencing of steps and events." |
| 5 | https://docs.pact.io/ | 2026-08-10 | official doc | WebFetch | The producer/consumer drift answer: derive the contract from the consumer's real behaviour, not from a hand-kept agreement. "The contract is generated during the execution of the automated consumer tests" and "only parts of the communication that are actually used by the consumer(s) get tested"; a provider change that breaks a consumer expectation fails verification instead of failing silently. |

| 6 | https://git-scm.com/docs/git-notes | 2026-08-10 | official doc | WebFetch | The annotate-don't-rewrite primitive, in the very VCS this archive lives in: notes "add, remove, or read notes attached to objects, **without touching the objects themselves**"; "A typical use of notes is to supplement a commit message without changing the commit itself." Amending would change the SHA; a note does not. |
| 7 | https://pdfs.semanticscholar.org/59d3/ec40b4f17ed94dc5ae510c316ac511915031.pdf (Zhang & Ali Babar, *On Searching Relevant Studies in Software Engineering*, EASE/BCS) | 2026-08-10 | peer-reviewed paper | curl + `pdfplumber` (43,452 chars extracted; per `.claude/rules/research-gate.md` step 3 -- not arXiv, so no `/html/` route) | The formal name for "validate the detector's recall before trusting its census": a **quasi-gold standard**. "as the gold standard for the subject is unknown, the corresponding sensitivity cannot be calculated ... our search approach uses the quasi-gold standard (from the manually selected sources) to measure sensitivity instead of the search universe." And a numeric bar: "we suggest a **threshold between 75% and 85%** as a reference for sensitivity evaluation of search performance." Also the crucial caveat that quasi-sensitivity is measured against the QGS pool, NOT the universe. |
| 8 | https://www.usenix.org/system/files/conference/osdi14/osdi14-paper-yuan.pdf (Yuan et al., *Simple Testing Can Prevent Most Critical Failures*, OSDI '14) | 2026-08-10 | peer-reviewed paper (USENIX) | curl + `pdfplumber` (80,839 chars extracted) | The empirical case for fail-loud. Finding 10: "**Almost all catastrophic failures (92%) are the result of incorrect handling of non-fatal errors explicitly signaled in software.**" And the exact anti-pattern this hook implements: "in 35% of the catastrophic failures, the faults in the error handling code fall into three trivial patterns: (i) **the error handler is simply empty or only contains a log printing statement**, (ii) the error handler aborts the cluster on an overly-general exception, and (iii) ... 'FIXME' or 'TODO'". |
| 9 | https://www.w3.org/TR/prov-dm/ | 2026-08-10 | official standard (W3C Recommendation) | WebFetch | Entities are defined by **fixed aspects** -- a changed thing is a NEW entity, not a mutated one. "A revision is a derivation for which the resulting entity is a revised version of some original", and both versions coexist in the provenance record. Also: "Provenance is information **about** entities, activities, and people involved in producing a piece of data or thing" -- i.e. the provenance record is a distinct object from the artifact, which is why you can annotate one without rewriting the other. |
| 10 | https://arxiv.org/html/2601.20727 (*Audit Trails for Accountability in Large Language Models*) | 2026-08-10 | preprint (arXiv, 2026-01 by ID) | WebFetch (arXiv native HTML per the doc chain) | Recency anchor. Store layer is a hash-chained append-only NDJSON log (`prev_hash -> curr_hash`, GENESIS anchor) so "in-place modification or deletion detectable"; "Once recorded, entries are tamper evident or read only". Names the failure mode we have: records that "**omit governance details such as who approved a change, under what conditions, and for what reasons**", scattered "across experiment trackers, CI logs, configuration files". Notably it offers **no** post-hoc correction mechanism -- only additional forward-appended governance events. (The fetch summary printed "2025"; the arXiv ID `2601.*` is 2026-01. I report the ID, not the summarizer's date.) |


### Snippet-only URLs (context; do NOT count toward the gate)

| URL | Kind | Why not fetched in full |
|---|---|---|
| https://medium.com/@veritaschain/append-only-is-the-easy-part-e25820208213 | community blog | tier-5; the operating-vs-architecting point is already covered by source #10 |
| https://dzone.com/articles/sql-server-ledger-tamper-evident-audit-trails | community blog | vendor-specific (SQL Server Ledger); no new principle |
| https://tracehold.ai/blog/immutable-audit-log-hmac-hash-chain/ | vendor blog | hash-chain mechanics duplicated by source #10 |
| https://audit-ready.eu/en/blog/audit-trail-best-practices | vendor blog | DORA retention; out of scope |
| https://www.endpointdev.com/blog/2016/12/bash-loop-wildcards-nullglob-failglob/ | practitioner blog | superseded by sources #1 + #2 (official + canonical) |
| https://blogs.gentoo.org/mgorny/2014/09/05/bash-pitfalls-globbing-everywhere/ | practitioner blog | same |
| https://bash-hackers.gabe565.com/syntax/expansion/globs/ | community wiki | same |
| https://intuitionlabs.ai/articles/audit-trail-requirements-ai-gxp-compliance | vendor article | secondary gloss on 21 CFR 11 -- source #4 is the primary |
| https://www.quanticate.com/blog/alcoa-principles | industry blog | ALCOA++ summary; primary reg text obtained instead |
| https://www.rephine.com/resources/blog/understanding-alcoa-and-data-integrity-in-gxp-compliance/ | industry blog | same |
| https://seattledataguy.substack.com/p/the-5-silent-failures-in-data-pipelines | community newsletter | tier-5; superseded by source #8 (peer-reviewed) |
| https://medium.com/towards-data-engineering/the-silent-failures-your-data-tests-arent-catching-09a56ecc0421 | community blog | tier-5 |
| https://blog.anomalyarmor.ai/data-pipeline-monitoring-how-to-stop-silent-failures-before-they-hit-production/ | vendor blog | tier-5; "never let 'it ran' stand in for 'it's correct'" is the only reusable line |
| https://www.e-informatyka.pl/index.php/einformatica/volumes/volume-2022/issue-1/article-3/ | peer-reviewed | **attempted full fetch and FAILED** -- the page served only metadata/abstract/references, no body text. Recorded honestly as snippet-only; the QGS method was obtained from the primary (source #7) instead. |
| https://dl.acm.org/doi/10.1016/j.infsof.2010.12.010 | peer-reviewed | paywalled ACM DL; the same authors' open PDF was fetched as source #7 |
| https://www.confluent.io/blog/schema-management-costs/ | industry blog | schema-registry-as-SoT; Pact (source #5) covers the mechanism at official-doc tier |
| https://www.conduktor.io/glossary/schema-registry-and-schema-management | vendor glossary | same |
| https://www.usenix.org/legacy/events/hotos03/tech/full_papers/candea/candea.pdf | peer-reviewed (crash-only software) | identified as prior art for fail-fast; source #8 is the stronger empirical anchor and was fetched instead |
| https://arxiv.org/pdf/2602.10916 | preprint (participation ledger) | governance-ledger adjacent; no incremental finding over #10 |

---

## Internal inventory -- PART 4: is the damage repairable, and from what?

Measured recoverability of a correctly-named `contract_<sid>.md` for each of the
818 archive dirs (worktree `handoff/current/` first, then all of git history):

| Route | dirs |
|---|---|
| present in worktree `handoff/current/` | 65 |
| present only in git history | 32 |
| not recoverable by that route | 721 |

Cross-tabulated against the classifier:

| state | worktree | git history | not recoverable |
|---|---|---|---|
| MATCH (373) | 11 | 15 | 347 |
| **MISMATCH (196)** | **54** | **17** | 125 |
| UNCLASSIFIED (225) | 0 | 0 | 225 |
| NO_CONTRACT (24) | 0 | 0 | 24 |

**All 32 dirs holding phase-82.54's contract have their correct
`contract_<sid>.md` sitting in `handoff/current/` right now.** They were never
archived because the MOVE glob is dead -- which is also why
`handoff/current/` has grown to **519 entries** (385 of them suffix-named).
The dead glob and the wrong archive content are the *same* bug seen from two
ends: the per-step file is not removed from `current/`, and the archive gets
the rolling file instead.

A second, independent recovery route exists for the prefix era: **245 of 818**
dirs already contain at least one prefix-form per-step artifact
(`phase-<sid>-contract.md` etc.) co-located next to the wrong `contract.md`.
Breakdown: 84 UNCLASSIFIED dirs, 13 MATCH, 12 of the 24 NO_CONTRACT dirs, 1
MISMATCH. So "NO_CONTRACT" does **not** equal "no evidence" -- half of that
bucket has the real per-step contract under the old name. Another reason not
to collapse buckets.

---

## External research -- rounds 5-6 (audit-class loop continues)

| # | URL | Accessed | Kind | Fetched how | Key finding |
|---|---|---|---|---|---|
| 11 | https://ccsds.org/Pubs/650x0m3.pdf (CCSDS 650.0-M-3, *Reference Model for an OAIS*, **December 2024** -- the ISO 14721 lineage) | 2026-08-10 | official standard | curl + `pdfplumber` (368,258 chars extracted) | **The single most design-deciding external finding.** Archival practice does NOT forbid altering an archived object; it forbids **undocumented** alteration: "Fixity Information documents the mechanisms that ensure that the Content Data Object has not been subject to **undocumented** alteration." And Provenance Information "documents the history of the Content Data Object ... **any changes that may have taken place since it was originated**, and who has had custody of it since it was originated, providing an audit trail for the Content Data Object. This gives future users some assurance as to the likely reliability of the Content Data Object as it contributes to evidence supporting **Authenticity**." So: repair-in-place is legitimate IFF the repair is itself recorded as Provenance Information. |

| 12 | https://www.sec.gov/investment/amendments-electronic-recordkeeping-requirements-broker-dealers + https://www.sec.gov/rules-regulations/staff-guidance/trading-markets-frequently-asked-questions/rule-amendments-broker | 2026-08-10 | official regulator guidance | WebFetch returned **HTTP 403**; re-fetched with `curl` + compliant User-Agent + tag-strip (11,539 + 17,868 chars) | The **adversarial** finding against naive "audit trails must be immutable". SEC Rule 17a-4 used to require records "exclusively in a non-rewriteable, non-erasable format (... WORM)". The 2023 amendments "**retained the WORM standard as an option**" and "**added an audit-trail alternative**" that "permits the **recreation of an original record if it is modified or deleted**", requiring "a complete time-stamped audit trail that includes: (1) all modifications to and deletions of a record or any part thereof; (2) the date and time of actions that create, modify, or delete the record; (3) if applicable, the identity of the individual creating, modifying, or deleting the record". Effective 2023-01-03, compliance 2023-05-03. **A financial regulator explicitly accepts mutation when recreation + a modification log exist.** |

| 13 | https://martinfowler.com/ieeeSoftware/failFast.pdf (Jim Shore, "Fail Fast", *IEEE Software* 21(5), 2004) | 2026-08-10 | peer-reviewed column | the jamesshore.com landing page carries only a summary, so: curl the linked PDF + `pdfplumber` (13,408 chars, 5pp) | The canonical statement, and it names our exact anti-pattern: "Some people recommend making your software robust by working around problems automatically. This results in the software '**failing slowly**.' The program continues working right after an error but fails in strange ways later on. A system that fails fast does exactly the opposite: when a problem occurs, it fails **immediately and visibly**." And the placement rule that matters here: "Assertions shine in their ability to flush out problems in the **seams of the system**. Use them to show mistakes in how the rest of the system interacts with your method." A producer/consumer naming convention IS a seam. |
| 14 | https://www.pantsbuild.org/stable/reference/global-options | 2026-08-10 | official doc | WebFetch | Shipped prior art for the exact knob: `unmatched_build_file_globs` -- values `ignore, warn, error`, **default `warn`** ("What to do when files and globs specified in BUILD files ... cannot be found") and `unmatched_cli_globs` -- same values, **default `error`**. The graduated policy is keyed on WHO named the target: a glob the *user* typed explicitly defaults to `error`; a glob in a config file defaults to `warn`. |
| 15 | https://github.com/backstage/backstage/issues/33326 | 2026-08-10 | community (GitHub issue), **2026-03-13** | WebFetch | **[COUNTER-POSITION]** A live 2026 instance of the class -- "Glob pattern in spec.targets with zero matches silently poisons entire Location entity, dropping all deferred entities" -- but the reporter's proposed fix argues the OPPOSITE of "always fail loud on zero match": "Zero-match glob patterns should be treated as **optional** (debug log only); only explicit file paths should trigger errors." The real harm there was the *second* bug: a target error set `ok:false` and 18 templates were dropped "with ... zero indication that other entities were dropped". Lesson: the loudness belongs on the **explicit-path** case and on the **dropped-work** case, not indiscriminately on every glob. |

