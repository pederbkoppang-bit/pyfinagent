# Research Brief -- step 86.29 (RERUN)

**Topic:** Silent no-match globs and archive provenance -- (a) fail-fast vs fail-silent
for globs that match zero files; (b) remediating already-wrong historical archive
entries (repair-in-place vs annotate vs regenerate) under OAIS + SEC 17a-4; (c)
deriving destination filenames from an identifier instead of hand-agreed conventions,
and detecting producer/consumer drift.

**Tier:** moderate (caller-specified). **Audit-class:** NO (coverage reported for
information only; `coverage.dry` not required).
**Started:** 2026-08-10.

---

## ENVELOPE (born inert -- phase-86.37; updated in place as sources land)

```json
{
  "brief_status": "COMPLETE",
  "tier": "moderate",
  "external_sources_read_in_full": 7,
  "snippet_only_sources": 33,
  "urls_collected": 40,
  "recency_scan_performed": true,
  "internal_files_inspected": 7,
  "coverage": {
    "audit_class": false,
    "rounds": 3,
    "dry_rounds": 0,
    "K_required": 2,
    "new_findings_last_round": 4,
    "dry": false
  },
  "summary": "MEASURED: bash leaves an unmatched pattern unexpanded ('If the pattern is unsuccessful, the word is left unchanged'), nullglob/failglob both off, so archive-handoff.sh:160 matches 0 files for every sid tested (86.29/86.6/4.5.9/82.54) while 896 files use the suffix convention vs 169 hyphen-prefix, all historical. Only the rolling COPY at :146-152 fires, so archives inherit contract.md, which declares phase-82.54. Pants ships the graduated policy verbatim: unmatched_build_file_globs default warn, unmatched_cli_globs default error -- split by who named the pattern. OAIS 650.0-M-3 (Dec 2024) DEFINES Transformation as 'an alteration to the Content Information or PDI' and requires Provenance to record 'any changes ... and who has had custody'; 17a-4(f)'s audit-trail alternative permits modification given date/time/identity and 're-creation of the original'. Undocumented alteration is the prohibition, not alteration. Snakemake is the derive-once model. Census precision is UNMEASURED and its 2 known positives are one failure instance.",
  "brief_path": "handoff/current/research_brief_86.29_rerun.md",
  "gate_passed": true
}
```

*(Envelope flipped INCOMPLETE -> COMPLETE as the final act, per phase-86.37.)*

---

## Status log

- [t0] Brief created; envelope written born-inert.
- [t1] Internal inventory round 1 complete (archive-handoff.sh both branches,
  glob behaviour MEASURED under bash, census script method reviewed, hook wiring).
- [t2] External reads 1-2 landed (Cornell LII 17 CFR 240.17a-4(f); Pants global options).

---

## INTERNAL CODE INVENTORY (measured, not assumed)

| File | Anchor | Role | Status |
|---|---|---|---|
| `.claude/hooks/archive-handoff.sh` | `:146-152` | ROLLING branch: `cp` of the 5 UNSUFFIXED names | LIVE -- this is the only branch that ever fires |
| `.claude/hooks/archive-handoff.sh` | `:160-169` | STEP-SPECIFIC branch: `git mv` of `${sid}-*.md` / `phase-${sid}-*.md` | **DEAD** -- 0 matches for every sid tested |
| `.claude/hooks/archive-handoff.sh` | `:26-27` | `trap 'exit 0' EXIT` + `set -uo pipefail` (no `-e`) | fail-open by design; also swallows the dead branch |
| `.claude/hooks/archive-handoff.sh` | `:54-116` | `NEWLY_DONE` python heredoc + `.claude/.archive-baseline.json` state | LIVE; self-seeding |
| `.claude/settings.json` | PostToolUse `Write` + `Edit` | fires `archive-handoff.sh` AND `auto-commit-and-push.sh` under the SAME matcher | LIVE (parallel -- no ordering guarantee) |
| `scripts/qa/derive_archive_misattribution_86_29.py` | `:44,:54,:60-68,:81-97` | census + recall gate | committed; method reviewed below |
| `scripts/housekeeping/verify_handoff_layout.py` | `:25-49,:100-149` | layout invariants (`ROLLING_KEEP`, `ROLLING_KEEP_PREFIXES`) | LIVE; does NOT check archive *content* attribution |

### The measured facts

1. **The step-specific glob matches nothing, and bash does not say so.** Under
   `bash`, `shopt nullglob` = **off** and `failglob` = **off** (measured
   2026-08-10 in this repo). So `for f in "$CURRENT_DIR/${sid}-"*.md` with no
   match assigns `f` the **literal unexpanded pattern**, `[ -f "$f" ]` is false,
   the loop body is skipped, and `moved` stays 0. No message, no non-zero exit.
   Measured matches for the hyphen-prefix globs: `86.29`->0, `86.6`->0,
   `phase-86.6`->0, `4.5.9`->0, `82.54`->0. Suffix-form files for the same ids:
   1, 5, 5, 0, 4. Repo-wide: **896** files match the suffix convention
   (`contract_86.6.md`) vs **169** the hyphen-prefix convention -- and of those
   169, **44 are in `handoff/archive/_quarantine_2026-04-21`, 7 in
   `handoff/archive/misc`, 5 loose at `handoff/` root, and 4 in the
   `phase-phase-6.N` double-prefixed dirs**; **0 are in `handoff/current/`**.
   The hyphen convention is *historical*, so the glob is not merely mismatched
   today, it has had an empty domain for the whole period the current naming has
   been in use.

2. **The silent fall-through is to a DIFFERENT SOURCE, not to nothing.** Because
   the MOVE branch no-ops, only the COPY branch at `:146-152` runs, and it copies
   the **rolling unsuffixed** files. Measured right now: `contract.md` exists and
   its `# Contract -- phase-82.54` header (line 1) declares **82.54**;
   `experiment_results.md` and `evaluator_critique.md` last moved 2026-08-06;
   `research_brief.md` last moved 2026-07-25 and still holds **phase-80.2**
   content; `research.md` is ABSENT (dead name kept in the loop). So every step
   closed after 2026-08-06 archives 82.54's contract under its own id. This is
   the exact failure shape in the objective: *a glob matches zero files and the
   code silently falls through to a different source rather than erroring.*
   Note the two branches are also asymmetric in verb -- COPY vs MOVE -- so the
   rolling file is **left in place** and re-copied into the next N archives.

3. **`sid` vs `short_sid` is a second latent defect in the same loop.** `:128`
   computes `local short_sid="${sid#phase-}"` and uses it for the target dir, but
   `:160` interpolates the **raw `$sid`** into both globs. For an id already
   carrying the prefix (`phase-6.1`) the second pattern expands to
   `phase-phase-6.1-*.md`. The archive contains `phase-phase-6.1/` .. `-6.4/`
   dirs, which is the fossil of that same confusion on the directory side.

4. **`verify_handoff_layout.py` cannot catch this.** Its invariants are about
   *where files live* (`ROLLING_KEEP` at `:25`, `ROLLING_KEEP_PREFIXES` at `:42`,
   the `*_audit.json*` / `*.log` rules), not about *whether an archived
   contract.md declares the step whose directory it sits in*. A green layout
   check is fully compatible with 100% misattribution.

5. **Hook wiring: same matcher, no ordering.** `.claude/settings.json` registers
   `archive-handoff.sh` and `auto-commit-and-push.sh` under **both** the `Write`
   and the `Edit` PostToolUse matchers. Hooks under one matcher run in parallel,
   so the archive copy and the auto-commit are racing; any fix that makes the
   archive step *fail loudly* must not turn that race into a blocked commit
   (the hook's `trap 'exit 0' EXIT` at `:26` is the existing fail-open contract).

### Critical review of `derive_archive_misattribution_86_29.py`

The method is **recall-gated but not precision-gated**, and the script says so
itself at `:50-53`: an earlier `[0-9]+`-only sid pattern truncated `phase-25.A`
to `25`, so 46 correct dirs were reported as mismatches while recall still read
2/2. That history is the strongest available evidence for the review point:

- **Recall gate (`:110-129`)** -- refuses to print a census unless BOTH
  `phase-86.6` and `phase-86.26` classify as `mismatch`. Good: a method that
  misses a known member is rejected, not tuned. But **n=2, and both known
  positives are the SAME failure instance** (both contain 82.54's contract), so
  the gate tests one shape, not the class. Per the standing lesson
  ("a guard from the instance is not a guard against the class") this is the
  weakest link in the census.
- **Precision is unmeasured.** There is no negative control -- no dir known to
  be CORRECT that the classifier must NOT flag. `agree`=386 is asserted by the
  same regex set that produced the 46 false positives before. A precision check
  needs a sample of `mismatch` dirs read by hand, or a second independent
  signal (e.g. git history of when the dir was created vs when the rolling file
  last changed).
- **`unclassified`=255 is honestly bucketed** (`:147-169`) into "harness
  per-cycle contract (declares no step, by design)" vs "genuinely opaque", which
  is the right treatment -- folding them into `agree` would be the flattering
  error the docstring names at `:22-25`.
- **First-hit-wins ordering (`:60-68`, `:72-78`)** over 7 patterns on the first
  4000 chars: pattern `[6]` (`^#.*?\bphase-(SID)\b`) is a broad catch-all and is
  last, which is correct ordering, but it can still bind to a *cross-reference*
  in a title rather than a declaration (e.g. a contract titled "... supersedes
  phase-82.12"). That is the residual precision risk and it is not currently
  measured.
- **`classify()` is called 2-3x per dir** (`:134`, `:152`, `:156`) -- pure
  inefficiency, not a correctness bug, but it means the "unclassified breakdown"
  re-reads every contract.

---

## EXTERNAL: READ IN FULL (counts toward the gate)

| # | URL | Accessed | Kind | Fetched how | Key finding |
|---|---|---|---|---|---|
| 1 | https://www.law.cornell.edu/cfr/text/17/240.17a-4 | 2026-08-10 | official reg text (CFR) | WebFetch, full | 17a-4(f) offers TWO alternatives; the audit-trail one **permits** modification/deletion provided the trail captures "**All modifications to and deletions of the record or any part thereof**", "**the date and time**", "**the identity of the individual**", and enough "**to permit re-creation of the original record if it is modified or deleted**". WORM is the *other* option, not the only one. |
| 2 | https://www.pantsbuild.org/stable/reference/global-options | 2026-08-10 | official docs | WebFetch, full | `unmatched_build_file_globs` = "What to do when files and globs specified in BUILD files, such as in the `sources` field, cannot be found", values `ignore, warn, error`, **default `warn`**. `unmatched_cli_globs` = "What to do when command line arguments, e.g. files and globs like `dir::`, cannot be found", values `ignore, warn, error`, **default `error`**. The graduated policy is real and the split is exactly by **who named the target**. |
| 3 | https://tiswww.case.edu/php/chet/bash/bashref.html | 2026-08-10 | official docs (bash maintainer's manual) | WebFetch, full page (tool truncated the tail; the load-bearing sentence is in the Filename Expansion section) | "**If the pattern is unsuccessful, the word is left unchanged.**" That single sentence is the entire defect: the unexpanded pattern becomes the loop variable, `[ -f ]` rejects it, and the loop body never runs. Corroborated in-repo: `shopt nullglob`=off, `failglob`=off. |
| 4 | https://snakemake.readthedocs.io/en/stable/snakefiles/rules.html | 2026-08-10 | official docs | WebFetch, full | Destination paths are **derived from one declared pattern + wildcards**, never typed twice: a request for `101/file.A.txt` binds `dataset=101, group=A` and the same binding drives input, output and the command. Ambiguity is an **error** (`AmbiguousRuleException`), and `wildcard_constraints` (e.g. `{dataset,\d+}`) constrain what an identifier may expand to. Outputs are **deleted before the job runs**, so a rule that fails to produce its declared output cannot silently pass off a stale file. |
| 5 | https://ccsds.org/Pubs/650x0m3.pdf | 2026-08-10 | standard (CCSDS 650.0-M-3 = ISO 14721:2025, Dec 2024) | WebFetch + **independently re-extracted with `pypdf` (150 pp, 374,127 chars) and regex-verified** | **Provenance Information**: "*The information that documents the history of the Content Data Object. This information tells the origin or source of the Content Data Object, any changes that may have taken place since it was originated, and who has had custody of it since it was originated.*" **Transformation**: "*A Digital Migration in which there is an alteration to the Content Information or PDI of an Archival Information Package.*" **Transformational Information Property**: "*An Information Property the preservation of the value of which is regarded as being necessary but not sufficient to verify that any Non-Reversible Transformation has adequately preserved information content. This could be important as contributing to evidence about Authenticity.*" Migration taxonomy: Refreshment / Replication (bit-preserving) vs **Repackaging / Transformation** ("Operations which change the bit sequences"). |
| 6 | https://www.dpconline.org/handbook/technical-solutions-and-tools/fixity-and-checksums | 2026-08-10 | practitioner handbook (DPC) | WebFetch, full | Preferred remedy for content discovered wrong is **replace from a known-good copy**, not in-place patching: "*if one of the copies has changed then one of the other copies can be used to create a known good replacement*". Requires "*Maintain logs of fixity info and supply audit on demand*". Fixity detects *that* something changed, never *where*. |
| 7 | https://docs.astro.build/en/reference/errors/astro-glob-no-match/ | 2026-08-10 | official docs | WebFetch, full | A mainstream framework chose hard-fail: "`Astro.glob(GLOB_STR)` did not return any matching files." Cause attributed to "*a typo in the glob pattern*" -- our exact case. **Caveat found:** the error was *deprecated and removed in Astro v6.0.0* along with the API, so it is prior art for the choice, not an endorsement that survived. |

### Identified but snippet-only / attempted (does NOT count toward the gate)

| URL | Kind | Why not read in full |
|---|---|---|
| https://github.com/pantsbuild/pants/issues/5430 | design debate | **Attempted WebFetch**; GitHub returned "there was an error while loading" -- only the opening comment (@stuhood, 2018-02-03, "long-standing TODO to warn or error (depending on an option) for unmatched globs") came through. Not counted. |
| https://www.loc.gov/standards/premis/understanding-premis.pdf | standard (PREMIS) | Attempted WebFetch -> HTTP 403 |
| https://www.ecfr.gov/current/title-17/section-240.17a-4 | official reg text | Attempted -> 302 to `unblock.federalregister.gov` (bot wall); substituted Cornell LII |
| https://www.gnu.org/software/bash/manual/html_node/The-Shopt-Builtin.html | official docs | Attempted 2x -> HTTP 429; substituted the maintainer's mirror |
| https://link.springer.com/article/10.1007/s10502-024-09462-w | peer-reviewed (Archival Science 2024) | Attempted -> 303 to Springer IdP auth wall |
| https://github.com/prettier/prettier/issues/7861 | ecosystem debate | snippet: Prettier 2.0.2 began throwing on non-matching globs where 1.x did not -- user pushback |
| https://github.com/yarnpkg/berry/issues/1813 | ecosystem debate | snippet: "[Feature] Glob support should **not** error on no matched files" -- the counter-position |
| https://github.com/pantsbuild/pants/issues/10574 | regression report | snippet: "Regression in 'unmatched glob' error" |
| https://github.com/pantsbuild/pants/issues/15655 | bug report | snippet: unmatched CLI glob error hitting a pre-commit hook |
| https://github.com/pantsbuild/pants/issues/11629 | usability | snippet: "Improve error message for unmatched globs" |
| https://github.com/pantsbuild/pants/issues/5863 | follow-up | snippet: "target sources glob expansion failure warning needs followup work" |
| https://github.com/pantsbuild/pants/pull/9010 | PR | snippet: "Improve unmatched globs error message" |
| https://github.com/pantsbuild/pants/pull/9013 | PR | snippet: "Describe the origin of failure when globs do not match" (`description_of_origin`) |
| https://www.pantsbuild.org/dev/docs/writing-plugins/the-rules-api/file-system | official docs | snippet: "By default, the engine will no-op for any globs that are unmatched"; `glob_match_error_behavior` requires `description_of_origin` |
| https://www.pantsbuild.org/v2.8/docs/rules-api-file-system | official docs (older) | snippet, superseded by the above |
| https://www.pantsbuild.org/dev/docs/tutorials/advanced-plugin-concepts | official docs | snippet |
| https://github.com/nushell/nushell/issues/10673 | ecosystem | snippet: no-match-found error handling debate |
| https://github.com/irongut/CodeCoverageSummary/issues/268 | ecosystem | snippet: "No files found matching glob pattern" |
| https://www.npmjs.com/package/fast-glob | library docs | snippet |
| https://discourse.roots.io/t/no-files-matching-the-pattern-then-module-build-failed-errors-for-new-user/20418 | forum (lowest tier) | snippet |
| https://en.wikipedia.org/wiki/Fail-fast_system | encyclopedia | snippet: fail-fast = "fail as soon as possible" |
| https://en.wikipedia.org/wiki/Open_Archival_Information_System | encyclopedia | snippet; primary standard read instead |
| https://msi.dublincore.org/standards/oais/ | standards index | snippet |
| https://siarchives.si.edu/sites/default/files/pdfs/650x0b1.PDF | superseded OAIS edition | snippet; read 650.0-M-3 instead |
| https://standards.iteh.ai/catalog/standards/sist/e69f8023-f60c-4b06-aa7a-1af3314b004e/sist-iso-14721-2025 | standards catalogue | snippet (paywalled) |
| https://cdn.standards.iteh.ai/samples/87471/14a5ffd9492e443f86440952aabecf69/ISO-14721-2025.pdf | ISO sample | snippet; identical text to CCSDS read in full |
| https://ufs.libguides.com/c.php?g=1113411&p=8118662 | libguide | snippet |
| https://arxiv.org/pdf/cs/0509084 | preprint | snippet; off-topic (MPEG-21 DID) |
| https://arxiv.org/pdf/2601.14823 | preprint 2026 | snippet: archival bond + IIIF; adjacent, not load-bearing |
| https://digital-preservation-a-critical-vocabulary.pubpub.org/pub/mrih3jw4 | scholarly vocabulary | snippet: provenance = origin, creator, chain of custody |
| https://www.archives.gov/preservation/digital-preservation/strategy | national archive policy 2022-2026 | snippet |
| https://www.archives.gov/preservation/digital-preservation | national archive policy | snippet |
| https://www.scoredetect.com/blog/posts/preserving-the-integrity-of-digital-archives-a-primer | vendor blog (lowest tier) | snippet |

**URLs collected: 40 unique (7 read in full, 33 snippet-only or attempted-and-blocked).**

### Search-query composition (three-variant discipline, all visible)

- **Year-less canonical:** `Pants build unmatched_build_file_globs unmatched_cli_globs warn error glob match zero files`; `OAIS CCSDS 650.0-M-3 reference model archival information package transformation provenance preservation description`
- **Current-year frontier (2026):** `unmatched glob matched no files silent no-op build system fail fast error 2026`
- **Last-2-year window (2025):** `digital preservation correcting erroneous archived records annotate versus repair in place provenance 2025`

### Recency scan (2024-2026) -- PERFORMED

**Result: 4 new findings that COMPLEMENT rather than supersede the canonical
sources; nothing overturns them.**

1. **The primary standard is itself brand new.** CCSDS 650.0-M-3 is dated
   **December 2024** and is identical to **ISO 14721:2025**. The canonical OAIS
   citation is therefore *inside* the recency window, not older prior art.
2. **The ecosystem is actively contested in both directions (2024-2026).**
   Prettier moved 1.x -> 2.0.2 from silent-pass to throwing on non-matching
   globs and drew user pushback (#7861); Yarn Berry carries an open feature
   request that glob support should **not** error on no matches (#1813); Pants
   has a logged **regression** in its unmatched-glob error (#10574) and a CLI-glob
   error breaking a pre-commit hook (#15655). Fail-fast is not a free win.
3. **Astro removed the hard error entirely in v6.0.0** (with the `Astro.glob()`
   API). The most recent datapoint is a *retreat* from hard-fail-on-no-match --
   the strongest qualifier against a naive "just make it error" fix.
4. **17a-4's audit-trail alternative is the 2022/2023 amendment**, i.e. the
   modern regulatory position explicitly *added* a lawful path for modifying
   records. No 2024-2026 source found reversing it.

### Consensus vs debate

**Consensus.** (i) A zero-match glob that silently no-ops is a recognised defect
class with a name and options in mature build systems, not a quirk. (ii) The
right unit of policy is *provenance*, not immutability: OAIS defines
Transformation as a legitimate migration and 17a-4(f) permits modification --
both conditional on a record of what changed, when, and by whom.

**Debate.** Whether the default should be `error` or `warn`. Pants ships **both
defaults in one product**, split by provenance of the pattern: author-declared
globs in BUILD files default to `warn`; user-typed CLI globs default to `error`.
The 2024-2026 ecosystem evidence (Prettier pushback, Yarn's counter-request,
Astro's removal) shows hard-fail imposed uniformly generates real friction --
so the graduated policy is the better-supported design, and the objective's
hypothesis about "who named the target" is **confirmed verbatim** by the Pants
option text.

### Pitfalls (from the literature and from this session)

1. **`ignore` is a real, shipped option.** Pants offers `ignore, warn, error`.
   Silence is a *choice* someone can select -- it should never be the accidental
   default arising from shell semantics.
2. **An error message without an origin is nearly useless.** Pants requires
   `description_of_origin` whenever behaviour is `warn`/`error` (PR #9013,
   "Describe the origin of failure when globs do not match"). Any pyfinagent
   warning must name the step id AND the pattern, or the operator cannot act.
3. **Fixity detects that something changed, never what** (DPC). A checksum over
   an archived `contract.md` would not have caught this: the file is intact, it
   is simply *the wrong file*. Attribution defects are invisible to integrity
   checks -- this is why `verify_handoff_layout.py` is green.
4. **Replace-from-known-good beats patch-in-place** (DPC) -- and here a
   known-good copy genuinely exists: git history holds the correct per-step
   contract at the commit that closed each step.
5. **A hard error inside a fail-open hook is a footgun.** The hook's
   `trap 'exit 0' EXIT` (`:26`) and its parallel scheduling with
   `auto-commit-and-push.sh` mean "error" must mean *loud + recorded*, not
   *non-zero exit*.
6. **[METHOD PITFALL, measured this session] A WebFetch summary of a large PDF
   can fabricate quotations.** The first CCSDS fetch returned, in quotation
   marks, "*All changes to the AIP must be documented to maintain the ability to
   establish authenticity.*" Re-extracting the actual 150-page PDF with `pypdf`
   (374,127 chars) gives **0 hits** for that string. The real definitions are
   different -- and stronger. This is the **second measured instance** of this
   failure in this project (cf. step 83.1.1). Any PDF-derived quote must be
   regex-verified against extracted text before it enters a contract.

### Application to pyfinagent

**(a) The glob.** `archive-handoff.sh:160` is the exact failure class. The fix
is not merely "correct the pattern" -- the pattern was corrected once before
(the `phase-4.16.2 fix` comment at `:156-158` added the second glob precisely
because "the old single-glob left 150 files stranded") and it silently went
stale again. The durable fix is to make **zero matches observable**. The Pants
model maps cleanly onto the two branches: the step-specific MOVE glob is
*author-declared* (the hook wrote the pattern) -> the analogue of
`unmatched_build_file_globs`, default **warn**; the rolling COPY of an
unsuffixed file into a *specific* step's directory is the higher-risk act and
deserves **error**-grade treatment, because that is the branch that fabricates
provenance. Mechanically, `shopt -s nullglob` + an explicit
`if (( moved == 0 )); then echo "[archive-handoff] WARN sid=$sid pattern=... matched 0 files" >&2; fi`
preserves the fail-open contract at `:26` while ending the silence. `set -e` /
`failglob` are the wrong tools here: both would fight the deliberate trap.

**(b) Remediating 818 archive dirs.** OAIS and 17a-4 agree and both permit the
repair: OAIS *defines* Transformation as "an alteration to the Content
Information or PDI", and 17a-4(f)'s audit-trail alternative allows modification
where the trail records "all modifications ... the date and time ... the identity
of the individual" and permits "re-creation of the original record if it is
modified or deleted". The prohibition is on **undocumented** alteration, exactly
as the objective hypothesised. Concretely: (1) do **not** silently overwrite;
(2) regenerate from the known-good source -- git history at each step's closing
commit -- which is DPC's replace-from-a-good-copy rather than hand-patching;
(3) write a machine-readable remediation record per touched dir (step id, what
was there, what it was replaced with, the commit it came from, timestamp, agent)
-- this is OAIS Provenance Information and simultaneously the 17a-4 audit trail;
(4) keep the displaced file rather than deleting it, so the original remains
re-creatable. Where regeneration is impossible (a step whose contract never
existed under a suffixed name), **annotate rather than fabricate**: a stub
recording "this directory's contract.md is the rolling file for phase-82.54, not
this step" is honest provenance; inventing a plausible contract is not.
Note the 255 `unclassified` and 24 `no_contract` dirs must stay in their own
buckets through remediation -- folding them into "repaired" would be the same
flattering error the census script warns about at `:22-25`.

**(c) Derive, don't hand-agree.** Snakemake is the cleanest prior art: one
declared pattern binds wildcards, and the *same* binding drives both ends, so
producer and consumer cannot drift because there is only one statement of the
convention. pyfinagent currently has the convention stated **twice and
differently** -- Main writes `contract_<sid>.md` (suffix, 896 files) while the
hook reads `<sid>-*.md` (prefix, 169 files, all historical). The fix shape is a
single derivation, e.g. one shell function `handoff_name(kind, sid)` used by
both the archive hook and any writer, or at minimum a checked-in
`ARTIFACT_PATTERNS` list that `verify_handoff_layout.py` asserts against
`handoff/current/` contents. Drift detection is the cheap, high-value half:
Snakemake deletes declared outputs before a job so a stale file cannot be passed
off as fresh -- the pyfinagent analogue is asserting **after** archiving that
`handoff/archive/phase-<sid>/contract.md` declares `<sid>`, i.e. running the
census script's `classify()` as a *post-condition*, not only as a one-off audit.
That single assertion would have caught this on the first occurrence.

**Scope caution for PLAN (not a recommendation to act):** the census's
`agree`=386 rests on an unmeasured-precision classifier (see review above); a
remediation that rewrites 153 dirs on that basis inherits its false-positive
rate. A precision control -- a hand-read sample of flagged dirs, plus a negative
control of dirs that must NOT be flagged -- belongs in the contract before any
write. Also note the two known positives are one failure instance, not two.

### Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (**7**)
- [x] 10+ unique URLs total incl. snippet-only (**40**)
- [x] Recency scan (last 2 years) performed + reported (4 findings)
- [x] Full pages/standards read, not abstracts (CCSDS additionally re-extracted
      and regex-verified with `pypdf`)
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every module in the caller's scope
      (`archive-handoff.sh` both branches, `contract.md`, the census script,
      `verify_handoff_layout.py`, `.claude/settings.json` wiring)
- [~] `backfill_handoff_archive.py` inspected only via the layout verifier's
      contract, not read line-by-line -- disclosed gap
- [x] Contradictions / consensus noted (warn-vs-error debate; Astro retreat)
- [x] All claims cited per-claim with URL + access date or file:line

---

## Status log (cont.)

- [t3] External reads 3-7 landed. CCSDS quote fabrication caught and corrected
  by independent `pypdf` extraction. Recency scan complete. Brief closed.

