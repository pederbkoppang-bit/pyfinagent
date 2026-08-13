# Research Brief -- step 86.68

**Topic:** Version-number semantics: the changelog hook's bump trigger moved from
per-attempt (`phase-X.Y:` subject -> patch) to per-shipped-work (masterplan
status flip). Verify the SHIPPED implementation (commit `fbac40d7`) rather than
trust it.

**Tier:** simple (caller-stated). **Audit-class:** NO (`coverage.dry` not required).
**Started:** 2026-08-14. **Researcher:** Layer-3 researcher, Workflow rail.

---

## Envelope (born inert -- phase-86.37)

```json
{
  "brief_status": "COMPLETE",
  "tier": "simple",
  "external_sources_read_in_full": 7,
  "snippet_only_sources": 29,
  "urls_collected": 36,
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

**Sources read in full (the 7 that count toward the gate):**

1. https://semver.org/
2. https://www.conventionalcommits.org/en/v1.0.0/
3. https://ar5iv.labs.arxiv.org/html/2110.07889
4. https://semantic-release.gitbook.io/semantic-release/
5. https://keepachangelog.com/en/1.1.0/
6. https://github.com/changesets/changesets/blob/main/docs/detailed-explanation.md
7. https://workos.com/blog/software-versioning-guide

---

## Status log (write-first, appended as work lands)

- [t0] Brief created; envelope born INCOMPLETE. Read `.claude/agents/researcher.md`
  and `.claude/rules/research-gate.md` in full.
- [t1] Beginning internal exploration of `.claude/hooks/post-commit-changelog.sh`.
- [t2] Internal half COMPLETE (7 files/artifacts inspected). Replay measured. Moving to external.
- [t3] External: 3-variant searches run; 5 sources READ IN FULL via WebFetch
  (semver.org, conventionalcommits.org, ar5iv/2110.07889, semantic-release,
  keepachangelog). 1 fetch FAILED (kroah.com, ECONNREFUSED). Fetching 2 more.
- [t4] External half COMPLETE. 7 read in full, 29 snippet-only, 36 URLs. Envelope
  flipped to COMPLETE.

---

# PART 2 -- EXTERNAL

## 2.1 Search queries run (three-variant discipline, `.claude/rules/research-gate.md`)

| Variant | Query |
|---|---|
| Year-less canonical | `semantic versioning specification what does a version increment mean` |
| Current-year frontier (2026) | `semantic-release automated version bump from commit history 2026` |
| Last-2-year window (2025) | `version number inflation per-commit versioning release cadence 2025` |
| Peer-reviewed | `arxiv empirical study semantic versioning compliance breaking changes Maven Central` |

## 2.2 Read in full (7; >=5 required -- counts toward the gate)

| URL | Accessed | Kind | Fetched how | Key quote / finding |
|---|---|---|---|---|
| https://semver.org/ | 2026-08-14 | Official spec (canonical, year-less) | WebFetch, full | Rules 6/7/8 all condition the increment on what "**is introduced to the public API**", never on activity: *"Patch version Z ... MUST be incremented if only backward compatible bug fixes are introduced."* And the purpose clause: *"version numbers and the way they change convey meaning about the underlying code and what has been modified from one version to the next."* Rule 3: *"Once a versioned package has been released, the contents of that version MUST NOT be modified."* |
| https://www.conventionalcommits.org/en/v1.0.0/ | 2026-08-14 | Official spec | WebFetch, full | The spec **disclaims reliability of its own signal**: *"In a worst case scenario, it's not the end of the world if a commit lands that does not meet the spec. It simply means that commit will be missed by tools based on the spec."* It maps `fix`->PATCH, `feat`->MINOR, `BREAKING CHANGE`->MAJOR but makes no claim that a declared type matches the actual change; misused types "create tooling failures without preventing merge." |
| https://ar5iv.labs.arxiv.org/html/2110.07889 | 2026-08-14 | **Peer-reviewed** (Ochoa et al., EMSE 2022) | WebFetch of ar5iv HTML (per the arXiv chain; `/pdf/` never attempted) | n = 119,879 library upgrades, 293,817 clients. *"83.4% of these upgrades do comply with semantic versioning"*; *"20.1% of non-major releases are breaking"*; non-compliance fell *"from 67.7% in 2005 to 16.0% in 2018."* Mechanism section: declared increments diverge from real change content when the increment is decoupled from a measurement of the change. |
| https://semantic-release.gitbook.io/semantic-release/ | 2026-08-14 | Official docs (reference implementation) | WebFetch, full | Rationale verbatim: *"This removes the immediate connection between human emotions and version numbers, strictly following the Semantic Versioning specification."* Gating verbatim: a release happens *"if there are codebase changes since the last release that **affect the package functionalities**"* -- i.e. non-releasing types produce **no version at all**. |
| https://keepachangelog.com/en/1.1.0/ | 2026-08-14 | Official spec | WebFetch, full | *"Using commit log diffs as changelogs is a bad idea: they're full of noise. Things like merge commits, commits with obscure titles, documentation changes, etc."* Also: changelogs are *"for humans, not machines"*; *"A changelog which only mentions some of the changes can be as dangerous as not having a changelog."* |
| https://github.com/changesets/changesets/blob/main/docs/detailed-explanation.md | 2026-08-14 | Official docs (the intent-declared alternative) | WebFetch, full | A changeset is an *"intent to change"* rather than a record that a change occurred; *"Git is a bad place to store this information, as it discourages writing detailed change descriptions."* The design **separates two moments**: declaring impact, and the later batched release that *"combines all changesets, creates one version bump."* |
| https://workos.com/blog/software-versioning-guide | 2026-08-14 (pub. 2025-04-17) | Industry blog -- **recency-window source** | WebFetch, full | *"Version numbers aren't just cosmetic. They communicate **intent**--to your team, your users, and your tools."* Positions CalVer for *"tools ... with frequent releases"* where *"release timing matters more than change type"*, SemVer for public APIs. Cautions to *"avoid unnecessary major version bumps ... when the changes are minor."* |

**Attempted and FAILED (recorded so the absence is proven, not asserted):**
`http://www.kroah.com/log/blog/2025/12/09/linux-kernel-version-numbers/` --
WebFetch returned `connect ECONNREFUSED 172.104.246.16:443` (the URL is
plain-HTTP; the fetcher upgrades to HTTPS and the host refused). Would have been
the strongest *contrasting* source (Linux deliberately assigns its version
numbers no semantic meaning). Counted as snippet-only, not read-in-full.

## 2.3 Identified but snippet-only (29; does NOT count toward the gate)

`https://www.tiny.cloud/blog/improving-our-engineering-best-practices-with-semantic-versioning/` /
`https://semantic-versioning.org/` / `https://semver.org/spec/v2.0.0-rc.1.html` /
`https://arjancodes.com/blog/how-to-implement-semantic-versioning-in-software-projects/` /
`https://embeddedartistry.com/blog/2017/12/07/use-semantic-versioning-and-give-your-version-numbers-meaning/` /
`https://volkanpaksoy.com/archive/2025/10/04/Semantic-Versioning/` /
`https://oneuptime.com/blog/post/2026-01-25-semantic-versioning-automation/view` /
`https://oneuptime.com/blog/post/2026-01-27-version-bumping-github-actions/view` /
`https://www.muhammedfayazts.in/blogs/semantic-versioning-how-commits-become-version-bumps` /
`https://python-semantic-release.readthedocs.io/` /
`https://www.techplained.com/semantic-versioning-conventional-commits` /
`https://devopsil.com/articles/2026-03-21-semantic-versioning-automated-releases` /
`https://khimananda.com/blog/automate-releases-with-semantic-release` /
`https://codingprotocols.com/tutorials/semantic-versioning-conventional-commits` /
`https://betaflight.com/blog/2025/09/01/Calendar%20Versioning%20Change` /
`https://deepwiki.com/goauthentik/authentik/9.2-release-notes-and-versioning` /
`https://en.wikipedia.org/wiki/Perl_5_version_history` /
`https://docs.gitscrum.com/en/best-practices/release-management-strategies` /
`https://grokipedia.com/page/Software_versioning` /
`http://www.kroah.com/log/blog/2025/12/09/linux-kernel-version-numbers/` (fetch failed) /
`https://dl.acm.org/doi/10.1145/3551349.3556956` /
`https://arxiv.org/abs/2110.07889` / `https://export.arxiv.org/abs/2110.07889v1` /
`https://link.springer.com/article/10.1007/s10664-021-10052-y` /
`https://jstvssr.github.io/assets/pdf/semantic-versioning-maven.pdf` /
`https://www.researchgate.net/publication/276269673_Semantic_Versioning_versus_Breaking_Changes_A_Study_of_the_Maven_Repository` /
`https://www.researchgate.net/publication/301742814_Semantic_Versioning_and_Impact_of_Breaking_Changes_in_the_Maven_Repository` /
`https://www.researchgate.net/publication/355367557_Breaking_Bad_Semantic_Versioning_and_Impact_of_Breaking_Changes_in_Maven_Central` /
`https://www.goodreads.com/notes/24836465-building-microservices/145381-brian/c4a523a1-0abc-42f7-a876-61ec6400d5fe`

Reason not fetched in full: paywalled/ratelimited aggregators (ACM, ResearchGate,
Springer), duplicate renderings of a paper already read via ar5iv, or lower-tier
restatements of the canonical specs already read.

## 2.4 Recency scan (2024-2026) -- MANDATORY, and non-empty

Searched the 2025 and 2026 windows explicitly (queries in 2.1). **Result: 3 new
findings that COMPLEMENT rather than supersede the canonical sources.**

1. **Commit-derived automated versioning is the 2026 default for public-API
   projects** -- multiple 2026 sources (oneuptime 2026-01-25/27, devopsil
   2026-03-21, codingprotocols 2026) state the SemVer + Conventional Commits +
   automated-release stack "is the default for any library with external
   consumers." pyfinagent is **not** that: it has no external consumers and no
   release artifact, which is precisely why importing the library-ecosystem
   default produced a meaningless number.
2. **High-cadence projects are moving AWAY from per-change numeric bumps**, not
   toward them: Betaflight moved to CalVer `YYYY.M.PATCH` with two majors a year
   (2025-09-01); authentik moved to a three-month cycle `2026.2`/`2026.5`;
   WorkOS (2025-04-17, read in full) recommends CalVer where "release timing
   matters more than change type." The 2024-2026 trend line is toward *fewer,
   more meaningful* increments.
3. **No source in the window advocates bumping on every commit.** The nearest
   is continuous deployment, where the *deploy* is per-commit but the *version*
   is not the unit of meaning. I searched for a defence of per-commit version
   increments and found none -- stated as a result, not as an absence I assumed.

Nothing in the window supersedes semver.org's rules 6/7/8 or Keep a Changelog's
commit-log-dump warning; both remain the governing statements.

## 2.5 Key findings (cited per claim)

1. **A version increment is a claim about the CODE, not about activity.** All
   three normative rules are conditioned on what "is introduced to the public
   API"; the stated purpose is that increments "convey meaning about the
   underlying code and what has been modified from one version to the next"
   (semver.org, accessed 2026-08-14). A trigger keyed to *commit count* makes
   the increment a function of process volume. That is a category error under
   the spec, independent of whether 40 or 4 is the right number.
2. **Conventional Commits does not warrant its own signal.** *"It simply means
   that commit will be missed by tools based on the spec"* -- the spec's own
   posture toward a mis-typed commit is that the tool silently gets it wrong
   (conventionalcommits.org v1.0.0, accessed 2026-08-14). Any versioning rule
   reading a subject string inherits an **unvalidated author assertion**. This
   is the external form of the project's own standing lesson that a subject is
   a claim and a diff is what happened.
3. **The reference implementation gates on functional change, and defaults to
   NO release.** semantic-release releases only "if there are codebase changes
   ... that affect the package functionalities" (accessed 2026-08-14); `chore:`
   and `docs:` release nothing. The retired pyfinagent rule inverted this: its
   fall-through was `return "patch"` with the comment "default safety"
   (`post-commit-changelog.sh:92`). **That default was the inflation engine** --
   it is the opposite of the reference tool's default, and it is why even
   `phase-observe:` and bare subjects minted versions.
4. **Empirically, declared increments diverge from real change content even
   under discipline.** 20.1% of non-major Maven releases were breaking despite
   the declaration, across 119,879 upgrades (Ochoa et al., EMSE 2022,
   ar5iv/2110.07889, accessed 2026-08-14). Peer-reviewed support for measuring
   the transition rather than trusting the declaration -- and a caution that
   even the fixed rule's number is a summary, not a contract.
5. **Keep a Changelog independently endorses the two-track split pyfinagent
   already has.** *"Using commit log diffs as changelogs is a bad idea: they're
   full of noise"* (keepachangelog.com 1.1.0, accessed 2026-08-14). The retired
   rule collapsed the tracks -- every commit minted a version header *and* a
   What's-New bullet, i.e. a commit-log dump wearing version headers. The
   shipped rule restores the separation, which externally justifies the
   otherwise-undocumented bullet change at internal finding 1.8.
6. **Changesets is the closest industry analogue to the shipped design.** It
   separates "a change was made" from "a release happened", storing the release
   intent *outside* the commit subject because *"Git is a bad place to store
   this information"* (changesets detailed-explanation, accessed 2026-08-14).
   pyfinagent's masterplan flip is the same separation, with one strengthening:
   a changeset is **author-declared**, whereas a `done` flip is reachable only
   behind a Q/A PASS under the harness protocol. The shipped trigger is
   therefore *less* self-asserted than the industry analogue.

## 2.6 Consensus vs debate (external)

**Consensus (unanimous across all 7 sources read in full):** the increment must
be a function of the change's content or impact, never of activity volume. No
source read defends per-commit bumping.

**Genuine debate -- WHERE the signal comes from.** Three live schools:
commit-message-derived (semantic-release, release-please, python-semantic-
release), author-declared-at-change-time (changesets), and calendar-driven
(CalVer; Betaflight, authentik). pyfinagent's shipped rule is a **fourth**:
derived from an externally-evaluated state machine. I found **no literature
treating that variant**; changesets is the nearest neighbour. Stated as a gap in
the evidence, not as novelty for its own sake.

**Qualification worth carrying into the contract:** Ochoa et al. show a declared
version is an imperfect signal even when the practice is followed correctly.
The fix makes the version *honest*; it does not make it a contract, and the
contract should not claim more than that.

## 2.7 Pitfalls (from literature) mapped to this hook

| Pitfall (source) | pyfinagent exposure | Anchor |
|---|---|---|
| "Missed by tools based on the spec" -- a signal the tool cannot see (Conventional Commits FAQ) | **LOW, and the design is why.** A masterplan flip made via `python`/`sed` bypasses the Write/Edit-matcher auto-commit hook (project memory), but `_flip_magnitude` reads **committed git state**, not tool events -- so any flip that lands in a commit is still detected. Parsed-state detection is robust to the exact channel that defeats the sibling hook. | `:129-147` |
| "A changelog which only mentions some of the changes can be as dangerous as not having a changelog" (Keep a Changelog) | **REAL.** `MAX_ROWS=20` makes the Recent Activity table a 20-row rolling window that structurally cannot be a complete record. Completeness lives in git, not the table. This is the mechanism behind the caller's criterion-4 caution. | `:17`, `:280-282` |
| Default-to-release inflates (semantic-release gates on functional change) | **FIXED** -- but the dead `return "patch"` default still sits in `classify_commit` at `:92` and its docstring still advertises it. Harmless today only because of the `:176` gate. | `:92`, `:176` |
| Declared increments diverge from real content (Ochoa et al.) | **RESIDUAL** -- a `done` -> reopened -> `done` cycle bumps twice; the version counts *closures*, not distinct shipped units. | `:156-157` |

## 2.8 Application to pyfinagent

- **The fix is externally well-founded and the direction is right.** Keying the
  increment to an evaluated state transition (`:129-157`) rather than to a
  subject string (`:79-81`, now dead) is the SemVer-conformant choice
  (semver.org rules 6/7/8) and matches changesets' separation of change from
  release. 40 -> 1 over the measured window (internal 1.5) is the magnitude.
- **The strongest evidence available is already on disk and needs no new run.**
  The natural experiment is a *differential*: `phase-86.58:` bumped and 34
  same-shaped `phase-8X.Y:` subjects did not. The contract can rest criterion 3
  on that rather than on a synthetic replay.
- **Three things the contract should carry that are NOT yet documented:**
  (i) `classify_commit`'s docstring at `:60-70` describes branches that are now
  unreachable as outputs (internal 1.3);
  (ii) the What's-New bullet also became rare via `is_chore` at `:180`, which is
  correct per Keep a Changelog but undocumented (internal 1.8);
  (iii) the flip-detect failure marker is written to stderr (`:171-172`) but its
  operator-visibility is **unverified** and PostToolUse stderr is historically
  invisible in this project -- a `systemMessage` would be the project's own
  idiom for making it seen (internal 1.4).
- **Criterion 4 must pair the row count with the cap** (`:17`, measured binding
  at exactly 20 data rows). A flat count is a probe that cannot dirty.
- **Criterion 6's mutation matrix needs 3+ cells with controls observed green
  first** (internal 1.11), run in an isolated repo because the hook's tail
  (`:295-296`) creates a commit.
- **Do not claim SemVer conformance beyond honesty.** pyfinagent has no public
  API and no release artifact; `MAJOR`/`MINOR`/`PATCH` here mean
  phase/kickoff/step, not compatibility. That is a defensible local convention
  and the contract should say so explicitly rather than imply spec conformance.

## 2.9 Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch -- **7**
- [x] 10+ unique URLs total incl. snippet-only -- **36** (7 full + 29 snippet)
- [x] Recency scan (last 2 years) performed + reported -- 2.4, non-empty
- [x] Full pages read, not abstracts -- the arXiv paper was read via **ar5iv
      HTML**, never the `/pdf/` URL; all others are full page fetches
- [x] file:line anchors for every internal claim -- Part 1 throughout

Soft checks:
- [x] Internal exploration covered every module in the caller's scope (hook incl.
      `:59`/`:98`/`:176`/`:17` and the error path; CLAUDE.md bullet; CHANGELOG;
      masterplan entries for 86.9/86.44) -- **plus** `.claude/settings.json:74`,
      which the scope did not name but the never-raise claim depends on
- [x] Contradictions / consensus noted (2.6), incl. one unfilled literature gap
- [x] All claims cited per-claim with URL + access date
- [ ] **Tier note:** the caller set `simple` (<=300 w). This brief is far longer
      because the caller made the INTERNAL half primary and enumerated 6
      sub-questions. Depth followed the mandate, not the word cap. Disclosed
      rather than silently overrun.

## 2.10 Honest limitations

1. `fbac40d7`'s own "348 commits -> 136 bumps -> 7 replayed" figures are **NOT
   re-verified**; that is a different window. My independent window
   (`fbac40d7..HEAD`, 107 commits) gives **40 -> 1**. Two windows, two rules --
   deliberately not reconciled into one number.
2. The stderr marker's operator-visibility is **not tested**, only its presence
   in source. I did not run the error path.
3. My `^|`-anchored whole-file count is **128**, not the caller's 116. I state my
   predicate rather than reconcile to a number whose rule I do not have; the
   *conclusion* (the count is capped and uninformative) is unaffected and is
   independently established by the measured 20 == `MAX_ROWS`.
4. One intended source (kroah.com) failed to fetch; recorded in 2.2, not padded
   over.
5. No literature was found treating "version derived from an evaluated state
   machine". Absence after a genuine search, reported as a finding.

---

# PART 1 -- INTERNAL (PRIMARY)

## 1.1 Internal inventory

| File | Lines cited | Role | Status |
|---|---|---|---|
| `.claude/hooks/post-commit-changelog.sh` | 1-297 (read in full) | The hook. Bash wrapper + embedded python3 heredoc `:43-286` | LIVE |
| `.claude/hooks/post-commit-changelog.sh:59-92` | `classify_commit` | Conventional-Commits / `phase-X.Y` classifier | LIVE but **demoted** (see 1.3) |
| `.claude/hooks/post-commit-changelog.sh:98-173` | `_flip_magnitude` | masterplan-diff flip detector (the 86.68 fix) | LIVE |
| `.claude/hooks/post-commit-changelog.sh:176-180` | the gate + `is_chore` alias | Where the two classifiers compose | LIVE |
| `.claude/hooks/post-commit-changelog.sh:17` | `MAX_ROWS=20` | Recent-Activity rolling cap | LIVE, binding (measured) |
| `.claude/settings.json:74-75` | hook wiring | `bash ".../post-commit-changelog.sh"`, `statusMessage: "Syncing changelog..."` | LIVE |
| `CLAUDE.md` (changelog bullet) | prose | Documents the rule | Matches code on the load-bearing points; 2 gaps (1.6) |
| `CHANGELOG.md` | `:33` current version header | Artifact under test | v6.93.221 |
| `.claude/masterplan.json` | 1370 id->status entries | The flip detector's INPUT | LIVE |
| commit `fbac40d7` | 2 files | The shipped fix (hook + CLAUDE.md only) | 2026-08-13 20:27:51 +0200 |

## 1.2 The ACTUAL decision logic (what triggers what)

The bump type is decided by **two** functions composed by a one-line gate at
`post-commit-changelog.sh:176-177`:

```
bump_type = classify_commit(subject, body)          # :95
if bump_type != "major":                            # :176
    bump_type = _flip_magnitude()                   # :177
```

So the real decision table is:

| Input | Result | Anchor |
|---|---|---|
| Subject matches `^[a-z]+(\(...\))?!:` (e.g. `feat!:`) | **major**, flip irrelevant -- `_flip_magnitude` is never called | `:74`, gate `:176` |
| Body has a line `^BREAKING CHANGE:` | **major**, flip irrelevant | `:76` |
| This commit flipped >=1 masterplan step to `done` AND that flip left its top-level group with zero non-`done` entries | **major** | `:161-165` |
| ...AND a newly-done id matches `^\d+\.0$` | **minor** | `:166-168` |
| ...any other newly-done id | **patch** | `:169` |
| No newly-done id | **none** -- no version header AND no What's-New bullet | `:158-159`, `:180`, `:212`, `:228` |
| Subject is `^chore: (auto-changelog\|changelog drift)` | hook exits 0 before python runs | `:27-29` (bash, case-insensitive) |

**Flip detection is on PARSED STATE, not on the subject and not on a text diff.**
Verified at `:129-147`: it shells `git show <ref>:.claude/masterplan.json` for
`HEAD` and `HEAD~1`, `json.loads` each, walks the tree collecting every object
carrying both a string `id` and a string `status` into an `id -> status` map, and
diffs the maps at `:156-157`:

```python
newly_done = [sid for sid, st in after.items()
              if st == "done" and before.get(sid) not in (None, "done")]
```

The commit **subject is not read by `_flip_magnitude` at all** -- it takes no
arguments. A subject-based detector would have reintroduced the defect; it did
not. This is confirmed by differential below (1.5), not only by reading.

Two properties that fall out of the predicate and are worth recording:

- `before.get(sid) is None` is EXCLUDED, so a step **added already-`done`**
  (backfill) does not bump. Deliberate-looking and correct.
- Any non-`done` -> `done` transition counts, including `deferred`,
  `superseded`, `dropped`, `blocked` (all present in the live file; see 1.4).
  Corollary residual: a step reopened and re-closed (`done` -> `pending` ->
  `done`) bumps **twice**. Far smaller than per-commit, but not zero.

## 1.3 `classify_commit` is now a BINARY major/not-major detector

Because of the `:176` gate, `classify_commit`'s `minor` / `patch` / `none`
return values are **unreachable as outputs** -- every one of them is
overwritten by `_flip_magnitude()` on the next line. Only `"major"` survives.

Its docstring at `:60-70` still documents the full old five-way semantics
(`feat: -> minor`, `phase-X.Y: -> patch`, `chore: -> none`, "anything else ->
patch (default safety)"). **That docstring is now stale/misleading**: those
branches compute values that are discarded. Not a behaviour defect -- the
behaviour is correct -- but a doc-vs-code drift a future reader will trip on.
This is the single most likely source of a future misreading of the hook.

## 1.4 The never-raise error path -- verified on both halves of the claim

CLAUDE.md claims: on any internal error the detector bumps nothing and prints
`[changelog] flip-detect FAILED` to stderr.

**(a) A failure cannot silently bump -- PROVEN by construction, 2 lines.**
`:170-173` catches `except Exception` and `return "none"`. The only consumer is
`:177`, which is reached only when `bump_type != "major"`. `"none"` then fails
the `bump_type != "none"` test at `:212`, so no version header is written, and
sets `is_chore = True` at `:180` so no bullet is written either. There is no
path from an exception inside `_flip_magnitude` to a version increment.

**(b) A failure cannot break the commit.** Three layers:
1. `except Exception` at `:170` swallows everything realistic -- including
   `subprocess.TimeoutExpired` (the `timeout=20` at `:131`) and
   `json.JSONDecodeError` (a `ValueError`). It does not catch `BaseException`
   (`KeyboardInterrupt`/`SystemExit`), which is standard and acceptable.
2. `git show` returning non-zero is handled *without* an exception at `:132-133`
   (`returncode != 0 or not stdout.strip()` -> `None` -> `"none"` at `:152-155`).
   So "masterplan.json absent at `HEAD~1`" and "first commit in repo" are
   ordinary control flow, not the error path.
3. Even if the whole python block died, `set -euo pipefail` (`:7`) would abort
   the *hook*, not the commit: this is a **PostToolUse** hook
   (`.claude/settings.json:74`) that runs AFTER the `git commit` the operator
   already ran. The commit exists before the hook is invoked.

**RESIDUAL, and I am flagging it rather than asserting the CLAUDE.md wording is
wrong:** the marker's *observability* is weaker than "prints to stderr"
suggests. The wiring at `.claude/settings.json:74` adds no redirect, so stderr
is not suppressed there -- but PostToolUse hook stderr is not surfaced to the
operator the way a `systemMessage` is (project memory
`reference_claude_code_hooks_run_in_parallel.md`: "log-only warnings are
invisible -- use systemMessage"). The claim "it prints the marker" is true and
verified in source; the claim "an operator will SEE it" is **not established**
and I did not test it. A silent-stop is exactly what the docstring at `:121-124`
says it wants to avoid, so this is the most useful residual in the internal half.

## 1.5 The natural experiment, REPLAYED (this is the strongest evidence)

I re-derived both rules over the identical window `fbac40d7..HEAD` by
re-implementing `classify_commit` and `_flip_magnitude` and running them over
every commit (replay script run 2026-08-14; `/usr/bin/git` pinned, no shell grep
involved).

- Window: **107 commits**; **54** skipped by the bash skip-list at `:27`
  (`chore: auto-changelog`); **53 classified**.
- **RETIRED rule would have bumped 40 times** (39 patch + 1 minor).
- **SHIPPED rule bumps exactly 1 time**: `2b50904a`, `patch`.

Independently corroborated from the artifact itself: the first version header in
`CHANGELOG.md` was `v6.93.220` at `fbac40d7` and is `v6.93.221` at HEAD -- one
increment across 105 commits. Walking every commit's CHANGELOG shows exactly one
transition point.

**Criterion 3, demonstrated LIVE and stronger than a synthetic replay:** the
window contains **35 `phase-8X.Y:`-subject commits**, including the 86.62 cycle
that FAILed repeatedly (`c6519b43`, `15720934`, `c5ad55d8`, `892983e9`,
`d5736cce`) and the 86.9 cycle (`ef4df9bd`, `7f20b59a`, `1ea5dc2f`, `ce8ac085`)
and 86.44 (`323f8c78`, `5769c366`, `1422d0ec`). Under the retired rule each was
a patch bump. Under the shipped rule **all of them produced zero bumps**, while
one commit with the *same subject shape* (`phase-86.58:`) DID bump -- because it,
and only it, carried a masterplan status flip. Same subject grammar, opposite
outcomes: that is a differential that the subject cannot explain, and it refutes
subject-based detection as the operative mechanism.

`.claude/masterplan.json` at HEAD confirms the premise: `86.9` -> `pending`,
`86.44` -> `pending` (both `retry_count: 0, max_retries: 3` -- corroborating the
separate known finding that `retry_count` is not maintained), `86.62` ->
`pending`, `86.58` -> `done`, `86.68` -> `pending` (`retry_count: null`).

## 1.6 Criterion 4 caution -- CONFIRMED, and the cap is the explanation

The caller warned that a flat Recent-Activity row count LOOKS like commits
stopped appearing. Measured at HEAD with the predicate stated:

- `/usr/bin/grep -c '^|' CHANGELOG.md` (**every** pipe-anchored line in the whole
  file, all tables) = **128**.
- Rows in the Recent Activity table specifically (awk, from the
  `### Recent Activity` heading to the first following non-`|` line) = **22**,
  i.e. **20 data rows + header + separator**.

**20 == `MAX_ROWS` at `:17`.** The cap is not merely configured, it is *binding*
and actively trimming (`:280-282` deletes `table_rows[max_rows:]`). So an
unchanged count is the expected steady state and carries **zero information**
about whether commits are being recorded. The row-count metric must always be
paired with the cap; on its own it is a probe that cannot dirty. (My 128 is a
different denominator than the caller's 116 -- I state my predicate rather than
reconcile to a number whose rule I do not have.)

Positive control for the grep itself: the same pinned binary counts **1021**
`^### v` version headers in the same file, so the pattern engine is not silently
returning zero.

## 1.7 Rung reachability -- both new rungs are live, plus one sharp edge

Computed over the live masterplan (1370 id->status entries):

- **87** top-level groups are already fully `done` -> the `major` rung at
  `:161-165` is reachable, not decorative.
- **39** ids match `^\d+\.0$` -> the `minor` rung at `:166-168` is reachable.
- Phase 86 itself: 75 entries, **56 pending / 19 done** -- so no near-term
  accidental major from this phase.

**SHARP EDGE worth putting in the contract.** The grouping key is
`sid.split(".")[0]` (`:162-163`). Phase *objects* carry ids of the form
`phase-86` (82 such dot-less ids exist) while *steps* carry `86.58`. Those two
namespaces do **not** collide -- `"phase-86".split(".")[0] == "phase-86"` !=
`"86"` -- which is fortunate (a phase object's own `status` therefore cannot
block a step-driven major). But the flip side: a phase object is its own
singleton group, so flipping `phase-86` itself to `done` makes
`all(st == "done")` trivially true and yields **major unconditionally**,
regardless of whether its 56 pending steps are done. Semantically that is
probably the intended reading of "the phase shipped", but it is reached by
accident of the grouping rule rather than by the documented test ("no pending
steps left in phase X"). Flagging, not asserting a defect.

## 1.8 Undocumented behaviour change beyond the version number

`is_chore = (bump_type == "none")` at `:180` feeds the **What's-New bullet**
block at `:228`. Under the retired rule a `phase-86.x:` attempt commit was
`patch` -> `is_chore = False` -> it got a bullet. Under the shipped rule it is
`none` -> `is_chore = True` -> **no bullet**. So the fix also made What's-New
bullets rare. This is load-bearing rather than incidental: the bullet loop at
`:229-231` attaches to the first `### vX.Y.Z` header containing today's date, so
without the `is_chore` guard an attempt commit's bullet would be filed under a
version header describing *different, already-shipped* work. CLAUDE.md's bullet
documents only the version number and says "Every commit gets a Recent-Activity
row" -- true (`:270`, unconditional) -- but is silent on the bullet change.

## 1.9 CLAUDE.md vs code -- reconciliation

Claim-by-claim against the shipped hook:

| CLAUDE.md claim | Verdict | Anchor |
|---|---|---|
| Both named functions exist | TRUE | `:59`, `:98` |
| Bump requires a flip detected from the masterplan diff, not the subject | TRUE (parsed state, stronger than "diff") | `:129-157` |
| major if the flip emptied a whole top-level phase | TRUE | `:161-165` |
| minor if the flipped step is `X.0` | TRUE | `:166-168` |
| patch otherwise | TRUE | `:169` |
| `feat!:`/`BREAKING CHANGE:` bumps major on its own authority, flip or no flip | TRUE | `:74-77` + gate `:176` |
| Every commit gets a Recent-Activity row | TRUE | `:270` (unconditional), subject to the `:17` cap |
| Detector never raises; bumps nothing; prints `[changelog] flip-detect FAILED` | TRUE in source; **observability unverified** | `:170-173`; see 1.4 residual |
| "Replayed under the new rule the same 348 commits produce 7 bumps instead of 136" | NOT re-verified by me (different window). My independent window gives 40 -> 1. | -- |

**Gaps found (both documentation, neither behavioural):** (i) `classify_commit`'s
docstring `:60-70` still describes discarded branches (1.3); (ii) the What's-New
bullet change (1.8) is undocumented.

## 1.10 Criterion 2 -- alternative triggers, and what each would have produced

Enumerated over the SAME 107-commit window, so the comparison is decidable:

| Alternative trigger | Bumps in window | Assessment |
|---|---|---|
| **A. Conventional-Commits type** (the RETIRED rule; `feat:`->minor, `fix:`/`phase-X.Y:`->patch, `chore:`/`docs:`->none) | **40** (39 patch + 1 minor) | This IS the defect. It counts *intent declared in a subject*; a subject is a claim, and 35 of the 40 were attempt commits on steps that never closed. |
| **B. Git tag** (bump only when an annotated tag is pushed) | **0** in this window -- and the repo has no release-tag practice; the standing rule is push-direct-to-main with no release ceremony (CLAUDE.md "ALWAYS work on main branch") | Correct-by-construction but inert here: it moves the decision to a manual act nobody performs, so the version would simply freeze. Rejected for *this* repo, not in general. |
| **C. Phase completion only** (bump only when a whole top-level phase empties) | **0** in this window (phase 86 is 56/75 pending) | Too coarse: 908 `done` steps have shipped across the project's life vs 87 completed phases, so ~90% of shipped work would be invisible. It is exactly the shipped rule's `major` rung, which is why the shipped rule keeps it as a *rung* rather than as the whole trigger. |
| **D. SHIPPED: masterplan status flip, magnitude by phase/kickoff/other** | **1** | Ties the version to an *evaluated state transition* (a step only reaches `done` behind a Q/A PASS per the harness protocol) rather than to a self-declared subject. |

The justification that matters is not "1 < 40" -- it is *which* commit survived.
A trigger should fire on evidence the author cannot unilaterally assert.
Alternative A reads a string the committer types; D reads a status field whose
transition to `done` is gated by the harness's own evaluation loop. B and C are
both sound-but-inert here for measurable reasons (0 bumps each).

## 1.11 Criterion 6 -- what the mutation test must show

The commit message for `fbac40d7` already claims a mutation was run ("removing
the gate makes an attempt commit bump again (v6.0.0 -> v6.0.1) -- KILLED, control
green first"). To *verify* rather than trust it, the mutation must be re-run with
these properties:

1. **CONTROL FIRST, and it must be observed GREEN before the mutant is scored.**
   Replay a parked step's attempt commits (86.9 and 86.44 are the named ones;
   both are `pending`, so their commits are guaranteed flip-free) through the
   UNMUTATED hook and observe **version unchanged**. If the control does not
   produce "unchanged", the cell is UNSCORABLE and nothing may be concluded from
   the mutant (project memory: a mutation cell survives when the control answer
   and the mutant's fail-safe answer coincide).
2. **The mutant:** delete or invert the `:176` gate so `classify_commit`'s value
   governs again. **Expected: the same replayed attempt commits now bump** --
   ~12 of them across 86.9/86.44/86.62 in the live window. A mutant that changes
   nothing means the gate is not the operative line.
3. **A SECOND cell is required**, because cell 1 alone cannot distinguish "the
   gate works" from "`_flip_magnitude` always returns none". Mutate
   `_flip_magnitude` to return `"none"` unconditionally and replay the
   **`2b50904a`** (86.58 flip) commit: the version must **stop** moving. If it
   still moves, some other path is bumping.
4. **A third cell for the never-raise path:** inject an exception inside the
   `try` (e.g. point the `git show` at a bogus ref) and assert BOTH halves --
   version unchanged AND `[changelog] flip-detect FAILED` present on stderr.
   Asserting only "no bump" cannot distinguish the error path from a normal
   no-flip commit, since both return `"none"` (the fail-safe and the control
   answer coincide -- exactly the trap).
5. **Isolation:** run in a scratch repo or worktree, never against the live
   `CHANGELOG.md`/`main` -- the hook's tail (`:295-296`) *creates a commit*.
   Isolation must cover the git channel, not just file paths.
</new_string>

</content>
