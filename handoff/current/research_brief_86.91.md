# Research Brief -- phase-86.91

**Topic:** Automated version-bump detection from state transitions, and the
ABSENT-vs-UNCHANGED conflation that makes such a detector silently under-count.
**Tier:** moderate (caller-specified). **Audit-class:** NO.
**Accessed:** 2026-08-16. All WebFetch reads dated 2026-08-16.

---

## STATUS ENVELOPE (born inert -- phase-86.37)

```json
{
  "brief_status": "COMPLETE",
  "tier": "moderate",
  "external_sources_read_in_full": 8,
  "snippet_only_sources": 20,
  "urls_collected": 28,
  "recency_scan_performed": true,
  "internal_files_inspected": 12,
  "coverage": {
    "audit_class": false,
    "rounds": 3,
    "dry_rounds": 0,
    "K_required": 2,
    "new_findings_last_round": 2,
    "dry": false
  },
  "gate_passed": true
}
```

---

## Search queries run (three-variant discipline, per research-gate.md)

| Variant | Query |
|---|---|
| year-less canonical | `semantic-release "no release" versus "failed to determine" release type decision` |
| year-less canonical | `null versus absent key conflation bug "missing" treated as "unchanged" sentinel object remedy` |
| year-less canonical | `changesets vs conventional commits critique "commit messages" unreliable release intent derived from changed files` |
| current-year frontier | `mutation testing 2026 regression test detects removed guard clause absent behaviour equivalent mutant` |
| last-2-year window | `fail-safe silent no-op observability 2025 structured stderr marker CI step must not fail build audit decision log` |

---

## Read in full (>=5 required; counts toward the gate)

| # | URL | Kind / tier | Key finding |
|---|-----|------|-------------|
| 1 | https://peps.python.org/pep-0661/ | Official language spec. **Final**, resolved 23-Apr-2026, target 3.15 | Sentinels are needed "usually when it needs to be distinct from `None` since `None` is a valid value in that context." Mandates identity: "Checking if a value is such a sentinel *should* be done using the `is` operator." Bare `object()` is rejected for its "uninformative and overly verbose repr" -- debuggability is part of the remedy, not a nicety. |
| 2 | https://python-patterns.guide/python/sentinel-object/ | Authoritative blog (Brandon Rhodes) | The pattern exists **exactly** to separate "value is None" from "key not found": `result = cache_get(key, sentinel); if result is not sentinel:`. A magic value fails because it "could legitimately appear in user data"; only a unique object's *identity* carries the meaning. Stdlib instances: `functools.lru_cache`, `bz2._sentinel`, `configparser._UNSET`. |
| 3 | https://semantic-release.org/support/faq/ | Official docs | "Only the codebase changes altering the published package will trigger a release... refactoring or changing code style would not." Skip is an **explicit marker**: `[skip release]`/`[release skip]` commits are "excluded from the commit analysis and won't participate in the release type determination." Verification path: "the dry-run options which prints to the console the next version to be published." **NOTABLE ABSENCE:** the FAQ never distinguishes "no release needed" from "failed to determine" -- see Finding 5. |
| 4 | https://git-scm.com/docs/githooks | Official docs (Git) | `post-commit` "is meant primarily for notification, and **cannot affect the outcome of `git commit`**." Two classes enumerated: abort-capable (`pre-commit`, `commit-msg`, `pre-push`...) vs exit-status-**ignored** (`post-commit`, `post-merge`, `post-rewrite`...). Both stdout and stderr are forwarded to the user. `--no-verify` covers only the pre-/msg- class. |
| 5 | https://brianschiller.com/blog/2023/09/18/changesets-vs-semantic-release/ | Practitioner blog, 2023-09-18 | The prefix critique in the author's words: "I shouldn't need to designate it as a 'fix' or a 'refactor' when the code needing a fix wasn't ever released." The subject prefix describes the **development act**, not the shipped user-visible change. |
| 6 | https://xnok.github.io/infra-bootstrap-tools/blog/intentional-releases-changesets/ | Practitioner blog, updated **2026-06-04** | Two named failure modes of commit-derived bumps. **Noise:** "A single feature might involve five feat commits and dozens of fix commits. Your changelog becomes a cluttered list of technical steps rather than a summary of value." **Intent gap:** "Commit parsers can't capture that nuance." |
| 7 | https://arxiv.org/html/2605.02033 | Preprint, submitted **2026-05-03** ("Conventional Commit Classification using LLMs and Prompt Engineering") | Problem framing: "software developers often ignore the conventional commits specification and write unstructured commit messages." **[PARTIALLY ADVERSARIAL]** -- it does NOT measure prefix-vs-diff disagreement and *assumes* the declared label is ground truth. Methodology is 3,200 commits from ONE repo (InfluxDB), no train/test split, no cross-validation, no overfitting analysis; authors concede "more datasets... can help us understand the generalization." |
| 8 | https://arxiv.org/html/2408.01760 | Peer-reviewed, **ISSTA '24** (Sept 16-20 2024) | Equivalent mutant = one that "exhibits the same behavior as the original program for all possible test cases"; detection "is undecidable". **Measured prevalence: "The rate of equivalent mutants in real-world development scenarios ranges from 4% to 39%."** Survived-but-equivalent mutants bias the score and burn reviewer time. |

---

## Identified but snippet-only (context; does NOT count toward gate)

| URL | Why not fetched in full |
|-----|-------------------------|
| https://committing-crimes.com/articles/2025-09-16-null-and-absence/ | Corroborates Finding 1; superseded by PEP 661 as citable source |
| https://blog.peterlamut.com/2020/04/01/distinguishing-between-none-and-missing-values-in-dictionaries-with-a-sentinel/ | Same claim as source 2, lower tier |
| https://www.pythonpool.com/sentinel-value-python/ | Community tier |
| https://medium.com/the-pythonworld/never-use-none-for-missing-values-again-do-this-instead-8affb146e42a | Aggregator, community tier |
| https://dev.to/kalio/cache-aside-and-the-null-sentinel-pattern-5gjc | Cache-specific framing of same pattern |
| https://github.com/semantic-release/semantic-release/blob/master/docs/support/troubleshooting.md | Overlaps source 3 |
| https://semantic-release.gitbook.io/semantic-release/support/faq | Mirror of source 3 |
| https://github.com/semantic-release/semantic-release/issues/1121 | Issue tracker, community tier |
| https://buglyst.com/learn/guides/semantic-release-debug | Community tier |
| https://shinesolutions.com/2021/07/21/learning-to-use-semantic-release-the-hard-way/ | Community tier |
| https://github.com/changesets/changesets/issues/577 | Issue tracker |
| https://github.com/changesets/changesets/issues/862 | Issue tracker |
| https://github.com/graphql/graphiql/issues/1531 | Issue tracker |
| https://lirantal.com/blog/introducing-changesets-simplify-project-versioning-with-semantic-releases | Overlaps sources 5/6 |
| https://arxiv.org/pdf/1803.07901 | "Selecting Fault Revealing Mutants" -- PDF-only, older than window |
| https://arxiv.org/pdf/2501.12862 | "Mutation-Guided LLM-based Test Generation at Meta" -- PDF-only |
| https://www.devopsroles.com/devops-production-grade-bash-strict-mode | Community tier; bash strict-mode + `trap ERR` structured logging |
| https://hackernoon.com/your-ci-pipeline-should-reject-any-service-without-an-observability-contract | Community tier |
| https://www.frugaltesting.com/blog/how-to-detect-silent-failures-in-microservices-using-advanced-observability-techniques | Community tier |
| https://dev.to/zvone187/5-silent-failure-modes-in-production-ai-agents-and-how-we-instrument-for-them-oca | Community tier; the "generic empty return reads as a valid no-op" framing |

---

## Recency scan (2024-2026)

**Performed; NON-EMPTY.** Four of the eight read-in-full sources fall inside the
window and two materially change the plan:

1. **PEP 661 reached `Final` on 2026-04-23** (target 3.15) -- the sentinel remedy
   is no longer a folk pattern but a language-level construct with a mandated
   `is`-identity check and a named `repr`. Supersedes the older `_UNSET = object()`
   blog advice as the citable authority.
2. **ISSTA '24 (arXiv:2408.01760) gives the first hard prevalence number** for
   equivalent mutants: **4%-39%**. This directly constrains criterion 6 -- a
   mutant that merely "survives" proves nothing without a control, and 86.68's
   Q/A already hit this (its cycle-1 residual was that the replay exited 0 while
   both cells SURVIVED).
3. **arXiv:2605.02033 (2026-05-03)** is the most recent work on commit
   classification and is *weaker* than the project's own evidence -- see
   Consensus vs debate.
4. **xnok (updated 2026-06-04)** restates the anti-inference case; no newer
   argument supersedes the 2023 Schiller framing.

No 2024-2026 source was found that defends deriving a release bump from commit
subject prefixes. The frontier has moved **away** from inference and toward
explicit intent.

---

## Key findings

1. **The defect class is named, and its canonical remedy is a sentinel.** "Whenever
   `None` means two different things in the same flow, one of them needs a name."
   PEP 661: sentinels are for when the value "needs to be distinct from `None`
   since `None` is a valid value in that context" (https://peps.python.org/pep-0661/).
   `before.get(sid)` returning `None` is exactly this: it means *both* "step absent
   at HEAD~1" and, to the reader, "nothing to compare".
2. **The correct primitive here is set-difference over key spaces, not per-key
   comparison.** Rhodes' formulation -- `result = cache_get(key, sentinel); if
   result is not sentinel` (https://python-patterns.guide/python/sentinel-object/)
   -- shows the fix has two shapes: pass a sentinel default, or ask the key space
   directly (`sid not in before`). The latter is stronger because it needs no
   sentinel discipline at the call site.
3. **The three states are ABSENT / UNCHANGED / TRANSITIONED, and only the middle
   one should suppress a bump.** The shipped predicate collapses ABSENT into
   UNCHANGED. Its *intent* (per the docstring) was "not a transition"; its
   *extension* is "ignore steps that appeared this commit".
4. **Prefix classification measures the development act, not the shipped change**
   -- and the literature says so independently of pyfinagent. Schiller: "I
   shouldn't need to designate it as a 'fix'... when the code needing a fix wasn't
   ever released" (https://brianschiller.com/blog/2023/09/18/changesets-vs-semantic-release/).
   xnok: "five feat commits and dozens of fix commits... a cluttered list of
   technical steps rather than a summary of value"
   (https://xnok.github.io/infra-bootstrap-tools/blog/intentional-releases-changesets/).
   **This vindicates 86.68's direction** -- the flip detector is the state-based
   analogue of a changeset. 86.91 must fix the predicate WITHOUT reverting to
   prefixes.
5. **Mature release tooling separates "nothing to release" from "failed to
   determine" only PARTIALLY -- and the gap is a known weakness, not a model to
   copy.** semantic-release makes *deliberate* skip an explicit, greppable marker
   (`[skip release]`) and offers `--dry-run` to print the decision
   (https://semantic-release.org/support/faq/), but its FAQ never names a
   "could-not-determine" state. Combined with the community debugging guides in the
   snippet table ("semantic-release Not Creating Release: Debug Guide" exists
   *because* the no-op is unexplained), the lesson is: **an explicit decision
   marker + a dry-run are the two affordances that make a declining detector
   auditable.** pyfinagent should adopt both rather than imitate the silence.
6. **Git's own contract confirms the design constraint AND reveals it does not
   apply here.** `post-commit` "cannot affect the outcome of `git commit`" and its
   exit status is ignored (https://git-scm.com/docs/githooks) -- so a
   never-raise notification hook is the *correct* shape. But git also guarantees
   stdout/stderr reach the user, and **that guarantee is broken on this project's
   actual dispatch path** (see Internal finding I3).
7. **A survived mutant is not evidence.** ISSTA '24 measures equivalent mutants at
   **4%-39%** of real-world mutants and notes equivalence detection is undecidable
   (https://arxiv.org/html/2408.01760). Therefore criterion 6's "control observed
   GREEN first" is not ceremony -- without it, a cell is unscorable. This is the
   same trap 86.68's Q/A caught (`MUTANT B exited 0 while both cells SURVIVED`).
8. **The one 2026 paper on commit classification is methodologically weaker than
   this project's own replay.** arXiv:2605.02033 uses one repo, no train/test
   split, no cross-validation, and *assumes* the declared prefix is ground truth
   (https://arxiv.org/html/2605.02033). pyfinagent's 86.68 replay -- full-corpus,
   three-arm, mutation-gated -- already exceeds it. **Do not import its
   methodology; keep the existing replay design.**

---

## Internal code inventory (file:line anchors)

| File | Anchor | Role | Status |
|------|--------|------|--------|
| `.claude/hooks/post-commit-changelog.sh` | `:98-173` `_flip_magnitude` | flip detector | **DEFECTIVE** |
| " | `:156-157` `before.get(sid) not in (None, "done")` | the conflation | **THE BUG** |
| " | `:161-169` | magnitude ladder (major/minor/patch) | correct |
| " | `:170-173` `except -> print("[changelog] flip-detect FAILED"...)` | never-raise marker | live, **never fired** |
| " | `:176-177` `if bump_type != "major": bump_type = _flip_magnitude()` | subject may only force MAJOR | correct |
| " | `:212` version header / `:228` bullet / `:252-270` row insert | row is UNCONDITIONAL, header+bullet gated on bump | correct |
| `scripts/qa/replay_changelog_rule_86_68.py` | `:54` | replay carries the **same** `not in (None,"done")` predicate | **DEFECT MIRRORED** |
| " | `:48-51` `enabled=False` mutant arm | mutation harness | reusable |
| " | `:108-116` control-green + killed gate | mutation scoring | correct (86.68 cycle-2 fix) |
| `.claude/hooks/auto-commit-and-push.sh` | `:396` `bash "$CHANGELOG_HOOK" >> "$LOG_FILE" 2>&1` | **stderr redirected to a gitignored log** | **CHANNEL DEFECT** |
| " | `:29` `trap 'exit 0' EXIT` | fail-open by design | correct |
| " | `:42` `LOG_FILE=.../auto-push.log` | 976,895 bytes | live |
| `.claude/settings.json` | `PostToolUse/Bash` + `PostToolUse/Write|Edit` | **two dispatch paths** into the same hook | live |
| `.git/hooks/` | only `pre-commit` present | **there is NO real git post-commit hook** | -- |
| `.claude/masterplan.json` | step `86.91` | 8 immutable criteria | pending |
| `handoff/archive/phase-86.68/contract.md` | `:106-107` | see I4 | archived |

### Measured internal findings

**I1 -- the defect reproduces exactly as filed.** On commit `e4f2e844`:
```
86.86 before: None -> after: done
SHIPPED rule newly_done: []
FIXED   rule newly_done: ['86.86']
```

**I2 -- the swallow is quantified, and the increase is TRACTABLE.** Over the 621
commits since 2026-08-11 (0 parse errors): **2** commits contain >=1
created-and-closed step (swallowed by the shipped rule); **5** contain >=1 normal
transition (counted). Criterion 3 demands the increase be "accounted for commit by
commit" -- that is at most 2 commits to justify, not a bulk claim. *Main must
re-derive the exact union at execution time; these two sets may overlap and I did
not compute the union.*

**I3 -- the never-raise marker is real but UNREACHABLE BY A HUMAN.** `grep -c
"flip-detect FAILED" handoff/logs/auto-push.log` = **0** over 976,895 bytes. So
today's frozen version is **not** caused by errors -- it is the silent `[]`.
Critically, git's "stderr reaches the user" guarantee (source 4) does **not**
hold: this is not a git hook (`.git/hooks/` has only `pre-commit`), and on the
auto-commit path `:396` redirects stderr into a **gitignored** log. Any fix that
adds a stderr marker and stops there will be **as invisible as the current
silence** on that path.

**I4 -- 86.68's own contract predicted this step.** `handoff/archive/phase-86.68/contract.md:106-107`:
> "A silent failure mode in a never-raise detector is worth a criterion of its own
> if it proves true; **file it, do not absorb it here** (only a criterion owns a...)"

86.91 is that criterion. This is prior-art *inside the repo* and should be cited in
the contract.

**I5 -- the replay script mirrors the defect at `:54`.** Fixing only the hook
leaves the replay harness measuring the OLD predicate, so criterion 3's "three
numbers" would silently compare the fixed hook against a stale baseline. **Both
files must change, and the replay must be able to express all three arms.**

---

## Consensus vs debate

- **Consensus (strong):** absence must not be encoded as the same value as a
  legitimate null; the remedy is a sentinel or a key-space membership test
  (sources 1, 2). No dissent found.
- **Consensus (strong):** subject-prefix classification is a poor proxy for
  shipped user-visible change (sources 5, 6). No 2024-2026 source defends it.
- **Debate / weakest link:** whether an automated detector should exist at all.
  Changesets' position is that inference is the problem and only human declaration
  is trustworthy. pyfinagent's flip detector is a **third way** -- inference from
  *verified state*, not from *prose* -- which neither camp discusses. This is the
  genuine gap in the literature and should be stated as such rather than papered
  over.
- **Adversarial note:** source 7 (the only 2026 paper) implicitly *endorses* prefix
  labels as ground truth. It is weak evidence (n=1 repo, no CV), and I record it as
  a qualification rather than a refutation.

---

## Pitfalls (from literature + prior cycles)

1. **Fixing the instance, not the class.** Criterion 2 explicitly fails a
   fix that special-cases `86.86`. The remedy must be the predicate.
2. **A mutant that survives proves nothing** (4%-39% equivalence, ISSTA '24). Observe
   the control GREEN first -- 86.68's cycle-1 residual was exactly this.
3. **Mirroring the fix into only one of two files** (I5).
4. **Adding a stderr marker that nobody can read** (I3). An unexplained no-op and
   an unread marker are the same defect.
5. **Reading a row count as coverage** -- `replay_changelog_rule_86_68.py:119-121`
   already warns: `MAX_ROWS=20`, so the table is trimmed and is not a census.
6. **Over-fitting the replay window.** 86.68's own Q/A found the corpus grew
   (348 -> 482 -> 496 -> 500 commits) between cycles; the "348" in CLAUDE.md is
   already stale. Criterion 3 says "the same 348-commit corpus" -- Main must
   state the corpus rule (`--since=2026-08-11`) and disclose the drift rather than
   quietly re-deriving a different number.

---

## Application to pyfinagent

- **The predicate fix** is a three-state model at `post-commit-changelog.sh:156-157`.
  `sid not in before` (key-space membership) is preferable to a sentinel default
  because it needs no discipline at the call site (Finding 2). A born-done step is
  a *creation*, which is a transition; only `before[sid] == "done"` is UNCHANGED.
- **Mirror it at `replay_changelog_rule_86_68.py:54`** or criterion 3's three
  numbers are not comparable (I5).
- **Criterion 4 (the silent-swallow class) needs a DECISION LOG, not just a
  marker.** The literature's two affordances are an explicit skip marker and a
  dry-run (source 3). Given I3, the marker must land somewhere a reader will look,
  and the outcome must be *distinguishable*: chore-by-classification vs
  no-flip-found vs error. Three distinct reasons, three distinct strings.
- **Criterion 7 (still never raises)** is satisfiable by fault injection into
  `_statuses`; the `try/except` at `:170-173` already returns `"none"` and prints.
  Verify by injection, not by reading -- the criterion says so.
- **Criterion 6's guard** must go RED when `None` is re-excluded. Build it as a
  fixture where a step is born-done, assert `newly_done` is non-empty, and mutate by
  restoring `not in (None, "done")`. Control GREEN first.
- **Do not touch verdict semantics or masterplan state** (criterion 8).

---

## Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch -- **8**
- [x] 10+ unique URLs total -- **28** (8 read-in-full + 20 snippet-only)
- [x] Recency scan (last 2 years) performed + reported -- non-empty, 4 in-window
- [x] Full pages read (not abstracts) for the read-in-full set; no `arxiv.org/pdf/` fetched (`/html/` route used for both preprints)
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every module in the caller's scope
- [x] Contradictions / consensus noted (source 7 flagged as partially adversarial)
- [x] All claims cited per-claim
- [!] **Gap disclosed:** I2's union of the two commit sets was not computed; Main
      must re-derive the exact three bump counts at execution time.
