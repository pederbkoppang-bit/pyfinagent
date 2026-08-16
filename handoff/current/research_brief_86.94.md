# Research Brief -- phase-86.94

**Topic:** Reproducible corpus and window definitions in measurement scripts:
git revision-range time semantics (approxidate; bare-date `--since` applied at
the *current time of day*; `--until`/`HEAD` as floating upper bounds), and how to
detect or forbid now-relative windows in analysis code so a reported count can be
regenerated.

**Tier:** moderate. **Audit-class:** YES (`coverage.audit_class = true`,
K_required = 2 consecutive dry rounds).
**Role:** Layer-3 Researcher (external literature + internal code inventory).
**Started:** 2026-08-16.

---

## ENVELOPE (born inert -- phase-86.37)

```json
{
  "brief_status": "COMPLETE",
  "tier": "moderate",
  "external_sources_read_in_full": 17,
  "snippet_only_sources": 28,
  "urls_collected": 45,
  "recency_scan_performed": true,
  "internal_files_inspected": 17,
  "coverage": {
    "audit_class": true,
    "rounds": 12,
    "dry_rounds": 3,
    "K_required": 2,
    "new_findings_last_round": 0,
    "dry": true
  },
  "summary": "Bare-date --since resolves at the current time of day (measured: git rev-parse --since=2026-08-11 -> 2026-08-11T22:53:12+0200), confirming 86.91. NEW: the 86.91 fix is incomplete -- a naive pinned timestamp is TZ-LOCAL, so 2026-08-11T00:00:00 spans 13h between Seoul and New York and yields 707 vs 787 commits with BOTH ends already pinned; use ...Z / +0000 / @epoch. Also measured: --since=today is NOT midnight (0 vs 64 commits) although git-log(1) and git-rev-list(1) both claim it is; @{date} is local-reflog-based, expires at 90d and FAILS OPEN; A.. and --all silently change the denominator (707 vs 766). --since early-stop is a latent hazard only -- inactive here (committer dates monotonic across 2,999 pairs). Class census: 6 executable sites, 3 analytical defects (replay_changelog_rule_86_68.py:99 TZ, verify_decision_log_86_97.py:360 floating upper, frontend_route_inventory.py:73 --since=30.days), 1 CORRECT relative use (scheduler.py:503) that any outright ban would break. No off-the-shelf detector exists (Ruff DTZ bans tz-naive now(), not now()); write one on the lint_limits_usage.py AST+allowlist template.",
  "brief_path": "handoff/current/research_brief_86.94.md",
  "gate_passed": true
}
```

**Envelope flipped to `COMPLETE` as the final act of the run.** All 17 claimed
read-in-full URLs were verified by grep to appear literally in this file before the
flip; no `arxiv.org/pdf/` URL was WebFetched (the gate's html-first chain was followed).

---

## Status log (write-first, incremental)

- [t0] Brief created; envelope born inert. Read `.claude/agents/researcher.md`
  and `.claude/rules/research-gate.md` in full as binding instructions.
- [t1] Round 1: read `git-log(1)` + Peattie approxidate blog IN FULL. Two
  design-deciding findings landed immediately (see Key findings F1, F2).
- [t1] Internal: target script read in full; repo-wide sweeps started.

---

## Read in full (>=5 required; counts toward the gate)

| # | URL | Accessed | Kind | Fetched how | Key quote / finding |
|---|-----|----------|------|-------------|---------------------|
| 1 | https://git-scm.com/docs/git-log | 2026-08-16 | Official docs (tier 2) | WebFetch, full page | `--since-as-filter=<date>`: *"Show all commits more recent than <date>. This visits all commits in the range, rather than stopping at the first commit which is older than <date>."* -> plain `--since` **STOPS traversal** at the first older commit; it is not a filter. Also: *"When no <revision-range> is specified, it defaults to `HEAD`"* -- the default upper bound is floating by construction. And *"As a special case, `today` means the last midnight."* (i.e. only the literal word `today` is snapped to midnight -- a bare *date* is not.) |
| 2 | https://alexpeattie.com/blog/working-with-dates-in-git/ | 2026-08-16 | Authoritative blog (tier 3) | WebFetch, full page | *"The parser, called 'approxidate' is very flexible, and allows both fixed dates in any format you can dream up ... and relative dates ('today', '1 month 2 days ago', 'six minutes ago')."* + *"Approxidate isn't really documented anywhere, but the code for the parser is very readable."* -> **the bare-date time-of-day rule is NOT in any documentation**; it is only in `date.c`. Separately: `git log` **displays author dates but `--since`/`--until` match COMMITTER dates**. |
| 3 | https://raw.githubusercontent.com/git/git/master/date.c | 2026-08-16 | Source code, tier 1 (upstream git) | WebFetch, full file | The mechanism: `approxidate_careful()` tries `parse_date_basic()`; on failure it calls `approxidate_str(date, &tv)` seeded with `get_time(&tv)` -> `localtime_r(&time_sec, &tm); now = tm;`. Year/mon/mday are initialised to `-1` and back-filled by `update_tm(&tm, &now, 0)` from `now`; **`tm_hour/tm_min/tm_sec` are not re-specified for a bare date, so the current wall-clock time of day survives into the cutoff.** CAVEAT: the fetch returns a summary of the file, so this mechanism is NOT quoted as gospel -- it is corroborated by direct local measurement (M1 below), which is the load-bearing evidence. |
| 4 | https://reproducible-builds.org/docs/source-date-epoch/ | 2026-08-16 | Official spec (tier 2) | WebFetch, full page | The canonical prior art for *forbidding* a clock read: *"If the user has set SOURCE_DATE_EPOCH then they are taking a position that 'this is the current time; please use this instead of whatever clock you normally use'."* Its stated rationale is exactly ours: *"builds may take varying amounts of time, [so] simply setting the system clock cannot reliably ensure reproducible timestamps."* Reference idiom: `int(os.environ.get('SOURCE_DATE_EPOCH', time.time()))` -- an **env-overridable clock**, not a banned one. |
| 5 | https://git-scm.com/docs/git-rev-list | 2026-08-16 | Official docs (tier 2) | WebFetch, full page | Confirms `--since-as-filter` *"visits all commits in the range, rather than stopping at the first commit which is older than <date>"*; `--max-age`/`--min-age` take a raw **timestamp** (epoch), i.e. the only date input git will not re-interpret. Also *"Note that these are applied before commit ordering and formatting options, such as `--reverse`"* -- date limiting happens during traversal, not as post-processing. **Repeats the same false claim as git-log(1): "As a special case, today means the last midnight."** (refuted by M2 below). |
| 6 | https://docs.astral.sh/ruff/rules/call-datetime-now-without-tzinfo/ | 2026-08-16 | Official tool docs (tier 2) | WebFetch, full page | Ruff **DTZ005** flags `datetime.datetime.now()` with no `tz=`. Rationale quoted from CPython: *"an aware object represents a specific moment in time, a naive object does not contain enough information to unambiguously locate itself relative to other datetime objects."* **Scope limit that matters for us:** DTZ005 bans *tz-naive* `now()`, NOT *now()* itself -- `datetime.now(tz=UTC)` is DTZ-clean and still produces a non-reproducible window. So off-the-shelf linting does **not** cover this step's requirement; the detector must be custom. |

---

## MEASUREMENTS (local, re-runnable -- these are the load-bearing evidence)

Environment: `git version 2.50.1 (Apple Git-155)`, repo `/Users/ford/.openclaw/workspace/pyfinagent`,
wall clock at probe time `2026-08-16T22:53-22:54 +0200 (CEST)`.

Probe technique: **`git rev-parse --since=<X>` prints the resolved `--max-age=<epoch>`**, which is a
direct read of approxidate's output and needs no repo. (`builtin/rev-parse.c` routes `--since`
through the same `approxidate()` that `revision.c` uses for `git log`.)

### M1 -- a bare date is resolved at the CURRENT TIME OF DAY (confirms the phase-86.91 finding)

| input to `--since` | resolved cutoff |
|---|---|
| `2026-08-11` | `2026-08-11T22:53:12+0200` <- **= the wall clock's 22:53:12** |
| `2026-08-11T00:00:00` | `2026-08-11T00:00:00+0200` |
| `2026-08-11 00:00:00` | `2026-08-11T00:00:00+0200` |
| `30.days` | `2026-07-17T22:53:12+0200` |

### M2 -- `--since=today` does NOT mean midnight (git-log(1) and git-rev-list(1) are BOTH wrong)

Both official man pages state *"As a special case, today means the last midnight."* Measured:

| input | resolved cutoff | `git log --since=... \| wc -l` |
|---|---|---|
| `today` | `2026-08-16T22:53:58+0200` | **0 commits** |
| `midnight` | `2026-08-16T00:00:00+0200` | **64 commits** |
| `2026-08-16T00:00:00` | `2026-08-16T00:00:00+0200` | 64 commits |
| `yesterday` | `2026-08-15T22:53:58+0200` | (also time-of-day, not midnight) |
| `noon` | `2026-08-16T12:00:00+0200` | |
| `now` | `2026-08-16T22:53:58+0200` | |

`midnight`, `noon`, `now`, `yesterday` are in `date.c`'s `special[]` table; **`today` is not**, so it
falls through to the generic parser and keeps the current time of day. Consequence: **the
documentation actively misleads here** -- a reader who "does the safe thing" per the man page and
writes `--since=today` gets a window that empties as the day advances. Do not cite the man page for
this; cite the measurement.

### M3 -- **NEW: a pinned naive timestamp is still not reproducible -- it is TZ-local**

Same string `2026-08-11T00:00:00`, same repo, different `TZ`:

| TZ | resolved epoch | = UTC instant |
|---|---|---|
| `Europe/Oslo` | 1786399200 | 2026-08-10T22:00:00Z |
| `UTC` | 1786406400 | 2026-08-11T00:00:00Z |
| `America/New_York` | 1786420800 | 2026-08-11T04:00:00Z |
| `Asia/Seoul` | 1786374000 | 2026-08-10T15:00:00Z |

**Spread = 46,800 s = 13 h** between Seoul and New York for the *same pinned string*. And it changes
the answer: with **BOTH ends already pinned** (`--since=2026-08-11T00:00:00 8dc70502`) the corpus is
**707 commits under `Europe/Oslo` and `UTC`, but 787 under `Asia/Seoul`** -- an 11.3% swing with no
code, no clock-of-day and no HEAD movement involved. TZ-invariant forms, measured identical across
Seoul/NY: `2026-08-11T00:00:00Z`, `2026-08-11T00:00:00+0000`, `@1786406400`.

### M4 -- the current pin does hold on repeat (the part that already works)

| command | result |
|---|---|
| `--since=2026-08-11T00:00:00 8dc70502` run twice | 707, 707 (stable) |
| `--since=2026-08-11T00:00:00 HEAD` | 766 (floats with every new commit) |
| `--since=2026-08-11 HEAD` (the 86.68 form) | **434** at 22:54, vs 621 @09:56 / 592 @10:17 recorded earlier the same day |
| `8dc70502` resolves to | `8dc705022fe7a7a0ade7cc1303f57aa04b1f5e61  2026-08-16T10:23:32+02:00` |

### M5b -- external sources 7 and 8 (appended after round 2)

| # | URL | Accessed | Kind | Fetched how | Key quote / finding |
|---|-----|----------|------|-------------|---------------------|
| 7 | https://arxiv.org/html/2306.11391 | 2026-08-16 | Peer-reviewed preprint (tier 1) | WebFetch, arXiv native HTML (per the gate's html-first chain; no PDF fetched) | *"the content of repositories changes over time, up to several times a day. This makes it difficult to reproduce the same dataset over time."* Proposes a **dataset fingerprint** `FP=(q,t)` -- *"a fingerprint composed of a query and a timestamp"* -- as the unit that makes a mined dataset regenerable. Quantified drift: re-running an identical query 13 months apart gave *"a non-negligible variation (+27.4%)"*; cross-version reproduction reached 96.8% precision. Directly validates "**a count is only meaningful when quoted with the (query, window) pair that produced it**". |
| 8 | https://docs.aws.amazon.com/durable-execution/patterns/best-practices/determinism/ | 2026-08-16 | Official vendor docs (tier 2) | WebFetch, full page | The canonical *detect-and-forbid* formulation: *"Anything that depends on wall-clock time, a random source, an external service, the local file system, or mutable global state is non-deterministic and must run inside a durable operation."* Explicitly names `time.time()` / `Date.now()`. **Crucially the remedy is NOT a ban but a checkpoint**: *"A step checkpoints its return value. On replay the step returns the checkpointed value instead of running the underlying code."* Plus the branching rule: *"Control flow decisions made outside steps must depend only on deterministic inputs."* Honest limit: the page describes the *symptom* of a violation, **not** an automated detector -- it has no determinism-assertion or replay-divergence tooling. |

### M5c -- external sources 9 and 10 (appended after round 3)

| # | URL | Accessed | Kind | Fetched how | Key quote / finding |
|---|-----|----------|------|-------------|---------------------|
| 9 | https://arxiv.org/html/2602.08561v3 | 2026-08-16 | Peer-reviewed preprint, **2026** (tier 1) | WebFetch, arXiv native HTML | *"A run was considered successful only if the script completed without errors and reproduced the original ground-truth outputs."* **[PARTIALLY ADVERSARIAL to the framing]** Its measured failure taxonomy is *"missing packages, incorrect file paths, missing objects or functions, shared library loading issues, package installation failures, and file-reading errors"* -- and it **does not examine time-dependent code, dates, or non-deterministic inputs as a category at all**. So the reproducibility literature's own failure taxonomies would NOT have caught our defect. Evidence that this class is under-studied, not that it is unimportant. |
| 10 | https://docs.oracle.com/en/java/javase/21/docs/api/java.base/java/time/Clock.html | 2026-08-16 | Official platform docs (tier 2) | WebFetch, full page | The canonical clock-injection prior art. *"The primary purpose of this abstraction is to allow alternate clocks to be plugged in as and when required. Applications use an object to obtain the current time rather than a static method."* / *"Best practice for applications is to pass a `Clock` into any method that requires the current instant and time-zone."* / on `fixed()`: *"the fixed clock ensures tests are not dependent on the current clock."* Note the design choice: `now()` is **not removed**, it is made **overridable**. |

### M5d -- external sources 11 and 12 (appended after rounds 4-5)

| # | URL | Accessed | Kind | Fetched how | Key quote / finding |
|---|-----|----------|------|-------------|---------------------|
| 11 | https://docs.databricks.com/aws/en/machine-learning/feature-store/time-series | 2026-08-16 | Official vendor docs (tier 2) | WebFetch, full page | The domain-adjacent analogue (cross-domain triangulation into ML/finance): *"Point-in-time correctness creates a training dataset that reflects feature values as of the time each label observation was recorded."* Leakage is *"when you use feature values ... that were not available at the time the label was recorded"* and *"This type of error can be hard to detect."* The enabling mechanism is a mandatory **timestamp key column** *"that ensures that each row ... represents the latest known feature values as of the row's timestamp"*. Same shape as our fix: **make the as-of instant an explicit, stored column instead of an implicit read of `now`.** |
| 12 | https://github.com/git/git/blob/master/Documentation/RelNotes/2.37.0.adoc | 2026-08-16 | Official upstream release notes (tier 2) | WebFetch, full file | Upstream's own statement of the early-stop hazard: *"`git log --since=X` will stop traversal upon seeing a commit that is older than X, but there may be commits behind it that is younger than X when the commit was created with a faulty clock. A new option is added to keep digging without stopping, and instead filter out commits with timestamp older than X."* Establishes the **minimum git version (2.37)** for `--since-as-filter` -- local git is 2.50.1, so it is available. |

### M5e -- external sources 13 and 14 (appended after rounds 6-7)

| # | URL | Accessed | Kind | Fetched how | Key quote / finding |
|---|-----|----------|------|-------------|---------------------|
| 13 | https://git-scm.com/docs/gitrevisions | 2026-08-16 | Official docs (tier 2) | WebFetch, full page | **A trap to forbid explicitly.** `<refname>@{<date>}` looks tempting as an upper-bound pin, but: *"Note that this looks up the state of your **local** ref at a given time; e.g., what was in your local master branch last week. If you want to look at commits made during certain times, see `--since` and `--until`."* It is reflog-backed (*"the ref must have an existing log ($GIT_DIR/logs/<ref>)"*), i.e. **per-clone, not pushed, and expiring**. Also: range syntax has the same floating-end trap -- *"you can omit one end and let it default to HEAD ... origin.. is a shorthand for origin..HEAD"*. |
| 14 | https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1003285 | 2026-08-16 | Peer-reviewed, PLOS Comput Biol (tier 1) | WebFetch, full article | Sandve et al., *Ten Simple Rules for Reproducible Computational Research* -- the **year-less canonical prior art**. Rule 1, *"For Every Result, Keep Track of How It Was Produced"*: *"you should ensure that every detail that may influence the execution of the step is recorded ... the critical details include the name and version of the program, as well as the exact parameters and inputs that were used."* Rule 9, *"Connect Textual Statements to Underlying Results"*: *"you will have to connect a given textual statement ... to the precise results underlying the statement."* Rule 6 treats a random seed as a first-class recorded input -- **the resolved window is our seed.** |

### M6 -- `@{date}` fails OPEN (measured; do not use it as an upper-bound pin)

```
$ git rev-parse --short 'HEAD@{2026-08-11}'   -> f3c06bf1
$ git rev-parse --short 'HEAD@{2026-01-01}'   -> warning: log for 'HEAD' only goes back to
                                                 Mon, 13 Jul 2026 21:44:59 +0200
                                                 838d2398          <- STILL EXITS 0 WITH A SHA
```
The warning goes to **stderr** and a sha is returned anyway. Reflog lives in `.git/logs`
(per-clone, never pushed); `gc.reflogExpire` is **unset here -> the 90-day default**, and
the oldest entry in this clone is 2026-07-13. A corpus pinned with `@{date}` therefore
resolves differently on a fresh clone and silently degrades after 90 days.

### M7 -- range syntax floats too

`git rev-list --count 8dc70502..` = **59** (omitted end defaults to `HEAD`);
`8dc70502..8dc70502` = 0. So `A..` is the same defect as an unpinned `--until`, in
different clothing. A detector keyed only on `--since`/`--until` would miss it.

### M5f -- external source 15 (appended after round 8)

| # | URL | Accessed | Kind | Fetched how | Key quote / finding |
|---|-----|----------|------|-------------|---------------------|
| 15 | https://docs.getdbt.com/docs/build/incremental-microbatch | 2026-08-16 | Official vendor docs (tier 2) | WebFetch, full page | The industry-standard remedy, stated as a hard rule: *"`--event-time-start` and `--event-time-end` are **mutually necessary**, meaning that if you specify one, you must specify the other."* And on the danger of implicit bounds: *"dbt doesn't know the minimum `event_time` in your data — it only uses the configs you provide... If you want to process data from the actual start of your dataset, you _must_ explicitly define it."* Standard runs *"process batches according to the current timestamp and the configured `lookback`"* -- i.e. **relative for operations, explicit two-ended for anything reproducible.** This is exactly the operational-vs-analytical split the pyfinagent rule needs. |

**Round-8 secondary angle (no new read-in-full):** searched for `GIT_COMMITTER_DATE`
rewriting as a corpus-stability threat. Only community-tier hits (GitHub Discussions,
gists, a pgsql-hackers thread). The authoritative form of that point is already held --
source 12's *"when the commit was created with a faulty clock"* plus measurement M5
(author-date != committer-date on 2 of 3,000 commits here). No source promoted.

### M5g -- external source 16 (appended after round 9)

| # | URL | Accessed | Kind | Fetched how | Key quote / finding |
|---|-----|----------|------|-------------|---------------------|
| 16 | https://prometheus.io/docs/prometheus/latest/querying/basics/ | 2026-08-16 | Official docs (tier 2) | WebFetch, full page | A fifth independent vendor arriving at the same design. PromQL's `@` modifier *"allows overriding the evaluation timestamp for queries"*, syntax `<expr> @ <timestamp>` with a Unix timestamp, plus `@ start()` / `@ end()` which *"resolve to the query's start and end times"*. Default without it: *"The value returned will be that of the most recent sample at or before the query's evaluation timestamp"* -- i.e. **relative by default, pinnable on demand**. Note the chosen unit is a **Unix timestamp**, not a local-time string -- the same TZ-hazard avoidance M3 argues for. |

### M5h -- external source 17 (appended after round 11; read in full, produced NO new finding)

| # | URL | Accessed | Kind | Fetched how | Key quote / finding |
|---|-----|----------|------|-------------|---------------------|
| 17 | https://arxiv.org/html/2604.25944 | 2026-08-16 | Peer-reviewed preprint, **2026** (tier 1) | WebFetch, arXiv native HTML | *From Code to Figure: A FAIR-Aligned Data Provenance Chain*. Normative rule: store *"the exact source state, input configuration, and execution context of the run"* together with the results, and *"the commit identifier of the analysis library can, in turn, be embedded in the figure metadata together with a reference to the originating dataset."* Required fields include git commit id, **local code modifications**, input files, runtime logs, environment. **[GAP-CONFIRMING]** It **does not address relative time windows or time-dependent parameters at all** -- a second 2026 paper, independent of source 9, whose provenance taxonomy would not have caught our defect. Recorded as a **de-dup** of sources 7 + 14, which is why round 11 is scored DRY. |

### M8 -- the traversal spec is ALSO part of the corpus definition (round 10, internal)

Same pinned window (`--since=2026-08-11T00:00:00Z`, upper `8dc70502`), four denominators:

| traversal | count |
|---|---|
| default (all parents, tip=`8dc70502`) | 707 |
| `--first-parent` | 707 |
| `--no-merges` | 707 |
| `--all` (every ref, not just the tip) | **766** |

So `(since, until)` is **not sufficient**; the reproducible unit is
`(since, until, tip/ref-scope, traversal flags)`. Also checked: this clone is **not
shallow** and **not grafted** (`git rev-parse --is-shallow-repository` -> `false`, 3,591
commits) -- a shallow clone would silently truncate any corpus, so "not shallow" belongs
in the recorded fingerprint too.

**Negative finding (round 5, reportable):** a targeted search for an off-the-shelf
static-analysis rule for this class (Semgrep registry, Ruff, Bandit, pylint) surfaced
**no rule that detects a now-relative *analysis window*.** Semgrep's own docs describe
only the general custom-rule machinery. Combined with source 6's scope limit (Ruff DTZ
bans *tz-naive* `now()`, not `now()` itself), the conclusion is firm: **the detector for
86.94 must be written, not adopted.** The repo already has the right template for it --
see the lint infrastructure row below.

---

## Internal code inventory (file:line anchors)

| File:line | Window form | Role | Status |
|---|---|---|---|
| `scripts/qa/replay_changelog_rule_86_68.py:99` | `CORPUS_SINCE = "2026-08-11T00:00:00"` | lower bound of the 86.68 replay corpus | **PARTLY FIXED.** Time-of-day pinned (86.91). **Still TZ-local** -> 707 commits in Oslo/UTC, 787 in Seoul (M3). Residual defect. |
| `scripts/qa/replay_changelog_rule_86_68.py:113` | `CORPUS_UNTIL = os.environ.get("CORPUS_UNTIL", "8dc70502")` | upper bound | **FIXED (86.91 cycle-2).** Pinned to a real commit (`8dc7050 2026-08-16T10:23:32+02:00`), env-overridable to `HEAD` on purpose. This is the SOURCE_DATE_EPOCH idiom (source 4) reinvented correctly: an env-overridable pin, not a ban. |
| `scripts/qa/replay_changelog_rule_86_68.py:124-129` | prints resolved endpoint next to the count | disclosure | **GOOD PRIOR ART.** `git rev-parse --short CORPUS_UNTIL` is printed with every headline number: *"every count below is quoted against the endpoint printed above."* This is source 7's `FP=(q,t)` fingerprint in miniature. Generalise this, don't reinvent it. |
| `backend/slack_bot/scheduler.py:499-503` | `git log --since-as-filter=midnight` | daily Slack digest "shipped today" | **CORRECT, and deliberately now-relative.** In-code comment: *"committer-date local tz; --since-as-filter avoids early traversal stop, git >= 2.37"*. Uses `midnight` (a real `date.c` special) not `today` (which is NOT one -- M2). **This file is the counter-example that proves the rule must not be "ban now-relative windows".** |
| `scripts/harness/frontend_route_inventory.py:70,73` | `git log --since=30.days` | 30-day per-route commit counts | **NOW-RELATIVE, figures reported.** Mitigation already present: the function returns `(counts, git_command_string)` so the command is disclosed -- but the command itself is relative, so disclosure does not make it regenerable. Prime candidate for the new rule. |
| `scripts/qa/verify_decision_log_86_97.py:360` | `git log --since={first_stamp}` | window derived from a data timestamp | Data-derived, not clock-derived -- needs classification by the rule, not automatic condemnation. |
| `scripts/qa/rail_drop_rate.py:184` | `--since` CLI arg, `default=None` | operator-supplied window | Operator-parameterised; the reported figure must carry the arg. |
| `scripts/qa/verify_changelog_flip_86_91.py:441` | prose only | the live gate for the 86.91 rule | Documents the hazard; no window of its own. |
| `.claude/masterplan.json` (step 79.2) | `--since='2026-07-24 08:39:03'` -> "15 non-test product modules" | operator-action text | Lower bound pinned but **naive/TZ-local**; upper bound floats (implicit `HEAD`). Same two residuals as the replay script. |
| `.claude/masterplan.json` (step 86.69) | `--since=2026-06-11 --until=2026-06-16 -- backend/` -> "only away-ops, Slack and alerting commits" | audit_basis for the empty-HOLD regression | **BOTH ends bare dates** -> both slide with the time of day. Qualitative claim, so low blast radius, but it is a member of the class. |
| `.claude/masterplan.json` (step 86.94) | quotes the 621/592/706 figures | this step's own `audit_basis` | The figures are correctly labelled as clock-dependent; note 706 is no longer reproducible either (M4 measures 707 at a pinned upper bound). |
| `handoff/current/experiment_results_86.91.md:115,122`, `live_check_86.91.md:75-83`, `evaluator_critique_86.91.md:46`, `day_report_2026-08-16.md:134`, `q1_binding_constraint_86.59.md:199`, `day_report_2026-08-13.md:117`, `research_brief_62.1.md:84` | quoted counts from now-relative windows | handoff artifacts | Each quotes a figure produced by a sliding window. `evaluator_critique_86.91.md:46` is the *documented* instance of the failure (Main claimed 706/250, Q/A re-ran and got 710/252). |
| `CHANGELOG.md` | **zero** `--since`/`--until` occurrences | -- | Clean. No window-derived figure is quoted there. |

Masterplan census (executed, not eyeballed -- walked every string in `.claude/masterplan.json`):
**4 strings** quote a git date window, across steps **79.2, 86.69, 86.94 (x2)**.

### Completed census -- every git-date-window site in EXECUTABLE code

Swept `scripts/`, `backend/`, `.claude/hooks/`, `.claude/workflows/`, `.github/workflows/`:

| Site | Form | Class |
|---|---|---|
| `scripts/qa/replay_changelog_rule_86_68.py:114` | `--since={CORPUS_SINCE}` + positional `CORPUS_UNTIL` | analytical, both ends pinned, **TZ-naive** |
| `scripts/qa/verify_decision_log_86_97.py:360` | `--since={first_stamp}` (data-derived), upper = HEAD | analytical, **upper floats** |
| `scripts/harness/frontend_route_inventory.py:73` | `--since=30.days` | **fully now-relative**, figures reported |
| `backend/slack_bot/scheduler.py:503` | `--since-as-filter=midnight` | **operational, correctly relative** |
| `scripts/qa/rail_drop_rate.py:184` + `:227` | `--since` CLI arg; prints `span` with the count | operator-parameterised, discloses window |
| `scripts/qa/census_qa_write_guard_log_86_31.py:22` | `--before` CLI arg | operator-parameterised |
| `.claude/hooks/post-commit-changelog.sh:20-23`, `auto-commit-and-push.sh:385` | `git log -1` | **no window** -- single commit, out of scope |
| `.github/workflows/*` (12 files) | -- | **zero** occurrences |
| `CHANGELOG.md` | -- | **zero** occurrences |

So the class has **6 members** in executable code, of which **3 are analytical defects
(replay:114 TZ, verify_decision_log:360 floating upper, frontend_route_inventory:73
fully relative)**, 1 is a correct operational use, and 2 are operator-parameterised.

Existing infrastructure the fix should reuse rather than reinvent:
- `scripts/governance/lint_limits_usage.py` -- the repo's established **AST-based
  repo-wide lint** shape: walks every tracked `.py`, explicit allowlist of files that
  legitimately violate, exit codes `0 / 1 (--strict) / 2 (misuse)`.
- `.github/workflows/governance-lint.yml` (+ `ascii-logger-lint.yml`,
  `env-syntax-lint.yml`) -- the established "one lint per class, one workflow" idiom.
- **No `pyproject.toml` exists**, so Ruff is not configured; "just enable Ruff DTZ" is
  not available and would not catch this class anyway (source 6).
- `scripts/qa/` holds 84 files on the `verify_*` / `replay_*` / `mutation_matrix_*`
  naming convention.
- Worktree is currently dirty (**18 modified tracked files**) -- source 17 names local
  modifications as a required provenance field, so a fingerprint should record it.

---

## Search-query composition (three-variant discipline, made visible)

| Variant | Queries run |
|---|---|
| **Year-less canonical** | `git log --since approxidate bare date current time of day semantics`; `deterministic analysis avoid wall clock now() reproducible query point-in-time snapshot`; `freezegun time-machine forbid datetime.now hermetic deterministic tests lint rule`; `point-in-time correctness as-of join feature store training serving skew leakage`; `dbt run_started_at relative date filter incremental model not reproducible backfill`; `GIT_COMMITTER_DATE rewritten rebase commit timestamps unreliable analysis git history`; `arguments against pinning fixed time windows rolling window metrics staleness monitoring drift`; `provenance metadata reported figures regenerable research artifact badging reproducible number`; `"reproducible" corpus definition commit range specification tooling regenerate exact count software study` |
| **Current-year (2026)** | `reproducibility mining software repositories dataset time window sampling bias 2026`; `semgrep custom rule detect relative date window analysis code static analysis reproducibility 2026` |
| **Last-2-year (2025-2026)** | `computational reproducibility time-dependent analysis scripts non-deterministic date range 2025 2026` |

The read-in-full set mixes all three: canonical/undated (git docs, Sandve 2013,
`java.time.Clock`, SOURCE_DATE_EPOCH, Prometheus, dbt), 2023-2024 (arXiv 2306.11391,
git 2.37 notes), and 2026 (arXiv 2602.08561v3, arXiv 2604.25944).

---

## Recency scan (last 2 years, 2024-2026) -- MANDATORY SECTION

**Performed.** Result: **3 new findings in the 2024-2026 window; none supersedes the
canonical sources, two of them are gap-confirmations rather than advances.**

1. **arXiv 2602.08561v3 (2026)** -- the most recent empirical study of computational
   reproducibility failure modes measured here does **not include time-dependent code
   or date windows in its taxonomy at all** (its categories are packages, paths,
   missing objects, library loading, install failures, file-read errors).
2. **arXiv 2604.25944 (2026)** -- a 2026 FAIR provenance chain for figures likewise
   **does not address relative time windows**; its normative rule ("record the exact
   source state, input configuration, and execution context") is a restatement of
   Sandve 2013 Rules 1 and 9.
3. **arXiv 2306.11391 (2023, within window on the arXiv-revision axis)** -- supplies
   the only *quantified* drift figure found anywhere: an identical query re-run 13
   months apart varied by **+27.4%**.

**Conclusion of the scan:** the 2024-2026 literature has **not** superseded Sandve et al.
2013 on this point, and has not produced a detector for it. The canonical rules still
govern; the newest work independently confirms the gap rather than closing it. The
practical state of the art lives in **tooling**, not papers -- SOURCE_DATE_EPOCH,
`java.time.Clock`, PromQL `@`, dbt `--event-time-start/--event-time-end`, AWS durable
steps -- all four of the vendor mechanisms read here converge on the same design.

---

## Identified but snippet-only (context; does NOT count toward the gate)

| URL | Kind | Why not fetched in full |
|---|---|---|
| https://arxiv.org/abs/2204.08108 | paper (MSR SLR) | Superseded for our purposes by 2306.11391, which quantifies drift |
| https://arxiv.org/html/2410.00639 | paper | About *representative sampling* of repos, not time windows |
| https://arxiv.org/pdf/2408.04782 | paper | Time-window bias in productivity metrics; adjacent, not on git semantics |
| https://dl.acm.org/doi/fullHtml/10.1145/3674805.3690747 | paper (ACM) | Duplicate of 2410.00639 |
| https://zenodo.org/records/5274208 | dataset | Duplicate of 2204.08108 |
| https://www.nature.com/articles/s42256-026-01219-7 | journal (Nature MI, 2026) | Recency candidate; paywalled, and the scan's conclusion was already established by 2 fetched 2026 papers |
| https://arxiv.org/pdf/2606.18320 | paper (2026) | Snippet already showed pure de-dup: "a data snapshot and counting protocol separate paper numbers from live recounts" = source 7 + source 14 |
| https://arxiv.org/pdf/2312.11028 | paper (4R survey) | Policy survey, no mechanism |
| https://arxiv.org/pdf/2605.21997 | paper | Agentic event-sourcing; the determinism contract is held better by source 8 |
| https://web.eecs.umich.edu/~pmchen/papers/devecsery17.pdf | paper | Deterministic *replay* of whole systems; out of scope |
| https://arxiv.org/pdf/2607.02529 | paper | Browser simulation determinism; out of scope |
| https://iclr-blogposts.github.io/2026/blog/2026/dissecting-non-determinism/ | blog (2026) | LLM/GPU non-determinism, a different mechanism |
| https://github.com/spulec/freezegun | code | Test-time clock freezing; complements but does not address analysis windows |
| https://betterstack.com/community/guides/testing/time-machine-vs-freezegun/ | community | Tier 5 |
| https://medium.com/pythoneers/mastering-time-dependent-tests-in-python-2025-freezegun-time-machine-the-clock-pattern-993b8a38f3c9 | blog | Tier 5; the "Clock Pattern" is held authoritatively by source 10 |
| https://semgrep.dev/docs/writing-rules/overview | vendor docs | Generic rule-authoring; no rule exists for this class (the negative finding) |
| https://github.com/semgrep/semgrep | code | Same |
| https://docs.getdbt.com/docs/build/incremental-models | vendor docs | Superset page; the microbatch page (source 15) is the on-point one |
| https://www.tobikodata.com/blog/dbt-incremental-but-incomplete | vendor blog | Competitor critique; partisan |
| https://discourse.getdbt.com/t/running-backfills-in-incremental-models-obsolete-records-may-persist/7661 | forum | Tier 5 |
| https://www.systemoverflow.com/learn/ml-feature-stores/feature-store-architecture/point-in-time-correctness-and-time-travel | tutorial | Tier 5; source 11 is the official form |
| https://docs.databricks.com/aws/en/machine-learning/feature-store/time-series (PIT section) | -- | (read in full, listed above -- not snippet) |
| https://github.com/orgs/community/discussions/22695 | forum | Tier 5 committer-date discussion |
| https://gist.github.com/ugultopu/0b6412674073a5b603f8227cb108441c | gist | Tier 5 |
| https://www.postgresql.org/message-id/558ACEC8.60000%40gmx.net | mailing list | Tier 5; interesting as prior art for a *push hook* that rejects bad timestamps |
| https://ladal.edu.au/tutorials/reproducibility/reproducibility.html | tutorial | Tier 5 |
| https://gitexamples.com/examples/93dd4a60-e155-47d9-90f9-14fe845e4c68 | tutorial | Tier 5 |
| https://www.oreilly.com/library/view/pragmatic-version-control/9781680500189/f_0049.html | book excerpt | Paywalled |
| https://en.wikipedia.org/wiki/Yesterday_(time) | encyclopedia | Irrelevant search noise |

---

## Key findings

1. **A bare `--since` date resolves to the current time of day** -- confirmed
   independently of the prior phase-86.91 observation by direct parser probe:
   `git rev-parse --since=2026-08-11` -> `2026-08-11T22:53:12+0200`, matching the wall
   clock to the second (M1). Mechanism per `date.c` (source 3): `approxidate_str` seeds
   from `localtime_r(now)` and `update_tm` back-fills only date fields.
2. **`--since=today` is NOT midnight, and both official man pages say it is.**
   git-log(1) and git-rev-list(1) both state *"As a special case, today means the last
   midnight"*; measured, `--since=today` returned **0 commits** while `--since=midnight`
   returned 64 (M2). `today` is absent from `date.c`'s `special[]` table. **The
   documentation is a trap here** -- the rule must name `midnight`, not `today`.
3. **NEW, and the highest-value finding: pinning the timestamp is not enough -- a naive
   timestamp is TZ-local.** The same string `2026-08-11T00:00:00` spans **13 hours**
   between Seoul and New York, and yields **707 vs 787 commits** on this repo with both
   ends already pinned (M3). The 86.91 fix is therefore **incomplete**. TZ-invariant
   forms measured: `...Z`, `...+0000`, `@<epoch>`.
4. **A window has more than two ends.** Same `(since, until)` gives 707 under the tip
   but **766 under `--all`** (M8); `A..` silently defaults its upper end to `HEAD`
   (M7, source 13); `@{date}` resolves against the **local, per-clone, 90-day-expiring
   reflog** and **fails open** with a stderr warning plus a sha (M6). The reproducible
   unit is `(since, until, ref-scope, traversal flags, non-shallow)`.
5. **`--since` stops traversal; `--since-as-filter` does not.** Upstream's own words:
   *"there may be commits behind it that is younger than X when the commit was created
   with a faulty clock"* (source 12). Measured **inactive on this repo** -- the two agree
   at three cutoffs, committer dates are monotonic across 2,999 adjacent pairs, and
   author != committer date on only 2 of 3,000 commits (M5). Guard against it; do not
   claim it as a present bug.
6. **The remedy in every mature system is OVERRIDE, not BAN.** Five independent
   mechanisms converge: SOURCE_DATE_EPOCH (source 4), `java.time.Clock` (10),
   AWS durable steps (8), PromQL `@` (16), dbt `--event-time-start/--event-time-end`
   (15). None removes `now()`; each makes the instant explicit, injectable and recorded.
   dbt states the two-ended rule hardest: the flags are *"mutually necessary."*
7. **A count is only meaningful with its window attached.** arXiv 2306.11391's
   fingerprint `FP=(q,t)` (source 7) and Sandve Rule 9 (source 14) say the same thing;
   the measured cost of not doing it is **+27.4%** drift on an identical query 13 months
   apart, and locally 621 -> 592 -> 434 on one command in one day (M4).
8. **No off-the-shelf detector exists for this class.** Ruff DTZ bans tz-naive `now()`,
   not `now()` (source 6); Semgrep has no registry rule for it; the 2026 reproducibility
   papers do not include it in their taxonomies (sources 9, 17). **It must be written.**

## Consensus vs debate

**Consensus (5 vendors + 2 papers):** make the as-of instant explicit, overridable and
recorded; a reported number carries its window.

**Genuine debate / the counter-case** (the honest adversarial position, and it is
supported *inside this repo*): fixed windows are **wrong** for monitoring. The drift-
detection literature argues fixed-length windows *"often fail to accommodate
nonstationarity"* and favours adaptive windows; dbt keeps relative windows for standard
runs and requires explicit ones only for backfills. `backend/slack_bot/scheduler.py:503`
is a correct, deliberate now-relative use -- a "shipped today" digest **must** slide.
**Therefore a rule that forbids now-relative windows outright would be wrong and would
break working code.** The defensible invariant is narrower and fully checkable:

> A now-relative window is legitimate for **operational** output and illegitimate the
> moment a number derived from it is **quoted somewhere durable** -- a criterion, an
> `audit_basis`, a handoff artifact, a CHANGELOG entry.

## Pitfalls (from literature + measurement)

- Trusting the man page on `today` (M2).
- "Pinned" naive timestamps that are TZ-local (M3) -- the failure mode that survives the
  obvious fix and looks fixed.
- Pinning one end only (`--since` pinned, `HEAD` floating) -- already realised on this
  step's own predecessor: `evaluator_critique_86.91.md:46` records Main claiming
  706/250 and the Q/A measuring 710/252 two hours later.
- "Fixing" a floating upper bound with `HEAD@{date}` -- strictly worse (M6, fail-open,
  per-clone, expires).
- A detector keyed only on `--since`/`--until` misses `A..`, `--all`, `30.days`, and
  every `datetime.now() - timedelta(...)` window (M7, M8; 193 clock-arithmetic hits
  across `scripts/`, 44 BigQuery `CURRENT_DATE()`/`INTERVAL` hits across `backend/` +
  `scripts/`).
- Banning `now()` outright: breaks `scheduler.py:503` and contradicts sources 8/10/15/16.
- A detector that cannot find its own known members is a failed gate (the step's own
  `audit_basis` says so) -- the positive control must be the **pre-86.91 form** of
  `replay_changelog_rule_86_68.py`, i.e. `--since=2026-08-11` with no upper bound.

## Application to pyfinagent (external findings -> file:line)

- **`scripts/qa/replay_changelog_rule_86_68.py:99`** -- change `CORPUS_SINCE` to a
  TZ-explicit form (`"2026-08-11T00:00:00Z"` or `"@1786406400"`). Measured effect: makes
  the count 707 everywhere instead of 707-or-787 (M3). This is the one *code* defect the
  research found that the step's `audit_basis` does not yet name.
- **`:113`** already implements source 4's env-overridable-pin idiom; **`:124-129`**
  already implements source 7's fingerprint-printing. Generalise these two into the
  rule; do not redesign them.
- **`scripts/harness/frontend_route_inventory.py:73`** (`--since=30.days`) and
  **`scripts/qa/verify_decision_log_86_97.py:360`** (upper = HEAD) are the two remaining
  analytical members.
- **`backend/slack_bot/scheduler.py:503`** is the **allowlist entry / negative test** --
  the detector must pass it, and a detector that flags it is over-broad.
- **Detector shape:** follow `scripts/governance/lint_limits_usage.py` (AST walk +
  allowlist + `0/1/2` exit codes) and add a `.github/workflows/*-lint.yml` sibling, per
  the repo's established one-lint-per-class idiom. Ruff is not an option (no
  `pyproject.toml`, and DTZ does not cover this).
- **Quoted-figure surface:** 4 masterplan strings (79.2, 86.69, 86.94 x2), ~7
  `handoff/current/*.md` files, **0** in `CHANGELOG.md`, **0** in `.github/workflows/`.

---

## Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch -- **17**
- [x] 10+ unique URLs total (incl. snippet-only) -- **~46** (17 full + 29 snippet-only)
- [x] Recency scan (last 2 years) performed + reported -- section above, 3 findings
- [x] Full papers / pages read (not abstracts); arXiv via `arxiv.org/html/` per the
      gate's html-first chain -- **no `arxiv.org/pdf/` URL was WebFetched**
- [x] file:line anchors for every internal claim -- inventory table + census table

Soft checks:
- [x] Internal exploration covered every relevant module (`scripts/`, `backend/`,
      `.claude/hooks/`, `.claude/workflows/`, `.github/workflows/`, `masterplan.json`,
      `handoff/current/`, `CHANGELOG.md`)
- [x] Contradictions / consensus noted -- incl. the **doc-vs-behaviour contradiction on
      `today`** and the **genuine counter-case for relative windows**
- [x] All claims cited per-claim (URL or `file:line` or measurement id)

## Adaptive-coverage log (audit-class, K=2)

| Round | New read-in-full findings | Note |
|---|---|---|
| 1 | 6 (sources 1-6) | git semantics + SOURCE_DATE_EPOCH + Ruff scope limit |
| 2 | 2 (7-8) | fingerprint FP=(q,t); AWS determinism contract |
| 3 | 2 (9-10) | 2026 taxonomy gap; `java.time.Clock` |
| 4 | 1 (11) | point-in-time correctness |
| 5 | 1 (12) | git 2.37 `--since-as-filter` provenance; **semgrep search returned nothing (negative finding)** |
| 6 | 1 (13) | `@{date}` reflog trap |
| 7 | 1 (14) | Sandve Rules 1/9 (year-less canonical) |
| 8 | 1 (15) | dbt two-ended backfill rule; GIT_COMMITTER_DATE angle yielded **0** (community tier only) |
| 9 | 1 (16) | PromQL `@` modifier |
| 10 | **0 -- DRY #1** | Adversarial search (fixed-vs-rolling windows) returned drift-detection literature that *supports the counter-case already recorded* -- no new authoritative source to promote. Internal M8 measured. |
| 11 | **0 -- DRY #2** | Source 17 (arXiv 2604.25944) **was** fetched and read in full, and produced **no new finding**: its normative rule de-dups Sandve Rule 1 / FP=(q,t), and its silence on time windows de-dups the gap already found in source 9. Counted as dry on FINDINGS, per the rule's wording; disclosed so an auditor can check the judgement rather than take it. |
| 12 | **0 -- DRY #3 (confirmatory)** | `TopVenues` (2606.18320) snippet was pure de-dup; final internal sweep closed the census with no new members. |

`dry_rounds = 3 >= K_required = 2` -> **`coverage.dry = true`**.

### M5 -- the early-stop hazard is real in general but currently INACTIVE here

`--since` stops traversal at the first older commit; `--since-as-filter` visits all. On this repo the
two agree exactly (`766/766`, `1907/1907`, `2112/2112` at three cutoffs), and committer dates are
**monotonic across all 2,999 adjacent pairs** in the last 3,000 commits, with author-date !=
committer-date on only **2 of 3,000**. So: a latent hazard to guard against in the rule, not a
present defect to fix. Say so honestly rather than claiming a bug that does not reproduce.

