# live_check — phase-86.94

**STATUS: COMPLETE.** Both measurements taken, ≥1h apart, as criterion 1 requires. Cycle-2 remediation in §I; **cycle-4 remediation in §J** (§E was replaced by it — the `mentions_reviewed` design it described no longer exists).

Every block is verbatim tool output from this session. **No count in this file
is quoted without the clock time and HEAD it was taken at** — that is the whole
subject of the step, and a bare number here would be the defect committed inside
its own remediation.

---

## A. THE DRIFT, REPRODUCED BY EXECUTION (criterion 1)

### A0. The decisive proof is not a count diff — it is git's own resolver

```
now: 2026-08-16 23:09:03 CEST

$ git rev-parse --since=2026-08-11
--max-age=1786482543          -> local 2026-08-11 23:09:03 CEST | UTC 2026-08-11T21:09:03Z

$ git rev-parse --since=today
--max-age=1786914543          -> local 2026-08-16 23:09:03 CEST | UTC 2026-08-16T21:09:03Z
```

The bare date **carried today's clock time (23:09:03) onto the target date**.
And `--since=today` resolved to **now**, not midnight:

```
--since=today                     : 0 commits
--since=<today>T00:00:00          : 64 commits
```

`git-log(1)` and `git-rev-list(1)` both describe `today` as midnight. They are
wrong, and this is measured against git's own resolver rather than inferred.

### A1. Measurement 1 — persisted at capture time, not retyped

```
MEASUREMENT 1
local_time: 2026-08-16 22:50:20 CEST
utc:        2026-08-16T20:50:20Z
HEAD:       a5cbfd678b4871c44f28a549793536171546f242
bare   --since=2026-08-13          : 376
pinned --since=2026-08-13T00:00:00 : 424
bare   --since=2026-08-11          : 434
pinned --since=2026-08-11T00:00:00 : 766
```

### A1b. The boundary date had to be chosen BY MEASUREMENT, and that is a finding

The drift is observable only when commits actually fall in the band the cutoff
slides across. The obvious choice, `2026-08-11`, would have shown **no change**
tonight — its last commit is at 22:36:46, already behind the 22:50 cutoff, so
that date's drift window is **exhausted**:

```
=== exact times of 2026-08-11 commits at/after 22:00 local ===
22:24:03 22:24:05 22:25:31 22:25:34 22:26:11 22:26:15 22:27:58 22:28:00 22:29:10 22:29:13
22:30:50 22:30:52 22:31:27 22:31:30 22:32:40 22:32:42 22:36:10 22:36:13 22:36:43 22:36:46
```

Two runs an hour apart on that date would have agreed — a **true** result that
would have looked like a refutation of the step's premise. `2026-08-13` has 22
commits in the 22:52–23:52 band, so it straddles:

```
22:52:42 22:52:42 22:53:11 22:53:14 22:55:23 22:55:24 23:10:53 23:10:54 23:11:14 23:11:16
23:12:39 23:12:41 23:13:33 23:13:35 23:22:49 23:22:51 23:35:39 23:35:40 23:39:47 23:39:49
23:52:09 23:52:11
```

**This is why a bare-date corpus is dangerous rather than merely wrong:** it is
silently stable whenever the slid band happens to be empty, and moves without
warning when it is not.

### A2. Measurement 2 — 1h00m49s after measurement 1

```
MEASUREMENT 2
local_time: 2026-08-16 23:51:09 CEST
utc:        2026-08-16T21:51:09Z
HEAD:       27f8c6f6371a5a6c8a391d7434206a13c19641f7
bare   --since=2026-08-13          : 360
pinned --since=2026-08-13T00:00:00 : 428
bare   --since=2026-08-11          : 438
pinned --since=2026-08-11T00:00:00 : 770
```

```
$ git rev-parse --since=2026-08-13
--max-age=1786657869   ->  2026-08-13 23:51:09 CEST
```

### A3. The result, and why it is the sharpest available statement of the defect

**The bare-date count went DOWN while the repository GREW.**

| form | 22:50:20 | 23:51:09 | change |
|---|---|---|---|
| `--since=2026-08-13` (bare) | 376 | **360** | **−16** |
| `--since=2026-08-13T00:00:00` (pinned) | 424 | **428** | +4 |
| `--since=2026-08-11` (bare) | 434 | 438 | +4 |
| `--since=2026-08-11T00:00:00` (pinned) | 766 | 770 | +4 |

A corpus that *shrinks* as history *grows* is not a corpus; it is a reading of
the clock. The pinned form does the only correct thing — it grows by exactly the
four commits that landed.

**The arithmetic closes exactly, so this is not a correlation:**

```
commits ADDED between the two measurements                   : 4
commits that SLID OUT of the 08-13 bare window (22:50->23:51): 20
predicted bare count = 376 + 4 - 20 = 360   | measured: 360
predicted pinned     = 424 + 4      = 428   | measured: 428
```

Both predictions land on the measured value with no residual.

And the third row is the control that makes the point complete: `--since=2026-08-11`
moved **+4**, exactly like the pinned forms — because that date's drift band was
already exhausted (§A1b). **The same command is reproducible on one date and not
on another**, decided entirely by where commits happen to sit relative to the
current time of day. That is what makes this class dangerous rather than merely
wrong, and it is why criterion 1 forbids pinning a count into it.

---

## B. THE PHASE-86.91 FIX WAS INCOMPLETE — a pinned timestamp is TZ-LOCAL

This is the finding the research surfaced and I re-measured. 86.91 answered its
criterion by pinning `CORPUS_SINCE = "2026-08-11T00:00:00"`. That is pinned to a
**clock** but not to a **timezone**:

```
=== TZ-stability of the corpus bound, before vs after ===
Europe/Oslo        naive=707     Z=707
Asia/Seoul         naive=787     Z=707
America/New_York   naive=707     Z=707
UTC                naive=707     Z=707
```

An **80-commit spread on the same repo and the same command**, decided by `$TZ`.

The asymmetry is explained by measurement rather than waved at — New York and
Oslo coincide because the band they straddle is empty:

```
commits in [2026-08-10T22:00Z, 2026-08-11T04:00Z): 0     <- Oslo vs NY band
commits in [2026-08-10T15:00Z, 2026-08-11T00:00Z): 80    <- Seoul band
```

So a TZ-naive pin can look perfectly stable for months and then move by 80 the
first time someone runs it elsewhere. **Silence is not evidence of safety.**

### The fix changes nothing on this machine, and that is the point

```
BEFORE: corpus: 707 commits in [2026-08-11T00:00:00 .. 8dc70502 = 8dc70502]
          version bumps under OLD rule (subject prefix)     : 251
          version bumps under SHIPPED flip rule (pre-86.91) : 9
          version bumps under FIXED flip rule (86.91)       : 11
        exit gate: control_green=True all_cells_killed=True cells_scored=2 -> exit 0

AFTER:  corpus: 707 commits in [2026-08-11T00:00:00Z .. 8dc70502 = 8dc70502]
          version bumps under OLD rule (subject prefix)     : 251
          version bumps under SHIPPED flip rule (pre-86.91) : 9
          version bumps under FIXED flip rule (86.91)       : 11
        exit gate: control_green=True all_cells_killed=True cells_scored=2 -> exit 0
```

Identical figures, now identical in **every** timezone. 86.91's published numbers
stand; they simply become regenerable off this laptop.

---

## C. THE CLASS, ENUMERATED FROM SOURCE (criterion 2)

**The rule, written down in the checker's own source:**

> A WINDOW SITE is any line in a tracked `*.py`/`*.sh` under `scripts/`,
> `.claude/hooks/` or `backend/` that passes `--since` / `--until` /
> `--since-as-filter` / `--until-as-filter` to git. It is SLIDING unless the
> value is (a) UTC-qualified (`Z`, `+0000`, `@epoch`), (b) an operator-supplied
> parameter with no hardcoded default, or (c) interpolated from a value read out
> of a pinned artifact.

**The rule is deliberately wider than "bare date"** — 86.91 pinned a naive
timestamp and believed it had closed this, so a bare-date-only rule would have
called that fix clean.

```
[2] ENUMERATION of live window sites, by the written-down rule (criterion 2)

  ok   [2] the self-exclusion covers exactly ONE file (this checker), so it cannot grow into a general escape hatch
  ok   [2] the file set is non-empty (a scan over nothing is not a clean bill of health)
  ok   [2] the rule finds at least one live window site
       backend/slack_bot/scheduler.py:503  'midnight'  -> ALLOWED
       scripts/harness/frontend_route_inventory.py:73  '30.days'  -> ALLOWED
       scripts/qa/replay_changelog_rule_86_68.py:114  '{CORPUS_SINCE}'  -> REPRODUCIBLE
       scripts/qa/verify_decision_log_86_97.py:360  '{first_stamp}'  -> ALLOWED
  ok   [2] every SLIDING site is either fixed or carries a RECORDED REASON in the allowlist
```

*(Cycle-2 correction: an earlier revision of this block listed
`frontend_route_inventory.py:70` as well as `:73`, and showed
`replay_changelog_rule_86_68.py:114` as `ALLOWED`. Neither reproduces. Line 70 is
a docstring and is correctly stripped, and `{CORPUS_SINCE}` resolves to
**REPRODUCIBLE** now that the constant carries a `Z` — so it needs no allowlist
entry at all. The block above is regenerated from a fresh run.)*

### Two defects the first version of my own rule had

1. **It matched its own documentation.** `replay_changelog_rule_86_68.py`
   *documents* this defect in a comment quoting `` `--since=2026-08-11` ``, and
   the scanner dutifully reported that prose as two SLIDING sites. Comments are
   now stripped — with a control in both directions, because a stripper that
   quietly does nothing looks identical to a correct one.
2. **It failed OPEN on indirection.** `--since={CORPUS_SINCE}` was classified
   REPRODUCIBLE on the reasoning that the value is "decided at the call site" —
   which is precisely how the TZ-naive constant stayed invisible. The resolver
   now reads the literal and **fails closed** when it cannot.

The second is not a cosmetic fix: it is the leg that found the real defect.

---

## D. KNOWN-MEMBER RECALL — the hard gate (criterion 3)

```
[1] KNOWN-MEMBER RECALL -- the pre-86.91 form, recovered from git (criterion 3)

  ok   [1] the pre-86.91 blob is recoverable from git
  ok   [1] the rule FINDS a window site in the pre-86.91 blob
  ok   [1] and classifies it SLIDING
       06c3265f:72  '2026-08-11'  -> SLIDING: bare date -- git applies the CURRENT TIME OF DAY
```

If the blob ever becomes unreachable this section **FAILS** rather than skipping:
a recall test that can silently not run is not a recall test.

---

## E. CLASSIFICATION AND THE CRITERION-4 JUDGEMENT

```
[3b] CRITERION 4 -- do any quoted figures derive from a SLIDING member?

  ok   [3b] the quote corpus is non-empty (an empty grep proves nothing)
       scheduler.py: mentioned outside this step's own artifacts in 282 file(s)
  ok   [3b] scheduler.py: the criterion-4 judgement is a STRUCTURED claim, not a sentence (quoted_as_evidence is an explicit bool)
  ok   [3b] scheduler.py: mentions_reviewed matches the measured count, so a drifting corpus RE-OPENS the judgement instead of ageing into a false statement
  ok   [3b] scheduler.py: the entry carries a stated REASON
       verify_decision_log_86_97.py: mentioned outside this step's own artifacts in 6 file(s)
  ok   [3b] verify_decision_log_86_97.py: the criterion-4 judgement is a STRUCTURED claim, not a sentence (quoted_as_evidence is an explicit bool)
  ok   [3b] verify_decision_log_86_97.py: mentions_reviewed matches the measured count, so a drifting corpus RE-OPENS the judgement instead of ageing into a false statement
  ok   [3b] verify_decision_log_86_97.py: the entry carries a stated REASON
       frontend_route_inventory.py: mentioned outside this step's own artifacts in 49 file(s)
  ok   [3b] frontend_route_inventory.py: the criterion-4 judgement is a STRUCTURED claim, not a sentence (quoted_as_evidence is an explicit bool)
  ok   [3b] frontend_route_inventory.py: mentions_reviewed matches the measured count, so a drifting corpus RE-OPENS the judgement instead of ageing into a false statement
  ok   [3b] frontend_route_inventory.py: the entry carries a stated REASON
```

**THE BLOCK ABOVE IS THE CYCLE-3 STATE AND IS SUPERSEDED. The design it shows —
`mentions_reviewed`, a pinned count of files whose text contains the member's
FILENAME — was removed in cycle 4 and is not merely annotated.** It was wrong in
two measured ways: it answered a different question from the one criterion 4
asks, so the bool it guarded stayed green when falsified; and it counted over the
working tree, 89.5% of which is gitignored, so it was a number about a machine.
Full account in §J. The current run prints:

```
[3b] CRITERION 4 -- do any quoted figures derive from a SLIDING member?

  ok   [3b] the quote corpus is the TRACKED set (a working-tree walk is a number about a machine, the same defect class this step closes)
  ok   [3b] the quote corpus is non-empty (an empty grep proves nothing)
  ok   [3b] scheduler.py: the criterion-4 judgement is a STRUCTURED claim, not a sentence (quoted_as_evidence is an explicit bool)
  ok   [3b] scheduler.py: the claim carries FIGURE PROBES, so it can be contradicted by a measurement
       scheduler.py: NAMED in 281 tracked file(s); a FIGURE it produced is QUOTED in 0
  ok   [3b] scheduler.py: quoted_as_evidence=False matches the measured figure evidence
  ok   [3b] scheduler.py: the entry carries a stated REASON
  ok   [3b] verify_decision_log_86_97.py: the criterion-4 judgement is a STRUCTURED claim, not a sentence (quoted_as_evidence is an explicit bool)
  ok   [3b] verify_decision_log_86_97.py: the claim carries FIGURE PROBES, so it can be contradicted by a measurement
       verify_decision_log_86_97.py: NAMED in 9 tracked file(s); a FIGURE it produced is QUOTED in 2: handoff/current/experiment_results_86.97.md, handoff/current/live_check_86.97.md
           handoff/current/experiment_results_86.97.md  ~commits=\d+\s+decision lines=\d+\s+gap=\d+  -> 'commits=47  decision lines=24  gap=23'
           handoff/current/live_check_86.97.md  ~commits=\d+\s+decision lines=\d+\s+gap=\d+  -> 'commits=51  decision lines=26  gap=25'
           handoff/current/live_check_86.97.md  ~commits matching the recursion guard=\d+  -> 'commits matching the recursion guard=26'
  ok   [3b] verify_decision_log_86_97.py: quoted_as_evidence=True matches the measured figure evidence
  ok   [3b] verify_decision_log_86_97.py: the entry carries a stated REASON
  ok   [3b] frontend_route_inventory.py: the criterion-4 judgement is a STRUCTURED claim, not a sentence (quoted_as_evidence is an explicit bool)
  ok   [3b] frontend_route_inventory.py: the claim carries FIGURE PROBES, so it can be contradicted by a measurement
       frontend_route_inventory.py: NAMED in 5 tracked file(s); a FIGURE it produced is QUOTED in 3: handoff/archive/phase-4.7.0/experiment_results.md, handoff/archive/phase-4.7.0/phase-4.7.0-contract.md, handoff/archive/phase-4.7.1/phase-4.7.1-contract.md
           handoff/archive/phase-4.7.0/experiment_results.md  ~"usage_source":\s*"git_activity_30d"  -> '"usage_source": "git_activity_30d"'
           handoff/archive/phase-4.7.0/experiment_results.md  ~\d+/\d+ integer opens_30d  -> '12/12 integer opens_30d'
           handoff/archive/phase-4.7.0/experiment_results.md  ~opens_30d=\d+  -> 'opens_30d=0'
  ok   [3b] frontend_route_inventory.py: quoted_as_evidence=True matches the measured figure evidence
  ok   [3b] frontend_route_inventory.py: the entry carries a stated REASON
```

The single most useful line is `scheduler.py: NAMED in 281 tracked file(s); a
FIGURE it produced is QUOTED in 0`. Those are the two predicates side by side:
the old tripwire tracked 281 and moved whenever anyone wrote a sentence; the
judgement actually at stake is 0.

| member | class | criterion-4 judgement (measured, cycle 4) |
|---|---|---|
| `backend/slack_bot/scheduler.py:503` `midnight` | **LEGITIMATELY RELATIVE** | A Slack "shipped today" digest must move with today. The window emits `d["commits_today"]` (`:501-507`), rendered as a bulleted list by `formatters.py:102-109`; **no count is ever formatted**. Named in 281 tracked files, **0** of which quote a figure it produced. `quoted_as_evidence: False`. |
| `frontend_route_inventory.py:73` `30.days` | **SLIDING, left** | A rolling 30 days is the intended semantics, so the window stays. **Its figures HAVE been quoted as evidence** — 3 tracked files, 5 hits: `handoff/archive/phase-4.7.0/experiment_results.md` carries `"usage_source": "git_activity_30d"`, `12/12 integer opens_30d` and `opens_30d=0`. Unreproducible (the window has slid months past them) and inert (closed step). `quoted_as_evidence: True`. |
| `verify_decision_log_86_97.py:360` `{first_stamp}` | **runtime-derived, allowed** | Figures **are** derived and **are** quoted — `commits=51  decision lines=26  gap=25` and `commits matching the recursion guard=26` — but each is quoted with the clock time it was taken at, and the checker asserts a *relationship*, never a pinned number. `quoted_as_evidence: True`. |
| `replay_changelog_rule_86_68.py:114` `{CORPUS_SINCE}` | **was SLIDING → FIXED** | The TZ-naive pin. Corrected to `...Z`; figures unchanged. |

**Two corrections to this section's own prior wording, replaced not accompanied.**
(i) The cycle-2/3 entry put `usage_source: git_activity_30d` and
`/portfolio 2 /login 1` inside quote marks; neither string is in the cited file.
It reads `"usage_source": "git_activity_30d"` (JSON form) and the second is
line-wrapped across `:62-63`. Paraphrase inside quote marks is not a quote, and
the probes above use the verbatim forms. (ii) The cited file itself,
`handoff/archive/_quarantine_2026-04-21/phase-3.7.5-v22/experiment_results.md`,
is **gitignored**, so it was never admissible evidence on a fresh clone; the
tracked carriers named above replace it.

**This step's own artifacts are excluded from the corpus, and the exclusion is
stated rather than quietly applied** — they necessarily discuss every member by
name, which would guarantee a hit for each and make the check meaningless.

---

## F. THE REGRESSION GUARD, MUTATION-TESTED (criterion 6)

```
[4] MUTATION -- a new sliding window must turn this guard RED (criterion 6)

  ok   [4] CONTROL: the tree's own replay has NO unlisted sliding window
  ok   [4] bare-date: KILLED -- introducing a bare date -- the original defect is flagged SLIDING
  ok   [4] relative-days: KILLED -- introducing a rolling N-day window is flagged SLIDING
  ok   [4] today-word: KILLED -- introducing `today`, which git does NOT resolve to midnight is flagged SLIDING
  ok   [4] tz-naive-pin: KILLED -- introducing a pinned but TZ-NAIVE timestamp -- the shape phase-86.91 believed was a fix is flagged SLIDING
  ok   [4] NEGATIVE CONTROL: a UTC-qualified window is NOT flagged
  ok   [4] STRIPPER: a window quoted in PROSE is not reported as a site
  ok   [4] STRIPPER CONTROL: the same text as CODE *is* reported
  ok   [4] RESOLVER: sees THROUGH indirection to a TZ-naive literal
  ok   [4] RESOLVER CONTROL: a UTC-qualified literal behind the same indirection is NOT flagged
```

Control observed GREEN first. Every kill has a paired **negative** control,
because a detector that flags everything discriminates nothing and gets switched
off — which is also why the guard is an allowlist rather than a ban.

---

## G. NO REGRESSION (criterion 7)

```
verify_no_sliding_windows_86_94.py   ALL GREEN: 45 passed, 0 failed
verify_changelog_flip_86_91.py       ALL GREEN: 42 passed, 0 failed
verify_workflow_args_boundary.mjs    ALL GREEN: 96 passed, 0 failed
immutable command                    green
```

No masterplan step was flipped and no verdict altered.

---

## H. THREE DEFECTS THE GUARD FOUND IN ITSELF, ON BEING COMMITTED

These were found by running the guard, not by review, and all three are the
step's own classes turned back on it.

### H1. It flagged ITSELF — and only once it was committed

`tracked_files()` uses `git ls-files`, so while the checker was untracked it was
invisible to its own scan. The moment it was committed, its section-[4] mutation
fixtures — deliberately-sliding literals whose whole job is to prove the rule
fires — were scanned as production code and reported as **14 false findings**.

A self-blind guard is worst exactly when it ships. It now excludes itself, and
the exclusion is bounded by an assertion rather than trusted:

```
  ok   [2] the self-exclusion covers exactly ONE file (this checker), so it cannot grow into a general escape hatch
```

```
tracked py/sh: 851   scanned: 850   excluded: 1
```

**Residual, stated rather than hidden:** a real sliding window introduced into
this checker is not caught by this checker. The mitigation is that its own
fixtures are asserted in both directions in [4], so a rule that stopped working
fails there instead.

### H2. Docstrings are a THIRD comment form

`is_prose` only knew `#` lines. This file's **own module docstring** quotes a
bare-date window while explaining the defect — so the scanner reported its own
explanation. `strip_docstrings` now blanks triple-quoted blocks while preserving
line numbering (so reported line numbers still point at real source), with its
own control pair:

```
  ok   [4] DOCSTRING STRIPPER: a window inside a triple-quoted block is not a site
  ok   [4] DOCSTRING STRIPPER CONTROL: the same window as CODE *is* reported
```

### H3. The rule covers `.sh`; every cell mutated `.py`

A guard demonstrated on one language is demonstrated on half its scope, and the
hooks are shell:

```
  ok   [4] SHELL: a sliding window in a .sh body is flagged
  ok   [4] SHELL NEGATIVE CONTROL: a UTC-qualified shell window is NOT flagged
  ok   [4] SHELL: a window in a `#` comment is not reported as a site
```

**24 → 30 assertions** at that point (37 after cycle 2), ruff clean, nothing weakened.

---

## I. CYCLE-2 REMEDIATION — five mutants survived, and three were named in my own filing

Cycle-1 verdict `wf_eb4c97d0-c34`: **FAIL**. Criteria 1, 2, 3 and 7 MET and the
product correct; criteria 4, 5 and 6 missed, each with counter-evidence the
evaluator **executed**. It also independently reproduced the TZ finding, the
drift arithmetic and the A1b near-refutation control, and took a third
measurement of its own (23:55:07, bare 362 / pinned 432) confirming the pattern
under a different observer.

### I1. Criterion 6 — five survivors (the cap)

| survivor | why it escaped | now |
|---|---|---|
| `--after=<bare date>` | an **exact synonym** of `--since` (measured: identical `--max-age`) that my option list never named | KILLED |
| `--before=<bare date>` | synonym of `--until` | KILLED |
| `--since <bare date>` (space form) | `WINDOW_RE` matched the line, `VALUE_RE` required `=`, so `raw==""` and the loop **`continue`d** | KILLED |
| now-relative arithmetic into git | `timedelta` reaching a window through an f-string | KILLED |
| bare date in an executed `"""…"""` | blanked by the H2 docstring fix — a false negative that fix itself opened | KILLED |

**Three of those shapes — `--after`, `--before`, now-relative arithmetic — are
named verbatim in this step's own `audit_basis`.** Their absence was a recall
failure against my own filing, not a scope decision, which is exactly what
criterion 3 exists to catch one level down.

**The space-form case is the sharpest.** A window option was *recognised* and
then *silently skipped* — a fail-**open** inside the module whose central claim
is that it fails closed. It now records `<unparsed>` and **fails**:

```
'sh("git","log","--since 2026-08-11")'
   WINDOW_RE matches: True   VALUE_RE: (none -> raw="" -> continue)
```

Widening the pattern had a consequence worth recording rather than smoothing
over: the space form is **ambiguous with English**, and
`print("... a bare --since date slides with the clock ...")` is executable code,
not a comment, so stripping cannot save it — the widened rule captured the word
`date` as a window value in two tracked files. The `=` form has no such
ambiguity, so it stays unconditional and fail-closed, while the space form
additionally requires a plausible value shape. **Residual, stated:** a
space-separated window with an exotic value is not detected.

### I2. Criterion 4 — my stated judgement was measurably FALSE

The allowlist said no figure from `frontend_route_inventory.py` had ever been
quoted. The scan behind that claim covered `handoff/current` **only**. Over the
whole tree the evaluator found 55 mentions and, decisively, an archived
`experiment_results.md` that quotes figures from that exact `--since=30.days`
window **as success-criteria evidence** (`usage_source: git_activity_30d`,
`every_route_has_usage_count | PASS (12/12 integer opens_30d)`).

`QUOTE_DIRS` now covers the whole `handoff/` tree (`rglob`), and the entry is
rewritten to what is true: **quoted, unreproducible, and inert** — the figures
are in a closed archived step and nothing live depends on them, which is why the
window is still left rather than pinned. "Never quoted" was wrong.

### I3. Criterion 5 — corrected in 2 of 7 carriers

Swept from the claim rather than my own list, the figure lives in 7 files; cycle
1 corrected 2. Now corrected **in place** in `day_report_2026-08-16.md`,
`escalation_86.90_86.91.md`, `harness_log.md` and two further occurrences in
`experiment_results_86.91.md`. Per-occurrence audit:

```
still unqualified: ['handoff/current/evaluator_critique_86.91.md:161', 'handoff/current/evaluator_critique_86.91.md:263']
```

**Those two are deliberately exempt, and here is the reason.** They are an
evaluator's own verdict transcript — a record of what that evaluator measured at
that time. Editing it would falsify the record rather than correct a claim, and
the no-self-eval guarantee depends on verdicts being immutable once returned.
The correct remedy for a superseded verdict is a later verdict, which is what
this document is.

### I4. Evidence integrity — my "verbatim" blocks did not reproduce

§G said `24 passed` against a measured 30 while §H in the *same file* said
`24 → 30`; §C listed `:70,73` where the run lists `:73`; §C showed
`{CORPUS_SINCE}` as `ALLOWED` where it measures `REPRODUCIBLE`. The cycle-1
commit that "corrected the stale assertion counts" touched
`experiment_results_86.94.md` only and left the operator-facing gate artifact
stale — the same defect the 86.91 cycle-2 Q/A raised, recurring inside the step
about numbers that do not reproduce. All three blocks are **regenerated from a
fresh run** above.

### I5. Scope — disclosed rather than silently omitted

Criterion 6 says "bare-date **or now-relative**". The rule's declared scope is
git revision-range windows; a now-relative expression that never reaches a git
window is out of it. Rather than leave that implicit, section `[5]` now runs a
**report-only census** and prints the surface it does not gate:

```
[5] SCOPE DISCLOSURE -- now-relative windows OUTSIDE the git surface

       245 now-relative expression(s) in 86 file(s), NOT gated -- outside the git revision-range scope
  ok   [5] the scope bound is REPORTED, not silently omitted (the census ran and found a non-empty surface, so 'guard green' cannot be read as 'no sliding windows anywhere')
```

Report-only on purpose: gating 245 sites would flood on legitimate uses
(schedulers, digests, TTLs) and the gate would be switched off — the same
reasoning that made the git rule an allowlist rather than a ban. The check fails
if the census finds *nothing*, since "clean repo" and "broken census" otherwise
look identical.

### Net

**30 → 37 assertions.** Nothing weakened, no criterion reinterpreted.

---

## J. CYCLE-4 REMEDIATION — the three named findings, and one the guard was hiding

The park note records the cycle-3 state as "the shipped guard is ALL GREEN 45/0".
**That was already false when it was written.** The guard was `44/1` at
`964b0255` (2026-08-17T00:51:13+02:00) — the commit that recorded it green — and
`42/3` by the time this cycle began. Preflight caught it; the provenance is in
`handoff/current/day_halt.md`.

### J0. Why it was red, and why that is the finding rather than a chore

`[3b]` pinned `mentions_reviewed`, a count of files whose text contains a
member's **filename**. Since the pinning commit exactly three handoff files were
added, and two of them merely *name* the guarded scripts: the park note itself
(`overnight_halt.md`) and `day_report_2026-08-17.md`; `day_halt.md` then moved all
three pins at once. **None quotes any figure derived from any window.** The
tripwire fired on writing prose about the thing it guards — the textbook
change-detector shape ("fails in the face of an unrelated change to production
code that does not introduce any real bugs", Google SWE-book ch.12).

### J1. Finding (a) — `quoted_as_evidence` was only `isinstance`-checked

Measured against the cycle-3 tree, both directions, scoring by FAIL-SET DELTA
because the base was already dirty:

```
--- M-D frontend_route_inventory quoted_as_evidence True -> FALSE (a wrong bool) ---
    rc=1  FAILED: 42 passed, 3 failed
    *** SURVIVED *** -- a factually WRONG criterion-4 judgement ships GREEN.

--- M-E scheduler quoted_as_evidence False -> TRUE (a wrong bool, other direction) ---
    rc=1  FAILED: 42 passed, 3 failed
    *** SURVIVED *** -- a factually WRONG criterion-4 judgement ships GREEN.
```

**Fix:** the bool is bound to a measurement of the property criterion 4 names.
`figure_probes` are patterns for a figure *produced by that member's window*,
each derived from the emitting expression in the member's own source — never
from my phrasing, which is the recall trap criterion 2 forbids for the
enumeration. The check asserts `quoted_as_evidence == bool(hits)`.

**And the corpus became the tracked set.** The research gate measured what I had
not: `handoff/` holds **49,094** `.md` of which **5,167** are tracked — **43,927
(89.5%) gitignored** via `.gitignore:80`. 45 of `frontend_route_inventory`'s 50
hits were in the ignored quarantine, and the allowlist's own smoking-gun citation
is itself gitignored. The count was a number about a machine in precisely the
class this step exists to close. Verified the True judgements survive the repair:
5 tracked hits across 3 tracked files, listed in §E.

### J2. Finding (b) — the fail-closed `<unparsed>` branch had no cell

```
--- M-A fail-OPEN: <unparsed> append -> bare continue ---
    rc=1  FAILED: 42 passed, 3 failed
    *** SURVIVED *** -- FAIL set identical to BASE; no cell covers this.
```

The module's central claim — an unparseable window fails closed rather than being
skipped — was asserted by a comment and executed by nothing.

That comment's own example was also **stale, and is replaced rather than left
standing**: it said `--since 2026-08-11` (space form) reaches the branch. It no
longer does; once `PLAUSIBLE_VALUE` landed, `window_value()` returns
`('2026-08-11', True)` and takes the ordinary value path. Re-measured, the shapes
that actually reach it are an argv list with a **variable** value, the
f-string-element form, `--since=` built by concatenation, and `--after` +
variable. An argv list with a runtime-computed bound is a realistic idiom, so
this was an uncovered branch, not a corner.

Four cells added, each asserting the reported value **is** `<unparsed>` — the
only signal that separates this branch from `classify()`.

### J3. Finding (c) — the argv cells were credited to the wrong leg

The argv widening has two separable mechanisms. Measured:

```
--- M-B neutralise VALUE_ARGV_RE  (argv VALUE-PARSE leg) ---
    *** SURVIVED *** -- FAIL set identical to BASE.

--- M-C neutralise WINDOW_RE argv alternative  (argv VISIBILITY leg) ---
    KILLED -- 3 NEW failure(s):
      + FAIL [4] argv-list-after: KILLED -- ... the argv-list spelling of the --after synonym ...
      + FAIL [4] argv-list-form: KILLED -- ... the ARGV-LIST spelling ...
```

So the cells are killed by **visibility** (`WINDOW_RE`'s `["']\s*,` alternative),
not by the value parse. `VALUE_ARGV_RE` was **entirely uncovered**, and the reason
is J2: with it neutralised, argv sites fall through to the fail-closed branch and
are flagged anyway. **The two uncovered mechanisms were masking each other.**

**Fix:** every cell now asserts its mechanism — value-classification cells assert
`value != "<unparsed>"`, fail-closed cells assert `value == "<unparsed>"`. This is
the distinction the mutation literature says is the whole game: assertion kills
"imply that the test oracles actually capture the correct program behaviour",
while others "may only show coincidental impacts" (arXiv:2306.02319, corroborated
arXiv:2511.11999).

### J4. The mutation matrix — control observed GREEN first

Every cell was named in `contract_86.94.md` **before** the work. A mutant whose
anchor does not apply is scored UNSCORABLE and counts as a failure, never a kill.

```
CONTROL -- the shipped tree, observed BEFORE any mutation is scored
  rc=0   ALL GREEN: 68 passed, 0 failed
  CONTROL GREEN. Kills below are differential against it.

--- M-A: restore the fail-OPEN `continue` in place of the <unparsed> append
    KILLED   rc=1  FAILED: 60 passed, 8 failed
--- M-B: neutralise VALUE_ARGV_RE (argv VALUE-PARSE leg)
    KILLED   rc=1  FAILED: 66 passed, 2 failed
--- M-C: neutralise WINDOW_RE's argv alternative (argv VISIBILITY leg)
    KILLED   rc=1  FAILED: 58 passed, 10 failed
--- M-D: frontend_route_inventory quoted_as_evidence True -> FALSE (wrong bool)
    KILLED   rc=1  FAILED: 67 passed, 1 failed
--- M-E: scheduler quoted_as_evidence False -> TRUE (wrong bool, other direction)
    KILLED   rc=1  FAILED: 67 passed, 1 failed
--- M-F: widen the corpus back to the WORKING TREE (untracked files included)
    KILLED   rc=1  FAILED: 67 passed, 1 failed
--- M-G: the quoted figures VANISH from the corpus (probes match nothing)
    KILLED   rc=1  FAILED: 67 passed, 1 failed

killed=7  survived=0  unscorable=0  of 7
PASS
```

**M-A, M-B, M-D and M-E all went from SURVIVED to KILLED.** That is the R8 proof
obligation discharged: the replacement is strictly stronger than what it removed,
not a loosening to get green. M-G is the specific check that drift detection was
**preserved** — if the quoted figures vanish, the judgement still re-opens. What
no longer re-opens it is someone writing a sentence.

### J5. Criterion 5 — the correction sweep, by claim class with a recall test

My cycle-3 sweep searched my own wording and missed survivors. This one is seeded
from the artifacts. **Recall test first**, on two members known to exist:

```
seed1: masterplan note phrase 'mentions_reviewed pinned'   -> .claude/masterplan.json:1
seed2: park-note phrase 'ALL GREEN 45/0'                   -> .claude/masterplan.json
                                                              handoff/current/contract_86.94.md
```

Both found, so the sweep is not blind. Class A (`mentions_reviewed`) over tracked
files returns 6 carriers, and each is dispositioned rather than counted:

| carrier | disposition |
|---|---|
| `scripts/qa/verify_no_sliding_windows_86_94.py` | **removed** — the field no longer exists |
| `.claude/masterplan.json` (86.94 note) | **corrected** — describes the replacement |
| `handoff/current/live_check_86.94.md` §E | **replaced** in place, not annotated |
| `handoff/current/experiment_results_86.94.md` | **replaced** in place |
| `handoff/current/contract_86.94.md` | current cycle; already states the replacement |
| `handoff/current/day_halt.md` | **RECORD, left verbatim** — it captures this morning's preflight output at the moment it was taken, and states that bumping the number is forbidden. Editing captured output would falsify the record, the same reasoning that left `evaluator_critique_86.91.md` untouched in cycle 3. |

Class B (`45/0`, `45 assertions`) returned 12 files, and **9 are coincidental** —
`all 45 heuristic names`, `all 45 trades`, `all 457 test files`, and a DOI
containing `46.4.268`. Reporting the raw hit count as if it were the class would
have been the over-sweep mirror of cycle 3's under-sweep. The real carriers are
`.claude/masterplan.json`, this file, and `experiment_results_86.94.md`;
`handoff/harness_log.md:35751` is an **append-only cycle record** and gets a new
entry rather than a rewrite.

### J6. What is NOT claimed

- The immutable command still runs the **86.91** checker and cannot fail on any
  defect in this step's class. It was green throughout and proves only that this
  work did not break 86.91. Disclosed in the contract; the evidence is this file.
- The guard still excludes itself from its own scan (§H1), and a real sliding
  window introduced *into it* would not be caught by it. Unchanged bound.
- `figure_probes` are a **judgement about which figures matter**, made by me and
  auditable in the source. They are not a proof that no other figure from these
  windows was ever quoted anywhere; they are a falsifiable statement about the
  figures each window actually emits, checked against the tracked corpus.

### Net

**45 → 68 assertions.** No criterion reinterpreted; no allowlist member added or
removed; no window rule relaxed.

### J7. Three corrections made AFTER the cycle-3 verdict — never graded by any Q/A

The cycle-3 evaluator returned CONDITIONAL, and I then made three further
corrections. **No verdict has ever been taken on them**, so they are re-stated
here rather than left for a reader to notice the timestamps. A fresh Q/A grades
the current tree, and this is what changed under it:

| id | what was wrong | disposition in the current tree |
|---|---|---|
| **W1** | `live_check_86.94.md:274` and `experiment_results_86.94.md:99` still read *"Name appears in **37** files"* — the cycle-1 figure — while the instrument printed 282. | Corrected, then superseded entirely by cycle 4: both now read the tracked-corpus figure (281 named / 0 figure-quoted). Verified: `grep "37 file"` over both artifacts returns nothing. |
| **W2** | Both artifacts claimed the widened `WINDOW_RE` *"immediately found a live site the old one missed"*. **I re-measured it myself and it is FALSE** — reverting only the widening leaves the live-site enumeration byte-identical, so it found **zero**. What I mistook for a find was `census_qa_write_guard_log_86_31.py:64`, an `argparse` flag for a non-git tool, i.e. a false positive I then excluded with the git-proximity rule. | **Retracted in place**, not annotated: `experiment_results_86.94.md:215` now carries the retraction and the measurement. The widening's real effect is confined to the mutation cells — a future-introduction gap, which is what criterion 6 governs. |
| **W3** | The allowlist prose said **55** files where the instrument measured 49. | Reconciled, then superseded by cycle 4 (the prose no longer carries a mention count at all — the count was the defect). Verified: `grep "55 file"` over both artifacts and the guard returns nothing. |

W2 is the one that matters, because it is a claim I made in my own favour and
then had to withdraw on my own measurement. It is recorded as a retraction rather
than quietly deleted.
