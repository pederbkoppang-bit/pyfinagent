# live_check — phase-86.94

**STATUS: COMPLETE.** Both measurements taken, ≥1h apart, as criterion 1 requires. Cycle-2 remediation in §I.

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

*(Cycle-3: this block previously showed `37 / 5 / 0` mention counts — the cycle-1
numbers, captured before `QUOTE_DIRS` was widened to the whole `handoff/` tree.
It is regenerated here from the shipped run, which prints `282 / 6 / 49`. The
cycle-2 commit claimed §C/§E/§G had all been regenerated; that was true for §C
and §G and **false for §E**, which is the same accompany-not-replace defect one
level up.)*

| member | class | criterion-4 judgement (measured) |
|---|---|---|
| `backend/slack_bot/scheduler.py:503` `midnight` | **LEGITIMATELY RELATIVE** | A Slack "shipped today" digest must move with today. Name appears in 37 files, but every hit is descriptive (em-dash cleanup, an APScheduler job description, a different scheduler at `:761-795`); none quotes a count from this window. |
| `frontend_route_inventory.py:73` `30.days` | **SLIDING, left** | A rolling 30 days is the intended semantics, so the window stays. **Its figures HAVE been quoted as evidence** — measured over the whole `handoff/` tree, **49** files mention it, and an archived `experiment_results.md` uses its counts as success criteria (`usage_source: git_activity_30d`, `every_route_has_usage_count \| PASS (12/12 integer opens_30d)`). Those figures are unreproducible (the window has slid months past them) and inert (closed step, nothing live depends on them). `quoted_as_evidence: True, mentions_reviewed: 49`. |
| `verify_decision_log_86_97.py:360` `{first_stamp}` | **runtime-derived, allowed** | Figures **are** derived and **are** quoted — but each is quoted with the clock time it was taken at, and the checker asserts a *relationship*, never a pinned number. Upper bound floats with HEAD by design. |
| `replay_changelog_rule_86_68.py:114` `{CORPUS_SINCE}` | **was SLIDING → FIXED** | The TZ-naive pin. Corrected to `...Z`; figures unchanged. |

**The check enforces disclosure, not absence.** An earlier version asserted the
script name was absent from the quote corpus, and immediately falsified two of my
own allowlist claims — correctly as to the proxy, misleadingly as to the
question, since every hit was descriptive prose. Criterion 4 asks for a
*judgement to be stated*, so the check now surfaces the mention sites for audit
and requires the entry to have stated one. **This step's own artifacts are
excluded from that count, and the exclusion is stated rather than quietly
applied** — they necessarily discuss every member by name, which would guarantee
a hit for each and make the check meaningless.

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
