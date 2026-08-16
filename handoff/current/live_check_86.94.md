# live_check — phase-86.94

**STATUS: IN PROGRESS** — §A2 (measurement 2) is filled in once ≥1h has elapsed
since measurement 1, as criterion 1 requires. Everything else is final.

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

### A2. Measurement 2 — ≥1h after measurement 1

*Pending: taken after 23:50:20 CEST.*

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

  ok   [2] the file set is non-empty (a scan over nothing is not a clean bill of health)
  ok   [2] the rule finds at least one live window site
       backend/slack_bot/scheduler.py:503  'midnight'  -> ALLOWED
       scripts/harness/frontend_route_inventory.py:70  '30.days'  -> ALLOWED
       scripts/harness/frontend_route_inventory.py:73  '30.days'  -> ALLOWED
       scripts/qa/replay_changelog_rule_86_68.py:114  '{CORPUS_SINCE}'  -> ALLOWED
       scripts/qa/verify_decision_log_86_97.py:360  '{first_stamp}'  -> ALLOWED
  ok   [2] every SLIDING site is either fixed or carries a RECORDED REASON in the allowlist
```

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
       scheduler.py: mentioned outside this step's own artifacts in 37 file(s)
  ok   [3b] scheduler.py: its allowlist entry states an explicit criterion-4 judgement about quoted figures
       verify_decision_log_86_97.py: mentioned outside this step's own artifacts in 5 file(s)
  ok   [3b] verify_decision_log_86_97.py: its allowlist entry states an explicit criterion-4 judgement about quoted figures
       frontend_route_inventory.py: mentioned outside this step's own artifacts in 0 file(s)
  ok   [3b] frontend_route_inventory.py: its allowlist entry states an explicit criterion-4 judgement about quoted figures
```

| member | class | criterion-4 judgement (measured) |
|---|---|---|
| `backend/slack_bot/scheduler.py:503` `midnight` | **LEGITIMATELY RELATIVE** | A Slack "shipped today" digest must move with today. Name appears in 37 files, but every hit is descriptive (em-dash cleanup, an APScheduler job description, a different scheduler at `:761-795`); none quotes a count from this window. |
| `frontend_route_inventory.py:70,73` `30.days` | **SLIDING, left** | A rolling 30 days is the intended semantics. Mentioned in **0** files outside this step's own artifacts, so no count from it is load-bearing. It *does* print per-route figures — they are simply never quoted as evidence. |
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
verify_no_sliding_windows_86_94.py   ALL GREEN: 24 passed, 0 failed
verify_changelog_flip_86_91.py       ALL GREEN: 42 passed, 0 failed
verify_workflow_args_boundary.mjs    ALL GREEN: 96 passed, 0 failed
immutable command                    green
```

No masterplan step was flipped and no verdict altered.
