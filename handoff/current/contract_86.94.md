# Contract — phase-86.94

**Written BEFORE any GENERATE work.** Research gate cleared first.

---

## Step

`86.94` (P2) — *"corpora and windows defined relative to `now` are not
reproducible: `git log --since=<bare date>` is applied at the CURRENT time of
day, so at least one measurement frozen into an immutable criterion cannot be
regenerated"*

---

## Research gate — PASSED (enforced, audit-class, loop-until-dry)

`.claude/workflows/research-gate.js` by `scriptPath` (rail R7), run
`wf_2c05296c-5d4`, 2 agents, 227,210 tokens, 1081s.

```
gate_passed: true          agent_self_reported_gate_passed: true
self_report_disagreed: false               violations: []
sources_floor_ok: 17 >= 5                  urls_floor_ok: 45 >= 10
recency_scan_ok                            audit_class_dry_ok
brief_on_disk_ok: handoff/current/research_brief_86.94.md (46121 chars, independently read)
brief_status_in_brief: COMPLETE            all_17_claimed_sources_present_in_brief
urls_collected_corroborated: 45 <= 45 distinct URLs in the brief
coverage: {audit_class: true, rounds: 12, dry_rounds: 3, K_required: 2, dry: true}
```

Read in full includes: `git-log(1)`, `git-rev-list(1)`, `gitrevisions(7)`, git's
own `date.c`, reproducible-builds `SOURCE_DATE_EPOCH`, Ruff `DTZ005`, Prometheus
querying basics, dbt incremental microbatch, Databricks time-series feature
store, PLOS Comp Biol reproducibility, arXiv 2306.11391 / 2602.08561v3 / 2604.25944.

---

## Hypothesis — REPRODUCED BY EXECUTION before this contract

### H1 — the class, proven from git's own resolution rather than from a count

The cleanest proof is not a count diff. `git rev-parse` prints what git actually
resolved:

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

— even though `git-log(1)` and `git-rev-list(1)` both describe `today` as
midnight. That is a documentation-vs-behaviour gap, not a misreading.

### H2 — the phase-86.91 fix is INCOMPLETE, and this is design-deciding

86.91 answered its criterion by pinning `CORPUS_SINCE = "2026-08-11T00:00:00"`.
**A naive pinned timestamp is still TZ-local.** Measured on this repo varying
only the machine timezone. NOTE the two pairs measure different corpora and the
label matters: the block below is the **open-ended** whole-history count (no
upper bound), so it drifts with every new commit — measured 766/846 at the time
of writing and 776/856 a few hours later. The **both-ends-pinned** pair, which is
the one the replay actually uses, is a stable 707 (Oslo/UTC/NY) vs 787 (Seoul):

```
Europe/Oslo        --since=2026-08-11T00:00:00 -> 766
Asia/Seoul         --since=2026-08-11T00:00:00 -> 846
America/New_York   --since=2026-08-11T00:00:00 -> 766
UTC                --since=2026-08-11T00:00:00 -> 766
(Seoul, Z form)    --since=2026-08-11T00:00:00Z -> 766
(NY,    Z form)    --since=2026-08-11T00:00:00Z -> 766
```

An 80-commit spread on the same repo and the same command, decided by `$TZ`. The
`Z` form normalises it.

**The asymmetry is explained by measurement, not waved at:** New York and Oslo
coincide because the band they straddle is empty, while Seoul's is not.

```
commits in [2026-08-10T22:00Z, 2026-08-11T04:00Z): 0     <- Oslo vs NY band
commits in [2026-08-10T15:00Z, 2026-08-11T00:00Z): 80    <- Seoul band
```

So the defect's *magnitude is data-dependent*: a TZ-naive pin can look perfectly
stable for months and then move by 80 commits when someone runs it elsewhere, or
when commits land in a previously-empty band. Silence is not evidence of safety.

### H3 — the drift is only observable when commits fall in the slid band

For criterion 1 I must show two bare-date runs ≥1h apart with **differing**
counts. That required choosing the boundary date by measurement: the last commit
on 2026-08-11 is at 22:36:46, already behind the current cutoff, so that date's
drift window is **exhausted** and two runs tonight would agree — a true result
that would have looked like a refutation. 2026-08-13 has 22 commits in the
22:52–23:52 band, so it straddles.

**Measurement 1** (persisted at capture time, not retyped):

```
MEASUREMENT 1
local_time: 2026-08-16 22:50:20 CEST
utc:        2026-08-16T20:50:20Z
HEAD:       a5cbfd678b4871c44f28a549793536171546f242
bare   --since=2026-08-13          : 376
pinned --since=2026-08-13T00:00:00 : 424
```

Measurement 2 is taken after ≥1h has elapsed, in GENERATE.

---

## Immutable success criteria — copied VERBATIM from `.claude/masterplan.json`

1. "the drift is REPRODUCED first, by EXECUTION and WITHOUT PINNED FIGURES: run the bare-date command twice at times of day that differ by at least an hour and show the two counts DIFFER, and show that the midnight-pinned form differs from both. Do NOT copy a specific count into this criterion -- by this step's own thesis no such count can be regenerated, and an earlier revision of this criterion pinned 621/592/706, which measured 560/712 the same day. That revision was the identical trap this step exists to close, committed inside the criterion written to prevent it"
2. "the class is enumerated FROM SOURCE, not hand-listed: the enumeration rule is written down, the command is quoted with its output, and each member is classified as REPRODUCIBLE or SLIDING with the reason per member"
3. "the enumeration finds its own known member -- the pre-86.91 form of replay_changelog_rule_86_68.py, recoverable from git -- and a scan that cannot is a FAILED gate"
4. "for each SLIDING member, state whether any figure derived from it has been quoted in a masterplan criterion, an audit_basis, a handoff artifact or CHANGELOG; a member whose numbers were never quoted is lower risk and may be left, but that judgement must be stated rather than silent"
5. "any figure found to be unreproducible is CORRECTED IN EVERY FILE THAT CARRIES IT, not merely annotated in one -- a correction must replace, not accompany"
6. "a regression guard is added that would go RED if a new bare-date or now-relative window is introduced into a measurement script, and it is mutation-tested with the control observed GREEN first"
7. "verdict semantics are UNCHANGED: nothing here may turn a non-PASS into a PASS"

Immutable command:
`bash -c 'source .venv/bin/activate && python scripts/qa/verify_changelog_flip_86_91.py > /dev/null && echo green'`

**Disclosed weakness:** this command runs the *86.91* checker, which is green
today and would stay green through every defect this step is about. It cannot
fail on the class. The real evidence goes in `live_check_86.94.md`.

---

## Plan

**P1 — reproduce (criterion 1).** Two bare-date runs ≥1h apart on a date whose
commits straddle the band, plus the midnight-pinned form, plus the
`git rev-parse` resolution proof. **No figure is pinned into any criterion**, and
every count is quoted with the clock time and HEAD it was taken at.

**P2 — enumerate FROM SOURCE with a written-down rule (criteria 2, 3).** The rule
is stated in the checker; the class is git revision-range windows and
now-relative corpus arithmetic in measurement scripts. **Known-member recall is a
hard gate:** the scan must find the pre-86.91 form of
`replay_changelog_rule_86_68.py`, recovered from git, and FAIL if it cannot.
Per-member classification REPRODUCIBLE / SLIDING with the reason.

**P3 — classify and judge each member (criterion 4).** For every SLIDING member,
state whether any figure derived from it was ever quoted in a masterplan
criterion, an `audit_basis`, a handoff artifact or the CHANGELOG. A member whose
numbers were never quoted may be left — **but the judgement is stated, not
silent.** The research names three analytical defects and, importantly, **one
CORRECT relative use** (`scheduler.py:503`) that a blanket ban would break: the
guard must therefore be an allowlist, not a prohibition.

**P4 — correct unreproducible figures everywhere (criterion 5).** Replace, never
annotate. The enumeration of sites is driven by the **claim class**, with a
known-member recall seed — the 86.97 FAIL earlier tonight was caused by
enumerating from my own wordings, and I will not repeat it in the step whose
criterion 3 is a recall test.

**P5 — the regression guard (criterion 6).** An AST + allowlist detector on the
`lint_limits_usage.py` template (research: no off-the-shelf rule exists — Ruff
`DTZ005` bans tz-naive `now()`, not `now()` itself). Mutation-tested with the
control observed GREEN first, and every mutant checked to BUILD before scoring.

---

## Non-goals

- Banning relative windows outright. One measured member is legitimately
  relative; a ban would break it.
- Re-running or amending phase-86.68's or 86.91's closed criteria.
- Fixing `frontend_route_inventory.py`'s behaviour beyond classification if no
  quoted figure derives from it — that judgement is stated under criterion 4.

---

## References

- `handoff/current/research_brief_86.94.md` (gate PASSED, 17 sources / 45 URLs, dry)
- Measurement 1 persisted at capture time; measurement 2 taken in GENERATE
- `reference_git_since_bare_date_slides_with_clock` (auto-memory, this class)
