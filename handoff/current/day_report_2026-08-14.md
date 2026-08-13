# Day report — 2026-08-14 (overnight session, 00:00–11:00 CEST)

Continues `day_report_2026-08-13.md`. Backend **pid 93024** (restarted by a peer at
20:30:59Z on 08-13, not by me). **No restarts, no manual cycles, no flag promotions, no
`.env` writes, no history rewrites by this session.**

---

## THE HEADLINE IS AN OPEN SECURITY INCIDENT, AND MY EARLIER ALL-CLEAR CAUSED IT TO SIT

**An `sk-ant-oat01…` Anthropic OAuth token is published on `origin/main` in FIVE tracked
files**, committed 2026-08-08 → 08-10, **public for six days**.
Full record: `handoff/current/INCIDENT_2026-08-14_credential_exposure.md` (commit `24fbcf9f`).

**Nothing remediated** — no rotation, no history rewrite, no `.gitignore` edit. All
operator-gated under ask **06-2**. **Rotate first**: removing it from history does not
un-publish it.

**The leak is closed** (6 clean files after 08-10) but **why it stopped is unverified** —
inferred from output, not demonstrated.

**I reported these exact files CLEAN at ~05:40** and told the operator the risk was "real
but not realised." Wrong, for a reason worth keeping:

```
mine     sk-[A-Za-z0-9]{20,}        vs  sk-ant-oat01-…  ->  0   ([A-Za-z0-9] can't cross a hyphen)
correct  sk-ant-[A-Za-z0-9_-]{10,}  vs  sk-ant-oat01-…  ->  1
```

The regex is the shallow error. **My "positive control" was a synthetic hyphen-free
`sk-abcdef…` token I wrote to match my own pattern** — it exercised exactly the case the
regex handled and could never expose the case it missed. **A control built from your own
pattern tests the pattern against itself.** Every other control I ran tonight came from
the real artifact, and several caught me.

**Action:** re-scan anything cleared with that pattern using the **vendor prefix**
(`sk-ant-`, `xoxb-`, `ghp_`, `AIza`), not a charset class.

---

## Shipped

| Step | Outcome |
|---|---|
| **86.58** | **PASS, closed** — attempt 3. The only step flipped in this session. |
| 86.59 | Research gate **PASSED** (attempt 3) — verified by pre-run diff, +14,854 bytes of real verification, not a marker flip |
| 86.63 | Gate **PASSED** (audit-class, dry) + contract written, criteria programmatically verbatim |
| 86.64 | Gate **PASSED** — and it **refuted the step's premise** |
| 86.67 | Gate ran; surfaced the incident above |
| 86.62 | **ESCALATED** at 4 attempts (FAIL/CONDITIONAL/FAIL/FAIL) |
| 86.9 | **ESCALATED** at 4 attempts (COND/COND/FAIL/FAIL); criteria 1–5 MET, criterion 6's evidence was a vacuous guard |
| 86.44 | live_check created; **criterion 5 measured and DISPROVEN** (14 collisions of 16) |
| 86.65 / 86.66 | Decisive criteria answered by measurement, without doing criterion work outside their gates |

---

## Four findings that change how a step should be fixed

1. **86.64 — the premise is wrong in a helpful direction.** "Write/Edit hooks do not
   intercept Bash" is true as written, false as read. PreToolUse runs for *every* tool
   except `EndConversation`; **the gap is `settings.json`'s matcher** (`Write|Edit`),
   verified by me. CWE-693 "ignored", not "missing" — so criterion 4 must rest on
   **decidability**, not capability. And it is not hypothetical: phase-82.39's Q/A
   already channel-switched — *"blocked my Write, so I moved everything to stdin."*
2. **86.44 — nothing reads the cycle number.** `harness_state_reader.py:143-149` splits
   on the literal delimiter and never parses it. And uniqueness is **disproven**: 16
   concurrent writers → 2 distinct numbers, 14 collisions, mechanism at
   `finalize.py:70-72`/`:83-85`. That explains 141 reused numbers and `Cycle 1` × 482.
3. **86.66 — sorting by count hid the live signal.** By count the `AttributeError` is
   1 of 63 and I nearly recorded the step as misaimed. **By date it is the only current
   failure** — every commoner class died in May–August, and the job ran six clean days
   before 08-13 broke. *Sort failure censuses by recency before frequency.*
4. **86.68 — shipped, live, governing every commit, with zero handoff artifacts.**
   Tonight was an unplanned natural experiment: **91 commits → exactly 1 version bump**,
   on the 1 step that flipped, while **86.62 failed 4× and 86.9 failed 2×** with
   `phase-86.x` subjects and produced **zero** bumps.

---

## What the Q/A caught, all mine

Eight verdicts, and the pattern is one class repeated:

- **86.58:** published a blast radius **without ever running with the flags on** — my
  script asserted them `False` and used a hand-set proxy. The number **inverted** (1-of-1
  → 0-of-2). Then I called a *real* measurement a dead end, and the Q/A corrected me **in
  my favour**.
- **86.62 (×4):** *"the third consecutive cycle of one class — a correction declared
  complete while its superseded text survives beside it."* Cycle 2 named four locations;
  I fixed two. Cycle 3 I ran a bulk `str.replace` that **silently didn't match** on three
  of four and printed "edits applied".
- **86.9:** evidenced "no `.env` write" with `git status`, on a **gitignored** file — a
  guard that **cannot fail**. And the mutation had already run: `.env` was written
  20:33:27Z, two minutes after the restart my own artifact records.

**Corrective method that finally held:** enumerate named locations **mechanically from
the verdict JSON**, assert every edit **landed** (non-zero exit otherwise), prove **zero
survivors**, and **negative-control the survivor probe**.

---

## Honest gaps

1. **The credential incident is unremediated** and the producer fix is unconfirmed.
2. **86.75 (yesterday's harness changes) remains unverified** — I authored and audited it.
3. **Three steps await operator decisions**: 86.62 and 86.9 (spawn again or hand to a
   fresh executor), 86.44 (D1-only scope, or fix the number producer).
4. **86.63's GENERATE is untouched** — production code on the live trade path.
5. **The cause of the 06-12 analysis-emptiness break (86.69, P0) is still unknown.**
6. **86.67's gate returned `gate_passed: false`** on a 38-vs-37 URL over-claim; its
   security finding is verified independently and stands regardless.

---

## Version

**6.93.221** — one bump, from 86.58's flip. 91 commits, and the changelog rule shipped
yesterday behaved exactly as designed.
