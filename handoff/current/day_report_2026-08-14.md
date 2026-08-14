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
   Tonight was an unplanned natural experiment: **99 commits → exactly 1 version bump**,
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

**6.93.221** — one bump, from 86.58's flip. **99 commits** (re-derived at push time; an
earlier draft said 91, which went stale while the report was being written), and the
changelog rule shipped
yesterday behaved exactly as designed.

---

# SESSION 2 (01:50–05:30 CEST) — the harness was rebuilding work it had already shipped

Operator reprioritised mid-session: *"add them later after this one we need our harness to
work correctly."* The picker chain was therefore **not** restarted — and Q1's committed
answer (`f6c2dbf4`, 08-13) independently says **STOP optimising the ranking**, because the
binding constraint is upstream analysis emptiness, not the ranking.

## The headline: a 1.45M-token audit rebuilt a step that had already shipped the fix

Four dated events on one defect, each measured with a positive control:

| date | event |
|---|---|
| 08-09 | **86.21** files the counter defect **and prescribes the fix verbatim** — "a small append-only per-step verdict ledger" |
| 08-09→11 | 86.21 **BUILDS it** and runs **five** Q/A cycles: `verdict_ledger.jsonl`, `verdict_history_86_21.py` (473 lines), `mutation_matrix_86_21.py` (**16/16 killed**, re-verified today) |
| 08-13 | **86.75**'s ultracode audit — 23 agents, **1,451,204 tokens** — reports that diagnosis as its headline discovery and ships the same fix. `'86.21'` appears **nowhere** in its step JSON (control: `'86.33'` does) |
| 08-14 | **I re-derived it again on top**, and only noticed because I overwrote 86.21's artifact and recovered 433 lines from `git show` |

**Filed as step 86.76 (P1).** The gap, positive-controlled: no instruction to search the
project's own defect register exists in `researcher.md`, `research-gate.js`, or
`rules/research-gate.md` (probes live — 10/9/27 hits for other terms). The gate's
prior-art discipline points entirely outward at literature.

## The re-derivation caused a regression

86.75 silently replaced CLAUDE.md:371-376's trigger (**3 consecutive CONDITIONALs**) with
**"3rd attempt or later."** Replayed on 36.17's real `C,F,F,C,C,PASS`: the correct rule
never fires (longest run 2); the shipped one **forces FAIL at attempts 4 and 5** and denies
the PASS 36.17 earned at attempt 6. It was also stricter than F1b's documented 5-attempt
budget, which *escalates* rather than auto-fails. **86.21 had flagged that exact ambiguity
on 08-11 and asked for it to be reconciled** — the re-derivation resolved it the wrong way.

## Fixed and pushed

1. **The counter inherited the defect it replaced.** A missing sink made
   `records_retained: 0` **byte-identical** to a genuine first attempt, silently disabling
   escalation. Added `source_present`; M1 killed, control green, M2 unscored *with its
   reason* (`prune_wip_records` deletes by design).
2. **Both rails restored** to the consecutive-CONDITIONAL trigger.
3. **Both rails repointed** at `verdict_history_86_21.py` — the purpose-built counter with
   a 5-value status vocabulary where `ledger_missing`/`ledger_empty`/`unparseable` return
   `None` and fail **closed**, instead of a hand-rolled grep.
4. **A staleness cross-check neither tool had alone:** `qa_wip` is automatic, the ledger is
   hand-written, so `records_retained > ledger count` ⇒ **STALE**. Live: **86.62 → 4 vs 0**.
5. **`mutation_matrix_86_31`: 20/24 → 24/24 KILLED.** Four cells had silently stopped
   testing anything since `6e8f3169` (08-11, anchor drift). Verified *not* caused by
   tonight's edits — all four already 0 at HEAD.
6. **86.75 live_check**: 6 of 8 criteria measured. Fixed two defects in my own audit —
   `run_memo.py`'s docstring asserted both a claim and its correction, and
   `cycle_prompt.md` still stated a **3-source** floor plus a research-skip carve-out the
   operator overruled in May.

## Three of my own probes were wrong, and each is a class

- **A probe matched its own documentation.** `grep 'phase=86.33 result=CONDITIONAL'`
  returned 1 — the audit's *prose quoting the grep*. Anchored: **0**, as the basis said.
  Corpus-wide, **121 prose lines** contaminate any unanchored grep of `harness_log.md`.
- **A false zero from zsh not word-splitting** an unquoted var, with stderr suppressed and
  `|| echo NONE` printing a clean result. It hid a live 3-source floor.
- **`exit=$?` after a pipe reads `tail`.** I reported a suite "still exits 0" with dead
  cells; it returns 1. The script was right, my measurement was not.

## Still open, and honestly

- **The OAuth token is still public.** Re-scanned all 76 session files with the vendor
  pattern: same 5 files, positive control fires, **all 5 still on `origin/main`**.
  Unremediated — operator-gated (ask 06-2 / #20, one credential, one fix).
- **86.75 criteria 1 and 7**: 1 needs a driven Q/A; 7 is separation-of-duties review, now
  covering **three** `qa.md` edits I authored.
- **Nothing appends `verdict_ledger.jsonl` automatically.** 86.21 said so on 08-11. The
  cross-check makes the staleness *visible*; it does not fix it.
- **No step was flipped and no Q/A graded any of this.** No self-certification.

## Book (server-side read-back, 05:25)

NAV **$23,920.63**, cash **$20,425.99 (85.4%)**, **2 positions** (DELL, NTAP),
P&L **+19.6%** vs benchmark **+9.09%**. The 95.6%-cash / 1-position framing that motivated
the picker urgency **no longer describes the book**.

`degradation` key: the **08-12 cycle wrote one**, and 08-13 did too — the peer's 86.38
stake holds.
