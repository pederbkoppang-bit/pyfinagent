# Experiment results -- step 86.44

**Step**: `86.44` (phase-86, **P3**) | **Phase**: GENERATE | **Date**: 2026-08-11
**Driver**: Main (`pyfinagent-06`) | **Contract**: `ea5b1cd5` (written BEFORE any code)
**Census tree**: `915d2cb0` | **HEAD at write time**: `cc682525b371b2efd31ae40a71a799035c3cfa0a`

**Three production files changed, all in producers or readers of
`harness_log.md`. No existing log content was edited.**

---

## 0. THE GATE'S HEADLINE CLAIM WAS WRONG, AND CRITERION 2 IS WHY I FOUND IT

The research gate concluded: *"The cycle number is WRITE-ONLY STATE: no consumer
keys, sorts or dedupes on it."* My contract adopted that and built a hypothesis on
it (*"the cycle number is not worth fixing"*).

**It is false.** `scripts/smoketest/steps/finalize.py:70-72`:

```python
def _next_cycle_number(text: str) -> int:
    nums = [int(m.group(1)) for m in re.finditer(r"^## Cycle\s+(\d+)", text, re.MULTILINE)]
    return (max(nums) + 1) if nums else 1
```

**A consumer parses the number as an integer and computes `max + 1` from it.**
Criterion 2 requires this be *determined by grep*, not assumed -- and running that
grep is exactly what refuted the claim I had been handed.

**Two of the gate's six cited consumer paths do not exist**
(`backend/services/harness_state_reader.py`, `scripts/harness/scheduler.py`); the
real modules are `backend/agents/harness_state_reader.py` and
`backend/slack_bot/scheduler.py`. A third (`HarnessDashboard.tsx`) is absent
entirely. The gate was right about the *modules* and wrong about the *paths*, and
wrong about the central claim.

## 1. Criterion 1 -- census RE-DERIVED at a named tree, with the extraction rule

**Rule stated with the number**: headers matched by `^## Cycle (.+?)\s*--`, token
stripped. Tree `915d2cb0`.

```
total headers            : 1224
  numeric                  : 1064
  NON-numeric              : 160
      58  unresolved template placeholder (N / N+k)
          e.g. '## Cycle N --'
      54  EMPTY token (header starts '## Cycle --')
          e.g. '## Cycle -- 2026-04-18 --'
      36  integer + parenthetical qualifier
          e.g. '## Cycle 30 (continued) --'
      10  step-id written where a cycle number belongs
          e.g. '## Cycle 4.15.3 --'
       2  other
          e.g. '## Cycle 7 Q/A correction (Main, 2026-05-27 09:05) --'

  => all four classes are DISTINCT FORMATS, not corruption:
     every one is a well-formed markdown header a human or a template wrote.
```

| quantity | value |
|---|---|
| `## Cycle` headers | **1,224** |
| numeric token | **1,064** |
| non-numeric token | **160** |
| token literally `1` | **481 (39.3%)** |
| distinct integers that duplicate | **141** |
| headers sharing a duplicated integer | **969** |

> **DISCREPANCY, DISCLOSED**: the research gate reported **482** literal `Cycle 1`;
> I measure **481**. Two defensible extraction rules disagree by one. Mine is stated
> above; I am not silently adopting either number.

**The 481 have at least TWO mechanical causes, derived by splitting the log into
blocks and testing each against the producer's own template:**

| | count | share |
|---|---|---|
| blocks whose token is `1` | **481** | 100% |
| contain `**Planner hypothesis:**` (emitted **unconditionally** by `run_harness`) | **418** | 86.9% |
| do **not** | **63** | 13.1% |
| of those 63, carry `phase=` **in the header line** | **62** | |

**`run_harness`'s header template cannot emit `phase=`** -- it is
`## Cycle {cycle} -- {timestamp}`, and `grep -c phase=` over the whole entry
f-string returns **0**. So those 62 are **manual protocol-format entries** that
restart per-step numbering at 1, a different mechanism entirely.

> **CORRECTED. This line used to say "The 481 have ONE mechanical cause".** That was
> asserted, not derived, and it is false for **≥63 of 481 (13.1%)**. Worse, §5's
> correction block -- written to fix exactly this class of error -- **cited this
> sentence approvingly** (*"which my own §1 already attributed correctly"*) and so
> inherited it. **A correction that leans on an underived claim propagates the
> defect it was written to remove.**

## 2. Criterion 2 -- ANSWERED: something DOES read it, and it is not display-only

| consumer | what it does with the token | reads the NUMBER? |
|---|---|---|
| **`scripts/smoketest/steps/finalize.py:70-72`** | **`int()` then `max()+1`** | **YES -- as a sequence** |
| `backend/api/backtest.py:1427-1434` | split key + display string | no |
| `backend/agents/harness_state_reader.py:143,149` | `content.split("## Cycle")` | no |
| `backend/slack_bot/scheduler.py:464` | `line.startswith("## Cycle")` + date | no |
| `scripts/qa/verdict_history_86_21.py:196` | `^## Cycle .*phase=<id> result=` | no |
| `.claude/hooks/lib/harness_log_gate.py:94` | keys on `phase=` | no |

> **AND THE Q/A FOUND A SECOND ONE I MISSED**, which is the right outcome of asking
> it to re-run my grep after I had just caught the gate missing one:
> **`finalize.py:113`** reads the number *again* as a split key --
> `split(f"## Cycle {cycle}")[-1]`. **That is where a D4 collision acquires a
> consequence**: with a duplicated number, the split lands on the wrong occurrence.
> It strengthens the case for 86.55 rather than weakening it.

**So renumbering history is not a no-op after all** -- it would move
`max()`, and therefore the next number `finalize.py` assigns. That strengthens the
case for *not* renumbering, on different grounds than my contract gave.

## 3. Criterion 3 -- the 160 characterised: DISTINCT FORMATS, not corruption

| n | class | example |
|---|---|---|
| **58** | unresolved template placeholder | `## Cycle N --`, `## Cycle N+42 --` |
| **54** | EMPTY token | `## Cycle -- 2026-04-18 --` |
| **36** | integer + parenthetical qualifier | `## Cycle 30 (continued) --` |
| **10** | step-id in the cycle slot | `## Cycle 4.15.3 --` |
| **2** | other | `## Cycle 7 Q/A correction (Main, ...) --` |

**Every one is a well-formed markdown header** that a human or a template wrote.
None is truncation or byte damage. The 58 placeholders trace to D3.

## 4. WHAT WAS FIXED

### D1 -- the producer destroyed concurrent writers' entries. FIXED.

`scripts/harness/run_harness.py` did `read_text()` then
`write_text(existing + entry)`. Anything another process appended in between was
**silently destroyed** -- the whole block. Replaced with an `O_APPEND` open.
`write(2)` specifies the seek-and-write is one atomic step; the old code opted out.

**Two Claude Code sessions work this repo concurrently. This was live data loss in
the harness's own audit trail.**

### D2 -- the reader silently dropped 13.1% and MISATTRIBUTED their bodies. FIXED.

`backend/api/backtest.py` split on `^## (Cycle \d+)\s*--`. A non-numeric header was
**not a split point**, so its body was glued onto the **preceding** cycle. The
Harness tab did not show a gap -- it showed those cycles' text **under the wrong
cycle**, which is worse, because it looks complete. Widened to `Cycle [^\n]+?`;
**the fixed code returns 1,224 of 1,224.**

> ### THE FIX IS COMMITTED BUT **NOT IN FORCE**, AND I FAILED TO SAY SO
>
> The cycle-1 Q/A caught this and it is the finding I am least comfortable with,
> because this project has a standing rule about exactly it and I still wrote
> "the parser now returns 1,224" as though it were live.
>
> | fact | value |
> |---|---|
> | backend pid | **66306** |
> | pid started | **2026-08-10 21:33:01** |
> | fix commit `fe9a6dad` | **2026-08-11T17:13:05+02:00** (~20h LATER) |
> | `GET /api/backtest/harness/log` **right now** | **1064 cycles** -- the PRE-FIX number |
> | the fixed code, in memory | 1224 |
>
> **So the Harness tab is still mis-attributing 160 headers as I write this.** The
> running process imported the old module at 21:33 yesterday and has never re-read
> it. Per the standing rule, restarts batch to session end and never near the 20:00
> cycle -- so the remedy is **a pending-restart entry, not a restart**:
> `handoff/current/pending_restart_2026-08-11.md`.
>
> Measure the running process, never the file. I had that written down.

### D3 -- the runbook was a copy-paste trap. FIXED.

`docs/runbooks/per-step-protocol.md` contained a fenced block whose first line was
literally `## Cycle N -- ...`, while the very next line already used `<id>`
placeholder syntax. Now `## Cycle <N>`, with an explicit substitute-this note and an
append-never-rewrite warning naming the D1 hazard.

## 5. D4 -- FOUND HERE, DEMONSTRATED, DELIBERATELY NOT FIXED

`finalize.py` computes the number **before** appending, with no lock:

```
seed max                 : 100
  concurrent appenders     : 16
  numbers assigned         : [101, 101, 101, 101, 102, 102, 103, 103, 103, 103, 104, 104, 105, 105, 106, 106]
  DISTINCT numbers         : 6 of 16
  collisions               : 10
  rows actually written    : 17 (seed 1 + 16 new)

  DATA: every append survived -- finalize.py already uses open('a'),
        so it never had D1's read-modify-write data loss.
  NUMBER: 10 writers claimed a number another writer also claimed.
        _next_cycle_number() reads max BEFORE the append, with no lock:
        scripts/smoketest/steps/finalize.py:70-72 then :83-85.
        THIS is the mechanism behind the duplicate integers in history.
```

**10 collisions of 16.** Data all survived (that producer already appends
correctly); the **number** is what races.

> **CORRECTED: I called this "the mechanism behind the 141 duplicate integers".
> It is ONE mechanism, and not the dominant one.** Derived after the Q/A challenged
> it:
>
> | | |
> |---|---|
> | headers sitting in a duplicate group | **969** |
> | of which the token is literally `1` | **481 (49.6%)** |
> | remaining | **488 across 140 integers** |
> | times `finalize.py` has written this file | **3** |
>
> **Roughly half the duplicates come from `run_harness.py`'s loop index** -- but
> not all of the `1`s do: §1 now derives the split as **418 run_harness-shaped, ≥62
> manual**. A producer that has run **3 times** cannot account for 969 headers.
> At least three mechanisms are in play: the loop index, this TOCTOU, and two
> sessions hand-numbering.
>
> **This is the same error as §7's range in step 86.9 earlier today**: a real
> finding stated over a population I had not derived.

> **The first run of this probe reported 0 collisions**, because process-startup
> jitter serialised the workers and the TOCTOU window never overlapped. **I did not
> report that as safety.** The barrier does not create the defect -- it stops
> startup jitter from hiding it, and every worker is inside the real, unmodified
> `append_cycle_row`.

**NOT fixed in this step, deliberately.** A correct fix needs an exclusive-lock
primitive; macOS has no `flock(1)`, and this project's own reference memo prescribes
atomic-`mkdir` with dead-pid *and* age breaking. That is a new concurrency primitive
in a shared writer, which is not a P3 side-quest at 17:30 on a cycle day. **Filed as
its own research-gated step** per the standing queue-discovered-defects rule.

## 6. Criterion 4 -- the renumbering decision, STATED

**DECISION: DO NOT RENUMBER HISTORY.** Reasons, in order:

1. **It would change behaviour, not just cosmetics.** `finalize.py` derives the next
   number from `max()` over history. Rewriting 1,224 headers moves that.
2. **The audit trail's value is that it is what was written.** Editing 1,224
   historical entries to tidy a field is exactly the "wrong while implying right"
   outcome criterion 4 warns against.
3. **The numbers were never unique and the history should say so.** 141 duplicated
   integers are evidence that **at least three mechanisms** wrote this file --
   `run_harness.py`'s loop index, manual protocol-format entries, and D4's TOCTOU --
   and normalising them destroys that evidence.

   > **SUPERSEDED, not annotated.** This clause used to read *"141 duplicated
   > integers are evidence of D4"*. The cycle-1 Q/A retracted that and I corrected
   > §5 but **left this sentence standing** -- in the very section carrying the
   > criterion-4 decision a future reader acts on. A correction that sits beside the
   > claim it retracts has not corrected anything.

**Leaving history wrong-but-honest, with the cause documented and the producer's
race filed.**

## 7. Criterion 5 -- the producer WAS changed; here is the proof, and its limit

**Changed**: the producer's **write** (D1). **Proven under concurrency**: 12
concurrent processes x 6 appends = **72/72 entries survived** against a
production-sized 1,064-cycle seed. Under the reverted mutant the same test lost
**1,033 of 1,064 seeded cycles**.

**NOT changed**: the **numbering**. So there is no "new numbering" to prove unique
-- and I will not stage a vacuous proof of one. Instead §5 **demonstrates the
existing numbering is NOT unique** (10/16 collisions) and files it.

## 8. Criterion 6 -- mutation matrix, **4** cells, ALL KILLED

```
==========================================================================
CONTROL -- every check must be GREEN before any cell is scored
==========================================================================
  GREEN  d1_concurrent_append       72/72 new entries survived 12 concurrent writers against a 1064-cycle seed
  GREEN  d2_parser_lossless         parser returned 1224 of 1224 headers
  GREEN  d3_runbook_placeholder     0 unallowed live file(s) carry the bare template (derived from 2 git-grep hits, 2 allowlisted)

  KILLED       M1_revert_d1_to_read_modify_write
               -> d1_concurrent_append: 49/72 new entries survived 12 concurrent writers against a 1064-cycle seed
               restore byte-identical: True
  KILLED       M2_revert_d2_to_digits_only
               -> d2_parser_lossless: parser returned 1064 of 1224 headers
               restore byte-identical: True
  KILLED       M3_restore_the_trap_in_the_RUNBOOK
               -> d3_runbook_placeholder: 1 unallowed live file(s) carry the bare template (derived from 3 git-grep hits, 2 allowlisted) ['docs/runbooks/per-step-protocol.md']
               restore byte-identical: True
  KILLED       M4_restore_the_trap_in_CLAUDE_md
               -> d3_runbook_placeholder: 1 unallowed live file(s) carry the bare template (derived from 3 git-grep hits, 2 allowlisted) ['CLAUDE.md']
               restore byte-identical: True

POST-RESTORE control: {'d1_concurrent_append': True, 'd2_parser_lossless': True, 'd3_runbook_placeholder': True}

ALL CELLS KILLED: True
```

Control observed **GREEN before any cell was scored**; every restore
**byte-identical**; post-restore control green.

> **THE GUARD'S POPULATION IS NOW DERIVED, AFTER I GOT THE CLASS WRONG TWICE.**
> Cycle 1 found the check scanning **one** file when the class had two. I "fixed"
> that by pinning a **two-file list** -- and cycle 2 then derived **five** live
> occurrences, including `.claude/hooks/lib/harness_log_gate.py` (a live hook
> docstring carrying the identical bare-`N`-beside-`<step_id>` inconsistency D3 was
> filed for) and `tests/_phase_24_helpers.py`, **whose comment literally reads
> "format from CLAUDE.md"** -- direct evidence the literal propagates.
>
> **A pinned list can only ever be as complete as the last person who edited it, and
> it can never fail on a NEW file that acquires the literal.** The check now derives
> its population by `git grep` and subtracts a **named allowlist** whose two members
> each carry their reason. My own edit missed one of the five; **the derived guard
> caught it** and named the file.

> **M4 EXISTS BECAUSE THE CYCLE-1 Q/A FOUND MY GUARD COULD NOT FAIL.** D3 was fixed
> in the runbook only, while **`CLAUDE.md` -- auto-loaded into every session, so the
> *more* likely copy-paste source -- still carried the literal**, and
> `check_d3_runbook_placeholder()` scanned one file. A guard whose population is one
> member of a two-member class is vacuous on the other. The check now scans a
> **pinned file list** (not a directory glob), and **M3 and M4 each name the file
> they broke**, so the two cells cannot be satisfied by the same fix.

> **THE M1 MAGNITUDE IS TIMING-DEPENDENT AND I QUOTED IT AS IF IT WERE STABLE.** My
> commit message said the mutant "loses 1,033 of the 1,064 seeded cycles". That was
> **one observation**. A second run of the same cell lost a different amount
> (**45/72**, then **49/72**, seed intact). Both are losses and the cell is
> KILLED either way, but **the honest claim is "the read-modify-write loses entries,
> in an amount that varies with interleaving", not a specific figure.** A race's
> damage is not a constant.

## 9. Files changed

| file | change |
|---|---|
| `scripts/harness/run_harness.py` | D1: `O_APPEND` instead of read-modify-write |
| `backend/api/backtest.py` | D2: parser accepts any cycle token |
| `docs/runbooks/per-step-protocol.md` | D3: `<N>` placeholder + append-never-rewrite note |
| `CLAUDE.md` | D3: same literal, in the file auto-loaded EVERY session |
| `.claude/hooks/lib/harness_log_gate.py` | D3: same literal in a live hook docstring |
| `tests/_phase_24_helpers.py` | D3: comment + assertion message (its comment says "from CLAUDE.md" -- direct evidence the literal propagates) |
| `scripts/qa/mutation_matrix_86_44.py` | NEW -- **4 cells**, control-gated; D3 population **derived by git grep**, not pinned |
| `scripts/qa/prove_cycle_number_toctou_86_44.py` | NEW -- D4 demonstration |
| `handoff/current/pending_restart_2026-08-11.md` | NEW -- D2 is committed but NOT in force |

> **This table said "3 cells" when there were 4, and OMITTED `CLAUDE.md` -- the file
> whose omission was cycle-1's sharpest finding.** A files-changed table that does
> not list the files changed is the same defect the step is about, one level up.

**NOT changed, deliberately**: `docs/audits/phase-24-2026-05-12/24.0-charter-findings.md`
carries the literal too, but it is a **dated audit record**. Editing a 2026-05-12
audit to tidy a template is the same category error as renumbering history, which
criterion 4 declined to do. It is **named in the guard's allowlist with that reason**,
so the exclusion is visible rather than silent.

## 10. What is NOT claimed

- **Not** that cycle numbers are now unique. They are not; D4 is unfixed and filed.
- **Not** that history was corrected. It was deliberately left as written.
- **Not** that 481 vs 482 is settled -- two rules disagree and mine is stated.
- **Not** that criterion 3's "111" is stale. **I called it stale and that was a
  mis-attribution.** It is a **rule difference**, not a drift: counting the token up
  to `--` gives **160**; counting the first whitespace-delimited field gives **123**
  by my rule and **112** by the Q/A's. The 48-ish delta is the classes that *begin*
  with a digit -- 36 parenthetical (`30 (continued)`), 10 step-ids (`4.15.3`), 2
  other. Three rules, three numbers, all defensible; the criterion's figure sits in
  that family. **The criterion was never amended and none of these numbers is wrong
  -- they answer different questions.**
- **Not** that the D4 probe proves the race fires in production timing; it proves
  the window exists and is losable when startup jitter is removed.
