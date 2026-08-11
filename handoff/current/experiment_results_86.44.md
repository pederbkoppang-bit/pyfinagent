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

The 481 have one mechanical cause: `run_harness.py:953` passes the **loop index** as
`cycle`, so every `--cycles 1` invocation writes `Cycle 1`.

## 2. Criterion 2 -- ANSWERED: something DOES read it, and it is not display-only

| consumer | what it does with the token | reads the NUMBER? |
|---|---|---|
| **`scripts/smoketest/steps/finalize.py:70-72`** | **`int()` then `max()+1`** | **YES -- as a sequence** |
| `backend/api/backtest.py:1427-1434` | split key + display string | no |
| `backend/agents/harness_state_reader.py:143,149` | `content.split("## Cycle")` | no |
| `backend/slack_bot/scheduler.py:464` | `line.startswith("## Cycle")` + date | no |
| `scripts/qa/verdict_history_86_21.py:196` | `^## Cycle .*phase=<id> result=` | no |
| `.claude/hooks/lib/harness_log_gate.py:94` | keys on `phase=` | no |

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
the parser now returns **1,224 of 1,224**.

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
correctly); the **number** is what races. This is the mechanism behind the 141
duplicate integers.

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
   integers are evidence of D4, and normalising them destroys that evidence.

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

## 8. Criterion 6 -- mutation matrix, 3 cells, ALL KILLED

```
==========================================================================
CONTROL -- every check must be GREEN before any cell is scored
==========================================================================
  GREEN  d1_concurrent_append       72/72 new entries survived 12 concurrent writers against a 1064-cycle seed
  GREEN  d2_parser_lossless         parser returned 1224 of 1224 headers
  GREEN  d3_runbook_placeholder     0 bare `## Cycle N --` template lines remain

  KILLED       M1_revert_d1_to_read_modify_write
               -> d1_concurrent_append: -1033/72 new entries survived 12 concurrent writers against a 1064-cycle seed
               restore byte-identical: True
  KILLED       M2_revert_d2_to_digits_only
               -> d2_parser_lossless: parser returned 1064 of 1224 headers
               restore byte-identical: True
  KILLED       M3_restore_the_copypaste_trap
               -> d3_runbook_placeholder: 1 bare `## Cycle N --` template lines remain
               restore byte-identical: True

POST-RESTORE control: {'d1_concurrent_append': True, 'd2_parser_lossless': True, 'd3_runbook_placeholder': True}

ALL CELLS KILLED: True
```

Control observed **GREEN before any cell was scored**; every restore
**byte-identical**; post-restore control green.

> **The matrix's own instrument was wrong twice before it was right, and both are
> recorded because both would have flattered the result.** (a) The D1 check first
> drove the real producer including `_reconciliation_log_line()`, which opens a
> **BigQuery client per call** -- 72 live calls, >120s, and a test touching live
> services. Stubbed that one unrelated dependency; the file write under test is not
> stubbed. (b) The check seeded a **14-byte** file, and a read-modify-write race is
> only lost when the read+write is slow enough to interleave -- i.e. a function of
> file size. A tiny seed would have let M1 **SURVIVE** and reported the old code as
> safe. It now seeds the real log's bulk.

## 9. Files changed

| file | change |
|---|---|
| `scripts/harness/run_harness.py` | D1: `O_APPEND` instead of read-modify-write |
| `backend/api/backtest.py` | D2: parser accepts any cycle token |
| `docs/runbooks/per-step-protocol.md` | D3: `<N>` placeholder + append-never-rewrite note |
| `scripts/qa/mutation_matrix_86_44.py` | NEW -- 3 cells, control-gated |
| `scripts/qa/prove_cycle_number_toctou_86_44.py` | NEW -- D4 demonstration |

## 10. What is NOT claimed

- **Not** that cycle numbers are now unique. They are not; D4 is unfixed and filed.
- **Not** that history was corrected. It was deliberately left as written.
- **Not** that 481 vs 482 is settled -- two rules disagree and mine is stated.
- **Not** that the D4 probe proves the race fires in production timing; it proves
  the window exists and is losable when startup jitter is removed.
