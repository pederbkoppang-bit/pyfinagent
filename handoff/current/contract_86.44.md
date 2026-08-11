# Contract -- step 86.44

**Step**: `86.44` (phase-86, **P3**) | **Phase**: PLAN | **Date**: 2026-08-11 (~17:0x CEST, read from `date`)
**Driver**: Main (`pyfinagent-06`) | **Tree**: `915d2cb0d35b96c0c304316e39f354232d8f19e0`
**Written BEFORE any code.** No file is modified at this moment.

---

## 1. Research gate

**PASSED** -- `wf_8945aab3-878`, tier `moderate`, script-enforced and recomputed:
**8 sources read in full** (floor 5), **16 URLs** (floor 10), recency scan present, all
8 claimed URLs verified present in the brief, `brief_status: COMPLETE`,
`rail_dropped: null`, 12 internal files inspected. Brief:
`handoff/current/research_brief_86.44.md` (24,788 chars).

Sources: RFC 9562 (UUIDv7 -- monotonic counters need a serialization point, s6.3),
RFC 9413 (protocol robustness -- strict parsing, s5.1), RFC 5424 (syslog -- MSGID
resets to 1), `write(2)` (O_APPEND seek+write is "an atomic step"), ULID spec, Git
object model (content addressing), CommonMark, OpenTelemetry log data model.

## 2. THE GATE CHANGED THE STEP. The filed defect is the least of what is here.

The step was filed as "cycle numbers are neither unique nor uniformly shaped". True,
and I re-derived it at tree `915d2cb0d35b96c0c304316e39f354232d8f19e0`. But the gate established the finding that
reframes the work:

**THE CYCLE NUMBER IS WRITE-ONLY STATE. Nothing reads it as an identifier.**

| consumer | what it does with the number |
|---|---|
| `backend/api/backtest.py:1415` | display string only |
| `backend/services/harness_state_reader.py:143` | splits the bare literal |
| `scripts/qa/verdict_history_86_21.py:196` | matches `.*` over it |
| `scripts/harness/scheduler.py:464` | keys on `phase=` / date |
| `.claude/hooks/harness_log_gate.py:94` | keys on `phase=` / date |
| `frontend/.../HarnessDashboard.tsx:448` | `key={i}` -- the array index |

**So renaming or renumbering has ZERO migration blast radius -- and equally, zero
benefit.** Fixing the *numbers* fixes nothing that anything depends on.

### What is actually broken, verified by me in source

**D1 -- THE APPEND IS A FULL-FILE READ-MODIFY-WRITE, AND TWO SESSIONS RUN THIS REPO.**
`scripts/harness/run_harness.py:976-980`:

```python
if HARNESS_LOG.exists():
    existing = HARNESS_LOG.read_text(encoding="utf-8")
HARNESS_LOG.write_text(existing + entry, encoding="utf-8")
```

Between the `read_text` and the `write_text`, anything another writer appends is
**silently destroyed** -- not a garbled line, the whole block. `write(2)` specifies
that an `O_APPEND` seek-and-write "is an atomic step"; this code opts out of it.
**This is a data-loss defect in the file that is the harness's own audit trail**, and
a peer session is live on this machine right now.

**D2 -- THE READER SILENTLY DROPS 13.1% OF CYCLES AND MISATTRIBUTES THEIR BODIES.**
`backtest.py:1415` splits on `^## (Cycle \d+)\s*--\s*(.+)$`. A header whose token is
not `\d+` is **not a split point**, so its body is **glued onto the preceding
cycle**. Measured at this tree: **1,224 headers, 1,064 matched, 160 dropped
(13.1%)**. The Harness tab therefore shows some cycles' text under the wrong cycle.

**D3 -- THE RUNBOOK IS A COPY-PASTE TRAP.** `docs/runbooks/per-step-protocol.md:334`
contains the literal `## Cycle N -- YYYY-MM-DD -- phase=X.Y result=PASS/CONDITIONAL/FAIL`,
which is the likely origin of the `Cycle N` / `Cycle N+k` headers.

### The shape census, re-derived at `915d2cb0d35b96c0c304316e39f354232d8f19e0`

**Extraction rule, stated with the number** (per my own standing lesson): headers
matched with `^## Cycle (.+?)\s*--` and the captured token stripped.

| quantity | value |
|---|---|
| `## Cycle` headers | **1,224** |
| token is literally `1` | **481 (39.3%)** |
| token is non-numeric | **160** |
| distinct integers that duplicate | **141** |
| headers involved in a duplicate | **969** |
| dropped by the reader's regex | **160 (13.1%)** |

> **A DISCREPANCY I AM NOT SMOOTHING OVER**: the gate reported **482** literal
> `Cycle 1`; I measure **481**. Different extraction rules, and I have stated mine.
> The step's criterion 1 asks for a range or a named tree; I give the tree and the
> rule, and flag that two defensible rules disagree by one.

**The 481 have a single mechanical cause**: `run_harness.py:953` passes the **loop
index** as `cycle`, so every `--cycles 1` invocation writes `Cycle 1`.

## 3. Immutable success criteria -- VERBATIM from `.claude/masterplan.json`

> Generated programmatically from the masterplan, not typed. Cycle 2 of step 86.9
> today failed on exactly this: a section headed "VERBATIM" whose criteria had drifted
> in 5 of 6 entries, one of which dropped the clause that produced that step's best
> finding.

> 1. The duplicate and malformed-header counts are RE-DERIVED with the commands stated, at a named tree, and reported as a range or with the tree named -- the file grows daily and any bare figure is stale on arrival.
> 2. Whether anything READS the cycle number is DETERMINED by grep across the repo, and the answer is stated plainly even if it is 'nothing does' -- that answer changes the correct fix and must not be skipped because it is unglamorous.
> 3. The 111 non-numeric headers are characterised (distinct format vs corruption) rather than counted and left opaque.
> 4. The decision on historical renumbering is STATED with its reason. Leaving history wrong-but-honest is an acceptable outcome; leaving it wrong while implying it is fixed is not.
> 5. If the producer is changed, the new numbering is proven UNIQUE under concurrent writers -- two sessions appending in the same second must not collide, demonstrated rather than argued.
> 6. Mutation-test any new guard: revert it and show the check goes red, with the control observed GREEN first.

## 4. Hypothesis

**The cycle number is not worth fixing; the three defects around it are.** Because no
consumer keys on it, renumbering history buys nothing and risks rewriting an audit
trail. The defensible outcome is: **leave history wrong-but-honest, make the producer
crash-safe and concurrency-safe (D1), make the reader lossless or loudly lossy (D2),
and remove the copy-paste trap (D3)** -- with criterion 4's decision stated and
reasoned rather than assumed.

## 5. Plan

- **P1 -- FIX D1 (the real defect).** Replace the read-modify-write with a true
  `O_APPEND` write. This is the only change with a live data-loss motive.
- **P2 -- FIX D2 or make it loud.** Either widen the reader to accept any token, or
  have it count and report what it skipped. **Silently dropping 13.1% is the part that
  must not survive**, whichever is chosen.
- **P3 -- FIX D3.** Make the runbook's template non-copy-pasteable as a literal.
- **P4 -- criterion 4: STATE the renumbering decision.** Recommendation: **do not
  renumber**. Reason: nothing reads it, and rewriting 1,224 historical headers edits an
  audit trail to fix a field with no consumer.
- **P5 -- criterion 5**: any new numbering must be proven unique under concurrent
  writers. **If P4 stands (no new numbering), say so plainly rather than staging a
  vacuous proof** -- and note that D1's fix is what actually makes concurrent appends
  safe.
- **P6 -- criterion 6**: mutation-test every guard, **control observed GREEN first**,
  byte-identical restore.

### Explicitly NOT doing

- **Not** renumbering 1,224 historical headers.
- **Not** touching `handoff/verdict_ledger.jsonl`'s `cycle` field -- that is **86.46**,
  a separate filed step with its own criteria.
- **Not** restarting anything. **Not** running the harness. The 19:30 freeze holds.

### Risk

`harness_log.md` is append-only history that two sessions write. Every change here is
to a **producer** or a **reader**, never to existing content. D1's fix reduces the risk
that the peer and I destroy each other's entries.

## 6. References

- `handoff/current/research_brief_86.44.md` (gate `wf_8945aab3-878`)
- RFC 9562 s6.3; RFC 9413 s5.1; RFC 5424 s7.3.1; `write(2)` O_APPEND; ULID spec
- `scripts/harness/run_harness.py:953,976-980`; `backend/api/backtest.py:1415`;
  `docs/runbooks/per-step-protocol.md:334`;
  `backend/services/harness_state_reader.py:143`; `scripts/qa/verdict_history_86_21.py:196`
