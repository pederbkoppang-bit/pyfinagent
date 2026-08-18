---
name: test-isolation-leak-86-110
description: 86.110 -- a patch-table denominator over- AND under-reports; the polluted heartbeat self-healed before the step ran; the two leaking tests build their own CycleHealthLog so a provider-only fix misses them
metadata:
  type: project
---

Step 86.110 (test isolation leaking into git-tracked `handoff/.cycle_heartbeat.json`).
Three findings that are not derivable from the step's own premise.

**1. The obvious denominator is wrong in BOTH directions.** The intuitive
enumeration is "who patches `_HISTORY_PATH` but not `_HEARTBEAT_PATH`". Measured,
that set has 4 members beyond the 2 the caller named, and it is wrong twice over:
- It OVER-reports. `backend/tests/test_cycle_heartbeat_alarm.py` patches only
  `_HISTORY_PATH` and is **not** a leak -- its tests call
  `cycle_health.cycle_heartbeat_alarm()`, a READER, and seed history through a local
  helper into `tmp_path`. No writer is reached.
- It UNDER-reports. `scripts/smoketest_stages_5_through_13.py` swaps the constant by
  plain assignment and is not pytest-collected, so a tests-only sweep never sees it.
Also `test_phase_36_17...` reaches `record_cycle_end` yet appears in NEITHER patch
list, because it cuts higher: `monkeypatch.setattr(cycle_health, "get_log", ...)`.
**The correct denominator is transitive reachability of the writer, then cross-check
the patch table against it** -- not the patch table alone.

**Why:** a patch-table sweep is a claim about a set whose membership rule
("this test writes the real file") was never actually checked. Same class as
[[measure-dont-assert]] and [[count-the-class-not-your-list]].

**How to apply:** for any "which tests leak into X" question, enumerate
`grep -rln '<writer_symbol>'` FIRST and adjudicate each hit by reachability; use the
patch table only as a cross-check.

**2. The polluted artifact self-healed before the step began.** The premise was that
`c2` sits in the tracked heartbeat. Measured: working tree held `3e5afddb`, a real
cycle present twice in `handoff/cycle_history.jsonl`, written by the live writer
hours earlier. `c2` has **0** rows in the 174-row ledger and was **never committed**
to the heartbeat across its 20 commits. So the "restore it?" question was moot.
**Rule extracted:** derived + last-writer-wins + live writer on a known cadence =>
take no content action, just stop the leak; hand-restoring would MANUFACTURE a
cycle_id with no ledger row, i.e. recreate the defect. Restore by hand only for
append-only/accumulating state. And the clean-check is not "equals HEAD" (it never
will be) but "is its `cycle_id` present in the ledger".

**3. The constraint that will bite GENERATE:** both leaking tests construct
`ch.CycleHealthLog()` directly rather than calling `get_log()`, so a provider-only
fix at `get_log` does **not** cover them. The fix must target the path binding
(or those two call sites change too).

**Trap, measured this session:** under zsh, unquoted `grep -rn X --include=*.py .`
dies with `(eval):1: no matches found` and emits NOTHING -- which reads exactly like
"this symbol does not exist anywhere". My first enumeration returned zero rows for
both constants; taking that at face value would have concluded there is no leak at
all. Always quote `--include='*.py'`. Related but distinct from
[[zsh-no-word-splitting]] (that one is word-splitting; this is NOMATCH on globs).

**External anchor worth reusing:** `infotroph/tree-is-clean` (GitHub Action) exists
for exactly this class -- its stated motivation is "detecting any files undesirably
written into the working directory, e.g. by tests that ought to be using a proper
tempdir". And the auto-repair literature deliberately does NOT cover filesystem
state: ODRepair (ICSE'22) says it focuses on "polluted heap-state" and sets file
system aside. No pytest plugin does snapshot-and-diff of tracked files around a
session -- the nearest ones (pytest-picked, pytest-run-changed, pytest-cagoule)
answer the INVERSE query. So the guard has to be hand-built; do not shop for it.
