# Live check — step 86.44

**Date:** 2026-08-14 (~05:55 CEST)
**Named tree:** `6fdc7f0ad88dac50706aa59b587d965193151845`
**Subject file mtime (UTC):** `2026-08-13T20:52:42Z`
**Backend:** pid 93024 — irrelevant here; every measurement below is against files on disk.

> **STATUS: PREREQUISITE ARTIFACT, created so 86.44's remaining attempt is not spent on
> a known-missing file.** The step was PARKED with code shipped and **no `live_check`**,
> and its ledger shows **3 prior Q/A spawns** — under the counter repointed in
> phase-86.75 the next attempt is **PASS or FAIL**, so spawning against a missing
> required artifact would have burned it. **I have NOT spawned that attempt.**
>
> **Scope honesty up front:** items 1–4 below are mine, measured at the named tree.
> **Item 5 I did NOT verify** — see §5.

---

## 1. Counts, re-derived, with the command beside each

The step's own criterion says any bare figure is stale because the file grows daily —
**including from me: I appended two cycle headers to this file during this session**
(`phase=86.75` and `phase=86.58`), so I am one of the writers being measured.

```bash
grep -c '^## Cycle' handoff/harness_log.md                 # 1229   total headers
grep -cE '^## Cycle [0-9]+' handoff/harness_log.md         # 1117   numeric-shaped
# non-numeric = 1229 - 1117                                #  112
```

| Quantity | Value at this tree |
|---|---:|
| Total `## Cycle` headers | **1229** |
| Numeric-shaped | **1117** |
| Non-numeric | **112** |
| Distinct numbers used **more than once** | **141** |

Duplicate concentration (`grep -oE '^## Cycle [0-9]+' … | awk '{print $3}' | sort -n | uniq -c | sort -rn`):

```
482  Cycle 1
 15  Cycle 4
  8  Cycle 2
  7  Cycle 7
  7  Cycle 5
```

**`Cycle 1` alone accounts for 482 of 1117 numeric headers (43.1%).** The numbering is
not merely non-unique — for most entries it carries no information at all.

---

## 2. Does anything READ the cycle number? **NO. Nothing does.**

This is the criterion that "changes the correct fix", and it resolves cleanly.

**The one real consumer** — `backend/agents/harness_state_reader.py:143-149`:

```python
cycles = content.split("## Cycle")
recent = cycles[-last_n:] if len(cycles) > last_n else cycles[1:]   # skip header
return {
    "available": True,
    "total_cycles": len(cycles) - 1,
    "recent": [f"## Cycle{c}" for c in recent],
}
```

**It splits on the literal delimiter `"## Cycle"` and counts segments. The number is
never parsed** — no `int()`, no regex capture, no ordering by it. `total_cycles` is
`len(cycles) - 1`, i.e. **the delimiter count**, and `recent` is a raw text slice.

**Consequence:** the consumer is *completely indifferent* to whether a header reads
`1`, `482`, `--` or `N+58`. It counts and slices identically either way. **Duplicate and
malformed numbers cause no observable defect in any reader.**

Search scope: all `*.py/*.js/*.mjs/*.sh/*.ts/*.tsx` outside `node_modules`, `.venv`,
`.git`. 15 files match `## Cycle`; **all others are tests, QA probes, this step's own
prior scripts, hook helpers, or the producer** — none parses the value.
**Positive control:** the same search shape finds `harness_log` in five known files, so
the probe is live and the "nothing parses it" result is not a false zero.

---

## 3. The 112 non-numeric headers are TWO FORMATS, not corruption

```
 54   ## Cycle -- …          bare, no number at all
~58   ## Cycle N+1 … N+58    a relative-placeholder convention
```

Both are **deliberate authoring conventions**, not damage: the `--` form omits the
number entirely, and the `N+k` form is a writer declining to guess the next absolute
number. **Both still begin `## Cycle`, so both split correctly** in the consumer above —
which is why 112 malformed headers have never produced a visible fault.

---

## 4. Renumbering decision: **DO NOT RENUMBER HISTORY.**

**Reason, following directly from §2:** no consumer reads the value, so renumbering
would change nothing observable while rewriting 1,229 entries of an append-only audit
log — trading a real risk (corrupting historical evidence, invalidating every existing
`file:line` citation into this file) for zero measurable benefit.

Per the criterion's own wording, this is **"wrong-but-honest"**: the numbers are
decorative, this artifact says so plainly, and nothing here implies they are fixed.
**The correct forward fix is to stop pretending the field is an identifier** — either
drop it or make the producer emit something genuinely unique — which is item 5's
territory, not history's.

---

## 5. Concurrent-writer uniqueness — **NOT VERIFIED BY ME. Conditional criterion.**

Criterion 5 applies **"if the producer is changed"**. Prior sessions shipped work here
that I did **not** re-run and am **not** certifying:

- `scripts/qa/prove_cycle_number_toctou_86_44.py` (4,119 b, 2026-08-11) — its header
  describes **D1 as FIXED**: `run_harness.py` previously did `read_text` + `write_text`,
  and the TOCTOU window between `_next_cycle_number(read_text())` and `open("a")` let
  two writers claim the same max.
- `scripts/qa/mutation_matrix_86_44.py` (11,347 b).

**What the next executor must do before claiming criterion 5:** re-run both, confirm the
producer change is actually in the tree, and demonstrate two concurrent appends in the
same second do not collide. **I did not do this**, and this artifact must not be read as
evidence that it holds.

Note the probe's own header records a self-caught trap worth preserving: *"the first run
of this probe reported 16/16 distinct and would have been read as a pass"* — the probe
was wrong before it was right.

---

## What this artifact does and does not license

- **Does:** unblock the step's final attempt by supplying the required file, with items
  1–4 measured at a named tree.
- **Does NOT:** certify criterion 5, certify the shipped producer change, or claim the
  step is ready to PASS. **One attempt remains and it is PASS-or-FAIL** — a Q/A should be
  spawned only after criterion 5 is genuinely settled.
- **Nothing was changed:** no file under `backend/` or `scripts/` was modified;
  `handoff/harness_log.md` was not edited by this artifact (the two headers I appended
  earlier were normal protocol logging for 86.75 and 86.58, before this measurement).
