# Live check — step 86.65


> **TIMESTAMP CORRECTION (2026-08-14 04:35 CEST).** Wall-clock times in this file were
> **narrated, not measured** — I read the clock once at session start and invented a
> progression from it. The real session spans **08-13 23:10 → 08-14 04:26** (~5h), not the
> 16+ hours the original times implied. Times below are now the **git commit timestamps**
> of this artifact, which are ground truth. Durations and orderings derived from the old
> figures should be disregarded; the measurements themselves are unaffected.
**Date:** 2026-08-14 ~02:59 CEST (git)
**Required shape:** *"the per-directory naming census and the grep-derived consumer list"*
**Immutable command:** `test -f docs/runbooks/per-step-protocol.md && echo runbook-present` → **`runbook-present`, exit 0**

---

## C1 — the census, SPLIT by directory, population rule beside each count

**RULE:** basename is exactly a protocol-artifact kind, or kind + `_`/`-` + a dotted step id.
Kinds: `contract`, `experiment_results`, `evaluator_critique`, `research_brief`, `live_check`.

```
handoff/current/  (non-recursive)            total matched: 488
    suffix_underscore     466      <- contract_86.63.md
    other                  18
    bare                    4

handoff/archive/  EXCLUDING _quarantine_2026-04-21   total matched: 4166
    bare                 3064      <- contract.md
    phase_prefix_dash     481
    suffix_underscore     444
    other                 177

EXCLUDED: _quarantine_2026-04-21 holds 43,905 .md files
```

**The exclusion is the point of criterion 1.** My first pass reported **48,114** archive
matches. That number is dominated by an April quarantine dump: **12,659 of 13,463**
`contract.md` files live under `_quarantine_2026-04-21/`, and only **4,390 of 48,301**
archive files are tracked in git at all. A tree-wide figure describes the dump, not the
convention.

## C2 — consumers, grep-derived, each classified

**CONVENTION-BOUND** (hardcode `<kind>_<step_id>`):

| consumer | binding |
|---|---|
| `.claude/hooks/lib/live_check_gate.py` (via `auto-commit-and-push.sh:154`) | `handoff/current/live_check_<sid>.md` |
| `.claude/hooks/lib/verdict_gate.py` (via `:257`) | `handoff/current/evaluator_critique_<sid>.json` |
| `.claude/hooks/archive-handoff.sh:215-233` | reads `<kind>_<sid>.md`, writes `<kind>.md` |
| `.claude/workflows/research-gate.js:211` | defaults to `research_brief_<sid>.md` |

**CONVENTION-AGNOSTIC**: `qa-write-guard.sh` (path-prefix only), `harness_log_gate.py`,
the agent files, and the `scripts/qa/*` probes (they name specific files, they do not
resolve a convention).

## C3 — ONE convention, and it is already coherent

**The three conventions are not competing — they are BY DIRECTORY, with a deliberate
transition between them:**

- `handoff/current/` → **`<kind>_<step_id>.md`** (466 of 488)
- `handoff/archive/phase-<sid>/` → **`<kind>.md`** (3,064 of 4,166)

`archive-handoff.sh:215` performs exactly that rename on snapshot, and its own comment
records why the old front-sid globs (`4.5.9-contract.md`) matched **zero** files for every
sid tested. So `phase_prefix_dash` (481) is **dead history**, not a live third convention.

**Historical files are NOT renamed**, and the reason is stated: no consumer requires it —
every convention-bound consumer above reads `handoff/current/`, never the archive — and
renaming would invalidate `file:line` citations across the archive and its
`provenance.md` tables. Cost with no reader.

## C4 — CLAUDE.md paths, all of them

```
distinct path-shaped refs checked : 61
resolve on disk                   : 39
globs skipped                     :  4
BROKEN                            :  1   -> CLAUDE.md:205 .claude/agents/per-step-protocol.md
```

Corrected to `docs/runbooks/per-step-protocol.md`, which the same file already cites
correctly at lines 28, 55 and 377. **Final sweep: 0 broken.**

> **The fix first appeared to FAIL.** The re-sweep still reported 1 broken — my own
> correction note, which quoted the dead path in backticks. Third instance this session of
> a probe matching its own documentation. The note now names the path without backticks.

**Negative control:** the checker reports a synthetic
`docs/runbooks/DEFINITELY_NOT_A_REAL_FILE_86_65.md` as absent, so the final 0 is a measured
zero, not a dead probe. **Positive control:** it still resolves the known-good runbook.

## C5 — no verification criteria edited

`git diff --stat -- .claude/masterplan.json` → **empty**. No step's criteria were touched.

## What this does NOT license

- **No Q/A has graded this**; the step is not flipped.
- The archive quarantine (43,905 files) is **described, not cleaned** — out of scope here.
