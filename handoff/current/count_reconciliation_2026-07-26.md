# Count re-derivation — 2026-07-26

The goal requires this explicitly: *"Re-derive them; do not inherit them"* and *"Measure,
don't assert. Re-derive every count in this document — including the 222/168/54 split."*

Re-derived from `.claude/masterplan.json` today. **The DRAFT's numbers are not wrong —
they use a narrower definition of "open" than a naive read suggests.**

## The definitional split (this is the whole discrepancy)

Full status distribution across the masterplan:

```
done 765 | pending 218 | deferred 15 | dropped 5 | superseded 4 | merged 2 | blocked 1
```

| definition | count | phases |
|---|---|---|
| `status == "pending"` — **the DRAFT's definition** | **218** | **21** |
| everything not `done`/`dropped` — a naive "open" | **240** | **32** |

`218 + 15 deferred + 1 blocked + 2 merged + 4 superseded = 240`. Both figures are correct;
they answer different questions. The DRAFT's "21 phases" matches the pending-only count
exactly, which confirms the definition.

**Anyone quoting a backlog number must state which one they mean.** The 11 extra phases in
the naive count (2, 3, 4, 4.14, 29, 36, 37, 39, 40, 53, 64) are almost entirely `deferred`.

## DRAFT (2026-07-25) vs measured (2026-07-26), pending-only

| metric | DRAFT | measured today | delta |
|---|---|---|---|
| open steps | 222 | **218** | −4 |
| phases | 21 | **21** | 0 |
| P0 | 26 | **22** | −4 |
| P1 | 45 | **44** | −1 |
| P2 | 91 | **93** | +2 |
| P3 | 40 | **40** | 0 |
| P4 | 8 | **8** | 0 |
| unset | 12 | **11** | −1 |
| `harness_required: false` (operator) | 53 | **53** | 0 |
| executor steps | 168 (derived) | **165** | −3 |

## Reconciliation of the P0 delta — residual **zero**

```
DRAFT pending-P0                      26
  − closed this session (P0 only)      5   80.2, 80.1, 80.27, 80.3, 80.4
  + added this session                 1   36.7 (kill switch cannot fire)
                                    ----
  expected                            22
  measured                            22   -> residual delta 0
```

Note `80.31` was **P2**, not P0 — six steps closed, but only five were P0. The P1/P2/unset
deltas (−1/+2/−1) are from a concurrent writer, exactly as the DRAFT warned would happen.

## Two metadata defects found while counting

Both are steps **named** `[OPERATOR ACTION -- not an executor task]` but **not flagged**
`harness_required: false`, so any executor filter picks them up as executor work:

| step | `harness_required` | priority |
|---|---|---|
| `79.55` | **missing (`None`)** | **P0** |
| `78.15` | **`true`** | P2 |

`79.55` is the consequential one: it is the P0 RESTART BLOCKER, and its mislabelling is
what inflated an executor-P0 count in this session's triage from 18 to 19. Fixing it is a
two-field metadata edit, not code.

**Deliberately not queued as its own masterplan step.** The queue-discovered-defects rule
exists for real defects; a research-gated step for two metadata fields would be
disproportionate. Recorded here and in the operator ask list instead — flagged so the
operator can decide, rather than silently absorbed.

## Method

```python
open_steps = [s for ph in masterplan["phases"] for s in ph["steps"]
              if s.get("status") == "pending"]           # DRAFT definition
priority   = Counter(s.get("priority", "unset") for s in open_steps)
operator   = [s for s in open_steps if s.get("harness_required") is False]
```
