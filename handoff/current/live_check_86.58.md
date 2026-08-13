# Live check — step 86.58

**Date:** 2026-08-13
**Backend:** pid **99231**, started `tir. 11 aug. 22.26.48 2026` — **not restarted** this session.
**Verdict context:** written after Q/A `wf_b127735e-55b` returned **FAIL**; this artifact
carries the corrected measurements, not the refuted ones.

The masterplan requires three things. Two are supplied from the live system. **The
third cannot be obtained through the permitted surface, and that is stated as a
limitation rather than substituted.**

---

## 1. The verbatim production log line

The phase-86.20 guard firing in production, from `backend.log` and its rotated
archives (population: JSON-format lines, `^{"timestamp"`):

```
2026-08-10 21:15:12,974  phase-86.20: UNRECOGNISED recommendation 'new_buy_signal' (held position row ticker=NTAP) -- treated as neither buy nor sell nor downgrade. A producer vocabulary drift is the usual cause.
2026-08-11 21:21:09,983  phase-86.20: UNRECOGNISED recommendation 'new_buy_signal' (held position row ticker=NTAP) -- treated as neither buy nor sell nor downgrade. A producer vocabulary drift is the usual cause.
2026-08-12 20:23:05,549  phase-86.20: UNRECOGNISED recommendation 'new_buy_signal' (held position row ticker=NTAP) -- treated as neither buy nor sell nor downgrade. A producer vocabulary drift is the usual cause.
2026-08-13 21:31:15,781  phase-86.20: UNRECOGNISED recommendation 'new_buy_signal' (held position row ticker=NTAP) -- treated as neither buy nor sell nor downgrade. A producer vocabulary drift is the usual cause.
```

**Four consecutive cycle days.** The 2026-08-13 occurrence required a **fresh** read of
the live log — my working snapshot was built before that day's cycle completed at
19:31Z, so reporting from the snapshot alone would have claimed a false absence for
08-13.

**Criterion 6 (the line is preserved, never quieted) is satisfied on live evidence:**
it is still firing in production today, and it fires twice more in the driven
reproduction below. Source occurrences of `UNRECOGNISED recommendation` in
`backend/services/portfolio_manager.py`: **1** — unchanged by this step, which
modified **no** file under `backend/`.

---

## 2. Flag values from the RUNNING process — **NOT OBTAINABLE. Stated, not faked.**

The masterplan asks for "the measured flag values read from the RUNNING process".
**I could not get them, and I did not substitute anything that looks like them.**

What I probed against pid 99231:

| Route | Result |
|---|---|
| `GET /api/settings/` | 200 — returns `FullSettings`, a **curated** model of 45 keys |
| `…/all`, `…/flags`, `…/debug`, `/api/paper-trading/config` | **404** |
| `grep` for `recommendation_fix\|recommendation_vocab` in `/api/settings/` | **0 hits** |
| same grep in `/api/paper-trading/portfolio` | **0 hits** |

**Positive control — the probe is live:** the same endpoint exposes **15 `paper_*`
keys** (`paper_starting_capital`, `paper_max_positions`, `paper_max_per_sector`,
`paper_screen_top_n`, …). So the absence is a real property of the endpoint's
response model, not a broken query. Only `@router.get("/")`, `/models` and
`/models/available` exist in `backend/api/settings_api.py`; the flags are writable
there but **not readable**.

Reading `backend/.env` is denied. A restart is prohibited mid-session and would not
help — it would change the process, not the endpoint's schema.

**What I CAN establish, and it is weaker:** both flags are declared with default
`False` at `backend/config/settings.py:210` and `:214`, and the driven test below
asserts `False` from a fresh `Settings()` — **the defaults path, not the running
process.** No runtime override endpoint was called by this step.

**Consequence for the verdict:** the claim "the flags are OFF in the running process"
is **UNVERIFIED**. Everything below about flag-OFF behaviour describes the defaults
path. A route that exposes these two flags read-only would close this gap; it is not
in this step's scope and is recorded rather than built.

---

## 3. Derived count of held rows carrying a reason-shaped recommendation

**Re-derived at publication time, 2026-08-13T20:24:53Z** — the earlier figure was
stale because DELL entered eight minutes before the first artifact was written.

Population rule, read from source (`portfolio_manager.py:60-64`): the closed
vocabulary the SELL path tests is
`_BUY_RECS ∪ _DOWNGRADE_RECS = {BUY, STRONG_BUY, HOLD, SELL, STRONG_SELL}`.

```
CURRENTLY HELD ROWS: 2
  NTAP   rec='new_buy_signal'  qty=5.346643  entry=2026-07-31T18:47:37Z  in_closed_set=False
  DELL   rec='new_buy_signal'  qty=4.806437  entry=2026-08-13T19:31:19Z  in_closed_set=False

  off-vocabulary: 2 of 2 = 100.0%   in closed set: 0 of 2
```

**2 of 2 currently-held rows (100%) carry a reason-shaped value. 0 of 2 carry a member
of the closed set.**

---

## 4. Corrected blast radius — measured with the flags genuinely ON

The refuted claim was **1 of 1 (100%)**, produced by a harness that asserted the flags
were `False` and hand-set `'BUY'` as a stand-in. The corrected script enters the
condition via `Settings().model_copy(update={both flags: True})` — in-process only,
no `.env` write, no promotion, no live-book contact:

```
=== FLAGS OFF ===
  A  pos='new_buy_signal'   fresh='HOLD' -> []                                    fired=False
  B  pos='BUY'              fresh='HOLD' -> [('NTAP','SELL','signal_downgrade')]  fired=True
  C  pos='swap_buy'         fresh='HOLD' -> []                                    fired=False
  D  pos='BUY' fresh='SELL' -> [('NTAP','SELL','sell_signal')]   (sell_signal PRE-EMPTS)

=== FLAGS ON ===
  E  pos='new_buy_signal'   fresh='HOLD' -> []                                    fired=False
  F  pos='BUY'              fresh='HOLD' -> [('NTAP','SELL','signal_downgrade')]  fired=True
  G  pos='swap_buy'         fresh='HOLD' -> []                                    fired=False

=== DISCRIMINATION CONTROL: pos='Strong Buy', fresh='HOLD' ===
  flags OFF -> []                                     fired=False
  flags ON  -> [('NTAP','SELL','signal_downgrade')]   fired=True
```

**MEASURED BLAST RADIUS AT PROMOTION TIME: 0 of 2 currently-held rows.**

Three green controls: **B** (OFF) and **F** (ON) both fire, so neither half is vacuous;
and the **discrimination control** — dead OFF, live ON — proves the probe reads flag
state. Without that third cell, E and G being empty would be indistinguishable from
the override silently not taking effect. That is precisely the hole in the first
version.

Mechanism, verified in source: flag-ON `_resolve_rec` maps `'new_buy_signal'` to
`_UNRECOGNISED_REC`, a member of none of the three sets; and `_pos_rec` is written
**only** by `execute_buy` (`paper_trader.py:488`, `:512`), while the partial-sell path
at `:676` preserves the stored value. **Flipping a flag does not rewrite existing
rows.** Exposure begins at the next `execute_buy`.

---

## 5. Nothing was promoted, nothing was written

- Both flags remain at their declared defaults; the driven test **asserts** it and the
  flags-ON cells use an in-process `model_copy` that touches no file.
- `git diff --stat` for `backend/` across this step: **empty**.
- No `.env` write, no `launchctl` action, no manual cycle, no backend restart.
- The live book was not contacted: the driven test constructs in-memory dicts and
  calls `decide_trades` directly.
