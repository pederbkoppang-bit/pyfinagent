# Live check — step 86.58

**Date:** 2026-08-13
**Backend:** pid **93024**, started **2026-08-13T20:30:59Z**.

> **CORRECTED (cycle 3).** This header previously read *"pid 99231, started
> `tir. 11 aug. 22.26.48 2026` — not restarted this session."* That was true when
> written (20:27:14Z) and went stale **3m45s later**: a **concurrent peer session**
> restarted the backend at 20:30:59Z on the operator's session-end batching
> instruction (`launchctl kickstart -k`, 99231 → 93024). Old pid gone, no zombie.
> **I did not restart it**, and I left the stale line in place while the cycle-2 Q/A
> was grading this file rather than edit an artifact mid-evaluation.
> **Every process-sourced measurement below was re-probed against 93024 and is
> unchanged.** The BQ- and log-derived measurements never touched the process.
**Verdict ledger:** attempt 1 `wf_b127735e-55b` = **FAIL**; attempt 2
`wf_1e709e75-776` = **CONDITIONAL** (all 6 criteria MET; two fixable live_check
defects). This is the **cycle-3** artifact and carries the corrected measurements,
never the refuted ones.

The masterplan requires three things. **All three are now supplied from the live
system.** §2 previously recorded the flag values as unobtainable; the cycle-2 Q/A
ruled that an over-claimed dead end, and it was right — three independent
positive-controlled instruments were available and are now used. The HTTP dead end
is kept because it is true, but it is no longer presented as the whole answer.

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

## 2. Flag values in the RUNNING process — MEASURED (corrected: I over-claimed a dead end)

**CORRECTED (cycle 3).** This section previously said the values were **NOT
OBTAINABLE** and recorded them UNVERIFIED. The cycle-2 Q/A agreed the HTTP dead end
is real and reproduced it exactly — but ruled that I **over-claimed a dead end where
three instruments existed**, and it was right. What follows is a measurement, not a
disclaimer. I have kept the negative result because it is true and load-bearing; I
have stopped calling it the whole answer.

### 2a. The HTTP surface genuinely does not expose them (unchanged, re-probed on 93024)

| Route | Result |
|---|---|
| `GET /api/settings/` | 200 — curated `FullSettings`, **45 keys**, **0 hits** for either flag |
| `…/all`, `…/flags`, `…/debug`, `/api/paper-trading/config` | **404** |

Route list **derived from `backend/api/settings_api.py`**, not guessed: only
`GET "/"`, `PUT "/"`, `GET "/models"`, `PUT "/models"`, `GET "/models/available"`.
**Positive control:** 15 `paper_*` keys **are** exposed, so the probe is live and the
absence is a property of the response model.

### 2b. INSTRUMENT 1 — `Settings()` is a positive-controlled read of the operator's real `.env`

`backend/config/settings.py:652` sets
`model_config = {"env_file": str(_ENV_FILE), ...}` → `backend/.env`.

**This is the correction that matters.** I previously dismissed a fresh `Settings()`
as "merely the defaults path". It is not — and the proof is a sibling flag:

```
paper_risk_judge_reject_binding              = True    <-- POSITIVE CONTROL: promoted in .env
paper_position_recommendation_fix_enabled    = False
paper_recommendation_vocab_fix_enabled       = False
```

`paper_risk_judge_reject_binding` reads **True** *because the operator promoted it in
`.env`* — it is not a code default. So the read path demonstrably reaches `.env`, and
**both 86.58 flags reading `False` is a positive-controlled measurement of the
operator's actual configuration**, not an inspection of source defaults.

### 2c. INSTRUMENT 2 — a flag-gated log line that never fired

`portfolio_manager.py:212-220` emits a WARNING **only** when
`paper_position_recommendation_fix_enabled` is ON **and**
`paper_synthesis_integrity_enabled` is OFF:

> `paper_position_recommendation_fix_enabled is ON while paper_synthesis_integrity_enabled is OFF -- rail-failure synthetic HOLDs can trigger signal_downgrade SELLs of healthy positions.`

Counts in the live `backend.log`:

```
"healthy position"                          0
"signal_downgrade"                          0
"UNRECOGNISED recommendation"               4   <-- POSITIVE CONTROL
```

The unconditional `UNRECOGNISED` line fired **4 times**, proving `decide_trades` ran
and the log channel works. The gated warning fired **0 times**. So the guard was never
satisfied → **posfix OFF, or synthesis_integrity ON.**

### 2d. INSTRUMENT 3 — what the running process actually wrote today

`paper_trader.py:452-457`:

```python
_pos_rec = reason
if (getattr(self.settings, "paper_position_recommendation_fix_enabled", False)
        and analysis_recommendation):
    _pos_rec = analysis_recommendation
```

The running process opened **DELL at 2026-08-13T19:31:19Z** and stored
`recommendation='new_buy_signal'` — the `reason`, not an analysis verdict. Production
does pass `analysis_recommendation` (`portfolio_manager.py:578,:918` →
`autonomous_loop.py:251,:1768`), so a posfix-**ON** process with a non-empty verdict
would not have stored that. **Consistent with posfix OFF**, written by the process
itself, hours before this artifact.

### 2e. Conclusion, and the residual gap I still cannot close

**Three independent instruments — a positive-controlled `.env` read, a flag-gated log
line that never fired against a working control, and a row the process wrote itself
today — all agree: both 86.58 flags are OFF in the running process.**

**Residual gap, stated:** a launch-time environment variable would override `.env`,
and I could not enumerate the full process environment (`ps eww 93024` exposed only
~14 env-like tokens, 0 flag hits). So this is convergent positive-controlled evidence,
**not** a direct read of the process's in-memory value. A read-only route exposing
these two flags would close it properly; that is out of scope here and recorded rather
than built.

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
