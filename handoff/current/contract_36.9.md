# Contract — masterplan step 36.9

**[P0] Three more ways the kill switch reports `armed:true` while a leg cannot fire.**

Step id: `36.9` · Phase: PLAN · Date: 2026-07-26 · HEAD at contract time: `edb67997`

## Immutable success criteria (copied VERBATIM from `.claude/masterplan.json`)

1. `evaluate_breach` compares a restored `sod_date` to the current UTC date and does not report `armed:true` / a daily breach based on a stale multi-day-old snapshot on GET /kill-switch, POST /resume, and the MCP risk_server tool -- a test proves this and FAILS against the current code
2. `nav_invalid` returns `armed` consistent with `any_breached` (an invalid NAV must not report `armed:true`) -- a test proves this and FAILS against the current code
3. `sod_nav=0.0` either does not latch as a valid baseline, or the re-anchor check is corrected to treat 0.0 the same as None -- a test proves the wedge is closed and the 409 remediation message is now true
4. The healthy path (valid, fresh `sod_date`, valid NAV, `sod_nav > 0`) is byte-for-byte unchanged -- assert against fixed fixtures
5. MUTATION-TEST all three fixes independently

Immutable verification command:
```
source .venv/bin/activate && python -m pytest backend/tests/ -q -k kill_switch
```
Immutable `live_check`: three test logs (one per finding) each showing the pre-fix failure
reproduced, then the post-fix behavior, against the real archived audit shapes measured 2026-07-26.

## Research-gate summary

`handoff/current/research_brief_36.9.md` — `gate_passed: true`, **11 sources read in full**
(floor 5), 42 URLs, 31 snippet-only, recency scan performed, 14 internal files inspected.

**The gate changed my design, and that is the point of running it.** I went in intending to make
all three fail loud and conservative, and explicitly asked whether the literature argues the
opposite — that a monitor which disarms on stale input is worse than one that keeps measuring
against a stale anchor. It does not. It says something sharper:

- **A stale anchor is the worst quadrant, not merely the less-safe one.** A 2-day move read as a
  1-day loss simultaneously *loses* coverage and *biases toward a spurious trip* — a nuisance trip
  and a diagnostic failure at once. Prior art: SEC market-wide circuit-breaker triggers are
  "calculated daily based on the prior day's closing price"; the daily anchor is definitionally
  date-scoped.
- **The real counterweight exists but constrains rather than opposes.** IEC 61511 Cl. 16.2.4 treats
  running with a safety function disarmed as a *bypass*, permitted only with compensating measures;
  16.2.3 bounds duration; 16.2.6/16.2.7 require authorization, indication and a bypass log; 11.7.3.2
  requires manual shutdown stay enabled. pyfinagent already satisfies these: 36.12's per-cycle order
  BLOCK is the compensating measure, and `BLOCK, NOT PAUSE` is exactly 11.7.3.2.
- **Therefore finding 3 is the clause violation, not just a bug**: a bypass *with no exit*. Highest-
  confidence fix of the three.
- **`nav_invalid` → unknown must not read as OK.** RuntimeAI (2026-05-12): *"If the policy plane is
  unreachable, the answer is no. Not 'best-effort.' Not 'log and pass.' Closed."* Kubernetes
  probe semantics and the three-state health model agree: unknown ≠ healthy.
- **`0.0` as a latched baseline is the semipredicate/sentinel anti-pattern** (`str.find() -> -1`).

Disclosed gaps, carried forward honestly: the CME Globex Kill Switch page (closest real-world
re-enable analogue) timed out and is snippet-only; no peer-reviewed source is in the read-in-full
set (both candidates paywalled/binary), so IEC clause text is quoted via two independent industry
sources that agree. The "4.0% measured on this book" figure is a **code comment**
(`paper_trader.py:1150-1152`) attributing the measurement to this very step — the researcher did
not reproduce it under a GET-only constraint and flagged it as a lead, not evidence. **I reproduced
it live before writing this contract** (below).

## Live pre-fix evidence (GET only, captured before any code change)

`handoff/current/captures_36.9/live_kill_switch_pre_fix.json`, from the running `:8000`:

```
today (UTC)          = 2026-07-26
sod_date on the wire = 2026-07-24   -> STALE BY 2 DAYS
armed                = True   any_breached = False
daily leg  : sod_nav 23838.19 x 0.96 = 22884.66   <- the 4% point, against a 2-day-old anchor
trailing leg: peak   24666.57 x 0.90 = 22199.91
```

`22884.66` reproduces the masterplan's measured figure exactly. **Finding 1 is not hypothetical: it
is live on the operator's badge endpoint right now.** This also re-verifies the goal's standing
precondition — the trailing leg still fires at NAV ≤ 22199.91 — before I touch kill-switch code.

## Hypothesis

`evaluate_breach` derives `armed` from *baseline presence* alone. Three inputs can make a leg
structurally unable to fire while presence still holds: the daily anchor can be **stale** (right
shape, wrong day), the current NAV can be **unmeasurable**, and a baseline can be **latched at
0.0** (present, positive-looking to an `is None` test, useless to the math). Making `armed` mean
"this leg can actually fire *now*" — per leg, additive keys, never a new falsy state — closes all
three without changing the healthy path.

## Plan

**F1 — stale `sod_date`.** In `evaluate_breach`, compare the snapshot's `sod_date` to today's UTC
date. A stale date makes the daily leg unevaluable: skip the daily math (never compute a percentage
from a stale anchor), set a new additive `daily_baseline_stale: true`, and fold it into `armed`.
Per-leg only — the trailing leg is a high-water mark, not date-scoped, so it keeps firing. This
matches the module's existing "the markers are PER LEG" doctrine and preserves protection rather
than removing it.

**F2 — `nav_invalid`.** `armed` is currently computed *before* the `nav_invalid` early return, so an
unmeasurable NAV returns `any_breached:False` **and** `armed:true`. In that branch `armed` becomes
`False` with an explicit reason: a leg that cannot measure cannot fire.

**F3 — `0.0` latch, fixed at the root AND at the consumer.** `update_sod_nav` refuses a
non-positive NAV — it does not assign, does not write a `sod_snapshot` audit row, and logs loud.
`sod_nav` therefore stays `None`, which the existing `is None` re-anchor predicate already handles,
so the next cycle genuinely re-anchors and the 409's promise becomes true. Additionally the
`paper_trader.py:1142` predicate is corrected to treat `<= 0` like `None` (defense in depth for any
0.0 already latched in a live process).

**Consumer safety (enumerated, not sampled).** 6 backend callers of `evaluate_breach`; exactly one
re-anchors first (`paper_trader.py:1154`), which is why the autonomous cycle is unaffected and the
three non-re-anchoring surfaces are the exposed ones. `armed` is consumed by
`paper_trading.py:598` and `kill_switch.py:873` as `.get("armed", True)` (**fails OPEN**) and by
`OpsStatusBar.tsx:318` / `KillSwitchPanel.tsx:137` as `armed === false`. **Therefore `armed` must
stay a plain bool and all new signals must be additive keys** — encoding a third state as
`armed: undefined` would render every surface ACTIVE. No consumer signature changes.

**Fixtures — a time bomb I must not ship.** Mechanical enumeration of every `_sod_date` assignment
in tests: **12 across 3 files** — 6 × `"2026-07-26"` (today), 4 × `"2026-05-22"`, 1 × `"2026-07-24"`,
1 × `'2026-04-20'`. The 6 hardcoding today would pass today and **fail tomorrow** once F1 lands.
Every fixture that wants the daily leg to fire is made date-relative (computed at test time); the
past-dated ones are either made relative or kept explicitly stale where staleness is the point.
These are genuine guards — **I fix the fixtures, not the fix.**

**Do not break.** `tests/verify_phase_23_2_19.py:47-50` is a *source-scan* pin requiring the literal
strings `snap.get("sod_date")` and `state.update_sod_nav(nav, date=today)` to remain in
`paper_trader.py`; the predicate edit must preserve both. `tests/services/test_sod_daily_roll.py`
re-implements the roll predicate inline at :80/:100/:156 and will drift silently.

**Out of scope → own steps.** The inline predicate duplication (3 sites + 1 source-scan pin, root
cause: no shared helper) is a real defect but not one of the three this step authorizes; it gets its
own research-gated step written for an executor with no memory of this discovery.

## Do-no-harm

Paper trading only. No `.env` edits, no flag flips, `historical_macro` frozen, no optimizer runs.
Kill-switch **limits**, stops, sector caps, DSR ≥ 0.95, PBO ≤ 0.5 byte-untouched — this step changes
only *whether a leg reports itself able to fire*, never a threshold, and every change is
more-conservative or fail-loud. No peak reset. `handoff/kill_switch_audit.jsonl` md5
`ce8fb93348bb9a3bbe26f2d91b1bc05e` verified before and after any experiment that could write it.
`:8000` GET-only, never restarted or POSTed to. `:3000` never driven.

## References

- `handoff/current/research_brief_36.9.md` (gate envelope, source tables, recency scan)
- IEC 61511 Cl. 16.2.3/16.2.4/16.2.6/16.2.7, 11.7.3.2 (bypass discipline) — via
  [instrumentationtools](https://instrumentationtools.com/iec-61511-standard-requirements-for-safety-bypass-and-override/)
- [SEC — market-wide circuit breakers](https://www.investor.gov/introduction-investing/investing-basics/glossary/stock-market-circuit-breakers) (daily anchor is date-scoped)
- [RuntimeAI — kill switch for autonomous AI, 2026-05-12](https://runtimeai.io/blog/2026-05-12-kill-switch.html) (unreachable → closed)
- [Kubernetes probes](https://kubernetes.io/docs/tasks/configure-pod-container/configure-liveness-readiness-startup-probes/), [health-check endpoint design](https://web-alert.io/blog/health-check-endpoint-design-livez-readyz-guide) (unknown ≠ healthy)
- [The Sentinel Object Pattern](https://python-patterns.guide/python/sentinel-object/) (0.0-as-absent)
- [gt-engineering — proof test & diagnostic coverage](https://www.gt-engineering.it/en/insights/process-safety-processi-gt-engineering/proof-test-diagnostic-coverage/), [Spurious Trip Rate](https://silsafe.net/spurious-trip-rate-explained/)
- Internal: `backend/services/kill_switch.py` (`evaluate_breach` :700-793, `update_sod_nav` :513-527,
  `_load_from_audit` sod branch :279-293, `_log_disarmed_once` :793), `backend/services/paper_trader.py:1138-1160`
